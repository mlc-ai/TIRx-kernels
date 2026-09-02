# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Persistent SM100 block-scaled GEMM with sReLU, probability, and amax.

Upstream source:
``python/cudnn/gemm/cutedsl/dense/srelu/``
``dense_blockscaled_gemm_persistent_srelu_quant.py``.
"""

import heapq
from functools import cache
from itertools import combinations

import tirx_kernels.kern as K

_TRY_WAIT_TICKS = 10_000_000
_SMEM_CAPACITY = 232_448
_MAX_ACTIVE_CLUSTERS = {1: 148, 2: 74, 4: 33, 8: 15, 16: 7}
_SF_MODES = (("float8_e8m0fnu", 16), ("float8_e8m0fnu", 32), ("float8_e4m3fn", 16))
_C_DTYPES = ("float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
_D_DTYPES = ("float32", "float16", "bfloat16")
_MODE_KEYS = (
    "ab_dtype",
    "sf_dtype",
    "sf_vec_size",
    "c_dtype",
    "d_dtype",
    "a_major",
    "b_major",
    "c_major",
    "mma_tiler_mn",
    "cluster_shape_mn",
    "vector_f32",
    "L",
)
_SHORT_DTYPE = {
    "float4_e2m1fn": "f4",
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
    "float8_e4m3fn": "e4",
    "float8_e5m2": "e5",
    "float8_e8m0fnu": "e8",
}


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align_up(value, alignment):
    return _ceil_div(value, alignment) * alignment


def _next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def _dtype_bits(dtype):
    return {
        "float4_e2m1fn": 4,
        "float8_e4m3fn": 8,
        "float8_e5m2": 8,
        "float8_e8m0fnu": 8,
        "bfloat16": 16,
        "float16": 16,
        "float32": 32,
    }[dtype]


def _descriptor_base(ldo, sdo, swizzle):
    arrangement_type = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = 0
    value |= (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement_type & 0x7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


def _instruction_descriptor(M, N, ab_dtype, sf_dtype, a_major, b_major):
    """Fold the source SM100 block-scaled MMA descriptor fields."""
    if M not in (128, 256) or N not in (64, 128, 192, 256):
        raise ValueError(f"unsupported instruction shape {(M, N)}")
    sf_format = {"float8_e4m3fn": 0, "float8_e8m0fnu": 1}[sf_dtype]
    value = 0
    if ab_dtype in {"float4_e2m1fn", "float8_e5m2"}:
        value |= 1 << 7
        value |= 1 << 10
    if a_major == "m":
        value |= 1 << 15
    if b_major == "n":
        value |= 1 << 16
    value |= ((N >> 3) & 0x3F) << 17
    value |= (sf_format & 1) << 23
    value |= ((M >> 4) & 0x1F) << 24
    return value & 0xFFFFFFFF


def _descriptor_with_address(base, shared_address):
    base_value = K.bitwise_or(
        K.shift_left(K.uint64(base >> 32), K.uint64(32)), K.uint64(base & 0xFFFFFFFF)
    )
    address_field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(base_value, address_field)


def _advance(state):
    state.advance()


def _try_wait_acquire(dst, barrier, phase):
    K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, K.cast(phase, "uint32")
    )


def _wait_plain(barrier, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, K.cast(phase, "uint32"), K.uint32(_TRY_WAIT_TICKS)
        )


def _wait_plain_if_needed(barrier, phase, speculative_ready):
    with K.If(speculative_ready == K.uint32(0)):
        with K.Then():
            _wait_plain(barrier, phase)


def _elected():
    elected_lane = K.local_scalar("uint32")
    elected_pred = K.local_scalar("uint32")
    K.ptx.elect_sync(elected_lane, elected_pred, K.uint32(0xFFFFFFFF))
    return elected_pred == K.uint32(1)


def _issue_ptx_if(predicate, mnemonic, *args):
    with K.If(predicate):
        with K.Then():
            K.ptx[mnemonic](*args)


def _valid_mode(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    if tile_m not in (128, 256) or tile_n not in (64, 128, 192, 256):
        return False
    if cluster_m not in (1, 2, 4) or cluster_n not in (1, 2, 4):
        return False
    if cluster_m * cluster_n > 16 or (tile_m == 256 and cluster_m % 2):
        return False
    if mode["ab_dtype"] == "float4_e2m1fn":
        if mode["a_major"] != "k" or mode["b_major"] != "k":
            return False
        if (mode["sf_dtype"], mode["sf_vec_size"]) not in _SF_MODES:
            return False
    else:
        if mode["a_major"] not in ("k", "m") or mode["b_major"] not in ("k", "n"):
            return False
        if (mode["sf_dtype"], mode["sf_vec_size"]) != ("float8_e8m0fnu", 32):
            return False
    return (
        mode["c_dtype"] in _C_DTYPES
        and mode["d_dtype"] in _D_DTYPES
        and mode["c_major"] == "n"
        and mode["L"] in (1, 2)
    )


def _mode_label(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    return (
        f"{_SHORT_DTYPE[mode['ab_dtype']]}_{_SHORT_DTYPE[mode['sf_dtype']]}v{mode['sf_vec_size']}_"
        f"{_SHORT_DTYPE[mode['c_dtype']]}_{_SHORT_DTYPE[mode['d_dtype']]}_"
        f"{mode['a_major']}{mode['b_major']}_{mode['c_major']}_"
        f"t{tile_m}x{tile_n}_c{cluster_m}x{cluster_n}_"
        f"{'v' if mode['vector_f32'] else 's'}f32_l{mode['L']}"
    )


# Exact PASS rows from ``source_results/capability_matrix.json``.  Each row was
# compiled in an isolated process and matched the FP32 oracle on three launches.
# Keeping this frozen roster here prevents single-axis source probes from being
# accidentally expanded into an unverified Cartesian product.
_VERIFIED_PROBES = (
    (
        "anchor",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "input_float4_e2m1fn_float8_e8m0fnu_v16",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "input_float4_e2m1fn_float8_e8m0fnu_v32",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "input_float4_e2m1fn_float8_e4m3fn_v16",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e4m3fn",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "input_float8_e4m3fn_float8_e8m0fnu_v32",
        256,
        256,
        512,
        2,
        "float8_e4m3fn",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "m",
        "n",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "input_float8_e5m2_float8_e8m0fnu_v32",
        256,
        256,
        512,
        2,
        "float8_e5m2",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "m",
        "n",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "c_float16",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "float16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "c_float32",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "float32",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "c_float8_e4m3fn",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "float8_e4m3fn",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "c_float8_e5m2",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "float8_e5m2",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "d_float16",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "float16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "d_float32",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "float32",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "c_fp8e4_d_f32",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "float8_e4m3fn",
        "float32",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "c_f32_d_f16",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "float32",
        "float16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "c_f16_d_bf16",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "float16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "vector_f32_false",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        False,
    ),
    (
        "batch_l1",
        256,
        256,
        512,
        1,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e4m3fn_kk",
        256,
        256,
        512,
        2,
        "float8_e4m3fn",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e4m3fn_kn",
        256,
        256,
        512,
        2,
        "float8_e4m3fn",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "k",
        "n",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e4m3fn_mk",
        256,
        256,
        512,
        2,
        "float8_e4m3fn",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "m",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e4m3fn_mn_c_n",
        256,
        256,
        512,
        2,
        "float8_e4m3fn",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "m",
        "n",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e5m2_kk",
        256,
        256,
        512,
        2,
        "float8_e5m2",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e5m2_kn",
        256,
        256,
        512,
        2,
        "float8_e5m2",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "k",
        "n",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e5m2_mk",
        256,
        256,
        512,
        2,
        "float8_e5m2",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "m",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "major_float8_e5m2_mn_c_n",
        256,
        256,
        512,
        2,
        "float8_e5m2",
        "float8_e8m0fnu",
        32,
        "bfloat16",
        "bfloat16",
        "m",
        "n",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "tile_128x64",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        64,
        1,
        1,
        True,
    ),
    (
        "tile_128x128",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        128,
        1,
        1,
        True,
    ),
    (
        "tile_128x192",
        256,
        192,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        192,
        1,
        1,
        True,
    ),
    (
        "tile_128x256",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        1,
        1,
        True,
    ),
    (
        "tile_256x64",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        64,
        2,
        1,
        True,
    ),
    (
        "tile_256x128",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        128,
        2,
        1,
        True,
    ),
    (
        "tile_256x192",
        256,
        192,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        192,
        2,
        1,
        True,
    ),
    (
        "tile_256x256",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "cluster_cta1_1x1",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        1,
        1,
        True,
    ),
    (
        "cluster_cta1_1x2",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        1,
        2,
        True,
    ),
    (
        "cluster_cta1_1x4",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        1,
        4,
        True,
    ),
    (
        "cluster_cta1_2x1",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        2,
        1,
        True,
    ),
    (
        "cluster_cta1_2x2",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        2,
        2,
        True,
    ),
    (
        "cluster_cta1_2x4",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        2,
        4,
        True,
    ),
    (
        "cluster_cta1_4x1",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        4,
        1,
        True,
    ),
    (
        "cluster_cta1_4x2",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        4,
        2,
        True,
    ),
    (
        "cluster_cta1_4x4",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        256,
        4,
        4,
        True,
    ),
    (
        "cluster_cta2_2x1",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "cluster_cta2_2x2",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        2,
        True,
    ),
    (
        "cluster_cta2_2x4",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        4,
        True,
    ),
    (
        "cluster_cta2_4x1",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        4,
        1,
        True,
    ),
    (
        "cluster_cta2_4x2",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        4,
        2,
        True,
    ),
    (
        "cluster_cta2_4x4",
        256,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        4,
        4,
        True,
    ),
    (
        "tail_m",
        272,
        256,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "tail_n",
        256,
        272,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "tail_k",
        256,
        256,
        544,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "tail_mnk_l2",
        272,
        272,
        544,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        256,
        2,
        1,
        True,
    ),
    (
        "tail_cta1_n64",
        272,
        208,
        544,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        64,
        1,
        1,
        True,
    ),
    (
        "tail_cta2_n192",
        272,
        208,
        544,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        192,
        2,
        1,
        True,
    ),
    (
        "persistent_cta1",
        4096,
        512,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        128,
        128,
        1,
        1,
        True,
    ),
    (
        "persistent_cta2",
        4096,
        512,
        512,
        2,
        "float4_e2m1fn",
        "float8_e8m0fnu",
        16,
        "bfloat16",
        "bfloat16",
        "k",
        "k",
        "n",
        256,
        128,
        2,
        1,
        True,
    ),
)


def _probe_config(probe):
    (
        label,
        M,
        N,
        K_dim,
        L,
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        d_dtype,
        a_major,
        b_major,
        c_major,
        tile_m,
        tile_n,
        cluster_m,
        cluster_n,
        vector_f32,
    ) = probe
    return {
        "label": f"source_{label}",
        "source_probe": label,
        "M": M,
        "N": N,
        "K": K_dim,
        "L": L,
        "alpha": 2.0 / 3.0,
        "ab_dtype": ab_dtype,
        "sf_dtype": sf_dtype,
        "sf_vec_size": sf_vec_size,
        "c_dtype": c_dtype,
        "d_dtype": d_dtype,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "mma_tiler_mn": (tile_m, tile_n),
        "cluster_shape_mn": (cluster_m, cluster_n),
        "vector_f32": vector_f32,
    }


@cache
def _verified_configs():
    return tuple(_probe_config(probe) for probe in _VERIFIED_PROBES)


def _verified_structure(mode):
    structural_keys = tuple(key for key in _MODE_KEYS if key != "L")
    return any(
        all(mode[key] == verified[key] for key in structural_keys)
        for verified in _verified_configs()
    )


def _pair_tokens(config):
    encoded = {key: repr(config[key]) for key in _MODE_KEYS}
    return frozenset(
        (left, encoded[left], right, encoded[right]) for left, right in combinations(_MODE_KEYS, 2)
    )


def _minimal_cover(candidates, token_fn):
    covers = [token_fn(mode) for mode in candidates]
    token_users = {}
    for index, tokens in enumerate(covers):
        for token in tokens:
            token_users.setdefault(token, []).append(index)
    uncovered = set(token_users)
    scores = [len(tokens) for tokens in covers]
    labels = [mode.get("label", _mode_label(mode)) for mode in candidates]
    heap = [(-score, labels[index], index) for index, score in enumerate(scores)]
    heapq.heapify(heap)
    selected = []
    while uncovered:
        while True:
            neg_score, _label, best = heapq.heappop(heap)
            if -neg_score == scores[best]:
                break
        if scores[best] == 0:
            raise AssertionError("structural coverage did not converge")
        selected.append(best)
        newly_covered = covers[best] & uncovered
        uncovered.difference_update(newly_covered)
        for token in newly_covered:
            for index in token_users[token]:
                scores[index] -= 1
                heapq.heappush(heap, (-scores[index], labels[index], index))
    counts = {}
    for index in selected:
        for token in covers[index]:
            counts[token] = counts.get(token, 0) + 1
    kept = []
    for index in reversed(selected):
        if all(counts[token] > 1 for token in covers[index]):
            for token in covers[index]:
                counts[token] -= 1
        else:
            kept.append(index)
    return [candidates[index] for index in reversed(kept)]


def _stage_parameters(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cta_group = tile_m // 128
    ab_bits = _dtype_bits(mode["ab_dtype"])
    c_bits = _dtype_bits(mode["c_dtype"])
    d_bits = _dtype_bits(mode["d_dtype"])
    k_tile = 256 if ab_bits == 4 else 128
    acc_stages = 1 if tile_n == 256 else 2
    a_stage = 128 * k_tile * ab_bits // 8
    b_stage = (tile_n // cta_group) * k_tile * ab_bits // 8
    sfa_stage = 128 * k_tile // mode["sf_vec_size"]
    sfb_stage = _align_up(tile_n, 128) * k_tile // mode["sf_vec_size"]
    c_stage = 128 * 32 * c_bits // 8
    d_stage = 128 * 32 * d_bits // 8
    ab_stage = a_stage + b_stage + sfa_stage + sfb_stage
    ab_stages = (_SMEM_CAPACITY - (1024 + 2 * c_stage + 2 * d_stage + 16)) // ab_stage
    c_offset = 1024
    d_offset = c_offset + 2 * c_stage
    a_offset = d_offset + 2 * d_stage
    b_offset = a_offset + ab_stages * a_stage
    sfa_offset = b_offset + ab_stages * b_stage
    sfb_offset = _align_up(sfa_offset + ab_stages * sfa_stage, 1024)
    amax_offset = _align_up(sfb_offset + ab_stages * sfb_stage, 1024)
    shared_bytes = _align_up(amax_offset + 16, 1024)
    return {
        "cta_group": cta_group,
        "k_tile": k_tile,
        "acc_stages": acc_stages,
        "ab_stages": ab_stages,
        "c_stages": 2,
        "d_stages": 2,
        "shared_bytes": shared_bytes,
    }


def _correctness_tokens(config):
    stages = _stage_parameters(config)
    tokens = set(_pair_tokens(config))
    tokens.add(
        (
            "stage_tuple",
            stages["acc_stages"],
            stages["ab_stages"],
            stages["c_stages"],
            stages["d_stages"],
            stages["shared_bytes"],
        )
    )
    source_probe = config["source_probe"]
    if source_probe == "anchor" or source_probe.startswith(("tail_", "persistent_")):
        tokens.add(("required_guard", source_probe))
    return frozenset(tokens)


def _correctness_configs():
    # Pairwise values, every stage tuple, and the frozen anchor/tail/persistent
    # guards are selected together, followed by reverse redundancy pruning.
    selected = _minimal_cover(_verified_configs(), _correctness_tokens)
    configs = []
    for selected_config in selected:
        config = dict(selected_config)
        config.pop("source_probe")
        configs.append(config)
    return sorted(configs, key=lambda config: config["label"])


def _performance_tokens(mode):
    stages = _stage_parameters(mode)
    return frozenset(
        {
            ("ab", mode["ab_dtype"]),
            ("sf", mode["sf_dtype"], mode["sf_vec_size"]),
            ("c", mode["c_dtype"]),
            ("d", mode["d_dtype"]),
            ("a_major", mode["a_major"]),
            ("b_major", mode["b_major"]),
            ("c_major", mode["c_major"]),
            ("cta_group", stages["cta_group"]),
            ("tile_n", mode["mma_tiler_mn"][1]),
            ("cluster", mode["cluster_shape_mn"]),
            ("mma", mode["ab_dtype"], mode["sf_dtype"], mode["sf_vec_size"]),
            ("store", mode["c_dtype"], mode["d_dtype"]),
            (
                "stages",
                stages["acc_stages"],
                stages["ab_stages"],
                stages["c_stages"],
                stages["d_stages"],
            ),
            ("amax", True),
        }
    )


def _benchmark_configs():
    # The independent PTX export proved vector_f32=true/false byte-identical,
    # so performance coverage keeps one representative (true) while CONFIGS
    # retain both correctness probes.
    unique_modes = {}
    for config in _verified_configs():
        if not config["vector_f32"]:
            continue
        key = tuple(config[name] for name in _MODE_KEYS if name != "L")
        unique_modes.setdefault(key, config)
    base_modes = tuple(sorted(unique_modes.values(), key=lambda config: config["label"]))
    representatives = _minimal_cover(base_modes, _performance_tokens)
    # Freeze one compact, deterministic instance roster after structural
    # representative selection. Every one of the 23 representatives remains
    # present, while seven extra rows cover small/large and L=1/L=2 boundaries.
    instances = {
        0: ((4096, 1), (8192, 1)),
        1: ((8192, 1),),
        2: ((1024, 1), (8192, 1)),
        3: ((1024, 1), (2048, 2), (4096, 1), (4096, 2), (8192, 2)),
        4: ((8192, 1),),
        5: ((4096, 2),),
        6: ((4096, 1),),
        7: ((1024, 2),),
        8: ((4096, 2),),
        9: ((8192, 1),),
        10: ((8192, 2),),
        11: ((8192, 2),),
        12: ((4096, 2),),
        13: ((4096, 1),),
        14: ((8192, 2),),
        15: ((4096, 2),),
        16: ((4096, 2),),
        17: ((2048, 2),),
        18: ((4096, 1),),
        19: ((4096, 1),),
        20: ((4096, 2),),
        21: ((4096, 2),),
        22: ((4096, 1), (8192, 2)),
    }
    assert len(representatives) == len(instances) == 23
    configs = []
    for index, base_mode in enumerate(representatives):
        for size, batch in instances[index]:
            mode = {**base_mode, "M": size, "N": size, "K": size, "L": batch}
            mode["label"] = f"perf{index:02d}_m{size}_n{size}_k{size}_l{batch}_{_mode_label(mode)}"
            mode.pop("source_probe")
            configs.append(mode)
    assert len(configs) == 30
    return sorted(configs, key=lambda config: config["label"])


KERNEL_META = {
    "name": "cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "nvidia-cudnn-frontend",
            "git": {
                "url": "https://github.com/NVIDIA/cudnn-frontend.git",
                "commit": "aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5",
            },
            "import": "cudnn",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}

CONFIGS = _correctness_configs()
BENCH_CONFIGS = _benchmark_configs()


@cache
def _make_kernel(
    M,
    N,
    K_dim,
    L,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    c_dtype,
    d_dtype,
    a_major,
    b_major,
    c_major,
    mma_tiler_mn,
    cluster_shape_mn,
    vector_f32,
):
    if ab_dtype == "uint8":
        raise ValueError("the non-running uint8 packed-FP4 alias is not supported")
    if sf_dtype == "int8":
        sf_dtype = "float8_e8m0fnu"
    mma_tiler_mn = tuple(mma_tiler_mn)
    cluster_shape_mn = tuple(cluster_shape_mn)
    mode = {
        "ab_dtype": ab_dtype,
        "sf_dtype": sf_dtype,
        "sf_vec_size": sf_vec_size,
        "c_dtype": c_dtype,
        "d_dtype": d_dtype,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "vector_f32": vector_f32,
        "L": L,
    }
    if min(M, N, K_dim, L) <= 0:
        raise ValueError("M/N/K/L must be positive")
    if not _valid_mode(mode):
        raise ValueError(f"unsupported source specialization: {_mode_label(mode)}")
    if not _verified_structure(mode):
        raise ValueError(
            f"source specialization was not independently verified: {_mode_label(mode)}"
        )
    ab_bits = _dtype_bits(ab_dtype)
    c_bits = _dtype_bits(c_dtype)
    d_bits = _dtype_bits(d_dtype)
    if (M if a_major == "m" else K_dim) % (128 // ab_bits):
        raise ValueError("A's contiguous dimension must be 16-byte aligned")
    if (N if b_major == "n" else K_dim) % (128 // ab_bits):
        raise ValueError("B's contiguous dimension must be 16-byte aligned")
    if (M if c_major == "m" else N) % (128 // c_bits):
        raise ValueError("C's contiguous dimension must be 16-byte aligned")
    if (M if c_major == "m" else N) % (128 // d_bits):
        raise ValueError("D's contiguous dimension must be 16-byte aligned")

    tile_m, n_tile = mma_tiler_mn
    cta_group = tile_m // 128
    cta_m = 128
    b_rows = n_tile // cta_group
    cluster_m, cluster_n = cluster_shape_mn
    cluster_size = cluster_m * cluster_n
    cluster_m_groups = cluster_m // cta_group
    k_tile = 256 if ab_dtype == "float4_e2m1fn" else 128
    k_tiles = _ceil_div(K_dim, k_tile)
    m_tiles = _ceil_div(M, cta_m)
    n_tiles = _ceil_div(N, n_tile)
    cluster_m_tiles = _ceil_div(m_tiles, cluster_m)
    cluster_n_tiles = _ceil_div(n_tiles, cluster_n)
    cluster_work = cluster_m_tiles * cluster_n_tiles * L
    num_clusters = min(cluster_work, _MAX_ACTIVE_CLUSTERS[cluster_size])
    acc_stages = 1 if n_tile == 256 else 2
    c_stages = 2
    d_stages = 2
    epi_n = 32
    epilogue_subtiles = n_tile // epi_n
    a_stage_bytes = cta_m * k_tile * ab_bits // 8
    b_stage_bytes = b_rows * k_tile * ab_bits // 8
    sfa_stage_bytes = cta_m * k_tile // sf_vec_size
    # The source SFB layout is padded to a whole 128-column scale tile.
    # This padding participates in both SMEM allocation and mbarrier bytes.
    sfb_stage_bytes = _align_up(n_tile, 128) * k_tile // sf_vec_size
    ab_stage_bytes = a_stage_bytes + b_stage_bytes + sfa_stage_bytes + sfb_stage_bytes
    c_stage_bytes = cta_m * epi_n * c_bits // 8
    d_stage_bytes = cta_m * epi_n * d_bits // 8
    ab_stages = (
        _SMEM_CAPACITY - (1024 + c_stages * c_stage_bytes + d_stages * d_stage_bytes + 16)
    ) // ab_stage_bytes
    if ab_stages <= 0:
        raise ValueError("source stage heuristic exceeds SM100 shared-memory capacity")

    ab_full_offset = 0
    ab_empty_offset = ab_full_offset + ab_stages * 8
    acc_full_offset = ab_empty_offset + ab_stages * 8
    acc_empty_offset = acc_full_offset + acc_stages * 8
    tmem_dealloc_offset = acc_empty_offset + acc_stages * 8
    tmem_ptr_offset = tmem_dealloc_offset + 8
    c_offset = 1024
    d_offset = c_offset + c_stages * c_stage_bytes
    a_offset = d_offset + d_stages * d_stage_bytes
    b_offset = a_offset + ab_stages * a_stage_bytes
    sfa_offset = b_offset + ab_stages * b_stage_bytes
    sfb_offset = _align_up(sfa_offset + ab_stages * sfa_stage_bytes, 1024)
    sfb_storage_stage_bytes = sfb_stage_bytes
    amax_offset = _align_up(sfb_offset + ab_stages * sfb_storage_stage_bytes, 1024)
    shared_bytes = _align_up(amax_offset + 16, 1024)
    if shared_bytes > _SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {shared_bytes} exceeds {_SMEM_CAPACITY}")

    ab_empty_arrivals = cluster_n + cluster_m_groups - 1
    acc_empty_arrivals = 4 * cta_group
    tmem_columns = 512
    tma_cache_hint = 0
    entry_max_registers = None

    def mn_smem_parameters(major_size):
        major_bits = major_size * ab_bits
        if ab_bits == 32 and major_bits % 1024 == 0:
            # MN_SW128_32B: S<2, 5, 2>, exposed to TMA as the
            # 128-byte-atom/32-byte-base swizzle.
            swizzle, chunk, sdo = 4, 32, 32
        elif major_bits % 1024 == 0:
            swizzle, chunk, sdo = 3, 1024 // ab_bits, 64
        elif major_bits % 512 == 0:
            swizzle, chunk, sdo = 2, 512 // ab_bits, 32
        elif major_bits % 256 == 0:
            swizzle, chunk, sdo = 1, 256 // ab_bits, 16
        else:
            raise ValueError("MN-major input tile must contain a 32-byte swizzle atom")
        copies = major_size // chunk
        ldo = 0 if copies == 1 else chunk * k_tile * (ab_bits // 8) // 16
        return swizzle, chunk, copies, ldo, sdo

    if a_major == "m":
        a_swizzle, a_chunk, a_copies, a_ldo, a_sdo = mn_smem_parameters(cta_m)
    else:
        a_swizzle, a_chunk, a_copies, a_ldo, a_sdo = 3, cta_m, 1, 1, 64
    if b_major == "n":
        b_swizzle, b_chunk, b_copies, b_ldo, b_sdo = mn_smem_parameters(b_rows)
    else:
        b_swizzle, b_chunk, b_copies, b_ldo, b_sdo = 3, b_rows, 1, 1, 64
    a_cluster_piece = (k_tile if a_major == "m" else cta_m) // cluster_n
    b_cluster_piece = (k_tile if b_major == "n" else b_rows) // cluster_m_groups
    a_piece_bytes = a_stage_bytes // (a_copies * cluster_n)
    b_piece_bytes = b_stage_bytes // (b_copies * cluster_m_groups)
    num_tma_load_bytes = ab_stage_bytes * cta_group

    a_desc_base = _descriptor_base(ldo=a_ldo, sdo=a_sdo, swizzle=a_swizzle)
    b_desc_base = _descriptor_base(ldo=b_ldo, sdo=b_sdo, swizzle=b_swizzle)
    sf_desc_base = _descriptor_base(ldo=1, sdo=8, swizzle=0)
    instr_desc = _instruction_descriptor(tile_m, n_tile, ab_dtype, sf_dtype, a_major, b_major)
    acc_columns = acc_stages * n_tile
    sfa_tmem_column = acc_columns
    sfa_chunks = sfa_stage_bytes // 512
    sfb_chunks = sfb_stage_bytes // 512
    sfb_tmem_column = sfa_tmem_column + sfa_chunks * 4
    if sfb_tmem_column + sfb_chunks * 4 > 512:
        raise ValueError("source TMEM scale regions exceed 512 columns")
    sf_k_box = k_tile // (4 * sf_vec_size)
    sfa_piece_values = 256 * sf_k_box // cluster_n
    sfb_n_box = _ceil_div(n_tile, 128)
    sfb_piece_values = 256 * sf_k_box * sfb_n_box // cluster_m_groups

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        sfa = params["sfa"]
        sfb = params["sfb"]
        c = params["c"]
        d = params["d"]
        a_map = K.stack_alloca("tensormap", 1)
        b_map = K.stack_alloca("tensormap", 1)
        sfa_map = K.stack_alloca("tensormap", 1)
        sfb_map = K.stack_alloca("tensormap", 1)
        c_map = K.stack_alloca("tensormap", 1)
        d_map = K.stack_alloca("tensormap", 1)

        def encode(descriptor, dtype, rank, data, *fields):
            K.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

        a_contiguous_bytes = (M if a_major == "m" else K_dim) * ab_bits // 8
        b_contiguous_bytes = (N if b_major == "n" else K_dim) * ab_bits // 8
        if a_major == "m":
            a_fields = (
                M,
                K_dim,
                L,
                a_contiguous_bytes,
                M * K_dim * ab_bits // 8,
                a_chunk,
                a_cluster_piece,
                1,
            )
        else:
            a_fields = (
                K_dim,
                M,
                L,
                a_contiguous_bytes,
                M * K_dim * ab_bits // 8,
                k_tile,
                a_cluster_piece,
                1,
            )
        if b_major == "n":
            b_fields = (
                N,
                K_dim,
                L,
                b_contiguous_bytes,
                N * K_dim * ab_bits // 8,
                b_chunk,
                b_cluster_piece,
                1,
            )
        else:
            b_fields = (
                K_dim,
                N,
                L,
                b_contiguous_bytes,
                N * K_dim * ab_bits // 8,
                k_tile,
                b_cluster_piece,
                1,
            )
        tensor_tail = (1, 1, 1, 0, a_swizzle, 2, 0)
        if ab_dtype == "float4_e2m1fn":
            tensor_tail += (13,)
        encode(a_map, ab_dtype, 3, a.data, *a_fields, *tensor_tail)
        tensor_tail = (1, 1, 1, 0, b_swizzle, 2, 0)
        if ab_dtype == "float4_e2m1fn":
            tensor_tail += (13,)
        encode(b_map, ab_dtype, 3, b.data, *b_fields, *tensor_tail)

        sf_k_groups = _ceil_div(K_dim, 4 * sf_vec_size)
        sf_m_groups = _ceil_div(M, 128)
        sf_n_groups = _ceil_div(N, 128)
        sfa_box_0 = min(256, sfa_piece_values)
        sfa_box_1 = sfa_piece_values // sfa_box_0
        encode(
            sfa_map,
            "uint16",
            4,
            sfa.data,
            256,
            sf_k_groups,
            sf_m_groups,
            L,
            512,
            sf_k_groups * 512,
            sf_m_groups * sf_k_groups * 512,
            sfa_box_0,
            sfa_box_1,
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
        sfb_box_0 = min(256, sfb_piece_values)
        sfb_remaining = sfb_piece_values // sfb_box_0
        sfb_box_1 = min(sf_k_box, sfb_remaining)
        sfb_box_2 = sfb_remaining // sfb_box_1
        encode(
            sfb_map,
            "uint16",
            4,
            sfb.data,
            256,
            sf_k_groups,
            sf_n_groups,
            L,
            512,
            sf_k_groups * 512,
            sf_n_groups * sf_k_groups * 512,
            sfb_box_0,
            sfb_box_1,
            sfb_box_2,
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

        def encode_output(descriptor, tensor, dtype, bits):
            element_bytes = bits // 8
            contiguous_bytes = N * element_bytes
            batch_bytes = M * N * element_bytes
            fields = (N, M, L, contiguous_bytes, batch_bytes, epi_n, cta_m, 1)
            row_bytes = epi_n * element_bytes
            swizzle = 3 if row_bytes >= 128 else 2 if row_bytes >= 64 else 1
            encode(descriptor, dtype, 3, tensor.data, *fields, 1, 1, 1, 0, swizzle, 2, 0)

        encode_output(c_map, c, c_dtype, c_bits)
        encode_output(d_map, d, d_dtype, d_bits)
        return a_map, b_map, sfa_map, sfb_map, c_map, d_map

    def kernel(a, b, sfa, sfb, c, d, prob, amax, alpha, *, host):
        del a, b, sfa, sfb, c, d
        a_map, b_map, sfa_map, sfb_map, c_map, d_map = host
        if entry_max_registers is None:
            _block_x, _block_y, cluster_work_id = K.cta_id()
        else:
            with K.attr({"tirx.max_registers": entry_max_registers}):
                _block_x, _block_y, cluster_work_id = K.cta_id()
        cluster_x_scope, cluster_y_scope = K.cta_id_in_cluster(
            [cluster_m, cluster_n], preferred=[cluster_m, cluster_n]
        )
        del _block_x, _block_y, cluster_x_scope, cluster_y_scope
        cluster_rank = K.local_scalar("int32", init=K.cuda.mov_sreg(32, "cluster_ctarank"))
        cluster_x = K.local_scalar("int32", init=cluster_rank & (cluster_m - 1))
        cluster_y = K.local_scalar("int32", init=cluster_rank >> (cluster_m.bit_length() - 1))
        cta_v = cluster_x % cta_group
        leader_cta = cta_v == 0
        cluster_m_group = cluster_x // cta_group
        pair_leader_x = cluster_m_group * cta_group
        leader_rank = pair_leader_x + cluster_m * cluster_y
        warp = K.warp_id()
        lane = K.lane_id()

        roles = K.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])

        smem = K.alloc_buffer((shared_bytes,), K.u8, scope="shared.dyn", align=1024)
        protocol_pool = K.smem_pool(base=smem)
        ab_pipe = K.Pipeline(
            protocol_pool,
            ab_stages,
            full="tma",
            empty="tcgen05",
            init_empty=ab_empty_arrivals,
            leader=K.bool(False),
        )
        acc_pipe = K.Pipeline(
            protocol_pool,
            acc_stages,
            full="tcgen05",
            empty="mbar",
            init_empty=acc_empty_arrivals,
            leader=K.bool(False),
        )
        if protocol_pool.bytes != tmem_dealloc_offset:
            raise AssertionError("protocol storage offsets changed")
        tmem_dealloc = protocol_pool.alloc((1,), K.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), K.u32, align=4)
        if protocol_pool.bytes != tmem_ptr_offset + 4:
            raise AssertionError("protocol storage header changed")

        with tma_role:
            K.ptx.prefetch.tensormap(K.address_of(a_map))
            K.ptx.prefetch.tensormap(K.address_of(b_map))
            K.ptx.prefetch.tensormap(K.address_of(sfa_map))
            K.ptx.prefetch.tensormap(K.address_of(sfb_map))
            K.ptx.prefetch.tensormap(K.address_of(c_map))
            K.ptx.prefetch.tensormap(K.address_of(d_map))

        with K.If(warp == 0):
            with K.Then():
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, ab_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, ab_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), K.uint32(ab_empty_arrivals)
                            )
        with K.If(warp == 0):
            with K.Then():
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, acc_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, acc_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), K.uint32(acc_empty_arrivals)
                            )
        if cta_group == 2:
            with K.If(warp == 0):
                with K.Then():
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx.mbarrier.init.shared.b64(tmem_dealloc.ptr_to([0]), K.uint32(32))
            K.ptx.fence.mbarrier_init.release.cluster()

        K.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            K.ptx.barrier.cluster.arrive.relaxed()

        smem_base = K.local_scalar("uint32")
        K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        tmem_slot_addr = K.uniform(smem_base + K.uint32(tmem_ptr_offset))
        cluster_smem_u64 = K.local_scalar("uint64")
        K.ptx.cvta.to.shared__cluster.u64(cluster_smem_u64, smem.ptr_to([0]))
        cluster_smem = K.local_scalar("uint32", init=K.cast(cluster_smem_u64, "uint32"))
        a_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + a_offset)
        )
        b_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + b_offset)
        )
        sfa_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + sfa_offset)
        )
        sfb_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + sfb_offset)
        )

        a_mcast_mask = K.local_scalar("uint32", init=K.uint32(0))
        for peer_n in range(cluster_n):
            K.assign(
                a_mcast_mask,
                K.bitwise_or(
                    a_mcast_mask, K.uint32(1) << K.cast(cluster_x + cluster_m * peer_n, "uint32")
                ),
            )
        b_mcast_mask = K.local_scalar("uint32", init=K.uint32(0))
        for peer_group in range(cluster_m_groups):
            peer_x = cta_v + cta_group * peer_group
            K.assign(
                b_mcast_mask,
                K.bitwise_or(
                    b_mcast_mask, K.uint32(1) << K.cast(peer_x + cluster_m * cluster_y, "uint32")
                ),
            )
        ab_consumer_mask = K.local_scalar("uint32", init=K.uint32(0))
        for pair_v in range(cta_group):
            for peer_n in range(cluster_n):
                K.assign(
                    ab_consumer_mask,
                    K.bitwise_or(
                        ab_consumer_mask,
                        K.uint32(1)
                        << K.cast(pair_leader_x + pair_v + cluster_m * peer_n, "uint32"),
                    ),
                )
            for peer_group in range(cluster_m_groups):
                peer_x = pair_v + cta_group * peer_group
                K.assign(
                    ab_consumer_mask,
                    K.bitwise_or(
                        ab_consumer_mask,
                        K.uint32(1) << K.cast(peer_x + cluster_m * cluster_y, "uint32"),
                    ),
                )
        acc_producer_mask = K.local_scalar(
            "uint32", init=K.uint32((1 << cta_group) - 1) << K.cast(leader_rank, "uint32")
        )
        ab_full_leader = ab_pipe.full.remote_view(leader_rank)
        acc_empty_leader = acc_pipe.empty.remote_view(leader_rank)

        if cluster_size > 1:
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0), K.uint32(192))

        def scheduler_coords(work):
            cluster_m_idx = work % cluster_m_tiles
            quotient = work // cluster_m_tiles
            cluster_n_idx = quotient % cluster_n_tiles
            batch_idx = quotient // cluster_n_tiles
            tile_m_idx = cluster_m_idx * cluster_m + cluster_x
            tile_n_idx = cluster_n_idx * cluster_n + cluster_y
            return tile_m_idx, tile_n_idx, batch_idx

        def advance_work(work):
            K.assign(work, work + num_clusters)

        # TMEM allocation is published through one physical 160-thread
        # barrier instruction shared by the four epilogue warps and the MMA
        # warp.  Keeping the arrival at one PC is required by synccheck and is
        # equivalent to the source's named-barrier-2 rendezvous.
        with K.If(warp < K.uint32(5)):
            with K.Then():
                with K.If(warp == 0):
                    with K.Then():
                        K.ptx[
                            "tcgen05.alloc.cta_group::"
                            + str(cta_group)
                            + ".sync.aligned.shared::cta.b32"
                        ](tmem_slot_addr, K.uint32(tmem_columns))
                K.ptx.bar.sync(K.uint32(2), K.uint32(160))

        with tma_role:
            tma_state = K.PipelineState(ab_stages, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            count = K.local_scalar("int32")
            speculative = K.local_scalar("uint32")
            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                K.assign(count, 0)
                K.assign(speculative, K.uint32(1))
                with K.If(count < k_tiles):
                    with K.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase
                        )
                with K.While(count < k_tiles):
                    _wait_plain_if_needed(
                        ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase, speculative
                    )
                    if k_tiles:
                        if ab_stages:
                            with K.If(leader_cta):
                                with K.Then():
                                    with K.If(_elected()):
                                        with K.Then():
                                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                                ab_pipe.full.ptr_to([tma_state.stage]),
                                                K.uint32(num_tma_load_bytes),
                                            )

                            a_elected = _elected()
                            for copy in range(a_copies):
                                a_coord_m = tile_m_idx * cta_m
                                a_coord_k = count * k_tile
                                if a_major == "m":
                                    a_coord_m = a_coord_m + copy * a_chunk
                                    a_coord_k = a_coord_k + cluster_y * a_cluster_piece
                                else:
                                    a_coord_m = a_coord_m + cluster_y * a_cluster_piece
                                a_coord_0 = a_coord_m if a_major == "m" else a_coord_k
                                a_coord_1 = a_coord_k if a_major == "m" else a_coord_m
                                a_smem_offset = (
                                    a_offset
                                    + tma_state.stage * a_stage_bytes
                                    + copy * (a_stage_bytes // a_copies)
                                    + cluster_y * a_piece_bytes
                                )
                                if cta_group == 1 and cluster_n == 1:
                                    _issue_ptx_if(
                                        a_elected,
                                        "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint",
                                        smem.ptr_to([a_smem_offset]),
                                        K.address_of(a_map),
                                        K.cast(a_coord_0, "int32"),
                                        K.cast(a_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                elif cta_group == 1:
                                    _issue_ptx_if(
                                        a_elected,
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint",
                                        cluster_smem + a_smem_offset,
                                        K.address_of(a_map),
                                        K.cast(a_coord_0, "int32"),
                                        K.cast(a_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.cast(a_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )
                                elif cluster_n == 1:
                                    _issue_ptx_if(
                                        a_elected,
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2",
                                        cluster_smem + a_smem_offset,
                                        K.address_of(a_map),
                                        K.cast(a_coord_0, "int32"),
                                        K.cast(a_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_full_leader.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    _issue_ptx_if(
                                        a_elected,
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2",
                                        cluster_smem + a_smem_offset,
                                        K.address_of(a_map),
                                        K.cast(a_coord_0, "int32"),
                                        K.cast(a_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_full_leader.ptr_to([tma_state.stage]),
                                        K.cast(a_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )

                            b_elected = _elected()
                            for copy in range(b_copies):
                                b_coord_n = tile_n_idx * n_tile + cta_v * b_rows
                                b_coord_k = count * k_tile
                                if b_major == "n":
                                    b_coord_n = b_coord_n + copy * b_chunk
                                    b_coord_k = b_coord_k + cluster_m_group * b_cluster_piece
                                else:
                                    b_coord_n = b_coord_n + cluster_m_group * b_cluster_piece
                                b_coord_0 = b_coord_n if b_major == "n" else b_coord_k
                                b_coord_1 = b_coord_k if b_major == "n" else b_coord_n
                                b_smem_offset = (
                                    b_offset
                                    + tma_state.stage * b_stage_bytes
                                    + copy * (b_stage_bytes // b_copies)
                                    + cluster_m_group * b_piece_bytes
                                )
                                if cta_group == 1 and cluster_m_groups == 1:
                                    _issue_ptx_if(
                                        b_elected,
                                        "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint",
                                        smem.ptr_to([b_smem_offset]),
                                        K.address_of(b_map),
                                        K.cast(b_coord_0, "int32"),
                                        K.cast(b_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                elif cta_group == 1:
                                    _issue_ptx_if(
                                        b_elected,
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint",
                                        cluster_smem + b_smem_offset,
                                        K.address_of(b_map),
                                        K.cast(b_coord_0, "int32"),
                                        K.cast(b_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.cast(b_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )
                                elif cluster_m_groups == 1:
                                    _issue_ptx_if(
                                        b_elected,
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2",
                                        cluster_smem + b_smem_offset,
                                        K.address_of(b_map),
                                        K.cast(b_coord_0, "int32"),
                                        K.cast(b_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_full_leader.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    _issue_ptx_if(
                                        b_elected,
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2",
                                        cluster_smem + b_smem_offset,
                                        K.address_of(b_map),
                                        K.cast(b_coord_0, "int32"),
                                        K.cast(b_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_full_leader.ptr_to([tma_state.stage]),
                                        K.cast(b_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )

                            sfa_linear = cluster_y * sfa_piece_values
                            sfa_coord_0 = sfa_linear % 256
                            sfa_coord_1 = count * sf_k_box + sfa_linear // 256
                            sfa_smem_offset = (
                                sfa_offset
                                + tma_state.stage * sfa_stage_bytes
                                + cluster_y * (sfa_stage_bytes // cluster_n)
                            )
                            sfa_elected = _elected()
                            if cta_group == 1 and cluster_n == 1:
                                _issue_ptx_if(
                                    sfa_elected,
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint",
                                    smem.ptr_to([sfa_smem_offset]),
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            elif cta_group == 1:
                                _issue_ptx_if(
                                    sfa_elected,
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint",
                                    cluster_smem + sfa_smem_offset,
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(a_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                            elif cluster_n == 1:
                                _issue_ptx_if(
                                    sfa_elected,
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2",
                                    cluster_smem + sfa_smem_offset,
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                _issue_ptx_if(
                                    sfa_elected,
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2",
                                    cluster_smem + sfa_smem_offset,
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    K.cast(a_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )

                            sfb_linear = cluster_m_group * sfb_piece_values
                            sfb_coord_0 = sfb_linear % 256
                            sfb_quotient = sfb_linear // 256
                            sfb_coord_1 = count * sf_k_box + sfb_quotient % sf_k_box
                            # SFB is globally blocked in 128-column units.  N64
                            # and N192 tiles can start halfway through a scale
                            # block; the TMEM shift below selects that half.
                            sfb_global_block = tile_n_idx * n_tile // 128
                            sfb_coord_2 = sfb_global_block + sfb_quotient // sf_k_box
                            sfb_smem_offset = (
                                sfb_offset
                                + tma_state.stage * sfb_stage_bytes
                                + cluster_m_group * (sfb_stage_bytes // cluster_m_groups)
                            )
                            sfb_elected = _elected()
                            if cta_group == 1 and cluster_m_groups == 1:
                                _issue_ptx_if(
                                    sfb_elected,
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint",
                                    smem.ptr_to([sfb_smem_offset]),
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            elif cta_group == 1:
                                _issue_ptx_if(
                                    sfb_elected,
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint",
                                    cluster_smem + sfb_smem_offset,
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(b_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                            elif cluster_m_groups == 1:
                                _issue_ptx_if(
                                    sfb_elected,
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2",
                                    cluster_smem + sfb_smem_offset,
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                _issue_ptx_if(
                                    sfb_elected,
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2",
                                    cluster_smem + sfb_smem_offset,
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    K.cast(b_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                    _advance(tma_state)
                    K.assign(count, count + 1)
                    K.assign(speculative, K.uint32(1))
                    with K.If(count < k_tiles):
                        with K.Then():
                            _try_wait_acquire(
                                speculative,
                                ab_pipe.empty.ptr_to([tma_state.stage]),
                                tma_state.phase,
                            )
                advance_work(work)
            with K.unroll(0, ab_stages) as unused_stage:
                _wait_plain(ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase)
                _advance(tma_state)
            del unused_stage

        with mma_role:
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            mma_state = K.PipelineState(ab_stages, phase=0)
            acc_state = K.PipelineState(acc_stages, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            count = K.local_scalar("int32")
            speculative = K.local_scalar("uint32")
            accumulate = K.local_scalar("uint32")
            with K.While(work < cluster_work):
                _tile_m_idx, tile_n_idx, _batch_idx = scheduler_coords(work)
                K.assign(count, 0)
                K.assign(speculative, K.uint32(1))
                with K.If((count < k_tiles) & leader_cta):
                    with K.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                        )
                with K.If(leader_cta):
                    with K.Then():
                        _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)
                K.assign(accumulate, K.uint32(0))
                with K.While(count < k_tiles):
                    with K.If(leader_cta):
                        with K.Then():
                            _wait_plain_if_needed(
                                ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase, speculative
                            )
                            for sf_chunk in range(sfa_chunks):
                                with K.If(_elected()):
                                    with K.Then():
                                        K.ptx[
                                            "tcgen05.cp.cta_group::"
                                            + str(cta_group)
                                            + ".32x128b.warpx4"
                                        ](
                                            K.cast(
                                                tmem_base + sfa_tmem_column + sf_chunk * 4, "uint32"
                                            ),
                                            sfa_descriptor
                                            + K.cast(
                                                mma_state.stage * (sfa_stage_bytes // 16)
                                                + sf_chunk * 32,
                                                "uint64",
                                            ),
                                        )
                            for sf_chunk in range(sfb_chunks):
                                sfb_shared_chunk = (
                                    sf_chunk % sfb_n_box
                                ) * sf_k_box + sf_chunk // sfb_n_box
                                with K.If(_elected()):
                                    with K.Then():
                                        K.ptx[
                                            "tcgen05.cp.cta_group::"
                                            + str(cta_group)
                                            + ".32x128b.warpx4"
                                        ](
                                            K.cast(
                                                tmem_base + sfb_tmem_column + sf_chunk * 4, "uint32"
                                            ),
                                            sfb_descriptor
                                            + K.cast(
                                                mma_state.stage * (sfb_stage_bytes // 16)
                                                + sfb_shared_chunk * 32,
                                                "uint64",
                                            ),
                                        )
                            for kblock in range(4):
                                if sf_vec_size == 16:
                                    sfa_scale_offset = kblock * 4
                                    sfb_scale_offset = kblock * 4 * sfb_n_box
                                    sf_selector = 0
                                elif ab_dtype == "float4_e2m1fn":
                                    sfa_scale_offset = (kblock // 2) * 4
                                    sfb_scale_offset = (kblock // 2) * 4 * sfb_n_box
                                    sf_selector = (kblock % 2) * 0x80000000
                                else:
                                    sfa_scale_offset = 0
                                    sfb_scale_offset = 0
                                    sf_selector = kblock * 0x40000000
                                sfb_tile_shift = 0
                                if n_tile in (64, 192):
                                    sfb_tile_shift = (tile_n_idx % 2) * 2
                                sfa_tmem_addr = K.cast(
                                    tmem_base
                                    + sfa_tmem_column
                                    + sfa_scale_offset
                                    + K.uint32(sf_selector),
                                    "uint32",
                                )
                                sfb_tmem_addr = K.cast(
                                    tmem_base
                                    + sfb_tmem_column
                                    + sfb_tile_shift
                                    + sfb_scale_offset
                                    + K.uint32(sf_selector),
                                    "uint32",
                                )
                                runtime_instr_desc = K.bitwise_and(
                                    K.uint32(instr_desc), K.uint32(0x9FFFFFCF)
                                )
                                runtime_instr_desc = K.bitwise_or(
                                    runtime_instr_desc,
                                    K.bitwise_and(
                                        K.shift_right(sfa_tmem_addr, K.uint32(1)),
                                        K.uint32(0x60000000),
                                    ),
                                )
                                runtime_instr_desc = K.bitwise_or(
                                    runtime_instr_desc,
                                    K.bitwise_and(
                                        K.shift_right(sfb_tmem_addr, K.uint32(26)), K.uint32(0x30)
                                    ),
                                )
                                with K.If(_elected()):
                                    with K.Then():
                                        mma_kind = (
                                            "mxf4nvf4"
                                            if ab_dtype == "float4_e2m1fn"
                                            else "mxf8f6f4"
                                        )
                                        a_kblock = (
                                            2 if a_major == "k" else a_chunk * 32 * ab_bits // 128
                                        )
                                        b_kblock = (
                                            2 if b_major == "k" else b_chunk * 32 * ab_bits // 128
                                        )
                                        K.ptx[
                                            "tcgen05.mma.cta_group::"
                                            + str(cta_group)
                                            + ".kind::"
                                            + mma_kind
                                            + ".block_scale.block"
                                            + str(sf_vec_size)
                                        ](
                                            K.cast(tmem_base + acc_state.stage * n_tile, "uint32"),
                                            a_descriptor
                                            + K.cast(
                                                mma_state.stage * (a_stage_bytes // 16)
                                                + kblock * a_kblock,
                                                "uint64",
                                            ),
                                            b_descriptor
                                            + K.cast(
                                                mma_state.stage * (b_stage_bytes // 16)
                                                + kblock * b_kblock,
                                                "uint64",
                                            ),
                                            runtime_instr_desc,
                                            sfa_tmem_addr,
                                            sfb_tmem_addr,
                                            K.ptx.pred(K.cast(accumulate, "bool")),
                                        )
                                K.assign(accumulate, K.uint32(1))
                            with K.If(_elected()):
                                with K.Then():
                                    if cta_group == 2:
                                        K.ptx[
                                            "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                                            ".shared::cluster.multicast::cluster.b64"
                                        ](
                                            ab_pipe.empty.ptr_to([mma_state.stage]),
                                            K.cast(ab_consumer_mask, "uint16"),
                                        )
                                    elif cluster_size > 1:
                                        K.ptx[
                                            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                            ".shared::cluster.multicast::cluster.b64"
                                        ](
                                            ab_pipe.empty.ptr_to([mma_state.stage]),
                                            K.cast(ab_consumer_mask, "uint16"),
                                        )
                                    else:
                                        K.ptx[
                                            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                            ".shared::cluster.b64"
                                        ](ab_pipe.empty.ptr_to([mma_state.stage]))
                    _advance(mma_state)
                    K.assign(count, count + 1)
                    K.assign(speculative, K.uint32(1))
                    with K.If((count < k_tiles) & leader_cta):
                        with K.Then():
                            _try_wait_acquire(
                                speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                            )
                with K.If(leader_cta):
                    with K.Then():
                        with K.If(_elected()):
                            with K.Then():
                                if cta_group == 2:
                                    K.ptx[
                                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                                        ".shared::cluster.multicast::cluster.b64"
                                    ](
                                        acc_pipe.full.ptr_to([acc_state.stage]),
                                        K.cast(acc_producer_mask, "uint16"),
                                    )
                                elif cluster_size > 1:
                                    K.ptx[
                                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                        ".shared::cluster.multicast::cluster.b64"
                                    ](
                                        acc_pipe.full.ptr_to([acc_state.stage]),
                                        K.cast(acc_producer_mask, "uint16"),
                                    )
                                else:
                                    K.ptx[
                                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                        ".shared::cluster.b64"
                                    ](acc_pipe.full.ptr_to([acc_state.stage]))
                _advance(acc_state)
                advance_work(work)
            with K.If(leader_cta):
                with K.Then():
                    for _ in range(acc_stages - 1):
                        _advance(acc_state)
                    _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            acc_state = K.PipelineState(acc_stages, phase=0)
            work = K.local_scalar("int32", init=cluster_work_id)
            executed_tiles = K.local_scalar("int32", init=0)

            values = K.alloc_local((32,), "float32")
            absolute_values = K.alloc_local((32,), "float32")
            maxima = K.alloc_local((11,), "float32")
            c_words = K.alloc_local((32,), "uint32")
            d_words = K.alloc_local((32,), "uint32")
            c_pairs = K.alloc_local((16,), "uint16")
            d_pairs = K.alloc_local((16,), "uint16")
            c_wide = K.alloc_local((16,), "uint64")
            d_wide = K.alloc_local((16,), "uint64")
            c_offsets = K.alloc_local((8,), "int32")
            d_offsets = K.alloc_local((8,), "int32")
            tile_amax = K.local_scalar("float32")
            warp_amax = K.local_scalar("float32")
            prob_value = K.local_scalar("float32")

            def load_accumulator(subtile):
                tmem_load_addr = K.local_scalar(
                    "uint32",
                    init=tmem_base + (warp << 21) + acc_state.stage * n_tile + subtile * epi_n,
                )
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                    *[values[index] for index in range(32)], tmem_load_addr
                )

            def multiply_pair(dst0, dst1, left0, left1, right0, right1):
                packed = K.local_scalar("uint64")
                K.ptx.mul.rn.f32x2(
                    packed, K.cuda.make_float2(left0, left1), K.cuda.make_float2(right0, right1)
                )
                K.ptx.mov.b64(dst0, dst1, packed)

            def apply_alpha():
                for index in range(0, 32, 2):
                    multiply_pair(
                        values[index],
                        values[index + 1],
                        values[index],
                        values[index + 1],
                        alpha,
                        alpha,
                    )

            def pack_output(dtype, source_values, words, pairs, wide):
                if dtype == "float32":
                    for index in range(16):
                        K.assign(
                            wide[index],
                            K.cuda.make_float2(
                                source_values[index * 2], source_values[index * 2 + 1]
                            ),
                        )
                elif dtype in {"float16", "bfloat16"}:
                    for index in range(16):
                        if dtype == "float16":
                            K.ptx.cvt.rn.f16x2.f32(
                                words[index], source_values[index * 2 + 1], source_values[index * 2]
                            )
                        else:
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[index], source_values[index * 2 + 1], source_values[index * 2]
                            )
                else:
                    for index in range(16):
                        if dtype == "float8_e4m3fn":
                            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                pairs[index], source_values[index * 2 + 1], source_values[index * 2]
                            )
                        else:
                            K.ptx.cvt.rn.satfinite.e5m2x2.f32(
                                pairs[index], source_values[index * 2 + 1], source_values[index * 2]
                            )
                    for index in range(8):
                        K.assign(
                            words[index],
                            K.bitwise_or(
                                K.cast(pairs[index * 2], "uint32"),
                                K.shift_left(K.cast(pairs[index * 2 + 1], "uint32"), K.uint32(16)),
                            ),
                        )

            def stage_output(dtype, bits, words, wide, offsets, region_offset, stage, stage_bytes):
                row_bytes = epi_n * bits // 8
                word_count = epi_n * bits // 32
                vector_count = row_bytes // 16
                # The dynamic arena's fixed base preserves every admitted row
                # alignment, so apply the vector swizzle directly to the
                # arena-relative row offset.
                row_offset = K.local_scalar(
                    "uint32",
                    init=region_offset
                    + stage * stage_bytes
                    + warp * (32 * row_bytes)
                    + lane * row_bytes,
                )
                swizzle_vector = K.local_scalar(
                    "uint32",
                    init=K.bitwise_and(
                        K.shift_right(row_offset, K.uint32(7)), K.uint32(vector_count - 1)
                    ),
                )
                for vector in range(vector_count):
                    vector_offset = K.shift_left(
                        K.bitwise_xor(K.uint32(vector), swizzle_vector), K.uint32(4)
                    )
                    if dtype == "float32":
                        K.assign(
                            offsets[vector],
                            K.cast(K.bitwise_or(row_offset, vector_offset), "int32"),
                        )
                    else:
                        K.assign(
                            offsets[vector],
                            K.cast(K.bitwise_xor(row_offset, vector_offset), "int32"),
                        )
                if dtype == "float32":
                    for vector in range(vector_count):
                        K.ptx["st.shared.v2.b64"](
                            smem.ptr_to([offsets[vector]]), wide[vector * 2], wide[vector * 2 + 1]
                        )
                else:
                    for vector in range(word_count // 4):
                        K.ptx.st.shared.v4.b32(
                            smem.ptr_to([offsets[vector]]),
                            words[vector * 4],
                            words[vector * 4 + 1],
                            words[vector * 4 + 2],
                            words[vector * 4 + 3],
                        )

            def tma_store_output(map_ptr, region_offset, stage, stage_bytes, subtile):
                K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"](
                    K.address_of(map_ptr),
                    K.cast(tile_n_idx * n_tile + subtile * epi_n, "int32"),
                    K.cast(tile_m_idx * cta_m, "int32"),
                    K.cast(batch_idx, "int32"),
                    smem.ptr_to([region_offset + stage * stage_bytes]),
                    K.uint64(tma_cache_hint),
                )

            def apply_srelu_and_amax():
                for index in range(32):
                    K.ptx.max.f32(values[index], values[index], K.float32(0.0))
                for index in range(0, 32, 2):
                    multiply_pair(
                        values[index],
                        values[index + 1],
                        values[index],
                        values[index + 1],
                        values[index],
                        values[index + 1],
                    )
                for index in range(0, 32, 2):
                    multiply_pair(
                        values[index],
                        values[index + 1],
                        values[index],
                        values[index + 1],
                        prob_value,
                        prob_value,
                    )
                for index in range(32):
                    K.ptx.abs.f32(absolute_values[index], values[index])

                # Fixed 17-instruction max.NaN tree for the source's 32-value
                # vector reduction, followed by one non-NaN running fmax.
                for index in range(10):
                    K.ptx.max.NaN.f32(
                        maxima[index],
                        absolute_values[index * 3],
                        absolute_values[index * 3 + 1],
                        absolute_values[index * 3 + 2],
                    )
                K.ptx.max.NaN.f32(
                    maxima[10], absolute_values[30], absolute_values[31], K.float32(0.0)
                )
                for index in range(3):
                    K.ptx.max.NaN.f32(
                        maxima[index],
                        maxima[index * 3],
                        maxima[index * 3 + 1],
                        maxima[index * 3 + 2],
                    )
                K.ptx.max.NaN.f32(maxima[3], maxima[9], maxima[10], K.float32(0.0))
                K.ptx.max.NaN.f32(maxima[0], maxima[0], maxima[1], maxima[2])
                K.ptx.max.NaN.f32(tile_amax, maxima[0], maxima[3], tile_amax)

            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                K.cuda.mbarrier_wait(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                K.assign(tile_amax, K.float32(0.0))
                global_row = tile_m_idx * cta_m + warp * 32 + lane
                prob_offset = batch_idx * (m_tiles * cta_m) + global_row
                K.ptx.ld.global_.b32(prob_value, prob.ptr_to([prob_offset]))
                subtile = K.local_scalar("int32", init=0)
                with K.While(subtile < epilogue_subtiles):
                    linear_subtile = executed_tiles * epilogue_subtiles + subtile
                    c_stage = K.bitwise_and(linear_subtile, K.int32(1))
                    d_stage = K.bitwise_and(linear_subtile, K.int32(1))

                    load_accumulator(subtile)
                    apply_alpha()

                    # C is converted and stored before sReLU; the FP32 values
                    # remain live and are the sole input to D.
                    pack_output(c_dtype, values, c_words, c_pairs, c_wide)
                    stage_output(
                        c_dtype,
                        c_bits,
                        c_words,
                        c_wide,
                        c_offsets,
                        c_offset,
                        c_stage,
                        c_stage_bytes,
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    with K.If(warp == 0):
                        with K.Then():
                            tma_store_output(c_map, c_offset, c_stage, c_stage_bytes, subtile)
                            K.ptx.cp.async_.bulk.commit_group()
                            K.ptx.cp.async_.bulk.wait_group.read(c_stages - 1)
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))

                    apply_srelu_and_amax()
                    pack_output(d_dtype, values, d_words, d_pairs, d_wide)
                    stage_output(
                        d_dtype,
                        d_bits,
                        d_words,
                        d_wide,
                        d_offsets,
                        d_offset,
                        d_stage,
                        d_stage_bytes,
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    with K.If(warp == 0):
                        with K.Then():
                            tma_store_output(d_map, d_offset, d_stage, d_stage_bytes, subtile)
                            K.ptx.cp.async_.bulk.commit_group()
                            K.ptx.cp.async_.bulk.wait_group.read(d_stages - 1)
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    K.assign(subtile, subtile + 1)

                K.idioms.warp_reduce_max_nan_f32(warp_amax, tile_amax)
                with K.If(lane == 0):
                    with K.Then():
                        K.ptx.st.shared.b32(smem.ptr_to([amax_offset + warp * 4]), warp_amax)
                K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                with K.If((warp == 0) & (lane == 0)):
                    with K.Then():
                        amax_words = K.alloc_local((4,), "uint32", align=16)
                        K.ptx.ld.shared.v4.b32(
                            amax_words[0],
                            amax_words[1],
                            amax_words[2],
                            amax_words[3],
                            smem.ptr_to([amax_offset]),
                        )
                        block_amax = K.local_scalar("float32")
                        K.ptx.max.f32(
                            block_amax, K.reinterpret("float32", amax_words[0]), K.float32(0.0)
                        )
                        for slot in range(1, 4):
                            K.ptx.max.f32(
                                block_amax, block_amax, K.reinterpret("float32", amax_words[slot])
                            )
                        K.ptx["red.global.max.s32"](
                            amax.ptr_to([0]), K.reinterpret("int32", block_amax)
                        )
                with K.If(_elected()):
                    with K.Then():
                        if cta_group == 2:
                            K.ptx.mbarrier.arrive.shared__cluster.b64(
                                acc_empty_leader.ptr_to([acc_state.stage]), K.uint32(1)
                            )
                        else:
                            K.ptx.mbarrier.arrive.shared.b64(
                                acc_pipe.empty.ptr_to([acc_state.stage]), K.uint32(1)
                            )
                _advance(acc_state)
                K.assign(executed_tiles, executed_tiles + 1)
                advance_work(work)

            with K.If(warp == 0):
                with K.Then():
                    K.ptx[
                        "tcgen05.relinquish_alloc_permit.cta_group::"
                        + str(cta_group)
                        + ".sync.aligned"
                    ]()
            K.ptx.bar.sync(K.uint32(1), K.uint32(128))
            with K.If(warp == 0):
                with K.Then():
                    if cta_group == 2:
                        remote_dealloc = K.local_scalar("uint32")
                        K.ptx.mapa.shared__cluster.u32(
                            remote_dealloc,
                            K.cuda.cvta_generic_to_shared(tmem_dealloc.ptr_to([0])),
                            K.cast(cluster_rank ^ 1, "uint32"),
                        )
                        K.ptx.mbarrier.arrive.shared__cluster.b64(remote_dealloc, K.uint32(1))
                        _wait_plain(tmem_dealloc.ptr_to([0]), K.uint32(0))
                    K.ptx["tcgen05.dealloc.cta_group::" + str(cta_group) + ".sync.aligned.b32"](
                        tmem_base, K.uint32(tmem_columns)
                    )
            K.ptx.cp.async_.bulk.wait_group.read(0)
            K.ptx.cp.async_.bulk.wait_group.read(0)

    kernel.__annotations__ = {
        "a": K.gptr[K.u8, (M * K_dim * L * ab_bits // 8,)],
        "b": K.gptr[K.u8, (N * K_dim * L * ab_bits // 8,)],
        "sfa": K.gptr[K.u8, (L * _ceil_div(M, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "sfb": K.gptr[K.u8, (L * _ceil_div(N, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "c": K.gptr[K.u8, (M * N * L * c_bits // 8,)],
        "d": K.gptr[K.u8, (M * N * L * d_bits // 8,)],
        "prob": K.gptr[K.f32, (m_tiles * cta_m * L,)],
        "amax": K.gptr[K.f32, (1,)],
        "alpha": K.f32,
    }
    return K.kernel(
        warps=6,
        arch="sm_100a",
        grid=[cluster_m, cluster_n, num_clusters],
        host_prelude=host_prelude,
    )(kernel)


def get_kernel(
    M,
    N,
    K,
    L,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    c_dtype,
    d_dtype,
    a_major,
    b_major,
    c_major,
    mma_tiler_mn,
    cluster_shape_mn,
    vector_f32,
    alpha=None,
):
    del alpha
    return _make_kernel(
        M,
        N,
        K,
        L,
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        d_dtype,
        a_major,
        b_major,
        c_major,
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn),
        vector_f32,
    ).func


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def _torch_dtype(torch, dtype):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e8m0fnu": torch.float8_e8m0fnu,
    }[dtype]


def _regular_tensor(torch, rows, columns, batch, dtype, first_major):
    bits = _dtype_bits(dtype)
    raw = torch.empty(rows * columns * batch * bits // 8, dtype=torch.uint8, device="cuda")
    storage = raw.view(_torch_dtype(torch, dtype))
    strides = (1, rows, rows * columns) if first_major else (columns, 1, rows * columns)
    logical = torch.as_strided(storage, (rows, columns, batch), strides)
    return raw, logical


def _signed_row_factors(torch, rows, batch, offset):
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
    batch_index = torch.arange(batch, device="cuda", dtype=torch.int64)[None, :]
    selector = ((row // 32) + batch_index + offset) % 4
    values = torch.tensor((-1.0, -0.5, 0.5, 1.0), dtype=torch.float32, device="cuda")
    return values[selector]


def _input_tensor(torch, rows, K_dim, batch, dtype, major, offset):
    factors = _signed_row_factors(torch, rows, batch, offset)
    if dtype == "float4_e2m1fn":
        if K_dim % 2:
            raise ValueError("packed FP4 K must be even")
        codes = (factors.abs() * 2.0).to(torch.uint8)
        codes = codes | ((factors < 0).to(torch.uint8) << 3)
        packed = codes | (codes << 4)
        physical = packed.transpose(0, 1)[:, :, None].expand(batch, rows, K_dim // 2).contiguous()
        logical = physical.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
        return {"raw": physical.reshape(-1), "source": logical, "factors": factors}
    raw, logical = _regular_tensor(torch, rows, K_dim, batch, dtype, first_major=major != "k")
    logical.copy_(factors[:, None, :].expand(rows, K_dim, batch).to(logical.dtype))
    return {"raw": raw, "source": logical, "factors": factors}


def _output_tensor(torch, rows, columns, batch, dtype, major):
    raw, logical = _regular_tensor(torch, rows, columns, batch, dtype, first_major=major == "m")
    logical.zero_()
    return {"raw": raw, "source": logical}


def _sf_storage(torch, rows, columns, batch, vec_size, dtype, fill):
    shape = (batch, _ceil_div(rows, 128), _ceil_div(columns, 4 * vec_size), 32, 4, 4)
    return torch.full(shape, fill, dtype=torch.float32, device="cuda").to(
        _torch_dtype(torch, dtype)
    )


def _prob_storage(torch, rows, padded_rows, batch, a_major):
    raw = torch.zeros(batch * padded_rows, dtype=torch.float32, device="cuda")
    strides = (1, padded_rows, padded_rows) if a_major == "m" else (1, 1, padded_rows)
    logical = torch.as_strided(raw, (rows, 1, batch), strides)
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
    batch_index = torch.arange(batch, device="cuda", dtype=torch.int64)[None, :]
    values = ((((row // 16) + batch_index) % 3).float() + 1.0) * 0.125
    logical.copy_(values[:, None, :])
    return {"raw": raw, "source": logical, "values": values}


def prepare_data(**config):
    """Allocate deterministic signed inputs and independent target/source outputs."""
    import math

    import torch

    config = _without_label(config)
    M, N, K_dim, L = (config[key] for key in ("M", "N", "K", "L"))
    sf_dtype = "float8_e8m0fnu" if config["sf_dtype"] == "int8" else config["sf_dtype"]
    a_data = _input_tensor(torch, M, K_dim, L, config["ab_dtype"], config["a_major"], 0)
    b_data = _input_tensor(torch, N, K_dim, L, config["ab_dtype"], config["b_major"], 1)

    scale_exponent = math.ceil(math.log2(max(1.0, K_dim / 512.0)))
    sfa_fill = 2.0 ** (-scale_exponent)
    sfb_fill = 0.5
    sfa = _sf_storage(torch, M, K_dim, L, config["sf_vec_size"], sf_dtype, sfa_fill)
    sfb = _sf_storage(torch, N, K_dim, L, config["sf_vec_size"], sf_dtype, sfb_fill)

    tirx_c = _output_tensor(torch, M, N, L, config["c_dtype"], config["c_major"])
    source_c = _output_tensor(torch, M, N, L, config["c_dtype"], config["c_major"])
    tirx_d = _output_tensor(torch, M, N, L, config["d_dtype"], config["c_major"])
    source_d = _output_tensor(torch, M, N, L, config["d_dtype"], config["c_major"])
    padded_m = _ceil_div(M, 128) * 128
    prob = _prob_storage(torch, M, padded_m, L, config["a_major"])
    tirx_amax = torch.full((1,), float("-inf"), dtype=torch.float32, device="cuda")
    source_amax = torch.full((1,), float("-inf"), dtype=torch.float32, device="cuda")

    return {
        "a": a_data,
        "b": b_data,
        "sfa": sfa,
        "sfb": sfb,
        "prob": prob,
        "tirx_c": tirx_c,
        "source_c": source_c,
        "tirx_d": tirx_d,
        "source_d": source_d,
        "tirx_amax": tirx_amax,
        "source_amax": source_amax,
    }


@cache
def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module(
        "cudnn.gemm.cutedsl.dense.srelu.dense_blockscaled_gemm_persistent_srelu_quant"
    )


def _compile_reference(data, config):
    from tirx_kernels.cudnn._reference import import_cutlass_reference

    cutlass = import_cutlass_reference()
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    module = _load_reference_source()
    config = _without_label(config)
    a = data["a"]["source"]
    b = data["b"]["source"]
    sfa = data["sfa"]
    sfb = data["sfb"]
    c = data["source_c"]["source"]
    d = data["source_d"]["source"]
    prob = data["prob"]["source"]
    amax = data["source_amax"]

    # Loading the reference can release stale TVM resources from an earlier
    # test, and CUDA resource cleanup is allowed to change the thread's current
    # device. DLPack requires it to match the tensor being exported.
    torch.cuda.set_device(a.device)

    def dynamic(tensor, leading_dim):
        return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)

    a_cute = dynamic(a, 0 if config["a_major"] == "m" else 1)
    b_cute = dynamic(b, 0 if config["b_major"] == "n" else 1)
    sfa_cute = from_dlpack(sfa, assumed_align=16)
    sfb_cute = from_dlpack(sfb, assumed_align=16)
    c_cute = dynamic(c, 0 if config["c_major"] == "m" else 1)
    d_cute = dynamic(d, 0 if config["c_major"] == "m" else 1)
    prob_cute = dynamic(prob, 0 if config["a_major"] == "m" else 1)
    amax_cute = from_dlpack(amax, assumed_align=16)

    kernel = module.Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=config["sf_vec_size"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
        vector_f32=config["vector_f32"],
    )
    cluster_size = config["cluster_shape_mn"][0] * config["cluster_shape_mn"][1]
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)

    def epilogue_op(x):
        return cute.where(x > 0, x, cute.full_like(x, 0)) ** 2

    executable = cute.compile(
        kernel,
        a_tensor=a_cute,
        b_tensor=b_cute,
        sfa_tensor=sfa_cute,
        sfb_tensor=sfb_cute,
        c_tensor=c_cute,
        d_tensor=d_cute,
        prob_tensor=prob_cute,
        amax_tensor=amax_cute,
        sfd_tensor=None,
        norm_const_tensor=None,
        alpha=config["alpha"],
        max_active_clusters=max_active_clusters,
        stream=make_fake_stream(use_tvm_ffi_env_stream=False),
        epilogue_op=epilogue_op,
        options="--enable-tvm-ffi",
    )
    torch.cuda.set_device(a.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def launch():
        executable(a, b, sfa, sfb, c, d, prob, amax, None, None, config["alpha"], stream)

    launch._keep_alive = (executable, a, b, sfa, sfb, c, d, prob, amax, stream)
    return launch


def _tirx_launch(executable, data, alpha):
    import torch

    a = data["a"]["raw"]
    b = data["b"]["raw"]
    sfa = data["sfa"].view(torch.uint8).reshape(-1)
    sfb = data["sfb"].view(torch.uint8).reshape(-1)
    c = data["tirx_c"]["raw"]
    d = data["tirx_d"]["raw"]
    prob = data["prob"]["raw"]
    amax = data["tirx_amax"]

    def launch():
        executable(a, b, sfa, sfb, c, d, prob, amax, alpha)

    launch._keep_alive = (a, b, sfa, sfb, c, d, prob, amax)
    return launch


def _reset_outputs(data, prefix):
    data[f"{prefix}_c"]["source"].zero_()
    data[f"{prefix}_d"]["source"].zero_()
    data[f"{prefix}_amax"].fill_(float("-inf"))


def _assert_close(torch, actual, expected, *, atol, rtol, label):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    if actual_f32.shape != expected_f32.shape:
        raise AssertionError(f"{label} shape mismatch: {actual_f32.shape} != {expected_f32.shape}")
    close = torch.isclose(actual_f32, expected_f32, atol=atol, rtol=rtol)
    if bool(torch.all(close).item()):
        return
    absolute_error = torch.abs(actual_f32 - expected_f32)
    allowed_error = atol + rtol * torch.abs(expected_f32)
    excess = absolute_error - allowed_error
    worst_index = int(torch.argmax(excess).item())
    raise AssertionError(
        f"{label} mismatch: value={float(actual_f32.reshape(-1)[worst_index].item())}, "
        f"expected={float(expected_f32.reshape(-1)[worst_index].item())}, "
        f"absolute_error={float(absolute_error.reshape(-1)[worst_index].item())}"
    )


def _output_tolerance(dtype):
    return (1.0e-4, 1.0e-4) if dtype == "float32" else (0.12, 0.02)


def _validate_outputs(data, config, *, with_source):
    """Hold TIRx to the upstream kernel's outputs on the same bytes."""
    if not with_source:
        return
    import torch

    c_atol, c_rtol = _output_tolerance(config["c_dtype"])
    d_atol, d_rtol = _output_tolerance(config["d_dtype"])
    _assert_close(
        torch,
        data["tirx_c"]["source"],
        data["source_c"]["source"],
        atol=c_atol,
        rtol=c_rtol,
        label="TIRx C versus standalone source",
    )
    _assert_close(
        torch,
        data["tirx_d"]["source"],
        data["source_d"]["source"],
        atol=d_atol,
        rtol=d_rtol,
        label="TIRx D versus standalone source",
    )
    _assert_close(
        torch,
        data["tirx_amax"],
        data["source_amax"],
        atol=0.12,
        rtol=0.02,
        label="TIRx amax versus standalone source",
    )


def run_test(**config):
    """Compare TIRx with the cuDNN Frontend kernel on identical inputs."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    executable = compile_kernel(get_kernel(**kernel_config))
    tirx_launch = _tirx_launch(executable, data, kernel_config["alpha"])
    source_launch = _compile_reference(data, kernel_config)
    _reset_outputs(data, "tirx")
    tirx_launch()
    _reset_outputs(data, "source")
    source_launch()
    torch.cuda.synchronize()
    _validate_outputs(data, kernel_config, with_source=True)
    return {"amax": float(data["source_amax"].item())}


def prepare_bench(**config):
    """Compile TIRx before entering the GPU benchmark child."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {"config": kernel_config, "executable": compile_kernel(get_kernel(**kernel_config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then time closures containing exactly one kernel launch."""
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = {**prepared["config"], **kwargs}
    kernel_config = _without_label(config)
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        data = prepare_data(**kernel_config)
        tirx_launch = _tirx_launch(prepared["executable"], data, kernel_config["alpha"])
        gpu_state = {
            "data": data,
            "tirx_launch": tirx_launch,
            "source_launch": None,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")
    data = gpu_state["data"]
    tirx_launch = gpu_state["tirx_launch"]
    source_launch = gpu_state["source_launch"]
    if not gpu_state["validated"]:
        _reset_outputs(data, "tirx")
        tirx_launch()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                if source_launch is None:
                    source_launch = _compile_reference(data, kernel_config)
                    gpu_state["source_launch"] = source_launch
                _reset_outputs(data, "source")
                source_launch()
                torch.cuda.synchronize()
        _validate_outputs(data, kernel_config, with_source=with_source)
        gpu_state["validated"] = True
    references = {"cudnn_frontend": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
