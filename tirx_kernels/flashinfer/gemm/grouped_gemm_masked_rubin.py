# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ 012cfdb97f217e0d48bc9352c17a74068c9e495b)
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM107 masked grouped block-scaled GEMM transcribed from FlashInfer CuTeDSL.

Upstream sources:
- flashinfer/gemm/kernels/grouped_gemm_masked_rubin.py
- flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py
- flashinfer/gemm/kernels/grouped_gemm_masked_wrapper.py
"""

import importlib
import importlib.util
import sys
from functools import cache
from pathlib import Path

import tirx_kernels.tirx_lite as txl

KERNEL_META = {
    "name": "grouped_gemm_masked_rubin",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_107a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "012cfdb97f217e0d48bc9352c17a74068c9e495b",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}

_SOURCE_ROOT = Path("/root-vol/aarch64-ws/kernel-libs/vr200/flashinfer")
_CUTLASS_ROOT = Path(__file__).resolve().parents[3] / ".reference-deps" / "cutlass-v4.8.0dev"
_CUTLASS_PARENT_RELATIVE = Path(
    "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/"
    "dense_blockscaled_gemm_persistent.py"
)
_CUTLASS_PARENT_MODULE = (
    "nvidia_cutlass_dsl.examples.CuTeDSL.cute.blackwell.kernel.blockscaled_gemm."
    "dense_blockscaled_gemm_persistent"
)

_SMEM_CAPACITY = 334_848
_AB_DTYPES = ("float4_e2m1fn", "float8_e4m3fn", "float8_e5m2")
_TMEM_COLUMNS = 576
_TRY_WAIT_TICKS = 10_000_000
_MAX_ACTIVE_CLUSTERS = {1: 200, 2: 100, 4: 40, 8: 20, 16: 10}
_SF_MODES = {
    "nvfp4": ("float8_e4m3fn", 16),
    "mxfp4": ("float8_e8m0fnu", 32),
    "e4m3_vec16": ("float8_e4m3fn", 16),
    "e4m3_vec32": ("float8_e4m3fn", 32),
    "e8m0_vec16": ("float8_e8m0fnu", 16),
    "e8m0_vec32": ("float8_e8m0fnu", 32),
    "e5m3_vec16": ("float8_e5m3fnu", 16),
    "e5m3_vec32": ("float8_e5m3fnu", 32),
    "mxfp8": ("float8_e8m0fnu", 32),
}
_OUT_DTYPES = ("float16", "bfloat16", "float32")


# Public Rubin specializations: (MMA tile MN, MMA instruction MN, cluster MN).
# Index zero is the production specialization. Keep the four high-value branch
# guards next, then expose the complete 120-combination low-level source domain.
def _source_tactics():
    preferred = (
        ((128, 128), (128, 128), (1, 1)),
        ((256, 128), (128, 128), (1, 1)),
        ((256, 128), (256, 128), (2, 1)),
        ((128, 192), (128, 192), (1, 2)),
        ((128, 64), (128, 64), (1, 1)),
    )
    legal = []
    for inst_m in (128, 256):
        for tile_m in (inst_m, 2 * inst_m):
            for tile_n in (64, 128, 192, 256):
                for cluster_m in (1, 2, 4):
                    for cluster_n in (1, 2, 4):
                        if cluster_m % (inst_m // 128) == 0 and cluster_m * cluster_n <= 16:
                            legal.append(
                                ((tile_m, tile_n), (inst_m, tile_n), (cluster_m, cluster_n))
                            )
    return preferred + tuple(tactic for tactic in legal if tactic not in preferred)


TACTICS = _source_tactics()
assert len(TACTICS) == 120

_PRODUCTION_NK = ((4096, 7168), (7168, 2048))


def _production_profiles():
    profiles = [(6, 1024), (6, 512), (1, 1024), (2, 512), (4, 256)]
    for num_ranks in (4, 8, 16, 32, 36, 48, 72):
        for num_tokens in (64, 128, 256, 384, 512, 768, 1024):
            groups = 288 // num_ranks
            profiles.append((groups, num_tokens * 8 // groups))
    return tuple(dict.fromkeys(profiles))


def _production_case(groups, expected, n, k):
    return {
        "label": f"bench_g{groups}_m{expected}_n{n}_k{k}",
        "num_groups": groups,
        "max_m": 4096,
        "expected_m_per_group": expected,
        "N": n,
        "K": k,
        "ab_dtype": "float4_e2m1fn",
        "sf_mode": "nvfp4",
        "out_dtype": "bfloat16",
        "alpha": False,
        "signals": False,
        "tactic": 0,
    }


BENCH_CONFIGS = [
    _production_case(g, expected, n, k)
    for g, expected in _production_profiles()
    for n, k in _PRODUCTION_NK
]
assert len(BENCH_CONFIGS) == 102

CONFIGS = [
    {**_production_case(2, 96, 256, 512), "label": "guard_production_small", "max_m": 256},
    {**_production_case(3, 80, 256, 512), "label": "guard_empty_trailing", "max_m": 256},
    {
        **_production_case(4, 80, 256, 384),
        "label": "guard_signals_alpha_fp32_tailk",
        "max_m": 256,
        "sf_mode": "e4m3_vec32",
        "out_dtype": "float32",
        "alpha": True,
        "signals": True,
    },
    {
        **_production_case(1, 190, 128, 256),
        "label": "guard_fp8_e5_breuse",
        "max_m": 256,
        "ab_dtype": "float8_e5m2",
        "sf_mode": "mxfp8",
        "tactic": 1,
    },
    {
        **_production_case(2, 250, 384, 384),
        "label": "guard_multicluster_groups",
        "max_m": 384,
        "tactic": TACTICS.index(((128, 128), (128, 128), (2, 2))),
    },
    {
        **_production_case(1, 140, 200, 384),
        "label": "guard_partial_cluster_edges",
        "max_m": 200,
        "tactic": TACTICS.index(((128, 128), (128, 128), (2, 4))),
    },
    {
        **_production_case(1, 390, 1024, 256),
        "label": "guard_cluster4x4_n256",
        "max_m": 512,
        "tactic": 37,
    },
    {
        **_production_case(1, 700, 1024, 256),
        "label": "guard_breuse_cluster4x4_n256",
        "max_m": 1024,
        "sf_mode": "e8m0_vec32",
        "out_dtype": "float16",
        "alpha": True,
        "tactic": 72,
    },
    {
        **_production_case(1, 390, 256, 192),
        "label": "guard_fp8_tailk_cta2_cluster4x4",
        "max_m": 512,
        "ab_dtype": "float8_e4m3fn",
        "sf_mode": "mxfp8",
        "tactic": 78,
    },
    {
        **_production_case(1, 700, 1024, 256),
        "label": "guard_fp4_cta2_breuse_cluster4x4_n256",
        "max_m": 1024,
        "tactic": 119,
    },
    {
        **_production_case(1, 350, 256, 256),
        "label": "guard_fp8_cta2_breuse_fp32_alpha_signals",
        "max_m": 512,
        "ab_dtype": "float8_e4m3fn",
        "sf_mode": "mxfp8",
        "out_dtype": "float32",
        "alpha": True,
        "signals": True,
        "tactic": 114,
    },
    {
        **_production_case(1, 700, 384, 512),
        "label": "guard_fp4_cta2_breuse_n192_cluster4x2",
        "max_m": 1024,
        "sf_mode": "e4m3_vec32",
        "out_dtype": "float32",
        "tactic": 112,
    },
    {
        **_production_case(1, 350, 128, 256),
        "label": "guard_fp4_cta2_breuse_n64",
        "max_m": 512,
        "sf_mode": "e8m0_vec16",
        "out_dtype": "float16",
        "alpha": True,
        "tactic": 96,
    },
    {
        **_production_case(1, 350, 256, 256),
        "label": "guard_fp8_e5_cta2",
        "max_m": 512,
        "ab_dtype": "float8_e5m2",
        "sf_mode": "mxfp8",
        "tactic": TACTICS.index(((256, 256), (256, 256), (2, 1))),
    },
    {
        **_production_case(1, 160, 128, 256),
        "label": "guard_fp4_e5m3_vec16",
        "max_m": 256,
        "sf_mode": "e5m3_vec16",
    },
    {
        **_production_case(1, 190, 128, 256),
        "label": "guard_fp4_e5m3_vec32_breuse",
        "max_m": 256,
        "sf_mode": "e5m3_vec32",
        "tactic": 1,
    },
]


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align_up(value, alignment):
    return _ceil_div(value, alignment) * alignment


def _descriptor_base(ldo, sdo, swizzle):
    arrangement_type = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = 0
    value |= (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement_type & 0x7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


def _descriptor_with_address(base, shared_address):
    address_field = txl.cast(
        txl.bitwise_and(txl.shift_right(shared_address, txl.uint32(4)), txl.uint32(0x7FFF)), "uint64"
    )
    return txl.bitwise_or(txl.uint64(base), address_field)


def _instruction_descriptor(inst_m, inst_n, ab_dtype, sf_dtype):
    sf_format = {"float8_e4m3fn": 0, "float8_e8m0fnu": 1, "float8_e5m3fnu": 2}[sf_dtype]
    if ab_dtype == "float4_e2m1fn":
        # PTX FP4 descriptor: K128 and SM107 sparsity-version v1 (also for dense MMA).
        value = (1 << 3) | (1 << 7) | (1 << 10) | (1 << 12)
    else:
        value = 1 << 31
        if ab_dtype == "float8_e5m2":
            value |= 0x480
    value |= ((inst_n >> 3) & 0x3F) << 17
    value |= (sf_format & 3) << 23
    value |= ((inst_m >> 4) & 0x1F) << 24
    return value & 0xFFFFFFFF


def _advance(state):
    state.advance()


def _try_wait_acquire(dst, barrier, phase):
    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, txl.cast(phase, "uint32")
    )


def _wait_plain(barrier, phase):
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, txl.cast(phase, "uint32"), txl.uint32(_TRY_WAIT_TICKS)
        )


def _wait_plain_if_needed(barrier, phase, speculative_ready):
    with txl.If(speculative_ready == txl.uint32(0)):
        with txl.Then():
            _wait_plain(barrier, phase)


def _elected():
    elected_lane = txl.local_scalar("uint32")
    elected_pred = txl.local_scalar("uint32")
    txl.ptx.elect_sync(elected_lane, elected_pred, txl.uint32(0xFFFFFFFF))
    return elected_pred == txl.uint32(1)


def _validate_problem(
    num_groups, max_m, N, K_dim, ab_dtype, sf_mode, out_dtype, alpha, signals, tactic
):
    if not 0 <= tactic < len(TACTICS):
        raise ValueError(f"tactic must be in [0, {len(TACTICS) - 1}], got {tactic}")
    if ab_dtype not in _AB_DTYPES:
        raise ValueError(f"unsupported input dtype {ab_dtype!r}")
    if sf_mode not in _SF_MODES:
        raise ValueError(f"unsupported scale mode {sf_mode!r}")
    if out_dtype not in _OUT_DTYPES:
        raise ValueError(f"unsupported output dtype {out_dtype!r}")
    if num_groups <= 0 or max_m <= 0 or N <= 0 or K_dim <= 0:
        raise ValueError("num_groups, max_m, N, and K must be positive")
    if N % 8 or K_dim % (16 if ab_dtype != "float4_e2m1fn" else 32):
        raise ValueError("source tensor alignment requirements are not satisfied")
    sf_dtype, sf_vec_size = _SF_MODES[sf_mode]
    is_fp8 = ab_dtype != "float4_e2m1fn"
    if ab_dtype != "float4_e2m1fn" and (sf_dtype, sf_vec_size) != ("float8_e8m0fnu", 32):
        raise ValueError("source FP8 inputs require E8M0 scale factors with vector size 32")
    if not isinstance(alpha, bool) or not isinstance(signals, bool):
        raise TypeError("alpha and signals must be compile-time booleans")
    if signals and num_groups > 8:
        raise ValueError("source completion signals pack at most eight experts")
    (tile_m, tile_n), (inst_m, inst_n), (cluster_m, cluster_n) = TACTICS[tactic]
    if tile_m not in (128, 256, 512) or tile_n not in (64, 128, 192, 256):
        raise ValueError("invalid source MMA tile")
    if inst_m not in (128, 256) or inst_n != tile_n or tile_m not in (inst_m, 2 * inst_m):
        raise ValueError("invalid source instruction/tile relation")
    cta_group = inst_m // 128
    if cluster_m % cta_group:
        raise ValueError("cluster M must be divisible by the CTA group")


@cache
def _make_kernel(
    num_groups, max_m, N, K_dim, ab_dtype, sf_mode, out_dtype, alpha, signals, tactic, num_sms
):
    _validate_problem(
        num_groups, max_m, N, K_dim, ab_dtype, sf_mode, out_dtype, alpha, signals, tactic
    )
    (tile_m, n_tile), (inst_m, inst_n), (cluster_m, cluster_n) = TACTICS[tactic]
    prefetch = None
    swap = False
    sf_dtype, sf_vec_size = _SF_MODES[sf_mode]
    is_fp8 = ab_dtype != "float4_e2m1fn"
    cta_group = inst_m // 128
    b_reuse = tile_m == 2 * inst_m
    cta_m = tile_m // cta_group
    # A CTA-group::2 instruction consumes a full N tile jointly, but each peer
    # TMA-loads one N/2 partition of B into its local shared memory.
    b_rows = n_tile // cta_group
    cluster_size = cluster_m * cluster_n
    cluster_m_groups = cluster_m // cta_group
    kernel_m, kernel_n = max_m, N
    m_tiles = _ceil_div(kernel_m, cta_m)
    n_tiles = _ceil_div(kernel_n, n_tile)
    cluster_m_tiles = _ceil_div(m_tiles, cluster_m)
    cluster_n_tiles = _ceil_div(n_tiles, cluster_n)
    cluster_work = cluster_m_tiles * cluster_n_tiles * num_groups
    active_cluster_cap = num_sms if cluster_size == 1 else _MAX_ACTIVE_CLUSTERS[cluster_size]
    num_clusters = min(cluster_work, active_cluster_cap)
    k_tile = 128 if is_fp8 else 256
    k_tiles = _ceil_div(K_dim, k_tile)

    acc_stages = 1 if b_reuse and n_tile in (192, 256) else 2
    input_bits = 8 if is_fp8 else 4
    a_stage_bytes = cta_m * k_tile * input_bits // 8
    b_stage_bytes = b_rows * k_tile * input_bits // 8
    sfa_stage_bytes = cta_m * k_tile // sf_vec_size
    sfb_stage_bytes = _align_up(n_tile, 128) * k_tile // sf_vec_size
    ab_stage_bytes = a_stage_bytes + b_stage_bytes + sfa_stage_bytes + sfb_stage_bytes
    epi_n = 32
    c_element_bytes = 4 if out_dtype == "float32" else 2
    c_stage_bytes = 128 * epi_n * c_element_bytes
    ab_stages = (_SMEM_CAPACITY - (1024 + 2 * c_stage_bytes)) // ab_stage_bytes
    c_stages = (
        2
        + (_SMEM_CAPACITY - ab_stages * ab_stage_bytes - (1024 + 2 * c_stage_bytes))
        // c_stage_bytes
    )
    prefetch_distance = 0
    alpha_is_one = not alpha

    ab_full_offset = 0
    ab_empty_offset = ab_stages * 8
    acc_full_offset = 2 * ab_stages * 8
    acc_empty_offset = acc_full_offset + acc_stages * 8
    tmem_dealloc_offset = acc_empty_offset + acc_stages * 8
    tmem_ptr_offset = tmem_dealloc_offset + 8
    c_offset = 1024
    a_offset = c_offset + c_stages * c_stage_bytes
    b_offset = a_offset + ab_stages * a_stage_bytes
    sfa_offset = b_offset + ab_stages * b_stage_bytes
    sfb_offset = _align_up(sfa_offset + ab_stages * sfa_stage_bytes, 1024)
    shared_bytes = _align_up(sfb_offset + ab_stages * sfb_stage_bytes, 1024)
    if shared_bytes > _SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {shared_bytes} exceeds {_SMEM_CAPACITY}")

    ab_empty_arrivals = cluster_n + cluster_m_groups - 1
    acc_empty_arrivals = 128 * cta_group
    num_tma_load_bytes = ab_stage_bytes * cta_group
    acc_columns = n_tile * acc_stages * (2 if b_reuse else 1)
    sfa_chunks = sfa_stage_bytes // 512
    sfb_chunks = sfb_stage_bytes // 512
    sfa_tmem_column = acc_columns
    sfb_tmem_column = sfa_tmem_column + sfa_chunks * 4
    tmem_columns = _TMEM_COLUMNS
    if sfb_tmem_column + sfb_chunks * 4 + (2 if n_tile in (64, 192) else 0) > tmem_columns:
        raise ValueError("source TMEM intervals exceed the SM107 allocation")

    a_desc_base = _descriptor_base(1, 64, 3)
    b_desc_base = _descriptor_base(1, 64, 3)
    sf_desc_base = _descriptor_base(1, 8, 0)
    instr_desc = _instruction_descriptor(inst_m, inst_n, ab_dtype, sf_dtype)
    sf_k_box = k_tile // (4 * sf_vec_size)
    sfb_n_box = _ceil_div(n_tile, 128)
    a_cluster_piece = cta_m // cluster_n
    b_cluster_piece = b_rows // cluster_m_groups
    a_piece_bytes = a_stage_bytes // cluster_n
    b_piece_bytes = b_stage_bytes // cluster_m_groups
    sfa_m_box = cta_m // 128
    sfa_m_partitions = min(sfa_m_box, cluster_n)
    sfa_remaining_partitions = cluster_n // sfa_m_partitions
    sfa_k_partitions = min(sf_k_box, sfa_remaining_partitions)
    sfa_inner_partitions = sfa_remaining_partitions // sfa_k_partitions
    if sfa_m_box % sfa_m_partitions or sf_k_box % sfa_k_partitions or 256 % sfa_inner_partitions:
        raise ValueError("source SFA cluster partition is not integral")
    sfa_box_0 = 256 // sfa_inner_partitions
    sfa_box_1 = sf_k_box // sfa_k_partitions
    sfa_box_2 = sfa_m_box // sfa_m_partitions
    sfb_piece_values = 256 * sf_k_box * sfb_n_box // cluster_m
    epilogue_subtiles = (n_tile // epi_n) if b_reuse else (cta_m // 128) * (n_tile // epi_n)
    tma_cache_hint = 0

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        c = params["c"]
        sfa = params["sfa"]
        sfb = params["sfb"]
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        sfa_map = txl.stack_alloca("tensormap", 1)
        sfb_map = txl.stack_alloca("tensormap", 1)
        c_map = txl.stack_alloca("tensormap", 1)

        def encode(descriptor, dtype, rank, data, *fields):
            txl.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

        ab_force_dtype = () if is_fp8 else (13,)

        encode(
            a_map,
            ab_dtype,
            3,
            a.data,
            K_dim,
            kernel_m,
            num_groups,
            K_dim * input_bits // 8,
            kernel_m * K_dim * input_bits // 8,
            k_tile,
            a_cluster_piece,
            1,
            1,
            1,
            1,
            0,
            3,
            2,
            0,
            *ab_force_dtype,
        )
        encode(
            b_map,
            ab_dtype,
            3,
            b.data,
            K_dim,
            kernel_n,
            num_groups,
            K_dim * input_bits // 8,
            kernel_n * K_dim * input_bits // 8,
            k_tile,
            b_cluster_piece,
            1,
            1,
            1,
            1,
            0,
            3,
            2,
            0,
            *ab_force_dtype,
        )
        sf_k_groups = _ceil_div(K_dim, 4 * sf_vec_size)
        sf_m_groups = _ceil_div(kernel_m, 128)
        sf_n_groups = _ceil_div(kernel_n, 128)
        encode(
            sfa_map,
            "uint16",
            4,
            sfa.data,
            256,
            sf_k_groups,
            sf_m_groups,
            num_groups,
            512,
            sf_k_groups * 512,
            sf_m_groups * sf_k_groups * 512,
            sfa_box_0,
            sfa_box_1,
            sfa_box_2,
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
            num_groups,
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
        element_bytes = 2 if out_dtype != "float32" else 4
        encode(
            c_map,
            out_dtype,
            3,
            c.data,
            kernel_n,
            kernel_m,
            num_groups,
            kernel_n * element_bytes,
            kernel_m * kernel_n * element_bytes,
            32,
            128,
            1,
            1,
            1,
            1,
            0,
            2 if element_bytes == 2 else 3,
            2,
            0,
        )
        return a_map, b_map, sfa_map, sfb_map, c_map

    def kernel(a, b, sfa, sfb, c, masked_m, alpha_ptr, dst_signals, *, host):
        del a, b, sfa, sfb, c
        if not signals:
            del dst_signals
        required_block_size = txl.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        a_map, b_map, sfa_map, sfb_map, c_map = host

        alpha_value = txl.local_scalar("float32", init=txl.float32(1.0))

        _block_x, _block_y, cluster_work_id = txl.cta_id()
        cluster_x_scope, cluster_y_scope = txl.cta_id_in_cluster(
            [cluster_m, cluster_n], preferred=[cluster_m, cluster_n]
        )
        del _block_x, _block_y, cluster_x_scope, cluster_y_scope
        cluster_rank = txl.local_scalar("int32", init=txl.cuda.mov_sreg(32, "cluster_ctarank"))
        if cluster_m == 1:
            cluster_x = txl.local_scalar("int32", init=0)
        else:
            cluster_x = txl.local_scalar("int32", init=cluster_rank & (cluster_m - 1))
        if cluster_n == 1:
            cluster_y = txl.local_scalar("int32", init=0)
        else:
            cluster_y = txl.local_scalar("int32", init=cluster_rank >> (cluster_m.bit_length() - 1))
        if cta_group == 2:
            cta_v = cluster_x & 1
            leader_cta = cta_v == 0
            cluster_m_group = cluster_x >> 1
            pair_leader_x = cluster_m_group << 1
            leader_rank = pair_leader_x + cluster_m * cluster_y
        else:
            cta_v = txl.local_scalar("int32", init=0)
            leader_cta = txl.bool(True)
            cluster_m_group = cluster_x
            pair_leader_x = cluster_x
            leader_rank = cluster_rank
        warp = txl.warp_id()
        lane = txl.lane_id()

        roles = txl.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])

        smem = txl.alloc_buffer((shared_bytes,), txl.u8, scope="shared.dyn", align=1024)
        protocol_pool = txl.smem_pool(base=smem)
        ab_pipe = txl.Pipeline(
            protocol_pool,
            ab_stages,
            full="tma",
            empty="tcgen05",
            init_empty=ab_empty_arrivals,
            leader=txl.bool(False),
        )
        acc_pipe = txl.Pipeline(
            protocol_pool,
            acc_stages,
            full="tcgen05",
            empty="mbar",
            init_empty=acc_empty_arrivals,
            leader=txl.bool(False),
        )
        if protocol_pool.bytes != tmem_dealloc_offset:
            raise AssertionError("protocol storage offsets changed")
        tmem_dealloc = protocol_pool.alloc((1,), txl.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), txl.u32, align=4)
        if protocol_pool.bytes != tmem_ptr_offset + 4:
            raise AssertionError("protocol storage header changed")

        with tma_role:
            txl.ptx.prefetch.tensormap(txl.address_of(a_map))
            txl.ptx.prefetch.tensormap(txl.address_of(b_map))
            txl.ptx.prefetch.tensormap(txl.address_of(sfa_map))
            txl.ptx.prefetch.tensormap(txl.address_of(sfb_map))
            txl.ptx.prefetch.tensormap(txl.address_of(c_map))

        with txl.If(warp == 0):
            with txl.Then():
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, ab_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, ab_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), txl.uint32(ab_empty_arrivals)
                            )
        txl.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            txl.ptx.barrier.cluster.arrive.relaxed()
            txl.ptx.barrier.cluster.wait()
        else:
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(192))

        with txl.If(warp == 0):
            with txl.Then():
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, acc_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, acc_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), txl.uint32(acc_empty_arrivals)
                            )
        txl.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            txl.ptx.barrier.cluster.arrive.relaxed()
            txl.ptx.barrier.cluster.wait()
        else:
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(192))

        if cta_group == 2:
            with tma_role:
                with txl.If(_elected()):
                    with txl.Then():
                        txl.ptx.mbarrier.init.shared.b64(tmem_dealloc.ptr_to([0]), txl.uint32(32))
        txl.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            txl.ptx.barrier.cluster.arrive.relaxed()

        smem_base = txl.local_scalar("uint32")
        txl.assign(smem_base, txl.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        tmem_slot_addr = txl.uniform(smem_base + txl.uint32(tmem_ptr_offset))
        cluster_smem_u64 = txl.local_scalar("uint64")
        txl.ptx.cvta.to.shared__cluster.u64(cluster_smem_u64, smem.ptr_to([0]))
        cluster_smem = txl.local_scalar("uint32", init=txl.cast(cluster_smem_u64, "uint32"))
        a_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + a_offset)
        )
        b_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + b_offset)
        )

        a_mcast_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for peer_n in range(cluster_n):
            txl.assign(
                a_mcast_mask,
                txl.bitwise_or(
                    a_mcast_mask, txl.uint32(1) << txl.cast(cluster_x + cluster_m * peer_n, "uint32")
                ),
            )
        b_mcast_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for peer_group in range(cluster_m_groups):
            peer_x = cta_v + cta_group * peer_group
            txl.assign(
                b_mcast_mask,
                txl.bitwise_or(
                    b_mcast_mask, txl.uint32(1) << txl.cast(peer_x + cluster_m * cluster_y, "uint32")
                ),
            )
        sfb_mcast_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for peer_x in range(cluster_m):
            txl.assign(
                sfb_mcast_mask,
                txl.bitwise_or(
                    sfb_mcast_mask, txl.uint32(1) << txl.cast(peer_x + cluster_m * cluster_y, "uint32")
                ),
            )
        ab_consumer_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for pair_v in range(cta_group):
            for peer_n in range(cluster_n):
                txl.assign(
                    ab_consumer_mask,
                    txl.bitwise_or(
                        ab_consumer_mask,
                        txl.uint32(1)
                        << txl.cast(pair_leader_x + pair_v + cluster_m * peer_n, "uint32"),
                    ),
                )
            for peer_group in range(cluster_m_groups):
                peer_x = pair_v + cta_group * peer_group
                txl.assign(
                    ab_consumer_mask,
                    txl.bitwise_or(
                        ab_consumer_mask,
                        txl.uint32(1) << txl.cast(peer_x + cluster_m * cluster_y, "uint32"),
                    ),
                )
        if cta_group == 2:
            acc_producer_mask = txl.local_scalar(
                "uint32", init=txl.uint32(3) << txl.cast(leader_rank, "uint32")
            )
        else:
            acc_producer_mask = txl.local_scalar(
                "uint32", init=txl.uint32(1) << txl.cast(cluster_rank, "uint32")
            )
        ab_full_leader = ab_pipe.full.remote_view(leader_rank)
        acc_empty_leader = acc_pipe.empty.remote_view(leader_rank)
        if cluster_size > 1:
            txl.ptx.barrier.cluster.wait()
        else:
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(192))

        def scheduler_coords(
            work, group_idx, accum_tile_m, valid, tile_m_idx, tile_n_idx, batch_idx
        ):
            keep_running = txl.local_scalar("uint32", init=txl.cast(group_idx < num_groups, "uint32"))
            rows = txl.local_scalar("int32", init=0)
            group_m_clusters = txl.local_scalar("int32", init=0)
            with txl.While(keep_running != txl.uint32(0)):
                txl.ptx.ld.global_.s32(rows, masked_m.ptr_to([group_idx]))
                txl.assign(group_m_clusters, (rows + cta_m - 1) // cta_m)
                with txl.If((accum_tile_m + group_m_clusters) * n_tiles <= work):
                    with txl.Then():
                        txl.assign(accum_tile_m, accum_tile_m + group_m_clusters)
                        txl.assign(group_idx, group_idx + 1)
                        txl.assign(keep_running, txl.cast(group_idx < num_groups, "uint32"))
                    with txl.Else():
                        txl.assign(keep_running, txl.uint32(0))
            txl.assign(valid, txl.cast(group_idx < num_groups, "uint32"))
            cluster_m_idx = txl.local_scalar("int32", init=work // n_tiles - accum_tile_m)
            cluster_n_idx = txl.local_scalar("int32", init=work % n_tiles)
            txl.assign(tile_m_idx, cluster_m_idx * cluster_m + cluster_x)
            txl.assign(tile_n_idx, cluster_n_idx * cluster_n + cluster_y)
            txl.assign(batch_idx, group_idx)

        def read_pending_byte(packed, index):
            shift = txl.cast(index * 8, "uint64")
            return txl.cast(txl.bitwise_and(txl.shift_right(packed, shift), txl.uint64(0xFF)), "uint32")

        def write_pending_byte(packed, index, value):
            shift = txl.cast(index * 8, "uint64")
            mask = txl.shift_left(txl.uint64(0xFF), shift)
            encoded = txl.shift_left(txl.cast(txl.bitwise_and(value, txl.uint32(0xFF)), "uint64"), shift)
            return txl.bitwise_or(txl.bitwise_and(packed, txl.bitwise_not(mask)), encoded)

        def scheduler_coords_epilogue(
            work,
            group_idx,
            accum_tile_m,
            valid,
            tile_m_idx,
            tile_n_idx,
            batch_idx,
            dsm_pending_packed,
            dsm_counter,
        ):
            keep_running = txl.local_scalar("uint32", init=txl.cast(group_idx < num_groups, "uint32"))
            rows = txl.local_scalar("int32", init=0)
            group_m_clusters = txl.local_scalar("int32", init=0)
            with txl.While(keep_running != txl.uint32(0)):
                txl.ptx.ld.global_.s32(rows, masked_m.ptr_to([group_idx]))
                txl.assign(group_m_clusters, (rows + cta_m - 1) // cta_m)
                with txl.If((accum_tile_m + group_m_clusters) * n_tiles <= work):
                    with txl.Then():
                        txl.assign(
                            dsm_pending_packed,
                            write_pending_byte(
                                dsm_pending_packed, group_idx, dsm_counter + txl.uint32(c_stages - 1)
                            ),
                        )
                        txl.assign(accum_tile_m, accum_tile_m + group_m_clusters)
                        txl.assign(group_idx, group_idx + 1)
                        txl.assign(keep_running, txl.cast(group_idx < num_groups, "uint32"))
                    with txl.Else():
                        txl.assign(keep_running, txl.uint32(0))
            txl.assign(valid, txl.cast(group_idx < num_groups, "uint32"))
            cluster_m_idx = txl.local_scalar("int32", init=work // n_tiles - accum_tile_m)
            cluster_n_idx = txl.local_scalar("int32", init=work % n_tiles)
            txl.assign(tile_m_idx, cluster_m_idx * cluster_m + cluster_x)
            txl.assign(tile_n_idx, cluster_n_idx * cluster_n + cluster_y)
            txl.assign(batch_idx, group_idx)

        def advance_work(work):
            txl.assign(work, work + num_clusters)

        def cta2_tma_barrier(stage):
            # Bit 24 is the CTA-in-pair selector; CTA2 completions target rank 0.
            return txl.bitwise_and(
                txl.cuda.cvta_generic_to_shared(ab_pipe.full.ptr_to([stage])), txl.uint32(0xFEFFFFF8)
            )

        def prefetch_inputs(tile_m_idx, tile_n_idx, future_k):
            a_coord_m = tile_m_idx * cta_m + cluster_y * a_cluster_piece
            a_coord_k = future_k * k_tile
            b_coord_n = tile_n_idx * n_tile + cta_v * b_rows + cluster_m_group * b_cluster_piece
            b_coord_k = future_k * k_tile
            sfa_inner_index = cluster_y % sfa_inner_partitions
            sfa_partition_outer = cluster_y // sfa_inner_partitions
            sfa_k_index = sfa_partition_outer % sfa_k_partitions
            sfa_m_index = sfa_partition_outer // sfa_k_partitions
            sfa_coord_0 = sfa_inner_index * sfa_box_0
            sfa_coord_1 = future_k * sf_k_box + sfa_k_index * sfa_box_1
            sfa_coord_2 = tile_m_idx * sfa_m_box + sfa_m_index * sfa_box_2
            sfb_linear = cluster_x * sfb_piece_values
            sfb_coord_0 = sfb_linear % 256
            sfb_quotient = sfb_linear // 256
            sfb_coord_1 = future_k * sf_k_box + sfb_quotient % sf_k_box
            sfb_tile_group = tile_n_idx * n_tile // 128
            sfb_coord_2 = sfb_tile_group + sfb_quotient // sf_k_box
            txl.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"](
                txl.address_of(a_map),
                txl.cast(a_coord_k, "int32"),
                txl.cast(a_coord_m, "int32"),
                txl.uint64(tma_cache_hint),
            )
            txl.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"](
                txl.address_of(b_map),
                txl.cast(b_coord_k, "int32"),
                txl.cast(b_coord_n, "int32"),
                txl.uint64(tma_cache_hint),
            )
            txl.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"](
                txl.address_of(sfa_map),
                txl.cast(sfa_coord_0, "int32"),
                txl.cast(sfa_coord_1, "int32"),
                txl.cast(sfa_coord_2, "int32"),
                txl.uint64(tma_cache_hint),
            )
            txl.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"](
                txl.address_of(sfb_map),
                txl.cast(sfb_coord_0, "int32"),
                txl.cast(sfb_coord_1, "int32"),
                txl.cast(sfb_coord_2, "int32"),
                txl.uint64(tma_cache_hint),
            )

        with tma_role:
            tma_state = txl.PipelineState(ab_stages, phase=1)
            work = txl.local_scalar("int32", init=cluster_work_id)
            count = txl.local_scalar("int32")
            speculative = txl.local_scalar("uint32")
            group_idx = txl.local_scalar("int32", init=0)
            accum_tile_m = txl.local_scalar("int32", init=0)
            valid = txl.local_scalar("uint32", init=0)
            tile_m_idx = txl.local_scalar("int32", init=0)
            tile_n_idx = txl.local_scalar("int32", init=0)
            batch_idx = txl.local_scalar("int32", init=0)
            scheduler_coords(
                work, group_idx, accum_tile_m, valid, tile_m_idx, tile_n_idx, batch_idx
            )
            with txl.While(valid != txl.uint32(0)):
                if prefetch_distance > 0:
                    prefetch_k = txl.local_scalar("int32", init=0)
                    with txl.While((prefetch_k < prefetch_distance) & (prefetch_k < k_tiles)):
                        prefetch_inputs(tile_m_idx, tile_n_idx, prefetch_k)
                        txl.assign(prefetch_k, prefetch_k + 1)
                txl.assign(count, 0)
                txl.assign(speculative, txl.uint32(1))
                with txl.If(count < k_tiles):
                    with txl.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase
                        )
                with txl.While(count < k_tiles):
                    _wait_plain_if_needed(
                        ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase, speculative
                    )
                    with txl.If(leader_cta):
                        with txl.Then():
                            with txl.If(_elected()):
                                with txl.Then():
                                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        txl.uint32(num_tma_load_bytes),
                                    )
                    with txl.If(_elected()):
                        with txl.Then():
                            a_coord_m = tile_m_idx * cta_m + cluster_y * a_cluster_piece
                            a_coord_k = count * k_tile
                            a_smem_offset = (
                                a_offset
                                + tma_state.stage * a_stage_bytes
                                + cluster_y * a_piece_bytes
                            )
                            if cta_group == 2:
                                a_barrier = cta2_tma_barrier(tma_state.stage)
                                if cluster_n == 1:
                                    txl.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + a_smem_offset,
                                        txl.address_of(a_map),
                                        txl.cast(a_coord_k, "int32"),
                                        txl.cast(a_coord_m, "int32"),
                                        txl.cast(batch_idx, "int32"),
                                        a_barrier,
                                        txl.uint64(tma_cache_hint),
                                    )
                                else:
                                    txl.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + a_smem_offset,
                                        txl.address_of(a_map),
                                        txl.cast(a_coord_k, "int32"),
                                        txl.cast(a_coord_m, "int32"),
                                        txl.cast(batch_idx, "int32"),
                                        a_barrier,
                                        txl.cast(a_mcast_mask, "uint16"),
                                        txl.uint64(tma_cache_hint),
                                    )
                            elif cluster_n == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([a_smem_offset]),
                                    txl.address_of(a_map),
                                    txl.cast(a_coord_k, "int32"),
                                    txl.cast(a_coord_m, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.uint64(tma_cache_hint),
                                )
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
                                    cluster_smem + a_smem_offset,
                                    txl.address_of(a_map),
                                    txl.cast(a_coord_k, "int32"),
                                    txl.cast(a_coord_m, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.cast(a_mcast_mask, "uint16"),
                                    txl.uint64(tma_cache_hint),
                                )
                    with txl.If(_elected()):
                        with txl.Then():
                            b_coord_n = (
                                tile_n_idx * n_tile
                                + cta_v * b_rows
                                + cluster_m_group * b_cluster_piece
                            )
                            b_coord_k = count * k_tile
                            b_smem_offset = (
                                b_offset
                                + tma_state.stage * b_stage_bytes
                                + cluster_m_group * b_piece_bytes
                            )
                            if cta_group == 2:
                                b_barrier = cta2_tma_barrier(tma_state.stage)
                                if cluster_m_groups == 1:
                                    txl.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + b_smem_offset,
                                        txl.address_of(b_map),
                                        txl.cast(b_coord_k, "int32"),
                                        txl.cast(b_coord_n, "int32"),
                                        txl.cast(batch_idx, "int32"),
                                        b_barrier,
                                        txl.uint64(tma_cache_hint),
                                    )
                                else:
                                    txl.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + b_smem_offset,
                                        txl.address_of(b_map),
                                        txl.cast(b_coord_k, "int32"),
                                        txl.cast(b_coord_n, "int32"),
                                        txl.cast(batch_idx, "int32"),
                                        b_barrier,
                                        txl.cast(b_mcast_mask, "uint16"),
                                        txl.uint64(tma_cache_hint),
                                    )
                            elif cluster_m_groups == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([b_smem_offset]),
                                    txl.address_of(b_map),
                                    txl.cast(b_coord_k, "int32"),
                                    txl.cast(b_coord_n, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.uint64(tma_cache_hint),
                                )
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
                                    cluster_smem + b_smem_offset,
                                    txl.address_of(b_map),
                                    txl.cast(b_coord_k, "int32"),
                                    txl.cast(b_coord_n, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.cast(b_mcast_mask, "uint16"),
                                    txl.uint64(tma_cache_hint),
                                )

                    with txl.If(_elected()):
                        with txl.Then():
                            sfa_inner_index = cluster_y % sfa_inner_partitions
                            sfa_partition_outer = cluster_y // sfa_inner_partitions
                            sfa_k_index = sfa_partition_outer % sfa_k_partitions
                            sfa_m_index = sfa_partition_outer // sfa_k_partitions
                            sfa_coord_0 = sfa_inner_index * sfa_box_0
                            sfa_coord_1 = count * sf_k_box + sfa_k_index * sfa_box_1
                            sfa_coord_2 = tile_m_idx * sfa_m_box + sfa_m_index * sfa_box_2
                            sfa_smem_offset = (
                                sfa_offset
                                + tma_state.stage * sfa_stage_bytes
                                + cluster_y * (sfa_stage_bytes // cluster_n)
                            )
                            if cta_group == 2:
                                if cluster_n == 1:
                                    txl.ptx[
                                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + sfa_smem_offset,
                                        txl.address_of(sfa_map),
                                        txl.cast(sfa_coord_0, "int32"),
                                        txl.cast(sfa_coord_1, "int32"),
                                        txl.cast(sfa_coord_2, "int32"),
                                        txl.cast(batch_idx, "int32"),
                                        cta2_tma_barrier(tma_state.stage),
                                        txl.uint64(tma_cache_hint),
                                    )
                                else:
                                    txl.ptx[
                                        "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + sfa_smem_offset,
                                        txl.address_of(sfa_map),
                                        txl.cast(sfa_coord_0, "int32"),
                                        txl.cast(sfa_coord_1, "int32"),
                                        txl.cast(sfa_coord_2, "int32"),
                                        txl.cast(batch_idx, "int32"),
                                        cta2_tma_barrier(tma_state.stage),
                                        txl.cast(a_mcast_mask, "uint16"),
                                        txl.uint64(tma_cache_hint),
                                    )
                            elif cluster_n == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([sfa_smem_offset]),
                                    txl.address_of(sfa_map),
                                    txl.cast(sfa_coord_0, "int32"),
                                    txl.cast(sfa_coord_1, "int32"),
                                    txl.cast(sfa_coord_2, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.uint64(tma_cache_hint),
                                )
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                ](
                                    cluster_smem + sfa_smem_offset,
                                    txl.address_of(sfa_map),
                                    txl.cast(sfa_coord_0, "int32"),
                                    txl.cast(sfa_coord_1, "int32"),
                                    txl.cast(sfa_coord_2, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.cast(a_mcast_mask, "uint16"),
                                    txl.uint64(tma_cache_hint),
                                )

                    with txl.If(_elected()):
                        with txl.Then():
                            sfb_linear = cluster_x * sfb_piece_values
                            sfb_coord_0 = sfb_linear % 256
                            sfb_quotient = sfb_linear // 256
                            sfb_coord_1 = count * sf_k_box + sfb_quotient % sf_k_box
                            sfb_tile_group = tile_n_idx * n_tile // 128
                            sfb_coord_2 = sfb_tile_group + sfb_quotient // sf_k_box
                            sfb_smem_offset = (
                                sfb_offset
                                + tma_state.stage * sfb_stage_bytes
                                + cluster_x * (sfb_stage_bytes // cluster_m)
                            )
                            if cta_group == 2:
                                txl.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2"
                                ](
                                    cluster_smem + sfb_smem_offset,
                                    txl.address_of(sfb_map),
                                    txl.cast(sfb_coord_0, "int32"),
                                    txl.cast(sfb_coord_1, "int32"),
                                    txl.cast(sfb_coord_2, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    cta2_tma_barrier(tma_state.stage),
                                    txl.cast(sfb_mcast_mask, "uint16"),
                                    txl.uint64(tma_cache_hint),
                                )
                            elif cluster_m_groups == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([sfb_smem_offset]),
                                    txl.address_of(sfb_map),
                                    txl.cast(sfb_coord_0, "int32"),
                                    txl.cast(sfb_coord_1, "int32"),
                                    txl.cast(sfb_coord_2, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.uint64(tma_cache_hint),
                                )
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                ](
                                    cluster_smem + sfb_smem_offset,
                                    txl.address_of(sfb_map),
                                    txl.cast(sfb_coord_0, "int32"),
                                    txl.cast(sfb_coord_1, "int32"),
                                    txl.cast(sfb_coord_2, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.cast(b_mcast_mask, "uint16"),
                                    txl.uint64(tma_cache_hint),
                                )
                    if prefetch_distance > 0:
                        with txl.If(count < k_tiles - prefetch_distance):
                            with txl.Then():
                                prefetch_inputs(tile_m_idx, tile_n_idx, count + prefetch_distance)
                    _advance(tma_state)
                    txl.assign(count, count + 1)
                    txl.assign(speculative, txl.uint32(1))
                    with txl.If(count < k_tiles):
                        with txl.Then():
                            _try_wait_acquire(
                                speculative,
                                ab_pipe.empty.ptr_to([tma_state.stage]),
                                tma_state.phase,
                            )
                advance_work(work)
                scheduler_coords(
                    work, group_idx, accum_tile_m, valid, tile_m_idx, tile_n_idx, batch_idx
                )
            with txl.unroll(0, ab_stages) as unused_stage:
                _wait_plain(ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase)
                _advance(tma_state)
            del unused_stage

        with mma_role:
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(160))
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            sfa_descriptor = txl.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + sfa_offset)
            )
            sfb_descriptor = txl.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + sfb_offset)
            )
            mma_state = txl.PipelineState(ab_stages, phase=0)
            acc_state = txl.PipelineState(acc_stages, phase=1)
            work = txl.local_scalar("int32", init=cluster_work_id)
            count = txl.local_scalar("int32")
            speculative = txl.local_scalar("uint32")
            accumulate = txl.local_scalar("uint32")

            def runtime_descriptor(sfa_addr, sfb_addr):
                desc = txl.bitwise_and(txl.uint32(instr_desc), txl.uint32(0x9FFFFFCF))
                desc = txl.bitwise_or(
                    desc, txl.bitwise_and(txl.shift_right(sfa_addr, txl.uint32(1)), txl.uint32(0x60000000))
                )
                return txl.bitwise_or(
                    desc, txl.bitwise_and(txl.shift_right(sfb_addr, txl.uint32(26)), txl.uint32(0x30))
                )

            group_idx = txl.local_scalar("int32", init=0)
            accum_tile_m = txl.local_scalar("int32", init=0)
            valid = txl.local_scalar("uint32", init=0)
            tile_m_idx = txl.local_scalar("int32", init=0)
            tile_n_idx = txl.local_scalar("int32", init=0)
            batch_idx = txl.local_scalar("int32", init=0)
            scheduler_coords(
                work, group_idx, accum_tile_m, valid, tile_m_idx, tile_n_idx, batch_idx
            )
            with txl.While(valid != txl.uint32(0)):
                txl.assign(count, 0)
                txl.assign(speculative, txl.uint32(1))
                with txl.If((count < k_tiles) & leader_cta):
                    with txl.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                        )
                with txl.If(leader_cta):
                    with txl.Then():
                        _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)
                txl.assign(accumulate, txl.uint32(0))
                with txl.While(count < k_tiles):
                    with txl.If(leader_cta):
                        with txl.Then():
                            _wait_plain_if_needed(
                                ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase, speculative
                            )
                            sfb_odd_shift = txl.local_scalar(
                                "uint32",
                                init=txl.Select(
                                    (tile_n_idx & 1) != 0,
                                    txl.uint32(2 if n_tile in (64, 192) else 0),
                                    txl.uint32(0),
                                ),
                            )
                            if b_reuse and is_fp8:
                                for sf_chunk in range(sfa_chunks):
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                            ](
                                                txl.cast(
                                                    tmem_base + sfa_tmem_column + sf_chunk * 4,
                                                    "uint32",
                                                ),
                                                sfa_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (sfa_stage_bytes // 16)
                                                    + sf_chunk * 32,
                                                    "uint64",
                                                ),
                                            )
                                for sf_chunk in range(sfb_chunks):
                                    sfb_shared_chunk = (
                                        sf_chunk % sfb_n_box
                                    ) * sf_k_box + sf_chunk // sfb_n_box
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                            ](
                                                txl.cast(
                                                    tmem_base + sfb_tmem_column + sf_chunk * 4,
                                                    "uint32",
                                                ),
                                                sfb_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (sfb_stage_bytes // 16)
                                                    + sfb_shared_chunk * 32,
                                                    "uint64",
                                                ),
                                            )
                                for kblock in range(2):
                                    sf_selector = txl.uint32(kblock * 0x80000000)
                                    sfa_keep = txl.cast(
                                        tmem_base + sfa_tmem_column + sf_selector, "uint32"
                                    )
                                    sfa_reuse = txl.cast(sfa_keep + 4, "uint32")
                                    sfb_addr = txl.cast(
                                        tmem_base + sfb_tmem_column + sfb_odd_shift + sf_selector,
                                        "uint32",
                                    )
                                    keep_desc = runtime_descriptor(sfa_keep, sfb_addr)
                                    reuse_desc = runtime_descriptor(sfa_reuse, sfb_addr)
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4"
                                                ".block_scale.collector::a::discard"
                                                ".collector::b::fill"
                                            ](
                                                txl.cast(
                                                    tmem_base + acc_state.stage * n_tile * 2,
                                                    "uint32",
                                                ),
                                                a_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                keep_desc,
                                                sfa_keep,
                                                sfb_addr,
                                                txl.ptx.pred(txl.cast(accumulate, "bool")),
                                            )
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4"
                                                ".block_scale.collector::a::discard"
                                                ".collector::b::lastuse"
                                            ](
                                                txl.cast(
                                                    tmem_base
                                                    + acc_state.stage * n_tile * 2
                                                    + n_tile,
                                                    "uint32",
                                                ),
                                                a_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + 1024
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                reuse_desc,
                                                sfa_reuse,
                                                sfb_addr,
                                                txl.ptx.pred(txl.cast(accumulate, "bool")),
                                            )
                                    txl.assign(accumulate, txl.uint32(1))
                            elif b_reuse:
                                sf_layout_scale = sf_vec_size // 16
                                sfa_segment_chunks = 2
                                sfb_kblock_chunks = sfb_chunks // 2
                                for kblock in range(2):
                                    for chunk in range(sfa_segment_chunks):
                                        with txl.If(_elected()):
                                            with txl.Then():
                                                txl.ptx[
                                                    f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                                ](
                                                    txl.cast(
                                                        tmem_base
                                                        + sfa_tmem_column
                                                        + kblock * (8 // sf_layout_scale)
                                                        + chunk * (16 // sf_layout_scale),
                                                        "uint32",
                                                    ),
                                                    sfa_descriptor
                                                    + txl.cast(
                                                        mma_state.stage * (sfa_stage_bytes // 16)
                                                        + kblock * (64 // sf_layout_scale)
                                                        + chunk * (128 // sf_layout_scale),
                                                        "uint64",
                                                    ),
                                                )
                                    for chunk in range(sfb_kblock_chunks):
                                        sfb_shared_chunk = (
                                            (chunk % sfb_n_box) * sf_k_box
                                            + kblock * (sfb_kblock_chunks // sfb_n_box)
                                            + chunk // sfb_n_box
                                        )
                                        with txl.If(_elected()):
                                            with txl.Then():
                                                txl.ptx[
                                                    f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                                ](
                                                    txl.cast(
                                                        tmem_base
                                                        + sfb_tmem_column
                                                        + kblock * sfb_kblock_chunks * 4
                                                        + chunk * 4,
                                                        "uint32",
                                                    ),
                                                    sfb_descriptor
                                                    + txl.cast(
                                                        mma_state.stage * (sfb_stage_bytes // 16)
                                                        + sfb_shared_chunk * 32,
                                                        "uint64",
                                                    ),
                                                )
                                    for chunk in range(sfa_segment_chunks):
                                        with txl.If(_elected()):
                                            with txl.Then():
                                                txl.ptx[
                                                    f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                                ](
                                                    txl.cast(
                                                        tmem_base
                                                        + sfa_tmem_column
                                                        + (4 if sf_vec_size == 16 else 0)
                                                        + kblock * (8 // sf_layout_scale)
                                                        + chunk * (16 // sf_layout_scale),
                                                        "uint32",
                                                    ),
                                                    sfa_descriptor
                                                    + txl.cast(
                                                        mma_state.stage * (sfa_stage_bytes // 16)
                                                        + kblock * (64 // sf_layout_scale)
                                                        + (32 if sf_vec_size == 16 else 0)
                                                        + chunk * (128 // sf_layout_scale),
                                                        "uint64",
                                                    ),
                                                )
                                    sfb_addr = txl.cast(
                                        tmem_base
                                        + sfb_tmem_column
                                        + sfb_odd_shift
                                        + kblock * sfb_kblock_chunks * 4,
                                        "uint32",
                                    )
                                    sfa_keep = txl.cast(
                                        tmem_base
                                        + sfa_tmem_column
                                        + kblock * (8 // sf_layout_scale),
                                        "uint32",
                                    )
                                    sfa_reuse = txl.cast(sfa_keep + 16 // sf_layout_scale, "uint32")
                                    keep_desc = runtime_descriptor(sfa_keep, sfb_addr)
                                    reuse_desc = runtime_descriptor(sfa_reuse, sfb_addr)
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4"
                                                f".block_scale.block{sf_vec_size}"
                                                ".collector::a::discard.collector::b::fill"
                                            ](
                                                txl.cast(
                                                    tmem_base + acc_state.stage * n_tile * 2,
                                                    "uint32",
                                                ),
                                                a_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                keep_desc,
                                                sfa_keep,
                                                sfb_addr,
                                                txl.ptx.pred(txl.cast(accumulate, "bool")),
                                            )
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4"
                                                f".block_scale.block{sf_vec_size}"
                                                ".collector::a::discard.collector::b::lastuse"
                                            ](
                                                txl.cast(
                                                    tmem_base
                                                    + acc_state.stage * n_tile * 2
                                                    + n_tile,
                                                    "uint32",
                                                ),
                                                a_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + 1024
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                reuse_desc,
                                                sfa_reuse,
                                                sfb_addr,
                                                txl.ptx.pred(txl.cast(accumulate, "bool")),
                                            )
                                    txl.assign(accumulate, txl.uint32(1))
                            else:
                                mma_name = (
                                    f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4"
                                    ".block_scale.collector::a::discard"
                                    if is_fp8
                                    else f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4"
                                    f".block_scale.block{sf_vec_size}.collector::a::discard"
                                )
                                for sf_chunk in range(sfa_chunks):
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                            ](
                                                txl.cast(
                                                    tmem_base + sfa_tmem_column + sf_chunk * 4,
                                                    "uint32",
                                                ),
                                                sfa_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (sfa_stage_bytes // 16)
                                                    + sf_chunk * 32,
                                                    "uint64",
                                                ),
                                            )
                                for sf_chunk in range(sfb_chunks):
                                    sfb_shared_chunk = (
                                        sf_chunk % sfb_n_box
                                    ) * sf_k_box + sf_chunk // sfb_n_box
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[
                                                f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                            ](
                                                txl.cast(
                                                    tmem_base + sfb_tmem_column + sf_chunk * 4,
                                                    "uint32",
                                                ),
                                                sfb_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (sfb_stage_bytes // 16)
                                                    + sfb_shared_chunk * 32,
                                                    "uint64",
                                                ),
                                            )
                                for kblock in range(2):
                                    if is_fp8:
                                        sf_selector = txl.uint32(kblock * 0x80000000)
                                        sfa_addr = txl.cast(
                                            tmem_base + sfa_tmem_column + sf_selector, "uint32"
                                        )
                                        sfb_addr = txl.cast(
                                            tmem_base
                                            + sfb_tmem_column
                                            + sfb_odd_shift
                                            + sf_selector,
                                            "uint32",
                                        )
                                    else:
                                        sfa_addr = txl.cast(
                                            tmem_base
                                            + sfa_tmem_column
                                            + kblock * (sfa_chunks // 2) * 4,
                                            "uint32",
                                        )
                                        sfb_addr = txl.cast(
                                            tmem_base
                                            + sfb_tmem_column
                                            + sfb_odd_shift
                                            + kblock * (sfb_chunks // 2) * 4,
                                            "uint32",
                                        )
                                    mma_desc = runtime_descriptor(sfa_addr, sfb_addr)
                                    with txl.If(_elected()):
                                        with txl.Then():
                                            txl.ptx[mma_name](
                                                txl.cast(
                                                    tmem_base + acc_state.stage * n_tile, "uint32"
                                                ),
                                                a_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + txl.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                mma_desc,
                                                sfa_addr,
                                                sfb_addr,
                                                txl.ptx.pred(txl.cast(accumulate, "bool")),
                                            )
                                    txl.assign(accumulate, txl.uint32(1))
                            with txl.If(_elected()):
                                with txl.Then():
                                    if cluster_size == 1:
                                        txl.ptx[
                                            f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                            "arrive::one.shared::cluster.b64"
                                        ](ab_pipe.empty.ptr_to([mma_state.stage]))
                                    else:
                                        txl.ptx[
                                            f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                            "arrive::one.shared::cluster.multicast::cluster.b64"
                                        ](
                                            ab_pipe.empty.ptr_to([mma_state.stage]),
                                            txl.cast(ab_consumer_mask, "uint16"),
                                        )
                    _advance(mma_state)
                    txl.assign(count, count + 1)
                    txl.assign(speculative, txl.uint32(1))
                    with txl.If((count < k_tiles) & leader_cta):
                        with txl.Then():
                            _try_wait_acquire(
                                speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                            )
                with txl.If(leader_cta):
                    with txl.Then():
                        with txl.If(_elected()):
                            with txl.Then():
                                if cluster_size == 1:
                                    txl.ptx[
                                        f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                        "arrive::one.shared::cluster.b64"
                                    ](acc_pipe.full.ptr_to([acc_state.stage]))
                                else:
                                    txl.ptx[
                                        f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                        "arrive::one.shared::cluster.multicast::cluster.b64"
                                    ](
                                        acc_pipe.full.ptr_to([acc_state.stage]),
                                        txl.cast(acc_producer_mask, "uint16"),
                                    )
                _advance(acc_state)
                advance_work(work)
                scheduler_coords(
                    work, group_idx, accum_tile_m, valid, tile_m_idx, tile_n_idx, batch_idx
                )
            with txl.If(leader_cta):
                with txl.Then():
                    for _ in range(acc_stages - 1):
                        _advance(acc_state)
                    _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            with txl.If(warp == 0):
                with txl.Then():
                    txl.ptx[
                        f"tcgen05.alloc.exclusive.cta_group::{cta_group}.sync.aligned."
                        "shared::cta.b32"
                    ](tmem_slot_addr, txl.uint32(tmem_columns))
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(160))
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            acc_state = txl.PipelineState(acc_stages, phase=0)
            work = txl.local_scalar("int32", init=cluster_work_id)
            executed_subtiles = txl.local_scalar("int32", init=0)
            dsm_pending_packed = txl.local_scalar("uint64", init=txl.uint64(0))
            dsm_pending_idx = txl.local_scalar("int32", init=0)
            dsm_counter = txl.local_scalar("uint32", init=txl.uint32(0))
            values = txl.alloc_local((32,), "float32")
            words = txl.alloc_local((32 if out_dtype == "float32" else 16,), "uint32")
            offsets = txl.alloc_local((8 if out_dtype == "float32" else 4,), "int32")

            def scale_and_pack():
                if not alpha_is_one:
                    for index in range(0, 32, 2):
                        packed = txl.local_scalar("uint64")
                        scale_pair = txl.local_scalar("uint64")
                        txl.ptx.mov.b64(packed, values[index], values[index + 1])
                        txl.ptx.mov.b64(scale_pair, alpha_value, alpha_value)
                        txl.ptx.mul.f32x2(packed, scale_pair, packed)
                        txl.ptx.mov.b64(values[index], values[index + 1], packed)
                if out_dtype == "float32":
                    for index in range(32):
                        txl.assign(words[index], txl.reinterpret("uint32", values[index]))
                else:
                    for index in range(16):
                        if out_dtype == "float16":
                            txl.ptx.cvt.rn.f16x2.f32(
                                words[index], values[index * 2 + 1], values[index * 2]
                            )
                        else:
                            txl.ptx.cvt.rn.bf16x2.f32(
                                words[index], values[index * 2 + 1], values[index * 2]
                            )

            def publish_ready_signals(final):
                if signals:
                    with txl.If((warp == 0) & (lane == 0)):
                        with txl.Then():
                            keep_publishing = txl.local_scalar(
                                "uint32", init=txl.cast(dsm_pending_idx < num_groups, "uint32")
                            )
                            with txl.While(keep_publishing != txl.uint32(0)):
                                if final:
                                    ready_to_publish = txl.bool(True)
                                else:
                                    ready_to_publish = (
                                        read_pending_byte(dsm_pending_packed, dsm_pending_idx)
                                        == dsm_counter
                                    )
                                with txl.If(ready_to_publish):
                                    with txl.Then():
                                        previous = txl.local_scalar("int32")
                                        txl.ptx.atom.release.gpu.global_.add.s32(
                                            previous,
                                            dst_signals.ptr_to([dsm_pending_idx]),
                                            txl.int32(1),
                                        )
                                        txl.assign(dsm_pending_idx, dsm_pending_idx + 1)
                                        txl.assign(
                                            keep_publishing,
                                            txl.cast(dsm_pending_idx < num_groups, "uint32"),
                                        )
                                    with txl.Else():
                                        txl.assign(keep_publishing, txl.uint32(0))

            def row_major_output_offset(stage, vector):
                row_bytes = epi_n * c_element_bytes
                unswizzled = (
                    smem_base
                    + c_offset
                    + stage * c_stage_bytes
                    + warp * (32 * row_bytes)
                    + lane * row_bytes
                    + vector * 16
                )
                swizzled = txl.bitwise_xor(
                    unswizzled,
                    txl.bitwise_and(
                        txl.shift_right(unswizzled, txl.uint32(3)),
                        txl.uint32(112 if c_element_bytes == 4 else 48),
                    ),
                )
                return txl.cast(swizzled - smem_base, "int32")

            group_idx = txl.local_scalar("int32", init=0)
            accum_tile_m = txl.local_scalar("int32", init=0)
            valid = txl.local_scalar("uint32", init=0)
            tile_m_idx = txl.local_scalar("int32", init=0)
            tile_n_idx = txl.local_scalar("int32", init=0)
            batch_idx = txl.local_scalar("int32", init=0)
            scheduler_coords_epilogue(
                work,
                group_idx,
                accum_tile_m,
                valid,
                tile_m_idx,
                tile_n_idx,
                batch_idx,
                dsm_pending_packed,
                dsm_counter,
            )
            with txl.While(valid != txl.uint32(0)):
                if not alpha_is_one:
                    txl.ptx.ld.global_.f32(alpha_value, alpha_ptr.ptr_to([batch_idx]))
                _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                subtile = txl.local_scalar("int32", init=0)
                with txl.While(subtile < epilogue_subtiles):
                    m_subtile = 0 if b_reuse else subtile % (cta_m // 128)
                    n_subtile = subtile if b_reuse else subtile // (cta_m // 128)
                    tmem_address = txl.local_scalar(
                        "uint32",
                        init=(
                            tmem_base
                            + (warp << 21)
                            + acc_state.stage * n_tile * (2 if b_reuse else 1)
                            + m_subtile * n_tile
                            + n_subtile * epi_n
                        ),
                    )
                    if swap:
                        txl.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[values[index] for index in range(16)], tmem_address
                        )
                        txl.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[values[index] for index in range(16, 32)],
                            tmem_address + txl.uint32(1 << 20),
                        )
                    else:
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                            *[values[index] for index in range(32)], tmem_address
                        )
                    scale_and_pack()
                    output_stage = executed_subtiles % c_stages
                    if not swap:
                        for vector in range(epi_n * c_element_bytes // 16):
                            txl.assign(offsets[vector], row_major_output_offset(output_stage, vector))
                        for vector in range(epi_n * c_element_bytes // 16):
                            txl.ptx.st.shared.v4.b32(
                                smem.ptr_to([offsets[vector]]),
                                words[vector * 4],
                                words[vector * 4 + 1],
                                words[vector * 4 + 2],
                                words[vector * 4 + 3],
                            )
                    else:
                        thread = warp * 32 + lane
                        temporary = txl.bitwise_or(
                            txl.bitwise_and(thread << 5, txl.int32(6144)),
                            txl.bitwise_and(thread, txl.int32(40)),
                        )
                        raw_address = txl.bitwise_or(
                            txl.bitwise_or(txl.bitwise_and(thread << 7, txl.int32(896)), temporary << 1),
                            txl.bitwise_and(thread << 6, txl.int32(1024)),
                        )
                        first_unswizzled = (
                            smem_base + c_offset + output_stage * c_stage_bytes + raw_address
                        )
                        first_swizzled = txl.bitwise_xor(
                            first_unswizzled,
                            txl.bitwise_and(
                                txl.shift_right(first_unswizzled, txl.uint32(3)), txl.uint32(112)
                            ),
                        )
                        second_unswizzled = first_unswizzled + 32
                        second_swizzled = txl.bitwise_xor(
                            second_unswizzled,
                            txl.bitwise_and(
                                txl.shift_right(second_unswizzled, txl.uint32(3)), txl.uint32(112)
                            ),
                        )
                        txl.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([txl.cast(first_swizzled - smem_base, "int32")]),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                        txl.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([txl.cast(first_swizzled - smem_base + 2048, "int32")]),
                            words[4],
                            words[5],
                            words[6],
                            words[7],
                        )
                        txl.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([txl.cast(second_swizzled - smem_base, "int32")]),
                            words[8],
                            words[9],
                            words[10],
                            words[11],
                        )
                        txl.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([txl.cast(second_swizzled - smem_base + 2048, "int32")]),
                            words[12],
                            words[13],
                            words[14],
                            words[15],
                        )
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                    with txl.If(warp == 0):
                        with txl.Then():
                            if swap:
                                for row_copy in range(2):
                                    txl.ptx[
                                        "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                                        ".bulk_group.L2::cache_hint"
                                    ](
                                        txl.address_of(c_map),
                                        txl.cast(
                                            tile_m_idx * cta_m + m_subtile * 128 + row_copy * 64,
                                            "int32",
                                        ),
                                        txl.cast(tile_n_idx * n_tile + n_subtile * epi_n, "int32"),
                                        smem.ptr_to(
                                            [
                                                c_offset
                                                + output_stage * c_stage_bytes
                                                + row_copy * 4096
                                            ]
                                        ),
                                        txl.uint64(tma_cache_hint),
                                    )
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                                    ".bulk_group.L2::cache_hint"
                                ](
                                    txl.address_of(c_map),
                                    txl.cast(tile_n_idx * n_tile + n_subtile * epi_n, "int32"),
                                    txl.cast(tile_m_idx * cta_m + m_subtile * 128, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    smem.ptr_to([c_offset + output_stage * c_stage_bytes]),
                                    txl.uint64(tma_cache_hint),
                                )
                            txl.ptx.cp.async_.bulk.commit_group()
                            if signals:
                                txl.assign(
                                    dsm_counter,
                                    txl.bitwise_and(dsm_counter + txl.uint32(1), txl.uint32(0xFF)),
                                )
                            txl.ptx.cp.async_.bulk.wait_group.read(c_stages - 1)
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                    publish_ready_signals(False)
                    txl.assign(executed_subtiles, executed_subtiles + 1)
                    txl.assign(subtile, subtile + 1)
                if cta_group == 2:
                    txl.ptx.mbarrier.arrive.shared__cluster.b64(
                        acc_empty_leader.ptr_to([acc_state.stage]), txl.uint32(1)
                    )
                else:
                    txl.ptx.mbarrier.arrive.shared.b64(
                        acc_pipe.empty.ptr_to([acc_state.stage]), txl.uint32(1)
                    )
                _advance(acc_state)
                advance_work(work)
                scheduler_coords_epilogue(
                    work,
                    group_idx,
                    accum_tile_m,
                    valid,
                    tile_m_idx,
                    tile_n_idx,
                    batch_idx,
                    dsm_pending_packed,
                    dsm_counter,
                )

            with txl.If(warp == 0):
                with txl.Then():
                    txl.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
            with txl.If(warp == 0):
                with txl.Then():
                    if cta_group == 2:
                        remote_dealloc = txl.local_scalar("uint32")
                        txl.ptx.mapa.shared__cluster.u32(
                            remote_dealloc,
                            txl.cuda.cvta_generic_to_shared(tmem_dealloc.ptr_to([0])),
                            txl.cast(cluster_rank ^ 1, "uint32"),
                        )
                        txl.ptx.mbarrier.arrive.shared__cluster.b64(remote_dealloc, txl.uint32(1))
                        _wait_plain(tmem_dealloc.ptr_to([0]), txl.uint32(0))
                    txl.ptx[f"tcgen05.dealloc.exclusive.cta_group::{cta_group}.sync.aligned.b32"](
                        tmem_base, txl.uint32(tmem_columns)
                    )
            txl.ptx.cp.async_.bulk.wait_group.read(0)
            publish_ready_signals(True)

        required_block_size.__exit__(None, None, None)

    kernel.__annotations__ = {
        "a": txl.gptr[txl.u8, (num_groups * kernel_m * K_dim * input_bits // 8,)],
        "b": txl.gptr[txl.u8, (num_groups * kernel_n * K_dim * input_bits // 8,)],
        "sfa": txl.gptr[
            txl.u8, (num_groups * _ceil_div(kernel_m, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)
        ],
        "sfb": txl.gptr[
            txl.u8, (num_groups * _ceil_div(kernel_n, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)
        ],
        "c": txl.gptr[txl.u8, (num_groups * kernel_m * kernel_n * c_element_bytes,)],
        "masked_m": txl.gptr[txl.i32, (num_groups,)],
        "alpha_ptr": txl.gptr[txl.f32, (num_groups,)],
        "dst_signals": txl.gptr[txl.i32, (num_groups,)],
    }
    return txl.kernel(
        warps=6,
        arch="sm_107a",
        min_blocks_per_sm=1,
        grid=[cluster_m, cluster_n, num_clusters],
        host_prelude=host_prelude,
    )(kernel)


def get_kernel(
    num_groups,
    max_m,
    N,
    K,
    ab_dtype="float4_e2m1fn",
    sf_mode="nvfp4",
    out_dtype="bfloat16",
    alpha=False,
    signals=False,
    tactic=0,
):
    """Return one concrete masked-grouped SM107 specialization."""
    from tirx_kernels.runner import hardware_num_sms

    return _make_kernel(
        num_groups,
        max_m,
        N,
        K,
        ab_dtype,
        sf_mode,
        out_dtype,
        alpha,
        signals,
        tactic,
        hardware_num_sms(216),
    ).func


_SOURCE_BENCHMARK = _SOURCE_ROOT / "benchmarks" / "bench_cute_dsl_blockscaled_gemm.py"


def _physical_u8(torch, tensor):
    return torch.as_strided(tensor, (tensor.numel(),), (1,)).view(torch.uint8)


def _torch_dtype(torch, dtype):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]


@cache
def _source_benchmark():
    source = _SOURCE_ROOT / "flashinfer/gemm/kernels/grouped_gemm_masked_rubin.py"
    if not source.is_file():
        raise RuntimeError(f"FlashInfer source is unavailable: {source}")
    source_root = str(_SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    name = "_tirx_grouped_gemm_masked_rubin_benchmark"
    spec = importlib.util.spec_from_file_location(name, _SOURCE_BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.rubin_impl = importlib.import_module("flashinfer.gemm.kernels.grouped_gemm_masked_rubin")
    source_get_cutlass_dtype = module.get_cutlass_dtype

    def get_cutlass_dtype(dtype):
        if dtype == "float8_e5m3fnu":
            return module.cutlass.FloatNV8E5M3FNU
        return source_get_cutlass_dtype(dtype)

    module.get_cutlass_dtype = get_cutlass_dtype
    module.rubin_impl.get_cutlass_dtype = get_cutlass_dtype
    return module


def _masked_counts(torch, num_groups, max_m, expected, label):
    if "empty_trailing" in label:
        values = [0 if index != 1 else min(max_m, max(1, expected)) for index in range(num_groups)]
    else:
        factors = (0.71, 0.83, 0.97, 1.09, 1.21, 1.29)
        values = [
            min(max_m, max(0, int(expected * factors[index % len(factors)])))
            for index in range(num_groups)
        ]
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _create_source_inputs_low_memory(
    source, torch, num_groups, max_m, N, K, ab_dtype, sf_dtype, sf_vec_size
):
    """Reproduce the frozen generator while releasing dead FP32 staging tensors."""

    def converted_matrix(mode0, mode1, is_mode0_major):
        reference = source.cutlass_torch.matrix(
            num_groups, mode0, mode1, is_mode0_major, source.cutlass.Float32
        )
        _, converted = source.cutlass_torch.cute_tensor_like(
            reference, source.get_cutlass_dtype(ab_dtype), is_dynamic_layout=True, assumed_align=16
        )
        torch.cuda.synchronize()
        del reference
        torch.cuda.empty_cache()
        return converted

    a_torch = converted_matrix(max_m, K, source.a_major == "m")
    b_torch = converted_matrix(N, K, source.b_major == "n")
    if ab_dtype == "float4_e2m1fn":
        m, k, groups = a_torch.shape
        n, _, _ = b_torch.shape
        a_torch = (
            a_torch.permute(2, 0, 1)
            .flatten()[: a_torch.numel() // 2]
            .reshape(groups, m, k // 2)
            .permute(1, 2, 0)
            .clone(memory_format=torch.preserve_format)
        )
        b_torch = (
            b_torch.permute(2, 0, 1)
            .flatten()[: b_torch.numel() // 2]
            .reshape(groups, n, k // 2)
            .permute(1, 2, 0)
            .clone(memory_format=torch.preserve_format)
        )
        torch.cuda.empty_cache()

    # The upstream generator randomizes a write-only C reference before scale
    # factors. Preserve that RNG advancement without retaining or converting C.
    c_reference = source.cutlass_torch.matrix(
        num_groups, max_m, N, source.c_major == "m", source.cutlass.Float32
    )
    del c_reference
    torch.cuda.empty_cache()

    sfa_ref, sfa_tensor, sfa_torch = source.create_scale_factor_tensor(
        num_groups,
        max_m,
        K,
        sf_vec_size,
        source.get_cutlass_dtype(sf_dtype),
        torch.device(f"cuda:{torch.cuda.current_device()}"),
    )
    torch.cuda.synchronize()
    del sfa_ref, sfa_tensor
    torch.cuda.empty_cache()
    sfb_ref, sfb_tensor, sfb_torch = source.create_scale_factor_tensor(
        num_groups,
        N,
        K,
        sf_vec_size,
        source.get_cutlass_dtype(sf_dtype),
        torch.device(f"cuda:{torch.cuda.current_device()}"),
    )
    torch.cuda.synchronize()
    del sfb_ref, sfb_tensor
    torch.cuda.empty_cache()
    return {"a": (a_torch, sfa_torch), "b": (b_torch, sfb_torch)}


def prepare_data(
    num_groups,
    max_m,
    expected_m_per_group,
    N,
    K,
    ab_dtype="float4_e2m1fn",
    sf_mode="nvfp4",
    out_dtype="bfloat16",
    alpha=False,
    signals=False,
    tactic=0,
    label="",
    **_,
):
    _validate_problem(num_groups, max_m, N, K, ab_dtype, sf_mode, out_dtype, alpha, signals, tactic)
    if sf_mode not in _SF_MODES:
        raise ValueError(sf_mode)
    import random

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        raise RuntimeError("grouped_gemm_masked_rubin requires an SM107 CUDA device")
    seed = (num_groups * 1000003 + max_m * 1009 + N * 17 + K) & 0x7FFFFFFF
    random.seed(seed)
    torch.manual_seed(seed)
    source = _source_benchmark()
    sf_dtype, sf_vec_size = _SF_MODES[sf_mode]
    source.ab_dtype = ab_dtype
    source.sf_dtype = sf_dtype
    source.sf_vec_size = sf_vec_size
    source.c_dtype = out_dtype
    created = _create_source_inputs_low_memory(
        source, torch, num_groups, max_m, N, K, ab_dtype, sf_dtype, sf_vec_size
    )
    masked_m = _masked_counts(torch, num_groups, max_m, expected_m_per_group, label)
    output_dtype = _torch_dtype(torch, out_dtype)
    tirx_c_storage = torch.full(
        (num_groups, max_m, N), float("nan"), device="cuda", dtype=output_dtype
    )
    source_c_storage = torch.full_like(tirx_c_storage, float("nan"))
    tirx_c = tirx_c_storage.permute(1, 2, 0)
    source_c = source_c_storage.permute(1, 2, 0)
    alpha_values = torch.linspace(0.5, 1.25, num_groups, device="cuda", dtype=torch.float32)
    tirx_signals = torch.zeros(num_groups, device="cuda", dtype=torch.int32)
    source_signals = torch.zeros_like(tirx_signals)
    return {
        "a": created["a"],
        "b": created["b"],
        "a_raw": _physical_u8(torch, created["a"][0]),
        "b_raw": _physical_u8(torch, created["b"][0]),
        "sfa_raw": _physical_u8(torch, created["a"][1]),
        "sfb_raw": _physical_u8(torch, created["b"][1]),
        "tirx_c": tirx_c,
        "source_c": source_c,
        "tirx_c_raw": _physical_u8(torch, tirx_c),
        "masked_m": masked_m,
        "alpha_values": alpha_values,
        "tirx_signals": tirx_signals,
        "source_signals": source_signals,
        "source": source,
    }


@cache
def _compile_executable(
    num_groups, max_m, N, K, ab_dtype, sf_mode, out_dtype, alpha, signals, tactic
):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(
        get_kernel(num_groups, max_m, N, K, ab_dtype, sf_mode, out_dtype, alpha, signals, tactic)
    )


def _tirx_launch(executable, data):
    def launch():
        executable(
            data["a_raw"],
            data["b_raw"],
            data["sfa_raw"],
            data["sfb_raw"],
            data["tirx_c_raw"],
            data["masked_m"],
            data["alpha_values"],
            data["tirx_signals"],
        )

    launch._keep_alive = data
    return launch


def _source_launch(data, config):
    source = data["source"]
    (_tile_m, tile_n), (_inst_m, _inst_n), cluster = TACTICS[config["tactic"]]
    kwargs = {
        "ab_dtype": config["ab_dtype"],
        "sf_dtype": _SF_MODES[config["sf_mode"]][0],
        "c_dtype": config["out_dtype"],
        "sf_vec_size": _SF_MODES[config["sf_mode"]][1],
        "alpha_dtype": "float32",
        "mma_tiler": (_tile_m, tile_n, 128 if config["ab_dtype"] != "float4_e2m1fn" else 256),
        "mma_inst_shape": (_inst_m, tile_n, 64 if config["ab_dtype"] != "float4_e2m1fn" else 128),
        "cluster_shape_mn": cluster,
    }
    if config["alpha"]:
        kwargs["alpha"] = data["alpha_values"]
    if config["signals"]:
        kwargs["dst_signals"] = data["source_signals"]

    def launch():
        source.rubin_impl._grouped_gemm_nt_masked_sm107(
            lhs=data["a"], rhs=data["b"], out=data["source_c"], masked_m=data["masked_m"], **kwargs
        )

    launch._keep_alive = data
    return launch


def _check_outputs(data, config, with_source):
    import torch

    actual = data["tirx_c"].permute(2, 0, 1)
    if not with_source:
        for group, rows in enumerate(data["masked_m"].tolist()):
            cta_m = TACTICS[config["tactic"]][0][0] // (TACTICS[config["tactic"]][1][0] // 128)
            cluster_m = TACTICS[config["tactic"]][2][0]
            rounded = min(config["max_m"], _ceil_div(rows, cta_m) * cta_m * cluster_m)
            if rounded and not bool(torch.isfinite(actual[group, :rounded].float()).all().item()):
                raise AssertionError(f"TIRx output has poison/nonfinite data in group {group}")
        return {"bitwise": None}
    expected = data["source_c"].permute(2, 0, 1)
    if config["signals"] and not torch.equal(data["tirx_signals"], data["source_signals"]):
        raise AssertionError(
            f"completion signal mismatch: tirx={data['tirx_signals'].tolist()} "
            f"source={data['source_signals'].tolist()}"
        )
    actual_bits = _physical_u8(torch, actual)
    expected_bits = _physical_u8(torch, expected)
    if torch.equal(actual_bits, expected_bits):
        return {"bitwise": True, "max_abs_diff": 0.0}
    differing = int((actual_bits != expected_bits).sum().item())
    both_finite = torch.isfinite(actual.float()) & torch.isfinite(expected.float())
    max_abs = 0.0
    if bool(both_finite.any().item()):
        max_abs = float(
            (actual.float()[both_finite] - expected.float()[both_finite]).abs().max().item()
        )
    raise AssertionError(
        f"grouped_gemm_masked_rubin full-allocation bitwise mismatch: "
        f"differing_bytes={differing}, max_abs_diff={max_abs}"
    )


def _config_dict(**config):
    return {
        "num_groups": int(config["num_groups"]),
        "max_m": int(config["max_m"]),
        "expected_m_per_group": int(config["expected_m_per_group"]),
        "N": int(config["N"]),
        "K": int(config["K"]),
        "ab_dtype": config.get("ab_dtype", "float4_e2m1fn"),
        "sf_mode": config.get("sf_mode", "nvfp4"),
        "out_dtype": config.get("out_dtype", "bfloat16"),
        "alpha": bool(config.get("alpha", False)),
        "signals": bool(config.get("signals", False)),
        "tactic": int(config.get("tactic", 0)),
        "label": config.get("label", ""),
    }


def run_test(**raw_config):
    import torch

    config = _config_dict(**raw_config)
    data = prepare_data(**config)
    executable = _compile_executable(
        *[
            config[name]
            for name in (
                "num_groups",
                "max_m",
                "N",
                "K",
                "ab_dtype",
                "sf_mode",
                "out_dtype",
                "alpha",
                "signals",
                "tactic",
            )
        ]
    )
    tirx = _tirx_launch(executable, data)
    source = _source_launch(data, config)
    tirx()
    source()
    torch.cuda.synchronize()
    result = _check_outputs(data, config, True)
    snapshot = data["tirx_c"].clone()
    signal_snapshot = data["tirx_signals"].clone()
    data["tirx_c"].fill_(float("nan"))
    data["tirx_signals"].zero_()
    tirx()
    torch.cuda.synchronize()
    if not torch.equal(_physical_u8(torch, data["tirx_c"]), _physical_u8(torch, snapshot)):
        raise AssertionError("TIRx repeat launch is not bitwise deterministic")
    if config["signals"] and not torch.equal(data["tirx_signals"], signal_snapshot):
        raise AssertionError("TIRx repeat completion signals are not deterministic")
    return result


def prepare_bench(**raw_config):
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = _config_dict(**raw_config)
    executable = _compile_executable(
        *[
            config[name]
            for name in (
                "num_groups",
                "max_m",
                "N",
                "K",
                "ab_dtype",
                "sf_mode",
                "out_dtype",
                "alpha",
                "signals",
                "tactic",
            )
        ]
    )
    return prepared_gpu_benchmark(run_gpu, {"config": config, "executable": executable})


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **_):
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = prepared["config"]
    data = prepare_data(**config)
    tirx = _tirx_launch(prepared["executable"], data)
    tirx()
    torch.cuda.synchronize()
    with_source = external_references_enabled()
    references = None
    if with_source:
        source = _source_launch(data, config)
        source()
        torch.cuda.synchronize()
        references = {"flashinfer": lambda: source}
    _check_outputs(data, config, with_source)
    torch.cuda.empty_cache()
    return bench(
        {"tirx": tirx},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "TACTICS",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
