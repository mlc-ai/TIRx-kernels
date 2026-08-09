# Copyright (c) 2025 DeepSeek
# Copyright (c) 2026 The TIRX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# This file is a TIRx port of DeepGEMM's
# `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`, together with
# the reachable helpers it instantiates (`scheduler/gemm.cuh`,
# `epilogue/sm100_store_cd{,_swap_ab}.cuh`) and the host-side layout heuristic in
# `csrc/jit_kernels/heuristics/sm100.hpp`.
# See NOTICE and THIRD_PARTY_LICENSES.md for upstream attribution.

"""Shared implementation of DeepGEMM's SM100 FP8/FP4 1D1D GEMM kernel template.

DeepGEMM exposes five host entries (normal, m-grouped contiguous, m-grouped
masked, k-grouped contiguous, batched) that all instantiate the *same* device
template `sm100_fp8_fp4_gemm_1d1d_impl`; the entries differ only in template
arguments, chiefly `kGemmType`.  This module mirrors that structure: one
parameterized kernel builder plus one layout heuristic, with the five registry
modules in this package acting as thin per-entry wrappers.

Module names starting with an underscore are skipped by the kernel registry
scan, so this file carries no ``KERNEL_META``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from typing import Literal

# NOTE: no `deep_gemm` / `torch` import at module import time -- registry
# discovery must succeed on hosts without the reference package.

__all__ = [
    "BLOCK_K",
    "LAYOUT_AD_M",
    "UMMA_K",
    "GemmSpec",
    "GemmType",
    "Major",
    "build_kernel",
    "get_best_config",
]


# --------------------------------------------------------------------------------------
# Source-mirrored enums and fixed constants
# --------------------------------------------------------------------------------------


class GemmType(IntEnum):
    """Mirrors `deep_gemm::GemmType` (common/types.cuh:20)."""

    NORMAL = 0
    M_GROUPED_CONTIGUOUS = 1
    M_GROUPED_MASKED = 2
    K_GROUPED_CONTIGUOUS = 3
    BATCHED = 4
    M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT = 5
    K_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT = 6

    @property
    def is_m_grouped_contiguous(self) -> bool:
        return self in (
            GemmType.M_GROUPED_CONTIGUOUS,
            GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT,
        )

    @property
    def is_k_grouped_contiguous(self) -> bool:
        return self in (
            GemmType.K_GROUPED_CONTIGUOUS,
            GemmType.K_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT,
        )

    @property
    def uses_m_grouped_fast_path(self) -> bool:
        """Layout candidates collapse to a single swap-AB layout (sm100.hpp:32)."""
        return self.is_m_grouped_contiguous or self is GemmType.M_GROUPED_MASKED


class Major(IntEnum):
    """Mirrors `cute::UMMA::Major`."""

    K = 0
    MN = 1


# `BLOCK_K = 128 / get_element_size(MmaKind::MXFP8FP4)`; the MXFP8FP4 element size is 1.
BLOCK_K = 128
UMMA_K = 32
LAYOUT_AD_M = 128
UMMA_STEP_N = 16
NUM_UTCCP_ALIGNED_ELEMS = 128
NUM_EPILOGUE_STAGES = 2
NUM_TMA_STORE_STAGES = 2
NUM_MAX_STAGES = 32
SMEM_CAPACITY = 232448
NUM_NON_EPILOGUE_THREADS = 128
NUM_EPILOGUE_THREADS = 128
NUM_THREADS = NUM_NON_EPILOGUE_THREADS + NUM_EPILOGUE_THREADS

#: `HeuristicsRuntime::kLegacyMKAlignmentForContiguousLayout` (runtime.hpp:10).
LEGACY_MK_ALIGNMENT = 128

#: Element sizes as `c10::elementSize` sees them.  Packed FP4 is stored as `int8`
#: in global memory (2 values per byte) and as one *unpacked* byte per value in
#: shared memory, so it reports 1 either way (`csrc/utils/math.hpp:11`).
_ELEM_SIZE = {"fp8": 1, "fp4": 1, "bf16": 2, "fp32": 4}

AbDtype = Literal["fp8", "fp4"]
CdDtype = Literal["bf16", "fp32"]


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def align_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def get_swizzle_mode(block_size: int, elem_size: int) -> int:
    """`deep_gemm::get_swizzle_mode` (heuristics/utils.hpp:13)."""
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise ValueError(f"unreachable swizzle mode for {block_size=} {elem_size=}")


def get_num_aligned_tmem_cols(num_cols: int) -> int:
    """`utils::get_num_aligned_tmem_cols` (common/utils.cuh:67)."""
    if num_cols > 512:
        raise ValueError(f"too many tensor memory columns: {num_cols}")
    for limit in (32, 64, 128, 256):
        if num_cols <= limit:
            return limit
    return 512


def gran_k_for(b_dtype: str) -> tuple[int, int]:
    """`(gran_k_a, gran_k_b)` for one B dtype (`enumerate_normal`).

    An FP4 B operand is quantized with a 32-element K granularity, everything
    else with 128.  Spec construction and data preparation must agree on this
    or the packed scale factors do not match the descriptor, so both read it
    from here.
    """
    return 128, 32 if b_dtype == "fp4" else 128


def get_theoretical_mk_alignment(expected_m: int | None = None) -> int:
    """`HeuristicsRuntime::get_theoretical_mk_alignment_for_contiguous_layout` (runtime.hpp:47).

    SM100 only; the SM90 branch returns the legacy 128.
    """
    block_m, mma_step = 224, 32
    if expected_m is not None:
        while block_m > 32 and block_m - mma_step >= expected_m:
            block_m -= mma_step
    return block_m


# --------------------------------------------------------------------------------------
# Config structures (mirror `csrc/jit_kernels/heuristics/config.hpp`)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GemmDesc:
    """Mirrors `deep_gemm::GemmDesc`; only the SM100 1D1D fields are kept."""

    gemm_type: GemmType
    m: int
    n: int
    k: int
    num_groups: int
    a_dtype: AbDtype = "fp8"
    b_dtype: AbDtype = "fp8"
    cd_dtype: CdDtype = "bf16"
    major_a: Major = Major.K
    major_b: Major = Major.K
    with_accumulation: bool = False
    num_sms: int = 148
    ensure_zero_padding: bool = True
    # Shape hints used only by the layout scorer.
    expected_m: int = 0
    expected_n: int = 0
    expected_k: int = 0
    expected_num_groups: int = 0
    # `HeuristicsRuntime` globals, made explicit instead of process-global.
    mk_alignment: int = LEGACY_MK_ALIGNMENT
    block_m_multiple_of: int = 1
    block_n_multiple_of: int = 1

    def get_expected_m(self) -> int:
        return self.expected_m if self.expected_m > 0 else self.m

    def get_expected_n(self) -> int:
        return self.expected_n if self.expected_n > 0 else self.n

    def get_expected_num_groups(self) -> int:
        return self.expected_num_groups if self.expected_num_groups > 0 else self.num_groups

    def check_validity(self) -> None:
        if self.a_dtype not in ("fp8", "fp4") or self.b_dtype not in ("fp8", "fp4"):
            raise ValueError(f"invalid AB dtypes: {self.a_dtype}/{self.b_dtype}")
        if self.cd_dtype not in ("bf16", "fp32"):
            raise ValueError(f"invalid CD dtype: {self.cd_dtype}")
        if self.num_sms % 2 != 0:
            raise ValueError(f"num_sms must be even, got {self.num_sms}")


@dataclass(frozen=True)
class Layout:
    swap_ab: bool
    block_m: int
    block_n: int
    block_k: int
    cluster_m: int
    cluster_n: int

    @property
    def cluster_size(self) -> int:
        return self.cluster_m * self.cluster_n


@dataclass(frozen=True)
class StorageConfig:
    load_block_m: int
    load_block_n: int
    store_block_m: int
    store_block_n: int
    swizzle_a_mode: int
    swizzle_b_mode: int
    swizzle_cd_mode: int


@dataclass(frozen=True)
class PipelineConfig:
    smem_size: int
    num_stages: int


@dataclass(frozen=True)
class LaunchConfig:
    num_sms: int
    num_sms_per_cluster: int
    num_threads: int = NUM_THREADS
    num_non_epilogue_threads: int = NUM_NON_EPILOGUE_THREADS
    num_epilogue_threads: int = NUM_EPILOGUE_THREADS


@dataclass(frozen=True)
class GemmConfig:
    layout: Layout
    storage_config: StorageConfig
    pipeline_config: PipelineConfig
    launch_config: LaunchConfig


# --------------------------------------------------------------------------------------
# Layout heuristic -- a transcription of `SM100ArchSpec` + `get_best_config`
# --------------------------------------------------------------------------------------


def _sf_utccp_aligned_block_sizes(block_m: int, block_n: int) -> tuple[int, int]:
    """`SM100ArchSpec::get_sf_uttcp_aligned_block_sizes` for `MmaKind::MXFP8FP4`."""
    return align_up(block_m, NUM_UTCCP_ALIGNED_ELEMS), align_up(block_n, NUM_UTCCP_ALIGNED_ELEMS)


def _get_storage_config(desc: GemmDesc, layout: Layout) -> StorageConfig:
    """`SM100ArchSpec::get_storage_config` (sm100.hpp:150)."""
    load_block_m = layout.block_m // layout.cluster_n
    load_block_n = layout.block_n // layout.cluster_m
    store_block_m = UMMA_STEP_N if layout.swap_ab else min(LAYOUT_AD_M, layout.block_m)
    store_block_n = layout.block_n

    swizzle_a = get_swizzle_mode(
        layout.block_k if desc.major_a is Major.K else load_block_m, _ELEM_SIZE[desc.a_dtype]
    )
    swizzle_b = get_swizzle_mode(
        layout.block_k if desc.major_b is Major.K else load_block_n, _ELEM_SIZE[desc.b_dtype]
    )
    swizzle_cd = get_swizzle_mode(store_block_n, _ELEM_SIZE[desc.cd_dtype])
    return StorageConfig(
        load_block_m, load_block_n, store_block_m, store_block_n, swizzle_a, swizzle_b, swizzle_cd
    )


def _get_layout_candidates(desc: GemmDesc) -> list[Layout]:
    """`SM100ArchSpec::get_layout_candidates` (sm100.hpp:27)."""
    block_k = BLOCK_K

    # Always enable swap A/B (and multicasting if possible) for m-grouped GEMMs.
    if desc.gemm_type.uses_m_grouped_fast_path:
        block_n = 128
        cluster_n = 2 if ceil_div(desc.n, block_n) % 2 == 0 and desc.num_sms % 2 == 0 else 1
        return [Layout(True, desc.mk_alignment, block_n, block_k, 1, cluster_n)]

    candidates: list[Layout] = []
    for swap_ab in (False, True):
        if swap_ab:
            step = math.lcm(16, desc.block_m_multiple_of)
            block_m_candidates = list(range(step, 256 + 1, step))
            block_n_candidates = [128]
        else:
            if desc.m <= 32:
                block_m_candidates = [32]
            elif desc.m <= 64:
                block_m_candidates = [64]
            else:
                block_m_candidates = [128]

            block_n_candidates = []
            if 16 % desc.block_n_multiple_of == 0:
                block_n_candidates.append(16)
            step = math.lcm(32, desc.block_n_multiple_of)
            # For small K, fewer store blocks improve store/compute overlap.
            end = 128 if desc.k <= 256 else 256
            block_n_candidates.extend(range(step, end + 1, step))

        for cluster_m in (1, 2):
            # After swapping, layout A/D can only do on cluster N.
            if swap_ab and cluster_m > 1:
                continue
            for cluster_n in (1, 2):
                if cluster_m * cluster_n > 2:
                    continue
                # Only support layout A/D.
                if not swap_ab and cluster_n > 1:
                    continue
                if desc.num_sms % (cluster_m * cluster_n) != 0:
                    continue

                for block_m in block_m_candidates:
                    swizzle_a_requirement = 128 if desc.a_dtype == "fp4" else 64
                    load_block_m_requirement = (
                        swizzle_a_requirement if desc.major_a is Major.MN else 8
                    )
                    if (block_m // cluster_n) % load_block_m_requirement != 0:
                        continue
                    if ceil_div(desc.m, block_m) % cluster_m != 0:
                        continue

                    for block_n in block_n_candidates:
                        swizzle_b_requirement = 128 if desc.b_dtype == "fp4" else 64
                        load_block_n_requirement = (
                            swizzle_b_requirement if desc.major_b is Major.MN else 8
                        )
                        if (block_n // cluster_m) % load_block_n_requirement != 0:
                            continue
                        if ceil_div(desc.n, block_n) % cluster_n != 0:
                            continue
                        # SwapAB requires block N to be layout A/D' UMMA M.
                        if swap_ab and block_n != LAYOUT_AD_M:
                            continue

                        sf_block_m, sf_block_n = _sf_utccp_aligned_block_sizes(block_m, block_n)
                        tmem_sf_cols = sf_block_m // 32 + sf_block_n // 32
                        umma_n = block_m if swap_ab else block_n
                        if 2 * umma_n + tmem_sf_cols > 512:
                            continue

                        layout = Layout(swap_ab, block_m, block_n, block_k, cluster_m, cluster_n)

                        # When neither A nor B is MN major, 128B swizzle is always feasible.
                        if desc.major_a is Major.K or desc.major_b is Major.K:
                            storage = _get_storage_config(desc, layout)
                            if storage.swizzle_a_mode != 128 or storage.swizzle_b_mode != 128:
                                continue

                        candidates.append(layout)

    if not candidates:
        raise ValueError(f"no layout candidate for {desc}")
    return candidates


def _get_pipeline_config(desc: GemmDesc, layout: Layout, storage: StorageConfig) -> PipelineConfig:
    """`SM100ArchSpec::get_pipeline_config` (sm100.hpp:176)."""
    cd_elem = _ELEM_SIZE[desc.cd_dtype]
    smem_cd = (
        storage.store_block_m * storage.store_block_n * cd_elem * 2
        if layout.swap_ab
        else storage.store_block_m * storage.swizzle_cd_mode * 2
    )
    # Space is reserved for the maximum stage count regardless of the chosen one.
    smem_barriers = NUM_MAX_STAGES * 8 * 3 + 2 * 8 * 2 + 8
    smem_tmem_ptr = 4

    smem_a_per_stage = storage.load_block_m * layout.block_k * _ELEM_SIZE[desc.a_dtype]
    smem_b_per_stage = storage.load_block_n * layout.block_k * _ELEM_SIZE[desc.b_dtype]
    sf_block_m, sf_block_n = _sf_utccp_aligned_block_sizes(layout.block_m, layout.block_n)
    smem_sfa_per_stage = sf_block_m * 4
    smem_sfb_per_stage = sf_block_n * 4

    smem_extra = smem_cd + smem_barriers + smem_tmem_ptr
    smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage
    num_stages = min((SMEM_CAPACITY - smem_extra) // smem_per_stage, NUM_MAX_STAGES)
    return PipelineConfig(smem_extra + num_stages * smem_per_stage, num_stages)


def _layout_info(desc: GemmDesc, layout: Layout) -> tuple[int, int, Layout]:
    """`SM100ArchSpec::get_layout_info` -> (num_waves, last_wave_util, layout)."""
    num_blocks = (
        ceil_div(desc.get_expected_m(), layout.block_m)
        * ceil_div(desc.get_expected_n(), layout.block_n)
        * desc.get_expected_num_groups()
    )
    num_waves = ceil_div(num_blocks, desc.num_sms)
    num_last_blocks = num_blocks % desc.num_sms
    last_wave_util = desc.num_sms if num_last_blocks == 0 else num_last_blocks
    return num_waves, last_wave_util, layout


def _better(a: tuple[int, int, Layout], b: tuple[int, int, Layout]) -> bool:
    """`SM100ArchSpec::compare` -- true when `a` is preferred over `b`."""
    a_waves, a_util, a_layout = a
    b_waves, b_util, b_layout = b
    if (a_waves == 1 or b_waves == 1) and a_waves != b_waves:
        return a_waves < b_waves
    if a_layout.cluster_size != b_layout.cluster_size:
        return a_layout.cluster_size > b_layout.cluster_size
    if a_waves != b_waves:
        return a_waves < b_waves
    if a_util != b_util:
        return a_util > b_util
    if a_layout.block_m + a_layout.block_n != b_layout.block_m + b_layout.block_n:
        return a_layout.block_m + a_layout.block_n < b_layout.block_m + b_layout.block_n
    return a_layout.block_m * a_layout.block_n < b_layout.block_m * b_layout.block_n


def get_best_config(desc: GemmDesc) -> GemmConfig:
    """`deep_gemm::get_best_config<SM100ArchSpec>` (heuristics/common.hpp:14)."""
    desc.check_validity()

    candidates = _get_layout_candidates(desc)
    best_layout = candidates[0]
    best_info = _layout_info(desc, best_layout)
    for candidate in candidates[1:]:
        info = _layout_info(desc, candidate)
        if _better(info, best_info):
            best_layout, best_info = candidate, info

    storage = _get_storage_config(desc, best_layout)
    pipeline = _get_pipeline_config(desc, best_layout, storage)
    launch = LaunchConfig(desc.num_sms, best_layout.cluster_size)
    return GemmConfig(best_layout, storage, pipeline, launch)


# --------------------------------------------------------------------------------------
# Kernel specialization
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GemmSpec:
    """One instantiation of `sm100_fp8_fp4_gemm_1d1d_impl`.

    Field order and meaning mirror the C++ template parameter list emitted by
    `SM100FP8FP4Gemm1D1DRuntime::generate_impl`
    (`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:46`).  `shape_m/n/k`
    are 0 when the dimension stays dynamic (DeepGEMM's `compiled_dims`).
    """

    major_a: Major
    major_b: Major
    gran_k_a: int
    gran_k_b: int
    k_alignment: int
    shape_m: int
    shape_n: int
    shape_k: int
    block_m: int
    block_n: int
    block_k: int
    num_groups: int
    swizzle_a_mode: int
    swizzle_b_mode: int
    swizzle_cd_mode: int
    num_stages: int
    num_non_epilogue_threads: int
    num_epilogue_threads: int
    num_multicast: int
    is_multicast_on_a: bool
    num_sms: int
    swap_ab: bool
    ensure_zero_padding: bool
    gemm_type: GemmType
    with_accumulation: bool
    a_dtype: AbDtype
    b_dtype: AbDtype
    cd_dtype: CdDtype
    smem_size: int

    # --- derived quantities (all `constexpr` in the source) ---

    @property
    def umma_m(self) -> int:
        return LAYOUT_AD_M * self.num_multicast

    @property
    def umma_n(self) -> int:
        return self.block_m if self.swap_ab else self.block_n

    @property
    def load_block_m(self) -> int:
        return self.block_m // (self.num_multicast if self.is_multicast_on_a else 1)

    @property
    def load_block_n(self) -> int:
        return self.block_n // (1 if self.is_multicast_on_a else self.num_multicast)

    @property
    def store_block_m(self) -> int:
        return UMMA_STEP_N if self.swap_ab else min(self.block_m, LAYOUT_AD_M)

    @property
    def store_block_n(self) -> int:
        return self.block_n if self.swap_ab else self.swizzle_cd_mode // _ELEM_SIZE[self.cd_dtype]

    @property
    def num_umma_store_threads(self) -> int:
        return self.num_epilogue_threads if self.swap_ab else self.store_block_m

    @property
    def sf_block_m(self) -> int:
        return align_up(self.block_m, NUM_UTCCP_ALIGNED_ELEMS)

    @property
    def sf_block_n(self) -> int:
        return align_up(self.block_n, NUM_UTCCP_ALIGNED_ELEMS)

    @property
    def num_sfa_stages_per_load(self) -> int:
        return 1 if self.gran_k_a == 32 else 4

    @property
    def num_sfb_stages_per_load(self) -> int:
        return 1 if self.gran_k_b == 32 else 4

    @property
    def num_accum_tmem_cols(self) -> int:
        return self.umma_n * NUM_EPILOGUE_STAGES

    @property
    def tmem_start_col_of_sfa(self) -> int:
        return self.num_accum_tmem_cols

    @property
    def tmem_start_col_of_sfb(self) -> int:
        return self.num_accum_tmem_cols + self.sf_block_m // 32

    @property
    def num_tmem_cols(self) -> int:
        return get_num_aligned_tmem_cols(
            self.num_accum_tmem_cols + self.sf_block_m // 32 + self.sf_block_n // 32
        )

    @property
    def may_have_tail_k_block(self) -> bool:
        """`kMayHaveTailKBlock` (kernel :338)."""
        if self.gemm_type.is_k_grouped_contiguous:
            return self.k_alignment % self.block_k != 0
        return self.shape_k == 0 or self.shape_k % self.block_k != 0

    # --- shared-memory byte offsets (kernel :130-159) ---

    @property
    def smem_cd_size_per_stage(self) -> int:
        return self.store_block_m * self.store_block_n * _ELEM_SIZE[self.cd_dtype]

    @property
    def smem_cd_size(self) -> int:
        return self.smem_cd_size_per_stage * NUM_TMA_STORE_STAGES

    @property
    def smem_a_size_per_stage(self) -> int:
        return self.load_block_m * self.block_k * _ELEM_SIZE[self.a_dtype]

    @property
    def smem_b_size_per_stage(self) -> int:
        return self.load_block_n * self.block_k * _ELEM_SIZE[self.b_dtype]

    @property
    def smem_sfa_size_per_stage(self) -> int:
        return self.sf_block_m * 4

    @property
    def smem_sfb_size_per_stage(self) -> int:
        return self.sf_block_n * 4

    @property
    def smem_a_offset(self) -> int:
        return self.smem_cd_size

    @property
    def smem_b_offset(self) -> int:
        return self.smem_a_offset + self.num_stages * self.smem_a_size_per_stage

    @property
    def smem_sfa_offset(self) -> int:
        return self.smem_b_offset + self.num_stages * self.smem_b_size_per_stage

    @property
    def smem_sfb_offset(self) -> int:
        return self.smem_sfa_offset + self.num_stages * self.smem_sfa_size_per_stage

    @property
    def smem_barrier_offset(self) -> int:
        return self.smem_sfb_offset + self.num_stages * self.smem_sfb_size_per_stage

    @property
    def smem_tmem_ptr_offset(self) -> int:
        return self.smem_barrier_offset + 8 * (self.num_stages * 3 + NUM_EPILOGUE_STAGES * 2)


def make_spec(
    desc: GemmDesc,
    config: GemmConfig,
    *,
    gran_k_a: int,
    gran_k_b: int,
    k_alignment: int,
    compiled_dims: str = "nk",
) -> GemmSpec:
    """Combine a descriptor and a chosen config into a kernel specialization.

    `compiled_dims` selects which of m/n/k are baked in as compile-time
    constants, matching DeepGEMM's `get_compiled_dim`.
    """
    layout, storage, pipeline, launch = (
        config.layout,
        config.storage_config,
        config.pipeline_config,
        config.launch_config,
    )
    return GemmSpec(
        major_a=desc.major_a,
        major_b=desc.major_b,
        gran_k_a=gran_k_a,
        gran_k_b=gran_k_b,
        k_alignment=k_alignment,
        shape_m=desc.m if "m" in compiled_dims else 0,
        shape_n=desc.n if "n" in compiled_dims else 0,
        shape_k=desc.k if "k" in compiled_dims else 0,
        block_m=layout.block_m,
        block_n=layout.block_n,
        block_k=layout.block_k,
        num_groups=desc.num_groups,
        swizzle_a_mode=storage.swizzle_a_mode,
        swizzle_b_mode=storage.swizzle_b_mode,
        swizzle_cd_mode=storage.swizzle_cd_mode,
        num_stages=pipeline.num_stages,
        num_non_epilogue_threads=launch.num_non_epilogue_threads,
        num_epilogue_threads=launch.num_epilogue_threads,
        num_multicast=layout.cluster_size,
        is_multicast_on_a=layout.cluster_n > 1,
        num_sms=launch.num_sms,
        swap_ab=layout.swap_ab,
        ensure_zero_padding=desc.ensure_zero_padding,
        gemm_type=desc.gemm_type,
        with_accumulation=desc.with_accumulation,
        a_dtype=desc.a_dtype,
        b_dtype=desc.b_dtype,
        cd_dtype=desc.cd_dtype,
        smem_size=pipeline.smem_size,
    )


# --------------------------------------------------------------------------------------
# Kernel builder
# --------------------------------------------------------------------------------------


def build_kernel(spec: GemmSpec):
    """Build the TIRx `PrimFunc` for one `sm100_fp8_fp4_gemm_1d1d_impl` instantiation.

    The body lives in the sibling `_sm100_fp8_fp4_gemm_1d1d_kernel` module; the
    import is deferred because that module imports this one for its constants.
    """
    from ._sm100_fp8_fp4_gemm_1d1d_kernel import build_kernel as _build

    return _build(spec)


__all__ += ["GemmConfig", "GemmDesc", "Layout", "make_spec"]


# --------------------------------------------------------------------------------------
# Deterministic per-group M/K derivation
# --------------------------------------------------------------------------------------


def make_actual_ms(num_groups: int, expected_m_per_group: int, seed: int) -> list[int]:
    """Per-group row counts, mirroring `generators.generate_m_grouped_contiguous`.

    DeepGEMM draws `int(expected_m_per_group * uniform(0.7, 1.3))` per group from
    the seeded global RNG; we use a private `random.Random` so a config is
    reproducible without depending on process-wide RNG state.
    """
    import random

    rng = random.Random(seed)
    return [int(expected_m_per_group * rng.uniform(0.7, 1.3)) for _ in range(num_groups)]


def make_aligned_ms(
    num_groups: int, expected_m_per_group: int, seed: int, alignment: int
) -> list[int]:
    return [align_up(m, alignment) for m in make_actual_ms(num_groups, expected_m_per_group, seed)]


def make_actual_ks(num_groups: int, expected_k_per_group: int, seed: int) -> list[int]:
    """Per-group K lengths, mirroring `generators.enumerate_k_grouped_contiguous`."""
    import random

    rng = random.Random(seed)
    return [max(1, int(expected_k_per_group * rng.uniform(0.7, 1.3))) for _ in range(num_groups)]


__all__ += ["make_actual_ks", "make_actual_ms", "make_aligned_ms"]


# --------------------------------------------------------------------------------------
# Host-side TMA descriptors
# --------------------------------------------------------------------------------------
#
# Transcribed from `csrc/jit_kernels/impls/runtime_utils.hpp`: `make_tma_2d_desc`,
# `make_tma_3d_desc`, `make_tma_{a,b,cd,sf}_desc`.  The kernel receives only these
# descriptors -- A, B, SFA, SFB and C/D are never passed as buffers -- so getting
# the box and stride arithmetic right is what makes the port addressable at all.

#: `CU_TENSOR_MAP_*` enum values used below.
_TMA_INTERLEAVE_NONE = 0
_TMA_L2_PROMOTION_256B = 3
_TMA_FLOAT_OOB_FILL_NONE = 0
_TMA_DATA_TYPE_16U4_ALIGN16B = 14

#: `mode_into_tensor_map_swizzle` (runtime_utils.hpp:94).
_SWIZZLE_ENUM = {0: 0, 16: 0, 32: 1, 64: 2, 128: 3}

_CUDA_DRIVER = None


def _cuda_driver():
    """`libcuda` with `cuTensorMapEncodeTiled` bound, for dtypes TVM cannot name."""
    global _CUDA_DRIVER
    if _CUDA_DRIVER is not None:
        return _CUDA_DRIVER
    import ctypes
    import ctypes.util

    driver = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    if driver.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    driver.cuTensorMapEncodeTiled.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuTensorMapEncodeTiled.restype = ctypes.c_int
    _CUDA_DRIVER = driver
    return driver


class AlignedTensorMap:
    """A 128-byte-aligned `CUtensorMap` staging buffer.

    Keep the Python object alive for as long as the kernel may be launched: `ptr`
    points into `_storage`.
    """

    __slots__ = ("_storage", "ptr")

    def __init__(self) -> None:
        import ctypes

        self._storage = ctypes.create_string_buffer(128 + 128)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 127) & ~127)


def tma_aligned_size(value: int, elem_size: int) -> int:
    """`get_tma_aligned_size`: round the MN extent up to a 16-byte multiple."""
    return align_up(value * elem_size, 16) // elem_size


def _encode_2d(
    tensor,
    *,
    gmem_inner: int,
    gmem_outer: int,
    smem_inner: int,
    smem_outer: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
    is_packed_fp4: bool = False,
) -> AlignedTensorMap:
    """`make_tma_2d_desc` (runtime_utils.hpp:113)."""
    import ctypes

    import tvm

    elem_size = tensor.element_size()
    if swizzle_mode != 0:
        # Swizzling fixes the inner box extent; the caller's value is advisory.
        smem_inner = swizzle_mode // elem_size
    if is_packed_fp4 and gmem_inner % 128 != 0:
        raise ValueError(
            f"unpacked-SMEM FP4 needs a global inner dim that is a multiple of 128, got {gmem_inner}"
        )

    desc = AlignedTensorMap()
    if is_packed_fp4:
        gmem_dims = (ctypes.c_uint64 * 2)(gmem_inner, gmem_outer)
        gmem_strides = (ctypes.c_uint64 * 1)(gmem_outer_stride * elem_size)
        smem_dims = (ctypes.c_uint32 * 2)(smem_inner, smem_outer)
        elem_strides = (ctypes.c_uint32 * 2)(1, 1)
        status = _cuda_driver().cuTensorMapEncodeTiled(
            desc.ptr,
            _TMA_DATA_TYPE_16U4_ALIGN16B,
            ctypes.c_uint32(2),
            ctypes.c_void_p(int(tensor.data_ptr())),
            gmem_dims,
            gmem_strides,
            smem_dims,
            elem_strides,
            _TMA_INTERLEAVE_NONE,
            _SWIZZLE_ENUM[swizzle_mode],
            _TMA_L2_PROMOTION_256B,
            _TMA_FLOAT_OOB_FILL_NONE,
        )
        if status != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed with {status}")
        return desc

    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        desc.ptr,
        _tvm_dtype_name(tensor),
        2,
        ctypes.c_void_p(int(tensor.data_ptr())),
        int(gmem_inner),
        int(gmem_outer),
        int(gmem_outer_stride * elem_size),
        int(smem_inner),
        int(smem_outer),
        1,
        1,
        _TMA_INTERLEAVE_NONE,
        _SWIZZLE_ENUM[swizzle_mode],
        _TMA_L2_PROMOTION_256B,
        _TMA_FLOAT_OOB_FILL_NONE,
    )
    return desc


def _encode_3d(
    tensor,
    *,
    gmem_inner: int,
    gmem_outer: int,
    batch: int,
    smem_inner: int,
    smem_outer: int,
    stride_outer: int,
    stride_batch: int,
    swizzle_mode: int,
) -> AlignedTensorMap:
    """`make_tma_3d_desc` (runtime_utils.hpp:187).

    Dims are `(inner, outer, batch)`; the two strides are the element strides
    between consecutive outer entries and between consecutive batches.  The batch
    box extent is always 1 -- one TMA moves one batch's tile.
    """
    import ctypes

    import tvm

    elem_size = tensor.element_size()
    if swizzle_mode != 0:
        smem_inner = swizzle_mode // elem_size

    desc = AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        desc.ptr,
        _tvm_dtype_name(tensor),
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        int(gmem_inner),
        int(gmem_outer),
        int(batch),
        int(stride_outer * elem_size),
        int(stride_batch * elem_size),
        int(smem_inner),
        int(smem_outer),
        1,
        1,
        1,
        1,
        _TMA_INTERLEAVE_NONE,
        _SWIZZLE_ENUM[swizzle_mode],
        _TMA_L2_PROMOTION_256B,
        _TMA_FLOAT_OOB_FILL_NONE,
    )
    return desc


def make_tma_a_desc_3d(tensor, spec: GemmSpec, *, shape_m: int, shape_k: int, batch: int):
    """Rank-3 A descriptor for `GemmType::Batched` (`sm100_..._1d1d.hpp:415`)."""
    inner, outer = (shape_k, shape_m) if spec.major_a is Major.K else (shape_m, shape_k)
    box_inner, box_outer = (
        (spec.block_k, spec.load_block_m)
        if spec.major_a is Major.K
        else (spec.load_block_m, spec.block_k)
    )
    return _encode_3d(
        tensor,
        gmem_inner=inner,
        gmem_outer=outer,
        batch=batch,
        smem_inner=box_inner,
        smem_outer=box_outer,
        stride_outer=tensor.stride(1 if spec.major_a is Major.K else 2),
        stride_batch=tensor.stride(0),
        swizzle_mode=spec.swizzle_a_mode,
    )


def make_tma_b_desc_3d(tensor, spec: GemmSpec, *, shape_n: int, shape_k: int, batch: int):
    """Rank-3 B descriptor for `GemmType::Batched`."""
    inner, outer = (shape_k, shape_n) if spec.major_b is Major.K else (shape_n, shape_k)
    box_inner, box_outer = (
        (spec.block_k, spec.load_block_n)
        if spec.major_b is Major.K
        else (spec.load_block_n, spec.block_k)
    )
    return _encode_3d(
        tensor,
        gmem_inner=inner,
        gmem_outer=outer,
        batch=batch,
        smem_inner=box_inner,
        smem_outer=box_outer,
        stride_outer=tensor.stride(1 if spec.major_b is Major.K else 2),
        stride_batch=tensor.stride(0),
        swizzle_mode=spec.swizzle_b_mode,
    )


def make_tma_cd_desc_3d(tensor, spec: GemmSpec, *, shape_m: int, shape_n: int, batch: int):
    """Rank-3 C/D descriptor: dims `(n, m, batch)`, box `(store_n, store_m, 1)`."""
    return _encode_3d(
        tensor,
        gmem_inner=shape_n,
        gmem_outer=shape_m,
        batch=batch,
        smem_inner=spec.store_block_n,
        smem_outer=spec.store_block_m,
        stride_outer=tensor.stride(1),
        stride_batch=tensor.stride(0),
        swizzle_mode=spec.swizzle_cd_mode,
    )


def _tvm_dtype_name(tensor) -> str:
    """TVM's name for a torch dtype.

    Every dtype this kernel passes to a tensor map is spelled the same by both
    sides, so the mapping is `str(dtype)` minus the namespace rather than a
    table that silently `KeyError`s on the next dtype added.
    """
    return str(tensor.dtype).removeprefix("torch.")


def make_tma_a_desc(tensor, spec: GemmSpec, *, shape_m: int, shape_k: int, num_groups: int = 1):
    """`make_tma_a_desc`: box `(BLOCK_K, load_block_m)` for K-major, swapped for MN-major."""
    if num_groups > 1 and spec.major_a is not Major.K:
        raise ValueError("grouped A must be K-major")
    inner, outer = (
        (shape_k, shape_m * num_groups)
        if spec.major_a is Major.K
        else (shape_m, shape_k * num_groups)
    )
    box_inner, box_outer = (
        (spec.block_k, spec.load_block_m)
        if spec.major_a is Major.K
        else (spec.load_block_m, spec.block_k)
    )
    stride = tensor.stride(-2 if spec.major_a is Major.K else -1)
    return _encode_2d(
        tensor,
        gmem_inner=inner,
        gmem_outer=outer,
        smem_inner=box_inner,
        smem_outer=box_outer,
        gmem_outer_stride=stride,
        swizzle_mode=spec.swizzle_a_mode,
        is_packed_fp4=spec.a_dtype == "fp4",
    )


def make_tma_b_desc(tensor, spec: GemmSpec, *, shape_n: int, shape_k: int, num_groups: int = 1):
    """`make_tma_b_desc`: the group count always multiplies the *outer* extent."""
    inner, outer = (shape_k, shape_n) if spec.major_b is Major.K else (shape_n, shape_k)
    box_inner, box_outer = (
        (spec.block_k, spec.load_block_n)
        if spec.major_b is Major.K
        else (spec.load_block_n, spec.block_k)
    )
    stride = tensor.stride(-2 if spec.major_b is Major.K else -1)
    return _encode_2d(
        tensor,
        gmem_inner=inner,
        gmem_outer=outer * num_groups,
        smem_inner=box_inner,
        smem_outer=box_outer,
        gmem_outer_stride=stride,
        swizzle_mode=spec.swizzle_b_mode,
        is_packed_fp4=spec.b_dtype == "fp4",
    )


def make_tma_cd_desc(tensor, spec: GemmSpec, *, shape_m: int, shape_n: int, num_groups: int = 1):
    """`make_tma_cd_desc`: C/D is always N-major, box `(store_block_n, store_block_m)`."""
    return _encode_2d(
        tensor,
        gmem_inner=shape_n,
        gmem_outer=shape_m * num_groups,
        smem_inner=spec.store_block_n,
        smem_outer=spec.store_block_m,
        gmem_outer_stride=tensor.stride(-2),
        swizzle_mode=spec.swizzle_cd_mode,
    )


def make_tma_sf_desc(
    tensor, *, shape_mn: int, shape_k: int, block_mn: int, gran_k: int, num_groups: int = 1
):
    """`make_tma_sf_desc`: MN-major, unswizzled, one `int32` per four `ue8m0`."""
    import torch

    elem_size = tensor.element_size()
    aligned_mn = tma_aligned_size(shape_mn, elem_size)
    packed = 1 if tensor.dtype == torch.float32 else 4
    return _encode_2d(
        tensor,
        gmem_inner=aligned_mn,
        gmem_outer=ceil_div(shape_k, gran_k * packed) * num_groups,
        smem_inner=block_mn,
        smem_outer=1,
        gmem_outer_stride=aligned_mn,
        swizzle_mode=0,
    )


__all__ += [
    "AlignedTensorMap",
    "make_tma_a_desc",
    "make_tma_a_desc_3d",
    "make_tma_b_desc",
    "make_tma_b_desc_3d",
    "make_tma_cd_desc",
    "make_tma_cd_desc_3d",
    "make_tma_sf_desc",
    "tma_aligned_size",
]


# --------------------------------------------------------------------------------------
# Compilation and launch
# --------------------------------------------------------------------------------------


@cache
def compile_spec(spec: GemmSpec):
    """Compile one specialization.  `GemmSpec` is frozen, so it keys the cache."""
    import tvm

    from ._sm100_fp8_fp4_gemm_1d1d_kernel import build_kernel

    tags = ["blockIdx.x"]
    if spec.num_multicast > 1:
        tags.append("clusterCtaIdx.x")
    tags += ["threadIdx.x", "tirx.use_dyn_shared_memory"]
    func = build_kernel(spec).with_attr("tirx.kernel_launch_params", tags)
    return tvm.compile(
        tvm.IRModule({"main": func}),
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100f"}),
        tir_pipeline="tirx",
    )


def build_launch(
    spec: GemmSpec,
    *,
    a,
    b,
    sfa,
    sfb,
    d,
    shape_m: int,
    shape_n: int,
    shape_k: int,
    grouped_layout=None,
    num_groups: int = 1,
    sf_num_groups_a: int = 1,
    sf_num_groups_b: int = 1,
):
    """Return a no-argument closure that issues exactly one kernel launch.

    Compilation and every TensorMap encode happen here, outside the closure, so a
    benchmark measures only the launch.  The descriptor objects are captured by
    the closure: they own the `CUtensorMap` storage the kernel dereferences.
    """
    import torch

    executable = compile_spec(spec)
    if spec.gemm_type is GemmType.BATCHED:
        # A, B and C/D are rank-3 here and A/D are permuted views, so the
        # descriptors must read their real strides rather than assume packing.
        tmap_a = make_tma_a_desc_3d(a, spec, shape_m=shape_m, shape_k=shape_k, batch=num_groups)
        tmap_b = make_tma_b_desc_3d(b, spec, shape_n=shape_n, shape_k=shape_k, batch=num_groups)
        tmap_cd = make_tma_cd_desc_3d(d, spec, shape_m=shape_m, shape_n=shape_n, batch=num_groups)
    else:
        tmap_a = make_tma_a_desc(
            a, spec, shape_m=shape_m, shape_k=shape_k, num_groups=1 if a.dim() == 2 else num_groups
        )
        tmap_b = make_tma_b_desc(
            b, spec, shape_n=shape_n, shape_k=shape_k, num_groups=1 if b.dim() == 2 else num_groups
        )
        tmap_cd = make_tma_cd_desc(
            d, spec, shape_m=shape_m, shape_n=shape_n, num_groups=1 if d.dim() == 2 else num_groups
        )
    # The k-grouped scale factors are packed per group, so their row count is
    # `sum(ceil_div(k_i, gran_k * 4))`, not `ceil_div(sum_k, gran_k * 4)`.  The
    # source derives the descriptor's K extent from the tensor itself
    # (`sm100_fp8_fp4_gemm_1d1d.hpp:365`).
    if spec.gemm_type.is_k_grouped_contiguous:
        sf_shape_k_a = int(sfa.size(0)) * spec.gran_k_a * 4
        sf_shape_k_b = int(sfb.size(0)) * spec.gran_k_b * 4
    else:
        sf_shape_k_a = shape_k
        sf_shape_k_b = shape_k
    tmap_sfa = make_tma_sf_desc(
        sfa,
        shape_mn=shape_m,
        shape_k=sf_shape_k_a,
        block_mn=spec.block_m,
        gran_k=spec.gran_k_a,
        num_groups=sf_num_groups_a,
    )
    tmap_sfb = make_tma_sf_desc(
        sfb,
        shape_mn=shape_n,
        shape_k=sf_shape_k_b,
        block_mn=spec.block_n,
        gran_k=spec.gran_k_b,
        num_groups=sf_num_groups_b,
    )
    layout = (
        grouped_layout
        if grouped_layout is not None
        else torch.zeros(1, device=d.device, dtype=torch.int32)
    )
    maps = (tmap_a, tmap_b, tmap_sfa, tmap_sfb, tmap_cd)
    # The kernel declares `grouped_layout` with a symbolic length, so the host
    # passes it explicitly; computing it here keeps it out of the timed closure.
    layout_len = int(layout.numel())

    def launch():
        executable.mod(
            layout,
            layout_len,
            shape_m,
            shape_n,
            shape_k,
            maps[0].ptr,
            maps[1].ptr,
            maps[2].ptr,
            maps[3].ptr,
            maps[4].ptr,
        )

    return launch


__all__ += ["build_launch", "compile_spec"]
