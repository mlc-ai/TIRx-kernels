# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Persistent SM100 block-scaled GEMM with an FP32 absolute-maximum output.

Upstream source:
``python/cudnn/gemm/cutedsl/dense/amax/dense_blockscaled_gemm_persistent_amax.py``.
"""

import json
import os
from functools import cache
from itertools import combinations, product

import tirx_kernels.kern as K

_TRY_WAIT_TICKS = 10_000_000
_SMEM_CAPACITY = 232_448
_MAX_ACTIVE_CLUSTERS = {1: 148, 2: 74, 4: 33, 8: 15, 16: 7}
_AB_DTYPES = ("float4_e2m1fn", "float8_e4m3fn", "float8_e5m2")
_SF_MODES = (("float8_e8m0fnu", 16), ("float8_e8m0fnu", 32), ("float8_e4m3fn", 16))
_C_DTYPES = ("bfloat16", "float16", "float32", "float4_e2m1fn", "float8_e4m3fn", "float8_e5m2")
_CLUSTERS = tuple(product((1, 2, 4), repeat=2))
_MODE_KEYS = (
    "ab_dtype",
    "sf_dtype",
    "sf_vec_size",
    "c_dtype",
    "a_major",
    "b_major",
    "c_major",
    "mma_tiler_mn",
    "cluster_shape_mn",
    "L",
)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align_up(value, alignment):
    return _ceil_div(value, alignment) * alignment


def _descriptor_base(ldo, sdo, swizzle):
    """Fold the SM100 shared descriptor fields except its 14-bit address."""
    arrangement_type = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = 0
    value |= (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement_type & 0x7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


def _instruction_descriptor(M, N, ab_dtype, sf_dtype, a_major, b_major):
    """Fold the static fields of the source block-scaled MMA descriptor."""
    if M != 128 or N not in (128, 256):
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


def _descriptor_with_address(base, shared_address):
    address_field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(base), address_field)


def _dtype_bits(dtype):
    return {
        "float4_e2m1fn": 4,
        "float8_e4m3fn": 8,
        "float8_e5m2": 8,
        "bfloat16": 16,
        "float16": 16,
        "float32": 32,
    }[dtype]


def _valid_mode(mode):
    fp4_ab = mode["ab_dtype"] == "float4_e2m1fn"
    fp4_c = mode["c_dtype"] == "float4_e2m1fn"
    fp8_c = mode["c_dtype"].startswith("float8_")
    if fp4_ab:
        if mode["a_major"] != "k" or mode["b_major"] != "k":
            return False
    else:
        if mode["sf_dtype"] != "float8_e8m0fnu" or mode["sf_vec_size"] != 32:
            return False
        if fp4_c or fp8_c:
            return False
    if fp4_c and (not fp4_ab or mode["c_major"] != "n"):
        return False
    if (
        mode["mma_tiler_mn"][1] == 256
        and mode["sf_vec_size"] == 16
        and mode["c_dtype"] in {"float32", "float16", "bfloat16"}
    ):
        return False
    return True


def _structural_modes(include_batch=True):
    modes = []
    batches = (1, 2) if include_batch else (1,)
    for ab_dtype in _AB_DTYPES:
        a_majors = ("k",) if ab_dtype == "float4_e2m1fn" else ("k", "m")
        b_majors = ("k",) if ab_dtype == "float4_e2m1fn" else ("k", "n")
        for (
            sf_dtype,
            sf_vec_size,
        ), c_dtype, a_major, b_major, c_major, tile_n, cluster_shape_mn, batch in product(
            _SF_MODES, _C_DTYPES, a_majors, b_majors, ("m", "n"), (128, 256), _CLUSTERS, batches
        ):
            mode = {
                "ab_dtype": ab_dtype,
                "sf_dtype": sf_dtype,
                "sf_vec_size": sf_vec_size,
                "c_dtype": c_dtype,
                "a_major": a_major,
                "b_major": b_major,
                "c_major": c_major,
                "mma_tiler_mn": (128, tile_n),
                "cluster_shape_mn": cluster_shape_mn,
                "L": batch,
            }
            if _valid_mode(mode):
                modes.append(mode)
    return sorted(modes, key=_mode_label)


def _mode_label(mode):
    short = {
        "float4_e2m1fn": "f4",
        "float8_e4m3fn": "e4",
        "float8_e5m2": "e5",
        "float8_e8m0fnu": "e8",
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
    }
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    return (
        f"{short[mode['ab_dtype']]}_{short[mode['sf_dtype']]}v{mode['sf_vec_size']}_"
        f"{short[mode['c_dtype']]}_{mode['a_major']}{mode['b_major']}_{mode['c_major']}_"
        f"t{tile_m}x{tile_n}_c{cluster_m}x{cluster_n}_l{mode['L']}"
    )


def _pair_tokens(mode):
    return frozenset(
        (left, json.dumps(mode[left]), right, json.dumps(mode[right]))
        for left, right in combinations(_MODE_KEYS, 2)
    )


def _minimal_pairwise_modes():
    candidates = _structural_modes(include_batch=True)
    cover = [_pair_tokens(mode) for mode in candidates]
    uncovered = set().union(*cover)
    selected = []
    while uncovered:
        best_coverage = max(len(tokens & uncovered) for tokens in cover)
        best = min(
            (
                index
                for index in range(len(candidates))
                if len(cover[index] & uncovered) == best_coverage
            ),
            key=lambda index: _mode_label(candidates[index]),
        )
        if not (cover[best] & uncovered):
            raise AssertionError("pairwise structural coverage did not converge")
        selected.append(best)
        uncovered.difference_update(cover[best])
    counts = {}
    for index in selected:
        for token in cover[index]:
            counts[token] = counts.get(token, 0) + 1
    kept = []
    for index in reversed(selected):
        if all(counts[token] > 1 for token in cover[index]):
            for token in cover[index]:
                counts[token] -= 1
        else:
            kept.append(index)
    return [candidates[index] for index in reversed(kept)]


def _make_case(mode, *, shape=(256, 256, 256), prefix="pair"):
    M, N, K_dim = shape
    return {
        "label": f"{prefix}_m{M}_n{N}_k{K_dim}_{_mode_label(mode)}",
        "M": M,
        "N": N,
        "K": K_dim,
        **mode,
    }


def _correctness_configs():
    configs = [_make_case(mode) for mode in _minimal_pairwise_modes()]
    tail_modes = [
        mode
        for mode in _structural_modes(include_batch=True)
        if (
            (mode["ab_dtype"], mode["mma_tiler_mn"][1], mode["L"])
            in {
                ("float4_e2m1fn", 128, 1),
                ("float4_e2m1fn", 256, 2),
                ("float8_e4m3fn", 128, 2),
                ("float8_e5m2", 256, 1),
            }
        )
    ]
    chosen = {}
    for mode in tail_modes:
        key = mode["ab_dtype"], mode["mma_tiler_mn"][1], mode["L"]
        chosen.setdefault(key, mode)
    configs.extend(
        _make_case(mode, shape=(144, 160, 160), prefix="tail") for mode in chosen.values()
    )
    return sorted(configs, key=lambda config: config["label"])


def _performance_tokens(mode):
    ab_bits = _dtype_bits(mode["ab_dtype"])
    c_bits = _dtype_bits(mode["c_dtype"])
    n_tile = mode["mma_tiler_mn"][1]
    k_tile = 256 if ab_bits == 4 else 128
    c_epi_n = 64 if c_bits == 4 else 32
    ab_stage_bytes = (
        128 * k_tile * ab_bits // 8
        + n_tile * k_tile * ab_bits // 8
        + 128 * k_tile // mode["sf_vec_size"]
        + n_tile * k_tile // mode["sf_vec_size"]
    )
    c_stage_bytes = 128 * c_epi_n * c_bits // 8
    ab_stages = (_SMEM_CAPACITY - (1024 + 2 * c_stage_bytes)) // ab_stage_bytes
    c_stages = (
        2
        + (_SMEM_CAPACITY - ab_stage_bytes * ab_stages - (1024 + 2 * c_stage_bytes))
        // c_stage_bytes
    )
    return frozenset(
        {
            ("ab", mode["ab_dtype"]),
            ("sf", mode["sf_dtype"], mode["sf_vec_size"]),
            ("mma", "fp4" if ab_bits == 4 else "fp8", mode["sf_vec_size"]),
            ("c", mode["c_dtype"]),
            ("a_tma", mode["ab_dtype"], mode["a_major"]),
            ("b_tma", mode["ab_dtype"], mode["b_major"]),
            ("store", mode["c_dtype"], mode["c_major"]),
            ("tile_n", n_tile),
            ("cluster", mode["cluster_shape_mn"]),
            ("acc_stages", 1 if n_tile == 256 else 2),
            ("ab_stages", ab_stages),
            ("c_stages", c_stages),
        }
    )


def _performance_representatives():
    candidates = _structural_modes(include_batch=False)
    cover = [_performance_tokens(mode) for mode in candidates]
    uncovered = set().union(*cover)
    selected = []
    while uncovered:
        best_coverage = max(len(tokens & uncovered) for tokens in cover)
        best = min(
            (
                index
                for index in range(len(candidates))
                if len(cover[index] & uncovered) == best_coverage
            ),
            key=lambda index: _mode_label(candidates[index]),
        )
        selected.append(best)
        uncovered.difference_update(cover[best])
    counts = {}
    for index in selected:
        for token in cover[index]:
            counts[token] = counts.get(token, 0) + 1
    kept = []
    for index in reversed(selected):
        if all(counts[token] > 1 for token in cover[index]):
            for token in cover[index]:
                counts[token] -= 1
        else:
            kept.append(index)
    return [candidates[index] for index in reversed(kept)]


def _benchmark_configs():
    shapes = tuple((size, size, size) for size in (1024, 2048, 4096, 8192))
    configs = []
    for representative_index, base_mode in enumerate(_performance_representatives()):
        for M, N, K_dim in shapes:
            for batch in (1, 2):
                mode = {**base_mode, "L": batch}
                configs.append(
                    _make_case(mode, shape=(M, N, K_dim), prefix=f"perf{representative_index:02d}")
                )
    return sorted(configs, key=lambda config: config["label"])


KERNEL_META = {
    "name": "cudnn_sm100_dense_blockscaled_gemm_persistent_amax",
    "category": "cudnn",
    "compute_capability": 10,
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
    a_major,
    b_major,
    c_major,
    mma_tiler_mn,
    cluster_shape_mn,
):
    if ab_dtype == "uint8" or c_dtype == "uint8":
        raise ValueError("the non-running public uint8 packed-FP4 alias is not supported")
    if sf_dtype == "int8":
        sf_dtype = "float8_e8m0fnu"
    mma_tiler_mn = tuple(mma_tiler_mn)
    cluster_shape_mn = tuple(cluster_shape_mn)
    mode = {
        "ab_dtype": ab_dtype,
        "sf_dtype": sf_dtype,
        "sf_vec_size": sf_vec_size,
        "c_dtype": c_dtype,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "L": L,
    }
    if min(M, N, K_dim, L) <= 0:
        raise ValueError("M/N/K/L must be positive")
    if not _valid_mode(mode):
        raise ValueError(f"unsupported source specialization: {_mode_label(mode)}")
    ab_bits = _dtype_bits(ab_dtype)
    c_bits = _dtype_bits(c_dtype)
    alignment = 128
    if (M if a_major == "m" else K_dim) % (alignment // ab_bits):
        raise ValueError("A's contiguous dimension must be 16-byte aligned")
    if (N if b_major == "n" else K_dim) % (alignment // ab_bits):
        raise ValueError("B's contiguous dimension must be 16-byte aligned")
    if (M if c_major == "m" else N) % (alignment // c_bits):
        raise ValueError("C's contiguous dimension must be 16-byte aligned")
    if ab_dtype == "float4_e2m1fn" and mma_tiler_mn[1] == 256 and K_dim <= 128:
        raise ValueError("FP4 N=256 instruction tiles require K > 128")

    m_tile = 128
    n_tile = mma_tiler_mn[1]
    k_tile = 256 if ab_dtype == "float4_e2m1fn" else 128
    cluster_m, cluster_n = cluster_shape_mn
    cluster_size = cluster_m * cluster_n
    m_tiles = _ceil_div(M, m_tile)
    n_tiles = _ceil_div(N, n_tile)
    cluster_m_tiles = _ceil_div(m_tiles, cluster_m)
    cluster_n_tiles = _ceil_div(n_tiles, cluster_n)
    cluster_work = cluster_m_tiles * cluster_n_tiles * L
    num_clusters = min(cluster_work, _MAX_ACTIVE_CLUSTERS[cluster_size])
    k_tiles = _ceil_div(K_dim, k_tile)

    acc_stages = 1 if n_tile == 256 else 2
    a_stage_bytes = m_tile * k_tile * ab_bits // 8
    b_stage_bytes = n_tile * k_tile * ab_bits // 8
    sfa_stage_bytes = m_tile * k_tile // sf_vec_size
    sfb_stage_bytes = n_tile * k_tile // sf_vec_size
    ab_stage_bytes = a_stage_bytes + b_stage_bytes + sfa_stage_bytes + sfb_stage_bytes
    epi_m = 128
    epi_n = 64 if c_dtype == "float4_e2m1fn" else 32
    epilogue_subtiles = n_tile // epi_n
    epilogue_values = 64 if c_dtype == "float4_e2m1fn" else 32
    c_packed_words = 32 if c_bits == 32 else 16 if c_bits == 16 else 8
    c_store_vectors = c_packed_words // 4
    c_stage_bytes = epi_m * epi_n * c_bits // 8
    ab_stages = (_SMEM_CAPACITY - (1024 + 2 * c_stage_bytes)) // ab_stage_bytes
    c_stages = (
        2
        + (_SMEM_CAPACITY - ab_stage_bytes * ab_stages - (1024 + 2 * c_stage_bytes))
        // c_stage_bytes
    )
    use_derived_c_stage = (
        c_dtype in {"float8_e4m3fn", "float8_e5m2"}
        and c_stages == 2
        and epilogue_subtiles % 2 == 0
        and cluster_work > 16 * num_clusters
    )
    if min(ab_stages, c_stages) <= 0:
        raise ValueError("source stage heuristic exceeds the SM100 shared-memory capacity")

    ab_full_offset = 0
    ab_empty_offset = ab_full_offset + ab_stages * 8
    acc_full_offset = ab_empty_offset + ab_stages * 8
    acc_empty_offset = acc_full_offset + acc_stages * 8
    tmem_dealloc_offset = acc_empty_offset + acc_stages * 8
    tmem_ptr_offset = tmem_dealloc_offset + 8
    c_offset = 1024
    a_offset = c_offset + c_stages * c_stage_bytes
    b_offset = a_offset + ab_stages * a_stage_bytes
    sfa_offset = b_offset + ab_stages * b_stage_bytes
    sfb_offset = sfa_offset + ab_stages * sfa_stage_bytes
    amax_offset = sfb_offset + ab_stages * sfb_stage_bytes
    shared_bytes = _align_up(amax_offset + 16, 1024)
    if shared_bytes > _SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {shared_bytes} exceeds {_SMEM_CAPACITY}")

    a_desc_base = _descriptor_base(ldo=1 if a_major == "k" else 0, sdo=64, swizzle=3)
    b_desc_base = _descriptor_base(
        ldo=(
            b_stage_bytes // 32 if b_major == "n" and n_tile == 256 else 1 if b_major == "k" else 0
        ),
        sdo=64,
        swizzle=3,
    )
    sf_desc_base = _descriptor_base(ldo=1, sdo=8, swizzle=0)
    instr_desc = _instruction_descriptor(128, n_tile, ab_dtype, sf_dtype, a_major, b_major)
    acc_columns = acc_stages * n_tile
    sfa_tmem_column = acc_columns
    sfa_chunks = sfa_stage_bytes // 512
    sfb_chunks = sfb_stage_bytes // 512
    sfb_tmem_column = sfa_tmem_column + sfa_chunks * 4
    tma_cache_hint = 0
    a_cluster_piece = (k_tile if a_major == "m" else m_tile) // cluster_n
    b_cluster_piece = (k_tile if b_major == "n" else n_tile) // cluster_m
    b_tma_copies = n_tile // 128 if b_major == "n" else 1
    b_tma_copy_bytes = b_stage_bytes // b_tma_copies
    sf_k_box = k_tile // (4 * sf_vec_size)
    sfa_piece_values = 256 * sf_k_box // cluster_n
    sfb_n_box = n_tile // 128
    sfb_piece_values = 256 * sf_k_box * sfb_n_box // cluster_m

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        sfa = params["sfa"]
        sfb = params["sfb"]
        c = params["c"]
        a_map = K.stack_alloca("tensormap", 1)
        b_map = K.stack_alloca("tensormap", 1)
        sfa_map = K.stack_alloca("tensormap", 1)
        sfb_map = K.stack_alloca("tensormap", 1)
        c_map = K.stack_alloca("tensormap", 1)

        def encode(descriptor, dtype, rank, data, *fields):
            K.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

        a_contiguous_bytes = (M if a_major == "m" else K_dim) * ab_bits // 8
        b_contiguous_bytes = (N if b_major == "n" else K_dim) * ab_bits // 8
        a_fields = (
            (M, K_dim, L, a_contiguous_bytes, M * K_dim * ab_bits // 8, m_tile, a_cluster_piece, 1)
            if a_major == "m"
            else (
                K_dim,
                M,
                L,
                a_contiguous_bytes,
                M * K_dim * ab_bits // 8,
                k_tile,
                a_cluster_piece,
                1,
            )
        )
        b_fields = (
            (
                N,
                K_dim,
                L,
                b_contiguous_bytes,
                N * K_dim * ab_bits // 8,
                n_tile // b_tma_copies,
                b_cluster_piece,
                1,
            )
            if b_major == "n"
            else (
                K_dim,
                N,
                L,
                b_contiguous_bytes,
                N * K_dim * ab_bits // 8,
                k_tile,
                b_cluster_piece,
                1,
            )
        )
        ab_tail = (1, 1, 1, 0, 3, 2, 0)
        if ab_dtype == "float4_e2m1fn":
            ab_tail += (13,)
        encode(a_map, ab_dtype, 3, a.data, *a_fields, *ab_tail)
        encode(b_map, ab_dtype, 3, b.data, *b_fields, *ab_tail)
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
        c_contiguous = (M if c_major == "m" else N) * c_bits // 8
        c_batch_stride = M * N * c_bits // 8
        if c_major == "m":
            c_row_copies = c_bits // 8
            c_fields = (M, N, L, c_contiguous, c_batch_stride, epi_m // c_row_copies, epi_n, 1)
            c_swizzle = 3
        else:
            c_fields = (N, M, L, c_contiguous, c_batch_stride, epi_n, epi_m, 1)
            c_swizzle = 3 if c_bits == 32 else 2 if c_bits == 16 else 1
        c_tail = (1, 1, 1, 0, c_swizzle, 2, 0)
        if c_dtype == "float4_e2m1fn":
            c_tail += (13,)
        encode(c_map, c_dtype, 3, c.data, *c_fields, *c_tail)
        return a_map, b_map, sfa_map, sfb_map, c_map

    def kernel(a, b, sfa, sfb, c, amax, *, host):
        del a, b, sfa, sfb, c
        a_map, b_map, sfa_map, sfb_map, c_map = host
        block_x, block_y, cluster_work_id = K.cta_id()
        cluster_x, cluster_y = K.cta_id_in_cluster(
            [cluster_m, cluster_n], preferred=[cluster_m, cluster_n]
        )
        cluster_rank = cluster_x + cluster_m * cluster_y
        del block_x, block_y
        warp = K.warp_id()
        lane = K.lane_id()

        roles = K.specialize(chain_dispatch=True)
        if c_dtype == "float32":
            epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3], regs=64)
        elif c_dtype == "bfloat16" and cluster_size == 1:
            epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3], regs=56)
        else:
            epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])

        smem = K.alloc_buffer((shared_bytes,), K.u8, scope="shared.dyn", align=1024)
        protocol_pool = K.smem_pool(base=smem)
        ab_pipe = K.Pipeline(
            protocol_pool, ab_stages, full="tma", empty="tcgen05", leader=K.bool(False)
        )
        acc_pipe = K.Pipeline(
            protocol_pool,
            acc_stages,
            full="tcgen05",
            empty="mbar",
            init_empty=4,
            leader=K.bool(False),
        )
        if protocol_pool.bytes != tmem_dealloc_offset:
            raise AssertionError("protocol storage offsets changed")
        tmem_dealloc = protocol_pool.alloc((1,), K.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), K.u32, align=4)
        if protocol_pool.bytes != tmem_ptr_offset + 4:
            raise AssertionError("protocol storage header changed")
        del tmem_dealloc

        with tma_role:
            K.ptx.prefetch.tensormap(K.address_of(a_map))
            K.ptx.prefetch.tensormap(K.address_of(b_map))
            K.ptx.prefetch.tensormap(K.address_of(sfa_map))
            K.ptx.prefetch.tensormap(K.address_of(sfb_map))
            K.ptx.prefetch.tensormap(K.address_of(c_map))

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
                                ab_pipe.empty.ptr_to([stage]), K.uint32(cluster_m + cluster_n - 1)
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
                                acc_pipe.empty.ptr_to([stage]), K.uint32(4)
                            )
        K.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            K.ptx.barrier.cluster.arrive.relaxed()

        smem_base = K.local_scalar("uint32")
        K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        cluster_smem_base_u64 = K.local_scalar("uint64")
        K.ptx.cvta.to.shared__cluster.u64(cluster_smem_base_u64, smem.ptr_to([0]))
        cluster_smem_base = K.local_scalar("uint32", init=K.cast(cluster_smem_base_u64, "uint32"))
        a_shared_base = K.local_scalar("uint32", init=smem_base + a_offset)
        b_shared_base = K.local_scalar("uint32", init=smem_base + b_offset)
        a_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, a_shared_base)
        )
        b_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, b_shared_base)
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
        for peer_m in range(cluster_m):
            K.assign(
                b_mcast_mask,
                K.bitwise_or(
                    b_mcast_mask, K.uint32(1) << K.cast(peer_m + cluster_m * cluster_y, "uint32")
                ),
            )
        ab_consumer_mask = K.local_scalar("uint32", init=K.bitwise_or(a_mcast_mask, b_mcast_mask))
        acc_producer_mask = K.local_scalar(
            "uint32", init=K.uint32(1) << K.cast(cluster_rank, "uint32")
        )

        if cluster_size > 1:
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0))

        def scheduler_coords(work):
            if c_dtype == "float4_e2m1fn":
                cluster_m_idx = work % cluster_m_tiles
                quotient = work // cluster_m_tiles
            elif cluster_m_tiles & (cluster_m_tiles - 1) == 0:
                cluster_m_idx = K.bitwise_and(work, K.int32(cluster_m_tiles - 1))
                quotient = K.shift_right(work, K.uint32(cluster_m_tiles.bit_length() - 1))
            else:
                cluster_m_idx = work % cluster_m_tiles
                quotient = work // cluster_m_tiles
            if c_dtype == "float4_e2m1fn":
                cluster_n_idx = quotient % cluster_n_tiles
                batch_idx = quotient // cluster_n_tiles
            elif L == 1:
                cluster_n_idx = quotient
                batch_idx = 0
            elif cluster_n_tiles & (cluster_n_tiles - 1) == 0:
                cluster_n_idx = K.bitwise_and(quotient, K.int32(cluster_n_tiles - 1))
                batch_idx = K.shift_right(quotient, K.uint32(cluster_n_tiles.bit_length() - 1))
            else:
                cluster_n_idx = quotient % cluster_n_tiles
                batch_idx = quotient // cluster_n_tiles
            tile_m_idx = cluster_m_idx * cluster_m + cluster_x
            tile_n_idx = cluster_n_idx * cluster_n + cluster_y
            return tile_m_idx, tile_n_idx, batch_idx

        def advance_work(work):
            K.assign(work, work + num_clusters)

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
                    tma_elected = _elected()
                    with K.If(tma_elected):
                        with K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([tma_state.stage]), K.uint32(ab_stage_bytes)
                            )
                    with K.If(tma_elected):
                        with K.Then():
                            a_coord_m = tile_m_idx * m_tile
                            a_coord_k = count * k_tile
                            if a_major == "m":
                                a_coord_k = a_coord_k + cluster_y * a_cluster_piece
                            else:
                                a_coord_m = a_coord_m + cluster_y * a_cluster_piece
                            a_coord_0 = a_coord_m if a_major == "m" else a_coord_k
                            a_coord_1 = a_coord_k if a_major == "m" else a_coord_m
                            if cluster_n == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([a_offset + tma_state.stage * a_stage_bytes]),
                                    K.address_of(a_map),
                                    K.cast(a_coord_0, "int32"),
                                    K.cast(a_coord_1, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
                                    cluster_smem_base
                                    + a_offset
                                    + tma_state.stage * a_stage_bytes
                                    + cluster_y * (a_stage_bytes // cluster_n),
                                    K.address_of(a_map),
                                    K.cast(a_coord_0, "int32"),
                                    K.cast(a_coord_1, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(a_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                    with K.If(tma_elected):
                        with K.Then():
                            for b_copy in range(b_tma_copies):
                                b_coord_n = tile_n_idx * n_tile
                                if b_major == "n":
                                    b_coord_n = b_coord_n + b_copy * 128
                                b_coord_k = count * k_tile
                                if b_major == "n":
                                    b_coord_k = b_coord_k + cluster_x * b_cluster_piece
                                else:
                                    b_coord_n = b_coord_n + cluster_x * b_cluster_piece
                                b_coord_0 = b_coord_n if b_major == "n" else b_coord_k
                                b_coord_1 = b_coord_k if b_major == "n" else b_coord_n
                                b_smem_offset = (
                                    b_offset
                                    + tma_state.stage * b_stage_bytes
                                    + b_copy * b_tma_copy_bytes
                                )
                                if cluster_m == 1:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                    ](
                                        smem.ptr_to([b_smem_offset]),
                                        K.address_of(b_map),
                                        K.cast(b_coord_0, "int32"),
                                        K.cast(b_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint"
                                    ](
                                        cluster_smem_base
                                        + b_smem_offset
                                        + cluster_x * (b_tma_copy_bytes // cluster_m),
                                        K.address_of(b_map),
                                        K.cast(b_coord_0, "int32"),
                                        K.cast(b_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.cast(b_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )
                    with K.If(tma_elected):
                        with K.Then():
                            sfa_linear = cluster_y * sfa_piece_values
                            sfa_coord_0 = sfa_linear % 256
                            sfa_coord_1 = count * sf_k_box + sfa_linear // 256
                            if cluster_n == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([sfa_offset + tma_state.stage * sfa_stage_bytes]),
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
                                    cluster_smem_base
                                    + sfa_offset
                                    + tma_state.stage * sfa_stage_bytes
                                    + cluster_y * (sfa_stage_bytes // cluster_n),
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(a_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                    with K.If(tma_elected):
                        with K.Then():
                            sfb_linear = cluster_x * sfb_piece_values
                            sfb_coord_0 = sfb_linear % 256
                            sfb_quotient = sfb_linear // 256
                            sfb_coord_1 = count * sf_k_box + sfb_quotient % sf_k_box
                            sfb_coord_2 = tile_n_idx * sfb_n_box + sfb_quotient // sf_k_box
                            if cluster_m == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([sfb_offset + tma_state.stage * sfb_stage_bytes]),
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
                                    cluster_smem_base
                                    + sfb_offset
                                    + tma_state.stage * sfb_stage_bytes
                                    + cluster_x * (sfb_stage_bytes // cluster_m),
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
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
            K.ptx.bar.sync(K.uint32(3), K.uint32(160))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            sfa_shared_base = K.local_scalar("uint32", init=smem_base + sfa_offset)
            sfb_shared_base = K.local_scalar("uint32", init=smem_base + sfb_offset)
            sfa_descriptor = K.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, sfa_shared_base)
            )
            sfb_descriptor = K.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, sfb_shared_base)
            )
            mma_state = K.PipelineState(ab_stages, phase=0)
            acc_state = K.PipelineState(acc_stages, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            count = K.local_scalar("int32")
            speculative = K.local_scalar("uint32")
            accumulate = K.local_scalar("uint32")
            with K.While(work < cluster_work):
                K.assign(count, 0)
                K.assign(speculative, K.uint32(1))
                with K.If(count < k_tiles):
                    with K.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                        )
                _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)
                K.assign(accumulate, K.uint32(0))
                with K.While(count < k_tiles):
                    _wait_plain_if_needed(
                        ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase, speculative
                    )
                    for sf_chunk in range(sfa_chunks):
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
                                    K.cast(tmem_base + sfa_tmem_column + sf_chunk * 4, "uint32"),
                                    sfa_descriptor
                                    + K.cast(
                                        mma_state.stage * (sfa_stage_bytes // 16) + sf_chunk * 32,
                                        "uint64",
                                    ),
                                )
                    for sf_chunk in range(sfb_chunks):
                        sfb_shared_chunk = (sf_chunk % sfb_n_box) * sf_k_box + sf_chunk // sfb_n_box
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
                                    K.cast(tmem_base + sfb_tmem_column + sf_chunk * 4, "uint32"),
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
                            sfb_scale_offset = kblock * 4 * (n_tile // 128)
                            sf_selector = 0
                        elif ab_dtype == "float4_e2m1fn":
                            sfa_scale_offset = (kblock // 2) * 4
                            sfb_scale_offset = (kblock // 2) * 4 * (n_tile // 128)
                            sf_selector = (kblock % 2) * 0x80000000
                        else:
                            sfa_scale_offset = 0
                            sfb_scale_offset = 0
                            sf_selector = kblock * 0x40000000
                        sfa_tmem_addr = K.cast(
                            tmem_base + sfa_tmem_column + sfa_scale_offset + K.uint32(sf_selector),
                            "uint32",
                        )
                        sfb_tmem_addr = K.cast(
                            tmem_base + sfb_tmem_column + sfb_scale_offset + K.uint32(sf_selector),
                            "uint32",
                        )
                        runtime_instr_desc = K.bitwise_and(
                            K.uint32(instr_desc), K.uint32(0x9FFFFFCF)
                        )
                        runtime_instr_desc = K.bitwise_or(
                            runtime_instr_desc,
                            K.bitwise_and(
                                K.shift_right(sfa_tmem_addr, K.uint32(1)), K.uint32(0x60000000)
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
                                mma_kind = "mxf4nvf4" if ab_dtype == "float4_e2m1fn" else "mxf8f6f4"
                                K.ptx[
                                    "tcgen05.mma.cta_group::1.kind::"
                                    + mma_kind
                                    + ".block_scale.block"
                                    + str(sf_vec_size)
                                ](
                                    K.cast(tmem_base + acc_state.stage * n_tile, "uint32"),
                                    a_descriptor
                                    + K.cast(
                                        mma_state.stage * (a_stage_bytes // 16)
                                        + kblock * (2 if a_major == "k" else 256),
                                        "uint64",
                                    ),
                                    b_descriptor
                                    + K.cast(
                                        mma_state.stage * (b_stage_bytes // 16)
                                        + kblock * (2 if b_major == "k" else 256),
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
                            if cluster_size == 1:
                                K.ptx[
                                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                    ".shared::cluster.b64"
                                ](ab_pipe.empty.ptr_to([mma_state.stage]))
                            else:
                                K.ptx[
                                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                    ".shared::cluster.multicast::cluster.b64"
                                ](
                                    ab_pipe.empty.ptr_to([mma_state.stage]),
                                    K.cast(ab_consumer_mask, "uint16"),
                                )
                    _advance(mma_state)
                    K.assign(count, count + 1)
                    K.assign(speculative, K.uint32(1))
                    with K.If(count < k_tiles):
                        with K.Then():
                            _try_wait_acquire(
                                speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                            )
                with K.If(_elected()):
                    with K.Then():
                        if cluster_size == 1:
                            K.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                ".shared::cluster.b64"
                            ](acc_pipe.full.ptr_to([acc_state.stage]))
                        else:
                            K.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                ".shared::cluster.multicast::cluster.b64"
                            ](
                                acc_pipe.full.ptr_to([acc_state.stage]),
                                K.cast(acc_producer_mask, "uint16"),
                            )
                _advance(acc_state)
                advance_work(work)
            with K.If((cluster_rank % 2) == 0):
                with K.Then():
                    for _ in range(acc_stages - 1):
                        _advance(acc_state)
                    _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            with K.If(warp == 0):
                with K.Then():
                    K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                        tmem_slot.ptr_to([0]), K.uint32(512)
                    )
            K.ptx.bar.sync(K.uint32(3), K.uint32(160))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            acc_state = K.PipelineState(acc_stages, phase=0)
            if use_derived_c_stage:
                c_stage = None
            else:
                c_stage = K.local_scalar("int32", init=0)
            work = K.local_scalar("int32", init=cluster_work_id)
            values = K.alloc_local((epilogue_values,), "float32")
            absolute = K.alloc_local((epilogue_values,), "float32")
            maxima = K.alloc_local((epilogue_values // 2,), "float32")
            packed_pair_dtype = (
                "uint8"
                if c_dtype == "float4_e2m1fn"
                else "uint16"
                if c_dtype.startswith("float8_")
                else "uint32"
            )
            packed_pairs = K.alloc_local((epilogue_values // 2,), packed_pair_dtype)
            packed = K.alloc_local((epilogue_values,), "uint32")
            if c_major == "n":
                c_store_offsets = K.alloc_local((c_store_vectors,), "int32")
            tile_amax = K.local_scalar("float32")
            warp_amax = K.local_scalar("float32")

            def emit_epilogue_subtile(subtile):
                if use_derived_c_stage:
                    c_stage_index = K.bitwise_and(subtile, K.int32(1))
                else:
                    c_stage_index = c_stage
                tmem_load_addr = K.local_scalar(
                    "uint32",
                    init=tmem_base + (warp << 21) + acc_state.stage * n_tile + subtile * epi_n,
                )
                if c_dtype == "float4_e2m1fn":
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"](
                        *[values[index] for index in range(64)], tmem_load_addr
                    )
                elif c_major == "m" and c_bits <= 16:
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[values[index] for index in range(16)], tmem_load_addr
                    )
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[values[index] for index in range(16, 32)],
                        tmem_load_addr + K.uint32(1 << 20),
                    )
                else:
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[values[index] for index in range(32)], tmem_load_addr
                    )
                if c_major == "n":
                    bytes_per_thread = c_packed_words * 4
                    swizzle_mask = {32: 16, 64: 48, 128: 112}[bytes_per_thread]
                    for vector in range(c_store_vectors):
                        unswizzled = (
                            smem_base
                            + c_offset
                            + c_stage_index * c_stage_bytes
                            + warp * (32 * bytes_per_thread)
                            + lane * bytes_per_thread
                            + vector * 16
                        )
                        swizzled = K.bitwise_xor(
                            unswizzled,
                            K.bitwise_and(
                                K.shift_right(unswizzled, K.uint32(3)), K.uint32(swizzle_mask)
                            ),
                        )
                        K.assign(c_store_offsets[vector], K.cast(swizzled - smem_base, "int32"))
                for index in range(epilogue_values):
                    K.ptx.abs.f32(absolute[index], values[index])
                ternary_groups = epilogue_values // 3
                for index in range(ternary_groups):
                    K.ptx.max.NaN.f32(
                        maxima[index],
                        absolute[index * 3],
                        absolute[index * 3 + 1],
                        absolute[index * 3 + 2],
                    )
                remainder = epilogue_values % 3
                if remainder == 1:
                    K.ptx.max.NaN.f32(
                        maxima[ternary_groups - 1],
                        maxima[ternary_groups - 1],
                        absolute[epilogue_values - 1],
                    )
                    reduction_width = ternary_groups
                elif remainder == 2:
                    K.ptx.max.NaN.f32(
                        maxima[ternary_groups],
                        absolute[epilogue_values - 2],
                        absolute[epilogue_values - 1],
                    )
                    reduction_width = ternary_groups + 1
                else:
                    reduction_width = ternary_groups
                while reduction_width > 1:
                    ternary_groups = reduction_width // 3
                    for index in range(ternary_groups):
                        K.ptx.max.NaN.f32(
                            maxima[index],
                            maxima[index * 3],
                            maxima[index * 3 + 1],
                            maxima[index * 3 + 2],
                        )
                    remainder = reduction_width % 3
                    if remainder == 1:
                        K.ptx.max.NaN.f32(
                            maxima[ternary_groups - 1],
                            maxima[ternary_groups - 1],
                            maxima[reduction_width - 1],
                        )
                        reduction_width = ternary_groups
                    elif remainder == 2:
                        K.ptx.max.NaN.f32(
                            maxima[ternary_groups],
                            maxima[reduction_width - 2],
                            maxima[reduction_width - 1],
                        )
                        reduction_width = ternary_groups + 1
                    else:
                        reduction_width = ternary_groups
                K.ptx.max.NaN.f32(maxima[0], maxima[0], K.float32(0.0))
                K.ptx.max.f32(tile_amax, tile_amax, maxima[0])

                if c_dtype == "float32":
                    for index in range(32):
                        K.assign(packed[index], K.reinterpret("uint32", values[index]))
                    packed_words = 32
                elif c_dtype in {"float16", "bfloat16"}:
                    for index in range(16):
                        if c_dtype == "float16":
                            K.ptx.cvt.rn.f16x2.f32(
                                packed[index], values[index * 2 + 1], values[index * 2]
                            )
                        else:
                            K.ptx.cvt.rn.bf16x2.f32(
                                packed[index], values[index * 2 + 1], values[index * 2]
                            )
                    packed_words = 16
                elif c_dtype in {"float8_e4m3fn", "float8_e5m2"}:
                    for index in range(16):
                        if c_dtype == "float8_e4m3fn":
                            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                packed_pairs[index], values[index * 2 + 1], values[index * 2]
                            )
                        else:
                            K.ptx.cvt.rn.satfinite.e5m2x2.f32(
                                packed_pairs[index], values[index * 2 + 1], values[index * 2]
                            )
                    if c_major != "n":
                        for index in range(8):
                            K.assign(
                                packed[index],
                                K.bitwise_or(
                                    K.bitwise_and(
                                        K.cast(packed_pairs[index * 2], "uint32"), K.uint32(0xFFFF)
                                    ),
                                    K.shift_left(
                                        K.bitwise_and(
                                            K.cast(packed_pairs[index * 2 + 1], "uint32"),
                                            K.uint32(0xFFFF),
                                        ),
                                        K.uint32(16),
                                    ),
                                ),
                            )
                    packed_words = 8
                else:
                    for index in range(32):
                        K.ptx.cvt.rn.satfinite.e2m1x2.f32(
                            packed_pairs[index], values[index * 2 + 1], values[index * 2]
                        )
                    for index in range(8):
                        K.assign(
                            packed[index],
                            K.bitwise_or(
                                K.bitwise_or(
                                    K.bitwise_and(
                                        K.cast(packed_pairs[index * 4], "uint32"), K.uint32(0xFF)
                                    ),
                                    K.shift_left(
                                        K.bitwise_and(
                                            K.cast(packed_pairs[index * 4 + 1], "uint32"),
                                            K.uint32(0xFF),
                                        ),
                                        K.uint32(8),
                                    ),
                                ),
                                K.bitwise_or(
                                    K.shift_left(
                                        K.bitwise_and(
                                            K.cast(packed_pairs[index * 4 + 2], "uint32"),
                                            K.uint32(0xFF),
                                        ),
                                        K.uint32(16),
                                    ),
                                    K.shift_left(
                                        K.bitwise_and(
                                            K.cast(packed_pairs[index * 4 + 3], "uint32"),
                                            K.uint32(0xFF),
                                        ),
                                        K.uint32(24),
                                    ),
                                ),
                            ),
                        )
                    packed_words = 8

                if c_major == "n":
                    if c_dtype in {"float8_e4m3fn", "float8_e5m2"}:
                        for vector in range(c_store_vectors):
                            pair = vector * 8
                            K.ptx["st.shared.v4.b16"](
                                smem.ptr_to([c_store_offsets[vector]]),
                                packed_pairs[pair],
                                packed_pairs[pair + 1],
                                packed_pairs[pair + 2],
                                packed_pairs[pair + 3],
                            )
                            K.ptx["st.shared.v4.b16"](
                                smem.ptr_to([c_store_offsets[vector] + 8]),
                                packed_pairs[pair + 4],
                                packed_pairs[pair + 5],
                                packed_pairs[pair + 6],
                                packed_pairs[pair + 7],
                            )
                    else:
                        for vector in range(c_store_vectors):
                            K.ptx.st.shared.v4.b32(
                                smem.ptr_to([c_store_offsets[vector]]),
                                packed[vector * 4],
                                packed[vector * 4 + 1],
                                packed[vector * 4 + 2],
                                packed[vector * 4 + 3],
                            )
                elif c_dtype == "float32":
                    scalar_base = (
                        smem_base
                        + c_offset
                        + c_stage_index * c_stage_bytes
                        + warp * 4096
                        + lane * 4
                    )
                    for index in range(32):
                        unswizzled = scalar_base + (index % 8) * 128 + (index // 8) * 1024
                        swizzled = K.bitwise_xor(
                            unswizzled,
                            K.bitwise_and(K.shift_right(unswizzled, K.uint32(3)), K.uint32(112)),
                        )
                        K.ptx.st.shared.b32(
                            smem.ptr_to([K.cast(swizzled - smem_base, "int32")]), packed[index]
                        )
                elif c_bits == 16:
                    thread = warp * 32 + lane
                    temporary = K.bitwise_or(
                        K.bitwise_and(thread << 5, K.int32(6144)),
                        K.bitwise_and(thread, K.int32(40)),
                    )
                    raw_address = K.bitwise_or(
                        K.bitwise_or(K.bitwise_and(thread << 7, K.int32(896)), temporary << 1),
                        K.bitwise_and(thread << 6, K.int32(1024)),
                    )
                    first_unswizzled = (
                        smem_base + c_offset + c_stage_index * c_stage_bytes + raw_address
                    )
                    first_swizzled = K.bitwise_xor(
                        first_unswizzled,
                        K.bitwise_and(K.shift_right(first_unswizzled, K.uint32(3)), K.uint32(112)),
                    )
                    second_unswizzled = first_unswizzled + 32
                    second_swizzled = K.bitwise_xor(
                        second_unswizzled,
                        K.bitwise_and(K.shift_right(second_unswizzled, K.uint32(3)), K.uint32(112)),
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        smem.ptr_to([K.cast(first_swizzled - smem_base, "int32")]),
                        packed[0],
                        packed[1],
                        packed[2],
                        packed[3],
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        smem.ptr_to([K.cast(first_swizzled - smem_base + 2048, "int32")]),
                        packed[4],
                        packed[5],
                        packed[6],
                        packed[7],
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        smem.ptr_to([K.cast(second_swizzled - smem_base, "int32")]),
                        packed[8],
                        packed[9],
                        packed[10],
                        packed[11],
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        smem.ptr_to([K.cast(second_swizzled - smem_base + 2048, "int32")]),
                        packed[12],
                        packed[13],
                        packed[14],
                        packed[15],
                    )
                else:
                    thread = warp * 32 + lane
                    raw_address = K.bitwise_or(
                        K.bitwise_and(thread, K.int32(224)),
                        K.bitwise_and(thread << 7, K.int32(3968)),
                    )
                    first_unswizzled = (
                        smem_base + c_offset + c_stage_index * c_stage_bytes + raw_address
                    )
                    first_swizzled = K.bitwise_xor(
                        first_unswizzled,
                        K.bitwise_and(K.shift_right(first_unswizzled, K.uint32(3)), K.uint32(112)),
                    )
                    second_unswizzled = first_unswizzled + 16
                    second_swizzled = K.bitwise_xor(
                        second_unswizzled,
                        K.bitwise_and(K.shift_right(second_unswizzled, K.uint32(3)), K.uint32(112)),
                    )
                    K.ptx.stmatrix.sync.aligned.m16n8.x4.trans.shared.b8(
                        smem.ptr_to([K.cast(first_swizzled - smem_base, "int32")]),
                        packed[0],
                        packed[1],
                        packed[2],
                        packed[3],
                    )
                    K.ptx.stmatrix.sync.aligned.m16n8.x4.trans.shared.b8(
                        smem.ptr_to([K.cast(second_swizzled - smem_base, "int32")]),
                        packed[4],
                        packed[5],
                        packed[6],
                        packed[7],
                    )
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                with K.If(warp == 0):
                    with K.Then():
                        if c_major == "n":
                            K.ptx[
                                "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                                ".bulk_group.L2::cache_hint"
                            ](
                                K.address_of(c_map),
                                K.cast(tile_n_idx * n_tile + subtile * epi_n, "int32"),
                                K.cast(tile_m_idx * m_tile, "int32"),
                                K.cast(batch_idx, "int32"),
                                smem.ptr_to([c_offset + c_stage_index * c_stage_bytes]),
                                K.uint64(tma_cache_hint),
                            )
                        else:
                            for row_copy in range(c_bits // 8):
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                                    ".bulk_group.L2::cache_hint"
                                ](
                                    K.address_of(c_map),
                                    K.cast(
                                        tile_m_idx * m_tile + row_copy * (epi_m // (c_bits // 8)),
                                        "int32",
                                    ),
                                    K.cast(tile_n_idx * n_tile + subtile * epi_n, "int32"),
                                    K.cast(batch_idx, "int32"),
                                    smem.ptr_to(
                                        [c_offset + c_stage_index * c_stage_bytes + row_copy * 4096]
                                    ),
                                    K.uint64(tma_cache_hint),
                                )
                        K.ptx.cp.async_.bulk.commit_group()
                        K.ptx.cp.async_.bulk.wait_group.read(c_stages - 1)
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                if not use_derived_c_stage:
                    if (c_stages & (c_stages - 1)) == 0:
                        K.assign(c_stage, K.bitwise_and(c_stage + 1, c_stages - 1))
                    else:
                        K.assign(c_stage, K.Select(c_stage == c_stages - 1, 0, c_stage + 1))

            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                K.assign(tile_amax, K.float32(0.0))
                subtile = K.local_scalar("int32", init=0)
                with K.While(subtile < epilogue_subtiles):
                    emit_epilogue_subtile(subtile)
                    K.assign(subtile, subtile + 1)

                K.ptx.redux_sync.max.NaN.f32(warp_amax, tile_amax, K.uint32(0xFFFFFFFF))
                with K.If(lane == 0):
                    with K.Then():
                        K.ptx.st.shared.b32(smem.ptr_to([amax_offset + warp * 4]), warp_amax)
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                with K.If((warp == 0) & (lane == 0)):
                    with K.Then():
                        block_amax = K.local_scalar("float32")
                        next_amax = K.local_scalar("float32")
                        K.ptx.ld.shared.b32(block_amax, smem.ptr_to([amax_offset]))
                        K.ptx.max.f32(block_amax, block_amax, K.float32(0.0))
                        for slot in range(1, 4):
                            K.ptx.ld.shared.b32(next_amax, smem.ptr_to([amax_offset + slot * 4]))
                            K.ptx.max.f32(block_amax, block_amax, next_amax)
                        atomic_old = K.local_scalar("int32")
                        K.ptx.atom.global_.max.s32(
                            atomic_old, amax.ptr_to([0]), K.reinterpret("int32", block_amax)
                        )
                with K.If(_elected()):
                    with K.Then():
                        K.ptx.mbarrier.arrive.shared.b64(
                            acc_pipe.empty.ptr_to([acc_state.stage]), K.uint32(1)
                        )
                _advance(acc_state)
                advance_work(work)

            K.ptx.bar.sync(K.uint32(2), K.uint32(128))
            K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](tmem_base, K.uint32(512))
            K.ptx.cp.async_.bulk.wait_group.read(0)

    kernel.__annotations__ = {
        "a": K.gptr[K.u8, (M * K_dim * L * ab_bits // 8,)],
        "b": K.gptr[K.u8, (N * K_dim * L * ab_bits // 8,)],
        "sfa": K.gptr[K.u8, (L * _ceil_div(M, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "sfb": K.gptr[K.u8, (L * _ceil_div(N, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "c": K.gptr[K.u8, (M * N * L * c_bits // 8,)],
        "amax": K.gptr[K.f32, (1,)],
    }
    return K.kernel(
        warps=6,
        arch="sm_100a",
        min_blocks_per_sm=1,
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
    a_major,
    b_major,
    c_major,
    mma_tiler_mn,
    cluster_shape_mn,
):
    return _make_kernel(
        M,
        N,
        K,
        L,
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        a_major,
        b_major,
        c_major,
        mma_tiler_mn,
        cluster_shape_mn,
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


def _fp4_tensor(torch, rows, columns, batch, *, zero=False):
    raw = (
        torch.zeros(batch * rows * columns // 2, dtype=torch.uint8, device="cuda")
        if zero
        else torch.empty(batch * rows * columns // 2, dtype=torch.uint8, device="cuda")
    )
    physical = raw.view(batch, rows, columns // 2)
    logical = physical.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
    return raw, logical, physical


def _sparse_input(torch, rows, K_dim, batch, dtype, major, salt):
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int64)
    k_indices = []
    input_values = []
    if dtype == "float4_e2m1fn":
        raw, logical, physical = _fp4_tensor(torch, rows, K_dim, batch, zero=True)
        for batch_index in range(batch):
            k_index = (row_ids * (17 + salt) + batch_index * 7 + salt) % K_dim
            value = 1 + ((row_ids + batch_index + salt) % 2)
            code = value.to(torch.uint8) * 2
            physical[batch_index, row_ids, k_index // 2] = code << (
                (k_index % 2).to(torch.uint8) * 4
            )
            k_indices.append(k_index)
            input_values.append(value.to(torch.float32))
    else:
        raw, logical = _regular_tensor(torch, rows, K_dim, batch, dtype, first_major=major != "k")
        logical.zero_()
        for batch_index in range(batch):
            k_index = (row_ids * (17 + salt) + batch_index * 7 + salt) % K_dim
            value = 1 + ((row_ids + batch_index + salt) % 2)
            logical[row_ids, k_index, batch_index] = value.to(logical.dtype)
            k_indices.append(k_index)
            input_values.append(value.to(torch.float32))
    return {"raw": raw, "source": logical, "k_indices": k_indices, "values": input_values}


def _output_tensor(torch, M, N, L, dtype, major):
    if dtype == "float4_e2m1fn":
        raw, logical, _ = _fp4_tensor(torch, M, N, L, zero=True)
        return {"raw": raw, "source": logical}
    raw, logical = _regular_tensor(torch, M, N, L, dtype, first_major=major == "m")
    logical.zero_()
    return {"raw": raw, "source": logical}


def _decode_fp4_output(torch, raw, M, N, L):
    physical = raw.view(L, M, N // 2)
    low = physical & 0xF
    high = physical >> 4
    codes = torch.stack((low, high), dim=-1).reshape(L, M, N).to(torch.int64)
    lookup = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
        dtype=torch.float32,
    )
    return lookup[codes].permute(1, 2, 0)


def _output_as_float(torch, output, M, N, L, dtype):
    if dtype == "float4_e2m1fn":
        return _decode_fp4_output(torch, output["raw"], M, N, L)
    return output["source"].float()


def prepare_data(**config):
    """Allocate one input set shared by TIRx, the source kernel, and the oracle."""
    import torch

    config = _without_label(config)
    M, N, K_dim, L = (config[key] for key in ("M", "N", "K", "L"))
    sf_dtype = "float8_e8m0fnu" if config["sf_dtype"] == "int8" else config["sf_dtype"]
    a_data = _sparse_input(torch, M, K_dim, L, config["ab_dtype"], config["a_major"], 0)
    b_data = _sparse_input(torch, N, K_dim, L, config["ab_dtype"], config["b_major"], 6)
    sf_shape_a = (L, _ceil_div(M, 128), _ceil_div(K_dim, 4 * config["sf_vec_size"]), 32, 4, 4)
    sf_shape_b = (L, _ceil_div(N, 128), _ceil_div(K_dim, 4 * config["sf_vec_size"]), 32, 4, 4)
    sf_torch_dtype = _torch_dtype(torch, sf_dtype)
    sfa = torch.ones(sf_shape_a, dtype=torch.float32, device="cuda").to(sf_torch_dtype)
    sfb = torch.ones(sf_shape_b, dtype=torch.float32, device="cuda").to(sf_torch_dtype)
    tirx_output = _output_tensor(torch, M, N, L, config["c_dtype"], config["c_major"])
    source_output = _output_tensor(torch, M, N, L, config["c_dtype"], config["c_major"])
    expected_batches = []
    for batch_index in range(L):
        matches = (
            a_data["k_indices"][batch_index][:, None] == b_data["k_indices"][batch_index][None, :]
        )
        expected_batches.append(
            matches.to(torch.float32)
            * a_data["values"][batch_index][:, None]
            * b_data["values"][batch_index][None, :]
        )
    expected = torch.stack(expected_batches, dim=2)
    return {
        "a": a_data,
        "b": b_data,
        "sfa": sfa,
        "sfb": sfb,
        "tirx_output": tirx_output,
        "source_output": source_output,
        "tirx_amax": torch.zeros(1, dtype=torch.float32, device="cuda"),
        "source_amax": torch.zeros(1, dtype=torch.float32, device="cuda"),
        "expected": expected,
    }


def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module(
        "cudnn.gemm.cutedsl.dense.amax.dense_blockscaled_gemm_persistent_amax"
    )


def _compile_reference(data, config):
    from tirx_kernels.cudnn._reference import import_cutlass_reference

    cutlass = import_cutlass_reference()
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    if os.environ.get("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0") != "0":
        raise RuntimeError("CUDNNFE_CLUSTER_OVERLAP_MARGIN must be 0")
    module = _load_reference_source()
    config = _without_label(config)
    a = data["a"]["source"]
    b = data["b"]["source"]
    sfa = data["sfa"]
    sfb = data["sfb"]
    c = data["source_output"]["source"]
    amax = data["source_amax"]
    a_cute = from_dlpack(a, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["a_major"] == "m" else 1
    )
    b_cute = from_dlpack(b, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["b_major"] == "n" else 1
    )
    sfa_cute = from_dlpack(sfa, assumed_align=16)
    sfb_cute = from_dlpack(sfb, assumed_align=16)
    c_cute = from_dlpack(c, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["c_major"] == "m" else 1
    )
    amax_cute = from_dlpack(amax, assumed_align=16)
    kernel = module.Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=config["sf_vec_size"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
    )
    cluster_size = config["cluster_shape_mn"][0] * config["cluster_shape_mn"][1]
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    executable = cute.compile(
        kernel,
        a_tensor=a_cute,
        b_tensor=b_cute,
        sfa_tensor=sfa_cute,
        sfb_tensor=sfb_cute,
        c_tensor=c_cute,
        amax_tensor=amax_cute,
        max_active_clusters=max_active_clusters,
        stream=make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def launch():
        executable(a, b, sfa, sfb, c, amax, stream)

    launch._keep_alive = (executable, a, b, sfa, sfb, c, amax, stream)
    return launch


def _tirx_launch(executable, data):
    import torch

    a = data["a"]["raw"]
    b = data["b"]["raw"]
    sfa = data["sfa"].view(torch.uint8).reshape(-1)
    sfb = data["sfb"].view(torch.uint8).reshape(-1)
    c = data["tirx_output"]["raw"]
    amax = data["tirx_amax"]

    def launch():
        executable(a, b, sfa, sfb, c, amax)

    launch._keep_alive = (a, b, sfa, sfb, c, amax)
    return launch


def _assert_close(torch, actual, expected, *, atol, rtol, label):
    if actual.shape != expected.shape:
        raise AssertionError(f"{label} shape mismatch: {actual.shape} != {expected.shape}")
    if actual.device != expected.device:
        raise AssertionError(f"{label} device mismatch: {actual.device} != {expected.device}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{label} dtype mismatch: {actual.dtype} != {expected.dtype}")
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    if bool(torch.all(close).item()):
        return
    absolute_error = torch.abs(actual - expected)
    allowed_error = atol + rtol * torch.abs(expected)
    excess = absolute_error - allowed_error
    worst_index = int(torch.argmax(excess).item())
    worst_absolute = float(absolute_error.reshape(-1)[worst_index].item())
    worst_allowed = float(allowed_error.reshape(-1)[worst_index].item())
    raise AssertionError(
        f"{label} mismatch: worst absolute error {worst_absolute} exceeds {worst_allowed}"
    )


def _validate_outputs(data, config, *, with_source):
    import torch

    config = _without_label(config)
    M, N, L = (config[key] for key in ("M", "N", "L"))
    tolerance = 0.1 if _dtype_bits(config["c_dtype"]) <= 8 else 0.01
    tirx_output = _output_as_float(torch, data["tirx_output"], M, N, L, config["c_dtype"])
    _assert_close(
        torch,
        tirx_output,
        data["expected"],
        atol=tolerance,
        rtol=tolerance,
        label="TIRx output versus FP32 oracle",
    )
    expected_amax = data["expected"].abs().amax().reshape(1)
    _assert_close(
        torch,
        data["tirx_amax"],
        expected_amax,
        atol=0.1,
        rtol=0.1,
        label="TIRx amax versus FP32 oracle",
    )
    if with_source:
        source_output = _output_as_float(torch, data["source_output"], M, N, L, config["c_dtype"])
        _assert_close(
            torch,
            source_output,
            data["expected"],
            atol=tolerance,
            rtol=tolerance,
            label="source output versus FP32 oracle",
        )
        _assert_close(
            torch,
            tirx_output,
            source_output,
            atol=tolerance,
            rtol=tolerance,
            label="TIRx output versus source output",
        )
        _assert_close(
            torch,
            data["source_amax"],
            expected_amax,
            atol=0.1,
            rtol=0.1,
            label="source amax versus FP32 oracle",
        )


def run_test(**config):
    """Compare TIRx with both the pinned source and the FP32 oracle."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _tirx_launch(compile_kernel(get_kernel(**kernel_config)), data)
    source_launch = _compile_reference(data, kernel_config)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    _validate_outputs(data, kernel_config, with_source=True)
    return {"max_abs": float(data["expected"].abs().amax().item())}


def prepare_bench(**config):
    """Compile only TIRx; reference imports and CUDA work stay in ``run_gpu``."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {"config": kernel_config, "executable": compile_kernel(get_kernel(**kernel_config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then time pure launches with the canonical timer."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = {**prepared["config"], **kwargs}
    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _tirx_launch(prepared["executable"], data)
    tirx_launch()
    torch.cuda.synchronize()
    with_source = external_references_enabled()
    references = None
    if with_source:
        source_launch = _compile_reference(data, kernel_config)
        source_launch()
        torch.cuda.synchronize()
        references = {"cudnn_frontend": lambda: source_launch}
    _validate_outputs(data, kernel_config, with_source=with_source)
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
