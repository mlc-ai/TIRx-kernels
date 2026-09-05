# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Persistent SM100 block-scaled GEMM with interleaved SwiGLU quantization.

Upstream source:
``python/cudnn/gemm/cutedsl/dense/swiglu/``
``dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py``.
"""

import heapq
from functools import cache
from itertools import combinations, product

import tirx_kernels.kern as K

_TRY_WAIT_TICKS = 10_000_000
_SMEM_CAPACITY = 232_448
_MAX_ACTIVE_CLUSTERS = {1: 148, 2: 74, 4: 33, 8: 15, 16: 7}
_AB_DTYPES = ("float4_e2m1fn", "float8_e4m3fn", "float8_e5m2")
_SF_MODES = (("float8_e8m0fnu", 16), ("float8_e8m0fnu", 32), ("float8_e4m3fn", 16))
_OUTPUT_DTYPES = ("float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
_CLUSTERS = tuple(product((1, 2, 4), repeat=2))
_MODE_KEYS = (
    "ab_dtype",
    "sf_dtype",
    "sf_vec_size",
    "ab12_dtype",
    "c_dtype",
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


def _valid_mode(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    if tile_m not in (128, 256) or tile_n not in (64, 128, 192, 256):
        return False
    if cluster_m not in (1, 2, 4) or cluster_n not in (1, 2, 4):
        return False
    if cluster_m * cluster_n > 16 or (tile_m == 256 and cluster_m % 2):
        return False
    # A CTA-group-1 cluster cannot acquire the source's exclusive 512-column
    # TMEM allocation under the current CUDA codegen launch contract.  The
    # standalone source emits .reqntid and runs this cross-mode; K emits the
    # repository-standard .maxntid launch bound, for which the allocation
    # permit does not complete.  Keep the fixed source TMEM ABI and reject the
    # non-runnable cross-mode instead of silently shrinking the allocation.
    if tile_m == 128 and cluster_m * cluster_n > 1:
        return False
    if mode["ab_dtype"] == "float4_e2m1fn":
        if mode["a_major"] != "k" or mode["b_major"] != "k":
            return False
        if (mode["sf_dtype"], mode["sf_vec_size"]) not in _SF_MODES:
            return False
    elif (mode["sf_dtype"], mode["sf_vec_size"]) != ("float8_e8m0fnu", 32):
        return False
    if mode["c_dtype"].startswith("float8_") and (tile_n != 256 or mode["c_major"] != "n"):
        return False
    if mode["c_major"] == "m" and mode["ab12_dtype"] in {"float32", "float8_e4m3fn", "float8_e5m2"}:
        return False
    if tile_n == 192:
        # These are the two standalone source specializations that pass the
        # N=192 mathematical probe.  Wider cross-axis combinations either
        # fail source serialization or execute an invalid persistent stage.
        if (
            mode["ab_dtype"] != "float4_e2m1fn"
            or (mode["sf_dtype"], mode["sf_vec_size"]) != ("float8_e8m0fnu", 16)
            or mode["ab12_dtype"] != "bfloat16"
            or mode["c_dtype"] != "bfloat16"
            or mode["c_major"] != "n"
            or not mode["vector_f32"]
            or (cluster_m, cluster_n) != (tile_m // 128, 1)
        ):
            return False
    if (
        mode["ab_dtype"].startswith("float8_")
        and mode["ab12_dtype"] == "float32"
        and not mode["vector_f32"]
    ):
        return False
    return mode["ab12_dtype"] in _OUTPUT_DTYPES and mode["c_dtype"] in _OUTPUT_DTYPES


def _mode_label(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    return (
        f"{_SHORT_DTYPE[mode['ab_dtype']]}_{_SHORT_DTYPE[mode['sf_dtype']]}v{mode['sf_vec_size']}_"
        f"{_SHORT_DTYPE[mode['ab12_dtype']]}_{_SHORT_DTYPE[mode['c_dtype']]}_"
        f"{mode['a_major']}{mode['b_major']}_{mode['c_major']}_"
        f"t{tile_m}x{tile_n}_c{cluster_m}x{cluster_n}_"
        f"{'v' if mode['vector_f32'] else 's'}f32_l{mode['L']}"
    )


@cache
def _structural_modes(include_batch=True):
    modes = []
    batches = (1, 2) if include_batch else (1,)
    for ab_dtype in _AB_DTYPES:
        sf_modes = _SF_MODES if ab_dtype == "float4_e2m1fn" else (("float8_e8m0fnu", 32),)
        a_majors = ("k",) if ab_dtype == "float4_e2m1fn" else ("k", "m")
        b_majors = ("k",) if ab_dtype == "float4_e2m1fn" else ("k", "n")
        for (
            sf_mode,
            ab12_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            tile_m,
            tile_n,
            cluster,
            vector_f32,
            batch,
        ) in product(
            sf_modes,
            _OUTPUT_DTYPES,
            _OUTPUT_DTYPES,
            a_majors,
            b_majors,
            ("m", "n"),
            (128, 256),
            (64, 128, 192, 256),
            _CLUSTERS,
            (False, True),
            batches,
        ):
            sf_dtype, sf_vec_size = sf_mode
            mode = {
                "ab_dtype": ab_dtype,
                "sf_dtype": sf_dtype,
                "sf_vec_size": sf_vec_size,
                "ab12_dtype": ab12_dtype,
                "c_dtype": c_dtype,
                "a_major": a_major,
                "b_major": b_major,
                "c_major": c_major,
                "mma_tiler_mn": (tile_m, tile_n),
                "cluster_shape_mn": cluster,
                "vector_f32": vector_f32,
                "L": batch,
            }
            if _valid_mode(mode):
                modes.append(mode)
    return tuple(sorted(modes, key=_mode_label))


def _pair_tokens(mode):
    encoded = {key: repr(mode[key]) for key in _MODE_KEYS}
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
    labels = [_mode_label(mode) for mode in candidates]
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
    for i in selected:
        for token in covers[i]:
            counts[token] = counts.get(token, 0) + 1
    kept = []
    for i in reversed(selected):
        if all(counts[token] > 1 for token in covers[i]):
            for token in covers[i]:
                counts[token] -= 1
        else:
            kept.append(i)
    return [candidates[i] for i in reversed(kept)]


def _make_case(mode, *, shape=(256, 256, 256), prefix="pair"):
    M, N, K_dim = shape
    if mode["mma_tiler_mn"][1] == 192:
        N = 192
    elif prefix == "pair":
        cluster_m, cluster_n = mode["cluster_shape_mn"]
        tile_n = mode["mma_tiler_mn"][1]
        M = _align_up(M, cluster_m * 128)
        N = _align_up(N, cluster_n * tile_n)
    if mode["c_dtype"].startswith("float8_") and not prefix.startswith("perf"):
        # Direct SFC stores inherit source scheduler validity rather than a
        # per-store predicate.  Use complete clusters for pairwise rows.
        M = max(M, 512)
        N = 1024
    return {
        "label": f"{prefix}_m{M}_n{N}_k{K_dim}_{_mode_label(mode)}",
        "M": M,
        "N": N,
        "K": K_dim,
        "alpha": 2.0 / 3.0,
        **mode,
    }


def _correctness_configs():
    configs = [_make_case(mode) for mode in _minimal_cover(_structural_modes(), _pair_tokens)]
    candidates = _structural_modes()
    extras = (
        ("tail_k", (256, 256, 320), lambda mode: mode["ab_dtype"] == "float4_e2m1fn"),
        ("tail_output", (272, 320, 320), lambda mode: not mode["c_dtype"].startswith("float8_")),
        ("n192", (256, 192, 256), lambda mode: mode["mma_tiler_mn"][1] == 192),
        ("sfc", (272, 448, 320), lambda mode: mode["c_dtype"].startswith("float8_")),
        (
            "amax",
            (272, 320, 320),
            lambda mode: mode["ab_dtype"] == "float4_e2m1fn" and mode["c_dtype"] == "bfloat16",
        ),
        (
            "cta1_l2",
            (256, 256, 256),
            lambda mode: mode["mma_tiler_mn"][0] == 128 and mode["L"] == 2,
        ),
        (
            "cta2_l2",
            (256, 256, 256),
            lambda mode: mode["mma_tiler_mn"][0] == 256 and mode["L"] == 2,
        ),
    )
    for prefix, shape, predicate in extras:
        mode = min((candidate for candidate in candidates if predicate(candidate)), key=_mode_label)
        configs.append(_make_case(mode, shape=shape, prefix=prefix))
    return sorted(configs, key=lambda config: config["label"])


def _stage_parameters(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cta_group = tile_m // 128
    ab_bits = _dtype_bits(mode["ab_dtype"])
    k_tile = 256 if ab_bits == 4 else 128
    b_rows = tile_n // cta_group
    epi_n = 32
    a_stage = 128 * k_tile * ab_bits // 8
    b_stage = b_rows * k_tile * ab_bits // 8
    sfa_stage = 128 * k_tile // mode["sf_vec_size"]
    sfb_stage = _align_up(tile_n, 128) * k_tile // mode["sf_vec_size"]
    ab12_stage = 128 * 64 * _dtype_bits(mode["ab12_dtype"]) // 8
    c_stage = 128 * epi_n * _dtype_bits(mode["c_dtype"]) // 8
    c_stages = tile_n // 64
    ab_stages = (_SMEM_CAPACITY - (1024 + 4 * ab12_stage + c_stages * c_stage + 16)) // (
        a_stage + b_stage + sfa_stage + sfb_stage
    )
    return cta_group, k_tile, c_stages, ab_stages


def _fits_shared_memory(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cta_group, _k_tile, c_stages, ab_stages = _stage_parameters(mode)
    if ab_stages <= 0:
        return False
    ab_bits = _dtype_bits(mode["ab_dtype"])
    ab12_bits = _dtype_bits(mode["ab12_dtype"])
    c_bits = _dtype_bits(mode["c_dtype"])
    k_tile = 256 if ab_bits == 4 else 128
    a_stage = 128 * k_tile * ab_bits // 8
    b_stage = (tile_n // cta_group) * k_tile * ab_bits // 8
    sfa_stage = 128 * k_tile // mode["sf_vec_size"]
    sfb_stage = _align_up(tile_n, 128) * k_tile // mode["sf_vec_size"]
    ab12_stage = 128 * 64 * ab12_bits // 8
    c_stage = 128 * 32 * c_bits // 8
    sfa_end = (
        1024 + c_stages * c_stage + 4 * ab12_stage + ab_stages * (a_stage + b_stage + sfa_stage)
    )
    sfb_offset = _align_up(sfa_end, 1024)
    return _align_up(sfb_offset + ab_stages * sfb_stage + 16, 1024) <= _SMEM_CAPACITY


def _supports_performance_workload(mode):
    tile_n = mode["mma_tiler_mn"][1]
    if not _fits_shared_memory(mode) or tile_n == 192:
        return False
    if tile_n != 64:
        return True
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    problem_clusters = _ceil_div(8, cluster_m) * _ceil_div(16, cluster_n)
    return problem_clusters <= _MAX_ACTIVE_CLUSTERS[cluster_m * cluster_n]


def _performance_tokens(mode):
    cta_group, k_tile, c_stages, ab_stages = _stage_parameters(mode)
    return frozenset(
        {
            ("input", mode["ab_dtype"]),
            ("sf", mode["sf_dtype"], mode["sf_vec_size"]),
            ("ab12", mode["ab12_dtype"]),
            ("c", mode["c_dtype"]),
            ("mma", mode["ab_dtype"], mode["sf_vec_size"], cta_group),
            ("a_tma", mode["ab_dtype"], mode["a_major"], cta_group),
            ("b_tma", mode["ab_dtype"], mode["b_major"], cta_group),
            ("store", mode["ab12_dtype"], mode["c_dtype"], mode["c_major"]),
            ("tile_n", mode["mma_tiler_mn"][1]),
            ("cluster", mode["cluster_shape_mn"]),
            ("k_tile", k_tile),
            ("c_stages", c_stages),
            ("ab_stages", ab_stages),
            ("vector_f32", mode["vector_f32"]),
            ("sfc", mode["c_dtype"].startswith("float8_")),
            ("amax", mode["ab_dtype"] == "float4_e2m1fn" and mode["c_dtype"] == "bfloat16"),
        }
    )


def _benchmark_configs():
    legal_modes = [
        mode
        for mode in _structural_modes(include_batch=False)
        if _supports_performance_workload(mode)
    ]
    representatives = _minimal_cover(legal_modes, _performance_tokens)
    configs = []
    for index, base_mode in enumerate(representatives):
        tile_n = base_mode["mma_tiler_mn"][1]
        sizes = (3072,) if tile_n == 192 else (1024, 2048, 4096, 8192)
        for size in sizes:
            for batch in (1, 2):
                mode = {**base_mode, "L": batch}
                if tile_n in (64, 192):
                    cluster_m, cluster_n = mode["cluster_shape_mn"]
                    problem_clusters = (
                        _ceil_div(_ceil_div(size, 128), cluster_m)
                        * _ceil_div(_ceil_div(size, tile_n), cluster_n)
                        * batch
                    )
                    if problem_clusters > _MAX_ACTIVE_CLUSTERS[cluster_m * cluster_n]:
                        continue
                configs.append(
                    _make_case(mode, shape=(size, size, size), prefix=f"perf{index:02d}")
                )
    return sorted(configs, key=lambda config: config["label"])


KERNEL_META = {
    "name": "cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant",
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
    ab12_dtype,
    c_dtype,
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
        "ab12_dtype": ab12_dtype,
        "c_dtype": c_dtype,
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
    if N % 64:
        raise ValueError("N must be divisible by 64")
    if c_dtype.startswith("float8_") and N % 256 not in (0, 192):
        raise ValueError("SFC output requires N modulo 256 to be 0 or 192")
    ab_bits = _dtype_bits(ab_dtype)
    ab12_bits = _dtype_bits(ab12_dtype)
    c_bits = _dtype_bits(c_dtype)
    if (M if a_major == "m" else K_dim) % (128 // ab_bits):
        raise ValueError("A's contiguous dimension must be 16-byte aligned")
    if (N if b_major == "n" else K_dim) % (128 // ab_bits):
        raise ValueError("B's contiguous dimension must be 16-byte aligned")
    if (M if c_major == "m" else N) % (128 // ab12_bits):
        raise ValueError("AB12's contiguous dimension must be 16-byte aligned")
    if (M if c_major == "m" else N // 2) % (128 // c_bits):
        raise ValueError("C's contiguous dimension must be 16-byte aligned")

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
    if n_tile in (64, 192) and cluster_work != num_clusters:
        raise ValueError(
            "tile-N 64/192 is unsafe when a persistent CTA executes more than one work tile"
        )
    if n_tile == 192 and N != 192:
        raise ValueError("tile-N 192 is source-correct only for N=192")

    acc_stages = 1 if n_tile == 256 else 2
    ab12_stages = 4
    c_stages = n_tile // 64
    epi_n = 32
    epilogue_subtiles = n_tile // epi_n
    a_stage_bytes = cta_m * k_tile * ab_bits // 8
    b_stage_bytes = b_rows * k_tile * ab_bits // 8
    sfa_stage_bytes = cta_m * k_tile // sf_vec_size
    # The source SFB layout is padded to a whole 128-column scale tile.
    # This padding participates in both SMEM allocation and mbarrier bytes.
    sfb_stage_bytes = _align_up(n_tile, 128) * k_tile // sf_vec_size
    ab_stage_bytes = a_stage_bytes + b_stage_bytes + sfa_stage_bytes + sfb_stage_bytes
    ab12_stage_bytes = cta_m * 64 * ab12_bits // 8
    c_stage_bytes = cta_m * epi_n * c_bits // 8
    ab_stages = (
        _SMEM_CAPACITY - (1024 + c_stages * c_stage_bytes + ab12_stages * ab12_stage_bytes + 16)
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
    ab12_offset = c_offset + c_stages * c_stage_bytes
    a_offset = ab12_offset + ab12_stages * ab12_stage_bytes
    b_offset = a_offset + ab_stages * a_stage_bytes
    sfa_offset = b_offset + ab_stages * b_stage_bytes
    sfb_offset = _align_up(sfa_offset + ab_stages * sfa_stage_bytes, 1024)
    sfb_storage_stage_bytes = sfb_stage_bytes
    amax_offset = sfb_offset + ab_stages * sfb_storage_stage_bytes
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
    generate_sfc = c_dtype.startswith("float8_")
    generate_amax = ab_dtype == "float4_e2m1fn" and c_dtype == "bfloat16"
    if generate_sfc and (m_tiles % cluster_m or n_tiles % cluster_n):
        raise ValueError("SFC output requires complete M/N cluster tiles")

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        sfa = params["sfa"]
        sfb = params["sfb"]
        ab12 = params["ab12"]
        c = params["c"]
        a_map = K.stack_alloca("tensormap", 1)
        b_map = K.stack_alloca("tensormap", 1)
        sfa_map = K.stack_alloca("tensormap", 1)
        sfb_map = K.stack_alloca("tensormap", 1)
        ab12_map = K.stack_alloca("tensormap", 1)
        c_map = K.stack_alloca("tensormap", 1)

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

        def encode_output(descriptor, tensor, dtype, columns, bits, logical_epi_n):
            element_bytes = bits // 8
            contiguous_bytes = (M if c_major == "m" else columns) * element_bytes
            batch_bytes = M * columns * element_bytes
            if c_major == "m":
                row_copies = element_bytes
                fields = (
                    M,
                    columns,
                    L,
                    contiguous_bytes,
                    batch_bytes,
                    cta_m // row_copies,
                    logical_epi_n,
                    1,
                )
                # FP32 MN-major epilogues use the source's
                # CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B encoding.  Its shared
                # address transform is S<2, 5, 2>, not the ordinary SW128
                # S<3, 4, 3> used by 16-bit outputs.
                swizzle = 4 if bits == 32 else 3
            else:
                fields = (columns, M, L, contiguous_bytes, batch_bytes, logical_epi_n, cta_m, 1)
                row_bytes = logical_epi_n * element_bytes
                swizzle = 3 if row_bytes >= 128 else 2 if row_bytes >= 64 else 1
            encode(descriptor, dtype, 3, tensor.data, *fields, 1, 1, 1, 0, swizzle, 2, 0)

        encode_output(
            ab12_map,
            ab12,
            ab12_dtype,
            N,
            ab12_bits,
            32 if c_major == "n" and ab12_bits == 32 else 64,
        )
        encode_output(c_map, c, c_dtype, N // 2, c_bits, epi_n)
        return a_map, b_map, sfa_map, sfb_map, ab12_map, c_map

    def kernel(a, b, sfa, sfb, c, ab12, amax, sfc, norm_const, alpha, *, host):
        del a, b, sfa, sfb, c, ab12
        a_map, b_map, sfa_map, sfb_map, ab12_map, c_map = host
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
            K.ptx.prefetch.tensormap(K.address_of(ab12_map))
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
                            with K.If(leader_cta):
                                with K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.uint32(num_tma_load_bytes),
                                    )

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
                                if cta_group == 1:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                    ](
                                        smem.ptr_to([a_smem_offset]),
                                        K.address_of(a_map),
                                        K.cast(a_coord_0, "int32"),
                                        K.cast(a_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                elif cluster_n == 1:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + a_smem_offset,
                                        K.address_of(a_map),
                                        K.cast(a_coord_0, "int32"),
                                        K.cast(a_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_full_leader.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + a_smem_offset,
                                        K.address_of(a_map),
                                        K.cast(a_coord_0, "int32"),
                                        K.cast(a_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_full_leader.ptr_to([tma_state.stage]),
                                        K.cast(a_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )

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
                                if cta_group == 1:
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
                                elif cluster_m_groups == 1:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + b_smem_offset,
                                        K.address_of(b_map),
                                        K.cast(b_coord_0, "int32"),
                                        K.cast(b_coord_1, "int32"),
                                        K.cast(batch_idx, "int32"),
                                        ab_full_leader.ptr_to([tma_state.stage]),
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
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
                            if cta_group == 1 and cluster_n == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
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
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
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
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                ](
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
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2"
                                ](
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
                            sfb_tile_group = tile_n_idx // 2 if n_tile == 64 else tile_n_idx
                            sfb_coord_2 = sfb_tile_group * sfb_n_box + sfb_quotient // sf_k_box
                            sfb_smem_offset = (
                                sfb_offset
                                + tma_state.stage * sfb_stage_bytes
                                + cluster_m_group * (sfb_stage_bytes // cluster_m_groups)
                            )
                            if cta_group == 1 and cluster_m_groups == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
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
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
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
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                ](
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
                                K.ptx[
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2"
                                ](
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
            K.ptx.bar.sync(K.uint32(2), K.uint32(160))
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
            with K.If(warp == 0):
                with K.Then():
                    K.ptx[
                        "tcgen05.alloc.cta_group::"
                        + str(cta_group)
                        + ".sync.aligned.shared::cta.b32"
                    ](tmem_slot_addr, K.uint32(tmem_columns))
            K.ptx.bar.sync(K.uint32(2), K.uint32(160))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            acc_state = K.PipelineState(acc_stages, phase=0)
            work = K.local_scalar("int32", init=cluster_work_id)
            executed_tiles = K.local_scalar("int32", init=0)

            x_values = K.alloc_local((32,), "float32")
            gate_values = K.alloc_local((32,), "float32")
            c_values = K.alloc_local((32,), "float32")
            gate_reciprocals = K.alloc_local((32,), "float32")
            absolute_values = K.alloc_local((32,), "float32")
            tile_amax = K.local_scalar("float32")
            warp_amax = K.local_scalar("float32")
            ab12_x_words = K.alloc_local((32,), "uint32")
            ab12_gate_words = K.alloc_local((32,), "uint32")
            ab12_x_pairs = K.alloc_local((16,), "uint16")
            ab12_gate_pairs = K.alloc_local((16,), "uint16")
            c_pairs = K.alloc_local((16,), "uint16")
            raw_scales = K.alloc_local((8,), "float32")
            decoded_scales = K.alloc_local((8,), "float32")
            sf_pairs = K.alloc_local((4,), "uint16")
            sfc_store_pairs = K.alloc_local((4,), "uint16")
            decoded_sf_pairs = K.alloc_local((4,), "uint32")
            sfc_words = K.alloc_local((2,), "uint32")
            group_amaxes = K.alloc_local((2,), "float32")
            quant_scales = K.alloc_local((2,), "float32")
            norm = K.local_scalar("float32")
            if generate_sfc:
                K.ptx.ld.global_.b32(norm, norm_const.ptr_to([0]))
            scalar_c_store = c_major == "m" and ab12_bits == 32 and c_bits == 16
            if scalar_c_store:
                c_halves = K.alloc_local((32,), "uint16")
            c_words = K.alloc_local((32,), "uint32")
            if c_major == "n":
                ab12_x_offsets = K.alloc_local((8,), "int32")
                ab12_gate_offsets = K.alloc_local((8,), "int32")
                c_offsets = K.alloc_local((8,), "int32")

            def load_accumulator_pair(subtile):
                pair_base = K.local_scalar(
                    "uint32",
                    init=(tmem_base + (warp << 21) + acc_state.stage * n_tile + subtile * epi_n),
                )
                if c_major == "m" and ab12_bits == 16:
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[x_values[index] for index in range(16)], pair_base
                    )
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[x_values[index] for index in range(16, 32)], pair_base + K.uint32(1 << 20)
                    )
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[gate_values[index] for index in range(16)], pair_base + epi_n
                    )
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[gate_values[index] for index in range(16, 32)],
                        pair_base + epi_n + K.uint32(1 << 20),
                    )
                else:
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[x_values[index] for index in range(32)], pair_base
                    )
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[gate_values[index] for index in range(32)], pair_base + epi_n
                    )

            def multiply_pair(dst0, dst1, left0, left1, right0, right1):
                packed = K.local_scalar("uint64")
                K.ptx.mul.rn.f32x2(
                    packed, K.cuda.make_float2(left0, left1), K.cuda.make_float2(right0, right1)
                )
                K.ptx.mov.b64(dst0, dst1, packed)

            def apply_alpha():
                for index in range(0, epi_n, 2):
                    multiply_pair(
                        x_values[index],
                        x_values[index + 1],
                        x_values[index],
                        x_values[index + 1],
                        alpha,
                        alpha,
                    )
                    multiply_pair(
                        gate_values[index],
                        gate_values[index + 1],
                        gate_values[index],
                        gate_values[index + 1],
                        alpha,
                        alpha,
                    )

            def apply_swiglu():
                if vector_f32:
                    for index in range(0, epi_n, 2):
                        multiply_pair(
                            gate_reciprocals[index],
                            gate_reciprocals[index + 1],
                            gate_values[index],
                            gate_values[index + 1],
                            K.float32(-1.4426950408889634),
                            K.float32(-1.4426950408889634),
                        )
                        K.ptx.ex2.approx.ftz.f32(gate_reciprocals[index], gate_reciprocals[index])
                        K.ptx.ex2.approx.ftz.f32(
                            gate_reciprocals[index + 1], gate_reciprocals[index + 1]
                        )
                        K.assign(gate_reciprocals[index], K.float32(1.0) + gate_reciprocals[index])
                        K.assign(
                            gate_reciprocals[index + 1],
                            K.float32(1.0) + gate_reciprocals[index + 1],
                        )
                        K.ptx.rcp.approx.ftz.f32(gate_reciprocals[index], gate_reciprocals[index])
                        K.ptx.rcp.approx.ftz.f32(
                            gate_reciprocals[index + 1], gate_reciprocals[index + 1]
                        )
                        multiply_pair(
                            c_values[index],
                            c_values[index + 1],
                            gate_reciprocals[index],
                            gate_reciprocals[index + 1],
                            gate_values[index],
                            gate_values[index + 1],
                        )
                        multiply_pair(
                            c_values[index],
                            c_values[index + 1],
                            c_values[index],
                            c_values[index + 1],
                            x_values[index],
                            x_values[index + 1],
                        )
                else:
                    for index in range(epi_n):
                        K.ptx.ex2.approx.ftz.f32(
                            gate_reciprocals[index],
                            gate_values[index] * K.float32(-1.4426950408889634),
                        )
                        K.assign(gate_reciprocals[index], K.float32(1.0) + gate_reciprocals[index])
                        K.ptx.rcp.approx.ftz.f32(gate_reciprocals[index], gate_reciprocals[index])
                        K.assign(
                            gate_reciprocals[index], gate_values[index] * gate_reciprocals[index]
                        )
                        K.assign(c_values[index], x_values[index] * gate_reciprocals[index])

            def accumulate_amax():
                subtile_amax = K.local_scalar("float32", init=K.float32(0.0))
                for index in range(epi_n):
                    K.ptx.abs.f32(absolute_values[index], c_values[index])
                    K.ptx.max.NaN.f32(subtile_amax, subtile_amax, absolute_values[index])
                K.ptx.max.f32(tile_amax, tile_amax, subtile_amax)

            def generate_sfc_for_pair(pair):
                scales_per_pair = epi_n // sf_vec_size
                scale_count = 4 * scales_per_pair
                for group in range(scales_per_pair):
                    group_amax = K.local_scalar("float32", init=K.float32(0.0))
                    for index in range(sf_vec_size):
                        value_index = group * sf_vec_size + index
                        K.ptx.abs.f32(absolute_values[value_index], c_values[value_index])
                        K.ptx.max.NaN.f32(group_amax, group_amax, absolute_values[value_index])
                    K.assign(group_amaxes[group], group_amax)

                scale_base = pair * scales_per_pair
                reciprocal_limit = K.float32(1.0 / (448.0 if c_dtype == "float8_e4m3fn" else 128.0))
                if vector_f32:
                    if scales_per_pair == 2:
                        multiply_pair(
                            raw_scales[scale_base],
                            raw_scales[scale_base + 1],
                            group_amaxes[0],
                            group_amaxes[1],
                            reciprocal_limit,
                            reciprocal_limit,
                        )
                        multiply_pair(
                            raw_scales[scale_base],
                            raw_scales[scale_base + 1],
                            raw_scales[scale_base],
                            raw_scales[scale_base + 1],
                            norm,
                            norm,
                        )
                    else:
                        unused_scale = K.local_scalar("float32")
                        multiply_pair(
                            raw_scales[scale_base],
                            unused_scale,
                            group_amaxes[0],
                            group_amaxes[0],
                            reciprocal_limit,
                            reciprocal_limit,
                        )
                        multiply_pair(
                            raw_scales[scale_base],
                            unused_scale,
                            raw_scales[scale_base],
                            unused_scale,
                            norm,
                            norm,
                        )
                else:
                    for group in range(scales_per_pair):
                        K.assign(
                            raw_scales[scale_base + group],
                            group_amaxes[group] * reciprocal_limit * norm,
                        )

                with K.If(pair == 3):
                    with K.Then():
                        for scale_pair in range(scale_count // 2):
                            if sf_dtype == "float8_e8m0fnu":
                                K.ptx.cvt.rp.satfinite.ue8m0x2.f32(
                                    sfc_store_pairs[scale_pair],
                                    raw_scales[scale_pair * 2 + 1],
                                    raw_scales[scale_pair * 2],
                                )
                            else:
                                K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                    sfc_store_pairs[scale_pair],
                                    raw_scales[scale_pair * 2 + 1],
                                    raw_scales[scale_pair * 2],
                                )
                        for word in range(scale_count // 4):
                            K.assign(
                                sfc_words[word],
                                K.bitwise_or(
                                    K.cast(sfc_store_pairs[word * 2], "uint32"),
                                    K.shift_left(
                                        K.cast(sfc_store_pairs[word * 2 + 1], "uint32"),
                                        K.uint32(16),
                                    ),
                                ),
                            )
                            global_row = tile_m_idx * cta_m + warp * 32 + lane
                            global_group = tile_n_idx * (n_tile // (2 * sf_vec_size)) + word * 4
                            sfc_rm = global_row // 128
                            sfc_m0 = global_row % 32
                            sfc_m1 = (global_row // 32) % 4
                            sfc_rk = global_group // 4
                            sfc_k0 = global_group % 4
                            sfc_rest_m = _ceil_div(M, 128)
                            sfc_rest_k = _ceil_div(N // 2, 4 * sf_vec_size)
                            sfc_offset = (
                                (
                                    ((batch_idx * sfc_rest_m + sfc_rm) * sfc_rest_k + sfc_rk) * 32
                                    + sfc_m0
                                )
                                * 16
                                + sfc_m1 * 4
                                + sfc_k0
                            )
                            K.ptx.st.global_.b32(sfc.ptr_to([sfc_offset]), sfc_words[word])

                for scale_pair in range(scale_count // 2):
                    if sf_dtype == "float8_e8m0fnu":
                        K.ptx.cvt.rp.satfinite.ue8m0x2.f32(
                            sf_pairs[scale_pair],
                            raw_scales[scale_pair * 2 + 1],
                            raw_scales[scale_pair * 2],
                        )
                        K.ptx.cvt.rn.bf16x2.ue8m0x2(
                            decoded_sf_pairs[scale_pair], sf_pairs[scale_pair]
                        )
                        K.ptx.cvt.f32.bf16(
                            decoded_scales[scale_pair * 2],
                            K.cast(
                                K.bitwise_and(decoded_sf_pairs[scale_pair], K.uint32(0xFFFF)),
                                "uint16",
                            ),
                        )
                        K.ptx.cvt.f32.bf16(
                            decoded_scales[scale_pair * 2 + 1],
                            K.cast(
                                K.shift_right(decoded_sf_pairs[scale_pair], K.uint32(16)), "uint16"
                            ),
                        )
                    else:
                        K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                            sf_pairs[scale_pair],
                            raw_scales[scale_pair * 2 + 1],
                            raw_scales[scale_pair * 2],
                        )
                        K.ptx.cvt.rn.f16x2.e4m3x2(
                            decoded_sf_pairs[scale_pair], sf_pairs[scale_pair]
                        )
                        K.ptx.cvt.f32.f16(
                            decoded_scales[scale_pair * 2],
                            K.cast(
                                K.bitwise_and(decoded_sf_pairs[scale_pair], K.uint32(0xFFFF)),
                                "uint16",
                            ),
                        )
                        K.ptx.cvt.f32.f16(
                            decoded_scales[scale_pair * 2 + 1],
                            K.cast(
                                K.shift_right(decoded_sf_pairs[scale_pair], K.uint32(16)), "uint16"
                            ),
                        )

                for group in range(scales_per_pair):
                    scale_index = pair * scales_per_pair + group
                    K.ptx.rcp.approx.ftz.f32(quant_scales[group], decoded_scales[scale_index])

                if vector_f32:
                    if scales_per_pair == 2:
                        multiply_pair(
                            quant_scales[0],
                            quant_scales[1],
                            quant_scales[0],
                            quant_scales[1],
                            norm,
                            norm,
                        )
                    else:
                        unused_scale = K.local_scalar("float32")
                        multiply_pair(
                            quant_scales[0],
                            unused_scale,
                            quant_scales[0],
                            quant_scales[0],
                            norm,
                            norm,
                        )
                else:
                    for group in range(scales_per_pair):
                        K.assign(quant_scales[group], norm * quant_scales[group])

                for group in range(scales_per_pair):
                    K.ptx["min.f32"](
                        quant_scales[group], quant_scales[group], K.float32(3.4028234663852886e38)
                    )
                    if vector_f32:
                        for index in range(group * sf_vec_size, (group + 1) * sf_vec_size, 2):
                            multiply_pair(
                                c_values[index],
                                c_values[index + 1],
                                c_values[index],
                                c_values[index + 1],
                                quant_scales[group],
                                quant_scales[group],
                            )
                    else:
                        for index in range(group * sf_vec_size, (group + 1) * sf_vec_size):
                            K.assign(c_values[index], c_values[index] * quant_scales[group])

            def pack_values(values, dtype, words, pairs):
                if dtype == "float32":
                    for index in range(epi_n):
                        K.assign(words[index], K.reinterpret("uint32", values[index]))
                elif dtype in {"float16", "bfloat16"}:
                    for index in range(epi_n // 2):
                        if dtype == "float16":
                            K.ptx.cvt.rn.f16x2.f32(
                                words[index], values[index * 2 + 1], values[index * 2]
                            )
                        else:
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[index], values[index * 2 + 1], values[index * 2]
                            )
                else:
                    for index in range(epi_n // 2):
                        if dtype == "float8_e4m3fn":
                            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                pairs[index], values[index * 2 + 1], values[index * 2]
                            )
                        else:
                            K.ptx.cvt.rn.satfinite.e5m2x2.f32(
                                pairs[index], values[index * 2 + 1], values[index * 2]
                            )
                    for index in range(epi_n // 4):
                        K.assign(
                            words[index],
                            K.bitwise_or(
                                K.cast(pairs[index * 2], "uint32"),
                                K.shift_left(K.cast(pairs[index * 2 + 1], "uint32"), K.uint32(16)),
                            ),
                        )

            def pack_ab12():
                pack_values(x_values, ab12_dtype, ab12_x_words, ab12_x_pairs)
                pack_values(gate_values, ab12_dtype, ab12_gate_words, ab12_gate_pairs)

            def pack_c():
                if scalar_c_store:
                    for index in range(epi_n):
                        if c_dtype == "float16":
                            K.ptx.cvt.rn.f16.f32(c_halves[index], c_values[index])
                        else:
                            K.ptx.cvt.rn.bf16.f32(c_halves[index], c_values[index])
                else:
                    pack_values(c_values, c_dtype, c_words, c_pairs)

            def n_major_offset(region_offset, stage, stage_bytes, row_bytes, vector, inner_bytes=0):
                unswizzled = (
                    smem_base
                    + region_offset
                    + stage * stage_bytes
                    + warp * (32 * row_bytes)
                    + lane * row_bytes
                    + inner_bytes
                    + vector * 16
                )
                swizzle_mask = {32: 16, 64: 48, 128: 112, 256: 112}[row_bytes]
                swizzled = K.bitwise_xor(
                    unswizzled,
                    K.bitwise_and(K.shift_right(unswizzled, K.uint32(3)), K.uint32(swizzle_mask)),
                )
                return K.cast(swizzled - smem_base, "int32")

            def store_n_major(words, offsets, word_count):
                for vector in range(word_count // 4):
                    K.ptx.st.shared.v4.b32(
                        smem.ptr_to([offsets[vector]]),
                        words[vector * 4],
                        words[vector * 4 + 1],
                        words[vector * 4 + 2],
                        words[vector * 4 + 3],
                    )

            def store_m_major(
                words, region_offset, stage, stage_bytes, bits, count, interleaved=False
            ):
                if bits == 32:
                    scalar_base = (
                        smem_base
                        + region_offset
                        + stage * stage_bytes
                        + warp * (8192 if interleaved else 4096)
                        + lane * 4
                    )
                    for index in range(count):
                        unswizzled = scalar_base + (index % 8) * 128 + (index // 8) * 1024
                        swizzled = K.bitwise_xor(
                            unswizzled,
                            K.bitwise_and(K.shift_right(unswizzled, K.uint32(2)), K.uint32(96)),
                        )
                        K.ptx.st.shared.b32(
                            smem.ptr_to([K.cast(swizzled - smem_base, "int32")]), words[index]
                        )
                elif bits == 16:
                    thread = warp * 32 + lane
                    temporary = K.bitwise_or(
                        K.bitwise_and(thread << 5, K.int32(6144)),
                        K.bitwise_and(thread, K.int32(40)),
                    )
                    raw_address = K.bitwise_or(
                        K.bitwise_or(K.bitwise_and(thread << 7, K.int32(896)), temporary << 1),
                        K.bitwise_and(thread << 6, K.int32(1024)),
                    )
                    if interleaved:
                        # A 16-bit M-major TMA box is split into two 64-row
                        # copies.  Keep each copy's up/gate halves adjacent:
                        # [up0, gate0, up1, gate1].
                        raw_address = raw_address + K.bitwise_and(thread << 6, K.int32(4096))
                    first_unswizzled = smem_base + region_offset + stage * stage_bytes + raw_address
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
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        smem.ptr_to([K.cast(first_swizzled - smem_base + 2048, "int32")]),
                        words[4],
                        words[5],
                        words[6],
                        words[7],
                    )
                    if count == 32:
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([K.cast(second_swizzled - smem_base, "int32")]),
                            words[8],
                            words[9],
                            words[10],
                            words[11],
                        )
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([K.cast(second_swizzled - smem_base + 2048, "int32")]),
                            words[12],
                            words[13],
                            words[14],
                            words[15],
                        )
                else:
                    thread = warp * 32 + lane
                    raw_address = K.bitwise_or(
                        K.bitwise_and(thread, K.int32(224)),
                        K.bitwise_and(thread << 7, K.int32(3968)),
                    )
                    first_unswizzled = smem_base + region_offset + stage * stage_bytes + raw_address
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
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
                    K.ptx.stmatrix.sync.aligned.m16n8.x4.trans.shared.b8(
                        smem.ptr_to([K.cast(second_swizzled - smem_base, "int32")]),
                        words[4],
                        words[5],
                        words[6],
                        words[7],
                    )

            def store_m_major_scalar_b16(halves, region_offset, stage, stage_bytes):
                thread = warp * 32 + lane
                row_group = K.bitwise_or(
                    K.bitwise_and(thread, K.int32(192)), K.bitwise_and(warp, K.int32(1))
                )
                scalar_base = (
                    smem_base
                    + region_offset
                    + stage * stage_bytes
                    + (row_group << 6)
                    + K.bitwise_and(thread << 1, K.int32(62))
                )
                for index in range(32):
                    unswizzled = scalar_base + (index % 16) * 128 + (index // 16) * 2048
                    swizzled = K.bitwise_xor(
                        unswizzled,
                        K.bitwise_and(K.shift_right(unswizzled, K.uint32(3)), K.uint32(112)),
                    )
                    K.ptx.st.shared.b16(
                        smem.ptr_to([K.cast(swizzled - smem_base, "int32")]), halves[index]
                    )

            def stage_ab12(ab12_stage):
                if c_major == "n":
                    ab12_row_bytes = 32 * ab12_bits // 8 if ab12_bits == 32 else 64 * ab12_bits // 8
                    ab12_word_count = 32 * ab12_bits // 32
                    for vector in range(ab12_word_count // 4):
                        K.assign(
                            ab12_x_offsets[vector],
                            n_major_offset(
                                ab12_offset, ab12_stage, ab12_stage_bytes, ab12_row_bytes, vector
                            ),
                        )
                        K.assign(
                            ab12_gate_offsets[vector],
                            n_major_offset(
                                (
                                    ab12_offset + ab12_stage_bytes // 2
                                    if ab12_bits == 32
                                    else ab12_offset
                                ),
                                ab12_stage,
                                ab12_stage_bytes,
                                ab12_row_bytes,
                                vector,
                                0 if ab12_bits == 32 else 32 * ab12_bits // 8,
                            ),
                        )
                    store_n_major(ab12_x_words, ab12_x_offsets, ab12_word_count)
                    store_n_major(ab12_gate_words, ab12_gate_offsets, ab12_word_count)
                else:
                    store_m_major(
                        ab12_x_words, ab12_offset, ab12_stage, ab12_stage_bytes, ab12_bits, 32, True
                    )
                    store_m_major(
                        ab12_gate_words,
                        ab12_offset + 4096,
                        ab12_stage,
                        ab12_stage_bytes,
                        ab12_bits,
                        32,
                        True,
                    )

            def stage_c(c_stage):
                if c_major == "n":
                    c_row_bytes = epi_n * c_bits // 8
                    c_word_count = epi_n * c_bits // 32
                    for vector in range(c_word_count // 4):
                        K.assign(
                            c_offsets[vector],
                            n_major_offset(c_offset, c_stage, c_stage_bytes, c_row_bytes, vector),
                        )
                    store_n_major(c_words, c_offsets, c_word_count)
                else:
                    if scalar_c_store:
                        store_m_major_scalar_b16(c_halves, c_offset, c_stage, c_stage_bytes)
                    else:
                        store_m_major(c_words, c_offset, c_stage, c_stage_bytes, c_bits, epi_n)

            def tma_store_output(
                map_ptr, region_offset, stage, stage_bytes, bits, columns, n_coord
            ):
                if c_major == "n":
                    K.ptx[
                        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                    ](
                        K.address_of(map_ptr),
                        K.cast(n_coord, "int32"),
                        K.cast(tile_m_idx * cta_m, "int32"),
                        K.cast(batch_idx, "int32"),
                        smem.ptr_to([region_offset + stage * stage_bytes]),
                        K.uint64(tma_cache_hint),
                    )
                else:
                    for row_copy in range(bits // 8):
                        K.ptx[
                            "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                            ".bulk_group.L2::cache_hint"
                        ](
                            K.address_of(map_ptr),
                            K.cast(tile_m_idx * cta_m + row_copy * (cta_m // (bits // 8)), "int32"),
                            K.cast(n_coord, "int32"),
                            K.cast(batch_idx, "int32"),
                            smem.ptr_to(
                                [
                                    region_offset
                                    + stage * stage_bytes
                                    + row_copy * (stage_bytes // (bits // 8))
                                ]
                            ),
                            K.uint64(tma_cache_hint),
                        )
                del columns

            def execute_epilogue_pair(subtile, pair):
                load_accumulator_pair(subtile)
                apply_alpha()
                pack_ab12()
                previous_subtiles = executed_tiles * epilogue_subtiles
                ab12_write_stage = K.bitwise_and(previous_subtiles + pair, K.int32(3))
                ab12_tma_stage = K.bitwise_and(pair, K.int32(3))
                stage_ab12(ab12_write_stage)
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                with K.If(warp == 0):
                    with K.Then():
                        ab12_n = tile_n_idx * n_tile + subtile * epi_n
                        tma_store_output(
                            ab12_map,
                            ab12_offset,
                            ab12_tma_stage,
                            ab12_stage_bytes,
                            ab12_bits,
                            N,
                            ab12_n,
                        )
                        if c_major == "n" and ab12_bits == 32:
                            tma_store_output(
                                ab12_map,
                                ab12_offset + ab12_stage_bytes // 2,
                                ab12_tma_stage,
                                ab12_stage_bytes,
                                ab12_bits,
                                N,
                                ab12_n + epi_n,
                            )
                        K.ptx.cp.async_.bulk.commit_group()
                        K.ptx.cp.async_.bulk.wait_group.read(3)
                K.ptx.bar.sync(K.uint32(1), K.uint32(128))

                apply_swiglu()
                if generate_amax:
                    accumulate_amax()
                if generate_sfc:
                    generate_sfc_for_pair(pair)
                pack_c()
                c_linear_stage = previous_subtiles + pair
                if (c_stages & (c_stages - 1)) == 0:
                    c_stage = K.bitwise_and(c_linear_stage, K.int32(c_stages - 1))
                else:
                    c_stage = c_linear_stage % c_stages
                stage_c(c_stage)
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                with K.If(warp == 0):
                    with K.Then():
                        c_n = tile_n_idx * (n_tile // 2) + pair * epi_n
                        tma_store_output(
                            c_map, c_offset, c_stage, c_stage_bytes, c_bits, N // 2, c_n
                        )
                        K.ptx.cp.async_.bulk.commit_group()
                        K.ptx.cp.async_.bulk.wait_group.read(c_stages - 1)
                K.ptx.bar.sync(K.uint32(1), K.uint32(128))

            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                if generate_amax:
                    K.assign(tile_amax, K.float32(0.0))
                if generate_sfc:
                    with K.unroll(0, epilogue_subtiles // 2) as pair:
                        execute_epilogue_pair(pair * 2, pair)
                else:
                    subtile = K.local_scalar("int32", init=0)
                    with K.While(subtile < epilogue_subtiles):
                        execute_epilogue_pair(subtile, subtile // 2)
                        K.assign(subtile, subtile + 2)
                if generate_amax:
                    K.ptx.redux_sync.max.NaN.f32(warp_amax, tile_amax, K.uint32(0xFFFFFFFF))
                    with K.If(lane == 0):
                        with K.Then():
                            K.ptx.st.shared.b32(smem.ptr_to([amax_offset + warp * 4]), warp_amax)
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    with K.If((warp == 0) & (lane == 0)):
                        with K.Then():
                            block_amax = K.local_scalar("float32")
                            next_amax = K.local_scalar("float32")
                            K.ptx.ld.shared.b32(block_amax, smem.ptr_to([amax_offset]))
                            K.ptx.max.f32(block_amax, block_amax, K.float32(0.0))
                            for slot in range(1, 4):
                                K.ptx.ld.shared.b32(
                                    next_amax, smem.ptr_to([amax_offset + slot * 4])
                                )
                                K.ptx.max.f32(block_amax, block_amax, next_amax)
                            atomic_old = K.local_scalar("int32")
                            K.ptx.atom.global_.max.s32(
                                atomic_old, amax.ptr_to([0]), K.reinterpret("int32", block_amax)
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
        "c": K.gptr[K.u8, (M * (N // 2) * L * c_bits // 8,)],
        "ab12": K.gptr[K.u8, (M * N * L * ab12_bits // 8,)],
        "amax": K.gptr[K.f32, (1,)],
        "sfc": K.gptr[K.u8, (L * _ceil_div(M, 128) * _ceil_div(N // 2, 4 * sf_vec_size) * 512,)],
        "norm_const": K.gptr[K.f32, (1,)],
        "alpha": K.f32,
    }
    return K.kernel(
        warps=6,
        arch="sm_100a",
        min_blocks_per_sm=3,
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
    ab12_dtype,
    c_dtype,
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
        ab12_dtype,
        c_dtype,
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


def _row_factors(torch, rows, batch, group_rows):
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
    batch_index = torch.arange(batch, device="cuda", dtype=torch.int64)[None, :]
    return ((((row // group_rows) + batch_index) % 3).float() + 1.0) * 0.5


def _input_tensor(torch, rows, K_dim, batch, dtype, major, group_rows):
    factors = _row_factors(torch, rows, batch, group_rows)
    if dtype == "float4_e2m1fn":
        if K_dim % 2:
            raise ValueError("packed FP4 K must be even")
        codes = (factors * 2.0).to(torch.uint8).transpose(0, 1)
        packed = codes | (codes << 4)
        physical = packed[:, :, None].expand(batch, rows, K_dim // 2).contiguous()
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


def _decode_sf_storage(torch, storage, rows, columns, batch, vec_size):
    row = torch.arange(rows, device="cuda", dtype=torch.int64)
    group = torch.arange(_ceil_div(columns, vec_size), device="cuda", dtype=torch.int64)
    m0 = row % 32
    m1 = (row // 32) % 4
    rm = row // 128
    k0 = group % 4
    rk = group // 4
    batches = []
    for batch_index in range(batch):
        batches.append(
            storage[
                batch_index, rm[:, None], rk[None, :], m0[:, None], m1[:, None], k0[None, :]
            ].float()
        )
    return torch.stack(batches, dim=2)


def prepare_data(**config):
    """Allocate one patterned input set shared by TIRx and the source kernel."""
    import torch

    config = _without_label(config)
    M, N, K_dim, L = (config[key] for key in ("M", "N", "K", "L"))
    sf_dtype = "float8_e8m0fnu" if config["sf_dtype"] == "int8" else config["sf_dtype"]
    a_data = _input_tensor(torch, M, K_dim, L, config["ab_dtype"], config["a_major"], 128)
    b_data = _input_tensor(torch, N, K_dim, L, config["ab_dtype"], config["b_major"], 32)
    input_scale = 1.0 if K_dim <= 320 else 256.0 / K_dim
    sfa = _sf_storage(torch, M, K_dim, L, config["sf_vec_size"], sf_dtype, input_scale)
    sfb = _sf_storage(torch, N, K_dim, L, config["sf_vec_size"], sf_dtype, 1.0)
    tirx_ab12 = _output_tensor(torch, M, N, L, config["ab12_dtype"], config["c_major"])
    source_ab12 = _output_tensor(torch, M, N, L, config["ab12_dtype"], config["c_major"])
    tirx_c = _output_tensor(torch, M, N // 2, L, config["c_dtype"], config["c_major"])
    source_c = _output_tensor(torch, M, N // 2, L, config["c_dtype"], config["c_major"])
    tirx_amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    source_amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    tirx_sfc = _sf_storage(torch, M, N // 2, L, config["sf_vec_size"], sf_dtype, 0.0)
    source_sfc = _sf_storage(torch, M, N // 2, L, config["sf_vec_size"], sf_dtype, 0.0)
    norm_const = torch.ones(1, dtype=torch.float32, device="cuda")
    return {
        "a": a_data,
        "b": b_data,
        "sfa": sfa,
        "sfb": sfb,
        "tirx_ab12": tirx_ab12,
        "source_ab12": source_ab12,
        "tirx_c": tirx_c,
        "source_c": source_c,
        "tirx_amax": tirx_amax,
        "source_amax": source_amax,
        "tirx_sfc": tirx_sfc,
        "source_sfc": source_sfc,
        "norm_const": norm_const,
        "input_scale": input_scale,
    }


def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module(
        "cudnn.gemm.cutedsl.dense.swiglu.dense_blockscaled_gemm_persistent_swiglu_interleaved_quant"
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
    ab12 = data["source_ab12"]["source"]
    c = data["source_c"]["source"]
    generate_amax = config["ab_dtype"] == "float4_e2m1fn" and config["c_dtype"] == "bfloat16"
    generate_sfc = config["c_dtype"].startswith("float8_")
    amax = data["source_amax"] if generate_amax else None
    sfc = data["source_sfc"] if generate_sfc else None
    norm_const = data["norm_const"] if generate_sfc else None
    a_cute = from_dlpack(a, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["a_major"] == "m" else 1
    )
    b_cute = from_dlpack(b, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["b_major"] == "n" else 1
    )
    sfa_cute = from_dlpack(sfa, assumed_align=16)
    sfb_cute = from_dlpack(sfb, assumed_align=16)
    ab12_cute = from_dlpack(ab12, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["c_major"] == "m" else 1
    )
    c_cute = from_dlpack(c, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["c_major"] == "m" else 1
    )
    amax_cute = from_dlpack(amax, assumed_align=16) if amax is not None else None
    sfc_cute = from_dlpack(sfc, assumed_align=16) if sfc is not None else None
    norm_const_cute = from_dlpack(norm_const, assumed_align=16) if norm_const is not None else None
    kernel = module.Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=config["sf_vec_size"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
        vector_f32=config["vector_f32"],
        ab12_stages=4,
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
        ab12_tensor=ab12_cute,
        amax_tensor=amax_cute,
        sfc_tensor=sfc_cute,
        norm_const_tensor=norm_const_cute,
        alpha=config["alpha"],
        max_active_clusters=max_active_clusters,
        stream=make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def launch():
        executable(a, b, sfa, sfb, c, ab12, amax, sfc, norm_const, config["alpha"], stream)

    launch._keep_alive = (executable, a, b, sfa, sfb, c, ab12, amax, sfc, norm_const, stream)
    return launch


def _tirx_launch(executable, data, alpha):
    import torch

    a = data["a"]["raw"]
    b = data["b"]["raw"]
    sfa = data["sfa"].view(torch.uint8).reshape(-1)
    sfb = data["sfb"].view(torch.uint8).reshape(-1)
    c = data["tirx_c"]["raw"]
    ab12 = data["tirx_ab12"]["raw"]
    amax = data["tirx_amax"]
    sfc = data["tirx_sfc"].view(torch.uint8).reshape(-1)
    norm_const = data["norm_const"]

    def launch():
        executable(a, b, sfa, sfb, c, ab12, amax, sfc, norm_const, alpha)

    launch._keep_alive = (a, b, sfa, sfb, c, ab12, amax, sfc, norm_const)
    return launch


def _assert_close(torch, actual, expected, *, atol, rtol, label):
    actual_f32 = actual.float()
    if actual_f32.shape != expected.shape:
        raise AssertionError(f"{label} shape mismatch: {actual_f32.shape} != {expected.shape}")
    close = torch.isclose(actual_f32, expected, atol=atol, rtol=rtol)
    if bool(torch.all(close).item()):
        return
    absolute_error = torch.abs(actual_f32 - expected)
    allowed_error = atol + rtol * torch.abs(expected)
    excess = absolute_error - allowed_error
    worst_index = int(torch.argmax(excess).item())
    raise AssertionError(
        f"{label} mismatch: value={float(actual_f32.reshape(-1)[worst_index].item())}, "
        f"expected={float(expected.reshape(-1)[worst_index].item())}, "
        f"absolute_error={float(absolute_error.reshape(-1)[worst_index].item())}"
    )


def _validate_outputs(data, config, *, with_source):
    """Hold TIRx to the upstream kernel's outputs on the same bytes.

    The upstream implementation is the sole arbiter; numeric validation only
    runs when the source ran.
    """
    if with_source:
        import torch

        ab12_tolerance = 0.1 if config["ab12_dtype"].startswith("float8_") else 0.01
        _assert_close(
            torch,
            data["tirx_ab12"]["source"],
            data["source_ab12"]["source"].float(),
            atol=ab12_tolerance,
            rtol=ab12_tolerance,
            label="TIRx AB12 versus standalone source",
        )
        c_tolerance = 0.1 if config["c_dtype"].startswith("float8_") else 0.01
        _assert_close(
            torch,
            data["tirx_c"]["source"],
            data["source_c"]["source"].float(),
            atol=c_tolerance,
            rtol=c_tolerance,
            label="TIRx C versus standalone source",
        )
        if config["c_dtype"].startswith("float8_"):
            tirx_sfc = _decode_sf_storage(
                torch,
                data["tirx_sfc"],
                config["M"],
                config["N"] // 2,
                config["L"],
                config["sf_vec_size"],
            )
            source_sfc = _decode_sf_storage(
                torch,
                data["source_sfc"],
                config["M"],
                config["N"] // 2,
                config["L"],
                config["sf_vec_size"],
            )
            _assert_close(
                torch,
                tirx_sfc,
                source_sfc,
                atol=0.1,
                rtol=0.1,
                label="TIRx SFC versus standalone source",
            )
        if config["ab_dtype"] == "float4_e2m1fn" and config["c_dtype"] == "bfloat16":
            _assert_close(
                torch,
                data["tirx_amax"],
                data["source_amax"].float(),
                atol=0.1,
                rtol=0.1,
                label="TIRx amax versus standalone source",
            )


def run_test(**config):
    """Compare TIRx with the cuDNN Frontend kernel on identical inputs."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _tirx_launch(
        compile_kernel(get_kernel(**kernel_config)), data, kernel_config["alpha"]
    )
    source_launch = _compile_reference(data, kernel_config)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    _validate_outputs(data, kernel_config, with_source=True)
    return {"max_abs": float(data["source_ab12"]["source"].float().abs().amax().item())}


def prepare_bench(**config):
    """Compile only TIRx; reference imports and CUDA work stay in ``run_gpu``."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {"config": kernel_config, "executable": compile_kernel(get_kernel(**kernel_config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then time pure launches with the canonical timer."""
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
        tirx_launch()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                if source_launch is None:
                    source_launch = _compile_reference(data, kernel_config)
                    gpu_state["source_launch"] = source_launch
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
