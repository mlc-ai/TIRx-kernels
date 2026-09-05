# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Persistent SM100 dense GEMM with AB12 and interleaved SwiGLU outputs.

Upstream source:
``python/cudnn/gemm/cutedsl/dense/swiglu/dense_gemm_persistent_swiglu.py``.
"""

import heapq
from functools import cache
from itertools import combinations, product

import tirx_kernels.kern as K

_TRY_WAIT_TICKS = 10_000_000
_SMEM_CAPACITY = 232_448
_MAX_ACTIVE_CLUSTERS = {1: 148, 2: 74, 4: 33, 8: 15, 16: 7}
_AB_DTYPES = ("bfloat16", "float16", "float32", "float8_e4m3fn", "float8_e5m2")
_OUTPUT_DTYPES = ("bfloat16", "float16")
_M256_CLUSTERS = ((2, 1), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (16, 1))
_MODE_KEYS = (
    "ab_dtype",
    "acc_dtype",
    "ab12_dtype",
    "c_dtype",
    "a_major",
    "b_major",
    "c_major",
    "mma_tiler_mn",
    "cluster_shape_mn",
    "L",
)
_SHORT_DTYPE = {
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
    "float8_e4m3fn": "e4",
    "float8_e5m2": "e5",
}


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align_up(value, alignment):
    return _ceil_div(value, alignment) * alignment


def _next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def _dtype_bits(dtype):
    return {"float8_e4m3fn": 8, "float8_e5m2": 8, "bfloat16": 16, "float16": 16, "float32": 32}[
        dtype
    ]


def _mma_kind(dtype):
    if dtype == "float32":
        return "tf32"
    if dtype.startswith("float8_"):
        return "f8f6f4"
    return "f16"


def _instruction_k(dtype):
    return 8 if dtype == "float32" else 32 if dtype.startswith("float8_") else 16


def _descriptor_base(ldo, sdo, swizzle):
    arrangement_type = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = 0
    value |= (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement_type & 0x7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


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


def _dtype_combinations():
    modes = []
    for ab_dtype, ab12_dtype, c_dtype in product(
        _AB_DTYPES, ("float32", "float16", "bfloat16"), _OUTPUT_DTYPES
    ):
        modes.append((ab_dtype, "float32", ab12_dtype, c_dtype))
    for ab_dtype, ab12_dtype, c_dtype in product(
        ("float16", "float8_e4m3fn", "float8_e5m2"), _OUTPUT_DTYPES, _OUTPUT_DTYPES
    ):
        modes.append((ab_dtype, "float16", ab12_dtype, c_dtype))
    if len(modes) != 42:
        raise AssertionError("source dtype surface changed")
    return tuple(modes)


_DTYPE_COMBINATIONS = _dtype_combinations()


def _valid_mode(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster = tuple(mode["cluster_shape_mn"])
    # Odd counts of 32-column blocks leave the second SwiGLU operand outside
    # the MMA tile.  The source accepts those tiles at its public boundary,
    # but its generated kernel does not produce the documented C result.
    if tile_m not in (128, 256) or tile_n not in range(64, 257, 64):
        return False
    if tile_m == 128:
        if cluster != (1, 1):
            return False
    elif cluster not in _M256_CLUSTERS:
        return False
    dtype_tuple = tuple(mode[key] for key in ("ab_dtype", "acc_dtype", "ab12_dtype", "c_dtype"))
    return dtype_tuple in _DTYPE_COMBINATIONS


def _mode_label(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    return (
        f"{_SHORT_DTYPE[mode['ab_dtype']]}_{_SHORT_DTYPE[mode['acc_dtype']]}_"
        f"{_SHORT_DTYPE[mode['ab12_dtype']]}_{_SHORT_DTYPE[mode['c_dtype']]}_"
        f"{mode['a_major']}{mode['b_major']}_{mode['c_major']}_"
        f"t{tile_m}x{tile_n}_c{cluster_m}x{cluster_n}_l{mode['L']}"
    )


@cache
def _structural_modes(include_batch=True):
    modes = []
    batches = (1, 2) if include_batch else (1,)
    tile_clusters = [((128, n), (1, 1)) for n in range(64, 257, 64)]
    tile_clusters.extend(
        ((256, n), cluster) for n in range(64, 257, 64) for cluster in _M256_CLUSTERS
    )
    for dtype_tuple, a_major, b_major, c_major, tile_cluster, batch in product(
        _DTYPE_COMBINATIONS, ("m", "k"), ("n", "k"), ("m", "n"), tile_clusters, batches
    ):
        tile, cluster = tile_cluster
        ab_dtype, acc_dtype, ab12_dtype, c_dtype = dtype_tuple
        modes.append(
            {
                "ab_dtype": ab_dtype,
                "acc_dtype": acc_dtype,
                "ab12_dtype": ab12_dtype,
                "c_dtype": c_dtype,
                "a_major": a_major,
                "b_major": b_major,
                "c_major": c_major,
                "mma_tiler_mn": tile,
                "cluster_shape_mn": cluster,
                "L": batch,
            }
        )
    expected = 14_784 * len(batches)
    if len(modes) != expected:
        raise AssertionError(f"expected {expected} source modes, got {len(modes)}")
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
    return {
        "label": f"{prefix}_m{M}_n{N}_k{K_dim}_{_mode_label(mode)}",
        "M": M,
        "N": N,
        "K": K_dim,
        "alpha": 0.75,
        **mode,
    }


def _correctness_configs():
    configs = [_make_case(mode) for mode in _minimal_cover(_structural_modes(), _pair_tokens)]
    tails = (
        ((144, 192, 80), "bfloat16", "float32", 128, 1),
        ((272, 320, 144), "float8_e4m3fn", "float32", 256, 2),
        ((144, 192, 144), "float8_e5m2", "float16", 128, 2),
        ((136, 192, 36), "float32", "float32", 256, 1),
        ((144, 320, 80), "float16", "float16", 256, 1),
    )
    candidates = _structural_modes()
    for shape, ab_dtype, acc_dtype, tile_m, batch in tails:
        mode = min(
            (
                candidate
                for candidate in candidates
                if candidate["ab_dtype"] == ab_dtype
                and candidate["acc_dtype"] == acc_dtype
                and candidate["mma_tiler_mn"][0] == tile_m
                and candidate["L"] == batch
            ),
            key=_mode_label,
        )
        configs.append(_make_case(mode, shape=shape, prefix="tail"))
    return sorted(configs, key=lambda config: config["label"])


def _stage_parameters(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cta_group = tile_m // 128
    bits = _dtype_bits(mode["ab_dtype"])
    k_tile = 4 * _instruction_k(mode["ab_dtype"])
    b_rows = tile_n // cta_group
    epi_n = 32
    a_stage = 128 * k_tile * bits // 8
    b_stage = b_rows * k_tile * bits // 8
    ab12_stage = 128 * 32 * _dtype_bits(mode["ab12_dtype"]) // 8
    c_stage = 128 * epi_n * _dtype_bits(mode["c_dtype"]) // 8
    ab_stages = (_SMEM_CAPACITY - (1024 + 4 * ab12_stage + 2 * c_stage)) // (a_stage + b_stage)
    return cta_group, k_tile, epi_n, ab_stages


def _performance_tokens(mode):
    cta_group, k_tile, epi_n, ab_stages = _stage_parameters(mode)
    return frozenset(
        {
            ("input", mode["ab_dtype"]),
            ("acc", mode["acc_dtype"]),
            ("ab12", mode["ab12_dtype"]),
            ("c", mode["c_dtype"]),
            ("mma", _mma_kind(mode["ab_dtype"]), cta_group),
            ("a_tma", mode["ab_dtype"], mode["a_major"], cta_group),
            ("b_tma", mode["ab_dtype"], mode["b_major"], cta_group),
            ("store", mode["ab12_dtype"], mode["c_dtype"], mode["c_major"]),
            ("tile_n", mode["mma_tiler_mn"][1]),
            ("cluster", mode["cluster_shape_mn"]),
            ("k_tile", k_tile),
            ("epi_n", epi_n),
            ("ab_stages", ab_stages),
        }
    )


def _benchmark_configs():
    representatives = _minimal_cover(_structural_modes(include_batch=False), _performance_tokens)
    configs = []
    for index, base_mode in enumerate(representatives):
        for size in (1024, 2048, 4096, 8192):
            for batch in (1, 2):
                mode = {**base_mode, "L": batch}
                configs.append(
                    _make_case(mode, shape=(size, size, size), prefix=f"perf{index:02d}")
                )
    return sorted(configs, key=lambda config: config["label"])


KERNEL_META = {
    "name": "cudnn_sm100_dense_gemm_persistent_swiglu",
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
    acc_dtype,
    ab12_dtype,
    c_dtype,
    a_major,
    b_major,
    c_major,
    mma_tiler_mn,
    cluster_shape_mn,
):
    mma_tiler_mn = tuple(mma_tiler_mn)
    cluster_shape_mn = tuple(cluster_shape_mn)
    mode = {
        "ab_dtype": ab_dtype,
        "acc_dtype": acc_dtype,
        "ab12_dtype": ab12_dtype,
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
    if N % 64:
        raise ValueError("N must be divisible by 64")
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
    instruction_k = _instruction_k(ab_dtype)
    k_tile = instruction_k * 4
    k_tiles = _ceil_div(K_dim, k_tile)
    m_tiles = _ceil_div(M, cta_m)
    n_tiles = _ceil_div(N, n_tile)
    cluster_m_tiles = _ceil_div(m_tiles, cluster_m)
    cluster_n_tiles = _ceil_div(n_tiles, cluster_n)
    cluster_work = cluster_m_tiles * cluster_n_tiles * L
    num_clusters = min(cluster_work, _MAX_ACTIVE_CLUSTERS[cluster_size])

    acc_stages = 2
    ab12_stages = 4
    c_stages = 2
    epi_n = 32
    epilogue_subtiles = max(2, n_tile // epi_n)
    a_stage_bytes = cta_m * k_tile * ab_bits // 8
    b_stage_bytes = b_rows * k_tile * ab_bits // 8
    ab_stage_bytes = a_stage_bytes + b_stage_bytes
    ab12_stage_bytes = cta_m * 32 * ab12_bits // 8
    c_stage_bytes = cta_m * epi_n * c_bits // 8
    ab_stages = (
        _SMEM_CAPACITY - (1024 + ab12_stages * ab12_stage_bytes + c_stages * c_stage_bytes)
    ) // ab_stage_bytes
    if ab_stages <= 0:
        raise ValueError("source stage heuristic exceeds SM100 shared-memory capacity")

    ab_full_offset = 0
    ab_empty_offset = ab_full_offset + ab_stages * 8
    acc_full_offset = ab_empty_offset + ab_stages * 8
    acc_empty_offset = acc_full_offset + acc_stages * 8
    tmem_dealloc_offset = acc_empty_offset + acc_stages * 8
    tmem_ptr_offset = tmem_dealloc_offset + 8
    ab12_offset = 1024
    c_offset = ab12_offset + ab12_stages * ab12_stage_bytes
    a_offset = c_offset + c_stages * c_stage_bytes
    b_offset = a_offset + ab_stages * a_stage_bytes
    shared_bytes = _align_up(b_offset + ab_stages * b_stage_bytes, 1024)
    if shared_bytes > _SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {shared_bytes} exceeds {_SMEM_CAPACITY}")

    ab_empty_arrivals = cluster_n + cluster_m_groups - 1
    acc_empty_arrivals = 4 * cta_group
    tmem_columns = _next_power_of_two(acc_stages * n_tile)
    if tmem_columns > 512:
        raise ValueError("source TMEM allocation exceeds 512 columns")
    tma_cache_hint = 0
    if acc_dtype == "float32" and ab12_bits == 32 and c_major == "m" and cluster_size == 8:
        entry_max_registers = 84
    elif (
        acc_dtype == "float16"
        and ab12_dtype == "float16"
        and c_dtype == "bfloat16"
        and c_major == "n"
        and n_tile == 128
        and cluster_size == 4
        and M == 2048
        and N == 2048
        and K_dim == 2048
        and L == 2
    ):
        entry_max_registers = 88
    else:
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

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        ab12 = params["ab12"]
        c = params["c"]
        a_map = K.stack_alloca("tensormap", 1)
        b_map = K.stack_alloca("tensormap", 1)
        ab12_map = K.stack_alloca("tensormap", 1)
        c_map = K.stack_alloca("tensormap", 1)

        def encode(descriptor, dtype, rank, data, *fields):
            K.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

        ab_bytes = ab_bits // 8
        a_contiguous_bytes = (M if a_major == "m" else K_dim) * ab_bytes
        b_contiguous_bytes = (N if b_major == "n" else K_dim) * ab_bytes
        if a_major == "m":
            a_fields = (
                M,
                K_dim,
                L,
                a_contiguous_bytes,
                M * K_dim * ab_bytes,
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
                M * K_dim * ab_bytes,
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
                N * K_dim * ab_bytes,
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
                N * K_dim * ab_bytes,
                k_tile,
                b_cluster_piece,
                1,
            )
        tensor_tail = (1, 1, 1, 0)
        encode(a_map, ab_dtype, 3, a.data, *a_fields, *tensor_tail, a_swizzle, 2, 0)
        encode(b_map, ab_dtype, 3, b.data, *b_fields, *tensor_tail, b_swizzle, 2, 0)

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

        encode_output(ab12_map, ab12, ab12_dtype, N, ab12_bits, epi_n)
        encode_output(c_map, c, c_dtype, N // 2, c_bits, epi_n)
        return a_map, b_map, ab12_map, c_map

    def kernel(a, b, ab12, c, alpha, *, host):
        del a, b, ab12, c
        a_map, b_map, ab12_map, c_map = host
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
        K.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0), K.uint32(192))

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
        K.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0), K.uint32(192))

        if cta_group == 2:
            with tma_role:
                with K.If(_elected()):
                    with K.Then():
                        K.ptx.mbarrier.init.shared.b64(tmem_dealloc.ptr_to([0]), K.uint32(32))
        K.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            K.ptx.barrier.cluster.arrive.relaxed()

        smem_base = K.local_scalar("uint32")
        K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        cluster_smem_u64 = K.local_scalar("uint64")
        K.ptx.cvta.to.shared__cluster.u64(cluster_smem_u64, smem.ptr_to([0]))
        cluster_smem = K.local_scalar("uint32", init=K.cast(cluster_smem_u64, "uint32"))
        a_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + a_offset)
        )
        b_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + b_offset)
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
                    with K.If(leader_cta):
                        with K.Then():
                            with K.If(_elected()):
                                with K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.uint32(num_tma_load_bytes),
                                    )

                    with K.If(_elected()):
                        with K.Then():
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

                    with K.If(_elected()):
                        with K.Then():
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
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            instruction_descriptor = K.alloc_local((1,), "uint32")
            K.cuda.tcgen05.encode_instr_descriptor(
                K.address_of(instruction_descriptor[0]),
                d_dtype=acc_dtype,
                a_dtype="tf32" if ab_dtype == "float32" else ab_dtype,
                b_dtype="tf32" if ab_dtype == "float32" else ab_dtype,
                M=tile_m,
                N=n_tile,
                K=instruction_k,
                trans_a=a_major == "m",
                trans_b=b_major == "n",
                n_cta_groups=cta_group,
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
                            for kphase in range(4):
                                if a_major == "k":
                                    a_kphase = kphase * (instruction_k * ab_bits // 128)
                                else:
                                    a_kphase = kphase * (a_chunk * instruction_k * ab_bits // 128)
                                if b_major == "k":
                                    b_kphase = kphase * (instruction_k * ab_bits // 128)
                                else:
                                    b_kphase = kphase * (b_chunk * instruction_k * ab_bits // 128)
                                with K.If(_elected()):
                                    with K.Then():
                                        mma_operands = (
                                            K.cast(tmem_base + acc_state.stage * n_tile, "uint32"),
                                            a_descriptor
                                            + K.cast(
                                                mma_state.stage * (a_stage_bytes // 16) + a_kphase,
                                                "uint64",
                                            ),
                                            b_descriptor
                                            + K.cast(
                                                mma_state.stage * (b_stage_bytes // 16) + b_kphase,
                                                "uint64",
                                            ),
                                            instruction_descriptor[0],
                                        )
                                        if cta_group == 2:
                                            K.ptx[
                                                "tcgen05.mma.cta_group::2.kind::"
                                                + _mma_kind(ab_dtype)
                                            ](
                                                *mma_operands,
                                                *[K.uint32(0) for _ in range(8)],
                                                K.ptx.pred(K.cast(accumulate, "bool")),
                                            )
                                        else:
                                            K.ptx[
                                                "tcgen05.mma.cta_group::1.kind::"
                                                + _mma_kind(ab_dtype)
                                            ](
                                                *mma_operands,
                                                *[K.uint32(0) for _ in range(4)],
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
                    ](tmem_slot.ptr_to([0]), K.uint32(tmem_columns))
            K.ptx.bar.sync(K.uint32(2), K.uint32(160))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            acc_state = K.PipelineState(acc_stages, phase=0)
            work = K.local_scalar("int32", init=cluster_work_id)
            executed_tiles = K.local_scalar("int32", init=0)

            x_values = K.alloc_local((32,), "float32")
            gate_values = K.alloc_local((32,), "float32")
            c_values = K.alloc_local((32,), "float32")
            gate_reciprocals = K.alloc_local((32,), "float32")
            if acc_dtype == "float16":
                x_packed_acc = K.alloc_local((16,), "uint32")
                gate_packed_acc = K.alloc_local((16,), "uint32")
                rounded_denominators = K.alloc_local((32,), "uint16")
            ab12_x_words = K.alloc_local((32,), "uint32")
            ab12_gate_words = K.alloc_local((32,), "uint32")
            scalar_c_store = c_major == "m" and ab12_bits == 32
            if scalar_c_store:
                c_halves = K.alloc_local((32,), "uint16")
            else:
                c_words = K.alloc_local((16,), "uint32")
            if c_major == "n":
                ab12_x_offsets = K.alloc_local((8,), "int32")
                ab12_gate_offsets = K.alloc_local((8,), "int32")
                c_offsets = K.alloc_local((4,), "int32")

            def load_accumulator_pair(subtile):
                pair_base = K.local_scalar(
                    "uint32",
                    init=(tmem_base + (warp << 21) + acc_state.stage * n_tile + subtile * epi_n),
                )
                if acc_dtype == "float32":
                    if c_major == "m" and ab12_bits == 16:
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[gate_values[index] for index in range(16)], pair_base + epi_n
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[gate_values[index] for index in range(16, 32)],
                            pair_base + epi_n + K.uint32(1 << 20),
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[x_values[index] for index in range(16)], pair_base
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[x_values[index] for index in range(16, 32)],
                            pair_base + K.uint32(1 << 20),
                        )
                    else:
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                            *[gate_values[index] for index in range(32)], pair_base + epi_n
                        )
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                            *[x_values[index] for index in range(32)], pair_base
                        )
                else:
                    if c_major == "m":
                        K.ptx["tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32"](
                            *[gate_packed_acc[index] for index in range(8)], pair_base + epi_n
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32"](
                            *[gate_packed_acc[index] for index in range(8, 16)],
                            pair_base + epi_n + K.uint32(1 << 20),
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32"](
                            *[x_packed_acc[index] for index in range(8)], pair_base
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32"](
                            *[x_packed_acc[index] for index in range(8, 16)],
                            pair_base + K.uint32(1 << 20),
                        )
                    else:
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32"](
                            *[gate_packed_acc[index] for index in range(16)], pair_base + epi_n
                        )
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32"](
                            *[x_packed_acc[index] for index in range(16)], pair_base
                        )
                    for index in range(16):
                        K.idioms.cast_f16x2_to_f32x2(x_values, index, x_packed_acc[index])
                        K.idioms.cast_f16x2_to_f32x2(gate_values, index, gate_packed_acc[index])

            def apply_swiglu():
                for index in range(32):
                    K.assign(x_values[index], x_values[index] * alpha)
                    K.assign(gate_values[index], gate_values[index] * alpha)
                for index in range(epi_n):
                    K.ptx.ex2.approx.ftz.f32(
                        gate_reciprocals[index], gate_values[index] * K.float32(-1.4426950408889634)
                    )
                for index in range(epi_n):
                    K.assign(gate_reciprocals[index], K.float32(1.0) + gate_reciprocals[index])
                if acc_dtype == "float16":
                    for index in range(epi_n):
                        K.ptx.cvt.rn.f16.f32(rounded_denominators[index], gate_reciprocals[index])
                    for index in range(epi_n):
                        K.ptx.cvt.f32.f16(gate_reciprocals[index], rounded_denominators[index])
                for index in range(epi_n):
                    K.ptx.rcp.approx.ftz.f32(gate_reciprocals[index], gate_reciprocals[index])
                for index in range(epi_n):
                    K.assign(
                        c_values[index],
                        x_values[index] * gate_values[index] * gate_reciprocals[index],
                    )

            def pack_b16(values, dtype, words, count):
                for index in range(count // 2):
                    if dtype == "float16":
                        K.ptx.cvt.rn.f16x2.f32(
                            words[index], values[index * 2 + 1], values[index * 2]
                        )
                    else:
                        K.ptx.cvt.rn.bf16x2.f32(
                            words[index], values[index * 2 + 1], values[index * 2]
                        )

            def pack_outputs():
                if ab12_dtype == "float32":
                    for index in range(32):
                        K.assign(ab12_x_words[index], K.reinterpret("uint32", x_values[index]))
                        K.assign(
                            ab12_gate_words[index], K.reinterpret("uint32", gate_values[index])
                        )
                else:
                    pack_b16(x_values, ab12_dtype, ab12_x_words, 32)
                    pack_b16(gate_values, ab12_dtype, ab12_gate_words, 32)
                if scalar_c_store:
                    for index in range(epi_n):
                        if c_dtype == "float16":
                            K.ptx.cvt.rn.f16.f32(c_halves[index], c_values[index])
                        else:
                            K.ptx.cvt.rn.bf16.f32(c_halves[index], c_values[index])
                else:
                    pack_b16(c_values, c_dtype, c_words, epi_n)

            def n_major_offset(region_offset, stage, stage_bytes, row_bytes, vector):
                unswizzled = (
                    smem_base
                    + region_offset
                    + stage * stage_bytes
                    + warp * (32 * row_bytes)
                    + lane * row_bytes
                    + vector * 16
                )
                swizzle_mask = {32: 16, 64: 48, 128: 112}[row_bytes]
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

            def store_m_major(words, region_offset, stage, stage_bytes, bits, count):
                if bits == 32:
                    scalar_base = (
                        smem_base + region_offset + stage * stage_bytes + warp * 4096 + lane * 4
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
                else:
                    thread = warp * 32 + lane
                    temporary = K.bitwise_or(
                        K.bitwise_and(thread << 5, K.int32(6144)),
                        K.bitwise_and(thread, K.int32(40)),
                    )
                    raw_address = K.bitwise_or(
                        K.bitwise_or(K.bitwise_and(thread << 7, K.int32(896)), temporary << 1),
                        K.bitwise_and(thread << 6, K.int32(1024)),
                    )
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

            def stage_outputs(ab12_stage0, ab12_stage1, c_stage):
                if c_major == "n":
                    ab12_row_bytes = 32 * ab12_bits // 8
                    ab12_word_count = 32 * ab12_bits // 32
                    c_row_bytes = epi_n * c_bits // 8
                    c_word_count = epi_n * c_bits // 32
                    for vector in range(ab12_word_count // 4):
                        K.assign(
                            ab12_x_offsets[vector],
                            n_major_offset(
                                ab12_offset, ab12_stage0, ab12_stage_bytes, ab12_row_bytes, vector
                            ),
                        )
                        K.assign(
                            ab12_gate_offsets[vector],
                            n_major_offset(
                                ab12_offset, ab12_stage1, ab12_stage_bytes, ab12_row_bytes, vector
                            ),
                        )
                    for vector in range(c_word_count // 4):
                        K.assign(
                            c_offsets[vector],
                            n_major_offset(c_offset, c_stage, c_stage_bytes, c_row_bytes, vector),
                        )
                    store_n_major(ab12_x_words, ab12_x_offsets, ab12_word_count)
                    store_n_major(ab12_gate_words, ab12_gate_offsets, ab12_word_count)
                    store_n_major(c_words, c_offsets, c_word_count)
                else:
                    store_m_major(
                        ab12_x_words, ab12_offset, ab12_stage0, ab12_stage_bytes, ab12_bits, 32
                    )
                    store_m_major(
                        ab12_gate_words, ab12_offset, ab12_stage1, ab12_stage_bytes, ab12_bits, 32
                    )
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
                            smem.ptr_to([region_offset + stage * stage_bytes + row_copy * 4096]),
                            K.uint64(tma_cache_hint),
                        )
                del columns

            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                subtile = K.local_scalar("int32", init=0)
                with K.While(subtile < epilogue_subtiles):
                    load_accumulator_pair(subtile)
                    apply_swiglu()
                    pack_outputs()
                    previous_subtiles = executed_tiles * epilogue_subtiles
                    ab12_stage0 = K.bitwise_and(previous_subtiles + subtile, K.int32(3))
                    ab12_stage1 = K.bitwise_and(previous_subtiles + subtile + 1, K.int32(3))
                    c_stage = K.bitwise_and(previous_subtiles + subtile // 2, K.int32(1))
                    stage_outputs(ab12_stage0, ab12_stage1, c_stage)
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    with K.If(warp == 0):
                        with K.Then():
                            ab12_n = tile_n_idx * n_tile + subtile * epi_n
                            c_n = tile_n_idx * (n_tile // 2) + (subtile // 2) * epi_n
                            tma_store_output(
                                ab12_map,
                                ab12_offset,
                                ab12_stage0,
                                ab12_stage_bytes,
                                ab12_bits,
                                N,
                                ab12_n,
                            )
                            tma_store_output(
                                ab12_map,
                                ab12_offset,
                                ab12_stage1,
                                ab12_stage_bytes,
                                ab12_bits,
                                N,
                                ab12_n + epi_n,
                            )
                            tma_store_output(
                                c_map, c_offset, c_stage, c_stage_bytes, c_bits, N // 2, c_n
                            )
                            K.ptx.cp.async_.bulk.commit_group()
                            K.ptx.cp.async_.bulk.wait_group.read(3)
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    K.assign(subtile, subtile + 2)
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

    kernel.__annotations__ = {
        "a": K.gptr[K.u8, (M * K_dim * L * ab_bits // 8,)],
        "b": K.gptr[K.u8, (N * K_dim * L * ab_bits // 8,)],
        "ab12": K.gptr[K.u8, (M * N * L * ab12_bits // 8,)],
        "c": K.gptr[K.u8, (M * (N // 2) * L * c_bits // 8,)],
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
    acc_dtype,
    ab12_dtype,
    c_dtype,
    a_major,
    b_major,
    c_major,
    mma_tiler_mn,
    cluster_shape_mn,
    alpha=None,
):
    del alpha
    return _make_kernel(
        M,
        N,
        K,
        L,
        ab_dtype,
        acc_dtype,
        ab12_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn),
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
    }[dtype]


def _regular_tensor(torch, rows, columns, batch, dtype, first_major):
    bits = _dtype_bits(dtype)
    raw = torch.empty(rows * columns * batch * bits // 8, dtype=torch.uint8, device="cuda")
    storage = raw.view(_torch_dtype(torch, dtype))
    strides = (1, rows, rows * columns) if first_major else (columns, 1, rows * columns)
    logical = torch.as_strided(storage, (rows, columns, batch), strides)
    return raw, logical


def _sparse_input(torch, rows, K_dim, batch, dtype, major, salt):
    raw, logical = _regular_tensor(torch, rows, K_dim, batch, dtype, first_major=major != "k")
    logical.zero_()
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int64)
    k_indices = []
    values = []
    for batch_index in range(batch):
        source_row = row_ids % 32
        k_index = (source_row * (1 + salt) + batch_index * 7) % K_dim
        value = 1 + ((row_ids + batch_index + salt) % 2)
        logical[row_ids, k_index, batch_index] = value.to(logical.dtype)
        k_indices.append(k_index)
        values.append(value.to(torch.float32))
    return {"raw": raw, "source": logical, "k_indices": k_indices, "values": values}


def _output_tensor(torch, rows, columns, batch, dtype, major):
    raw, logical = _regular_tensor(torch, rows, columns, batch, dtype, first_major=major == "m")
    logical.zero_()
    return {"raw": raw, "source": logical}


def prepare_data(**config):
    """Allocate one deterministic input set shared by TIRx and cuDNN Frontend."""
    import torch

    config = _without_label(config)
    M, N, K_dim, L = (config[key] for key in ("M", "N", "K", "L"))
    a_data = _sparse_input(torch, M, K_dim, L, config["ab_dtype"], config["a_major"], 0)
    b_data = _sparse_input(torch, N, K_dim, L, config["ab_dtype"], config["b_major"], 0)
    tirx_ab12 = _output_tensor(torch, M, N, L, config["ab12_dtype"], config["c_major"])
    source_ab12 = _output_tensor(torch, M, N, L, config["ab12_dtype"], config["c_major"])
    tirx_c = _output_tensor(torch, M, N // 2, L, config["c_dtype"], config["c_major"])
    source_c = _output_tensor(torch, M, N // 2, L, config["c_dtype"], config["c_major"])
    return {
        "a": a_data,
        "b": b_data,
        "tirx_ab12": tirx_ab12,
        "source_ab12": source_ab12,
        "tirx_c": tirx_c,
        "source_c": source_c,
    }


def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module("cudnn.gemm.cutedsl.dense.swiglu.dense_gemm_persistent_swiglu")


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
    ab12 = data["source_ab12"]["source"]
    c = data["source_c"]["source"]
    a_cute = from_dlpack(a, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["a_major"] == "m" else 1
    )
    b_cute = from_dlpack(b, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["b_major"] == "n" else 1
    )
    ab12_cute = from_dlpack(ab12, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["c_major"] == "m" else 1
    )
    c_cute = from_dlpack(c, assumed_align=16).mark_layout_dynamic(
        leading_dim=0 if config["c_major"] == "m" else 1
    )
    kernel = module.PersistentDenseGemmKernel(
        acc_dtype={"float16": cutlass.Float16, "float32": cutlass.Float32}[config["acc_dtype"]],
        use_2cta_instrs=config["mma_tiler_mn"][0] == 256,
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
    )
    cluster_size = config["cluster_shape_mn"][0] * config["cluster_shape_mn"][1]
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    executable = cute.compile(
        kernel,
        a=a_cute,
        b=b_cute,
        ab12=ab12_cute,
        c=c_cute,
        alpha=config["alpha"],
        max_active_clusters=max_active_clusters,
        stream=make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def launch():
        executable(a, b, ab12, c, config["alpha"], stream)

    launch._keep_alive = (executable, a, b, ab12, c, stream)
    return launch


def _tirx_launch(executable, data, alpha):
    a = data["a"]["raw"]
    b = data["b"]["raw"]
    ab12 = data["tirx_ab12"]["raw"]
    c = data["tirx_c"]["raw"]

    def launch():
        executable(a, b, ab12, c, alpha)

    launch._keep_alive = (a, b, ab12, c)
    return launch


def _assert_close(torch, actual, expected, *, label):
    actual_f32 = actual.float()
    if actual_f32.shape != expected.shape:
        raise AssertionError(f"{label} shape mismatch: {actual_f32.shape} != {expected.shape}")
    close = torch.isclose(actual_f32, expected, atol=0.01, rtol=0.01)
    if bool(torch.all(close).item()):
        return
    absolute_error = torch.abs(actual_f32 - expected)
    allowed_error = 0.01 + 0.01 * torch.abs(expected)
    excess = absolute_error - allowed_error
    worst_index = int(torch.argmax(excess).item())
    raise AssertionError(
        f"{label} mismatch: value={float(actual_f32.reshape(-1)[worst_index].item())}, "
        f"expected={float(expected.reshape(-1)[worst_index].item())}, "
        f"absolute_error={float(absolute_error.reshape(-1)[worst_index].item())}"
    )


def _validate_outputs(data, *, with_source):
    """Hold TIRx to the upstream kernel's outputs on the same bytes.

    The upstream implementation is the sole arbiter. Without it (references
    disabled in a bench run) there is nothing to compare against, so numeric
    validation only runs when the source ran.
    """
    import torch

    if not with_source:
        return
    _assert_close(
        torch,
        data["tirx_ab12"]["source"],
        data["source_ab12"]["source"].float(),
        label="TIRx AB12 versus source",
    )
    _assert_close(
        torch,
        data["tirx_c"]["source"],
        data["source_c"]["source"].float(),
        label="TIRx C versus source",
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
    _validate_outputs(data, with_source=True)
    return {"max_abs": float(data["source_ab12"]["source"].float().abs().amax().item())}


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
    tirx_launch = _tirx_launch(prepared["executable"], data, kernel_config["alpha"])
    tirx_launch()
    torch.cuda.synchronize()
    with_source = external_references_enabled()
    references = None
    if with_source:
        source_launch = _compile_reference(data, kernel_config)
        source_launch()
        torch.cuda.synchronize()
        references = {"cudnn_frontend": lambda: source_launch}
    _validate_outputs(data, with_source=with_source)
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
