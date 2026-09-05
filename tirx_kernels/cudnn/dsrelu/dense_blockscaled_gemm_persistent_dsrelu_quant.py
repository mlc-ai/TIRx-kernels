# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Persistent SM100 block-scaled GEMM with dSReLU and probability reduction.

Upstream source:
python/cudnn/gemm/cutedsl/dense/dsrelu/
dense_blockscaled_gemm_persistent_dsrelu_quant.py.
"""

import tirx_kernels.kern as K
from tirx_kernels.runner import prepare_cluster_shape, prepare_cuda_arch

_TRY_WAIT_TICKS = 10_000_000
_SMEM_CAPACITY = 232_448
_MAX_ACTIVE_CLUSTERS = {1: 148, 2: 74, 4: 33, 8: 15, 16: 7}
_SF_MODES = (("float8_e8m0fnu", 16), ("float8_e8m0fnu", 32), ("float8_e4m3fn", 16))
_C_DTYPES = ("float32", "float16", "bfloat16")
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


def _swizzled(row_offset, chunk, row_bytes):
    """Return one 16-byte chunk offset under the epilogue TensorMap swizzle."""
    chunks = row_bytes // 16
    if chunks == 1:
        return row_offset + K.int32(chunk * 16)
    twist = K.local_scalar(
        "int32", init=((row_offset // K.int32(128)) % K.int32(chunks)) * K.int32(16)
    )
    return K.local_scalar("int32", init=row_offset | (K.int32(chunk * 16) ^ twist))


def _unpack_input(values, words, dtype, bits):
    """Expand one thread's 32-element C row to FP32 registers."""
    if bits == 16:
        if dtype == "bfloat16":
            for word in range(16):
                low_bits = K.local_scalar("uint32")
                high_bits = K.local_scalar("uint32")
                K.ptx.shl.b32(low_bits, words[word], K.uint32(16))
                K.ptx.and_.b32(high_bits, words[word], K.uint32(0xFFFF0000))
                K.assign(values[2 * word], K.reinterpret("float32", low_bits))
                K.assign(values[2 * word + 1], K.reinterpret("float32", high_bits))
            return
        for word in range(16):
            low = K.local_scalar("uint16")
            high = K.local_scalar("uint16")
            K.ptx.mov.b32(low, high, words[word])
            K.ptx.cvt.f32.f16(values[2 * word], low)
            K.ptx.cvt.f32.f16(values[2 * word + 1], high)
    else:
        for word in range(32):
            K.ptx.mov.b32(values[word], words[word])


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
        and mode["with_sfd"] is False
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


# Every base record below corresponds to a PASS row in the preserved source
# capability manifest. Shapes may be replaced by BENCH_CONFIGS only after the
# structure itself has passed the source compile/run/oracle probe.
def _base_config(label="anchor", **updates):
    config = {
        "label": label,
        "M": 256,
        "N": 256,
        "K": 512,
        "L": 2,
        "ab_dtype": "float4_e2m1fn",
        "sf_dtype": "float8_e8m0fnu",
        "sf_vec_size": 16,
        "c_dtype": "bfloat16",
        "d_dtype": "bfloat16",
        "a_major": "k",
        "b_major": "k",
        "c_major": "n",
        "mma_tiler_mn": (256, 256),
        "cluster_shape_mn": (2, 1),
        "vector_f32": False,
        "with_amax": True,
        "with_sfd": False,
        "alpha": 2.0 / 3.0,
    }
    config.update(updates)
    return config


def _source_probe_configs():
    probes = [_base_config()]
    probes.extend(
        [
            _base_config("fp4_e8v32", sf_vec_size=32),
            _base_config("fp4_e4v16", sf_dtype="float8_e4m3fn"),
            _base_config(
                "fp8e4_e8v32_mn", ab_dtype="float8_e4m3fn", sf_vec_size=32, a_major="m", b_major="n"
            ),
            _base_config(
                "fp8e5_e8v32_mn", ab_dtype="float8_e5m2", sf_vec_size=32, a_major="m", b_major="n"
            ),
            _base_config("c_f16_d_bf16", c_dtype="float16"),
            _base_config("c_f32_d_f16", c_dtype="float32", d_dtype="float16"),
            _base_config("c_bf16_d_f32", d_dtype="float32"),
            _base_config("c_f16_d_f32", c_dtype="float16", d_dtype="float32"),
            _base_config("l1", L=1),
            _base_config("no_amax_fp4", with_amax=False),
            _base_config(
                "no_amax_fp8",
                ab_dtype="float8_e4m3fn",
                sf_vec_size=32,
                a_major="m",
                b_major="n",
                with_amax=False,
            ),
            _base_config(
                "explicit_amax_fp8",
                ab_dtype="float8_e5m2",
                sf_vec_size=32,
                a_major="m",
                b_major="n",
            ),
        ]
    )
    for dtype in ("float8_e4m3fn", "float8_e5m2"):
        for a_major in ("k", "m"):
            for b_major in ("k", "n"):
                probes.append(
                    _base_config(
                        f"major_{dtype}_{a_major}{b_major}",
                        ab_dtype=dtype,
                        sf_vec_size=32,
                        a_major=a_major,
                        b_major=b_major,
                        with_amax=False,
                    )
                )
    for tile_m in (128, 256):
        for tile_n in (64, 128, 192, 256):
            probes.append(
                _base_config(
                    f"tile_{tile_m}x{tile_n}",
                    N=192 if tile_n == 192 else 256,
                    mma_tiler_mn=(tile_m, tile_n),
                    cluster_shape_mn=(1, 1) if tile_m == 128 else (2, 1),
                )
            )
    for cluster_m in (1, 2, 4):
        for cluster_n in (1, 2, 4):
            probes.append(
                _base_config(
                    f"cluster_cta1_{cluster_m}x{cluster_n}",
                    mma_tiler_mn=(128, 256),
                    cluster_shape_mn=(cluster_m, cluster_n),
                )
            )
    for cluster_m in (2, 4):
        for cluster_n in (1, 2, 4):
            probes.append(
                _base_config(
                    f"cluster_cta2_{cluster_m}x{cluster_n}", cluster_shape_mn=(cluster_m, cluster_n)
                )
            )
    probes.extend(
        [
            _base_config("tail_m", M=272),
            _base_config("tail_n", N=272),
            _base_config("tail_k", K=544),
            _base_config("tail_mnk", M=272, N=272, K=544),
            _base_config(
                "tail_cta1_n64",
                M=272,
                N=208,
                K=544,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
            ),
            _base_config("tail_cta2_n192", M=272, N=208, K=544, mma_tiler_mn=(256, 192)),
            _base_config(
                "persistent_cta1", M=4096, N=512, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1)
            ),
            _base_config("persistent_cta2", M=4096, N=512, mma_tiler_mn=(256, 128)),
        ]
    )
    by_label = {}
    for probe in probes:
        by_label[probe["label"]] = probe
    return [by_label[label] for label in sorted(by_label)]


_STRUCTURE_KEYS = (
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
    "with_amax",
    "with_sfd",
    "L",
)


def _verified_structure(mode):
    for probe in _source_probe_configs():
        if all(probe[key] == mode[key] for key in _STRUCTURE_KEYS):
            return True
    return False


def _pair_tokens(config):
    tokens = set()
    for left in range(len(_STRUCTURE_KEYS)):
        for right in range(left + 1, len(_STRUCTURE_KEYS)):
            key_a = _STRUCTURE_KEYS[left]
            key_b = _STRUCTURE_KEYS[right]
            tokens.add((key_a, repr(config[key_a]), key_b, repr(config[key_b])))
    return tokens


def _minimal_pairwise_cover(candidates, forced_labels):
    uncovered = set()
    for candidate in candidates:
        uncovered.update(_pair_tokens(candidate))
    selected = []
    remaining = list(candidates)
    for label in forced_labels:
        for candidate in tuple(remaining):
            if candidate["label"] == label:
                selected.append(candidate)
                remaining.remove(candidate)
                uncovered.difference_update(_pair_tokens(candidate))
                break
    while uncovered:
        best = max(
            remaining,
            key=lambda candidate: (
                len(_pair_tokens(candidate) & uncovered),
                tuple(-ord(char) for char in candidate["label"]),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(_pair_tokens(best))
    return selected


def _correctness_configs():
    forced = (
        "anchor",
        "l1",
        "no_amax_fp4",
        "tail_mnk",
        "tail_cta1_n64",
        "tail_cta2_n192",
        "persistent_cta1",
        "persistent_cta2",
        "cluster_cta1_4x4",
        "cluster_cta2_4x4",
    )
    return _minimal_pairwise_cover(_source_probe_configs(), forced)


def _structure_tokens(config):
    return {
        ("ab", config["ab_dtype"]),
        ("sf", config["sf_dtype"], config["sf_vec_size"]),
        ("cd", config["c_dtype"], config["d_dtype"]),
        ("major", config["a_major"], config["b_major"]),
        ("tile", config["mma_tiler_mn"]),
        ("cluster", config["cluster_shape_mn"]),
        ("amax", config["with_amax"]),
        ("batch", config["L"]),
    }


def _benchmark_configs():
    probes = _source_probe_configs()
    uncovered = set()
    for probe in probes:
        uncovered.update(_structure_tokens(probe))
    representatives = []
    while uncovered:
        best = max(
            probes,
            key=lambda probe: (
                len(_structure_tokens(probe) & uncovered),
                tuple(-ord(char) for char in probe["label"]),
            ),
        )
        representatives.append(best)
        probes.remove(best)
        uncovered.difference_update(_structure_tokens(best))

    configs = []
    anchor = _base_config()
    for size, batch in ((1024, 1), (2048, 2), (4096, 1), (8192, 2)):
        config = dict(anchor)
        config.update(
            label=f"anchor_m{size}_n{size}_k{size}_l{batch}", M=size, N=size, K=size, L=batch
        )
        configs.append(config)
    for index, base in enumerate(representatives):
        size = (1024, 2048, 4096, 8192)[index % 4]
        config = dict(base)
        config.update(
            label=f"cover{index:02d}_m{size}_n{size}_k{size}_{base['label']}",
            M=size,
            N=size,
            K=size,
        )
        configs.append(config)
    by_label = {}
    for config in configs:
        by_label[config["label"]] = config
    return [by_label[label] for label in sorted(by_label)]


KERNEL_META = {
    "name": "cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
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
    with_amax,
    with_sfd,
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
        "with_amax": with_amax,
        "with_sfd": with_sfd,
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
    c_full_offset = acc_empty_offset + acc_stages * 8
    c_empty_offset = c_full_offset + c_stages * 8
    tmem_dealloc_offset = c_empty_offset + c_stages * 8
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

    def kernel(a, b, sfa, sfb, c, d, prob, dprob, amax, alpha, *, host):
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
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3], regs=88)
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])
        c_role = roles.role("c_load", warps=[6])

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
        if protocol_pool.bytes != c_full_offset:
            raise AssertionError("protocol storage changed before the C pipeline")
        c_pipe = K.Pipeline(
            protocol_pool, c_stages, full="tma", empty="mbar", init_empty=4, leader=K.bool(False)
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
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, c_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(c_pipe.full.ptr_to([stage]), K.uint32(1))
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, c_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                c_pipe.empty.ptr_to([stage]), K.uint32(4)
                            )

        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.sync(K.uint32(0), K.uint32(224))

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
            K.ptx.bar.sync(K.uint32(0), K.uint32(224))

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

        # Warp 6 follows the same persistent cursor and produces read-only C
        # subtiles into the two-stage mbarrier ring.
        with c_role:
            c_state = K.PipelineState(c_stages, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                subtile = K.local_scalar("int32", init=0)
                with K.While(subtile < epilogue_subtiles):
                    _wait_plain(c_pipe.empty.ptr_to([c_state.stage]), c_state.phase)
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                c_pipe.full.ptr_to([c_state.stage]), K.uint32(c_stage_bytes)
                            )
                            K.ptx[
                                "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint"
                            ](
                                smem.ptr_to([c_offset + c_state.stage * c_stage_bytes]),
                                K.address_of(c_map),
                                K.cast(tile_n_idx * n_tile + subtile * epi_n, "int32"),
                                K.cast(tile_m_idx * cta_m, "int32"),
                                K.cast(batch_idx, "int32"),
                                c_pipe.full.ptr_to([c_state.stage]),
                                K.uint64(tma_cache_hint),
                            )
                    _advance(c_state)
                    K.assign(subtile, subtile + 1)
                advance_work(work)
            with K.unroll(0, c_stages):
                _wait_plain(c_pipe.empty.ptr_to([c_state.stage]), c_state.phase)
                _advance(c_state)

        with epilogue_role:
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            acc_state = K.PipelineState(acc_stages, phase=0)
            c_state = K.PipelineState(c_stages, phase=0)
            work = K.local_scalar("int32", init=cluster_work_id)
            executed_tiles = K.local_scalar("int32", init=0)

            values = K.alloc_local((32,), "float32")
            c_values = K.alloc_local((32,), "float32")
            d_values = K.alloc_local((32,), "float32")
            dprob_terms = K.alloc_local((32,), "float32")
            c_words = K.alloc_local((32,), "uint32")
            d_words = K.alloc_local((32,), "uint32")
            d_pairs = K.alloc_local((16,), "uint16")
            d_wide = K.alloc_local((16,), "uint64")
            d_offsets = K.alloc_local((8,), "int32")
            absolute_values = K.alloc_local((32,), "float32")
            maxima = K.alloc_local((11,), "float32")
            tile_amax = K.local_scalar("float32")
            warp_amax = K.local_scalar("float32")
            prob_value = K.local_scalar("float32")
            dprob_value = K.local_scalar("float32")

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

            def add_pair(dst0, dst1, left0, left1, right0, right1):
                packed = K.local_scalar("uint64")
                K.ptx.add.rn.f32x2(
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

            def load_c():
                row_bytes = epi_n * c_bits // 8
                word_count = epi_n * c_bits // 32
                c_slot = c_offset + c_state.stage * c_stage_bytes
                row_offset = (warp * 32 + lane) * row_bytes
                for word in range(0, word_count, 4):
                    K.ptx.ld.shared.v4.b32(
                        c_words[word],
                        c_words[word + 1],
                        c_words[word + 2],
                        c_words[word + 3],
                        smem.ptr_to([c_slot + _swizzled(row_offset, word // 4, row_bytes)]),
                    )
                K.ptx.fence.proxy.async_.shared__cta()
                with K.If(lane == 0):
                    with K.Then():
                        K.ptx.mbarrier.arrive.shared.b64(c_pipe.empty.ptr_to([c_state.stage]))
                _advance(c_state)
                _unpack_input(c_values, c_words, c_dtype, c_bits)

            def pack_output(dtype, source_values, words, pairs, wide):
                if dtype == "float32":
                    for index in range(16):
                        K.assign(
                            wide[index],
                            K.cuda.make_float2(
                                source_values[index * 2], source_values[index * 2 + 1]
                            ),
                        )
                else:
                    for index in range(16):
                        if dtype == "float16":
                            K.ptx.cvt.rn.f16x2.f32(
                                words[index], source_values[index * 2 + 1], source_values[index * 2]
                            )
                        else:
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[index], source_values[index * 2 + 1], source_values[index * 2]
                            )

            def stage_output(dtype, bits, words, wide, offsets, stage):
                row_bytes = epi_n * bits // 8
                word_count = epi_n * bits // 32
                vector_count = row_bytes // 16
                row_offset = K.local_scalar(
                    "int32",
                    init=d_offset
                    + stage * d_stage_bytes
                    + warp * (32 * row_bytes)
                    + lane * row_bytes,
                )
                for vector in range(vector_count):
                    K.assign(
                        offsets[vector],
                        d_offset
                        + stage * d_stage_bytes
                        + _swizzled((warp * 32 + lane) * row_bytes, vector, row_bytes),
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
                del row_offset

            def tma_store_d(stage, subtile):
                K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"](
                    K.address_of(d_map),
                    K.cast(tile_n_idx * n_tile + subtile * epi_n, "int32"),
                    K.cast(tile_m_idx * cta_m, "int32"),
                    K.cast(batch_idx, "int32"),
                    smem.ptr_to([d_offset + stage * d_stage_bytes]),
                    K.uint64(tma_cache_hint),
                )

            def apply_dsrelu_and_dprob():
                for index in range(32):
                    K.ptx.max.f32(values[index], values[index], K.float32(0.0))
                for index in range(0, 32, 2):
                    c_prob_0 = K.local_scalar("float32")
                    c_prob_1 = K.local_scalar("float32")
                    doubled_0 = K.local_scalar("float32")
                    doubled_1 = K.local_scalar("float32")
                    squared_0 = K.local_scalar("float32")
                    squared_1 = K.local_scalar("float32")
                    multiply_pair(
                        c_prob_0,
                        c_prob_1,
                        c_values[index],
                        c_values[index + 1],
                        prob_value,
                        prob_value,
                    )
                    add_pair(
                        doubled_0,
                        doubled_1,
                        values[index],
                        values[index + 1],
                        values[index],
                        values[index + 1],
                    )
                    multiply_pair(
                        d_values[index],
                        d_values[index + 1],
                        doubled_0,
                        doubled_1,
                        c_prob_0,
                        c_prob_1,
                    )
                    multiply_pair(
                        squared_0,
                        squared_1,
                        values[index],
                        values[index + 1],
                        values[index],
                        values[index + 1],
                    )
                    multiply_pair(
                        dprob_terms[index],
                        dprob_terms[index + 1],
                        squared_0,
                        squared_1,
                        c_values[index],
                        c_values[index + 1],
                    )

                subtile_sum = K.local_scalar("float32", init=K.float32(0.0))
                for index in range(31, -1, -1):
                    K.ptx.add.f32(subtile_sum, subtile_sum, dprob_terms[index])
                K.ptx.add.f32(dprob_value, dprob_value, subtile_sum)

            def update_amax():
                for index in range(32):
                    K.ptx.abs.f32(absolute_values[index], d_values[index])
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
                subtile_amax = K.local_scalar("float32")
                K.ptx.max.NaN.f32(subtile_amax, maxima[0], maxima[3], K.float32(0.0))
                K.ptx.max.f32(tile_amax, tile_amax, subtile_amax)

            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                K.assign(tile_amax, K.float32(0.0))
                K.assign(dprob_value, K.float32(0.0))
                K.assign(prob_value, K.float32(0.0))
                global_row = tile_m_idx * cta_m + warp * 32 + lane
                with K.If(global_row < M):
                    with K.Then():
                        prob_offset = batch_idx * (m_tiles * cta_m) + global_row
                        K.ptx.ld.global_.b32(prob_value, prob.ptr_to([prob_offset]))

                subtile = K.local_scalar("int32", init=0)
                with K.While(subtile < epilogue_subtiles):
                    linear_subtile = executed_tiles * epilogue_subtiles + subtile
                    d_stage = K.bitwise_and(linear_subtile, K.int32(1))
                    load_accumulator(subtile)
                    apply_alpha()
                    _wait_plain(c_pipe.full.ptr_to([c_state.stage]), c_state.phase)
                    load_c()
                    apply_dsrelu_and_dprob()
                    if with_amax:
                        update_amax()

                    pack_output(d_dtype, d_values, d_words, d_pairs, d_wide)
                    stage_output(d_dtype, d_bits, d_words, d_wide, d_offsets, d_stage)
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    with K.If(warp == 0):
                        with K.Then():
                            tma_store_d(d_stage, subtile)
                            K.ptx.cp.async_.bulk.commit_group()
                            K.ptx.cp.async_.bulk.wait_group.read(d_stages - 1)
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    K.assign(subtile, subtile + 1)

                if with_amax:
                    K.idioms.warp_reduce_max_nan_f32(warp_amax, tile_amax)
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

                with K.If(global_row < M):
                    with K.Then():
                        atomic_old = K.local_scalar("float32")
                        K.ptx.atom.global_.add.f32(
                            atomic_old, dprob.ptr_to([global_row * L + batch_idx]), dprob_value
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

    kernel.__annotations__ = {
        "a": K.gptr[K.u8, (M * K_dim * L * ab_bits // 8,)],
        "b": K.gptr[K.u8, (N * K_dim * L * ab_bits // 8,)],
        "sfa": K.gptr[K.u8, (L * _ceil_div(M, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "sfb": K.gptr[K.u8, (L * _ceil_div(N, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "c": K.gptr[K.u8, (M * N * L * c_bits // 8,)],
        "d": K.gptr[K.u8, (M * N * L * d_bits // 8,)],
        "prob": K.gptr[K.f32, (m_tiles * cta_m * L,)],
        "dprob": K.gptr[K.f32, (M * L,)],
        "amax": K.gptr[K.f32, (1,)],
        "alpha": K.f32,
    }
    return K.kernel(
        warps=7,
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
    d_dtype,
    a_major,
    b_major,
    c_major,
    mma_tiler_mn,
    cluster_shape_mn,
    vector_f32,
    with_amax=True,
    with_sfd=False,
    alpha=None,
):
    del alpha
    if with_sfd:
        raise ValueError("the source SFD branch is not implemented")
    cluster_shape_mn = prepare_cluster_shape(cluster_shape_mn)
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
        with_amax,
        with_sfd,
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


def _c_input_tensor(torch, rows, columns, batch, dtype, major):
    raw, logical = _regular_tensor(torch, rows, columns, batch, dtype, first_major=major == "m")
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None, None]
    column = torch.arange(columns, device="cuda", dtype=torch.int64)[None, :, None]
    batch_index = torch.arange(batch, device="cuda", dtype=torch.int64)[None, None, :]
    # Keep both signs and seven deterministic levels while leaving headroom for
    # the source and target's different inter-CTA FP32 atomic reduction order.
    values = (((row // 17 + column // 13 + batch_index) % 7).float() - 3.0) * 0.0625
    logical.copy_(values.to(logical.dtype))
    return {"raw": raw, "source": logical}


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
    sfa = _sf_storage(torch, M, K_dim, L, config["sf_vec_size"], sf_dtype, 2.0 ** (-scale_exponent))
    sfb = _sf_storage(torch, N, K_dim, L, config["sf_vec_size"], sf_dtype, 0.5)
    c = _c_input_tensor(torch, M, N, L, config["c_dtype"], config["c_major"])
    tirx_d = _output_tensor(torch, M, N, L, config["d_dtype"], config["c_major"])
    source_d = _output_tensor(torch, M, N, L, config["d_dtype"], config["c_major"])

    padded_m = _ceil_div(M, 128) * 128
    prob = _prob_storage(torch, M, padded_m, L, config["a_major"])
    tirx_dprob = torch.zeros((M, 1, L), dtype=torch.float32, device="cuda")
    source_dprob = torch.zeros((M, 1, L), dtype=torch.float32, device="cuda")
    tirx_amax = torch.full((1,), float("-inf"), dtype=torch.float32, device="cuda")
    source_amax = torch.full((1,), float("-inf"), dtype=torch.float32, device="cuda")

    return {
        "a": a_data,
        "b": b_data,
        "sfa": sfa,
        "sfb": sfb,
        "c": c,
        "prob": prob,
        "tirx_d": tirx_d,
        "source_d": source_d,
        "tirx_dprob": tirx_dprob,
        "source_dprob": source_dprob,
        "tirx_amax": tirx_amax,
        "source_amax": source_amax,
    }


def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module(
        "cudnn.gemm.cutedsl.dense.dsrelu.dense_blockscaled_gemm_persistent_dsrelu_quant"
    )


def _compile_reference(data, config):
    from tirx_kernels.cudnn._reference import from_dlpack_typed, import_cutlass_reference

    cutlass = import_cutlass_reference()
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import make_fake_stream

    from_dlpack = from_dlpack_typed

    module = _load_reference_source()
    config = _without_label(config)
    a = data["a"]["source"]
    b = data["b"]["source"]
    sfa = data["sfa"]
    sfb = data["sfb"]
    c = data["c"]["source"]
    d = data["source_d"]["source"]
    prob = data["prob"]["source"]
    dprob = data["source_dprob"]
    amax = data["source_amax"] if config["with_amax"] else None

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
    dprob_cute = from_dlpack(dprob, assumed_align=16)
    amax_cute = from_dlpack(amax, assumed_align=16) if amax is not None else None

    cluster_shape_mn = prepare_cluster_shape(config["cluster_shape_mn"])
    kernel = module.Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=config["sf_vec_size"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=cluster_shape_mn,
        vector_f32=config["vector_f32"],
    )
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)

    def epilogue_op(x, y):
        return cute.where(x > 0, x, cute.full_like(x, 0)) * 2 * y

    executable = cute.compile(
        kernel,
        a_tensor=a_cute,
        b_tensor=b_cute,
        sfa_tensor=sfa_cute,
        sfb_tensor=sfb_cute,
        c_tensor=c_cute,
        d_tensor=d_cute,
        prob_tensor=prob_cute,
        dprob_tensor=dprob_cute,
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
        executable(a, b, sfa, sfb, c, d, prob, dprob, amax, None, None, config["alpha"], stream)

    launch._keep_alive = (executable, a, b, sfa, sfb, c, d, prob, dprob, amax, stream)
    return launch


def _tirx_launch(executable, data, alpha):
    import torch

    a = data["a"]["raw"]
    b = data["b"]["raw"]
    sfa = data["sfa"].view(torch.uint8).reshape(-1)
    sfb = data["sfb"].view(torch.uint8).reshape(-1)
    c = data["c"]["raw"]
    d = data["tirx_d"]["raw"]
    prob = data["prob"]["raw"]
    dprob = data["tirx_dprob"].reshape(-1)
    amax = data["tirx_amax"]

    def launch():
        executable(a, b, sfa, sfb, c, d, prob, dprob, amax, alpha)

    launch._keep_alive = (a, b, sfa, sfb, c, d, prob, dprob, amax)
    return launch


def _reset_outputs(data, prefix):
    data[f"{prefix}_d"]["source"].zero_()
    data[f"{prefix}_dprob"].zero_()
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


# Exact tolerances used by the pinned dense dSReLU source test.
_LOW_PRECISION_TOLERANCE = (0.12, 0.02)
_FP32_TOLERANCE = (0.12, 0.02)


def _validate_outputs(data, config, *, with_source):
    """Hold D, dprob, and enabled amax to the pinned source on identical bytes."""
    if not with_source:
        return
    import torch

    d_atol, d_rtol = _FP32_TOLERANCE if config["d_dtype"] == "float32" else _LOW_PRECISION_TOLERANCE
    _assert_close(
        torch,
        data["tirx_d"]["source"],
        data["source_d"]["source"],
        atol=d_atol,
        rtol=d_rtol,
        label="TIRx D versus pinned source",
    )
    _assert_close(
        torch,
        data["tirx_dprob"],
        data["source_dprob"],
        atol=_FP32_TOLERANCE[0],
        rtol=_FP32_TOLERANCE[1],
        label="TIRx dprob versus pinned source",
    )
    if config["with_amax"]:
        _assert_close(
            torch,
            data["tirx_amax"],
            data["source_amax"],
            atol=_FP32_TOLERANCE[0],
            rtol=_FP32_TOLERANCE[1],
            label="TIRx amax versus pinned source",
        )


def _validate_with_oracle(data, config):
    """Validate the structured block-scaled GEMM and dSReLU equation."""
    import math

    import torch

    M, N, K_dim, L = (config[key] for key in ("M", "N", "K", "L"))
    scale_exponent = math.ceil(math.log2(max(1.0, K_dim / 512.0)))
    scale_product = 2.0 ** (-scale_exponent) * 0.5
    accumulator = (
        data["a"]["factors"][:, None, :]
        * data["b"]["factors"][None, :, :]
        * (K_dim * scale_product * config["alpha"])
    )
    relu = torch.relu(accumulator)
    c = data["c"]["source"].float()
    prob = data["prob"]["values"][:, None, :]
    expected_d_f32 = 2.0 * relu * c * prob
    expected_d = expected_d_f32.to(_torch_dtype(torch, config["d_dtype"]))
    expected_dprob = (relu.square() * c).sum(dim=1, keepdim=True)
    d_atol, d_rtol = _FP32_TOLERANCE if config["d_dtype"] == "float32" else _LOW_PRECISION_TOLERANCE
    _assert_close(
        torch,
        data["tirx_d"]["source"],
        expected_d,
        atol=d_atol,
        rtol=d_rtol,
        label="TIRx D versus FP32 oracle",
    )
    _assert_close(
        torch,
        data["tirx_dprob"],
        expected_dprob,
        atol=_FP32_TOLERANCE[0],
        rtol=_FP32_TOLERANCE[1],
        label="TIRx dprob versus FP32 oracle",
    )
    if config["with_amax"]:
        _assert_close(
            torch,
            data["tirx_amax"],
            expected_d_f32.abs().amax().reshape(1),
            atol=_FP32_TOLERANCE[0],
            rtol=_FP32_TOLERANCE[1],
            label="TIRx amax versus FP32 oracle",
        )


def run_test(**config):
    """Compare TIRx with the pinned source, or the FP32 oracle on Thor."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    executable = compile_kernel(get_kernel(**kernel_config))
    tirx_launch = _tirx_launch(executable, data, kernel_config["alpha"])
    _reset_outputs(data, "tirx")
    tirx_launch()
    if prepare_cuda_arch() == "sm_110a":
        torch.cuda.synchronize()
        _validate_with_oracle(data, kernel_config)
        if not kernel_config["with_amax"]:
            # Only the optional amax branch emits redux.sync.max.NaN.f32,
            # which the pinned source compiler rejects for sm_110a.
            source_launch = _compile_reference(data, kernel_config)
            _reset_outputs(data, "source")
            source_launch()
            torch.cuda.synchronize()
            _validate_outputs(data, kernel_config, with_source=True)
        result = data["tirx_amax"]
    else:
        source_launch = _compile_reference(data, kernel_config)
        _reset_outputs(data, "source")
        source_launch()
        torch.cuda.synchronize()
        _validate_outputs(data, kernel_config, with_source=True)
        result = data["source_amax"]
    return {"amax": float(result.item())}


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
    with_source = external_references_enabled() and (
        prepare_cuda_arch() != "sm_110a" or not kernel_config["with_amax"]
    )
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
