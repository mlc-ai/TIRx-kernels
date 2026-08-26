# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Static specialization domain for the BF16 MoE grouped GEMM dGLU+dBias port.

The upstream kernel carries no configuration dataclass -- its domain is the
constructor keyword set guarded by ``can_implement_bf16_grouped_gemm``. This
module reproduces that domain as an explicit table, generates the correctness
and benchmark matrices from it, and derives the launch, stage and storage facts
each specialization needs.

Every value here is a compile-time fact for the port: shapes, expert count and
the activation scalars are baked into a specialization rather than passed at
runtime, which is why ``derive`` can return an exact byte map.
"""

import json
from itertools import combinations, product

# Upstream fixed pad size, decoupled from the tile size (source ``FIX_PAD_SIZE``).
FIX_PAD_SIZE = 256

# ``get_smem_capacity_in_bytes("sm_100")``.
SMEM_CAPACITY = 232_448

# Upstream ``epi_tile``; the epilogue walks 32-column subtiles of a 128-row block.
EPI_TILE = (128, 32)

# Instruction K of ``tcgen05.mma.kind::f16`` times the upstream ``mma_inst_tile_k``.
MMA_INST_K = 16
MMA_INST_TILE_K = 4
K_TILE = MMA_INST_K * MMA_INST_TILE_K

# Stand-in for the upstream ``HardwareInfo().get_max_active_clusters`` query, so the
# persistent grid stays a static specialization fact.
MAX_ACTIVE_CLUSTERS = {1: 148, 2: 74, 4: 33, 8: 15, 16: 7}

C_DTYPES = ("bfloat16", "float16", "float32")
D_DTYPES = ("bfloat16", "float16", "float32")
ACT_FUNCS = ("dswiglu", "dgeglu")
WEIGHT_MODES = ("dense", "discrete")
SCHED_MODES = ("static", "dynamic")
B_MAJORS = ("k", "n")

# ``linear_offset`` is the only activation scalar the upstream bf16 kernel leaves
# adjustable; 1.702 and the +-7.0 clamps are hard-coded in its ``dgeglu``. Upstream
# passes this one at runtime as a Float32; the port bakes it, like every other
# shape fact, and the reference call still supplies it at runtime.
DEFAULT_LINEAR_OFFSET = 1.0

MODE_KEYS = (
    "weight_mode",
    "sched",
    "act",
    "c_dtype",
    "d_dtype",
    "b_major",
    "mma_tiler_mn",
    "cluster_shape_mn",
    "vectorized_f32",
    "with_dbias",
)

_DTYPE_BITS = {"float16": 16, "bfloat16": 16, "float32": 32}

_SHORT = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "dswiglu": "sw",
    "dgeglu": "ge",
    "dense": "dn",
    "discrete": "dc",
    "static": "st",
    "dynamic": "dy",
}


def ceil_div(value, divisor):
    return -(-value // divisor)


def align_up(value, alignment):
    return ceil_div(value, alignment) * alignment


def dtype_bits(dtype):
    return _DTYPE_BITS[dtype]


def atom_thr(mode):
    """CTAs per MMA atom: two for the 256-row tile, one otherwise."""
    return 2 if mode["mma_tiler_mn"][0] == 256 else 1


def valid_mode(mode):
    """``can_implement_bf16_grouped_gemm`` plus the constructor's own checks.

    Mirrors ``moe_kernel_helpers.can_implement_bf16_grouped_gemm`` called with
    ``fix_pad_size=256, n_align=32, tile_n_align=32`` -- the values the bf16
    class pins -- and ``act_func in ("dswiglu", "dgeglu")``.
    """
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    use_2cta = tile_m == 256
    thr = 2 if use_2cta else 1

    if mode["act"] not in ACT_FUNCS:
        return False
    if mode["c_dtype"] not in C_DTYPES or mode["d_dtype"] not in D_DTYPES:
        return False
    if mode["b_major"] not in B_MAJORS:
        return False
    # ``mma_tiler_mn[0]`` is 128 for one CTA and 256 for two, with no other value.
    if tile_m not in (128, 256):
        return False
    if tile_n not in range(32, 257, 32):
        return False
    if cluster_m % thr != 0:
        return False
    if cluster_m <= 0 or cluster_n <= 0 or cluster_m * cluster_n > 16:
        return False
    if not _is_power_of_two(cluster_m) or not _is_power_of_two(cluster_n):
        return False
    if (cluster_m // thr) * tile_m not in (128, 256):
        return False
    # ``m_aligned`` is the fixed pad size, so it must divide the row tile exactly.
    if FIX_PAD_SIZE % tile_m != 0:
        return False
    return True


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def valid_shape(mode, *, tokens_total, N, K_dim, L):
    """Alignment rules the problem extents must satisfy for a valid mode."""
    if L < 1 or L > 1024:
        return False
    if tokens_total % 256 != 0:
        return False
    if N % 32 != 0:
        return False
    # ``is_valid_tensor_alignment``: the contiguous extent of each operand must
    # carry a whole 128-bit vector.
    if K_dim % (128 // 16) != 0:
        return False
    b_contig = K_dim if mode["b_major"] == "k" else N
    if b_contig % (128 // 16) != 0:
        return False
    if (2 * N) % (128 // dtype_bits(mode["d_dtype"])) != 0:
        return False
    if (2 * N) % (128 // dtype_bits(mode["c_dtype"])) != 0:
        return False
    return True


def mode_label(mode):
    tile_m, tile_n = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    parts = [
        _SHORT[mode["weight_mode"]] + _SHORT[mode["sched"]],
        _SHORT[mode["act"]],
        "c" + _SHORT[mode["c_dtype"]],
        "d" + _SHORT[mode["d_dtype"]],
        "b" + mode["b_major"],
        f"t{tile_m}x{tile_n}",
        f"c{cluster_m}x{cluster_n}",
        ("v" if mode["vectorized_f32"] else "s") + ("b" if mode["with_dbias"] else ""),
    ]
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Covering array over the independent axis groups
# ---------------------------------------------------------------------------


def _tile_group_values():
    """Every valid (tile_m, cluster) pair, with tile_n pinned to the API default.

    tile_n is deliberately excluded from the pairwise universe: coupling its
    eight values into this group would force at least one row per value per
    cluster, tripling the matrix. The tile_n axis is covered by an explicit
    sweep in the tail cases instead.
    """
    values = []
    for tile_m, cluster_m, cluster_n in product((128, 256), (1, 2, 4, 8, 16), (1, 2, 4, 8, 16)):
        mode = {"mma_tiler_mn": (tile_m, 256), "cluster_shape_mn": (cluster_m, cluster_n)}
        probe = dict(mode, act="dswiglu", c_dtype="bfloat16", d_dtype="bfloat16", b_major="k")
        if valid_mode(probe):
            values.append(mode)
    return values


def _groups():
    groups = [_tile_group_values()]
    groups.extend(
        [{key: value} for value in values]
        for key, values in (
            ("weight_mode", WEIGHT_MODES),
            ("sched", SCHED_MODES),
            ("act", ACT_FUNCS),
            ("c_dtype", C_DTYPES),
            ("d_dtype", D_DTYPES),
            ("b_major", B_MAJORS),
            ("vectorized_f32", (False, True)),
            ("with_dbias", (False, True)),
        )
    )
    return groups


def _token(key, value):
    return key, json.dumps(value)


def _pair(left, right):
    return (left, right) if left <= right else (right, left)


def _pair_universe(groups):
    universe = set()
    for index, values in enumerate(groups):
        keys = tuple(values[0])
        for value in values:
            for left, right in combinations(keys, 2):
                universe.add(_pair(_token(left, value[left]), _token(right, value[right])))
        for other_index in range(index + 1, len(groups)):
            other = groups[other_index]
            other_keys = tuple(other[0])
            left_tokens = {_token(key, value[key]) for value in values for key in keys}
            right_tokens = {_token(key, value[key]) for value in other for key in other_keys}
            for left in left_tokens:
                for right in right_tokens:
                    universe.add(_pair(left, right))
    return universe


def _row_pairs(mode):
    return {
        _pair(_token(left, mode[left]), _token(right, mode[right]))
        for left, right in combinations(MODE_KEYS, 2)
    }


def _group_gain(value, fixed, uncovered):
    keys = tuple(value)
    gain = 0
    for left, right in combinations(keys, 2):
        if _pair(_token(left, value[left]), _token(right, value[right])) in uncovered:
            gain += 1
    for key in keys:
        token = _token(key, value[key])
        for fixed_key, fixed_value in fixed.items():
            if _pair(_token(fixed_key, fixed_value), token) in uncovered:
                gain += 1
    return gain


def _matches(value, token):
    key, encoded = token
    return key in value and json.dumps(value[key]) == encoded


def minimal_pairwise_modes():
    """Greedy pairwise covering array over the domain's independent groups."""
    groups = _groups()
    uncovered = _pair_universe(groups)
    order = sorted(range(len(groups)), key=lambda index: -len(groups[index]))
    rows = []
    while uncovered:
        seed = min(sorted(uncovered))
        mode = {}
        for index in order:
            candidates = [
                value
                for value in groups[index]
                if all(_matches(value, token) for token in seed if token[0] in value)
            ]
            if not candidates:
                raise AssertionError(f"seed pair {seed} is unreachable in group {index}")
            best = max(
                candidates,
                key=lambda value: (
                    _group_gain(value, mode, uncovered),
                    tuple(sorted(map(str, value.items()))),
                ),
            )
            mode.update(best)
        pairs = _row_pairs(mode)
        if not (pairs & uncovered):
            raise AssertionError("pairwise coverage did not converge")
        uncovered.difference_update(pairs)
        rows.append(mode)
    return sorted(rows, key=mode_label)


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def make_case(mode, *, group_m_list, N, K_dim, prefix, index=None, linear_offset=None):
    """Turn one mode plus one problem shape into a config dictionary."""
    label_index = "" if index is None else f"{index:02d}"
    tokens = sum(group_m_list)
    label = "_".join(
        part
        for part in (
            f"{prefix}{label_index}",
            f"e{len(group_m_list)}",
            f"t{tokens}",
            f"n{N}",
            f"k{K_dim}",
            mode_label(mode),
        )
        if part
    )
    case = dict(mode)
    case.update(
        {
            "label": label,
            "group_m_list": tuple(group_m_list),
            "N": N,
            "K_dim": K_dim,
            "linear_offset": (
                DEFAULT_LINEAR_OFFSET if linear_offset is None else float(linear_offset)
            ),
        }
    )
    return case


def _base_mode(**overrides):
    mode = {
        "weight_mode": "dense",
        "sched": "static",
        "act": "dswiglu",
        "c_dtype": "bfloat16",
        "d_dtype": "bfloat16",
        "b_major": "k",
        "mma_tiler_mn": (256, 256),
        "cluster_shape_mn": (2, 1),
        "vectorized_f32": True,
        "with_dbias": True,
    }
    mode.update(overrides)
    return mode


def correctness_configs():
    """Pairwise cover at a small shape, plus boundary and tile_n tail cases."""
    cases = []
    for index, mode in enumerate(minimal_pairwise_modes()):
        cases.append(
            make_case(mode, group_m_list=(256,) * 4, N=512, K_dim=512, prefix="pair", index=index)
        )

    tails = [
        # Ragged and empty expert row ranges through the padded-offset walk.
        (_base_mode(), {"group_m_list": (256, 512, 256, 256), "N": 512, "K_dim": 512}),
        (_base_mode(), {"group_m_list": (256, 0, 512, 256), "N": 512, "K_dim": 512}),
        # Many k tiles with an N that is not a whole tile.
        (_base_mode(), {"group_m_list": (256,) * 2, "N": 320, "K_dim": 2048}),
        # More experts than the scheduler caches at once.
        (_base_mode(), {"group_m_list": (256,) * 8, "N": 512, "K_dim": 512}),
        # Discrete weights driven by the dynamic scheduler over a ragged range.
        (
            _base_mode(weight_mode="discrete", sched="dynamic"),
            {"group_m_list": (256, 512, 256), "N": 512, "K_dim": 512},
        ),
        # A single expert: one tile row, no cross-expert scheduling at all.
        (_base_mode(), {"group_m_list": (256,), "N": 512, "K_dim": 512}),
    ]
    for index, (mode, shape) in enumerate(tails):
        cases.append(make_case(mode, prefix="tail", index=index, **shape))

    # tile_n sweep, each value rotated through other axes so the rows also carry
    # incidental coverage rather than only exercising the tile.
    tile_n_rows = (
        (32, _base_mode(mma_tiler_mn=(128, 32), cluster_shape_mn=(1, 1))),
        (64, _base_mode(mma_tiler_mn=(128, 64), cluster_shape_mn=(1, 4), weight_mode="discrete")),
        (96, _base_mode(mma_tiler_mn=(256, 96), cluster_shape_mn=(2, 1))),
        (128, _base_mode(mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 2), b_major="n")),
        (160, _base_mode(mma_tiler_mn=(128, 160), cluster_shape_mn=(1, 1), c_dtype="float32")),
        (192, _base_mode(mma_tiler_mn=(128, 192), cluster_shape_mn=(1, 1), sched="dynamic")),
        (224, _base_mode(mma_tiler_mn=(128, 224), cluster_shape_mn=(1, 1), act="dgeglu")),
    )
    for index, (_tile_n, mode) in enumerate(tile_n_rows):
        cases.append(
            make_case(mode, group_m_list=(256,) * 2, N=512, K_dim=512, prefix="tilen", index=index)
        )

    # The one non-default activation scalar the bf16 kernel still exposes.
    cases.append(
        make_case(
            _base_mode(act="dgeglu"),
            group_m_list=(256,) * 2,
            N=512,
            K_dim=512,
            prefix="knob",
            index=0,
            linear_offset=0.0,
        )
    )

    for case in cases:
        if not valid_mode(case):
            raise AssertionError(f"invalid mode generated: {case['label']}")
        if not valid_shape(
            case,
            tokens_total=sum(case["group_m_list"]),
            N=case["N"],
            K_dim=case["K_dim"],
            L=len(case["group_m_list"]),
        ):
            raise AssertionError(f"invalid shape generated: {case['label']}")
    return cases


# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

HEADLINE_MODE = _base_mode()

# Small / medium / large MoE backward points, matching the blockscaled port so the
# two kernels' matrices stay comparable.
_PERF_SHAPES = (
    {"group_m_list": (1024,) * 4, "N": 2048, "K_dim": 2048},
    {"group_m_list": (2048,) * 8, "N": 4096, "K_dim": 8192},
    {"group_m_list": (4096,) * 8, "N": 4096, "K_dim": 7168},
)

_PERF_SWEEP = tuple(
    {"group_m_list": (tokens,) * experts, "N": 4096, "K_dim": K_dim}
    for experts in (4, 8)
    for tokens in (512, 1024, 2048, 4096)
    for K_dim in (2048, 8192)
)


def _performance_token(key, value):
    return f"{key}={json.dumps(value)}"


def _performance_tokens(mode):
    """Axis values that plausibly change the emitted program or its schedule."""
    tile_m, tile_n = mode["mma_tiler_mn"]
    return {
        _performance_token("weight_mode", mode["weight_mode"]),
        _performance_token("sched", mode["sched"]),
        _performance_token("act", mode["act"]),
        _performance_token("c_dtype", mode["c_dtype"]),
        _performance_token("d_dtype", mode["d_dtype"]),
        _performance_token("b_major", mode["b_major"]),
        _performance_token("tile_m", tile_m),
        _performance_token("tile_n", tile_n),
        _performance_token("cluster_shape_mn", list(mode["cluster_shape_mn"])),
        _performance_token("vectorized_f32", mode["vectorized_f32"]),
        _performance_token("with_dbias", mode["with_dbias"]),
    }


def performance_representatives():
    """Greedy cover of the performance-relevant axis values, headline first."""
    pool = list(minimal_pairwise_modes())
    extras = [
        _base_mode(mma_tiler_mn=(128, 64), cluster_shape_mn=(1, 1)),
        _base_mode(cluster_shape_mn=(2, 8)),
        _base_mode(mma_tiler_mn=(128, 256), cluster_shape_mn=(1, 16)),
    ]
    for extra in extras:
        if valid_mode(extra):
            pool.append(extra)

    covered = set(_performance_tokens(HEADLINE_MODE))
    chosen = []
    remaining = [mode for mode in pool if _performance_tokens(mode) - covered]
    while remaining:
        best = max(
            remaining, key=lambda mode: (len(_performance_tokens(mode) - covered), mode_label(mode))
        )
        covered |= _performance_tokens(best)
        chosen.append(best)
        remaining = [mode for mode in remaining if _performance_tokens(mode) - covered]
    return chosen


def benchmark_configs():
    """One family per representative mode across three shapes, plus a token sweep."""
    families = [HEADLINE_MODE, *performance_representatives()]
    cases = []
    for index, mode in enumerate(families):
        for shape in _PERF_SHAPES:
            cases.append(make_case(mode, prefix="perf", index=index, **shape))
    for shape in _PERF_SWEEP:
        cases.append(make_case(HEADLINE_MODE, prefix="sweep", index=0, **shape))

    for case in cases:
        if not valid_mode(case):
            raise AssertionError(f"invalid benchmark mode: {case['label']}")
        if not valid_shape(
            case,
            tokens_total=sum(case["group_m_list"]),
            N=case["N"],
            K_dim=case["K_dim"],
            L=len(case["group_m_list"]),
        ):
            raise AssertionError(f"invalid benchmark shape: {case['label']}")
    return cases


def default_bench_labels():
    """The three headline rows the bench-suite runs by default."""
    labels = []
    for shape in _PERF_SHAPES:
        labels.append(make_case(HEADLINE_MODE, prefix="perf", index=0, **shape)["label"])
    return labels


def upstream_launch_is_flaky(mode):
    """No known unreliable upstream corner for the bf16 kernel.

    The blockscaled sibling skips its upstream comparison on one corner whose
    trigger involves ``discrete_col_sfd``, which this kernel does not have. The
    hook stays so the correctness gate can pin one down if bring-up finds it.
    """
    del mode
    return False


# ---------------------------------------------------------------------------
# Derived launch, stage and storage facts
# ---------------------------------------------------------------------------


def derive(config):
    """Closed form of the upstream ``_setup_attributes`` and ``_compute_stages``.

    Validated element-for-element against the upstream class over the whole
    configuration matrix during the correctness gate.
    """
    tile_m, tile_n = config["mma_tiler_mn"]
    cluster_m, cluster_n = config["cluster_shape_mn"]
    thr = atom_thr(config)
    group_m_list = tuple(config["group_m_list"])
    L = len(group_m_list)
    N = config["N"]
    K_dim = config["K_dim"]

    padded = []
    running = 0
    for rows in group_m_list:
        running += align_up(rows, FIX_PAD_SIZE)
        padded.append(running)
    tokens_total = running

    cta_tile_m = tile_m // thr
    c_bits = dtype_bits(config["c_dtype"])
    d_bits = dtype_bits(config["d_dtype"])

    num_acc_stage = 2
    num_c_stage = 2
    num_d_stage = 2
    num_tile_stage = 2

    epi_m, epi_n = EPI_TILE
    c_bytes = epi_m * epi_n * num_c_stage * c_bits // 8
    d_bytes = epi_m * epi_n * num_d_stage * d_bits // 8
    dbias_bytes = 128 * epi_n * 2 * 4 if config["with_dbias"] else 4

    a_stage_bytes = cta_tile_m * K_TILE * 2
    b_stage_bytes = tile_n * K_TILE * 2
    ab_stage_bytes = a_stage_bytes + b_stage_bytes

    mbar_helpers_bytes = 1024
    sinfo_bytes = 4 * 4 * num_tile_stage
    reserved = mbar_helpers_bytes + sinfo_bytes + c_bytes + d_bytes + dbias_bytes
    num_ab_stage = (SMEM_CAPACITY - reserved) // ab_stage_bytes
    if num_ab_stage < 1:
        raise AssertionError(f"no shared-memory budget for one AB stage: {config['label']}")

    cluster_size = cluster_m * cluster_n
    grid = (cluster_m, cluster_n, MAX_ACTIVE_CLUSTERS[cluster_size])

    needs_helper = config["weight_mode"] == "discrete" or config["sched"] == "dynamic"
    helper_grid = (L if config["weight_mode"] == "discrete" else 1, 1, 1)
    workspace_bytes = (128 * L if config["weight_mode"] == "discrete" else 0) + (
        4 if config["sched"] == "dynamic" else 0
    )

    return {
        "L": L,
        "N": N,
        "K": K_dim,
        "tokens_total": tokens_total,
        "padded_offsets": tuple(padded),
        "use_2cta": thr == 2,
        "atom_thr": thr,
        "cta_tile_shape_mnk": (cta_tile_m, tile_n, K_TILE),
        "k_tile": K_TILE,
        "k_tile_count": ceil_div(K_dim, K_TILE),
        "epi_tile": EPI_TILE,
        "num_acc_stage": num_acc_stage,
        "num_ab_stage": num_ab_stage,
        "num_c_stage": num_c_stage,
        "num_d_stage": num_d_stage,
        "num_tile_stage": num_tile_stage,
        "grid": grid,
        "threads_per_cta": 256,
        "helper_grid": helper_grid,
        "needs_helper": needs_helper,
        "workspace_bytes": workspace_bytes,
        "smem": {
            "c_bytes": c_bytes,
            "d_bytes": d_bytes,
            "dbias_bytes": dbias_bytes,
            "a_stage_bytes": a_stage_bytes,
            "b_stage_bytes": b_stage_bytes,
        },
    }
