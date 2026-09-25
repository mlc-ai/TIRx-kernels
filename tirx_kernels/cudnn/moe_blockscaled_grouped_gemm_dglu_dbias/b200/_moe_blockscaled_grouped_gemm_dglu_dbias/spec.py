# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Static specialization domain for the MoE block-scaled grouped GEMM with dGLU backward.

Upstream sources:
``python/cudnn/gemm/cutedsl/grouped/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py``
(kernel, ``can_implement``, ``_setup_attributes``, ``_compute_stages``),
``python/cudnn/gemm/cutedsl/grouped/moe_kernel_helpers.py`` (validity predicates),
``python/cudnn/gemm/cutedsl/grouped/dglu/_blockscaled_api.py`` (``check_support``
dispatch domain).
"""

import json
from itertools import combinations, product

# Fixed row padding the upstream kernel requires of every expert range
# (``BlockScaledMoEGroupedGemmDgluDbiasKernel.FIX_PAD_SIZE``).
FIX_PAD_SIZE = 256

# ``utils.get_smem_capacity_in_bytes("sm_100")``.
SMEM_CAPACITY = 232_448
# SM100 tensor-memory columns, hardcoded by the upstream kernel.
TMEM_CAPACITY_COLUMNS = 512
# Epilogue subtile the upstream kernel pins in ``_setup_attributes``.
EPI_TILE = (128, 32)
# ``cluster_size -> max_active_clusters`` on the target part, used in place of a
# runtime ``HardwareInfo`` query so the launch grid stays a static specialization.
MAX_ACTIVE_CLUSTERS = {1: 148, 2: 74, 4: 33, 8: 15, 16: 7}

AB_DTYPES = ("float4_e2m1fn", "float8_e4m3fn", "float8_e5m2")
SF_MODES = (("float8_e8m0fnu", 16), ("float8_e8m0fnu", 32), ("float8_e4m3fn", 16))
C_DTYPES = ("bfloat16", "float16", "float32", "float8_e4m3fn", "float8_e5m2")
D_DTYPES = ("bfloat16", "float16", "float32", "float8_e4m3fn", "float8_e5m2")
ACT_FUNCS = ("dswiglu", "dgeglu", "dsituglu")
WEIGHT_MODES = ("dense", "discrete")
SCHED_MODES = ("static", "dynamic")

# Activation scalar knobs are compile-time constants on the upstream ``__call__``.
DEFAULT_ACT_KNOBS = {
    "linear_offset": 1.0,
    "geglu_alpha": 1.702,
    "glu_clamp_max": 7.0,
    "glu_clamp_min": -7.0,
    "situ_beta1": 4.0,
    "situ_beta2": 25.0,
}

MODE_KEYS = (
    "weight_mode",
    "sched",
    "act",
    "ab_dtype",
    "sf_dtype",
    "sf_vec_size",
    "c_dtype",
    "d_dtype",
    "b_major",
    "mma_tiler_mn",
    "cluster_shape_mn",
    "vectorized_f32",
    "with_dbias",
    "with_prob",
    "with_amax",
    "discrete_col_sfd",
)

_DTYPE_BITS = {
    "float4_e2m1fn": 4,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e8m0fnu": 8,
    "float16": 16,
    "bfloat16": 16,
    "float32": 32,
}

_SHORT = {
    "float4_e2m1fn": "f4",
    "float8_e4m3fn": "e4",
    "float8_e5m2": "e5",
    "float8_e8m0fnu": "e8",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "dswiglu": "sw",
    "dgeglu": "ge",
    "dsituglu": "si",
    "dense": "dn",
    "discrete": "dc",
    "static": "st",
    "dynamic": "dy",
}


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def align_up(value, alignment):
    return ceil_div(value, alignment) * alignment


def dtype_bits(dtype):
    return _DTYPE_BITS[dtype]


def is_fp8(dtype):
    return dtype in ("float8_e4m3fn", "float8_e5m2")


def is_fp4(dtype):
    return dtype == "float4_e2m1fn"


def generates_sfd(mode):
    """Upstream ``generate_sfd``: FP8 A/B with E8M0 scale factors and FP8 output."""
    return (
        is_fp8(mode["ab_dtype"])
        and mode["sf_dtype"] == "float8_e8m0fnu"
        and is_fp8(mode["d_dtype"])
    )


def valid_mode(mode):
    """The dispatch domain accepted by the upstream API's ``check_support``."""
    ab_dtype = mode["ab_dtype"]
    sf_dtype = mode["sf_dtype"]
    sf_vec_size = mode["sf_vec_size"]
    d_dtype = mode["d_dtype"]
    c_dtype = mode["c_dtype"]

    if sf_vec_size not in (16, 32):
        return False
    if sf_dtype == "float8_e4m3fn" and sf_vec_size == 32:
        return False
    if is_fp8(ab_dtype) and sf_vec_size == 16:
        return False

    if is_fp4(ab_dtype):
        if mode["b_major"] != "k":
            return False
        if d_dtype not in ("float16", "bfloat16", "float32"):
            return False
    elif is_fp8(ab_dtype):
        if not is_fp8(d_dtype):
            return False
        # ``cvt_f32x4_to_f8x4_pack_i32`` (source 1232-1249) selects its inline
        # PTX from the output dtype and covers only UE8M0 and E4M3; every other
        # type falls into the ``else`` at 1246-1249, which prints "unsupported
        # fp8 element type" and returns ``None``. Both epilogue call sites
        # (1389, 1539) pass ``self.d_dtype``, so an E5M2 output does not compile
        # upstream -- this excludes it from the source's real dispatch domain,
        # not merely from this port's.
        if d_dtype == "float8_e5m2":
            return False
    else:
        return False

    # Rejected outright by the upstream API validator: ``_blockscaled_api.py``
    # 577-578 raises "fp8 c_dtype and vector_f32 is not supported".
    if is_fp8(c_dtype) and mode["vectorized_f32"]:
        return False

    # discrete_col_sfd is only read when the kernel emits scale factors.
    if mode["discrete_col_sfd"] and not generates_sfd(mode):
        return False

    tile_m, tile_n = mode["mma_tiler_mn"]
    if tile_n != 256:
        return False
    if tile_m not in (128, 256):
        return False
    use_2cta = tile_m == 256
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    if cluster_m <= 0 or cluster_n <= 0 or cluster_m > 4 or cluster_n > 4:
        return False
    if cluster_m & (cluster_m - 1) or cluster_n & (cluster_n - 1):
        return False
    if cluster_m * cluster_n > 16:
        return False
    if cluster_m % (2 if use_2cta else 1) != 0:
        return False
    if (cluster_m // (2 if use_2cta else 1)) * tile_m not in (128, 256):
        return False
    if FIX_PAD_SIZE % tile_m != 0:
        return False
    return True


def upstream_launch_is_flaky(mode):
    """Does the upstream kernel fault intermittently on this specialization?

    One corner of the upstream dispatch domain -- dense weights, the dynamic
    scheduler, FP32 ``C`` and ``discrete_col_sfd`` -- fails intermittently with
    ``cudaErrorLaunchFailure``. Across fresh processes it has passed roughly two
    runs in seven; ``compute-sanitizer --tool memcheck`` reports zero errors over
    three launches and never reproduces it, which fits a race and rules out an
    out-of-bounds access. Resources are not the cause (about 220,160 of 232,448
    shared bytes). Neighbouring dense + dynamic configurations, including the
    same shape without ``discrete_col_sfd``, run clean over ten launches.

    The fault is sticky: once it fires it invalidates the CUDA context, so every
    later launch in that process fails too and a retry inside the same process is
    worthless. ``run_test`` therefore does not launch the upstream kernel here at
    all -- it validates TIRx against the FP32 oracle, which is independent of the
    source and constrains every output this specialization writes.

    Re-confirmed after this port's own memory-safety defects were fixed, since one
    of them wrote past the end of a scale-factor buffer and could have been what
    destabilized a shared process: with TIRx never launched at all, the upstream
    kernel still faults three runs out of three.
    """
    return (
        mode["weight_mode"] == "dense"
        and mode["sched"] == "dynamic"
        and mode["c_dtype"] == "float32"
        and mode["discrete_col_sfd"]
    )


def valid_shape(mode, *, tokens_total, N, K_dim, L):
    """Upstream ``is_valid_tensor_alignment`` plus the dGLU N constraint."""
    if N % 32 != 0:
        return False
    if L <= 0 or tokens_total <= 0 or K_dim <= 0:
        return False
    ab_contiguous = 16 * 8 // dtype_bits(mode["ab_dtype"])
    d_contiguous = 16 * 8 // dtype_bits(mode["d_dtype"])
    # A is k-major, so K is the contiguous extent.
    if K_dim % ab_contiguous != 0:
        return False
    # B is k-major or n-major depending on the mode.
    if (K_dim if mode["b_major"] == "k" else N) % ab_contiguous != 0:
        return False
    # C/D are n-major over the interleaved 2N output.
    if (2 * N) % d_contiguous != 0:
        return False
    return True


def mode_label(mode):
    tile_m, _ = mode["mma_tiler_mn"]
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    flags = "".join(
        (
            "b" if mode["with_dbias"] else "",
            "p" if mode["with_prob"] else "",
            "a" if mode["with_amax"] else "",
            "v" if mode["vectorized_f32"] else "",
            "s" if mode["discrete_col_sfd"] else "",
        )
    )
    return (
        f"{_SHORT[mode['weight_mode']]}{_SHORT[mode['sched']]}_{_SHORT[mode['act']]}_"
        f"{_SHORT[mode['ab_dtype']]}_{_SHORT[mode['sf_dtype']]}v{mode['sf_vec_size']}_"
        f"c{_SHORT[mode['c_dtype']]}_d{_SHORT[mode['d_dtype']]}_b{mode['b_major']}_"
        f"t{tile_m}x256_c{cluster_m}x{cluster_n}_{flags or 'none'}"
    )


# The dispatch domain factors into independent groups: constraints couple the
# dtype/layout keys and the tile/cluster keys, and leave the remaining keys free.
# Every combination of one value per group is a valid mode, so the covering array
# below never has to repair a constrained row.
_COUPLED_DTYPE_KEYS = (
    "ab_dtype",
    "sf_dtype",
    "sf_vec_size",
    "c_dtype",
    "d_dtype",
    "b_major",
    "vectorized_f32",
    "discrete_col_sfd",
)
_COUPLED_TILE_KEYS = ("mma_tiler_mn", "cluster_shape_mn")
_FREE_KEYS = ("weight_mode", "sched", "act", "with_dbias", "with_prob", "with_amax")


def _dtype_group_values():
    values = []
    for ab_dtype in AB_DTYPES:
        b_majors = ("k",) if is_fp4(ab_dtype) else ("k", "n")
        d_dtypes = ("float16", "bfloat16", "float32") if is_fp4(ab_dtype) else ("float8_e4m3fn",)
        for (
            sf_dtype,
            sf_vec_size,
        ), c_dtype, d_dtype, b_major, vectorized_f32, discrete_col_sfd in product(
            SF_MODES, C_DTYPES, d_dtypes, b_majors, (False, True), (False, True)
        ):
            value = {
                "ab_dtype": ab_dtype,
                "sf_dtype": sf_dtype,
                "sf_vec_size": sf_vec_size,
                "c_dtype": c_dtype,
                "d_dtype": d_dtype,
                "b_major": b_major,
                "vectorized_f32": vectorized_f32,
                "discrete_col_sfd": discrete_col_sfd,
            }
            if _valid_dtype_group(value):
                values.append(value)
    return values


def _valid_dtype_group(value):
    sf_dtype, sf_vec_size = value["sf_dtype"], value["sf_vec_size"]
    if sf_dtype == "float8_e4m3fn" and sf_vec_size == 32:
        return False
    if is_fp8(value["ab_dtype"]) and sf_vec_size == 16:
        return False
    if is_fp8(value["c_dtype"]) and value["vectorized_f32"]:
        return False
    if value["discrete_col_sfd"] and not generates_sfd(value):
        return False
    return True


def _tile_group_values():
    values = []
    for tile_m, cluster_m, cluster_n in product((128, 256), (1, 2, 4), (1, 2, 4)):
        use_2cta = tile_m == 256
        if cluster_m % (2 if use_2cta else 1) != 0:
            continue
        if (cluster_m // (2 if use_2cta else 1)) * tile_m not in (128, 256):
            continue
        if cluster_m * cluster_n > 16:
            continue
        values.append({"mma_tiler_mn": (tile_m, 256), "cluster_shape_mn": (cluster_m, cluster_n)})
    return values


def _groups():
    groups = [_dtype_group_values(), _tile_group_values()]
    groups.extend(
        [{key: value} for value in values]
        for key, values in (
            ("weight_mode", WEIGHT_MODES),
            ("sched", SCHED_MODES),
            ("act", ACT_FUNCS),
            ("with_dbias", (False, True)),
            ("with_prob", (False, True)),
            ("with_amax", (False, True)),
        )
    )
    return groups


def structural_modes():
    """Every point of the upstream dispatch domain, as mode dictionaries."""
    modes = []
    for parts in product(*_groups()):
        mode = {}
        for part in parts:
            mode.update(part)
        modes.append(mode)
    return sorted(modes, key=mode_label)


def _token(key, value):
    return key, json.dumps(value)


def _pair(left, right):
    return (left, right) if left <= right else (right, left)


def _pair_universe(groups):
    """Every key-pair value combination reachable in the domain."""
    universe = set()
    for index, values in enumerate(groups):
        keys = tuple(values[0])
        # Pairs inside one group only exist where the group's values pair them.
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
    """Greedy pairwise covering array over the domain's independent groups.

    Each row is seeded with one still-uncovered pair so the construction always
    makes progress, then filled group by group by maximum newly-covered pairs.
    """
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


def _base_shape(mode):
    """Smallest correctness shape that satisfies the mode's alignment rules."""
    group_m_list = (256,) * 4
    N, K_dim = 512, 512
    return {"group_m_list": group_m_list, "N": N, "K_dim": K_dim}


def make_case(mode, *, group_m_list, N, K_dim, prefix, index=None, act_knobs=None):
    L = len(group_m_list)
    tokens_total = sum(align_up(rows, FIX_PAD_SIZE) for rows in group_m_list)
    if not valid_shape(mode, tokens_total=tokens_total, N=N, K_dim=K_dim, L=L):
        raise ValueError(f"invalid shape for mode {mode_label(mode)}")
    tag = prefix if index is None else f"{prefix}{index:02d}"
    knobs = dict(DEFAULT_ACT_KNOBS)
    if act_knobs:
        knobs.update(act_knobs)
    knob_tag = "" if not act_knobs else "_knob"
    case = {
        "label": f"{tag}_e{L}_t{tokens_total}_n{N}_k{K_dim}_{mode_label(mode)}{knob_tag}",
        "group_m_list": list(group_m_list),
        "N": N,
        "K": K_dim,
        **mode,
        **knobs,
    }
    case["mma_tiler_mn"] = list(mode["mma_tiler_mn"])
    case["cluster_shape_mn"] = list(mode["cluster_shape_mn"])
    return case


def _first_mode(predicate):
    for mode in structural_modes():
        if predicate(mode):
            return mode
    raise AssertionError("no structural mode satisfies the predicate")


def correctness_configs():
    """Pairwise cover of the dispatch domain plus explicit boundary cases."""
    configs = [
        make_case(mode, prefix="pair", **_base_shape(mode)) for mode in minimal_pairwise_modes()
    ]

    def add(mode, *, group_m_list=(256,) * 4, N=512, K_dim=512, act_knobs=None):
        configs.append(
            make_case(
                mode,
                group_m_list=group_m_list,
                N=N,
                K_dim=K_dim,
                prefix="tail",
                act_knobs=act_knobs,
            )
        )

    fp4 = _first_mode(
        lambda mode: (
            mode["ab_dtype"] == "float4_e2m1fn"
            and mode["weight_mode"] == "dense"
            and mode["sched"] == "static"
            and mode["act"] == "dswiglu"
            and mode["sf_vec_size"] == 16
            and mode["sf_dtype"] == "float8_e8m0fnu"
            and mode["c_dtype"] == "bfloat16"
            and mode["d_dtype"] == "bfloat16"
            and mode["mma_tiler_mn"] == (128, 256)
            and mode["cluster_shape_mn"] == (1, 1)
            and not mode["vectorized_f32"]
            and mode["with_dbias"]
            and mode["with_prob"]
            and mode["with_amax"]
        )
    )
    fp8 = _first_mode(
        lambda mode: (
            mode["ab_dtype"] == "float8_e4m3fn"
            and mode["weight_mode"] == "dense"
            and mode["sched"] == "static"
            and mode["act"] == "dswiglu"
            and mode["c_dtype"] == "bfloat16"
            and mode["d_dtype"] == "float8_e4m3fn"
            and mode["b_major"] == "k"
            and mode["mma_tiler_mn"] == (256, 256)
            and mode["cluster_shape_mn"] == (2, 1)
            and mode["vectorized_f32"]
            and mode["with_dbias"]
            and mode["with_prob"]
            and not mode["with_amax"]
            and not mode["discrete_col_sfd"]
        )
    )

    # Ragged expert ranges: unaligned token counts, and an expert with no tokens.
    add(fp4, group_m_list=(96, 320, 128, 256))
    add(fp8, group_m_list=(256, 0, 512, 256))
    # Multi-k-tile plus a non-256-multiple N.
    add(fp4, N=320, K_dim=2048)
    # Expert count beyond one scheduler cache line.
    add(fp8, group_m_list=(256,) * 8)
    # Discrete weights combined with the dynamic scheduler and ragged ranges.
    add({**fp8, "weight_mode": "discrete", "sched": "dynamic"}, group_m_list=(96, 320, 128, 256))
    # Column scale factors written through the discrete path.
    add({**fp8, "discrete_col_sfd": True})
    # Activation knobs away from their defaults.
    add(
        {**fp4, "act": "dgeglu"},
        act_knobs={
            "linear_offset": 0.0,
            "geglu_alpha": 1.0,
            "glu_clamp_max": 5.0,
            "glu_clamp_min": -5.0,
        },
    )
    # SiTU-GLU off the beta1 == 4.0 packed fast path.
    add({**fp4, "act": "dsituglu"}, act_knobs={"situ_beta1": 2.0, "situ_beta2": 16.0})
    # No probability weighting at all.
    add({**fp4, "with_prob": False})
    # Single expert, offsets read at runtime.
    add(fp4, group_m_list=(256,))

    seen = set()
    unique = []
    for config in sorted(configs, key=lambda config: config["label"]):
        if config["label"] in seen:
            continue
        seen.add(config["label"])
        unique.append(config)
    return unique


def _performance_tokens(mode):
    """Structural knobs that change the emitted performance path."""
    ab_bits = dtype_bits(mode["ab_dtype"])
    c_bits = dtype_bits(mode["c_dtype"])
    d_bits = dtype_bits(mode["d_dtype"])
    tile_m, tile_n = mode["mma_tiler_mn"]
    return frozenset(
        {
            ("ab", mode["ab_dtype"]),
            ("sf", mode["sf_dtype"], mode["sf_vec_size"]),
            ("mma", "fp4" if ab_bits == 4 else "fp8", mode["sf_vec_size"], tile_m),
            ("c", mode["c_dtype"]),
            ("d", mode["d_dtype"]),
            ("b_tma", mode["ab_dtype"], mode["b_major"]),
            ("cluster", mode["cluster_shape_mn"], tile_m),
            ("weight", mode["weight_mode"]),
            ("sched", mode["sched"]),
            ("act", mode["act"], mode["vectorized_f32"]),
            ("dbias", mode["with_dbias"]),
            ("prob", mode["with_prob"]),
            ("amax", mode["with_amax"]),
            ("sfd", generates_sfd(mode), mode["discrete_col_sfd"]),
            ("c_stages", 4 if ab_bits == 8 else 2, c_bits),
            ("d_stages", d_bits, generates_sfd(mode)),
        }
    )


def _complete(groups, assignment):
    """Fill unassigned groups with their first value, yielding a full mode."""
    mode = {}
    for index, values in enumerate(groups):
        mode.update(assignment.get(index, values[0]))
    return mode


def _token_universe(groups, token_fn):
    """Every token reachable in the domain.

    No token in ``_performance_tokens`` reads more than two groups, so pinning
    every other group to its first value while sweeping one or two groups at a
    time reaches all of them without enumerating the full product.
    """
    universe = set()
    for index, values in enumerate(groups):
        for value in values:
            universe |= token_fn(_complete(groups, {index: value}))
    for index, values in enumerate(groups):
        for other in range(index + 1, len(groups)):
            for value, other_value in product(values, groups[other]):
                universe |= token_fn(_complete(groups, {index: value, other: other_value}))
    return universe


def _greedy_token_rows(groups, token_fn):
    """Greedy cover of ``token_fn``'s universe, one group choice at a time."""
    uncovered = _token_universe(groups, token_fn)
    order = sorted(range(len(groups)), key=lambda index: -len(groups[index]))
    rows = []
    while uncovered:
        assignment = {}
        for index in order:
            best = max(
                groups[index],
                key=lambda value: (
                    len(token_fn(_complete(groups, {**assignment, index: value})) & uncovered),
                    tuple(sorted(map(str, value.items()))),
                ),
            )
            assignment[index] = best
        mode = _complete(groups, assignment)
        covered = token_fn(mode) & uncovered
        if not covered:
            raise AssertionError("performance token coverage did not converge")
        uncovered -= covered
        rows.append(mode)
    return rows


def performance_representatives():
    return sorted(_greedy_token_rows(_groups(), _performance_tokens), key=mode_label)


# The mode the upstream module documents as its headline invocation:
# ``--ab_dtype Float8E4M3FN --sf_dtype Float8E8M0FNU --c_dtype BFloat16
#   --d_dtype Float8E4M3FN --sf_vec_size 32 --mma_tiler_mn 256,256
#   --cluster_shape_mn 2,1 --nkl 4096,7168,8 --use_2cta_instrs --m_aligned 256``.
HEADLINE_MODE = {
    "weight_mode": "dense",
    "sched": "static",
    "act": "dswiglu",
    "ab_dtype": "float8_e4m3fn",
    "sf_dtype": "float8_e8m0fnu",
    "sf_vec_size": 32,
    "c_dtype": "bfloat16",
    "d_dtype": "float8_e4m3fn",
    "b_major": "k",
    "mma_tiler_mn": (256, 256),
    "cluster_shape_mn": (2, 1),
    "vectorized_f32": True,
    "with_dbias": True,
    "with_prob": True,
    "with_amax": False,
    "discrete_col_sfd": False,
}

# Realistic MoE benchmark shapes, sized from the upstream fusion benchmark and the
# module docstring's example invocation (8 experts, 4096 tokens per expert,
# K 7168-8192, N 4096).
_PERF_SHAPES = (
    ("small", 4, 1024, 2048, 2048),
    ("medium", 8, 2048, 4096, 8192),
    ("large", 8, 4096, 4096, 7168),
)

# Headline sweep across expert count, tokens per expert, and K.
_PERF_SWEEP = tuple(
    (experts, tokens, 4096, K_dim)
    for experts in (4, 8)
    for tokens in (512, 1024, 2048, 4096)
    for K_dim in (2048, 8192)
)


def _perf_shape_ok(mode, experts, tokens, N, K_dim):
    tokens_total = experts * align_up(tokens, FIX_PAD_SIZE)
    return valid_shape(mode, tokens_total=tokens_total, N=N, K_dim=K_dim, L=experts)


def benchmark_configs():
    """Deterministic covering matrix; the perf gate requires every row above 0.99x."""
    representatives = [HEADLINE_MODE] + [
        mode
        for mode in performance_representatives()
        if mode_label(mode) != mode_label(HEADLINE_MODE)
    ]
    configs = []
    for index, mode in enumerate(representatives):
        for _role, experts, tokens, N, K_dim in _PERF_SHAPES:
            if not _perf_shape_ok(mode, experts, tokens, N, K_dim):
                continue
            configs.append(
                make_case(
                    mode,
                    group_m_list=(tokens,) * experts,
                    N=N,
                    K_dim=K_dim,
                    prefix="perf",
                    index=index,
                )
            )
    for experts, tokens, N, K_dim in _PERF_SWEEP:
        if not _perf_shape_ok(HEADLINE_MODE, experts, tokens, N, K_dim):
            continue
        configs.append(
            make_case(
                HEADLINE_MODE,
                group_m_list=(tokens,) * experts,
                N=N,
                K_dim=K_dim,
                prefix="sweep",
                index=0,
            )
        )
    seen = set()
    unique = []
    for config in sorted(configs, key=lambda config: config["label"]):
        if config["label"] in seen:
            continue
        seen.add(config["label"])
        unique.append(config)
    return unique


def default_bench_labels():
    """The three bench-suite default rows, tagged small/medium/large."""
    labels = {}
    for role, experts, tokens, N, K_dim in _PERF_SHAPES:
        if not _perf_shape_ok(HEADLINE_MODE, experts, tokens, N, K_dim):
            continue
        case = make_case(
            HEADLINE_MODE,
            group_m_list=(tokens,) * experts,
            N=N,
            K_dim=K_dim,
            prefix="perf",
            index=0,
        )
        labels[role] = case["label"]
    return labels


# ---------------------------------------------------------------------------
# Derived static attributes
# ---------------------------------------------------------------------------
#
# Closed form of the upstream ``_setup_attributes`` / ``_compute_stages`` results,
# validated element-for-element against the source class over the full CONFIGS and
# BENCH_CONFIGS matrices.


def derive(mode, *, group_m_list, N, K_dim):
    """Static launch, staging, and storage facts for one specialization."""
    L = len(group_m_list)
    tokens_total = sum(align_up(rows, FIX_PAD_SIZE) for rows in group_m_list)
    if not valid_mode(mode):
        raise ValueError(f"unsupported source specialization: {mode_label(mode)}")
    if not valid_shape(mode, tokens_total=tokens_total, N=N, K_dim=K_dim, L=L):
        raise ValueError(f"unsupported shape for {mode_label(mode)}")

    ab_bits = dtype_bits(mode["ab_dtype"])
    c_bits = dtype_bits(mode["c_dtype"])
    d_bits = dtype_bits(mode["d_dtype"])
    sf_vec_size = mode["sf_vec_size"]
    tile_m, tile_n = mode["mma_tiler_mn"]
    use_2cta = tile_m == 256
    atom_thr = 2 if use_2cta else 1
    cluster_m, cluster_n = mode["cluster_shape_mn"]
    cluster_size = cluster_m * cluster_n

    # MMA instruction K times the fixed 4-instruction K tiling.
    k_tile = 256 if ab_bits == 4 else 128
    cta_tile_m = tile_m // atom_thr
    cta_tile_n = tile_n
    # SFB rides a separate cta_group::1 MMA whose M is the CTA tile.
    cta_tile_m_sfb = (tile_m // atom_thr) // atom_thr
    epi_m, epi_n = EPI_TILE
    epi_tile_cnt = (cta_tile_m // epi_m, cta_tile_n // epi_n)

    num_acc_stage = 1 if tile_n == 256 else 2
    num_c_stage = 4 if ab_bits == 8 else 2
    num_d_stage = 2
    num_tile_stage = 2
    overlapping_accum = num_acc_stage == 1 and tile_n == 256

    a_stage_bytes = cta_tile_m * k_tile * ab_bits // 8
    b_stage_bytes = (cta_tile_n // atom_thr) * k_tile * ab_bits // 8
    sfa_stage_bytes = cta_tile_m * k_tile // sf_vec_size
    sfb_stage_bytes = cta_tile_n * k_tile // sf_vec_size
    ab_stage_bytes = a_stage_bytes + b_stage_bytes + sfa_stage_bytes + sfb_stage_bytes

    c_stage_bytes = epi_m * epi_n * c_bits // 8
    d_stage_bytes = epi_m * epi_n * d_bits // 8
    c_bytes = c_stage_bytes * num_c_stage
    d_bytes = d_stage_bytes * num_d_stage * (2 if d_bits == 8 else 1)
    amax_bytes = 16 if mode["d_dtype"] == "bfloat16" else 0
    dbias_bytes = 128 * 2 * epi_n * 4 if mode["with_dbias"] else 0
    sinfo_bytes = 4 * 4 * num_tile_stage
    mbar_helpers_bytes = 1024
    epi_bytes = c_bytes + d_bytes + amax_bytes + dbias_bytes
    num_ab_stage = (
        SMEM_CAPACITY - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)
    ) // ab_stage_bytes
    if num_ab_stage <= 0:
        raise ValueError("the source stage heuristic exceeds the SM100 shared-memory capacity")

    sf_atom_mn = 32
    num_sfa_tmem_cols = (cta_tile_m // sf_atom_mn) * 4
    num_sfb_tmem_cols = (align_up(cta_tile_n, 128) // sf_atom_mn) * 4
    num_sf_tmem_cols = num_sfa_tmem_cols + num_sfb_tmem_cols
    num_accumulator_tmem_cols = (
        cta_tile_n * 2 - num_sf_tmem_cols if overlapping_accum else cta_tile_n * num_acc_stage
    )
    iter_acc_early_release = ceil_div(num_sf_tmem_cols, epi_n) - 1

    generate_sfd = generates_sfd(mode)
    # Shared-memory byte map, in the upstream ``SharedStorage`` declaration order
    # with its alignments. Verified against the anchor export's mbarrier and TMA
    # operands: sA at 50176, sB at 115712, sSFA at 181248, sSFB at 183296.
    offsets = {}
    cursor = 0

    def place(name, byte_count, alignment=8):
        nonlocal cursor
        cursor = align_up(cursor, alignment)
        offsets[name] = cursor
        cursor += byte_count
        return offsets[name]

    place("ab_full", num_ab_stage * 8)
    place("ab_empty", num_ab_stage * 8)
    place("acc_full", num_acc_stage * 8)
    place("acc_empty", num_acc_stage * 8)
    place("tile_full", num_tile_stage * 8)
    place("tile_empty", num_tile_stage * 8)
    place("sinfo", 4 * 4 * num_tile_stage, 16)
    if mode["sched"] == "dynamic":
        place("cluster_mbar", 2 * 8)
        place("cluster_broadcast", 4 * 4, 16)
    place("c_full", num_c_stage * 8)
    place("c_empty", num_c_stage * 8)
    place("tmem_dealloc", 8)
    place("tmem_slot", 4, 4)
    protocol_bytes = cursor
    place("sC", num_c_stage * c_stage_bytes, 1024)
    place("sD", num_d_stage * d_stage_bytes, 1024)
    place("sD_col", num_d_stage * d_stage_bytes if generate_sfd else 0, 1024)
    place("sA", num_ab_stage * a_stage_bytes, 1024)
    place("sB", num_ab_stage * b_stage_bytes, 1024)
    place("sSFA", num_ab_stage * sfa_stage_bytes, 1024)
    place("sSFB", num_ab_stage * sfb_stage_bytes, 1024)
    place("sAmax", 4 * 4, 4)
    place("sDbias", dbias_bytes if dbias_bytes else 4, 128)
    shared_bytes = align_up(cursor, 1024)
    if shared_bytes > SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {shared_bytes} exceeds {SMEM_CAPACITY}")

    n_out = 2 * N
    workspace_bytes = (2 * L * 128 if mode["weight_mode"] == "discrete" else 0) + (
        4 if mode["sched"] == "dynamic" else 0
    )

    return {
        "L": L,
        "tokens_total": tokens_total,
        "group_m_list": tuple(group_m_list),
        "N": N,
        "n_out": n_out,
        "K": K_dim,
        "k_tiles": ceil_div(K_dim, k_tile),
        "k_tile": k_tile,
        "mma_tiler": (tile_m, tile_n, k_tile),
        "cta_tile_shape_mnk": (cta_tile_m, cta_tile_n, k_tile),
        "cta_tile_m_sfb": cta_tile_m_sfb,
        "use_2cta": use_2cta,
        "atom_thr": atom_thr,
        "epi_tile": EPI_TILE,
        "epi_tile_cnt": epi_tile_cnt,
        "num_acc_stage": num_acc_stage,
        "num_ab_stage": num_ab_stage,
        "num_c_stage": num_c_stage,
        "num_d_stage": num_d_stage,
        "num_tile_stage": num_tile_stage,
        "overlapping_accum": overlapping_accum,
        "a_stage_bytes": a_stage_bytes,
        "b_stage_bytes": b_stage_bytes,
        "sfa_stage_bytes": sfa_stage_bytes,
        "sfb_stage_bytes": sfb_stage_bytes,
        "ab_stage_bytes": ab_stage_bytes,
        "c_stage_bytes": c_stage_bytes,
        "d_stage_bytes": d_stage_bytes,
        "dbias_bytes": dbias_bytes,
        "num_sfa_tmem_cols": num_sfa_tmem_cols,
        "num_sfb_tmem_cols": num_sfb_tmem_cols,
        "num_sf_tmem_cols": num_sf_tmem_cols,
        "num_accumulator_tmem_cols": num_accumulator_tmem_cols,
        "num_tmem_alloc_cols": TMEM_CAPACITY_COLUMNS,
        "iter_acc_early_release": iter_acc_early_release,
        "grid": (cluster_m, cluster_n, MAX_ACTIVE_CLUSTERS[cluster_size]),
        "cluster_size": cluster_size,
        "threads_per_cta": 256,
        "generate_sfd": generate_sfd,
        "generate_amax": mode["with_amax"],
        "generate_dbias": mode["with_dbias"],
        "generate_dprob": mode["with_prob"],
        "smem_offsets": offsets,
        "protocol_bytes": protocol_bytes,
        "shared_bytes": shared_bytes,
        "workspace_bytes": workspace_bytes,
        "needs_helper": mode["weight_mode"] == "discrete" or mode["sched"] == "dynamic",
        "helper_grid": (L if mode["weight_mode"] == "discrete" else 1, 1, 1),
        "sf_shape_a": (1, ceil_div(tokens_total, 128), ceil_div(K_dim, 4 * sf_vec_size), 32, 4, 4),
        "sf_shape_b": (L, ceil_div(N, 128), ceil_div(K_dim, 4 * sf_vec_size), 32, 4, 4),
        "sf_shape_d_row": (
            1,
            ceil_div(tokens_total, 128),
            ceil_div(n_out, 4 * sf_vec_size),
            32,
            4,
            4,
        ),
        "sf_shape_d_col": (
            1,
            ceil_div(n_out, 128),
            ceil_div(tokens_total, 4 * sf_vec_size),
            32,
            4,
            4,
        ),
    }


def derive_for_config(config):
    mode = {key: config[key] for key in MODE_KEYS}
    mode["mma_tiler_mn"] = tuple(mode["mma_tiler_mn"])
    mode["cluster_shape_mn"] = tuple(mode["cluster_shape_mn"])
    return derive(mode, group_m_list=config["group_m_list"], N=config["N"], K_dim=config["K"])
