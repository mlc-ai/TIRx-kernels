# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025
# DeepSeek, licensed under the MIT License. The upstream sources carry no
# per-file license header; see licenses/LICENSE.deepgemm.txt for the full
# license text.
#
# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# Data preparation and reference oracles for the TIRx port of DeepGEMM's
# sm100_fp8_fp4_gemm_1d1d. The quantization recipes, scale-factor layout
# transform and accuracy metric mirror deep_gemm/utils/math.py,
# csrc/apis/layout.hpp and deep_gemm/testing/numeric.py.
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

"""Operand construction, scale-factor packing and correctness oracles.

Everything here runs on the host, outside any timed closure.  The scale-factor
layout is produced by DeepGEMM's own `transform_sf_into_required_layout`, so our
kernel and the reference consume a byte-identical tensor and the TMA descriptors
we build cannot drift from the layout the reference expects.

`deep_gemm` is imported lazily inside each function: kernel discovery must work
on hosts without it.
"""

from __future__ import annotations

from ._sm100_fp8_fp4_gemm_1d1d import Major, ceil_div, gran_k_for

__all__ = [
    "assert_within_threshold",
    "bench_against_deepgemm",
    "calc_diff",
    "dequantize",
    "quantize_a",
    "quantize_b",
    "require_sm100",
    "transform_sf",
]


def require_sm100() -> None:
    """Skip rather than fail when the host cannot run the kernel."""
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("no CUDA device")
    if torch.cuda.get_device_capability()[0] != 10:
        raise SkipTest(
            f"needs SM100, got sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )


def require_deep_gemm():
    """Return the `deep_gemm` module, or skip if it is not installed."""
    from unittest import SkipTest

    try:
        import deep_gemm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SkipTest(f"deep_gemm is not installed: {exc}") from exc
    return deep_gemm


def calc_diff(x, y) -> float:
    """DeepGEMM's accuracy metric: `1 - 2 * sum(xy) / sum(x^2 + y^2)`.

    Zero for identical tensors; DeepGEMM's own thresholds are 1e-3 for
    FP8 x FP8 and 1e-2 when one operand is FP4.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denominator)


def max_diff_threshold(a_dtype: str, b_dtype: str) -> float:
    """`QuantConfig.max_diff` from DeepGEMM's test generators."""
    if a_dtype == "fp4" and b_dtype == "fp4":
        return 0.02
    if a_dtype == "fp4" or b_dtype == "fp4":
        return 0.01
    return 0.001


def quantize_a(x, *, gran_k: int = 128, dtype: str = "fp8") -> tuple:
    """Per-token quantization of the A operand (`recipe_a = (1, gran_k)`)."""
    from deep_gemm.utils.math import per_token_cast_to_fp4, per_token_cast_to_fp8

    if dtype == "fp8":
        return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)
    if dtype == "fp4":
        return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=gran_k)
    raise ValueError(f"unsupported A dtype: {dtype}")


def quantize_b(x, *, gran_k: int = 128, dtype: str = "fp8", per_block: bool | None = None) -> tuple:
    """Quantize the B operand.

    DeepGEMM pairs FP8 B with a `(gran_k, gran_k)` block recipe and FP4 B with a
    per-token `(1, gran_k)` recipe (`QuantConfig.get_recipes`), so `per_block`
    defaults to "FP8 and not FP4".
    """
    from deep_gemm.utils.math import (
        per_block_cast_to_fp8,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    if per_block is None:
        per_block = dtype == "fp8"
    if dtype == "fp4":
        if per_block:
            raise ValueError("FP4 operands use a per-token recipe")
        return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=gran_k)
    if dtype != "fp8":
        raise ValueError(f"unsupported B dtype: {dtype}")
    if per_block:
        return per_block_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)
    return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)


def recipe_for(dtype: str, gran_k: int, *, is_a: bool, is_wgrad: bool = False) -> tuple[int, int]:
    """`(gran_mn, gran_k)` for one operand, matching `QuantConfig.get_recipes`."""
    if is_a:
        return (1, gran_k)
    if dtype == "fp4" or is_wgrad:
        return (1, gran_k)
    return (gran_k, gran_k)


def transform_sf(sf, mn: int, k: int, recipe: tuple[int, int], num_groups: int | None = None):
    """Pack scales into the MN-major, TMA-aligned, `ue8m0`-in-`int32` layout.

    Delegating to DeepGEMM guarantees our kernel and the reference read the exact
    same bytes; `csrc/apis/layout.hpp:49-53` is the SM100 branch that runs.
    """
    deep_gemm = require_deep_gemm()
    if num_groups is None:
        return deep_gemm.transform_sf_into_required_layout(sf, mn, k, recipe)
    return deep_gemm.transform_sf_into_required_layout(sf, mn, k, recipe, num_groups)


def broadcast_block_sf(sf, mn: int, gran_mn: int):
    """Expand a `(mn / gran_mn, k / gran_k)` block scale to one row per MN element.

    `cast_back_*` and DeepGEMM's own reference path both want a per-row scale;
    `csrc/apis/layout.hpp:51-52` performs the same broadcast on the device side.
    """
    if gran_mn == 1:
        return sf
    return sf.repeat_interleave(gran_mn, dim=0)[:mn]


def to_major(x, major: str):
    """Return the same logical `[MN, K]` tensor laid out K-major or MN-major.

    "K-major" means the K index is contiguous (`stride(-1) == 1`), which is what
    `get_major_type_ab` tests.  "MN-major" keeps the logical shape but stores the
    transpose, giving `stride == (1, MN)` -- a real relayout, not a view, because
    the TMA descriptor addresses storage.
    """
    if major == "k":
        return x
    if major != "mn":
        raise ValueError(f"unsupported major: {major}")
    if x.dtype.itemsize == 1 and x.dtype.is_signed and str(x.dtype) == "torch.int8":
        # Packed FP4 cannot be transposed elementwise; DeepGEMM keeps FP4 operands
        # K-major everywhere on SM100, so this combination should never be built.
        raise ValueError("packed FP4 operands are K-major only")
    return x.t().contiguous().t()


def dequantize(quantized, sf_raw, *, dtype: str, gran_k: int, gran_mn: int = 1):
    """Reconstruct the value a perfect GEMM would consume from this operand."""
    from deep_gemm.utils.math import cast_back_from_fp4, cast_back_from_fp8

    sf_rows = broadcast_block_sf(sf_raw, quantized.size(0), gran_mn)
    if dtype == "fp8":
        return cast_back_from_fp8(quantized, sf_rows, gran_k=gran_k)
    if dtype == "fp4":
        return cast_back_from_fp4(quantized, sf_rows, gran_k=gran_k)
    raise ValueError(f"unsupported dtype: {dtype}")


# --------------------------------------------------------------------------------------
# Normal (`GemmType::Normal`)
# --------------------------------------------------------------------------------------


def prepare_normal(
    *,
    M: int,
    N: int,
    K: int,
    major_a: str = "k",
    major_b: str = "k",
    accumulate: bool = False,
    cd_dtype: str = "bf16",
    b_dtype: str = "fp8",
    seed: int = 0,
) -> dict:
    """Build one dense-GEMM case, mirroring `generators.generate_normal`.

    Returns the tensors in the orientation the TMA descriptors address, the
    packed scales, an output buffer, an optional accumulator input, and a
    dequantized-matmul reference.
    """
    import torch

    require_sm100()
    torch.manual_seed(seed)

    gran_k_a, gran_k_b = gran_k_for(b_dtype)
    is_wgrad = accumulate and cd_dtype == "fp32"

    a_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

    a_q, a_sf = quantize_a(a_bf16, gran_k=gran_k_a, dtype="fp8")
    recipe_b = recipe_for(b_dtype, gran_k_b, is_a=False, is_wgrad=is_wgrad)
    b_q, b_sf = quantize_b(b_bf16, gran_k=gran_k_b, dtype=b_dtype, per_block=recipe_b[0] != 1)

    # The reference consumes exactly what the kernel will: the dequantized values.
    a_ref = dequantize(a_q, a_sf, dtype="fp8", gran_k=gran_k_a)
    b_ref = dequantize(b_q, b_sf, dtype=b_dtype, gran_k=gran_k_b, gran_mn=recipe_b[0])

    out_dtype = torch.float32 if cd_dtype == "fp32" else torch.bfloat16
    ref = (a_ref.float() @ b_ref.float().t()).to(out_dtype)

    c = None
    if accumulate:
        c = torch.randn((M, N), device="cuda", dtype=out_dtype)
        ref = (ref.float() + c.float()).to(out_dtype)
    d = torch.empty((M, N), device="cuda", dtype=out_dtype)

    sfa = transform_sf(a_sf, M, K, recipe_for("fp8", gran_k_a, is_a=True))
    sfb = transform_sf(b_sf, N, K, recipe_b)

    a_view = to_major(a_q, major_a)
    b_view = to_major(b_q, major_b)

    return {
        "M": M,
        "N": N,
        "K": K,
        "major_a": major_a,
        "major_b": major_b,
        "a_dtype": "fp8",
        "b_dtype": b_dtype,
        "cd_dtype": cd_dtype,
        "gran_k_a": gran_k_a,
        "gran_k_b": gran_k_b,
        "accumulate": accumulate,
        "a": a_view,
        "b": b_view,
        "sfa": sfa,
        "sfb": sfb,
        "a_raw_sf": a_sf,
        "b_raw_sf": b_sf,
        "c": c,
        "d": d,
        "ref": ref,
        "recipe_a": recipe_for("fp8", gran_k_a, is_a=True),
        "recipe_b": recipe_b,
        # Once `transform_sf` has run, the scales are per-row and already packed,
        # so a DeepGEMM call handed `sfa`/`sfb` must declare a `(1, gran_k)`
        # recipe -- the int32 branch of `check_sf_layout` asserts `gran_mn == 1`.
        # This is the "prepacked" fairness mode: neither side pays for the
        # layout transform inside a timed region.
        "dg_recipe_a": (1, gran_k_a),
        "dg_recipe_b": (1, gran_k_b),
        "num_sf_k_a": ceil_div(K, gran_k_a * 4),
        "num_sf_k_b": ceil_div(K, gran_k_b * 4),
        "major_a_enum": Major.K if major_a == "k" else Major.MN,
        "major_b_enum": Major.K if major_b == "k" else Major.MN,
    }


def _quantize_grouped_b(b_bf16, *, gran_k: int, dtype: str, per_block: bool):
    """Quantize a `[G, N, K]` weight slab one expert at a time.

    `per_block_cast_to_fp8` and `per_token_cast_to_fp4` are 2-D, and DeepGEMM's
    `grouped_cast_fp8_fp4_with_major` loops the same way.
    """
    import torch

    groups = [
        quantize_b(b_bf16[i], gran_k=gran_k, dtype=dtype, per_block=per_block)
        for i in range(b_bf16.size(0))
    ]
    data = torch.stack([g[0] for g in groups])
    sf = torch.stack([g[1] for g in groups])
    return data, sf


def _dequantize_grouped_b(b_q, b_sf, *, dtype: str, gran_k: int, gran_mn: int):
    import torch

    return torch.stack(
        [
            dequantize(b_q[i], b_sf[i], dtype=dtype, gran_k=gran_k, gran_mn=gran_mn)
            for i in range(b_q.size(0))
        ]
    )


# --------------------------------------------------------------------------------------
# M-grouped contiguous (`MGroupedContiguous[WithPsumLayout]`)
# --------------------------------------------------------------------------------------


def prepare_m_grouped_contiguous(
    *,
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    major_b: str = "k",
    use_psum_layout: bool = False,
    ensure_zero_padding: bool = False,
    b_dtype: str = "fp8",
    seed: int = 0,
) -> dict:
    """Mirror `generators.generate_m_grouped_contiguous`.

    Rows between a group's actual end and its aligned end are padding: the source
    zeroes them so the quantized scales there are regular rather than
    uninitialized, and marks them `-1` in the non-psum grouped layout.
    """
    import torch

    require_sm100()
    from ._sm100_fp8_fp4_gemm_1d1d import align_up, get_theoretical_mk_alignment, make_actual_ms

    torch.manual_seed(seed)
    alignment = get_theoretical_mk_alignment()
    actual_ms = make_actual_ms(num_groups, expected_m_per_group, seed)
    aligned_ms = [align_up(m, alignment) for m in actual_ms]
    M = sum(aligned_ms)

    gran_k_a, gran_k_b = gran_k_for(b_dtype)
    recipe_b = recipe_for(b_dtype, gran_k_b, is_a=False)

    a_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, N, K), device="cuda", dtype=torch.bfloat16)
    grouped_layout = torch.empty(
        num_groups if use_psum_layout else M, device="cuda", dtype=torch.int32
    )

    start = 0
    spans = []
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end, aligned_end = start + actual_m, start + aligned_m
        if use_psum_layout:
            grouped_layout[i] = actual_end
        else:
            grouped_layout[start:actual_end] = i
            grouped_layout[actual_end:aligned_end] = -1
        a_bf16[actual_end:aligned_end] = 0
        spans.append((start, actual_end, aligned_end))
        start = aligned_end

    a_q, a_sf = quantize_a(a_bf16, gran_k=gran_k_a, dtype="fp8")
    b_q, b_sf = _quantize_grouped_b(
        b_bf16, gran_k=gran_k_b, dtype=b_dtype, per_block=recipe_b[0] != 1
    )
    a_ref = dequantize(a_q, a_sf, dtype="fp8", gran_k=gran_k_a)
    b_ref = _dequantize_grouped_b(b_q, b_sf, dtype=b_dtype, gran_k=gran_k_b, gran_mn=recipe_b[0])

    ref = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    for i, (start, _actual_end, aligned_end) in enumerate(spans):
        ref[start:aligned_end] = (a_ref[start:aligned_end].float() @ b_ref[i].float().t()).to(
            torch.bfloat16
        )
    d = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    psum_layout = grouped_layout if use_psum_layout else None
    sfa = transform_sf_psum(
        a_sf, M, K, recipe_for("fp8", gran_k_a, is_a=True), psum_layout, alignment
    )
    sfb = transform_sf(b_sf, N, K, recipe_b, num_groups)

    return {
        "M": M,
        "N": N,
        "K": K,
        "num_groups": num_groups,
        "a": a_q,
        "b": to_major_grouped(b_q, major_b),
        "sfa": sfa,
        "sfb": sfb,
        "grouped_layout": grouped_layout,
        "d": d,
        "ref": ref,
        "c": None,
        "alignment": alignment,
        "use_psum_layout": use_psum_layout,
        "ensure_zero_padding": ensure_zero_padding,
        "a_dtype": "fp8",
        "b_dtype": b_dtype,
        "cd_dtype": "bf16",
        "gran_k_a": gran_k_a,
        "gran_k_b": gran_k_b,
        "major_a": "k",
        "major_b": major_b,
        "recipe_a": recipe_for("fp8", gran_k_a, is_a=True),
        "recipe_b": recipe_b,
        "dg_recipe_a": (1, gran_k_a),
        "dg_recipe_b": (1, gran_k_b),
    }


def transform_sf_psum(sf, mn: int, k: int, recipe, psum_layout, alignment: int | None = None):
    """`transform_sf` plus the optional psum layout that makes SFA skip gap rows.

    The psum group starts are aligned by `smxx_layout.hpp:195`, which reads the
    *process-global* MK alignment rather than taking it as an argument.  The
    kernel places group `j` at `align(end[j-1], BLOCK_M)`, so the packing has to
    see the same value; leaving it at DeepGEMM's default silently lays the scale
    factors out on a different grid than A and D.
    """
    deep_gemm = require_deep_gemm()
    if alignment is not None:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    if psum_layout is None:
        return deep_gemm.transform_sf_into_required_layout(sf, mn, k, recipe)
    # Positional signature: (sf, mn, k, recipe, num_groups, is_sfa,
    #                        disable_ue8m0_cast, psum_layout)
    return deep_gemm.transform_sf_into_required_layout(
        sf, mn, k, recipe, None, None, False, psum_layout
    )


def to_major_grouped(x, major: str):
    """`to_major` for a `[G, N, K]` slab."""
    if major == "k":
        return x
    if major != "mn":
        raise ValueError(f"unsupported major: {major}")
    return x.mT.contiguous().mT


# --------------------------------------------------------------------------------------
# M-grouped masked (`MGroupedMasked`)
# --------------------------------------------------------------------------------------


def prepare_m_grouped_masked(
    *,
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    max_m: int = 4096,
    b_dtype: str = "fp8",
    seed: int = 0,
) -> dict:
    """Mirror `generators.generate_m_grouped_masked`."""
    import torch

    require_sm100()
    from ._sm100_fp8_fp4_gemm_1d1d import get_theoretical_mk_alignment, make_actual_ms

    torch.manual_seed(seed)
    expected_m = int(expected_m_per_group * 1.2)
    alignment = get_theoretical_mk_alignment(expected_m)

    gran_k_a, gran_k_b = gran_k_for(b_dtype)
    recipe_b = recipe_for(b_dtype, gran_k_b, is_a=False)

    a_bf16 = torch.randn((num_groups, max_m, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, N, K), device="cuda", dtype=torch.bfloat16)
    counts = make_actual_ms(num_groups, expected_m_per_group, seed)
    if max(counts) > max_m:
        raise ValueError(f"masked_m {max(counts)} exceeds max_m {max_m}")
    masked_m = torch.tensor(counts, device="cuda", dtype=torch.int32)

    a_flat = a_bf16.view(num_groups * max_m, K)
    a_q_flat, a_sf_flat = quantize_a(a_flat, gran_k=gran_k_a, dtype="fp8")
    a_q = a_q_flat.view(num_groups, max_m, K)
    a_sf = a_sf_flat.view(num_groups, max_m, -1)
    b_q, b_sf = _quantize_grouped_b(
        b_bf16, gran_k=gran_k_b, dtype=b_dtype, per_block=recipe_b[0] != 1
    )

    # Only `[g, :masked_m[g]]` is ever compared (`masked_slice_diff`), and
    # `max_m` is the padded capacity -- typically ~16x the largest count -- so
    # the oracle covers only the rows that can be read.  The compared values are
    # unchanged; the rest of `ref` stays zero.
    used_m = max(counts)
    a_ref = dequantize(a_q_flat, a_sf_flat, dtype="fp8", gran_k=gran_k_a).view(
        num_groups, max_m, K
    )[:, :used_m]
    b_ref = _dequantize_grouped_b(b_q, b_sf, dtype=b_dtype, gran_k=gran_k_b, gran_mn=recipe_b[0])
    ref = torch.zeros((num_groups, max_m, N), device="cuda", dtype=torch.bfloat16)
    ref[:, :used_m] = torch.einsum("gmk,gnk->gmn", a_ref.float(), b_ref.float()).to(torch.bfloat16)
    d = torch.empty((num_groups, max_m, N), device="cuda", dtype=torch.bfloat16)

    sfa = transform_sf(a_sf, max_m, K, recipe_for("fp8", gran_k_a, is_a=True), num_groups)
    sfb = transform_sf(b_sf, N, K, recipe_b, num_groups)

    return {
        "M": max_m,
        "N": N,
        "K": K,
        "num_groups": num_groups,
        "expected_m": expected_m,
        "alignment": alignment,
        "a": a_q,
        "b": b_q,
        "sfa": sfa,
        "sfb": sfb,
        "masked_m": masked_m,
        "grouped_layout": masked_m,
        "d": d,
        "ref": ref,
        "c": None,
        "a_dtype": "fp8",
        "b_dtype": b_dtype,
        "cd_dtype": "bf16",
        "gran_k_a": gran_k_a,
        "gran_k_b": gran_k_b,
        "major_a": "k",
        "major_b": "k",
        "recipe_a": recipe_for("fp8", gran_k_a, is_a=True),
        "recipe_b": recipe_b,
        "dg_recipe_a": (1, gran_k_a),
        "dg_recipe_b": (1, gran_k_b),
    }


# --------------------------------------------------------------------------------------
# Reference launch closures
# --------------------------------------------------------------------------------------
#
# Each builder performs every piece of one-off work -- global alignment state, a
# warm-up call that triggers DeepGEMM's JIT compile, and output allocation --
# and then returns a no-argument closure that issues exactly one GEMM.  That is
# what `tvm.tirx.bench.bench` expects from a `references={...}` entry, and it
# keeps compilation and setup out of the timed region on both sides.


def assert_within_threshold(diff, data, *, kernel: str, detail: str, **extra) -> dict:
    """Raise unless `diff` clears the dtype-derived threshold; else return the row.

    Every entry module compares one number against `max_diff_threshold` and
    reports the same two keys, so the check lives here and each module supplies
    only its own identifying `detail`.
    """
    threshold = max_diff_threshold(data["a_dtype"], data["b_dtype"])
    if not diff < threshold:
        raise AssertionError(f"{kernel} {detail}: diff {diff:.3e} >= {threshold:.0e}")
    return {"diff": diff, "threshold": threshold, **extra}


def bench_against_deepgemm(
    tirx_launch, reference_launcher, data, *, warmup, repeat, timer, rounds, cooldown_s, **extra
) -> dict:
    """Time our launch against the DeepGEMM entry `reference_launcher` wraps.

    The reference is built lazily, so a missing or failing DeepGEMM is reported
    by `bench` as a baseline error instead of losing our own measurement.
    `extra` is merged into the result as the shape columns the report prints.
    """
    from tvm.tirx.bench import bench

    def build_reference():
        launch, _out = reference_launcher(data)
        return launch

    result = bench(
        {"tirx": tirx_launch},
        references={"deepgemm": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )
    result.update(extra)
    return result


def _warm_up(launch) -> None:
    import torch

    launch()
    torch.cuda.synchronize()


def deepgemm_launch_normal(data, *, out=None):
    """`deep_gemm.fp8_fp4_gemm_nt` over an already-prepared normal case."""
    import torch

    deep_gemm = require_deep_gemm()
    out = torch.empty_like(data["d"]) if out is None else out
    a, sfa = data["a"], data["sfa"]
    b, sfb = data["b"], data["sfb"]
    # Our kernel accumulates in place into a D that already holds C.  Handing
    # DeepGEMM a *separate* C makes `gemm.hpp:43` run a full `d.copy_(c)` inside
    # the launch: the copy is a second kernel that per-kernel attribution does not
    # charge to the GEMM, but it leaves D hot in L2 and measures ~2.4% faster than
    # the same GEMM run in place.  Pass the output as C so both sides do the same
    # read-modify-write on the same bytes.
    c = out if data["c"] is not None else None
    if c is not None:
        out.copy_(data["c"])
    recipe_a, recipe_b = data["dg_recipe_a"], data["dg_recipe_b"]

    def launch():
        deep_gemm.fp8_fp4_gemm_nt(
            (a, sfa),
            (b, sfb),
            out,
            c=c,
            disable_ue8m0_cast=False,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    _warm_up(launch)
    return launch, out


def deepgemm_launch_m_grouped_contiguous(data, *, out=None):
    """`deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`."""
    import torch

    deep_gemm = require_deep_gemm()
    # The alignment is process-global state that selects BLOCK_M; set it once,
    # outside timing, and identically for our kernel.
    deep_gemm.set_mk_alignment_for_contiguous_layout(data["alignment"])
    out = torch.empty_like(data["d"]) if out is None else out
    a, sfa, b, sfb = data["a"], data["sfa"], data["b"], data["sfb"]
    layout = data["grouped_layout"]
    use_psum, zero_pad = data["use_psum_layout"], data["ensure_zero_padding"]
    recipe_a, recipe_b = data["dg_recipe_a"], data["dg_recipe_b"]

    def launch():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a, sfa),
            (b, sfb),
            out,
            layout,
            disable_ue8m0_cast=False,
            use_psum_layout=use_psum,
            ensure_zero_padding=zero_pad,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    _warm_up(launch)
    return launch, out


def deepgemm_launch_m_grouped_masked(data, *, out=None):
    """`deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked`."""
    import torch

    deep_gemm = require_deep_gemm()
    deep_gemm.set_mk_alignment_for_contiguous_layout(data["alignment"])
    out = torch.empty_like(data["d"]) if out is None else out
    a, sfa, b, sfb = data["a"], data["sfa"], data["b"], data["sfb"]
    masked_m, expected_m = data["masked_m"], data["expected_m"]
    recipe_a, recipe_b = data["dg_recipe_a"], data["dg_recipe_b"]

    def launch():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
            (a, sfa),
            (b, sfb),
            out,
            masked_m,
            expected_m,
            disable_ue8m0_cast=False,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    _warm_up(launch)
    return launch, out


def masked_slice_diff(actual, expected, masked_m) -> float:
    """Worst per-group diff over only the valid rows of a masked output."""
    worst = 0.0
    for j in range(masked_m.numel()):
        rows = int(masked_m[j].item())
        if rows == 0:
            continue
        worst = max(worst, calc_diff(actual[j, :rows], expected[j, :rows]))
    return worst


def psum_slice_diff(actual, expected, grouped_layout, alignment, *, zero_padding=False) -> float:
    """Worst per-group diff over the valid rows of a psum-layout output.

    Group `j` occupies `[align(end[j-1], alignment), end[j])`.  The rows between
    one group's end and the next group's aligned start are padding; with
    `zero_padding` the kernel promises to write zeros there, so that is asserted
    here rather than left to a caller.
    """
    from ._sm100_fp8_fp4_gemm_1d1d import align_up

    ends = [int(v) for v in grouped_layout.tolist()]
    worst = 0.0
    for j, end in enumerate(ends):
        start = 0 if j == 0 else align_up(ends[j - 1], alignment)
        if end > start:
            worst = max(worst, calc_diff(actual[start:end], expected[start:end]))
        if zero_padding:
            gap_end = align_up(end, alignment)
            gap = actual[end:gap_end]
            if gap.numel() and bool(gap.any()):
                raise AssertionError(
                    f"ensure_zero_padding: group {j} rows [{end}, {gap_end}) are not zero"
                )
    return worst


# --------------------------------------------------------------------------------------
# Batched (`GemmType::Batched`, reached through `fp8_einsum`)
# --------------------------------------------------------------------------------------


def prepare_bmm(*, expr: str, H: int, R: int, D: int, B: int, seed: int = 0) -> dict:
    """Build one batched case, mirroring `tests/test_einsum.py`.

    `fp8_einsum` permutes its operands into `(batch, m, n, k)` order before
    calling `fp8_bmm`; we build the same permuted views, so A and C/D are
    non-contiguous in the batch dimension and the rank-3 TMA descriptors have to
    read their strides rather than assume packing.
    """
    import torch
    from deep_gemm.utils.math import ceil_div, per_block_cast_to_fp8, per_token_cast_to_fp8

    require_sm100()
    torch.manual_seed(seed)
    gran_k = 128

    if expr == "bhd,hdr->bhr":
        return _prepare_bmm_bhd_hdr_bhr(H=H, R=R, D=D, B=B, gran_k=gran_k)
    if expr == "bhd,bhr->hdr":
        return _prepare_bmm_bhd_bhr_hdr(H=H, R=R, D=D, B=B, gran_k=gran_k)
    if expr != "bhr,hdr->bhd":
        raise NotImplementedError(f"unsupported batched expression: {expr}")

    # (batch, m, n, k) = (H, B, D, R)
    x = torch.randn((B, H, R), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((H, D, R), device="cuda", dtype=torch.bfloat16)

    x_q, x_sf = per_token_cast_to_fp8(x.view(-1, R), use_ue8m0=True)
    x_q = x_q.view(B, H, R)
    x_sf = x_sf.view(B, H, ceil_div(R, gran_k))
    y_q = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_sf = torch.empty(
        (H, ceil_div(D, gran_k), ceil_div(R, gran_k)), device="cuda", dtype=torch.float
    )
    for i in range(H):
        y_q[i], y_sf[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)

    x_ref = dequantize(
        x_q.view(-1, R), x_sf.view(-1, ceil_div(R, gran_k)), dtype="fp8", gran_k=gran_k
    ).view(B, H, R)
    y_ref = torch.stack(
        [dequantize(y_q[i], y_sf[i], dtype="fp8", gran_k=gran_k, gran_mn=gran_k) for i in range(H)]
    )
    ref = torch.einsum("bhr,hdr->bhd", x_ref.float(), y_ref.float()).to(torch.bfloat16)
    z = torch.empty((B, H, D), device="cuda", dtype=torch.bfloat16)

    # The `(batch, m, n, k)` views `fp8_bmm` actually sees.
    a = x_q.permute(1, 0, 2)
    sfa_raw = x_sf.permute(1, 0, 2)
    d_view = z.permute(1, 0, 2)
    ref_view = ref.permute(1, 0, 2)

    sfa = transform_sf(sfa_raw, B, R, (1, gran_k), H)
    sfb = transform_sf(y_sf, D, R, (gran_k, gran_k), H)

    return {
        "expr": expr,
        "batch": H,
        "M": B,
        "N": D,
        "K": R,
        "a": a,
        "b": y_q,
        "sfa": sfa,
        "sfb": sfb,
        "d": d_view,
        "ref": ref_view,
        "c": None,
        "z": z,
        # `fp8_einsum` permutes each SF operand before handing it to `fp8_bmm`,
        # and permutation is its own inverse here, so passing the already-packed
        # `int32` scales through the inverse permutation lands them in exactly
        # the layout `fp8_bmm` wants.  Without this the reference repacks
        # float32 scales inside the timed region, which the other four entries
        # deliberately avoid; verified bit-identical to the float32 path.
        "x_fp8": (x_q, sfa.permute(1, 0, 2)),
        "y_fp8": (y_q, sfb),
        "a_dtype": "fp8",
        "b_dtype": "fp8",
        "cd_dtype": "bf16",
        "gran_k_a": gran_k,
        "gran_k_b": gran_k,
        "major_a": "k",
        "major_b": "k",
        "dg_recipe": (1, 1, gran_k),
    }


def deepgemm_launch_bmm(data, *, out=None):
    """`deep_gemm.fp8_einsum` over an already-prepared batched case."""
    import torch

    deep_gemm = require_deep_gemm()
    out = torch.empty_like(data["z"]) if out is None else out
    expr, x_fp8, y_fp8 = data["expr"], data["x_fp8"], data["y_fp8"]
    accumulate = data["c"] is not None
    # `(gran_mn_a, gran_mn_b, gran_k)`.  Both MN granularities are 1 because the
    # scales arrive already packed and the int32 branch of `check_sf_layout`
    # asserts `gran_mn == 1` -- the same rule the four non-batched entries follow
    # with their `(1, gran_k)` pairs.
    recipe = data["dg_recipe"]

    def launch():
        if accumulate:
            deep_gemm.fp8_einsum(expr, x_fp8, y_fp8, out, out, recipe=recipe)
        else:
            deep_gemm.fp8_einsum(expr, x_fp8, y_fp8, out, recipe=recipe)

    _warm_up(launch)
    return launch, out


def _prepare_bmm_bhd_hdr_bhr(*, H: int, R: int, D: int, B: int, gran_k: int) -> dict:
    """`bhd,hdr->bhr` -> `(batch, m, n, k) = (H, B, R, D)`, B is MN-major."""
    import torch
    from deep_gemm.utils.math import ceil_div, per_block_cast_to_fp8, per_token_cast_to_fp8

    x = torch.randn((B, H, D), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((H, D, R), device="cuda", dtype=torch.bfloat16)

    x_q, x_sf = per_token_cast_to_fp8(x.view(-1, D), use_ue8m0=True)
    x_q = x_q.view(B, H, D)
    x_sf = x_sf.view(B, H, ceil_div(D, gran_k))
    y_q = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_sf = torch.empty(
        (H, ceil_div(D, gran_k), ceil_div(R, gran_k)), device="cuda", dtype=torch.float
    )
    for i in range(H):
        y_q[i], y_sf[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)

    x_ref = dequantize(
        x_q.view(-1, D), x_sf.view(-1, ceil_div(D, gran_k)), dtype="fp8", gran_k=gran_k
    ).view(B, H, D)
    y_ref = torch.stack(
        [dequantize(y_q[i], y_sf[i], dtype="fp8", gran_k=gran_k, gran_mn=gran_k) for i in range(H)]
    )
    ref = torch.einsum("bhd,hdr->bhr", x_ref.float(), y_ref.float()).to(torch.bfloat16)
    z = torch.empty((B, H, R), device="cuda", dtype=torch.bfloat16)

    sfa = transform_sf(x_sf.permute(1, 0, 2), B, D, (1, gran_k), H)
    sfb = transform_sf(y_sf.permute(0, 2, 1), R, D, (gran_k, gran_k), H)

    return {
        "expr": "bhd,hdr->bhr",
        "batch": H,
        "M": B,
        "N": R,
        "K": D,
        "a": x_q.permute(1, 0, 2),
        "b": y_q.permute(0, 2, 1),
        "sfa": sfa,
        "sfb": sfb,
        "d": z.permute(1, 0, 2),
        "ref": ref.permute(1, 0, 2),
        "c": None,
        "z": z,
        # See the note in `_prepare_bmm_bhr_hdr_bhd`: the already-packed scales
        # go through `fp8_einsum`'s inverse permutation so the reference does not
        # repack float32 scales inside the timed region.
        "x_fp8": (x_q, sfa.permute(1, 0, 2)),
        "y_fp8": (y_q, sfb.permute(0, 2, 1)),
        "a_dtype": "fp8",
        "b_dtype": "fp8",
        "cd_dtype": "bf16",
        "gran_k_a": gran_k,
        "gran_k_b": gran_k,
        "major_a": "k",
        "major_b": "mn",
        "dg_recipe": (1, 1, gran_k),
    }


def _prepare_bmm_bhd_bhr_hdr(*, H: int, R: int, D: int, B: int, gran_k: int) -> dict:
    """`bhd,bhr->hdr` -> `(H, D, R, B)`: both operands MN-major, FP32 + accumulate."""
    import torch
    from deep_gemm.utils.math import ceil_div, per_channel_cast_to_fp8

    x = torch.randn((B, H, D), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((B, H, R), device="cuda", dtype=torch.bfloat16)
    z0 = torch.randn((H, D, R), device="cuda", dtype=torch.float32) * 10

    x_q, x_sf = per_channel_cast_to_fp8(x.view(B, -1), use_ue8m0=True)
    y_q, y_sf = per_channel_cast_to_fp8(y.view(B, -1), use_ue8m0=True)
    x_q, x_sf = x_q.view(B, H, D), x_sf.view(ceil_div(B, gran_k), H, D)
    y_q, y_sf = y_q.view(B, H, R), y_sf.view(ceil_div(B, gran_k), H, R)

    # `per_channel` scales along the reduction axis (B), so dequantizing means
    # broadcasting each 128-row block's scale back over its rows.
    x_ref = (
        x_q.float().view(B, -1)
        * x_sf.view(ceil_div(B, gran_k), -1).repeat_interleave(gran_k, dim=0)[:B]
    ).view(B, H, D)
    y_ref = (
        y_q.float().view(B, -1)
        * y_sf.view(ceil_div(B, gran_k), -1).repeat_interleave(gran_k, dim=0)[:B]
    ).view(B, H, R)
    ref = z0 + torch.einsum("bhd,bhr->hdr", x_ref, y_ref)
    z = z0.clone()

    sfa = transform_sf(x_sf.permute(1, 2, 0), D, B, (1, gran_k), H)
    sfb = transform_sf(y_sf.permute(1, 2, 0), R, B, (1, gran_k), H)

    return {
        "expr": "bhd,bhr->hdr",
        "batch": H,
        "M": D,
        "N": R,
        "K": B,
        "a": x_q.permute(1, 2, 0),
        "b": y_q.permute(1, 2, 0),
        "sfa": sfa,
        "sfb": sfb,
        "d": z,
        "ref": ref,
        "c": z,
        "z": z,
        "z0": z0,
        # See the note in `_prepare_bmm_bhr_hdr_bhd`; `(1, 2, 0)` inverts to
        # `(2, 0, 1)`.
        "x_fp8": (x_q, sfa.permute(2, 0, 1)),
        "y_fp8": (y_q, sfb.permute(2, 0, 1)),
        "a_dtype": "fp8",
        "b_dtype": "fp8",
        "cd_dtype": "fp32",
        "gran_k_a": gran_k,
        "gran_k_b": gran_k,
        "major_a": "mn",
        "major_b": "mn",
        "dg_recipe": (1, 1, gran_k),
    }


# --------------------------------------------------------------------------------------
# K-grouped contiguous (`KGroupedContiguous[WithPsumLayout]`)
# --------------------------------------------------------------------------------------


def k_grouped_quantize(x, ks: list[int], *, gran_k: int, group_ends: list[int] | None = None):
    """Per-channel quantization, one group at a time.

    Mirrors `generators.k_grouped_per_channel_cast_to_fp8`: each group is padded
    up to `gran_k` before quantizing so its scale rows stay compact
    (`ceil_div(k, gran_k)` rows per group), and empty groups contribute nothing.
    """
    import itertools

    import torch
    from deep_gemm.utils.math import align, per_channel_cast_to_fp8

    if x.dim() != 2:
        raise ValueError("k-grouped operands are 2-D `[sum_k, mn]`")
    if group_ends is None:
        group_ends = list(itertools.accumulate(ks))
        if (group_ends[-1] if ks else 0) != x.size(0):
            raise ValueError("group lengths do not cover the tensor")

    n = x.size(1)
    quantized = torch.zeros(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    sf_groups = []
    for k, end in zip(ks, group_ends):
        if k == 0:
            continue
        start = end - k
        padded = torch.zeros((align(k, gran_k), n), dtype=x.dtype, device=x.device)
        padded[:k] = x[start:end]
        q, sf = per_channel_cast_to_fp8(padded, use_ue8m0=True, gran_k=gran_k)
        quantized[start:end] = q[:k]
        sf_groups.append(sf)
    sf = (
        torch.cat(sf_groups)
        if sf_groups
        else torch.empty((0, n), dtype=torch.float, device=x.device)
    )
    return quantized, sf


def k_grouped_dequantize(q, sf, ks: list[int], *, gran_k: int, group_ends: list[int] | None = None):
    """Undo `k_grouped_quantize`, giving the values the MMA actually consumes."""
    import itertools

    import torch
    from deep_gemm.utils.math import ceil_div

    if group_ends is None:
        group_ends = list(itertools.accumulate(ks))
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    sf_row = 0
    for k, end in zip(ks, group_ends):
        if k == 0:
            continue
        start = end - k
        rows = ceil_div(k, gran_k)
        block = sf[sf_row : sf_row + rows]
        sf_row += rows
        out[start:end] = q[start:end].float() * block.repeat_interleave(gran_k, dim=0)[:k]
    return out


def prepare_k_grouped(
    *,
    num_groups: int,
    M: int,
    N: int,
    expected_k_per_group: int,
    gran_k: int,
    k_alignment: int,
    use_psum_layout: bool = False,
    empty_group: bool = False,
    short_first_group: bool = False,
    seed: int = 0,
) -> dict:
    """Mirror `generators.generate_k_grouped_contiguous` (TN, FP32 + accumulate)."""
    import torch

    require_sm100()
    deep_gemm = require_deep_gemm()
    from ._sm100_fp8_fp4_gemm_1d1d import align_up
    from .k_grouped_fp8_gemm_contiguous import make_ks

    torch.manual_seed(seed)
    real_ks, aligned_ks = make_ks(
        num_groups=num_groups,
        expected_k_per_group=expected_k_per_group,
        k_alignment=k_alignment,
        seed=seed,
        empty_group=empty_group,
        short_first_group=short_first_group,
    )
    if use_psum_layout:
        # Group `i` starts at `align(end[i-1], k_alignment)` and runs for its real K.
        ends, cursor = [], 0
        for k in real_ks:
            cursor = align_up(cursor, k_alignment) + k
            ends.append(cursor)
        total_k = align_up(ends[-1], k_alignment) if ends else 0
        ks, group_ends = real_ks, ends
        layout = torch.tensor(ends, device="cuda", dtype=torch.int32)
    else:
        total_k = sum(aligned_ks)
        ks, group_ends = aligned_ks, None
        layout = torch.tensor(aligned_ks, device="cuda", dtype=torch.int32)

    a_bf16 = torch.randn((total_k, M), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((total_k, N), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((num_groups, M, N), device="cuda", dtype=torch.float32) * 32

    a_q, a_sf = k_grouped_quantize(a_bf16, ks, gran_k=gran_k, group_ends=group_ends)
    b_q, b_sf = k_grouped_quantize(b_bf16, ks, gran_k=gran_k, group_ends=group_ends)
    a_ref = k_grouped_dequantize(a_q, a_sf, ks, gran_k=gran_k, group_ends=group_ends)
    b_ref = k_grouped_dequantize(b_q, b_sf, ks, gran_k=gran_k, group_ends=group_ends)

    import itertools

    ends_it = group_ends or list(itertools.accumulate(ks))
    ref = c.clone()
    for i, (k, end) in enumerate(zip(ks, ends_it)):
        if k == 0:
            continue
        ref[i] += a_ref[end - k : end].t() @ b_ref[end - k : end]
    d = c.clone()

    sfa = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        a_sf, layout, aligned_ks, gran_k, k_alignment, use_psum_layout
    )
    sfb = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        b_sf, layout, aligned_ks, gran_k, k_alignment, use_psum_layout
    )

    return {
        "num_groups": num_groups,
        "M": M,
        "N": N,
        "K": total_k,
        "ks": ks,
        "real_ks": real_ks,
        "aligned_ks": aligned_ks,
        # Stored `[sum_k, MN]`; the descriptor wants the logical `[MN, sum_k]`
        # MN-major view, whose outer stride is MN rather than 1.
        "a": a_q.t(),
        "b": b_q.t(),
        "a_store": a_q,
        "b_store": b_q,
        "sfa": sfa,
        "sfb": sfb,
        "grouped_layout": layout,
        "c": c,
        "d": d,
        "ref": ref,
        "use_psum_layout": use_psum_layout,
        "gran_k_a": gran_k,
        "gran_k_b": gran_k,
        "k_alignment": k_alignment,
        "a_dtype": "fp8",
        "b_dtype": "fp8",
        "cd_dtype": "fp32",
        "major_a": "mn",
        "major_b": "mn",
        "dg_recipe": (1, 1, gran_k),
    }


def deepgemm_launch_k_grouped(data, *, out=None):
    """`deep_gemm.k_grouped_fp8_gemm_tn_contiguous`."""
    import torch

    deep_gemm = require_deep_gemm()
    deep_gemm.set_mk_alignment_for_contiguous_layout(data["k_alignment"])
    # Its own accumulator: sharing `data["d"]` with our launch would put two
    # implementations' read-modify-writes on one buffer, which the other four
    # entries avoid.
    out = torch.empty_like(data["d"]) if out is None else out
    # DeepGEMM takes the `[sum_k, MN]` storage view, not the transposed one our
    # descriptors address.
    a, sfa, b, sfb = data["a_store"], data["sfa"], data["b_store"], data["sfb"]
    aligned_ks, layout, c = data["aligned_ks"], data["grouped_layout"], data["c"]
    recipe, use_psum = data["dg_recipe"], data["use_psum_layout"]

    # Seed the accumulator once, outside the timed closure: our kernel is handed a
    # D that already holds C, so charging the reference for the copy -- or letting
    # the copy warm its output in L2 -- would compare different work.
    out.copy_(c)

    def launch():
        deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
            (a, sfa),
            (b, sfb),
            out,
            aligned_ks,
            layout,
            out,
            recipe=recipe,
            use_psum_layout=use_psum,
        )

    _warm_up(launch)
    return launch, out
