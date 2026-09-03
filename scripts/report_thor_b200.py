#!/usr/bin/env python3
# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Render a Thor versus repository SM100/B200 benchmark comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path


def _config(row: dict) -> str:
    return row.get("config") or row.get("label") or "?"


def _ours(row: dict | None) -> tuple[str | None, float | None]:
    if row is None:
        return None, None
    for name, value in (row.get("impls") or {}).items():
        if name in {"tir", "tirx"} or name.startswith(("tir-", "tirx-")):
            return name, float(value)
    return None, None


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _cv(row: dict, impl: str) -> float:
    samples = [float(value) for value in (row.get("round_samples") or {}).get(impl, [])]
    if len(samples) < 2 or statistics.fmean(samples) == 0:
        return 0.0
    return 100.0 * statistics.pstdev(samples) / statistics.fmean(samples)


def _gemm_dims(config: str) -> tuple[int, int, int]:
    shape = config.rsplit("_", 1)[-1]
    return tuple(int(value) for value in shape.split("x"))


def _markdown(
    thor: dict,
    b200: dict,
    *,
    thor_path: Path,
    b200_path: Path,
    thor_device: str,
    thor_sms: int,
    b200_sms: int,
    validated_configs: int,
    prior_thor: dict | None,
    prior_thor_path: Path | None,
) -> str:
    b200_rows = {(row["kernel"], _config(row)): row for row in b200["results"]}
    rows = []
    by_kernel: dict[str, list[float]] = defaultdict(list)
    for thor_row in thor["results"]:
        thor_impl, thor_us = _ours(thor_row)
        if thor_impl is None or thor_us is None:
            raise ValueError(
                f"missing TIR/TIRx timing for {thor_row['kernel']}/{_config(thor_row)}"
            )
        b200_row = b200_rows.get((thor_row["kernel"], _config(thor_row)))
        _b200_impl, b200_us = _ours(b200_row)
        ratio = thor_us / b200_us if b200_us is not None else None
        if ratio is not None:
            by_kernel[thor_row["kernel"]].append(ratio)
        rows.append((thor_row, thor_impl, thor_us, b200_us, ratio, _cv(thor_row, thor_impl)))

    statuses: dict[str, int] = defaultdict(int)
    for row in thor["results"]:
        statuses[row.get("status") or "?"] += 1
    matched = [ratio for *_prefix, ratio, _cv_value in rows if ratio is not None]
    noisy = sum(cv > 10.0 for *_prefix, cv in rows)
    missing_b200 = sum((row["kernel"], _config(row)) not in b200_rows for row, *_rest in rows)
    unusable_b200 = len(rows) - len(matched) - missing_b200
    overall = _geomean(matched)
    median = statistics.median(matched)
    thor_git = thor.get("git") or {}
    b200_git = b200.get("git") or {}
    selection = thor.get("selection") or {}
    pipeline = thor.get("pipeline") or {}
    protocol = pipeline.get("measurement_protocol") or {}
    thor_by_kernel: dict[str, int] = defaultdict(int)
    for row, *_rest in rows:
        thor_by_kernel[row["kernel"]] += 1

    session_comparison = None
    if prior_thor is not None:
        prior_rows = {(row["kernel"], _config(row)): _ours(row)[1] for row in prior_thor["results"]}
        shifts = []
        for row, _impl, thor_us, *_rest in rows:
            key = (row["kernel"], _config(row))
            prior_us = prior_rows.get(key)
            if prior_us is not None:
                ratio = thor_us / prior_us
                shifts.append((max(ratio, 1 / ratio), ratio, key, prior_us, thor_us))
        if shifts:
            largest = max(shifts)
            session_comparison = {
                "count": len(shifts),
                "geomean": _geomean([shift[1] for shift in shifts]),
                "median": statistics.median(shift[1] for shift in shifts),
                "largest": largest,
            }

    lines = [
        "# NVIDIA Thor versus B200 TIRx performance",
        "",
        f"Measured on {date.today().isoformat()} using the default representative workload roster.",
        "",
        "## Result",
        "",
        f"- Thor completed **{statuses.get('ok', 0)}/{len(rows)}** workloads with "
        f"**{pipeline.get('interference_retry_count', 0)} interference retries**.",
        f"- **{len(matched)}** rows have a usable TIR/TIRx timing in the repository's historical "
        f"SM100/B200 baseline; **{unusable_b200}** exact matched baseline rows failed, and "
        f"**{missing_b200}** Thor workload rows are absent there.",
        f"- Across the {len(matched)} matched rows, geometric-mean Thor/B200 latency is "
        f"**{overall:.3f}x**; equivalently, Thor delivers **{1 / overall:.1%}** of B200's "
        "throughput on this workload mix.",
        f"- The median latency ratio is **{median:.3f}x** and the observed range is "
        f"**{min(matched):.3f}x--{max(matched):.3f}x**.",
        f"- Thor has {thor_sms} SMs versus {b200_sms} on B200; B200 has "
        f"{b200_sms / thor_sms:.1f}x as many. The aggregate per-SM-normalized throughput is about "
        f"**{(1 / overall) * (b200_sms / thor_sms):.1%}** of the B200 baseline, but this is "
        "only a rough diagnostic because the table mixes compute-, bandwidth-, and latency-bound kernels.",
        "",
        "The geometric mean is a descriptive summary, not a model-level score. The detailed rows below "
        "are the authoritative data.",
        "",
        "## Measurement provenance",
        "",
        "| Field | Thor | Repository SM100/B200 baseline |",
        "|---|---|---|",
        f"| GPU | {thor_device} | B200 attribution inferred from `sm_100a` and the suite's 148-SM B200 annotations; the JSON does not store a product name |",
        f"| SM count | {thor_sms} | {b200_sms} |",
        f"| CUDA architecture | `{selection.get('cuda_arch', 'sm_110a')}` | `sm_100a` |",
        "| Power/clock state | `MAXN`; dynamic clocks (`jetson_clocks` was not locked) | Not recorded |",
        f"| Timer | `{selection.get('timer_override', 'proton')}` | `proton` |",
        f"| Rounds | {protocol.get('rounds', 5)}, arithmetic mean | 5, arithmetic mean |",
        "| Warmup / repeat budget | 25 ms / 100 ms | 25 ms / 100 ms |",
        f"| TVM/TIR revision | `{thor_git.get('tir', '-')}` | `{b200_git.get('tir', '-')}` |",
        f"| TIRx-kernels revision | `{thor_git.get('tirx-kernels', '-')}` | `{b200_git.get('tirx-kernels', '-')}` |",
        "| CUDA/PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 | CUDA 13.2 / PyTorch 2.13.0+cu132 |",
        "",
        "Thor's Triton 3.5.1 package bundles CUDA 12.8 CUPTI, which cannot initialize against this "
        "CUDA 13.1 Thor stack. The run therefore set "
        "`TRITON_CUPTI_LIB_PATH=/usr/local/cuda-13.1/extras/CUPTI/lib64`; both final columns still "
        "use the same Proton timer protocol.",
        "",
        "This is a historical cross-machine comparison, not a controlled hardware-only A/B: the TVM, "
        "TIRx, CUDA, PyTorch, and CUPTI revisions differ, and the repository baseline labels its "
        "TIRx checkout as dirty. A publication-grade comparison requires rerunning this exact TIRx/TVM "
        "revision on a B200.",
        "",
    ]
    if session_comparison is not None:
        _largest_factor, largest_ratio, largest_key, largest_prior, largest_current = (
            session_comparison["largest"]
        )
        lines.extend(
            [
                "## Cross-session stability",
                "",
                f"Against the preceding same-protocol Thor run, {session_comparison['count']} common "
                f"rows have geometric-mean current/prior latency **{session_comparison['geomean']:.3f}x** "
                f"and median **{session_comparison['median']:.3f}x**. The largest individual shift is "
                f"`{largest_key[0]}/{largest_key[1]}`: {largest_prior:.3f} µs to "
                f"{largest_current:.3f} µs ({largest_ratio:.3f}x).",
                "",
                "The aggregate is repeatable, but individual absolute times can move substantially "
                "between sessions under dynamic clocks. The complete table uses only the final, "
                f"single-piece {len(rows)}-row run; no samples were spliced from the earlier run.",
                "",
            ]
        )
    lines.extend(
        [
            "## Kernel summary",
            "",
            "`Thor/B200 latency > 1` means Thor is slower. Relative throughput is its reciprocal.",
            "",
            "| Kernel | Thor rows | Matched B200 rows | Geomean Thor/B200 latency | Thor relative throughput |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for kernel in sorted(thor_by_kernel):
        ratios = by_kernel[kernel]
        if ratios:
            geomean = _geomean(ratios)
            lines.append(
                f"| `{kernel}` | {thor_by_kernel[kernel]} | {len(ratios)} | "
                f"{geomean:.3f}x | {1 / geomean:.1%} |"
            )
        else:
            lines.append(f"| `{kernel}` | {thor_by_kernel[kernel]} | 0 | — | — |")

    lines.extend(
        [
            "",
            "## GEMM effective throughput",
            "",
            "Throughput uses the conventional `2*M*N*K` operation count. NVFP4 values are effective "
            "throughput and include neither scale-processing operations nor any sparsity multiplier.",
            "",
            "| Kernel | Config | Thor µs | B200 µs | Thor effective TFLOP/s | B200 effective TFLOP/s | Thor/B200 throughput |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    gemm_thor_tflops: dict[tuple[str, str], float] = {}
    for row, _impl, thor_us, b200_us, ratio, _row_cv in rows:
        if row["kernel"] not in {"fp16_bf16_gemm", "nvfp4_gemm"} or b200_us is None:
            continue
        m, n, k = _gemm_dims(_config(row))
        thor_tflops = 2 * m * n * k / thor_us / 1e6
        b200_tflops = 2 * m * n * k / b200_us / 1e6
        gemm_thor_tflops[(row["kernel"], _config(row))] = thor_tflops
        lines.append(
            f"| `{row['kernel']}` | `{_config(row)}` | {thor_us:.3f} | {b200_us:.3f} | "
            f"{thor_tflops:.3f} | {b200_tflops:.3f} | {1 / ratio:.1%} |"
        )

    lines.extend(
        [
            "",
            f"The FP16 16384³ row reaches only "
            f"{gemm_thor_tflops[('fp16_bf16_gemm', 'fp16_16384x16384x16384')]:.2f} "
            "effective TFLOP/s on Thor, below the 4096³ BF16 row's "
            f"{gemm_thor_tflops[('fp16_bf16_gemm', 'bf16_4096x4096x4096')]:.2f} TFLOP/s. "
            "That inversion is a concrete tuning target: the current "
            "B200-oriented schedule does not scale well to Thor's 20-SM device at that shape.",
            "",
            "## Complete workload table",
            "",
            f"Thor CV is the population coefficient of variation across five round means. **{noisy}** "
            "rows exceed 10% and are marked `†`; repeat those rows under locked clocks before using "
            "small differences for tuning decisions.",
            "",
            "| Kernel | Config | Thor Proton µs | Thor CV | B200 Proton µs | Thor/B200 latency | Thor relative throughput |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row, _impl, thor_us, b200_us, ratio, row_cv in sorted(
        rows, key=lambda item: (item[0]["kernel"], _config(item[0]))
    ):
        cv_cell = f"{row_cv:.1f}%" + (" †" if row_cv > 10.0 else "")
        if b200_us is None or ratio is None:
            b200_row = b200_rows.get((row["kernel"], _config(row)))
            b200_cell = f"`{b200_row.get('status', 'unusable')}`" if b200_row else "—"
            lines.append(
                f"| `{row['kernel']}` | `{_config(row)}` | {thor_us:.3f} | {cv_cell} | "
                f"{b200_cell} | — | — |"
            )
        else:
            lines.append(
                f"| `{row['kernel']}` | `{_config(row)}` | {thor_us:.3f} | {cv_cell} | "
                f"{b200_us:.3f} | {ratio:.3f}x | {1 / ratio:.1%} |"
            )

    lines.extend(
        [
            "",
            "## Correctness scope",
            "",
            f"The performance roster contains only the {len(thor_by_kernel)} kernels already admitted "
            "for exact `sm_110a` runtime support after their complete correctness matrices passed: "
            f"{validated_configs}/{validated_configs} configurations. The {len(rows)} timed rows are "
            "the suite's selected representative performance shapes; they do not replace the "
            "complete numerical validation matrices.",
            "",
            "## Raw evidence",
            "",
            f"- Thor run: `{thor_path}`",
            f"- B200 baseline: `{b200_path}`",
            *((f"- Prior Thor stability run: `{prior_thor_path}`",) if prior_thor_path else ()),
            f"- Thor run status: {statuses.get('ok', 0)} `ok`, "
            f"{pipeline.get('failure_count', 0)} failures, "
            f"{pipeline.get('interference_retry_count', 0)} interference retries",
            f"- Usable B200 matches: {len(matched)} rows; failed B200 baseline rows: "
            f"{unusable_b200}; workload rows absent from the B200 baseline: {missing_b200}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thor", type=Path, required=True)
    parser.add_argument("--b200", type=Path, default=Path("tirx_kernels/bench_suite/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("THOR_B200_PERFORMANCE.md"))
    parser.add_argument("--thor-device", default="NVIDIA Jetson AGX Thor Developer Kit")
    parser.add_argument("--thor-sms", type=int, default=20)
    parser.add_argument("--b200-sms", type=int, default=148)
    parser.add_argument(
        "--validated-configs",
        type=int,
        required=True,
        help="Number of correctness configurations passed by the admitted Thor kernels",
    )
    parser.add_argument("--prior-thor", type=Path, default=None)
    args = parser.parse_args()

    thor = json.loads(args.thor.read_text())
    b200 = json.loads(args.b200.read_text())
    prior_thor = json.loads(args.prior_thor.read_text()) if args.prior_thor else None
    report = _markdown(
        thor,
        b200,
        thor_path=args.thor.resolve(),
        b200_path=args.b200.resolve(),
        thor_device=args.thor_device,
        thor_sms=args.thor_sms,
        b200_sms=args.b200_sms,
        validated_configs=args.validated_configs,
        prior_thor=prior_thor,
        prior_thor_path=args.prior_thor.resolve() if args.prior_thor else None,
    )
    args.output.write_text(report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
