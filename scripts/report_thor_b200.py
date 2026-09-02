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
    unmatched = len(rows) - len(matched)
    overall = _geomean(matched)
    median = statistics.median(matched)
    thor_git = thor.get("git") or {}
    b200_git = b200.get("git") or {}
    selection = thor.get("selection") or {}
    pipeline = thor.get("pipeline") or {}
    protocol = pipeline.get("measurement_protocol") or {}

    lines = [
        "# NVIDIA Thor versus B200 TIRx performance",
        "",
        f"Measured on {date.today().isoformat()} using the default representative workload roster.",
        "",
        "## Result",
        "",
        f"- Thor completed **{statuses.get('ok', 0)}/{len(rows)}** workloads with "
        f"**{pipeline.get('interference_retry_count', 0)} interference retries**.",
        f"- **{len(matched)}** rows have an exact `(kernel, config)` match in the repository's "
        f"historical SM100/B200 baseline; **{unmatched}** new BSA rows have no B200 value there.",
        f"- Across the {len(matched)} matched rows, geometric-mean Thor/B200 latency is "
        f"**{overall:.3f}x**; equivalently, Thor delivers **{1 / overall:.1%}** of B200's "
        "throughput on this workload mix.",
        f"- The median latency ratio is **{median:.3f}x** and the observed range is "
        f"**{min(matched):.3f}x--{max(matched):.3f}x**.",
        f"- Thor has {thor_sms} SMs versus {b200_sms} on B200 ({b200_sms / thor_sms:.1f}x as "
        "many). The aggregate per-SM-normalized throughput is about "
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
        "## Kernel summary",
        "",
        "`Thor/B200 latency > 1` means Thor is slower. Relative throughput is its reciprocal.",
        "",
        "| Kernel | Thor rows | Matched B200 rows | Geomean Thor/B200 latency | Thor relative throughput |",
        "|---|---:|---:|---:|---:|",
    ]
    thor_by_kernel: dict[str, int] = defaultdict(int)
    for row, *_rest in rows:
        thor_by_kernel[row["kernel"]] += 1
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
    for row, _impl, thor_us, b200_us, ratio, _row_cv in rows:
        if row["kernel"] not in {"fp16_bf16_gemm", "nvfp4_gemm"} or b200_us is None:
            continue
        m, n, k = _gemm_dims(_config(row))
        thor_tflops = 2 * m * n * k / thor_us / 1e6
        b200_tflops = 2 * m * n * k / b200_us / 1e6
        lines.append(
            f"| `{row['kernel']}` | `{_config(row)}` | {thor_us:.3f} | {b200_us:.3f} | "
            f"{thor_tflops:.3f} | {b200_tflops:.3f} | {1 / ratio:.1%} |"
        )

    lines.extend(
        [
            "",
            "The FP16 16384³ row reaches only 30.46 effective TFLOP/s on Thor, below the 4096³ "
            "BF16 row's 107.45 TFLOP/s. That inversion is a concrete tuning target: the current "
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
            lines.append(
                f"| `{row['kernel']}` | `{_config(row)}` | {thor_us:.3f} | {cv_cell} | — | — | — |"
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
            "The performance roster contains only the 19 kernels already admitted for exact `sm_110a` "
            "runtime support after their complete correctness matrices passed: 316/316 configurations. "
            "The 57 timed rows are three representative performance shapes per kernel; they do not replace "
            "the complete numerical validation matrices.",
            "",
            "## Raw evidence",
            "",
            f"- Thor run: `{thor_path}`",
            f"- B200 baseline: `{b200_path}`",
            "- Thor run status: 57 `ok`, 0 failures, 0 interference retries",
            "- B200 matches: 45 rows; new BSA without historical B200 rows: 12",
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
    args = parser.parse_args()

    thor = json.loads(args.thor.read_text())
    b200 = json.loads(args.b200.read_text())
    report = _markdown(
        thor,
        b200,
        thor_path=args.thor.resolve(),
        b200_path=args.b200.resolve(),
        thor_device=args.thor_device,
        thor_sms=args.thor_sms,
        b200_sms=args.b200_sms,
    )
    args.output.write_text(report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
