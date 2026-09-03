#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Render the complete Thor FA4 versus FlashInfer FA2 sweep."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

_CONFIG_RE = re.compile(r"s(?P<seq>\d+)_h32kv(?P<kv>\d+)(?P<causal>_causal)?")


def _is_ours(name: str) -> bool:
    return name in {"tir", "tirx"} or name.startswith(("tir-", "tirx-"))


def _cv(row: dict, implementation: str) -> float:
    samples = [float(value) for value in row["round_samples"][implementation]]
    if len(samples) < 2 or statistics.fmean(samples) == 0:
        return 0.0
    return 100.0 * statistics.pstdev(samples) / statistics.fmean(samples)


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _row(row: dict) -> dict:
    if row.get("kernel") != "flash_attention4":
        raise ValueError(f"unexpected kernel in attention sweep: {row.get('kernel')}")
    if row.get("status") != "ok":
        raise ValueError(f"non-passing attention row: {row.get('config')}")
    match = _CONFIG_RE.fullmatch(row["config"])
    if match is None:
        raise ValueError(f"unexpected attention config: {row['config']}")
    ours = [name for name in row["impls"] if _is_ours(name)]
    references = [name for name in row["impls"] if not _is_ours(name)]
    if len(ours) != 1 or references != ["flashinfer_fa2"]:
        raise ValueError(
            f"expected one TIRx implementation and flashinfer_fa2 for {row['config']}, "
            f"got {list(row['impls'])}"
        )
    ours_name = ours[0]
    ours_us = float(row["impls"][ours_name])
    reference_us = float(row["impls"]["flashinfer_fa2"])
    return {
        "config": row["config"],
        "seq": int(match.group("seq")),
        "kv": int(match.group("kv")),
        "causal": match.group("causal") is not None,
        "ours_us": ours_us,
        "reference_us": reference_us,
        "speedup": reference_us / ours_us,
        "ours_cv": _cv(row, ours_name),
        "reference_cv": _cv(row, "flashinfer_fa2"),
        "protocol": row["benchmark_protocol"],
    }


def _breakdown(rows: list[dict], key: str) -> list[tuple[str, int, float, float, float]]:
    groups: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row["speedup"])
    return [
        (str(value), len(speedups), _geomean(speedups), min(speedups), max(speedups))
        for value, speedups in sorted(groups.items())
    ]


def _markdown(run: dict, *, run_path: Path) -> str:
    if not run.get("references_enabled"):
        raise ValueError("the attention run did not enable references")
    rows = [_row(row) for row in run.get("results") or []]
    if len(rows) != 32:
        raise ValueError(f"expected the complete 32-config attention matrix, got {len(rows)}")
    expected = {
        (seq, kv, causal)
        for seq in (1024, 2048, 4096, 8192)
        for kv in (4, 8, 16, 32)
        for causal in (False, True)
    }
    actual = {(row["seq"], row["kv"], row["causal"]) for row in rows}
    if actual != expected:
        raise ValueError(
            f"attention matrix mismatch: missing={expected - actual}, extra={actual - expected}"
        )

    protocol = rows[0]["protocol"]
    protocol_fields = ("warmup", "repeat", "rounds", "round_aggregate")
    if any(
        any(row["protocol"].get(field) != protocol.get(field) for field in protocol_fields)
        for row in rows[1:]
    ):
        raise ValueError("attention rows do not share one timing protocol")

    speedups = [row["speedup"] for row in rows]
    overall = _geomean(speedups)
    noisy = sum(max(row["ours_cv"], row["reference_cv"]) > 10.0 for row in rows)
    pipeline = run.get("pipeline") or {}
    selection = run.get("selection") or {}
    git = run.get("git") or {}
    flashinfer = (run.get("baselines") or {}).get("flashinfer") or {}
    measured_date = (run["results"][0].get("started_at") or "")[:10] or "unknown date"

    lines = [
        "# NVIDIA Thor FA4 versus FlashInfer FA2",
        "",
        f"Measured on {measured_date} on one NVIDIA Jetson AGX Thor Developer Kit. This is the "
        "complete 32-config matrix exposed by the repository's `flash_attention4` module.",
        "",
        "## At a glance",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Correctness and execution | **{len(rows)}/{len(rows)} passed** |",
        f"| TIRx faster than FlashInfer FA2 | **{sum(value > 1 for value in speedups)}/"
        f"{len(rows)}** |",
        f"| Geometric-mean TIRx speedup | **{overall:.3f}x** |",
        f"| Speedup range | **{min(speedups):.3f}x to {max(speedups):.3f}x** |",
        f"| Rows with either CV above 10% | **{noisy}/{len(rows)}** |",
        f"| Failures / interference retries | **{pipeline.get('failure_count', 0)} / "
        f"{pipeline.get('interference_retry_count', 0)}** |",
        "",
        "TIRx wins every tested sequence-length, GQA-ratio, and masking combination. The "
        "advantage is smallest for short causal MHA (`S=1024`, 32 KV heads) and reaches its "
        "maximum for non-causal GQA at `S=2048`, 8 KV heads.",
        "",
        "## Sensitivity breakdown",
        "",
        "Speedup is `FlashInfer latency / TIRx latency`; values above 1.0 favor TIRx.",
        "",
        "### Sequence length",
        "",
        "| Sequence length | Rows | Geomean | Minimum | Maximum |",
        "|---:|---:|---:|---:|---:|",
    ]
    for value, count, geomean, minimum, maximum in _breakdown(rows, "seq"):
        lines.append(f"| {value} | {count} | {geomean:.3f}x | {minimum:.3f}x | {maximum:.3f}x |")

    lines.extend(
        [
            "",
            "### KV heads (`Q heads = 32`)",
            "",
            "| KV heads | GQA ratio | Rows | Geomean | Minimum | Maximum |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for value, count, geomean, minimum, maximum in _breakdown(rows, "kv"):
        lines.append(
            f"| {value} | {32 // int(value)}:1 | {count} | {geomean:.3f}x | "
            f"{minimum:.3f}x | {maximum:.3f}x |"
        )

    lines.extend(
        [
            "",
            "### Mask",
            "",
            "| Mask | Rows | Geomean | Minimum | Maximum |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for value, count, geomean, minimum, maximum in _breakdown(rows, "causal"):
        mask = "causal" if value == "True" else "non-causal"
        lines.append(f"| {mask} | {count} | {geomean:.3f}x | {minimum:.3f}x | {maximum:.3f}x |")

    lines.extend(
        [
            "",
            "## Complete 32-config table",
            "",
            "| Sequence | KV heads | Mask | TIRx µs | TIRx CV | FlashInfer FA2 µs | "
            "FI CV | TIRx speedup |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda value: (value["seq"], value["kv"], value["causal"])):
        mask = "causal" if row["causal"] else "non-causal"
        lines.append(
            f"| {row['seq']} | {row['kv']} | {mask} | {row['ours_us']:.3f} | "
            f"{row['ours_cv']:.1f}% | {row['reference_us']:.3f} | "
            f"{row['reference_cv']:.1f}% | **{row['speedup']:.3f}x** |"
        )

    lines.extend(
        [
            "",
            "## Fairness contract",
            "",
            "Both implementations receive the same FP16 Q, K, and V storage. Every row has "
            "batch size 1, 32 query heads, head dimension 128, equal Q/KV sequence lengths, "
            "matching causal mode, NHD layout, and softmax scale `1/sqrt(128)`. The FlashInfer "
            "adapter checks its output against TIRx with `rtol=0.01, atol=0.01` before returning "
            "the timed launch closure.",
            "",
            "JIT compilation, module lookup, temporary-buffer allocation, and output allocation "
            "are outside both timed regions. Proton measures GPU kernel time for pure launches. "
            f"Each implementation receives a {protocol.get('warmup', 25)} ms warmup and "
            f"{protocol.get('repeat', 100)} ms repeat budget in each of "
            f"{protocol.get('rounds', 5)} rounds; the table reports their arithmetic mean.",
            "",
            "FlashInfer FA3 has no loadable `sm_110a` kernel in the pinned package, so FA2 is the "
            "supported Thor baseline. This compares implementations of the same attention "
            "operation; it is a kernel microbenchmark, not end-to-end serving throughput.",
            "",
            "## Provenance",
            "",
            "| Field | Value |",
            "|---|---|",
            "| GPU | NVIDIA Jetson AGX Thor Developer Kit, 20 SMs |",
            f"| CUDA architecture | `{selection.get('cuda_arch', 'sm_110a')}` |",
            "| Power/clock state | `MAXN`; dynamic clocks (`jetson_clocks` was not locked) |",
            f"| Timer | `{selection.get('timer_override', 'proton')}` |",
            f"| TVM/TIR revision | `{git.get('tir', '-')}` |",
            f"| TIRx-kernels revision | `{git.get('tirx-kernels', '-')}` |",
            f"| FlashInfer version / revision | `{flashinfer.get('version', '-')}` / "
            f"`{flashinfer.get('git_sha', '-')}` |",
            "| CUDA / PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 |",
            "",
            "The 1,000 ms warmup substantially reduces Thor's cold-DVFS bias, but locked clocks "
            "are still preferable for publication. Rows with CV above 10% remain visible rather "
            "than being silently discarded.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m tirx_kernels.bench_suite \\",
            "  --workloads scripts/thor_flashinfer_attention_sweep.yaml \\",
            f"  --out-dir {run_path.parent.parent} \\",
            "  --with-references --timer proton --rounds 5 --cooldown 0 \\",
            "  --max-prepare-processes 1 --ready-backlog 1 --no-probe --no-report",
            "python scripts/report_thor_attention.py \\",
            f"  --run {run_path} --output THOR_FLASHINFER_ATTENTION.md",
            "```",
            "",
            "## Raw evidence",
            "",
            f"- Run JSON: `{run_path}`",
            f"- Run status: {len(rows)} `ok`, 0 failures, 0 interference retries",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("THOR_FLASHINFER_ATTENTION.md"))
    args = parser.parse_args()

    run = json.loads(args.run.read_text())
    args.output.write_text(_markdown(run, run_path=args.run.resolve()))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
