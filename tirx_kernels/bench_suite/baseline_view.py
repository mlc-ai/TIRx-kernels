#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Render a single bench-suite run JSON as a human-readable markdown summary.

Each workload uses one row per config and one TIRx timing column.

Usage:
    python baseline_view.py [run.json] [-o PATH]

Default input: baseline.json in this directory
Default output: baseline.md (next to the baseline)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _natural_sort_key(value: str) -> tuple[str | int, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def render_markdown(payload: dict, src_name: str) -> str:
    results = payload.get("results") or []
    ok_results = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    lines: list[str] = []
    lines.append(f"# bench-suite baseline view: `{src_name}`")
    lines.append("")
    lines.append(f"- Timestamp: `{payload.get('timestamp')}`")
    lines.append(f"- Label:     `{payload.get('label')}`")
    runner = payload.get("runner") or "unclaimed — full fixed-runner refresh required"
    lines.append(f"- Runner:    `{runner}`")
    lines.append(f"- Git:       `{payload.get('git')}`")
    lines.append(f"- Workloads: {len(ok_results)} ok, {len(failed)} failed")
    lines.append("")
    lines.append("One row per config with the pinned TIRx absolute GPU time.")
    lines.append("")

    by_kernel: dict[str, list[dict]] = {}
    for result in sorted(
        ok_results, key=lambda r: (r["kernel"], r.get("label") or r.get("config"))
    ):
        by_kernel.setdefault(result["kernel"], []).append(result)

    for kernel, kernel_results in by_kernel.items():
        lines.append(f"## {kernel}")
        lines.append("")
        kernel_results.sort(
            key=lambda result: _natural_sort_key(result.get("label") or result.get("config"))
        )
        lines.append("| config | timer | tirx (µs) |")
        lines.append("|---|---|---:|")
        for result in kernel_results:
            impls = result.get("impls") or {}
            timing = f"{impls['tirx']:.4f}" if "tirx" in impls else "—"
            config = result.get("label") or result.get("config")
            lines.append(f"| `{config}` | `{result.get('timer') or '—'}` | {timing} |")
        lines.append("")

    if failed:
        lines.append(f"## Failed ({len(failed)})")
        lines.append("")
        for result in failed:
            first = (result.get("error") or "?").splitlines()[0]
            config = result.get("label") or result.get("config")
            lines.append(f"- `{result['kernel']}/{config}`: {first}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Baseline JSON; default is baseline.json in this directory",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write markdown path (default: baseline.md next to the baseline)",
    )
    args = ap.parse_args()

    if args.input is not None:
        in_path = Path(args.input)
        payload = json.loads(in_path.read_text())
        src_name = in_path.name
        default_out = in_path.with_suffix(".md")
    else:
        baseline_path = here / "baseline.json"
        payload = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        src_name = "baseline.json"
        default_out = here / "baseline.md"
    md = render_markdown(payload, src_name)
    out_path = args.output if args.output else default_out
    out_path.write_text(md)
    print(f"[baseline_view] written: {out_path}", file=sys.stderr)
    print(md[:1200])  # head preview to stdout


if __name__ == "__main__":
    main()
