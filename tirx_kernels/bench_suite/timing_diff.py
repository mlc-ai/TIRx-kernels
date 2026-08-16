#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Absolute timing regression report for candidate-only bench-suite runs.

Compare each current TIRx timing with the pinned timing from the same fixed
runner and workload. ``current / pinned`` above one is a slowdown; below one
is an improvement.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

DEFAULT_TIMING_THRESHOLD = 4.0
DEFAULT_ABSOLUTE_THRESHOLD_US = 5.0
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_LATEST_RUN = REPO_ROOT / ".bench-suite" / "latest.json"
DEFAULT_BASELINE = HERE / "baseline.json"


def index(payload: dict) -> dict[tuple[str, str], tuple[float, str]]:
    """Return the TIRx timing and method keyed by ``(kernel, config)``."""
    if payload.get("schema_version") != 2:
        raise ValueError(f"expected timing schema version 2, got {payload.get('schema_version')!r}")
    out: dict[tuple[str, str], tuple[float, str]] = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        key = (row["kernel"], row.get("label") or row.get("config"))
        if key in out:
            raise ValueError(f"duplicate candidate timing for {key}")
        impls = row.get("impls") or {}
        if set(impls) != {"tirx"}:
            raise ValueError(f"{key} must contain exactly one 'tirx' timing, got {list(impls)}")
        timing = impls.get("tirx")
        if not isinstance(timing, (int, float)) or isinstance(timing, bool) or timing <= 0:
            raise ValueError(f"{key} has invalid TIRx timing {timing!r}")
        timer = row.get("timer")
        if not isinstance(timer, str) or not timer:
            raise ValueError(f"{key} has no timing method")
        out[key] = float(timing), timer
    return out


def build_report(
    baseline_path: Path | str | dict,
    current: dict | Path | str,
    *,
    threshold_pct: float = DEFAULT_TIMING_THRESHOLD,
    threshold_us: float = DEFAULT_ABSOLUTE_THRESHOLD_US,
) -> tuple[str, int]:
    """Build the candidate timing report and return it with its regression count."""
    if threshold_pct < 0 or threshold_us < 0:
        raise ValueError("timing thresholds must be non-negative")
    if isinstance(baseline_path, dict):
        baseline_payload = baseline_path
        baseline_label = (
            f"{baseline_payload.get('timestamp')} ({baseline_payload.get('label') or '-'})"
        )
    else:
        baseline_payload = json.loads(Path(baseline_path).read_text())
        baseline_label = str(baseline_path)
    if isinstance(current, str | Path):
        current_payload = json.loads(Path(current).read_text())
        current_label = str(current)
    else:
        current_payload = current
        current_label = "(in-memory)"

    baseline_runner = baseline_payload.get("runner")
    current_runner = current_payload.get("runner")
    if not isinstance(baseline_runner, dict) or not baseline_runner:
        raise ValueError(
            "pinned baseline has no fixed runner identity; replace it with a fresh run"
        )
    if not isinstance(current_runner, dict) or not current_runner:
        raise ValueError("current run has no fixed runner identity")
    if current_runner != baseline_runner:
        raise ValueError(f"runner mismatch: pinned={baseline_runner!r}, current={current_runner!r}")

    baseline_index = index(baseline_payload)
    current_index = index(current_payload)
    current_status = {
        (row["kernel"], row.get("label") or row.get("config")): row
        for row in current_payload.get("results") or []
    }

    rows: list[tuple[str, str, float, float, float, float, float]] = []
    missing: list[tuple[str, str, str]] = []
    for key, (pinned_us, pinned_timer) in baseline_index.items():
        current_measurement = current_index.get(key)
        if current_measurement is None:
            record = current_status.get(key)
            if record is not None:
                status = record.get("status") or "?"
                errors = (record.get("error") or "").strip().splitlines()
                reason = f"{status}: {errors[0]}" if errors else status
                missing.append((*key, reason))
            else:
                missing.append((*key, "missing from current run"))
            continue
        current_us, current_timer = current_measurement
        if current_timer != pinned_timer:
            raise ValueError(
                f"timing method mismatch for {key}: "
                f"pinned={pinned_timer!r}, current={current_timer!r}"
            )
        timing_ratio = current_us / pinned_us
        delta_us = current_us - pinned_us
        slowdown_pct = (timing_ratio - 1.0) * 100.0
        rows.append((key[0], key[1], pinned_us, current_us, delta_us, timing_ratio, slowdown_pct))

    rows.sort(key=lambda row: row[6])
    regressions = sum(1 for row in rows if row[6] >= threshold_pct and row[4] >= threshold_us)
    improvements = sum(1 for row in rows if row[6] <= -threshold_pct and row[4] <= -threshold_us)

    out = io.StringIO()

    def write(line: str = "") -> None:
        out.write(line + "\n")

    write("# bench-suite timing report")
    write()
    write(f"- Pinned run: `{baseline_label}`")
    write(f"- Current run: `{current_label}`")
    write("- Oracle: TIRx GPU time on the same fixed runner; current/pinned above 1 is slower.")
    write(
        f"- Summary: {len(rows)} comparable candidate measurements; "
        f"{improvements} improved by at least {threshold_pct:g}% and {threshold_us:g}µs, "
        f"{regressions} regressed by at least {threshold_pct:g}% and {threshold_us:g}µs"
        + (f"; {len(missing)} not comparable (see below)" if missing else "")
        + "."
    )
    write()

    if rows:
        write(
            "| kernel | config | pinned (µs) | current (µs) | delta (µs) | "
            "current/pinned | slowdown |"
        )
        write("|---|---|---:|---:|---:|---:|---:|")
        for kernel, config, pinned_us, current_us, delta_us, ratio, slowdown in rows:
            write(
                f"| {kernel} | {config} | {pinned_us:.2f} | {current_us:.2f} | "
                f"{delta_us:+.2f} | {ratio:.3f} | {slowdown:+.1f}% |"
            )
        write()

    if missing:
        write(f"## Not comparable in current run ({len(missing)})")
        write()
        for kernel, config, reason in sorted(missing):
            reason = reason if len(reason) <= 160 else reason[:157] + "..."
            write(f"- `{kernel}/{config}` — {reason}")
        write()

    return out.getvalue(), regressions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("current", nargs="?", default=str(DEFAULT_LATEST_RUN))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--threshold", type=float, default=DEFAULT_TIMING_THRESHOLD)
    parser.add_argument(
        "--absolute-threshold-us", type=float, default=DEFAULT_ABSOLUTE_THRESHOLD_US
    )
    parser.add_argument("--output", "-o", type=Path, default=None)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    report, _ = build_report(
        baseline,
        args.current,
        threshold_pct=args.threshold,
        threshold_us=args.absolute_threshold_us,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"[timing_diff] written: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
