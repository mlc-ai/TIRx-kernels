#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Promote a bench-suite run JSON to a checked-in baseline and refresh baseline.md.

See README.md in this directory for the full baseline refresh workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

_BASELINE_AGG_KEYS = ("rounds", "method", "max_retry", "min_ok_rounds", "ok_rounds")


def _result_key(row: dict) -> tuple[str, str]:
    return row["kernel"], row.get("label") or row["config"]


def _default_keys() -> set[tuple[str, str]]:
    from tirx_kernels.bench_suite.run import load_config_dir

    return {(row["kernel"], row["config"]) for row in load_config_dir()}


def _validate_full_pin(doc: dict) -> None:
    if doc.get("references_enabled"):
        raise ValueError("diagnostic reference runs cannot be promoted to the TIRx timing pin")
    expected = _default_keys()
    rows = doc.get("results") or []
    failed = [_result_key(row) for row in rows if row.get("status", "ok") != "ok"]
    if failed:
        raise ValueError(f"full timing pin contains failed workloads: {failed[:5]!r}")
    keys = [_result_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("full timing pin contains duplicate workloads")
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"full timing pin must cover the {len(expected)} default workloads exactly; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )


def slim_baseline_row(row: dict) -> dict:
    """Drop per-run metadata; keep only checked-in baseline fields."""
    agg = row.get("aggregated") or {}
    impls = row.get("impls") or {}
    timer = row.get("timer")
    if not isinstance(timer, str) or not timer:
        raise ValueError(f"{_result_key(row)} has no timing method")
    if set(impls) != {"tirx"}:
        raise ValueError(f"{_result_key(row)} must contain exactly one 'tirx' timing")
    timing = impls.get("tirx")
    if not isinstance(timing, (int, float)) or isinstance(timing, bool) or timing <= 0:
        raise ValueError(f"{_result_key(row)} has no positive tirx timing")
    return {
        "kernel": row["kernel"],
        "config": row["config"],
        "label": row.get("label") or row["config"],
        "status": "ok",
        "timer": timer,
        "impls": {"tirx": timing},
        "aggregated": {k: agg[k] for k in _BASELINE_AGG_KEYS if k in agg},
    }


def slim_baseline_doc(doc: dict) -> dict:
    runner = doc.get("runner")
    if not isinstance(runner, dict) or not runner:
        raise ValueError("timing pin requires a fixed runner identity")
    return {
        "schema_version": 2,
        "timestamp": doc.get("timestamp"),
        "label": doc.get("label"),
        "runner": runner,
        "git": doc.get("git") or {},
        "results": sorted(
            [
                slim_baseline_row(row)
                for row in doc.get("results") or []
                if row.get("status", "ok") == "ok"
            ],
            key=_result_key,
        ),
    }


def _write_baseline(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(slim_baseline_doc(doc), indent=2) + "\n")


def merge_baseline(run_json: Path, baseline_path: Path) -> int:
    """Patch ok rows from run_json into baseline_path by (kernel, config).

    If ``baseline_path`` does not exist yet, the slimmed run is written as a
    fresh baseline."""
    run = json.loads(run_json.read_text())
    if run.get("references_enabled"):
        print(
            "[promote] merge: diagnostic reference runs cannot update the TIRx timing pin",
            file=sys.stderr,
        )
        return 1
    if not baseline_path.exists():
        try:
            _validate_full_pin(run)
            _write_baseline(baseline_path, run)
        except ValueError as error:
            print(f"[promote] merge: {error}", file=sys.stderr)
            return 1
        print(f"[promote] merge: no existing baseline, wrote {baseline_path.relative_to(HERE)}")
        return 0
    baseline = json.loads(baseline_path.read_text())
    run_runner = run.get("runner")
    baseline_runner = baseline.get("runner")
    if not isinstance(run_runner, dict) or not run_runner:
        print("[promote] merge: run has no fixed runner identity", file=sys.stderr)
        return 1
    if not isinstance(baseline_runner, dict) or not baseline_runner:
        print(
            "[promote] merge: baseline has no fixed runner identity; replace it with a full run",
            file=sys.stderr,
        )
        return 1
    if run_runner != baseline_runner:
        print(
            f"[promote] merge: runner mismatch: baseline={baseline_runner!r}, run={run_runner!r}",
            file=sys.stderr,
        )
        return 1
    try:
        _validate_full_pin(baseline)
    except ValueError as error:
        print(f"[promote] merge: {error}", file=sys.stderr)
        return 1
    patch = {
        _result_key(r): slim_baseline_row(r)
        for r in run.get("results") or []
        if r.get("status") == "ok"
    }
    if not patch:
        print("[promote] merge: no ok rows in run JSON", file=sys.stderr)
        return 1
    unexpected = sorted(set(patch) - _default_keys())
    if unexpected:
        print(
            f"[promote] merge: run contains non-default workload(s): {unexpected[:5]!r}",
            file=sys.stderr,
        )
        return 1

    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in baseline.get("results") or []:
        key = _result_key(row)
        if key in patch:
            merged.append(patch[key])
            seen.add(key)
        else:
            merged.append(slim_baseline_row(row))
    for key, row in patch.items():
        if key not in seen:
            merged.append(row)

    baseline["results"] = merged
    for key in ("timestamp", "label", "runner", "git"):
        if key in run:
            baseline[key] = run[key]
    _write_baseline(baseline_path, baseline)
    print(
        f"[promote] merged {len(patch)} ok row(s) from {run_json} "
        f"-> {baseline_path.relative_to(HERE)}"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_json",
        type=Path,
        nargs="?",
        help="run JSON to promote (e.g. .bench-suite/runs/18.json)",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="patch ok rows from run_json into the existing baseline.json instead of replacing it",
    )
    ap.add_argument(
        "--slim",
        action="store_true",
        help="strip run metadata from the checked-in baseline JSON (no run_json needed)",
    )
    args = ap.parse_args()

    baseline_path = HERE / "baseline.json"

    if args.slim:
        if not baseline_path.exists():
            ap.error(f"baseline not found: {baseline_path}")
        _write_baseline(baseline_path, json.loads(baseline_path.read_text()))
        print(f"[promote] slimmed {baseline_path.relative_to(HERE)}")
    else:
        if not args.run_json:
            ap.error("run_json required (or pass --slim to clean the existing baseline)")
        if not args.run_json.exists():
            ap.error(f"run JSON not found: {args.run_json}")
        if args.merge:
            rc = merge_baseline(args.run_json, baseline_path)
            if rc:
                sys.exit(1)
        else:
            run = json.loads(args.run_json.read_text())
            if not isinstance(run.get("runner"), dict) or not run["runner"]:
                ap.error("run JSON has no fixed runner identity")
            try:
                _validate_full_pin(run)
            except ValueError as error:
                ap.error(str(error))
            _write_baseline(baseline_path, run)
            print(f"[promote] {args.run_json} -> {baseline_path.relative_to(HERE)}")

    # Always regenerate the human-facing baseline.md so it never drifts from the
    # JSON baseline. This is the whole reason to promote through this helper.
    subprocess.run(
        [sys.executable, str(HERE / "baseline_view.py")], check=True, stdout=subprocess.DEVNULL
    )
    print(f"[promote] regenerated {(HERE / 'baseline.md').relative_to(HERE)}")


if __name__ == "__main__":
    main()
