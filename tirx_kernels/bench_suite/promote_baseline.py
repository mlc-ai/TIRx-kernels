#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Promote a complete bench-suite run to the checked-in before baseline."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MERGE_DISABLED_REASON = (
    "incremental baseline merge is disabled for the direct before/after campaign; "
    "promote one complete replacement run"
)


def _result_key(row: dict) -> tuple[str, str]:
    return row["kernel"], row.get("label") or row["config"]


def slim_baseline_row(row: dict) -> dict:
    """Compatibility entry point that preserves the complete row evidence."""
    return copy.deepcopy(row)


def slim_baseline_doc(doc: dict) -> dict:
    """Return a deterministically ordered copy without discarding evidence."""
    out = copy.deepcopy({key: value for key, value in doc.items() if key != "results"})
    out["results"] = sorted(
        [slim_baseline_row(row) for row in doc.get("results") or []], key=_result_key
    )
    return out


def _write_baseline(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(slim_baseline_doc(doc), indent=2) + "\n")


def merge_baseline(run_json: Path, baseline_path: Path) -> int:
    """Reject partial promotion while retaining the historical callable."""
    del run_json, baseline_path
    print(f"[promote] {MERGE_DISABLED_REASON}", file=sys.stderr)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_json",
        type=Path,
        nargs="?",
        help="run JSON to promote (e.g. .bench-suite/runs/18.json)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="disabled: the direct campaign requires a complete replacement baseline",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="normalize ordering without dropping raw evidence (no run_json needed)",
    )
    args = parser.parse_args()

    baseline_path = HERE / "baseline.json"

    if args.slim:
        if not baseline_path.exists():
            parser.error(f"baseline not found: {baseline_path}")
        _write_baseline(baseline_path, json.loads(baseline_path.read_text()))
        print(f"[promote] normalized {baseline_path.relative_to(HERE)} without dropping evidence")
    else:
        if not args.run_json:
            parser.error("run_json required (or pass --slim to normalize the existing baseline)")
        if not args.run_json.exists():
            parser.error(f"run JSON not found: {args.run_json}")
        if args.merge:
            if merge_baseline(args.run_json, baseline_path):
                sys.exit(1)
        else:
            _write_baseline(baseline_path, json.loads(args.run_json.read_text()))
            print(f"[promote] {args.run_json} -> {baseline_path.relative_to(HERE)}")

    subprocess.run(
        [sys.executable, str(HERE / "baseline_view.py")], check=True, stdout=subprocess.DEVNULL
    )
    print(f"[promote] regenerated {(HERE / 'baseline.md').relative_to(HERE)}")


if __name__ == "__main__":
    main()
