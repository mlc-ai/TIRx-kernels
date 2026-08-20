#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Direct before/after performance gate for the pinned bench-suite sweep.

The verdict uses only arithmetic means of raw ``round_samples`` for the same
TIR/TIRx implementation in the pinned baseline and current run.  A row passes
iff ``after_us / before_us < 1.01``.  External implementations remain useful
diagnostics and are required on both sides, but never affect the verdict.

Usage:
    python ratio_diff.py [current.json] [--baseline PATH] [-o PATH]

Importable as ``build_report(baseline, current)`` for use from ``run.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tirx_kernels.bench_suite.impls import is_our_impl
except ModuleNotFoundError:  # Support direct script execution.
    from impls import is_our_impl

DEFAULT_RATIO_THRESHOLD = 1.0
MAX_AFTER_OVER_BEFORE = 1.01
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_LATEST_RUN = REPO_ROOT / ".bench-suite" / "latest.json"
DEFAULT_BASELINE = HERE / "baseline.json"

_ROW_PROVENANCE_FIELDS = (
    "timer",
    "benchmark_protocol",
    "num_gpus",
    "physical_gpu_uuids",
    "execution_mode",
    "process_model",
)
_PIPELINE_PROVENANCE_FIELDS = ("execution_mode", "process_model", "measurement_protocol")
_INTENTIONALLY_CHANGED_GIT_KEYS = {"tirx-kernels"}
_INTENTIONALLY_CHANGED_TREE_KEYS = {"tirx-kernels:tirx_kernels"}
_REQUIRED_GIT_KEYS = ("tir", "tirx-kernels")
_EPHEMERAL_BENCHMARK_PROTOCOL_FIELDS = {"timing_stream"}


def _load_config_dir() -> list[dict]:
    try:
        from tirx_kernels.bench_suite.run import load_config_dir
    except ModuleNotFoundError:  # Support direct script execution.
        from run import load_config_dir

    return load_config_dir()


def _key(row: dict, *, location: str) -> tuple[str, str]:
    kernel = row.get("kernel")
    config = row.get("label") or row.get("config")
    if not isinstance(kernel, str) or not kernel:
        raise ValueError(f"{location}: invalid kernel")
    if not isinstance(config, str) or not config:
        raise ValueError(f"{location}: invalid label/config")
    label = row.get("label")
    declared_config = row.get("config")
    if label is not None and declared_config is not None and label != declared_config:
        raise ValueError(f"{location}: conflicting label/config")
    return kernel, config


def _expected_keys() -> tuple[set[tuple[str, str]], list[str]]:
    selected: list[tuple[str, str]] = []
    errors: list[str] = []
    try:
        workloads = _load_config_dir()
    except Exception as error:
        return set(), [f"default workload discovery failed: {error}"]
    for index, workload in enumerate(workloads):
        try:
            selected.append(_key(workload, location=f"default workloads[{index}]"))
        except ValueError as error:
            errors.append(str(error))
    duplicates = sorted(key for key, count in Counter(selected).items() if count > 1)
    for kernel, config in duplicates:
        errors.append(f"{kernel}/{config}: duplicate default workload")
    return set(selected), errors


def _selected_after_keys(
    payload: dict, default_keys: set[tuple[str, str]]
) -> tuple[set[tuple[str, str]], str, list[str]]:
    selection = payload.get("selection")
    if selection is None:
        return set(default_keys), "default (legacy payload)", []
    if not isinstance(selection, dict):
        return set(default_keys), "invalid", ["after: selection must be a mapping"]

    mode = selection.get("mode")
    raw_keys = selection.get("keys")
    errors: list[str] = []
    if mode not in {"default", "targeted"}:
        errors.append("after: selection.mode must be 'default' or 'targeted'")
    if not isinstance(raw_keys, list):
        return (
            set(default_keys),
            str(mode or "invalid"),
            [*errors, "after: selection.keys must be a list"],
        )

    parsed: list[tuple[str, str]] = []
    for index, item in enumerate(raw_keys):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
        ):
            errors.append(f"after: selection.keys[{index}] must be a [kernel, config] string pair")
            continue
        parsed.append((item[0], item[1]))

    duplicates = sorted(key for key, count in Counter(parsed).items() if count > 1)
    for kernel, config in duplicates:
        errors.append(f"after: duplicate selected workload {kernel}/{config}")

    selected = set(parsed)
    unknown = sorted(selected - default_keys)
    for kernel, config in unknown:
        errors.append(f"after: selected workload is not in the default roster: {kernel}/{config}")

    if mode == "default":
        missing = default_keys - selected
        if missing or unknown:
            errors.append(
                "after: default selection must match the complete default roster "
                f"(missing {len(missing)}, unknown {len(unknown)})"
            )
        return set(default_keys), "default", errors

    if mode == "targeted":
        if not parsed:
            errors.append("after: targeted selection must not be empty")
        return selected & default_keys, "targeted", errors

    return set(default_keys), "invalid", errors


def _records(payload: Any, label: str) -> tuple[dict[tuple[str, str], list[dict]], list[str]]:
    if not isinstance(payload, dict):
        return {}, [f"{label}: payload must be a JSON object"]
    results = payload.get("results")
    if not isinstance(results, list):
        return {}, [f"{label}: results must be a list"]
    indexed: dict[tuple[str, str], list[dict]] = {}
    errors: list[str] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            errors.append(f"{label}: results[{index}] must be a JSON object")
            continue
        try:
            key = _key(row, location=f"{label}: results[{index}]")
        except ValueError as error:
            errors.append(str(error))
            continue
        indexed.setdefault(key, []).append(row)
    return indexed, errors


def _validate_record_set(
    records: dict[tuple[str, str], list[dict]], expected: set[tuple[str, str]], label: str
) -> list[str]:
    errors: list[str] = []
    for kernel, config in sorted(expected):
        matches = records.get((kernel, config), [])
        if len(matches) != 1:
            detail = "missing" if not matches else f"duplicate ({len(matches)})"
            errors.append(f"{kernel}/{config}: {label} result is {detail}")
    for kernel, config in sorted(set(records) - expected):
        errors.append(f"{kernel}/{config}: {label} result is unexpected")
    return errors


def index(payload: dict) -> dict[tuple[str, str], dict[str, float]]:
    """Compatibility view of unique, successful aggregate rows.

    Duplicate keys are rejected instead of silently overwriting evidence.
    The direct gate itself does not consume this aggregate view.
    """

    records, errors = _records(payload, "payload")
    if errors:
        raise ValueError("; ".join(errors))
    duplicates = [key for key, rows in records.items() if len(rows) != 1]
    if duplicates:
        rendered = ", ".join(f"{kernel}/{config}" for kernel, config in sorted(duplicates))
        raise ValueError(f"duplicate result keys: {rendered}")
    return {
        key: dict(rows[0].get("impls") or {})
        for key, rows in records.items()
        if rows[0].get("status") == "ok"
    }


def _sample_means(row: dict) -> dict[str, float]:
    samples = row.get("round_samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError("round_samples must be a non-empty mapping")
    protocol = row.get("benchmark_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("benchmark_protocol must be a mapping")
    if protocol.get("round_aggregate") != "mean":
        raise ValueError("benchmark_protocol must declare round_aggregate='mean'")
    rounds = protocol.get("rounds")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("benchmark_protocol rounds must be a positive integer")

    sample_order = list(samples)
    if "order" in protocol:
        if protocol["order"] != sample_order:
            raise ValueError("benchmark_protocol order must match round_samples")
    else:
        round_orders = protocol.get("round_orders")
        if (
            not isinstance(round_orders, list)
            or len(round_orders) != rounds
            or any(
                not isinstance(order, list)
                or len(order) != len(sample_order)
                or set(order) != set(sample_order)
                for order in round_orders
            )
        ):
            raise ValueError("benchmark_protocol must declare valid per-round orders")

    means: dict[str, float] = {}
    for impl, values in samples.items():
        if not isinstance(impl, str) or not impl:
            raise ValueError("round_samples implementation names must be non-empty strings")
        if not isinstance(values, list) or len(values) != rounds:
            count = len(values) if isinstance(values, list) else type(values).__name__
            raise ValueError(f"{impl} has {count} round(s), expected {rounds}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError(f"{impl} round samples must all be finite and positive")
        means[impl] = float(statistics.fmean(values))
    return means


def _ours(means: dict[str, float]) -> str:
    names = [name for name in means if is_our_impl(name)]
    if len(names) != 1:
        raise ValueError(f"expected exactly one ours implementation, found {names}")
    return names[0]


def _refs_only(impls: dict[str, float]) -> dict[str, float]:
    return {name: value for name, value in impls.items() if not is_our_impl(name) and value > 0}


def pick_ref(base_impls: dict[str, float]) -> str | None:
    """Pick the fastest external baseline implementation for diagnostics."""

    refs = _refs_only(base_impls)
    return min(refs, key=refs.__getitem__) if refs else None


def _validate_git(payload: dict, label: str) -> list[str]:
    git = payload.get("git")
    if not isinstance(git, dict) or not git:
        return [f"{label}: git provenance must be a non-empty mapping"]
    errors = []
    for name, value in git.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: git[{name!r}] must be null or a non-empty string")
        elif value.strip().endswith("-dirty"):
            errors.append(f"{label}: dirty git provenance for {name}")
    for name in _REQUIRED_GIT_KEYS:
        value = git.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: git[{name!r}] must be a non-empty string")
    return errors


def _compare_mapping(
    before: dict, after: dict, field: str, *, ignored_keys: set[str] = frozenset()
) -> list[str]:
    before_value = before.get(field)
    after_value = after.get(field)
    if (
        not isinstance(before_value, dict)
        or not before_value
        or not isinstance(after_value, dict)
        or not after_value
    ):
        return [f"provenance: {field} must be a non-empty mapping in both runs"]
    if field == "kernel_tree" and any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value.strip()
        for mapping in (before_value, after_value)
        for key, value in mapping.items()
    ):
        return [f"provenance: {field} must contain non-empty string identities"]
    before_cmp = {key: value for key, value in before_value.items() if key not in ignored_keys}
    after_cmp = {key: value for key, value in after_value.items() if key not in ignored_keys}
    if before_cmp != after_cmp:
        return [f"provenance: {field} differs between before and after"]
    return []


def _compare_run_provenance(before: dict, after: dict) -> list[str]:
    errors = [*_validate_git(before, "before"), *_validate_git(after, "after")]
    errors.extend(
        _compare_mapping(before, after, "git", ignored_keys=_INTENTIONALLY_CHANGED_GIT_KEYS)
    )
    errors.extend(
        _compare_mapping(
            before, after, "kernel_tree", ignored_keys=_INTENTIONALLY_CHANGED_TREE_KEYS
        )
    )
    errors.extend(_compare_mapping(before, after, "baselines"))

    before_pipeline = before.get("pipeline")
    after_pipeline = after.get("pipeline")
    if not isinstance(before_pipeline, dict) or not isinstance(after_pipeline, dict):
        errors.append("provenance: pipeline must be a mapping in both runs")
    else:
        for field in _PIPELINE_PROVENANCE_FIELDS:
            if field not in before_pipeline or field not in after_pipeline:
                errors.append(f"provenance: pipeline.{field} must be present in both runs")
            elif before_pipeline[field] != after_pipeline[field]:
                errors.append(f"provenance: pipeline.{field} differs between before and after")
    return errors


def _interfered_keys(payload: dict) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    pipeline = payload.get("pipeline")
    retries = pipeline.get("interference_retries") if isinstance(pipeline, dict) else None
    if not isinstance(retries, list):
        return result
    for index, retry in enumerate(retries):
        if not isinstance(retry, dict):
            continue
        try:
            result.add(_key(retry, location=f"interference_retries[{index}]"))
        except ValueError:
            continue
    return result


def _validate_row(row: dict, row_name: str) -> tuple[dict[str, float] | None, list[str]]:
    errors: list[str] = []
    if row.get("status") != "ok":
        errors.append(f"{row_name}: status={row.get('status')!r}, expected 'ok'")
    if row.get("clean") is False:
        errors.append(f"{row_name}: clean=false")
    if row.get("interfered") or row.get("retry_in_place"):
        errors.append(f"{row_name}: interfered/retried measurement")
    benchmark_errors = row.get("errors")
    if not isinstance(benchmark_errors, dict):
        errors.append(f"{row_name}: errors must be a mapping")
    elif benchmark_errors:
        errors.append(f"{row_name}: benchmark errors are present")
    try:
        means = _sample_means(row)
    except ValueError as error:
        errors.append(f"{row_name}: {error}")
        means = None
    for field in _ROW_PROVENANCE_FIELDS:
        if field not in row:
            errors.append(f"{row_name}: missing provenance field {field}")
    if not isinstance(row.get("timer"), str) or not row.get("timer"):
        errors.append(f"{row_name}: timer must be a non-empty string")
    for field in ("execution_mode", "process_model"):
        if not isinstance(row.get(field), str) or not row[field]:
            errors.append(f"{row_name}: {field} must be a non-empty string")
    num_gpus = row.get("num_gpus")
    gpu_uuids = row.get("physical_gpu_uuids")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus < 1:
        errors.append(f"{row_name}: num_gpus must be a positive integer")
    elif (
        not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != num_gpus
        or any(not isinstance(uuid, str) or not uuid for uuid in gpu_uuids)
    ):
        errors.append(f"{row_name}: physical_gpu_uuids must identify every GPU")
    return means, errors


def _compare_row_provenance(before: dict, after: dict, row_name: str) -> list[str]:
    def comparable(row: dict, field: str):
        value = row.get(field)
        if field != "benchmark_protocol" or not isinstance(value, dict):
            return value
        return {
            key: item
            for key, item in value.items()
            if key not in _EPHEMERAL_BENCHMARK_PROTOCOL_FIELDS
        }

    return [
        f"{row_name}: provenance field {field} differs between before and after"
        for field in _ROW_PROVENANCE_FIELDS
        if comparable(before, field) != comparable(after, field)
    ]


def _load_payload(value: dict | Path | str) -> tuple[dict, str]:
    if isinstance(value, dict):
        return value, f"{value.get('timestamp')} ({value.get('label') or '-'})"
    path = Path(value)
    return json.loads(path.read_text()), str(path)


def build_report(
    baseline_path: Path | str | dict,
    current: dict | Path | str,
    *,
    threshold_pct: float = DEFAULT_RATIO_THRESHOLD,
) -> tuple[str, int]:
    """Return the direct-gate markdown and its total failure count.

    ``threshold_pct`` remains accepted for report-call compatibility and marks
    large external ratio movement.  The verdict threshold is fixed at 1%.
    """

    before_payload, before_label = _load_payload(baseline_path)
    after_payload, after_label = _load_payload(current)
    default_keys, failures = _expected_keys()
    expected, selection_mode, selection_errors = _selected_after_keys(after_payload, default_keys)
    failures.extend(selection_errors)
    before_records, before_errors = _records(before_payload, "before")
    after_records, after_errors = _records(after_payload, "after")
    failures.extend(before_errors)
    failures.extend(after_errors)
    failures.extend(_validate_record_set(before_records, default_keys, "before"))
    failures.extend(_validate_record_set(after_records, expected, "after"))
    failures.extend(_compare_run_provenance(before_payload, after_payload))

    before_interfered = _interfered_keys(before_payload)
    after_interfered = _interfered_keys(after_payload)
    rows: list[dict[str, Any]] = []
    for kernel, config in sorted(expected):
        key = kernel, config
        row_name = f"{kernel}/{config}"
        before_matches = before_records.get(key, [])
        after_matches = after_records.get(key, [])
        if len(before_matches) != 1 or len(after_matches) != 1:
            continue

        before_row, after_row = before_matches[0], after_matches[0]
        if key in before_interfered or key in after_interfered:
            failures.append(f"{row_name}: interference retry recorded")
        before_means, row_errors = _validate_row(before_row, f"{row_name} before")
        failures.extend(row_errors)
        after_means, row_errors = _validate_row(after_row, f"{row_name} after")
        failures.extend(row_errors)
        failures.extend(_compare_row_provenance(before_row, after_row, row_name))
        if before_means is None or after_means is None:
            continue
        try:
            before_ours = _ours(before_means)
            after_ours = _ours(after_means)
        except ValueError as error:
            failures.append(f"{row_name}: {error}")
            continue
        if before_ours != after_ours:
            failures.append(
                f"{row_name}: ours implementation mismatch: {before_ours!r} != {after_ours!r}"
            )
            continue

        before_refs = set(_refs_only(before_means))
        after_refs = set(_refs_only(after_means))
        missing_ref_sides = [
            label for label, refs in (("before", before_refs), ("after", after_refs)) if not refs
        ]
        if missing_ref_sides:
            failures.append(
                f"{row_name}: external reference required in {' and '.join(missing_ref_sides)}"
            )
        elif before_refs != after_refs:
            failures.append(
                f"{row_name}: external reference names differ: "
                f"{sorted(before_refs)} != {sorted(after_refs)}"
            )

        before_us = before_means[before_ours]
        after_us = after_means[after_ours]
        direct_ratio = after_us / before_us
        speedup_pct = (before_us - after_us) / before_us * 100.0
        passed = direct_ratio < MAX_AFTER_OVER_BEFORE
        if not passed:
            failures.append(
                f"{row_name}: after/before={direct_ratio:.6f}, required < {MAX_AFTER_OVER_BEFORE}"
            )

        ref = pick_ref(before_means)
        diagnostic: dict[str, Any] = {}
        if ref is not None and ref in after_means:
            before_ref_us = before_means[ref]
            after_ref_us = after_means[ref]
            before_ref_ratio = before_ref_us / before_us
            after_ref_ratio = after_ref_us / after_us
            diagnostic = {
                "ref": ref,
                "after_ref_us": after_ref_us,
                "ratio": after_ref_ratio,
                "ratio_delta_pct": (after_ref_ratio - before_ref_ratio) / before_ref_ratio * 100.0,
                "ref_drift_pct": (after_ref_us - before_ref_us) / before_ref_us * 100.0,
            }
        rows.append(
            {
                "kernel": kernel,
                "config": config,
                "ours": before_ours,
                "before_us": before_us,
                "after_us": after_us,
                "direct_ratio": direct_ratio,
                "speedup_pct": speedup_pct,
                "passed": passed,
                **diagnostic,
            }
        )

    rows.sort(key=lambda row: row["direct_ratio"], reverse=True)
    out = io.StringIO()

    def write(line: str = "") -> None:
        out.write(line + "\n")

    write("# bench-suite direct before/after report")
    write()
    write(f"- Before: `{before_label}`")
    write(f"- After: `{after_label}`")
    write(
        f"- Scope: {selection_mode}; {len(expected)} after row(s) selected against "
        f"a complete {len(default_keys)}-row before baseline."
    )
    write(
        f"- Gate: arithmetic mean of raw samples for the same ours implementation; "
        f"strict `after/before < {MAX_AFTER_OVER_BEFORE}`."
    )
    write(
        f"- Summary: {len(rows)}/{len(expected)} expected rows evaluated; "
        f"{sum(row['passed'] for row in rows)} direct passes; {len(failures)} failure(s)."
    )
    write("- External reference ratios and drift are diagnostic only.")
    write()

    if rows:
        write(
            "| kernel | config | ours | before (us) | after (us) | after/before | "
            "speedup | direct gate | ref diagnostic |"
        )
        write("|---|---|---|---:|---:|---:|---:|---:|---|")
        for row in rows:
            diagnostic = "-"
            if "ref" in row:
                warning = (
                    " warning"
                    if abs(row["ratio_delta_pct"]) >= float(threshold_pct)
                    or abs(row["ref_drift_pct"]) > 20.0
                    else ""
                )
                diagnostic = (
                    f"{row['ref']}: {row['after_ref_us']:.2f} us, "
                    f"ref/ours={row['ratio']:.3f}, ratio delta={row['ratio_delta_pct']:+.1f}%, "
                    f"ref drift={row['ref_drift_pct']:+.1f}%{warning}"
                )
            write(
                f"| {row['kernel']} | {row['config']} | {row['ours']} | "
                f"{row['before_us']:.2f} | {row['after_us']:.2f} | "
                f"{row['direct_ratio']:.6f} | {row['speedup_pct']:+.2f}% | "
                f"{'PASS' if row['passed'] else 'FAIL'} | {diagnostic} |"
            )
        write()

    if failures:
        write(f"## Failures ({len(failures)})")
        write()
        for failure in failures:
            write(f"- {failure}")
        write()

    return out.getvalue(), len(failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "current",
        nargs="?",
        default=str(DEFAULT_LATEST_RUN),
        help=f"Current run JSON (default: {DEFAULT_LATEST_RUN})",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help=f"Baseline JSON to diff against (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help="External ratio diagnostic threshold in percent; direct gate remains fixed at 1%%",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write report path (default: .bench-suite/reports/<run>/bench.md)",
    )
    args = parser.parse_args()

    report, failures = build_report(args.baseline, args.current, threshold_pct=args.threshold)
    print(report)
    if args.output is not None:
        output_path = args.output
    else:
        current_path = Path(args.current).resolve()
        reports_dir = current_path.parent.parent / "reports" / current_path.stem
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "bench.md"
    output_path.write_text(report)
    print(f"[ratio_diff] written: {output_path}", file=sys.stderr)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
