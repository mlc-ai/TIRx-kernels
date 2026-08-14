#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Build reviewable evidence for the AC-external full-suite timing supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tirx_kernels.bench_suite.run import workload_phase_breakdown  # noqa: E402

DEFAULT_ROUNDS = 5
DEFAULT_COOLDOWN_S = 1.0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _source(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(root))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_matrix(path: Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    rows = data.get("workloads") if isinstance(data, dict) else None
    supplement = data.get("supplement") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"supplemental matrix must contain workloads: {path}")
    if not isinstance(supplement, dict):
        raise ValueError(f"supplemental matrix lacks selection metadata: {path}")

    matrix = []
    for row in rows:
        if not isinstance(row, dict) or row.get("num_gpus", 1) != 1:
            raise ValueError(f"supplemental matrix must be single-GPU: {row!r}")
        identity = (row.get("kernel"), row.get("config"))
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError(f"invalid workload identity: {row!r}")
        matrix.append(identity)
    if len(set(matrix)) != len(matrix):
        raise ValueError("supplemental matrix contains duplicate workloads")

    exclusions = supplement.get("excluded_workloads")
    if not isinstance(exclusions, list) or not exclusions:
        raise ValueError("supplemental matrix lacks excluded-workload provenance")
    excluded_identities = []
    for exclusion in exclusions:
        if not isinstance(exclusion, dict):
            raise ValueError(f"invalid exclusion: {exclusion!r}")
        identity = (exclusion.get("kernel"), exclusion.get("config"))
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError(f"invalid exclusion identity: {exclusion!r}")
        if not exclusion.get("scope") or not exclusion.get("reason"):
            raise ValueError(f"incomplete exclusion provenance: {exclusion!r}")
        excluded_identities.append(identity)
    if len(set(excluded_identities)) != len(excluded_identities):
        raise ValueError("supplemental matrix contains duplicate exclusions")
    if set(matrix) & set(excluded_identities):
        raise ValueError("excluded workload is still present in supplemental matrix")

    historical = supplement.get("historical_pre_exclusion_workload_count")
    canonical = supplement.get("canonical_default_workload_count")
    temporary_count = sum(exclusion["scope"] == "this_supplement_only" for exclusion in exclusions)
    if historical != len(matrix) + len(exclusions):
        raise ValueError("historical workload count does not match matrix plus exclusions")
    if canonical != len(matrix) + temporary_count:
        raise ValueError("canonical default count does not match supplemental-only exclusions")
    return matrix, supplement


def _validate_cold_cache(outer: dict[str, Any], path: Path) -> dict[str, Any]:
    roots = outer.get("cold_cache_roots")
    before = roots.get("before") if isinstance(roots, dict) else None
    if not isinstance(before, list) or not before:
        raise ValueError(f"outer artifact has no cold-cache preflight: {path}")
    if any(root.get("entry_count") != 0 for root in before):
        raise ValueError(f"outer artifact did not start from cold cache roots: {path}")
    return {
        "state_at_command_start": "cold",
        "root_count": len(before),
        "roots": before,
        "environment": outer.get("cache_environment"),
    }


def _monitor_summary(outer: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    monitor = outer.get("all_gpu_monitor")
    samples = monitor.get("samples") if isinstance(monitor, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"outer artifact has no all-GPU occupancy timeline: {path}")
    errors = monitor.get("errors")
    if errors:
        raise ValueError(f"outer GPU monitor contains sampling errors: {path}: {errors}")

    uuid_by_index: dict[str, str] = {}
    for sample in samples:
        for gpu in sample.get("gpus") or ():
            index = str(gpu["index"])
            uuid = gpu["uuid"]
            previous = uuid_by_index.setdefault(index, uuid)
            if previous != uuid:
                raise ValueError(f"physical GPU identity changed in {path}: {index}")

    def apparently_idle(sample: dict[str, Any]) -> list[str]:
        return sorted(
            (
                str(gpu["index"])
                for gpu in sample["gpus"]
                if gpu["utilization_gpu_percent"] == 0.0 and not gpu["compute_processes"]
            ),
            key=int,
        )

    return (
        {
            "schema_version": monitor.get("schema_version"),
            "interval_s": monitor.get("interval_s"),
            "sample_count": len(samples),
            "apparently_idle_indices_at_start": apparently_idle(samples[0]),
            "apparently_idle_indices_at_end": apparently_idle(samples[-1]),
            "gpu_uuid_by_index": dict(sorted(uuid_by_index.items(), key=lambda item: int(item[0]))),
        },
        uuid_by_index,
    )


def _participating_indices(stdout: str) -> list[str]:
    indices: set[str] = set()
    for match in re.finditer(r"gpus=([0-9,]+) (?:START|GPU_START)", stdout):
        indices.update(match.group(1).split(","))
    return sorted(indices, key=int)


def _protocol(record: dict[str, Any]) -> tuple[int, float]:
    protocol = record.get("benchmark_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            f"workload lacks benchmark protocol: {record.get('kernel')}/{record.get('config')}"
        )
    rounds = protocol.get("rounds")
    cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
    if rounds != DEFAULT_ROUNDS or cooldown != DEFAULT_COOLDOWN_S:
        raise ValueError(
            "workload lost the default 5-round/1.0s protocol: "
            f"{record.get('kernel')}/{record.get('config')}: {protocol!r}"
        )
    return int(rounds), float(cooldown)


def _failure_summary(record: dict[str, Any], log_text: str = "") -> dict[str, Any]:
    error = str(record.get("error") or "unknown failure")
    lines = [line.strip() for line in f"{error}\n{log_text}".splitlines() if line.strip()]
    ptxas = next(
        (line for line in reversed(lines) if "ptxas" in line and re.search(r"\berror\s*:", line)),
        next((line for line in reversed(lines) if "ptxas" in line), None),
    )
    runtime = next((line for line in lines if "illegal instruction" in line.lower()), None)
    return {
        "kernel": record.get("kernel"),
        "config": record.get("config"),
        "error_first_line": lines[0] if lines else "unknown failure",
        "error_last_line": lines[-1] if lines else "unknown failure",
        "ptxas_diagnostic": ptxas,
        "runtime_diagnostic": runtime,
    }


def _measurement_row(record: dict[str, Any]) -> dict[str, Any]:
    rounds, cooldown = _protocol(record)
    impls = record.get("impls")
    samples = record.get("round_samples")
    if not isinstance(impls, dict) or not impls:
        raise ValueError(f"workload has no implementation means: {record!r}")
    if not isinstance(samples, dict) or set(samples) != set(impls):
        raise ValueError(f"workload samples do not match implementation means: {record!r}")
    for impl, values in samples.items():
        if not isinstance(values, list) or len(values) != rounds:
            raise ValueError(f"{impl} does not contain exactly {rounds} round samples")
        if not math.isclose(
            statistics.mean(float(value) for value in values),
            float(impls[impl]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{impl} aggregate is not recomputable from round samples")
    return {
        "kernel": record["kernel"],
        "config": record["config"],
        "gpu_indices": [int(value) for value in record.get("gpus") or [record.get("gpu")]],
        "physical_gpu_uuids": record.get("physical_gpu_uuids"),
        "attempt": record.get("attempt"),
        "retry_in_place": record.get("retry_in_place"),
        "benchmark_protocol": {"rounds": rounds, "cooldown_s": cooldown},
        "impls_us": {name: float(value) for name, value in impls.items()},
        "round_samples_us": {
            name: [float(value) for value in values] for name, values in samples.items()
        },
    }


def _side_summary(
    *,
    run: dict[str, Any],
    outer: dict[str, Any],
    stdout: str,
    matrix: list[tuple[str, str]],
    outer_path: Path,
    failure_logs: dict[tuple[str, str], str],
    eligibility_policy: str,
) -> dict[str, Any]:
    results = run.get("results")
    if not isinstance(results, list):
        raise ValueError("run artifact has no results list")
    identities = [(record.get("kernel"), record.get("config")) for record in results]
    if len(set(identities)) != len(identities) or not set(identities).issubset(set(matrix)):
        raise ValueError("run results are not a unique subset of the supplemental matrix")
    for record in results:
        if record.get("status") == "ok":
            _protocol(record)

    ok_records = [record for record in results if record.get("status") == "ok"]
    full_sweep_complete = (
        outer.get("returncode") == 0
        and set(identities) == set(matrix)
        and len(ok_records) == len(matrix)
    )
    monitor, uuid_by_index = _monitor_summary(outer, outer_path)
    participating = _participating_indices(stdout)
    pipeline = run.get("pipeline") if isinstance(run.get("pipeline"), dict) else None
    cost = pipeline.get("cost_model") if pipeline else None
    cost_measured = isinstance(cost, dict) and cost.get("measurement_status") == "measured"
    records_by_identity = {
        (record.get("kernel"), record.get("config")): record for record in ok_records
    }

    summary: dict[str, Any] = {
        "measurement_status": "measured" if full_sweep_complete else "incomplete",
        "full_sweep_complete": full_sweep_complete,
        "requested_workload_count": len(matrix),
        "terminal_record_count": len(results),
        "successful_record_count": len(ok_records),
        "status_counts": dict(Counter(str(record.get("status")) for record in results)),
        "command_wall_s": outer.get("command_wall_s"),
        "command_wall_minutes": outer.get("command_wall_s") / 60.0,
        "returncode": outer.get("returncode"),
        "git": run.get("git"),
        "kernel_tree": run.get("kernel_tree"),
        "runtime_environment": outer.get("runtime_environment"),
        "failures": [
            _failure_summary(
                record, failure_logs.get((record.get("kernel"), record.get("config")), "")
            )
            for record in results
            if record.get("status") == "FAIL"
        ],
        "cache": _validate_cold_cache(outer, outer_path),
        "gpu_occupancy": monitor,
        "gpu_eligibility_policy": eligibility_policy,
        "participating_gpu_indices": participating,
        "participating_gpu_count": len(participating),
        "participating_gpu_uuids": [uuid_by_index[index] for index in participating],
        "workload_measurements": [
            _measurement_row(records_by_identity[identity])
            for identity in matrix
            if identity in records_by_identity
        ],
    }
    if pipeline:
        summary["interference_retry_count"] = pipeline.get("interference_retry_count")
        summary["cost_model_measurement_status"] = (
            cost.get("measurement_status") if isinstance(cost, dict) else "missing"
        )
        if cost_measured:
            summary["gpu_busy_s_by_index"] = cost["gpu_busy_s_by_index"]
            summary["foreign_wait_s"] = cost["foreign_wait_s"]
    else:
        unavailable = {"measurement_status": "not_available_in_migration_before_artifact"}
        summary["interference_retry_count"] = unavailable
        summary["gpu_busy_s_by_index"] = unavailable
        summary["foreign_wait_s"] = unavailable
    return summary


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile without values")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _pipeline_phase_evidence(run: dict[str, Any], matrix: list[tuple[str, str]]) -> dict[str, Any]:
    records = {
        (record.get("kernel"), record.get("config")): record
        for record in run.get("results") or ()
        if record.get("status") == "ok"
    }
    rows = []
    for identity in matrix:
        record = records.get(identity)
        if record is None:
            continue
        phases = workload_phase_breakdown(record)
        if phases is None:
            raise ValueError(f"pipeline workload lacks a complete phase timeline: {identity}")
        rounds, cooldown = _protocol(record)
        impl_count = len(record["impls"])
        protocol_floor = impl_count * rounds * cooldown
        residual = phases["gpu_stage_s"] - protocol_floor
        rows.append(
            {
                "kernel": identity[0],
                "config": identity[1],
                "gpu_indices": [int(value) for value in record.get("gpus") or [record.get("gpu")]],
                "physical_gpu_uuids": record.get("physical_gpu_uuids"),
                "retry_in_place": record.get("retry_in_place"),
                **phases,
                "implementation_count": impl_count,
                "protocol_cooldown_floor_s": protocol_floor,
                "gpu_stage_minus_protocol_floor_s": residual,
                "impls_us": {name: float(value) for name, value in record["impls"].items()},
            }
        )
    residuals = sorted(
        (
            {
                "rank": rank,
                "kernel": row["kernel"],
                "config": row["config"],
                "gpu_stage_s": row["gpu_stage_s"],
                "implementation_count": row["implementation_count"],
                "protocol_cooldown_floor_s": row["protocol_cooldown_floor_s"],
                "gpu_stage_minus_protocol_floor_s": row["gpu_stage_minus_protocol_floor_s"],
            }
            for rank, row in enumerate(
                sorted(rows, key=lambda row: row["gpu_stage_minus_protocol_floor_s"], reverse=True),
                start=1,
            )
        ),
        key=lambda row: row["rank"],
    )
    result = {
        "measurement_status": "measured" if len(rows) == len(matrix) else "missing",
        "workload_count": len(rows),
        "definition": (
            "gpu_stage_minus_protocol_floor_s = measured GPU-stage wall - "
            "implementation_count * rounds * cooldown_s; this is a cooldown-only lower-bound "
            "residual, not a claim that the entire residual is movable CPU work"
        ),
        "workloads_in_matrix_order": rows,
        "residuals_descending": residuals,
    }
    if len(rows) != len(matrix):
        result["missing_reason"] = (
            "phase coverage is incomplete; residual percentiles and maxima are not published"
        )
        return result
    values = [row["gpu_stage_minus_protocol_floor_s"] for row in rows]
    result["residual_summary_s"] = {
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }
    return result


def _format_impls(impls: dict[str, float]) -> str:
    return "; ".join(f"{name}={value:.6f}" for name, value in impls.items())


def _breakdown_markdown(payload: dict[str, Any]) -> str:
    before = payload["before"]
    after = payload["after"]
    phases = payload["phase_breakdown"]["after"]
    lines = [
        "# 106-workload supplemental sweep breakdown",
        "",
        "This is supplemental evidence outside every acceptance criterion.",
        "",
        "## End-to-end result",
        "",
        f"- migration-before: `{before['command_wall_s']:.9f}s` "
        f"(`{before['command_wall_minutes']:.4f} min`)",
        f"- pipeline: `{after['command_wall_s']:.9f}s` (`{after['command_wall_minutes']:.4f} min`)",
        f"- before / after: `{payload['wall_speedup']:.6f}x`",
        f"- wall reduction: `{payload['wall_saved_s']:.9f}s` "
        f"(`{payload['wall_reduction_percent']:.4f}%`)",
        "",
        "Both sides used isolated cold caches. This increases the prepare share and makes the "
        "measured speedup an upper bound for ordinary warm-cache use.",
        "",
        "The pipeline command measured commit 5429283 before the later NVFP4 cuBLASLt "
        "build-only move; no adjusted suite wall is inferred from the targeted follow-up.",
        "",
        "The migration-before runner has no equivalent phase timeline or card-time cost model; "
        "those fields are unavailable, not zero.",
        "",
        "## Pipeline workload phase breakdown",
        "",
        "All durations are wall-clock seconds.",
        "",
        "| workload | startup | CLI | framework import | exact import | config | "
        "specialize/compile | CPU prepare | READY wait | ASSIGN | GPU stage | result | reap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in phases["workloads_in_matrix_order"]:
        identity = f"{row['kernel']}/{row['config']}"
        lines.append(
            f"| `{identity}` | {row['process_startup_s']:.3f} | "
            f"{row['cli_bootstrap_s']:.3f} | {row['framework_import_s']:.3f} | "
            f"{row['exact_import_s']:.3f} | {row['config_resolve_s']:.3f} | "
            f"{row['specialize_generate_compile_s']:.3f} | {row['cpu_prepare_s']:.3f} | "
            f"{row['ready_wait_s']:.3f} | {row['assignment_handoff_s']:.3f} | "
            f"{row['gpu_stage_s']:.3f} | {row['result_handoff_s']:.3f} | "
            f"{row['process_reap_tail_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## GPU-stage residual, descending",
            "",
            "The floor includes only mandatory cooldown: implementations x 5 rounds x 1.0s. "
            "Timer setup, correctness, allocation, loading, warmup/repeat and real GPU execution "
            "remain in the residual, so it is a triage signal rather than movable-work accounting.",
            "",
            "| rank | workload | GPU stage | cooldown floor | residual |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in phases["residuals_descending"]:
        identity = f"{row['kernel']}/{row['config']}"
        lines.append(
            f"| {row['rank']} | `{identity}` | {row['gpu_stage_s']:.3f} | "
            f"{row['protocol_cooldown_floor_s']:.3f} | "
            f"{row['gpu_stage_minus_protocol_floor_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-implementation sweep means",
            "",
            "Values are microseconds and are independently recomputable from the five round "
            "samples embedded in the JSON evidence and its hashed raw run artifacts.",
            "",
            "| workload | migration-before impls (us) | pipeline impls (us) |",
            "|---|---|---|",
        ]
    )
    before_rows = {(row["kernel"], row["config"]): row for row in before["workload_measurements"]}
    after_rows = {(row["kernel"], row["config"]): row for row in after["workload_measurements"]}
    for row in phases["workloads_in_matrix_order"]:
        identity = (row["kernel"], row["config"])
        lines.append(
            f"| `{identity[0]}/{identity[1]}` | "
            f"{_format_impls(before_rows[identity]['impls_us'])} | "
            f"{_format_impls(after_rows[identity]['impls_us'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--before-run", type=Path, required=True)
    parser.add_argument("--before-outer", type=Path, required=True)
    parser.add_argument("--after-run", type=Path, required=True)
    parser.add_argument("--after-outer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--breakdown-output", type=Path)
    args = parser.parse_args()

    root = REPO_ROOT
    matrix, supplement = _load_matrix(args.workloads)
    before_run = _load_json(args.before_run)
    before_outer = _load_json(args.before_outer)
    after_run = _load_json(args.after_run)
    after_outer = _load_json(args.after_outer)
    before_stdout_path = Path(before_outer["stdout_log"])
    before_stderr_path = Path(before_outer["stderr_log"])
    after_stdout_path = Path(after_outer["stdout_log"])
    after_stderr_path = Path(after_outer["stderr_log"])

    def failure_log_sources(
        run_path: Path, run: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
        log_dir = run_path.resolve().parents[1] / "logs"
        sources = []
        texts = {}
        for record in run.get("results") or ():
            if record.get("status") != "FAIL":
                continue
            identity = (record.get("kernel"), record.get("config"))
            matches = sorted(log_dir.glob(f"{identity[0]}__{identity[1]}__a*.log"))
            if len(matches) != 1:
                raise ValueError(f"expected one failure log for {identity}, found {matches}")
            sources.append(_source(matches[0], root))
            texts[identity] = matches[0].read_text()
        return sources, texts

    before_failure_sources, before_failure_logs = failure_log_sources(args.before_run, before_run)
    after_failure_sources, after_failure_logs = failure_log_sources(args.after_run, after_run)

    before = _side_summary(
        run=before_run,
        outer=before_outer,
        stdout=before_stdout_path.read_text(),
        matrix=matrix,
        outer_path=args.before_outer,
        failure_logs=before_failure_logs,
        eligibility_policy=(
            "migration-before compute-process/utilization policy; resident VRAM without a "
            "listed compute process was not rejected"
        ),
    )
    after = _side_summary(
        run=after_run,
        outer=after_outer,
        stdout=after_stdout_path.read_text(),
        matrix=matrix,
        outer_path=args.after_outer,
        failure_logs=after_failure_logs,
        eligibility_policy=(
            "pipeline device-level resident-VRAM policy; a card is rejected when memory.used "
            "exceeds the 512 MiB allowance even without a listed compute process"
        ),
    )

    locked_runtime_keys = (
        "TVM_LIBRARY_PATH",
        "NVSHMEM_HOME",
        "TIRX_NCCL_LIBRARY",
        "TIRX_CUBLAS_LIBRARY",
        "TIRX_CUBLASMP_LIBRARY",
        "TIRX_NVSHMEM_LIBRARY",
    )
    before_runtime = before["runtime_environment"]
    after_runtime = after["runtime_environment"]
    if not isinstance(before_runtime, dict) or not isinstance(after_runtime, dict):
        raise ValueError("both outer artifacts must persist their runtime environments")
    runtime_locks = {}
    for key in locked_runtime_keys:
        if not before_runtime.get(key) or before_runtime[key] != after_runtime.get(key):
            raise ValueError(f"before/after runtime lock differs for {key}")
        runtime_locks[key] = before_runtime[key]

    measured = before["full_sweep_complete"] and after["full_sweep_complete"]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "measurement_status": "measured" if measured else "missing",
        "acceptance_use": supplement["acceptance_use"],
        "ac_ledger_inclusion": False,
        "matrix_workload_count": len(matrix),
        "canonical_default_workload_count": supplement["canonical_default_workload_count"],
        "historical_pre_exclusion_workload_count": supplement[
            "historical_pre_exclusion_workload_count"
        ],
        "excluded_workloads": supplement["excluded_workloads"],
        "cache_state": "cold_on_both_sides",
        "common_runtime_locks": runtime_locks,
        "sources": {
            "workloads": _source(args.workloads, root),
            "before": {
                "run": _source(args.before_run, root),
                "outer_timer": _source(args.before_outer, root),
                "stdout": _source(before_stdout_path, root),
                "stderr": _source(before_stderr_path, root),
                "failure_logs": before_failure_sources,
            },
            "after": {
                "run": _source(args.after_run, root),
                "outer_timer": _source(args.after_outer, root),
                "stdout": _source(after_stdout_path, root),
                "stderr": _source(after_stderr_path, root),
                "failure_logs": after_failure_sources,
            },
        },
        "before": before,
        "after": after,
        "phase_breakdown": {
            "before": {
                "measurement_status": "not_available_in_migration_before_artifact",
                "missing_reason": (
                    "the a91a1b7 one-stage runner did not persist equivalent phase timestamps"
                ),
            },
            "after": _pipeline_phase_evidence(after_run, matrix),
        },
        "attribution": {
            "pipeline_overlap_included": True,
            "multi_gpu_worker_parallelism_is_migration_gain": False,
            "coverage_reduction_included": False,
            "available_gpu_sets_may_differ": True,
            "gpu_eligibility_policies_differ": True,
            "cold_cache_speedup_is_upper_bound_for_warm_cache_use": True,
            "post_measurement_nvfp4_cublaslt_prepare_move_included": False,
        },
        "card_time_ratio_unavailable_reason": (
            "the migration-before artifact has no equivalent gpu_busy_s_by_index cost model"
        ),
    }
    if measured:
        before_wall = float(before["command_wall_s"])
        after_wall = float(after["command_wall_s"])
        payload["wall_speedup"] = before_wall / after_wall
        payload["wall_saved_s"] = before_wall - after_wall
        payload["wall_reduction_percent"] = (1.0 - after_wall / before_wall) * 100.0
    else:
        payload["missing_reason"] = (
            "both sides must complete the identical matrix with status=ok for every workload; "
            "no speedup is published from partial or skipped results"
        )

    _write_json(args.output, payload)
    if args.breakdown_output is not None:
        if not measured:
            raise ValueError("cannot publish a breakdown report for incomplete A/B evidence")
        _write_text(args.breakdown_output, _breakdown_markdown(payload))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
