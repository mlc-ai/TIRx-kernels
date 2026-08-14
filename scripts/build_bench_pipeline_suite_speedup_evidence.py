#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Build the AC-external 112-workload suite-speedup evidence artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import yaml

DEFAULT_ROUNDS = 5
DEFAULT_COOLDOWN_S = 1.0
EXPECTED_WORKLOADS = 112


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _source(path: Path, root: Path) -> dict:
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_matrix(path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(path.read_text())
    rows = data.get("workloads") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_WORKLOADS:
        raise ValueError(f"expected exactly {EXPECTED_WORKLOADS} workloads: {path}")
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
    return matrix


def _validate_cold_cache(outer: dict, path: Path) -> dict:
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


def _monitor_summary(outer: dict, path: Path) -> tuple[dict, dict[str, str]]:
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

    def available(sample: dict) -> list[str]:
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
            "available_indices_at_start": available(samples[0]),
            "available_indices_at_end": available(samples[-1]),
            "gpu_uuid_by_index": dict(sorted(uuid_by_index.items(), key=lambda item: int(item[0]))),
        },
        uuid_by_index,
    )


def _participating_indices(stdout: str) -> list[str]:
    indices: set[str] = set()
    for match in re.finditer(r"gpus=([0-9,]+) (?:START|GPU_START)", stdout):
        indices.update(match.group(1).split(","))
    return sorted(indices, key=int)


def _protocol_is_default(record: dict) -> bool:
    protocol = record.get("benchmark_protocol")
    if not isinstance(protocol, dict):
        return False
    cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
    return protocol.get("rounds") == DEFAULT_ROUNDS and cooldown == DEFAULT_COOLDOWN_S


def _failure_summary(record: dict, log_text: str = "") -> dict:
    error = str(record.get("error") or "unknown failure")
    lines = [line.strip() for line in f"{error}\n{log_text}".splitlines() if line.strip()]
    ptxas = next(
        (
            line
            for line in reversed(lines)
            if "ptxas" in line and re.search(r"\berror\s*:", line)
        ),
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


def _side_summary(
    *,
    run: dict,
    outer: dict,
    stdout: str,
    matrix: list[tuple[str, str]],
    outer_path: Path,
    failure_logs: dict[tuple[str, str], str],
) -> dict:
    results = run.get("results")
    if not isinstance(results, list):
        raise ValueError("run artifact has no results list")
    identities = [(record.get("kernel"), record.get("config")) for record in results]
    if len(set(identities)) != len(identities) or not set(identities).issubset(set(matrix)):
        raise ValueError("run results are not a unique subset of the supplemental matrix")
    successful = [record for record in results if record.get("status") in ("ok", "SKIP")]
    if any(record.get("status") == "ok" and not _protocol_is_default(record) for record in results):
        raise ValueError("successful workload lost the default 5-round/1.0s protocol")

    monitor, uuid_by_index = _monitor_summary(outer, outer_path)
    participating = _participating_indices(stdout)
    pipeline = run.get("pipeline") if isinstance(run.get("pipeline"), dict) else None
    cost = pipeline.get("cost_model") if pipeline else None
    cost_measured = isinstance(cost, dict) and cost.get("measurement_status") == "measured"
    full_sweep_complete = (
        outer.get("returncode") == 0
        and len(results) == len(matrix)
        and len(successful) == len(matrix)
    )
    summary = {
        "measurement_status": "measured" if full_sweep_complete else "incomplete",
        "full_sweep_complete": full_sweep_complete,
        "requested_workload_count": len(matrix),
        "terminal_record_count": len(results),
        "successful_record_count": len(successful),
        "status_counts": dict(Counter(str(record.get("status")) for record in results)),
        "partial_command_wall_s": outer.get("command_wall_s"),
        "returncode": outer.get("returncode"),
        "git": run.get("git"),
        "kernel_tree": run.get("kernel_tree"),
        "runtime_environment": outer.get("runtime_environment"),
        "failures": [
            _failure_summary(
                record,
                failure_logs.get((record.get("kernel"), record.get("config")), ""),
            )
            for record in results
            if record.get("status") == "FAIL"
        ],
        "cache": _validate_cold_cache(outer, outer_path),
        "gpu_occupancy": monitor,
        "participating_gpu_indices": participating,
        "participating_gpu_uuids": [uuid_by_index[index] for index in participating],
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
        summary["interference_retry_count"] = {
            "measurement_status": "not_available_in_migration_before_artifact"
        }
        summary["gpu_busy_s_by_index"] = {
            "measurement_status": "not_available_in_migration_before_artifact"
        }
        summary["foreign_wait_s"] = {
            "measurement_status": "not_available_in_migration_before_artifact"
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--before-run", type=Path, required=True)
    parser.add_argument("--before-outer", type=Path, required=True)
    parser.add_argument("--after-run", type=Path, required=True)
    parser.add_argument("--after-outer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    matrix = _load_matrix(args.workloads)
    before_run = _load_json(args.before_run)
    before_outer = _load_json(args.before_outer)
    after_run = _load_json(args.after_run)
    after_outer = _load_json(args.after_outer)
    before_stdout_path = Path(before_outer["stdout_log"])
    before_stderr_path = Path(before_outer["stderr_log"])
    after_stdout_path = Path(after_outer["stdout_log"])
    after_stderr_path = Path(after_outer["stderr_log"])

    def failure_log_sources(run_path: Path, run: dict) -> tuple[list[dict], dict]:
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

    before_failure_sources, before_failure_logs = failure_log_sources(
        args.before_run, before_run
    )
    after_failure_sources, after_failure_logs = failure_log_sources(args.after_run, after_run)

    before = _side_summary(
        run=before_run,
        outer=before_outer,
        stdout=before_stdout_path.read_text(),
        matrix=matrix,
        outer_path=args.before_outer,
        failure_logs=before_failure_logs,
    )
    after = _side_summary(
        run=after_run,
        outer=after_outer,
        stdout=after_stdout_path.read_text(),
        matrix=matrix,
        outer_path=args.after_outer,
        failure_logs=after_failure_logs,
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

    payload = {
        "schema_version": 1,
        "measurement_status": (
            "measured"
            if before["full_sweep_complete"] and after["full_sweep_complete"]
            else "missing"
        ),
        "acceptance_use": "supplemental_outside_all_acceptance_criteria",
        "ac_ledger_inclusion": False,
        "matrix_workload_count": len(matrix),
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
        "attribution": {
            "pipeline_overlap_included": True,
            "multi_gpu_worker_parallelism_is_migration_gain": False,
            "default_coverage_reduction_234_to_112_included": False,
            "available_gpu_sets_may_differ": True,
        },
    }
    if payload["measurement_status"] == "measured":
        before_wall = before["partial_command_wall_s"]
        after_wall = after["partial_command_wall_s"]
        payload["wall_speedup"] = before_wall / after_wall
        payload["wall_reduction_percent"] = (1.0 - after_wall / before_wall) * 100.0
        before_busy = before.get("gpu_busy_s_by_index")
        after_busy = after.get("gpu_busy_s_by_index")
        if isinstance(before_busy, dict) and isinstance(after_busy, dict) and all(
            isinstance(value, (int, float))
            for value in [*before_busy.values(), *after_busy.values()]
        ):
            payload["card_time_ratio"] = sum(before_busy.values()) / sum(after_busy.values())
    else:
        payload["missing_reason"] = (
            "neither side completed the identical 112-workload sweep; partial command walls "
            "are diagnostic only and no end-to-end speedup or card-time ratio is published"
        )

    _write_json(args.output, payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
