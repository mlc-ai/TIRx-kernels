#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Validate raw before/after artifacts and build portable AC-10 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import yaml

DEFAULT_ROUNDS = 5
DEFAULT_COOLDOWN_S = 1.0
MULTI_GPU_EXEMPTION = "exempted_by_human_unmeasured"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required raw artifact is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _source(path: Path, root: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required raw artifact is missing: {path}")
    return {"path": _portable_path(path, root), "sha256": _sha256(path)}


def _load_matrix(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"workload matrix is missing: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    defaults = data.get("defaults") or {}
    workloads = []
    for entry in data.get("workloads") or []:
        workload = {**defaults, **entry}
        if not isinstance(workload.get("kernel"), str) or not workload["kernel"]:
            raise ValueError(f"invalid workload kernel in {path}: {workload}")
        if not isinstance(workload.get("config"), str) or not workload["config"]:
            raise ValueError(f"invalid workload config in {path}: {workload}")
        num_gpus = workload.get("num_gpus", 1)
        if type(num_gpus) is not int or num_gpus != 1:
            raise ValueError(f"AC-10 targeted evidence must be single-GPU: {workload}")
        workload["num_gpus"] = num_gpus
        workload["timer"] = workload.get("timer") or "proton"
        workloads.append(workload)
    if not workloads:
        raise ValueError(f"workload matrix is empty: {path}")
    identities = {(row["kernel"], row["config"]) for row in workloads}
    if len(identities) != len(workloads):
        raise ValueError(f"workload matrix contains duplicate identities: {path}")
    return workloads


def _validate_outer(path: Path) -> tuple[dict, str, str, list[dict]]:
    outer = _load_json(path)
    if outer.get("status") != "completed":
        raise ValueError(f"outer timer did not complete: {path}: {outer.get('status')!r}")
    if outer.get("returncode") != 0 or outer.get("wrapper_returncode") != 0:
        raise ValueError(f"outer command failed: {path}")
    for prefix in ("command", "wrapper"):
        started = outer.get(f"{prefix}_started_monotonic_ns")
        finished = outer.get(f"{prefix}_finished_monotonic_ns")
        wall_ns = outer.get(f"{prefix}_wall_ns")
        wall_s = outer.get(f"{prefix}_wall_s")
        if not all(isinstance(value, int) for value in (started, finished, wall_ns)):
            raise ValueError(f"outer timer has incomplete {prefix} monotonic data: {path}")
        if wall_ns != finished - started:
            raise ValueError(f"outer timer has inconsistent {prefix} wall time: {path}")
        if not isinstance(wall_s, (int, float)) or not math.isclose(
            wall_s, wall_ns / 1_000_000_000, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"outer timer has inconsistent {prefix}_wall_s: {path}")

    physical = outer.get("physical_gpu")
    if not isinstance(physical, dict):
        raise ValueError(f"outer timer has no physical-GPU provenance: {path}")
    requested_index = physical.get("requested_index")
    before = physical.get("before")
    after = physical.get("after")
    if not isinstance(requested_index, str) or not isinstance(before, dict) or not isinstance(
        after, dict
    ):
        raise ValueError(f"outer timer has incomplete physical-GPU provenance: {path}")
    before_uuid = before.get("uuid")
    after_uuid = after.get("uuid")
    if not isinstance(before_uuid, str) or not before_uuid.startswith("GPU-"):
        raise ValueError(f"outer timer has invalid preflight UUID: {path}")
    if before_uuid != after_uuid or physical.get("same_uuid_before_after") is not True:
        raise ValueError(f"outer timer physical UUID changed during the command: {path}")
    if before.get("compute_processes"):
        raise ValueError(f"selected GPU had compute processes before launch: {path}")
    if after.get("compute_processes"):
        raise ValueError(f"selected GPU had compute processes after completion: {path}")
    if outer.get("cuda_visible_devices_at_wrapper_start") != requested_index:
        raise ValueError(
            f"outer command was not restricted to physical index {requested_index}: {path}"
        )

    log_sources = []
    for key in ("stdout_log", "stderr_log"):
        declared_path = Path(outer.get(key, ""))
        candidates = [
            declared_path if declared_path.is_absolute() else path.parent / declared_path,
            path.parent / declared_path.name,
        ]
        log_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if log_path is None:
            raise FileNotFoundError(
                f"outer timer references a missing {key}: {declared_path}; "
                f"relocated sibling also missing: {path.parent / declared_path.name}"
            )
        log_sources.append(
            {"kind": key, "path": log_path, "declared_path": str(declared_path)}
        )
    return outer, requested_index, before_uuid, log_sources


def _records_by_identity(run: dict, path: Path) -> dict[tuple[str, str], dict]:
    records = run.get("results")
    if not isinstance(records, list):
        raise ValueError(f"run JSON has no result list: {path}")
    by_identity = {}
    for record in records:
        identity = (record.get("kernel"), record.get("config") or record.get("label"))
        if identity in by_identity:
            raise ValueError(f"run JSON has duplicate result {identity}: {path}")
        by_identity[identity] = record
    return by_identity


def _validate_record(record: dict, workload: dict, path: Path, gpu_index: str) -> dict:
    identity = f"{workload['kernel']}/{workload['config']}"
    if record.get("status") != "ok":
        raise ValueError(f"{identity} is not an ok measurement in {path}")
    if record.get("errors"):
        raise ValueError(f"{identity} contains benchmark errors in {path}")
    if record.get("timer") != workload["timer"]:
        raise ValueError(f"{identity} used the wrong timer in {path}")
    if record.get("num_gpus") != 1 or record.get("gpus") != [gpu_index]:
        raise ValueError(f"{identity} did not use exactly physical GPU {gpu_index} in {path}")

    protocol = record.get("benchmark_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{identity} has no benchmark protocol in {path}")
    protocol_cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
    if protocol.get("rounds") != DEFAULT_ROUNDS or not isinstance(
        protocol_cooldown, (int, float)
    ) or isinstance(protocol_cooldown, bool) or not math.isclose(
        protocol_cooldown, DEFAULT_COOLDOWN_S, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{identity} did not use the default 5/1.0 protocol in {path}")
    if protocol.get("round_aggregate") != "mean":
        raise ValueError(f"{identity} did not use mean round aggregation in {path}")
    aggregated = record.get("aggregated")
    if aggregated != {"rounds": DEFAULT_ROUNDS, "method": "mean"}:
        raise ValueError(f"{identity} has inconsistent aggregate metadata in {path}")

    impls = record.get("impls")
    samples = record.get("round_samples")
    if not isinstance(impls, dict) or not impls or not isinstance(samples, dict):
        raise ValueError(f"{identity} has incomplete implementation samples in {path}")
    if set(impls) != set(samples):
        raise ValueError(f"{identity} implementation/sample keys differ in {path}")
    implementation_order = list(impls)
    if "order" in protocol:
        if protocol["order"] != implementation_order:
            raise ValueError(
                f"{identity} implementation order differs from result order in {path}"
            )
    else:
        round_orders = protocol.get("round_orders")
        if (
            not isinstance(round_orders, list)
            or len(round_orders) != DEFAULT_ROUNDS
            or any(
                not isinstance(order, list)
                or len(order) != len(implementation_order)
                or set(order) != set(implementation_order)
                for order in round_orders
            )
        ):
            raise ValueError(
                f"{identity} has invalid per-round implementation orders in {path}"
            )
    for implementation, values in samples.items():
        if not isinstance(values, list) or len(values) != DEFAULT_ROUNDS:
            raise ValueError(f"{identity}/{implementation} lacks five raw samples in {path}")
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError(f"{identity}/{implementation} has non-numeric samples in {path}")
        if not math.isclose(
            impls[implementation], statistics.fmean(values), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"{identity}/{implementation} mean is not reproducible in {path}")
    return {
        "kernel": workload["kernel"],
        "config": workload["config"],
        "timer": workload["timer"],
        "implementation_order": implementation_order,
        "impls_us": impls,
    }


def _validate_run(
    path: Path,
    workloads: list[dict],
    gpu_index: str,
    gpu_uuid: str,
    *,
    pipeline: bool,
) -> tuple[dict, list[dict]]:
    run = _load_json(path)
    by_identity = _records_by_identity(run, path)
    expected = {(row["kernel"], row["config"]) for row in workloads}
    if set(by_identity) != expected:
        raise ValueError(
            f"run result matrix differs from workload input: expected {sorted(expected)}, "
            f"got {sorted(by_identity)}"
        )
    summaries = []
    for workload in workloads:
        identity = (workload["kernel"], workload["config"])
        record = by_identity[identity]
        summary = _validate_record(record, workload, path, gpu_index)
        if pipeline:
            if record.get("execution_mode") != "pipeline":
                raise ValueError(f"pipeline result lacks execution_mode: {identity}")
            if record.get("physical_gpu_uuids") != [gpu_uuid]:
                raise ValueError(f"pipeline result ran on an unexpected physical GPU: {identity}")
            retry_in_place = record.get("retry_in_place")
            if type(retry_in_place) is not bool:
                raise ValueError(f"pipeline result lacks retry_in_place provenance: {identity}")
            if retry_in_place != (record.get("attempt") > 1):
                raise ValueError(
                    f"pipeline retry provenance contradicts attempt number: {identity}"
                )
            summary["retry_in_place"] = retry_in_place
            summary["attempt"] = record.get("attempt")
        summaries.append(summary)

    if pipeline:
        pipeline_data = run.get("pipeline")
        if not isinstance(pipeline_data, dict):
            raise ValueError(f"pipeline run has no pipeline metadata: {path}")
        measurement = pipeline_data.get("measurement_protocol")
        if not isinstance(measurement, dict) or measurement.get("is_default") is not True:
            raise ValueError(f"pipeline run is not marked as default protocol: {path}")
        multi_gpu = pipeline_data.get("multi_gpu_runtime_validation")
        if not isinstance(multi_gpu, dict) or (
            multi_gpu.get("validation_status") != MULTI_GPU_EXEMPTION
        ):
            raise ValueError(f"pipeline run lost the multi-GPU exemption state: {path}")
        cost = pipeline_data.get("cost_model")
        if not isinstance(cost, dict) or cost.get("measurement_status") != "measured":
            raise ValueError(f"pipeline run has no measured cost model: {path}")
        if cost.get("schema_version") != 3:
            raise ValueError(
                f"pipeline run uses an unsupported cost-model schema: {path}: "
                f"{cost.get('schema_version')!r}"
            )
        if cost.get("complete_timeline_count") != len(workloads):
            raise ValueError(f"pipeline run has incomplete timelines: {path}")
        complete_measurements = cost.get("complete_measurement_count", len(workloads))
        if complete_measurements != len(workloads):
            raise ValueError(f"pipeline run has incomplete measurements: {path}")
        numeric_fields = (
            "observed_critical_s",
            "first_ready_s",
            "ideal_gpu_list_schedule_s",
            "cpu_ready_constrained_gpu_list_schedule_s",
            "ready_constrained_gpu_list_schedule_s",
            "eligibility_constrained_gpu_list_schedule_s",
            "foreign_wait_s",
            "expected_s",
            "unexplained_s",
            "ready_starvation_s",
            "interference_retry_ready_delay_s",
            "interference_retry_count",
            "interference_retry_gpu_ownership_s",
            "interference_retry_gpu_execution_s",
        )
        if not all(isinstance(cost.get(field), (int, float)) for field in numeric_fields):
            raise ValueError(f"pipeline cost model contains missing numeric evidence: {path}")
        for mapping_field in ("gpu_busy_s_by_index", "gpu_execution_s_by_index"):
            mapping = cost.get(mapping_field)
            if not isinstance(mapping, dict) or not all(
                isinstance(value, (int, float)) for value in mapping.values()
            ):
                raise ValueError(
                    f"pipeline cost model contains missing {mapping_field}: {path}"
                )
        if not math.isclose(
            cost["unexplained_s"],
            cost["observed_critical_s"] - cost["expected_s"] - cost["foreign_wait_s"],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"pipeline unexplained time is not reproducible: {path}")
        dispatch = cost.get("dispatch_latency_s")
        if not isinstance(dispatch, dict) or not isinstance(dispatch.get("p95"), (int, float)):
            raise ValueError(f"pipeline dispatch latency is missing: {path}")
    return run, summaries


def _relative_delta_percent(before: float, after: float) -> float:
    if before == 0:
        raise ValueError("cannot compare an implementation mean against zero")
    return (after / before - 1.0) * 100.0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build AC-10 evidence only from complete tracked raw artifacts"
    )
    parser.add_argument("--timer-family", required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--before-run", type=Path, required=True)
    parser.add_argument("--before-outer", type=Path, required=True)
    parser.add_argument("--after-run", type=Path, required=True)
    parser.add_argument("--after-outer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    try:
        workloads = _load_matrix(args.workloads.resolve())
        timers = {row["timer"] for row in workloads}
        if len(timers) != 1 or args.timer_family not in timers:
            raise ValueError(
                f"--timer-family {args.timer_family!r} does not match workload timers {timers}"
            )

        before_outer, before_index, before_uuid, before_logs = _validate_outer(
            args.before_outer.resolve()
        )
        after_outer, after_index, after_uuid, after_logs = _validate_outer(
            args.after_outer.resolve()
        )
        if before_index != after_index or before_uuid != after_uuid:
            raise ValueError(
                "before/after did not run on the same verified physical GPU: "
                f"before={before_index}/{before_uuid}, after={after_index}/{after_uuid}"
            )

        before_run, before_results = _validate_run(
            args.before_run.resolve(), workloads, before_index, before_uuid, pipeline=False
        )
        after_run, after_results = _validate_run(
            args.after_run.resolve(), workloads, after_index, after_uuid, pipeline=True
        )
        if before_run.get("git", {}).get("tir") != after_run.get("git", {}).get("tir"):
            raise ValueError("before/after use different TIR commits")
        if before_run.get("baselines") != after_run.get("baselines"):
            raise ValueError("before/after use different baseline dependency provenance")

        implementation_deltas = []
        for before_result, after_result in zip(before_results, after_results, strict=True):
            if before_result["implementation_order"] != after_result["implementation_order"]:
                raise ValueError(
                    "before/after implementation order differs for "
                    f"{before_result['kernel']}/{before_result['config']}"
                )
            for implementation in before_result["implementation_order"]:
                before_us = before_result["impls_us"][implementation]
                after_us = after_result["impls_us"][implementation]
                implementation_deltas.append(
                    {
                        "kernel": before_result["kernel"],
                        "config": before_result["config"],
                        "implementation": implementation,
                        "before_us": before_us,
                        "after_us": after_us,
                        "delta_percent": _relative_delta_percent(before_us, after_us),
                    }
                )

        before_wall = before_outer["command_wall_s"]
        after_wall = after_outer["command_wall_s"]
        if before_wall <= 0 or after_wall <= 0:
            raise ValueError("outer command wall times must be positive")
        pipeline_data = after_run["pipeline"]
        if after_wall < pipeline_data["critical_wall_s"]:
            raise ValueError("outer after wall is shorter than the pipeline critical wall")

        sources = {
            "workloads": _source(args.workloads.resolve(), root),
            "before": {
                "run": _source(args.before_run.resolve(), root),
                "outer_timer": _source(args.before_outer.resolve(), root),
                "logs": [
                    {
                        "kind": log["kind"],
                        "declared_path": log["declared_path"],
                        **_source(log["path"], root),
                    }
                    for log in before_logs
                ],
            },
            "after": {
                "run": _source(args.after_run.resolve(), root),
                "outer_timer": _source(args.after_outer.resolve(), root),
                "logs": [
                    {
                        "kind": log["kind"],
                        "declared_path": log["declared_path"],
                        **_source(log["path"], root),
                    }
                    for log in after_logs
                ],
            },
        }
        cost = pipeline_data["cost_model"]
        unexplained_bound = max(0.5, 0.05 * cost["observed_critical_s"])
        payload = {
            "schema_version": 1,
            "measurement_status": "measured",
            "acceptance_use": "ac_10_performance_evidence",
            "timer_family": args.timer_family,
            "matrix": [f"{row['kernel']}/{row['config']}" for row in workloads],
            "fixed_conditions": {
                "physical_gpu_index": int(before_index),
                "physical_gpu_uuid": before_uuid,
                "rounds": DEFAULT_ROUNDS,
                "cooldown_s": DEFAULT_COOLDOWN_S,
                "multi_gpu_runtime_validation": MULTI_GPU_EXEMPTION,
            },
            "sources": sources,
            "before": {
                "git": before_run.get("git"),
                "kernel_tree": before_run.get("kernel_tree"),
                "outer_wall_s": before_wall,
                "results": before_results,
            },
            "after": {
                "git": after_run.get("git"),
                "kernel_tree": after_run.get("kernel_tree"),
                "outer_wall_s": after_wall,
                "results": after_results,
                "pipeline_cost_model": cost,
                "critical_wall_s": pipeline_data["critical_wall_s"],
                "final_reap_tail_s": pipeline_data["final_reap_tail_s"],
                "interference_retry_count": pipeline_data["interference_retry_count"],
            },
            "derived": {
                "wall_speedup": before_wall / after_wall,
                "wall_reduction_percent": (1.0 - after_wall / before_wall) * 100.0,
                "implementation_deltas": implementation_deltas,
                "pipeline_overlap_only": True,
                "default_coverage_reduction_included": False,
                "unexplained_bound_s": unexplained_bound,
                "acceptance_checks": {
                    "unexplained_within_bound": cost["unexplained_s"] <= unexplained_bound,
                    "dispatch_p95_below_100ms": cost["dispatch_latency_s"]["p95"] < 0.1,
                    "ready_starvation_absent": cost["ready_starvation_s"] == 0.0,
                },
            },
        }
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        print(f"AC-10 evidence rejected: {error}", file=sys.stderr)
        return 2

    _write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
