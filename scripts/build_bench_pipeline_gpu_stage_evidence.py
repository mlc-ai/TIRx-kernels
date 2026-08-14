#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Build evidence for CPU work found inside the late-bound GPU stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tirx_kernels.bench_suite.run import workload_phase_breakdown  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _residual(record: dict[str, Any]) -> tuple[float, float, float]:
    phases = workload_phase_breakdown(record)
    if phases is None:
        raise ValueError(f"incomplete phase timeline: {record['kernel']}/{record['config']}")
    protocol = record["benchmark_protocol"]
    cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
    floor = len(record["impls"]) * protocol["rounds"] * cooldown
    return phases["gpu_stage_s"], floor, phases["gpu_stage_s"] - floor


def _nvfp4_comparisons(baseline: dict[str, Any], probe: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_records = {(row["kernel"], row["config"]): row for row in baseline["results"]}
    comparisons = []
    for optimized in probe["results"]:
        identity = (optimized["kernel"], optimized["config"])
        if identity[0] != "nvfp4_gemm":
            raise ValueError(f"unexpected targeted probe workload: {identity}")
        original = baseline_records[identity]
        original_phases = workload_phase_breakdown(original)
        optimized_phases = workload_phase_breakdown(optimized)
        if original_phases is None or optimized_phases is None:
            raise ValueError(f"incomplete NVFP4 comparison timeline: {identity}")
        per_impl_change = {
            impl: (float(optimized["impls"][impl]) / float(original["impls"][impl]) - 1.0) * 100.0
            for impl in optimized["impls"]
        }
        retries = len(optimized.get("gpu_attempts") or ()) - 1
        comparisons.append(
            {
                "kernel": identity[0],
                "config": identity[1],
                "baseline_cpu_prepare_s": original_phases["cpu_prepare_s"],
                "optimized_cpu_prepare_s": optimized_phases["cpu_prepare_s"],
                "cpu_prepare_added_s": (
                    optimized_phases["cpu_prepare_s"] - original_phases["cpu_prepare_s"]
                ),
                "baseline_gpu_stage_s": original_phases["gpu_stage_s"],
                "optimized_gpu_stage_s": optimized_phases["gpu_stage_s"],
                "gpu_stage_saved_s": (
                    original_phases["gpu_stage_s"] - optimized_phases["gpu_stage_s"]
                ),
                "attempt": optimized.get("attempt"),
                "retry_in_place": optimized.get("retry_in_place"),
                "interference_retry_count": retries,
                "attribution_status": (
                    "not_attributable_due_to_in_place_retries"
                    if optimized.get("retry_in_place")
                    else "clean_targeted_comparison"
                ),
                "impl_mean_change_percent": per_impl_change,
                "round_sample_counts": {
                    impl: len(values) for impl, values in optimized["round_samples"].items()
                },
            }
        )
    return sorted(comparisons, key=lambda row: row["config"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-run", type=Path, required=True)
    parser.add_argument("--nvfp4-probe-run", type=Path, required=True)
    parser.add_argument("--nvfp4-probe-outer", type=Path, required=True)
    parser.add_argument("--nvfp4-probe-stdout", type=Path, required=True)
    parser.add_argument("--nvfp4-probe-stderr", type=Path, required=True)
    parser.add_argument("--prepare-import-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite = _load_json(args.suite_run)
    nvfp4_probe = _load_json(args.nvfp4_probe_run)
    import_probe = _load_json(args.prepare_import_probe)
    residuals = []
    for record in suite["results"]:
        gpu_stage, floor, residual = _residual(record)
        residuals.append(
            {
                "kernel": record["kernel"],
                "config": record["config"],
                "gpu_stage_s": gpu_stage,
                "protocol_cooldown_floor_s": floor,
                "gpu_stage_minus_protocol_floor_s": residual,
            }
        )
    threshold = _percentile([row["gpu_stage_minus_protocol_floor_s"] for row in residuals], 0.90)
    significant = sorted(
        (row for row in residuals if row["gpu_stage_minus_protocol_floor_s"] >= threshold),
        key=lambda row: row["gpu_stage_minus_protocol_floor_s"],
        reverse=True,
    )
    comparisons = _nvfp4_comparisons(suite, nvfp4_probe)
    clean_comparisons = [
        row for row in comparisons if row["attribution_status"] == "clean_targeted_comparison"
    ]
    payload = {
        "schema_version": 1,
        "measurement_status": "measured",
        "sources": {
            "suite_run": _source(args.suite_run),
            "nvfp4_probe_run": _source(args.nvfp4_probe_run),
            "nvfp4_probe_outer": _source(args.nvfp4_probe_outer),
            "nvfp4_probe_stdout": _source(args.nvfp4_probe_stdout),
            "nvfp4_probe_stderr": _source(args.nvfp4_probe_stderr),
            "prepare_import_probe": _source(args.prepare_import_probe),
        },
        "residual_definition": (
            "measured GPU-stage wall - implementation_count * rounds * cooldown_s; "
            "the residual also contains timer setup, correctness, allocation, loading, "
            "warmup/repeat and real GPU execution"
        ),
        "significant_threshold": {
            "rule": "at_or_above_suite_p90",
            "p90_s": threshold,
            "workload_count": len(significant),
        },
        "significant_workloads": significant,
        "moved_work": [
            {
                "work_class": "nvfp4_cublaslt_pytorch_extension_build",
                "from": "GPU stage reference builder",
                "to": "CPU prepare before READY",
                "gpu_stage_action_after_move": "load the exact prepared shared library",
                "cuda_prepare_guard": "passed in the cold targeted probe",
                "clean_gpu_stage_saved_s": {
                    "min": min(row["gpu_stage_saved_s"] for row in clean_comparisons),
                    "max": max(row["gpu_stage_saved_s"] for row in clean_comparisons),
                    "comparisons": len(clean_comparisons),
                },
                "comparisons": comparisons,
            }
        ],
        "prepare_import_guard": import_probe,
        "not_moved": [
            {
                "work_class": "flashinfer_fp4_jit_and_autotune",
                "reason": (
                    "the reachable FlashInfer FP4 import changes CUDA initialization state; "
                    "tensor-dependent autotuning also requires the assigned device"
                ),
            },
            {
                "work_class": "flashinfer_selective_state_reference",
                "reason": (
                    "the reachable reference import changes CUDA initialization state; case "
                    "allocation and reference warmup/correctness are device work"
                ),
            },
            {
                "work_class": "flashinfer_trtllm_sparse_mla_reference",
                "reason": (
                    "the reachable TRT-LLM decode import changes CUDA initialization state; "
                    "the 128 MiB workspace, KV tensors and tactic probe are device work"
                ),
            },
            {
                "work_class": "flashkda_peer_import_and_provenance",
                "reason": (
                    "prepare-safe but measured at only 24 ms, below the stopping threshold; "
                    "the material residual is device allocation, TMA setup and reference work"
                ),
            },
            {
                "work_class": "deepgemm_mega_import",
                "reason": (
                    "prepare-safe but approximately 0.2s and not process-portable to spawned "
                    "rank workers; process-group, case allocation and barriers remain after claim"
                ),
            },
        ],
        "impl_mean_change_percent_abs_max_for_clean_nvfp4_comparisons": max(
            abs(value)
            for row in clean_comparisons
            for value in row["impl_mean_change_percent"].values()
        ),
        "all_nvfp4_probe_rounds_present": all(
            all(count == 5 for count in row["round_sample_counts"].values()) for row in comparisons
        ),
    }
    _write_json(args.output, payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
