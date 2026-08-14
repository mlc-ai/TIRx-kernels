# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_outer(path: Path, *, uuid: str, command_wall_ns: int) -> None:
    stdout_log = path.with_name(f"{path.stem}-stdout.log")
    stderr_log = path.with_name(f"{path.stem}-stderr.log")
    stdout_log.write_text("complete\n")
    stderr_log.write_text("")
    wrapper_wall_ns = command_wall_ns + 100_000_000
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "returncode": 0,
                "wrapper_returncode": 0,
                "command_started_monotonic_ns": 100,
                "command_finished_monotonic_ns": 100 + command_wall_ns,
                "command_wall_ns": command_wall_ns,
                "command_wall_s": command_wall_ns / 1_000_000_000,
                "wrapper_started_monotonic_ns": 50,
                "wrapper_finished_monotonic_ns": 50 + wrapper_wall_ns,
                "wrapper_wall_ns": wrapper_wall_ns,
                "wrapper_wall_s": wrapper_wall_ns / 1_000_000_000,
                "cuda_visible_devices_at_wrapper_start": "1",
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "physical_gpu": {
                    "requested_index": "1",
                    "same_uuid_before_after": True,
                    "before": {"uuid": uuid, "compute_processes": []},
                    "after": {"uuid": uuid, "compute_processes": []},
                },
            }
        )
    )


def _record(*, pipeline: bool) -> dict:
    record = {
        "kernel": "kernel",
        "config": "config",
        "label": "config",
        "status": "ok",
        "errors": {},
        "timer": "proton",
        "num_gpus": 1,
        "gpus": ["1"],
        "attempt": 1,
        "benchmark_protocol": {
            "rounds": 5,
            "cooldown_s": 1.0,
            "round_aggregate": "mean",
            "order": ["tir"],
        },
        "aggregated": {"rounds": 5, "method": "mean"},
        "impls": {"tir": 3.0},
        "round_samples": {"tir": [1.0, 2.0, 3.0, 4.0, 5.0]},
    }
    if pipeline:
        record.update(
            execution_mode="pipeline",
            physical_gpu_uuids=["GPU-test"],
            retry_in_place=False,
        )
    return record


def _write_run(path: Path, *, pipeline: bool) -> None:
    run = {
        "git": {"tir": "tir-sha", "tirx-kernels": "after" if pipeline else "before"},
        "baselines": {"torch": {"version": "test"}},
        "kernel_tree": {"tirx-kernels:tirx_kernels": "tree"},
        "results": [_record(pipeline=pipeline)],
    }
    if pipeline:
        run["pipeline"] = {
            "measurement_protocol": {"is_default": True},
            "multi_gpu_runtime_validation": {
                "validation_status": "exempted_by_human_unmeasured"
            },
            "cost_model": {
                "measurement_status": "measured",
                "complete_timeline_count": 1,
                "complete_measurement_count": 1,
                "observed_critical_s": 1.0,
                "first_ready_s": 0.1,
                "ideal_gpu_list_schedule_s": 0.5,
                "ready_constrained_gpu_list_schedule_s": 0.5,
                "eligibility_constrained_gpu_list_schedule_s": 0.5,
                "foreign_wait_s": 0.0,
                "expected_s": 0.6,
                "unexplained_s": 0.4,
                "ready_starvation_s": 0.0,
                "dispatch_latency_s": {"p95": 0.01},
            },
            "critical_wall_s": 1.0,
            "final_reap_tail_s": 0.1,
            "interference_retry_count": 0,
        }
    path.write_text(json.dumps(run))


def _command(repo_root: Path, tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "scripts/build_bench_pipeline_ac10_evidence.py"),
        "--timer-family",
        "proton",
        "--workloads",
        str(tmp_path / "workloads.yaml"),
        "--before-run",
        str(tmp_path / "before-run.json"),
        "--before-outer",
        str(tmp_path / "before-outer.json"),
        "--after-run",
        str(tmp_path / "after-run.json"),
        "--after-outer",
        str(tmp_path / "after-outer.json"),
        "--output",
        str(tmp_path / "evidence.json"),
    ]


def test_ac10_evidence_builder_recomputes_complete_raw_artifacts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text("workloads:\n  - kernel: kernel\n    config: config\n")
    _write_outer(tmp_path / "before-outer.json", uuid="GPU-test", command_wall_ns=2_000_000_000)
    _write_outer(tmp_path / "after-outer.json", uuid="GPU-test", command_wall_ns=1_500_000_000)
    _write_run(tmp_path / "before-run.json", pipeline=False)
    _write_run(tmp_path / "after-run.json", pipeline=True)

    completed = subprocess.run(_command(repo_root, tmp_path), check=False, capture_output=True)

    assert completed.returncode == 0, completed.stderr.decode()
    evidence_path = tmp_path / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["measurement_status"] == "measured"
    assert evidence["fixed_conditions"]["physical_gpu_uuid"] == "GPU-test"
    assert evidence["derived"]["wall_speedup"] == pytest.approx(2.0 / 1.5)
    assert evidence["derived"]["default_coverage_reduction_included"] is False
    assert evidence["derived"]["acceptance_checks"] == {
        "dispatch_p95_below_100ms": True,
        "ready_starvation_absent": True,
        "unexplained_within_bound": True,
    }
    assert evidence["after"]["pipeline_cost_model"]["unexplained_s"] == 0.4
    run_source = evidence["sources"]["after"]["run"]
    assert run_source["sha256"] == hashlib.sha256(
        (tmp_path / "after-run.json").read_bytes()
    ).hexdigest()


def test_ac10_evidence_builder_preserves_measured_acceptance_failures(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "workloads.yaml").write_text(
        "workloads:\n  - kernel: kernel\n    config: config\n"
    )
    _write_outer(tmp_path / "before-outer.json", uuid="GPU-test", command_wall_ns=2_000_000_000)
    _write_outer(tmp_path / "after-outer.json", uuid="GPU-test", command_wall_ns=1_500_000_000)
    _write_run(tmp_path / "before-run.json", pipeline=False)
    after_path = tmp_path / "after-run.json"
    _write_run(after_path, pipeline=True)
    after = json.loads(after_path.read_text())
    cost = after["pipeline"]["cost_model"]
    cost.update(
        observed_critical_s=1.4,
        unexplained_s=0.8,
        ready_starvation_s=0.1,
        dispatch_latency_s={"p95": 0.2},
    )
    after_path.write_text(json.dumps(after))

    completed = subprocess.run(_command(repo_root, tmp_path), check=False, capture_output=True)

    assert completed.returncode == 0, completed.stderr.decode()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["measurement_status"] == "measured"
    assert evidence["derived"]["acceptance_checks"] == {
        "dispatch_p95_below_100ms": False,
        "ready_starvation_absent": False,
        "unexplained_within_bound": False,
    }


@pytest.mark.parametrize("failure", ["missing", "uuid_mismatch"])
def test_ac10_evidence_builder_rejects_incomplete_or_mismatched_sources(
    tmp_path: Path, failure: str
):
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "workloads.yaml").write_text(
        "workloads:\n  - kernel: kernel\n    config: config\n"
    )
    _write_outer(tmp_path / "before-outer.json", uuid="GPU-test", command_wall_ns=2_000_000_000)
    _write_outer(
        tmp_path / "after-outer.json",
        uuid="GPU-other" if failure == "uuid_mismatch" else "GPU-test",
        command_wall_ns=1_500_000_000,
    )
    _write_run(tmp_path / "before-run.json", pipeline=False)
    if failure != "missing":
        _write_run(tmp_path / "after-run.json", pipeline=True)

    completed = subprocess.run(_command(repo_root, tmp_path), check=False, capture_output=True)

    assert completed.returncode == 2
    assert not (tmp_path / "evidence.json").exists()
    assert b"AC-10 evidence rejected:" in completed.stderr
