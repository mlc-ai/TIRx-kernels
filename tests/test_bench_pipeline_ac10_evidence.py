# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tirx_kernels.bench_suite.run import _pipeline_cost_model


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


def _record(*, pipeline: bool, timer: str = "proton") -> dict:
    if timer == "megamoe":
        samples = {
            "tirx": [1.0, 2.0, 3.0, 4.0, 5.0],
            "deepgemm": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
        protocol = {
            "rounds": 5,
            "round_cooldown_s": 1.0,
            "round_aggregate": "mean",
            "round_orders": [
                ["tirx", "deepgemm"] if round_index % 2 == 0 else ["deepgemm", "tirx"]
                for round_index in range(5)
            ],
        }
    else:
        samples = {"tir": [1.0, 2.0, 3.0, 4.0, 5.0]}
        protocol = {
            "rounds": 5,
            "cooldown_s": 1.0,
            "round_aggregate": "mean",
            "order": ["tir"],
        }
    record = {
        "kernel": "kernel",
        "config": "config",
        "label": "config",
        "status": "ok",
        "errors": {},
        "timer": timer,
        "num_gpus": 1,
        "gpus": ["1"],
        "attempt": 1,
        "benchmark_protocol": protocol,
        "aggregated": {"rounds": 5, "method": "mean"},
        "impls": {
            implementation: statistics.fmean(values)
            for implementation, values in samples.items()
        },
        "round_samples": samples,
    }
    if pipeline:
        record.update(
            execution_mode="pipeline",
            physical_gpu_uuids=["GPU-test"],
            retry_in_place=False,
        )
    return record


def _write_run(path: Path, *, pipeline: bool, timer: str = "proton") -> None:
    run = {
        "git": {"tir": "tir-sha", "tirx-kernels": "after" if pipeline else "before"},
        "baselines": {"torch": {"version": "test"}},
        "kernel_tree": {"tirx-kernels:tirx_kernels": "tree"},
        "results": [_record(pipeline=pipeline, timer=timer)],
    }
    if pipeline:
        run["pipeline"] = {
            "measurement_protocol": {"is_default": True},
            "multi_gpu_runtime_validation": {
                "validation_status": "exempted_by_human_unmeasured"
            },
            "cost_model": {
                "schema_version": 3,
                "measurement_status": "measured",
                "complete_timeline_count": 1,
                "complete_measurement_count": 1,
                "observed_critical_s": 1.0,
                "first_ready_s": 0.1,
                "ideal_gpu_list_schedule_s": 0.5,
                "cpu_ready_constrained_gpu_list_schedule_s": 0.5,
                "ready_constrained_gpu_list_schedule_s": 0.5,
                "eligibility_constrained_gpu_list_schedule_s": 0.5,
                "gpu_busy_s_by_index": {"1": 0.5},
                "gpu_execution_s_by_index": {"1": 0.4},
                "foreign_wait_s": 0.0,
                "expected_s": 0.6,
                "unexplained_s": 0.4,
                "ready_starvation_s": 0.0,
                "interference_retry_ready_delay_s": 0.0,
                "interference_retry_count": 0,
                "interference_retry_gpu_ownership_s": 0.0,
                "interference_retry_gpu_execution_s": 0.0,
                "dispatch_latency_s": {"p95": 0.01},
            },
            "critical_wall_s": 1.0,
            "final_reap_tail_s": 0.1,
            "interference_retry_count": 0,
        }
    path.write_text(json.dumps(run))


def _command(repo_root: Path, tmp_path: Path, *, timer_family: str = "proton") -> list[str]:
    return [
        sys.executable,
        str(repo_root / "scripts/build_bench_pipeline_ac10_evidence.py"),
        "--timer-family",
        timer_family,
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


def test_tracked_ac10_proton_before_artifact_is_complete():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = repo_root / "bench_pipeline_ac10_artifacts/proton/before"
    outer = json.loads((artifact_root / "outer_timer.json").read_text())
    run = json.loads((artifact_root / "suite/runs/1.json").read_text())
    matrix_data = yaml.safe_load((repo_root / "bench_pipeline_ac10_workloads.yaml").read_text())
    matrix = [(row["kernel"], row["config"]) for row in matrix_data["workloads"]]

    assert outer["status"] == "completed"
    assert outer["returncode"] == outer["wrapper_returncode"] == 0
    assert outer["command_wall_ns"] == (
        outer["command_finished_monotonic_ns"] - outer["command_started_monotonic_ns"]
    )
    assert outer["command_wall_s"] == pytest.approx(
        outer["command_wall_ns"] / 1_000_000_000
    )
    physical = outer["physical_gpu"]
    assert physical["requested_index"] == "1"
    assert physical["same_uuid_before_after"] is True
    assert physical["before"]["uuid"] == physical["after"]["uuid"]
    assert physical["before"]["compute_processes"] == []
    assert physical["after"]["compute_processes"] == []
    assert physical["before"]["memory_used_mib"] <= 512
    assert physical["after"]["memory_used_mib"] <= 512

    assert run["git"]["tirx-kernels"] == "a91a1b76"
    assert [(row["kernel"], row["config"]) for row in run["results"]] == matrix
    sample_count = sum(
        len(values) for row in run["results"] for values in row["round_samples"].values()
    )
    assert sample_count == 30
    for row in run["results"]:
        assert row["status"] == "ok"
        assert row["errors"] == {}
        assert row["gpus"] == ["1"]
        assert row["timer"] == "proton"
        assert row["benchmark_protocol"]["rounds"] == 5
        assert row["benchmark_protocol"]["cooldown_s"] == 1.0
        assert row["aggregated"] == {"rounds": 5, "method": "mean"}
        for implementation, values in row["round_samples"].items():
            assert len(values) == 5
            assert row["impls"][implementation] == pytest.approx(
                statistics.fmean(values), abs=1e-9
            )


def test_tracked_ac10_proton_attempt_one_evidence_matches_every_raw_source():
    repo_root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (repo_root / "bench_pipeline_ac10_artifacts/proton/evidence-attempt-1.json").read_text()
    )

    assert evidence["measurement_status"] == "measured"
    assert evidence["fixed_conditions"]["physical_gpu_uuid"] == (
        "GPU-e8754e6d-624e-e1d0-595a-f9444588960a"
    )
    assert evidence["after"]["interference_retry_count"] == 7
    assert evidence["derived"]["acceptance_checks"] == {
        "dispatch_p95_below_100ms": False,
        "ready_starvation_absent": False,
        "unexplained_within_bound": True,
    }
    assert evidence["derived"]["wall_speedup"] == pytest.approx(
        evidence["before"]["outer_wall_s"] / evidence["after"]["outer_wall_s"]
    )

    sources = [evidence["sources"]["workloads"]]
    for side in ("before", "after"):
        sources.extend(
            [
                evidence["sources"][side]["run"],
                evidence["sources"][side]["outer_timer"],
                *evidence["sources"][side]["logs"],
            ]
        )
    for source in sources:
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        assert source_path.is_file()
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]

    before_run = json.loads(
        (repo_root / evidence["sources"]["before"]["run"]["path"]).read_text()
    )
    after_run = json.loads(
        (repo_root / evidence["sources"]["after"]["run"]["path"]).read_text()
    )
    before_results = {row["config"]: row for row in before_run["results"]}
    after_results = {row["config"]: row for row in after_run["results"]}
    for delta in evidence["derived"]["implementation_deltas"]:
        before_us = before_results[delta["config"]]["impls"][delta["implementation"]]
        after_us = after_results[delta["config"]]["impls"][delta["implementation"]]
        assert delta["before_us"] == before_us
        assert delta["after_us"] == after_us
        assert delta["delta_percent"] == pytest.approx((after_us / before_us - 1.0) * 100.0)

    committed_tree = subprocess.run(
        ["git", "rev-parse", "b5f63b5:tirx_kernels"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert after_run["kernel_tree"]["tirx-kernels:tirx_kernels"] == committed_tree

    pipeline = after_run["pipeline"]
    external_history = [
        (row["timestamp"], tuple(row["occupied_gpu_indices"]))
        for row in pipeline["external_occupancy_timeline"]
    ]
    recomputed_cost = _pipeline_cost_model(
        after_run["results"],
        run_started=pipeline["started"],
        critical_finished=pipeline["critical_finished"],
        known_gpu_indices=("1",),
        external_history=external_history,
    )
    embedded_cost = pipeline["cost_model"]
    assert embedded_cost["ready_starvation_s"] == pytest.approx(8.329377889633179)
    assert recomputed_cost["ready_starvation_s"] == 0.0
    assert recomputed_cost["interference_retry_ready_delay_s"] == pytest.approx(
        4.888591766357422
    )
    assert sum(recomputed_cost["gpu_busy_s_by_index"].values()) == pytest.approx(
        sum(embedded_cost["gpu_busy_s_by_index"].values())
        + sum(
            attempt["gpu_started"] - attempt["assigned"]
            for record in after_run["results"]
            for attempt in record["gpu_attempts"]
        )
    )
    assert recomputed_cost["interference_retry_count"] == 7
    assert recomputed_cost["schema_version"] == 3
    assert recomputed_cost["expected_s"] == pytest.approx(63.68443465232849)
    assert recomputed_cost["unexplained_s"] == pytest.approx(0.6604344844818115)
    assert recomputed_cost["unexplained_s"] == pytest.approx(
        recomputed_cost["observed_critical_s"]
        - recomputed_cost["expected_s"]
        - recomputed_cost["foreign_wait_s"]
    )


def test_tracked_ac10_proton_schema_three_evidence_passes_from_raw_sources():
    repo_root = Path(__file__).resolve().parents[1]
    evidence_path = (
        repo_root / "bench_pipeline_ac10_artifacts/proton/evidence-gpu2-schema3.json"
    )
    evidence = json.loads(evidence_path.read_text())

    assert evidence["measurement_status"] == "measured"
    assert evidence["fixed_conditions"] == {
        "cooldown_s": 1.0,
        "multi_gpu_runtime_validation": "exempted_by_human_unmeasured",
        "physical_gpu_index": 2,
        "physical_gpu_uuid": "GPU-f8a4f1df-8b46-4cbf-3244-a33b90e06aa9",
        "rounds": 5,
    }
    assert evidence["derived"]["acceptance_checks"] == {
        "dispatch_p95_below_100ms": True,
        "ready_starvation_absent": True,
        "unexplained_within_bound": True,
    }
    assert evidence["derived"]["wall_speedup"] == pytest.approx(1.269215059469141)

    sources = [evidence["sources"]["workloads"]]
    for side in ("before", "after"):
        sources.extend(
            [
                evidence["sources"][side]["run"],
                evidence["sources"][side]["outer_timer"],
                *evidence["sources"][side]["logs"],
            ]
        )
    for source in sources:
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        assert source_path.is_file()
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]

    after_run = json.loads(
        (repo_root / evidence["sources"]["after"]["run"]["path"]).read_text()
    )
    committed_tree = subprocess.run(
        ["git", "rev-parse", "0400e58:tirx_kernels"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert after_run["kernel_tree"]["tirx-kernels:tirx_kernels"] == committed_tree
    cost = evidence["after"]["pipeline_cost_model"]
    assert cost["schema_version"] == 3
    assert cost["ready_starvation_s"] == 0.0
    assert cost["interference_retry_count"] == 0
    assert cost["unexplained_s"] == pytest.approx(0.047116994857788086)
    assert sum(cost["gpu_busy_s_by_index"].values()) > sum(
        cost["gpu_execution_s_by_index"].values()
    )


def test_ac10_evidence_builder_recomputes_complete_raw_artifacts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text("workloads:\n  - kernel: kernel\n    config: config\n")
    _write_outer(tmp_path / "before-outer.json", uuid="GPU-test", command_wall_ns=2_000_000_000)
    _write_outer(tmp_path / "after-outer.json", uuid="GPU-test", command_wall_ns=1_500_000_000)
    for name in ("before-outer.json", "after-outer.json"):
        outer_path = tmp_path / name
        outer = json.loads(outer_path.read_text())
        outer["stdout_log"] = f"/unavailable/original/{Path(outer['stdout_log']).name}"
        outer["stderr_log"] = f"/unavailable/original/{Path(outer['stderr_log']).name}"
        outer_path.write_text(json.dumps(outer))
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
    stdout_source = next(
        source
        for source in evidence["sources"]["before"]["logs"]
        if source["kind"] == "stdout_log"
    )
    assert stdout_source["declared_path"].startswith("/unavailable/original/")
    assert Path(stdout_source["path"]).name == "before-outer-stdout.log"


def test_ac10_evidence_builder_accepts_megamoe_alternating_round_protocol(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "workloads.yaml").write_text(
        "defaults:\n  timer: megamoe\n"
        "workloads:\n  - kernel: kernel\n    config: config\n"
    )
    _write_outer(tmp_path / "before-outer.json", uuid="GPU-test", command_wall_ns=2_000_000_000)
    _write_outer(tmp_path / "after-outer.json", uuid="GPU-test", command_wall_ns=1_500_000_000)
    _write_run(tmp_path / "before-run.json", pipeline=False, timer="megamoe")
    _write_run(tmp_path / "after-run.json", pipeline=True, timer="megamoe")

    completed = subprocess.run(
        _command(repo_root, tmp_path, timer_family="megamoe"),
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["timer_family"] == "megamoe"
    assert evidence["before"]["results"][0]["implementation_order"] == [
        "tirx",
        "deepgemm",
    ]


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
