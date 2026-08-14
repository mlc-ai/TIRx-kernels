# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import hashlib
import json
import statistics
from pathlib import Path

import pytest

from tirx_kernels.bench_suite.run import workload_phase_breakdown


def test_tracked_full_suite_supplemental_evidence_is_explicitly_missing():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "bench_pipeline_suite_speedup_evidence.json").read_text())

    assert evidence["measurement_status"] == "missing"
    assert evidence["acceptance_use"] == "supplemental_outside_all_acceptance_criteria"
    assert evidence["ac_ledger_inclusion"] is False
    assert evidence["matrix_workload_count"] == 112
    assert evidence["cache_state"] == "cold_on_both_sides"
    assert evidence["common_runtime_locks"]["TVM_LIBRARY_PATH"] == (
        "/tmp/tvm-nvshmem-ea0950ab-20260814/lib"
    )
    assert "wall_speedup" not in evidence
    assert "wall_reduction_percent" not in evidence
    assert "card_time_ratio" not in evidence

    before = evidence["before"]
    after = evidence["after"]
    assert before["full_sweep_complete"] is False
    assert before["successful_record_count"] == 7
    assert before["partial_command_wall_s"] == pytest.approx(85.087617807)
    assert before["failures"][0]["runtime_diagnostic"] is not None
    assert "illegal instruction" in before["failures"][0]["runtime_diagnostic"]
    assert after["full_sweep_complete"] is False
    assert after["successful_record_count"] == 0
    assert after["partial_command_wall_s"] == pytest.approx(21.092392288)
    assert ".scale_vec::1X" in after["failures"][0]["ptxas_diagnostic"]
    assert after["participating_gpu_indices"] == []
    assert after["cost_model_measurement_status"] == "missing"

    sources = [evidence["sources"]["workloads"]]
    for side in ("before", "after"):
        side_sources = evidence["sources"][side]
        sources.extend(
            [
                side_sources["run"],
                side_sources["outer_timer"],
                side_sources["stdout"],
                side_sources["stderr"],
                *side_sources["failure_logs"],
            ]
        )
    for source in sources:
        path = Path(source["path"])
        if not path.is_absolute():
            path = root / path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_tracked_106_workload_supplement_recomputes_from_raw_artifacts():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "bench_pipeline_suite_speedup_106_evidence.json").read_text())

    assert evidence["measurement_status"] == "measured"
    assert evidence["acceptance_use"] == "supplemental_outside_all_acceptance_criteria"
    assert evidence["ac_ledger_inclusion"] is False
    assert evidence["matrix_workload_count"] == 106
    assert evidence["canonical_default_workload_count"] == 109
    assert evidence["historical_pre_exclusion_workload_count"] == 112
    assert len(evidence["excluded_workloads"]) == 6
    assert {row["scope"] for row in evidence["excluded_workloads"]} == {
        "canonical_default_sweep",
        "this_supplement_only",
    }
    assert "card_time_ratio" not in evidence

    sources = [evidence["sources"]["workloads"]]
    for side in ("before", "after"):
        side_sources = evidence["sources"][side]
        sources.extend(
            [
                side_sources["run"],
                side_sources["outer_timer"],
                side_sources["stdout"],
                side_sources["stderr"],
                *side_sources["failure_logs"],
            ]
        )
    for source in sources:
        path = Path(source["path"])
        if not path.is_absolute():
            path = root / path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]

    before_outer = json.loads(
        (root / evidence["sources"]["before"]["outer_timer"]["path"]).read_text()
    )
    after_outer = json.loads(
        (root / evidence["sources"]["after"]["outer_timer"]["path"]).read_text()
    )
    before_wall = before_outer["command_wall_s"]
    after_wall = after_outer["command_wall_s"]
    assert evidence["before"]["command_wall_s"] == before_wall
    assert evidence["after"]["command_wall_s"] == after_wall
    assert evidence["wall_speedup"] == pytest.approx(before_wall / after_wall)
    assert evidence["wall_saved_s"] == pytest.approx(before_wall - after_wall)
    assert evidence["wall_reduction_percent"] == pytest.approx(
        (1.0 - after_wall / before_wall) * 100.0
    )

    after_run = json.loads((root / evidence["sources"]["after"]["run"]["path"]).read_text())
    raw_records = {(record["kernel"], record["config"]): record for record in after_run["results"]}
    phase_rows = evidence["phase_breakdown"]["after"]["workloads_in_matrix_order"]
    assert len(phase_rows) == 106
    for row in phase_rows:
        record = raw_records[(row["kernel"], row["config"])]
        assert record["status"] == "ok"
        phases = workload_phase_breakdown(record)
        assert phases is not None
        for name, value in phases.items():
            assert row[name] == pytest.approx(value)
        protocol = record["benchmark_protocol"]
        cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
        floor = len(record["impls"]) * protocol["rounds"] * cooldown
        assert row["protocol_cooldown_floor_s"] == pytest.approx(floor)
        assert row["gpu_stage_minus_protocol_floor_s"] == pytest.approx(
            phases["gpu_stage_s"] - floor
        )

    for side in ("before", "after"):
        run = json.loads((root / evidence["sources"][side]["run"]["path"]).read_text())
        raw_records = {(record["kernel"], record["config"]): record for record in run["results"]}
        measurements = evidence[side]["workload_measurements"]
        assert len(measurements) == 106
        for measurement in measurements:
            record = raw_records[(measurement["kernel"], measurement["config"])]
            assert measurement["round_samples_us"] == record["round_samples"]
            for impl, values in measurement["round_samples_us"].items():
                assert measurement["impls_us"][impl] == pytest.approx(statistics.mean(values))

    assert evidence["phase_breakdown"]["before"]["measurement_status"] == (
        "not_available_in_migration_before_artifact"
    )


def test_tracked_gpu_stage_cpu_work_evidence_recomputes_nvfp4_move():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "bench_pipeline_gpu_stage_cpu_work_evidence.json").read_text())
    for source in evidence["sources"].values():
        path = Path(source["path"])
        if not path.is_absolute():
            path = root / path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]

    suite = json.loads((root / evidence["sources"]["suite_run"]["path"]).read_text())
    probe = json.loads((root / evidence["sources"]["nvfp4_probe_run"]["path"]).read_text())
    suite_records = {(record["kernel"], record["config"]): record for record in suite["results"]}
    comparisons = evidence["moved_work"][0]["comparisons"]
    assert len(comparisons) == 3
    for comparison in comparisons:
        identity = (comparison["kernel"], comparison["config"])
        original = suite_records[identity]
        optimized = next(
            record
            for record in probe["results"]
            if (record["kernel"], record["config"]) == identity
        )
        original_phases = workload_phase_breakdown(original)
        optimized_phases = workload_phase_breakdown(optimized)
        assert original_phases is not None and optimized_phases is not None
        assert comparison["cpu_prepare_added_s"] == pytest.approx(
            optimized_phases["cpu_prepare_s"] - original_phases["cpu_prepare_s"]
        )
        assert comparison["gpu_stage_saved_s"] == pytest.approx(
            original_phases["gpu_stage_s"] - optimized_phases["gpu_stage_s"]
        )
        assert comparison["retry_in_place"] is optimized["retry_in_place"]

    import_results = {row["probe"]: row for row in evidence["prepare_import_guard"]["results"]}
    for name in ("flashinfer_fp4_jit", "flashinfer_selective_state", "flashinfer_trtllm_decode"):
        assert import_results[name]["guard_result"] == "rejected"
        assert import_results[name]["cuda_initialized_before"] is False
        assert import_results[name]["cuda_initialized_after"] is True
    for name in ("flashkda_peer", "deepgemm_mega"):
        assert import_results[name]["guard_result"] == "passed"
        assert import_results[name]["cuda_initialized_after"] is False
