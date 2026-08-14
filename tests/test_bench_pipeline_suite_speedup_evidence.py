# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import hashlib
import json
from pathlib import Path

import pytest


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
