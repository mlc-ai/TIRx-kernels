import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tirx_kernels.bench_suite import run
from tirx_kernels.bench_suite.baseline_view import render_markdown
from tirx_kernels.bench_suite.ratio_diff import build_report
from tirx_kernels.gemm_comm.allgather_gemm import CONFIGS as ALLGATHER_GEMM_CONFIGS
from tirx_kernels.gemm_comm.gemm_reduce_scatter import CONFIGS as GEMM_RS_CONFIGS
from tirx_kernels.registry import discover_kernels

_RMSNORM_DEFAULT = ("rmsnorm", "hs4096_bs4113")
_FLASH_MLA_TARGETED = ("flash_mla_sparse_fwd", "bench_dqk512_hq64_s4096_kv8192_topk512")


class _ScheduledJobsPool:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def total_visible(self) -> int:
        return self.capacity


def test_bench_suite_standard_sampling_defaults() -> None:
    assert run.DEFAULT_ROUNDS == 5
    assert run.DEFAULT_COOLDOWN_S == 1.0


def test_run_protocol_records_exact_workload_sampling_and_baseline_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "workloads.yaml"
    manifest.write_text(
        "workloads:\n- {kernel: example, config: case, timer: kineto, num_gpus: 1}\n"
    )
    baseline = tmp_path / "before.json"
    baseline.write_text('{"results": []}\n')
    kernel_artifact = tmp_path / "tirx-kernels.whl"
    kernel_artifact.write_bytes(b"exact kernel package")
    monkeypatch.setenv("TVM_CUDA_COMPILE_MODE", "nvcc")
    monkeypatch.setenv("TIRX_NVFP4_FLASHINFER_BACKEND", "cutlass")
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", str(tmp_path / "fa-cache"))
    monkeypatch.setenv("TIRX_BENCH_KERNEL_ARTIFACT", str(kernel_artifact))
    workloads = run.load_workloads(manifest)

    protocol = run.build_run_protocol(
        manifest,
        workloads,
        rounds=5,
        cooldown=1.0,
        threshold=1.0,
        baseline_path=baseline,
        workload_filter=None,
    )

    assert (
        protocol["workload_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert protocol["workload_manifest"]["selected_count"] == 1
    assert protocol["workload_manifest"]["selected_workloads"] == workloads
    assert protocol["sampling"] == {"rounds": 5, "aggregate": "mean", "cooldown_s": 1.0}
    assert protocol["comparison"]["threshold_pct"] == 1.0
    assert (
        protocol["comparison"]["baseline"]["sha256"]
        == hashlib.sha256(baseline.read_bytes()).hexdigest()
    )
    assert protocol["execution"]["cuda_compile_mode"] == "nvcc"
    assert protocol["execution"]["nvfp4_flashinfer_backend"] == "cutlass"
    assert protocol["execution"]["flash_attention_cute_dsl_cache"] == {
        "enabled": True,
        "directory": str(tmp_path / "fa-cache"),
    }
    assert (
        protocol["execution"]["kernel_artifact"]["sha256"]
        == hashlib.sha256(kernel_artifact.read_bytes()).hexdigest()
    )
    assert protocol["scheduling"] == {
        "mode": "automatic",
        "assignment_baseline": None,
        "assignments": None,
        "max_interference_retries": run.DEFAULT_MAX_INTERFERENCE_RETRIES,
    }


def test_run_protocol_records_exact_baseline_gpu_replay_assignments(tmp_path: Path) -> None:
    manifest = tmp_path / "workloads.yaml"
    manifest.write_text("workloads:\n- {kernel: example, config: case, num_gpus: 2}\n")
    baseline = tmp_path / "before.json"
    baseline.write_text('{"results": []}\n')
    workloads = run.load_workloads(manifest)
    assignments = {("example", "case"): ("3", "1")}
    loaded, snapshot_evidence = run.load_baseline_snapshot(baseline)
    assignment_evidence = {
        key: value for key, value in snapshot_evidence.items() if key != "exists"
    }
    original_payload = baseline.read_bytes()

    # Protocol provenance must remain tied to the bytes that produced the
    # assignments, even if the path changes while a long benchmark is running.
    baseline.write_text('{"results": [{"different": true}]}\n')

    protocol = run.build_run_protocol(
        manifest,
        workloads,
        rounds=5,
        cooldown=1.0,
        threshold=1.0,
        baseline_path=baseline,
        workload_filter=None,
        gpu_assignments=assignments,
        gpu_assignment_baseline_path=baseline,
        baseline_evidence=snapshot_evidence,
        gpu_assignment_baseline_evidence=assignment_evidence,
    )

    assert loaded == {"results": []}
    assert protocol["scheduling"] == {
        "mode": "baseline_gpu_replay",
        "assignment_baseline": {
            "path": str(baseline.resolve()),
            "size": len(original_payload),
            "sha256": hashlib.sha256(original_payload).hexdigest(),
        },
        "assignments": [{"kernel": "example", "config": "case", "gpus": ["3", "1"]}],
        "max_interference_retries": run.DEFAULT_MAX_INTERFERENCE_RETRIES,
    }
    assert protocol["comparison"]["baseline"] == snapshot_evidence
    assert (
        protocol["comparison"]["baseline"]["sha256"]
        == protocol["scheduling"]["assignment_baseline"]["sha256"]
    )


def test_baseline_gpu_replay_requires_an_explicit_baseline_cli_argument(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["bench-suite", "--replay-baseline-gpus"])

    with pytest.raises(SystemExit) as error:
        run.main()

    assert error.value.code == 2
    assert "requires an explicit --baseline JSON" in capsys.readouterr().err


def test_baseline_gpu_replay_requires_one_successful_exact_assignment_per_workload() -> None:
    workloads = [
        {"kernel": "single", "config": "a", "num_gpus": 1},
        {"kernel": "multi", "config": "b", "num_gpus": 2},
    ]
    baseline = {
        "results": [
            {"kernel": "single", "label": "a", "status": "ok", "gpus": ["5"]},
            {"kernel": "multi", "config": "b", "status": "ok", "num_gpus": 2, "gpus": ["3", "1"]},
        ]
    }

    assert run.build_baseline_gpu_assignments(baseline, workloads) == {
        ("single", "a"): ("5",),
        ("multi", "b"): ("3", "1"),
    }

    baseline["results"].pop()
    with pytest.raises(ValueError, match="lacks GPU assignments"):
        run.build_baseline_gpu_assignments(baseline, workloads)


def test_finalize_bench_record_uses_all_rounds_arithmetic_mean() -> None:
    row = {"round_samples": {"tir": [1.0, 2.0, 100.0], "reference": [4.0, 5.0, 6.0]}}

    run._finalize_bench_record(row, rounds=3)

    assert row["status"] == "ok"
    assert row["impls"] == {"tir": 103.0 / 3.0, "reference": 5.0}
    assert row["aggregated"] == {"rounds": 3, "method": "mean"}


def test_finalize_bench_record_rejects_baseline_errors() -> None:
    row = {
        "impls": {"tir": 10.0},
        "round_samples": {"tir": [10.0] * 5},
        "errors": {"deepgemm": "setup failed"},
    }

    run._finalize_bench_record(row, rounds=5)

    assert row["status"] == "FAIL"
    assert row["error"] == "baseline error(s): deepgemm: setup failed"


def test_default_workloads_include_manual_tp1_gemm_comm_kineto_profiles() -> None:
    workloads = run.load_config_dir()
    selected = [
        workload
        for workload in workloads
        if workload["kernel"] in {"allgather_gemm", "gemm_reduce_scatter"}
    ]

    configs_by_kernel = {
        "allgather_gemm": {config["label"]: config for config in ALLGATHER_GEMM_CONFIGS},
        "gemm_reduce_scatter": {config["label"]: config for config in GEMM_RS_CONFIGS},
    }
    assert len(selected) == 8
    assert all(workload["config"] in configs_by_kernel[workload["kernel"]] for workload in selected)
    assert all(workload["timer"] == "kineto" for workload in selected)
    assert all(
        workload["num_gpus"]
        == configs_by_kernel[workload["kernel"]][workload["config"]]["world_size"]
        for workload in selected
    )


def test_default_workloads_keep_existing_sweep_and_add_representative_rmsnorm() -> None:
    workloads = run.load_config_dir()
    workload_keys = [(workload["kernel"], workload["config"]) for workload in workloads]

    assert len(workload_keys) == len(set(workload_keys)) == 128
    assert workload_keys.count(_RMSNORM_DEFAULT) == 1
    assert len([key for key in workload_keys if key != _RMSNORM_DEFAULT]) == 127


def test_flash_mla_dispatcher_target_is_available_without_joining_default_sweep() -> None:
    targeted = [
        workload
        for workload in run.load_kernel_configs(_FLASH_MLA_TARGETED[0])
        if workload["config"] == _FLASH_MLA_TARGETED[1]
    ]
    default_keys = {(workload["kernel"], workload["config"]) for workload in run.load_config_dir()}

    assert len(targeted) == 1
    assert targeted[0]["default"] is False
    assert targeted[0]["num_gpus"] == 1
    assert _FLASH_MLA_TARGETED not in default_keys

    from tirx_kernels.flashmla.flash_mla_sparse_fwd import CONFIGS

    assert _FLASH_MLA_TARGETED[1] in {config["label"] for config in CONFIGS}

    covered_registry_names = {kernel for kernel, _ in default_keys}
    covered_registry_names.add(_FLASH_MLA_TARGETED[0])
    assert covered_registry_names == set(discover_kernels(strict=True))


def test_bench_suite_defaults_to_five_round_arithmetic_mean() -> None:
    row = {"round_samples": {"tirx": [1.0, 2.0, 3.0, 4.0, 100.0]}}

    run._finalize_bench_record(row, rounds=run.DEFAULT_ROUNDS)

    assert run.DEFAULT_ROUNDS == 5
    assert row["impls"] == {"tirx": 22.0}
    assert row["aggregated"] == {"rounds": 5, "method": "mean"}
    assert row["status"] == "ok"


def test_default_workloads_do_not_override_standard_timer_budgets() -> None:
    workloads = run.load_config_dir()

    for workload in workloads:
        if workload.get("timer") in {"kineto", "megamoe"}:
            assert "warmup" not in workload
            assert "repeat" not in workload
            continue
        assert "warmup" not in workload
        assert "repeat" not in workload
        assert "timer" not in workload


def test_run_summary_uses_flashinfer_as_rmsnorm_reference(tmp_path: Path) -> None:
    summary_path = run.write_summary(
        tmp_path,
        {
            "timestamp": "now",
            "label": "test",
            "git": {},
            "results": [
                {
                    "kernel": "rmsnorm",
                    "label": _RMSNORM_DEFAULT[1],
                    "status": "ok",
                    "impls": {"tir": 10.0, "flashinfer": 12.0},
                }
            ],
        },
    )

    markdown = summary_path.read_text()
    assert "_baseline impl_: `flashinfer` · _ours_: `tir`" in markdown
    assert "flashinfer/tir" in markdown
    assert "1.200" in markdown


def test_ratio_report_keeps_grouped_tir_schedulers_out_of_references() -> None:
    baseline = {
        "results": [
            {
                "kernel": "grouped_moe",
                "label": "moe_a3b_bs1_all",
                "status": "ok",
                "impls": {
                    "tir_static": 10.0,
                    "tir_dynamic": 11.0,
                    "tir_unfused": 12.0,
                    "sglang_full": 13.0,
                    "flashinfer_full": 14.0,
                },
            }
        ]
    }
    current = {
        "results": [
            {
                "kernel": "grouped_moe",
                "label": "moe_a3b_bs1_all",
                "status": "ok",
                "impls": {
                    "tir_static": 10.0,
                    "tir_dynamic": 11.0,
                    "tir_unfused": 12.0,
                    "sglang_full": 13.0,
                    "flashinfer_full": 14.0,
                },
            }
        ]
    }

    report, regressions = build_report(baseline, current)

    assert regressions == 0
    assert "| grouped_moe | moe_a3b_bs1_all | tir_static | sglang_full |" in report
    assert "| grouped_moe | moe_a3b_bs1_all | tir_dynamic | sglang_full |" in report
    assert "| grouped_moe | moe_a3b_bs1_all | tir_unfused | sglang_full |" in report


def test_ratio_gate_accepts_an_absolute_speedup_despite_reference_drift() -> None:
    baseline = {
        "results": [
            {
                "kernel": "example",
                "label": "shape",
                "status": "ok",
                "impls": {"tir": 100.0, "reference": 100.0},
            }
        ]
    }
    current = {
        "results": [
            {
                "kernel": "example",
                "label": "shape",
                "status": "ok",
                "impls": {"tir": 99.0, "reference": 95.0},
            }
        ]
    }

    report, regressions = build_report(baseline, current, threshold_pct=1.0)

    assert regressions == 0
    assert "1 negative-ratio row(s) passed because ours is absolutely faster" in report


def test_ratio_gate_rejects_a_ratio_regression_without_absolute_speedup() -> None:
    baseline = {
        "results": [
            {
                "kernel": "example",
                "label": "shape",
                "status": "ok",
                "impls": {"tir": 100.0, "reference": 100.0},
            }
        ]
    }
    current = {
        "results": [
            {
                "kernel": "example",
                "label": "shape",
                "status": "ok",
                "impls": {"tir": 101.0, "reference": 98.0},
            }
        ]
    }

    _, regressions = build_report(baseline, current, threshold_pct=1.0)

    assert regressions == 1


def test_baseline_view_renders_grouped_implementations_in_one_row() -> None:
    payload = {
        "timestamp": "now",
        "label": "test",
        "git": {},
        "results": [
            {
                "kernel": "grouped_moe",
                "label": "moe_a3b_bs128_all",
                "status": "ok",
                "impls": {
                    "tir_static": 20.0,
                    "tir_dynamic": 21.0,
                    "tir_unfused": 22.0,
                    "sglang_full": 23.0,
                    "flashinfer_full": 24.0,
                },
            },
            {
                "kernel": "grouped_moe",
                "label": "moe_a3b_bs1_all",
                "status": "ok",
                "impls": {
                    "tir_static": 10.0,
                    "tir_dynamic": 11.0,
                    "tir_unfused": 12.0,
                    "sglang_full": 13.0,
                    "flashinfer_full": 14.0,
                },
            },
        ],
    }

    markdown = render_markdown(payload, "test.json")

    assert (
        "| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | "
        "sglang_full (µs) | flashinfer_full (µs) |" in markdown
    )
    assert "| `moe_a3b_bs1_all` | 10.0000 | 11.0000 | 12.0000 | 13.0000 | 14.0000 |" in markdown
    assert markdown.count("`moe_a3b_bs1_all`") == 1
    assert markdown.index("`moe_a3b_bs1_all`") < markdown.index("`moe_a3b_bs128_all`")


def test_baseline_view_keeps_single_tir_ratio_table() -> None:
    payload = {
        "results": [
            {
                "kernel": "gemm",
                "label": "m1024",
                "status": "ok",
                "impls": {"tir": 10.0, "reference": 12.0},
            }
        ]
    }

    markdown = render_markdown(payload, "test.json")

    assert "| config | ours impl | ours (µs) | ref impl |" in markdown
    assert "| `m1024` | tir | 10.0000 | reference | 12.0000 | 1.200 | — |" in markdown


def test_load_workloads_accepts_multigpu_megamoe(tmp_path: Path) -> None:
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        """
defaults: {}
workloads:
  - kernel: deepgemm_fp8_fp4_mega_moe
    config: t64_m64_h7168_i3072_e384_k6_g6
    timer: megamoe
    num_gpus: 6
"""
    )

    assert run.load_workloads(workloads) == [
        {
            "kernel": "deepgemm_fp8_fp4_mega_moe",
            "config": "t64_m64_h7168_i3072_e384_k6_g6",
            "timer": "megamoe",
            "num_gpus": 6,
        }
    ]


@pytest.mark.parametrize("num_gpus", [0, -1, True, "2"])
def test_load_workloads_rejects_invalid_gpu_count(tmp_path: Path, num_gpus) -> None:
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        f"workloads:\n  - {{kernel: kernel, config: config, num_gpus: {json.dumps(num_gpus)}}}\n"
    )

    with pytest.raises(ValueError, match="num_gpus must be a positive integer"):
        run.load_workloads(workloads)


def test_load_workloads_rejects_megamoe_budget_override(tmp_path: Path) -> None:
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        """
workloads:
  - {kernel: kernel, config: config, timer: megamoe, warmup: 10}
"""
    )

    with pytest.raises(ValueError, match="fixed DeepGEMM protocol"):
        run.load_workloads(workloads)


def test_gpu_pool_acquires_and_releases_multiple_cards_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = run.GpuPool(allowed={"0", "1", "2"})
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0"), ("1", "GPU-1"), ("2", "GPU-2")])
    monkeypatch.setattr(run.random, "sample", lambda population, count: population[:count])

    assert pool.acquire_many(2) == ("0", "1")
    assert pool.acquire() == "2"
    assert pool._owned == {"0", "1", "2"}

    pool.release_many(("0", "1"))
    pool.release("2")
    assert pool._owned == set()


def test_gpu_pool_replays_an_exact_assignment_without_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = run.GpuPool(allowed={"0", "1", "2"})
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0"), ("1", "GPU-1"), ("2", "GPU-2")])

    assert pool.acquire_many(2, required_indices=("2", "0")) == ("2", "0")
    pool.release_many(("2", "0"))


def test_gpu_pool_prioritizes_larger_waiting_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = run.GpuPool(allowed={"0", "1", "2"})
    occupied = {"0", "1", "2"}
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set(occupied))
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0"), ("1", "GPU-1"), ("2", "GPU-2")])
    monkeypatch.setattr(run.random, "sample", lambda population, count: population[:count])
    monkeypatch.setattr(run, "POLL_INTERVAL", 0.001)
    results = {}
    large_done = threading.Event()
    single_done = threading.Event()

    def acquire_large() -> None:
        results["large"] = pool.acquire_many(3)
        large_done.set()

    def acquire_single() -> None:
        results["single"] = pool.acquire_many(1)
        single_done.set()

    large_thread = threading.Thread(target=acquire_large)
    single_thread = threading.Thread(target=acquire_single)
    large_thread.start()
    while True:
        with pool._lock:
            if pool._waiters:
                break
        time.sleep(0.001)
    single_thread.start()
    while True:
        with pool._lock:
            if len(pool._waiters) == 2:
                break
        time.sleep(0.001)
    occupied.clear()
    assert not single_done.is_set()

    assert large_done.wait(1)
    assert results["large"] == ("0", "1", "2")
    assert not single_done.is_set()

    pool.release_many(results["large"])
    assert single_done.wait(1)
    pool.release(results["single"])
    large_thread.join()
    single_thread.join()


def test_unsatisfiable_affinity_does_not_block_an_unrelated_free_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = run.GpuPool(allowed={"0", "1", "2"})
    occupied = {"0"}
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set(occupied))
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0"), ("1", "GPU-1"), ("2", "GPU-2")])
    monkeypatch.setattr(run, "POLL_INTERVAL", 0.001)
    results = {}
    large_done = threading.Event()
    single_done = threading.Event()

    def acquire_large() -> None:
        results["large"] = pool.acquire_many(2, required_indices=("0", "1"))
        large_done.set()

    def acquire_single() -> None:
        results["single"] = pool.acquire_many(1, required_indices=("2",))
        single_done.set()

    large_thread = threading.Thread(target=acquire_large)
    single_thread = threading.Thread(target=acquire_single)
    large_thread.start()
    while True:
        with pool._lock:
            if pool._waiters:
                break
        time.sleep(0.001)
    single_thread.start()

    assert single_done.wait(1)
    assert results["single"] == ("2",)
    assert not large_done.is_set()
    pool.release_many(results["single"])
    occupied.clear()
    assert large_done.wait(1)
    assert results["large"] == ("0", "1")
    pool.release_many(results["large"])
    large_thread.join()
    single_thread.join()


def test_resident_strangers_include_idle_compute_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run,
        "_compute_pids_by_gpu_uuid",
        lambda: {"GPU-0": {101, 999}, "GPU-1": {202}, "GPU-unassigned": {303}},
    )

    assert run._resident_strangers_on_gpu_uuids(("GPU-0", "GPU-1"), {999}) == {101, 202}


def test_gpu_telemetry_failure_is_not_interpreted_as_an_idle_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_query(*args, **kwargs):
        return type("FailedQuery", (), {"returncode": 1, "stdout": "", "stderr": "NVML error"})()

    monkeypatch.setattr(run.subprocess, "run", failed_query)

    with pytest.raises(RuntimeError, match="nvidia-smi telemetry failed"):
        run.GpuPool._nvidia_smi(["--query-gpu=index"])
    with pytest.raises(RuntimeError, match="compute-process telemetry failed"):
        run._compute_pids_by_gpu_uuid()


@pytest.mark.parametrize(
    "stdout",
    [
        "N/A, GPU-0\n",
        "123, N/A\n",
        "123, [Not Supported]\n",
        "123, Unknown\n",
        "123\n",
        "123, \n",
        "0, GPU-0\n",
        "123, GPU-0, unexpected\n",
    ],
)
def test_malformed_compute_process_telemetry_fails_closed(
    stdout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def malformed_query(*args, **kwargs):
        return type("MalformedQuery", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(run.subprocess, "run", malformed_query)

    with pytest.raises(RuntimeError, match="malformed compute-process telemetry row"):
        run._compute_pids_by_gpu_uuid()


@pytest.mark.parametrize("stdout", ["", "\n", " \n\t\n"])
def test_empty_compute_process_telemetry_means_no_resident_contexts(
    stdout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def empty_query(*args, **kwargs):
        return type("EmptyQuery", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(run.subprocess, "run", empty_query)

    assert run._compute_pids_by_gpu_uuid() == {}


def test_monitored_subprocess_requeues_resident_context_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "_gpu_uuid_of", lambda gpu_index: f"GPU-{gpu_index}")
    monkeypatch.setattr(
        run,
        "_resident_strangers_on_gpu_uuids",
        lambda gpu_uuids, our_pids: {123} if gpu_uuids == ("GPU-4",) else set(),
    )
    log_path = tmp_path / "subprocess.log"

    result = run._run_subprocess_monitored(
        [sys.executable, "-c", "raise AssertionError('must not spawn')"],
        os.environ.copy(),
        str(tmp_path),
        log_path,
        ("4",),
        0.01,
        0.0,
    )

    assert result == (-1, True, [123], False)
    assert "foreign compute contexts" in log_path.read_text()


def test_monitored_subprocess_requeues_when_gpu_uuid_lookup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "_gpu_uuid_of", lambda gpu_index: None)
    log_path = tmp_path / "subprocess.log"

    result = run._run_subprocess_monitored(
        [sys.executable, "-c", "raise AssertionError('must not spawn')"],
        os.environ.copy(),
        str(tmp_path),
        log_path,
        ("4",),
        0.01,
        0.0,
    )

    assert result == (-1, True, [], False)
    assert "could not resolve every assigned GPU UUID" in log_path.read_text()


def test_monitored_subprocess_checks_for_interference_after_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = 0
    monkeypatch.setattr(run, "_gpu_uuid_of", lambda gpu_index: f"GPU-{gpu_index}")

    def resident_after_exit(gpu_uuids, our_pids):
        nonlocal checks
        checks += 1
        return set() if checks == 1 else {321}

    monkeypatch.setattr(run, "_resident_strangers_on_gpu_uuids", resident_after_exit)

    result = run._run_subprocess_monitored(
        [sys.executable, "-c", "pass"],
        os.environ.copy(),
        str(tmp_path),
        tmp_path / "subprocess.log",
        ("2",),
        0.01,
        0.0,
    )

    assert result == (0, True, [321], False)
    assert checks >= 2


def test_monitored_subprocess_rejects_a_compute_child_that_outlives_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_path = tmp_path / "child.pid"
    monkeypatch.setattr(run, "_gpu_uuid_of", lambda gpu_index: f"GPU-{gpu_index}")

    def resident_outliving_child(gpu_uuids, our_pids):
        if not pid_path.exists():
            return set()
        child_pid = int(pid_path.read_text())
        return set() if child_pid in our_pids else {child_pid}

    monkeypatch.setattr(run, "_resident_strangers_on_gpu_uuids", resident_outliving_child)
    child_program = "import time; time.sleep(10)"
    root_program = (
        "import subprocess,sys; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_program!r}]); "
        f"Path({str(pid_path)!r}).write_text(str(p.pid))"
    )

    try:
        result = run._run_subprocess_monitored(
            [sys.executable, "-c", root_program],
            os.environ.copy(),
            str(tmp_path),
            tmp_path / "subprocess.log",
            ("2",),
            0.01,
            0.0,
        )
        child_pid = int(pid_path.read_text())

        assert result == (0, True, [child_pid], False)
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), 15)
            except ProcessLookupError:
                pass


def test_run_one_passes_multigpu_assignment_to_megamoe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePool:
        util_threshold = 0.0

        def __init__(self) -> None:
            self.released = None

        def acquire_many(
            self,
            count: int,
            *,
            required_indices: tuple[str, ...] | None = None,
            cancel_event: threading.Event | None = None,
        ) -> tuple[str, ...]:
            assert count == 2
            assert required_indices == ("4", "2")
            assert cancel_event is None
            return required_indices

        def release_many(self, indices: tuple[str, ...]) -> None:
            self.released = indices

    pool = FakePool()
    captured = {}

    def fake_run_subprocess_monitored(
        cmd, env, cwd, log_path, gpu_indices, monitor_interval, sm_threshold, cancel_event
    ):
        assert cancel_event is None
        captured.update(cmd=cmd, env=env, gpu_indices=gpu_indices)
        json_path = Path(cmd[cmd.index("--json-file") + 1])
        json_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "kernel": "deepgemm_fp8_fp4_mega_moe",
                            "label": "t64_m64_h7168_i3072_e384_k6_g2",
                            "status": "OK",
                            "impls": {"deepgemm": 10.5, "tirx": 10.0},
                            "round_samples": {"deepgemm": [10.0, 11.0], "tirx": [9.5, 10.5]},
                            "errors": {},
                        }
                    ]
                }
            )
        )
        return 0, False, [], False

    monkeypatch.setattr(run, "_run_subprocess_monitored", fake_run_subprocess_monitored)

    record = run.run_one(
        {
            "kernel": "deepgemm_fp8_fp4_mega_moe",
            "config": "t64_m64_h7168_i3072_e384_k6_g2",
            "timer": "megamoe",
            "num_gpus": 2,
        },
        pool,
        tmp_path,
        rounds=2,
        cooldown=0,
        required_gpus=("4", "2"),
    )

    assert record["status"] == "ok"
    assert record["gpu"] == "4,2"
    assert record["gpus"] == ["4", "2"]
    assert record["num_gpus"] == 2
    assert record["impls"] == {"deepgemm": 10.5, "tirx": 10.0}
    assert record["round_samples"] == {"deepgemm": [10.0, 11.0], "tirx": [9.5, 10.5]}
    assert captured["gpu_indices"] == ("4", "2")
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "4,2"
    assert Path(captured["env"]["TIRX_BENCH_CACHE_DIR"]).name == "cache"
    assert Path(captured["env"]["TIRX_BENCH_CACHE_DIR"]).is_absolute()
    assert captured["cmd"][captured["cmd"].index("--timer") + 1] == "megamoe"
    assert captured["cmd"][captured["cmd"].index("--cooldown") + 1] == "0"
    assert "--round-cooldown" not in captured["cmd"]
    assert pool.released == ("4", "2")


def test_required_benchmark_evidence_rejects_an_adapter_skip() -> None:
    record = run.finalize_required_benchmark_record(
        {
            "kernel": "fixture_kernel",
            "label": "fixture_config",
            "status": "SKIP",
            "reason": "reference dependency is unavailable",
        },
        rounds=5,
    )

    assert record["status"] == "FAIL"
    assert record["error"] == (
        "benchmark adapter skipped required evidence: reference dependency is unavailable"
    )


def test_gpu_pool_wait_is_cancelled_promptly(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = run.GpuPool(allowed={"0"})
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0")])
    with pool._lock:
        pool._owned.add("0")

    cancel_event = threading.Event()
    timer = threading.Timer(0.05, cancel_event.set)
    timer.start()
    try:
        with pytest.raises(run._BenchSuiteCancelled):
            pool.acquire_many(1, cancel_event=cancel_event)
    finally:
        timer.cancel()


def test_monitored_subprocess_is_terminated_on_cancel(tmp_path: Path) -> None:
    cancel_event = threading.Event()
    timer = threading.Timer(0.05, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        returncode, interfered, intruders, cancelled = run._run_subprocess_monitored(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            str(tmp_path),
            tmp_path / "subprocess.log",
            (),
            0.01,
            0.0,
            cancel_event,
        )
    finally:
        timer.cancel()

    assert cancelled
    assert returncode != 0
    assert not interfered
    assert intruders == []
    assert time.monotonic() - started < 2


def test_run_scheduled_jobs_stops_after_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run_one(workload, pool, log_dir, **kwargs):
        calls.append(workload["config"])
        return {
            "kernel": workload["kernel"],
            "config": workload["config"],
            "status": "FAIL",
            "error": "deterministic failure",
        }

    monkeypatch.setattr(run, "run_one", fake_run_one)
    records, retry_log = run.run_scheduled_jobs(
        [{"kernel": "kernel", "config": "fails"}, {"kernel": "kernel", "config": "must_not_start"}],
        _ScheduledJobsPool(1),
        tmp_path,
        rounds=1,
        cooldown=0,
    )

    assert calls == ["fails"]
    assert retry_log == []
    assert [(record["config"], record["status"], record["attempt"]) for record in records] == [
        ("fails", "FAIL", 1)
    ]


def test_run_scheduled_jobs_retries_interference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = []

    def fake_run_one(workload, pool, log_dir, *, attempt, **kwargs):
        attempts.append(attempt)
        if attempt == 1:
            return {
                "kernel": workload["kernel"],
                "config": workload["config"],
                "status": "INTERFERED",
                "error": "active neighbor",
                "intruder_pids": [123],
            }
        return {"kernel": workload["kernel"], "config": workload["config"], "status": "ok"}

    monkeypatch.setattr(run, "run_one", fake_run_one)
    records, retry_log = run.run_scheduled_jobs(
        [{"kernel": "kernel", "config": "config"}],
        _ScheduledJobsPool(1),
        tmp_path,
        rounds=1,
        cooldown=0,
    )

    assert attempts == [1, 2]
    assert records[0]["status"] == "ok"
    assert records[0]["attempt"] == 2
    assert retry_log == [("kernel", "config", 1, "intruders [123]")]


def test_run_scheduled_jobs_fails_after_the_interference_retry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = []

    def fake_run_one(workload, pool, log_dir, *, attempt, **kwargs):
        attempts.append(attempt)
        return {
            "kernel": workload["kernel"],
            "config": workload["config"],
            "status": "INTERFERED",
            "intruder_pids": [123],
        }

    monkeypatch.setattr(run, "run_one", fake_run_one)
    records, retry_log = run.run_scheduled_jobs(
        [{"kernel": "kernel", "config": "config"}],
        _ScheduledJobsPool(1),
        tmp_path,
        rounds=1,
        cooldown=0,
        max_interference_retries=1,
    )

    assert attempts == [1, 2]
    assert retry_log == [("kernel", "config", 1, "intruders [123]")]
    assert records[0]["status"] == "FAIL"
    assert records[0]["attempt"] == 2
    assert "retry limit is 1" in records[0]["error"]


def test_run_scheduled_jobs_cancels_inflight_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_started = threading.Event()
    calls = []

    def fake_run_one(workload, pool, log_dir, *, cancel_event, **kwargs):
        config = workload["config"]
        calls.append(config)
        if config == "fails":
            assert active_started.wait(1)
            return {
                "kernel": workload["kernel"],
                "config": config,
                "status": "FAIL",
                "error": "deterministic failure",
            }
        if config == "active":
            active_started.set()
            assert cancel_event.wait(1)
            return {"kernel": workload["kernel"], "config": config, "status": "CANCELLED"}
        raise AssertionError(f"unexpectedly started {config}")

    monkeypatch.setattr(run, "run_one", fake_run_one)
    records, retry_log = run.run_scheduled_jobs(
        [
            {"kernel": "kernel", "config": "fails"},
            {"kernel": "kernel", "config": "active"},
            {"kernel": "kernel", "config": "must_not_start"},
        ],
        _ScheduledJobsPool(2),
        tmp_path,
        rounds=1,
        cooldown=0,
    )

    assert set(calls) == {"fails", "active"}
    assert retry_log == []
    assert [(record["config"], record["status"]) for record in records] == [("fails", "FAIL")]
