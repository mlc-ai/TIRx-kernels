# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import hashlib
import importlib
import json
import os
import pickle
import statistics
import subprocess
import sys
import textwrap
import time
from contextlib import nullcontext
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

from tirx_kernels import registry
from tirx_kernels import runner as kernel_runner
from tirx_kernels.basic.utils import _runtime
from tirx_kernels.bench.__main__ import _validated_gpu_assignment
from tirx_kernels.bench_suite import run as bench_run
from tirx_kernels.bench_suite.run import (
    GpuPool,
    _audit_custom_cache_adapter_source,
    _audit_driver_safe_cuda_targets,
    _audit_generic_adapter_source,
    _audit_strict_cache_adapter_source,
    _finalize_bench_record,
    _pipeline_cost_model,
    _validate_pipeline_timelines,
    audit_pipeline_capabilities,
    check_workload_capabilities,
    write_summary,
)
from tirx_kernels.runner import (
    PreparedKernelBenchmark,
    PreparedRunBench,
    _offline_nvcc_arch,
    bind_cuda_assignment,
    compile_kernel_lazy,
    consume_prepared_cache,
    cuda_initialization_guard,
    current_cuda_assignment,
    gpu_stage_compile_guard,
    physical_cuda_uuids,
    replay_compiled_kernels,
    replay_prepared_cache,
    run_kernel_bench,
    run_prepared_kernel_bench,
    validate_current_cuda_assignment,
)

_FAKE_PREPARED_CHILD = textwrap.dedent(
    r"""
    import json
    import signal
    import socket
    import sys
    import time

    control = socket.socket(fileno=int(sys.argv[1]))
    reader = control.makefile("r", encoding="utf-8")
    workload = json.loads(sys.argv[2])

    def send(message):
        control.sendall(json.dumps(message, separators=(",", ":")).encode() + b"\n")

    started = time.time()
    print('{"type":"READY","channel":"stdout noise only"}', flush=True)
    time.sleep(float(workload.get("prepare_s", 0.02)))
    if workload.get("mode") == "fail_prepare":
        send({"type": "FAIL", "phase": "prepare", "error": "synthetic prepare failure"})
        raise SystemExit(1)

    ready = time.time()
    send(
        {
            "type": "READY",
            "child_started": started,
            "prepare_started": started,
            "framework_import_started": started,
            "framework_loaded": started,
            "module_loaded": started,
            "config_resolved": started,
            "ready": ready,
            "required_num_gpus": int(workload.get("num_gpus", 1)),
        }
    )
    class Interrupted(BaseException):
        pass

    signal.signal(signal.SIGUSR1, lambda *_args: (_ for _ in ()).throw(Interrupted()))
    gpu_attempt = 1
    while True:
        command = json.loads(reader.readline())
        if command.get("type") == "CANCEL":
            raise SystemExit(0)
        if command.get("type") != "ASSIGN":
            raise RuntimeError(f"unexpected command: {command}")
        gpu_uuids = command.get("gpu_uuids")
        if not isinstance(gpu_uuids, list) or len(gpu_uuids) != int(
            workload.get("num_gpus", 1)
        ):
            raise RuntimeError(f"invalid test GPU UUID assignment: {gpu_uuids}")

        gpu_started = time.time()
        if workload.get("mode") == "wrong_gpu_uuid":
            gpu_uuids = ["GPU-wrong"] * len(gpu_uuids)
        send(
            {
                "type": "RUNNING_GPU",
                "gpu_attempt": gpu_attempt,
                "gpu_started": gpu_started,
                "physical_gpu_uuids": gpu_uuids,
            }
        )
        try:
            time.sleep(float(workload.get("gpu_s", 0.05)))
        except Interrupted:
            send(
                {
                    "type": "INTERFERED",
                    "gpu_attempt": gpu_attempt,
                    "gpu_started": gpu_started,
                    "gpu_finished": time.time(),
                    "physical_gpu_uuids": gpu_uuids,
                    "resident_context_bytes_after_cleanup": {
                        str(index): 4096 for index in command["gpu_indices"]
                    },
                }
            )
            gpu_attempt += 1
            continue
        if workload.get("mode") == "fail_gpu":
            send(
                {
                    "type": "FAIL",
                    "phase": "gpu",
                    "gpu_started": gpu_started,
                    "gpu_finished": time.time(),
                    "error": "synthetic GPU failure",
                }
            )
            raise SystemExit(1)

        gpu_finished = time.time()
        rounds = int(workload["_rounds"])
        print("compiler/reference log after assignment", flush=True)
        send(
            {
                "type": "RESULT_READY",
                "gpu_attempt": gpu_attempt,
                "gpu_finished": gpu_finished,
                "result": {
                    "retry_in_place": gpu_attempt > 1,
                    "round_samples": {"tir": [1.0] * rounds},
                    "errors": {},
                    "timer": "proton",
                    "benchmark_protocol": {
                        "rounds": rounds,
                        "round_aggregate": "mean",
                        "order": ["tir"],
                        "cooldown_s": float(workload["_cooldown"]),
                    },
                },
            }
        )
        decision = json.loads(reader.readline())
        if decision.get("type") == "ACCEPT_RESULT":
            break
        if decision.get("type") != "RETRY_GPU":
            raise RuntimeError(f"unexpected result decision: {decision}")
        gpu_attempt += 1
    """
)


def _fake_pipeline(
    monkeypatch,
    tmp_path: Path,
    workloads: list[dict],
    *,
    gpu_indices=("0",),
    max_prepare_processes=2,
    ready_backlog=2,
    active_strangers=None,
):
    def command(workload, *, control_fd, rounds, cooldown):
        payload = {**workload, "_rounds": rounds, "_cooldown": cooldown}
        return [sys.executable, "-c", _FAKE_PREPARED_CHILD, str(control_fd), json.dumps(payload)]

    monkeypatch.setattr(bench_run, "_prepared_child_command", command)
    monkeypatch.setattr(
        bench_run, "_active_strangers", active_strangers or (lambda *_args, **_kwargs: {})
    )
    pool = GpuPool(allowed=set(gpu_indices))
    monkeypatch.setattr(
        pool, "_all_gpus", lambda: [(index, f"GPU-{index}") for index in gpu_indices]
    )
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return bench_run.run_scheduled_jobs(
        workloads,
        pool,
        log_dir,
        rounds=5,
        cooldown=1.0,
        compile_profile={"cuda_arch": "sm_100a", "num_sms": 148},
        max_prepare_processes=max_prepare_processes,
        ready_backlog=ready_backlog,
    )


def test_one_shot_pipeline_parallelizes_across_logical_gpus_and_isolates_logs(
    monkeypatch, tmp_path: Path
):
    workloads = [
        {"kernel": "fake", "config": f"w{index}", "num_gpus": 1, "gpu_s": 0.12}
        for index in range(4)
    ]

    records, retries, pipeline = _fake_pipeline(
        monkeypatch,
        tmp_path,
        workloads,
        gpu_indices=("0", "1"),
        max_prepare_processes=2,
        ready_backlog=2,
    )

    assert retries == []
    assert len(records) == 4
    assert all(record["status"] == "ok" for record in records)
    assert all(record["execution_mode"] == "pipeline" for record in records)
    assert len({record["process_pid"] for record in records}) == 4
    assert pipeline["max_observed_preparing"] <= 2
    assert pipeline["max_observed_buffered"] <= 2
    assert pipeline["max_observed_active_children"] <= 4
    assert pipeline["max_observed_process_tree"] >= 3
    assert pipeline["max_observed_rss_bytes"] > 0
    assert pipeline["max_observed_open_fds"] > 0

    intervals = [
        (record["phase_timestamps"]["gpu_started"], record["phase_timestamps"]["gpu_finished"])
        for record in records
    ]
    assert any(
        left_start < right_end and right_start < left_end
        for index, (left_start, left_end) in enumerate(intervals)
        for right_start, right_end in intervals[index + 1 :]
    )
    for gpu in ("0", "1"):
        per_gpu = sorted(
            (record["phase_timestamps"]["gpu_started"], record["phase_timestamps"]["gpu_finished"])
            for record in records
            if record["gpu"] == gpu
        )
        assert all(current[0] >= previous[1] for previous, current in pairwise(per_gpu))

    logs = "\n".join(path.read_text() for path in (tmp_path / "logs").glob("*.log"))
    assert "stdout noise only" in logs
    assert "compiler/reference log after assignment" in logs


def test_ready_backpressure_resumes_without_dropping_or_reusing_workloads(
    monkeypatch, tmp_path: Path
):
    workloads = [
        {"kernel": "fake", "config": f"w{index}", "num_gpus": 1, "prepare_s": 0.01, "gpu_s": 0.08}
        for index in range(6)
    ]

    records, retries, pipeline = _fake_pipeline(
        monkeypatch, tmp_path, workloads, max_prepare_processes=2, ready_backlog=2
    )

    assert retries == []
    assert sorted(record["config"] for record in records) == [f"w{index}" for index in range(6)]
    assert len({record["process_pid"] for record in records}) == len(workloads)
    assert all(record["attempt"] == 1 for record in records)
    assert pipeline["max_observed_preparing"] <= 2
    assert pipeline["max_observed_buffered"] <= 2
    first_gpu_started = min(record["phase_timestamps"]["gpu_started"] for record in records)
    assert any(
        record["phase_timestamps"]["process_started"] > first_gpu_started for record in records
    )


def test_pipeline_assigns_a_complete_multigpu_claim_before_gpu_stage(monkeypatch, tmp_path: Path):
    records, retries, pipeline = _fake_pipeline(
        monkeypatch,
        tmp_path,
        [{"kernel": "fake", "config": "tp2", "num_gpus": 2}],
        gpu_indices=("0", "1"),
        max_prepare_processes=1,
        ready_backlog=1,
    )

    assert retries == []
    assert records[0]["status"] == "ok"
    assert records[0]["gpus"] == ["0", "1"]
    timeline = records[0]["phase_timestamps"]
    assert timeline["ready"] <= timeline["assigned"] <= timeline["gpu_started"]


def test_pipeline_rejects_physical_gpu_identity_mismatch_before_gpu_stage(
    monkeypatch, tmp_path: Path
):
    records, retries, pipeline = _fake_pipeline(
        monkeypatch,
        tmp_path,
        [{"kernel": "fake", "config": "wrong-uuid", "num_gpus": 1, "mode": "wrong_gpu_uuid"}],
        max_prepare_processes=1,
        ready_backlog=1,
    )

    assert retries == []
    assert records[0]["status"] == "FAIL"
    assert "physical GPU UUID mismatch" in records[0]["error"]
    assert "gpu_started" not in records[0]["phase_timestamps"]
    assert bench_run._BenchPidRegistry._roots == set()


def test_pipeline_fail_fast_cancels_nonterminal_children(monkeypatch, tmp_path: Path):
    workloads = [
        {"kernel": "fake", "config": "fail", "num_gpus": 1, "mode": "fail_prepare"},
        {"kernel": "fake", "config": "slow", "num_gpus": 1, "prepare_s": 5.0},
        {"kernel": "fake", "config": "never", "num_gpus": 1},
    ]

    records, retries, _pipeline = _fake_pipeline(
        monkeypatch, tmp_path, workloads, max_prepare_processes=2, ready_backlog=2
    )

    assert retries == []
    assert len(records) == 1
    assert records[0]["config"] == "fail"
    assert records[0]["status"] == "FAIL"
    assert "synthetic prepare failure" in records[0]["error"]
    assert bench_run._BenchPidRegistry._roots == set()


def test_pipeline_fail_fast_cancels_ready_and_running_children(monkeypatch, tmp_path: Path):
    cleanup_events = []
    original_terminate = bench_run._terminate_subprocess
    original_release_many = GpuPool.release_many

    def terminate(proc):
        cleanup_events.append(("terminate", proc.pid))
        original_terminate(proc)

    def release_many(pool, indices):
        cleanup_events.append(("release", tuple(indices)))
        original_release_many(pool, indices)

    monkeypatch.setattr(bench_run, "_terminate_subprocess", terminate)
    monkeypatch.setattr(GpuPool, "release_many", release_many)
    workloads = [
        {"kernel": "fake", "config": "running", "num_gpus": 1, "gpu_s": 5.0},
        {
            "kernel": "fake",
            "config": "fail",
            "num_gpus": 1,
            "prepare_s": 0.15,
            "mode": "fail_prepare",
        },
        {"kernel": "fake", "config": "ready", "num_gpus": 1},
    ]

    started = time.monotonic()
    records, retries, _pipeline = _fake_pipeline(
        monkeypatch, tmp_path, workloads, max_prepare_processes=3, ready_backlog=3
    )
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert retries == []
    assert len(records) == 1
    assert records[0]["config"] == "fail"
    release_index = cleanup_events.index(("release", ("0",)))
    assert any(kind == "terminate" for kind, _value in cleanup_events[:release_index])
    assert bench_run._BenchPidRegistry._roots == set()


def test_keyboard_interrupt_reaps_preparing_children_and_resources(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        bench_run,
        "_prepared_child_command",
        lambda workload, *, control_fd, rounds, cooldown: [
            sys.executable,
            "-c",
            _FAKE_PREPARED_CHILD,
            str(control_fd),
            json.dumps({**workload, "prepare_s": 5.0, "_rounds": rounds, "_cooldown": cooldown}),
        ],
    )
    monkeypatch.setattr(bench_run, "_active_strangers", lambda *_args, **_kwargs: {})
    pool = GpuPool(allowed={"0"})
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0")])
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    original_select = bench_run.select.select
    interrupted = False

    def interrupt_once(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_select(*args, **kwargs)

    monkeypatch.setattr(bench_run.select, "select", interrupt_once)

    with pytest.raises(KeyboardInterrupt):
        bench_run.run_scheduled_jobs(
            [{"kernel": "fake", "config": "interrupt", "num_gpus": 1}],
            pool,
            log_dir,
            rounds=5,
            cooldown=1.0,
            compile_profile={"cuda_arch": "sm_100a", "num_sms": 148},
            max_prepare_processes=1,
            ready_backlog=1,
        )

    assert bench_run._BenchPidRegistry._roots == set()
    assert list(tmp_path.glob("bench-suite-*")) == []
    assert pool._owned == set()


def test_external_busy_to_eligible_poll_dispatches_ready_child(monkeypatch, tmp_path: Path):
    samples = 0

    def occupied_indices():
        nonlocal samples
        samples += 1
        return {"0"} if samples < 3 else set()

    monkeypatch.setattr(bench_run, "POLL_INTERVAL", 0.03)
    monkeypatch.setattr(
        bench_run,
        "_prepared_child_command",
        lambda workload, *, control_fd, rounds, cooldown: [
            sys.executable,
            "-c",
            _FAKE_PREPARED_CHILD,
            str(control_fd),
            json.dumps({**workload, "_rounds": rounds, "_cooldown": cooldown}),
        ],
    )
    monkeypatch.setattr(bench_run, "_active_strangers", lambda *_args, **_kwargs: {})
    pool = GpuPool(allowed={"0"})
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0")])
    monkeypatch.setattr(pool, "_occupied_indices", occupied_indices)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    records, retries, pipeline = bench_run.run_scheduled_jobs(
        [{"kernel": "fake", "config": "wait", "num_gpus": 1}],
        pool,
        log_dir,
        rounds=5,
        cooldown=1.0,
        compile_profile={"cuda_arch": "sm_100a", "num_sms": 148},
        max_prepare_processes=1,
        ready_backlog=1,
    )

    assert retries == []
    assert records[0]["status"] == "ok"
    assert records[0]["phase_timestamps"]["assigned"] > records[0]["phase_timestamps"]["ready"]
    assert any(
        entry["occupied_gpu_indices"] == ["0"] for entry in pipeline["external_occupancy_timeline"]
    )
    assert pipeline["external_occupancy_timeline"][-1]["occupied_gpu_indices"] == []
    assert bench_run._BenchPidRegistry._roots == set()


def test_interference_retry_reuses_prepared_child_and_releases_claim_after_ack(
    monkeypatch, tmp_path: Path
):
    samples = 0

    def active_strangers(*_args, **_kwargs):
        nonlocal samples
        samples += 1
        return {4242: 100.0} if samples == 2 else {}

    monkeypatch.setattr(bench_run, "MONITOR_INTERVAL", 0.01)

    records, retries, pipeline = _fake_pipeline(
        monkeypatch,
        tmp_path,
        [{"kernel": "fake", "config": "retry", "num_gpus": 1, "gpu_s": 0.5}],
        max_prepare_processes=1,
        ready_backlog=1,
        active_strangers=active_strangers,
    )

    assert len(retries) == 1
    assert retries[0]["status"] == "INTERFERED"
    assert retries[0]["intruder_pids"] == [4242]
    assert retries[0]["retry_attempt"] == 2
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["attempt"] == 2
    assert records[0]["retry_in_place"] is True
    assert records[0]["process_pid"] == retries[0]["process_pid"]
    assert len(records[0]["gpu_attempts"]) == 2
    first_attempt, second_attempt = records[0]["gpu_attempts"]
    assert first_attempt["status"] == "INTERFERED"
    assert first_attempt["resident_context_bytes_after_cleanup"] == {"0": 4096}
    assert first_attempt["ownership_released"] <= second_attempt["assigned"]
    intervals = pipeline["foreign_interference_intervals"]
    assert len(intervals) == 1
    assert intervals[0]["gpu_index"] == "0"
    assert intervals[0]["intruder_pids"] == [4242]
    assert intervals[0]["sources"] == ["running_gpu_monitor"]
    assert intervals[0]["closed_by"] == "predispatch_verified_clear"
    assert intervals[0]["started"] <= first_attempt["ownership_released"]
    assert intervals[0]["finished"] <= second_attempt["assigned"]
    assert pipeline["cost_model"]["foreign_wait_s"] > 0.0
    assert pipeline["cost_model"]["raw_dispatch_wait_s"]["p95"] >= pipeline[
        "cost_model"
    ]["dispatch_latency_s"]["p95"]
    timeline = records[0]["phase_timestamps"]
    assert set(
        ("prepare_started", "framework_import_started", "framework_loaded", "module_loaded")
    ) <= timeline.keys()
    attempt_logs = sorted((tmp_path / "logs").glob("fake__retry__a*.log"))
    assert [path.stem for path in attempt_logs] == ["fake__retry__a1"]


def test_exact_alias_load_imports_only_the_target_module():
    script = textwrap.dedent(
        """
        import json
        import sys

        before = set(sys.modules)
        from tirx_kernels.registry import load_kernel
        after_registry = set(sys.modules)
        module = load_kernel("flash_attention_backward_sm100", strict=True)
        after_load = set(sys.modules)
        print(
            json.dumps(
                {
                    "module": module.__name__,
                    "registry_imports": sorted(
                        name for name in after_registry - before
                        if name.startswith("tirx_kernels.")
                    ),
                    "load_imports": sorted(
                        name for name in after_load - after_registry
                        if name.startswith("tirx_kernels.")
                    ),
                }
            )
        )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)

    assert payload["module"] == "tirx_kernels.flashattention.flash_attention_backward"
    assert payload["registry_imports"] == ["tirx_kernels.registry"]
    assert payload["load_imports"] == [
        "tirx_kernels.flashattention",
        "tirx_kernels.flashattention.flash_attention_backward",
    ]


def test_registry_rejects_duplicate_and_invalid_source_metadata(tmp_path: Path):
    valid = tmp_path / "one.py"
    duplicate = tmp_path / "two.py"
    invalid = tmp_path / "invalid.py"
    valid.write_text(
        'KERNEL_META = {"name": "same", "category": "fake", "compute_capability": 10}\n'
    )
    duplicate.write_text(valid.read_text())
    invalid.write_text(
        'KERNEL_META = {"name": "bad", "category": "wrong", "compute_capability": True}\n'
    )

    duplicate_snapshot = (
        ("fake", "one", str(valid), valid.stat().st_mtime_ns, valid.stat().st_size),
        ("fake", "two", str(duplicate), duplicate.stat().st_mtime_ns, duplicate.stat().st_size),
    )
    with pytest.raises(ValueError, match="duplicate kernel registry name"):
        registry._build_kernel_index(duplicate_snapshot)

    invalid_snapshot = (
        ("fake", "invalid", str(invalid), invalid.stat().st_mtime_ns, invalid.stat().st_size),
    )
    records, diagnostics = registry._build_kernel_index(invalid_snapshot)
    assert records == {}
    assert "invalid KERNEL_META" in diagnostics[0]


def test_registry_cache_key_changes_with_source_provenance(tmp_path: Path):
    source = tmp_path / "kernel.py"
    source.write_text(
        'KERNEL_META = {"name": "before", "category": "fake", "compute_capability": 10}\n'
    )
    first_stat = source.stat()
    first_snapshot = (("fake", "kernel", str(source), first_stat.st_mtime_ns, first_stat.st_size),)
    first, _ = registry._build_kernel_index(first_snapshot)

    source.write_text(
        'KERNEL_META = {"name": "after_longer", "category": "fake", "compute_capability": 10}\n'
    )
    second_stat = source.stat()
    second_snapshot = (
        ("fake", "kernel", str(source), second_stat.st_mtime_ns, second_stat.st_size),
    )
    second, _ = registry._build_kernel_index(second_snapshot)

    assert first_snapshot != second_snapshot
    assert set(first) == {"before"}
    assert set(second) == {"after_longer"}


def test_assignment_rejects_partial_duplicate_and_invalid_claims():
    with pytest.raises(ValueError, match="invalid GPU assignment"):
        _validated_gpu_assignment(["0"], 2)
    with pytest.raises(ValueError, match="duplicates"):
        _validated_gpu_assignment(["0", "0"], 2)
    with pytest.raises(ValueError, match="invalid physical index"):
        _validated_gpu_assignment(["0", "GPU-deadbeef"], 2)
    assert _validated_gpu_assignment([3, "7"], 2) == ["3", "7"]


def test_physical_cuda_uuids_uses_driver_identity_without_creating_a_context(monkeypatch):
    from cuda.bindings import driver

    calls = []
    uuid_bytes = (
        bytes.fromhex("ef5a8300123434567890abcdefabcdef"),
        bytes.fromhex("e56ad157aaaabbbbccccdddd11112222"),
    )
    success = driver.CUresult.CUDA_SUCCESS
    monkeypatch.setattr(driver, "cuInit", lambda flags: calls.append(("init", flags)) or (success,))
    monkeypatch.setattr(
        driver, "cuDeviceGetCount", lambda: calls.append(("count",)) or (success, 2)
    )
    monkeypatch.setattr(
        driver,
        "cuDeviceGet",
        lambda index: calls.append(("device", index)) or (success, index + 10),
    )
    monkeypatch.setattr(
        driver,
        "cuDeviceGetUuid",
        lambda device: (
            calls.append(("uuid", device))
            or (success, SimpleNamespace(bytes=uuid_bytes[device - 10]))
        ),
    )

    assert physical_cuda_uuids((1, 0)) == (
        "GPU-e56ad157-aaaa-bbbb-cccc-dddd11112222",
        "GPU-ef5a8300-1234-3456-7890-abcdefabcdef",
    )
    assert calls == [
        ("init", 0),
        ("count",),
        ("device", 1),
        ("uuid", 11),
        ("device", 0),
        ("uuid", 10),
    ]


def test_physical_cuda_uuids_rejects_out_of_range_assignment(monkeypatch):
    from cuda.bindings import driver

    success = driver.CUresult.CUDA_SUCCESS
    monkeypatch.setattr(driver, "cuInit", lambda _flags: (success,))
    monkeypatch.setattr(driver, "cuDeviceGetCount", lambda: (success, 1))
    with pytest.raises(RuntimeError, match=r"requested device\(s\) \[2\].*only 1"):
        physical_cuda_uuids((2,))


def test_bind_assignment_happens_after_selection_and_rejects_uuid_mismatch(monkeypatch):
    monkeypatch.setattr(kernel_runner, "_CUDA_ASSIGNMENT", None)
    events = []
    fake_cuda = SimpleNamespace(
        set_device=lambda index: events.append(("set", index)),
        current_device=lambda: 3,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(kernel_runner, "physical_cuda_uuids", lambda indices: ("GPU-actual",))

    with pytest.raises(RuntimeError, match="identity mismatch"):
        bind_cuda_assignment((3,), ("GPU-expected",))

    assert events == [("set", 3)]


def test_external_device_override_is_restored_and_revalidated(monkeypatch):
    monkeypatch.setattr(kernel_runner, "_CUDA_ASSIGNMENT", None)
    current = 5

    def set_device(index):
        nonlocal current
        current = int(index)

    fake_cuda = SimpleNamespace(set_device=set_device, current_device=lambda: current)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(kernel_runner, "physical_cuda_uuids", lambda indices: ("GPU-5",))
    monkeypatch.setattr(kernel_runner, "_current_process_cuda_gpus", lambda **_kwargs: (5,))

    assert bind_cuda_assignment((5,), ("GPU-5",)) == ("GPU-5",)
    current = 0
    assert validate_current_cuda_assignment("after external call", restore=True) == ("GPU-5",)
    assert current == 5
    assert current_cuda_assignment() == ((5,), ("GPU-5",))


def test_bench_restores_external_reference_device_before_timing(monkeypatch):
    monkeypatch.setattr(kernel_runner, "_CUDA_ASSIGNMENT", None)
    current = 5
    events = []

    def set_device(index):
        nonlocal current
        current = int(index)
        events.append(("set", current))

    fake_cuda = SimpleNamespace(set_device=set_device, current_device=lambda: current)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(kernel_runner, "physical_cuda_uuids", lambda indices: ("GPU-5",))
    monkeypatch.setattr(kernel_runner, "_current_process_cuda_gpus", lambda **_kwargs: (5,))
    bind_cuda_assignment((5,), ("GPU-5",))

    def canonical_bench(*_args, references=None, **_kwargs):
        reference = references["external"]()
        events.append(("before_timing", current))
        reference()
        return {"impls": {"tir": 1.0, "external": 2.0}}

    canonical_module = importlib.import_module("tvm.tirx.bench")
    monkeypatch.setattr(canonical_module, "bench", canonical_bench)

    def build_external():
        set_device(0)
        return lambda: None

    result = kernel_runner.bench(
        {"tir": lambda: None}, references={"external": build_external}
    )

    assert result["impls"]["external"] == 2.0
    assert ("before_timing", 5) in events


def test_assignment_rejects_context_on_never_assigned_card(monkeypatch):
    monkeypatch.setattr(kernel_runner, "_CUDA_ASSIGNMENT", None)
    current = 4
    fake_cuda = SimpleNamespace(
        set_device=lambda index: None,
        current_device=lambda: current,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(kernel_runner, "physical_cuda_uuids", lambda indices: ("GPU-4",))
    monkeypatch.setattr(kernel_runner, "_current_process_cuda_gpus", lambda **_kwargs: (4, 1))

    with pytest.raises(RuntimeError, match=r"never-assigned physical GPU.*1"):
        bind_cuda_assignment((4,), ("GPU-4",))


def test_prepared_benchmark_is_process_local_and_not_serializable(monkeypatch):
    prepared = PreparedKernelBenchmark(
        kernel="fake",
        label="shape",
        benchmark=SimpleNamespace(run_gpu=lambda **_kwargs: {"impls": {"tir": 1.0}}),
    )

    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(prepared)

    monkeypatch.setattr("tirx_kernels.runner.os.getpid", lambda: prepared.owner_pid + 1)
    with pytest.raises(RuntimeError, match="process-local"):
        run_prepared_kernel_bench(prepared)


def test_standalone_runner_composes_prepare_and_gpu_stage(monkeypatch):
    events = []
    module = SimpleNamespace(
        __name__="fake_module",
        prepare_bench=lambda value: SimpleNamespace(
            run_gpu=lambda **kwargs: (
                events.append(("gpu", value, kwargs))
                or {"round_samples": {"tir": [1.0] * kwargs["rounds"]}}
            ),
            close=lambda: events.append("close"),
        ),
    )

    monkeypatch.setattr(kernel_runner, "cuda_initialization_guard", lambda **_kwargs: nullcontext())
    result = run_kernel_bench(
        "fake",
        {"label": "shape", "value": 7},
        registry={"fake": module},
        timer="proton",
        rounds=5,
        cooldown=1.0,
    )

    assert events == [
        ("gpu", 7, {"timer": "proton", "rounds": 5, "cooldown_s": 1.0}),
        "close",
    ]
    assert result["kernel"] == "fake"
    assert result["label"] == "shape"


def test_compile_kernel_lazy_builds_and_compiles_without_replay(monkeypatch):
    events = []
    prim_func = object()
    executable = object()

    def builder():
        events.append("build")
        return prim_func

    def compile_kernel(func):
        events.append(("compile", func))
        return executable

    monkeypatch.setattr(kernel_runner, "compile_kernel", compile_kernel)

    assert compile_kernel_lazy(builder) is executable
    assert events == ["build", ("compile", prim_func)]


def test_compile_kernel_lazy_replay_skips_builder_and_consumes_exactly_once():
    executable = object()

    def builder():
        raise AssertionError("GPU-stage replay must not rebuild the PrimFunc")

    with replay_compiled_kernels(((object(), executable),)):
        assert compile_kernel_lazy(builder) is executable

    with pytest.raises(RuntimeError, match="more kernel compilations"):
        with replay_compiled_kernels(((object(), executable),)):
            assert compile_kernel_lazy(builder) is executable
            compile_kernel_lazy(builder)

    with pytest.raises(RuntimeError, match="consumed 0 of 1"):
        with replay_compiled_kernels(((object(), executable),)):
            pass


def test_prepared_cache_replay_requires_exact_key_and_consumption():
    executable = object()

    def builder():
        raise AssertionError("GPU-stage replay must not compile a cached artifact")

    with replay_prepared_cache(
        (kernel_runner.PreparedCacheEntry("compile_spec", (1, 2), executable),)
    ):
        assert consume_prepared_cache("compile_spec", (1, 2), builder) is executable

    with pytest.raises(RuntimeError, match="does not match CPU prepare"):
        with replay_prepared_cache(
            (kernel_runner.PreparedCacheEntry("compile_spec", (1, 2), executable),)
        ):
            consume_prepared_cache("compile_spec", (1, 3), builder)

    with pytest.raises(RuntimeError, match="consumed 0 of 1"):
        with replay_prepared_cache(
            (kernel_runner.PreparedCacheEntry("compile_spec", (1, 2), executable),)
        ):
            pass

    with pytest.raises(RuntimeError, match="unprepared cached artifact"):
        with replay_prepared_cache(()):
            consume_prepared_cache("compile_spec", (1, 2), builder)


def test_prepared_run_bench_enforces_custom_cache_consumption():
    executable = object()
    module = SimpleNamespace(
        __name__="fake_cached_module",
        run_bench=lambda **_kwargs: consume_prepared_cache(
            "compile_spec", "shape", lambda: pytest.fail("must not build")
        ),
    )
    prepared = PreparedRunBench(
        module=module,
        params={},
        compiled=(),
        cached=(kernel_runner.PreparedCacheEntry("compile_spec", "shape", executable),),
    )

    assert prepared.run_gpu() is executable


def test_gpu_stage_compile_guard_rejects_direct_tvm_compile(monkeypatch):
    original_compile = kernel_runner.tvm.compile
    with pytest.raises(RuntimeError, match="completed before READY"):
        with gpu_stage_compile_guard():
            kernel_runner.tvm.compile(object())
    assert kernel_runner.tvm.compile is original_compile


def test_offline_nvcc_arch_preserves_family_specific_blackwell_features():
    assert _offline_nvcc_arch("sm_100a") == ["-gencode", "arch=compute_100a,code=sm_100a"]
    assert _offline_nvcc_arch("sm_100f") == ["-gencode", "arch=compute_100f,code=sm_100f"]
    assert _offline_nvcc_arch("sm_100") == "sm_100"


def test_prepare_cuda_toolchain_keeps_nvrtc_with_selected_nvcc(tmp_path):
    cuda_home = tmp_path / "cuda-13.2"
    cuda_bin = cuda_home / "bin"
    cuda_lib = cuda_home / "lib64"
    cuda_bin.mkdir(parents=True)
    cuda_lib.mkdir()
    for tool in ("nvcc", "ptxas"):
        path = cuda_bin / tool
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    nvrtc_library = cuda_lib / "libnvrtc.so.13"
    nvrtc_library.write_bytes(b"")
    wheel_lib = tmp_path / "wheel-cuda-13.0"
    wheel_lib.mkdir()
    env = {
        "PATH": os.pathsep.join([str(cuda_bin), "/usr/bin"]),
        "LD_LIBRARY_PATH": os.pathsep.join([str(wheel_lib), str(cuda_lib)]),
    }

    bench_run.pin_prepare_cuda_toolchain(env)

    assert env["PATH"].split(os.pathsep)[0] == str(cuda_bin)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(cuda_lib), str(wheel_lib)]
    assert env["CUDA_HOME"] == str(cuda_home)
    assert env["CUDA_PATH"] == str(cuda_home)
    assert env["TIRX_PREPARE_NVRTC_LIBRARY"] == str(nvrtc_library)


def test_megamoe_block_scale_compile_uses_arch_specific_sm100_target(monkeypatch):
    import tvm
    from tirx_kernels.deepgemm import mega_moe

    compiled = object()
    captured = {}
    monkeypatch.setattr(mega_moe, "get_kernel", lambda **_kwargs: object())
    monkeypatch.setattr(tvm, "IRModule", lambda functions: functions)

    def fake_compile(module, *, target, tir_pipeline):
        captured.update(module=module, target=target, tir_pipeline=tir_pipeline)
        return compiled

    monkeypatch.setattr(tvm, "compile", fake_compile)
    monkeypatch.setattr(mega_moe, "_cuda_compile_mode", lambda _mode: nullcontext())
    mega_moe._compile_tirx_mega_moe_for_config.cache_clear()
    try:
        result = mega_moe._compile_tirx_mega_moe_for_config(
            num_processes=1,
            num_max_tokens_per_rank=64,
            num_tokens=64,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=384,
            num_topk=6,
            activation_clamp=10.0,
            fast_math=1,
            collect_stats=False,
            cuda_compile_mode="nvcc",
        )
    finally:
        mega_moe._compile_tirx_mega_moe_for_config.cache_clear()

    assert result is compiled
    assert captured["target"].arch == "sm_100a"
    assert captured["tir_pipeline"] == "tirx"


def test_megamoe_distributed_rank_is_independent_of_physical_device(monkeypatch):
    from tirx_kernels.deepgemm import mega_moe

    calls = {}
    set_devices = []
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.9")
    monkeypatch.setenv("MASTER_PORT", "9123")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(mega_moe.torch.cuda, "set_device", set_devices.append)
    monkeypatch.setattr(
        mega_moe.torch,
        "set_default_device",
        lambda device: calls.update(default_device=device),
    )

    def fake_init_process_group(*, backend, init_method, world_size, rank, device_id=None):
        calls.update(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            device_id=device_id,
        )

    monkeypatch.setattr(
        mega_moe.torch.distributed, "init_process_group", fake_init_process_group
    )
    monkeypatch.setattr(mega_moe.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(mega_moe.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        mega_moe.torch.distributed,
        "new_group",
        lambda ranks: ("group", tuple(ranks)),
    )

    rank, world_size, group = mega_moe._init_dist_on_assigned_device(0, 1, 5)

    assert (rank, world_size, group) == (0, 1, ("group", (0,)))
    assert calls == {
        "backend": "nccl",
        "init_method": "tcp://127.0.0.9:9123",
        "world_size": 1,
        "rank": 0,
        "device_id": mega_moe.torch.device("cuda", 5),
        "default_device": "cuda",
    }
    assert set_devices == [5, 5]


def test_run_prepared_benchmark_applies_gpu_stage_compile_guard(monkeypatch):
    module = SimpleNamespace(run_gpu=lambda **_kwargs: kernel_runner.tvm.compile(object()))
    prepared = PreparedKernelBenchmark(kernel="fake", label="shape", benchmark=module)

    with pytest.raises(RuntimeError, match=r"GPU stage attempted tvm\.compile"):
        run_prepared_kernel_bench(prepared)


@pytest.mark.parametrize(
    ("module_name", "config_name", "compile_name", "key_name", "compile_args"),
    [
        (
            "tirx_kernels.deepgemm.tf32_hc_prenorm_gemm",
            "TF32HCPrenormGemmConfig",
            "_compile_tirx_tf32_hc",
            "_compile_tirx_tf32_hc_key",
            (),
        ),
        (
            "tirx_kernels.deepgemm.mqa_logits_fp4",
            "MQALogitsConfig",
            "_compile_tirx_mqa",
            "_compile_tirx_mqa_key",
            (0,),
        ),
        (
            "tirx_kernels.deepgemm.mqa_logits_fp8",
            "MQALogitsFP8Config",
            "_compile_tirx_mqa",
            "_compile_tirx_mqa_key",
            (0,),
        ),
        (
            "tirx_kernels.deepgemm.paged_mqa_logits_fp4",
            "PagedMQALogitsFP4Config",
            "_compile_tirx_paged_mqa",
            "_compile_tirx_paged_mqa_key",
            (),
        ),
        (
            "tirx_kernels.deepgemm.paged_mqa_logits_fp8",
            "PagedMQALogitsFP8Config",
            "_compile_tirx_paged_mqa",
            "_compile_tirx_paged_mqa_key",
            (),
        ),
    ],
)
def test_deepgemm_custom_compilers_consume_exact_prepared_artifact(
    monkeypatch,
    module_name: str,
    config_name: str,
    compile_name: str,
    key_name: str,
    compile_args: tuple,
):
    module = importlib.import_module(module_name)
    config = getattr(module, config_name)()
    compile_fn = getattr(module, compile_name)
    key_fn = getattr(module, key_name)
    namespace = module._COMPILE_CACHE_NAMESPACE
    raw_compiler_name = next(
        name for name in vars(module) if name.startswith(compile_name + "_for_config")
    )
    monkeypatch.setattr(
        module,
        raw_compiler_name,
        lambda **_kwargs: pytest.fail("prepared replay must not call the raw compiler"),
    )
    executable = object()

    with replay_prepared_cache(
        (kernel_runner.PreparedCacheEntry(namespace, key_fn(config), executable),)
    ):
        assert compile_fn(config, *compile_args) is executable

    mismatched_config = replace(config, num_sms=config.num_sms + 1)
    with pytest.raises(RuntimeError, match="does not match CPU prepare"):
        with replay_prepared_cache(
            (kernel_runner.PreparedCacheEntry(namespace, key_fn(config), executable),)
        ):
            compile_fn(mismatched_config, *compile_args)


def test_generic_adapter_static_gate_rejects_gpu_stage_builder_work(tmp_path: Path):
    valid = tmp_path / "valid.py"
    valid.write_text(
        textwrap.dedent(
            """
            def prepare_bench(**kwargs):
                return prepare_module_bench(__name__, kwargs)

            def run_test(**kwargs):
                return compile_kernel(get_kernel(**kwargs))

            def run_bench(**kwargs):
                return compile_kernel_lazy(lambda: get_kernel(**kwargs))
            """
        )
    )
    assert _audit_generic_adapter_source("valid", valid) is True

    eager = tmp_path / "eager.py"
    eager.write_text(
        valid.read_text().replace(
            "return compile_kernel_lazy(lambda: get_kernel(**kwargs))",
            "kernel = get_kernel(**kwargs)\n    return compile_kernel(kernel)",
        )
    )
    with pytest.raises(TypeError, match="compile_kernel_lazy"):
        _audit_generic_adapter_source("eager", eager)


def test_driver_safe_target_gate_rejects_import_time_device_detection(tmp_path: Path):
    safe_helper = tmp_path / "safe_helper.py"
    safe_helper.write_text("target = cuda_target()\n")
    _audit_driver_safe_cuda_targets("safe-helper", safe_helper)

    safe_explicit = tmp_path / "safe_explicit.py"
    safe_explicit.write_text(
        'target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})\n'
        "tvm.compile(mod, target=target)\n"
    )
    _audit_driver_safe_cuda_targets("safe-explicit", safe_explicit)

    unsafe_target = tmp_path / "unsafe_target.py"
    unsafe_target.write_text(
        'first = tvm.target.Target("cuda")\nsecond = tvm.target.Target({"kind": "cuda"})\n'
    )
    with pytest.raises(TypeError, match=r"Target line\(s\) \[1, 2\]"):
        _audit_driver_safe_cuda_targets("unsafe-target", unsafe_target)

    unsafe_compile = tmp_path / "unsafe_compile.py"
    unsafe_compile.write_text(
        'first = tvm.compile(mod, target="cuda")\nsecond = tvm.compile(mod, {"kind": "cuda"})\n'
    )
    with pytest.raises(TypeError, match=r"compile line\(s\) \[1, 2\]"):
        _audit_driver_safe_cuda_targets("unsafe-compile", unsafe_compile)


def test_strict_cache_adapter_gate_rejects_direct_compile_spec_prepare(tmp_path: Path):
    valid = tmp_path / "valid_cache.py"
    valid.write_text(
        textwrap.dedent(
            """
            def prepare_bench(**config):
                return prepare_compile_spec_bench(__name__, config, _spec_for(config))

            def _tirx_launch(data, config):
                return build_launch(_spec_for(config), a=data)

            def run_bench(**config):
                return _tirx_launch({}, config)
            """
        )
    )
    assert _audit_strict_cache_adapter_source("valid", valid) is True

    invalid = tmp_path / "invalid_cache.py"
    invalid.write_text(
        valid.read_text().replace(
            "return prepare_compile_spec_bench(__name__, config, _spec_for(config))",
            "compile_spec(_spec_for(config))\n"
            "    return prepared_cached_run_bench(__name__, config)",
        )
    )
    with pytest.raises(TypeError, match="strict compile-spec cache entry"):
        _audit_strict_cache_adapter_source("invalid", invalid)

    mismatched = tmp_path / "mismatched_cache.py"
    mismatched.write_text(
        valid.read_text().replace(
            "return build_launch(_spec_for(config), a=data)",
            "return build_launch(_spec_for(other_config), a=data)",
        )
    )
    with pytest.raises(TypeError, match=r"canonical _spec_for\(config\)"):
        _audit_strict_cache_adapter_source("mismatched", mismatched)

    bypassed = tmp_path / "bypassed_launch.py"
    bypassed.write_text(
        valid.read_text().replace("return _tirx_launch({}, config)", "return object()")
    )
    with pytest.raises(TypeError, match=r"exactly one _tirx_launch\(\) call"):
        _audit_strict_cache_adapter_source("bypassed", bypassed)


def test_custom_cache_adapter_gate_rejects_missing_registration_and_bypass(tmp_path: Path):
    valid = tmp_path / "valid_custom_cache.py"
    valid.write_text(
        textwrap.dedent(
            """
            CACHE_NAMESPACE = "custom.compile"

            def compile_key(config):
                return config.shape

            def raw_compile(**kwargs):
                return object()

            def compile_artifact(config):
                kwargs = {"shape": config.shape}
                return consume_prepared_cache(
                    CACHE_NAMESPACE,
                    compile_key(config),
                    lambda: raw_compile(**kwargs),
                )

            def prepare_bench(**kwargs):
                config = make_config(**kwargs)
                executable = compile_artifact(config)
                return prepared_cached_run_bench(
                    __name__,
                    kwargs,
                    cached=((CACHE_NAMESPACE, compile_key(config), executable),),
                )

            def prepare_invocation(config):
                return compile_artifact(config)

            def run_bench(**kwargs):
                return prepare_invocation(make_config(**kwargs))
            """
        )
    )
    assert _audit_custom_cache_adapter_source("valid", valid) is True

    unregistered = tmp_path / "unregistered_custom_cache.py"
    unregistered.write_text(
        valid.read_text().replace(
            "cached=((CACHE_NAMESPACE, compile_key(config), executable),),", "cached=(),"
        )
    )
    with pytest.raises(TypeError, match="exactly one cache entry"):
        _audit_custom_cache_adapter_source("unregistered", unregistered)

    wrong_key = tmp_path / "wrong_key_custom_cache.py"
    wrong_key.write_text(
        valid.read_text().replace(
            "cached=((CACHE_NAMESPACE, compile_key(config), executable),),",
            "cached=((CACHE_NAMESPACE, other_key(config), executable),),",
        )
    )
    with pytest.raises(TypeError, match="canonical key helper"):
        _audit_custom_cache_adapter_source("wrong-key", wrong_key)

    bypassed = tmp_path / "bypassed_custom_cache.py"
    bypassed.write_text(
        valid.read_text().replace(
            "return prepare_invocation(make_config(**kwargs))",
            "return raw_compile(shape=kwargs['shape'])",
        )
    )
    with pytest.raises(TypeError, match="does not consume"):
        _audit_custom_cache_adapter_source("bypassed", bypassed)


def test_cuda_initialization_guard_rejects_prepare_side_effect(monkeypatch):
    initialized = False
    fake_cuda = SimpleNamespace(is_initialized=lambda: initialized)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    with pytest.raises(RuntimeError, match="CPU prepare changed CUDA initialization state"):
        with cuda_initialization_guard(require_uninitialized=True):
            initialized = True


def test_cuda_initialization_guard_rejects_existing_driver_context(monkeypatch):
    monkeypatch.setattr("tirx_kernels.runner._current_process_cuda_gpus", lambda: (3,))

    with pytest.raises(RuntimeError, match="CUDA still uninitialized"):
        with cuda_initialization_guard(require_uninitialized=True):
            pass


@pytest.mark.parametrize(
    "missing_field", ["round_samples", "errors", "timer", "benchmark_protocol"]
)
def test_finalize_rejects_incomplete_measurement_schema(missing_field):
    row = {
        "round_samples": {"tir": [1.0] * 5, "ref": [2.0] * 5},
        "errors": {},
        "timer": "proton",
        "benchmark_protocol": {
            "rounds": 5,
            "round_aggregate": "mean",
            "order": ["tir", "ref"],
            "cooldown_s": 1.0,
        },
    }
    row.pop(missing_field)

    _finalize_bench_record(row, rounds=5, cooldown=1.0)

    assert row["status"] == "FAIL"
    assert missing_field in row["error"]


def test_finalize_requires_raw_samples_and_protocol_order():
    row = {
        "impls": {"tir": 1.0},
        "round_samples": {"tir": [1.0] * 5},
        "errors": {},
        "timer": "proton",
        "benchmark_protocol": {
            "rounds": 5,
            "round_aggregate": "mean",
            "order": ["ref", "tir"],
            "cooldown_s": 1.0,
        },
    }

    _finalize_bench_record(row, rounds=5, cooldown=1.0)

    assert row["status"] == "FAIL"
    assert "implementation order" in row["error"]


def test_finalize_accepts_megamoe_alternating_round_orders():
    row = {
        "round_samples": {"tir": [1.0] * 5, "deepgemm": [2.0] * 5},
        "errors": {},
        "timer": "megamoe",
        "benchmark_protocol": {
            "rounds": 5,
            "round_aggregate": "mean",
            "round_cooldown_s": 1.0,
            "round_orders": [
                ["tir", "deepgemm"],
                ["deepgemm", "tir"],
                ["tir", "deepgemm"],
                ["deepgemm", "tir"],
                ["tir", "deepgemm"],
            ],
        },
    }

    _finalize_bench_record(row, rounds=5, cooldown=1.0)

    assert row["status"] == "ok"
    assert row["impls"] == {"tir": 1.0, "deepgemm": 2.0}


@pytest.mark.parametrize(
    ("timer", "timer_attr"),
    [
        ("event", "_do_bench_event"),
        ("proton", "_do_bench_proton"),
        ("cudagraph_proton", "_do_bench_cudagraph_proton"),
    ],
)
def test_local_timer_families_emit_the_strict_measurement_schema(monkeypatch, timer, timer_attr):
    bench_module = importlib.import_module("tvm.tirx.bench")

    def fake_event_or_proton(fn, warmup=25, rep=100):
        del warmup, rep
        fn()
        return 0.001

    def fake_cudagraph(fn, rep=20):
        del rep
        fn()
        return 0.001

    fake_timer = fake_cudagraph if timer == "cudagraph_proton" else fake_event_or_proton
    monkeypatch.setattr(bench_module, timer_attr, fake_timer)
    monkeypatch.setattr(bench_module, "_sleep_before_impl", lambda _seconds: None)

    result = bench_module.bench(
        {"tirx": lambda: None},
        references={"reference": lambda: lambda: None},
        timer=timer,
        rounds=5,
        cooldown_s=1.0,
    )
    _finalize_bench_record(result, rounds=5, cooldown=1.0)

    assert result["status"] == "ok"
    assert result["timer"] == timer
    assert result["benchmark_protocol"]["order"] == ["tirx", "reference"]
    assert result["round_samples"] == {"tirx": [1.0] * 5, "reference": [1.0] * 5}


def test_kineto_schema_retains_complete_span_barrier_and_rank_max(monkeypatch):
    bench_module = importlib.import_module("tvm.tirx.bench")
    barriers = []
    reductions = []

    monkeypatch.setattr(bench_module, "_sleep_before_impl", lambda _seconds: None)
    monkeypatch.setattr(
        bench_module,
        "_profile_distributed_kineto_span",
        lambda *_args, **_kwargs: ([1.0, 3.0], [1, 2]),
    )

    def max_reduce(value):
        reductions.append(value)
        return value + 10.0

    distributed = bench_module.DistributedBenchContext(
        rank=0,
        world_size=2,
        barrier=lambda: barriers.append("barrier"),
        max_reduce=max_reduce,
        stream=SimpleNamespace(cuda_stream=7),
    )
    result = bench_module._bench_distributed_kineto_span(
        {"tirx": lambda: None, "reference": lambda: None}, {}, distributed, rounds=5, cooldown_s=1.0
    )
    _finalize_bench_record(result, rounds=5, cooldown=1.0)

    assert result["status"] == "ok"
    assert result["round_samples"] == {"tirx": [12.0] * 5, "reference": [12.0] * 5}
    assert reductions == [1.0, 3.0] * 10
    assert len(barriers) == 10
    protocol = result["benchmark_protocol"]
    assert protocol["timing_scope"] == "complete correlated GPU activity span"
    assert protocol["all_correlated_streams"] is True
    assert protocol["rank_barrier_before_each_launch"] is True
    assert protocol["rank_aggregate"] == "sample_wise_max"
    assert protocol["sample_aggregate"] == "median"


def test_megamoe_timer_schema_retains_barriers_spans_and_round_orders(monkeypatch):
    from tirx_kernels.deepgemm import mega_moe

    launches = []
    barriers = []
    resets = []

    def bench_kineto(run_pair, kernel_names, barrier, num_tests=30):
        del num_tests
        run_pair()
        barrier()
        launches.append(kernel_names)
        return [1e-6, 2e-6]

    monkeypatch.setattr(mega_moe.time, "sleep", lambda _seconds: None)
    result = mega_moe._bench_megamoe_mode(
        {"tirx": lambda: launches.append("tirx"), "deepgemm": lambda: launches.append("deepgemm")},
        {"tirx": "tirx_kernel", "deepgemm": "deepgemm_kernel"},
        bench_kineto,
        lambda: barriers.append("barrier"),
        lambda: resets.append("reset"),
        rounds=5,
        cooldown_s=1.0,
    )
    _finalize_bench_record(result, rounds=5, cooldown=1.0)

    assert result["status"] == "ok"
    assert len(barriers) == 5
    assert len(resets) == 5
    protocol = result["benchmark_protocol"]
    assert protocol["source"] == "deep_gemm.testing.bench_kineto"
    assert protocol["rank_barrier_outside_kernel_timing"] is True
    assert protocol["paired_profile_session"] is True
    assert protocol["round_orders"][0] == ["tirx", "deepgemm"]
    assert protocol["round_orders"][1] == ["deepgemm", "tirx"]


def test_gpu_pool_never_retains_a_partial_claim():
    pool = GpuPool(allowed={"0", "1"})
    with pool._changed:
        pool._external_occupied = set()

    assert pool.try_acquire_many(3) is None
    assert pool._owned == set()

    claim = pool.try_acquire_many(2)
    assert set(claim or ()) == {"0", "1"}
    assert pool._owned == {"0", "1"}
    assert pool.try_acquire_many(1) is None
    assert pool._owned == {"0", "1"}


def _megamoe_rank_result(*, tirx, deepgemm):
    return {
        "status": "OK",
        "impls": {"tirx": sum(tirx) / len(tirx), "deepgemm": sum(deepgemm) / len(deepgemm)},
        "round_samples": {"tirx": tirx, "deepgemm": deepgemm},
        "errors": {},
        "timer": "megamoe",
        "benchmark_protocol": {
            "rounds": len(tirx),
            "round_aggregate": "mean",
            "round_cooldown_s": 1.0,
            "round_orders": [
                ["tirx", "deepgemm"] if index % 2 == 0 else ["deepgemm", "tirx"]
                for index in range(len(tirx))
            ],
        },
        "deepgemm_max_abs_diff": 0.0,
    }


def test_megamoe_aggregates_rank_samples_by_sample_wise_max():
    from tirx_kernels.deepgemm import mega_moe

    result = mega_moe._aggregate_rank_results(
        [
            (1, _megamoe_rank_result(tirx=[5.0, 2.0], deepgemm=[3.0, 8.0])),
            (0, _megamoe_rank_result(tirx=[1.0, 7.0], deepgemm=[4.0, 6.0])),
        ]
    )

    assert result["round_samples"] == {"tirx": [5.0, 7.0], "deepgemm": [4.0, 8.0]}
    assert result["impls"] == {"tirx": 6.0, "deepgemm": 6.0}
    assert [entry["rank"] for entry in result["rank_results"]] == [0, 1]


def test_megamoe_rejects_mismatched_rank_round_counts():
    from tirx_kernels.deepgemm import mega_moe

    with pytest.raises(RuntimeError, match="different round counts"):
        mega_moe._aggregate_rank_results(
            [
                (0, _megamoe_rank_result(tirx=[1.0, 2.0], deepgemm=[3.0, 4.0])),
                (1, _megamoe_rank_result(tirx=[5.0], deepgemm=[6.0])),
            ]
        )


def test_explicit_workload_gpu_count_must_match_module_config():
    workload = {
        "kernel": "allgather_gemm",
        "config": "tp4_m8192_n24576_k4096_fp16_dynamic",
        "num_gpus": 1,
        "timer": "kineto",
    }
    with pytest.raises(ValueError, match="module config requires 4"):
        check_workload_capabilities([workload])


def test_distributed_ranks_start_after_visibility_validation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(kernel_runner, "_CUDA_ASSIGNMENT", None)
    monkeypatch.setattr(kernel_runner, "physical_cuda_uuids", lambda indices: tuple(
        f"GPU-{index}" for index in indices
    ))
    current_device = 6
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=lambda index: None,
                current_device=lambda: current_device,
            )
        ),
    )
    monkeypatch.setattr(
        kernel_runner, "_current_process_cuda_gpus", lambda **_kwargs: (6, 4, 2, 0)
    )
    bind_cuda_assignment((6, 4, 2, 0), ("GPU-6", "GPU-4", "GPU-2", "GPU-0"))
    events = []
    temporary_directory = SimpleNamespace(cleanup=lambda: events.append("cleanup"))
    prepared = _runtime.PreparedDistributedBench(
        executable=object(),
        library_path=tmp_path / "kernel.so",
        temporary_directory=temporary_directory,
        world_size=4,
        worker=lambda *_args, **_kwargs: {},
        worker_kwargs={"shape": "tp4"},
        required_timer="kineto",
    )

    monkeypatch.setattr(
        _runtime,
        "require_sm100",
        lambda device_indices: events.append(("validate_visible", device_indices)),
    )

    def launch(
        library_path, *, world_size, worker, mode, worker_kwargs, device_indices, device_uuids
    ):
        events.append(
            (
                "launch_ranks",
                library_path,
                world_size,
                mode,
                worker_kwargs,
                device_indices,
                device_uuids,
            )
        )
        return {"status": "OK"}

    monkeypatch.setattr(_runtime, "_run_distributed_library", launch)
    result = prepared.run_gpu(timer="kineto", rounds=5, cooldown_s=1.0)

    assert result == {"status": "OK"}
    assert events[0] == ("validate_visible", (6, 4, 2, 0))
    assert events[1][0] == "launch_ranks"
    assert events[1][4]["rounds"] == 5
    assert events[1][4]["cooldown_s"] == 1.0
    assert events[1][5:] == ((6, 4, 2, 0), ("GPU-6", "GPU-4", "GPU-2", "GPU-0"))
    assert "cleanup" not in events
    prepared.close()
    assert events[-1] == "cleanup"


def test_distributed_rank_cleanup_runs_when_worker_fails(monkeypatch):
    monkeypatch.setattr(kernel_runner, "_CUDA_ASSIGNMENT", None)
    events = []
    fake_runtime = SimpleNamespace(barrier=lambda: events.append("barrier"))
    fake_dist = SimpleNamespace(
        init_process_group=lambda **_kwargs: events.append("init_process_group"),
        is_available=lambda: True,
        is_initialized=lambda: True,
        destroy_process_group=lambda: events.append("destroy_process_group"),
    )
    current_device = 6

    def set_device(index):
        nonlocal current_device
        current_device = int(index)
        events.append(("set_device", index))

    fake_cuda = SimpleNamespace(
        set_device=set_device,
        current_device=lambda: current_device,
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, device=lambda *args: args)
    fake_tvm = SimpleNamespace(
        get_global_func=lambda name: (
            (lambda *_args: events.append("init_nvshmem"))
            if name.endswith("init_nvshmem")
            else (lambda: events.append("finalize_nvshmem"))
        ),
        runtime=SimpleNamespace(load_module=lambda _path: object()),
    )

    monkeypatch.setattr(_runtime, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(_runtime, "dist", fake_dist)
    monkeypatch.setattr(_runtime, "tvm", fake_tvm)
    monkeypatch.setattr(kernel_runner, "physical_cuda_uuids", lambda indices: ("GPU-6",))
    monkeypatch.setattr(kernel_runner, "_current_process_cuda_gpus", lambda **_kwargs: (6,))
    monkeypatch.setattr(_runtime, "_broadcast_nvshmem_uid", lambda _rank, _device: object())
    monkeypatch.setattr(_runtime, "_create_runtime", lambda *_args: fake_runtime)
    monkeypatch.setattr(
        _runtime, "_cleanup_runtime", lambda runtime: events.append(("cleanup_runtime", runtime))
    )

    def worker(*_args):
        events.append("worker")
        raise RuntimeError("synthetic rank failure")

    with pytest.raises(RuntimeError, match="synthetic rank failure"):
        _runtime._rank_entry(
            0,
            1,
            "file:///tmp/mock-init",
            "/tmp/mock-kernel.so",
            worker,
            "bench",
            {},
            SimpleNamespace(put=lambda _result: events.append("put")),
            (6,),
            ("GPU-6",),
        )

    assert ("set_device", 6) in events
    assert "worker" in events
    assert ("cleanup_runtime", fake_runtime) in events
    assert "finalize_nvshmem" in events
    assert "destroy_process_group" in events
    assert "put" not in events


def test_cost_model_charges_ready_starvation_and_result_handoff():
    def record(*, ready, assigned, gpu_started, gpu_finished, result_received, reaped):
        return {
            "status": "ok",
            "gpus": ["0"],
            "phase_timestamps": {
                "process_started": 0.0,
                "child_started": 0.1,
                "prepare_started": 0.1,
                "framework_import_started": 0.12,
                "framework_loaded": 0.15,
                "module_loaded": 0.2,
                "config_resolved": 0.3,
                "ready": ready,
                "assigned": assigned,
                "gpu_started": gpu_started,
                "gpu_finished": gpu_finished,
                "result_received": result_received,
                "process_reaped": reaped,
            },
        }

    records = [
        record(
            ready=1.0,
            assigned=1.0,
            gpu_started=1.0,
            gpu_finished=3.0,
            result_received=3.5,
            reaped=3.6,
        ),
        record(
            ready=4.0,
            assigned=4.0,
            gpu_started=4.0,
            gpu_finished=5.0,
            result_received=5.2,
            reaped=5.3,
        ),
    ]
    model = _pipeline_cost_model(
        records,
        run_started=0.0,
        critical_finished=5.2,
        known_gpu_indices=("0",),
        external_history=[(0.0, ())],
    )

    assert model["measurement_status"] == "measured"
    assert model["complete_timeline_count"] == model["record_count"] == 2
    assert model["ideal_gpu_list_schedule_s"] == pytest.approx(3.7)
    assert model["ready_constrained_gpu_list_schedule_s"] == pytest.approx(4.2)
    assert model["ready_starvation_s"] == pytest.approx(0.5)
    assert model["expected_s"] == pytest.approx(5.2)
    assert model["unexplained_s"] == pytest.approx(0.0)
    assert model["dispatch_latency_s"]["max"] == pytest.approx(0.0)

    records[1]["phase_timestamps"].update(ready=3.0, assigned=3.5)
    model = _pipeline_cost_model(
        records,
        run_started=0.0,
        critical_finished=5.2,
        known_gpu_indices=("0",),
        external_history=[(0.0, ())],
    )
    assert model["dispatch_latency_s"]["max"] == pytest.approx(0.0)


def test_cost_model_includes_interrupted_gpu_attempts():
    record = {
        "status": "ok",
        "num_gpus": 1,
        "gpus": ["1"],
        "phase_timestamps": {
            "process_started": 0.0,
            "child_started": 0.1,
            "prepare_started": 0.1,
            "framework_import_started": 0.12,
            "framework_loaded": 0.15,
            "module_loaded": 0.2,
            "config_resolved": 0.3,
            "ready": 1.0,
            "assigned": 2.1,
            "gpu_started": 2.1,
            "gpu_finished": 4.1,
            "result_received": 4.1,
            "process_reaped": 4.2,
        },
        "gpu_attempts": [
            {
                "attempt": 1,
                "ready": 1.0,
                "assigned": 1.0,
                "gpu_started": 1.1,
                "gpu_finished": 2.0,
                "ownership_released": 2.0,
                "gpus": ["0"],
                "status": "INTERFERED",
            },
            {
                "attempt": 2,
                "ready": 2.0,
                "assigned": 2.1,
                "gpu_started": 2.1,
                "gpu_finished": 4.1,
                "ownership_released": 4.1,
                "gpus": ["1"],
                "status": "RESULT",
            },
        ],
    }

    model = _pipeline_cost_model(
        [record],
        run_started=0.0,
        critical_finished=4.1,
        known_gpu_indices=("0", "1"),
        external_history=[(0.0, ())],
    )

    assert model["measurement_status"] == "measured"
    assert model["gpu_busy_s_by_index"] == pytest.approx({"0": 1.0, "1": 2.0})
    assert model["gpu_execution_s_by_index"] == pytest.approx({"0": 0.9, "1": 2.0})
    assert model["cpu_ready_constrained_gpu_list_schedule_s"] == pytest.approx(3.0)
    assert model["ready_constrained_gpu_list_schedule_s"] == pytest.approx(3.0)
    assert model["ready_starvation_s"] == pytest.approx(0.0)
    assert model["interference_retry_ready_delay_s"] == pytest.approx(0.0)
    assert model["interference_retry_count"] == 1
    assert model["interference_retry_gpu_ownership_s"] == pytest.approx(1.0)
    assert model["interference_retry_gpu_execution_s"] == pytest.approx(0.9)
    assert model["initial_dispatch_latency_s"]["p95"] == pytest.approx(0.0)
    assert model["retry_dispatch_latency_s"]["p95"] == pytest.approx(0.1)
    assert model["expected_s"] == pytest.approx(4.0)
    assert model["unexplained_s"] == pytest.approx(0.1)

    model = _pipeline_cost_model(
        [record],
        run_started=0.0,
        critical_finished=4.1,
        known_gpu_indices=("0", "1"),
        external_history=[(0.0, ())],
        foreign_intervals=[
            {
                "gpu_index": "0",
                "started": 2.0,
                "finished": 2.1,
            },
            {
                "gpu_index": "1",
                "started": 2.0,
                "finished": 2.1,
            }
        ],
    )
    assert model["foreign_wait_s"] == pytest.approx(0.1)
    assert model["raw_dispatch_wait_s"]["p95"] == pytest.approx(0.1)
    assert model["foreign_dispatch_wait_s"]["p95"] == pytest.approx(0.1)
    assert model["dispatch_latency_s"]["p95"] == pytest.approx(0.0)
    assert model["expected_with_foreign_s"] == pytest.approx(4.1)
    assert model["unexplained_s"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        ([], "no terminal workload records"),
        (
            [{"status": "FAIL", "phase_timestamps": {"process_started": 0.0}}],
            ("no workload produced an ok result with a complete timeline and valid GPU assignment"),
        ),
    ],
)
def test_cost_model_never_publishes_zeroes_for_missing_gpu_evidence(records, reason):
    model = _pipeline_cost_model(
        records,
        run_started=0.0,
        critical_finished=5.0,
        known_gpu_indices=("0",),
        external_history=[(0.0, ())],
    )

    assert model == {
        "schema_version": 3,
        "measurement_status": "missing",
        "record_count": len(records),
        "complete_timeline_count": 0,
        "complete_measurement_count": 0,
        "missing_reason": reason,
    }
    for field in (
        "expected_s",
        "expected_with_foreign_s",
        "unexplained_s",
        "dispatch_latency_s",
        "raw_dispatch_wait_s",
        "foreign_dispatch_wait_s",
        "initial_dispatch_latency_s",
        "retry_dispatch_latency_s",
        "assignment_handoff_s",
        "ready_starvation_s",
        "interference_retry_ready_delay_s",
        "interference_retry_count",
        "interference_retry_gpu_ownership_s",
        "interference_retry_gpu_execution_s",
        "cpu_ready_constrained_gpu_list_schedule_s",
        "gpu_busy_s_by_index",
        "gpu_execution_s_by_index",
        "foreign_wait_s",
    ):
        assert field not in model


def test_cost_model_requires_complete_coverage_for_every_record():
    complete = {
        "status": "ok",
        "gpus": ["0"],
        "phase_timestamps": {
            "process_started": 0.0,
            "child_started": 0.1,
            "prepare_started": 0.1,
            "framework_import_started": 0.12,
            "framework_loaded": 0.15,
            "module_loaded": 0.2,
            "config_resolved": 0.3,
            "ready": 1.0,
            "assigned": 1.0,
            "gpu_started": 1.0,
            "gpu_finished": 2.0,
            "result_received": 2.0,
            "process_reaped": 2.1,
        },
    }
    incomplete = {"status": "FAIL", "phase_timestamps": {"process_started": 0.0}}

    model = _pipeline_cost_model(
        [complete, incomplete],
        run_started=0.0,
        critical_finished=2.0,
        known_gpu_indices=("0",),
        external_history=[(0.0, ())],
    )

    assert model["measurement_status"] == "missing"
    assert model["record_count"] == 2
    assert model["complete_timeline_count"] == 1
    assert model["complete_measurement_count"] == 1
    assert "ok result, complete GPU timeline" in model["missing_reason"]
    assert "unexplained_s" not in model


@pytest.mark.parametrize(
    "record_update",
    [
        {"gpus": []},
        {"gpus": ["1"]},
        {"gpus": ["0", "0"], "num_gpus": 2},
        {"gpus": ["0"], "num_gpus": 2},
    ],
)
def test_cost_model_requires_a_valid_gpu_assignment_for_every_record(record_update):
    record = {
        "status": "ok",
        "gpus": ["0"],
        "num_gpus": 1,
        "phase_timestamps": {
            "process_started": 0.0,
            "child_started": 0.1,
            "prepare_started": 0.1,
            "framework_import_started": 0.12,
            "framework_loaded": 0.15,
            "module_loaded": 0.2,
            "config_resolved": 0.3,
            "ready": 1.0,
            "assigned": 1.0,
            "gpu_started": 1.0,
            "gpu_finished": 2.0,
            "result_received": 2.0,
            "process_reaped": 2.1,
        },
    }
    record.update(record_update)

    model = _pipeline_cost_model(
        [record],
        run_started=0.0,
        critical_finished=2.0,
        known_gpu_indices=("0",),
        external_history=[(0.0, ())],
    )

    assert model["measurement_status"] == "missing"
    assert model["complete_timeline_count"] == 1
    assert model["complete_measurement_count"] == 0
    assert "valid GPU assignment" in model["missing_reason"]
    assert "expected_s" not in model


def test_summary_marks_nondefault_protocol_and_missing_cost_evidence(tmp_path: Path):
    current = {
        "timestamp": "diagnostic",
        "label": "diagnostic-run",
        "git": {},
        "results": [{"kernel": "fake", "config": "shape", "status": "FAIL"}],
        "pipeline": {
            "process_model": "one_shot_child_per_workload_attempt",
            "measurement_protocol": {
                "rounds": 1,
                "cooldown_s": 0.0,
                "default_rounds": 5,
                "default_cooldown_s": 1.0,
                "is_default": False,
            },
            "cost_model": {
                "measurement_status": "missing",
                "record_count": 1,
                "complete_timeline_count": 0,
                "complete_measurement_count": 0,
                "missing_reason": "no workload reached a complete GPU timeline",
            },
        },
    }

    report = write_summary(tmp_path, current).read_text()

    assert "DIAGNOSTIC, NON-DEFAULT MEASUREMENT PROTOCOL" in report
    assert "used `1` round(s) and `0.0`s cooldown" in report
    assert "cost-model evidence: `missing`" in report
    assert "complete GPU timelines: `0` / `1`" in report
    assert "complete cost-model measurements: `0` / `1`" in report
    assert "unexplained residual:" not in report
    assert "dispatch latency p95/max:" not in report


def test_summary_derives_diagnostic_watermark_for_older_run_json(tmp_path: Path):
    current = {
        "timestamp": "legacy-diagnostic",
        "label": "legacy-diagnostic",
        "git": {},
        "results": [
            {
                "kernel": "fake",
                "config": "shape",
                "status": "ok",
                "benchmark_protocol": {"rounds": 1, "cooldown_s": 0.0},
            }
        ],
        "pipeline": {
            "process_model": "one_shot_child_per_workload_attempt",
            "cost_model": {
                "measurement_status": "missing",
                "record_count": 1,
                "complete_timeline_count": 0,
                "complete_measurement_count": 0,
                "missing_reason": "no workload reached a complete GPU timeline",
            },
        },
    }

    report = write_summary(tmp_path, current).read_text()

    assert "DIAGNOSTIC, NON-DEFAULT MEASUREMENT PROTOCOL" in report
    assert "used `1` round(s) and `0.0`s cooldown" in report


def test_summary_does_not_mark_default_protocol_as_diagnostic(tmp_path: Path):
    current = {
        "timestamp": "default",
        "label": "default-run",
        "git": {},
        "results": [],
        "pipeline": {
            "process_model": "one_shot_child_per_workload_attempt",
            "measurement_protocol": {
                "rounds": 5,
                "cooldown_s": 1.0,
                "default_rounds": 5,
                "default_cooldown_s": 1.0,
                "is_default": True,
            },
            "cost_model": {
                "measurement_status": "missing",
                "record_count": 0,
                "complete_timeline_count": 0,
                "complete_measurement_count": 0,
                "missing_reason": "no terminal workload records",
            },
        },
    }

    report = write_summary(tmp_path, current).read_text()

    assert "DIAGNOSTIC, NON-DEFAULT" not in report


def test_cost_model_attributes_lost_parallelism_to_external_occupancy():
    def record(*, gpu, assigned, gpu_finished, result_received, reaped):
        return {
            "status": "ok",
            "gpus": [gpu],
            "phase_timestamps": {
                "process_started": 0.0,
                "child_started": 0.1,
                "prepare_started": 0.1,
                "framework_import_started": 0.12,
                "framework_loaded": 0.15,
                "module_loaded": 0.2,
                "config_resolved": 0.3,
                "ready": 1.0,
                "assigned": assigned,
                "gpu_started": assigned,
                "gpu_finished": gpu_finished,
                "result_received": result_received,
                "process_reaped": reaped,
            },
        }

    records = [
        record(gpu="0", assigned=1.0, gpu_finished=3.0, result_received=3.0, reaped=3.1),
        record(gpu="0", assigned=3.0, gpu_finished=5.0, result_received=5.0, reaped=5.1),
        record(gpu="1", assigned=4.0, gpu_finished=6.0, result_received=6.0, reaped=6.1),
    ]
    model = _pipeline_cost_model(
        records,
        run_started=0.0,
        critical_finished=6.2,
        known_gpu_indices=("0", "1"),
        external_history=[(0.0, ("1",)), (4.0, ())],
    )

    assert model["ready_constrained_gpu_list_schedule_s"] == pytest.approx(4.0)
    assert model["eligibility_constrained_gpu_list_schedule_s"] == pytest.approx(5.0)
    assert model["foreign_wait_s"] == pytest.approx(1.0)
    assert model["expected_with_foreign_s"] == pytest.approx(6.0)
    assert model["unexplained_s"] == pytest.approx(0.2)


def test_cost_model_treats_pre_snapshot_gpus_as_ineligible():
    record = {
        "status": "ok",
        "gpus": ["0"],
        "phase_timestamps": {
            "process_started": 0.0,
            "child_started": 0.1,
            "prepare_started": 0.1,
            "framework_import_started": 0.12,
            "framework_loaded": 0.15,
            "module_loaded": 0.2,
            "config_resolved": 0.3,
            "ready": 1.0,
            "assigned": 2.0,
            "gpu_started": 2.0,
            "gpu_finished": 3.0,
            "result_received": 3.0,
            "process_reaped": 3.1,
        },
    }
    model = _pipeline_cost_model(
        [record],
        run_started=0.0,
        critical_finished=3.0,
        known_gpu_indices=("0",),
        external_history=[(2.0, ())],
    )

    assert model["ready_constrained_gpu_list_schedule_s"] == pytest.approx(1.0)
    assert model["eligibility_constrained_gpu_list_schedule_s"] == pytest.approx(2.0)
    assert model["foreign_wait_s"] == pytest.approx(1.0)
    assert model["unexplained_s"] == pytest.approx(0.0)


def _complete_timeline_record(*, label: str, gpu: str, assigned: float, received: float):
    return {
        "kernel": "fake",
        "config": label,
        "status": "ok",
        "gpus": [gpu],
        "phase_timestamps": {
            "process_started": 0.0,
            "child_started": 0.1,
            "prepare_started": 0.1,
            "framework_import_started": 0.12,
            "framework_loaded": 0.15,
            "module_loaded": 0.2,
            "config_resolved": 0.3,
            "ready": 1.0,
            "assigned": assigned,
            "gpu_started": assigned,
            "gpu_finished": received,
            "result_received": received,
            "process_reaped": received + 0.1,
        },
    }


def test_timeline_validation_rejects_missing_out_of_order_and_overlapping_records():
    missing = _complete_timeline_record(label="missing", gpu="0", assigned=1.0, received=2.0)
    missing["phase_timestamps"].pop("ready")
    with pytest.raises(RuntimeError, match="missing timeline phases"):
        _validate_pipeline_timelines([missing])

    reversed_timeline = _complete_timeline_record(
        label="reversed", gpu="0", assigned=1.0, received=2.0
    )
    reversed_timeline["phase_timestamps"]["gpu_started"] = 0.5
    with pytest.raises(RuntimeError, match="out-of-order timeline"):
        _validate_pipeline_timelines([reversed_timeline])

    first = _complete_timeline_record(label="first", gpu="0", assigned=1.0, received=3.0)
    second = _complete_timeline_record(label="second", gpu="0", assigned=2.0, received=4.0)
    with pytest.raises(RuntimeError, match="ownership intervals overlap"):
        _validate_pipeline_timelines([first, second])

    retried = _complete_timeline_record(label="retried", gpu="0", assigned=3.0, received=4.0)
    retried["gpu_attempts"] = [
        {
            "attempt": 1,
            "ready": 1.0,
            "assigned": 1.0,
            "gpu_started": 1.1,
            "gpu_finished": 2.0,
            "ownership_released": 2.1,
            "gpus": ["0"],
        },
        {
            "attempt": 2,
            "ready": 2.1,
            "assigned": 2.2,
            "gpu_started": 2.3,
            "gpu_finished": 4.0,
            "ownership_released": 4.0,
            "gpus": ["0"],
        },
    ]
    _validate_pipeline_timelines([retried])
    retried["gpu_attempts"][1].update(ready=2.0, assigned=2.0)
    with pytest.raises(RuntimeError, match="ownership intervals overlap"):
        _validate_pipeline_timelines([retried])


def test_capability_audit_accounts_for_every_adapter_and_curated_selection():
    capability = audit_pipeline_capabilities()

    assert capability["kernel_count"] == 41
    adapter_sets = [set(names) for names in capability["adapter_kernels"].values()]
    assert sum(map(len, adapter_sets)) == len(set.union(*adapter_sets))
    assert set.union(*adapter_sets) == set(registry.kernel_index(strict=True))
    assert [len(names) for names in adapter_sets] == [21, 10, 10]
    assert all(
        set(selection["configs"]) == {"small", "medium", "large"} and selection["rationale"]
        for selection in capability["curated_default_selections"]
    )


def test_tracked_default_protocol_ab_evidence_is_internally_consistent():
    evidence_path = Path(__file__).resolve().parents[1] / "bench_pipeline_ab_evidence.json"
    evidence = json.loads(evidence_path.read_text())

    assert (
        evidence["evidence_status"]
        == "recorded_default_protocol_run_invalidated_physical_identity_unverified"
    )
    assert evidence["acceptance_use"]["ac_10_performance_evidence"] is False
    conditions = evidence["fixed_conditions"]
    assert conditions["requested_physical_gpu_index"] == 6
    assert conditions["physical_gpu_identity"]["same_physical_gpu_verified"] is False
    assert conditions["rounds"] == 5
    assert conditions["cooldown_s"] == 1.0
    assert conditions["implementation_order"] == ["tir", "torch-cublas"]
    assert conditions["round_aggregate"] == "mean"

    before_wall = evidence["before"]["outer_wall_s"]
    after_wall = evidence["after"]["outer_wall_s"]
    assert evidence["derived"]["wall_speedup"] == pytest.approx(before_wall / after_wall, abs=1e-6)
    assert evidence["derived"]["wall_reduction_percent"] == pytest.approx(
        (1.0 - after_wall / before_wall) * 100.0, abs=1e-4
    )
    assert (
        evidence["derived"]["performance_claim_status"]
        == "arithmetic_only_invalidated_for_acceptance"
    )

    before_by_config = {row["config"]: row for row in evidence["before"]["results"]}
    after_by_config = {row["config"]: row for row in evidence["after"]["results"]}
    assert (
        list(before_by_config)
        == list(after_by_config)
        == [workload.split("/", 1)[1] for workload in evidence["matrix"]]
    )
    for config, before in before_by_config.items():
        after = after_by_config[config]
        for row in (before, after):
            assert list(row["round_samples_us"]) == conditions["implementation_order"]
            assert all(
                len(samples) == conditions["rounds"] for samples in row["round_samples_us"].values()
            )
        before_ratio = statistics.mean(before["round_samples_us"]["torch-cublas"]) / (
            statistics.mean(before["round_samples_us"]["tir"])
        )
        after_ratio = statistics.mean(after["round_samples_us"]["torch-cublas"]) / (
            statistics.mean(after["round_samples_us"]["tir"])
        )
        assert evidence["derived"]["ratio_delta_percent"][config] == pytest.approx(
            (after_ratio / before_ratio - 1.0) * 100.0, abs=1e-4
        )

    intervals = sorted(
        (
            row["timeline_s_from_scheduler_start"]["gpu_started"],
            row["timeline_s_from_scheduler_start"]["result_received"],
        )
        for row in evidence["after"]["results"]
    )
    assert all(current[0] >= previous[1] for previous, current in pairwise(intervals))
    cost = evidence["after"]["pipeline_cost_model"]
    first_ready = min(
        row["timeline_s_from_scheduler_start"]["ready"] for row in evidence["after"]["results"]
    )
    measured_gpu_schedule = sum(
        row["timeline_s_from_scheduler_start"]["result_received"]
        - row["timeline_s_from_scheduler_start"]["gpu_started"]
        for row in evidence["after"]["results"]
    )
    assert cost["first_ready_s"] == pytest.approx(first_ready, abs=1e-6)
    assert cost["ideal_gpu_list_schedule_s"] == pytest.approx(measured_gpu_schedule, abs=1e-6)
    assert cost["expected_s"] == pytest.approx(first_ready + measured_gpu_schedule, abs=1e-6)
    assert cost["unexplained_s"] == pytest.approx(
        cost["observed_critical_s"] - cost["expected_s"] - cost["foreign_wait_s"], abs=1e-6
    )
    assert cost["ready_starvation_s"] == 0.0
    assert cost["foreign_wait_s"] == 0.0
    assert cost["dispatch_latency_p95_s"] < 0.1
    assert cost["unexplained_s"] <= max(0.5, 0.05 * cost["observed_critical_s"])


@pytest.mark.parametrize("side", ["before", "after"])
def test_tracked_default_protocol_ab_evidence_matches_local_source_artifact_when_present(side: str):
    repo_root = Path(__file__).resolve().parents[1]
    evidence = json.loads((repo_root / "bench_pipeline_ab_evidence.json").read_text())
    source = evidence[side]["source_artifact"]
    artifact_path = Path(source["path"])
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    if not artifact_path.is_file():
        pytest.skip(
            f"{side} source artifact is gitignored local state and is unavailable: {artifact_path}"
        )

    artifact_bytes = artifact_path.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == evidence[side]["source_sha256"]
    artifact = json.loads(artifact_bytes)
    artifact_results = artifact["results"]
    tracked_results = evidence[side]["results"]
    assert [row["config"] for row in artifact_results] == [
        workload.split("/", 1)[1] for workload in evidence["matrix"]
    ]
    assert [row["config"] for row in tracked_results] == [row["config"] for row in artifact_results]

    conditions = evidence["fixed_conditions"]
    for artifact_row, tracked_row in zip(artifact_results, tracked_results, strict=True):
        assert artifact_row["kernel"] == "fp16_bf16_gemm"
        assert artifact_row["status"] == "ok"
        assert artifact_row["timer"] == conditions["timer"]
        protocol = artifact_row["benchmark_protocol"]
        assert protocol["rounds"] == conditions["rounds"]
        assert protocol["cooldown_s"] == conditions["cooldown_s"]
        assert protocol["warmup"] == conditions["warmup_ms"]
        assert protocol["repeat"] == conditions["repeat_ms"]
        assert protocol["round_aggregate"] == conditions["round_aggregate"]
        assert protocol["order"] == conditions["implementation_order"]
        for implementation in conditions["implementation_order"]:
            assert artifact_row["round_samples"][implementation] == pytest.approx(
                tracked_row["round_samples_us"][implementation], abs=1e-9
            )

    if side == "after":
        pipeline = artifact["pipeline"]
        assert pipeline["measurement_protocol"]["is_default"] is True
        for artifact_row, tracked_row in zip(artifact_results, tracked_results, strict=True):
            timeline = {
                name: timestamp - pipeline["started"]
                for name, timestamp in artifact_row["phase_timestamps"].items()
            }
            tracked_timeline = tracked_row["timeline_s_from_scheduler_start"]
            assert set(tracked_timeline) <= set(timeline)
            for phase, timestamp in tracked_timeline.items():
                assert timestamp == pytest.approx(timeline[phase], abs=1e-6)

        artifact_cost = pipeline["cost_model"]
        tracked_cost = evidence["after"]["pipeline_cost_model"]
        direct_fields = {
            "observed_critical_s",
            "first_ready_s",
            "ideal_gpu_list_schedule_s",
            "ready_constrained_gpu_list_schedule_s",
            "eligibility_constrained_gpu_list_schedule_s",
            "ready_starvation_s",
            "foreign_wait_s",
            "expected_s",
            "unexplained_s",
        }
        for field in direct_fields:
            assert tracked_cost[field] == pytest.approx(artifact_cost[field], abs=1e-9)
        assert tracked_cost["dispatch_latency_p95_s"] == pytest.approx(
            artifact_cost["dispatch_latency_s"]["p95"], abs=1e-9
        )
        assert tracked_cost["assignment_handoff_p95_s"] == pytest.approx(
            artifact_cost["assignment_handoff_s"]["p95"], abs=1e-9
        )


def test_tracked_set_device_evidence_preserves_retry_provenance():
    evidence_path = Path(__file__).resolve().parents[1] / "bench_pipeline_set_device_evidence.json"
    evidence = json.loads(evidence_path.read_text())

    assert evidence["acceptance_use"] == {
        "implementation_validation": True,
        "ac_10_migration_before_ab": False,
        "reason": (
            "These are pipeline-only targeted runs; no persisted migration-before run "
            "on the same verified physical GPU is paired with them."
        ),
    }
    fresh = evidence["artifacts"]["fresh"]
    retry = evidence["artifacts"]["retry"]
    assert fresh["protocol"]["is_default"] is True
    assert retry["protocol"]["is_default"] is True
    assert fresh["record"]["retry_in_place"] is False
    assert retry["record"]["retry_in_place"] is True
    assert fresh["record"]["attempt"] == 1
    assert retry["record"]["attempt"] == 2
    assert len(fresh["record"]["gpu_attempts"]) == 1
    assert [attempt["status"] for attempt in retry["record"]["gpu_attempts"]] == [
        "INTERFERED",
        "RESULT",
    ]
    first_attempt, second_attempt = retry["record"]["gpu_attempts"]
    assert first_attempt["ownership_released"] <= second_attempt["assigned"]
    assert first_attempt["resident_context_bytes_after_cleanup"] == {"2": 645_922_816}
    assert second_attempt["abandoned_gpu_resident_bytes_after_reassignment"] == {
        "2": 645_922_816
    }
    assert retry["interference_retries"][0]["process_pid"] == retry["record"]["process_pid"]
    assert retry["interference_retries"][0]["retry_in_place"] is True
    assert evidence["post_exit_observation"]["abandoned_gpu"]["memory_used_mib"] == 124
    assert evidence["post_exit_observation"]["abandoned_gpu"]["listed_compute_pid"] is False


@pytest.mark.parametrize("name", ["fresh", "retry"])
def test_tracked_set_device_evidence_matches_local_artifact_when_present(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    evidence = json.loads((repo_root / "bench_pipeline_set_device_evidence.json").read_text())
    tracked = evidence["artifacts"][name]
    artifact_path = repo_root / tracked["local_path"]
    if not artifact_path.is_file():
        pytest.skip(f"set-device source artifact is gitignored local state: {artifact_path}")

    artifact_bytes = artifact_path.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == tracked["sha256"]
    artifact = json.loads(artifact_bytes)
    assert artifact["pipeline"]["measurement_protocol"] == tracked["protocol"]
    assert artifact["pipeline"]["cost_model"] == tracked["cost_model"]
    assert artifact["pipeline"]["interference_retries"] == tracked.get(
        "interference_retries", []
    )
    artifact_record = artifact["records"][0]
    for field in (
        "kernel",
        "config",
        "status",
        "process_pid",
        "attempt",
        "retry_in_place",
        "physical_gpu_uuids",
        "impls",
        "round_samples",
        "benchmark_protocol",
        "phase_timestamps",
        "gpu_attempts",
    ):
        assert artifact_record[field] == tracked["record"][field]
    assert tracked["outer_wall_s"] == pytest.approx(
        artifact["pipeline"]["processes_reaped"] - artifact["pipeline"]["started"],
        abs=1e-9,
    )


def test_tracked_large_prepare_evidence_respects_resource_bounds():
    evidence_path = Path(__file__).resolve().parents[1] / "bench_pipeline_cpu_prepare_evidence.json"
    evidence = json.loads(evidence_path.read_text())

    assert evidence["evidence_status"] == "measured_cpu_only_ready_then_cancelled"
    assert evidence["protocol"] == {
        "concurrency_limit": 3,
        "gpu_assignment_sent": False,
        "terminal_command": "CANCEL",
        "rounds_forwarded_but_not_executed": 5,
        "cooldown_s_forwarded_but_not_executed": 1.0,
    }
    assert len(evidence["workloads"]) == 3
    assert evidence["wall_to_all_ready_s"] == pytest.approx(
        max(workload["ready_s"] for workload in evidence["workloads"])
    )
    assert evidence["resource_peaks"]["owned_process_tree"] <= 4
    assert evidence["resource_peaks"]["rss_bytes"] > 0
    assert evidence["resource_peaks"]["open_fds"] > 0
    assert evidence["cleanup"] == {
        "remaining_registered_pids": [],
        "temporary_directories_removed": True,
        "cuda_initialized": False,
    }


def test_tracked_custom_cache_prepare_evidence_has_no_gpu_claim():
    evidence_path = (
        Path(__file__).resolve().parents[1]
        / "bench_pipeline_custom_cache_prepare_evidence.json"
    )
    evidence = json.loads(evidence_path.read_text())

    assert evidence["evidence_status"] == "measured_cpu_only_ready_then_cancelled"
    assert evidence["protocol"] == {
        "fresh_one_shot_child_per_workload": True,
        "gpu_assignment_sent": False,
        "terminal_command": "CANCEL",
        "rounds_forwarded_but_not_executed": 5,
        "cooldown_s_forwarded_but_not_executed": 1.0,
    }
    expected_workloads = [
        (
            "deepgemm_sm100_tf32_hc_prenorm_gemm",
            "m64_n24_k28672_s112",
        ),
        (
            "deepgemm_sm100_fp4_mqa_logits",
            "s2048_skv4096_h64_d128_f32_dense_cp",
        ),
        (
            "deepgemm_sm100_fp8_mqa_logits",
            "s2048_skv4096_h64_d128_f32_dense_cp",
        ),
        (
            "deepgemm_sm100_fp4_paged_mqa_logits",
            "b1_n1_mp1_ps32_h64_d128_f32_fixed",
        ),
        (
            "deepgemm_sm100_fp8_paged_mqa_logits",
            "b1_n1_mp4_ps64_h64_d128_f32_fixed",
        ),
    ]
    workloads = evidence["workloads"]
    assert [(row["kernel"], row["config"]) for row in workloads] == expected_workloads
    assert all(row["exit_code"] == 0 for row in workloads)
    assert all(row["wall_to_ready_s"] >= row["cpu_prepare_s"] > 0 for row in workloads)
    assert all(row["specialize_generate_compile_s"] > 0 for row in workloads)
    assert "AC-10" in evidence["evidence_boundary"]["does_not_prove"]
    assert "no raw run JSON" in evidence["evidence_boundary"]["raw_artifact"]


def test_outer_timer_persists_complete_command_provenance(tmp_path: Path):
    artifact = tmp_path / "outer.json"
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    script = Path(__file__).resolve().parents[1] / "scripts/run_with_outer_timer.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact",
            str(artifact),
            "--stdout-log",
            str(stdout_log),
            "--stderr-log",
            str(stderr_log),
            "--",
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
        ],
        check=False,
    )

    assert completed.returncode == 7
    evidence = json.loads(artifact.read_text())
    assert evidence["status"] == "completed"
    assert evidence["returncode"] == 7
    assert evidence["wrapper_returncode"] == 7
    assert evidence["command_wall_ns"] == (
        evidence["command_finished_monotonic_ns"]
        - evidence["command_started_monotonic_ns"]
    )
    assert evidence["command_wall_s"] == pytest.approx(
        evidence["command_wall_ns"] / 1_000_000_000
    )
    assert evidence["wrapper_wall_ns"] == (
        evidence["wrapper_finished_monotonic_ns"]
        - evidence["wrapper_started_monotonic_ns"]
    )
    assert evidence["wrapper_wall_s"] == pytest.approx(
        evidence["wrapper_wall_ns"] / 1_000_000_000
    )
    assert evidence["wrapper_wall_ns"] >= evidence["command_wall_ns"]
    assert evidence["cwd"] == str(Path.cwd())
    assert evidence["argv"][-2:] == [
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
    ]
    assert stdout_log.read_text() == "out\n"
    assert stderr_log.read_text() == "err\n"


def test_outer_timer_records_one_physical_gpu_without_masking_the_child(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_smi = fake_bin / "nvidia-smi"
    fake_smi.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "query = next(arg for arg in sys.argv if arg.startswith('--query-'))\n"
        "if query.startswith('--query-gpu='):\n"
        "    print('1, GPU-test-uuid, NVIDIA B200, 0, 128, 183359')\n"
        "elif query.startswith('--query-compute-apps='):\n"
        "    pass\n"
    )
    fake_smi.chmod(0o755)
    artifact = tmp_path / "outer.json"
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    script = Path(__file__).resolve().parents[1] / "scripts/run_with_outer_timer.py"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.pop("CUDA_VISIBLE_DEVICES", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact",
            str(artifact),
            "--stdout-log",
            str(stdout_log),
            "--stderr-log",
            str(stderr_log),
            "--physical-gpu-index",
            "1",
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))",
        ],
        check=False,
        env=env,
    )

    assert completed.returncode == 0
    evidence = json.loads(artifact.read_text())
    assert evidence["status"] == "completed"
    assert evidence["physical_gpu"]["requested_index"] == "1"
    assert evidence["physical_gpu"]["before"]["uuid"] == "GPU-test-uuid"
    assert evidence["physical_gpu"]["after"]["uuid"] == "GPU-test-uuid"
    assert evidence["physical_gpu"]["same_uuid_before_after"] is True
    assert evidence["cuda_visible_devices_at_wrapper_start"] is None
    assert "environment_overrides" not in evidence
    assert stdout_log.read_text() == "<unset>\n"


def test_outer_timer_rejects_a_nonidle_gpu_before_launch(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_smi = fake_bin / "nvidia-smi"
    fake_smi.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "query = next(arg for arg in sys.argv if arg.startswith('--query-'))\n"
        "if query.startswith('--query-gpu='):\n"
        "    print('1, GPU-test-uuid, NVIDIA B200, 0, 2048, 183359')\n"
        "elif query.startswith('--query-compute-apps='):\n"
        "    pass\n"
    )
    fake_smi.chmod(0o755)
    artifact = tmp_path / "outer.json"
    marker = tmp_path / "launched"
    script = Path(__file__).resolve().parents[1] / "scripts/run_with_outer_timer.py"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact",
            str(artifact),
            "--stdout-log",
            str(tmp_path / "stdout.log"),
            "--stderr-log",
            str(tmp_path / "stderr.log"),
            "--physical-gpu-index",
            "1",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        check=False,
        env=env,
    )

    assert completed.returncode == 75
    assert not marker.exists()
    evidence = json.loads(artifact.read_text())
    assert evidence["status"] == "preflight_rejected"
    assert "memory 2048.0 MiB exceeds 512.0 MiB" in evidence["error"]
