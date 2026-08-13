# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import importlib
import json
import pickle
import subprocess
import sys
import textwrap
import time
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
    _audit_generic_adapter_source,
    _audit_strict_cache_adapter_source,
    _finalize_bench_record,
    _pipeline_cost_model,
    check_workload_capabilities,
    write_summary,
)
from tirx_kernels.runner import (
    PreparedKernelBenchmark,
    PreparedRunBench,
    compile_kernel_lazy,
    consume_prepared_cache,
    cuda_initialization_guard,
    replay_compiled_kernels,
    replay_prepared_cache,
    run_prepared_kernel_bench,
)

_FAKE_PREPARED_CHILD = textwrap.dedent(
    r"""
    import json
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
    command = json.loads(reader.readline())
    if command.get("type") == "CANCEL":
        raise SystemExit(0)
    if command.get("type") != "ASSIGN":
        raise RuntimeError(f"unexpected command: {command}")

    gpu_started = time.time()
    time.sleep(float(workload.get("gpu_s", 0.05)))
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
            "type": "RESULT",
            "gpu_started": gpu_started,
            "gpu_finished": gpu_finished,
            "result": {
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
        bench_run,
        "_active_strangers",
        active_strangers or (lambda *_args, **_kwargs: {}),
    )
    pool = GpuPool(allowed=set(gpu_indices))
    monkeypatch.setattr(
        pool,
        "_all_gpus",
        lambda: [(index, f"GPU-{index}") for index in gpu_indices],
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
        compile_profile={"num_sms": 148},
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
            (
                record["phase_timestamps"]["gpu_started"],
                record["phase_timestamps"]["gpu_finished"],
            )
            for record in records
            if record["gpu"] == gpu
        )
        assert all(current[0] >= previous[1] for previous, current in pairwise(per_gpu))

    logs = "\n".join(path.read_text() for path in (tmp_path / "logs").glob("*.log"))
    assert "stdout noise only" in logs
    assert "compiler/reference log after assignment" in logs


def test_pipeline_assigns_a_complete_multigpu_claim_before_gpu_stage(monkeypatch, tmp_path: Path):
    records, retries, _pipeline = _fake_pipeline(
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


def test_pipeline_fail_fast_cancels_nonterminal_children(monkeypatch, tmp_path: Path):
    workloads = [
        {"kernel": "fake", "config": "fail", "num_gpus": 1, "mode": "fail_prepare"},
        {"kernel": "fake", "config": "slow", "num_gpus": 1, "prepare_s": 5.0},
        {"kernel": "fake", "config": "never", "num_gpus": 1},
    ]

    records, retries, _pipeline = _fake_pipeline(
        monkeypatch,
        tmp_path,
        workloads,
        max_prepare_processes=2,
        ready_backlog=2,
    )

    assert retries == []
    assert len(records) == 1
    assert records[0]["config"] == "fail"
    assert records[0]["status"] == "FAIL"
    assert "synthetic prepare failure" in records[0]["error"]
    assert bench_run._BenchPidRegistry._roots == set()


def test_pipeline_fail_fast_cancels_ready_and_running_children(monkeypatch, tmp_path: Path):
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
        monkeypatch,
        tmp_path,
        workloads,
        max_prepare_processes=3,
        ready_backlog=3,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert retries == []
    assert len(records) == 1
    assert records[0]["config"] == "fail"
    assert bench_run._BenchPidRegistry._roots == set()


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
        compile_profile={"num_sms": 148},
        max_prepare_processes=1,
        ready_backlog=1,
    )

    assert retries == []
    assert records[0]["status"] == "ok"
    assert records[0]["phase_timestamps"]["assigned"] > records[0]["phase_timestamps"]["ready"]
    assert any(
        entry["occupied_gpu_indices"] == ["0"]
        for entry in pipeline["external_occupancy_timeline"]
    )
    assert pipeline["external_occupancy_timeline"][-1]["occupied_gpu_indices"] == []
    assert bench_run._BenchPidRegistry._roots == set()


def test_interference_retry_uses_a_fresh_child(monkeypatch, tmp_path: Path):
    samples = 0

    def active_strangers(*_args, **_kwargs):
        nonlocal samples
        samples += 1
        return {4242: 100.0} if samples == 1 else {}

    records, retries, _pipeline = _fake_pipeline(
        monkeypatch,
        tmp_path,
        [{"kernel": "fake", "config": "retry", "num_gpus": 1}],
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
    assert records[0]["process_pid"] != retries[0]["process_pid"]
    attempt_logs = sorted((tmp_path / "logs").glob("fake__retry__a*.log"))
    assert [path.stem for path in attempt_logs] == ["fake__retry__a1", "fake__retry__a2"]


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
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
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
        (
            "fake",
            "two",
            str(duplicate),
            duplicate.stat().st_mtime_ns,
            duplicate.stat().st_size,
        ),
    )
    with pytest.raises(ValueError, match="duplicate kernel registry name"):
        registry._build_kernel_index(duplicate_snapshot)

    invalid_snapshot = (
        (
            "fake",
            "invalid",
            str(invalid),
            invalid.stat().st_mtime_ns,
            invalid.stat().st_size,
        ),
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
    first_snapshot = (
        ("fake", "kernel", str(source), first_stat.st_mtime_ns, first_stat.st_size),
    )
    first, _ = registry._build_kernel_index(first_snapshot)

    source.write_text(
        'KERNEL_META = {"name": "after_longer", "category": "fake", '
        '"compute_capability": 10}\n'
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
    mismatched.write_text(valid.read_text().replace(
        "return build_launch(_spec_for(config), a=data)",
        "return build_launch(_spec_for(other_config), a=data)",
    ))
    with pytest.raises(TypeError, match=r"canonical _spec_for\(config\)"):
        _audit_strict_cache_adapter_source("mismatched", mismatched)

    bypassed = tmp_path / "bypassed_launch.py"
    bypassed.write_text(valid.read_text().replace(
        "return _tirx_launch({}, config)",
        "return object()",
    ))
    with pytest.raises(TypeError, match=r"exactly one _tirx_launch\(\) call"):
        _audit_strict_cache_adapter_source("bypassed", bypassed)


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
    "missing_field",
    ["round_samples", "errors", "timer", "benchmark_protocol"],
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
def test_local_timer_families_emit_the_strict_measurement_schema(
    monkeypatch, timer, timer_attr
):
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
        references={"reference": lambda: (lambda: None)},
        timer=timer,
        rounds=5,
        cooldown_s=1.0,
    )
    _finalize_bench_record(result, rounds=5, cooldown=1.0)

    assert result["status"] == "ok"
    assert result["timer"] == timer
    assert result["benchmark_protocol"]["order"] == ["tirx", "reference"]
    assert result["round_samples"] == {
        "tirx": [1.0] * 5,
        "reference": [1.0] * 5,
    }


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
        {"tirx": lambda: None, "reference": lambda: None},
        {},
        distributed,
        rounds=5,
        cooldown_s=1.0,
    )
    _finalize_bench_record(result, rounds=5, cooldown=1.0)

    assert result["status"] == "ok"
    assert result["round_samples"] == {
        "tirx": [12.0] * 5,
        "reference": [12.0] * 5,
    }
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
        {
            "tirx": lambda: launches.append("tirx"),
            "deepgemm": lambda: launches.append("deepgemm"),
        },
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
        "impls": {
            "tirx": sum(tirx) / len(tirx),
            "deepgemm": sum(deepgemm) / len(deepgemm),
        },
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

    assert result["round_samples"] == {
        "tirx": [5.0, 7.0],
        "deepgemm": [4.0, 8.0],
    }
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
        lambda world_size: events.append(("validate_visible", world_size)),
    )

    def launch(library_path, *, world_size, worker, mode, worker_kwargs):
        events.append(("launch_ranks", library_path, world_size, mode, worker_kwargs))
        return {"status": "OK"}

    monkeypatch.setattr(_runtime, "_run_distributed_library", launch)
    result = prepared.run_gpu(timer="kineto", rounds=5, cooldown_s=1.0)

    assert result == {"status": "OK"}
    assert events[0] == ("validate_visible", 4)
    assert events[1][0] == "launch_ranks"
    assert events[1][4]["rounds"] == 5
    assert events[1][4]["cooldown_s"] == 1.0
    assert events[-1] == "cleanup"


def test_distributed_rank_cleanup_runs_when_worker_fails(monkeypatch):
    events = []
    fake_runtime = SimpleNamespace(barrier=lambda: events.append("barrier"))
    fake_dist = SimpleNamespace(
        init_process_group=lambda **_kwargs: events.append("init_process_group"),
        is_available=lambda: True,
        is_initialized=lambda: True,
        destroy_process_group=lambda: events.append("destroy_process_group"),
    )
    fake_cuda = SimpleNamespace(set_device=lambda rank: events.append(("set_device", rank)))
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
    monkeypatch.setattr(_runtime, "dist", fake_dist)
    monkeypatch.setattr(_runtime, "tvm", fake_tvm)
    monkeypatch.setattr(_runtime, "_broadcast_nvshmem_uid", lambda _rank: object())
    monkeypatch.setattr(_runtime, "_create_runtime", lambda *_args: fake_runtime)
    monkeypatch.setattr(
        _runtime,
        "_cleanup_runtime",
        lambda runtime: events.append(("cleanup_runtime", runtime)),
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
        )

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


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        ([], "no terminal workload records"),
        (
            [{"status": "FAIL", "phase_timestamps": {"process_started": 0.0}}],
            (
                "no workload produced an ok result with a complete timeline and valid "
                "GPU assignment"
            ),
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
        "assignment_handoff_s",
        "ready_starvation_s",
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
