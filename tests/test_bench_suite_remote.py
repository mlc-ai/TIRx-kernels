# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Unit tests for the kcoral bench-suite backend (no network, no kcoral install)."""

from __future__ import annotations

import ast
import io
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tirx_kernels.bench_suite import remote
from tirx_kernels.bench_suite import run as bench_run

# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeKCoralError(Exception):
    def __init__(self, status_code, message, *, kind=None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.kind = kind
        self.message = message


class _FakeTransportError(Exception):
    pass


class _FakeProtocolError(Exception):
    pass


class _RecordingProgram:
    def __init__(self):
        self.instructions = []

    def upload(self, *, id, kind, source=None, value=None):
        self.instructions.append(("upload", id, kind))
        return ("ref", id)

    def get_function(self, *, id, module, name, cpu_only=False):
        self.instructions.append(("get_function", id, name, cpu_only))
        return ("ref", id)

    def run(self, *, id, fn, args):
        self.instructions.append(("run", id, fn, tuple(args)))
        return ("ref", id)

    def return_(self, *, key, value):
        self.instructions.append(("return", key, value))


@dataclass
class _FakeOutcome:
    status: str
    request_id: str = "req-1"
    queue_ms: float = 0.1
    elapsed_ms: float = 5000.0
    lease_wait_ms: float = 100.0
    lease_held_ms: float = 3400.0
    results: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error: dict | None = None


def _api():
    return remote.KcoralApi(
        Client=None,
        Program=_RecordingProgram,
        KCoralError=_FakeKCoralError,
        TransportError=_FakeTransportError,
        ProtocolError=_FakeProtocolError,
    )


def _profile(**probe):
    base = {
        "device": {
            "uuid": "GPU-abc",
            "name": "B200",
            "multi_processor_count": 148,
            "arch": "sm_100a",
        },
        "versions": {"tvm": "0.26.dev0"},
        "tvm": {"git_label": "f6726b02", "tirx_tree": "sha256:deadbeef"},
        "baselines": {"torch": {"version": "2.14"}},
    }
    base.update(probe)
    return remote.ServerProfile(
        url="http://server:1",
        health={"status": "ok", "gpu_count": 1, "target": {"arch": "sm_100a"}, "workers": [{}] * 3},
        probe=base,
    )


def _tree():
    return remote.TreeArchive(
        name="tirx_kernels", sha256="a" * 64, data=b"gz", file_count=1, byte_count=1
    )


def _workload():
    return {"kernel": "rmsnorm", "config": "hs128_bs32", "num_gpus": 1}


def _bench_result(rounds=3):
    return {
        "kernel": "rmsnorm",
        "label": "hs128_bs32",
        "round_samples": {"tir": [2.4, 2.41, 2.39][:rounds]},
        "errors": {},
        "timer": "proton",
        "benchmark_protocol": {
            "rounds": rounds,
            "round_aggregate": "mean",
            "cooldown_s": 0.0,
            "order": ["tir"],
        },
    }


def _submission(outcome=None, error=None, attempts=1, category=None):
    return remote.Submission(
        outcome=outcome,
        error=error,
        attempts=attempts,
        started_at="2026-09-11T00:00:00+00:00",
        finished_at="2026-09-11T00:00:05+00:00",
        wall_s=5.0,
        error_category=category,
    )


def _record(submission, **overrides):
    kwargs = dict(
        profile=_profile(),
        tree_sha256="a" * 64,
        before_tree_sha256=None,
        prepare_mode="cpu",
        rounds=3,
        cooldown=0.0,
        references_enabled=False,
        request_timeout_s=1800.0,
        log_path=Path("/tmp/x.log"),
    )
    kwargs.update(overrides)
    return remote.outcome_to_record(_workload(), submission, **kwargs)


# ── tree archive ─────────────────────────────────────────────────────────────


def _make_package(root: Path) -> Path:
    pkg = root / "tirx_kernels"
    (pkg / "basic").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "basic" / "rmsnorm.py").write_text("x = 1\n")
    (pkg / "basic" / "kernel.cu").write_text("// cuda\n")
    (pkg / "basic" / "__pycache__").mkdir()
    (pkg / "basic" / "__pycache__" / "rmsnorm.cpython-311.pyc").write_bytes(b"\x00")
    (pkg / "basic" / "stale.pyc").write_bytes(b"\x00")
    (pkg / "bench_suite").mkdir()
    (pkg / "bench_suite" / "baseline.json").write_text("{}")
    (pkg / "bench_suite" / "config.yaml").write_text("kernel: rmsnorm\n")
    return pkg


def test_build_tree_archive_is_deterministic_and_excludes_caches(tmp_path):
    pkg = _make_package(tmp_path)
    first = remote.build_tree_archive(pkg)
    (pkg / "basic" / "rmsnorm.py").touch()  # mtime changes must not change the archive
    second = remote.build_tree_archive(pkg)
    assert first.sha256 == second.sha256
    assert first.data == second.data
    with tarfile.open(fileobj=io.BytesIO(first.data), mode="r:gz") as archive:
        names = archive.getnames()
    assert names == sorted(names)
    assert "tirx_kernels/basic/rmsnorm.py" in names
    assert "tirx_kernels/basic/kernel.cu" in names
    assert "tirx_kernels/bench_suite/config.yaml" in names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert "tirx_kernels/bench_suite/baseline.json" not in names
    assert first.file_count == len(names)


def test_build_tree_archive_content_change_changes_hash(tmp_path):
    pkg = _make_package(tmp_path)
    first = remote.build_tree_archive(pkg)
    (pkg / "basic" / "rmsnorm.py").write_text("x = 2\n")
    assert remote.build_tree_archive(pkg).sha256 != first.sha256


# ── program shape ────────────────────────────────────────────────────────────


def _spec(side="after", prepare_mode="cpu"):
    return remote.workload_spec(
        {"kernel": "rmsnorm", "config": "hs128_bs32", "timer": "proton", "warmup": None},
        cuda_arch="sm_100a",
        num_sms=148,
        side=side,
        references_enabled=False,
        prepare_mode=prepare_mode,
        rounds=5,
        cooldown=0.0,
    )


def test_workload_spec_only_forwards_explicit_bench_overrides():
    spec = _spec()
    assert spec["bench"] == {"rounds": 5, "cooldown": 0.0, "timer": "proton"}
    assert spec["cuda_compile_mode"] == "nvcc"
    assert spec["side"] == "after"
    with pytest.raises(ValueError):
        _spec(prepare_mode="tpu")


@pytest.mark.parametrize("prepare_mode, cpu_only", [("cpu", True), ("gpu", False)])
def test_workload_program_shape(prepare_mode, cpu_only):
    program = remote.build_workload_program(
        _RecordingProgram(),
        tree=_tree(),
        before_tree=None,
        shim="def prepare(): pass",
        spec=_spec(prepare_mode=prepare_mode),
    )
    ops = program.instructions
    assert ops[0] == ("upload", "tree", "bytes")
    assert ops[1] == ("upload", "shim", "module")
    assert ops[2] == ("get_function", "prepare_fn", "prepare", cpu_only)
    assert ops[3] == ("get_function", "run_fn", "run", False)
    assert ops[4][:2] == ("run", "prepare")
    assert ops[4][3][1] is None  # no before tree on the after side
    assert ops[5] == ("return", "prepare", ("ref", "prepare"))
    assert ops[6][:2] == ("run", "result")
    assert ops[7] == ("return", "result", ("ref", "result"))


def test_before_side_uploads_both_trees():
    before = remote.TreeArchive(
        name="tirx_kernels", sha256="b" * 64, data=b"gz2", file_count=1, byte_count=1
    )
    program = remote.build_workload_program(
        _RecordingProgram(), tree=_tree(), before_tree=before, shim="", spec=_spec(side="before")
    )
    ops = program.instructions
    assert ops[1] == ("upload", "before_tree", "bytes")
    run_prepare = next(op for op in ops if op[0] == "run" and op[1] == "prepare")
    assert run_prepare[3][1] == ("ref", "before_tree")
    with pytest.raises(ValueError):
        remote.build_workload_program(
            _RecordingProgram(), tree=_tree(), before_tree=None, shim="", spec=_spec(side="before")
        )


# ── outcome mapping ──────────────────────────────────────────────────────────


def test_outcome_completed_finalizes_samples():
    payload = json.dumps(
        {"result": _bench_result(), "device": {"uuid": "GPU-xyz"}, "timings": {"gpu_s": 3.4}}
    )
    outcome = _FakeOutcome(
        "COMPLETED", results={"result": payload, "prepare": json.dumps({"cupti": {"target": "/x"}})}
    )
    row = _record(_submission(outcome))
    assert row["status"] == "ok"
    assert row["impls"] == {"tir": pytest.approx(2.4)}
    assert row["aggregated"] == {"rounds": 3, "method": "mean"}
    assert row["execution_mode"] == remote.EXECUTION_MODE
    assert row["process_model"] == remote.PROCESS_MODEL
    assert row["physical_gpu_uuids"] == ["GPU-xyz"]
    assert row["num_gpus"] == 1
    assert row["remote"]["lease_held_ms"] == 3400.0
    assert row["remote"]["cupti"] == {"target": "/x"}
    assert "retry_in_place" not in row and "interfered" not in row


def test_outcome_completed_round_mismatch_fails_row():
    payload = json.dumps({"result": _bench_result(rounds=2)})
    row = _record(_submission(_FakeOutcome("COMPLETED", results={"result": payload})), rounds=3)
    assert row["status"] == "FAIL"
    assert "round count" in row["error"]


def test_outcome_skip_row():
    payload = json.dumps(
        {
            "result": {
                "kernel": "rmsnorm",
                "label": "hs128_bs32",
                "status": "SKIP",
                "reason": "no ref",
            }
        }
    )
    row = _record(_submission(_FakeOutcome("COMPLETED", results={"result": payload})))
    assert row["status"] == "SKIP"
    assert row["reason"] == "no ref"
    assert row["physical_gpu_uuids"] == ["GPU-abc"]


def test_outcome_failed_gpu_access_hint():
    outcome = _FakeOutcome(
        "FAILED",
        error={
            "kind": "gpu_access",
            "message": "cpu_only function called cuInit",
            "instruction_id": "prepare",
            "cuda_call": "cuInit",
            "location": "<uploaded>:10",
            "traceback": "Traceback ...",
        },
    )
    row = _record(_submission(outcome))
    assert row["status"] == "FAIL"
    assert row["error"].startswith("prepare: gpu_access: cpu_only function called cuInit")
    assert "--prepare gpu" in row["error"]
    assert "Traceback" in row["error"]
    assert row["remote"]["error"]["instruction_id"] == "prepare"


def test_outcome_failed_gpu_phase():
    outcome = _FakeOutcome(
        "FAILED", error={"kind": "runtime", "message": "boom", "instruction_id": "result"}
    )
    row = _record(_submission(outcome))
    assert row["error"].startswith("gpu: runtime: boom")


def test_outcome_timeout_busy_transport():
    timeout = _record(_submission(error=_FakeKCoralError(504, "timed out", kind="timeout")))
    assert timeout["status"] == "FAIL" and timeout["error"].startswith("timeout:")
    busy = _record(_submission(error=_FakeKCoralError(503, "saturated", kind="busy"), attempts=4))
    assert busy["error"].startswith("busy: server saturated after 4 attempt(s)")
    transport = _record(
        _submission(error=_FakeTransportError("connection reset"), category="transport")
    )
    assert transport["error"].startswith("transport: connection reset")
    protocol = _record(_submission(error=_FakeProtocolError("bad body"), category="protocol"))
    assert protocol["error"].startswith("protocol: _FakeProtocolError")


# ── retries ──────────────────────────────────────────────────────────────────


class _FlakyClient:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def execute(self, program, *, timeout_seconds, output_limit_bytes):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return _FakeOutcome(
            "COMPLETED", results={"result": json.dumps({"result": _bench_result()})}
        )


def test_execute_with_retry_backs_off_on_busy():
    api = _api()
    client = _FlakyClient(
        [_FakeKCoralError(503, "busy", kind="busy")] * 2 + [_FakeTransportError("reset")]
    )
    sleeps = []
    policy = remote.RetryPolicy(busy_backoff_s=(1.0, 2.0), jitter=0.0, transport_backoff_s=0.5)
    submission = remote.execute_with_retry(
        api, client, object(), timeout_s=10, policy=policy, sleep=sleeps.append, clock=lambda: 0.0
    )
    assert submission.outcome is not None and submission.error is None
    assert submission.attempts == 4
    assert submission.busy_retries == 2
    assert submission.transport_retries == 1
    assert sleeps == [1.0, 2.0, 0.5]


def test_execute_with_retry_gives_up():
    api = _api()
    client = _FlakyClient([_FakeTransportError("reset")] * 5)
    policy = remote.RetryPolicy(transport_attempts=2, jitter=0.0)
    submission = remote.execute_with_retry(
        api, client, object(), timeout_s=10, policy=policy, sleep=lambda s: None
    )
    assert isinstance(submission.error, _FakeTransportError)
    assert client.calls == 2
    ticks = iter([0.0, 0.0, 10_000.0, 10_000.0])
    client = _FlakyClient([_FakeKCoralError(503, "busy", kind="busy")] * 3)
    submission = remote.execute_with_retry(
        api,
        client,
        object(),
        timeout_s=10,
        policy=policy,
        sleep=lambda s: None,
        clock=lambda: next(ticks),
    )
    assert isinstance(submission.error, _FakeKCoralError)
    assert submission.error.status_code == 503


# ── shim, provenance, pipeline ───────────────────────────────────────────────


def test_shim_source_is_stdlib_only_and_defines_entry_points():
    source = remote.shim_source()
    assert source.startswith("# SPDX-License-Identifier: Apache-2.0\n")
    module = ast.parse(source)
    names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {"probe", "prepare", "run"} <= names
    for node in module.body:
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert roots <= sys.stdlib_module_names, roots


def test_pipeline_metadata_carries_gate_provenance():
    pipeline = remote.pipeline_metadata(
        profile=_profile(),
        tree=_tree(),
        rounds=5,
        cooldown=0.0,
        prepare_mode="cpu",
        max_in_flight=3,
        request_timeout_s=1800.0,
        records=[{"status": "FAIL"}, {"status": "ok"}],
    )
    assert pipeline["execution_mode"] == "remote"
    assert pipeline["process_model"] == "kcoral_worker_per_request"
    assert pipeline["measurement_protocol"]["rounds"] == 5
    assert pipeline["measurement_protocol"]["is_default"] is True
    assert pipeline["failure_count"] == 1
    assert pipeline["server"]["arch"] == "sm_100a"
    assert "interference_retries" not in pipeline


def test_merge_provenance_takes_tir_from_server():
    git, tree = remote.merge_provenance(
        _profile(),
        local_git={"tir": "7a8c0703-dirty", "tirx-kernels": "2eff7b34", "tirx-bench-ci": None},
        local_tree={"tir:python/tvm/tirx": "local", "tirx-kernels:tirx_kernels": "kt"},
    )
    assert git == {"tir": "f6726b02", "tirx-kernels": "2eff7b34", "tirx-bench-ci": None}
    assert tree == {"tir:python/tvm/tirx": "sha256:deadbeef", "tirx-kernels:tirx_kernels": "kt"}
    fallback = _profile(tvm={"git_label": None, "tirx_tree": None})
    assert fallback.tvm_label == "tvm-0.26.dev0"
    assert fallback.tirx_tree == "tvm-0.26.dev0"


def test_validate_health_rejects_bad_servers():
    with pytest.raises(remote.RemoteBenchError):
        remote.validate_health({"status": "down"}, "u")
    with pytest.raises(remote.RemoteBenchError):
        remote.validate_health({"status": "ok", "gpu_count": 0, "target": {"arch": "sm_100a"}}, "u")
    with pytest.raises(remote.RemoteBenchError):
        remote.validate_health({"status": "ok", "gpu_count": 1, "target": {}}, "u")
    assert remote.default_max_in_flight({"workers": [{}] * 8}) == 8
    assert remote.default_max_in_flight({}) == 1


def test_main_rejects_multi_gpu_workloads_before_connecting(monkeypatch, tmp_path):
    workloads = tmp_path / "w.yaml"
    workloads.write_text(
        "workloads:\n  - {kernel: allgather_gemm, config: tp4, num_gpus: 4}\n"
        "  - {kernel: rmsnorm, config: hs128_bs32}\n"
    )
    monkeypatch.setattr(remote, "connect", lambda *a, **k: pytest.fail("connected"))
    monkeypatch.setattr(
        sys, "argv", ["bench_suite", "--workloads", str(workloads), "--out-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as info:
        bench_run.main()
    assert info.value.code == 2


def test_run_reexports_provenance_helpers():
    from tirx_kernels.bench_suite import provenance

    assert bench_run.collect_repo_git is provenance.collect_repo_git
    assert bench_run.package_provenance is provenance.package_provenance
    assert bench_run.git_label is provenance.git_label
    assert bench_run._tir_repo_root is provenance.tir_repo_root
