# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tirx_kernels.bench_suite import _local_only_runtime, run
from tirx_kernels.bench_suite.run import (
    CONFIG_DIR,
    DEFAULT_COOLDOWN_S,
    DEFAULT_MEM_THRESHOLD,
    DEFAULT_PAIRED_SPEEDUP_THRESHOLD,
    DEFAULT_UTIL_THRESHOLD,
    IR_BUILDER_MIGRATION_ROUNDS,
    GpuPool,
    _assert_checkout_unchanged,
    _capture_checkout_snapshot,
    _checkout_pythonpath,
    _daily_bench_env,
    _derive_ir_builder_migration_scope,
    _direct_speedup_pct,
    _finalize_bench_record,
    _local_samples,
    _materialize_local_only_runtime_hook,
    _next_run_id,
    _nvrtc_preflight,
    _paired_local_only_env,
    _paired_order,
    _read_bench_result,
    _run_subprocess_monitored,
    _validate_ir_builder_migration_gate_options,
    _validate_ir_builder_migration_workloads,
    _validate_paired_config_resolution,
    load_all_config_dir,
    load_config_dir,
    run_one_paired,
    run_scheduled_jobs,
)


def test_canonical_config_loader_covers_every_declared_benchmark_config():
    all_configs = load_all_config_dir(CONFIG_DIR)
    daily_configs = load_config_dir(CONFIG_DIR)

    identities = [(row["kernel"], row["config"]) for row in all_configs]
    assert len(all_configs) == 676
    assert len(set(identities)) == len(identities)
    assert sum(row["default"] for row in all_configs) == 195
    assert len(daily_configs) == 195
    assert {(row["kernel"], row["config"]) for row in daily_configs} == {
        (row["kernel"], row["config"]) for row in all_configs if row["default"]
    }
    exclusive = [row for row in all_configs if row.get("exclusive_resource") == "nvshmem"]
    assert len(exclusive) == 8
    assert {row["kernel"] for row in exclusive} == {"allgather_gemm", "gemm_reduce_scatter"}


def test_canonical_loader_rejects_duplicate_identity_across_files(tmp_path: Path):
    body = """
kernel: duplicated
configs:
  - config: same
    default: true
"""
    (tmp_path / "a.yaml").write_text(body)
    (tmp_path / "b.yaml").write_text(body)

    with pytest.raises(ValueError, match="duplicate bench config"):
        load_all_config_dir(tmp_path)


def test_canonical_loader_requires_explicit_default_flag(tmp_path: Path):
    (tmp_path / "kernel.yaml").write_text(
        """
kernel: missing_default
configs:
  - config: case
"""
    )

    with pytest.raises(ValueError, match="must set default"):
        load_all_config_dir(tmp_path)


def test_paired_checkout_environment_precedes_existing_pythonpath(tmp_path: Path):
    (tmp_path / "tirx_kernels").mkdir()
    (tmp_path / "tirx_kernels" / "__init__.py").write_text("")

    env = _checkout_pythonpath(tmp_path, {"PYTHONPATH": "/shared/dependencies"})

    assert env["PYTHONPATH"].split(":") == [str(tmp_path.resolve()), "/shared/dependencies"]


def test_daily_bench_environment_clears_local_only_state():
    env = _daily_bench_env(
        {"PATH": "/bin", "TIRX_BENCH_LOCAL_ONLY": "1", "TIRX_BENCH_LOCAL_ONLY_KERNEL": "fixture"}
    )

    assert env == {"PATH": "/bin"}


def test_gpu_pool_uses_global_framebuffer_and_fails_closed_on_missing_rows(monkeypatch):
    pool = GpuPool(mem_threshold=0.5)

    def fake_nvidia_smi(args):
        if args == ["--query-gpu=index,uuid"]:
            return ["0, GPU-0", "1, GPU-1", "2, GPU-2"]
        if args == ["--query-gpu=index,utilization.gpu"]:
            return ["0, 0", "1, 0", "2, 0"]
        if args == ["--query-gpu=index,memory.used,memory.total"]:
            return ["0, 116, 183359", "1, 7520, 183359"]
        raise AssertionError(args)

    monkeypatch.setattr(pool, "_nvidia_smi", fake_nvidia_smi)

    assert pool._mem_used_pct()["0"] == pytest.approx(116 / 183359 * 100)
    assert pool._occupied_indices() == {"1", "2"}


def test_gpu_pool_waits_for_framebuffer_release(monkeypatch):
    pool = GpuPool(mem_threshold=0.5)
    samples = iter([{"0": 10.0}, {"0": 0.1}])
    monkeypatch.setattr(pool, "_mem_used_pct", lambda: next(samples))

    assert pool.wait_for_memory_release(("0",), timeout_s=1.0, poll_interval_s=0.0) == {}


def test_gpu_pool_allows_observed_nvshmem_teardown_tail(monkeypatch):
    pool = GpuPool(mem_threshold=0.5)
    samples = iter([{"0": 10.0}, {"0": 0.1}])
    times = iter([0.0, 240.0, 240.0])
    monkeypatch.setattr(pool, "_mem_used_pct", lambda: next(samples))
    monkeypatch.setattr(run.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(run.time, "sleep", lambda _seconds: None)

    assert pool.wait_for_memory_release(("0",)) == {}


def test_gpu_pool_framebuffer_release_timeout_fails_closed(monkeypatch):
    pool = GpuPool(mem_threshold=0.5)
    times = iter([100.0, 101.0])
    monkeypatch.setattr(pool, "_mem_used_pct", lambda: {})
    monkeypatch.setattr(run.time, "monotonic", lambda: next(times))

    busy = pool.wait_for_memory_release(("0",), timeout_s=0.5, poll_interval_s=0.0)

    assert math.isinf(busy["0"])


def _write_fake_bench_packages(checkout: Path) -> None:
    for package in (checkout / "tvm", checkout / "tvm" / "tirx", checkout / "tirx_kernels"):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
    (checkout / "tvm" / "tirx" / "bench.py").write_text(
        """
def bench(funcs, *, references=None, **_kwargs):
    for builder in (references or {}).values():
        builder()
    return {
        "impls": {name: 1.0 for name in funcs},
        "round_samples": {name: [1.0] for name in funcs},
        "errors": {},
    }
"""
    )


def test_local_only_runtime_hook_skips_references_and_reaches_spawned_children(tmp_path: Path):
    checkout = tmp_path / "checkout"
    _write_fake_bench_packages(checkout)
    hook_dir = _materialize_local_only_runtime_hook(tmp_path / "cache")
    env = _paired_local_only_env(
        checkout, hook_dir, {"kernel": "fixture", "config": "case"}, {"PATH": os.environ["PATH"]}
    )
    script = tmp_path / "spawn_contract.py"
    script.write_text(
        """
import json
import multiprocessing as mp


def child(queue):
    from tvm.tirx.bench import bench

    def forbidden_reference():
        raise RuntimeError("reference builder ran")

    queue.put(bench({"tirx": lambda: None}, references={"reference": forbidden_reference}))


if __name__ == "__main__":
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    process.join(30)
    if process.exitcode != 0:
        raise SystemExit(process.exitcode)
    print(json.dumps(queue.get(), sort_keys=True))
"""
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=40,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["local_only"] is True
    assert result["impls"] == {"tirx": 1.0}
    assert result["round_samples"] == {"tirx": [1.0]}


def test_local_only_gemmcomm_hook_replaces_baseline_construction(tmp_path: Path):
    checkout = tmp_path / "checkout"
    _write_fake_bench_packages(checkout)
    package = checkout / "tirx_kernels" / "basic" / "utils"
    package.mkdir(parents=True)
    for parent in (package.parent, package):
        (parent / "__init__.py").write_text("")
    (package / "_baselines.py").write_text(
        """
def create_baseline_suite(*_args, **_kwargs):
    raise RuntimeError("real GemmComm baseline suite was constructed")
"""
    )
    hook_dir = _materialize_local_only_runtime_hook(tmp_path / "cache")
    env = _paired_local_only_env(
        checkout,
        hook_dir,
        {"kernel": "allgather_gemm", "config": "case"},
        {"PATH": os.environ["PATH"]},
    )
    code = """
import json
from tirx_kernels.basic.utils._baselines import create_baseline_suite

class Runtime:
    def __init__(self):
        self.barriers = 0
    def barrier(self):
        self.barriers += 1

runtime = Runtime()
suite = create_baseline_suite(runtime)
suite.close()
print(json.dumps({
    "references": suite.references(),
    "metadata": suite.metadata(),
    "barriers": runtime.barriers,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {"barriers": 1, "metadata": {"local_only": True}, "references": {}}


def test_local_only_tinygemm_bench_never_constructs_flashinfer(monkeypatch):
    counters = {"launches": 0, "references": 0}

    class Cuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_device):
            return SimpleNamespace(multi_processor_count=148)

        @staticmethod
        def synchronize():
            return None

    class Executable:
        def __call__(self, *_args):
            counters["launches"] += 1

    module = SimpleNamespace(
        _require_sm100=lambda: None,
        prepare_data=lambda batch, output, inner: {"B": batch, "O": output, "K": inner},
        torch=SimpleNamespace(cuda=Cuda()),
        _select_stage=lambda _B, _O, _K, _sms: 4,
        _compile_executable=lambda _B, _O, _stage, _pdl: Executable(),
        _tirx_args=lambda _case: (object(),),
        _run_flashinfer=lambda *_args: counters.__setitem__("references", 1),
    )

    def fake_bench(funcs, **_kwargs):
        assert set(funcs) == {"tirx"}
        funcs["tirx"]()
        return {
            "impls": {"tirx": 2.0},
            "round_samples": {"tirx": [2.0]},
            "errors": {},
            "local_only": True,
        }

    monkeypatch.setattr("tvm.tirx.bench.bench", fake_bench)
    result = _local_only_runtime._local_only_tinygemm_bench(
        module, 1, 128, 720, rounds=1, cooldown_s=0.0
    )

    assert counters == {"launches": 2, "references": 0}
    assert result["local_only"] is True


def test_local_only_selective_state_bench_never_runs_reference(monkeypatch):
    counters = {"launches": 0, "references": 0}

    class Executable:
        def __call__(self, *_args):
            counters["launches"] += 1

    module = SimpleNamespace(
        prepare_data=lambda **kwargs: dict(kwargs),
        get_kernel=lambda **_kwargs: object(),
        _tirx_args=lambda _case: (object(),),
        _run_reference=lambda *_args: counters.__setitem__("references", 1),
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
    )

    monkeypatch.setattr("tirx_kernels.runner.compile_kernel", lambda _kernel: Executable())

    def fake_bench(funcs, **_kwargs):
        assert set(funcs) == {"tirx"}
        funcs["tirx"]()
        return {
            "impls": {"tirx": 3.0},
            "round_samples": {"tirx": [3.0]},
            "errors": {},
            "local_only": True,
        }

    monkeypatch.setattr("tvm.tirx.bench.bench", fake_bench)
    result = _local_only_runtime._local_only_selective_state_bench(
        module, rounds=1, cooldown_s=0.0, batch_size=4
    )

    assert counters == {"launches": 2, "references": 0}
    assert result["local_only"] is True


def test_local_only_fp8_paged_mqa_bench_never_runs_reference(monkeypatch):
    counters = {"launches": 0, "references": 0}
    config = object()
    data = object()
    invocation = object()

    module = SimpleNamespace(
        _make_config=lambda **_kwargs: config,
        _compile_tirx_paged_mqa=lambda value: ("executable", value),
        _prepare_data=lambda value, *, compute_reference: (
            data if value is config and compute_reference is False else None
        ),
        _prepare_tirx_invocation=lambda value, *, executable: (
            invocation if value is data and executable == ("executable", config) else None
        ),
        _run_tirx_invocation=lambda value, prepared: counters.__setitem__(
            "launches", counters["launches"] + int(value is data and prepared is invocation)
        ),
        _run_deepgemm_paged_mqa=lambda *_args, **_kwargs: counters.__setitem__("references", 1),
        _make_sglang_cutedsl_runner=lambda *_args, **_kwargs: counters.__setitem__("references", 1),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
        ),
    )

    def fake_bench(funcs, **_kwargs):
        assert set(funcs) == {"tirx"}
        funcs["tirx"]()
        return {
            "impls": {"tirx": 4.0},
            "round_samples": {"tirx": [4.0]},
            "errors": {},
            "local_only": True,
        }

    monkeypatch.setattr("tvm.tirx.bench.bench", fake_bench)
    result = _local_only_runtime._local_only_fp8_paged_mqa_bench(
        module, rounds=1, cooldown_s=0.0, batch_size=1
    )

    assert counters == {"launches": 2, "references": 0}
    assert result["local_only"] is True


def test_local_only_nvfp4_gemm_bench_never_runs_flashinfer(monkeypatch):
    counters = {"launches": 0, "references": 0}

    class Target:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Executable:
        @staticmethod
        def mod(*_args):
            counters["launches"] += 1

    class Tensor:
        def item(self):
            return 1.0

        def to(self, *_args):
            return self

    module = SimpleNamespace(
        tir_ws_kernel=lambda *_args: object(),
        tvm=SimpleNamespace(
            target=SimpleNamespace(Target=lambda *_args: Target()),
            IRModule=lambda value: value,
            compile=lambda *_args, **_kwargs: Executable(),
        ),
        prepare_data=lambda *_args: (Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()),
        torch=SimpleNamespace(
            tensor=lambda *_args, **_kwargs: Tensor(),
            empty_like=lambda _value: Tensor(),
            float=object(),
            bfloat16=object(),
        ),
        flashinfer=SimpleNamespace(
            mm_fp4=lambda *_args, **_kwargs: counters.__setitem__("references", 1)
        ),
        _load_cublaslt_nvfp4_ext=lambda: counters.__setitem__("references", 1),
    )

    def fake_bench(funcs, **_kwargs):
        assert set(funcs) == {"tir"}
        funcs["tir"]()
        return {
            "impls": {"tir": 5.0},
            "round_samples": {"tir": [5.0]},
            "errors": {},
            "local_only": True,
        }

    monkeypatch.setattr("tvm.tirx.bench.bench", fake_bench)
    result = _local_only_runtime._local_only_nvfp4_gemm_bench(
        module, 1024, 1024, 1024, rounds=1, cooldown_s=0.0
    )

    assert counters == {"launches": 1, "references": 0}
    assert result["local_only"] is True


def test_local_only_megamoe_worker_constructs_and_times_only_tirx(monkeypatch):
    counters = {"cases": 0, "launches": 0, "cleanups": 0, "destroy": 0}

    class Config:
        num_processes = 1
        num_tokens = 2
        hidden = 128
        num_experts_per_rank = 1

        def validate(self):
            return None

    class Cuda:
        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def set_device(_device):
            return None

    class Distributed:
        @staticmethod
        def is_initialized():
            return False

    fake_torch = SimpleNamespace(
        cuda=Cuda(),
        distributed=Distributed(),
        get_default_device=lambda: "cpu",
        set_default_device=lambda _device: None,
    )
    deep_gemm = SimpleNamespace(
        utils=SimpleNamespace(dist=SimpleNamespace(init_dist=lambda _rank, _size: (0, 1, object())))
    )
    case = SimpleNamespace(config=Config())
    invocation = SimpleNamespace()

    def create_case(*_args):
        counters["cases"] += 1
        return case

    def launch(*_args):
        counters["launches"] += 1

    module = SimpleNamespace(
        MegaMoeConfig=lambda **_kwargs: Config(),
        SkipTest=RuntimeError,
        torch=fake_torch,
        load_deep_gemm_mega=lambda: (deep_gemm, "fixture"),
        _destroy_process_group=lambda: counters.__setitem__("destroy", counters["destroy"] + 1),
        create_case=create_case,
        _copy_inputs_into_symm_buffer=lambda _case: None,
        _prepare_tirx_invocation=lambda _case: invocation,
        _launch_tirx_mega_moe=launch,
        _cleanup_distinct_cases=lambda *_cases: counters.__setitem__(
            "cleanups", counters["cleanups"] + 1
        ),
    )

    def fake_bench(funcs, **_kwargs):
        assert set(funcs) == {"tirx"}
        funcs["tirx"]()
        return {
            "impls": {"tirx": 2.0},
            "round_samples": {"tirx": [2.0]},
            "errors": {},
            "timer": "event",
            "benchmark_protocol": {},
            "local_only": True,
        }

    monkeypatch.setattr("tvm.tirx.bench.bench", fake_bench)
    result = _local_only_runtime._local_only_megamoe_worker(
        module, 0, {"rounds": 1, "cooldown_s": 0.0}, "bench"
    )

    assert counters == {"cases": 1, "launches": 2, "cleanups": 1, "destroy": 1}
    assert result["local_only"] is True
    assert result["impls"] == {"tirx": 2.0}


def test_paired_result_requires_one_unambiguous_local_implementation():
    assert _local_samples({"round_samples": {"tirx": [1.25]}}) == ("tirx", [1.25])

    with pytest.raises(ValueError, match="exactly one"):
        _local_samples({"round_samples": {"tirx": [1.25], "tir_debug": [1.5]}})
    for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="invalid local samples"):
            _local_samples({"round_samples": {"tirx": [invalid]}})


def test_daily_finalization_rejects_reference_errors():
    row = {"round_samples": {"tirx": [1.25]}, "errors": {"missing_reference": "not installed"}}
    _finalize_bench_record(row, rounds=1)
    assert row["status"] == "FAIL"
    assert "missing_reference" in row["error"]


def test_paired_result_requires_explicit_local_only_and_no_extra_implementation(tmp_path: Path):
    path = tmp_path / "result.json"
    workload = {"kernel": "fixture", "config": "case"}
    row = {
        "kernel": "fixture",
        "label": "case",
        "impls": {"tirx": 1.25},
        "round_samples": {"tirx": [1.25]},
        "errors": {},
    }
    path.write_text(json.dumps({"results": [row]}))
    with pytest.raises(ValueError, match="local_only=true"):
        _read_bench_result(path, workload, rounds=1, require_local_only=True)

    row["local_only"] = True
    row["impls"]["reference"] = 1.0
    row["round_samples"]["reference"] = [1.0]
    path.write_text(json.dumps({"results": [row]}))
    parsed = _read_bench_result(path, workload, rounds=1, require_local_only=True)
    with pytest.raises(ValueError, match="must report only"):
        _local_samples(parsed, require_only=True)


def test_paired_result_preserves_real_failure_before_local_only_marker_check(tmp_path: Path):
    path = tmp_path / "result.json"
    row = {
        "kernel": "fixture",
        "label": "case",
        "status": "FAIL",
        "error": "reference import unexpectedly ran",
    }
    path.write_text(json.dumps({"results": [row]}))

    assert (
        _read_bench_result(
            path, {"kernel": "fixture", "config": "case"}, rounds=1, require_local_only=True
        )
        == row
    )


def test_paired_order_is_counterbalanced_and_speedup_uses_direct_times():
    assert [_paired_order(index) for index in range(5)] == [
        ("old", "current"),
        ("current", "old"),
        ("old", "current"),
        ("current", "old"),
        ("old", "current"),
    ]
    assert _direct_speedup_pct(99.0, 100.0) == pytest.approx(-1.0)
    assert _direct_speedup_pct(101.0, 100.0) == pytest.approx(1.0)
    for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="finite and positive"):
            _direct_speedup_pct(invalid, 100.0)


def test_ir_builder_migration_gate_rejects_scope_and_threshold_overrides(tmp_path: Path):
    valid = {
        "enabled": True,
        "old_checkout": tmp_path / "old",
        "workloads_path": None,
        "all_configs": False,
        "kernel_filter": None,
        "rounds": IR_BUILDER_MIGRATION_ROUNDS,
        "cooldown": DEFAULT_COOLDOWN_S,
        "speedup_threshold": DEFAULT_PAIRED_SPEEDUP_THRESHOLD,
        "util_threshold": DEFAULT_UTIL_THRESHOLD,
        "mem_threshold": DEFAULT_MEM_THRESHOLD,
        "check_imports": False,
    }
    _validate_ir_builder_migration_gate_options(**valid)

    invalid_overrides = {
        "old_checkout": None,
        "workloads_path": tmp_path / "subset.yaml",
        "all_configs": True,
        "kernel_filter": "one_kernel",
        "rounds": IR_BUILDER_MIGRATION_ROUNDS - 1,
        "cooldown": 0.0,
        "speedup_threshold": 0.0,
        "util_threshold": 100.0,
        "mem_threshold": 100.0,
        "check_imports": True,
    }
    for field, value in invalid_overrides.items():
        options = dict(valid)
        options[field] = value
        with pytest.raises(ValueError):
            _validate_ir_builder_migration_gate_options(**options)


def test_ir_builder_migration_gate_requires_exact_canonical_workload_sequence():
    canonical = load_config_dir(CONFIG_DIR)
    assert len(canonical) == 195
    required, scope = _derive_ir_builder_migration_scope(canonical)
    assert len(required) == 187
    assert scope["canonical_default_count"] == 195
    assert scope["required_count"] == 187
    assert scope["user_exempted_count"] == 8
    assert {row["status"] for row in scope["user_exempted"]} == {"user-exempted"}
    assert {row["kernel"] for row in scope["user_exempted"]} == {
        "allgather_gemm",
        "gemm_reduce_scatter",
    }
    assert all(
        "requires NVSHMEM/distributed communication libraries" in row["reasons"]
        for row in scope["user_exempted"]
    )
    _validate_ir_builder_migration_workloads(required, required)

    narrowed = required[:-1]
    with pytest.raises(ValueError, match="canonical"):
        _validate_ir_builder_migration_workloads(narrowed, required)


def test_ir_builder_gate_nvrtc_preflight_records_loaded_library(monkeypatch, tmp_path: Path):
    library = tmp_path / "libnvrtc.so.13.2.51"
    library.write_bytes(b"nvrtc fixture")
    builtins = tmp_path / "libnvrtc-builtins.so.13.2.51"
    builtins.write_bytes(b"nvrtc builtins fixture")
    monkeypatch.setenv(run.NVRTC_LIBRARY_ENV, str(library))
    monkeypatch.setenv(run.NVRTC_BUILTINS_LIBRARY_ENV, str(builtins))
    monkeypatch.setattr(run, "_query_nvrtc_version", lambda: (13, 2))
    monkeypatch.setattr(run, "_mapped_nvrtc_paths", lambda: [library.resolve()])
    monkeypatch.setattr(run, "_mapped_nvrtc_builtins_paths", lambda: [builtins.resolve()])

    evidence = _nvrtc_preflight()

    assert evidence["version"] == [13, 2]
    assert evidence["required_version"] == [13, 2]
    assert evidence["library_path"] == str(library.resolve())
    assert evidence["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert evidence["builtins_library_path"] == str(builtins.resolve())
    assert evidence["applies_to"] == ["old", "current"]


def test_ir_builder_gate_nvrtc_preflight_rejects_wrong_version(monkeypatch, tmp_path: Path):
    library = tmp_path / "libnvrtc.so.13.0.88"
    library.write_bytes(b"old nvrtc fixture")
    builtins = tmp_path / "libnvrtc-builtins.so.13.0"
    builtins.write_bytes(b"old nvrtc builtins fixture")
    monkeypatch.setenv(run.NVRTC_LIBRARY_ENV, str(library))
    monkeypatch.setenv(run.NVRTC_BUILTINS_LIBRARY_ENV, str(builtins))
    monkeypatch.setattr(run, "_query_nvrtc_version", lambda: (13, 0))
    monkeypatch.setattr(run, "_mapped_nvrtc_paths", lambda: [library.resolve()])
    monkeypatch.setattr(run, "_mapped_nvrtc_builtins_paths", lambda: [builtins.resolve()])

    with pytest.raises(ValueError, match="requires NVRTC"):
        _nvrtc_preflight()


def test_paired_config_resolution_requires_identical_actual_parameters():
    workloads = [{"kernel": "fixture", "config": "case"}]
    old_configs = [
        {
            "kernel": "fixture",
            "config": "case",
            "params": ["dict", [[["str", "size"], ["int", "128"]]]],
        }
    ]
    assert _validate_paired_config_resolution(workloads, old_configs, old_configs)

    current_configs = json.loads(json.dumps(old_configs))
    current_configs[0]["params"] = ["dict", [[["str", "size"], ["int", "256"]]]]
    with pytest.raises(ValueError, match="parameters differ"):
        _validate_paired_config_resolution(workloads, old_configs, current_configs)


def test_paired_checkout_snapshot_detects_source_changes(tmp_path: Path):
    package = tmp_path / "tirx_kernels"
    package.mkdir()
    source = package / "kernel.py"
    source.write_text("VALUE = 1\n")
    snapshot = _capture_checkout_snapshot(tmp_path)
    _assert_checkout_unchanged(tmp_path, snapshot, full=True)

    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="changed during paired sweep"):
        _assert_checkout_unchanged(tmp_path, snapshot)


def test_next_run_id_reserves_incomplete_and_complete_artifacts(tmp_path: Path):
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    runs.mkdir()
    reports.mkdir()
    (runs / "2.json").write_text("{}")
    (runs / "5.log").write_text("cancelled")
    (reports / "4").mkdir()
    (runs / "latest.log").write_text("ignored")
    (reports / "notes").mkdir()

    assert _next_run_id(tmp_path) == "6"


def test_scheduler_serializes_named_resource_without_blocking_unrelated_work(
    tmp_path: Path, monkeypatch
):
    exclusive_running = threading.Event()
    ordinary_started = threading.Event()
    state_lock = threading.Lock()
    active_exclusive = 0
    max_active_exclusive = 0
    ordinary_overlapped = False

    class Pool:
        @staticmethod
        def total_visible():
            return 3

    def fake_run_one(workload, *_args, **_kwargs):
        nonlocal active_exclusive, max_active_exclusive, ordinary_overlapped
        if workload.get("exclusive_resource") == "nvshmem":
            with state_lock:
                active_exclusive += 1
                max_active_exclusive = max(max_active_exclusive, active_exclusive)
            exclusive_running.set()
            ordinary_started.wait(2)
            time.sleep(0.02)
            with state_lock:
                active_exclusive -= 1
        else:
            assert exclusive_running.wait(2)
            with state_lock:
                ordinary_overlapped = active_exclusive == 1
            ordinary_started.set()
        return {"kernel": workload["kernel"], "config": workload["config"], "status": "ok"}

    monkeypatch.setattr(run, "run_one", fake_run_one)
    workloads = [
        {"kernel": "exclusive_a", "config": "case", "exclusive_resource": "nvshmem"},
        {"kernel": "exclusive_b", "config": "case", "exclusive_resource": "nvshmem"},
        {"kernel": "ordinary", "config": "case"},
    ]

    records, retries = run_scheduled_jobs(workloads, Pool(), tmp_path, rounds=1, cooldown=0.0)

    assert retries == []
    assert len(records) == 3
    assert max_active_exclusive == 1
    assert ordinary_overlapped


def test_monitored_subprocess_fails_closed_on_timeout(tmp_path: Path):
    log_path = tmp_path / "timeout.log"
    returncode, interfered, intruders, cancelled = _run_subprocess_monitored(
        [run.sys.executable, "-c", "import time; time.sleep(30)"],
        run.os.environ.copy(),
        str(tmp_path),
        log_path,
        (),
        0.01,
        0.0,
        timeout_s=0.05,
    )

    assert returncode != 0
    assert not interfered
    assert intruders == []
    assert not cancelled
    assert "TIMEOUT" in log_path.read_text()
    for invalid in (math.nan, math.inf, -math.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match="finite and positive"):
            _direct_speedup_pct(invalid, 100.0)
        with pytest.raises(ValueError, match="finite and positive"):
            _direct_speedup_pct(100.0, invalid)


def test_paired_result_rejects_nonfinite_local_samples():
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="invalid local samples"):
            _local_samples({"round_samples": {"tirx": [invalid]}})


def test_paired_runner_reuses_one_gpu_claim_and_aggregates_direct_samples(
    tmp_path: Path, monkeypatch
):
    old_checkout = tmp_path / "old"
    current_checkout = tmp_path / "current"
    for checkout in (old_checkout, current_checkout):
        (checkout / "tirx_kernels").mkdir(parents=True)
        (checkout / "tirx_kernels" / "__init__.py").write_text("")

    class Pool:
        util_threshold = 0.0

        def __init__(self):
            self.acquired = 0
            self.released = 0
            self.release_checks = 0
            self.mem_threshold = DEFAULT_MEM_THRESHOLD

        def acquire_many(self, count, cancel_event=None):
            assert count == 1
            self.acquired += 1
            return ("7",)

        def release_many(self, gpus):
            assert gpus == ("7",)
            self.released += 1

        def wait_for_memory_release(self, gpus, cancel_event=None):
            assert gpus == ("7",)
            self.release_checks += 1
            return {}

    observed_sides = []
    side_samples = {"old": iter([101.0, 99.0, 100.0]), "current": iter([100.0, 100.0, 100.0])}

    def fake_monitored(cmd, env, cwd, log_path, gpu_indices, *args):
        pythonpath = [Path(path) for path in env["PYTHONPATH"].split(os.pathsep)]
        assert pythonpath[0].name.startswith("local-only-runtime-")
        side = "old" if pythonpath[1] == old_checkout else "current"
        observed_sides.append(side)
        json_path = Path(cmd[cmd.index("--json-file") + 1])
        sample = next(side_samples[side])
        json_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "kernel": "fixture",
                            "label": "case",
                            "impls": {"tirx": sample},
                            "round_samples": {"tirx": [sample]},
                            "errors": {},
                            "local_only": True,
                        }
                    ]
                }
            )
        )
        return 0, False, [], False

    monkeypatch.setattr(run, "_run_subprocess_monitored", fake_monitored)
    pool = Pool()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = run_one_paired(
        {"kernel": "fixture", "config": "case", "num_gpus": 1},
        pool,
        log_dir,
        old_checkout=old_checkout,
        current_checkout=current_checkout,
        rounds=3,
        cooldown=0,
    )

    assert observed_sides == ["old", "current", "current", "old", "old", "current"]
    assert pool.acquired == pool.released == 1
    assert pool.release_checks == 6
    assert result["status"] == "ok"
    assert result["paired"]["old_samples_us"] == [101.0, 99.0, 100.0]
    assert result["paired"]["current_samples_us"] == [100.0, 100.0, 100.0]
    assert result["paired"]["speedup_pct"] == pytest.approx(0.0)
    assert result["paired"]["local_only"] is True
    assert result["local_only"] is True


def test_paired_runner_fails_when_framebuffer_does_not_release(tmp_path: Path, monkeypatch):
    old_checkout = tmp_path / "old"
    current_checkout = tmp_path / "current"
    for checkout in (old_checkout, current_checkout):
        (checkout / "tirx_kernels").mkdir(parents=True)
        (checkout / "tirx_kernels" / "__init__.py").write_text("")

    class Pool:
        util_threshold = 0.0
        mem_threshold = DEFAULT_MEM_THRESHOLD

        @staticmethod
        def acquire_many(count, cancel_event=None):
            assert count == 1
            return ("3",)

        @staticmethod
        def release_many(gpus):
            assert gpus == ("3",)

        @staticmethod
        def wait_for_memory_release(gpus, cancel_event=None):
            assert gpus == ("3",)
            return {"3": 64.0}

    def fake_monitored(cmd, env, cwd, log_path, gpu_indices, *args):
        json_path = Path(cmd[cmd.index("--json-file") + 1])
        json_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "kernel": "fixture",
                            "label": "case",
                            "impls": {"tirx": 1.0},
                            "round_samples": {"tirx": [1.0]},
                            "errors": {},
                            "local_only": True,
                        }
                    ]
                }
            )
        )
        return 0, False, [], False

    monkeypatch.setattr(run, "_run_subprocess_monitored", fake_monitored)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = run_one_paired(
        {"kernel": "fixture", "config": "case", "num_gpus": 1},
        Pool(),
        log_dir,
        old_checkout=old_checkout,
        current_checkout=current_checkout,
        rounds=1,
        cooldown=0,
    )

    assert result["status"] == "FAIL"
    assert "framebuffer did not return" in result["error"]
