# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import subprocess
import sys
from types import SimpleNamespace

import pytest

from tirx_kernels import runner
from tvm.backend.config import merge_backend_configs


def test_explicit_config_defaults_and_forwarding(monkeypatch):
    config = {
        "cuda": {
            "arch": "sm_107a",
            "compiler": "nvcc",
            "nvcc": ["--use_fast_math", "--ftz=false"],
            "nvrtc": ["--use_fast_math", "--ftz=false"],
        }
    }
    assert runner.resolve_backend_config(
        config,
        defaults={
            "cuda": {
                "nvcc": ["--ftz=true"],
                "nvrtc": ["--ftz=true"],
                "ptxas": ["-v", "--warn-on-local-memory-usage", "--register-usage-level=4"],
            }
        },
    ) == (
        merge_backend_configs(
            config,
            {
                "cuda": {
                    "ptxas": [
                        "-v",
                        "--warn-on-local-memory-usage",
                        "--register-usage-level=" + str(4),
                    ]
                }
            },
        )
    )
    assert runner.cuda_target(backend_config=config).arch == "sm_107a"
    calls = []
    monkeypatch.setattr(runner.tvm, "compile", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(runner.tvm, "IRModule", lambda value: value)
    runner.compile_kernel(object(), backend_config=config)
    assert calls[0]["backend_config"] == config


def test_cpu_prepare_requires_explicit_arch():
    with pytest.raises(ValueError, match="requires.*arch"):
        runner.prepare_kernel_bench("unused", {})


def test_prepare_and_gpu_stage_keep_same_config(monkeypatch):
    monkeypatch.setattr(runner, "cuda_is_initialized", lambda: False)
    seen = []

    def run_gpu(state, *, backend_config, **kwargs):
        seen.append(backend_config)
        return {"value": state}

    def prepare_bench(value, *, backend_config):
        seen.append(backend_config)
        return runner.prepared_gpu_benchmark(run_gpu, value, backend_config=backend_config)

    config = {
        "cuda": {
            "arch": "sm_100a",
            "nvcc": ["--use_fast_math", "--ftz=false"],
            "nvrtc": ["--use_fast_math", "--ftz=false"],
        }
    }
    prepared = runner.prepare_kernel_bench(
        "fake",
        {"value": 7},
        module=SimpleNamespace(prepare_bench=prepare_bench),
        backend_config=config,
    )
    assert runner.run_prepared_kernel_bench(prepared)["value"] == 7
    assert seen == [merge_backend_configs(config, {"cuda": {"compiler": "nvcc"}})] * 2


def test_real_cpu_prepare_does_not_initialize_cuda():
    script = """
from tvm.backend.cuda import BackendConfig
from tirx_kernels.runner import cuda_is_initialized, prepare_kernel_bench
assert not cuda_is_initialized()
prepared = prepare_kernel_bench(
    "rmsnorm", {"hidden_size": 128, "batch_size": 32},
    backend_config={"cuda": BackendConfig(arch="sm_100a")},
)
assert not cuda_is_initialized()
assert prepared.benchmark.backend_config["cuda"]["compiler"] == "nvcc"
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_factory_cache_uses_config_snapshots():
    seen = []

    @runner.cache_backend_config
    def factory(size, *, backend_config=None):
        seen.append(backend_config)
        return object()

    config = {"cuda": {"arch": "sm_100a", "nvrtc": ["--ftz=false"]}}
    first = factory(32, backend_config=config)
    assert (
        factory(32, backend_config={"cuda": {"nvrtc": ["--ftz=false"], "arch": "sm_100a"}}) is first
    )
    config["cuda"]["nvrtc"].append("--generate-line-info")
    assert factory(32, backend_config=config) is not first
    assert seen[0] == {"cuda": {"arch": "sm_100a", "nvrtc": ["--ftz=false"]}}
    assert factory.cache_info().hits == 1
    assert "compiler" not in seen[0]["cuda"]


def test_factory_cache_keeps_target_defaults():
    import tvm

    @runner.cache_backend_config
    def factory(*, backend_config=None):
        return backend_config

    base = {"kind": "cuda", "arch": "sm_100a"}
    with tvm.target.Target(base):
        plain = factory()
    with tvm.target.Target({**base, "backend_config": {"cuda": {"nvrtc": []}}}):
        configured = factory()
        assert runner.cuda_target().attrs["backend_config"]["cuda"]["nvrtc"] == []
    assert plain == {"cuda": {"arch": "sm_100a"}}
    assert configured == plain
    assert configured is not plain
    assert factory.cache_info().misses == 2


def test_manual_cache_key_tracks_presence_and_target_defaults():
    import tvm

    config = {"cuda": {"arch": "sm_100a"}}
    first = runner.backend_config_key(config)
    assert hash(first) == hash(runner.backend_config_key({"cuda": {"arch": "sm_100a"}}))
    config["cuda"]["nvrtc"] = ["--use_fast_math"]
    # An explicit list must still override factory-local math defaults.
    assert first != runner.backend_config_key(config)
    with tvm.target.Target(
        {"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"ptxas": []}}}
    ):
        assert first != runner.backend_config_key({"cuda": {"arch": "sm_100a"}})
