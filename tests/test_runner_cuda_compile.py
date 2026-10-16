# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tirx_kernels import runner
from tvm_tirx.backend.config import copy_backend_config, merge_backend_configs


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
    monkeypatch.setattr(runner.tvm_tirx.tirx, "compile", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(runner.tvm_tirx, "IRModule", lambda value: value)
    runner.compile_kernel(object(), backend_config=config)
    assert copy_backend_config(calls[0]["target"].attrs["backend_config"]) == config


@pytest.mark.parametrize("precision,register_level", [("fp4", 10), ("fp8", 6)])
@pytest.mark.parametrize("explicit_options", [False, True])
def test_mqa_compile_target_preserves_tuning(
    monkeypatch, precision, register_level, explicit_options
):
    module = importlib.import_module(f"tirx_kernels.ported.deepgemm.mqa_logits_{precision}")
    monkeypatch.setattr(module, "get_kernel", lambda **kwargs: object())
    monkeypatch.setattr(runner.tvm_tirx, "IRModule", lambda value: value)
    monkeypatch.setattr(runner.tvm_tirx.tirx, "compile", lambda mod, *, target, pipeline: target)
    config = {"cuda": {"arch": "sm_100a"}}
    if explicit_options:
        config["cuda"].update(nvcc=[], nvrtc=["--ftz=true"], ptxas=[])
    target = module._compile_tirx_mqa_for_config.__wrapped__(
        seq_len=128,
        seq_len_kv=128,
        num_heads=64,
        head_dim=128,
        logits_dtype="float32",
        compressed_logits=False,
        disable_cp=True,
        num_sms=1,
        logits_stride_override=None,
        backend_config=config,
    )
    compiled_config = copy_backend_config(target.attrs["backend_config"])["cuda"]
    if explicit_options:
        assert compiled_config == config["cuda"]
    else:
        assert compiled_config["nvcc"] == ["--use_fast_math", "--ftz=false"]
        assert compiled_config["nvrtc"] == ["--use_fast_math", "--ftz=false"]
        assert f"--register-usage-level={register_level}" in compiled_config["ptxas"]


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
import sys
from tvm_tirx.backend.cuda import BackendConfig
from tirx_kernels.runner import cuda_is_initialized, prepare_kernel_bench
assert "tvm" not in sys.modules
assert not cuda_is_initialized()
prepared = prepare_kernel_bench(
    "rmsnorm", {"hidden_size": 128, "batch_size": 32},
    backend_config={"cuda": BackendConfig(arch="sm_100a")},
)
assert not cuda_is_initialized()
assert "tvm" not in sys.modules
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
    import tvm_tirx

    @runner.cache_backend_config
    def factory(*, backend_config=None):
        return backend_config

    base = {"kind": "cuda", "arch": "sm_100a"}
    with tvm_tirx.target.Target(base):
        plain = factory()
    with tvm_tirx.target.Target({**base, "backend_config": {"cuda": {"nvrtc": []}}}):
        configured = factory()
        assert runner.cuda_target().attrs["backend_config"]["cuda"]["nvrtc"] == []
    assert plain == {"cuda": {"arch": "sm_100a"}}
    assert configured == plain
    assert configured is not plain
    assert factory.cache_info().misses == 2


def test_manual_cache_key_tracks_presence_and_target_defaults():
    import tvm_tirx

    config = {"cuda": {"arch": "sm_100a"}}
    first = runner.backend_config_key(config)
    assert hash(first) == hash(runner.backend_config_key({"cuda": {"arch": "sm_100a"}}))
    config["cuda"]["nvrtc"] = ["--use_fast_math"]
    # An explicit list must still override factory-local math defaults.
    assert first != runner.backend_config_key(config)
    with tvm_tirx.target.Target(
        {"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"ptxas": []}}}
    ):
        assert first != runner.backend_config_key({"cuda": {"arch": "sm_100a"}})
