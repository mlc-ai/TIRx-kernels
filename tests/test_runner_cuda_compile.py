# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import subprocess
import sys
from types import SimpleNamespace

import pytest

from tirx_kernels import runner
from tvm.backend.cuda import CompileConfig


def test_explicit_config_defaults_and_forwarding(monkeypatch):
    config = CompileConfig(arch="sm_107a", compiler="nvcc", ftz=False)
    assert runner.resolve_compile_config(config, ftz=True, ptxas_reg_usage_level=4) == (
        config.with_overrides(ptxas_reg_usage_level=4)
    )
    assert runner.cuda_target(compile_config=config).arch == "sm_107a"
    calls = []
    monkeypatch.setattr(runner.tvm, "compile", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(runner.tvm, "IRModule", lambda value: value)
    runner.compile_kernel(object(), compile_config=config)
    assert calls[0]["compile_config"] == config


def test_cpu_prepare_requires_explicit_arch():
    with pytest.raises(ValueError, match="requires.*arch"):
        runner.prepare_kernel_bench("unused", {})


def test_prepare_and_gpu_stage_keep_same_config(monkeypatch):
    monkeypatch.setattr(runner, "cuda_is_initialized", lambda: False)
    seen = []

    def run_gpu(state, *, compile_config, **kwargs):
        seen.append(compile_config)
        return {"value": state}

    def prepare_bench(value, *, compile_config):
        seen.append(compile_config)
        return runner.prepared_gpu_benchmark(run_gpu, value, compile_config=compile_config)

    config = CompileConfig(arch="sm_100a", ftz=False)
    prepared = runner.prepare_kernel_bench(
        "fake",
        {"value": 7},
        module=SimpleNamespace(prepare_bench=prepare_bench),
        compile_config=config,
    )
    assert runner.run_prepared_kernel_bench(prepared)["value"] == 7
    assert seen == [config.with_overrides(compiler="nvcc")] * 2


def test_real_cpu_prepare_does_not_initialize_cuda():
    script = """
from tvm.backend.cuda import CompileConfig
from tirx_kernels.runner import cuda_is_initialized, prepare_kernel_bench
assert not cuda_is_initialized()
prepared = prepare_kernel_bench(
    "rmsnorm", {"hidden_size": 128, "batch_size": 32},
    compile_config=CompileConfig(arch="sm_100a"),
)
assert not cuda_is_initialized()
assert prepared.benchmark.compile_config.compiler == "nvcc"
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
