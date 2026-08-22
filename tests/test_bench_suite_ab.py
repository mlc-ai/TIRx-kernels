# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from types import ModuleType

from tirx_kernels.bench.__main__ import _find_bench_config
from tirx_kernels.bench_suite import ab
from tirx_kernels.bench_suite.ratio_diff import build_report
from tirx_kernels.runner import (
    AB_CURRENT_BENCHMARK_ROOT_ENV,
    ExplicitPreparedBenchmark,
    ab_current_benchmark_module,
    close_prepared_kernel_bench,
    prepare_kernel_bench,
    prepared_gpu_benchmark,
)


def _row(kernel: str, config: str, gpu_uuid: str, samples: list[float]) -> dict:
    return {
        "kernel": kernel,
        "config": config,
        "label": config,
        "status": "ok",
        "errors": {},
        "round_samples": {"tirx": samples},
        "impls": {"tirx": sum(samples) / len(samples)},
        "timer": "proton",
        "benchmark_protocol": {
            "rounds": len(samples),
            "round_aggregate": "mean",
            "cooldown_s": 0.0,
            "order": ["tirx"],
        },
        "num_gpus": 1,
        "physical_gpu_uuids": [gpu_uuid],
        "execution_mode": "pipeline",
        "process_model": "one_shot_child_per_workload",
        "retry_in_place": False,
    }


def _payload(label: str, revision: str, rows: list[dict]) -> dict:
    keys = [[row["kernel"], row["config"]] for row in rows]
    return {
        "timestamp": label,
        "label": label,
        "git": {"tir": "tvm-dirty", "tirx-kernels": revision, "tirx-bench-ci": None},
        "kernel_tree": {
            "tir:python/tvm/tirx": "shared-tir-tree",
            "tirx-kernels:tirx_kernels": f"{revision}-tree",
        },
        "references_enabled": True,
        "baselines": {"torch": {"version": "test"}},
        "selection": {"mode": "targeted", "keys": keys},
        "pipeline": {
            "execution_mode": "pipeline",
            "process_model": "one_shot_child_per_workload",
            "measurement_protocol": {"rounds": 2, "cooldown_s": 0.0},
            "interference_retry_count": 0,
            "interference_retries": [],
        },
        "results": rows,
    }


def test_paired_report_allows_different_gpus_across_workloads() -> None:
    before = _payload(
        "before",
        "before-rev",
        [
            _row("kernel_a", "config_a", "GPU-a", [10.0, 10.0]),
            _row("kernel_b", "config_b", "GPU-b", [20.0, 20.0]),
        ],
    )
    after = _payload(
        "after",
        "after-rev",
        [
            _row("kernel_a", "config_a", "GPU-a", [10.05, 10.05]),
            _row("kernel_b", "config_b", "GPU-b", [19.0, 19.0]),
        ],
    )

    report, failures = build_report(before, after, paired=True)

    assert failures == 0
    assert "2/2 expected rows evaluated; 2 direct passes" in report


def test_paired_report_allows_empty_baselines_when_references_are_disabled() -> None:
    before = _payload(
        "before", "before-rev", [_row("kernel", "config", "GPU-a", [10.0, 10.0])]
    )
    after = copy.deepcopy(before)
    after["label"] = "after"
    after["git"]["tirx-kernels"] = "after-rev"
    after["kernel_tree"]["tirx-kernels:tirx_kernels"] = "after-tree"
    for payload in (before, after):
        payload["references_enabled"] = False
        payload["baselines"] = {}

    report, failures = build_report(before, after, paired=True)

    assert failures == 0
    assert "1/1 expected rows evaluated; 1 direct passes" in report


def test_paired_report_rejects_a_cross_gpu_pair() -> None:
    before = _payload(
        "before", "before-rev", [_row("kernel", "config", "GPU-before", [10.0, 10.0])]
    )
    after = copy.deepcopy(before)
    after["label"] = "after"
    after["git"]["tirx-kernels"] = "after-rev"
    after["kernel_tree"]["tirx-kernels:tirx_kernels"] = "after-tree"
    after["results"][0]["physical_gpu_uuids"] = ["GPU-after"]

    report, failures = build_report(before, after, paired=True)

    assert failures == 1
    assert "provenance field physical_gpu_uuids differs" in report


def test_pair_restarts_both_sides_after_interference(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def run_side(side, *_args, **_kwargs):
        calls.append(side)
        if calls == ["before", "after"]:
            raise ab._InterferenceError("polluted")
        return {"side": side}

    monkeypatch.setattr(ab, "_run_side", run_side)
    rejected: list[dict] = []
    result = ab._run_pair(
        1,
        {"kernel": "kernel", "config": "config", "num_gpus": 1},
        gpu_index="3",
        gpu_uuid="GPU-3",
        campaign_root=tmp_path,
        roots={"before": tmp_path, "after": tmp_path},
        revisions={"before": "before", "after": "after"},
        rounds=2,
        cooldown=0.0,
        util_threshold=0.0,
        mem_threshold=0.0,
        rejected=rejected,
        rejected_lock=threading.Lock(),
    )

    assert calls == ["before", "after", "before", "after"]
    assert result.attempt == 2
    assert len(rejected) == 1


def test_before_uses_current_config_with_own_run_gpu(monkeypatch, tmp_path) -> None:
    current_source = tmp_path / "tirx_kernels" / "fake_kernel.py"
    current_source.parent.mkdir()
    current_source.write_text(
        "CONFIGS = [{'label': 'same', 'value': 'current'}]\n"
        "def run_gpu(state, **kwargs):\n"
        "    raise AssertionError('paired A/B must not rebind the current run_gpu')\n"
    )
    monkeypatch.setenv(AB_CURRENT_BENCHMARK_ROOT_ENV, str(tmp_path))

    closed: list[str] = []
    old_module = ModuleType("tirx_kernels.fake_kernel")
    old_module.__package__ = "tirx_kernels"

    def old_run_gpu(state, **kwargs):
        return {"callback": "old", "state": state, "kwargs": kwargs}

    def old_prepare_bench(**config):
        return prepared_gpu_benchmark(
            old_run_gpu,
            {"compiled_by": "old", "config": config},
            required_num_gpus=2,
            close=lambda: closed.append("old"),
        )

    old_module.prepare_bench = old_prepare_bench
    current_module = ab_current_benchmark_module(old_module)
    config = _find_bench_config(current_module, "same")
    prepared = prepare_kernel_bench(
        "fake", config, module=old_module, require_cuda_uninitialized=False
    )

    assert isinstance(prepared.benchmark, ExplicitPreparedBenchmark)
    assert prepared.required_num_gpus == 2
    # Config comes from the current module; the old module's own run_gpu packs
    # launch arguments for the executables it compiled itself.
    assert prepared.benchmark.run_gpu(marker=1) == {
        "callback": "old",
        "state": {"compiled_by": "old", "config": {"value": "current"}},
        "kwargs": {"marker": 1},
    }
    close_prepared_kernel_bench(prepared)
    assert closed == ["old"]


def test_only_before_side_receives_current_benchmark_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(AB_CURRENT_BENCHMARK_ROOT_ENV, "stale")
    before = ab._side_environment(Path("/before"), "3", current_benchmark_root=tmp_path)
    after = ab._side_environment(Path("/after"), "3")

    assert before[AB_CURRENT_BENCHMARK_ROOT_ENV] == str(tmp_path)
    assert AB_CURRENT_BENCHMARK_ROOT_ENV not in after


def test_current_contract_uses_after_kern_without_rebinding_before_kern(monkeypatch, tmp_path):
    monkeypatch.setenv(AB_CURRENT_BENCHMARK_ROOT_ENV, str(tmp_path))
    package_root = tmp_path / "tirx_kernels"
    kern_root = package_root / "kern"
    kern_root.mkdir(parents=True)
    (kern_root / "__init__.py").write_text("MARKER = 'after'\n")
    (package_root / "ab_kern_isolation.py").write_text(
        "import tirx_kernels.kern as K\n"
        "KERN_MARKER = K.MARKER\n"
        "CONFIGS = [{'label': 'same'}]\n"
    )

    before_kern = ModuleType("tirx_kernels.kern")
    before_kern.MARKER = "before"
    monkeypatch.setitem(sys.modules, "tirx_kernels.kern", before_kern)
    old_module = ModuleType("tirx_kernels.ab_kern_isolation")
    old_module.__package__ = "tirx_kernels"

    current_module = ab_current_benchmark_module(old_module)

    assert current_module.KERN_MARKER == "after"
    assert sys.modules["tirx_kernels.kern"] is before_kern
