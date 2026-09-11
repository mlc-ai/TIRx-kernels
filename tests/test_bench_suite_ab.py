# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
        "execution_mode": "remote",
        "process_model": "kcoral_worker_per_request",
        "remote": {"request_id": f"req-{kernel}-{config}"},
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
            "execution_mode": "remote",
            "process_model": "kcoral_worker_per_request",
            "measurement_protocol": {"rounds": 2, "cooldown_s": 0.0},
            "prepare_mode": "cpu",
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
    before = _payload("before", "before-rev", [_row("kernel", "config", "GPU-a", [10.0, 10.0])])
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


def test_pair_order_alternates_by_index(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def submit_side(side, index, workload, **_kwargs):
        calls.append((index, side))
        return _row(workload["kernel"], workload["config"], "GPU-0", [10.0, 10.0])

    monkeypatch.setattr(ab, "_submit_side", submit_side)
    workload = {"kernel": "kernel", "config": "config", "num_gpus": 1}
    odd = ab._run_pair(1, workload, campaign_root=Path("/unused"))
    even = ab._run_pair(2, workload, campaign_root=Path("/unused"))

    assert calls == [(1, "before"), (1, "after"), (2, "after"), (2, "before")]
    assert odd.order == ("before", "after")
    assert even.order == ("after", "before")
    assert odd.gpu_uuid == "GPU-0"
    assert odd.before["status"] == "ok" and odd.after["status"] == "ok"


def test_pair_rejects_cross_gpu_sides(monkeypatch) -> None:
    def submit_side(side, index, workload, **_kwargs):
        return _row(workload["kernel"], workload["config"], f"GPU-{side}", [10.0, 10.0])

    monkeypatch.setattr(ab, "_submit_side", submit_side)
    with pytest.raises(RuntimeError, match="before ran on"):
        ab._run_pair(1, {"kernel": "kernel", "config": "config"})


def test_validate_side_row_requires_ok_single_impl() -> None:
    workload = {"kernel": "kernel", "config": "config"}
    ab._validate_side_row("after", _row("kernel", "config", "GPU-0", [1.0, 1.0]), workload)
    failed = _row("kernel", "config", "GPU-0", [1.0, 1.0])
    failed.update({"status": "FAIL", "error": "gpu: runtime: boom"})
    with pytest.raises(RuntimeError, match="gpu: runtime: boom"):
        ab._validate_side_row("after", failed, workload)
    two_impls = _row("kernel", "config", "GPU-0", [1.0, 1.0])
    two_impls["round_samples"]["tir"] = [1.0, 1.0]
    with pytest.raises(RuntimeError, match="one TIR/TIRx implementation"):
        ab._validate_side_row("after", two_impls, workload)
    wrong_identity = _row("other", "config", "GPU-0", [1.0, 1.0])
    with pytest.raises(RuntimeError, match="identity differs"):
        ab._validate_side_row("after", wrong_identity, workload)


def test_aggregate_side_builds_gate_payload() -> None:
    pairs = [
        ab._PairResult(
            index=1,
            workload={"kernel": "kernel", "config": "config"},
            order=("before", "after"),
            gpu_uuid="GPU-0",
            before=_row("kernel", "config", "GPU-0", [10.0, 10.0]),
            after=_row("kernel", "config", "GPU-0", [10.05, 10.05]),
        )
    ]
    common = {
        "selection": {"mode": "targeted", "keys": [["kernel", "config"]], "cuda_arch": "sm_100a"},
        "timestamp": "stamp",
        "git": {"tir": "f6726b02", "tirx-kernels": "ignored", "tirx-bench-ci": None},
        "kernel_tree": {"tir:python/tvm/tirx": "sha256:x", "tirx-kernels:tirx_kernels": "ignored"},
        "probe": {"server": {}},
        "pipeline": {
            "execution_mode": "remote",
            "process_model": "kcoral_worker_per_request",
            "measurement_protocol": {"rounds": 2, "cooldown_s": 0.0},
        },
    }
    before = ab._aggregate_side("before", pairs, revision="b" * 40, tree="before-tree", **common)
    after = ab._aggregate_side("after", pairs, revision="a" * 40, tree="after-tree", **common)

    assert before["git"] == {"tir": "f6726b02", "tirx-kernels": "b" * 8, "tirx-bench-ci": None}
    assert after["kernel_tree"]["tirx-kernels:tirx_kernels"] == "after-tree"
    assert before["results"][0]["impls"] == {"tirx": 10.0}
    report, failures = build_report(before, after, paired=True)
    assert failures == 0
    assert "1/1 expected rows evaluated; 1 direct passes" in report


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


def test_current_contract_uses_after_kern_without_rebinding_before_kern(monkeypatch, tmp_path):
    monkeypatch.setenv(AB_CURRENT_BENCHMARK_ROOT_ENV, str(tmp_path))
    package_root = tmp_path / "tirx_kernels"
    kern_root = package_root / "kern"
    kern_root.mkdir(parents=True)
    (kern_root / "__init__.py").write_text("MARKER = 'after'\n")
    (package_root / "ab_kern_isolation.py").write_text(
        "import tirx_kernels.kern as K\nKERN_MARKER = K.MARKER\nCONFIGS = [{'label': 'same'}]\n"
    )

    before_kern = ModuleType("tirx_kernels.kern")
    before_kern.MARKER = "before"
    monkeypatch.setitem(sys.modules, "tirx_kernels.kern", before_kern)
    old_module = ModuleType("tirx_kernels.ab_kern_isolation")
    old_module.__package__ = "tirx_kernels"

    current_module = ab_current_benchmark_module(old_module)

    assert current_module.KERN_MARKER == "after"
    assert sys.modules["tirx_kernels.kern"] is before_kern
