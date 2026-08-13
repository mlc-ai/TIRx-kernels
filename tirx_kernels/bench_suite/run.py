#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""bench-suite: pre-commit regression benchmark for TIRx kernels.

See README.md in this directory for setup, baseline workflow, and flags.

Quick start:
    python -m tirx_kernels.bench_suite
    python tirx_kernels/bench_suite/promote_baseline.py .bench-suite/runs/<id>.json --merge

Exit codes:
    0  no regressions (or no baseline yet)
    1  workload failure (suite stopped immediately)
    2  config error (no workloads / bad YAML)
    3  one or more regressions exceeded the threshold
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import os
import random
import select
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.util import find_spec
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, ClassVar

import yaml

from tirx_kernels.runner import DEFAULT_BENCH_COOLDOWN_S as DEFAULT_COOLDOWN_S
from tirx_kernels.runner import DEFAULT_BENCH_ROUNDS as DEFAULT_ROUNDS
from tirx_kernels.runner import PREPARE_NUM_SMS_ENV

try:
    from tirx_kernels.bench_suite.impls import our_impls
except ModuleNotFoundError:  # Support `python tirx_kernels/bench_suite/run.py`.
    from impls import our_impls

SCRIPT_DIR = Path(__file__).resolve().parent


def _kernels_repo_root() -> Path:
    """Git root of the tirx-kernels repo (parent of the tirx_kernels package)."""
    return SCRIPT_DIR.parent.parent


DEFAULT_OUT_DIR = _kernels_repo_root() / ".bench-suite"
# One file per kernel, every config it can bench, each flagged `default:` for
# whether the pinned regression sweep includes it. With no --workloads, the
# flagged ones are assembled into a generated workload file and benched.
CONFIG_DIR = SCRIPT_DIR / "config"
GENERATED_WORKLOADS_NAME = "workloads.generated.yaml"
EXPECTED_DEFAULT_WORKLOADS = 112
MAX_DEFAULT_CONFIGS_PER_KERNEL = 3
# Single pinned baseline: every run benches our kernel + all reference impls,
# so one JSON holds both. Promote a run over it via promote_baseline.py.
DEFAULT_BASELINE = SCRIPT_DIR / "baseline.json"
DEFAULT_REGRESSION_THRESHOLD = 1.0
POLL_INTERVAL = 5.0  # seconds between GPU re-checks when none is free
MONITOR_INTERVAL = 0.5  # seconds between nvidia-smi polls during a workload
DEFAULT_UTIL_THRESHOLD = 0.0  # % GPU util above which a card counts as busy.
DEFAULT_MEM_THRESHOLD = 0.0  # % compute-app memory above which a card counts as busy.


# Tiny real workload used to decide whether a GPU is actually usable.
# Catches: driver hangs, ECC errors when touching memory, cuBLAS init
# failures, MIG/cgroup restrictions, fragmentation surprises — issues that
# nvidia-smi "free" status alone won't surface.
PROBE_SCRIPT = r"""
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("PROBE_FAIL: torch.cuda.is_available()=False", file=sys.stderr)
        sys.exit(1)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    del a, b, c
    torch.cuda.empty_cache()
except Exception as e:
    print(f"PROBE_FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
print("PROBE_OK")
"""


# ── Workload loading ─────────────────────────────────────────────────────────


def _normalize_workload(workload: dict) -> dict:
    """Apply the field rules every loader shares."""
    if "kernel" not in workload or "config" not in workload:
        raise ValueError(f"workload missing kernel/config: {workload}")
    num_gpus = workload.get("num_gpus", 1)
    if type(num_gpus) is not int or num_gpus < 1:
        raise ValueError(f"workload num_gpus must be a positive integer: {workload}")
    workload["num_gpus"] = num_gpus
    if workload.get("timer") == "megamoe" and (
        workload.get("warmup") is not None or workload.get("repeat") is not None
    ):
        raise ValueError(
            "timer='megamoe' uses a fixed DeepGEMM protocol and cannot override "
            f"warmup/repeat: {workload}"
        )
    return workload


def _read_kernel_config(path: Path) -> tuple[str, list[dict]]:
    data = yaml.safe_load(path.read_text()) or {}
    kernel = data.get("kernel")
    if not kernel:
        raise ValueError(f"{path.name}: missing top-level 'kernel'")
    defaults = data.get("defaults") or {}
    entries = []
    labels: set[str] = set()
    default_count = 0
    for entry in data.get("configs") or []:
        if "config" not in entry:
            raise ValueError(f"{path.name}: config entry missing 'config': {entry}")
        label = entry["config"]
        if not isinstance(label, str) or not label:
            raise ValueError(f"{path.name}: config label must be a non-empty string: {entry}")
        if label in labels:
            raise ValueError(f"{path.name}: duplicate config label {label!r}")
        labels.add(label)
        if entry.get("default", False):
            default_count += 1
        entries.append({"kernel": kernel, **defaults, **entry})
    if default_count > MAX_DEFAULT_CONFIGS_PER_KERNEL:
        raise ValueError(
            f"{path.name}: kernel {kernel!r} has {default_count} default configs; "
            f"maximum is {MAX_DEFAULT_CONFIGS_PER_KERNEL}"
        )
    return kernel, entries


def load_kernel_configs(kernel: str, config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Every config one kernel can bench, flagged for the pinned sweep or not.

    Entries keep their ``default`` flag; :func:`load_config_dir` is the filtered
    view the regression gate runs.
    """
    matches = sorted(config_dir.rglob(f"{kernel}.yaml"))
    if not matches:
        raise FileNotFoundError(f"no config file for kernel {kernel!r} under {config_dir}")
    if len(matches) > 1:
        raise ValueError(f"kernel {kernel!r} has more than one config file: {matches}")
    path = matches[0]
    _, entries = _read_kernel_config(path)
    return [_normalize_workload(e) for e in entries]


def load_config_dir(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """The pinned sweep: every `default: true` config across ``config/**/*.yaml``.

    Each file is one kernel's complete benchable matrix, so which configs the
    regression gate covers is a per-line flag rather than a separate file.  The
    files are bucketed to mirror the kernel tree, so the walk is recursive.
    """
    files = sorted(config_dir.rglob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no kernel config files under {config_dir}")
    out: list[dict] = []
    for path in files:
        _, entries = _read_kernel_config(path)
        for entry in entries:
            if not entry.pop("default", False):
                continue
            workload = _normalize_workload(entry)
            if workload["num_gpus"] != 1:
                raise ValueError(
                    "default measured sweep must remain single-GPU; run multi-GPU "
                    f"workload explicitly instead: {workload}"
                )
            out.append(workload)
    if config_dir.resolve() == CONFIG_DIR.resolve() and len(out) != EXPECTED_DEFAULT_WORKLOADS:
        raise ValueError(
            f"default workload inventory has {len(out)} entries; "
            f"expected {EXPECTED_DEFAULT_WORKLOADS}"
        )
    return out


def write_generated_workloads(workloads: list[dict], path: Path) -> Path:
    """Materialize the assembled sweep so a run's exact input is inspectable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by bench_suite from tirx_kernels/bench_suite/config/**/*.yaml\n"
        "# (every config flagged `default: true`). Do not edit -- rewritten each run.\n"
    )
    path.write_text(
        header + yaml.safe_dump({"defaults": {}, "workloads": workloads}, sort_keys=False)
    )
    return path


def load_workloads(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    defaults = data.get("defaults") or {}
    return [_normalize_workload({**defaults, **entry}) for entry in data.get("workloads") or []]


def _bench_configs(module: Any) -> list[dict]:
    return list(getattr(module, "BENCH_CONFIGS", getattr(module, "CONFIGS", [])))


def _config_num_gpus(config: dict) -> int:
    values = [config[key] for key in ("world_size", "num_processes") if key in config]
    if not values:
        return 1
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError(f"invalid distributed config GPU count: {config}")
    if len(set(values)) != 1:
        raise ValueError(f"conflicting distributed config GPU counts: {config}")
    return values[0]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _audit_generic_adapter_source(kernel: str, source_path: Path) -> bool:
    """Validate the CPU/GPU boundary of a ``prepare_module_bench`` adapter.

    Returns whether the module uses the generic one-PrimFunc adapter. Custom
    prepared adapters own their boundary explicitly and are audited by their
    runtime/capability contracts instead.
    """
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    prepare_bench = functions.get("prepare_bench")
    if prepare_bench is None:
        return False
    uses_generic_adapter = any(
        isinstance(node, ast.Call) and _call_name(node) == "prepare_module_bench"
        for node in ast.walk(prepare_bench)
    )
    if not uses_generic_adapter:
        return False

    run_bench = functions.get("run_bench")
    if run_bench is None:
        raise TypeError(f"kernel {kernel!r} generic adapter has no run_bench()")
    lazy_calls = [
        node
        for node in ast.walk(run_bench)
        if isinstance(node, ast.Call) and _call_name(node) == "compile_kernel_lazy"
    ]
    if len(lazy_calls) != 1:
        raise TypeError(
            f"kernel {kernel!r} generic run_bench() must contain exactly one "
            f"compile_kernel_lazy() call, found {len(lazy_calls)}"
        )
    eager_compiles = [
        node.lineno
        for node in ast.walk(run_bench)
        if isinstance(node, ast.Call) and _call_name(node) == "compile_kernel"
    ]
    if eager_compiles:
        raise TypeError(
            f"kernel {kernel!r} generic run_bench() eagerly compiles on line(s) "
            f"{eager_compiles}"
        )

    lazy_call = lazy_calls[0]
    if len(lazy_call.args) != 1 or not isinstance(lazy_call.args[0], ast.Lambda):
        raise TypeError(
            f"kernel {kernel!r} compile_kernel_lazy() must receive one builder lambda"
        )
    builder_lambda = lazy_call.args[0]
    if not isinstance(builder_lambda.body, ast.Call) or _call_name(builder_lambda.body) != "get_kernel":
        raise TypeError(
            f"kernel {kernel!r} generic lazy builder must call canonical get_kernel()"
        )
    builder_names = {
        name
        for node in ast.walk(builder_lambda.body)
        if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
    }
    lambda_node_ids = {id(node) for node in ast.walk(builder_lambda)}
    eager_builders = [
        (name, node.lineno)
        for node in ast.walk(run_bench)
        if id(node) not in lambda_node_ids
        and isinstance(node, ast.Call)
        and (name := _call_name(node)) in builder_names
    ]
    if eager_builders:
        raise TypeError(
            f"kernel {kernel!r} generic run_bench() invokes lazy builder work "
            f"outside its lambda: {eager_builders}"
        )

    run_test = functions.get("run_test")
    if run_test is not None and any(
        isinstance(node, ast.Call) and _call_name(node) == "compile_kernel_lazy"
        for node in ast.walk(run_test)
    ):
        raise TypeError(
            f"kernel {kernel!r} run_test() must keep standalone eager compilation"
        )
    return True


def _audit_strict_cache_adapter_source(kernel: str, source_path: Path) -> bool:
    """Require DeepGEMM ``build_launch`` adapters to consume their prepared spec."""
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    prepare_bench = functions.get("prepare_bench")
    if prepare_bench is None:
        return False
    tirx_launch = functions.get("_tirx_launch")
    run_bench = functions.get("run_bench")
    if tirx_launch is None or run_bench is None:
        return False

    build_launch_calls = [
        node
        for node in ast.walk(tirx_launch)
        if isinstance(node, ast.Call) and _call_name(node) == "build_launch"
    ]
    if not build_launch_calls:
        return False
    if len(build_launch_calls) != 1:
        raise TypeError(
            f"kernel {kernel!r} must contain exactly one build_launch() call in "
            f"_tirx_launch(), found {len(build_launch_calls)}"
        )

    def is_config_spec(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and _call_name(node) == "_spec_for"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "config"
            and not node.keywords
        )

    canonical_spec_names = {
        target.id
        for node in ast.walk(tirx_launch)
        if isinstance(node, ast.Assign) and is_config_spec(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    launch_call = build_launch_calls[0]
    launch_spec = launch_call.args[0] if launch_call.args else None
    launch_uses_config_spec = is_config_spec(launch_spec) or (
        isinstance(launch_spec, ast.Name) and launch_spec.id in canonical_spec_names
    )
    if not launch_uses_config_spec:
        raise TypeError(
            f"kernel {kernel!r} _tirx_launch() must derive build_launch()'s spec "
            "from canonical _spec_for(config)"
        )
    helper_calls = [
        node
        for node in ast.walk(prepare_bench)
        if isinstance(node, ast.Call) and _call_name(node) == "prepare_compile_spec_bench"
    ]
    if len(helper_calls) != 1:
        raise TypeError(
            f"kernel {kernel!r} uses DeepGEMM build_launch() and must prepare exactly "
            f"one strict compile-spec cache entry, found {len(helper_calls)}"
        )
    helper_call = helper_calls[0]
    if len(helper_call.args) < 3 or not is_config_spec(helper_call.args[2]):
        raise TypeError(
            f"kernel {kernel!r} prepare_bench() must derive its strict cache key "
            "from canonical _spec_for(config)"
        )
    if any(
        isinstance(node, ast.Call) and _call_name(node) == "compile_spec"
        for node in ast.walk(prepare_bench)
    ):
        raise TypeError(
            f"kernel {kernel!r} prepare_bench() bypasses strict compile-spec replay"
        )
    run_launch_calls = [
        node
        for node in ast.walk(run_bench)
        if isinstance(node, ast.Call) and _call_name(node) == "_tirx_launch"
    ]
    if len(run_launch_calls) != 1:
        raise TypeError(
            f"kernel {kernel!r} run_bench() must consume the prepared spec through "
            f"exactly one _tirx_launch() call, found {len(run_launch_calls)}"
        )
    return True


def audit_pipeline_capabilities(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Statically prove exact resolution and explicit pipeline coverage."""
    from tirx_kernels.registry import kernel_index, load_kernel
    from tirx_kernels.runner import cuda_is_initialized

    before_cuda = cuda_is_initialized()
    records = kernel_index(strict=True)
    modules: dict[str, Any] = {}
    labels_by_kernel: dict[str, dict[str, dict]] = {}
    multi_gpu_exemptions: list[dict[str, Any]] = []
    module_config_count = 0
    generic_adapter_count = 0
    strict_cache_adapter_count = 0

    for name in sorted(records):
        module = load_kernel(name, strict=True)
        modules[name] = module
        if not callable(getattr(module, "prepare_bench", None)):
            raise TypeError(f"kernel {name!r} has no explicit prepare_bench()")
        source_path = Path(inspect.getsourcefile(module) or "")
        if not source_path.is_file():
            raise FileNotFoundError(f"kernel {name!r} has no inspectable source file")
        generic_adapter_count += int(_audit_generic_adapter_source(name, source_path))
        strict_cache_adapter_count += int(
            _audit_strict_cache_adapter_source(name, source_path)
        )
        configs = _bench_configs(module)
        prepare_bench = module.prepare_bench
        labels: dict[str, dict] = {}
        for config in configs:
            label = config.get("label", "default")
            if not isinstance(label, str) or not label:
                raise ValueError(f"kernel {name!r} has an invalid config label: {config}")
            if label in labels:
                raise ValueError(f"kernel {name!r} has duplicate config label {label!r}")
            labels[label] = config
            params = {key: value for key, value in config.items() if key != "label"}
            try:
                inspect.signature(prepare_bench).bind(**params)
            except TypeError as error:
                raise TypeError(
                    f"kernel {name!r} config {label!r} cannot bind to prepare_bench: {error}"
                ) from error
            num_gpus = _config_num_gpus(config)
            if num_gpus > 1:
                multi_gpu_exemptions.append(
                    {
                        "kernel": name,
                        "config": label,
                        "num_gpus": num_gpus,
                        "validation_status": "exempted_by_human_unmeasured",
                    }
                )
        labels_by_kernel[name] = labels
        module_config_count += len(configs)

    yaml_config_count = 0
    yaml_kernel_names: set[str] = set()
    yaml_labels_by_kernel: dict[str, set[str]] = {}
    for path in sorted(config_dir.rglob("*.yaml")):
        kernel, entries = _read_kernel_config(path)
        yaml_kernel_names.add(kernel)
        if kernel not in modules:
            raise KeyError(f"{path}: unknown kernel {kernel!r}")
        labels = labels_by_kernel[kernel]
        yaml_labels = yaml_labels_by_kernel.setdefault(kernel, set())
        for entry in entries:
            yaml_config_count += 1
            label = entry["config"]
            yaml_labels.add(label)
            if label not in labels:
                raise KeyError(f"{path}: unknown config {kernel}/{label}")
            expected_num_gpus = _config_num_gpus(labels[label])
            declared_num_gpus = _normalize_workload(dict(entry))["num_gpus"]
            if declared_num_gpus != expected_num_gpus:
                raise ValueError(
                    f"{path}: {kernel}/{label} declares num_gpus={declared_num_gpus}, "
                    f"module config requires {expected_num_gpus}"
                )

    missing_yaml = sorted(set(records) - yaml_kernel_names)
    if missing_yaml:
        raise ValueError(f"registered kernels missing config YAML: {missing_yaml}")
    missing_yaml_configs = {
        kernel: sorted(set(labels) - yaml_labels_by_kernel.get(kernel, set()))
        for kernel, labels in labels_by_kernel.items()
        if set(labels) - yaml_labels_by_kernel.get(kernel, set())
    }
    if missing_yaml_configs:
        raise ValueError(
            "module benchmark configs missing from YAML inventory: "
            f"{missing_yaml_configs}"
        )
    if cuda_is_initialized() != before_cuda:
        raise RuntimeError("pipeline capability audit changed CUDA initialization state")

    return {
        "validation_status": "static_pass",
        "execution_mode": "pipeline",
        "kernel_count": len(records),
        "module_config_count": module_config_count,
        "yaml_config_count": yaml_config_count,
        "generic_adapter_count": generic_adapter_count,
        "strict_cache_adapter_count": strict_cache_adapter_count,
        "multi_gpu_runtime_validation": {
            "validation_status": "exempted_by_human_unmeasured",
            "workload_count": len(multi_gpu_exemptions),
            "workloads": multi_gpu_exemptions,
        },
    }


def write_pipeline_capability_report(out_dir: Path, capability: dict[str, Any]) -> Path:
    """Write the all-config static audit without adding it to timed runs."""
    report_dir = out_dir.resolve() / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    exemption = capability["multi_gpu_runtime_validation"]
    lines = [
        "# bench-suite pipeline capability audit",
        "",
        f"- static migration status: `{capability['validation_status']}`",
        f"- registered kernels: `{capability['kernel_count']}`",
        f"- module configs: `{capability['module_config_count']}`",
        f"- YAML configs: `{capability['yaml_config_count']}`",
        f"- generic lazy-replay adapters: `{capability['generic_adapter_count']}`",
        f"- strict custom-cache replay adapters: `{capability['strict_cache_adapter_count']}`",
        "",
        "## Multi-GPU runtime validation exemption",
        "",
        f"- validation status: `{exemption['validation_status']}`",
        f"- exempt module configs: `{exemption['workload_count']}`",
        "- runtime evidence: not collected, by explicit human direction; this is neither "
        "`passed` nor `missing`",
        "- migrated semantics: pipeline-only late assignment, full atomic claim before "
        "rank/CUDA startup, barriers, sample-wise max aggregation, Kineto spans, and "
        "process-group cleanup",
        "- no-card structural checks: assignment-count rejection, atomic-claim failure, "
        "and rank lifecycle ordering are covered by "
        "`tests/test_bench_pipeline_protocol.py`; this static audit does not execute them",
        "",
        "| workload | GPUs | runtime validation |",
        "|---|---:|---|",
    ]
    for workload in exemption["workloads"]:
        lines.append(
            f"| `{workload['kernel']}/{workload['config']}` | "
            f"{workload['num_gpus']} | `{workload['validation_status']}` |"
        )
    lines.append("")
    path = report_dir / "pipeline-capability.md"
    path.write_text("\n".join(lines))
    return path


def check_workload_capabilities(workloads: list[dict]) -> list[str]:
    """Validate an explicit workload list against module-owned config facts."""
    from tirx_kernels.registry import load_kernel

    names: list[str] = []
    seen: set[str] = set()
    for workload in workloads:
        normalized = _normalize_workload(dict(workload))
        name = normalized["kernel"]
        module = load_kernel(name, strict=True)
        if not callable(getattr(module, "prepare_bench", None)):
            raise TypeError(f"kernel {name!r} has no explicit prepare_bench()")
        matches = [
            config
            for config in _bench_configs(module)
            if config.get("label", "default") == normalized["config"]
        ]
        if len(matches) != 1:
            raise KeyError(
                f"{name}/{normalized['config']}: expected exactly one module config, "
                f"found {len(matches)}"
            )
        required_num_gpus = _config_num_gpus(matches[0])
        if normalized["num_gpus"] != required_num_gpus:
            raise ValueError(
                f"{name}/{normalized['config']} declares num_gpus={normalized['num_gpus']}, "
                f"module config requires {required_num_gpus}"
            )
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


# ── GPU pool ─────────────────────────────────────────────────────────────────


class GpuPool:
    """Exclusive GPU resource pool for late assignment by the orchestrator.

    A READY workload atomically acquires all required GPUs, holds them only for
    its GPU stage, and releases them when RESULT reaches the parent. At most one
    orchestrator GPU stage owns a card at a time, including multi-GPU jobs.

    A card is ineligible when internally owned or when the background external
    occupancy snapshot exceeds the utilization/memory gates. Startup probing
    filters broken cards into `allowed` before the pool is created.
    """

    def __init__(
        self,
        allowed: set[str] | None = None,
        util_threshold: float = DEFAULT_UTIL_THRESHOLD,
        mem_threshold: float = DEFAULT_MEM_THRESHOLD,
    ):
        self._owned: set[str] = set()
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._allowed = allowed
        self._known_indices: tuple[str, ...] = tuple(
            sorted(allowed, key=int) if allowed is not None else ()
        )
        self._external_occupied: set[str] | None = None
        self._external_history: list[tuple[float, tuple[str, ...]]] = []
        self._change_generation = 0
        self.util_threshold = util_threshold
        self.mem_threshold = mem_threshold

    @staticmethod
    def _nvidia_smi(args: list[str]) -> list[str]:
        try:
            out = subprocess.run(
                ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            # Transient nvidia-smi stall under cluster load: degrade to an empty
            # reading instead of killing the whole sweep. Callers treat an empty
            # utilization map as "no occupancy info this tick"; a real co-run
            # conflict is still caught by the per-PID interference check.
            return []
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def _all_gpus(self) -> list[tuple[str, str]]:
        rows = self._nvidia_smi(["--query-gpu=index,uuid"])
        result = []
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result.append((parts[0], parts[1]))
        return result

    def _busy_indices(self) -> set[str]:
        """GPU indices with at least one compute-app PID (anyone's). Kept for
        the informational startup banner only — selection uses _occupied_indices
        (utilization), since a PID may just be parking idle VRAM."""
        rows = self._nvidia_smi(["--query-compute-apps=gpu_uuid"])
        busy_uuids = {row for row in rows if row}
        return {idx for idx, uuid in self._all_gpus() if uuid in busy_uuids}

    def _utils(self) -> dict[str, float]:
        """Map GPU index -> current utilization.gpu (percent)."""
        rows = self._nvidia_smi(["--query-gpu=index,utilization.gpu"])
        out: dict[str, float] = {}
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return out

    def _mem_used_pct(self) -> dict[str, float]:
        """Map GPU index -> compute-app used_memory / memory.total (percent)."""
        gpus = self._nvidia_smi(["--query-gpu=index,uuid,memory.total"])
        uuid_to_idx: dict[str, str] = {}
        total_by_idx: dict[str, float] = {}
        for line in gpus:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    uuid_to_idx[parts[1]] = parts[0]
                    total_by_idx[parts[0]] = float(parts[2])
                except ValueError:
                    pass
        used_by_idx: dict[str, float] = {idx: 0.0 for idx in total_by_idx}
        rows = self._nvidia_smi(["--query-compute-apps=gpu_uuid,used_memory"])
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                idx = uuid_to_idx.get(parts[0])
                if idx is None:
                    continue
                try:
                    used_by_idx[idx] += float(parts[1])
                except ValueError:
                    pass
        out: dict[str, float] = {}
        for idx, used in used_by_idx.items():
            total = total_by_idx.get(idx, 0.0)
            out[idx] = 100.0 * used / total if total > 0 else 0.0
        return out

    def _occupied_indices(self) -> set[str]:
        """GPU indices over the configured SM or memory threshold."""
        util_busy = {idx for idx, u in self._utils().items() if u > self.util_threshold}
        mem_busy = {idx for idx, m in self._mem_used_pct().items() if m > self.mem_threshold}
        return util_busy | mem_busy

    def total_visible(self) -> int:
        gpus = self._all_gpus()
        if self._allowed is not None:
            gpus = [g for g in gpus if g[0] in self._allowed]
        return len(gpus)

    def try_acquire_many(self, count: int) -> tuple[str, ...] | None:
        """Atomically claim currently eligible GPUs without waiting."""
        if type(count) is not int or count < 1:
            raise ValueError(f"GPU count must be a positive integer, got {count!r}")
        with self._changed:
            if self._external_occupied is None:
                return None
            free = [
                idx
                for idx in self._known_indices
                if (self._allowed is None or idx in self._allowed)
                and idx not in self._owned
                and idx not in self._external_occupied
            ]
            if len(free) < count:
                return None
            selected = tuple(sorted(random.sample(free, count), key=int))
            self._owned.update(selected)
            return selected

    def refresh_external_occupancy(self) -> set[str]:
        """Poll foreign-visible occupancy and wake dispatch if eligibility changed."""
        rows = self._all_gpus()
        sampled = self._occupied_indices()
        with self._changed:
            if self._allowed is None:
                self._known_indices = tuple(sorted((idx for idx, _uuid in rows), key=int))
            previous = self._external_occupied or set()
            # Aggregate utilization/memory includes our RUNNING children. Keep
            # each internally-owned card's pre-claim external state; the per-PID
            # monitor below detects new foreign activity on owned cards.
            occupied = {
                idx
                for idx in self._known_indices
                if (idx in previous if idx in self._owned else idx in sampled)
            }
            changed = occupied != self._external_occupied
            self._external_occupied = occupied
            if changed:
                self._change_generation += 1
                self._external_history.append((time.time(), tuple(sorted(occupied, key=int))))
                self._changed.notify_all()
        return occupied

    def external_timeline(self) -> tuple[tuple[str, ...], list[tuple[float, tuple[str, ...]]]]:
        """Return an immutable snapshot of known GPUs and external occupancy changes."""
        with self._changed:
            return self._known_indices, list(self._external_history)

    def release_many(self, indices: tuple[str, ...] | list[str]) -> None:
        with self._changed:
            previous = set(self._owned)
            self._owned.difference_update(indices)
            if self._owned != previous:
                self._change_generation += 1
                self._changed.notify_all()

    def change_generation(self) -> int:
        with self._changed:
            return self._change_generation

    def wait_for_change(self, generation: int, stop: threading.Event) -> int:
        """Block a notifier thread until eligibility/ownership changes or stop."""
        with self._changed:
            self._changed.wait_for(
                lambda: self._change_generation != generation or stop.is_set()
            )
            return self._change_generation

    def wake(self) -> None:
        """Wake assignment waiters after cancellation or an internal event."""
        with self._changed:
            self._changed.notify_all()

    def release(self, idx: str) -> None:
        self.release_many((idx,))


# ── Tee stdout → run log ─────────────────────────────────────────────────────


class _Tee:
    """Write to multiple streams; flush on every write so the log is live.

    Locks per write so two threads' simultaneous writes don't interleave
    bytes. For atomic *lines*, callers should still hold _log_lock around
    the full print+flush sequence — see log() below.
    """

    def __init__(self, *streams):
        self._streams = streams
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
            for st in self._streams:
                st.write(s)
                st.flush()
        return len(s)

    def flush(self):
        with self._lock:
            for st in self._streams:
                st.flush()


# Thread-safe one-liner emitter. `print()` calls file.write() multiple times
# (once for the message, once for the trailing newline), so without this
# lock concurrent prints from worker threads can interleave halfway through
# a line. Use log() for any [bench-suite] status print from a worker thread.
_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)


# ── GPU probe ────────────────────────────────────────────────────────────────


def probe_gpu(idx: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Run PROBE_SCRIPT on a single GPU. Returns (ok, error_message)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = idx
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout:.0f}s"
    except Exception as e:
        return False, repr(e)
    if proc.returncode == 0 and "PROBE_OK" in proc.stdout:
        return True, ""
    msg = (proc.stderr or proc.stdout).strip().splitlines()
    return False, msg[-1] if msg else f"exit {proc.returncode}"


def detect_usable_gpus(
    candidates: list[str], probe_timeout: float
) -> tuple[set[str], dict[str, str]]:
    """Probe candidates in parallel. Returns (usable_set, failures)."""
    usable: set[str] = set()
    failures: dict[str, str] = {}
    if not candidates:
        return usable, failures
    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futs = {ex.submit(probe_gpu, idx, probe_timeout): idx for idx in candidates}
        for fut in as_completed(futs):
            idx = futs[fut]
            ok, err = fut.result()
            if ok:
                usable.add(idx)
                log(f"[bench-suite]   gpu {idx}: ok")
            else:
                failures[idx] = err
                log(f"[bench-suite]   gpu {idx}: FAIL — {err}")
    return usable, failures


def gpu_compile_profile(indices: set[str]) -> dict:
    """Read the homogeneous compile profile for late-bindable GPUs via NVML."""
    if not indices:
        raise ValueError("cannot build a GPU compile profile for an empty GPU set")
    try:
        import pynvml

        pynvml.nvmlInit()
        profiles = []
        for index in sorted(indices, key=int):
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            name = str(pynvml.nvmlDeviceGetName(handle))
            compute_capability = tuple(pynvml.nvmlDeviceGetCudaComputeCapability(handle))
            core_count = int(pynvml.nvmlDeviceGetNumGpuCores(handle))
            if compute_capability == (10, 0):
                cores_per_sm = 128
            else:
                raise ValueError(
                    f"GPU {index} {name!r} has unsupported compile profile "
                    f"sm{compute_capability[0]}{compute_capability[1]}"
                )
            if core_count % cores_per_sm:
                raise ValueError(
                    f"GPU {index} {name!r} reports {core_count} cores, not divisible "
                    f"by {cores_per_sm} cores/SM"
                )
            profiles.append(
                {
                    "name": name,
                    "compute_capability": list(compute_capability),
                    "num_sms": core_count // cores_per_sm,
                }
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    first = profiles[0]
    if any(profile != first for profile in profiles[1:]):
        raise ValueError(
            "eligible GPUs have heterogeneous compile profiles; late binding requires "
            f"one profile per prepared child, got {profiles}"
        )
    return first


# ── Workload execution ───────────────────────────────────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _visible_gpu_rows(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Apply the process CUDA visibility filter to physical NVML rows."""

    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        return rows
    visible = {item.strip() for item in configured.split(",") if item.strip()}
    return [(index, uuid) for index, uuid in rows if index in visible or uuid in visible]


def _pid_sm_on_gpus(gpu_indices: tuple[str, ...]) -> dict[int, float] | None:
    """Return the maximum per-process SM utilization on assigned GPUs.

    ``None`` means the utilization snapshot failed and the caller must fail
    closed.  Inactive rows use ``-`` for SM utilization and are equivalent to
    zero, so a process that only keeps a CUDA context resident is shareable.
    """
    if not gpu_indices:
        return {}
    try:
        import pynvml

        pynvml.nvmlInit()
        result: dict[int, float] = {}
        # The API reports samples since a wall-clock timestamp in microseconds.
        # A one-second window covers the monitor cadence without retaining stale
        # activity long enough to delay a newly eligible assignment.
        since_us = int((time.time() - 1.0) * 1_000_000)
        for gpu_index in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            try:
                samples = pynvml.nvmlDeviceGetProcessUtilization(handle, since_us)
            except pynvml.NVMLError_NotFound:
                samples = ()
            for sample in samples:
                pid = int(sample.pid)
                result[pid] = max(result.get(pid, 0.0), float(sample.smUtil))
        return result
    except (ImportError, AttributeError, ValueError):
        pass
    except Exception:
        # Keep the existing fail-closed subprocess sampler as a compatibility
        # fallback for NVML versions without process-utilization support.
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "pmon", "-i", ",".join(gpu_indices), "-c", "1", "-s", "u"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None

    result: dict[int, float] = {}
    assigned = set(gpu_indices)
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0] not in assigned:
            continue
        try:
            pid = int(fields[1])
            sm = float(fields[3])
        except ValueError:
            continue
        result[pid] = max(result.get(pid, 0.0), sm)
    return result


def _active_strangers(
    gpu_indices: tuple[str, ...], our_pids: set[int], sm_threshold: float
) -> dict[int, float] | None:
    """Foreign PIDs whose sampled SM utilization exceeds the threshold."""
    snapshot = _pid_sm_on_gpus(gpu_indices)
    if snapshot is None:
        return None
    return {pid: sm for pid, sm in snapshot.items() if pid not in our_pids and sm > sm_threshold}


class _BenchPidRegistry:
    """Bench subprocess PIDs spawned by this orchestrator (register at Popen)."""

    _lock = threading.Lock()
    _roots: ClassVar[set[int]] = set()
    _our_pids_cache: ClassVar[tuple[float, set[int]] | None] = None

    @classmethod
    def register(cls, pid: int) -> None:
        with cls._lock:
            cls._roots.add(pid)
            cls._our_pids_cache = None

    @classmethod
    def unregister(cls, pid: int) -> None:
        with cls._lock:
            cls._roots.discard(pid)
            cls._our_pids_cache = None


def _proc_children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            pid = int(entry)
            with open(f"/proc/{entry}/stat") as f:
                data = f.read()
            rparen = data.rfind(")")
            fields = data[rparen + 2 :].split()
            ppid = int(fields[1])
            children.setdefault(ppid, []).append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return children


def _descendants_of(roots: set[int]) -> set[int]:
    if not roots:
        return set()
    children = _proc_children_map()
    out: set[int] = set()
    stack = list(roots)
    while stack:
        p = stack.pop()
        for c in children.get(p, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


_OUR_PIDS_TTL = 0.25  # seconds; amortize /proc walks across pmon polls
_RESOURCE_SAMPLE_INTERVAL = 0.25


def _our_pids() -> set[int]:
    """Orchestrator + every registered bench subprocess and its descendants."""
    now = time.time()
    with _BenchPidRegistry._lock:
        hit = _BenchPidRegistry._our_pids_cache
        if hit is not None and now - hit[0] < _OUR_PIDS_TTL:
            return hit[1]
        roots = set(_BenchPidRegistry._roots)
    pids = {os.getpid()} | roots | _descendants_of(roots)
    with _BenchPidRegistry._lock:
        _BenchPidRegistry._our_pids_cache = (now, pids)
    return pids


def _process_tree_resources(root_pids: set[int]) -> dict[str, int]:
    """Sample RSS, open FDs, and process count for owned process trees."""
    pids = set(root_pids) | _descendants_of(root_pids)
    rss_bytes = 0
    open_fds = 0
    observed_processes = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        rss_bytes += int(line.split()[1]) * 1024
                        break
            open_fds += len(os.listdir(f"/proc/{pid}/fd"))
            observed_processes += 1
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return {
        "rss_bytes": rss_bytes,
        "open_fds": open_fds,
        "processes": observed_processes,
    }


def _terminate_subprocess(proc: subprocess.Popen) -> None:
    """Terminate a bench process group, escalating to SIGKILL when needed."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()


@dataclass
class _PreparedAttempt:
    """One process-local prepare/GPU attempt and its owned resources."""

    workload: dict
    attempt: int
    process: subprocess.Popen
    control: socket.socket
    log_file: object
    log_path: Path
    workdir: str
    state: str = "PREPARING_CPU"
    buffer: bytearray = field(default_factory=bytearray)
    gpus: tuple[str, ...] = ()
    gpu_ownership_released: bool = False
    timeline: dict[str, float] = field(default_factory=dict)
    record: dict | None = None
    terminal_at: float | None = None

    @property
    def label(self) -> str:
        return f"{self.workload['kernel']}/{self.workload['config']}"


def _prepared_child_command(
    workload: dict,
    *,
    control_fd: int,
    rounds: int,
    cooldown: float,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tirx_kernels.bench",
        "--kernel",
        workload["kernel"],
        "--config",
        workload["config"],
        "--prepared-control-fd",
        str(control_fd),
        "--prepared-num-gpus",
        str(workload.get("num_gpus", 1)),
        "--rounds",
        str(rounds),
        "--cooldown",
        str(cooldown),
    ]
    if workload.get("warmup") is not None:
        command += ["--warmup", str(workload["warmup"])]
    if workload.get("repeat") is not None:
        command += ["--repeat", str(workload["repeat"])]
    if workload.get("timer") is not None:
        command += ["--timer", workload["timer"]]
    return command


def _spawn_prepared_attempt(
    workload: dict,
    attempt: int,
    log_dir: Path,
    *,
    rounds: int,
    cooldown: float,
    compile_profile: dict,
) -> _PreparedAttempt:
    """Spawn a GPU-unbound child that immediately begins CPU prepare."""
    parent_control, child_control = socket.socketpair()
    parent_control.setblocking(False)
    kernel = workload["kernel"]
    config = workload["config"]
    log_path = log_dir / f"{kernel}__{config}__a{attempt}.log"
    log_file = open(log_path, "w")
    workdir = tempfile.mkdtemp(prefix=f"bench-suite-{kernel}-{config}-")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["TIRX_BENCH_JSON"] = "1"
    env[PREPARE_NUM_SMS_ENV] = str(compile_profile["num_sms"])
    repo_root = str(_kernels_repo_root())
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not python_path else f"{repo_root}{os.pathsep}{python_path}"
    cache_dir = (log_dir.parent / "cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["TIRX_BENCH_CACHE_DIR"] = str(cache_dir)
    command = _prepared_child_command(
        workload,
        control_fd=child_control.fileno(),
        rounds=rounds,
        cooldown=cooldown,
    )
    process_started = time.time()
    try:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=workdir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            pass_fds=(child_control.fileno(),),
            start_new_session=True,
        )
    except BaseException:
        parent_control.close()
        child_control.close()
        log_file.close()
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    child_control.close()
    _BenchPidRegistry.register(process.pid)
    return _PreparedAttempt(
        workload=workload,
        attempt=attempt,
        process=process,
        control=parent_control,
        log_file=log_file,
        log_path=log_path,
        workdir=workdir,
        timeline={"process_started": process_started},
    )


def _send_child(attempt: _PreparedAttempt, message: dict) -> None:
    attempt.control.sendall(json.dumps(message, separators=(",", ":")).encode() + b"\n")


def _receive_child_messages(attempt: _PreparedAttempt) -> tuple[list[dict], bool]:
    """Drain complete control messages; return ``(messages, eof)``."""
    eof = False
    while True:
        try:
            chunk = attempt.control.recv(65536)
        except BlockingIOError:
            break
        if not chunk:
            eof = True
            break
        attempt.buffer.extend(chunk)
    messages: list[dict] = []
    while True:
        newline = attempt.buffer.find(b"\n")
        if newline < 0:
            break
        line = bytes(attempt.buffer[:newline])
        del attempt.buffer[: newline + 1]
        if line:
            messages.append(json.loads(line))
    if eof and attempt.buffer:
        messages.append(json.loads(bytes(attempt.buffer)))
        attempt.buffer.clear()
    return messages, eof


def _base_attempt_record(attempt: _PreparedAttempt) -> dict:
    workload = attempt.workload
    gpu_csv = ",".join(attempt.gpus)
    return {
        "kernel": workload["kernel"],
        "config": workload["config"],
        "label": workload["config"],
        "gpu": gpu_csv,
        "gpus": list(attempt.gpus),
        "num_gpus": workload.get("num_gpus", 1),
        "attempt": attempt.attempt,
        "process_pid": attempt.process.pid,
        "execution_mode": "pipeline",
        "process_model": "one_shot_child_per_workload_attempt",
        "started_at": datetime.fromtimestamp(
            attempt.timeline["process_started"], timezone.utc
        ).isoformat(timespec="seconds"),
        "phase_timestamps": attempt.timeline,
    }


def _finish_attempt_process(attempt: _PreparedAttempt) -> None:
    """Close host resources after the already-reaped child."""
    attempt.timeline.setdefault("process_reaped", time.time())
    _BenchPidRegistry.unregister(attempt.process.pid)
    try:
        attempt.control.close()
    finally:
        attempt.log_file.close()
        shutil.rmtree(attempt.workdir, ignore_errors=True)


def _record_child_failure(attempt: _PreparedAttempt, message: dict | None = None) -> dict:
    record = _base_attempt_record(attempt)
    if message is None:
        error = f"prepared child exited {attempt.process.returncode} without a terminal message"
    else:
        for phase_name in ("gpu_started", "gpu_finished"):
            if phase_name in message:
                attempt.timeline[phase_name] = message[phase_name]
        if "gpu_finished" in message:
            attempt.timeline["result_received"] = time.time()
        phase = message.get("phase", "unknown")
        error = f"{phase}: {message.get('error', 'unknown child failure')}"
        if message.get("traceback"):
            error += "\n" + message["traceback"]
    record.update({"status": "FAIL", "error": error, "finished_at": now_iso()})
    return record


# ── Output ───────────────────────────────────────────────────────────────────


def git_label(repo: Path) -> str | None:
    if not repo.exists():
        return None
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def _tir_repo_root() -> Path | None:
    """TVM git root: TVM_PATH env, else installed tvm package checkout."""
    env = os.environ.get("TVM_PATH")
    if env:
        p = Path(env).resolve()
        if (p / "python" / "tvm").is_dir():
            return p
    return _module_repo_root("tvm")


def _module_repo_root(import_name: str) -> Path | None:
    """Git root of an importable package, if it's a local checkout."""
    try:
        mod = __import__(import_name)
    except Exception:
        return None
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        return None
    for p in [Path(pkg_file).resolve().parent, *Path(pkg_file).resolve().parents]:
        if (p / ".git").exists():
            return p
    return None


def collect_repo_git() -> dict[str, str | None]:
    """SHAs for the three repos involved: tvm, tirx-kernels, tirx-bench-ci."""
    tir_root = _tir_repo_root()
    tirx_root = _module_repo_root("tirx_kernels") or _kernels_repo_root()
    bench_ci_root: Path | None = None
    for base in (tirx_root, tir_root):
        if base is None:
            continue
        candidate = base.parent / "tirx-bench-ci"
        if (candidate / ".git").exists():
            bench_ci_root = candidate
            break
    return {
        "tir": git_label(tir_root) if tir_root else None,
        "tirx-kernels": git_label(tirx_root) if tirx_root else None,
        "tirx-bench-ci": git_label(bench_ci_root) if bench_ci_root else None,
    }


def collect_kernel_fingerprint() -> dict[str, str | None]:
    """Merge-stable content fingerprints (git *tree* SHAs) of the source that
    determines kernel codegen + perf.

    The commit SHAs in ``collect_repo_git`` are rewritten by a squash/rebase
    merge, so a baseline that records only commit SHAs can't be mapped back to a
    mainline commit afterwards. A git tree SHA is content-addressed (Merkle): it
    is identical before and after a merge as long as the directory's content is
    unchanged. Confirm a checkout matches a recorded baseline with
    ``git rev-parse HEAD:<path>``.
    """
    tir_root = _tir_repo_root()
    tirx_root = _module_repo_root("tirx_kernels") or _kernels_repo_root()

    def _tree(root: Path | None, path: str) -> str | None:
        if root is None:
            return None
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", f"HEAD:{path}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        return out.stdout.strip() or None

    return {
        "tir:python/tvm/tirx": _tree(tir_root, "python/tvm/tirx"),
        "tirx-kernels:tirx_kernels": _tree(tirx_root, "tirx_kernels"),
    }


# Packages used as baselines in workloads.yaml — anything our regression
# numbers compare against, so the recorded version pins the comparison.
BASELINE_PACKAGES = [
    "torch",
    "deep_gemm",
    "flashinfer",
    "flash_kda",
    "flash_attn",
    "sglang",
    "cutlass",
]


def package_provenance(import_name: str) -> dict | None:
    """Probe a Python package: version + (if editable git install) repo + SHA.

    Returns None when neither the package nor distribution metadata exists.
    """

    def _record_git(path: Path, info: dict) -> None:
        try:
            root = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if not root:
                return
            sha = subprocess.run(
                ["git", "-C", root, "rev-parse", "--short=8", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if not sha:
                return
            dirty = subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            info["git_dir"] = root
            info["git_sha"] = sha + ("-dirty" if dirty else "")
        except Exception:
            pass

    dists: list[str] = []
    try:
        from importlib.metadata import distribution as _probe_dist

        _probe_dist(import_name)
        dists.append(import_name)
    except Exception:
        pass
    try:
        from importlib.metadata import packages_distributions

        for dist_name in packages_distributions().get(import_name) or []:
            if dist_name not in dists:
                dists.append(dist_name)
    except Exception:
        pass
    if not dists:
        dists = [import_name]

    mod = None
    try:
        mod = __import__(import_name)
    except Exception:
        pass
    info: dict = {"importable": mod is not None}
    # Heavy optional baselines can fail during their top-level import even when
    # their source checkout is discoverable on PYTHONPATH. Resolve the module
    # spec without executing it so provenance still records that checkout.
    try:
        spec = find_spec(import_name)
    except Exception:
        spec = None
    if spec is not None:
        source_dir = None
        if spec.origin and spec.origin not in ("built-in", "frozen"):
            source_dir = Path(spec.origin).resolve().parent
        elif spec.submodule_search_locations:
            source_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        if source_dir is not None:
            info.setdefault("source_dir", str(source_dir))
            _record_git(source_dir, info)
    # Version: prefer __version__, else importlib.metadata. Top-level import
    # name and the distribution name often disagree (e.g. flash_attn ↔
    # flash-attn-4) — use packages_distributions() to bridge.
    version = getattr(mod, "__version__", None) if mod is not None else None
    if version is None:
        try:
            from importlib.metadata import version as _meta_version

            for d in dists:
                try:
                    version = _meta_version(d)
                    if version is not None:
                        info["dist"] = d
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if version is not None:
        info["version"] = str(version)
    if import_name == "torch":
        cuda = getattr(getattr(mod, "version", None), "cuda", None)
        git_v = getattr(getattr(mod, "version", None), "git_version", None)
        if cuda:
            info["cuda"] = str(cuda)
        if git_v:
            info["torch_git_version"] = str(git_v)
    # PEP 610 direct_url.json: when a package was `pip install -e <path>` or
    # `pip install <path>`, pip writes the source path/URL into the dist-info.
    # This catches the editable case (the package lives outside the repo it
    # was built from, so the __file__ walk below misses it). dist resolution:
    # prefer `info["dist"]` if we set it above, else default to import_name.
    try:
        from importlib.metadata import distribution as _meta_dist

        dist = None
        for dist_name in [info.get("dist"), *dists, import_name]:
            if not dist_name:
                continue
            try:
                dist = _meta_dist(dist_name)
                info.setdefault("dist", dist.metadata["Name"])
                break
            except Exception:
                continue
        if dist is not None:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct = json.loads(direct_url_text)
                url = direct.get("url") or ""
                if url.startswith("file://"):
                    src_path = Path(url[len("file://") :]).resolve()
                    info["source_dir"] = str(src_path)
                    if direct.get("dir_info", {}).get("editable"):
                        info["editable"] = True
                    _record_git(src_path, info)
    except Exception:
        pass
    if mod is None:
        return info if "version" in info or "source_dir" in info else None
    # Resolve a directory we can git-probe. Namespace packages and some
    # __init__.py-less namespaces set mod.__file__ to None — fall back to
    # __path__[0] then to a known submodule's file.
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        # Last resort: try to import a likely submodule with a real file.
        for sub in (".cute", ".csrc", ".jit_kernels", ".jit"):
            try:
                submod = __import__(import_name + sub, fromlist=["__file__"])
                if getattr(submod, "__file__", None):
                    pkg_file = submod.__file__
                    break
            except Exception:
                continue
    if pkg_file:
        pkg_dir = Path(pkg_file).resolve().parent
        # Walk up looking for a git repo. .git can be a dir (regular clone)
        # or a file (worktree); both are fine for `git rev-parse`.
        _record_git(pkg_dir, info)
    return info


def collect_baseline_provenance() -> dict:
    return {name: package_provenance(name) or {"installed": False} for name in BASELINE_PACKAGES}


def write_run(
    out_dir: Path,
    stamp: str,
    results: list[dict],
    label: str | None,
    probe: dict | None = None,
    pipeline: dict | None = None,
) -> Path:
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": stamp,
        "label": label,
        "git": collect_repo_git(),
        "kernel_tree": collect_kernel_fingerprint(),
        "baselines": collect_baseline_provenance(),
        "probe": probe or {},
        "pipeline": pipeline or {},
        "results": results,
    }
    path = runs_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


#: Which reference a kernel's ratio is quoted against, for the kernels that
#: benchmark against more than one.  A kernel with a single non-ours impl needs
#: no entry -- `_baseline_impl` derives it from the row.
BASELINE_IMPL_BY_KERNEL = {
    "fp16_bf16_gemm": "torch-cublas",
    "nvfp4_gemm": "flashinfer",
    "flashkda_bf16_fused_m128": "flashinfer_m128",
    "deepgemm_sm100_fp8_paged_mqa_logits": "deepgemm",
    "sparse_flashmla_prefill_head64_phase1": "flashmla",
    "sparse_flashmla_prefill_head128_phase1": "flashmla",
}


def _baseline_impl(kernel: str, impl_names: list[str]) -> str | None:
    """The reference a kernel's ratio is quoted against.

    Most kernels have exactly one non-ours implementation, so naming it in a
    hand-maintained table only creates a row that can be forgotten -- and four
    kernels already had been, printing a bare `ratio` header.  The table now
    only breaks genuine ties.
    """
    pinned = BASELINE_IMPL_BY_KERNEL.get(kernel)
    if pinned:
        return pinned
    ours = set(our_impls(dict.fromkeys(impl_names)))
    others = [name for name in impl_names if name not in ours]
    return others[0] if len(others) == 1 else None


def _our_impl(row_impls: dict) -> str | None:
    """Pick the first TIR/TIRx implementation from a row's impls dict."""
    return next(iter(our_impls(row_impls)), None)


def _run_measurement_protocol(current: dict) -> dict[str, Any] | None:
    """Return run-level rounds/cooldown, deriving older artifacts from result rows."""
    pipeline = current.get("pipeline") or {}
    declared = pipeline.get("measurement_protocol")
    if isinstance(declared, dict):
        return declared

    observed = set()
    for result in current.get("results") or []:
        protocol = result.get("benchmark_protocol") or {}
        rounds = protocol.get("rounds")
        cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
        if (
            isinstance(rounds, int)
            and not isinstance(rounds, bool)
            and isinstance(cooldown, (int, float))
            and not isinstance(cooldown, bool)
        ):
            observed.add((rounds, float(cooldown)))
    if len(observed) != 1:
        return None
    rounds, cooldown = observed.pop()
    return {
        "rounds": rounds,
        "cooldown_s": cooldown,
        "default_rounds": DEFAULT_ROUNDS,
        "default_cooldown_s": DEFAULT_COOLDOWN_S,
        "is_default": rounds == DEFAULT_ROUNDS
        and math.isclose(
            cooldown,
            DEFAULT_COOLDOWN_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "source": "derived_from_result_protocols",
    }


def write_summary(out_dir: Path, current: dict) -> Path:
    """Human-readable per-run report, grouped by kernel.

    Times are in µs to match the existing bench-suite doc convention. Per row:
    config, one column per impl present in that kernel, baseline/ours ratio
    (against the kernel's reference impl from BASELINE_IMPL_BY_KERNEL),
    then attempt + gpu.
    """
    stamp = current["timestamp"]
    reports_dir = out_dir / "reports" / stamp
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# bench-suite run {stamp}")
    lines.append("")
    label = current.get("label") or "-"
    git = current.get("git") or {}
    lines.append(f"- label: `{label}`")
    lines.append(
        f"- git: tir=`{git.get('tir') or '-'}`  "
        f"tirx-kernels=`{git.get('tirx-kernels') or '-'}`  "
        f"tirx-bench-ci=`{git.get('tirx-bench-ci') or '-'}`"
    )
    statuses: dict[str, int] = {}
    for r in current.get("results") or []:
        s = r.get("status") or "?"
        statuses[s] = statuses.get(s, 0) + 1
    status_line = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    lines.append(f"- status: {status_line} (over {sum(statuses.values())} workloads)")
    lines.append("")

    pipeline = current.get("pipeline") or {}
    cost = pipeline.get("cost_model") or {}
    if pipeline:
        measurement = _run_measurement_protocol(current) or {}
        if measurement and not measurement.get("is_default", False):
            lines.append(
                "> **DIAGNOSTIC, NON-DEFAULT MEASUREMENT PROTOCOL:** "
                f"this run used `{measurement.get('rounds', '?')}` round(s) and "
                f"`{measurement.get('cooldown_s', '?')}`s cooldown; the acceptance "
                f"default is `{measurement.get('default_rounds', DEFAULT_ROUNDS)}` "
                f"rounds and `{measurement.get('default_cooldown_s', DEFAULT_COOLDOWN_S)}`s. "
                "Do not use this run as default-protocol performance evidence."
            )
            lines.append("")
        lines.append("## Pipeline critical path")
        lines.append("")
        lines.append(
            "Parallel CPU preparation overlaps across one-shot children; only the first "
            "READY latency is charged before the GPU schedule."
        )
        lines.append("")
        lines.append(f"- process model: `{pipeline.get('process_model', '?')}`")
        lines.append(f"- cost-model evidence: `{cost.get('measurement_status', 'missing')}`")
        if cost.get("measurement_status") == "measured":
            lines.append(f"- observed critical wall: `{cost['observed_critical_s']:.3f}s`")
            lines.append(
                f"- first child spawn / spawn span: "
                f"`{cost['first_process_started_s']:.3f}s` / "
                f"`{cost['prepare_spawn_span_s']:.3f}s`"
            )
            lines.append(f"- first READY latency: `{cost['first_ready_s']:.3f}s`")
            lines.append(
                "- ideal GPU list schedule (all workloads READY at time zero): "
                f"`{cost['ideal_gpu_list_schedule_s']:.3f}s`"
            )
            lines.append(
                "- READY-constrained GPU list schedule: "
                f"`{cost['ready_constrained_gpu_list_schedule_s']:.3f}s`"
            )
            lines.append(
                "- eligibility-constrained GPU list schedule: "
                f"`{cost['eligibility_constrained_gpu_list_schedule_s']:.3f}s`"
            )
            lines.append(f"- READY starvation: `{cost['ready_starvation_s']:.3f}s`")
            lines.append(f"- foreign GPU wait: `{cost['foreign_wait_s']:.3f}s`")
            lines.append(f"- expected critical wall: `{cost['expected_s']:.3f}s`")
            lines.append(
                "- expected critical wall including foreign occupancy: "
                f"`{cost['expected_with_foreign_s']:.3f}s`"
            )
            lines.append(f"- unexplained residual: `{cost['unexplained_s']:.3f}s`")
            dispatch = cost["dispatch_latency_s"]
            handoff = cost["assignment_handoff_s"]
            lines.append(
                f"- dispatch latency p95/max: `{dispatch['p95']:.3f}s` / "
                f"`{dispatch['max']:.3f}s`"
            )
            lines.append(
                f"- ASSIGN-to-GPU-start p95/max: `{handoff['p95']:.3f}s` / "
                f"`{handoff['max']:.3f}s`"
            )
        else:
            lines.append(f"- missing reason: {cost.get('missing_reason', 'unknown')}")
            lines.append(
                f"- complete GPU timelines: `{cost.get('complete_timeline_count', 0)}` / "
                f"`{cost.get('record_count', 0)}`"
            )
            lines.append(
                "- complete cost-model measurements: "
                f"`{cost.get('complete_measurement_count', 0)}` / "
                f"`{cost.get('record_count', 0)}`"
            )
            lines.append(
                "- expected wall, residual, starvation, foreign wait, and latency "
                "percentiles are intentionally unpublished because the run lacks "
                "complete measurement evidence"
            )
        lines.append(
            f"- final process-reap tail: `{pipeline.get('final_reap_tail_s', 0.0):.3f}s`"
        )
        lines.append(
            f"- observed bounds: preparing={pipeline.get('max_observed_preparing', 0)}, "
            f"READY={pipeline.get('max_observed_ready', 0)}, "
            f"buffered={pipeline.get('max_observed_buffered', 0)}, "
            f"active children={pipeline.get('max_observed_active_children', 0)}"
        )
        lines.append(
            f"- host resource peaks: process tree="
            f"{pipeline.get('max_observed_process_tree', 0)}, "
            f"RSS={pipeline.get('max_observed_rss_bytes', 0) / (1024**3):.3f} GiB, "
            f"FDs={pipeline.get('max_observed_open_fds', 0)}"
        )
        lines.append("")

        phase_rows = []
        for record in current.get("results") or []:
            breakdown = _workload_phase_breakdown(record)
            if breakdown is not None:
                phase_rows.append((record, breakdown))
        phase_rows.sort(key=lambda pair: pair[0]["phase_timestamps"]["gpu_started"])
        if phase_rows:
            lines.append("### Workload phase breakdown")
            lines.append("")
            lines.append(
                "Times are wall-clock seconds. `specialize/compile` includes workload "
                "specialization, IR generation, and compilation after config resolution."
            )
            lines.append("")
            lines.append(
                "| workload | GPU | startup | CLI bootstrap | framework import | "
                "exact import | config resolve | specialize/compile | CPU prepare | "
                "READY wait | ASSIGN handoff | GPU stage | result handoff | reap tail |"
            )
            lines.append(
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            )
            for record, breakdown in phase_rows:
                workload = f"{record['kernel']}/{record.get('config') or record.get('label')}"
                lines.append(
                    f"| `{workload}` | `{record.get('gpu') or '-'}` | "
                    f"{breakdown['process_startup_s']:.3f} | "
                    f"{breakdown['cli_bootstrap_s']:.3f} | "
                    f"{breakdown['framework_import_s']:.3f} | "
                    f"{breakdown['exact_import_s']:.3f} | "
                    f"{breakdown['config_resolve_s']:.3f} | "
                    f"{breakdown['specialize_generate_compile_s']:.3f} | "
                    f"{breakdown['cpu_prepare_s']:.3f} | "
                    f"{breakdown['ready_wait_s']:.3f} | "
                    f"{breakdown['assignment_handoff_s']:.3f} | "
                    f"{breakdown['gpu_stage_s']:.3f} | "
                    f"{breakdown['result_handoff_s']:.3f} | "
                    f"{breakdown['process_reap_tail_s']:.3f} |"
                )
            lines.append("")

        exemption = pipeline.get("multi_gpu_runtime_validation") or {}
        if exemption:
            lines.append("## Multi-GPU runtime validation exemption")
            lines.append("")
            lines.append(
                f"- validation status: `{exemption.get('validation_status', '?')}`"
            )
            lines.append(
                "- runtime evidence: not collected, by explicit human direction; this is "
                "neither a pass nor a missing-result placeholder"
            )
            lines.append(
                "- migrated semantics: pipeline-only late assignment, atomic full-GPU "
                "claim before rank/CUDA startup, barriers, sample-wise max aggregation, "
                "Kineto spans, and process-group cleanup"
            )
            lines.append(
                "- non-multi-GPU evidence: protocol assignment-count rejection, atomic "
                "claim failure, and rank lifecycle ordering"
            )
            lines.append("")

    baselines = current.get("baselines") or {}
    if baselines:
        lines.append("## Baseline impl provenance")
        lines.append("")
        for name, info in sorted(baselines.items()):
            if not info or info.get("installed") is False:
                lines.append(f"- `{name}`: not installed")
                continue
            bits = []
            if "version" in info:
                bits.append(f"v{info['version']}")
            if "cuda" in info:
                bits.append(f"cuda={info['cuda']}")
            if "torch_git_version" in info:
                bits.append(f"torch_git={info['torch_git_version'][:12]}")
            if "git_sha" in info:
                bits.append(f"@`{info['git_sha']}`")
            if "git_dir" in info:
                bits.append(f"({info['git_dir']})")
            lines.append(f"- `{name}`: {' '.join(bits) if bits else '?'}")
        lines.append("")

    # Group by kernel
    by_kernel: dict[str, list[dict]] = {}
    for r in current.get("results") or []:
        by_kernel.setdefault(r["kernel"], []).append(r)

    for kernel in sorted(by_kernel):
        rows = sorted(by_kernel[kernel], key=lambda r: r.get("label") or r.get("config") or "")
        # Discover all impl names that appear in this kernel
        impl_names: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for impl in r.get("impls") or {}:
                if impl not in seen:
                    seen.add(impl)
                    impl_names.append(impl)
        impl_names.sort()
        baseline_impl = _baseline_impl(kernel, impl_names)
        # Determine "ours" impl name once for the whole kernel (constant per kernel)
        ours_impl = None
        for r in rows:
            ours_impl = _our_impl(r.get("impls") or {})
            if ours_impl:
                break
        ratio_label = f"{baseline_impl}/{ours_impl}" if baseline_impl and ours_impl else "ratio"
        lines.append(f"## `{kernel}`")
        if baseline_impl and ours_impl:
            lines.append("")
            lines.append(
                f"_baseline impl_: `{baseline_impl}` · _ours_: `{ours_impl}` · "
                f"_ratio_ = baseline/ours · `>1` means ours is faster"
            )
        lines.append("")
        # Table header
        header = ["config", *impl_names, ratio_label, "attempt", "gpus"]
        align = ["---"] + ["---:"] * len(impl_names) + ["---:", "---:", "---:"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(align) + "|")
        for r in rows:
            cfg = r.get("label") or r.get("config") or "?"
            status = r.get("status", "ok")
            impls = r.get("impls") or {}
            row = [cfg]
            for impl in impl_names:
                us = impls.get(impl)
                row.append(f"{us:.2f}" if us is not None else "—")
            # Ratio column
            ratio_cell = "—"
            if baseline_impl and ours_impl:
                base_us = impls.get(baseline_impl)
                ours_us = impls.get(ours_impl)
                if base_us is not None and ours_us is not None and ours_us > 0:
                    ratio = base_us / ours_us
                    # Bold values that flag a regression risk (we're slower)
                    ratio_cell = f"**{ratio:.3f}**" if ratio < 1.0 else f"{ratio:.3f}"
            row.append(ratio_cell)
            if status != "ok":
                row[0] = f"{cfg} **[{status}]**"
            row.append(str(r.get("attempt", 1)))
            row.append(str(r.get("gpu", "-")))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path = reports_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def load_baseline(path=None):
    """Load the pinned baseline.json, or None if no baseline exists yet.

    ``path`` (optional) overrides the default baseline location."""
    p = Path(path) if path is not None else DEFAULT_BASELINE
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ── Main ─────────────────────────────────────────────────────────────────────


def _finalize_bench_record(row: dict, *, rounds: int, cooldown: float) -> None:
    """Validate in-bench round samples and write aggregated impl times (microseconds)."""
    required_fields = ("round_samples", "errors", "timer", "benchmark_protocol")
    missing_fields = [field for field in required_fields if field not in row]
    if missing_fields:
        row["status"] = "FAIL"
        row["error"] = f"bench result is missing required field(s): {missing_fields}"
        return
    if not isinstance(row["errors"], dict):
        row["status"] = "FAIL"
        row["error"] = "bench result field 'errors' must be a mapping"
        return
    if not isinstance(row["timer"], str) or not row["timer"]:
        row["status"] = "FAIL"
        row["error"] = "bench result field 'timer' must be a non-empty string"
        return
    protocol = row["benchmark_protocol"]
    if not isinstance(protocol, dict):
        row["status"] = "FAIL"
        row["error"] = "bench result field 'benchmark_protocol' must be a mapping"
        return
    if protocol.get("rounds") != rounds:
        row["status"] = "FAIL"
        row["error"] = (
            "benchmark protocol round count does not match suite request: "
            f"{protocol.get('rounds')!r} != {rounds}"
        )
        return
    if protocol.get("round_aggregate") != "mean":
        row["status"] = "FAIL"
        row["error"] = "benchmark protocol must declare round_aggregate='mean'"
        return
    protocol_cooldown = protocol.get(
        "cooldown_s",
        protocol.get("round_cooldown_s"),
    )
    if (
        not isinstance(protocol_cooldown, (int, float))
        or isinstance(protocol_cooldown, bool)
        or not math.isclose(float(protocol_cooldown), cooldown, rel_tol=0.0, abs_tol=1e-9)
    ):
        row["status"] = "FAIL"
        row["error"] = (
            "benchmark protocol cooldown does not match suite request: "
            f"{protocol_cooldown!r} != {cooldown}"
        )
        return

    baseline_errors = row["errors"]
    if baseline_errors:
        details = "; ".join(f"{name}: {error}" for name, error in baseline_errors.items())
        row["status"] = "FAIL"
        row["error"] = f"baseline error(s): {details}"
        return
    samples = row["round_samples"]
    if not isinstance(samples, dict) or not samples:
        row["status"] = "FAIL"
        row["error"] = "bench result field 'round_samples' must be a non-empty mapping"
        return
    bad = {
        impl: len(vals) if isinstance(vals, list) else type(vals).__name__
        for impl, vals in samples.items()
        if not isinstance(vals, list) or len(vals) != rounds
    }
    if bad:
        row["status"] = "FAIL"
        row["error"] = f"expected {rounds} round(s) per impl, got {bad}"
        return
    invalid = {
        impl: value
        for impl, values in samples.items()
        for value in values
        if not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    }
    if invalid:
        row["status"] = "FAIL"
        row["error"] = f"round samples must be finite positive numbers: {invalid}"
        return
    sample_order = list(samples)
    if "order" in protocol:
        if protocol["order"] != sample_order:
            row["status"] = "FAIL"
            row["error"] = (
                "benchmark protocol implementation order does not match round samples: "
                f"{protocol['order']!r} != {sample_order!r}"
            )
            return
    else:
        round_orders = protocol.get("round_orders")
        if (
            not isinstance(round_orders, list)
            or len(round_orders) != rounds
            or any(
                not isinstance(order, list)
                or len(order) != len(sample_order)
                or set(order) != set(sample_order)
                for order in round_orders
            )
        ):
            row["status"] = "FAIL"
            row["error"] = (
                "benchmark protocol must declare one implementation order or one valid "
                "permutation per round"
            )
            return
    row["impls"] = {impl: statistics.mean(vals) for impl, vals in samples.items()}
    row["aggregated"] = {"rounds": rounds, "method": "mean"}
    row["status"] = "ok"


def _pipeline_cost_model(
    records: list[dict],
    *,
    run_started: float,
    critical_finished: float,
    known_gpu_indices: tuple[str, ...],
    external_history: list[tuple[float, tuple[str, ...]]],
) -> dict[str, Any]:
    required_timeline_phases = (
        "process_started",
        "child_started",
        "prepare_started",
        "framework_import_started",
        "framework_loaded",
        "module_loaded",
        "config_resolved",
        "ready",
        "assigned",
        "gpu_started",
        "gpu_finished",
        "result_received",
        "process_reaped",
    )
    known_gpus = set(known_gpu_indices)

    def has_complete_timeline(record: dict) -> bool:
        timeline = record.get("phase_timestamps") or {}
        return record.get("status") == "ok" and all(
            name in timeline for name in required_timeline_phases
        )

    def has_valid_assignment(record: dict) -> bool:
        gpus = record.get("gpus")
        if not isinstance(gpus, (list, tuple)) or not gpus:
            return False
        assigned_gpus = tuple(str(gpu) for gpu in gpus)
        required_num_gpus = record.get("num_gpus", len(assigned_gpus))
        return (
            type(required_num_gpus) is int
            and required_num_gpus == len(assigned_gpus)
            and len(set(assigned_gpus)) == len(assigned_gpus)
            and set(assigned_gpus).issubset(known_gpus)
        )

    timelines = [record.get("phase_timestamps") or {} for record in records]
    complete_timelines = [
        timeline
        for record, timeline in zip(records, timelines, strict=True)
        if has_complete_timeline(record)
    ]
    complete_measurements = [
        timeline
        for record, timeline in zip(records, timelines, strict=True)
        if has_complete_timeline(record) and has_valid_assignment(record)
    ]
    coverage = {
        "measurement_status": "missing",
        "record_count": len(records),
        "complete_timeline_count": len(complete_timelines),
        "complete_measurement_count": len(complete_measurements),
    }
    if not records:
        return {**coverage, "missing_reason": "no terminal workload records"}
    if not complete_measurements:
        return {
            **coverage,
            "missing_reason": (
                "no workload produced an ok result with a complete timeline and valid "
                "GPU assignment"
            ),
        }
    if len(complete_measurements) != len(records):
        return {
            **coverage,
            "missing_reason": (
                "cost model requires an ok result, complete GPU timeline, and valid "
                "nonempty GPU assignment from this run for every record"
            ),
        }
    process_starts = [timeline["process_started"] for timeline in complete_measurements]
    first_ready = min(timeline["ready"] for timeline in complete_measurements)
    assignment_handoffs = [
        max(0.0, timeline["gpu_started"] - timeline["assigned"])
        for timeline in complete_measurements
    ]
    gpu_intervals_by_index: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        timeline = record.get("phase_timestamps") or {}
        if "gpu_started" not in timeline or "gpu_finished" not in timeline:
            continue
        for gpu in record.get("gpus") or []:
            gpu_intervals_by_index.setdefault(str(gpu), []).append(
                (timeline["gpu_started"], timeline["gpu_finished"])
            )
    gpu_busy_by_index = {
        gpu: sum(end - start for start, end in gpu_intervals_by_index.get(gpu, []))
        for gpu in known_gpu_indices
    }
    scheduled_records = records

    def external_occupied_at(timestamp: float) -> set[str]:
        occupied: tuple[str, ...] | None = None
        for changed_at, value in external_history:
            if changed_at > timestamp:
                break
            occupied = value
        # Before the first complete snapshot, no physical GPU is eligible for
        # assignment. CPU preparation may still proceed in parallel.
        return set(known_gpu_indices if occupied is None else occupied)
    actual_available_at: dict[str, float] = {}
    dispatch_latencies: list[float] = []
    for record in sorted(
        scheduled_records, key=lambda item: item["phase_timestamps"]["assigned"]
    ):
        timeline = record["phase_timestamps"]
        available_at = max(
            [timeline["ready"]]
            + [actual_available_at.get(str(gpu), timeline["ready"]) for gpu in record["gpus"]]
        )
        dispatch_latencies.append(max(0.0, timeline["assigned"] - available_at))
        for gpu in record["gpus"]:
            actual_available_at[str(gpu)] = timeline["result_received"]

    gpu_indices = sorted(known_gpu_indices, key=int)

    external_change_offsets = sorted(
        max(0.0, timestamp - first_ready)
        for timestamp, _occupied in external_history
        if timestamp > first_ready
    )

    def earliest_eligible_start(selected: tuple[str, ...], earliest: float) -> float:
        candidate = earliest
        while external_occupied_at(first_ready + candidate).intersection(selected):
            next_changes = [offset for offset in external_change_offsets if offset > candidate]
            if not next_changes:
                return math.inf
            candidate = next_changes[0]
        return candidate

    def list_schedule(*, respect_ready: bool, respect_external: bool = False) -> float:
        available = {gpu: 0.0 for gpu in gpu_indices}
        finish = 0.0
        ordered = sorted(
            scheduled_records,
            key=lambda item: (
                item["phase_timestamps"]["ready"],
                -len(item.get("gpus") or ()),
            ),
        )
        for record in ordered:
            count = len(record["gpus"])
            if count > len(gpu_indices):
                return math.inf
            ready_offset = (
                max(0.0, record["phase_timestamps"]["ready"] - first_ready)
                if respect_ready
                else 0.0
            )
            if respect_external:
                candidates = []
                for selected in combinations(gpu_indices, count):
                    internally_available = max(
                        [ready_offset] + [available[gpu] for gpu in selected]
                    )
                    start = earliest_eligible_start(selected, internally_available)
                    candidates.append((start, selected))
                start, selected = min(
                    candidates,
                    key=lambda candidate: (
                        candidate[0],
                        tuple(int(gpu) for gpu in candidate[1]),
                    ),
                )
                if not math.isfinite(start):
                    return math.inf
            else:
                selected = tuple(
                    sorted(gpu_indices, key=lambda gpu: (available[gpu], int(gpu)))[:count]
                )
                start = max([ready_offset] + [available[gpu] for gpu in selected])
            duration = (
                record["phase_timestamps"]["result_received"]
                - record["phase_timestamps"]["gpu_started"]
            )
            finish = max(finish, start + duration)
            for gpu in selected:
                available[gpu] = start + duration
        return finish

    ideal_gpu_s = list_schedule(respect_ready=False)
    ready_constrained_gpu_s = list_schedule(respect_ready=True)
    eligibility_constrained_gpu_s = list_schedule(
        respect_ready=True,
        respect_external=True,
    )
    foreign_wait_s = max(0.0, eligibility_constrained_gpu_s - ready_constrained_gpu_s)
    first_ready_s = max(0.0, first_ready - run_started)
    expected_s = first_ready_s + ready_constrained_gpu_s
    expected_with_foreign_s = first_ready_s + eligibility_constrained_gpu_s
    observed_s = max(0.0, critical_finished - run_started)
    unexplained_s = observed_s - first_ready_s - ready_constrained_gpu_s - foreign_wait_s

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
        return ordered[index]

    return {
        "measurement_status": "measured",
        "record_count": len(records),
        "observed_critical_s": observed_s,
        "first_process_started_s": max(0.0, min(process_starts) - run_started),
        "prepare_spawn_span_s": max(process_starts) - min(process_starts),
        "first_ready_s": first_ready_s,
        "ideal_gpu_list_schedule_s": ideal_gpu_s,
        "ready_constrained_gpu_list_schedule_s": ready_constrained_gpu_s,
        "eligibility_constrained_gpu_list_schedule_s": eligibility_constrained_gpu_s,
        "gpu_busy_s_by_index": gpu_busy_by_index,
        "foreign_wait_s": foreign_wait_s,
        "expected_s": expected_s,
        "expected_with_foreign_s": expected_with_foreign_s,
        "unexplained_s": unexplained_s,
        "dispatch_latency_s": {
            "count": len(dispatch_latencies),
            "max": max(dispatch_latencies),
            "p95": percentile(dispatch_latencies, 0.95),
        },
        "assignment_handoff_s": {
            "count": len(assignment_handoffs),
            "max": max(assignment_handoffs),
            "p95": percentile(assignment_handoffs, 0.95),
        },
        "ready_starvation_s": max(0.0, ready_constrained_gpu_s - ideal_gpu_s),
        "complete_timeline_count": len(complete_timelines),
        "complete_measurement_count": len(complete_measurements),
    }


def _workload_phase_breakdown(record: dict) -> dict[str, float] | None:
    """Derive workload phase durations from the canonical transition timestamps."""
    timeline = record.get("phase_timestamps") or {}
    required = (
        "process_started",
        "child_started",
        "prepare_started",
        "framework_import_started",
        "framework_loaded",
        "module_loaded",
        "config_resolved",
        "ready",
        "assigned",
        "gpu_started",
        "gpu_finished",
        "result_received",
        "process_reaped",
    )
    if any(name not in timeline for name in required):
        return None
    return {
        "process_startup_s": timeline["child_started"] - timeline["process_started"],
        "cli_bootstrap_s": timeline["framework_import_started"] - timeline["child_started"],
        "framework_import_s": (
            timeline["framework_loaded"] - timeline["framework_import_started"]
        ),
        "exact_import_s": timeline["module_loaded"] - timeline["framework_loaded"],
        "config_resolve_s": timeline["config_resolved"] - timeline["module_loaded"],
        "specialize_generate_compile_s": timeline["ready"] - timeline["config_resolved"],
        "cpu_prepare_s": timeline["ready"] - timeline["prepare_started"],
        "ready_wait_s": timeline["assigned"] - timeline["ready"],
        "assignment_handoff_s": timeline["gpu_started"] - timeline["assigned"],
        "gpu_stage_s": timeline["gpu_finished"] - timeline["gpu_started"],
        "result_handoff_s": timeline["result_received"] - timeline["gpu_finished"],
        "process_reap_tail_s": timeline["process_reaped"] - timeline["result_received"],
    }


def _validate_pipeline_timelines(records: list[dict]) -> None:
    ordered_phases = (
        "process_started",
        "child_started",
        "prepare_started",
        "framework_import_started",
        "framework_loaded",
        "module_loaded",
        "config_resolved",
        "ready",
        "assigned",
        "gpu_started",
        "gpu_finished",
        "result_received",
        "process_reaped",
    )
    ownership_by_gpu: dict[str, list[tuple[float, float, str]]] = {}
    for record in records:
        if record.get("status") not in ("ok", "SKIP"):
            continue
        timeline = record.get("phase_timestamps") or {}
        missing = [phase for phase in ordered_phases if phase not in timeline]
        if missing:
            raise RuntimeError(
                f"{record.get('kernel')}/{record.get('config')} missing timeline phases {missing}"
            )
        values = [timeline[phase] for phase in ordered_phases]
        if any(right < left for left, right in pairwise(values)):
            raise RuntimeError(
                f"{record.get('kernel')}/{record.get('config')} has out-of-order timeline "
                f"{timeline}"
            )
        label = f"{record.get('kernel')}/{record.get('config')}"
        for gpu in record.get("gpus") or []:
            ownership_by_gpu.setdefault(str(gpu), []).append(
                (timeline["assigned"], timeline["result_received"], label)
            )
    for gpu, intervals in ownership_by_gpu.items():
        intervals.sort()
        for previous, current in pairwise(intervals):
            if current[0] < previous[1]:
                raise RuntimeError(
                    f"GPU {gpu} ownership intervals overlap: {previous[2]} and {current[2]}"
                )


def run_scheduled_jobs(
    workloads: list[dict],
    pool: GpuPool,
    log_dir: Path,
    *,
    rounds: int,
    cooldown: float,
    compile_profile: dict,
    max_prepare_processes: int | None = None,
    ready_backlog: int | None = None,
) -> tuple[list[dict], list[dict[str, Any]], dict]:
    """Run a bounded CPU-prepare/READY/GPU pipeline with fail-fast semantics."""
    n_jobs = len(workloads)
    if not n_jobs:
        return [], [], {}
    visible = pool.total_visible()
    if max_prepare_processes is None:
        max_prepare_processes = min(
            n_jobs, max(1, min(os.cpu_count() or 1, max(4, visible * 2)))
        )
    if ready_backlog is None:
        ready_backlog = max(max_prepare_processes, visible * 2)
    if max_prepare_processes < 1 or ready_backlog < 1:
        raise ValueError("max_prepare_processes and ready_backlog must be positive")
    if ready_backlog < max_prepare_processes:
        raise ValueError("ready_backlog must be >= max_prepare_processes")

    pending = deque((workload, 1) for workload in workloads)
    active: dict[int, _PreparedAttempt] = {}
    ready: deque[_PreparedAttempt] = deque()
    records: list[dict] = []
    retry_log: list[dict[str, Any]] = []
    completed = 0
    failed = False
    last_interference_poll = 0.0
    max_observed_preparing = 0
    max_observed_ready = 0
    max_observed_buffered = 0
    max_observed_active_children = 0
    max_observed_process_tree = 0
    max_observed_rss_bytes = 0
    max_observed_open_fds = 0
    last_resource_sample = 0.0
    run_started = time.time()
    critical_finished: float | None = None

    def preparing_count() -> int:
        return sum(item.state == "PREPARING_CPU" for item in active.values())

    def buffered_count() -> int:
        return sum(item.state in ("PREPARING_CPU", "READY") for item in active.values())

    def sample_host_resources() -> None:
        nonlocal max_observed_process_tree, max_observed_rss_bytes, max_observed_open_fds
        roots = {os.getpid()} | {item.process.pid for item in active.values()}
        sample = _process_tree_resources(roots)
        max_observed_process_tree = max(max_observed_process_tree, sample["processes"])
        max_observed_rss_bytes = max(max_observed_rss_bytes, sample["rss_bytes"])
        max_observed_open_fds = max(max_observed_open_fds, sample["open_fds"])

    def remove_ready(item: _PreparedAttempt) -> None:
        try:
            ready.remove(item)
        except ValueError:
            pass

    def release_gpus(item: _PreparedAttempt) -> None:
        if item.gpus and not item.gpu_ownership_released:
            pool.release_many(item.gpus)
            item.gpu_ownership_released = True

    def terminate_and_cleanup(item: _PreparedAttempt) -> None:
        fd = item.control.fileno()
        remove_ready(item)
        release_gpus(item)
        if item.process.poll() is None:
            _terminate_subprocess(item.process)
        item.timeline.setdefault("process_reaped", time.time())
        _finish_attempt_process(item)
        active.pop(fd, None)

    def fail(item: _PreparedAttempt, message: dict | None = None) -> None:
        nonlocal completed, failed
        item.state = "FAILED"
        release_gpus(item)
        item.record = _record_child_failure(item, message)
        records.append(item.record)
        completed += 1
        failed = True
        detail = item.record.get("error") or "unknown workload failure"
        log(
            f"[bench-suite] >>> FAIL-FAST {item.label} attempt {item.attempt}: "
            f"{detail[:160]} <<<"
        )

    def requeue_interfered(item: _PreparedAttempt, intruders: list[int], detail: str) -> None:
        retry_log.append(
            {
                "status": "INTERFERED",
                "kernel": item.workload["kernel"],
                "config": item.workload["config"],
                "attempt": item.attempt,
                "process_pid": item.process.pid,
                "intruder_pids": intruders,
                "detail": detail[:240],
                "retry_attempt": item.attempt + 1,
            }
        )
        log("[bench-suite] " + "*" * 70)
        log(f"[bench-suite] *** INTERFERED *** {item.label} attempt {item.attempt}")
        log(f"[bench-suite] ***   intruder PIDs: {intruders}")
        log("[bench-suite] ***   child killed; retry starts from CPU prepare")
        log("[bench-suite] " + "*" * 70)
        pending.append((item.workload, item.attempt + 1))
        terminate_and_cleanup(item)

    def spawn_available() -> None:
        while (
            pending
            and not failed
            and preparing_count() < max_prepare_processes
            # Every spawned child reserves one future READY slot. This makes
            # ready_backlog a hard bound even when all prepares finish together.
            and buffered_count() < ready_backlog
            # RUNNING children can add at most one per visible GPU. Terminal
            # children waiting for host teardown also count, so their tail can
            # never turn the one-shot process model into unbounded fan-out.
            and len(active) < ready_backlog + visible
        ):
            workload, attempt_number = pending.popleft()
            item = _spawn_prepared_attempt(
                workload,
                attempt_number,
                log_dir,
                rounds=rounds,
                cooldown=cooldown,
                compile_profile=compile_profile,
            )
            active[item.control.fileno()] = item
            log(
                f"[bench-suite] {now_iso()} prepare pid={item.process.pid} "
                f"START {item.label} (attempt {attempt_number})"
            )

    def dispatch_ready() -> None:
        if failed or not ready:
            return
        # Larger atomic claims first, preserving READY order within equal sizes.
        ordered = sorted(
            list(ready),
            key=lambda item: (-item.workload.get("num_gpus", 1), item.timeline["ready"]),
        )
        for item in ordered:
            count = item.workload.get("num_gpus", 1)
            gpus = pool.try_acquire_many(count)
            if gpus is None:
                continue
            strangers = _active_strangers(gpus, _our_pids(), pool.util_threshold)
            if strangers is None or strangers:
                pool.release_many(gpus)
                detail = (
                    "could not sample assigned GPU utilization"
                    if strangers is None
                    else f"intruders {sorted(strangers)}"
                )
                requeue_interfered(item, sorted(strangers or {}), detail)
                continue
            remove_ready(item)
            item.gpus = gpus
            item.gpu_ownership_released = False
            item.state = "RUNNING_GPU"
            item.timeline["assigned"] = time.time()
            _send_child(item, {"type": "ASSIGN", "gpu_indices": list(gpus)})
            log(
                f"[bench-suite] {now_iso()} gpus={','.join(gpus)} GPU_START "
                f"{item.label} (attempt {item.attempt})"
            )

    external_stop = threading.Event()
    dispatch_stop = threading.Event()
    dispatch_reader, dispatch_writer = socket.socketpair()
    dispatch_reader.setblocking(False)
    dispatch_writer.setblocking(False)

    def notify_dispatch_on_pool_change() -> None:
        generation = pool.change_generation()
        while not dispatch_stop.is_set():
            generation = pool.wait_for_change(generation, dispatch_stop)
            if dispatch_stop.is_set():
                return
            try:
                dispatch_writer.send(b"\0")
            except BlockingIOError:
                pass
            except OSError:
                return

    def monitor_external_occupancy() -> None:
        while not external_stop.is_set():
            pool.refresh_external_occupancy()
            if external_stop.wait(POLL_INTERVAL):
                return

    external_thread = threading.Thread(
        target=monitor_external_occupancy,
        name="bench-gpu-occupancy",
        daemon=True,
    )
    dispatch_thread = threading.Thread(
        target=notify_dispatch_on_pool_change,
        name="bench-gpu-dispatch-notifier",
        daemon=True,
    )
    dispatch_thread.start()
    external_thread.start()
    try:
        while completed < n_jobs and not failed:
            spawn_available()
            dispatch_ready()
            max_observed_preparing = max(max_observed_preparing, preparing_count())
            max_observed_ready = max(max_observed_ready, len(ready))
            max_observed_buffered = max(max_observed_buffered, buffered_count())
            max_observed_active_children = max(max_observed_active_children, len(active))

            now = time.monotonic()
            if now - last_resource_sample >= _RESOURCE_SAMPLE_INTERVAL:
                last_resource_sample = now
                sample_host_resources()
            if now - last_interference_poll >= MONITOR_INTERVAL:
                last_interference_poll = now
                for item in list(active.values()):
                    if item.state != "RUNNING_GPU" or not item.gpus:
                        continue
                    strangers = _active_strangers(item.gpus, _our_pids(), pool.util_threshold)
                    if strangers is None or strangers:
                        detail = (
                            "could not sample assigned GPU utilization"
                            if strangers is None
                            else f"intruders {sorted(strangers)}"
                        )
                        requeue_interfered(item, sorted(strangers or {}), detail)

            sockets = [
                dispatch_reader,
                *(item.control for item in active.values() if item.state != "REAPED"),
            ]
            if sockets:
                readable, _, _ = select.select(sockets, [], [], 0.1)
            else:
                readable = []
                time.sleep(0.05)

            if dispatch_reader in readable:
                while True:
                    try:
                        if not dispatch_reader.recv(4096):
                            break
                    except BlockingIOError:
                        break
                readable.remove(dispatch_reader)

            for control in readable:
                item = active.get(control.fileno())
                if item is None:
                    continue
                try:
                    messages, eof = _receive_child_messages(item)
                except (json.JSONDecodeError, OSError) as error:
                    fail(
                        item,
                        {"phase": "protocol", "error": f"invalid control message: {error}"},
                    )
                    break
                for message in messages:
                    message_type = message.get("type")
                    if message_type == "READY" and item.state == "PREPARING_CPU":
                        required_num_gpus = message.get("required_num_gpus")
                        declared_num_gpus = item.workload.get("num_gpus", 1)
                        if required_num_gpus != declared_num_gpus:
                            fail(
                                item,
                                {
                                    "phase": "prepare",
                                    "error": (
                                        f"READY requires {required_num_gpus!r} GPU(s), "
                                        f"workload declares {declared_num_gpus!r}"
                                    ),
                                },
                            )
                            break
                        item.timeline["child_started"] = message["child_started"]
                        item.timeline["prepare_started"] = message["prepare_started"]
                        item.timeline["framework_import_started"] = message[
                            "framework_import_started"
                        ]
                        item.timeline["framework_loaded"] = message["framework_loaded"]
                        item.timeline["module_loaded"] = message["module_loaded"]
                        item.timeline["config_resolved"] = message["config_resolved"]
                        item.timeline["ready"] = message["ready"]
                        item.state = "READY"
                        ready.append(item)
                        log(
                            f"[bench-suite] {now_iso()} READY {item.label} "
                            f"prepare={item.timeline['ready'] - item.timeline['prepare_started']:.3f}s"
                        )
                    elif message_type == "RESULT" and item.state == "RUNNING_GPU":
                        item.timeline["gpu_started"] = message["gpu_started"]
                        item.timeline["gpu_finished"] = message["gpu_finished"]
                        item.timeline["result_received"] = time.time()
                        release_gpus(item)
                        record = _base_attempt_record(item)
                        result = message.get("result") or {}
                        status = result.get("status")
                        if status == "SKIP":
                            record.update(result)
                        elif status == "FAIL":
                            record.update(result)
                            record.setdefault("status", "FAIL")
                        else:
                            _finalize_bench_record(result, rounds=rounds, cooldown=cooldown)
                            record.update(result)
                        record.setdefault("label", item.workload["config"])
                        record["finished_at"] = now_iso()
                        item.record = record
                        item.state = "RESULT"
                        item.terminal_at = time.monotonic()
                        records.append(record)
                        completed += 1
                        critical_finished = item.timeline["result_received"]
                        impls = record.get("impls") or {}
                        impl_str = ", ".join(f"{k}={v:.3f}µs" for k, v in impls.items())
                        log(
                            f"[bench-suite] {record['finished_at']} "
                            f"gpus={record.get('gpu') or '-'} {record.get('status', 'ok'):4s} "
                            f"{item.label} {impl_str}"
                        )
                        if record.get("status") not in ("ok", "SKIP"):
                            failed = True
                            log(
                                f"[bench-suite] >>> FAIL-FAST {item.label} "
                                f"attempt {item.attempt}: {record.get('error', 'bench failed')[:160]} <<<"
                            )
                    elif message_type == "FAIL":
                        fail(item, message)
                        break
                    else:
                        fail(
                            item,
                            {
                                "phase": "protocol",
                                "error": (
                                    f"unexpected {message_type!r} message in state {item.state}"
                                ),
                            },
                        )
                        break
                if failed:
                    break
                if eof and item.state not in ("RESULT", "FAILED"):
                    item.process.poll()
                    fail(item)
                    break

            # Reap terminal children asynchronously. Their GPU ownership was
            # already released when RESULT arrived.
            for item in list(active.values()):
                if item.state != "RESULT":
                    continue
                if item.process.poll() is not None:
                    fd = item.control.fileno()
                    item.timeline["process_reaped"] = time.time()
                    _finish_attempt_process(item)
                    active.pop(fd, None)
    finally:
        external_stop.set()
        dispatch_stop.set()
        pool.wake()
        external_thread.join(timeout=1.0)
        dispatch_thread.join(timeout=1.0)
        dispatch_reader.close()
        dispatch_writer.close()
        # Failure and KeyboardInterrupt cancel every nonterminal state. A READY
        # child receives CANCEL for protocol completeness, then the process group
        # is reaped so no compiler/CUDA descendants survive the suite.
        for item in list(active.values()):
            if item.process.poll() is None and item.state == "RESULT" and not failed:
                try:
                    item.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _terminate_subprocess(item.process)
            elif item.process.poll() is None and item.state in ("PREPARING_CPU", "READY"):
                try:
                    _send_child(item, {"type": "CANCEL"})
                except OSError:
                    pass
            if item.process.poll() is None:
                _terminate_subprocess(item.process)
            fd = item.control.fileno()
            release_gpus(item)
            item.timeline.setdefault("process_reaped", time.time())
            _finish_attempt_process(item)
            active.pop(fd, None)
    run_finished = time.time()
    sample_host_resources()
    if critical_finished is None:
        critical_finished = run_finished
    _validate_pipeline_timelines(records)
    known_gpu_indices, external_history = pool.external_timeline()
    pipeline_meta = {
        "execution_mode": "pipeline",
        "process_model": "one_shot_child_per_workload_attempt",
        "measurement_protocol": {
            "rounds": rounds,
            "cooldown_s": cooldown,
            "default_rounds": DEFAULT_ROUNDS,
            "default_cooldown_s": DEFAULT_COOLDOWN_S,
            "is_default": (
                rounds == DEFAULT_ROUNDS
                and math.isclose(
                    cooldown,
                    DEFAULT_COOLDOWN_S,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ),
        },
        "compile_profile": compile_profile,
        "max_prepare_processes": max_prepare_processes,
        "ready_backlog": ready_backlog,
        "max_observed_preparing": max_observed_preparing,
        "max_observed_ready": max_observed_ready,
        "max_observed_buffered": max_observed_buffered,
        "max_observed_active_children": max_observed_active_children,
        "max_observed_process_tree": max_observed_process_tree,
        "max_observed_rss_bytes": max_observed_rss_bytes,
        "max_observed_open_fds": max_observed_open_fds,
        "started": run_started,
        "critical_finished": critical_finished,
        "processes_reaped": run_finished,
        "critical_wall_s": critical_finished - run_started,
        "final_reap_tail_s": run_finished - critical_finished,
        "external_occupancy_timeline": [
            {"timestamp": timestamp, "occupied_gpu_indices": list(occupied)}
            for timestamp, occupied in external_history
        ],
        "interference_retry_count": len(retry_log),
        "interference_retries": retry_log,
        "multi_gpu_runtime_validation": {
            "validation_status": "exempted_by_human_unmeasured",
            "runtime_evidence": "not_collected_by_human_direction",
            "structural_evidence": (
                "assignment_count_rejection_atomic_claim_and_rank_lifecycle_ordering"
            ),
        },
        "cost_model": _pipeline_cost_model(
            records,
            run_started=run_started,
            critical_finished=critical_finished,
            known_gpu_indices=known_gpu_indices,
            external_history=external_history,
        ),
    }
    return records, retry_log, pipeline_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="bench-suite: pre-commit regression benchmark")
    ap.add_argument(
        "--workloads",
        type=Path,
        default=None,
        help="YAML file listing kernels/configs to bench (default: assemble every "
        "`default: true` config from bench_suite/config/**/*.yaml)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to store runs/, logs/, reports/, latest.json "
        "(default: <tirx-kernels>/.bench-suite)",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON to diff against instead of the pinned baseline.json",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help=f"Regression threshold in percent slowdown (default {DEFAULT_REGRESSION_THRESHOLD:g})",
    )
    ap.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only keep workloads whose kernel contains this substring",
    )
    # NOTE: there is intentionally no --gpus flag. GPU selection is automatic
    # (util-gated probe + per-acquire utilization scan); a human pinning cards
    # defeats that and can land work on a busy card. See acquire()/_occupied_indices.
    ap.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-form label for this run (default: git short sha)",
    )
    ap.add_argument("--no-report", action="store_true", help="Skip regression report generation")
    ap.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the per-GPU probe (use nvidia-smi free-status only)",
    )
    ap.add_argument(
        "--probe-timeout",
        type=float,
        default=60.0,
        help="Per-GPU probe timeout in seconds (default 60)",
    )
    ap.add_argument(
        "--util-threshold",
        type=float,
        default=DEFAULT_UTIL_THRESHOLD,
        help="%% GPU utilization above which assignment skips a card; during a run, "
        "a foreign process above this per-PID SM utilization requeues the workload "
        f"(default {DEFAULT_UTIL_THRESHOLD:g})",
    )
    ap.add_argument(
        "--mem-threshold",
        type=float,
        default=DEFAULT_MEM_THRESHOLD,
        help="%% GPU memory used by compute apps above which a card counts as occupied "
        f"(default {DEFAULT_MEM_THRESHOLD:g})",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Independent standard-timer samples per workload (default {DEFAULT_ROUNDS}). "
        "Compile/prepare once; each round cools down and runs a complete timer call.",
    )
    ap.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help=f"Seconds to sleep before every implementation (default {DEFAULT_COOLDOWN_S:g}).",
    )
    ap.add_argument(
        "--max-prepare-processes",
        type=int,
        default=None,
        help=(
            "Maximum concurrently CPU-preparing one-shot child processes "
            "(default: host/GPU-derived bound)"
        ),
    )
    ap.add_argument(
        "--ready-backlog",
        type=int,
        default=None,
        help=(
            "Maximum PREPARING+READY one-shot children waiting for GPU assignment "
            "(default: at least max-prepare-processes and 2x visible GPUs)"
        ),
    )
    ap.add_argument(
        "--check-imports",
        action="store_true",
        help=(
            "Audit exact imports, explicit pipeline adapters, all module/YAML config labels, "
            "and workload GPU counts, then exit without benchmarking"
        ),
    )
    args = ap.parse_args()
    if args.rounds < 1:
        print("[bench-suite] --rounds must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.cooldown < 0:
        print("[bench-suite] --cooldown must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.util_threshold < 0 or args.mem_threshold < 0:
        print("[bench-suite] --util-threshold/--mem-threshold must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.max_prepare_processes is not None and args.max_prepare_processes < 1:
        print("[bench-suite] --max-prepare-processes must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.ready_backlog is not None and args.ready_backlog < 1:
        print("[bench-suite] --ready-backlog must be >= 1", file=sys.stderr)
        sys.exit(2)
    if (
        args.max_prepare_processes is not None
        and args.ready_backlog is not None
        and args.ready_backlog < args.max_prepare_processes
    ):
        print(
            "[bench-suite] --ready-backlog must be >= --max-prepare-processes",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.workloads is None:
        assembled = load_config_dir()
        workloads_path = write_generated_workloads(
            assembled, args.out_dir.resolve() / GENERATED_WORKLOADS_NAME
        )
        print(
            f"[bench-suite] assembled {len(assembled)} default workload(s) "
            f"from {CONFIG_DIR}/*.yaml -> {workloads_path}"
        )
    else:
        workloads_path = args.workloads

    workloads = load_workloads(workloads_path)
    if args.filter:
        workloads = [w for w in workloads if args.filter in w["kernel"]]
    if not workloads:
        print("[bench-suite] no workloads to run.", file=sys.stderr)
        sys.exit(2)

    if args.check_imports:
        capability = audit_pipeline_capabilities()
        names = check_workload_capabilities(workloads)
        print(
            "[bench-suite] pipeline capability audit ok: "
            f"{capability['kernel_count']} kernels, "
            f"{capability['module_config_count']} module configs, "
            f"{capability['yaml_config_count']} YAML configs; "
            f"selected workload file references {len(names)} kernel(s)"
        )
        exemption = capability["multi_gpu_runtime_validation"]
        print(
            "[bench-suite] multi-GPU runtime validation: "
            f"{exemption['validation_status']} ({exemption['workload_count']} module configs)"
        )
        report_path = write_pipeline_capability_report(args.out_dir, capability)
        print(f"[bench-suite] wrote {report_path}")
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Run id: incrementing integer — one more than the highest existing numeric
    # run in runs/ (runs/7.json, reports/7/, latest -> 7).
    _existing = [int(p.stem) for p in runs_dir.glob("*.json") if p.stem.isdigit()]
    stamp = str(max(_existing, default=0) + 1)
    run_log_path = runs_dir / f"{stamp}.log"
    run_log_fh = open(run_log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, run_log_fh)
    sys.stderr = _Tee(sys.stderr, run_log_fh)
    # Repoint `latest.log` symlink immediately so `tail -f .bench-suite/latest.log`
    # picks up this run before any output happens.
    latest_log = out_dir / "latest.log"
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
    latest_log.symlink_to(run_log_path.relative_to(out_dir))

    print(f"[bench-suite] live log: {run_log_path}")
    print(f"[bench-suite]   tail : tail -f {latest_log}")
    print(f"[bench-suite] run id : {stamp}")

    # ── Automatic GPU selection (no manual override on purpose) ──
    # 1. Startup probe: run a tiny fp16 matmul on every visible card
    #    (including busy ones — the probe is light, finishes fine on a
    #    contended card; this catches broken drivers / ECC). Probe failures
    #    are banned for the rest of the run.
    # 2. Per-workload acquire: re-scan utilization/memory every time we need a card.
    listing_pool = GpuPool(util_threshold=args.util_threshold, mem_threshold=args.mem_threshold)
    in_filter = [idx for idx, _ in _visible_gpu_rows(listing_pool._all_gpus())]
    if not in_filter:
        print("[bench-suite] no visible GPUs.", file=sys.stderr)
        sys.exit(1)
    utils_now = listing_pool._utils()
    mem_now = listing_pool._mem_used_pct()
    occupied_now = sorted(listing_pool._occupied_indices() & set(in_filter), key=int)
    resident = sorted(listing_pool._busy_indices() & set(in_filter), key=int)
    util_str = " ".join(f"{i}:{utils_now.get(i, 0):.0f}%" for i in sorted(in_filter, key=int))
    mem_str = " ".join(f"{i}:{mem_now.get(i, 0):.1f}%" for i in sorted(in_filter, key=int))
    print(
        f"[bench-suite] visible: {len(in_filter)} {sorted(in_filter, key=int)}; "
        f"util now [{util_str}]; mem now [{mem_str}]",
        flush=True,
    )
    print(
        f"[bench-suite] gate: util-threshold={args.util_threshold:g}%, "
        f"mem-threshold={args.mem_threshold:g}% — "
        f"occupied (skip): {occupied_now if occupied_now else 'none'}; "
        f"shareable: "
        f"{sorted((set(in_filter) - set(occupied_now)), key=int)} "
        f"(resident-VRAM cards: {resident if resident else 'none'})",
        flush=True,
    )

    if args.no_probe:
        usable = set(in_filter)
        probe_failures: dict[str, str] = {}
    else:
        print(
            f"[bench-suite] probing {len(in_filter)} GPU(s) with fp16 512x512 matmul ...",
            flush=True,
        )
        usable, probe_failures = detect_usable_gpus(in_filter, args.probe_timeout)

    if not usable:
        print("[bench-suite] no usable GPUs (all probes failed).", file=sys.stderr)
        for idx, err in probe_failures.items():
            print(f"[bench-suite]   gpu {idx}: {err}", file=sys.stderr)
        sys.exit(1)

    max_required_gpus = max(workload.get("num_gpus", 1) for workload in workloads)
    if max_required_gpus > len(usable):
        print(
            f"[bench-suite] workload requires {max_required_gpus} GPU(s), but only "
            f"{len(usable)} passed the startup probe.",
            file=sys.stderr,
        )
        sys.exit(2)

    pool = GpuPool(
        allowed=usable, util_threshold=args.util_threshold, mem_threshold=args.mem_threshold
    )
    n_gpus = len(usable)
    compile_profile = gpu_compile_profile(usable)
    _repo_git = collect_repo_git()
    label = args.label or _repo_git.get("tirx-kernels") or _repo_git.get("tir") or "local"
    agg_note = (
        f", {args.rounds} standard-timer round(s), aggregate=mean, "
        f"cooldown={args.cooldown:g}s before every impl/round"
        if args.rounds > 1 or args.cooldown > 0
        else ""
    )
    print(
        f"[bench-suite] {len(workloads)} workloads, {n_gpus} probe-OK GPU(s) in pool, "
        f"one-shot prepared children, compile-profile={compile_profile}, "
        f"label={label}{agg_note}",
        flush=True,
    )

    results, retry_log, pipeline_meta = run_scheduled_jobs(
        workloads,
        pool,
        log_dir,
        rounds=args.rounds,
        cooldown=args.cooldown,
        compile_profile=compile_profile,
        max_prepare_processes=args.max_prepare_processes,
        ready_backlog=args.ready_backlog,
    )

    if retry_log:
        log(f"[bench-suite] interference retry summary: {len(retry_log)} attempt(s)")
        for retry in retry_log:
            log(
                f"[bench-suite]   - {retry['kernel']}/{retry['config']}: "
                f"attempt {retry['attempt']} → {retry['detail']}"
            )
    else:
        log("[bench-suite] interference retry summary: none")

    results.sort(key=lambda r: (r["kernel"], r.get("label") or r.get("config")))
    probe_meta = {"enabled": not args.no_probe, "usable": sorted(usable), "failed": probe_failures}
    run_path = write_run(
        out_dir,
        stamp,
        results,
        label,
        probe=probe_meta,
        pipeline=pipeline_meta,
    )
    current = json.loads(run_path.read_text())

    latest = out_dir / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_path.relative_to(out_dir))

    summary_path = write_summary(out_dir, current)
    print(f"[bench-suite] wrote {run_path}")
    print(f"[bench-suite] wrote {summary_path}")

    failures = [record for record in results if record.get("status") == "FAIL"]
    if failures:
        first = failures[0]
        print(
            f"[bench-suite] stopped after workload failure: "
            f"{first['kernel']}/{first.get('config') or first.get('label')}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.no_report:
        return

    # Single pinned baseline (baseline.json). Promote a fresh run over it via
    # promote_baseline.py.
    baseline = load_baseline(args.baseline)
    if baseline is None:
        print("[bench-suite] no baseline (baseline.json) — skipping regression report")
        print(f"[bench-suite]   set baseline: promote_baseline.py {run_path} --merge")
        return

    reports_dir = out_dir / "reports" / current["timestamp"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    # keep reports/latest pointing at the most recent run's folder
    reports_latest = out_dir / "reports" / "latest"
    if reports_latest.exists() or reports_latest.is_symlink():
        reports_latest.unlink()
    reports_latest.symlink_to(current["timestamp"])

    sys.path.insert(0, str(SCRIPT_DIR))
    from ratio_diff import build_report as _build_bench_report

    try:
        bench_md, n_regress = _build_bench_report(baseline, current, threshold_pct=args.threshold)
    except Exception as e:
        print(f"[bench-suite] bench report failed: {e}", file=sys.stderr)
        sys.exit(3)

    bench_path = reports_dir / "bench.md"
    bench_path.write_text(bench_md)
    print(f"[bench-suite] wrote {bench_path}\n")
    print(bench_md)

    if n_regress > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
