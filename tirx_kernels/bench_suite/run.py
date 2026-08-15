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
import ctypes
import hashlib
import json
import math
import os
import queue
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
from pathlib import Path
from typing import Any, ClassVar

import yaml

from tirx_kernels.bench_suite._local_only_runtime import LOCAL_ONLY_ENV, LOCAL_ONLY_KERNEL_ENV
from tirx_kernels.bench_suite._nvrtc132_sitecustomize import (
    NVRTC_BUILTINS_LIBRARY_ENV,
    NVRTC_LIBRARY_ENV,
)
from tirx_kernels.runner import DEFAULT_BENCH_COOLDOWN_S as DEFAULT_COOLDOWN_S
from tirx_kernels.runner import DEFAULT_BENCH_ROUNDS as DEFAULT_ROUNDS
from tirx_kernels.runner import (
    PREPARE_CUDA_ARCH_ENV,
    PREPARE_NUM_SMS_ENV,
    TVM_FFI_DISABLE_TORCH_C_DLPACK_ENV,
)

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
MAX_DEFAULT_CONFIGS_PER_KERNEL = 3
DEFAULT_SELECTION_ROLES = ("small", "medium", "large")
# Single pinned baseline: every run benches our kernel + all reference impls,
# so one JSON holds both. Promote a run over it via promote_baseline.py.
DEFAULT_BASELINE = SCRIPT_DIR / "baseline.json"
DEFAULT_REGRESSION_THRESHOLD = 1.0
DEFAULT_PAIRED_SPEEDUP_THRESHOLD = -1.0
IR_BUILDER_MIGRATION_ROUNDS = 5
IR_BUILDER_MIGRATION_MEM_THRESHOLD = 0.5
IR_BUILDER_MIGRATION_USER_EXEMPTION_TIMESTAMP = "2026-08-12T22:14:00Z"
IR_BUILDER_MIGRATION_REQUIRED_NVRTC_VERSION = (13, 2)
PAIRED_SIDE_TIMEOUT_S = 10 * 60.0
GPU_MEMORY_RELEASE_TIMEOUT_S = 5 * 60.0
POLL_INTERVAL = 5.0  # seconds between GPU re-checks when none is free
MONITOR_INTERVAL = 0.5  # seconds between nvidia-smi polls during a workload
DEFAULT_UTIL_THRESHOLD = 0.0  # % GPU util above which a card counts as busy.
DEFAULT_MEM_THRESHOLD = 0.0  # % physical memory above the idle floor that counts as busy.
IDLE_GPU_MEMORY_FLOOR_MIB = 512.0


class _BenchSuiteCancelled(RuntimeError):
    """Internal signal used to unwind paired workers after the first failure."""


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
    exclusive_resource = workload.get("exclusive_resource")
    if exclusive_resource is not None and (
        not isinstance(exclusive_resource, str) or not exclusive_resource
    ):
        raise ValueError(f"workload exclusive_resource must be a non-empty string: {workload}")
    return workload


def _read_kernel_config(path: Path) -> tuple[str, list[dict], str | None]:
    data = yaml.safe_load(path.read_text()) or {}
    kernel = data.get("kernel")
    if not kernel:
        raise ValueError(f"{path.name}: missing top-level 'kernel'")
    defaults = data.get("defaults") or {}
    entries: list[dict] = []
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
        if type(entry.get("default")) is not bool:
            raise ValueError(f"{path.name}: config {label!r} must set default: true|false")
        if entry.get("default", False):
            default_count += 1
        selection_role = entry.get("selection_role")
        if selection_role is not None and selection_role not in DEFAULT_SELECTION_ROLES:
            raise ValueError(
                f"{path.name}: selection_role must be one of "
                f"{DEFAULT_SELECTION_ROLES}, got {selection_role!r}"
            )
        entries.append({"kernel": kernel, **defaults, **entry})
    if default_count > MAX_DEFAULT_CONFIGS_PER_KERNEL:
        raise ValueError(
            f"{path.name}: kernel {kernel!r} has {default_count} default configs; "
            f"maximum is {MAX_DEFAULT_CONFIGS_PER_KERNEL}"
        )
    selection_rationale = data.get("selection_rationale")
    if selection_rationale is not None and (
        not isinstance(selection_rationale, str) or not selection_rationale.strip()
    ):
        raise ValueError(f"{path.name}: selection_rationale must be a non-empty string")
    if len(entries) > MAX_DEFAULT_CONFIGS_PER_KERNEL and default_count == 3:
        if selection_rationale is None:
            raise ValueError(
                f"{path.name}: curated three-point default selection requires selection_rationale"
            )
        default_roles = {
            entry.get("selection_role") for entry in entries if entry.get("default", False)
        }
        if default_roles != set(DEFAULT_SELECTION_ROLES):
            raise ValueError(
                f"{path.name}: curated defaults must have exactly the roles "
                f"{DEFAULT_SELECTION_ROLES}, got {sorted(default_roles, key=str)}"
            )
        nondefault_roles = [
            entry["config"]
            for entry in entries
            if not entry.get("default", False) and entry.get("selection_role") is not None
        ]
        if nondefault_roles:
            raise ValueError(
                f"{path.name}: non-default configs cannot have selection_role: {nondefault_roles}"
            )
    elif selection_rationale is not None:
        raise ValueError(
            f"{path.name}: selection_rationale is only valid for a curated "
            "three-of-many default selection"
        )
    elif any(entry.get("selection_role") is not None for entry in entries):
        raise ValueError(
            f"{path.name}: selection_role is only valid for a curated "
            "three-of-many default selection"
        )
    return kernel, entries, selection_rationale


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
    _, entries, _selection_rationale = _read_kernel_config(path)
    return [
        _normalize_workload({key: value for key, value in entry.items() if key != "selection_role"})
        for entry in entries
    ]


def load_all_config_dir(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Inventory every benchable entry across ``config/**/*.yaml``."""
    files = sorted(config_dir.rglob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no kernel config files under {config_dir}")
    out: list[dict] = []
    seen: dict[tuple[str, str], Path] = {}
    for path in files:
        _, entries, _selection_rationale = _read_kernel_config(path)
        for entry in entries:
            key = (entry["kernel"], entry["config"])
            if key in seen:
                raise ValueError(f"duplicate bench config {key!r} in {seen[key]} and {path}")
            seen[key] = path
            entry.pop("selection_role", None)
            out.append(_normalize_workload(dict(entry)))
    return out


def load_config_dir(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """The pinned daily sweep: ``default: true`` rows from the canonical set.

    Each file is one kernel's complete benchable matrix, so which configs the
    regression gate covers is a per-line flag rather than a separate file.  The
    files are bucketed to mirror the kernel tree, so the walk is recursive.
    """
    out: list[dict] = []
    for entry in load_all_config_dir(config_dir):
        if not entry.pop("default"):
            continue
        out.append(entry)
    return out


def _ir_builder_migration_exemption_reasons(workload: dict) -> list[str]:
    reasons = []
    if workload.get("num_gpus", 1) > 1:
        reasons.append("num_gpus > 1")
    if workload.get("exclusive_resource") == "nvshmem":
        reasons.append("requires NVSHMEM/distributed communication libraries")
    return reasons


def _derive_ir_builder_migration_scope(
    canonical_workloads: list[dict],
) -> tuple[list[dict], dict]:
    """Derive the formal single-GPU scope from the canonical default sweep."""
    required = []
    exempted = []
    for workload in canonical_workloads:
        reasons = _ir_builder_migration_exemption_reasons(workload)
        if not reasons:
            required.append(workload)
            continue
        exempted.append(
            {
                "kernel": workload["kernel"],
                "config": workload["config"],
                "status": "user-exempted",
                "reasons": reasons,
            }
        )
    scope = {
        "kind": "ir-builder-migration",
        "canonical_default_count": len(canonical_workloads),
        "required_count": len(required),
        "user_exempted_count": len(exempted),
        "user_exemption_timestamp": IR_BUILDER_MIGRATION_USER_EXEMPTION_TIMESTAMP,
        "derivation": "default:true and not (num_gpus > 1 or exclusive_resource == nvshmem)",
        "user_exempted": exempted,
    }
    return required, scope


def write_generated_workloads(
    workloads: list[dict], path: Path, *, selection: str = "every config flagged default: true"
) -> Path:
    """Materialize the assembled sweep so a run's exact input is inspectable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by bench_suite from tirx_kernels/bench_suite/config/**/*.yaml\n"
        f"# ({selection}). Do not edit -- rewritten each run.\n"
    )
    path.write_text(
        header + yaml.safe_dump({"defaults": {}, "workloads": workloads}, sort_keys=False)
    )
    return path


def load_workloads(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    defaults = data.get("defaults") or {}
    return [_normalize_workload({**defaults, **entry}) for entry in data.get("workloads") or []]


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
        idle_memory_floor_mib: float = IDLE_GPU_MEMORY_FLOOR_MIB,
    ):
        self._owned: set[str] = set()
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._allowed = allowed
        self._known_indices: tuple[str, ...] = tuple(
            sorted(allowed, key=int) if allowed is not None else ()
        )
        self._external_occupied: set[str] | None = None
        self._change_generation = 0
        self.util_threshold = util_threshold
        self.mem_threshold = mem_threshold
        self.idle_memory_floor_mib = idle_memory_floor_mib

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
        """GPU indices with physical VRAM above the idle-driver allowance."""
        return {idx for idx, used_pct in self._mem_used_pct().items() if used_pct > 0.0}

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
        """Map GPU index -> physical VRAM above the configured idle floor / total."""
        rows = self._nvidia_smi(["--query-gpu=index,memory.used,memory.total"])
        out: dict[str, float] = {}
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    used = max(0.0, float(parts[1]) - self.idle_memory_floor_mib)
                    total = float(parts[2])
                    out[parts[0]] = 100.0 * used / total if total > 0 else 0.0
                except ValueError:
                    pass
        return out

    def _occupied_indices(self) -> set[str]:
        """GPU indices over the configured SM or memory threshold."""
        indices = {idx for idx, _uuid in self._all_gpus()}
        util_busy = {idx for idx, u in self._utils().items() if u > self.util_threshold}
        memory = self._mem_used_pct()
        # Missing framebuffer telemetry is not evidence that a card is free.
        mem_busy = {
            idx for idx in indices if idx not in memory or memory[idx] > self.mem_threshold
        }
        return util_busy | mem_busy

    def total_visible(self) -> int:
        gpus = self._all_gpus()
        if self._allowed is not None:
            gpus = [g for g in gpus if g[0] in self._allowed]
        return len(gpus)

    def wait_for_memory_release(
        self,
        indices: tuple[str, ...] | list[str],
        *,
        timeout_s: float = GPU_MEMORY_RELEASE_TIMEOUT_S,
        poll_interval_s: float = MONITOR_INTERVAL,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, float]:
        """Wait for claimed GPUs to return below the configured memory gate."""
        selected = tuple(indices)
        deadline = time.monotonic() + timeout_s
        while True:
            memory = self._mem_used_pct()
            busy = {
                idx: memory.get(idx, math.inf)
                for idx in selected
                if memory.get(idx, math.inf) > self.mem_threshold
            }
            if not busy:
                return {}
            if time.monotonic() >= deadline:
                return busy
            if cancel_event is not None and cancel_event.is_set():
                return busy
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))

    def acquire_many(
        self, count: int, *, cancel_event: threading.Event | None = None
    ) -> tuple[str, ...]:
        """Block until ``count`` GPUs are eligible and claim them atomically."""
        if type(count) is not int or count < 1:
            raise ValueError(f"GPU count must be a positive integer, got {count!r}")
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise _BenchSuiteCancelled
            self.refresh_external_occupancy()
            selected = self.try_acquire_many(count)
            if selected is not None:
                return selected
            if cancel_event is None:
                time.sleep(POLL_INTERVAL)
            elif cancel_event.wait(POLL_INTERVAL):
                raise _BenchSuiteCancelled

    def acquire(self) -> str:
        """Backward-compatible single-GPU acquisition."""
        return self.acquire_many(1)[0]

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

    def try_acquire_exact(self, indices: tuple[str, ...]) -> tuple[str, ...] | None:
        """Atomically reclaim one exact previously assigned GPU set."""
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(f"exact GPU claim must contain unique indices, got {indices!r}")
        requested = tuple(indices)
        with self._changed:
            if self._external_occupied is None:
                return None
            if any(
                idx not in self._known_indices
                or (self._allowed is not None and idx not in self._allowed)
                or idx in self._owned
                or idx in self._external_occupied
                for idx in requested
            ):
                return None
            self._owned.update(requested)
            return requested

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
                self._changed.notify_all()
        return occupied

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
            self._changed.wait_for(lambda: self._change_generation != generation or stop.is_set())
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
                    "cuda_arch": "sm_100a",
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
    """One process-local workload prepare and its GPU-attempt resources."""

    workload: dict
    attempt: int
    process: subprocess.Popen
    control: socket.socket
    log_file: object
    log_path: Path
    workdir: str
    started_at: float
    state: str = "PREPARING_CPU"
    buffer: bytearray = field(default_factory=bytearray)
    gpus: tuple[str, ...] = ()
    gpu_affinity: tuple[str, ...] = ()
    physical_gpu_uuids: tuple[str, ...] = ()
    gpu_ownership_released: bool = False
    pending_interference: dict[str, Any] | None = None
    ready_since: float | None = None
    interference_stop_deadline: float | None = None

    @property
    def label(self) -> str:
        return f"{self.workload['kernel']}/{self.workload['config']}"


def _prepared_child_command(
    workload: dict, *, control_fd: int, rounds: int, cooldown: float
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
    env[TVM_FFI_DISABLE_TORCH_C_DLPACK_ENV] = "1"
    env[PREPARE_CUDA_ARCH_ENV] = str(compile_profile["cuda_arch"])
    env[PREPARE_NUM_SMS_ENV] = str(compile_profile["num_sms"])
    repo_root = str(_kernels_repo_root())
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not python_path else f"{repo_root}{os.pathsep}{python_path}"
    configured_cache = env.get("TIRX_BENCH_CACHE_DIR")
    if configured_cache:
        cache_dir = Path(configured_cache).expanduser().resolve()
    else:
        user_cache = Path(env.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = (user_cache / "tirx-kernels" / "bench-suite").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["TIRX_BENCH_CACHE_DIR"] = str(cache_dir)
    command = _prepared_child_command(
        workload, control_fd=child_control.fileno(), rounds=rounds, cooldown=cooldown
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
        started_at=process_started,
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
        "process_model": "one_shot_child_per_workload",
        "started_at": datetime.fromtimestamp(attempt.started_at, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "retry_in_place": attempt.attempt > 1,
    }


def _finish_attempt_process(attempt: _PreparedAttempt) -> None:
    """Close host resources after the already-reaped child."""
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
        phase = message.get("phase", "unknown")
        error = f"{phase}: {message.get('error', 'unknown child failure')}"
        if message.get("traceback"):
            error += "\n" + message["traceback"]
    record.update({"status": "FAIL", "error": error, "finished_at": now_iso()})
    return record

def _reap_subprocess(proc: subprocess.Popen) -> None:
    """Ensure the child is reaped so it cannot linger as a zombie holding VRAM."""
    try:
        proc.wait(timeout=0)
    except subprocess.TimeoutExpired:
        _terminate_subprocess(proc)
    except ChildProcessError:
        pass


def _run_subprocess_monitored(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
    log_path: Path,
    gpu_indices: tuple[str, ...],
    monitor_interval: float,
    sm_threshold: float,
    cancel_event: threading.Event | None = None,
    timeout_s: float | None = None,
) -> tuple[int, bool, list[int], bool]:
    """Spawn ``cmd`` on assigned GPUs and watch all of them for active intruders.

    Returns (returncode, interfered, intruder_pids, cancelled).

    Interference means a foreign CUDA process exceeds ``sm_threshold`` on an
    assigned card.  Per-process ``pmon`` utilization stays meaningful while
    the benchmark itself drives aggregate device utilization to 100%, and it
    permits idle resident contexts when a nonzero threshold is requested.

    Protection is applied immediately before spawn and after every process wait
    poll.  Registered bench subprocesses and their descendants (for example
    nvcc) are excluded.
    """
    proc: subprocess.Popen | None = None
    registered_pid: int | None = None
    intruders: list[int] = []
    interfered = False
    cancelled = False
    started_at = time.monotonic()
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError(f"timeout_s must be finite and positive, got {timeout_s}")
    if cancel_event is not None and cancel_event.is_set():
        return -1, False, [], True
    if gpu_indices:
        pre = _active_strangers(gpu_indices, _our_pids(), sm_threshold)
        if pre is None:
            with open(log_path, "w") as lf:
                lf.write("RACE_LOST: could not sample assigned GPU utilization\n")
            return -1, True, [], False
        if pre:
            with open(log_path, "w") as lf:
                lf.write(f"RACE_LOST: pre-spawn check — active foreign processes {pre}\n")
            return -1, True, sorted(pre), False
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True
        )
    registered_pid = proc.pid
    _BenchPidRegistry.register(registered_pid)
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                _terminate_subprocess(proc)
                break
            try:
                proc.wait(timeout=monitor_interval)
                break  # subprocess exited normally
            except subprocess.TimeoutExpired:
                pass
            if timeout_s is not None and time.monotonic() - started_at >= timeout_s:
                with open(log_path, "a") as lf:
                    lf.write(f"\nTIMEOUT: subprocess exceeded {timeout_s:g} seconds\n")
                _terminate_subprocess(proc)
                break
            if not gpu_indices:
                continue
            active = _active_strangers(gpu_indices, _our_pids(), sm_threshold)
            if active is None:
                interfered = True
                _terminate_subprocess(proc)
                break
            if active:
                interfered = True
                intruders = sorted(active)
                _terminate_subprocess(proc)
                break
    except KeyboardInterrupt:
        _terminate_subprocess(proc)
        raise
    finally:
        if registered_pid is not None:
            _BenchPidRegistry.unregister(registered_pid)
        if proc is not None:
            _reap_subprocess(proc)
    return proc.returncode, interfered, intruders, cancelled


_LOCAL_ONLY_SITECUSTOMIZE = """\
from _tirx_bench_nvrtc132_sitecustomize import install as install_nvrtc
from _tirx_bench_local_only_runtime import install

install_nvrtc()
install()
"""
_LOCAL_ONLY_RUNTIME_MODULE = "_tirx_bench_local_only_runtime.py"
_NVRTC132_RUNTIME_MODULE = "_tirx_bench_nvrtc132_sitecustomize.py"
_local_only_hook_lock = threading.Lock()


def _bench_command(workload: dict, json_path: Path, *, rounds: int, cooldown: float) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "tirx_kernels.bench",
        "--kernel",
        workload["kernel"],
        "--config",
        workload["config"],
        "--json-file",
        str(json_path),
        "--rounds",
        str(rounds),
        "--cooldown",
        str(cooldown),
    ]
    for flag, field in (("--warmup", "warmup"), ("--repeat", "repeat"), ("--timer", "timer")):
        if workload.get(field) is not None:
            cmd += [flag, str(workload[field])]
    return cmd


def _prepend_pythonpath(path: Path, env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    existing = out.get("PYTHONPATH")
    out["PYTHONPATH"] = str(path.resolve()) + (os.pathsep + existing if existing else "")
    return out


def _materialize_local_only_runtime_hook(cache_root: Path) -> Path:
    """Copy the gate-owned runtime hook outside both measured checkouts."""

    source = SCRIPT_DIR / "_local_only_runtime.py"
    runtime_source = source.read_text()
    nvrtc_source = (SCRIPT_DIR / "_nvrtc132_sitecustomize.py").read_text()
    digest = hashlib.sha256((runtime_source + "\0" + nvrtc_source).encode()).hexdigest()
    hook_dir = cache_root / f"local-only-runtime-{digest[:16]}"
    expected = {
        hook_dir / "sitecustomize.py": _LOCAL_ONLY_SITECUSTOMIZE,
        hook_dir / _LOCAL_ONLY_RUNTIME_MODULE: runtime_source,
        hook_dir / _NVRTC132_RUNTIME_MODULE: nvrtc_source,
    }
    with _local_only_hook_lock:
        hook_dir.mkdir(parents=True, exist_ok=True)
        for path, content in expected.items():
            if path.exists():
                if path.read_text() != content:
                    raise ValueError(f"paired local-only runtime hook changed unexpectedly: {path}")
                continue
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            temporary.write_text(content)
            os.replace(temporary, path)
    return hook_dir


def _local_only_runtime_sha256(hook_dir: Path) -> str:
    return hashlib.sha256((hook_dir / _LOCAL_ONLY_RUNTIME_MODULE).read_bytes()).hexdigest()


def _paired_local_only_env(
    checkout: Path, hook_dir: Path, workload: dict, env: dict[str, str]
) -> dict[str, str]:
    out = _checkout_pythonpath(checkout, env)
    out = _prepend_pythonpath(hook_dir, out)
    out[LOCAL_ONLY_ENV] = "1"
    out[LOCAL_ONLY_KERNEL_ENV] = workload["kernel"]
    return out


def _daily_bench_env(env: dict[str, str]) -> dict[str, str]:
    """Keep local-only gate state out of ordinary reference benchmarks."""

    out = dict(env)
    out.pop(LOCAL_ONLY_ENV, None)
    out.pop(LOCAL_ONLY_KERNEL_ENV, None)
    return out


def _checkout_pythonpath(checkout: Path, env: dict[str, str]) -> dict[str, str]:
    """Return an environment whose first ``tirx_kernels`` import comes from checkout."""
    checkout = checkout.resolve()
    if not (checkout / "tirx_kernels" / "__init__.py").is_file():
        raise ValueError(f"not a tirx-kernels checkout: {checkout}")
    return _prepend_pythonpath(checkout, env)


def _read_bench_result(
    json_path: Path,
    workload: dict,
    *,
    rounds: int,
    cooldown: float = DEFAULT_COOLDOWN_S,
    require_local_only: bool = False,
) -> dict:
    payload = json.loads(json_path.read_text())
    rows = payload.get("results") or []
    row = next(
        (
            item
            for item in rows
            if item.get("kernel") == workload["kernel"] and item.get("label") == workload["config"]
        ),
        None,
    )
    if row is None:
        raise ValueError(f"no matching row in bench JSON ({len(rows)} rows)")
    if row.get("status") in ("FAIL", "SKIP"):
        return row
    if require_local_only and row.get("local_only") is not True:
        raise ValueError("paired benchmark result did not confirm local_only=true")
    _finalize_bench_record(row, rounds=rounds, cooldown=cooldown)
    return row


def _local_samples(row: dict, *, require_only: bool = False) -> tuple[str, list[float]]:
    impls = our_impls(row.get("round_samples") or row.get("impls") or {})
    if len(impls) != 1:
        raise ValueError(f"expected exactly one local TIR/TIRx implementation, got {impls}")
    name = impls[0]
    if require_only:
        reported = set(row.get("round_samples") or row.get("impls") or {})
        if reported != {name}:
            raise ValueError(
                f"local-only benchmark must report only {name!r}, got {sorted(reported)}"
            )
    values = row.get("round_samples", {}).get(name)
    if values is None:
        value = (row.get("impls") or {}).get(name)
        values = [value] if value is not None else []
    if not values or any(
        value is None or not math.isfinite(float(value)) or float(value) <= 0 for value in values
    ):
        raise ValueError(f"invalid local samples for {name!r}: {values}")
    return name, [float(value) for value in values]


def _checkout_source_manifest(checkout: Path) -> tuple[tuple[str, int, int], ...]:
    """Cheap source-tree identity used to detect edits during a paired sweep."""
    checkout = checkout.resolve()
    rows = []
    for path in sorted((checkout / "tirx_kernels").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        stat = path.stat()
        rows.append((str(path.relative_to(checkout)), stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


def _checkout_provenance(checkout: Path) -> dict[str, str | None]:
    checkout = checkout.resolve()
    digest = hashlib.sha256()
    for path in sorted((checkout / "tirx_kernels").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(checkout)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"path": str(checkout), "git": git_label(checkout), "source_sha256": digest.hexdigest()}


def _capture_checkout_snapshot(checkout: Path) -> dict:
    """Capture a stable content hash plus a cheap manifest for repeated checks."""
    checkout = checkout.resolve()
    for _ in range(3):
        manifest = _checkout_source_manifest(checkout)
        provenance = _checkout_provenance(checkout)
        if _checkout_source_manifest(checkout) == manifest:
            return {"provenance": provenance, "manifest": manifest}
    raise ValueError(f"checkout source changed while snapshotting: {checkout}")


def _assert_checkout_unchanged(checkout: Path, snapshot: dict, *, full: bool = False) -> None:
    """Fail closed if a paired checkout changes after its initial snapshot."""
    checkout = checkout.resolve()
    if _checkout_source_manifest(checkout) != snapshot["manifest"]:
        raise ValueError(f"checkout source changed during paired sweep: {checkout}")
    if full and _checkout_provenance(checkout) != snapshot["provenance"]:
        raise ValueError(f"checkout provenance changed during paired sweep: {checkout}")


def _checkout_cache_key(provenance: dict[str, str | None]) -> str:
    checkout_name = Path(str(provenance["path"])).name
    return f"{provenance['git'] or checkout_name}-{provenance['source_sha256'][:16]}".replace(
        os.sep, "_"
    )


_PAIRED_CONFIG_RESOLVER = r"""
import json
import math
import sys

from tirx_kernels.registry import discover_kernels


def encode(value):
    value_type = type(value)
    if value is None:
        return ["none"]
    if value_type is bool:
        return ["bool", value]
    if value_type is int:
        return ["int", str(value)]
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite config float: {value}")
        return ["float", value.hex()]
    if value_type is str:
        return ["str", value]
    if value_type is list:
        return ["list", [encode(item) for item in value]]
    if value_type is tuple:
        return ["tuple", [encode(item) for item in value]]
    if value_type is dict:
        items = [[encode(key), encode(item)] for key, item in value.items()]
        items.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return ["dict", items]
    raise TypeError(f"unsupported bench config value {value!r}: {value_type}")


workloads = json.loads(sys.stdin.read())
kernels = discover_kernels(strict=True)
resolved = []
for workload in workloads:
    kernel = workload["kernel"]
    label = workload["config"]
    try:
        module = kernels[kernel]
    except KeyError:
        raise ValueError(f"unknown kernel in bench workload: {kernel}") from None
    configs = getattr(module, "BENCH_CONFIGS", getattr(module, "CONFIGS", []))
    matches = [config for config in configs if config.get("label", "default") == label]
    if len(matches) != 1:
        raise ValueError(
            f"{kernel}/{label}: expected exactly one BENCH_CONFIGS/CONFIGS match, got {len(matches)}"
        )
    params = {key: value for key, value in matches[0].items() if key != "label"}
    resolved.append({"kernel": kernel, "config": label, "params": encode(params)})
json.dump(resolved, sys.stdout, separators=(",", ":"))
"""


def _resolve_checkout_bench_configs(checkout: Path, workloads: list[dict]) -> list[dict]:
    """Resolve the actual module bench parameters selected by each YAML label."""
    checkout = checkout.resolve()
    env = _checkout_pythonpath(checkout, os.environ.copy())
    payload = [{"kernel": row["kernel"], "config": row["config"]} for row in workloads]
    result = subprocess.run(
        [sys.executable, "-c", _PAIRED_CONFIG_RESOLVER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=checkout,
        env=env,
        timeout=600,
    )
    if result.returncode != 0:
        raise ValueError(
            f"failed to resolve bench configs in {checkout}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid bench config resolution from {checkout}: {error}") from error


def _validate_paired_config_resolution(
    workloads: list[dict], old_configs: list[dict], current_configs: list[dict]
) -> str:
    """Require both checkouts to map every requested label to identical parameters."""
    expected = [(row["kernel"], row["config"]) for row in workloads]
    for role, configs in (("old", old_configs), ("current", current_configs)):
        actual = [(row.get("kernel"), row.get("config")) for row in configs]
        if actual != expected:
            raise ValueError(f"paired {role} config resolution identities differ from workloads")
    if old_configs != current_configs:
        mismatch = next(
            index
            for index, (old_config, current_config) in enumerate(
                zip(old_configs, current_configs, strict=True)
            )
            if old_config != current_config
        )
        kernel, config = expected[mismatch]
        raise ValueError(
            f"paired config parameters differ for {kernel}/{config}: "
            f"old={old_configs[mismatch]['params']} current={current_configs[mismatch]['params']}"
        )
    canonical = json.dumps(old_configs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _validate_ir_builder_migration_gate_options(
    *,
    enabled: bool,
    old_checkout: Path | None,
    workloads_path: Path | None,
    all_configs: bool,
    kernel_filter: str | None,
    rounds: int,
    cooldown: float,
    speedup_threshold: float,
    util_threshold: float,
    mem_threshold: float,
    check_imports: bool,
) -> None:
    """Keep the formal migration acceptance mode exact and non-overridable."""
    if not enabled:
        return
    errors = []
    if old_checkout is None:
        errors.append("--paired-old-checkout is required")
    if workloads_path is not None:
        errors.append("--workloads cannot replace the canonical config directory")
    if all_configs:
        errors.append("--all-configs cannot widen the canonical default:true sweep")
    if kernel_filter is not None:
        errors.append("--filter cannot narrow the canonical config directory")
    if rounds != IR_BUILDER_MIGRATION_ROUNDS:
        errors.append(f"--rounds must be exactly {IR_BUILDER_MIGRATION_ROUNDS}")
    if cooldown != DEFAULT_COOLDOWN_S:
        errors.append(f"--cooldown must be exactly {DEFAULT_COOLDOWN_S:g}")
    if speedup_threshold != DEFAULT_PAIRED_SPEEDUP_THRESHOLD:
        errors.append(f"--speedup-threshold must be exactly {DEFAULT_PAIRED_SPEEDUP_THRESHOLD:g}")
    if util_threshold != DEFAULT_UTIL_THRESHOLD:
        errors.append(f"--util-threshold must be exactly {DEFAULT_UTIL_THRESHOLD:g}")
    if mem_threshold != IR_BUILDER_MIGRATION_MEM_THRESHOLD:
        errors.append(
            f"--mem-threshold must be exactly {IR_BUILDER_MIGRATION_MEM_THRESHOLD:g}"
        )
    if check_imports:
        errors.append("--check-imports does not execute the performance gate")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_ir_builder_migration_workloads(
    workloads: list[dict], required_workloads: list[dict]
) -> None:
    """Require the exact mechanically derived non-exempt workload sequence."""
    if workloads != required_workloads:
        raise ValueError(
            "migration gate workloads differ from the canonical non-exempt default:true sweep"
        )


def _mapped_nvrtc_paths() -> list[Path]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        raise ValueError("cannot inspect loaded NVRTC path: /proc/self/maps is unavailable")
    paths = set()
    for line in maps_path.read_text().splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1].removesuffix(" (deleted)"))
        if path.name.startswith("libnvrtc.so"):
            paths.add(path.resolve())
    return sorted(paths)


def _mapped_nvrtc_builtins_paths() -> list[Path]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        raise ValueError("cannot inspect loaded NVRTC builtins path: /proc/self/maps is unavailable")
    paths = set()
    for line in maps_path.read_text().splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1].removesuffix(" (deleted)"))
        if path.name.startswith("libnvrtc-builtins.so"):
            paths.add(path.resolve())
    return sorted(paths)


def _query_nvrtc_version() -> tuple[int, int]:
    library = ctypes.CDLL("libnvrtc.so.13")
    major = ctypes.c_int()
    minor = ctypes.c_int()
    version = library.nvrtcVersion
    version.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    version.restype = ctypes.c_int
    result = version(ctypes.byref(major), ctypes.byref(minor))
    if result != 0:
        raise ValueError(f"nvrtcVersion failed with result {result}")
    return major.value, minor.value


def _nvrtc_preflight() -> dict:
    selected = os.environ.get(NVRTC_LIBRARY_ENV)
    if not selected:
        raise ValueError(
            f"{NVRTC_LIBRARY_ENV} is not set; launch the formal gate with "
            "python -m tirx_kernels.bench_suite.run_ir_builder_gate"
        )
    selected_path = Path(selected).expanduser().resolve()
    if not selected_path.is_file():
        raise ValueError(f"selected NVRTC library does not exist: {selected_path}")
    selected_builtins = os.environ.get(NVRTC_BUILTINS_LIBRARY_ENV)
    if not selected_builtins:
        raise ValueError(f"{NVRTC_BUILTINS_LIBRARY_ENV} is not set by the formal gate wrapper")
    selected_builtins_path = Path(selected_builtins).expanduser().resolve()
    if not selected_builtins_path.is_file():
        raise ValueError(f"selected NVRTC builtins library does not exist: {selected_builtins_path}")

    version = _query_nvrtc_version()
    loaded_paths = _mapped_nvrtc_paths()
    loaded_builtins_paths = _mapped_nvrtc_builtins_paths()
    matching = [path for path in loaded_paths if path.samefile(selected_path)]
    unexpected = [path for path in loaded_paths if not path.samefile(selected_path)]
    if not matching or unexpected:
        raise ValueError(
            f"loaded NVRTC paths do not match {NVRTC_LIBRARY_ENV}={selected_path}: "
            f"loaded={[str(path) for path in loaded_paths]}"
        )
    matching_builtins = [
        path for path in loaded_builtins_paths if path.samefile(selected_builtins_path)
    ]
    unexpected_builtins = [
        path for path in loaded_builtins_paths if not path.samefile(selected_builtins_path)
    ]
    if not matching_builtins or unexpected_builtins:
        raise ValueError(
            f"loaded NVRTC builtins paths do not match "
            f"{NVRTC_BUILTINS_LIBRARY_ENV}={selected_builtins_path}: "
            f"loaded={[str(path) for path in loaded_builtins_paths]}"
        )
    if version != IR_BUILDER_MIGRATION_REQUIRED_NVRTC_VERSION:
        raise ValueError(
            f"formal gate requires NVRTC {IR_BUILDER_MIGRATION_REQUIRED_NVRTC_VERSION}, "
            f"loaded {version} from {selected_path}"
        )
    return {
        "required_version": list(IR_BUILDER_MIGRATION_REQUIRED_NVRTC_VERSION),
        "version": list(version),
        "library_path": str(selected_path),
        "library_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "loaded_paths": [str(path) for path in loaded_paths],
        "builtins_library_path": str(selected_builtins_path),
        "builtins_library_sha256": hashlib.sha256(
            selected_builtins_path.read_bytes()
        ).hexdigest(),
        "loaded_builtins_paths": [str(path) for path in loaded_builtins_paths],
        "selection_env": NVRTC_LIBRARY_ENV,
        "applies_to": ["old", "current"],
    }


def _paired_order(round_index: int) -> tuple[str, str]:
    return ("old", "current") if round_index % 2 == 0 else ("current", "old")


def _direct_speedup_pct(old_us: float, current_us: float) -> float:
    if not math.isfinite(old_us) or not math.isfinite(current_us) or old_us <= 0 or current_us <= 0:
        raise ValueError(
            f"paired times must be finite and positive: old={old_us}, current={current_us}"
        )
    speedup_pct = (old_us / current_us - 1.0) * 100.0
    if not math.isfinite(speedup_pct):
        raise ValueError(f"paired speedup must be finite: old={old_us}, current={current_us}")
    return speedup_pct


def run_one_paired(
    workload: dict,
    pool: GpuPool,
    log_dir: Path,
    *,
    old_checkout: Path,
    current_checkout: Path,
    attempt: int = 1,
    rounds: int = DEFAULT_ROUNDS,
    cooldown: float = DEFAULT_COOLDOWN_S,
    speedup_threshold: float = DEFAULT_PAIRED_SPEEDUP_THRESHOLD,
    checkout_snapshots: dict[str, dict] | None = None,
    config_params_sha256: str | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Counterbalanced old/current benchmark within one exclusive GPU claim.

    Each side is invoked once per round with ``--rounds 1``.  The order is
    AB/BA/AB/... so thermal or clock drift is not owned by one checkout.  Each
    subprocess retains the existing timer, GPU-interference monitor, cache root
    and provenance conventions while excluding external reference work.
    """
    kernel, config = workload["kernel"], workload["config"]
    checkouts = {"old": old_checkout.resolve(), "current": current_checkout.resolve()}
    if checkout_snapshots is None:
        checkout_snapshots = {
            side: _capture_checkout_snapshot(checkout) for side, checkout in checkouts.items()
        }
    for side, checkout in checkouts.items():
        _assert_checkout_unchanged(checkout, checkout_snapshots[side])
    gpus = pool.acquire_many(workload.get("num_gpus", 1), cancel_event=cancel_event)
    gpu_csv = ",".join(gpus)
    gpu_uuid_by_index = dict(pool._all_gpus()) if hasattr(pool, "_all_gpus") else {}
    record: dict = {
        "kernel": kernel,
        "config": config,
        "label": config,
        "gpu": gpu_csv,
        "gpus": list(gpus),
        "gpu_uuids": [gpu_uuid_by_index.get(index) for index in gpus],
        "num_gpus": workload.get("num_gpus", 1),
        "started_at": now_iso(),
        "paired": {
            "old": checkout_snapshots["old"]["provenance"],
            "current": checkout_snapshots["current"]["provenance"],
            "order": [],
            "threshold_pct": speedup_threshold,
            "config_params_sha256": config_params_sha256,
            "local_only": True,
        },
    }
    samples: dict[str, list[float]] = {"old": [], "current": []}
    diagnostics: dict[str, list[dict]] = {"old": [], "current": []}
    try:
        base_env = os.environ.copy()
        base_env["CUDA_VISIBLE_DEVICES"] = gpu_csv
        cache_root = (log_dir.parent / "cache" / "paired").resolve()
        hook_dir = _materialize_local_only_runtime_hook(cache_root)
        record["paired"]["runtime_hook_sha256"] = _local_only_runtime_sha256(hook_dir)
        cache_keys = {
            side: _checkout_cache_key(snapshot["provenance"])
            for side, snapshot in checkout_snapshots.items()
        }
        for cache_key in cache_keys.values():
            (cache_root / cache_key).mkdir(parents=True, exist_ok=True)

        for round_index in range(rounds):
            order = _paired_order(round_index)
            record["paired"]["order"].append(list(order))
            for side in order:
                checkout = checkouts[side]
                _assert_checkout_unchanged(checkout, checkout_snapshots[side])
                env = _paired_local_only_env(checkout, hook_dir, workload, base_env)
                env["TIRX_BENCH_CACHE_DIR"] = str(cache_root / cache_keys[side])
                with tempfile.TemporaryDirectory(
                    prefix=f"bench-suite-paired-{kernel}-{config}-{side}-"
                ) as workdir:
                    json_path = Path(workdir) / "result.json"
                    log_path = log_dir / (
                        f"{kernel}__{config}__a{attempt}__r{round_index + 1}__{side}.log"
                    )
                    cmd = _bench_command(workload, json_path, rounds=1, cooldown=cooldown)
                    returncode, interfered, intruders, cancelled = _run_subprocess_monitored(
                        cmd,
                        env,
                        workdir,
                        log_path,
                        gpus,
                        MONITOR_INTERVAL,
                        pool.util_threshold,
                        cancel_event,
                        PAIRED_SIDE_TIMEOUT_S,
                    )
                    memory_busy = pool.wait_for_memory_release(gpus, cancel_event=cancel_event)
                    if cancelled:
                        record.update(status="CANCELLED", error="suite stopped after failure")
                        return record
                    if interfered:
                        record.update(
                            status="INTERFERED",
                            intruder_pids=intruders,
                            error=f"gpus {gpu_csv}: intruder PIDs {intruders}",
                        )
                        return record
                    if returncode != 0:
                        tail = "\n".join(log_path.read_text().splitlines()[-30:])
                        memory_detail = (
                            f"\nGPU framebuffer did not return below {pool.mem_threshold:g}%: "
                            f"{memory_busy}"
                            if memory_busy
                            else ""
                        )
                        record.update(
                            status="FAIL", error=f"{side} exit {returncode}\n{tail}{memory_detail}"
                        )
                        return record
                    if memory_busy:
                        record.update(
                            status="FAIL",
                            error=(
                                f"{side}: GPU framebuffer did not return below "
                                f"{pool.mem_threshold:g}% after subprocess exit: {memory_busy}"
                            ),
                        )
                        return record
                    row = _read_bench_result(
                        json_path,
                        workload,
                        rounds=1,
                        cooldown=cooldown,
                        require_local_only=True,
                    )
                    if row.get("status") != "ok":
                        record.update(
                            status="FAIL",
                            error=f"{side}: {row.get('error') or row.get('reason') or row}",
                        )
                        return record
                    local_impl, values = _local_samples(row, require_only=True)
                    samples[side].append(values[0])
                    diagnostics[side].append(
                        {
                            "local_impl": local_impl,
                            "impls": row.get("impls") or {},
                            "round_samples": row.get("round_samples") or {},
                            "local_only": row.get("local_only"),
                        }
                    )

        old_us = statistics.mean(samples["old"])
        current_us = statistics.mean(samples["current"])
        speedup_pct = _direct_speedup_pct(old_us, current_us)
        record["paired"].update(
            old_samples_us=samples["old"],
            current_samples_us=samples["current"],
            old_us=old_us,
            current_us=current_us,
            speedup_pct=speedup_pct,
            diagnostics=diagnostics,
        )
        record["impls"] = {"old": old_us, "current": current_us}
        record["round_samples"] = {"old": samples["old"], "current": samples["current"]}
        record["aggregated"] = {"rounds": rounds, "method": "mean"}
        record["local_only"] = True
        if speedup_pct <= speedup_threshold:
            record.update(
                status="REGRESSION",
                error=(
                    f"direct old/current speedup {speedup_pct:.3f}% must be "
                    f"> {speedup_threshold:.3f}%"
                ),
            )
        else:
            record["status"] = "ok"
        return record
    except Exception as error:
        record.update(status="FAIL", error=repr(error))
        return record
    finally:
        try:
            for side, checkout in checkouts.items():
                _assert_checkout_unchanged(checkout, checkout_snapshots[side])
        except Exception as error:
            record.update(status="FAIL", error=repr(error))
        record["finished_at"] = now_iso()
        pool.release_many(gpus)


def run_one(
    workload: dict,
    pool: GpuPool,
    log_dir: Path,
    *,
    attempt: int = 1,
    rounds: int = DEFAULT_ROUNDS,
    cooldown: float = DEFAULT_COOLDOWN_S,
    cancel_event: threading.Event | None = None,
) -> dict:
    kernel = workload["kernel"]
    config = workload["config"]
    warmup = workload.get("warmup")
    repeat = workload.get("repeat")
    timer = workload.get("timer")
    num_gpus = workload.get("num_gpus", 1)

    gpus = pool.acquire_many(num_gpus, cancel_event=cancel_event)
    gpu_csv = ",".join(gpus)
    json_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    json_tmp.close()
    log_path = log_dir / f"{kernel}__{config}__a{attempt}.log"

    cmd = _bench_command(workload, Path(json_tmp.name), rounds=rounds, cooldown=cooldown)

    env = _daily_bench_env(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_csv
    # Reference-library autotune caches must survive the per-workload scratch
    # cwd and must be populated before timing. Individual adapters use
    # per-op/per-shape files below this absolute directory, so concurrent suite
    # workers do not overwrite one another's selections.
    cache_dir = (log_dir.parent / "cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["TIRX_BENCH_CACHE_DIR"] = str(cache_dir)

    # Each workload gets its own scratch cwd so concurrent runs don't race on
    # proton's <proton_name>.hatchet file.
    workdir = tempfile.mkdtemp(prefix=f"bench-suite-{kernel}-{config}-")

    label = f"{kernel}/{config}"
    worker = threading.current_thread().name
    started = now_iso()
    record: dict = {
        "kernel": kernel,
        "config": config,
        "gpu": gpu_csv,
        "gpus": list(gpus),
        "num_gpus": num_gpus,
        "started_at": started,
    }
    interfered = False
    intruder_pids: list[int] = []
    try:
        log(f"[bench-suite] {started} {worker} gpus={gpu_csv} START {label} (attempt {attempt})")
        # Pass every physical GPU index; the monitor uses per-PID sm-util (pmon).
        returncode, interfered, intruder_pids, cancelled = _run_subprocess_monitored(
            cmd, env, workdir, log_path, gpus, MONITOR_INTERVAL, pool.util_threshold, cancel_event
        )
        memory_busy = pool.wait_for_memory_release(gpus, cancel_event=cancel_event)
        if cancelled:
            record["status"] = "CANCELLED"
            record["error"] = "suite stopped after another workload failed"
        elif interfered:
            record["status"] = "INTERFERED"
            record["intruder_pids"] = intruder_pids
            record["error"] = f"gpus {gpu_csv}: intruder PIDs {intruder_pids}"
        elif returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            memory_detail = (
                f"\nGPU framebuffer did not return below {pool.mem_threshold:g}%: {memory_busy}"
                if memory_busy
                else ""
            )
            record["status"] = "FAIL"
            record["error"] = f"exit {returncode}\n{tail}{memory_detail}"
        elif memory_busy:
            record["status"] = "FAIL"
            record["error"] = (
                f"GPU framebuffer did not return below {pool.mem_threshold:g}% "
                f"after subprocess exit: {memory_busy}"
            )
        else:
            payload = json.loads(Path(json_tmp.name).read_text())
            rows = payload.get("results") or []
            match = next(
                (r for r in rows if r.get("kernel") == kernel and r.get("label") == config), None
            )
            if match is None:
                record["status"] = "FAIL"
                record["error"] = f"no matching row in bench JSON ({len(rows)} rows)"
            else:
                st = match.get("status")
                if st == "SKIP":
                    record.update(match)
                elif st == "FAIL":
                    record.update(match)
                    record.setdefault("status", "FAIL")
                else:
                    _finalize_bench_record(match, rounds=rounds, cooldown=cooldown)
                    record.update(match)
                    record.setdefault("label", config)
                    if record.get("status") != "ok":
                        record["error"] = match.get("error", "bench finalize failed")
    except Exception as e:
        record["status"] = "FAIL"
        record["error"] = repr(e)
    finally:
        try:
            os.unlink(json_tmp.name)
        except FileNotFoundError:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
        pool.release_many(gpus)

    record["finished_at"] = now_iso()
    status = record.get("status", "ok")
    impls = record.get("impls") or {}
    impl_str = ", ".join(f"{k}={v:.3f}µs" for k, v in impls.items())
    if interfered:
        # Make INTERFERED stand out — easy to spot when scrolling.
        log("[bench-suite] " + "*" * 70)
        log(f"[bench-suite] *** INTERFERED *** {worker} gpus={gpu_csv} {label} attempt {attempt}")
        log(f"[bench-suite] ***   intruder PIDs on gpus {gpu_csv}: {intruder_pids}")
        log("[bench-suite] ***   subprocess killed, will retry until ok")
        log("[bench-suite] " + "*" * 70)
    else:
        log(
            f"[bench-suite] {record['finished_at']} {worker} gpus={gpu_csv} "
            f"{status:4s} {label} {impl_str}"
        )
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
    environment: dict | None = None,
    scope: dict | None = None,
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
        "environment": environment or {},
        "scope": scope or {},
        "results": results,
    }
    path = runs_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _next_run_id(out_dir: Path) -> str:
    """Return an unused monotonically increasing numeric artifact id."""
    occupied: set[int] = set()
    runs_dir = out_dir / "runs"
    for path in runs_dir.iterdir() if runs_dir.is_dir() else ():
        if path.suffix in {".json", ".log"} and path.stem.isdigit():
            occupied.add(int(path.stem))
    reports_dir = out_dir / "reports"
    for path in reports_dir.iterdir() if reports_dir.is_dir() else ():
        if path.is_dir() and path.name.isdigit():
            occupied.add(int(path.name))
    return str(max(occupied, default=0) + 1)


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


def _finalize_bench_record(
    row: dict, *, rounds: int, cooldown: float = DEFAULT_COOLDOWN_S
) -> None:
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
    protocol_cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
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


def _run_paired_scheduled_jobs(
    workloads: list[dict],
    pool: GpuPool,
    log_dir: Path,
    *,
    rounds: int,
    cooldown: float,
    old_checkout: Path | None = None,
    current_checkout: Path | None = None,
    speedup_threshold: float = DEFAULT_PAIRED_SPEEDUP_THRESHOLD,
    checkout_snapshots: dict[str, dict] | None = None,
    config_params_sha256: str | None = None,
) -> tuple[list[dict], list[tuple[str, str, int, str]]]:
    """Run one subprocess per workload and stop the suite on the first failure.

    External interference is retried because it is not a workload failure. Any
    actual FAIL stops new scheduling and cancels subprocesses still in flight.
    """
    n_jobs = len(workloads)
    if not n_jobs:
        return [], []

    pending: queue.Queue[tuple[dict, int] | None] = queue.Queue()
    for w in workloads:
        pending.put((w, 1))

    records: list[dict] = []
    retry_log: list[tuple[str, str, int, str]] = []
    state_lock = threading.Lock()
    done_cv = threading.Condition(state_lock)
    cancel_event = threading.Event()
    resource_lock = threading.Lock()
    active_resources: set[str] = set()
    n_done = 0

    def try_acquire_resource(workload: dict) -> str | None | bool:
        """Reserve a named process-global resource without blocking a worker.

        ``None`` means the workload has no exclusive resource, a string is a
        successful reservation, and ``False`` asks the worker to requeue the
        workload so unrelated work can continue on the available GPUs.
        """
        resource = workload.get("exclusive_resource")
        if resource is None:
            return None
        with resource_lock:
            if resource in active_resources:
                return False
            active_resources.add(resource)
        return resource

    def release_resource(resource: str | None) -> None:
        if resource is None:
            return
        with resource_lock:
            active_resources.remove(resource)

    def worker() -> None:
        nonlocal n_done
        while not cancel_event.is_set():
            with state_lock:
                if n_done >= n_jobs:
                    return
            try:
                item = pending.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                pending.task_done()
                return
            if cancel_event.is_set():
                pending.task_done()
                return

            workload, attempt = item
            resource = try_acquire_resource(workload)
            if resource is False:
                pending.put(item)
                pending.task_done()
                time.sleep(0.01)
                continue
            kernel = workload["kernel"]
            config = workload["config"]
            try:
                if old_checkout is None:
                    record = run_one(
                        workload,
                        pool,
                        log_dir,
                        attempt=attempt,
                        rounds=rounds,
                        cooldown=cooldown,
                        cancel_event=cancel_event,
                    )
                else:
                    assert current_checkout is not None
                    record = run_one_paired(
                        workload,
                        pool,
                        log_dir,
                        old_checkout=old_checkout,
                        current_checkout=current_checkout,
                        attempt=attempt,
                        rounds=rounds,
                        cooldown=cooldown,
                        speedup_threshold=speedup_threshold,
                        checkout_snapshots=checkout_snapshots,
                        config_params_sha256=config_params_sha256,
                        cancel_event=cancel_event,
                    )
            except _BenchSuiteCancelled:
                pending.task_done()
                return
            except Exception as e:
                record = {
                    "kernel": kernel,
                    "config": config,
                    "label": config,
                    "status": "FAIL",
                    "error": repr(e),
                    "finished_at": now_iso(),
                }
            finally:
                release_resource(resource)

            status = record.get("status", "FAIL")
            if status in ("ok", "SKIP", "REGRESSION"):
                record["attempt"] = attempt
                with state_lock:
                    records.append(record)
                    n_done += 1
                    done_cv.notify_all()
            elif status == "INTERFERED":
                detail = record.get("error") or ""
                if record.get("intruder_pids"):
                    detail = f"intruders {record['intruder_pids']}"
                if not cancel_event.is_set():
                    with state_lock:
                        retry_log.append((kernel, config, attempt, detail[:240]))
                    log(
                        f"[bench-suite] >>> REQUEUE {kernel}/{config} "
                        f"attempt {attempt} (INTERFERED): {detail[:160]} <<<"
                    )
                    pending.put((workload, attempt + 1))
            elif status != "CANCELLED":
                record["status"] = "FAIL"
                record["attempt"] = attempt
                detail = record.get("error") or "unknown workload failure"
                with state_lock:
                    records.append(record)
                    n_done += 1
                    first_failure = not cancel_event.is_set()
                    cancel_event.set()
                    done_cv.notify_all()
                if first_failure:
                    log(
                        f"[bench-suite] >>> FAIL-FAST {kernel}/{config} "
                        f"attempt {attempt}: {detail[:160]} <<<"
                    )
            pending.task_done()

    n_workers = min(pool.total_visible(), n_jobs)
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="bench") as ex:
        futs = [ex.submit(worker) for _ in range(n_workers)]
        with state_lock:
            while n_done < n_jobs and not cancel_event.is_set():
                done_cv.wait(timeout=1.0)
        if not cancel_event.is_set():
            for _ in range(n_workers):
                pending.put(None)
        for fut in as_completed(futs):
            fut.result()
    return records, retry_log


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
    """Run bounded CPU preparation and late-bound GPU stages."""
    n_jobs = len(workloads)
    if not n_jobs:
        return [], [], {}

    visible = pool.total_visible()
    if max_prepare_processes is None:
        max_prepare_processes = min(n_jobs, max(1, min(os.cpu_count() or 1, max(4, visible * 2))))
    if ready_backlog is None:
        ready_backlog = max(max_prepare_processes, visible * 2)
    if max_prepare_processes < 1 or ready_backlog < max_prepare_processes:
        raise ValueError("ready_backlog must be >= max_prepare_processes >= 1")

    pending = deque(workloads)
    active: dict[int, _PreparedAttempt] = {}
    ready: deque[_PreparedAttempt] = deque()
    records: list[dict] = []
    retry_log: list[dict[str, Any]] = []
    completed = 0
    failed = False
    last_interference_poll = 0.0
    physical_uuid_by_index = dict(pool._all_gpus())

    def preparing_count() -> int:
        return sum(item.state == "PREPARING_CPU" for item in active.values())

    def buffered_count() -> int:
        return sum(item.state in ("PREPARING_CPU", "READY") for item in active.values())

    def remove_ready(item: _PreparedAttempt) -> None:
        try:
            ready.remove(item)
        except ValueError:
            pass

    def release_gpus(item: _PreparedAttempt) -> None:
        if item.gpus and not item.gpu_ownership_released:
            pool.release_many(item.gpus)
            item.gpu_ownership_released = True

    def fail(item: _PreparedAttempt, message: dict | None = None) -> None:
        nonlocal completed, failed
        if item.state == "FAILED":
            return
        item.state = "FAILED"
        record = _record_child_failure(item, message)
        records.append(record)
        completed += 1
        failed = True
        detail = record.get("error") or "unknown workload failure"
        log(f"[bench-suite] >>> FAIL-FAST {item.label} attempt {item.attempt}: {detail[:160]} <<<")

    def request_interference_stop(
        item: _PreparedAttempt, intruders: list[int], detail: str
    ) -> None:
        if item.state not in ("ASSIGNED", "RUNNING_GPU") or item.pending_interference:
            return
        item.pending_interference = {
            "intruder_pids": intruders,
            "detail": detail[:240],
        }
        item.interference_stop_deadline = time.monotonic() + 30.0
        item.state = "STOPPING_INTERFERED_GPU"
        log(
            f"[bench-suite] >>> INTERFERED {item.label} attempt {item.attempt}: "
            f"{detail[:160]}; retrying GPU stage in place <<<"
        )
        try:
            os.kill(item.process.pid, signal.SIGUSR1)
        except ProcessLookupError:
            fail(item, {"phase": "interference", "error": "child exited before stop signal"})

    def acknowledge_interference(item: _PreparedAttempt, message: dict) -> None:
        pending_interference = item.pending_interference or {}
        retry_log.append(
            {
                "kernel": item.workload["kernel"],
                "config": item.workload["config"],
                "attempt": item.attempt,
                "intruder_pids": pending_interference.get("intruder_pids", []),
                "detail": pending_interference.get("detail", "")[:240],
                "resident_context_bytes_after_cleanup": message.get(
                    "resident_context_bytes_after_cleanup", {}
                ),
                "retry_in_place": True,
            }
        )
        release_gpus(item)
        item.gpus = ()
        item.physical_gpu_uuids = ()
        item.pending_interference = None
        item.interference_stop_deadline = None
        item.attempt += 1
        item.state = "READY"
        item.ready_since = time.time()
        ready.append(item)

    def spawn_available() -> None:
        while (
            pending
            and not failed
            and preparing_count() < max_prepare_processes
            and buffered_count() < ready_backlog
            and len(active) < ready_backlog + visible
        ):
            workload = pending.popleft()
            item = _spawn_prepared_attempt(
                workload,
                1,
                log_dir,
                rounds=rounds,
                cooldown=cooldown,
                compile_profile=compile_profile,
            )
            active[item.control.fileno()] = item
            log(f"[bench-suite] {now_iso()} prepare pid={item.process.pid} START {item.label}")

    def dispatch_ready() -> None:
        if failed:
            return
        ordered = sorted(
            ready,
            key=lambda item: (
                -item.workload.get("num_gpus", 1),
                item.ready_since or item.started_at,
            ),
        )
        for item in ordered:
            count = item.workload.get("num_gpus", 1)
            gpus = (
                pool.try_acquire_exact(item.gpu_affinity)
                if item.gpu_affinity
                else pool.try_acquire_many(count)
            )
            if gpus is None:
                continue
            strangers = _active_strangers(gpus, _our_pids(), pool.util_threshold)
            if strangers is None or strangers:
                pool.release_many(gpus)
                continue
            remove_ready(item)
            if not item.gpu_affinity:
                item.gpu_affinity = gpus
            item.gpus = gpus
            item.gpu_ownership_released = False
            item.state = "ASSIGNED"
            _send_child(
                item,
                {
                    "type": "ASSIGN",
                    "gpu_indices": list(gpus),
                    "gpu_uuids": [physical_uuid_by_index[gpu] for gpu in gpus],
                },
            )

    occupancy_stop = threading.Event()
    dispatch_stop = threading.Event()
    dispatch_reader, dispatch_writer = socket.socketpair()
    dispatch_reader.setblocking(False)
    dispatch_writer.setblocking(False)

    def monitor_external_occupancy() -> None:
        while not occupancy_stop.is_set():
            pool.refresh_external_occupancy()
            if occupancy_stop.wait(POLL_INTERVAL):
                return

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

    occupancy_thread = threading.Thread(
        target=monitor_external_occupancy, name="bench-gpu-occupancy", daemon=True
    )
    dispatch_thread = threading.Thread(
        target=notify_dispatch_on_pool_change, name="bench-gpu-dispatch", daemon=True
    )
    occupancy_thread.start()
    dispatch_thread.start()
    try:
        while completed < n_jobs and not failed:
            spawn_available()
            dispatch_ready()

            now = time.monotonic()
            if now - last_interference_poll >= MONITOR_INTERVAL:
                last_interference_poll = now
                for item in active.values():
                    if item.state != "RUNNING_GPU" or not item.gpus:
                        continue
                    strangers = _active_strangers(item.gpus, _our_pids(), pool.util_threshold)
                    if strangers is None or strangers:
                        detail = (
                            "could not sample assigned GPU utilization"
                            if strangers is None
                            else f"intruders {sorted(strangers)}"
                        )
                        request_interference_stop(item, sorted(strangers or {}), detail)

            for item in active.values():
                if (
                    item.state == "STOPPING_INTERFERED_GPU"
                    and item.interference_stop_deadline is not None
                    and time.monotonic() >= item.interference_stop_deadline
                ):
                    fail(
                        item,
                        {
                            "phase": "interference",
                            "error": "GPU attempt did not acknowledge interruption within 30s",
                        },
                    )
                    break
            if failed:
                break

            sockets = [dispatch_reader, *(item.control for item in active.values())]
            readable, _, _ = select.select(sockets, [], [], 0.1)
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
                    fail(item, {"phase": "protocol", "error": f"invalid control message: {error}"})
                    break

                for message in messages:
                    message_type = message.get("type")
                    if message_type == "READY" and item.state == "PREPARING_CPU":
                        required = message.get("required_num_gpus")
                        declared = item.workload.get("num_gpus", 1)
                        if required != declared:
                            fail(
                                item,
                                {
                                    "phase": "prepare",
                                    "error": (
                                        f"READY requires {required!r} GPU(s), "
                                        f"workload declares {declared!r}"
                                    ),
                                },
                            )
                            break
                        item.ready_since = message.get("ready", time.time())
                        item.state = "READY"
                        ready.append(item)
                        log(f"[bench-suite] {now_iso()} READY {item.label}")
                    elif message_type == "RUNNING_GPU" and item.state == "ASSIGNED":
                        actual_uuids = message.get("physical_gpu_uuids")
                        expected_uuids = [physical_uuid_by_index[gpu] for gpu in item.gpus]
                        if actual_uuids != expected_uuids:
                            fail(
                                item,
                                {
                                    "phase": "assignment",
                                    "error": (
                                        f"physical GPU UUID mismatch: expected {expected_uuids}, "
                                        f"got {actual_uuids!r}"
                                    ),
                                },
                            )
                            break
                        item.physical_gpu_uuids = tuple(actual_uuids)
                        item.state = "RUNNING_GPU"
                        log(
                            f"[bench-suite] {now_iso()} gpus={','.join(item.gpus)} "
                            f"GPU_START {item.label} (attempt {item.attempt})"
                        )
                    elif message_type == "INTERFERED" and item.state == "STOPPING_INTERFERED_GPU":
                        acknowledge_interference(item, message)
                    elif message_type == "RESULT_READY" and item.state in (
                        "RUNNING_GPU",
                        "STOPPING_INTERFERED_GPU",
                    ):
                        if item.state == "STOPPING_INTERFERED_GPU":
                            acknowledge_interference(item, message)
                            _send_child(item, {"type": "RETRY_GPU"})
                            continue

                        _send_child(item, {"type": "ACCEPT_RESULT"})
                        release_gpus(item)
                        record = _base_attempt_record(item)
                        record["physical_gpu_uuids"] = list(item.physical_gpu_uuids)
                        result = message.get("result") or {}
                        status = result.get("status")
                        if status in ("SKIP", "FAIL"):
                            record.update(result)
                        else:
                            _finalize_bench_record(result, rounds=rounds, cooldown=cooldown)
                            record.update(result)
                        record.setdefault("label", item.workload["config"])
                        record["finished_at"] = now_iso()
                        item.state = "RESULT"
                        records.append(record)
                        completed += 1
                        impls = record.get("impls") or {}
                        impl_str = ", ".join(f"{name}={value:.3f}µs" for name, value in impls.items())
                        log(
                            f"[bench-suite] {record['finished_at']} "
                            f"gpus={record.get('gpu') or '-'} {record.get('status', 'ok'):4s} "
                            f"{item.label} {impl_str}"
                        )
                        if record.get("status") not in ("ok", "SKIP"):
                            failed = True
                    elif message_type == "FAIL":
                        fail(item, message)
                        break
                    else:
                        fail(
                            item,
                            {
                                "phase": "protocol",
                                "error": f"unexpected {message_type!r} message in state {item.state}",
                            },
                        )
                        break
                if failed:
                    break
                if eof and item.state not in ("RESULT", "FAILED"):
                    item.process.poll()
                    fail(item)
                    break

            for item in list(active.values()):
                if item.state == "RESULT" and item.process.poll() is not None:
                    fd = item.control.fileno()
                    _finish_attempt_process(item)
                    active.pop(fd, None)
    finally:
        occupancy_stop.set()
        dispatch_stop.set()
        pool.wake()
        occupancy_thread.join(timeout=1.0)
        dispatch_thread.join(timeout=1.0)
        dispatch_reader.close()
        dispatch_writer.close()
        for item in list(active.values()):
            fd = item.control.fileno()
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
            release_gpus(item)
            _finish_attempt_process(item)
            active.pop(fd, None)

    pipeline = {
        "execution_mode": "pipeline",
        "process_model": "one_shot_child_per_workload",
        "max_prepare_processes": max_prepare_processes,
        "ready_backlog": ready_backlog,
        "measurement_protocol": {
            "rounds": rounds,
            "cooldown_s": cooldown,
            "default_rounds": DEFAULT_ROUNDS,
            "default_cooldown_s": DEFAULT_COOLDOWN_S,
            "is_default": rounds == DEFAULT_ROUNDS
            and math.isclose(cooldown, DEFAULT_COOLDOWN_S, rel_tol=0.0, abs_tol=1e-9),
        },
        "interference_retry_count": len(retry_log),
        "interference_retries": retry_log,
    }
    return records, retry_log, pipeline


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
        "--all-configs",
        action="store_true",
        help="Assemble every config/**/*.yaml row, including default: false.",
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
        "--paired-old-checkout",
        type=Path,
        default=None,
        help="Run direct counterbalanced old/current A/B using this old checkout.",
    )
    ap.add_argument(
        "--paired-current-checkout",
        type=Path,
        default=None,
        help="Current checkout for paired A/B (default: this checkout).",
    )
    ap.add_argument(
        "--ir-builder-migration-gate",
        action="store_true",
        help="Run the formal IR Builder acceptance gate.",
    )
    ap.add_argument(
        "--speedup-threshold",
        type=float,
        default=DEFAULT_PAIRED_SPEEDUP_THRESHOLD,
        help="Strict per-config direct old/current speedup floor in percent.",
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
        help="Import every unique kernel in --workloads and exit (for CI import gates)",
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
        print("[bench-suite] --ready-backlog must be >= --max-prepare-processes", file=sys.stderr)
        sys.exit(2)

    if args.paired_current_checkout is not None and args.paired_old_checkout is None:
        print(
            "[bench-suite] --paired-current-checkout requires --paired-old-checkout",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        _validate_ir_builder_migration_gate_options(
            enabled=args.ir_builder_migration_gate,
            old_checkout=args.paired_old_checkout,
            workloads_path=args.workloads,
            all_configs=args.all_configs,
            kernel_filter=args.filter,
            rounds=args.rounds,
            cooldown=args.cooldown,
            speedup_threshold=args.speedup_threshold,
            util_threshold=args.util_threshold,
            mem_threshold=args.mem_threshold,
            check_imports=args.check_imports,
        )
    except ValueError as error:
        print(f"[bench-suite] invalid IR Builder migration gate: {error}", file=sys.stderr)
        sys.exit(2)

    migration_scope = None
    migration_required_workloads = None
    if args.ir_builder_migration_gate:
        migration_required_workloads, migration_scope = _derive_ir_builder_migration_scope(
            load_config_dir()
        )

    if args.workloads is None:
        if args.ir_builder_migration_gate:
            assembled = migration_required_workloads
            selection = (
                f"{migration_scope['required_count']} required single-GPU rows derived from "
                f"{migration_scope['canonical_default_count']} default:true rows; "
                f"{migration_scope['user_exempted_count']} user-exempted"
            )
        else:
            assembled = load_all_config_dir() if args.all_configs else load_config_dir()
            selection = (
                "every config entry; default flags retained but do not filter"
                if args.all_configs
                else "every config flagged default: true"
            )
        workloads_path = write_generated_workloads(
            assembled,
            args.out_dir.resolve() / GENERATED_WORKLOADS_NAME,
            selection=selection,
        )
        if args.ir_builder_migration_gate:
            print(
                f"[bench-suite] assembled {len(assembled)} required migration workload(s) from "
                f"{migration_scope['canonical_default_count']} canonical default:true rows; "
                f"{migration_scope['user_exempted_count']} user-exempted -> {workloads_path}"
            )
        else:
            print(
                f"[bench-suite] assembled {len(assembled)} "
                f"{'all-config' if args.all_configs else 'default'} workload(s) "
                f"from {CONFIG_DIR}/**/*.yaml -> {workloads_path}"
            )
    else:
        workloads_path = args.workloads

    workloads = load_workloads(workloads_path)
    if args.filter:
        workloads = [w for w in workloads if args.filter in w["kernel"]]
    if not workloads:
        print("[bench-suite] no workloads to run.", file=sys.stderr)
        sys.exit(2)
    if args.ir_builder_migration_gate:
        try:
            _validate_ir_builder_migration_workloads(workloads, migration_required_workloads)
        except ValueError as error:
            print(f"[bench-suite] invalid IR Builder migration coverage: {error}", file=sys.stderr)
            sys.exit(2)

    if args.check_imports:
        from tirx_kernels.registry import check_workload_imports

        names = check_workload_imports(workloads, strict=True)
        print(f"[bench-suite] import check ok ({len(names)} kernels from {workloads_path})")
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    stamp = _next_run_id(out_dir)
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

    environment_meta = {}
    if args.ir_builder_migration_gate:
        try:
            environment_meta["nvrtc"] = _nvrtc_preflight()
        except ValueError as error:
            print(f"[bench-suite] NVRTC preflight failed: {error}", file=sys.stderr)
            sys.exit(2)
        nvrtc = environment_meta["nvrtc"]
        print(
            f"[bench-suite] NVRTC preflight: version={nvrtc['version']} "
            f"path={nvrtc['library_path']} sha256={nvrtc['library_sha256'][:16]} "
            "applies-to=old,current",
            flush=True,
        )

    # ── Automatic GPU selection (no manual override on purpose) ──
    # 1. Startup probe: run a tiny fp16 matmul on every visible card
    #    (including busy ones — the probe is light, finishes fine on a
    #    contended card; this catches broken drivers / ECC). Probe failures
    #    are banned for the rest of the run.
    # 2. Per-workload acquire: re-scan utilization/memory every time we need a card.
    idle_memory_floor_mib = 0.0 if args.ir_builder_migration_gate else IDLE_GPU_MEMORY_FLOOR_MIB
    listing_pool = GpuPool(
        util_threshold=args.util_threshold,
        mem_threshold=args.mem_threshold,
        idle_memory_floor_mib=idle_memory_floor_mib,
    )
    in_filter = [idx for idx, _ in _visible_gpu_rows(listing_pool._all_gpus())]
    if not in_filter:
        print("[bench-suite] no visible GPUs.", file=sys.stderr)
        sys.exit(1)
    utils_now = listing_pool._utils()
    mem_now = listing_pool._mem_used_pct()
    occupied_now = sorted(listing_pool._occupied_indices() & set(in_filter), key=int)
    candidates = sorted(set(in_filter) - set(occupied_now), key=int)
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
        f"{candidates} "
        f"(resident-VRAM cards: {resident if resident else 'none'})",
        flush=True,
    )

    probe_candidates = candidates if args.ir_builder_migration_gate else in_filter
    if args.no_probe:
        usable = set(probe_candidates)
        probe_failures: dict[str, str] = {}
    else:
        print(
            f"[bench-suite] probing {len(probe_candidates)} GPU(s) with fp16 512x512 matmul ...",
            flush=True,
        )
        usable, probe_failures = detect_usable_gpus(probe_candidates, args.probe_timeout)

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
        allowed=usable,
        util_threshold=args.util_threshold,
        mem_threshold=args.mem_threshold,
        idle_memory_floor_mib=idle_memory_floor_mib,
    )
    n_gpus = len(usable)
    _repo_git = collect_repo_git()
    label = args.label or _repo_git.get("tirx-kernels") or _repo_git.get("tir") or "local"
    agg_note = (
        f", {args.rounds} standard-timer round(s), aggregate=mean, "
        f"cooldown={args.cooldown:g}s before every impl/round"
        if args.rounds > 1 or args.cooldown > 0
        else ""
    )
    old_checkout = args.paired_old_checkout.resolve() if args.paired_old_checkout else None
    current_checkout = (
        (args.paired_current_checkout or _kernels_repo_root()).resolve()
        if old_checkout is not None
        else None
    )
    checkout_snapshots = None
    config_params_sha256 = None
    if old_checkout is not None:
        for role, checkout in (("old", old_checkout), ("current", current_checkout)):
            try:
                _checkout_pythonpath(checkout, {})
            except ValueError as error:
                print(f"[bench-suite] invalid paired {role} checkout: {error}", file=sys.stderr)
                sys.exit(2)
        try:
            checkout_snapshots = {
                "old": _capture_checkout_snapshot(old_checkout),
                "current": _capture_checkout_snapshot(current_checkout),
            }
            old_configs = _resolve_checkout_bench_configs(old_checkout, workloads)
            current_configs = _resolve_checkout_bench_configs(current_checkout, workloads)
            config_params_sha256 = _validate_paired_config_resolution(
                workloads, old_configs, current_configs
            )
            for role, checkout in (("old", old_checkout), ("current", current_checkout)):
                _assert_checkout_unchanged(checkout, checkout_snapshots[role], full=True)
        except ValueError as error:
            print(f"[bench-suite] invalid paired checkout contract: {error}", file=sys.stderr)
            sys.exit(2)
        print(
            f"[bench-suite] {len(workloads)} workloads, {n_gpus} probe-OK GPU(s) in pool, "
            f"paired workers, label={label}{agg_note}",
            flush=True,
        )
        print(
            f"[bench-suite] paired direct gate: old={old_checkout} current={current_checkout} "
            f"speedup>{args.speedup_threshold:g}% config-params={config_params_sha256[:16]}",
            flush=True,
        )
        results, retry_log = _run_paired_scheduled_jobs(
            workloads,
            pool,
            log_dir,
            rounds=args.rounds,
            cooldown=args.cooldown,
            old_checkout=old_checkout,
            current_checkout=current_checkout,
            speedup_threshold=args.speedup_threshold,
            checkout_snapshots=checkout_snapshots,
            config_params_sha256=config_params_sha256,
        )
        pipeline_meta = {}
    else:
        compile_profile = gpu_compile_profile(usable)
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

    if old_checkout is not None:
        try:
            for role, checkout in (("old", old_checkout), ("current", current_checkout)):
                _assert_checkout_unchanged(checkout, checkout_snapshots[role], full=True)
        except ValueError as error:
            print(f"[bench-suite] paired checkout changed: {error}", file=sys.stderr)
            sys.exit(1)

    if retry_log:
        log(f"[bench-suite] interference retry summary: {len(retry_log)} attempt(s)")
        for retry in retry_log:
            if isinstance(retry, dict):
                kernel, config = retry["kernel"], retry["config"]
                attempt, detail = retry["attempt"], retry["detail"]
            else:
                kernel, config, attempt, detail = retry
            log(f"[bench-suite]   - {kernel}/{config}: attempt {attempt} → {detail}")
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
        environment=environment_meta,
        scope=migration_scope,
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

    if old_checkout is not None:
        expected = {(workload["kernel"], workload["config"]) for workload in workloads}
        actual = {(row["kernel"], row.get("config") or row.get("label")) for row in results}
        if actual != expected or len(results) != len(workloads):
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            print(
                f"[bench-suite] paired coverage mismatch: missing={missing[:10]} "
                f"extra={extra[:10]} rows={len(results)}/{len(workloads)}",
                file=sys.stderr,
            )
            sys.exit(3)
        regressions = [row for row in results if row.get("status") == "REGRESSION"]
        if regressions:
            worst = min(regressions, key=lambda row: row["paired"]["speedup_pct"])
            print(
                f"[bench-suite] paired direct gate failed for {len(regressions)}/"
                f"{len(results)} workload(s); worst={worst['kernel']}/"
                f"{worst.get('config') or worst.get('label')} "
                f"{worst['paired']['speedup_pct']:.3f}%",
                file=sys.stderr,
            )
            sys.exit(3)
        print(
            f"[bench-suite] paired direct gate passed for all {len(results)} workload(s): "
            f"speedup > {args.speedup_threshold:g}%",
            flush=True,
        )
        return

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
