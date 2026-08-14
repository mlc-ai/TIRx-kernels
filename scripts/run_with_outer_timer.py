#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Run one command and persist its complete-command wall-time provenance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CACHE_ENV_KEYS = (
    "TMPDIR",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUTE_DSL_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "TRTLLM_DG_CACHE_DIR",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _nvidia_smi_rows(arguments: list[str]) -> list[list[str]]:
    completed = subprocess.run(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(completed.stdout.splitlines())
        if row
    ]


def _descendant_pids(root_pid: int | None) -> set[int]:
    if root_pid is None:
        return set()
    children: dict[int, set[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parent_line = next(
                line
                for line in (entry / "status").read_text().splitlines()
                if line.startswith("PPid:")
            )
            parent = int(parent_line.split()[1])
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
        children.setdefault(parent, set()).add(int(entry.name))
    descendants = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, ()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _all_gpu_snapshots(suite_pids: set[int] | None = None) -> list[dict]:
    gpu_rows = _nvidia_smi_rows(
        ["--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total"]
    )
    process_rows = _nvidia_smi_rows(
        ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory"]
    )
    processes_by_uuid: dict[str, list[dict]] = {}
    for process in process_rows:
        if len(process) != 4:
            continue
        parsed = {
            "gpu_uuid": process[0],
            "pid": int(process[1]),
            "process_name": process[2],
            "used_memory_mib": float(process[3]),
        }
        if suite_pids is not None:
            parsed["owner"] = "suite" if parsed["pid"] in suite_pids else "foreign"
        processes_by_uuid.setdefault(process[0], []).append(parsed)
    snapshots = []
    for row in gpu_rows:
        if len(row) != 6:
            raise RuntimeError(f"malformed physical GPU row: {row!r}")
        processes = processes_by_uuid.get(row[1], [])
        snapshot = {
            "index": row[0],
            "uuid": row[1],
            "name": row[2],
            "utilization_gpu_percent": float(row[3]),
            "memory_used_mib": float(row[4]),
            "memory_total_mib": float(row[5]),
            "compute_processes": processes,
        }
        if suite_pids is not None:
            snapshot["suite_compute_pids"] = sorted(
                process["pid"] for process in processes if process["owner"] == "suite"
            )
            snapshot["foreign_compute_pids"] = sorted(
                process["pid"] for process in processes if process["owner"] == "foreign"
            )
        snapshots.append(snapshot)
    return snapshots


def _gpu_snapshot(index: str) -> dict:
    matches = [row for row in _all_gpu_snapshots() if row["index"] == index]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU index {index!r} resolved to {matches!r}")
    return matches[0]


def _all_gpu_timeline_sample(root_pid: int | None) -> dict:
    sampled_ns = time.monotonic_ns()
    suite_pids = _descendant_pids(root_pid)
    return {
        "sampled_utc": _utc_now(),
        "sampled_monotonic_ns": sampled_ns,
        "suite_process_tree_pids": sorted(suite_pids),
        "gpus": _all_gpu_snapshots(suite_pids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and persist an independent outer-wall timer artifact"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument(
        "--physical-gpu-index",
        help="Preflight one physical GPU and persist its UUID before and after; does not mask it",
    )
    parser.add_argument(
        "--max-preflight-gpu-utilization",
        type=float,
        default=0.0,
        help="Reject the command when the selected GPU exceeds this utilization (default: 0)",
    )
    parser.add_argument(
        "--max-preflight-memory-used-mib",
        type=float,
        default=512.0,
        help="Reject the command when the selected GPU exceeds this memory use (default: 512)",
    )
    parser.add_argument(
        "--monitor-all-gpus-interval",
        type=float,
        default=None,
        help=(
            "Non-rejecting all-GPU occupancy sampling interval in seconds. Persists "
            "GPU UUIDs plus suite/foreign compute PIDs for supplemental sweep evidence."
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    artifact = args.artifact.resolve()
    stdout_log = args.stdout_log.resolve()
    stderr_log = args.stderr_log.resolve()
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    wrapper_started_utc = _utc_now()
    wrapper_started_ns = time.monotonic_ns()
    payload = {
        "schema_version": 1,
        "status": "running",
        "cwd": os.getcwd(),
        "argv": command,
        "wrapper_started_utc": wrapper_started_utc,
        "wrapper_started_monotonic_ns": wrapper_started_ns,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "cuda_visible_devices_at_wrapper_start": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cache_environment": {key: os.environ.get(key) for key in CACHE_ENV_KEYS},
    }
    if args.monitor_all_gpus_interval is not None and args.monitor_all_gpus_interval <= 0:
        parser.error("--monitor-all-gpus-interval must be positive")
    child_env = os.environ.copy()
    if args.physical_gpu_index is not None:
        if not args.physical_gpu_index.isdigit():
            parser.error("--physical-gpu-index must be a non-negative integer")
        if args.max_preflight_gpu_utilization < 0 or args.max_preflight_memory_used_mib < 0:
            parser.error("GPU preflight thresholds must be non-negative")
        try:
            before_gpu = _gpu_snapshot(args.physical_gpu_index)
        except BaseException as error:
            payload.update(
                {
                    "status": "preflight_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            _write_json(artifact, payload)
            return 75
        payload["physical_gpu"] = {
            "requested_index": args.physical_gpu_index,
            "before": before_gpu,
            "preflight_limits": {
                "utilization_gpu_percent": args.max_preflight_gpu_utilization,
                "memory_used_mib": args.max_preflight_memory_used_mib,
            },
        }
        violations = []
        if before_gpu["utilization_gpu_percent"] > args.max_preflight_gpu_utilization:
            violations.append(
                f"utilization {before_gpu['utilization_gpu_percent']}% exceeds "
                f"{args.max_preflight_gpu_utilization}%"
            )
        if before_gpu["memory_used_mib"] > args.max_preflight_memory_used_mib:
            violations.append(
                f"memory {before_gpu['memory_used_mib']} MiB exceeds "
                f"{args.max_preflight_memory_used_mib} MiB"
            )
        if before_gpu["compute_processes"]:
            violations.append("compute processes are already present")
        if violations:
            payload.update(
                {
                    "status": "preflight_rejected",
                    "error": "; ".join(violations),
                }
            )
            _write_json(artifact, payload)
            return 75
    _write_json(artifact, payload)

    child: subprocess.Popen | None = None
    monitor_stop = threading.Event()
    monitor_samples: list[dict] = []
    monitor_errors: list[dict] = []
    monitor_thread: threading.Thread | None = None

    def sample_all_gpus() -> None:
        try:
            monitor_samples.append(
                _all_gpu_timeline_sample(child.pid if child is not None else None)
            )
        except BaseException as error:
            monitor_errors.append(
                {
                    "sampled_utc": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    def monitor_all_gpus() -> None:
        while not monitor_stop.wait(args.monitor_all_gpus_interval):
            sample_all_gpus()

    def forward(signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    previous_handlers = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        if args.monitor_all_gpus_interval is not None:
            sample_all_gpus()
        with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
            command_started_utc = _utc_now()
            command_started_ns = time.monotonic_ns()
            child = subprocess.Popen(
                command,
                env=child_env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            if args.monitor_all_gpus_interval is not None:
                sample_all_gpus()
                monitor_thread = threading.Thread(
                    target=monitor_all_gpus,
                    name="outer-gpu-monitor",
                    daemon=True,
                )
                monitor_thread.start()
            returncode = child.wait()
            command_finished_ns = time.monotonic_ns()
            command_finished_utc = _utc_now()
    except BaseException as error:
        wrapper_finished_ns = time.monotonic_ns()
        payload.update(
            {
                "status": "wrapper_error",
                "wrapper_finished_utc": _utc_now(),
                "wrapper_finished_monotonic_ns": wrapper_finished_ns,
                "wrapper_wall_ns": wrapper_finished_ns - wrapper_started_ns,
                "wrapper_wall_s": (wrapper_finished_ns - wrapper_started_ns)
                / 1_000_000_000,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(artifact, payload)
        raise
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=15)
        if args.monitor_all_gpus_interval is not None:
            sample_all_gpus()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    wrapper_returncode = returncode
    if args.physical_gpu_index is not None:
        try:
            after_gpu = _gpu_snapshot(args.physical_gpu_index)
            payload["physical_gpu"]["after"] = after_gpu
            same_uuid = after_gpu["uuid"] == payload["physical_gpu"]["before"]["uuid"]
            payload["physical_gpu"]["same_uuid_before_after"] = same_uuid
            if not same_uuid:
                payload["status"] = "postflight_identity_mismatch"
                wrapper_returncode = returncode or 76
        except BaseException as error:
            payload["status"] = "postflight_error"
            payload["postflight_error"] = f"{type(error).__name__}: {error}"
            wrapper_returncode = returncode or 76
    final_status = payload.get("status")
    if final_status == "running":
        final_status = "completed"
    wrapper_finished_ns = time.monotonic_ns()
    payload.update(
        {
            "status": final_status or "completed",
            "command_started_utc": command_started_utc,
            "command_started_monotonic_ns": command_started_ns,
            "command_finished_utc": command_finished_utc,
            "command_finished_monotonic_ns": command_finished_ns,
            "command_wall_ns": command_finished_ns - command_started_ns,
            "command_wall_s": (command_finished_ns - command_started_ns)
            / 1_000_000_000,
            "wrapper_finished_utc": _utc_now(),
            "wrapper_finished_monotonic_ns": wrapper_finished_ns,
            "wrapper_wall_ns": wrapper_finished_ns - wrapper_started_ns,
            "wrapper_wall_s": (wrapper_finished_ns - wrapper_started_ns)
            / 1_000_000_000,
            "returncode": returncode,
            "wrapper_returncode": wrapper_returncode,
        }
    )
    if args.monitor_all_gpus_interval is not None:
        payload["all_gpu_monitor"] = {
            "schema_version": 1,
            "interval_s": args.monitor_all_gpus_interval,
            "samples": monitor_samples,
            "errors": monitor_errors,
        }
    _write_json(artifact, payload)
    return wrapper_returncode


if __name__ == "__main__":
    sys.exit(main())
