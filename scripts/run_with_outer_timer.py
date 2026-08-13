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
import time
from datetime import datetime, timezone
from pathlib import Path


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


def _gpu_snapshot(index: str) -> dict:
    gpu_rows = _nvidia_smi_rows(
        ["--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total"]
    )
    matches = [row for row in gpu_rows if row[0] == index]
    if len(matches) != 1 or len(matches[0]) != 6:
        raise RuntimeError(f"physical GPU index {index!r} resolved to {matches!r}")
    row = matches[0]
    uuid = row[1]
    process_rows = _nvidia_smi_rows(
        ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory"]
    )
    processes = [
        {
            "gpu_uuid": process[0],
            "pid": int(process[1]),
            "process_name": process[2],
            "used_memory_mib": float(process[3]),
        }
        for process in process_rows
        if len(process) == 4 and process[0] == uuid
    ]
    return {
        "index": index,
        "uuid": uuid,
        "name": row[2],
        "utilization_gpu_percent": float(row[3]),
        "memory_used_mib": float(row[4]),
        "memory_total_mib": float(row[5]),
        "compute_processes": processes,
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
    }
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

    def forward(signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    previous_handlers = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
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
    _write_json(artifact, payload)
    return wrapper_returncode


if __name__ == "__main__":
    sys.exit(main())
