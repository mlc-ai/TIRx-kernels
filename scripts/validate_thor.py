#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Run the registered SM100 correctness matrix on a single NVIDIA Thor.

The parent process inventories every registry config and launches each selected
case in an isolated child.  Isolation keeps a failed CUDA context from
contaminating later cases, while JSONL output makes long sweeps resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest import SkipTest

from tirx_kernels.registry import discover_kernels, load_kernel


def _config_num_gpus(config: dict) -> int:
    topology = [config[key] for key in ("world_size", "num_processes") if key in config]
    if len(topology) == 2 and topology[0] != topology[1]:
        raise ValueError(
            f"conflicting world_size={topology[0]!r} and num_processes={topology[1]!r}"
        )
    value = topology[0] if topology else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"invalid config GPU count: {value!r}")
    return value


def _worker(kernel_name: str, label: str, *, launch_only: bool) -> int:
    started = time.monotonic()
    result = {"kernel": kernel_name, "config": label}
    try:
        module = load_kernel(kernel_name, strict=True)
        matches = [config for config in module.CONFIGS if config.get("label", "default") == label]
        if len(matches) != 1:
            raise ValueError(f"expected one config named {label!r}, found {len(matches)}")
        params = {key: value for key, value in matches[0].items() if key != "label"}
        if launch_only:
            if not hasattr(module, "prepare_bench"):
                raise SkipTest("kernel has no prepare_bench launch contract")
            prepared = module.prepare_bench(**params)
            from tirx_kernels.runner import bind_cuda_assignment, physical_cuda_uuids

            required_gpus = int(getattr(prepared, "required_num_gpus", 1))
            device_indices = tuple(range(required_gpus))
            try:
                bind_cuda_assignment(device_indices, physical_cuda_uuids(device_indices))
                prepared.run_gpu(warmup=1, repeat=1, timer="event", rounds=1, cooldown_s=0.0)
            finally:
                close = getattr(prepared, "close", None)
                if close is not None:
                    close()
            result.update(status="LAUNCH_ONLY", validation="compile-and-launch without oracle")
        else:
            module.run_test(**params)
            result.update(status="PASS", validation="correctness")
    except SkipTest as error:
        result.update(status="SKIP", reason=str(error))
    except Exception as error:
        result.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
    result["duration_s"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] != "FAIL" else 1


def _load_completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            result = json.loads(line)
            completed.add((result["kernel"], result["config"]))
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return completed


def _append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, sort_keys=True) + "\n")


def _cases(*, smoke: bool, kernel_filter: set[str] | None):
    kernels = discover_kernels(strict=True)
    for kernel_name, module in sorted(kernels.items()):
        if "sm_100a" not in module.KERNEL_META["runtime_cuda_archs"]:
            continue
        if kernel_filter is not None and kernel_name not in kernel_filter:
            continue
        configs = list(module.CONFIGS)
        if smoke:
            config = next(
                (config for config in configs if _config_num_gpus(config) == 1), configs[0]
            )
            configs = [config]
        for config in configs:
            yield kernel_name, config.get("label", "default"), _config_num_gpus(config)


def _parent(args: argparse.Namespace) -> int:
    args.results = args.results.resolve()
    completed = _load_completed(args.results) if args.resume else set()
    kernel_filter = set(args.kernel) if args.kernel else None
    counts = {"PASS": 0, "LAUNCH_ONLY": 0, "FAIL": 0, "SKIP": 0, "TIMEOUT": 0}
    selected = list(_cases(smoke=args.smoke, kernel_filter=kernel_filter))
    print(
        f"selected={len(selected)} completed={len(completed)} "
        f"results={args.results}",
        flush=True,
    )

    child_env = os.environ.copy()
    child_env.setdefault("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    child_env.setdefault("TIRX_PREPARE_NUM_SMS", "20")
    # NVRTC emits both .maxntid and .reqntid for some large SM100 kernels when
    # retargeted to sm_110a.  CUDA 13.1 nvcc compiles and runs the same TIRx IR,
    # so make that tested backend the Thor default while retaining an explicit
    # caller override.
    if child_env["TIRX_PREPARE_CUDA_ARCH"] == "sm_110a":
        child_env.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")
    child_env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + child_env.get("PATH", "")
    for index, (kernel_name, label, required_gpus) in enumerate(selected, 1):
        key = (kernel_name, label)
        if key in completed:
            continue
        if required_gpus > args.available_gpus:
            result = {
                "kernel": kernel_name,
                "config": label,
                "status": "SKIP",
                "reason": f"requires {required_gpus} GPUs, only {args.available_gpus} available",
                "duration_s": 0.0,
            }
        else:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                kernel_name,
                label,
            ]
            if args.launch_only:
                command.append("--launch-only")
            started = time.monotonic()
            try:
                process = subprocess.run(
                    command,
                    env=child_env,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout,
                    check=False,
                )
                output_lines = [line for line in process.stdout.splitlines() if line.strip()]
                result = json.loads(output_lines[-1])
                if process.returncode < 0:
                    result.update(
                        status="FAIL", error=f"worker terminated by signal {-process.returncode}"
                    )
                if process.stderr:
                    result["stderr"] = process.stderr
            except subprocess.TimeoutExpired as error:
                result = {
                    "kernel": kernel_name,
                    "config": label,
                    "status": "TIMEOUT",
                    "error": f"exceeded {args.timeout}s",
                    "stdout": error.stdout or "",
                    "stderr": error.stderr or "",
                    "duration_s": round(time.monotonic() - started, 3),
                }
            except (IndexError, json.JSONDecodeError) as error:
                result = {
                    "kernel": kernel_name,
                    "config": label,
                    "status": "FAIL",
                    "error": f"worker returned no valid JSON: {error}",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "duration_s": round(time.monotonic() - started, 3),
                }
        _append_result(args.results, result)
        counts[result["status"]] += 1
        print(
            f"[{index}/{len(selected)}] {result['status']:7} "
            f"{kernel_name} [{label}] ({result['duration_s']}s)",
            flush=True,
        )
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 1 if counts["FAIL"] or counts["TIMEOUT"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the first single-GPU config")
    parser.add_argument("--kernel", action="append", help="restrict to a registry kernel name")
    parser.add_argument("--results", type=Path, default=Path("thor-validation.jsonl"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=900, help="per-config timeout in seconds")
    parser.add_argument("--available-gpus", type=int, default=1)
    parser.add_argument(
        "--launch-only",
        action="store_true",
        help="compile and launch without a correctness oracle; reports LAUNCH_ONLY, never PASS",
    )
    parser.add_argument("--worker", nargs=2, metavar=("KERNEL", "CONFIG"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(*args.worker, launch_only=args.launch_only)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
