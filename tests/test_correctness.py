# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tirx_kernels.registry import discover_kernels


def _correctness_cases() -> list[Any]:
    cases = []
    for kernel_name, module in sorted(discover_kernels().items()):
        labels = [config.get("label", "default") for config in module.CONFIGS]
        if len(labels) != len(set(labels)):
            raise RuntimeError(f"correctness config labels are not unique for {kernel_name}")
        for config, label in zip(module.CONFIGS, labels, strict=True):
            required_devices = int(
                config.get("num_processes", config.get("world_size", 1))
            )
            cases.append(
                pytest.param(
                    kernel_name,
                    label,
                    required_devices,
                    id=f"{kernel_name}[{label}]",
                )
            )
    return cases


def _visible_gpu_memory() -> list[tuple[str, int, bool]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    compute_processes = {
        line.strip()
        for line in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        if line.strip()
    }
    rows = []
    for line in output.splitlines():
        index, uuid, free_memory = (field.strip() for field in line.split(","))
        rows.append((index, uuid, int(free_memory), uuid in compute_processes))

    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        visible = {index for index, _uuid, _free_memory in rows}
    else:
        visible = {token.strip() for token in configured.split(",") if token.strip()}
    return [
        (uuid if uuid in visible else index, free_memory, has_compute_process)
        for index, uuid, free_memory, has_compute_process in rows
        if index in visible or uuid in visible
    ]


def _try_lock(path: Path):
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@contextmanager
def _reserve_gpus(test_run_uid: str, count: int):
    lock_root = Path("/tmp") / f"tirx-correctness-{test_run_uid}"
    lock_root.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    gpu_handles = []
    while not selected:
        with (lock_root / "allocator.lock").open("a+") as allocator:
            fcntl.flock(allocator, fcntl.LOCK_EX)
            gpu_memory = sorted(_visible_gpu_memory(), key=lambda item: item[1], reverse=True)
            if not gpu_memory:
                raise RuntimeError("no CUDA GPU is visible to correctness tests")
            if count > len(gpu_memory):
                raise RuntimeError(
                    f"correctness case requires {count} GPUs, "
                    f"but only {len(gpu_memory)} are visible"
                )
            # Kernel resource use has no divisible per-card contract: one case
            # may consume all memory, SMs, or TMEM. Own each assigned GPU as a
            # unit and reject externally occupied cards.
            candidates = []
            for gpu, free_memory, has_compute_process in gpu_memory:
                gpu_handle = _try_lock(lock_root / f"gpu-{gpu}.lock")
                if gpu_handle is None:
                    continue
                if has_compute_process:
                    gpu_handle.close()
                    continue
                candidates.append((free_memory, gpu, gpu_handle))

            candidates.sort(key=lambda item: item[0], reverse=True)
            if len(candidates) >= count:
                for _free_memory, gpu, gpu_handle in candidates[:count]:
                    selected.append(gpu)
                    gpu_handles.append(gpu_handle)
                candidates = candidates[count:]
            for _free_memory, _gpu, gpu_handle in candidates:
                gpu_handle.close()
        if not selected:
            time.sleep(0.1)

    try:
        yield selected
    finally:
        for gpu_handle in gpu_handles:
            gpu_handle.close()


def _last_json_object(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for offset in range(len(output) - 1, -1, -1):
        if output[offset] != "{":
            continue
        try:
            payload, end = decoder.raw_decode(output[offset:])
        except json.JSONDecodeError:
            continue
        if not output[offset + end :].strip() and isinstance(payload, dict):
            return payload
    raise AssertionError(f"correctness child did not emit JSON:\n{output[-12000:]}")


@pytest.mark.parametrize(
    ("kernel_name", "config_label", "required_devices"), _correctness_cases()
)
def test_kernel_correctness(
    kernel_name: str, config_label: str, required_devices: int, testrun_uid: str
) -> None:
    with _reserve_gpus(testrun_uid, required_devices) as selected_gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tirx_kernels.test",
                "--json",
                "--kernel",
                kernel_name,
                "--config",
                config_label,
            ],
            env=env,
            text=True,
            capture_output=True,
        )

    payload = _last_json_object(completed.stdout)
    diagnostic = (
        f"assigned GPUs: {','.join(selected_gpus)}\n"
        + completed.stdout[-12000:]
        + completed.stderr[-12000:]
    )
    assert completed.returncode == 0, diagnostic
    assert payload["passed"] == 1, payload
    assert payload["failed"] == 0, payload
    assert payload["skipped"] == 0, payload
