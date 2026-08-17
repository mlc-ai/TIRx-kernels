# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import fcntl
import json
import math
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


def _visible_gpu_memory() -> list[tuple[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, uuid, free_memory = (field.strip() for field in line.split(","))
        rows.append((index, uuid, int(free_memory)))

    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        visible = {index for index, _uuid, _free_memory in rows}
    else:
        visible = {token.strip() for token in configured.split(",") if token.strip()}
    return [
        (uuid if uuid in visible else index, free_memory)
        for index, uuid, free_memory in rows
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
    worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))

    selected: list[str] = []
    slot_handles = []
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
            # Any four visible GPUs can absorb all xdist workers, while free
            # cards remain eligible to take work from cards occupied externally.
            active_gpu_target = min(4, len(gpu_memory))
            slots_per_gpu = math.ceil(worker_count / active_gpu_target)
            candidates = []
            for gpu, free_memory in gpu_memory:
                active_slots = 0
                candidate_handle = None
                for slot in range(slots_per_gpu):
                    slot_handle = _try_lock(lock_root / f"gpu-{gpu}-slot-{slot}.lock")
                    if slot_handle is None:
                        active_slots += 1
                    elif candidate_handle is None:
                        candidate_handle = slot_handle
                    else:
                        slot_handle.close()
                if candidate_handle is not None:
                    candidates.append(
                        (free_memory // (active_slots + 1), free_memory, gpu, candidate_handle)
                    )

            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if len(candidates) >= count:
                for _score, _free_memory, gpu, slot_handle in candidates[:count]:
                    selected.append(gpu)
                    slot_handles.append(slot_handle)
                candidates = candidates[count:]
            for _score, _free_memory, _gpu, slot_handle in candidates:
                slot_handle.close()
        if not selected:
            time.sleep(0.1)

    try:
        yield selected
    finally:
        for slot_handle in slot_handles:
            slot_handle.close()


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
    assert completed.returncode == 0, completed.stdout[-12000:] + completed.stderr[-12000:]
    assert payload["passed"] == 1, payload
    assert payload["failed"] == 0, payload
    assert payload["skipped"] == 0, payload
