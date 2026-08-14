#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Record whether candidate READY-time imports preserve an uninitialized CUDA state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tirx_kernels.runner import cuda_initialization_guard, cuda_is_initialized  # noqa: E402


def _flashinfer_fp4_jit() -> dict:
    from flashinfer.jit.gemm import gen_gemm_sm100_module_cutlass_fp4

    spec = gen_gemm_sm100_module_cutlass_fp4()
    return {"jit_spec": spec.name, "library_path": str(spec.get_library_path())}


def _flashinfer_selective_state() -> dict:
    from flashinfer.mamba import selective_state_update

    return {"callable": selective_state_update.__name__}


def _flashinfer_trtllm_decode() -> dict:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    return {"callable": trtllm_batch_decode_with_kv_cache_mla.__name__}


def _flashkda_peer() -> dict:
    from tirx_kernels.flashinfer.utils._flashkda_bench import _load_flash_kda_peer

    _module, provenance = _load_flash_kda_peer()
    return {
        key: provenance.get(key) for key in ("package_version", "source_commit", "cutlass_commit")
    }


def _deepgemm_mega() -> dict:
    from tirx_kernels.deepgemm.mega_moe import load_deep_gemm_mega

    _module, source = load_deep_gemm_mega()
    return {"source": source}


PROBES: dict[str, Callable[[], dict]] = {
    "flashinfer_fp4_jit": _flashinfer_fp4_jit,
    "flashinfer_selective_state": _flashinfer_selective_state,
    "flashinfer_trtllm_decode": _flashinfer_trtllm_decode,
    "flashkda_peer": _flashkda_peer,
    "deepgemm_mega": _deepgemm_mega,
}


def _run_child(name: str) -> int:
    before = cuda_is_initialized()
    started = time.monotonic()
    details = None
    error = None
    try:
        with cuda_initialization_guard(require_uninitialized=True):
            details = PROBES[name]()
    except Exception as exc:  # The guard rejection is the evidence being collected.
        error = f"{type(exc).__name__}: {exc}"
    after = cuda_is_initialized()
    print(
        json.dumps(
            {
                "probe": name,
                "cuda_initialized_before": before,
                "cuda_initialized_after": after,
                "guard_result": "passed" if error is None else "rejected",
                "elapsed_s": time.monotonic() - started,
                "details": details,
                "error": error,
            },
            sort_keys=True,
        )
    )
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=sorted(PROBES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        return _run_child(args.child)
    if args.output is None:
        parser.error("--output is required unless --child is used")

    environment = os.environ.copy()
    environment.setdefault("FLASHINFER_CUDA_ARCH_LIST", "10.0a")
    results = []
    for name in PROBES:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", name],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        results.append(json.loads(completed.stdout))
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "measurement_kind": "cpu_prepare_cuda_initialization_guard",
            "uses_gpu_measurement": False,
            "environment": {
                key: environment.get(key)
                for key in ("FLASHINFER_CUDA_ARCH_LIST", "PYTHONPATH", "LD_LIBRARY_PATH")
            },
            "results": results,
        },
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
