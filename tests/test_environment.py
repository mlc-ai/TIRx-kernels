# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tirx_kernels.environment import (
    BEFORE_KERNEL_ROOT,
    FORBIDDEN_KERNEL_ROOT,
    KERNEL_SHA,
    MAIN_KERNEL_ROOT,
    TVM_FFI_ROOT,
    TVM_LIBRARY_ROOT,
    TVM_PYTHON_ROOT,
    TVM_ROOT,
    TVM_SHA,
    make_environment_report,
    resolve_kernel_root,
    validate_environment_snapshot,
)

_DIGEST = "a" * 64
_DEPENDENCY_NAMES = (
    "torch",
    "deep_gemm",
    "flashinfer",
    "flash_attn",
    "flash_kda",
    "sglang",
    "flash_mla",
    "cutlass",
    "nccl",
)
_LOCK_NAMES = (
    "TIRX_NCCL_LIBRARY",
    "TIRX_CUBLAS_LIBRARY",
    "TIRX_CUBLASMP_LIBRARY",
    "TIRX_NVSHMEM_LIBRARY",
)


def _library(path: Path) -> dict:
    return {"path": str(path), "size": 1, "sha256": _DIGEST, "error": None}


def _repository(root: Path, sha: str) -> dict:
    return {
        "root": str(root),
        "sha": sha,
        "expected_sha": sha,
        "revision_error": None,
        "submodules": [],
        "submodule_error": None,
    }


def _valid_snapshot() -> dict:
    return {
        "schema_version": 1,
        "captured_at": "2026-08-07T00:00:00+00:00",
        "host": {"hostname": "test-host", "platform": "test-platform"},
        "python": {
            "executable": "/usr/bin/python3",
            "version": "3.13.0",
            "implementation": "CPython",
        },
        "repositories": {
            "kernel": _repository(MAIN_KERNEL_ROOT, KERNEL_SHA),
            "tvm": _repository(TVM_ROOT, TVM_SHA),
        },
        "tvm": {
            "root": str(TVM_ROOT),
            "python_root": str(TVM_PYTHON_ROOT),
            "ffi_root": str(TVM_FFI_ROOT),
            "library_root": str(TVM_LIBRARY_ROOT),
            "python_search_path": {
                "raw": os.pathsep.join((str(TVM_PYTHON_ROOT), str(TVM_FFI_ROOT))),
                "paths": [str(TVM_PYTHON_ROOT), str(TVM_FFI_ROOT)],
            },
            "library_search_path": {"raw": str(TVM_LIBRARY_ROOT), "paths": [str(TVM_LIBRARY_ROOT)]},
            "forbidden_source_paths": [],
            "module_path": str(TVM_PYTHON_ROOT / "tvm" / "__init__.py"),
            "module_version": "0.26.dev0",
            "ffi_module_path": str(TVM_FFI_ROOT / "tvm_ffi" / "__init__.py"),
            "ffi_core_path": str(TVM_FFI_ROOT / "tvm_ffi" / "core.cpython.so"),
            "ffi_version": "0.1.13.post2",
            "import_error": None,
            "loaded_libraries": {
                "source": "/proc/self/maps",
                "error": None,
                "tvm": [
                    _library(TVM_LIBRARY_ROOT / "libtvm_compiler.so"),
                    _library(TVM_LIBRARY_ROOT / "libtvm_runtime.so"),
                ],
                "ffi": [
                    _library(TVM_FFI_ROOT / "tvm_ffi" / "core.cpython.so"),
                    _library(TVM_FFI_ROOT / "tvm_ffi" / "lib" / "libtvm_ffi.so"),
                ],
            },
        },
        "cuda": {
            "nvidia_smi": {"executable": None, "returncode": None, "error": "unavailable"},
            "gpus": [],
            "nvcc": {
                "executable": None,
                "returncode": None,
                "release": None,
                "build": None,
                "output": None,
            },
        },
        "dependencies": {
            name: {
                "module": name,
                "installed": True,
                "module_path": f"/opt/python/{name}/__init__.py",
                "distribution": name,
                "version": "1.0",
                "error": None,
            }
            for name in _DEPENDENCY_NAMES
        },
        "runtime": {
            "library_locks": {
                name: {"set": False, "path": None, "exists": False, "sha256": None, "error": None}
                for name in _LOCK_NAMES
            },
            "nvshmem_home": {
                "set": False,
                "path": None,
                "exists": False,
                "sha256": None,
                "error": None,
            },
            "optional_source_paths": {
                name: {"set": False, "path": None, "exists": False, "sha256": None, "error": None}
                for name in ("FLASHKDA_PR_WORKTREE", "FLASHKDA_SOURCE_DIR", "FLASH_MLA_PATH")
            },
        },
    }


def test_authorized_kernel_root_aliases_resolve_to_the_fixed_worktrees() -> None:
    assert resolve_kernel_root("main") == MAIN_KERNEL_ROOT.resolve()
    assert resolve_kernel_root("before") == BEFORE_KERNEL_ROOT.resolve()

    with pytest.raises(ValueError, match="not an allowed kernel root"):
        resolve_kernel_root("/tmp/not-the-kernel-repository")
    with pytest.raises(ValueError, match="refused forbidden kernel root"):
        resolve_kernel_root(str(FORBIDDEN_KERNEL_ROOT))


def test_complete_fixed_snapshot_is_accepted_without_optional_host_capabilities() -> None:
    snapshot = _valid_snapshot()
    snapshot["tvm"]["loaded_libraries"]["tvm"].append(
        _library(TVM_LIBRARY_ROOT / "libtvm_ffi_testing.so")
    )

    assert validate_environment_snapshot(snapshot) == ()
    report = make_environment_report(snapshot)
    assert report["ok"] is True
    assert report["issues"] == []


def test_validation_rejects_wrong_revisions_and_nonisolated_tvm_sources() -> None:
    snapshot = _valid_snapshot()
    snapshot["repositories"]["kernel"]["sha"] = "0" * 40
    snapshot["tvm"]["module_path"] = "/home/hongyij/tir/python/tvm/__init__.py"
    snapshot["tvm"]["ffi_module_path"] = "/opt/site-packages/tvm_ffi/__init__.py"
    snapshot["tvm"]["python_search_path"]["paths"].reverse()
    snapshot["tvm"]["loaded_libraries"]["ffi"][1]["path"] = (
        "/opt/site-packages/tvm_ffi/lib/libtvm_ffi.so"
    )

    issues = validate_environment_snapshot(snapshot)

    assert any(KERNEL_SHA in issue for issue in issues)
    assert any(str(TVM_PYTHON_ROOT) in issue for issue in issues)
    assert any("isolated target" in issue for issue in issues)
    assert any("in that order" in issue for issue in issues)
    assert any("loaded outside isolated target" in issue for issue in issues)


def test_validation_rejects_forbidden_kernel_paths_in_nested_provenance() -> None:
    snapshot = _valid_snapshot()
    snapshot["runtime"]["optional_source_paths"]["FLASH_MLA_PATH"].update(
        {
            "set": True,
            "path": str(FORBIDDEN_KERNEL_ROOT / "FlashMLA"),
            "exists": True,
            "error": None,
        }
    )

    issues = validate_environment_snapshot(snapshot)

    assert any(str(FORBIDDEN_KERNEL_ROOT) in issue for issue in issues)


def test_validation_requires_actual_tvm_and_ffi_library_hashes() -> None:
    snapshot = _valid_snapshot()
    snapshot["tvm"]["loaded_libraries"]["tvm"][0]["sha256"] = None
    snapshot["tvm"]["loaded_libraries"]["ffi"] = []

    issues = validate_environment_snapshot(snapshot)

    assert any("invalid TVM library path/hash provenance" in issue for issue in issues)
    assert any("libtvm_ffi.so" in issue for issue in issues)
    assert any("core.*.so" in issue for issue in issues)


def test_optional_cuda_and_benchmark_requirements_become_hard_gates() -> None:
    snapshot = _valid_snapshot()
    snapshot["dependencies"]["sglang"].update(
        {"installed": False, "module_path": None, "distribution": None, "version": None}
    )

    assert validate_environment_snapshot(snapshot) == ()

    issues = validate_environment_snapshot(snapshot, require_cuda=True, require_benchmark_deps=True)
    assert any("fully identified NVIDIA GPU" in issue for issue in issues)
    assert any("working nvcc" in issue for issue in issues)
    assert any("sglang" in issue for issue in issues)
    for lock in _LOCK_NAMES:
        assert any(lock in issue for issue in issues)
    assert any("NVSHMEM_HOME" in issue for issue in issues)


def test_cli_emits_machine_readable_failure_and_writes_the_same_artifact(tmp_path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "environment.json"
    command = [
        sys.executable,
        "-m",
        "tirx_kernels.environment",
        "--kernel-root",
        str(tmp_path),
        "--format",
        "json",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    stdout_report = json.loads(result.stdout)
    artifact_report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_report == artifact_report
    assert stdout_report["ok"] is False
    assert stdout_report["snapshot"] is None
    assert any("not an allowed kernel root" in issue for issue in stdout_report["issues"])


def test_cli_text_failure_is_human_readable(tmp_path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tirx_kernels.environment",
            "--kernel-root",
            str(tmp_path),
            "--format",
            "text",
        ],
        cwd=repository_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stdout.startswith("environment preflight: FAIL\n")
    assert "not an allowed kernel root" in result.stdout


def test_validation_does_not_mutate_a_recorded_snapshot() -> None:
    snapshot = _valid_snapshot()
    before = copy.deepcopy(snapshot)

    validate_environment_snapshot(snapshot, require_cuda=True, require_benchmark_deps=True)

    assert snapshot == before
