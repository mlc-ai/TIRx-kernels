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
"""Capture and validate the reproducible TIRx kernel build environment.

The public snapshot is deliberately made only from observable state: git
revisions, imported module and mapped shared-library paths, file digests,
tool output, installed distributions, and explicitly locked runtime paths.
It is suitable for preserving alongside correctness and benchmark artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KERNEL_SHA = "f495258bdd188dc2584b59a3f94bcf6849b2c652"
TVM_SHA = "4b5598a74fb164000e624eafcbfce7ba85d3efd8"

MAIN_KERNEL_ROOT = Path("/home/hongyij/workspace/tirx-kernels")
BEFORE_KERNEL_ROOT = Path("/home/hongyij/workspace/tirx-kernels-before")
TVM_ROOT = Path("/home/hongyij/workspace/tvm-4b5598a74f")
TVM_PYTHON_ROOT = TVM_ROOT / "python"
TVM_FFI_ROOT = TVM_ROOT / "build" / "python"
TVM_LIBRARY_ROOT = TVM_ROOT / "build" / "lib"
FORBIDDEN_TIR_ROOT = Path("/home/hongyij/tir")
FORBIDDEN_KERNEL_ROOT = Path("/home/hongyij/tirx-kernels")
_FORBIDDEN_ROOTS = (FORBIDDEN_TIR_ROOT, FORBIDDEN_KERNEL_ROOT)

SCHEMA_VERSION = 1
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_DEPENDENCIES = {
    "torch": ("torch", ("torch",)),
    "deep_gemm": ("deep_gemm", ("deep-gemm", "deep_gemm")),
    "flashinfer": ("flashinfer", ("flashinfer-python", "flashinfer")),
    # FA4 deliberately shares the ``flash_attn`` import namespace with the
    # legacy extension package.  Prefer the distribution that supplies the
    # CuTe baseline used by this repository when both are installed.
    "flash_attn": ("flash_attn", ("flash-attn-4", "flash-attn")),
    "flash_kda": ("flash_kda", ("flash-kda", "flash_kda")),
    "sglang": ("sglang", ("sglang",)),
    "flash_mla": ("flash_mla", ("flash-mla", "flash_mla")),
    "cutlass": ("cutlass", ("nvidia-cutlass-dsl",)),
    "nccl": ("nccl", ("nccl4py", "nvidia-nccl-cu13", "nvidia-nccl-cu12", "nccl")),
}
_BENCHMARK_DEPENDENCIES = tuple(_DEPENDENCIES)
_COMMUNICATION_LIBRARY_LOCKS = (
    "TIRX_NCCL_LIBRARY",
    "TIRX_CUBLAS_LIBRARY",
    "TIRX_CUBLASMP_LIBRARY",
    "TIRX_NVSHMEM_LIBRARY",
)
_OPTIONAL_SOURCE_PATHS = ("FLASHKDA_PR_WORKTREE", "FLASHKDA_SOURCE_DIR", "FLASH_MLA_PATH")


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    """Return an absolute normalized path without touching the filesystem."""
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: str | os.PathLike[str], root: Path) -> bool:
    candidate = _lexical_absolute(path)
    root = _lexical_absolute(root)
    return candidate == root or root in candidate.parents


def _forbidden_root(path: str | os.PathLike[str]) -> Path | None:
    return next((root for root in _FORBIDDEN_ROOTS if _is_within(path, root)), None)


def resolve_kernel_root(value: str | os.PathLike[str] = "main") -> Path:
    """Resolve one of the two authorized kernel worktrees.

    Rejection is lexical and happens before filesystem resolution, so an
    arbitrary or forbidden path is never probed as a fallback.
    """
    if os.fspath(value) == "main":
        candidate = MAIN_KERNEL_ROOT
    elif os.fspath(value) == "before":
        candidate = BEFORE_KERNEL_ROOT
    else:
        candidate = _lexical_absolute(value)

    forbidden = _forbidden_root(candidate)
    if forbidden is not None:
        raise ValueError(f"refused forbidden kernel root without accessing it: {forbidden}")
    allowed = (MAIN_KERNEL_ROOT, BEFORE_KERNEL_ROOT)
    if candidate not in allowed:
        choices = ", ".join(str(path) for path in allowed)
        raise ValueError(f"{candidate} is not an allowed kernel root; expected one of: {choices}")

    try:
        if candidate.is_symlink():
            raise ValueError(f"allowed kernel root must not be a symlink: {candidate}")
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"allowed kernel root is unavailable: {candidate}: {exc}") from exc
    if resolved != candidate or not resolved.is_dir():
        raise ValueError(f"allowed kernel root did not resolve to a directory: {candidate}")
    return resolved


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    forbidden_search_paths = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and _forbidden_root(entry) is not None
    ]
    if forbidden_search_paths:
        return {
            "command": command,
            "executable": None,
            "returncode": None,
            "stdout": "",
            "stderr": "refused executable discovery while a forbidden root is on PATH",
        }
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "executable": None,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} was not found",
        }
    forbidden = _forbidden_root(executable)
    if forbidden is not None:
        return {
            "command": command,
            "executable": executable,
            "returncode": None,
            "stdout": "",
            "stderr": f"refused executable from forbidden root without accessing it: {forbidden}",
        }
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "executable": executable,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": command,
        "executable": executable,
        "returncode": completed.returncode,
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.strip(),
    }


def _git_repository(root: Path, expected_sha: str) -> dict[str, Any]:
    revision = _run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=root)
    sha = revision["stdout"] if revision["returncode"] == 0 else None
    revision_error = None if revision["returncode"] == 0 else revision["stderr"]

    submodule_result = _run(["git", "submodule", "status", "--recursive"], cwd=root)
    submodules: list[dict[str, Any]] = []
    submodule_error = None
    if submodule_result["returncode"] == 0:
        states = {" ": "clean", "-": "uninitialized", "+": "revision-mismatch", "U": "conflict"}
        for line in submodule_result["stdout"].splitlines():
            match = re.match(r"^([ +\-U])([0-9a-f]{40})\s+(\S+)(?:\s+\((.*)\))?$", line)
            if match is None:
                submodules.append({"raw": line, "status": "unparsed", "sha": None, "path": None})
                continue
            prefix, submodule_sha, path, description = match.groups()
            submodules.append(
                {
                    "path": path,
                    "sha": submodule_sha,
                    "status": states[prefix],
                    "description": description,
                }
            )
    else:
        submodule_error = submodule_result["stderr"] or "git submodule status failed"

    return {
        "root": str(root),
        "sha": sha,
        "expected_sha": expected_sha,
        "revision_error": revision_error,
        "submodules": submodules,
        "submodule_error": submodule_error,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    lexical = _lexical_absolute(path)
    forbidden = _forbidden_root(lexical)
    if forbidden is not None:
        return {
            "path": str(lexical),
            "size": None,
            "sha256": None,
            "error": f"refused file from forbidden root without accessing it: {forbidden}",
        }
    try:
        resolved = lexical.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a regular file")
        return {
            "path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
            "error": None,
        }
    except OSError as exc:
        return {"path": str(lexical), "size": None, "sha256": None, "error": str(exc)}


def _safe_module_path(module: Any) -> tuple[str | None, str | None]:
    raw = getattr(module, "__file__", None)
    if not raw:
        return None, "imported module has no __file__"
    lexical = _lexical_absolute(raw)
    forbidden = _forbidden_root(lexical)
    if forbidden is not None:
        return str(lexical), f"refused module from forbidden root without accessing it: {forbidden}"
    try:
        return str(lexical.resolve(strict=True)), None
    except OSError as exc:
        return str(lexical), str(exc)


def _mapped_libraries(ffi_core_path: str | None) -> dict[str, Any]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        return {"source": str(maps_path), "tvm": [], "ffi": [], "error": "unavailable"}

    try:
        mapped_paths: set[Path] = set()
        for line in maps_path.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            raw_path = fields[5].removesuffix(" (deleted)")
            lexical = _lexical_absolute(raw_path)
            name = lexical.name
            if name.startswith("libtvm") or (ffi_core_path and lexical == Path(ffi_core_path)):
                mapped_paths.add(lexical)
    except OSError as exc:
        return {"source": str(maps_path), "tvm": [], "ffi": [], "error": str(exc)}

    tvm_records: list[dict[str, Any]] = []
    ffi_records: list[dict[str, Any]] = []
    ffi_core = _lexical_absolute(ffi_core_path) if ffi_core_path else None
    for path in sorted(mapped_paths):
        record = _file_record(path)
        # Only the tvm-ffi runtime library and Python extension belong to the
        # isolated tvm_ffi package.  TVM's build tree also exposes support
        # libraries such as libtvm_ffi_testing.so; those are ordinary TVM
        # build products and must be validated against TVM_LIBRARY_ROOT.
        is_ffi_runtime = path.name == "libtvm_ffi.so" or path.name.startswith(
            "libtvm_ffi.so."
        )
        if is_ffi_runtime or (ffi_core is not None and path == ffi_core):
            ffi_records.append(record)
        else:
            tvm_records.append(record)
    return {"source": str(maps_path), "tvm": tvm_records, "ffi": ffi_records, "error": None}


def _library_search_path() -> dict[str, Any]:
    raw = os.environ.get("TVM_LIBRARY_PATH")
    paths = (
        []
        if raw is None
        else [str(_lexical_absolute(item)) for item in raw.split(os.pathsep) if item]
    )
    return {"raw": raw, "paths": paths}


def _python_search_path() -> dict[str, Any]:
    raw = os.environ.get("PYTHONPATH")
    paths = (
        []
        if raw is None
        else [str(_lexical_absolute(item)) for item in raw.split(os.pathsep) if item]
    )
    return {"raw": raw, "paths": paths}


def _collect_tvm() -> dict[str, Any]:
    search_path = _library_search_path()
    python_search_path = _python_search_path()
    forbidden_paths = []
    for entry in [*sys.path, *python_search_path["paths"], *search_path["paths"]]:
        if entry and _forbidden_root(entry) is not None:
            forbidden_paths.append(entry)
    result: dict[str, Any] = {
        "root": str(TVM_ROOT),
        "python_root": str(TVM_PYTHON_ROOT),
        "ffi_root": str(TVM_FFI_ROOT),
        "library_root": str(TVM_LIBRARY_ROOT),
        "python_search_path": python_search_path,
        "library_search_path": search_path,
        "forbidden_source_paths": sorted(set(forbidden_paths)),
        "module_path": None,
        "module_version": None,
        "ffi_module_path": None,
        "ffi_version": None,
        "import_error": None,
        "loaded_libraries": {"source": "/proc/self/maps", "tvm": [], "ffi": [], "error": None},
    }
    if forbidden_paths:
        result["import_error"] = "refused import while a forbidden root is on a source path"
        return result

    try:
        tvm = importlib.import_module("tvm")
        tvm_path, tvm_path_error = _safe_module_path(tvm)
        result["module_path"] = tvm_path
        result["module_version"] = getattr(tvm, "__version__", None)
        if tvm_path_error:
            result["import_error"] = tvm_path_error
            return result

        tvm_ffi = importlib.import_module("tvm_ffi")
        ffi_path, ffi_path_error = _safe_module_path(tvm_ffi)
        result["ffi_module_path"] = ffi_path
        result["ffi_version"] = getattr(tvm_ffi, "__version__", None)
        if ffi_path_error:
            result["import_error"] = ffi_path_error
            return result

        ffi_core = importlib.import_module("tvm_ffi.core")
        ffi_core_path, ffi_core_error = _safe_module_path(ffi_core)
        if ffi_core_error:
            result["import_error"] = ffi_core_error
            return result
        result["ffi_core_path"] = ffi_core_path
        result["loaded_libraries"] = _mapped_libraries(ffi_core_path)
    except Exception as exc:  # The report must survive and explain import failures.
        result["import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _collect_cuda() -> dict[str, Any]:
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, str]] = []
    if smi["returncode"] == 0:
        for row in csv.reader(smi["stdout"].splitlines(), skipinitialspace=True):
            if len(row) == 5:
                gpus.append(
                    {
                        "index": row[0],
                        "uuid": row[1],
                        "name": row[2],
                        "driver_version": row[3],
                        "compute_capability": row[4],
                    }
                )

    nvcc = _run(["nvcc", "--version"])
    nvcc_text = "\n".join(part for part in (nvcc["stdout"], nvcc["stderr"]) if part)
    release = re.search(r"release\s+([0-9.]+)", nvcc_text)
    build = re.search(r"\bV([0-9.]+)", nvcc_text)
    return {
        "nvidia_smi": {
            "executable": smi["executable"],
            "returncode": smi["returncode"],
            "error": None if smi["returncode"] == 0 else smi["stderr"],
        },
        "gpus": gpus,
        "nvcc": {
            "executable": nvcc["executable"],
            "returncode": nvcc["returncode"],
            "release": release.group(1) if release else None,
            "build": build.group(1) if build else None,
            "output": nvcc_text or None,
        },
    }


def _distribution_version(candidates: tuple[str, ...]) -> tuple[str | None, str | None]:
    for distribution in candidates:
        try:
            return distribution, importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None, None


def _collect_dependencies() -> dict[str, Any]:
    dependencies: dict[str, Any] = {}
    forbidden_paths = [entry for entry in sys.path if entry and _forbidden_root(entry) is not None]
    if forbidden_paths:
        for name, (module_name, _) in _DEPENDENCIES.items():
            dependencies[name] = {
                "module": module_name,
                "installed": False,
                "module_path": None,
                "distribution": None,
                "version": None,
                "error": "refused dependency discovery while a forbidden root is on sys.path",
            }
        return dependencies
    for name, (module_name, distributions) in _DEPENDENCIES.items():
        error = None
        try:
            spec = importlib.util.find_spec(module_name)
            module_path = None if spec is None else spec.origin
        except (ImportError, AttributeError, ValueError) as exc:
            spec = None
            module_path = None
            error = f"{type(exc).__name__}: {exc}"
        if module_path and (forbidden := _forbidden_root(module_path)) is not None:
            spec = None
            error = f"refused dependency module from forbidden root: {forbidden}"
            module_path = None
        distribution, version = _distribution_version(distributions)
        dependencies[name] = {
            "module": module_name,
            "installed": spec is not None,
            "module_path": module_path,
            "distribution": distribution,
            "version": version,
            "error": error,
        }
    return dependencies


def _collect_path_setting(name: str, *, file_required: bool) -> dict[str, Any]:
    raw = os.environ.get(name)
    if raw is None:
        return {"set": False, "path": None, "exists": False, "sha256": None, "error": None}
    lexical = _lexical_absolute(raw)
    if not Path(raw).is_absolute():
        return {
            "set": True,
            "path": str(lexical),
            "exists": False,
            "sha256": None,
            "error": "path must be absolute",
        }
    forbidden = _forbidden_root(lexical)
    if forbidden is not None:
        return {
            "set": True,
            "path": str(lexical),
            "exists": False,
            "sha256": None,
            "error": f"refused path from forbidden root without accessing it: {forbidden}",
        }
    try:
        resolved = lexical.resolve(strict=True)
        valid_kind = resolved.is_file() if file_required else resolved.is_dir()
        if not valid_kind:
            expected = "file" if file_required else "directory"
            raise OSError(f"not a {expected}")
        return {
            "set": True,
            "path": str(resolved),
            "exists": True,
            "sha256": _sha256(resolved) if file_required else None,
            "error": None,
        }
    except OSError as exc:
        return {
            "set": True,
            "path": str(lexical),
            "exists": False,
            "sha256": None,
            "error": str(exc),
        }


def _collect_runtime_settings() -> dict[str, Any]:
    return {
        "library_locks": {
            name: _collect_path_setting(name, file_required=True)
            for name in _COMMUNICATION_LIBRARY_LOCKS
        },
        "nvshmem_home": _collect_path_setting("NVSHMEM_HOME", file_required=False),
        "optional_source_paths": {
            name: _collect_path_setting(name, file_required=False)
            for name in _OPTIONAL_SOURCE_PATHS
        },
    }


def collect_environment_snapshot(kernel_root: str | os.PathLike[str] = "main") -> dict[str, Any]:
    """Collect the live environment into a JSON-serializable snapshot."""
    selected_kernel_root = resolve_kernel_root(kernel_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "platform": platform.platform()},
        "python": {
            "executable": str(_lexical_absolute(sys.executable)),
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "repositories": {
            "kernel": _git_repository(selected_kernel_root, KERNEL_SHA),
            "tvm": _git_repository(TVM_ROOT, TVM_SHA),
        },
        "tvm": _collect_tvm(),
        "cuda": _collect_cuda(),
        "dependencies": _collect_dependencies(),
        "runtime": _collect_runtime_settings(),
    }


def _valid_hash(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and isinstance(record.get("size"), int)
        and record["size"] >= 0
        and isinstance(record.get("sha256"), str)
        and _SHA256_RE.fullmatch(record["sha256"]) is not None
        and record.get("error") is None
    )


def _declared_paths(value: Any, key: str | None = None):
    path_keys = {
        "executable",
        "ffi_core_path",
        "ffi_module_path",
        "library_root",
        "module_path",
        "path",
        "python_root",
        "root",
        "source",
    }
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _declared_paths(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _declared_paths(child, key)
    elif isinstance(value, str) and (key in path_keys or key == "paths"):
        yield value


def _validate_repository(
    issues: list[str], name: str, repository: Any, roots: tuple[Path, ...], expected_sha: str
) -> None:
    if not isinstance(repository, dict):
        issues.append(f"missing {name} repository provenance")
        return
    root = repository.get("root")
    allowed_roots = {str(_lexical_absolute(candidate)) for candidate in roots}
    if root not in allowed_roots:
        issues.append(f"{name} repository root is not allowed: {root!r}")
    sha = repository.get("sha")
    if sha != expected_sha or not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None:
        issues.append(f"{name} repository SHA must be {expected_sha}, got {sha!r}")
    if repository.get("expected_sha") != expected_sha:
        issues.append(f"{name} expected-SHA provenance is inconsistent")
    if repository.get("revision_error") is not None:
        issues.append(
            f"{name} repository revision collection failed: {repository['revision_error']}"
        )
    submodules = repository.get("submodules")
    if not isinstance(submodules, list) or repository.get("submodule_error") is not None:
        issues.append(f"{name} submodule provenance is incomplete")
        return
    for submodule in submodules:
        if (
            not isinstance(submodule, dict)
            or submodule.get("status") != "clean"
            or not isinstance(submodule.get("sha"), str)
            or _SHA_RE.fullmatch(submodule["sha"]) is None
            or not submodule.get("path")
        ):
            issues.append(f"{name} has non-reproducible submodule state: {submodule!r}")


def validate_environment_snapshot(
    snapshot: Any, *, require_cuda: bool = False, require_benchmark_deps: bool = False
) -> tuple[str, ...]:
    """Return all violations of the fixed environment contract.

    Missing optional capabilities are recorded but accepted by default.  The
    two requirement flags turn CUDA execution and the complete benchmark host
    dependency set into hard gates.
    """
    issues: list[str] = []
    if not isinstance(snapshot, dict):
        return ("environment snapshot must be a JSON object",)
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"unsupported or missing schema_version: {snapshot.get('schema_version')!r}")
    if not isinstance(snapshot.get("captured_at"), str) or not snapshot["captured_at"]:
        issues.append("capture timestamp provenance is missing")
    host = snapshot.get("host")
    if not isinstance(host, dict) or not all(host.get(field) for field in ("hostname", "platform")):
        issues.append("host provenance is incomplete")
    for path in sorted(set(_declared_paths(snapshot))):
        if (forbidden := _forbidden_root(path)) is not None:
            issues.append(f"snapshot declares a path inside forbidden root {forbidden}: {path}")

    repositories = snapshot.get("repositories")
    if not isinstance(repositories, dict):
        issues.append("missing repository provenance")
    else:
        _validate_repository(
            issues,
            "kernel",
            repositories.get("kernel"),
            (MAIN_KERNEL_ROOT, BEFORE_KERNEL_ROOT),
            KERNEL_SHA,
        )
        _validate_repository(issues, "TVM", repositories.get("tvm"), (TVM_ROOT,), TVM_SHA)

    python = snapshot.get("python")
    if not isinstance(python, dict) or not all(
        isinstance(python.get(field), str) and python[field]
        for field in ("executable", "version", "implementation")
    ):
        issues.append("Python provenance is incomplete")

    tvm = snapshot.get("tvm")
    if not isinstance(tvm, dict):
        issues.append("missing TVM import provenance")
    else:
        if tvm.get("root") != str(TVM_ROOT):
            issues.append(f"TVM root must be {TVM_ROOT}, got {tvm.get('root')!r}")
        if tvm.get("import_error") is not None:
            issues.append(f"TVM import failed validation: {tvm['import_error']}")
        forbidden_paths = tvm.get("forbidden_source_paths")
        if not isinstance(forbidden_paths, list) or forbidden_paths:
            issues.append(f"forbidden TVM source path detected: {forbidden_paths!r}")
        module_path = tvm.get("module_path")
        if not isinstance(module_path, str) or not _is_within(module_path, TVM_PYTHON_ROOT):
            issues.append(f"TVM must import from {TVM_PYTHON_ROOT}, got {module_path!r}")
        elif _is_within(module_path, FORBIDDEN_TIR_ROOT):
            issues.append(f"TVM import came from forbidden source {FORBIDDEN_TIR_ROOT}")
        if not isinstance(tvm.get("module_version"), str) or not tvm["module_version"]:
            issues.append("TVM version provenance is missing")

        python_search = tvm.get("python_search_path")
        expected_python_path = [str(TVM_PYTHON_ROOT), str(TVM_FFI_ROOT)]
        if (
            not isinstance(python_search, dict)
            or python_search.get("paths") != expected_python_path
        ):
            paths = None if not isinstance(python_search, dict) else python_search.get("paths")
            issues.append(
                f"PYTHONPATH must resolve to {expected_python_path!r} in that order, got {paths!r}"
            )
        ffi_module_path = tvm.get("ffi_module_path")
        if not isinstance(ffi_module_path, str) or not _is_within(ffi_module_path, TVM_FFI_ROOT):
            issues.append(
                f"tvm_ffi must import from isolated target {TVM_FFI_ROOT}, got {ffi_module_path!r}"
            )
        if not isinstance(tvm.get("ffi_version"), str) or not tvm["ffi_version"]:
            issues.append("tvm_ffi version provenance is missing")
        ffi_core_path = tvm.get("ffi_core_path")
        if not isinstance(ffi_core_path, str) or not _is_within(ffi_core_path, TVM_FFI_ROOT):
            issues.append(
                f"tvm_ffi core must load from isolated target {TVM_FFI_ROOT}, got {ffi_core_path!r}"
            )

        search = tvm.get("library_search_path")
        expected_library_path = str(TVM_LIBRARY_ROOT)
        if not isinstance(search, dict) or search.get("paths") != [expected_library_path]:
            paths = None if not isinstance(search, dict) else search.get("paths")
            issues.append(
                f"TVM_LIBRARY_PATH must resolve only to {expected_library_path}, got {paths!r}"
            )

        loaded = tvm.get("loaded_libraries")
        if not isinstance(loaded, dict) or loaded.get("error") is not None:
            issues.append("actual TVM/FFI loaded-library provenance is unavailable")
        else:
            tvm_libraries = loaded.get("tvm")
            ffi_libraries = loaded.get("ffi")
            if not isinstance(tvm_libraries, list) or not isinstance(ffi_libraries, list):
                issues.append("actual TVM/FFI loaded-library provenance is incomplete")
            else:
                names = {Path(record.get("path", "")).name for record in tvm_libraries}
                for required in ("libtvm_compiler.so", "libtvm_runtime.so"):
                    if required not in names:
                        issues.append(f"required loaded TVM library is missing: {required}")
                for record in tvm_libraries:
                    if not _valid_hash(record):
                        issues.append(f"invalid TVM library path/hash provenance: {record!r}")
                    elif not _is_within(record["path"], TVM_LIBRARY_ROOT):
                        issues.append(
                            f"TVM library was loaded outside {TVM_LIBRARY_ROOT}: {record['path']}"
                        )
                ffi_names = {Path(record.get("path", "")).name for record in ffi_libraries}
                if "libtvm_ffi.so" not in ffi_names:
                    issues.append("required loaded FFI library is missing: libtvm_ffi.so")
                if not any(name.startswith("core.") and name.endswith(".so") for name in ffi_names):
                    issues.append("required loaded FFI extension is missing: core.*.so")
                for record in ffi_libraries:
                    if not _valid_hash(record):
                        issues.append(f"invalid FFI library path/hash provenance: {record!r}")
                    elif not _is_within(record["path"], TVM_FFI_ROOT):
                        issues.append(
                            f"FFI library was loaded outside isolated target {TVM_FFI_ROOT}: "
                            f"{record['path']}"
                        )

    dependencies = snapshot.get("dependencies")
    if not isinstance(dependencies, dict):
        issues.append("dependency provenance is missing")
    else:
        for name in _DEPENDENCIES:
            dependency = dependencies.get(name)
            if not isinstance(dependency, dict) or not isinstance(
                dependency.get("installed"), bool
            ):
                issues.append(f"dependency provenance is missing for {name}")
                continue
            if dependency.get("error") is not None:
                issues.append(f"dependency provenance failed for {name}: {dependency['error']}")
            if dependency["installed"] and (
                not dependency.get("module_path") or not dependency.get("version")
            ):
                issues.append(f"installed dependency lacks path/version provenance: {name}")
            if require_benchmark_deps and not dependency["installed"]:
                issues.append(f"required benchmark dependency is unavailable: {name}")

    cuda = snapshot.get("cuda")
    if (
        not isinstance(cuda, dict)
        or not isinstance(cuda.get("nvidia_smi"), dict)
        or not isinstance(cuda.get("gpus"), list)
        or not isinstance(cuda.get("nvcc"), dict)
    ):
        issues.append("CUDA provenance is missing")
    elif require_cuda:
        if not cuda["gpus"] or any(
            not all(
                gpu.get(field) for field in ("uuid", "name", "driver_version", "compute_capability")
            )
            for gpu in cuda["gpus"]
        ):
            issues.append("--require-cuda needs at least one fully identified NVIDIA GPU")
        if cuda["nvcc"].get("returncode") != 0 or not cuda["nvcc"].get("release"):
            issues.append("--require-cuda needs a working nvcc with a reported CUDA release")

    runtime = snapshot.get("runtime")
    if not isinstance(runtime, dict):
        issues.append("runtime dependency provenance is missing")
    else:
        locks = runtime.get("library_locks")
        if not isinstance(locks, dict):
            issues.append("communication library-lock provenance is missing")
        else:
            for name in _COMMUNICATION_LIBRARY_LOCKS:
                lock = locks.get(name)
                if not isinstance(lock, dict) or not isinstance(lock.get("set"), bool):
                    issues.append(f"communication library-lock provenance is missing for {name}")
                    continue
                if lock.get("set") and (
                    not lock.get("exists")
                    or lock.get("error") is not None
                    or not isinstance(lock.get("sha256"), str)
                    or _SHA256_RE.fullmatch(lock["sha256"]) is None
                ):
                    issues.append(f"configured communication library lock is invalid: {name}")
                if require_benchmark_deps and not lock.get("set"):
                    issues.append(f"required benchmark library lock is unset: {name}")
        nvshmem_home = runtime.get("nvshmem_home")
        if not isinstance(nvshmem_home, dict) or not isinstance(nvshmem_home.get("set"), bool):
            issues.append("NVSHMEM_HOME provenance is missing")
        elif nvshmem_home.get("set") and (
            not nvshmem_home.get("exists") or nvshmem_home.get("error") is not None
        ):
            issues.append("configured NVSHMEM_HOME is invalid")
        elif require_benchmark_deps and not nvshmem_home.get("set"):
            issues.append("required benchmark path is unset: NVSHMEM_HOME")
        optional_paths = runtime.get("optional_source_paths")
        if not isinstance(optional_paths, dict):
            issues.append("optional benchmark source-path provenance is missing")
        else:
            for name in _OPTIONAL_SOURCE_PATHS:
                setting = optional_paths.get(name)
                if not isinstance(setting, dict) or not isinstance(setting.get("set"), bool):
                    issues.append(f"optional source-path provenance is missing for {name}")
                elif setting.get("set") and (
                    not setting.get("exists") or setting.get("error") is not None
                ):
                    issues.append(f"configured optional source path is invalid: {name}")

    return tuple(issues)


def make_environment_report(
    snapshot: dict[str, Any], *, require_cuda: bool = False, require_benchmark_deps: bool = False
) -> dict[str, Any]:
    """Validate a snapshot and return the stable CLI report object."""
    issues = validate_environment_snapshot(
        snapshot, require_cuda=require_cuda, require_benchmark_deps=require_benchmark_deps
    )
    return {
        "ok": not issues,
        "requirements": {"cuda": require_cuda, "benchmark_dependencies": require_benchmark_deps},
        "issues": list(issues),
        "snapshot": snapshot,
    }


def _text_report(report: dict[str, Any]) -> str:
    lines = [f"environment preflight: {'PASS' if report['ok'] else 'FAIL'}"]
    snapshot = report.get("snapshot")
    if isinstance(snapshot, dict):
        repositories = snapshot.get("repositories", {})
        for label in ("kernel", "tvm"):
            repository = repositories.get(label, {})
            lines.append(f"{label}: {repository.get('root', '?')} @ {repository.get('sha', '?')}")
        python = snapshot.get("python", {})
        lines.append(
            f"python: {python.get('executable', '?')} ({python.get('implementation', '?')})"
        )
        tvm = snapshot.get("tvm", {})
        lines.append(f"tvm import: {tvm.get('module_path', '?')}")
        loaded = tvm.get("loaded_libraries", {})
        for family in ("tvm", "ffi"):
            for library in loaded.get(family, []):
                lines.append(
                    f"{family} library: {library.get('path', '?')} sha256={library.get('sha256', '?')}"
                )
        cuda = snapshot.get("cuda", {})
        lines.append(
            f"cuda: nvcc={cuda.get('nvcc', {}).get('release', 'unavailable')} "
            f"gpus={len(cuda.get('gpus', []))}"
        )
        for name, dependency in snapshot.get("dependencies", {}).items():
            state = dependency.get("version") if dependency.get("installed") else "unavailable"
            lines.append(f"dependency {name}: {state}")
        runtime = snapshot.get("runtime", {})
        for name, lock in runtime.get("library_locks", {}).items():
            state = lock.get("path") if lock.get("set") else "unset"
            lines.append(f"library lock {name}: {state}")
    if report.get("issues"):
        lines.append("issues:")
        lines.extend(f"- {issue}" for issue in report["issues"])
    return "\n".join(lines) + "\n"


def _render(report: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report, indent=2, sort_keys=True) + "\n"
    return _text_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the fixed TIRx kernel environment")
    parser.add_argument(
        "--kernel-root",
        default="main",
        help="main, before, or the exact absolute path of either authorized worktree",
    )
    parser.add_argument("--format", choices=("json", "text"), default="text")
    parser.add_argument("--output", type=Path, default=None, help="Also write the report here")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-benchmark-deps", action="store_true")
    args = parser.parse_args(argv)

    try:
        snapshot = collect_environment_snapshot(args.kernel_root)
        report = make_environment_report(
            snapshot,
            require_cuda=args.require_cuda,
            require_benchmark_deps=args.require_benchmark_deps,
        )
    except Exception as exc:  # Preserve a machine-readable negative result.
        report = {
            "ok": False,
            "requirements": {
                "cuda": args.require_cuda,
                "benchmark_dependencies": args.require_benchmark_deps,
            },
            "issues": [f"{type(exc).__name__}: {exc}"],
            "snapshot": None,
        }

    rendered = _render(report, args.format)
    if args.output is not None:
        try:
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"failed to write environment report to {args.output}: {exc}", file=sys.stderr)
            return 2
    sys.stdout.write(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BEFORE_KERNEL_ROOT",
    "FORBIDDEN_KERNEL_ROOT",
    "KERNEL_SHA",
    "MAIN_KERNEL_ROOT",
    "TVM_FFI_ROOT",
    "TVM_LIBRARY_ROOT",
    "TVM_PYTHON_ROOT",
    "TVM_ROOT",
    "TVM_SHA",
    "collect_environment_snapshot",
    "main",
    "make_environment_report",
    "resolve_kernel_root",
    "validate_environment_snapshot",
]
