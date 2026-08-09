# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Compile codegen evidence or hard-gate a baseline/candidate manifest pair.

The audit starts from the public ``get_kernel(**config)`` return value and the
normal TIRx compilation pipeline.  It writes deterministic CUDA, PTX, cubin,
SASS, and compiler-log artifacts plus a machine-readable manifest.  Missing
tools, source-dump failures, or unparseable resource data are incomplete hard
failures; the partial manifest records the stage rather than treating absent
evidence as success.  Existing manifests can then be compared at observable
launch, resource, and semantic instruction-family layers; absent or regressed
evidence fails the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import tvm
from tvm import tirx
from tvm.ir.instrument import pass_instrument

_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_ARTIFACT_NAMES = (
    "kernel.cu",
    "kernel.ptx",
    "kernel.cubin",
    "kernel.sass",
    "nvcc.txt",
    "ptxas.txt",
)
_RUN_ARTIFACT_NAMES = ("kernel.nvcc.ptx", "ptxas.raw.txt", "timing.json")
_FORBIDDEN_ROOTS = (Path("/home/hongyij/tir"), Path("/home/hongyij/tirx-kernels"))
_ARCH_RE = re.compile(r"^sm_[0-9]{2,3}a?$")
_FLAG_TAGS = {
    "tirx.use_programtic_dependent_launch": "programmatic_dependent_launch",
    "tirx.use_cooperative_launch": "cooperative_launch",
}
_DYNAMIC_SMEM_TAG = "tirx.use_dyn_shared_memory"
_PTX_INSTRUCTION_RE = re.compile(
    r"^\s*(?:@[!A-Za-z0-9_%.$]+\s+)?([A-Za-z][A-Za-z0-9_:.]*(?:\.[A-Za-z0-9_:.]+)*)"
)
_SASS_INSTRUCTION_RE = re.compile(
    r"/\*[0-9a-fA-F]+\*/\s*(?:@[!A-Za-z0-9_.]+\s+)?([A-Z][A-Z0-9_.]*)"
)
_CACHE_MODIFIER_RE = re.compile(r"(?:L[12]::[a-z_]+|(?:(?<=\.)|^)(?:ca|cg|cs|lu|cv|wb|wt)(?=\.|$))")
_MEMORY_ORDER_QUALIFIERS = ("acquire", "release", "acq_rel", "relaxed", "volatile", "weak")
_PTX_SEMANTIC_FAMILIES = (
    "atomic",
    "barrier",
    "descriptor",
    "direct_memory",
    "fence",
    "tcgen05",
    "tma",
)
_SASS_SEMANTIC_FAMILIES = ("atomic", "barrier", "descriptor_or_tma", "direct_memory", "tcgen05")
_LAUNCH_COMPARISON_FIELDS = (
    "grid",
    "block",
    "cluster",
    "preferred_cluster",
    "dynamic_shared_memory_bytes",
    "flags",
    "parameter_tags",
)
_RESOURCE_LIMIT_FIELDS = ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes")
# CUDA 13.2's occupancy API allocates compute-major 10 GPRs in 256-register
# units per warp.  Compare that physical allocation class rather than a raw
# ptxas registers-per-thread value that can move within the same class.
_REGISTER_ALLOCATION_GRANULARITY = {"sm_100": 256, "sm_100a": 256}
_MISSING = object()


class CodegenAuditSelectionError(ValueError):
    """Raised when a requested registry/configuration selection is invalid."""


@dataclass(frozen=True)
class CodegenComparisonCheck:
    """One public-contract check between two codegen manifests."""

    path: str
    rule: str
    baseline: Any
    candidate: Any
    passed: bool
    message: str
    delta: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "path": self.path,
            "rule": self.rule,
            "status": "pass" if self.passed else "fail",
            "baseline": self.baseline,
            "candidate": self.candidate,
            "message": self.message,
        }
        if self.delta is not None:
            result["delta"] = self.delta
        return result


@dataclass(frozen=True)
class CodegenComparisonReport:
    """Deterministic hard-gate result for a baseline/candidate manifest pair."""

    inputs: dict[str, Any]
    checks: tuple[CodegenComparisonCheck, ...]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def status(self) -> str:
        return "pass" if self.ok else "fail"

    def to_dict(self) -> dict[str, Any]:
        checks = [check.to_dict() for check in self.checks]
        failures = [check for check in checks if check["status"] == "fail"]
        changes = [
            check
            for check in checks
            if check["baseline"] != check["candidate"]
            and check["rule"] not in {"required_type", "valid_evidence"}
        ]
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": "tirx_codegen_manifest_comparison",
            "status": self.status,
            "ok": self.ok,
            "inputs": self.inputs,
            "policy": {
                "resource_rule": (
                    "physical register allocation per warp and other candidate resource values "
                    "may not newly appear or increase; raw registers per thread are recorded"
                ),
                "semantic_rule": (
                    "launch and synchronization/tensor instruction-family evidence must be "
                    "unchanged"
                ),
                "direct_memory_rule": (
                    "PTX and SASS count changes are recorded because explicit typed/vectorized "
                    "forms may change statement count; performance is gated independently"
                ),
                "ptx_comparison_layer": "semantic family counts",
                "ignored_instruction_details": [
                    "PTX instruction_count",
                    "PTX family mnemonics",
                    "SASS instruction_count",
                    "SASS family mnemonics",
                ],
            },
            "summary": {
                "checks": len(checks),
                "passed": len(checks) - len(failures),
                "failed": len(failures),
                "changes": len(changes),
            },
            "checks": checks,
            "changes": changes,
            "failures": failures,
        }

    def summary(self) -> str:
        failed = sum(not check.passed for check in self.checks)
        return (
            f"{'PASS' if self.ok else 'FAIL'} codegen comparison: "
            f"{len(self.checks) - failed}/{len(self.checks)} checks passed"
        )


@dataclass(frozen=True)
class CodegenAuditError:
    """One hard failure while collecting the requested evidence."""

    stage: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CodegenAuditReport:
    """Complete or partial result for one registry configuration."""

    kernel: dict[str, Any]
    config: dict[str, Any]
    target: dict[str, Any]
    status: str
    errors: tuple[CodegenAuditError, ...]
    launches: tuple[dict[str, Any], ...]
    resources: dict[str, Any]
    instructions: dict[str, Any]
    tools: dict[str, Any]
    artifacts: dict[str, Any]
    run_artifacts: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "complete" and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "status": self.status,
            "ok": self.ok,
            "kernel": self.kernel,
            "config": self.config,
            "target": self.target,
            "errors": [error.to_dict() for error in self.errors],
            "launches": list(self.launches),
            "resources": self.resources,
            "instructions": self.instructions,
            "tools": self.tools,
            "artifacts": self.artifacts,
            "run_artifacts": self.run_artifacts,
            "normalization": {
                "kernel.ptx": "NVCC private-symbol nonces are replaced by fixed-width zeros",
                "ptxas.txt": "compile-time samples are stored only in timing.json",
            },
        }

    def summary(self) -> str:
        label = self.config.get("label", "?")
        headline = (
            f"{'PASS' if self.ok else 'FAIL'} codegen audit: "
            f"{self.kernel.get('name', '?')}[{label}] ({self.status})"
        )
        if not self.errors:
            return headline
        return "\n".join(
            [headline]
            + [f"ERROR {error.stage}: {error.error_type}: {error.message}" for error in self.errors]
        )


@pass_instrument
class _LoweredModuleCapture:
    """Keep immutable pass outputs so launch metadata can be inspected."""

    def __init__(self) -> None:
        self.modules: list[tvm.IRModule] = []

    def run_after_pass(self, mod: tvm.IRModule, _info: Any) -> None:
        self.modules.append(mod)


def _error(stage: str, value: BaseException | str, error_type: str | None = None):
    if isinstance(value, BaseException):
        message = str(value)
        # Compiler failures can include the entire generated translation unit.
        # Keep the manifest readable while the full stage log remains an artifact.
        if len(message) > 4000:
            message = f"{message[:4000]}... [truncated]"
        return CodegenAuditError(stage, type(value).__name__, message)
    return CodegenAuditError(stage, error_type or "RuntimeError", value)


def _json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"value is not JSON serializable: {exc}") from exc


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    raw = os.path.expanduser(os.fspath(path))
    return Path(os.path.abspath(raw))


def _is_within(path: str | os.PathLike[str], root: Path) -> bool:
    absolute = _lexical_absolute(path)
    try:
        absolute.relative_to(root)
    except ValueError:
        return False
    return True


def _guard_path(path: str | os.PathLike[str], *, purpose: str) -> Path:
    absolute = _lexical_absolute(path)
    for root in _FORBIDDEN_ROOTS:
        if _is_within(absolute, root):
            raise ValueError(f"refusing {purpose} under forbidden root {root}")
    return absolute


def _guard_executable_search_path() -> None:
    for item in os.environ.get("PATH", "").split(os.pathsep):
        if not item:
            continue
        _guard_path(item, purpose="executable search")


def _tool(name: str) -> Path:
    _guard_executable_search_path()
    result = shutil.which(name)
    if result is None:
        raise FileNotFoundError(f"required CUDA tool {name!r} was not found on PATH")
    path = _guard_path(result, purpose=f"using {name}")
    if not path.is_file():
        raise FileNotFoundError(f"required CUDA tool {name!r} is not a file: {path}")
    return path


def _run(
    command: Sequence[str | os.PathLike[str]], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(item) for item in command],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _run_disassembler(
    command: Sequence[str | os.PathLike[str]], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(item) for item in command], cwd=cwd, check=False, text=True, capture_output=True
    )


def _version(tool: Path, *, cwd: Path) -> str:
    result = _run([tool, "--version"], cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"{tool.name} --version exited {result.returncode}: {result.stdout.strip()}"
        )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"{tool.name} --version returned no output")
    return value


def _iter_prim_funcs(value: Any, path: str = "root") -> Iterator[tuple[str, tirx.PrimFunc]]:
    if isinstance(value, tirx.PrimFunc):
        yield path, value
        return
    if isinstance(value, tvm.IRModule):
        functions = sorted(
            value.functions.items(), key=lambda item: str(getattr(item[0], "name_hint", item[0]))
        )
        for global_var, func in functions:
            name = str(getattr(global_var, "name_hint", global_var))
            if not isinstance(func, tirx.PrimFunc):
                raise TypeError(f"{path}.{name} is not a tvm.tirx.PrimFunc")
            yield f"{path}.{name}", func
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=repr):
            yield from _iter_prim_funcs(value[key], f"{path}[{key!r}]")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _iter_prim_funcs(item, f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} is {type(value).__name__}, expected a TIRx PrimFunc, IRModule, "
        "or a list/tuple/dict containing them"
    )


def _compile_module(value: Any) -> tuple[tvm.IRModule, tuple[str, ...]]:
    functions: dict[str, tirx.PrimFunc] = {}
    paths: list[str] = []
    for index, (path, func) in enumerate(_iter_prim_funcs(value)):
        paths.append(path)
        symbol = func.attrs.get("global_symbol") if func.attrs is not None else None
        name = str(symbol) if symbol is not None else ("main" if index == 0 else f"kernel_{index}")
        if name in functions:
            raise ValueError(f"duplicate compiled function symbol {name!r}")
        functions[name] = func
    if not functions:
        raise TypeError("get_kernel returned no TIRx PrimFunc objects")
    return tvm.IRModule(functions), tuple(paths)


def _calling_conv(func: Any) -> int:
    if not isinstance(func, tirx.PrimFunc) or func.attrs is None:
        return 0
    value = func.attrs.get("calling_conv", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _lowered_launch_module(modules: Sequence[tvm.IRModule]) -> tvm.IRModule:
    for mod in reversed(modules):
        conventions = {_calling_conv(func) for func in mod.functions.values()}
        if 1 in conventions and 2 in conventions:
            return mod
    raise RuntimeError("the TIRx pipeline did not expose combined host/device launch metadata")


def _expr_json(value: Any) -> Any:
    if isinstance(value, tvm.tirx.IntImm):
        return int(value.value)
    if isinstance(value, tvm.tirx.FloatImm):
        return float(value.value)
    if isinstance(value, tvm.tirx.StringImm):
        return str(value.value)
    return {"expression": str(value)}


def _launch_slot(tag: str) -> tuple[str, int]:
    axis_name, dot, axis = tag.rpartition(".")
    if dot != "." or axis not in ("x", "y", "z"):
        raise ValueError(f"unsupported launch parameter tag {tag!r}")
    index = "xyz".index(axis)
    if axis_name == "blockIdx":
        return "grid", index
    if axis_name == "threadIdx":
        return "block", index
    if axis_name == "clusterCtaIdx":
        return "cluster", index
    if axis_name in ("preferredClusterCtaIdx", "preferredClusterIdx"):
        return "preferred_cluster", index
    raise ValueError(f"unsupported launch parameter tag {tag!r}")


def _packed_calls(func: tirx.PrimFunc) -> list[tvm.ir.Call]:
    calls: list[tvm.ir.Call] = []

    def visit(node: Any) -> None:
        if (
            isinstance(node, tvm.ir.Call)
            and node.args
            and isinstance(node.args[0], tvm.tirx.StringImm)
        ):
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def extract_launch_topology(lowered: tvm.IRModule) -> list[dict[str, Any]]:
    """Return the observable launch arguments from a lowered host/device module."""
    devices: dict[str, tirx.PrimFunc] = {}
    host_funcs: list[tirx.PrimFunc] = []
    for global_var, func in lowered.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            continue
        convention = _calling_conv(func)
        if convention == 1:
            host_funcs.append(func)
        elif convention == 2:
            name = str(getattr(global_var, "name_hint", global_var))
            symbol = func.attrs.get("global_symbol") if func.attrs is not None else None
            devices[str(symbol) if symbol is not None else name] = func

    calls_by_name: dict[str, list[tvm.ir.Call]] = {name: [] for name in devices}
    for host_func in host_funcs:
        for call in _packed_calls(host_func):
            name = str(call.args[0].value)
            if name in calls_by_name:
                calls_by_name[name].append(call)

    launches: list[dict[str, Any]] = []
    for name in sorted(devices):
        func = devices[name]
        raw_tags = func.attrs.get("tirx.kernel_launch_params") if func.attrs is not None else None
        if raw_tags is None:
            raise RuntimeError(
                f"compiled device function {name!r} has no launch parameter metadata"
            )
        tags = [str(tag) for tag in raw_tags]
        calls = calls_by_name.get(name, [])
        if not calls:
            raise RuntimeError(f"compiled device function {name!r} has no host launch")
        for launch_index, call in enumerate(calls):
            values = list(call.args[1 + len(func.params) :])
            dimensions: dict[str, list[Any]] = {
                "grid": [1, 1, 1],
                "block": [1, 1, 1],
                "cluster": [1, 1, 1],
                "preferred_cluster": [1, 1, 1],
            }
            flags = {"cooperative_launch": False, "programmatic_dependent_launch": False}
            dynamic_smem: Any = 0
            value_index = 0
            for tag in tags:
                flag = _FLAG_TAGS.get(tag)
                if flag is not None:
                    flags[flag] = True
                    continue
                if value_index >= len(values):
                    raise RuntimeError(f"host launch for {name!r} is missing value for {tag!r}")
                value = _expr_json(values[value_index])
                value_index += 1
                if tag == _DYNAMIC_SMEM_TAG:
                    dynamic_smem = value
                    continue
                dimension, axis = _launch_slot(tag)
                dimensions[dimension][axis] = value
            if value_index != len(values):
                raise RuntimeError(
                    f"host launch for {name!r} has {len(values) - value_index} "
                    "unclassified launch argument(s)"
                )
            launches.append(
                {
                    "function": name,
                    "launch_index": launch_index,
                    **dimensions,
                    "dynamic_shared_memory_bytes": dynamic_smem,
                    "flags": flags,
                    "parameter_tags": tags,
                }
            )
    if not launches:
        raise RuntimeError("compiled module contains no observable CUDA launches")
    return launches


def _runtime_modules(root: Any) -> Iterator[Any]:
    pending = [root]
    seen: set[int] = set()
    while pending:
        module = pending.pop(0)
        identity = id(module)
        if identity in seen:
            continue
        seen.add(identity)
        yield module
        pending.extend(getattr(module, "imports", ()))


def _cuda_source(executable: Any) -> str:
    root = getattr(executable, "mod", executable)
    sources: list[str] = []
    for module in _runtime_modules(root):
        if str(getattr(module, "kind", "")) != "cuda":
            continue
        try:
            source = module.inspect_source("cuda")
        except (AttributeError, RuntimeError):
            source = module.inspect_source()
        if source:
            sources.append(str(source))
    if len(sources) != 1:
        raise RuntimeError(f"expected exactly one CUDA source module, found {len(sources)}")
    return sources[0]


def _ptx_mnemonics(source: str) -> list[str]:
    result: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith((".", "{", "}")) or line.endswith(":"):
            continue
        match = _PTX_INSTRUCTION_RE.match(line)
        if match is not None:
            result.append(match.group(1))
    return result


def _family(mnemonics: Sequence[str], predicate: Any) -> dict[str, Any]:
    selected = [mnemonic for mnemonic in mnemonics if predicate(mnemonic)]
    return {"count": len(selected), "mnemonics": dict(sorted(Counter(selected).items()))}


def _sass_mnemonics(source: str) -> list[str]:
    return [match.group(1) for match in _SASS_INSTRUCTION_RE.finditer(source)]


def analyze_instruction_features(
    cuda_source: str, ptx_source: str, sass_source: str
) -> dict[str, Any]:
    """Classify generated PTX/SASS instruction families used by AC-8 evidence."""
    ptx = _ptx_mnemonics(ptx_source)
    cache_modifiers = [
        modifier
        for mnemonic in ptx
        for modifier in _CACHE_MODIFIER_RE.findall(mnemonic)
        if mnemonic.startswith(("ld.", "st."))
    ]
    qualifiers = [
        qualifier
        for mnemonic in ptx
        for qualifier in _MEMORY_ORDER_QUALIFIERS
        if qualifier in mnemonic.split(".")
    ]
    ptx_families = {
        "barrier": _family(ptx, lambda value: value.startswith(("bar.", "barrier.", "mbarrier."))),
        "fence": _family(ptx, lambda value: value.startswith("fence.")),
        "descriptor": _family(ptx, lambda value: "tensormap" in value),
        "tma": _family(
            ptx,
            lambda value: (
                value.startswith(("cp.async.bulk.", "cp.reduce.async.bulk."))
                and not value.startswith(
                    (
                        "cp.async.bulk.commit_group",
                        "cp.async.bulk.wait_group",
                        "cp.reduce.async.bulk.commit_group",
                        "cp.reduce.async.bulk.wait_group",
                    )
                )
            ),
        ),
        "tcgen05": _family(ptx, lambda value: value.startswith("tcgen05.")),
        "atomic": _family(ptx, lambda value: value.startswith(("atom.", "red."))),
        "direct_memory": _family(
            ptx,
            lambda value: value.startswith(("ld.global", "st.global", "ld.shared", "st.shared")),
        ),
    }

    sass = _sass_mnemonics(sass_source)
    sass_families = {
        "barrier": _family(sass, lambda value: "BAR" in value),
        "descriptor_or_tma": _family(
            sass, lambda value: value.startswith("UTMA") or "TENSORMAP" in value
        ),
        "tcgen05": _family(sass, lambda value: "MMA" in value),
        "atomic": _family(
            sass, lambda value: "ATOM" in value or value.startswith(("REDG", "REDS"))
        ),
        "direct_memory": _family(
            sass, lambda value: value.startswith(("LDG", "STG", "LDS", "STS"))
        ),
    }
    return {
        "cuda": {"tensor_map_type_occurrences": cuda_source.count("CUtensorMap")},
        "ptx": {
            "instruction_count": len(ptx),
            "families": ptx_families,
            "cache_modifiers": {
                "count": len(cache_modifiers),
                "values": dict(sorted(Counter(cache_modifiers).items())),
            },
            "memory_order_qualifiers": {
                "count": len(qualifiers),
                "values": dict(sorted(Counter(qualifiers).items())),
            },
        },
        "sass": {"instruction_count": len(sass), "families": sass_families},
    }


def parse_ptxas_verbose(log: str) -> dict[str, dict[str, Any]]:
    """Parse per-entry register, barrier, stack, and spill facts from ``ptxas -v``."""
    entries: dict[str, dict[str, Any]] = {}
    current: str | None = None
    property_seen: set[str] = set()
    for line in log.splitlines():
        entry_match = re.search(r"Compiling entry function ['\"]([^'\"]+)['\"]", line)
        if entry_match is not None:
            current = entry_match.group(1)
            entries.setdefault(current, {"memory_bytes": {}})
            continue
        property_match = re.search(r"Function properties for\s+['\"]?([^'\"\s]+)", line)
        if property_match is not None:
            property_name = property_match.group(1)
            # ptxas also reports out-of-line device helpers (for example the
            # CUDA printf shim).  They are not launch entries and may omit the
            # entry-only register line, so keep the evidence scoped to names
            # introduced by "Compiling entry function".
            current = property_name if property_name in entries else None
            if current is not None:
                property_seen.add(current)
            continue
        if current is None:
            continue
        spill_match = re.search(
            r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
            r"(\d+) bytes spill loads",
            line,
        )
        if spill_match is not None:
            entry = entries[current]
            entry["stack_frame_bytes"] = int(spill_match.group(1))
            entry["spill_store_bytes"] = int(spill_match.group(2))
            entry["spill_load_bytes"] = int(spill_match.group(3))
            continue
        used_match = re.search(r"Used\s+(\d+)\s+registers", line)
        if used_match is None:
            continue
        entry = entries[current]
        entry["registers"] = int(used_match.group(1))
        barrier_match = re.search(r"used\s+(\d+)\s+barriers", line)
        if barrier_match is not None:
            entry["barriers"] = int(barrier_match.group(1))
        for count, kind in re.findall(
            r"(\d+)\s+bytes\s+(cumulative stack size|[A-Za-z_]+(?:\[[0-9]+\])?)", line
        ):
            if kind == "cumulative stack size":
                entry["cumulative_stack_size_bytes"] = int(count)
            else:
                entry["memory_bytes"][kind] = int(count)

    if not entries:
        raise ValueError("ptxas -v output contains no compiled entry functions")
    required = {"registers", "stack_frame_bytes", "spill_store_bytes", "spill_load_bytes"}
    for name, entry in entries.items():
        missing = sorted(required - entry.keys())
        if name not in property_seen:
            missing.append("function_properties")
        if missing:
            raise ValueError(f"ptxas resources for {name!r} are missing: {', '.join(missing)}")
        entry.setdefault("barriers", 0)
        entry["memory_bytes"] = dict(sorted(entry["memory_bytes"].items()))
    return {name: entries[name] for name in sorted(entries)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _artifact_map(output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _ARTIFACT_NAMES:
        path = output_dir / name
        if path.is_file():
            result[name] = _artifact(path, output_dir)
    return result


def _run_artifact_map(output_dir: Path) -> dict[str, Any]:
    return {
        name: {"path": name, "deterministic": False, "sha256_excluded_from_manifest": True}
        for name in _RUN_ARTIFACT_NAMES
        if (output_dir / name).is_file()
    }


def _canonical_ptx(source: str) -> str:
    # nvcc salts private libcu++ symbols even when source and flags are
    # identical.  The salt is not referenced outside the translation unit;
    # replacing its fixed-width hexadecimal field preserves symbol identity
    # while making the evidence and the cubin assembled from it reproducible.
    return re.sub(r"(?<=_INTERNAL_)[0-9a-fA-F]{8}(?=_)", "00000000", source)


def _stable_ptxas_log(log: str) -> tuple[str, list[dict[str, Any]]]:
    current: str | None = None
    stable_lines: list[str] = []
    samples: list[dict[str, Any]] = []
    for line in log.splitlines():
        entry_match = re.search(r"Compiling entry function ['\"]([^'\"]+)['\"]", line)
        if entry_match is not None:
            current = entry_match.group(1)
        timing_match = re.search(r"Compile time\s*=\s*([0-9.]+)\s*ms", line)
        if timing_match is not None:
            samples.append({"function": current, "milliseconds": float(timing_match.group(1))})
            continue
        stable_lines.append(line)
    stable = "\n".join(stable_lines)
    if log.endswith("\n"):
        stable += "\n"
    return stable, samples


def _default_arch(compute_capability: int) -> str:
    suffix = compute_capability * 10
    return f"sm_{suffix}a" if compute_capability >= 9 else f"sm_{suffix}"


def _select(
    registry: Mapping[str, Any] | None, kernel_name: str, config_label: str
) -> tuple[Any, dict[str, Any], int, dict[str, Any]]:
    if registry is None:
        from tirx_kernels.registry import discover_kernels

        registry = discover_kernels(strict=True)
    if not isinstance(registry, Mapping):
        raise TypeError("registry must be a mapping from kernel names to modules")
    if kernel_name not in registry:
        raise CodegenAuditSelectionError(f"kernel {kernel_name!r} was not found")
    module = registry[kernel_name]
    raw_meta = getattr(module, "KERNEL_META", None)
    if not isinstance(raw_meta, Mapping) or raw_meta.get("name") != kernel_name:
        raise CodegenAuditSelectionError(
            f"kernel {kernel_name!r} has missing or inconsistent KERNEL_META"
        )
    meta = _json_value(dict(raw_meta))
    compute_capability = meta.get("compute_capability")
    if isinstance(compute_capability, bool) or not isinstance(compute_capability, int):
        raise CodegenAuditSelectionError(
            f"kernel {kernel_name!r} has an invalid compute_capability"
        )
    configs = getattr(module, "CONFIGS", None)
    if not isinstance(configs, list) or not configs:
        raise CodegenAuditSelectionError(f"kernel {kernel_name!r} has no declared CONFIGS")
    matches = [
        config
        for config in configs
        if isinstance(config, Mapping) and config.get("label") == config_label
    ]
    if not matches:
        raise CodegenAuditSelectionError(
            f"configuration label {config_label!r} was not found for kernel {kernel_name!r}"
        )
    if len(matches) != 1:
        raise CodegenAuditSelectionError(
            f"configuration label {config_label!r} is duplicated for kernel {kernel_name!r}"
        )
    config = _json_value(dict(matches[0]))
    params = {key: value for key, value in matches[0].items() if key != "label"}
    get_kernel = getattr(module, "get_kernel", None)
    if not callable(get_kernel):
        raise CodegenAuditSelectionError(f"kernel {kernel_name!r} has no callable get_kernel")
    return get_kernel, params, compute_capability, {"meta": meta, "config": config}


def _prepare_output(output: str | os.PathLike[str]) -> Path:
    output_dir = _guard_path(output, purpose="writing codegen evidence")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    return output_dir


def _write_manifest(report: CodegenAuditReport, output_dir: Path) -> None:
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    (output_dir / _MANIFEST_NAME).write_text(rendered, encoding="utf-8")


def _compile_options(cuda_source: str) -> list[str]:
    options = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    if not os.environ.get("TVM_CUDA_NVCC_NO_FAST_MATH"):
        options.append("--use_fast_math")
    if "#include <nvshmem" in cuda_source:
        home = os.environ.get("NVSHMEM_HOME")
        if not home:
            raise RuntimeError("generated CUDA requires NVSHMEM_HOME for its headers")
        include = _guard_path(Path(home) / "include", purpose="using NVSHMEM headers")
        if not include.is_dir():
            raise FileNotFoundError(f"NVSHMEM include directory does not exist: {include}")
        options.extend([f"-I{include}", "--relocatable-device-code=true"])
    return options


def _ptxas_options() -> list[str]:
    options = [
        "-v",
        f"--register-usage-level={os.environ.get('TVM_CUDA_PTXAS_REG_LEVEL', '10')}",
        "--warn-on-local-memory-usage",
    ]
    extra = os.environ.get("TVM_CUDA_PTXAS_EXTRA_OPTS", "").strip()
    if extra:
        options.extend(shlex.split(extra))
    return options


def audit_registered_codegen(
    kernel: str,
    config: str,
    output: str | os.PathLike[str],
    *,
    registry: Mapping[str, Any] | None = None,
    arch: str | None = None,
) -> CodegenAuditReport:
    """Collect deterministic codegen evidence for one original registry config.

    The output directory must be absent or empty.  Selection errors are raised;
    compilation/tool failures return an incomplete report and are also written
    to ``manifest.json``.
    """
    get_kernel, params, compute_capability, selected = _select(registry, kernel, config)
    selected_arch = arch or _default_arch(compute_capability)
    if not _ARCH_RE.fullmatch(selected_arch):
        raise CodegenAuditSelectionError(f"invalid CUDA architecture {selected_arch!r}")
    output_dir = _prepare_output(output)

    module_name = getattr(get_kernel, "__module__", None)
    kernel_record = {
        "name": kernel,
        "module": module_name if isinstance(module_name, str) else type(get_kernel).__name__,
        "metadata": selected["meta"],
    }
    config_record = selected["config"]
    target_record = {
        "kind": "cuda",
        "arch": selected_arch,
        "compute_capability": compute_capability,
        "pipeline": "tirx",
        "tvm_version": str(getattr(tvm, "__version__", "unknown")),
        "tvm_python": str(Path(tvm.__file__).absolute()),
    }
    errors: list[CodegenAuditError] = []
    launches: list[dict[str, Any]] = []
    resources: dict[str, Any] = {}
    instructions: dict[str, Any] = {}
    tools: dict[str, Any] = {}

    def make_report() -> CodegenAuditReport:
        return CodegenAuditReport(
            kernel=kernel_record,
            config=config_record,
            target=target_record,
            status="incomplete" if errors else "complete",
            errors=tuple(errors),
            launches=tuple(launches),
            resources=resources,
            instructions=instructions,
            tools=tools,
            artifacts=_artifact_map(output_dir),
            run_artifacts=_run_artifact_map(output_dir),
        )

    try:
        value = get_kernel(**params)
    except Exception as exc:
        errors.append(_error("get_kernel", exc))
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    try:
        ir_module, source_functions = _compile_module(value)
        kernel_record["source_functions"] = list(source_functions)
    except Exception as exc:
        errors.append(_error("ir_normalization", exc))
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    capture = _LoweredModuleCapture()
    current = tvm.transform.PassContext.current()
    try:
        with tvm.transform.PassContext(
            opt_level=current.opt_level,
            required_pass=list(current.required_pass),
            disabled_pass=list(current.disabled_pass),
            config=dict(current.config),
            instruments=[*current.instruments, capture],
        ):
            executable = tvm.compile(
                ir_module,
                target=tvm.target.Target({"kind": "cuda", "arch": selected_arch}),
                tir_pipeline="tirx",
            )
    except Exception as exc:
        errors.append(_error("tvm_compile", exc))
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    try:
        cuda_source = _cuda_source(executable)
        cuda_path = output_dir / "kernel.cu"
        cuda_path.write_text(cuda_source, encoding="utf-8")
    except Exception as exc:
        errors.append(_error("cuda_source", exc))
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    try:
        lowered = _lowered_launch_module(capture.modules)
        launches.extend(extract_launch_topology(lowered))
    except Exception as exc:
        errors.append(_error("launch_topology", exc))

    tool_paths: dict[str, Path] = {}
    try:
        for name in ("nvcc", "ptxas", "nvdisasm"):
            path = _tool(name)
            tool_paths[name] = path
            tools[name] = {"path": str(path), "version": _version(path, cwd=output_dir)}
    except Exception as exc:
        errors.append(_error("cuda_tools", exc))
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    try:
        compile_options = _compile_options(cuda_source)
    except Exception as exc:
        errors.append(_error("cuda_dependencies", exc))
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    nvcc_command = [
        tool_paths["nvcc"],
        "--ptx",
        "-O3",
        f"-arch={selected_arch}",
        *compile_options,
        "-o",
        "kernel.nvcc.ptx",
        "kernel.cu",
    ]
    nvcc_result = _run(nvcc_command, cwd=output_dir)
    (output_dir / "nvcc.txt").write_text(nvcc_result.stdout, encoding="utf-8")
    raw_ptx_path = output_dir / "kernel.nvcc.ptx"
    if nvcc_result.returncode != 0 or not raw_ptx_path.is_file():
        errors.append(
            _error(
                "nvcc_ptx", f"nvcc exited {nvcc_result.returncode}: {nvcc_result.stdout.strip()}"
            )
        )
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    raw_ptx = raw_ptx_path.read_text(encoding="utf-8")
    (output_dir / "kernel.ptx").write_text(_canonical_ptx(raw_ptx), encoding="utf-8")

    ptxas_command = [
        tool_paths["ptxas"],
        *_ptxas_options(),
        f"-arch={selected_arch}",
        "kernel.ptx",
        "-o",
        "kernel.cubin",
    ]
    ptxas_result = _run(ptxas_command, cwd=output_dir)
    (output_dir / "ptxas.raw.txt").write_text(ptxas_result.stdout, encoding="utf-8")
    stable_ptxas_log, timing_samples = _stable_ptxas_log(ptxas_result.stdout)
    (output_dir / "ptxas.txt").write_text(stable_ptxas_log, encoding="utf-8")
    (output_dir / "timing.json").write_text(
        json.dumps({"ptxas_compile_time_ms": timing_samples}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if ptxas_result.returncode != 0 or not (output_dir / "kernel.cubin").is_file():
        errors.append(
            _error(
                "ptxas", f"ptxas exited {ptxas_result.returncode}: {ptxas_result.stdout.strip()}"
            )
        )
        report = make_report()
        _write_manifest(report, output_dir)
        return report

    try:
        resources.update(parse_ptxas_verbose(stable_ptxas_log))
        launch_functions = {launch["function"] for launch in launches}
        missing_resources = sorted(launch_functions - resources.keys())
        if missing_resources:
            raise ValueError(
                "ptxas output is missing launched function(s): " + ", ".join(missing_resources)
            )
    except Exception as exc:
        errors.append(_error("ptxas_resources", exc))

    disassembly = _run_disassembler(
        [tool_paths["nvdisasm"], "--print-code", "--print-instruction-encoding", "kernel.cubin"],
        cwd=output_dir,
    )
    if disassembly.returncode != 0 or not disassembly.stdout.strip():
        message = disassembly.stderr.strip() or disassembly.stdout.strip()
        errors.append(_error("sass", f"nvdisasm exited {disassembly.returncode}: {message}"))
    else:
        (output_dir / "kernel.sass").write_text(disassembly.stdout, encoding="utf-8")

    if (output_dir / "kernel.sass").is_file():
        try:
            ptx_source = (output_dir / "kernel.ptx").read_text(encoding="utf-8")
            sass_source = (output_dir / "kernel.sass").read_text(encoding="utf-8")
            instructions.update(analyze_instruction_features(cuda_source, ptx_source, sass_source))
        except Exception as exc:
            errors.append(_error("instruction_analysis", exc))

    report = make_report()
    _write_manifest(report, output_dir)
    return report


def _comparison_value(value: Any) -> Any:
    if value is _MISSING:
        return {"missing": True}
    return _json_value(value)


def _json_pointer(*parts: Any) -> str:
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _get(value: Any, key: str) -> Any:
    return value.get(key, _MISSING) if isinstance(value, Mapping) else _MISSING


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class _ComparisonBuilder:
    def __init__(self) -> None:
        self.checks: list[CodegenComparisonCheck] = []

    def add(
        self,
        path: str,
        rule: str,
        baseline: Any,
        candidate: Any,
        passed: bool,
        message: str,
        *,
        delta: int | None = None,
    ) -> None:
        self.checks.append(
            CodegenComparisonCheck(
                path=path,
                rule=rule,
                baseline=_comparison_value(baseline),
                candidate=_comparison_value(candidate),
                passed=passed,
                message=message,
                delta=delta,
            )
        )

    def valid(
        self, path: str, baseline: Any, candidate: Any, predicate: Any, description: str
    ) -> bool:
        baseline_valid = baseline is not _MISSING and predicate(baseline)
        candidate_valid = candidate is not _MISSING and predicate(candidate)
        passed = baseline_valid and candidate_valid
        self.add(
            path,
            "valid_evidence",
            baseline,
            candidate,
            passed,
            f"both manifests provide {description}"
            if passed
            else f"both manifests must provide {description}",
        )
        return passed

    def expected(
        self, path: str, baseline: Any, candidate: Any, expected: Any, description: str
    ) -> bool:
        passed = baseline == expected and candidate == expected
        self.add(
            path,
            "both_equal_expected",
            baseline,
            candidate,
            passed,
            f"both values are {description}" if passed else f"both values must be {description}",
        )
        return passed

    def unchanged(self, path: str, baseline: Any, candidate: Any, description: str) -> bool:
        passed = baseline is not _MISSING and candidate is not _MISSING and baseline == candidate
        self.add(
            path,
            "unchanged",
            baseline,
            candidate,
            passed,
            f"{description} is unchanged"
            if passed
            else f"{description} must be present and unchanged",
        )
        return passed

    def unchanged_integer(self, path: str, baseline: Any, candidate: Any, description: str) -> bool:
        valid = _is_nonnegative_integer(baseline) and _is_nonnegative_integer(candidate)
        passed = valid and baseline == candidate
        self.add(
            path,
            "unchanged_nonnegative_integer",
            baseline,
            candidate,
            passed,
            f"{description} is unchanged"
            if passed
            else f"{description} must be present, nonnegative, and unchanged",
        )
        return passed

    def not_increased(self, path: str, baseline: Any, candidate: Any, description: str) -> bool:
        valid = _is_nonnegative_integer(baseline) and _is_nonnegative_integer(candidate)
        passed = valid and candidate <= baseline
        delta = candidate - baseline if valid else None
        self.add(
            path,
            "not_increased",
            baseline,
            candidate,
            passed,
            f"{description} did not increase"
            if passed
            else f"{description} must be present, nonnegative, and may not increase",
            delta=delta,
        )
        return passed

    def observed_integer(self, path: str, baseline: Any, candidate: Any, description: str) -> bool:
        valid = _is_nonnegative_integer(baseline) and _is_nonnegative_integer(candidate)
        delta = candidate - baseline if valid else None
        self.add(
            path,
            "observed_nonnegative_integer",
            baseline,
            candidate,
            valid,
            f"{description} change is recorded"
            if valid
            else f"{description} must be present and nonnegative",
            delta=delta,
        )
        return valid

    def report(self, inputs: dict[str, Any]) -> CodegenComparisonReport:
        checks = tuple(sorted(self.checks, key=lambda item: (item.path, item.rule)))
        return CodegenComparisonReport(inputs=inputs, checks=checks)


def _source_label(value: Any) -> str:
    if isinstance(value, Mapping):
        return "mapping"
    try:
        return Path(os.fspath(value)).as_posix()
    except TypeError:
        return f"<{type(value).__name__}>"


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        manifest, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_codegen_manifest(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _source_label(value)
    if isinstance(value, Mapping):
        manifest = json.loads(json.dumps(dict(value), allow_nan=False))
    else:
        path = _guard_path(os.fspath(value), purpose="reading codegen evidence")
        manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("codegen manifest root must be a JSON object")
    return manifest, {"source": source, "sha256": _manifest_digest(manifest), "loaded": True}


def _mapping_pair(
    builder: _ComparisonBuilder, path: str, baseline: Any, candidate: Any, *, nonempty: bool = False
) -> bool:
    def valid(value: Any) -> bool:
        return isinstance(value, Mapping) and (bool(value) or not nonempty)

    description = "a non-empty JSON object" if nonempty else "a JSON object"
    return builder.valid(path, baseline, candidate, valid, description)


def _valid_launches(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    keys: set[tuple[str, int]] = set()
    for launch in value:
        if not isinstance(launch, Mapping):
            return False
        function = launch.get("function")
        index = launch.get("launch_index")
        if not isinstance(function, str) or not function or not _is_nonnegative_integer(index):
            return False
        key = (function, index)
        if key in keys:
            return False
        keys.add(key)
    return True


def _launch_index(value: list[Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {(launch["function"], launch["launch_index"]): launch for launch in value}


def _compare_launches(builder: _ComparisonBuilder, baseline: Any, candidate: Any) -> None:
    if not builder.valid(
        "/launches",
        baseline,
        candidate,
        _valid_launches,
        "a non-empty launch list with unique function/index keys",
    ):
        return
    baseline_index = _launch_index(baseline)
    candidate_index = _launch_index(candidate)
    baseline_keys = sorted(baseline_index)
    candidate_keys = sorted(candidate_index)
    builder.unchanged(
        "/launches/keys",
        [list(key) for key in baseline_keys],
        [list(key) for key in candidate_keys],
        "launched function/index set",
    )

    for function, index in sorted(baseline_index.keys() & candidate_index.keys()):
        baseline_launch = baseline_index[(function, index)]
        candidate_launch = candidate_index[(function, index)]
        prefix = _json_pointer("launches", function, index)
        for field in _LAUNCH_COMPARISON_FIELDS:
            baseline_value = _get(baseline_launch, field)
            candidate_value = _get(candidate_launch, field)
            path = f"{prefix}/{field}"
            if field in {"grid", "block", "cluster", "preferred_cluster"}:
                builder.valid(
                    path,
                    baseline_value,
                    candidate_value,
                    lambda value: isinstance(value, list) and len(value) == 3,
                    "a three-axis launch dimension",
                )
            elif field == "dynamic_shared_memory_bytes":
                builder.valid(
                    path,
                    baseline_value,
                    candidate_value,
                    lambda value: (
                        _is_nonnegative_integer(value)
                        or (
                            isinstance(value, Mapping)
                            and isinstance(value.get("expression"), str)
                            and bool(value["expression"])
                        )
                    ),
                    "a dynamic shared-memory value",
                )
            elif field == "flags":
                builder.valid(
                    path,
                    baseline_value,
                    candidate_value,
                    lambda value: (
                        isinstance(value, Mapping)
                        and all(isinstance(item, bool) for item in value.values())
                        and {"cooperative_launch", "programmatic_dependent_launch"}.issubset(value)
                    ),
                    "launch flags",
                )
            else:
                builder.valid(
                    path,
                    baseline_value,
                    candidate_value,
                    lambda value: (
                        isinstance(value, list) and all(isinstance(item, str) for item in value)
                    ),
                    "launch parameter tags",
                )
            builder.unchanged(path, baseline_value, candidate_value, field.replace("_", " "))


def _valid_resources(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(name, str) and isinstance(entry, Mapping) for name, entry in value.items()
        )
    )


def _valid_memory_map(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(name, str) and _is_nonnegative_integer(amount) for name, amount in value.items()
    )


def _allocated_registers_per_warp(registers: Any, arch: Any) -> Any:
    if not _is_nonnegative_integer(registers) or not isinstance(arch, str):
        return _MISSING
    granularity = _REGISTER_ALLOCATION_GRANULARITY.get(arch)
    if granularity is None:
        return _MISSING
    registers_per_warp = registers * 32
    return ((registers_per_warp + granularity - 1) // granularity) * granularity


def _compare_resources(
    builder: _ComparisonBuilder,
    baseline: Any,
    candidate: Any,
    *,
    baseline_arch: Any,
    candidate_arch: Any,
) -> None:
    if not builder.valid(
        "/resources",
        baseline,
        candidate,
        _valid_resources,
        "non-empty per-function resource evidence",
    ):
        return
    baseline_functions = sorted(baseline)
    candidate_functions = sorted(candidate)
    builder.unchanged(
        "/resources/functions", baseline_functions, candidate_functions, "resource function set"
    )
    for function in sorted(set(baseline) & set(candidate)):
        baseline_entry = baseline[function]
        candidate_entry = candidate[function]
        prefix = _json_pointer("resources", function)
        builder.unchanged_integer(
            f"{prefix}/barriers",
            _get(baseline_entry, "barriers"),
            _get(candidate_entry, "barriers"),
            "barrier count",
        )
        baseline_registers = _get(baseline_entry, "registers")
        candidate_registers = _get(candidate_entry, "registers")
        builder.observed_integer(
            f"{prefix}/registers",
            baseline_registers,
            candidate_registers,
            "reported registers per thread",
        )
        builder.not_increased(
            f"{prefix}/allocated_registers_per_warp",
            _allocated_registers_per_warp(baseline_registers, baseline_arch),
            _allocated_registers_per_warp(candidate_registers, candidate_arch),
            "physical register allocation per warp",
        )
        for field in _RESOURCE_LIMIT_FIELDS:
            builder.not_increased(
                f"{prefix}/{field}",
                _get(baseline_entry, field),
                _get(candidate_entry, field),
                field.replace("_", " "),
            )

        baseline_cumulative = _get(baseline_entry, "cumulative_stack_size_bytes")
        candidate_cumulative = _get(candidate_entry, "cumulative_stack_size_bytes")
        if baseline_cumulative is not _MISSING or candidate_cumulative is not _MISSING:
            builder.not_increased(
                f"{prefix}/cumulative_stack_size_bytes",
                baseline_cumulative,
                candidate_cumulative,
                "cumulative stack size bytes",
            )

        baseline_memory = _get(baseline_entry, "memory_bytes")
        candidate_memory = _get(candidate_entry, "memory_bytes")
        memory_path = f"{prefix}/memory_bytes"
        if not builder.valid(
            memory_path,
            baseline_memory,
            candidate_memory,
            _valid_memory_map,
            "a per-kind memory byte map",
        ):
            continue
        for kind in sorted(set(baseline_memory) | set(candidate_memory)):
            baseline_present = kind in baseline_memory
            candidate_present = kind in candidate_memory
            baseline_amount = baseline_memory.get(kind, 0)
            candidate_amount = candidate_memory.get(kind, 0)
            valid = _is_nonnegative_integer(baseline_amount) and _is_nonnegative_integer(
                candidate_amount
            )
            passed = (
                valid
                and (not candidate_present or baseline_present)
                and (candidate_amount <= baseline_amount)
            )
            builder.add(
                f"{memory_path}/{str(kind).replace('~', '~0').replace('/', '~1')}",
                "not_new_or_increased",
                baseline_amount if baseline_present else _MISSING,
                candidate_amount if candidate_present else _MISSING,
                passed,
                "memory allocation did not newly appear or increase"
                if passed
                else "memory allocation may not newly appear or increase",
                delta=candidate_amount - baseline_amount if valid else None,
            )


def _valid_histogram(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    count = value.get("count")
    values = value.get("values")
    return (
        _is_nonnegative_integer(count)
        and isinstance(values, Mapping)
        and all(
            isinstance(name, str) and _is_nonnegative_integer(amount)
            for name, amount in values.items()
        )
        and count == sum(values.values())
    )


def _compare_histogram(
    builder: _ComparisonBuilder, path: str, baseline: Any, candidate: Any, description: str
) -> None:
    if not builder.valid(
        path, baseline, candidate, _valid_histogram, f"a self-consistent {description} histogram"
    ):
        return
    builder.unchanged_integer(
        f"{path}/count", baseline["count"], candidate["count"], f"{description} count"
    )
    builder.unchanged(
        f"{path}/values", baseline["values"], candidate["values"], f"{description} values"
    )


def _compare_family_counts(
    builder: _ComparisonBuilder,
    path: str,
    baseline: Any,
    candidate: Any,
    expected_families: Sequence[str],
    *,
    observed_families: Sequence[str] = (),
    nonincreasing_families: Sequence[str] = (),
) -> None:
    if not _mapping_pair(builder, path, baseline, candidate, nonempty=True):
        return
    expected = sorted(expected_families)
    builder.expected(
        f"{path}/keys",
        sorted(baseline),
        sorted(candidate),
        expected,
        "the comparison schema's semantic family set",
    )
    observed = set(observed_families)
    nonincreasing = set(nonincreasing_families)
    if not observed.isdisjoint(nonincreasing):
        raise ValueError("an instruction family cannot have two comparison policies")
    if not (observed | nonincreasing) <= set(expected):
        raise ValueError("instruction family policy names must be in the expected family set")
    for family in expected:
        baseline_family = _get(baseline, family)
        candidate_family = _get(candidate, family)
        family_path = f"{path}/{family}"
        if not _mapping_pair(
            builder, family_path, baseline_family, candidate_family, nonempty=True
        ):
            continue
        count_path = f"{family_path}/count"
        baseline_count = _get(baseline_family, "count")
        candidate_count = _get(candidate_family, "count")
        description = f"{family} semantic-family count"
        if family in observed:
            builder.observed_integer(count_path, baseline_count, candidate_count, description)
        elif family in nonincreasing:
            builder.not_increased(count_path, baseline_count, candidate_count, description)
        else:
            builder.unchanged_integer(count_path, baseline_count, candidate_count, description)


def _compare_instructions(builder: _ComparisonBuilder, baseline: Any, candidate: Any) -> None:
    if not _mapping_pair(builder, "/instructions", baseline, candidate, nonempty=True):
        return
    baseline_cuda = _get(baseline, "cuda")
    candidate_cuda = _get(candidate, "cuda")
    if _mapping_pair(builder, "/instructions/cuda", baseline_cuda, candidate_cuda, nonempty=True):
        builder.observed_integer(
            "/instructions/cuda/tensor_map_type_occurrences",
            _get(baseline_cuda, "tensor_map_type_occurrences"),
            _get(candidate_cuda, "tensor_map_type_occurrences"),
            "CUDA tensor-map type occurrence count",
        )

    baseline_ptx = _get(baseline, "ptx")
    candidate_ptx = _get(candidate, "ptx")
    if _mapping_pair(builder, "/instructions/ptx", baseline_ptx, candidate_ptx, nonempty=True):
        _compare_family_counts(
            builder,
            "/instructions/ptx/families",
            _get(baseline_ptx, "families"),
            _get(candidate_ptx, "families"),
            _PTX_SEMANTIC_FAMILIES,
            observed_families=("direct_memory",),
        )
        _compare_histogram(
            builder,
            "/instructions/ptx/cache_modifiers",
            _get(baseline_ptx, "cache_modifiers"),
            _get(candidate_ptx, "cache_modifiers"),
            "PTX cache-modifier",
        )
        _compare_histogram(
            builder,
            "/instructions/ptx/memory_order_qualifiers",
            _get(baseline_ptx, "memory_order_qualifiers"),
            _get(candidate_ptx, "memory_order_qualifiers"),
            "PTX memory-order qualifier",
        )

    baseline_sass = _get(baseline, "sass")
    candidate_sass = _get(candidate, "sass")
    if _mapping_pair(builder, "/instructions/sass", baseline_sass, candidate_sass, nonempty=True):
        _compare_family_counts(
            builder,
            "/instructions/sass/families",
            _get(baseline_sass, "families"),
            _get(candidate_sass, "families"),
            _SASS_SEMANTIC_FAMILIES,
            observed_families=("direct_memory",),
        )


def _compare_manifest_payloads(
    builder: _ComparisonBuilder, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    builder.expected(
        "/schema_version",
        _get(baseline, "schema_version"),
        _get(candidate, "schema_version"),
        _SCHEMA_VERSION,
        f"schema version {_SCHEMA_VERSION}",
    )
    builder.expected(
        "/status", _get(baseline, "status"), _get(candidate, "status"), "complete", "complete"
    )
    builder.expected("/ok", _get(baseline, "ok"), _get(candidate, "ok"), True, "true")
    builder.expected(
        "/errors", _get(baseline, "errors"), _get(candidate, "errors"), [], "an empty list"
    )

    baseline_kernel = _get(baseline, "kernel")
    candidate_kernel = _get(candidate, "kernel")
    if builder.valid(
        "/kernel",
        baseline_kernel,
        candidate_kernel,
        lambda value: (
            isinstance(value, Mapping)
            and isinstance(value.get("name"), str)
            and bool(value["name"])
        ),
        "kernel identity evidence",
    ):
        builder.unchanged("/kernel", baseline_kernel, candidate_kernel, "kernel identity")

    baseline_config = _get(baseline, "config")
    candidate_config = _get(candidate, "config")
    if builder.valid(
        "/config",
        baseline_config,
        candidate_config,
        lambda value: (
            isinstance(value, Mapping)
            and isinstance(value.get("label"), str)
            and bool(value["label"])
        ),
        "declared configuration evidence",
    ):
        builder.unchanged("/config", baseline_config, candidate_config, "declared configuration")

    baseline_target = _get(baseline, "target")
    candidate_target = _get(candidate, "target")
    if _mapping_pair(builder, "/target", baseline_target, candidate_target, nonempty=True):
        builder.valid(
            "/target/arch",
            _get(baseline_target, "arch"),
            _get(candidate_target, "arch"),
            lambda value: isinstance(value, str) and _ARCH_RE.fullmatch(value) is not None,
            "a CUDA architecture",
        )
        builder.unchanged(
            "/target/arch",
            _get(baseline_target, "arch"),
            _get(candidate_target, "arch"),
            "target architecture",
        )
        builder.expected(
            "/target/kind",
            _get(baseline_target, "kind"),
            _get(candidate_target, "kind"),
            "cuda",
            "cuda",
        )
        builder.expected(
            "/target/pipeline",
            _get(baseline_target, "pipeline"),
            _get(candidate_target, "pipeline"),
            "tirx",
            "tirx",
        )
        builder.valid(
            "/target/compute_capability",
            _get(baseline_target, "compute_capability"),
            _get(candidate_target, "compute_capability"),
            _is_nonnegative_integer,
            "a nonnegative compute capability",
        )
        builder.unchanged(
            "/target/compute_capability",
            _get(baseline_target, "compute_capability"),
            _get(candidate_target, "compute_capability"),
            "compute capability",
        )

    _compare_launches(builder, _get(baseline, "launches"), _get(candidate, "launches"))
    _compare_resources(
        builder,
        _get(baseline, "resources"),
        _get(candidate, "resources"),
        baseline_arch=_get(baseline_target, "arch")
        if isinstance(baseline_target, Mapping)
        else _MISSING,
        candidate_arch=_get(candidate_target, "arch")
        if isinstance(candidate_target, Mapping)
        else _MISSING,
    )
    _compare_instructions(builder, _get(baseline, "instructions"), _get(candidate, "instructions"))


def compare_codegen_manifests(
    baseline: Mapping[str, Any] | str | os.PathLike[str],
    candidate: Mapping[str, Any] | str | os.PathLike[str],
) -> CodegenComparisonReport:
    """Compare two audit manifests using AC-8's observable hard-gate policy.

    Inputs may be decoded manifest mappings or paths to existing manifests.
    Unreadable, incomplete, malformed, or mismatched evidence is represented by
    a failing report rather than accepted as absent/zero evidence.  PTX and SASS
    typed mnemonics may change; instruction semantics are compared at the family,
    cache-modifier, and memory-order layers.
    """
    builder = _ComparisonBuilder()
    manifests: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        try:
            manifest, descriptor = _load_codegen_manifest(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            descriptor = {
                "source": _source_label(value),
                "sha256": None,
                "loaded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            manifests[name] = manifest
        inputs[name] = descriptor

    baseline_loaded = "baseline" in manifests
    candidate_loaded = "candidate" in manifests
    builder.add(
        "/inputs",
        "readable_json_manifests",
        baseline_loaded,
        candidate_loaded,
        baseline_loaded and candidate_loaded,
        "both inputs are readable JSON manifests"
        if baseline_loaded and candidate_loaded
        else "both inputs must be readable JSON manifests",
    )
    if baseline_loaded and candidate_loaded:
        _compare_manifest_payloads(builder, manifests["baseline"], manifests["candidate"])
    return builder.report(inputs)


def _write_comparison(report: CodegenComparisonReport, output: str | os.PathLike[str]) -> None:
    output_path = _guard_path(output, purpose="writing codegen comparison")
    if output_path.exists():
        raise FileExistsError(f"comparison output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Compile one registry config or hard-gate two existing CUDA codegen manifests")
    )
    parser.add_argument("--kernel", help="Exact public registry kernel name")
    parser.add_argument("--config", help="Exact label from that module's CONFIGS")
    parser.add_argument("--baseline", help="Existing baseline manifest to compare")
    parser.add_argument("--candidate", help="Existing candidate manifest to compare")
    parser.add_argument(
        "--output", required=True, help="New comparison JSON file or new/empty evidence directory"
    )
    parser.add_argument("--arch", help="CUDA architecture, defaulted from KERNEL_META")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: Mapping[str, Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the codegen evidence CLI and return 0/1/2 for pass/fail/usage."""
    args = _parser().parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    comparison_mode = args.baseline is not None or args.candidate is not None
    if comparison_mode:
        if args.baseline is None or args.candidate is None:
            print("ERROR: comparison mode requires both --baseline and --candidate", file=stderr)
            return 2
        incompatible = [
            flag
            for flag, value in (
                ("--kernel", args.kernel),
                ("--config", args.config),
                ("--arch", args.arch),
            )
            if value is not None
        ]
        if incompatible:
            print("ERROR: comparison mode does not accept " + ", ".join(incompatible), file=stderr)
            return 2
        report = compare_codegen_manifests(args.baseline, args.candidate)
        try:
            _write_comparison(report, args.output)
        except (FileExistsError, OSError, TypeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=stderr)
            return 2
        rendered = (
            json.dumps(report.to_dict(), indent=2, sort_keys=True)
            if args.format == "json"
            else report.summary()
        )
        print(rendered, file=stdout)
        return 0 if report.ok else 1

    if args.kernel is None or args.config is None:
        print("ERROR: compile mode requires both --kernel and --config", file=stderr)
        return 2
    try:
        report = audit_registered_codegen(
            args.kernel, args.config, args.output, registry=registry, arch=args.arch
        )
    except (CodegenAuditSelectionError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    rendered = (
        json.dumps(report.to_dict(), indent=2, sort_keys=True)
        if args.format == "json"
        else report.summary()
    )
    print(rendered, file=stdout)
    return 0 if report.ok else 1


__all__ = [
    "CodegenAuditError",
    "CodegenAuditReport",
    "CodegenAuditSelectionError",
    "CodegenComparisonCheck",
    "CodegenComparisonReport",
    "analyze_instruction_features",
    "audit_registered_codegen",
    "compare_codegen_manifests",
    "extract_launch_topology",
    "main",
    "parse_ptxas_verbose",
]


if __name__ == "__main__":
    raise SystemExit(main())
