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
"""Capture and compare the public, pre-lowering kernel registry contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import tvm
from tvm import tirx

_SCHEMA_VERSION = 1
_REQUIRED_METADATA_TYPES = {"name": str, "category": str, "compute_capability": int}


class RegistryManifestError(ValueError):
    """Base class for failures that make a registry manifest unusable."""


class RegistryManifestValidationError(RegistryManifestError):
    """Raised when a module contract or serialized manifest is malformed."""


class RegistryManifestCaptureError(RegistryManifestError):
    """Raised when a public kernel cannot be constructed for the manifest."""


@dataclass(frozen=True)
class ManifestDifference:
    """One observable incompatibility between two registry manifests."""

    code: str
    path: str
    message: str
    before: Any = None
    after: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistryCompatibilityReport:
    """Compatibility result for a baseline and candidate manifest."""

    before_summary: dict[str, int]
    after_summary: dict[str, int]
    differences: tuple[ManifestDifference, ...]

    @property
    def compatible(self) -> bool:
        return not self.differences

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "compatible": self.compatible,
            "before_summary": self.before_summary,
            "after_summary": self.after_summary,
            "differences": [difference.to_dict() for difference in self.differences],
        }

    def summary(self) -> str:
        if self.compatible:
            return (
                "PASS registry compatibility: "
                f"{self.after_summary['kernels']} kernels, "
                f"{self.after_summary['configs']} configs, "
                f"{self.after_summary['functions']} pre-lowering functions"
            )

        lines = [f"FAIL registry compatibility: {len(self.differences)} difference(s)"]
        lines.extend(
            f"- {difference.path}: {difference.message}" for difference in self.differences
        )
        return "\n".join(lines)


def _validation_error(path: str, message: str) -> RegistryManifestValidationError:
    return RegistryManifestValidationError(f"{path}: {message}")


def _json_value(value: Any, path: str) -> Any:
    """Normalize a value to strict JSON data, rejecting lossy object encodings."""
    try:
        rendered = json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True)
        return json.loads(rendered)
    except (TypeError, ValueError) as exc:
        raise _validation_error(path, f"value is not JSON serializable: {exc}") from exc


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise _validation_error(path, "object keys must be strings")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing keys {missing!r}")
        if unexpected:
            details.append(f"unexpected keys {unexpected!r}")
        raise _validation_error(path, "; ".join(details))


def _validate_string_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _validation_error(path, f"expected an object, got {type(value).__name__}")
    if any(not isinstance(key, str) for key in value):
        raise _validation_error(path, "object keys must be strings")
    normalized = _json_value(dict(value), path)
    if not isinstance(normalized, dict):
        raise _validation_error(path, "expected an object")
    return normalized


def _public_metadata(registry_name: str, module: Any) -> dict[str, Any]:
    raw_metadata = getattr(module, "KERNEL_META", None)
    if not isinstance(raw_metadata, Mapping):
        raise _validation_error(
            f"kernel[{registry_name!r}].metadata", "KERNEL_META must be a mapping"
        )
    if any(not isinstance(key, str) for key in raw_metadata):
        raise _validation_error(
            f"kernel[{registry_name!r}].metadata", "KERNEL_META keys must be strings"
        )

    metadata = {key: value for key, value in raw_metadata.items() if not key.startswith("_")}
    for field, expected_type in _REQUIRED_METADATA_TYPES.items():
        value = metadata.get(field)
        if not isinstance(value, expected_type) or isinstance(value, bool):
            raise _validation_error(
                f"kernel[{registry_name!r}].metadata.{field}", f"must be a {expected_type.__name__}"
            )
        if expected_type is str and not value:
            raise _validation_error(
                f"kernel[{registry_name!r}].metadata.{field}", "must not be empty"
            )
    if metadata["name"] != registry_name:
        raise _validation_error(
            f"kernel[{registry_name!r}].metadata.name",
            f"must match registry name {registry_name!r}",
        )
    return _validate_string_mapping(metadata, f"kernel[{registry_name!r}].metadata")


def _global_name(global_var: Any) -> str:
    return str(getattr(global_var, "name_hint", global_var))


def _describe_kernel_return(value: Any, path: str) -> tuple[dict[str, Any], int]:
    if isinstance(value, tirx.PrimFunc):
        return {"kind": "prim_func"}, 1

    if isinstance(value, tvm.IRModule):
        global_names: list[str] = []
        for global_var, base_func in value.functions.items():
            name = _global_name(global_var)
            if not isinstance(base_func, tirx.PrimFunc):
                raise _validation_error(
                    f"{path}.{name}", f"expected tvm.tirx.PrimFunc, got {type(base_func).__name__}"
                )
            global_names.append(name)
        if not global_names:
            raise _validation_error(path, "IRModule must contain at least one TIRx PrimFunc")
        if len(set(global_names)) != len(global_names):
            raise _validation_error(path, "IRModule contains duplicate global names")
        global_names.sort()
        return {"kind": "ir_module", "globals": global_names}, len(global_names)

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _validation_error(path, "returned mapping keys must be strings")
        entries = []
        function_count = 0
        for key in sorted(value):
            container, count = _describe_kernel_return(value[key], f"{path}[{key!r}]")
            entries.append({"key": key, "container": container})
            function_count += count
        return {"kind": "mapping", "entries": entries}, function_count

    if isinstance(value, list | tuple):
        items = []
        function_count = 0
        for index, item in enumerate(value):
            container, count = _describe_kernel_return(item, f"{path}[{index}]")
            items.append(container)
            function_count += count
        kind = "list" if isinstance(value, list) else "tuple"
        return {"kind": kind, "items": items}, function_count

    raise _validation_error(
        path,
        f"expected a TIRx PrimFunc, IRModule, or nested list/tuple/mapping; "
        f"got {type(value).__name__}",
    )


def _capture_config(
    registry_name: str, config: Any, get_kernel: Any, *, index: int
) -> dict[str, Any]:
    path = f"kernel[{registry_name!r}].configs[{index}]"
    if not isinstance(config, Mapping):
        raise _validation_error(path, f"must be a mapping, got {type(config).__name__}")
    if any(not isinstance(key, str) for key in config):
        raise _validation_error(path, "configuration keys must be strings")
    label = config.get("label")
    if not isinstance(label, str) or not label:
        raise _validation_error(f"{path}.label", "must be a non-empty string")

    raw_parameters = {key: value for key, value in config.items() if key != "label"}
    parameters = _validate_string_mapping(raw_parameters, f"{path}.parameters")
    try:
        value = get_kernel(**raw_parameters)
    except Exception as exc:
        raise RegistryManifestCaptureError(
            f"{path}: get_kernel failed with {type(exc).__name__}: {exc}"
        ) from exc
    container, function_count = _describe_kernel_return(value, f"{path}.return")
    if function_count == 0:
        raise _validation_error(f"{path}.return", "must contain at least one TIRx PrimFunc")
    return {
        "label": label,
        "parameters": parameters,
        "result": {"function_count": function_count, "container": container},
    }


def _capture_kernel(registry_name: str, module: Any) -> dict[str, Any]:
    metadata = _public_metadata(registry_name, module)
    configs = getattr(module, "CONFIGS", None)
    if not isinstance(configs, list) or not configs:
        raise _validation_error(
            f"kernel[{registry_name!r}].configs", "CONFIGS must be a non-empty list"
        )
    get_kernel = getattr(module, "get_kernel", None)
    if not callable(get_kernel):
        raise _validation_error(
            f"kernel[{registry_name!r}].get_kernel", "get_kernel must be callable"
        )

    captured = [
        _capture_config(registry_name, config, get_kernel, index=index)
        for index, config in enumerate(configs)
    ]
    labels = [config["label"] for config in captured]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise _validation_error(
            f"kernel[{registry_name!r}].configs",
            f"configuration labels must be unique; duplicates: {duplicates!r}",
        )
    captured.sort(key=lambda config: config["label"])
    return {"name": registry_name, "metadata": metadata, "configs": captured}


def _manifest_summary(kernels: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    configs = [config for kernel in kernels for config in kernel["configs"]]
    return {
        "kernels": len(kernels),
        "configs": len(configs),
        "functions": sum(config["result"]["function_count"] for config in configs),
    }


def capture_registry_manifest(registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Construct every public config and return a deterministic registry manifest.

    When ``registry`` is omitted, discovery is strict so an import failure cannot
    silently reduce coverage.  Each public ``get_kernel`` is called directly and
    its return value is inspected before any lowering or compilation.
    """
    if registry is None:
        from tirx_kernels.registry import discover_kernels

        try:
            registry = discover_kernels(strict=True)
        except Exception as exc:
            raise RegistryManifestCaptureError(
                f"registry discovery failed with {type(exc).__name__}: {exc}"
            ) from exc
    if not isinstance(registry, Mapping):
        raise _validation_error("registry", "must be a mapping from names to kernel modules")
    if not registry:
        raise _validation_error("registry", "must contain at least one kernel")
    if any(not isinstance(name, str) or not name for name in registry):
        raise _validation_error("registry", "kernel names must be non-empty strings")

    kernels = [_capture_kernel(name, registry[name]) for name in sorted(registry)]
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "summary": _manifest_summary(kernels),
        "kernels": kernels,
    }
    return validate_registry_manifest(manifest)


def _validate_container_shape(value: Any, path: str) -> tuple[dict[str, Any], int]:
    if not isinstance(value, Mapping):
        raise _validation_error(path, "container shape must be an object")
    if any(not isinstance(key, str) for key in value):
        raise _validation_error(path, "container shape keys must be strings")
    kind = value.get("kind")
    if kind == "prim_func":
        _require_exact_keys(value, {"kind"}, path)
        return {"kind": "prim_func"}, 1
    if kind == "ir_module":
        _require_exact_keys(value, {"kind", "globals"}, path)
        globals_value = value["globals"]
        if not isinstance(globals_value, list) or not globals_value:
            raise _validation_error(path, "IRModule globals must be a non-empty list")
        if any(not isinstance(name, str) or not name for name in globals_value):
            raise _validation_error(path, "IRModule global names must be non-empty strings")
        if len(set(globals_value)) != len(globals_value):
            raise _validation_error(path, "IRModule global names must be unique")
        names = sorted(globals_value)
        return {"kind": "ir_module", "globals": names}, len(names)
    if kind in {"list", "tuple"}:
        _require_exact_keys(value, {"kind", "items"}, path)
        items_value = value["items"]
        if not isinstance(items_value, list):
            raise _validation_error(path, "container items must be a list")
        items = []
        count = 0
        for index, item in enumerate(items_value):
            normalized, item_count = _validate_container_shape(item, f"{path}.items[{index}]")
            items.append(normalized)
            count += item_count
        return {"kind": kind, "items": items}, count
    if kind == "mapping":
        _require_exact_keys(value, {"kind", "entries"}, path)
        entries_value = value["entries"]
        if not isinstance(entries_value, list):
            raise _validation_error(path, "mapping entries must be a list")
        entries = []
        keys = []
        count = 0
        for index, entry in enumerate(entries_value):
            entry_path = f"{path}.entries[{index}]"
            if not isinstance(entry, Mapping):
                raise _validation_error(entry_path, "mapping entry must be an object")
            _require_exact_keys(entry, {"key", "container"}, entry_path)
            key = entry["key"]
            if not isinstance(key, str):
                raise _validation_error(f"{entry_path}.key", "must be a string")
            normalized, item_count = _validate_container_shape(
                entry["container"], f"{entry_path}.container"
            )
            keys.append(key)
            entries.append({"key": key, "container": normalized})
            count += item_count
        if len(set(keys)) != len(keys):
            raise _validation_error(path, "mapping entry keys must be unique")
        entries.sort(key=lambda entry: entry["key"])
        return {"kind": "mapping", "entries": entries}, count
    raise _validation_error(path, f"unknown container kind {kind!r}")


def _validate_serialized_config(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _validation_error(path, "configuration must be an object")
    _require_exact_keys(value, {"label", "parameters", "result"}, path)
    label = value["label"]
    if not isinstance(label, str) or not label:
        raise _validation_error(f"{path}.label", "must be a non-empty string")
    parameters = _validate_string_mapping(value["parameters"], f"{path}.parameters")

    result = value["result"]
    if not isinstance(result, Mapping):
        raise _validation_error(f"{path}.result", "must be an object")
    _require_exact_keys(result, {"function_count", "container"}, f"{path}.result")
    function_count = result["function_count"]
    if (
        not isinstance(function_count, int)
        or isinstance(function_count, bool)
        or function_count < 1
    ):
        raise _validation_error(f"{path}.result.function_count", "must be a positive integer")
    container, counted_functions = _validate_container_shape(
        result["container"], f"{path}.result.container"
    )
    if function_count != counted_functions:
        raise _validation_error(
            f"{path}.result.function_count",
            f"declares {function_count}, but container describes {counted_functions}",
        )
    return {
        "label": label,
        "parameters": parameters,
        "result": {"function_count": function_count, "container": container},
    }


def _validate_serialized_kernel(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _validation_error(path, "kernel must be an object")
    _require_exact_keys(value, {"name", "metadata", "configs"}, path)
    name = value["name"]
    if not isinstance(name, str) or not name:
        raise _validation_error(f"{path}.name", "must be a non-empty string")
    metadata = _validate_string_mapping(value["metadata"], f"{path}.metadata")
    if any(key.startswith("_") for key in metadata):
        raise _validation_error(f"{path}.metadata", "must contain only public fields")
    for field, expected_type in _REQUIRED_METADATA_TYPES.items():
        field_value = metadata.get(field)
        if not isinstance(field_value, expected_type) or isinstance(field_value, bool):
            raise _validation_error(
                f"{path}.metadata.{field}", f"must be a {expected_type.__name__}"
            )
        if expected_type is str and not field_value:
            raise _validation_error(f"{path}.metadata.{field}", "must not be empty")
    if metadata["name"] != name:
        raise _validation_error(f"{path}.metadata.name", "must match the kernel name")

    configs_value = value["configs"]
    if not isinstance(configs_value, list) or not configs_value:
        raise _validation_error(f"{path}.configs", "must be a non-empty list")
    configs = [
        _validate_serialized_config(config, f"{path}.configs[{index}]")
        for index, config in enumerate(configs_value)
    ]
    labels = [config["label"] for config in configs]
    if len(set(labels)) != len(labels):
        raise _validation_error(f"{path}.configs", "configuration labels must be unique")
    configs.sort(key=lambda config: config["label"])
    return {"name": name, "metadata": metadata, "configs": configs}


def validate_registry_manifest(value: Any) -> dict[str, Any]:
    """Validate and canonically order a serialized registry manifest."""
    if not isinstance(value, Mapping):
        raise _validation_error("manifest", "must be an object")
    _require_exact_keys(value, {"schema_version", "summary", "kernels"}, "manifest")
    schema_version = value["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _SCHEMA_VERSION
    ):
        raise _validation_error(
            "manifest.schema_version", f"expected {_SCHEMA_VERSION}, got {schema_version!r}"
        )
    kernels_value = value["kernels"]
    if not isinstance(kernels_value, list) or not kernels_value:
        raise _validation_error("manifest.kernels", "must be a non-empty list")
    kernels = [
        _validate_serialized_kernel(kernel, f"manifest.kernels[{index}]")
        for index, kernel in enumerate(kernels_value)
    ]
    names = [kernel["name"] for kernel in kernels]
    if len(set(names)) != len(names):
        raise _validation_error("manifest.kernels", "kernel names must be unique")
    kernels.sort(key=lambda kernel: kernel["name"])

    expected_summary = _manifest_summary(kernels)
    summary = _validate_string_mapping(value["summary"], "manifest.summary")
    if summary != expected_summary:
        raise _validation_error(
            "manifest.summary", f"expected derived summary {expected_summary!r}, got {summary!r}"
        )
    return {"schema_version": _SCHEMA_VERSION, "summary": expected_summary, "kernels": kernels}


def load_registry_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a registry manifest JSON file."""
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RegistryManifestValidationError(f"cannot read {str(manifest_path)!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryManifestValidationError(
            f"{str(manifest_path)!r} is not valid JSON: {exc}"
        ) from exc
    return validate_registry_manifest(value)


def _difference(
    code: str, path: str, message: str, before: Any = None, after: Any = None
) -> ManifestDifference:
    return ManifestDifference(code=code, path=path, message=message, before=before, after=after)


def compare_registry_manifests(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> RegistryCompatibilityReport:
    """Compare two valid manifests for exact public registry compatibility."""
    canonical_before = validate_registry_manifest(before)
    canonical_after = validate_registry_manifest(after)
    before_kernels = {kernel["name"]: kernel for kernel in canonical_before["kernels"]}
    after_kernels = {kernel["name"]: kernel for kernel in canonical_after["kernels"]}
    differences: list[ManifestDifference] = []

    for name in sorted(before_kernels.keys() - after_kernels.keys()):
        differences.append(
            _difference(
                "kernel_missing",
                f"kernels[{name!r}]",
                "baseline kernel is missing from the candidate manifest",
                before=before_kernels[name],
            )
        )
    for name in sorted(after_kernels.keys() - before_kernels.keys()):
        differences.append(
            _difference(
                "kernel_added",
                f"kernels[{name!r}]",
                "candidate manifest contains a kernel absent from the baseline",
                after=after_kernels[name],
            )
        )

    for name in sorted(before_kernels.keys() & after_kernels.keys()):
        baseline_kernel = before_kernels[name]
        candidate_kernel = after_kernels[name]
        kernel_path = f"kernels[{name!r}]"
        if baseline_kernel["metadata"] != candidate_kernel["metadata"]:
            differences.append(
                _difference(
                    "metadata_changed",
                    f"{kernel_path}.metadata",
                    "public KERNEL_META fields changed",
                    baseline_kernel["metadata"],
                    candidate_kernel["metadata"],
                )
            )

        before_configs = {config["label"]: config for config in baseline_kernel["configs"]}
        after_configs = {config["label"]: config for config in candidate_kernel["configs"]}
        for label in sorted(before_configs.keys() - after_configs.keys()):
            differences.append(
                _difference(
                    "config_missing",
                    f"{kernel_path}.configs[{label!r}]",
                    "baseline configuration is missing from the candidate manifest",
                    before=before_configs[label],
                )
            )
        for label in sorted(after_configs.keys() - before_configs.keys()):
            differences.append(
                _difference(
                    "config_added",
                    f"{kernel_path}.configs[{label!r}]",
                    "candidate manifest contains a configuration absent from the baseline",
                    after=after_configs[label],
                )
            )
        for label in sorted(before_configs.keys() & after_configs.keys()):
            baseline_config = before_configs[label]
            candidate_config = after_configs[label]
            config_path = f"{kernel_path}.configs[{label!r}]"
            if baseline_config["parameters"] != candidate_config["parameters"]:
                differences.append(
                    _difference(
                        "parameters_changed",
                        f"{config_path}.parameters",
                        "configuration parameter protocol changed",
                        baseline_config["parameters"],
                        candidate_config["parameters"],
                    )
                )
            baseline_result = baseline_config["result"]
            candidate_result = candidate_config["result"]
            if baseline_result["function_count"] != candidate_result["function_count"]:
                differences.append(
                    _difference(
                        "function_count_changed",
                        f"{config_path}.result.function_count",
                        "pre-lowering PrimFunc count changed",
                        baseline_result["function_count"],
                        candidate_result["function_count"],
                    )
                )
            if baseline_result["container"] != candidate_result["container"]:
                differences.append(
                    _difference(
                        "container_shape_changed",
                        f"{config_path}.result.container",
                        "pre-lowering return container shape changed",
                        baseline_result["container"],
                        candidate_result["container"],
                    )
                )

    return RegistryCompatibilityReport(
        before_summary=canonical_before["summary"],
        after_summary=canonical_after["summary"],
        differences=tuple(differences),
    )


def render_registry_manifest(manifest: Mapping[str, Any]) -> str:
    """Render a validated manifest as deterministic JSON."""
    canonical = validate_registry_manifest(manifest)
    return json.dumps(canonical, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_or_print(rendered: str, output: str | None, stdout: TextIO) -> None:
    if output is None:
        print(rendered, file=stdout, end="")
        return
    try:
        Path(output).write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise RegistryManifestError(f"cannot write {output!r}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and compare the public pre-lowering kernel registry contract"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Capture the live public registry")
    export_parser.add_argument("--output", help="Write JSON to this file instead of stdout")

    compare_parser = subparsers.add_parser("compare", help="Compare before and after manifests")
    compare_parser.add_argument("before", help="Baseline manifest JSON")
    compare_parser.add_argument("after", help="Candidate manifest JSON")
    compare_parser.add_argument("--format", choices=("text", "json"), default="text")
    compare_parser.add_argument("--output", help="Write the report to this file instead of stdout")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: Mapping[str, Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the registry manifest CLI and return its process exit code."""
    args = _parser().parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        if args.command == "export":
            manifest = capture_registry_manifest(registry)
            _write_or_print(render_registry_manifest(manifest), args.output, stdout)
            return 0

        before = load_registry_manifest(args.before)
        after = load_registry_manifest(args.after)
        report = compare_registry_manifests(before, after)
        rendered = (
            json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            if args.format == "json"
            else report.summary() + "\n"
        )
        _write_or_print(rendered, args.output, stdout)
        return 0 if report.compatible else 1
    except RegistryManifestError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2


__all__ = [
    "ManifestDifference",
    "RegistryCompatibilityReport",
    "RegistryManifestCaptureError",
    "RegistryManifestError",
    "RegistryManifestValidationError",
    "capture_registry_manifest",
    "compare_registry_manifests",
    "load_registry_manifest",
    "main",
    "render_registry_manifest",
    "validate_registry_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
