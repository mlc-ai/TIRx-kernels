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
"""Audit every registered kernel configuration's pre-lowering TIRx IR."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from tirx_kernels.low_level_ir import LowLevelIRReport, inspect_low_level_ir

_SCHEMA_VERSION = 1
_MISSING = object()


class AuditSelectionError(ValueError):
    """Raised when a requested kernel or configuration does not exist."""


@dataclass(frozen=True)
class RegistryAuditError:
    """One structured error encountered outside the IR contract itself."""

    stage: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "type": self.error_type, "message": self.message}


@dataclass(frozen=True)
class ConfigAuditResult:
    """Audit result for one declared registry configuration."""

    index: int
    label: str | None
    parameters: dict[str, Any] | None
    status: str
    errors: tuple[RegistryAuditError, ...]
    ir: LowLevelIRReport | None

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "parameters": self.parameters,
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
            "ir": None if self.ir is None else _ir_report_dict(self.ir),
        }


@dataclass(frozen=True)
class KernelAuditResult:
    """Audit result for one registry module."""

    name: str
    module: str
    meta: dict[str, Any] | None
    declared_config_count: int
    audited_config_count: int
    complete: bool
    decision: str
    errors: tuple[RegistryAuditError, ...]
    configs: tuple[ConfigAuditResult, ...]

    @property
    def ok(self) -> bool:
        return not self.errors and all(config.ok for config in self.configs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "meta": self.meta,
            "declared_config_count": self.declared_config_count,
            "audited_config_count": self.audited_config_count,
            "complete": self.complete,
            "decision": self.decision,
            "errors": [error.to_dict() for error in self.errors],
            "configs": [config.to_dict() for config in self.configs],
        }


@dataclass(frozen=True)
class RegistryAuditReport:
    """Machine-readable result for a dynamic registry audit."""

    complete_registry: bool
    kernel_filter: str | None
    config_filter: str | None
    discovered_kernel_count: int
    registry_errors: tuple[RegistryAuditError, ...]
    kernels: tuple[KernelAuditResult, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.registry_errors
            and bool(self.kernels)
            and all(kernel.ok for kernel in self.kernels)
            and any(kernel.audited_config_count for kernel in self.kernels)
        )

    def counts(self) -> dict[str, int]:
        configs = [config for kernel in self.kernels for config in kernel.configs]
        ir_reports = [config.ir for config in configs if config.ir is not None]
        violations = [finding for report in ir_reports for finding in report.violations]
        return {
            "discovered_kernels": self.discovered_kernel_count,
            "selected_kernels": len(self.kernels),
            "declared_configs": sum(kernel.declared_config_count for kernel in self.kernels),
            "audited_configs": len(configs),
            "passed_configs": sum(config.status == "pass" for config in configs),
            "violation_configs": sum(config.status == "violation" for config in configs),
            "error_configs": sum(config.status == "error" for config in configs),
            "checked_functions": sum(len(report.checked_functions) for report in ir_reports),
            "violations": len(violations),
            "tile_primitives": sum(finding.kind == "tile_primitive" for finding in violations),
            "forbidden_loads": sum(finding.kind == "buffer_load" for finding in violations),
            "forbidden_stores": sum(finding.kind == "buffer_store" for finding in violations),
            "address_only_loads": sum(len(report.address_only_loads) for report in ir_reports),
            "registry_errors": len(self.registry_errors),
            "kernel_errors": sum(len(kernel.errors) for kernel in self.kernels),
            "skip_kernels": sum(kernel.decision == "skip" for kernel in self.kernels),
            "rewrite_kernels": sum(kernel.decision == "rewrite" for kernel in self.kernels),
            "incomplete_kernels": sum(kernel.decision == "incomplete" for kernel in self.kernels),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "scope": {
                "complete_registry": self.complete_registry,
                "kernel": self.kernel_filter,
                "config": self.config_filter,
            },
            "ok": self.ok,
            "summary": self.counts(),
            "registry_errors": [error.to_dict() for error in self.registry_errors],
            "kernels": [kernel.to_dict() for kernel in self.kernels],
        }

    def summary(self) -> str:
        counts = self.counts()
        scope = "full registry" if self.complete_registry else "filtered/incomplete registry"
        lines = [
            f"{'PASS' if self.ok else 'FAIL'} low-level IR audit ({scope}): "
            f"{counts['audited_configs']}/{counts['declared_configs']} configs, "
            f"{counts['checked_functions']} functions, {counts['violations']} violations, "
            f"{counts['error_configs']} config errors, {counts['kernel_errors']} kernel errors, "
            f"{counts['registry_errors']} registry errors"
        ]
        for error in self.registry_errors:
            lines.append(f"ERROR registry/{error.stage}: {error.error_type}: {error.message}")
        for kernel in self.kernels:
            lines.append(
                f"{kernel.decision.upper():10s} {kernel.name}: "
                f"{kernel.audited_config_count}/{kernel.declared_config_count} configs"
            )
            for error in kernel.errors:
                lines.append(f"  ERROR {error.stage}: {error.error_type}: {error.message}")
            for config in kernel.configs:
                if config.status == "pass":
                    continue
                label = config.label if config.label is not None else f"index {config.index}"
                lines.append(f"  {config.status.upper()} [{label}]")
                for error in config.errors:
                    lines.append(f"    {error.stage}: {error.error_type}: {error.message}")
                if config.ir is not None:
                    for finding in config.ir.violations:
                        scope_text = f", scope={finding.scope}" if finding.scope else ""
                        span_text = f", span={finding.span}" if finding.span else ""
                        lines.append(
                            f"    {finding.function}: {finding.kind} "
                            f"({finding.node_type}{scope_text}{span_text})"
                        )
        return "\n".join(lines)


def _error(stage: str, error: BaseException | str, error_type: str | None = None):
    if isinstance(error, BaseException):
        return RegistryAuditError(stage, type(error).__name__, str(error))
    return RegistryAuditError(stage, error_type or "ValueError", error)


def _json_value(value: Any) -> Any:
    """Return the value normalized to JSON types, or raise ``TypeError``."""
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"value is not JSON serializable: {exc}") from exc


def _ir_counts(report: LowLevelIRReport) -> dict[str, int]:
    return {
        "checked_functions": len(report.checked_functions),
        "violations": len(report.violations),
        "tile_primitives": sum(finding.kind == "tile_primitive" for finding in report.violations),
        "forbidden_loads": sum(finding.kind == "buffer_load" for finding in report.violations),
        "forbidden_stores": sum(finding.kind == "buffer_store" for finding in report.violations),
        "address_only_loads": len(report.address_only_loads),
    }


def _ir_report_dict(report: LowLevelIRReport) -> dict[str, Any]:
    result = report.to_dict()
    # TVM's fallback SequentialSpan rendering embeds SourceName object addresses.
    # They are process-local diagnostics, not source locations, and would make
    # otherwise identical audit JSON differ across invocations.
    for key in ("violations", "address_only_loads"):
        for finding in result[key]:
            span = finding.get("span")
            if span is not None:
                finding["span"] = re.sub(r", 0x[0-9a-fA-F]+\)", ")", span)
    result["counts"] = _ir_counts(report)
    return result


def _module_name(module: Any) -> str:
    name = getattr(module, "__name__", None)
    return name if isinstance(name, str) else type(module).__name__


def _validate_meta(
    registry_name: str, module: Any
) -> tuple[dict[str, Any] | None, list[RegistryAuditError]]:
    errors: list[RegistryAuditError] = []
    raw_meta = getattr(module, "KERNEL_META", _MISSING)
    if raw_meta is _MISSING:
        return None, [_error("metadata", "KERNEL_META is required")]
    if not isinstance(raw_meta, Mapping):
        return None, [_error("metadata", "KERNEL_META must be a mapping", "TypeError")]

    for key in ("name", "category", "compute_capability"):
        if key not in raw_meta:
            errors.append(_error("metadata", f"KERNEL_META[{key!r}] is required"))
    meta_name = raw_meta.get("name")
    if "name" in raw_meta and (not isinstance(meta_name, str) or not meta_name):
        errors.append(_error("metadata", "KERNEL_META['name'] must be a non-empty string"))
    elif meta_name != registry_name:
        errors.append(
            _error(
                "metadata",
                f"registry key {registry_name!r} does not match KERNEL_META['name']={meta_name!r}",
            )
        )
    category = raw_meta.get("category")
    if "category" in raw_meta and (not isinstance(category, str) or not category):
        errors.append(_error("metadata", "KERNEL_META['category'] must be a non-empty string"))
    compute_capability = raw_meta.get("compute_capability")
    if "compute_capability" in raw_meta and (
        isinstance(compute_capability, bool) or not isinstance(compute_capability, int)
    ):
        errors.append(_error("metadata", "KERNEL_META['compute_capability'] must be an integer"))

    try:
        normalized = _json_value(dict(raw_meta))
    except TypeError as exc:
        errors.append(_error("metadata", exc))
        normalized = None
    return normalized, errors


def _validate_configs(module: Any) -> tuple[list[Any] | None, list[RegistryAuditError]]:
    raw_configs = getattr(module, "CONFIGS", _MISSING)
    if raw_configs is _MISSING:
        return None, [_error("configs", "CONFIGS is required")]
    if not isinstance(raw_configs, list):
        return None, [_error("configs", "CONFIGS must be a list", "TypeError")]
    if not raw_configs:
        return raw_configs, [_error("configs", "CONFIGS must not be empty")]
    return raw_configs, []


def _config_errors(
    config: Any, duplicate_labels: set[str]
) -> tuple[str | None, dict[str, Any] | None, list[RegistryAuditError]]:
    if not isinstance(config, Mapping):
        return None, None, [_error("config", "configuration must be a mapping", "TypeError")]

    errors: list[RegistryAuditError] = []
    non_string_keys = [key for key in config if not isinstance(key, str)]
    if non_string_keys:
        errors.append(_error("config", "configuration keys must be strings", "TypeError"))

    label = config.get("label")
    if not isinstance(label, str) or not label:
        errors.append(_error("config", "configuration label must be a non-empty string"))
        normalized_label = None
    else:
        normalized_label = label
        if label in duplicate_labels:
            errors.append(_error("config", f"duplicate configuration label {label!r}"))

    parameters = {key: value for key, value in config.items() if key != "label"}
    try:
        normalized_parameters = _json_value(parameters)
    except TypeError as exc:
        errors.append(_error("config", exc))
        normalized_parameters = None
    return normalized_label, normalized_parameters, errors


def _duplicate_labels(configs: Sequence[Any]) -> set[str]:
    labels = [
        config.get("label")
        for config in configs
        if isinstance(config, Mapping) and isinstance(config.get("label"), str)
    ]
    return {label for label, count in Counter(labels).items() if count > 1}


def _audit_kernel(
    registry_name: str, module: Any, *, config_filter: str | None
) -> KernelAuditResult:
    meta, meta_errors = _validate_meta(registry_name, module)
    configs, config_list_errors = _validate_configs(module)
    kernel_errors = [*meta_errors, *config_list_errors]

    get_kernel = getattr(module, "get_kernel", None)
    if not callable(get_kernel):
        kernel_errors.append(_error("get_kernel", "get_kernel must be callable", "TypeError"))

    declared_config_count = len(configs) if configs is not None else 0
    duplicate_labels = _duplicate_labels(configs or [])
    results: list[ConfigAuditResult] = []
    for index, config in enumerate(configs or []):
        raw_label = config.get("label") if isinstance(config, Mapping) else None
        if config_filter is not None and raw_label != config_filter:
            continue

        label, normalized_parameters, errors = _config_errors(config, duplicate_labels)
        errors = [*kernel_errors, *errors]
        if errors:
            results.append(
                ConfigAuditResult(
                    index=index,
                    label=label,
                    parameters=normalized_parameters,
                    status="error",
                    errors=tuple(errors),
                    ir=None,
                )
            )
            continue

        parameters = {key: value for key, value in config.items() if key != "label"}
        try:
            value = get_kernel(**parameters)
        except Exception as exc:
            results.append(
                ConfigAuditResult(
                    index=index,
                    label=label,
                    parameters=normalized_parameters,
                    status="error",
                    errors=(_error("get_kernel", exc),),
                    ir=None,
                )
            )
            continue

        try:
            ir_report = inspect_low_level_ir(value)
        except Exception as exc:
            results.append(
                ConfigAuditResult(
                    index=index,
                    label=label,
                    parameters=normalized_parameters,
                    status="error",
                    errors=(_error("inspection", exc),),
                    ir=None,
                )
            )
            continue

        if not ir_report.checked_functions:
            results.append(
                ConfigAuditResult(
                    index=index,
                    label=label,
                    parameters=normalized_parameters,
                    status="error",
                    errors=(
                        _error(
                            "inspection",
                            "get_kernel returned no TIRx PrimFunc objects",
                            "TypeError",
                        ),
                    ),
                    ir=ir_report,
                )
            )
            continue

        results.append(
            ConfigAuditResult(
                index=index,
                label=label,
                parameters=normalized_parameters,
                status="pass" if ir_report.ok else "violation",
                errors=(),
                ir=ir_report,
            )
        )

    complete = (
        not kernel_errors
        and len(results) == declared_config_count
        and all(result.status != "error" for result in results)
    )
    has_config_error = any(result.status == "error" for result in results)
    if kernel_errors or has_config_error:
        decision = "incomplete"
    elif any(result.status == "violation" for result in results):
        decision = "rewrite"
    elif not complete:
        decision = "incomplete"
    else:
        decision = "skip"

    return KernelAuditResult(
        name=registry_name,
        module=_module_name(module),
        meta=meta,
        declared_config_count=declared_config_count,
        audited_config_count=len(results),
        complete=complete,
        decision=decision,
        errors=tuple(kernel_errors),
        configs=tuple(results),
    )


def audit_registered_kernels(
    registry: Mapping[str, Any] | None = None,
    *,
    kernel: str | None = None,
    config: str | None = None,
) -> RegistryAuditReport:
    """Dynamically construct and inspect every selected public kernel config.

    Discovery is strict: an import error makes the report fail rather than
    silently reducing coverage.  ``label`` identifies a configuration but is
    not passed to the public ``get_kernel`` call, matching the test and bench
    runners.
    """
    registry_errors: list[RegistryAuditError] = []
    if registry is None:
        from tirx_kernels.registry import discover_kernels

        try:
            discovered: Mapping[str, Any] = discover_kernels(strict=True)
        except Exception as exc:
            registry_errors.append(_error("discovery", exc))
            discovered = {}
    elif not isinstance(registry, Mapping):
        raise TypeError("registry must be a mapping from kernel names to modules")
    else:
        discovered = registry

    discovered_count = len(discovered)
    if not discovered and not registry_errors:
        registry_errors.append(_error("discovery", "no registered kernels were discovered"))

    if kernel is not None:
        if kernel not in discovered:
            if registry_errors:
                selected = {}
            else:
                raise AuditSelectionError(f"kernel {kernel!r} was not found")
        else:
            selected = {kernel: discovered[kernel]}
    else:
        selected = dict(discovered)

    kernels = tuple(
        _audit_kernel(name, selected[name], config_filter=config) for name in sorted(selected)
    )
    if (
        config is not None
        and not registry_errors
        and not any(result.audited_config_count for result in kernels)
    ):
        raise AuditSelectionError(f"configuration label {config!r} was not found")

    return RegistryAuditReport(
        complete_registry=kernel is None and config is None and not registry_errors,
        kernel_filter=kernel,
        config_filter=config,
        discovered_kernel_count=discovered_count,
        registry_errors=tuple(registry_errors),
        kernels=kernels,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit registered kernels against the pre-lowering low-level IR contract"
    )
    parser.add_argument("--kernel", help="Audit only one registry kernel")
    parser.add_argument("--config", help="Audit only matching configuration labels")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", help="Write the report to this file instead of stdout")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: Mapping[str, Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the registry audit CLI and return its process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    try:
        report = audit_registered_kernels(registry, kernel=args.kernel, config=args.config)
    except AuditSelectionError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2

    rendered = (
        json.dumps(report.to_dict(), indent=2, sort_keys=True)
        if args.format == "json"
        else report.summary()
    )
    if args.output:
        try:
            Path(args.output).write_text(f"{rendered}\n", encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: cannot write {args.output!r}: {exc}", file=stderr)
            return 2
    else:
        print(rendered, file=stdout)
    return 0 if report.ok else 1


__all__ = [
    "AuditSelectionError",
    "ConfigAuditResult",
    "KernelAuditResult",
    "RegistryAuditError",
    "RegistryAuditReport",
    "audit_registered_kernels",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
