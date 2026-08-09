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
"""Run public kernel correctness checks and emit auditable results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO
from unittest import SkipTest

from tirx_kernels.registry import discover_kernels
from tirx_kernels.runner import run_kernel_test

SCHEMA_VERSION = 1


class CorrectnessSelectionError(ValueError):
    """Raised when the requested public correctness scope is absent or malformed."""


def _declared_configs(name: str, module: ModuleType) -> list[tuple[str, dict[str, Any]]]:
    configs = getattr(module, "CONFIGS", None)
    if not isinstance(configs, list) or not configs:
        raise CorrectnessSelectionError(f"{name}: CONFIGS must be a non-empty list")

    declared: list[tuple[str, dict[str, Any]]] = []
    for index, config in enumerate(configs):
        if not isinstance(config, Mapping):
            raise CorrectnessSelectionError(f"{name}: CONFIGS[{index}] must be a mapping")
        label = config.get("label")
        if not isinstance(label, str) or not label:
            raise CorrectnessSelectionError(
                f"{name}: CONFIGS[{index}] must have a non-empty string label"
            )
        declared.append((label, dict(config)))

    duplicates = sorted(
        label for label, count in Counter(label for label, _ in declared).items() if count > 1
    )
    if duplicates:
        raise CorrectnessSelectionError(f"{name}: duplicate CONFIGS labels: {duplicates}")
    return declared


def run_correctness_suite(
    registry: Mapping[str, ModuleType],
    *,
    kernel_filter: str | None = None,
    config_filter: str | None = None,
    compute_capability_filter: int | None = None,
) -> dict[str, Any]:
    """Execute the selected public configurations and return a strict report.

    A report is successful only when at least one declared configuration ran and
    every result passed.  A reference skip is preserved as evidence and is a
    hard failure, rather than being silently accepted as coverage.
    """
    if not registry:
        raise CorrectnessSelectionError("no registered kernels were discovered")
    if kernel_filter is not None and kernel_filter not in registry:
        raise CorrectnessSelectionError(
            f"kernel {kernel_filter!r} was not found; available: {sorted(registry)}"
        )

    selected_registry = (
        {kernel_filter: registry[kernel_filter]} if kernel_filter is not None else dict(registry)
    )
    results: list[dict[str, str]] = []
    declared_config_count = 0
    selected_config_count = 0

    for name, module in sorted(selected_registry.items()):
        declared = _declared_configs(name, module)
        declared_config_count += len(declared)
        for label, config in declared:
            if config_filter is not None and label != config_filter:
                continue
            selected_config_count += 1
            try:
                run_kernel_test(name, config, registry=registry)
            except SkipTest as exc:
                results.append(
                    {"kernel": name, "config": label, "status": "SKIP", "reason": str(exc)}
                )
            except Exception as exc:
                results.append(
                    {
                        "kernel": name,
                        "config": label,
                        "status": "FAIL",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                results.append({"kernel": name, "config": label, "status": "PASS"})

    if selected_config_count == 0:
        scope = f"config {config_filter!r}" if config_filter is not None else "requested scope"
        raise CorrectnessSelectionError(f"{scope} selected no declared configurations")

    passed = sum(result["status"] == "PASS" for result in results)
    failed = sum(result["status"] == "FAIL" for result in results)
    skipped = sum(result["status"] == "SKIP" for result in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": failed == 0 and skipped == 0 and passed == selected_config_count,
        "scope": {
            "complete_registry": kernel_filter is None
            and config_filter is None
            and compute_capability_filter is None,
            "kernel": kernel_filter,
            "config": config_filter,
            "compute_capability": compute_capability_filter,
        },
        "discovered_kernels": len(registry),
        "selected_kernels": len(selected_registry),
        "declared_configs": declared_config_count,
        "selected_configs": selected_config_count,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }


def write_correctness_report(report: Mapping[str, Any], output: Path) -> Path:
    """Write one JSON evidence artifact and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _text_report(report: Mapping[str, Any]) -> str:
    lines = []
    for result in report["results"]:
        detail = result.get("reason") or result.get("error")
        suffix = f": {detail}" if detail else ""
        lines.append(f"{result['status']:<5} {result['kernel']} [{result['config']}]{suffix}")
    lines.extend(
        (
            "",
            "=" * 60,
            f"Total: {report['selected_configs']}  Passed: {report['passed']}  "
            f"Failed: {report['failed']}  Skipped: {report['skipped']}",
        )
    )
    return "\n".join(lines) + "\n"


def main(
    argv: list[str] | None = None,
    *,
    registry: Mapping[str, ModuleType] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run kernel correctness tests")
    parser.add_argument("--kernel", type=str, default=None, help="Run only this kernel")
    parser.add_argument("--config", type=str, default=None, help="Run only this config label")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument("--cc", type=int, default=None, help="Compute capability filter")
    args = parser.parse_args(argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    try:
        if registry is None:
            registry = discover_kernels(min_compute_capability=args.cc, strict=True)
        report = run_correctness_suite(
            registry,
            kernel_filter=args.kernel,
            config_filter=args.config,
            compute_capability_filter=args.cc,
        )
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "scope": {
                "complete_registry": False,
                "kernel": args.kernel,
                "config": args.config,
                "compute_capability": args.cc,
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
            "results": [],
        }
        if args.output is not None:
            write_correctness_report(report, args.output)
        stream = stdout if args.json else stderr
        print(
            json.dumps(report, indent=2, sort_keys=True) if args.json else f"ERROR: {exc}",
            file=stream,
        )
        return 2

    if args.output is not None:
        write_correctness_report(report, args.output)
    rendered = (
        json.dumps(report, indent=2, sort_keys=True) + "\n" if args.json else _text_report(report)
    )
    print(rendered, file=stdout, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
