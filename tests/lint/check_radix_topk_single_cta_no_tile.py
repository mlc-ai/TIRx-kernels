#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every radix top-k single-CTA specialization.

Scans the kernel module and the shared instruction-level helpers it imports,
then every pre-dispatch specialization built from ``CONFIGS``.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tvm
from tirx_kernels.flashinfer.topk import radix_topk_single_cta as target
from tvm.tirx.stmt_functor import StmtExprVisitor

REPO = Path(__file__).resolve().parents[2]
TARGETS = (
    REPO / "tirx_kernels/flashinfer/topk/radix_topk_single_cta.py",
    REPO / "tirx_kernels/flashinfer/utils/topk_radix.py",
)

_TILE_OPS = {
    "add",
    "binary_chain",
    "binary_reduce",
    "cast",
    "compose_op",
    "copy",
    "copy_async",
    "exp",
    "exp2",
    "fdiv",
    "fill",
    "fma",
    "gemm",
    "gemm_async",
    "max",
    "maximum",
    "memset",
    "min",
    "minimum",
    "mul",
    "permute_layout",
    "reciprocal",
    "reduce_negate",
    "select",
    "silu",
    "sqrt",
    "sub",
    "sum",
    "unary_reduce",
    "zero",
}
_SCOPES = {"thread", "warp", "wg", "warpgroup", "cta", "cluster"}


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    detail: str


def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    return None


def _is_plain_cast(call: ast.Call) -> bool:
    """Recognize ordinary expression ``T.cast(value, "dtype")`` calls."""
    return (
        len(call.args) == 2
        and not call.keywords
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
    )


def _tile_call_reason(name: str | None, call: ast.Call) -> str | None:
    if name is None:
        return None
    parts = name.split(".")
    leaf = parts[-1]
    if leaf == "TilePrimitiveCall" or ".tile." in name:
        return f"explicit tile primitive call {name}"
    if "tile_primitive" in parts and leaf not in {"SwizzleMode", "sf_smem_layout"}:
        return f"tile-primitive helper call {name}"

    # TIRx script exposes both direct and execution-scope tile APIs.  Cast has
    # a scalar overload, which is legal when its second argument is a dtype.
    script_root = parts[0] in {"T", "Tx", "tirx"}
    scoped = len(parts) >= 3 and parts[-2] in _SCOPES
    direct = len(parts) == 2
    if script_root and (direct or scoped) and leaf in _TILE_OPS:
        if leaf == "cast" and _is_plain_cast(call):
            return None
        return f"TIRx tile API call {name}"
    return None


class _FunctionScanner(ast.NodeVisitor):
    def __init__(
        self,
        source: str,
        aliases: dict[str, str],
        function_names: set[str],
        function: str,
        target: Path,
    ) -> None:
        self.source = source
        self.target = target
        self.aliases = dict(aliases)
        self.function_names = function_names
        self.function = function
        self.findings: list[Finding] = []
        self.calls: list[tuple[str, ast.Call]] = []

    def _path(self, node: ast.AST) -> str:
        return f"{self.target.relative_to(REPO)}:{node.lineno}:{node.col_offset + 1}"

    def visit_Assign(self, node: ast.Assign) -> None:
        value_name = _qualified_name(node.value, self.aliases)
        if value_name is not None:
            for assignment in node.targets:
                if isinstance(assignment, ast.Name):
                    self.aliases[assignment.id] = value_name
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            value_name = _qualified_name(node.value, self.aliases)
            if value_name is not None:
                self.aliases[node.target.id] = value_name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _qualified_name(node.func, self.aliases)
        reason = _tile_call_reason(name, node)
        if reason is not None:
            self.findings.append(Finding("source", self._path(node), reason))

        local_name = name.rsplit(".", 1)[-1] if name else None
        if local_name in self.function_names:
            self.calls.append((local_name, node))

        for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("tirx.tile."):
                    self.findings.append(
                        Finding(
                            "source",
                            self._path(argument),
                            f"dynamic tile operator name {argument.value!r}",
                        )
                    )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested function bodies are scanned by their own scanner.
        if node.name != self.function:
            return
        self.generic_visit(node)


def _source_findings_for(target: Path) -> list[Finding]:
    source = target.read_text()
    tree = ast.parse(source, filename=str(target))
    aliases: dict[str, str] = {}
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for item in node.names:
                aliases[item.asname or item.name] = f"{module}.{item.name}"
        elif isinstance(node, ast.Assign):
            name = _qualified_name(node.value, aliases)
            if name is not None:
                for assignment in node.targets:
                    if isinstance(assignment, ast.Name):
                        aliases[assignment.id] = name

    scans: dict[str, _FunctionScanner] = {}
    findings: list[Finding] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        scanner = _FunctionScanner(source, aliases, functions, node.name, target)
        scanner.visit(node)
        scans[node.name] = scanner
        findings.extend(scanner.findings)

    # Propagate taint through local wrappers so both the primitive and every
    # wrapper call site are visible in diagnostics.
    tainted = {name for name, scanner in scans.items() if scanner.findings}
    changed = True
    while changed:
        changed = False
        for name, scanner in scans.items():
            if name in tainted:
                continue
            if any(callee in tainted for callee, _ in scanner.calls):
                tainted.add(name)
                changed = True

    for name, scanner in scans.items():
        for callee, call in scanner.calls:
            if callee in tainted:
                findings.append(
                    Finding(
                        "source",
                        scanner._path(call),
                        f"wrapper {name} calls tile-tainted helper {callee}",
                    )
                )
    return findings


def _source_findings() -> list[Finding]:
    findings: list[Finding] = []
    for target_path in TARGETS:
        findings.extend(_source_findings_for(target_path))
    return findings


class _IRScanner(StmtExprVisitor):
    def __init__(self, specialization: str) -> None:
        super().__init__()
        self.specialization = specialization
        self.findings: list[Finding] = []

    def _record(self, node: Any, detail: str) -> None:
        span = getattr(node, "span", None)
        suffix = f" span={span}" if span is not None else ""
        self.findings.append(Finding("IR", self.specialization, detail + suffix))

    def visit_op_call_(self, op: Any) -> None:
        op_name = getattr(getattr(op, "op", None), "name", "<unknown>")
        self._record(op, f"TilePrimitiveCall op={op_name}")
        super().visit_op_call_(op)

    def visit_call_(self, call: tvm.ir.Call) -> None:
        op = getattr(call, "op", None)
        op_name = getattr(op, "name", None)
        category = op.get_attr("TIRxOpCategory") if isinstance(op, tvm.ir.Op) else None
        if op_name is not None and op_name.startswith("tirx.tile."):
            self._record(call, f"forbidden operator {op_name}")
        elif category == "tile_primitive":
            self._record(call, f"operator {op_name} has TIRxOpCategory=tile_primitive")
        super().visit_call_(call)


def _ir_findings() -> list[Finding]:
    findings: list[Finding] = []
    for config in target.CONFIGS:
        label = str(config["label"])
        kernel = target.get_kernel(**config)
        scanner = _IRScanner(f"CONFIGS[{label!r}]")
        scanner(kernel.body)
        findings.extend(scanner.findings)
    return findings


def main() -> int:
    findings = _source_findings() + _ir_findings()
    if findings:
        for finding in findings:
            print(f"ERROR [{finding.kind}] {finding.path}: {finding.detail}")
        print(f"\n{len(findings)} tile-primitive violation(s).")
        return 1
    print(
        f"PASS: {len(TARGETS)} source file(s) and all "
        f"{len(target.CONFIGS)} pre-dispatch specializations contain no tile primitive"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
