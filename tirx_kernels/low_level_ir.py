# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Validate the pre-lowering IR contract for low-level TIRx kernels."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import StmtExprVisitor

_FORBIDDEN_SCOPE_ROOTS = ("global", "shared")
_ADDRESS_OF_OP = "tirx.address_of"
_FUNC_CALL_OP = "tirx.cuda.func_call"

NVSHMEM_RUNTIME_FUNC_CALLS = frozenset(
    {
        "enqueue_remote",
        "exit_barrier_arrive_and_wait",
        "ld_reduce_8_fp16",
        "semaphore_notify_remote",
    }
)
LOW_LEVEL_IR_FUNC_CALL_EXCEPTIONS_BY_KERNEL: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "cudnn_sm100_csa_compressor_fwd": frozenset(
            {
                "tirx_csa_exp_constants",
                "tirx_csa_load4_bf16x2",
                "tirx_csa_ordered_max",
                "tirx_csa_scan_boundary",
                "tirx_csa_source_exp",
            }
        ),
        "gemm_reduce_scatter": NVSHMEM_RUNTIME_FUNC_CALLS,
    }
)


def _is_forbidden_scope(scope: str) -> bool:
    return any(scope == root or scope.startswith(f"{root}.") for root in _FORBIDDEN_SCOPE_ROOTS)


def _node_name(node: Any) -> str:
    return type(node).__name__.removesuffix("Node")


def _span_text(node: Any) -> str | None:
    span = getattr(node, "span", None)
    if span is None:
        return None

    source_name = getattr(span, "source_name", None)
    source = getattr(source_name, "name", None)
    line = getattr(span, "line", None)
    column = getattr(span, "column", None)
    end_line = getattr(span, "end_line", None)
    end_column = getattr(span, "end_column", None)
    if source is not None and line is not None and column is not None:
        start = f"{source}:{line}:{column}"
        if end_line is not None and end_column is not None:
            return f"{start}-{end_line}:{end_column}"
        return start
    return str(span)


@dataclass(frozen=True)
class LowLevelIRFinding:
    """One observable contract finding in a particular TIRx function."""

    function: str
    kind: str
    node_type: str
    scope: str | None
    callee: str | None
    span: str | None


@dataclass(frozen=True)
class LowLevelIRReport:
    """Structured result of inspecting one kernel return value."""

    checked_functions: tuple[str, ...]
    violations: tuple[LowLevelIRFinding, ...]
    func_calls: tuple[LowLevelIRFinding, ...]
    address_only_loads: tuple[LowLevelIRFinding, ...]

    @property
    def ok(self) -> bool:
        """Whether every visited function satisfies the low-level IR contract."""
        return not self.violations

    def summary(self) -> str:
        """Return a compact human-readable result."""
        address_count = len(self.address_only_loads)
        func_call_count = len(self.func_calls)
        if self.ok:
            return (
                f"low-level IR contract passed for {len(self.checked_functions)} function(s); "
                f"recorded {func_call_count} func_call(s) and "
                f"{address_count} address-only load(s)"
            )

        lines = [
            f"low-level IR contract failed with {len(self.violations)} violation(s) "
            f"across {len(self.checked_functions)} function(s); "
            f"recorded {func_call_count} func_call(s) and "
            f"{address_count} address-only load(s)"
        ]
        for finding in self.violations:
            scope = f", scope={finding.scope}" if finding.scope is not None else ""
            callee = f", callee={finding.callee}" if finding.callee is not None else ""
            span = f", span={finding.span}" if finding.span is not None else ""
            lines.append(
                f"- {finding.function}: {finding.kind} ({finding.node_type}{scope}{callee}{span})"
            )
        return "\n".join(lines)


class LowLevelIRContractError(ValueError):
    """Raised when pre-lowering TIRx violates the low-level IR contract."""

    def __init__(self, report: LowLevelIRReport):
        self.report = report
        super().__init__(report.summary())


class _LowLevelIRVisitor(StmtExprVisitor):
    """Collect forbidden operations from one pre-lowering ``PrimFunc`` body."""

    def __init__(self, function: str, allowed_func_calls: frozenset[str]):
        super().__init__()
        self.function = function
        self.allowed_func_calls = allowed_func_calls
        self.violations: list[LowLevelIRFinding] = []
        self.func_calls: list[LowLevelIRFinding] = []
        self.address_only_loads: list[LowLevelIRFinding] = []

    def _finding(
        self, node: Any, kind: str, scope: str | None = None, callee: str | None = None
    ) -> LowLevelIRFinding:
        return LowLevelIRFinding(
            function=self.function,
            kind=kind,
            node_type=_node_name(node),
            scope=scope,
            callee=callee,
            span=_span_text(node),
        )

    def visit_op_call_(self, op: Any) -> None:
        self.violations.append(self._finding(op, "tile_primitive"))
        super().visit_op_call_(op)

    def visit_buffer_load_(self, op: tirx.BufferLoad) -> None:
        scope = str(op.buffer.scope())
        if _is_forbidden_scope(scope):
            self.violations.append(self._finding(op, "buffer_load", scope))
        # The fixed TIRx visitor only walks BufferLoad.indices.  A predicated
        # load can itself contain loads, and those are real accesses even when
        # the outer load is in an allowed scope.
        for index in op.indices:
            self.visit_expr(index)
        if op.predicate is not None:
            self.visit_expr(op.predicate)

    def visit_buffer_store_(self, op: tirx.BufferStore) -> None:
        scope = str(op.buffer.scope())
        if _is_forbidden_scope(scope):
            self.violations.append(self._finding(op, "buffer_store", scope))
        self.visit_expr(op.value)
        for index in op.indices:
            self.visit_expr(index)
        if op.predicate is not None:
            self.visit_expr(op.predicate)

    def visit_call_(self, op: tvm.ir.Call) -> None:
        op_name = getattr(op.op, "name", None)
        if op_name == _FUNC_CALL_OP:
            callee = str(getattr(op.args[0], "value", op.args[0])) if op.args else "<missing>"
            finding = self._finding(op, "func_call", callee=callee)
            self.func_calls.append(finding)
            if callee not in self.allowed_func_calls:
                self.violations.append(finding)
        if op_name == _ADDRESS_OF_OP and len(op.args) == 1:
            addressed = op.args[0]
            if isinstance(addressed, tirx.BufferLoad):
                scope = str(addressed.buffer.scope())
                if _is_forbidden_scope(scope):
                    self.address_only_loads.append(
                        self._finding(addressed, "address_only_buffer_load", scope)
                    )
                # The BufferLoad is pointer syntax rather than a memory read.  Its
                # indices are still expressions, and any loads inside them remain
                # real accesses that must be checked normally.
                for index in addressed.indices:
                    self.visit_expr(index)
                if addressed.predicate is not None:
                    self.visit_expr(addressed.predicate)
                return
        super().visit_call_(op)


def _global_name(global_var: Any) -> str:
    return str(getattr(global_var, "name_hint", global_var))


def _iter_prim_funcs(value: Any, path: str = "root") -> Iterator[tuple[str, tirx.PrimFunc]]:
    if isinstance(value, tirx.PrimFunc):
        yield path, value
        return

    if isinstance(value, tvm.IRModule):
        for global_var, base_func in value.functions.items():
            function_path = f"{path}.{_global_name(global_var)}"
            if not isinstance(base_func, tirx.PrimFunc):
                raise TypeError(
                    f"{function_path} is {_node_name(base_func)}, expected tvm.tirx.PrimFunc"
                )
            yield function_path, base_func
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_prim_funcs(item, f"{path}[{key!r}]")
        return

    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _iter_prim_funcs(item, f"{path}[{index}]")
        return

    raise TypeError(
        f"{path} is {_node_name(value)}, expected a TIRx PrimFunc, IRModule, "
        "or a list/tuple/dict containing them"
    )


def inspect_low_level_ir(
    value: Any, *, allowed_func_calls: frozenset[str] | set[str] | tuple[str, ...] = ()
) -> LowLevelIRReport:
    """Inspect a public ``get_kernel`` return value without lowering it.

    ``value`` may be a TIRx ``PrimFunc``, ``IRModule``, or nested list, tuple,
    or mapping containing those objects.  Direct ``tirx.address_of`` operands
    are reported separately and do not count as memory reads.
    """
    checked_functions: list[str] = []
    violations: list[LowLevelIRFinding] = []
    func_calls: list[LowLevelIRFinding] = []
    address_only_loads: list[LowLevelIRFinding] = []
    allowed_func_calls = frozenset(allowed_func_calls)

    for path, prim_func in _iter_prim_funcs(value):
        checked_functions.append(path)
        visitor = _LowLevelIRVisitor(path, allowed_func_calls)
        visitor(prim_func.body)
        violations.extend(visitor.violations)
        func_calls.extend(visitor.func_calls)
        address_only_loads.extend(visitor.address_only_loads)

    if not checked_functions:
        raise TypeError("root must contain at least one TIRx PrimFunc")

    return LowLevelIRReport(
        checked_functions=tuple(checked_functions),
        violations=tuple(violations),
        func_calls=tuple(func_calls),
        address_only_loads=tuple(address_only_loads),
    )


def check_low_level_ir(
    value: Any, *, allowed_func_calls: frozenset[str] | set[str] | tuple[str, ...] = ()
) -> LowLevelIRReport:
    """Return a report for valid IR, or raise ``LowLevelIRContractError``."""
    report = inspect_low_level_ir(value, allowed_func_calls=allowed_func_calls)
    if not report.ok:
        raise LowLevelIRContractError(report)
    return report


__all__ = [
    "LOW_LEVEL_IR_FUNC_CALL_EXCEPTIONS_BY_KERNEL",
    "NVSHMEM_RUNTIME_FUNC_CALLS",
    "LowLevelIRContractError",
    "LowLevelIRFinding",
    "LowLevelIRReport",
    "check_low_level_ir",
    "inspect_low_level_ir",
]
