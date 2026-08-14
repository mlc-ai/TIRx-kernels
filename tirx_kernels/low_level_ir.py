# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Validate the pre-lowering IR contract for low-level TIRx kernels."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import StmtExprVisitor

_FORBIDDEN_SCOPE_ROOTS = ("global", "shared")
_ADDRESS_OF_OP = "tirx.address_of"


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
    span: str | None


@dataclass(frozen=True)
class LowLevelIRReport:
    """Structured result of inspecting one kernel return value."""

    checked_functions: tuple[str, ...]
    violations: tuple[LowLevelIRFinding, ...]
    address_only_loads: tuple[LowLevelIRFinding, ...]

    @property
    def ok(self) -> bool:
        """Whether every visited function satisfies the low-level IR contract."""
        return not self.violations

    def summary(self) -> str:
        """Return a compact human-readable result."""
        address_count = len(self.address_only_loads)
        if self.ok:
            return (
                f"low-level IR contract passed for {len(self.checked_functions)} function(s); "
                f"recorded {address_count} address-only load(s)"
            )

        lines = [
            f"low-level IR contract failed with {len(self.violations)} violation(s) "
            f"across {len(self.checked_functions)} function(s); "
            f"recorded {address_count} address-only load(s)"
        ]
        for finding in self.violations:
            scope = f", scope={finding.scope}" if finding.scope is not None else ""
            span = f", span={finding.span}" if finding.span is not None else ""
            lines.append(f"- {finding.function}: {finding.kind} ({finding.node_type}{scope}{span})")
        return "\n".join(lines)


class LowLevelIRContractError(ValueError):
    """Raised when pre-lowering TIRx violates the low-level IR contract."""

    def __init__(self, report: LowLevelIRReport):
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True)
class BuilderContractFinding:
    """One static authoring-contract violation reachable from a registry entry."""

    kernel: str
    function: str
    kind: str
    detail: str
    source: str | None


@dataclass(frozen=True)
class BuilderContractReport:
    """Static registry-wide proof that public kernels use explicit IR Builder paths."""

    checked_kernels: tuple[str, ...]
    reachable_functions: tuple[str, ...]
    builder_functions: tuple[str, ...]
    violations: tuple[BuilderContractFinding, ...]

    @property
    def ok(self) -> bool:
        """Whether all registry construction paths satisfy the builder-only contract."""
        return not self.violations

    def summary(self) -> str:
        """Return a compact, actionable result."""
        if self.ok:
            return (
                f"builder-only contract passed for {len(self.checked_kernels)} kernel(s); "
                f"reached {len(self.reachable_functions)} function(s) and "
                f"{len(self.builder_functions)} explicit builder endpoint(s)"
            )
        lines = [
            f"builder-only contract failed with {len(self.violations)} violation(s) "
            f"across {len(self.checked_kernels)} kernel(s)"
        ]
        for finding in self.violations:
            source = f" ({finding.source})" if finding.source else ""
            lines.append(
                f"- {finding.kernel}: {finding.kind} in {finding.function}{source}: "
                f"{finding.detail}"
            )
        return "\n".join(lines)


class BuilderContractError(ValueError):
    """Raised when a registry construction path is not explicitly builder-only."""

    def __init__(self, report: BuilderContractReport):
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True)
class _SourceFunction:
    module: str
    name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    path: Path

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass(frozen=True)
class _SourceModule:
    name: str
    path: Path
    tree: ast.Module
    imports: Mapping[str, str]
    functions: Mapping[str, _SourceFunction]
    parser_bindings: Mapping[str, tuple[str, int]]


_PARSER_DECORATORS = frozenset(
    {"tvm.script.tir.prim_func", "tvm.script.tirx.jit", "tvm.script.tirx.prim_func"}
)
_IR_BUILDER = "tvm.script.ir_builder.IRBuilder"


def _module_name(package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([package_root.name, *parts])


def _resolve_import(
    module: str, imported: str | None, level: int, *, module_is_package: bool = False
) -> str:
    if level == 0:
        return imported or ""
    package = module if module_is_package else module.rpartition(".")[0]
    request = "." * level + (imported or "")
    return importlib.util.resolve_name(request, package)


def _imports_for(module: str, tree: ast.Module, *, module_is_package: bool = False) -> dict[str, str]:
    imports: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(statement, ast.ImportFrom):
            base = _resolve_import(
                module,
                statement.module,
                statement.level,
                module_is_package=module_is_package,
            )
            for alias in statement.names:
                if alias.name == "*":
                    continue
                imports[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
    return imports


def _parser_bindings(tree: ast.Module, imports: Mapping[str, str]) -> dict[str, tuple[str, int]]:
    bindings: dict[str, tuple[str, int]] = {}
    for statement in tree.body:
        value: ast.expr | None = None
        targets: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            value = statement.value
            targets = list(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
            targets = [statement.target]
        if not isinstance(value, ast.Call):
            continue
        fake_source = _SourceModule("", Path(), tree, imports, {}, {})
        origin = _qualified_expr(value.func, fake_source, imports)
        if origin not in _PARSER_DECORATORS:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = (origin, statement.lineno)
    return bindings


def _load_source_modules(package_root: Path) -> dict[str, _SourceModule]:
    modules: dict[str, _SourceModule] = {}
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = _module_name(package_root, path)
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = _imports_for(module, tree, module_is_package=path.name == "__init__.py")
        functions = {
            statement.name: _SourceFunction(module, statement.name, statement, path)
            for statement in tree.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        modules[module] = _SourceModule(
            name=module,
            path=path,
            tree=tree,
            imports=imports,
            functions=functions,
            parser_bindings=_parser_bindings(tree, imports),
        )
    return modules


def _qualified_expr(
    expr: ast.expr, source: _SourceModule, imports: Mapping[str, str] | None = None
) -> str | None:
    imports = imports or source.imports
    if isinstance(expr, ast.Name):
        return imports.get(expr.id, expr.id)
    if isinstance(expr, ast.Attribute):
        base = _qualified_expr(expr.value, source, imports)
        return f"{base}.{expr.attr}" if base else None
    return None


def _visible_imports(function: _SourceFunction, source: _SourceModule) -> dict[str, str]:
    imports = dict(source.imports)
    for node in _runtime_nodes(function):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import(
                function.module,
                node.module,
                node.level,
                module_is_package=source.path.name == "__init__.py",
            )
            for alias in node.names:
                if alias.name != "*":
                    imports[alias.asname or alias.name] = (
                        f"{base}.{alias.name}" if base else alias.name
                    )
    return imports


def _runtime_nodes(function: _SourceFunction) -> Iterator[ast.AST]:
    """Yield nodes evaluated by one call, excluding uncalled nested bodies."""
    stack: list[ast.AST] = list(reversed(function.node.body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            children: list[ast.AST] = []
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                children.extend(node.decorator_list)
                children.extend(node.args.defaults)
                children.extend(value for value in node.args.kw_defaults if value is not None)
                if node.returns is not None:
                    children.append(node.returns)
            stack.extend(reversed(children))
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _resolve_source_function(
    expr: ast.expr,
    source: _SourceModule,
    modules: Mapping[str, _SourceModule],
    imports: Mapping[str, str],
) -> _SourceFunction | None:
    if isinstance(expr, ast.Name) and expr.id in source.functions:
        return source.functions[expr.id]
    qualified = _qualified_expr(expr, source, imports)
    if not qualified or "." not in qualified:
        return None
    module, _, name = qualified.rpartition(".")
    target = modules.get(module)
    return _source_entrypoint(target, modules, name) if target is not None else None


def _source_entrypoint(
    source: _SourceModule,
    modules: Mapping[str, _SourceModule],
    name: str,
    seen: frozenset[tuple[str, str]] = frozenset(),
) -> _SourceFunction | None:
    """Resolve a function defined locally or through an unambiguous re-export chain."""
    key = (source.name, name)
    if key in seen:
        return None
    seen = seen | {key}
    local = source.functions.get(name)
    if local is not None:
        return local
    qualified = source.imports.get(name)
    if qualified and "." in qualified:
        module, _, imported_name = qualified.rpartition(".")
        target = modules.get(module)
        if target is not None:
            resolved = _source_entrypoint(target, modules, imported_name, seen)
            if resolved is not None:
                return resolved

    candidates: set[_SourceFunction] = set()
    for statement in source.tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        if not any(alias.name == "*" for alias in statement.names):
            continue
        module = _resolve_import(
            source.name,
            statement.module,
            statement.level,
            module_is_package=source.path.name == "__init__.py",
        )
        target = modules.get(module)
        if target is None:
            continue
        resolved = _source_entrypoint(target, modules, name, seen)
        if resolved is not None:
            candidates.add(resolved)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _decorator_origin(
    decorator: ast.expr, source: _SourceModule, imports: Mapping[str, str]
) -> str | None:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return _qualified_expr(decorator, source, imports)


def _function_dependencies(
    function: _SourceFunction, source: _SourceModule, modules: Mapping[str, _SourceModule]
) -> set[_SourceFunction]:
    """Resolve source-defined call/value edges from one reachable function.

    Value references are intentional: returning a module-level parser-created
    ``PrimFunc`` without calling it is still a construction dependency.
    """
    dependencies: set[_SourceFunction] = set()
    imports = _visible_imports(function, source)
    for node in _runtime_nodes(function):
        expr: ast.expr | None = None
        if isinstance(node, ast.Call):
            expr = node.func
        elif isinstance(node, ast.Name | ast.Attribute) and isinstance(node.ctx, ast.Load):
            expr = node
        if expr is not None:
            resolved = _resolve_source_function(expr, source, modules, imports)
            if resolved is not None and resolved != function:
                dependencies.add(resolved)
    return dependencies


def _explicit_builder_endpoint(function: _SourceFunction, source: _SourceModule) -> bool:
    imports = _visible_imports(function, source)
    builder_vars: set[str] = set()
    constructed = False
    for node in _runtime_nodes(function):
        value: ast.expr | None = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        elif isinstance(node, ast.withitem):
            value = node.context_expr
            if node.optional_vars is not None:
                targets = [node.optional_vars]
        if (
            isinstance(value, ast.Call)
            and _qualified_expr(value.func, source, imports) == _IR_BUILDER
        ):
            constructed = True
            for target in targets:
                if isinstance(target, ast.Name):
                    builder_vars.add(target.id)

    if not constructed:
        return False
    for node in _runtime_nodes(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get":
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id in builder_vars:
            return True
        if (
            isinstance(owner, ast.Call)
            and _qualified_expr(owner.func, source, imports) == _IR_BUILDER
        ):
            return True
    return False


def _package_root_for_registry(registry: Mapping[str, ModuleType]) -> Path:
    paths: list[Path] = []
    for module in registry.values():
        file = getattr(module, "__file__", None)
        name = getattr(module, "__name__", "")
        if not file or not name:
            continue
        path = Path(file).resolve()
        package_depth = max(len(name.split(".")) - 1, 0)
        paths.append(path.parents[package_depth - 1] if package_depth else path.parent)
    if not paths:
        raise ValueError("registry modules must expose __name__ and __file__")
    root = paths[0]
    if any(path != root for path in paths[1:]):
        raise ValueError(f"registry modules do not share one package root: {paths}")
    return root


def inspect_registry_builder_contract(
    registry: Mapping[str, ModuleType] | None = None, *, package_root: Path | None = None
) -> BuilderContractReport:
    """Inspect every registry ``get_kernel`` construction path using Python AST.

    The registry is the scope authority.  Starting from each public
    ``get_kernel``, this builds a cross-module graph of source-defined call and
    returned-value dependencies.  A valid path must reach at least one explicit
    ``IRBuilder()``/``builder.get()`` endpoint and must not reach a TVMScript
    parser decorator.  Consequently a builder wrapper around a parser-produced
    function remains invalid.
    """
    if registry is None:
        from tirx_kernels.registry import discover_kernels

        registry = discover_kernels(strict=True)
    if not registry:
        raise ValueError("registry must contain at least one kernel")

    package_root = (package_root or _package_root_for_registry(registry)).resolve()
    modules = _load_source_modules(package_root)
    findings: list[BuilderContractFinding] = []
    all_reachable: set[str] = set()
    all_builders: set[str] = set()

    for kernel, runtime_module in sorted(registry.items()):
        module_name = runtime_module.__name__
        source = modules.get(module_name)
        if source is None:
            findings.append(
                BuilderContractFinding(
                    kernel, module_name, "source_unavailable", str(runtime_module.__file__), None
                )
            )
            continue
        root = _source_entrypoint(source, modules, "get_kernel")
        if root is None:
            findings.append(
                BuilderContractFinding(
                    kernel,
                    module_name,
                    "missing_entrypoint",
                    "registry module has no source-resolvable get_kernel",
                    str(source.path),
                )
            )
            continue

        pending = [root]
        visited: set[_SourceFunction] = set()
        builders: set[_SourceFunction] = set()
        while pending:
            function = pending.pop()
            if function in visited:
                continue
            visited.add(function)
            function_source = modules[function.module]
            imports = _visible_imports(function, function_source)
            if _explicit_builder_endpoint(function, function_source):
                builders.add(function)
            for decorator in function.node.decorator_list:
                origin = _decorator_origin(decorator, function_source, imports)
                if origin in _PARSER_DECORATORS:
                    findings.append(
                        BuilderContractFinding(
                            kernel=kernel,
                            function=function.qualified_name,
                            kind="parser_decorator",
                            detail=f"reachable @{origin} definition {function.name!r}",
                            source=f"{function.path}:{getattr(decorator, 'lineno', '?')}",
                        )
                    )
            for node in _runtime_nodes(function):
                if isinstance(node, ast.Call):
                    origin = _qualified_expr(node.func, function_source, imports)
                    if origin in _PARSER_DECORATORS:
                        findings.append(
                            BuilderContractFinding(
                                kernel=kernel,
                                function=function.qualified_name,
                                kind="parser_call",
                                detail=f"reachable call to parser API {origin}",
                                source=f"{function.path}:{getattr(node, 'lineno', '?')}",
                            )
                        )
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    parser_binding = function_source.parser_bindings.get(node.id)
                    if parser_binding is not None:
                        origin, line = parser_binding
                        findings.append(
                            BuilderContractFinding(
                                kernel=kernel,
                                function=function.qualified_name,
                                kind="parser_value",
                                detail=f"reachable value produced by parser API {origin}",
                                source=f"{function.path}:{line}",
                            )
                        )
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    qualified = _qualified_expr(node, function_source, imports)
                    if qualified and "." in qualified:
                        target_module, _, target_name = qualified.rpartition(".")
                        parser_binding = (
                            modules[target_module].parser_bindings.get(target_name)
                            if target_module in modules
                            else None
                        )
                        if parser_binding is not None:
                            origin, line = parser_binding
                            findings.append(
                                BuilderContractFinding(
                                    kernel=kernel,
                                    function=function.qualified_name,
                                    kind="parser_value",
                                    detail=f"reachable value produced by parser API {origin}",
                                    source=f"{modules[target_module].path}:{line}",
                                )
                            )
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                for decorator in node.decorator_list:
                    origin = _decorator_origin(decorator, function_source, imports)
                    if origin in _PARSER_DECORATORS:
                        findings.append(
                            BuilderContractFinding(
                                kernel=kernel,
                                function=function.qualified_name,
                                kind="parser_decorator",
                                detail=f"reachable @{origin} definition {node.name!r}",
                                source=f"{function.path}:{getattr(decorator, 'lineno', '?')}",
                            )
                        )
            pending.extend(_function_dependencies(function, function_source, modules) - visited)

        if not builders:
            findings.append(
                BuilderContractFinding(
                    kernel=kernel,
                    function=root.qualified_name,
                    kind="missing_builder_endpoint",
                    detail="no reachable explicit IRBuilder() followed by builder.get()",
                    source=f"{root.path}:{root.node.lineno}",
                )
            )
        all_reachable.update(function.qualified_name for function in visited)
        all_builders.update(function.qualified_name for function in builders)

    return BuilderContractReport(
        checked_kernels=tuple(sorted(registry)),
        reachable_functions=tuple(sorted(all_reachable)),
        builder_functions=tuple(sorted(all_builders)),
        violations=tuple(findings),
    )


def check_registry_builder_contract(
    registry: Mapping[str, ModuleType] | None = None, *, package_root: Path | None = None
) -> BuilderContractReport:
    """Return a valid registry builder report, or raise ``BuilderContractError``."""
    report = inspect_registry_builder_contract(registry, package_root=package_root)
    if not report.ok:
        raise BuilderContractError(report)
    return report


class _LowLevelIRVisitor(StmtExprVisitor):
    """Collect forbidden operations from one pre-lowering ``PrimFunc`` body."""

    def __init__(self, function: str):
        super().__init__()
        self.function = function
        self.violations: list[LowLevelIRFinding] = []
        self.address_only_loads: list[LowLevelIRFinding] = []

    def _finding(self, node: Any, kind: str, scope: str | None = None) -> LowLevelIRFinding:
        return LowLevelIRFinding(
            function=self.function,
            kind=kind,
            node_type=_node_name(node),
            scope=scope,
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


def iter_prim_funcs(value: Any, path: str = "root") -> Iterator[tuple[str, tirx.PrimFunc]]:
    """Recursively yield every TIRx ``PrimFunc`` in a public kernel return value."""
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
            yield from iter_prim_funcs(item, f"{path}[{key!r}]")
        return

    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from iter_prim_funcs(item, f"{path}[{index}]")
        return

    raise TypeError(
        f"{path} is {_node_name(value)}, expected a TIRx PrimFunc, IRModule, "
        "or a list/tuple/dict containing them"
    )


def inspect_low_level_ir(value: Any) -> LowLevelIRReport:
    """Inspect a public ``get_kernel`` return value without lowering it.

    ``value`` may be a TIRx ``PrimFunc``, ``IRModule``, or nested list, tuple,
    or mapping containing those objects.  Direct ``tirx.address_of`` operands
    are reported separately and do not count as memory reads.
    """
    checked_functions: list[str] = []
    violations: list[LowLevelIRFinding] = []
    address_only_loads: list[LowLevelIRFinding] = []

    for path, prim_func in iter_prim_funcs(value):
        checked_functions.append(path)
        visitor = _LowLevelIRVisitor(path)
        visitor(prim_func.body)
        violations.extend(visitor.violations)
        address_only_loads.extend(visitor.address_only_loads)

    if not checked_functions:
        raise TypeError("root must contain at least one TIRx PrimFunc")

    return LowLevelIRReport(
        checked_functions=tuple(checked_functions),
        violations=tuple(violations),
        address_only_loads=tuple(address_only_loads),
    )


def check_low_level_ir(value: Any) -> LowLevelIRReport:
    """Return a report for valid IR, or raise ``LowLevelIRContractError``."""
    report = inspect_low_level_ir(value)
    if not report.ok:
        raise LowLevelIRContractError(report)
    return report


__all__ = [
    "BuilderContractError",
    "BuilderContractFinding",
    "BuilderContractReport",
    "LowLevelIRContractError",
    "LowLevelIRFinding",
    "LowLevelIRReport",
    "check_low_level_ir",
    "check_registry_builder_contract",
    "inspect_low_level_ir",
    "inspect_registry_builder_contract",
    "iter_prim_funcs",
]
