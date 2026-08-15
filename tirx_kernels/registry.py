# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Kernel registry derived from literal ``KERNEL_META`` declarations.

Public kernel identity has one authority: the ``KERNEL_META`` dictionary in the
kernel module itself.  The source index below extracts only that literal
expression, so resolving one public name never imports the complete kernel tree.
The imported module is validated against the same metadata before it is returned.
"""

from __future__ import annotations

import ast
import importlib
import io
import logging
import pkgutil
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import tirx_kernels

_log = logging.getLogger(__name__)

# CLI / package infra — not kernel categories.
_SKIP_CATEGORIES = frozenset({"bench", "test", "bench_suite", "experimental"})

# Shared-helper subpackages inside a category — they hold no kernel modules.
_SKIP_SUBPACKAGES = frozenset({"utils"})

_REQUIRED_META_TYPES = {"name": str, "category": str, "compute_capability": int}


@dataclass(frozen=True)
class KernelRecord:
    """Source-derived identity needed to import exactly one kernel module."""

    name: str
    category: str
    compute_capability: int
    module_name: str
    source_path: Path


def _kernels_root() -> Path:
    return Path(tirx_kernels.__file__).resolve().parent


def discover_categories() -> list[str]:
    """Return sorted category subpackage names under the installed package."""
    root = _kernels_root()
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith("_") or child.name in _SKIP_CATEGORIES or not child.is_dir():
            continue
        if (child / "__init__.py").is_file():
            out.append(child.name)
    return out


def _iter_kernel_module_names(pkg_path: Path) -> list[str]:
    """Return possible kernel module names below one category package."""
    out: list[str] = []
    for _importer, mod_name, is_pkg in pkgutil.iter_modules([str(pkg_path)]):
        if mod_name.startswith("_"):
            continue
        if not is_pkg:
            out.append(mod_name)
        elif mod_name not in _SKIP_SUBPACKAGES:
            out.extend(
                f"{mod_name}.{sub}" for sub in _iter_kernel_module_names(pkg_path / mod_name)
            )
    return out


def _module_source_path(category: str, relative_name: str) -> Path:
    return _kernels_root() / category / Path(*relative_name.split(".")).with_suffix(".py")


def _candidate_sources() -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for category in discover_categories():
        for relative_name in _iter_kernel_module_names(_kernels_root() / category):
            out.append((category, relative_name, _module_source_path(category, relative_name)))
    return out


def _literal_kernel_meta(path: Path) -> dict | None:
    """Extract the literal assigned to ``KERNEL_META`` without parsing the module AST."""
    with tokenize.open(path) as source_file:
        source = source_file.read()
    tokens = iter(tokenize.generate_tokens(io.StringIO(source).readline))
    matches: list[dict] = []

    for token in tokens:
        if token.type != tokenize.NAME or token.string != "KERNEL_META":
            continue

        try:
            equals = next(tokens)
        except StopIteration:
            break
        if equals.type != tokenize.OP or equals.string != "=":
            continue

        expression = []
        depth = 0
        for item in tokens:
            if item.type == tokenize.OP:
                if item.string in "([{":
                    depth += 1
                elif item.string in ")]} ".rstrip():
                    depth -= 1
            if item.type == tokenize.NEWLINE and depth == 0:
                break
            expression.append(item)

        value = ast.literal_eval(
            tokenize.untokenize((item.type, item.string) for item in expression)
        )
        if not isinstance(value, dict):
            raise ValueError(f"{path} KERNEL_META must be a literal dict")
        matches.append(value)

    if len(matches) > 1:
        raise ValueError(f"{path} assigns KERNEL_META more than once")
    return matches[0] if matches else None


def _validate_meta(meta: object, *, category: str, owner: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(meta, dict):
        return [f"expected dict, got {type(meta).__name__}"]

    for field, expected_type in _REQUIRED_META_TYPES.items():
        value = meta.get(field)
        if not isinstance(value, expected_type) or isinstance(value, bool):
            errors.append(f"{field!r} must be {expected_type.__name__}")
        elif expected_type is str and not value:
            errors.append(f"{field!r} must not be empty")
    if meta.get("category") != category:
        errors.append(
            f"'category' is {meta.get('category')!r}, expected containing category {category!r}"
        )
    return errors


def _source_snapshot() -> tuple[tuple[str, str, str, int, int], ...]:
    """Fingerprint candidates so editable installs invalidate the derived index."""
    rows = []
    for category, relative_name, path in _candidate_sources():
        stat = path.stat()
        rows.append((category, relative_name, str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(rows)


@lru_cache(maxsize=1)
def _build_kernel_index(
    snapshot: tuple[tuple[str, str, str, int, int], ...],
) -> tuple[dict[str, KernelRecord], tuple[str, ...]]:
    records: dict[str, KernelRecord] = {}
    diagnostics: list[str] = []
    for category, relative_name, path_string, _mtime_ns, _size in snapshot:
        path = Path(path_string)
        try:
            meta = _literal_kernel_meta(path)
        except (SyntaxError, ValueError, tokenize.TokenError) as error:
            diagnostics.append(f"{path} has invalid KERNEL_META: {error}")
            continue
        if meta is None:
            continue

        module_name = f"tirx_kernels.{category}.{relative_name}"
        errors = _validate_meta(meta, category=category, owner=module_name)
        if errors:
            diagnostics.append(f"{module_name} has invalid KERNEL_META: " + "; ".join(errors))
            continue

        name = meta["name"]
        record = KernelRecord(
            name=name,
            category=category,
            compute_capability=meta["compute_capability"],
            module_name=module_name,
            source_path=path,
        )
        previous = records.get(name)
        if previous is not None:
            raise ValueError(
                f"duplicate kernel registry name {name!r}: {previous.module_name} and {module_name}"
            )
        records[name] = record
    return records, tuple(diagnostics)


def kernel_index(*, strict: bool = False) -> dict[str, KernelRecord]:
    """Return the canonical source-derived public-name index."""
    records, diagnostics = _build_kernel_index(_source_snapshot())
    if diagnostics:
        if strict:
            raise ValueError(diagnostics[0])
        for message in diagnostics:
            _log.warning(message)
    return dict(records)


def _import_record(record: KernelRecord, *, strict: bool) -> ModuleType | None:
    try:
        module = importlib.import_module(record.module_name)
    except Exception as error:
        if strict:
            raise
        _log.warning("tirx_kernels: failed to import %s: %s", record.module_name, error)
        return None

    meta = getattr(module, "KERNEL_META", None)
    errors = _validate_meta(meta, category=record.category, owner=record.module_name)
    if isinstance(meta, dict):
        if meta.get("name") != record.name:
            errors.append(
                f"runtime name {meta.get('name')!r} does not match source index {record.name!r}"
            )
        if meta.get("compute_capability") != record.compute_capability:
            errors.append(
                "runtime compute_capability "
                f"{meta.get('compute_capability')!r} does not match source index "
                f"{record.compute_capability!r}"
            )
    if errors:
        message = f"{record.module_name} has invalid runtime KERNEL_META: " + "; ".join(errors)
        if strict:
            raise ValueError(message)
        _log.warning(message)
        return None
    return module


def load_kernel(name: str, *, strict: bool = False) -> ModuleType:
    """Import exactly the module identified by ``KERNEL_META['name']``.

    Raises ``KeyError`` when no matching valid kernel is installed.  The source
    index is complete and duplicate-checked, but unrelated kernel modules are
    never imported.
    """
    record = kernel_index(strict=strict).get(name)
    if record is None:
        raise KeyError(name)
    module = _import_record(record, strict=strict)
    if module is None:
        raise KeyError(name)
    return module


def check_workload_imports(workloads: list[dict], *, strict: bool = True) -> list[str]:
    """Import every unique kernel referenced in a workloads list. Returns names."""
    names = sorted({workload["kernel"] for workload in workloads})
    for name in names:
        load_kernel(name, strict=strict)
    return names


def discover_kernels(
    *, min_compute_capability: int | None = None, category: str | None = None, strict: bool = False
) -> dict[str, ModuleType]:
    """Return ``{public_name: module}`` for all matching registered kernels."""
    records = kernel_index(strict=strict)
    result: dict[str, ModuleType] = {}
    for name, record in records.items():
        if category is not None and record.category != category:
            continue
        if (
            min_compute_capability is not None
            and record.compute_capability != min_compute_capability
        ):
            continue
        module = _import_record(record, strict=strict)
        if module is not None:
            result[name] = module
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="List registered kernels")
    parser.add_argument("--list", action="store_true", help="List all kernels")
    parser.add_argument(
        "--format", choices=["text", "json", "benchrun"], default="text", help="Output format"
    )
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--cc", type=int, default=None, help="Compute capability filter")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first kernel import failure (for CI import gates)",
    )
    args = parser.parse_args()

    all_kernels = discover_kernels(
        min_compute_capability=args.cc, category=args.category, strict=args.strict
    )

    if args.format == "json":
        out = []
        for name, mod in sorted(all_kernels.items()):
            configs = getattr(mod, "CONFIGS", [])
            out.append({"name": name, "meta": mod.KERNEL_META, "configs": configs})
        print(json.dumps(out, indent=2))
    elif args.format == "benchrun":
        for name, mod in sorted(all_kernels.items()):
            for cfg in getattr(mod, "CONFIGS", []):
                label = cfg.get("label", "default")
                print(
                    f"{name}|{label}|python -m tirx_kernels.bench --kernel {name} "
                    f"--config {label} --json-file {{json_file}}"
                )
    else:
        for name, mod in sorted(all_kernels.items()):
            meta = mod.KERNEL_META
            n_configs = len(getattr(mod, "CONFIGS", []))
            print(
                f"  {name:30s}  {meta.get('category', '?'):12s}  "
                f"sm{meta.get('compute_capability', '?') * 10}  ({n_configs} configs)"
            )
