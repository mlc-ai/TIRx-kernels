# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Kernel registry — auto-discovers kernel modules with KERNEL_META."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from types import ModuleType

import tirx_kernels

_log = logging.getLogger(__name__)

# CLI / package infra — not kernel categories.
_SKIP_CATEGORIES = frozenset({"bench", "test", "bench_suite", "experimental"})

# Shared-helper subpackages inside a category — they hold no kernel modules.
_SKIP_SUBPACKAGES = frozenset({"utils"})

_REQUIRED_META_TYPES = {"name": str, "category": str, "compute_capability": int}


def _kernels_root() -> Path:
    return Path(tirx_kernels.__file__).resolve().parent


def discover_categories() -> list[str]:
    """Return sorted category subpackage names under the installed ``tirx_kernels`` package."""
    root = _kernels_root()
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith("_") or child.name in _SKIP_CATEGORIES or not child.is_dir():
            continue
        if (child / "__init__.py").is_file():
            out.append(child.name)
    return out


def _iter_kernel_module_names(pkg_path: Path) -> list[str]:
    """Return kernel module names under ``pkg_path``, relative to it and dot-separated.

    Categories may group their kernels into subpackages (``flashinfer/quantization/``
    and friends bucket the ports by the FlashInfer Python entry point they back), so
    this walks into subpackages as well.  ``utils`` subpackages hold shared helpers,
    not kernels, and are skipped.
    """
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


def _import_kernel_module(category: str, mod_name: str, *, strict: bool) -> ModuleType | None:
    pkg_name = f"tirx_kernels.{category}"
    try:
        mod = importlib.import_module(f"{pkg_name}.{mod_name}")
    except Exception as e:
        if strict:
            raise
        _log.warning("tirx_kernels: failed to import %s.%s: %s", pkg_name, mod_name, e)
        return None
    if not hasattr(mod, "KERNEL_META"):
        return None

    meta = mod.KERNEL_META
    errors: list[str] = []
    if not isinstance(meta, dict):
        errors.append(f"expected dict, got {type(meta).__name__}")
    else:
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

    if errors:
        message = f"tirx_kernels.{category}.{mod_name} has invalid KERNEL_META: " + "; ".join(
            errors
        )
        if strict:
            raise ValueError(message)
        _log.warning(message)
        return None

    return mod


def load_kernel(name: str, *, strict: bool = False) -> ModuleType:
    """Import a single kernel module by ``KERNEL_META['name']``.

    The complete registry is validated before selecting the requested name, so
    duplicate public identities can never be hidden by scan order or a cache.
    Raises ``KeyError`` when no matching kernel is installed.
    """
    kernels = discover_kernels(strict=strict)
    try:
        return kernels[name]
    except KeyError:
        raise KeyError(name) from None


def check_workload_imports(workloads: list[dict], *, strict: bool = True) -> list[str]:
    """Import every unique kernel referenced in a workloads list. Returns kernel names."""
    names = sorted({w["kernel"] for w in workloads})
    kernels = discover_kernels(strict=strict)
    for name in names:
        if name not in kernels:
            raise KeyError(name)
    return names


def _scan_category(category: str, *, strict: bool) -> dict[str, ModuleType]:
    """Import all modules under tirx_kernels/<category>/ that expose KERNEL_META."""
    result: dict[str, ModuleType] = {}
    pkg_path = _kernels_root() / category
    if not pkg_path.is_dir():
        return result
    for mod_name in _iter_kernel_module_names(pkg_path):
        mod = _import_kernel_module(category, mod_name, strict=strict)
        if mod is not None:
            name = mod.KERNEL_META["name"]
            if name in result:
                raise ValueError(
                    f"duplicate kernel registry name {name!r}: "
                    f"{result[name].__name__} and {mod.__name__}"
                )
            result[name] = mod
    return result


def discover_kernels(
    *, min_compute_capability: int | None = None, category: str | None = None, strict: bool = False
) -> dict[str, ModuleType]:
    """Return ``{name: module}`` for all registered kernels in the installed package.

    Parameters
    ----------
    min_compute_capability : int, optional
        If given, only include kernels whose ``compute_capability`` is
        *exactly* this value.  (We use exact match because sm100a kernels
        won't run on sm90a, etc.)
    category : str, optional
        If given, only scan this category subdirectory.
    strict : bool, optional
        If True, raise on the first kernel module import failure instead of
        logging a warning and continuing.
    """
    categories = [category] if category else discover_categories()
    result: dict[str, ModuleType] = {}
    for cat in categories:
        category_kernels = _scan_category(cat, strict=strict)
        for name, mod in category_kernels.items():
            if name in result:
                raise ValueError(
                    f"duplicate kernel registry name {name!r}: "
                    f"{result[name].__name__} and {mod.__name__}"
                )
            result[name] = mod
    if min_compute_capability is not None:
        result = {
            name: mod
            for name, mod in result.items()
            if mod.KERNEL_META.get("compute_capability") == min_compute_capability
        }
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
        # Output in bench-run.sh format: KERNEL|SIZE|COMMAND
        for name, mod in sorted(all_kernels.items()):
            for cfg in getattr(mod, "CONFIGS", []):
                label = cfg.get("label", "default")
                print(
                    f"{name}|{label}|python -m tirx_kernels.bench --kernel {name} --config {label} --json-file {{json_file}}"
                )
    else:
        for name, mod in sorted(all_kernels.items()):
            meta = mod.KERNEL_META
            n_configs = len(getattr(mod, "CONFIGS", []))
            print(
                f"  {name:30s}  {meta.get('category', '?'):12s}  sm{meta.get('compute_capability', '?') * 10}  ({n_configs} configs)"
            )
