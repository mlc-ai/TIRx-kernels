#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Check that the README kernel section matches the kernel registry.

The ``## Kernels`` section groups kernels by upstream library and entry point.
Every registered kernel must be linked exactly once, optionally followed by one
annotation in angle brackets. Without an annotation a kernel runs on the default
architectures (``sm_100a``, ``sm_103a``, ``sm_107a``). The architecture tokens of
an annotation are derived from ``KERNEL_META["runtime_cuda_archs"]``: the full
list when it is not a superset of the default set, or ``+sm_xxx`` for each
architecture beyond it.

The summary table at the top of the section must give, per architecture, how
many kernels run on it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
ARCH_ORDER = ("sm_100a", "sm_103a", "sm_107a", "sm_110a")
DEFAULT_ARCHS = ("sm_100a", "sm_103a", "sm_107a")

_LINK = re.compile(r"\[`([^`]+)`\]\((tirx_kernels/[^)]+\.py)\)(?: ⟨([^⟩]*)⟩)?")
_SUMMARY_ROW = re.compile(r"^\| `(sm_[0-9]+[af]?)` \| (\d+) \|$", re.MULTILINE)


def _registry() -> dict[str, tuple[str, tuple[str, ...]]]:
    sys.path.insert(0, str(REPO_ROOT))
    from tirx_kernels import registry

    index, diagnostics = registry._build_kernel_index(registry._source_snapshot())
    if diagnostics:
        raise SystemExit("\n".join(diagnostics))
    return {
        name: (str(record.source_path.relative_to(REPO_ROOT)), tuple(record.runtime_cuda_archs))
        for name, record in index.items()
    }


def _ordered(archs: tuple[str, ...]) -> list[str]:
    ordered = [arch for arch in ARCH_ORDER if arch in archs]
    return ordered + sorted(arch for arch in archs if arch not in ARCH_ORDER)


def _expected_annotation(archs: tuple[str, ...]) -> str:
    extra = [arch for arch in _ordered(archs) if arch not in DEFAULT_ARCHS]
    if set(archs) - set(extra) == set(DEFAULT_ARCHS):
        return " ".join(f"+{arch}" for arch in extra)
    return " ".join(_ordered(archs))


def main() -> int:
    kernels = _registry()
    text = README.read_text()
    start = text.index("## Kernels")
    end = text.index("\n## ", start + 1)
    section = text[start:end]
    errors: list[str] = []

    rows: dict[str, tuple[str, str]] = {}
    for name, path, annotation in _LINK.findall(section):
        if name in rows:
            errors.append(f"{name}: listed more than once")
        rows[name] = (path, annotation.strip())

    for name, (path, archs) in sorted(kernels.items()):
        if name not in rows:
            errors.append(f"{name}: not listed in README.md")
            continue
        readme_path, annotation = rows[name]
        if readme_path != path:
            errors.append(f"{name}: README links {readme_path}, module is {path}")
        expected = _expected_annotation(archs)
        if annotation != expected:
            errors.append(f"{name}: annotation is {annotation!r}, expected {expected!r}")
    for name in sorted(set(rows) - set(kernels)):
        errors.append(f"{name}: listed in README.md but not registered")

    summary = {arch: int(runs) for arch, runs in _SUMMARY_ROW.findall(section)}
    for arch in ARCH_ORDER:
        runs = sum(arch in archs for _, archs in kernels.values())
        if runs == 0 and arch not in summary:
            continue
        if arch not in summary:
            errors.append(f"summary table: missing row for {arch}")
        elif summary[arch] != runs:
            errors.append(f"summary table: {arch} should read {runs}, found {summary[arch]}")
    for arch in set(summary) - set(ARCH_ORDER):
        errors.append(f"summary table: unexpected row for {arch}")

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
