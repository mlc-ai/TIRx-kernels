#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Check that the README kernel section matches the kernel registry.

Three things must hold:

* every registered kernel is linked exactly once, and the link opens its module;
* untagged kernels declare the three default CUDA architectures; ``**[+sm_110a]**``
  adds Thor to that set, while other tags list the complete supported set;
* the architecture overview table lists the correct kernel count and the exact
  set of single-architecture kernels for each architecture.

``KERNEL_META["runtime_cuda_archs"]`` is the authority for all three.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
DEFAULT_ARCHS = ("sm_100a", "sm_103a", "sm_107a")
ALL_ARCHS = (*DEFAULT_ARCHS, "sm_110a")

_LINK = re.compile(r"\[`([^`]+)`\]\((tirx_kernels/[^)]+\.py)\)( \*\*\[([^\]\n]*)\]\*\*)?")
_TABLE_ROW = re.compile(r"^\| `(sm_[0-9]+[af]?)`[^|]*\| (\d+) \|([^|]*)\|$", re.MULTILINE)


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


def main() -> int:
    kernels = _registry()
    text = README.read_text()
    errors: list[str] = []

    linked: dict[str, tuple[str, str | None]] = {}
    for name, path, tag_markup, tag in _LINK.findall(text):
        if name in linked:
            errors.append(f"{name}: linked more than once")
        linked[name] = (path, tag if tag_markup else None)

    for name, (path, archs) in sorted(kernels.items()):
        if name not in linked:
            errors.append(f"{name}: not linked in README.md")
            continue
        readme_path, tag = linked[name]
        if readme_path != path:
            errors.append(f"{name}: README links {readme_path}, module is {path}")
        if archs == DEFAULT_ARCHS:
            expected_tag = None
        elif archs == ALL_ARCHS:
            expected_tag = "+sm_110a"
        else:
            expected_tag = ", ".join(archs)
        if tag != expected_tag:
            errors.append(
                f"{name}: README tag is {tag}, runtime_cuda_archs {archs} needs {expected_tag}"
            )
    for name in sorted(set(linked) - set(kernels)):
        errors.append(f"{name}: linked in README.md but not registered")

    table = {
        arch: (int(count), set(re.findall(r"`([^`]+)`", names)))
        for arch, count, names in _TABLE_ROW.findall(text)
    }
    for arch in ALL_ARCHS:
        count = sum(arch in archs for _, archs in kernels.values())
        only = {name for name, (_, archs) in kernels.items() if archs == (arch,)}
        if arch not in table:
            errors.append(f"overview table: missing row for {arch}")
        elif table[arch] != (count, only):
            errors.append(
                f"overview table: {arch} row should list {count} kernels and "
                f"{sorted(only)}, found {table[arch][0]} and {sorted(table[arch][1])}"
            )

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
