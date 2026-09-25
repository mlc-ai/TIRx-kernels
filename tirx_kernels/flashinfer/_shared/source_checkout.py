# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Locate the FlashInfer source checkout that reference launchers import from.

Some ports drive upstream's own CuTe DSL launchers as their reference, which
means importing modules (and, for one port, a benchmark script) straight out of
a FlashInfer *source tree* rather than an installed wheel.  The tree is resolved
in this order:

1. ``TIRX_FLASHINFER_SOURCE_ROOT`` -- an explicit override.
2. The checkout the importable ``flashinfer`` package lives in.  The registry's
   ``reference_requirements`` gate already pins that checkout to the port's
   upstream commit, so this is the normal case.
3. ``.reference-deps/flashinfer`` under the repository root, which is where
   ``scripts/install_reference_dependencies.py`` clones the lock's revision.
"""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from pathlib import Path

from tirx_kernels.bench.provenance import kernels_repo_root

ENV_OVERRIDE = "TIRX_FLASHINFER_SOURCE_ROOT"
_REPO_ROOT = kernels_repo_root()


def _is_source_root(root: Path) -> bool:
    return (root / "flashinfer" / "__init__.py").is_file()


@cache
def flashinfer_source_root() -> Path:
    """Return the FlashInfer checkout root, raising when none can be found."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        root = Path(override).expanduser().resolve()
        if not _is_source_root(root):
            raise RuntimeError(f"{ENV_OVERRIDE}={override!r} is not a FlashInfer source checkout")
        return root

    tried: list[Path] = []
    spec = importlib.util.find_spec("flashinfer")
    locations = list(getattr(spec, "submodule_search_locations", None) or ()) if spec else []
    for location in locations:
        root = Path(location).resolve().parent
        if _is_source_root(root):
            return root
        tried.append(root)

    installed = _REPO_ROOT / ".reference-deps" / "flashinfer"
    if _is_source_root(installed):
        return installed
    tried.append(installed)

    rendered = ", ".join(str(path) for path in tried) or "no candidates"
    raise RuntimeError(
        "FlashInfer source checkout is unavailable; set "
        f"{ENV_OVERRIDE} or run scripts/install_reference_dependencies.py (tried: {rendered})"
    )
