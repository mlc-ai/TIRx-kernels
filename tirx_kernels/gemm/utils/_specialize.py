# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Load implementation-preserving, import-time kernel specializations."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

_SPECIALIZATION_LOCK = threading.RLock()


def load_specialized_module(
    *, package: str, stem: str, source: str, key: tuple[int, ...], environment: Mapping[str, int]
) -> ModuleType:
    """Execute one kernel source file under a compile-time integer environment."""

    suffix = "_".join(str(value) for value in key)
    module_name = f"{package}._{stem}_specialized_{suffix}"
    with _SPECIALIZATION_LOCK:
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached

        previous = {name: os.environ.get(name) for name in environment}
        os.environ.update({name: str(value) for name, value in environment.items()})
        try:
            spec = importlib.util.spec_from_file_location(module_name, Path(source))
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create specialization module {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
            return module
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


__all__ = ["load_specialized_module"]
