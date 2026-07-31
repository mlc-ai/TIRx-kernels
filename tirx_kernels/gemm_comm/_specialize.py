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
