# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Redirect PyTorch's absolute-path NVRTC preload to the gate-selected CUDA toolkit."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_ORIGINAL_CDLL = ctypes.CDLL
NVRTC_LIBRARY_ENV = "TIRX_BENCH_NVRTC_LIBRARY"
NVRTC_BUILTINS_LIBRARY_ENV = "TIRX_BENCH_NVRTC_BUILTINS_LIBRARY"


def _selected_library(requested: object) -> str | None:
    name = Path(os.fsdecode(requested)).name if isinstance(requested, (str, bytes, os.PathLike)) else ""
    if name.startswith("libnvrtc-builtins.so"):
        return os.environ.get(NVRTC_BUILTINS_LIBRARY_ENV)
    if name.startswith("libnvrtc.so"):
        return os.environ.get(NVRTC_LIBRARY_ENV)
    return None


def _gate_cdll(requested, *args, **kwargs):
    selected = _selected_library(requested)
    return _ORIGINAL_CDLL(selected or requested, *args, **kwargs)


def install() -> None:
    if ctypes.CDLL is not _gate_cdll:
        ctypes.CDLL = _gate_cdll
