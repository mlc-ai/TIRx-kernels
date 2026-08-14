# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Lightweight environment contract for driver-free benchmark preparation."""

from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path

PREPARE_NUM_SMS_ENV = "TIRX_PREPARE_NUM_SMS"
PREPARE_CUDA_ARCH_ENV = "TIRX_PREPARE_CUDA_ARCH"
PREPARE_NVRTC_LIBRARY_ENV = "TIRX_PREPARE_NVRTC_LIBRARY"


def pin_prepare_cuda_toolchain(env: dict[str, str]) -> None:
    """Make NVCC, PTXAS, and NVRTC resolve from one CUDA toolkit."""
    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc is None:
        raise FileNotFoundError("prepare requires nvcc on PATH")
    cuda_home = Path(nvcc).resolve().parent.parent
    cuda_bin = cuda_home / "bin"
    cuda_lib = cuda_home / "lib64"
    nvrtc_library = cuda_lib / "libnvrtc.so.13"
    if not (cuda_bin / "ptxas").is_file():
        raise FileNotFoundError(f"prepare CUDA toolkit has no ptxas: {cuda_bin}")
    if not nvrtc_library.is_file():
        raise FileNotFoundError(f"prepare CUDA toolkit has no libnvrtc.so.13: {cuda_lib}")

    def prepend(name: str, entry: Path) -> None:
        parts = [part for part in env.get(name, "").split(os.pathsep) if part]
        resolved_entry = str(entry.resolve())
        retained = [part for part in parts if str(Path(part).resolve()) != resolved_entry]
        env[name] = os.pathsep.join([str(entry), *retained])

    prepend("PATH", cuda_bin)
    prepend("LD_LIBRARY_PATH", cuda_lib)
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env[PREPARE_NVRTC_LIBRARY_ENV] = str(nvrtc_library)


def preload_prepare_nvrtc() -> ctypes.CDLL | None:
    """Load the selected NVRTC before TVM or torch can select another copy."""
    library_path = os.environ.get(PREPARE_NVRTC_LIBRARY_ENV)
    if library_path is None:
        return None
    return ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)


def validate_prepare_nvrtc_binding(preloaded: ctypes.CDLL | None) -> dict[str, object]:
    """Prove cuda-bindings selected the preloaded toolkit's NVRTC instance."""
    if preloaded is None:
        raise RuntimeError(f"prepared child is missing {PREPARE_NVRTC_LIBRARY_ENV}")

    expected_major = ctypes.c_int()
    expected_minor = ctypes.c_int()
    if preloaded.nvrtcVersion(ctypes.byref(expected_major), ctypes.byref(expected_minor)) != 0:
        raise RuntimeError("selected prepare NVRTC failed nvrtcVersion()")

    from cuda.bindings import nvrtc

    result, actual_major, actual_minor = nvrtc.nvrtcVersion()
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"cuda-bindings nvrtcVersion() failed: {result}")
    expected = (expected_major.value, expected_minor.value)
    actual = (actual_major, actual_minor)
    if actual != expected:
        raise RuntimeError(
            "cuda-bindings selected a different NVRTC than the prepare toolkit: "
            f"expected {expected[0]}.{expected[1]}, got {actual[0]}.{actual[1]}"
        )

    nvcc = shutil.which("nvcc")
    ptxas = shutil.which("ptxas")
    return {
        "cuda_home": os.environ.get("CUDA_HOME"),
        "nvcc": str(Path(nvcc).resolve()) if nvcc else None,
        "ptxas": str(Path(ptxas).resolve()) if ptxas else None,
        "nvrtc_library": str(Path(preloaded._name).resolve()),
        "nvrtc_version": [actual_major, actual_minor],
    }
