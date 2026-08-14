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

"""Launch the formal IR Builder gate with CUDA 13.2 NVRTC preloaded."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from tirx_kernels.bench_suite._nvrtc132_sitecustomize import (
    NVRTC_BUILTINS_LIBRARY_ENV,
    NVRTC_LIBRARY_ENV,
)

CUDA_HOME_ENV = "TIRX_BENCH_CUDA_HOME"
DEFAULT_CUDA_HOME = Path("/usr/local/cuda-13.2")


def _cuda_library(cuda_home: Path, name: str) -> Path:
    candidates = (cuda_home / "targets" / "x86_64-linux" / "lib" / name, cuda_home / "lib64" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"CUDA 13.2 {name} was not found under {cuda_home}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _launcher_environment(
    base: dict[str, str], nvrtc_library: Path, builtins_library: Path, cuda_home: Path, hook_dir: Path
) -> dict[str, str]:
    env = dict(base)
    env[NVRTC_LIBRARY_ENV] = str(nvrtc_library)
    env[NVRTC_BUILTINS_LIBRARY_ENV] = str(builtins_library)
    env["CUDA_PATH"] = str(cuda_home)
    env["CUDA_HOME"] = str(cuda_home)
    env.pop("CONDA_PREFIX", None)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(hook_dir) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    return env


def _materialize_torch_nvrtc_hook(hook_dir: Path) -> None:
    hook_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("_nvrtc132_sitecustomize.py")
    shutil.copy2(source, hook_dir / "_tirx_bench_nvrtc132_sitecustomize.py")
    (hook_dir / "sitecustomize.py").write_text(
        "from _tirx_bench_nvrtc132_sitecustomize import install\n\ninstall()\n"
    )


def main() -> None:
    if "--ir-builder-migration-gate" in sys.argv[1:]:
        raise SystemExit("the wrapper supplies --ir-builder-migration-gate; do not pass it twice")
    cuda_home = Path(os.environ.get(CUDA_HOME_ENV, DEFAULT_CUDA_HOME)).expanduser().resolve()
    nvrtc_library = _cuda_library(cuda_home, "libnvrtc.so.13")
    builtins_library = _cuda_library(cuda_home, "libnvrtc-builtins.so.13.2")
    hook_dir = Path(tempfile.mkdtemp(prefix="tirx-bench-nvrtc132-"))
    _materialize_torch_nvrtc_hook(hook_dir)
    env = _launcher_environment(
        os.environ, nvrtc_library, builtins_library, cuda_home, hook_dir
    )
    command = [
        sys.executable,
        "-m",
        "tirx_kernels.bench_suite",
        "--ir-builder-migration-gate",
        "--mem-threshold",
        "0.5",
        *sys.argv[1:],
    ]
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
