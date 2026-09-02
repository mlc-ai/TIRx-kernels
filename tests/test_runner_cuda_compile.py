# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import pytest

from tirx_kernels import runner


def test_offline_cuda_compile_defaults_to_nvrtc(monkeypatch) -> None:
    monkeypatch.delenv(runner.TVM_CUDA_COMPILE_MODE_ENV, raising=False)
    monkeypatch.delenv(runner.TVM_CUDA_NVRTC_EXTRA_OPTS_ENV, raising=False)

    assert runner._offline_cuda_compile_parameters("sm_107a") == {
        "target_format": "cubin",
        "arch": "sm_107a",
        "options": None,
        "compiler": "nvrtc",
    }


def test_offline_cuda_compile_allows_explicit_nvcc(monkeypatch) -> None:
    monkeypatch.setenv(runner.TVM_CUDA_COMPILE_MODE_ENV, "nvcc")
    monkeypatch.setenv(runner.TVM_CUDA_NVRTC_EXTRA_OPTS_ENV, "-lineinfo")

    assert runner._offline_cuda_compile_parameters("sm_107a") == {
        "target_format": "fatbin",
        "arch": ["-gencode", "arch=compute_107a,code=sm_107a"],
        "options": ["-lineinfo"],
        "compiler": "nvcc",
    }


def test_offline_cuda_compile_rejects_unknown_mode(monkeypatch) -> None:
    monkeypatch.setenv(runner.TVM_CUDA_COMPILE_MODE_ENV, "unknown")

    with pytest.raises(
        ValueError, match=r"Invalid TVM_CUDA_COMPILE_MODE: unknown\. Expected 'nvcc' or 'nvrtc'\."
    ):
        runner._offline_cuda_compile_parameters("sm_107a")
