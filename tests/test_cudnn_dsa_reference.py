# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tirx_kernels.cudnn.dsa._sparse_attention_backward import reference


def test_non_thor_does_not_load_or_change_the_compiler(monkeypatch):
    monkeypatch.setattr(reference, "prepare_cuda_arch", lambda: "sm_100a")
    loader = Mock(side_effect=AssertionError("unexpected source import"))
    monkeypatch.setattr(reference, "load_reference_module", loader)
    reference._prepare_reference_target()
    loader.assert_not_called()


def test_thor_only_adds_its_exact_host_target(monkeypatch, tmp_path):
    source = tmp_path / "compiler.py"
    source.write_text("frozen host compiler")
    original = {(9, 0): "sm_90a", (10, 0): "sm_100a", (10, 3): "sm_103a"}
    compiler = SimpleNamespace(
        __file__=str(source), _ARCH_MAP=dict(original), gpu_arch_flag=Mock()
    )
    monkeypatch.setattr(reference, "prepare_cuda_arch", lambda: "sm_110a")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (11, 0))
    monkeypatch.setattr(reference, "load_reference_module", lambda _: compiler)
    monkeypatch.setattr(
        reference, "_COMPILER_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    reference._prepare_reference_target()
    reference._prepare_reference_target()
    assert compiler._ARCH_MAP == {**original, (11, 0): "sm_110a"}
    compiler.gpu_arch_flag.cache_clear.assert_called_once()
    source.write_text("changed source")
    with pytest.raises(RuntimeError, match="pinned compiler"):
        reference._prepare_reference_target()


def test_thor_target_rejects_another_actual_device(monkeypatch):
    monkeypatch.setattr(reference, "prepare_cuda_arch", lambda: "sm_110a")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    with pytest.raises(RuntimeError, match="actual sm_110"):
        reference._prepare_reference_target()
