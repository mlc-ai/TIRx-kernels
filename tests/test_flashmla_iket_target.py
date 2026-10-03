# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from types import SimpleNamespace

import pytest

from tirx_kernels import registry, runner
from tirx_kernels.ported.flashmla import flash_mla_sparse_fwd as flashmla


def test_dispatch_architectures_match_implementations():
    index = registry.kernel_index(strict=True)
    dispatch_archs = set(index["flash_mla_sparse_fwd"].runtime_cuda_archs)
    for name in (
        "sparse_flashmla_prefill_head64_phase1",
        "sparse_flashmla_prefill_head128_phase1",
        "sparse_flashmla_prefill_head128_small_topk_phase1",
    ):
        assert dispatch_archs == set(index[name].runtime_cuda_archs)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_iket_entry_uses_requested_architecture(monkeypatch, arch):
    monkeypatch.setenv(runner.PREPARE_CUDA_ARCH_ENV, arch)

    class ReachedCompile(Exception):
        pass

    def compile_for_target(_self, _module, *, target, tir_pipeline):
        assert target.arch == arch
        assert tir_pipeline == "tirx"
        raise ReachedCompile

    # Stop at compilation so this test checks target selection without a GPU.
    monkeypatch.setattr(flashmla.iket.IketProfiler, "compile", compile_for_target)
    monkeypatch.setattr(flashmla, "get_kernel", lambda **_kwargs: None)
    monkeypatch.setattr(flashmla.tvm, "IRModule", lambda _functions: object())
    args = SimpleNamespace(s_q=1, s_kv=8192, repeat=1, kernel="all")
    with pytest.raises(ReachedCompile):
        flashmla._profile_iket_workload(args)
