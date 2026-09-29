# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tirx_kernels.bench_suite import ratio_diff
from tirx_kernels.bench_suite import run as bench_run


def test_validate_workload_archs_accepts_exact_arch(monkeypatch):
    records = {"kernel": SimpleNamespace(runtime_cuda_archs=("sm_100a",))}
    monkeypatch.setattr(bench_run, "kernel_index", lambda strict: records)

    bench_run.validate_workload_archs([{"kernel": "kernel"}], "sm_100a")


def test_partition_workloads_by_arch_keeps_only_exact_matches(monkeypatch):
    records = {
        "blackwell": SimpleNamespace(runtime_cuda_archs=("sm_100a",)),
        "rubin": SimpleNamespace(runtime_cuda_archs=("sm_107a",)),
    }
    monkeypatch.setattr(bench_run, "kernel_index", lambda strict: records)
    workloads = [{"kernel": "blackwell"}, {"kernel": "rubin"}]

    supported, incompatible = bench_run.partition_workloads_by_arch(workloads, "sm_107a")

    assert supported == [{"kernel": "rubin"}]
    assert incompatible == [{"kernel": "blackwell"}]


def test_expected_keys_scopes_default_roster_to_arch(monkeypatch):
    records = {
        "blackwell": SimpleNamespace(runtime_cuda_archs=("sm_100a",)),
        "rubin": SimpleNamespace(runtime_cuda_archs=("sm_107a",)),
    }
    monkeypatch.setattr(bench_run, "kernel_index", lambda strict: records)
    monkeypatch.setattr(
        ratio_diff,
        "_load_config_dir",
        lambda: [
            {"kernel": "blackwell", "config": "blackwell_config"},
            {"kernel": "rubin", "config": "rubin_config"},
        ],
    )

    keys, errors = ratio_diff._expected_keys("sm_107a")

    assert errors == []
    assert keys == {("rubin", "rubin_config")}


def test_default_roster_includes_curated_rubin_bmm():
    workloads = bench_run.load_config_dir()
    labels = {workload["config"] for workload in workloads if workload["kernel"] == "bmm_fp8_rubin"}

    assert labels == {
        "bench_t2_e4m3_bf16_b1_m512_n4096_k2720",
        "bench_t4_e5m2_fp16_b1_m1024_n4096_k3072",
        "bench_t7_e4m3_fp32_b2_m4096_n1024_k3072",
    }


def test_default_roster_includes_curated_dense_blockscaled_gemm_sm107():
    workloads = bench_run.load_config_dir()
    labels = {
        workload["config"]
        for workload in workloads
        if workload["kernel"] == "dense_blockscaled_gemm_sm107"
    }

    assert labels == {
        "bench_t1_mxfp4_bf16_m4096_n1024_k3072",
        "bench_t3_mxfp4_fp16_m256_n10304_k2688",
        "bench_t4_nvfp4_bf16_m4096_n2048_k7168",
    }


def test_default_roster_is_partitioned_by_registered_architecture():
    workloads = bench_run.load_config_dir()
    records = bench_run.kernel_index(strict=True)

    def identity(workload):
        return workload["kernel"], workload["config"]

    expected = {identity(workload) for workload in workloads}
    for cuda_arch in ("sm_100a", "sm_103a", "sm_107a", "sm_110a"):
        compatible, incompatible = bench_run.partition_workloads_by_arch(workloads, cuda_arch)
        compatible_ids = {identity(workload) for workload in compatible}
        incompatible_ids = {identity(workload) for workload in incompatible}
        assert compatible_ids.isdisjoint(incompatible_ids)
        assert compatible_ids | incompatible_ids == expected
        assert all(
            cuda_arch in records[workload["kernel"]].runtime_cuda_archs for workload in compatible
        )
        assert all(
            cuda_arch not in records[workload["kernel"]].runtime_cuda_archs
            for workload in incompatible
        )


def test_validate_workload_archs_rejects_mismatch_before_prepare(monkeypatch):
    records = {"rubin": SimpleNamespace(runtime_cuda_archs=("sm_107a",))}
    monkeypatch.setattr(bench_run, "kernel_index", lambda strict: records)

    with pytest.raises(ValueError, match=r"sm_100a.*rubin"):
        bench_run.validate_workload_archs([{"kernel": "rubin"}], "sm_100a")
