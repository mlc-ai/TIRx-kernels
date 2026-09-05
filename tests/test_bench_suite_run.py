# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tirx_kernels.bench_suite import ratio_diff
from tirx_kernels.bench_suite import run as bench_run

_OUR_CHILD_PID = 4001
_FOREIGN_PID = 9001


def _pool_with_fake_smi(monkeypatch, apps_rows: list[str]) -> bench_run.GpuPool:
    def fake_smi(args: list[str]) -> list[str]:
        query = args[0]
        if query == "--query-gpu=index,utilization.gpu":
            return ["0, 0", "1, 0"]
        if query == "--query-gpu=index,memory.used,memory.total":
            return ["0, 7000, 180000", "1, 400, 180000"]
        if query == "--query-gpu=index,memory.total":
            return ["0, 180000", "1, 180000"]
        if query == "--query-gpu=index,uuid":
            return ["0, GPU-aaa", "1, GPU-bbb"]
        if query == "--query-compute-apps=pid,gpu_uuid,used_memory":
            return apps_rows
        raise AssertionError(f"unexpected nvidia-smi query: {args!r}")

    monkeypatch.setattr(bench_run.GpuPool, "_nvidia_smi", staticmethod(fake_smi))
    monkeypatch.setattr(bench_run, "_our_pids", lambda: {_OUR_CHILD_PID})
    return bench_run.GpuPool(allowed={"0", "1"})


def test_parked_own_child_does_not_occupy_its_affinity_card(monkeypatch):
    """An interference-parked bench child's residual context must not mark the
    card externally occupied, or the child starves waiting to reacquire it."""
    pool = _pool_with_fake_smi(monkeypatch, [f"{_OUR_CHILD_PID}, GPU-aaa, 7000"])
    assert pool._occupied_indices() == set()
    assert pool.try_acquire_exact(("0",)) is None  # occupancy not refreshed yet
    pool.refresh_external_occupancy()
    assert pool.try_acquire_exact(("0",)) == ("0",)


def test_foreign_resident_memory_occupies_card(monkeypatch):
    pool = _pool_with_fake_smi(monkeypatch, [f"{_FOREIGN_PID}, GPU-aaa, 53000"])
    assert pool._occupied_indices() == {"0"}
    pool.refresh_external_occupancy()
    assert pool.try_acquire_exact(("0",)) is None
    assert pool.try_acquire_exact(("1",)) == ("1",)


def test_small_foreign_residual_is_forgiven_by_idle_floor(monkeypatch):
    pool = _pool_with_fake_smi(monkeypatch, [f"{_FOREIGN_PID}, GPU-aaa, 300"])
    assert pool._occupied_indices() == set()


def test_mixed_own_and_foreign_memory_counts_only_foreign(monkeypatch):
    pool = _pool_with_fake_smi(
        monkeypatch,
        [
            f"{_OUR_CHILD_PID}, GPU-aaa, 4000",
            f"{_FOREIGN_PID}, GPU-aaa, 2000",
            f"{_OUR_CHILD_PID}, GPU-bbb, 6000",
        ],
    )
    assert pool._occupied_indices() == {"0"}


def test_gpu_compile_profile_supports_sm107(monkeypatch):
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda _handle: "NVIDIA Graphics Device",
        nvmlDeviceGetCudaComputeCapability=lambda _handle: (10, 7),
        nvmlDeviceGetNumGpuCores=lambda _handle: 216 * 128,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    assert bench_run.gpu_compile_profile({"0", "1"}) == {
        "name": "NVIDIA Graphics Device",
        "compute_capability": [10, 7],
        "cuda_arch": "sm_107a",
        "num_sms": 216,
    }


def test_gpu_compile_profile_supports_sm103(monkeypatch):
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda _handle: "NVIDIA GB300",
        nvmlDeviceGetCudaComputeCapability=lambda _handle: (10, 3),
        nvmlDeviceGetNumGpuCores=lambda _handle: 152 * 128,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    assert bench_run.gpu_compile_profile({"0", "1"}) == {
        "name": "NVIDIA GB300",
        "compute_capability": [10, 3],
        "cuda_arch": "sm_103a",
        "num_sms": 152,
    }


def test_gpu_compile_profile_rejects_mixed_arch_pool(monkeypatch):
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: f"GPU {handle}",
        nvmlDeviceGetCudaComputeCapability=lambda handle: (10, 0) if handle == 0 else (10, 7),
        nvmlDeviceGetNumGpuCores=lambda handle: (148 if handle == 0 else 216) * 128,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    with pytest.raises(ValueError, match="heterogeneous compile profiles"):
        bench_run.gpu_compile_profile({"0", "1"})


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


def test_default_roster_is_available_on_sm103_and_sm107():
    workloads = bench_run.load_config_dir()

    sm107, sm107_incompatible = bench_run.partition_workloads_by_arch(workloads, "sm_107a")
    sm103, sm103_incompatible = bench_run.partition_workloads_by_arch(workloads, "sm_103a")
    sm100, sm100_incompatible = bench_run.partition_workloads_by_arch(workloads, "sm_100a")

    def kernels(rows):
        return {workload["kernel"] for workload in rows}

    # 262 rows run everywhere; twelve single-architecture kernels contribute three default rows
    # each: cake_vsa_{blk128_compact,longseq,ultrasparse_bsr}_sm100 (sm_100a),
    # {bmm_fp8_rubin,dense_blockscaled_gemm_sm107,grouped_gemm_masked_rubin} (sm_107a), and
    # blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103,
    # blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103,
    # blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103, cake_vsa_longseq_sm103,
    # fastcu_nvfp4_gemm_gb300, and flash_attention4_fp4 (sm_103a).
    assert len(sm107) == 271
    assert len(sm107_incompatible) == 27
    assert kernels(sm107_incompatible) == {
        "blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103",
        "blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103",
        "blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103",
        "cake_vsa_blk128_compact_sm100",
        "cake_vsa_longseq_sm100",
        "cake_vsa_longseq_sm103",
        "cake_vsa_ultrasparse_bsr_sm100",
        "fastcu_nvfp4_gemm_gb300",
        "flash_attention4_fp4",
    }
    assert len(sm103) == 280
    assert len(sm103_incompatible) == 18
    assert kernels(sm103_incompatible) == {
        "bmm_fp8_rubin",
        "cake_vsa_blk128_compact_sm100",
        "cake_vsa_longseq_sm100",
        "cake_vsa_ultrasparse_bsr_sm100",
        "dense_blockscaled_gemm_sm107",
        "grouped_gemm_masked_rubin",
    }
    assert len(sm100) == 271
    assert len(sm100_incompatible) == 27
    assert kernels(sm100_incompatible) == {
        "blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103",
        "blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103",
        "bmm_fp8_rubin",
        "blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103",
        "cake_vsa_longseq_sm103",
        "dense_blockscaled_gemm_sm107",
        "grouped_gemm_masked_rubin",
        "fastcu_nvfp4_gemm_gb300",
        "flash_attention4_fp4",
    }


def test_validate_workload_archs_rejects_mismatch_before_prepare(monkeypatch):
    records = {"rubin": SimpleNamespace(runtime_cuda_archs=("sm_107a",))}
    monkeypatch.setattr(bench_run, "kernel_index", lambda strict: records)

    with pytest.raises(ValueError, match=r"sm_100a.*rubin"):
        bench_run.validate_workload_archs([{"kernel": "rubin"}], "sm_100a")
