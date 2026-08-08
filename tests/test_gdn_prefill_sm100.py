# Copyright (c) 2026 The TIRX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from pathlib import Path

from tirx_kernels.bench_suite import run as bench_suite_run
from tirx_kernels.flashinfer import gdn_prefill_sm100


def test_gdn_prefill_sm100_public_contract() -> None:
    assert gdn_prefill_sm100.KERNEL_META == {
        "name": "gdn_prefill_sm100",
        "category": "flashinfer",
        "compute_capability": 10,
    }
    assert not hasattr(gdn_prefill_sm100, "BENCH_CONFIGS")
    assert set(gdn_prefill_sm100.__all__) == {
        "CONFIGS",
        "KERNEL_META",
        "get_kernel",
        "prepare_data",
        "run_bench",
        "run_test",
    }


def test_gdn_prefill_sm100_workloads_exactly_match_configs() -> None:
    config_labels = [config["label"] for config in gdn_prefill_sm100.CONFIGS]
    workloads = bench_suite_run.load_kernel_configs("gdn_prefill_sm100")
    workload_labels = [workload["config"] for workload in workloads]

    assert len(config_labels) == len(set(config_labels)) == 120
    assert len(workload_labels) == len(set(workload_labels)) == 120
    assert all(workload["kernel"] == "gdn_prefill_sm100" for workload in workloads)
    assert all(workload["num_gpus"] == 1 for workload in workloads)
    assert all("timer" not in workload for workload in workloads)
    assert all("warmup" not in workload and "repeat" not in workload for workload in workloads)
    assert workload_labels == config_labels
    assert bench_suite_run.BASELINE_IMPL_BY_KERNEL["gdn_prefill_sm100"] == "flashinfer_cutedsl"


def test_gdn_prefill_sm100_covers_frozen_cartesian_matrix() -> None:
    expected_heads = {(2, 8), (4, 16), (8, 32), (16, 64), (16, 32), (16, 48), (16, 16), (32, 32)}
    expected_sequences = {
        (65536,),
        (32768,),
        (16384,),
        (8192,),
        (4096,),
        (2048,),
        (6144, 2048),
        (4096, 4096),
        (2048, 6144),
        (1024, 7168),
        (2048,) * 4,
        (1024,) * 8,
        (8192,) * 8,
        (8192,) * 16,
        (8192,) * 32,
    }
    observed = {
        (config["hq"], config["hv"], tuple(config["seq_lens"]))
        for config in gdn_prefill_sm100.CONFIGS
    }

    assert observed == {
        (hq, hv, seq_lens) for hq, hv in expected_heads for seq_lens in expected_sequences
    }


def test_gdn_prefill_sm100_rejects_tile_primitive_execution_apis() -> None:
    source = Path(gdn_prefill_sm100.__file__).read_text()
    forbidden = (
        "Tx.copy(",
        "Tx.copy_async(",
        "Tx.gemm(",
        "Tx.gemm_async(",
        "tvm.backend.cuda.tile_primitive",
        "T.Kernel(",
        "T.Parallel(",
        "T.Pipelined(",
        "T.alloc_fragment(",
        "@T.prim_func",
        "/home/",
        ".porting/",
    )

    assert "T.device_entry()" in source
    assert not {needle for needle in forbidden if needle in source}
    assert '"evict_normal"' not in source
    # The five TMA copies pass their zero cache policy as the ptx chain's
    # trailing positional operand (the legacy kwarg spelling is retired).
    assert source.count("T.uint64(0),") == 5
    assert source.count("_descriptor_copy_payload(o_map, descriptor_o)") == 1
    assert "% num_sequences" not in source
