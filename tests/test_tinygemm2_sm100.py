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

from __future__ import annotations

import torch

from tirx_kernels.bench_suite import run as bench_suite_run
from tirx_kernels.flashinfer import tinygemm2_sm100
from tvm.tirx.tile_primitive import TilePrimitiveCall


def test_tinygemm2_sm100_public_contract() -> None:
    assert tinygemm2_sm100.KERNEL_META == {
        "name": "tinygemm2_sm100",
        "category": "flashinfer",
        "compute_capability": 10,
    }
    assert set(tinygemm2_sm100.__all__) == {
        "BENCH_CONFIGS",
        "CONFIGS",
        "KERNEL_META",
        "get_kernel",
        "prepare_data",
        "run_bench",
        "run_test",
    }


def test_tinygemm2_sm100_workloads_match_configs() -> None:
    labels = [config["label"] for config in tinygemm2_sm100.CONFIGS]
    workloads = bench_suite_run.load_kernel_configs("tinygemm2_sm100")

    assert [workload["config"] for workload in workloads] == labels
    assert all(workload["kernel"] == "tinygemm2_sm100" for workload in workloads)
    assert all(workload["default"] for workload in workloads)
    assert bench_suite_run.BASELINE_IMPL_BY_KERNEL["tinygemm2_sm100"] == "flashinfer_sm100"


def test_tinygemm2_sm100_b200_dispatch_boundaries() -> None:
    select = tinygemm2_sm100._select_stage
    assert select(8, 1024, 1024, 148) == 4
    assert select(8, 1024, 1025, 148) == 8
    assert select(8, 4736, 2048, 148) == 8  # exactly 2 * num_sms CTAs
    assert select(8, 4752, 2048, 148) == 4  # first large-grid CTA beyond the boundary
    assert [select(c["B"], c["O"], c["K"], 148) for c in tinygemm2_sm100.CONFIGS] == [
        4,
        4,
        8,
        8,
        4,
        8,
        4,
        4,
    ]


def test_tinygemm2_sm100_four_predispatch_variants_have_no_tile_calls() -> None:
    import tvm

    for stage in (4, 8):
        for use_pdl in (False, True):
            calls = []
            func = tinygemm2_sm100.get_kernel(8, 1024, 2048, stage=stage, use_pdl=use_pdl)
            tvm.tirx.stmt_functor.post_order_visit(
                func.body,
                lambda node: calls.append(node) if isinstance(node, TilePrimitiveCall) else None,
            )
            assert not calls


def test_tinygemm2_sm100_uses_u32_thread_id_and_runtime_loops() -> None:
    import tvm

    func = tinygemm2_sm100.get_kernel(13, 1024, 2048, stage=8, use_pdl=False)
    scope_ids = []
    runtime_k_loops = []

    def visit(node) -> None:
        if isinstance(node, tvm.tirx.ScopeIdDefStmt):
            scope_ids.extend(getattr(node, "def").def_ids)
        elif isinstance(node, tvm.tirx.For) and node.loop_var.name == "ki":
            runtime_k_loops.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)

    tid = next(var for var in scope_ids if var.name == "tid_u32")
    assert str(tid.ty) == "uint32"
    assert len(runtime_k_loops) == 3
    assert all(str(loop.loop_var.ty) == "uint32" for loop in runtime_k_loops)


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        import pytest

        pytest.skip("TinyGEMM2 SM100 GPU tests require B200")


def _assert_direct_variant(B: int, output_size: int, K: int, stage: int, use_pdl: bool) -> None:
    case = tinygemm2_sm100.prepare_data(B, output_size, K)
    actual = torch.zeros_like(case["out"])
    expected = torch.zeros_like(case["out"])
    tinygemm2_sm100._run_tirx(case, stage, use_pdl, actual)
    tinygemm2_sm100._run_flashinfer(case, stage, use_pdl, expected)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


def test_tinygemm2_sm100_direct_variants() -> None:
    _require_sm100()
    for B, output_size, K in ((1, 128, 720), (8, 1024, 2048)):
        for stage in (4, 8):
            for use_pdl in (False, True):
                _assert_direct_variant(B, output_size, K, stage, use_pdl)


def test_tinygemm2_sm100_batch_oob_and_k_tail() -> None:
    _require_sm100()
    for B in range(1, 8):
        _assert_direct_variant(B, 128, 312, 4, False)
        _assert_direct_variant(B, 128, 312, 4, True)


def test_tinygemm2_sm100_pdl_back_to_back() -> None:
    _require_sm100()
    for num_launches in (2, 8):
        cases = [tinygemm2_sm100.prepare_data(8, 1024, 2048, seed=i) for i in range(num_launches)]
        actual = [torch.zeros_like(case["out"]) for case in cases]
        expected = [torch.zeros_like(case["out"]) for case in cases]
        for case, output in zip(cases, actual, strict=True):
            tinygemm2_sm100._run_tirx(case, 8, True, output)
        for case, output in zip(cases, expected, strict=True):
            tinygemm2_sm100._run_flashinfer(case, 8, True, output)
        torch.cuda.synchronize()
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected, strict=True))
