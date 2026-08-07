# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Runtime tests for MegaKernelMOE."""

import pytest

from tirx_kernels.megakernel import moe
from tirx_kernels.runner import run_kernel_test


@pytest.mark.parametrize("batch_size", [1, 128])
def test_megakernel_moe(batch_size):
    run_kernel_test("megakernel_moe", {"batch_size": batch_size, "world_size": 1})


def test_run_test_all_schedulers_uses_flashinfer_reference_once(monkeypatch):
    compiled = []
    tir_runs = []
    reference_builds = {}
    reference_calls = {"flashinfer": 0}
    validations = []
    synchronizations = []

    def compile_schedulers(schedulers, batch_size, world_size):
        compiled.append((schedulers, batch_size, world_size))
        return object(), {scheduler: object() for scheduler in schedulers}

    def make_tir_case(*, scheduler, **kwargs):
        return {"scheduler": scheduler}

    def build_reference(name):
        def build(*args, **kwargs):
            reference_builds[name] = (args, kwargs)

            def run(case):
                reference_calls[name] += 1
                return name

            return run

        return build

    monkeypatch.setattr(moe, "_require_cuda_sm100", lambda: None)
    monkeypatch.setattr(moe, "_compile_moe_schedulers", compile_schedulers)
    monkeypatch.setattr(moe, "_reset_prepare_data_cache", lambda: None)
    monkeypatch.setattr(moe, "prepare_data", lambda batch_size, mk: {})
    monkeypatch.setattr(moe, "_make_tir_case", make_tir_case)
    monkeypatch.setattr(
        moe,
        "_run_tir_case_and_check_finite",
        lambda case: tir_runs.append(case["tir"]["scheduler"]),
    )
    monkeypatch.setattr(moe, "_build_flashinfer_full_reference", build_reference("flashinfer"))
    monkeypatch.setattr(
        moe,
        "_build_sglang_full_reference",
        lambda mk: pytest.fail("run_test must not build the SGLang benchmark reference"),
    )
    monkeypatch.setattr(
        moe,
        "_validate_tir_matches_reference",
        lambda case, reference: validations.append((case["tir"]["scheduler"], reference)),
    )
    monkeypatch.setattr(moe.torch.cuda, "synchronize", lambda: synchronizations.append(True))

    moe.run_test(batch_size=512)

    assert compiled == [(moe._SUPPORTED_SCHEDULERS, 512, 1)]
    assert tir_runs == list(moe._SUPPORTED_SCHEDULERS)
    assert reference_builds["flashinfer"][1] == {"tune": False}
    assert reference_calls == {"flashinfer": 1}
    assert validations == [(scheduler, "flashinfer") for scheduler in moe._SUPPORTED_SCHEDULERS]
    assert synchronizations == [True]
