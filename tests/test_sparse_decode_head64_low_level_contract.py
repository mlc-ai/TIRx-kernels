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
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tirx_kernels.flashmla import sparse_decode_head64
from tirx_kernels.low_level_ir import check_low_level_ir
from tvm.tirx import analysis


@pytest.mark.parametrize("config", sparse_decode_head64.CONFIGS, ids=lambda config: config["label"])
def test_public_sparse_decode_pair_satisfies_the_low_level_launch_contract(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object]
) -> None:
    # Exercise the public factory without allocating or touching a CUDA device.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )
    parameters = {key: value for key, value in config.items() if key != "label"}

    kernels = sparse_decode_head64.get_kernel(**parameters, device="cuda:0")

    assert isinstance(kernels, list)
    assert len(kernels) == 2
    report = check_low_level_ir(kernels)
    assert report.checked_functions == ("root[0]", "root[1]")
    for kernel in kernels:
        assert analysis.verify_tirx_well_formed(kernel)
        assert analysis.verify_well_formed(kernel)
        assert analysis.verify_ssa(kernel)
        assert analysis.verify_memory(kernel)

    main_launch = tuple(str(tag) for tag in kernels[0].attrs["tirx.kernel_launch_params"])
    combine_launch = tuple(str(tag) for tag in kernels[1].attrs["tirx.kernel_launch_params"])
    assert main_launch == (
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.x",
        "tirx.use_dyn_shared_memory",
    )
    expected_combine = ("blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x")
    if config["b"] == 2:
        expected_combine += ("tirx.use_programtic_dependent_launch",)
    assert combine_launch == expected_combine
