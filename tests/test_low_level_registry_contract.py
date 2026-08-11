# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from tirx_kernels.low_level_ir import check_low_level_ir
from tirx_kernels.registry import discover_kernels


def _registry_cases() -> list:
    registry = discover_kernels(strict=True)
    assert registry, "no registered kernels discovered"

    cases = []
    for kernel_name, module in sorted(registry.items()):
        assert callable(getattr(module, "get_kernel", None)), (
            f"{kernel_name}: get_kernel must be callable"
        )
        configs = getattr(module, "CONFIGS", None)
        assert isinstance(configs, list) and configs, (
            f"{kernel_name}: CONFIGS must be a non-empty list"
        )

        labels = []
        for config in configs:
            assert isinstance(config, dict), f"{kernel_name}: config must be a dict"
            assert all(isinstance(key, str) for key in config), (
                f"{kernel_name}: config keys must be strings"
            )
            label = config.get("label")
            assert isinstance(label, str) and label, f"{kernel_name}: config label must be set"
            labels.append(label)
            parameters = {key: value for key, value in config.items() if key != "label"}
            cases.append(
                pytest.param(kernel_name, module, parameters, id=f"{kernel_name}[{label}]")
            )

        assert len(labels) == len(set(labels)), f"{kernel_name}: duplicate config labels"
    return cases


@pytest.mark.parametrize(("kernel_name", "module", "parameters"), _registry_cases())
def test_registered_public_configs_satisfy_low_level_ir_contract(
    monkeypatch: pytest.MonkeyPatch,
    kernel_name: str,
    module: ModuleType,
    parameters: dict[str, object],
) -> None:
    # Kernel construction is target-specific but must not require allocating a
    # CUDA tensor.  Pin the two factories that derive compile-time scheduling
    # constants from the target GPU to the repository's SM100 reference target.
    monkeypatch.setenv("TIRX_DEEPGEMM_NUM_SMS_OVERRIDE", "148")
    if kernel_name == "sparse_flashmla_decode_head64":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _device: SimpleNamespace(multi_processor_count=148),
        )
        parameters = {**parameters, "device": "cuda:0"}

    report = check_low_level_ir(module.get_kernel(**parameters))

    assert report.ok
