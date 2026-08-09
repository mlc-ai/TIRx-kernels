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

import pytest

from tirx_kernels.attention import flash_attention_backward
from tirx_kernels.deepgemm import paged_mqa_logits_fp4, paged_mqa_logits_fp8
from tirx_kernels.flashkda import bf16_fused_m128 as flashkda_bf16_fused_m128
from tirx_kernels.low_level_ir import check_low_level_ir


@pytest.mark.parametrize(
    "config", flash_attention_backward.CONFIGS, ids=lambda config: config["label"]
)
def test_flash_attention_backward_public_configs_satisfy_low_level_contract(config) -> None:
    parameters = {key: value for key, value in config.items() if key != "label"}

    report = check_low_level_ir(flash_attention_backward.get_kernel(**parameters))

    assert report.checked_functions


@pytest.mark.parametrize(
    "config", flashkda_bf16_fused_m128.CONFIGS, ids=lambda config: config["label"]
)
def test_flashkda_public_configs_satisfy_low_level_contract(config) -> None:
    parameters = {key: value for key, value in config.items() if key != "label"}

    report = check_low_level_ir(flashkda_bf16_fused_m128.get_kernel(**parameters))

    assert report.checked_functions


PAGED_MQA_KERNELS = (paged_mqa_logits_fp4, paged_mqa_logits_fp8)
PAGED_MQA_CONFIGS = [(module, config) for module in PAGED_MQA_KERNELS for config in module.CONFIGS]


@pytest.mark.parametrize(
    ("module", "config"),
    PAGED_MQA_CONFIGS,
    ids=lambda value: value["label"] if isinstance(value, dict) else value.KERNEL_META["name"],
)
def test_paged_mqa_public_configs_satisfy_low_level_contract(module, config) -> None:
    parameters = {key: value for key, value in config.items() if key != "label"}

    report = check_low_level_ir(module.get_kernel(**parameters))

    assert report.checked_functions
