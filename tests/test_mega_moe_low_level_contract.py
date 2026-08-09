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

from tirx_kernels.deepgemm import mega_moe
from tirx_kernels.low_level_ir import check_low_level_ir


def _kernel_parameters(config: dict[str, object], **overrides: object) -> dict[str, object]:
    parameters = {key: value for key, value in config.items() if key != "label"}
    parameters.update(overrides)
    return parameters


@pytest.mark.parametrize("config", mega_moe.CONFIGS, ids=lambda config: config["label"])
def test_public_configs_satisfy_low_level_contract(config) -> None:
    report = check_low_level_ir(mega_moe.get_kernel(**_kernel_parameters(config)))

    assert report.checked_functions


_P1_CONFIG = next(config for config in mega_moe.CONFIGS if config["num_processes"] == 1)
_P2_CONFIG = next(config for config in mega_moe.CONFIGS if config["num_processes"] == 2)


@pytest.mark.parametrize(
    "config", (_P1_CONFIG, _P2_CONFIG), ids=lambda config: f"{config['label']}-collect-stats"
)
def test_collect_stats_specializations_satisfy_low_level_contract(config) -> None:
    report = check_low_level_ir(
        mega_moe.get_kernel(**_kernel_parameters(config, collect_stats=True))
    )

    assert report.checked_functions


def test_multi_process_no_timeout_printf_specialization_satisfies_low_level_contract() -> None:
    report = check_low_level_ir(
        mega_moe.get_kernel(**_kernel_parameters(_P2_CONFIG, emit_nvl_barrier_timeout_printf=False))
    )

    assert report.checked_functions
