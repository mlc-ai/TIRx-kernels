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

import pytest

from tirx_kernels.deepgemm import tf32_hc_prenorm_gemm
from tirx_kernels.low_level_ir import check_low_level_ir


@pytest.mark.parametrize("config", tf32_hc_prenorm_gemm.CONFIGS, ids=lambda config: config["label"])
def test_public_configs_satisfy_low_level_ir_contract(config) -> None:
    parameters = {key: value for key, value in config.items() if key != "label"}

    report = check_low_level_ir(tf32_hc_prenorm_gemm.get_kernel(**parameters))

    assert report.checked_functions
