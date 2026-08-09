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

from tirx_kernels.flashmla import sparse_prefill_head128_phase1
from tirx_kernels.low_level_ir import check_low_level_ir
from tvm import tirx


@pytest.mark.parametrize(
    "config", sparse_prefill_head128_phase1.CONFIGS, ids=lambda config: config["label"]
)
def test_public_phase1_kernel_satisfies_the_low_level_launch_and_abi_contract(
    config: dict[str, object],
) -> None:
    parameters = {key: value for key, value in config.items() if key != "label"}

    kernel = sparse_prefill_head128_phase1.get_kernel(**parameters)

    report = check_low_level_ir(kernel)
    assert report.checked_functions == ("root",)
    assert len(kernel.params) == 8
    assert all(tirx.is_buffer_var(parameter) for parameter in kernel.params)
    assert tuple(str(tag) for tag in kernel.attrs["tirx.kernel_launch_params"]) == (
        "blockIdx.x",
        "clusterCtaIdx.x",
        "threadIdx.x",
        "tirx.use_dyn_shared_memory",
    )
