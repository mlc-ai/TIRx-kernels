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

from tirx_kernels.deepgemm import mqa_logits_fp8
from tirx_kernels.low_level_ir import check_low_level_ir

_PUBLIC_CONFIGS = tuple(mqa_logits_fp8.CONFIGS)


@pytest.mark.parametrize(
    "config", _PUBLIC_CONFIGS, ids=[config["label"] for config in _PUBLIC_CONFIGS]
)
def test_public_mqa_logits_fp8_configs_satisfy_low_level_ir_contract(config) -> None:
    assert _PUBLIC_CONFIGS
    assert len({item["label"] for item in _PUBLIC_CONFIGS}) == len(_PUBLIC_CONFIGS)

    report = check_low_level_ir(mqa_logits_fp8.get_kernel(**config))

    assert report.checked_functions
