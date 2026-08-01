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

from tirx_kernels.gemm_comm import allgather_gemm, gemm_reduce_scatter
from tirx_kernels.gemm_comm._model_shapes import (
    ALLGATHER_GEMM_MODEL_SHAPES,
    GEMM_RS_MODEL_SHAPES,
    SUPPORTED_WORLD_SIZES,
    make_configs,
)
from tirx_kernels.registry import discover_kernels


def test_gemm_comm_registry_entries() -> None:
    kernels = discover_kernels(category="gemm_comm")
    assert set(kernels) == {"allgather_gemm", "gemm_reduce_scatter"}


@pytest.mark.parametrize(
    "module, model_shapes",
    [(allgather_gemm, ALLGATHER_GEMM_MODEL_SHAPES), (gemm_reduce_scatter, GEMM_RS_MODEL_SHAPES)],
)
def test_gemm_comm_configs_cover_model_shapes_and_tp_degrees(module, model_shapes) -> None:
    assert module.CONFIGS == make_configs(model_shapes)
    assert len(module.CONFIGS) == len(model_shapes) * len(SUPPORTED_WORLD_SIZES)
    assert {config["world_size"] for config in module.CONFIGS} == set(SUPPORTED_WORLD_SIZES)


@pytest.mark.parametrize("module", [allgather_gemm, gemm_reduce_scatter])
def test_gemm_comm_registered_configs_build(module) -> None:
    for config in module.CONFIGS:
        kwargs = {key: value for key, value in config.items() if key != "label"}
        assert module.get_kernel(**kwargs) is not None


@pytest.mark.parametrize(
    "module, overrides",
    [
        (allgather_gemm, {"M": allgather_gemm.M + 1}),
        (allgather_gemm, {"world_size": 2}),
        (gemm_reduce_scatter, {"world_size": 2}),
        (gemm_reduce_scatter, {"scheduler": "static"}),
    ],
)
def test_gemm_comm_rejects_unsupported_public_configs(module, overrides) -> None:
    with pytest.raises(ValueError, match="does not support|supports world_size|supports only"):
        module.get_kernel(**overrides)
