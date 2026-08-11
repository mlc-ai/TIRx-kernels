# Copyright (c) 2026 The TIRx Authors
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

"""Model-shape matrix shared by the GemmComm registry entries."""

from __future__ import annotations

SUPPORTED_WORLD_SIZES = (1, 4)

GEMM_RS_MODEL_SHAPES = (
    ("qwen3_8b", 8192, 4096, 12288),
    ("llama_3_1_8b", 8192, 4096, 14336),
    ("gemma_2_9b", 8192, 3584, 14336),
    ("gemma_2_27b", 8192, 4608, 36864),
    ("qwen3_32b", 8192, 5120, 25600),
    ("llama_3_1_70b", 8192, 8192, 28672),
    ("gpt_3_175b", 8192, 12288, 49152),
    ("llama_3_1_405b", 8192, 16384, 53248),
)

ALLGATHER_GEMM_MODEL_SHAPES = (
    ("qwen3_8b", 8192, 24576, 4096),
    ("llama_3_1_8b", 8192, 28672, 4096),
    ("gemma_2_9b", 8192, 28672, 3584),
    ("gemma_2_27b", 8192, 73728, 4608),
    ("qwen3_32b", 8192, 51200, 5120),
    ("llama_3_1_70b", 8192, 57344, 8192),
    ("gpt_3_175b", 8192, 98304, 12288),
    ("llama_3_1_405b", 8192, 106496, 16384),
)


def make_configs(model_shapes: tuple[tuple[str, int, int, int], ...]) -> list[dict[str, object]]:
    """Expand eight model shapes over the supported TP degrees."""

    return [
        {
            "M": M,
            "N": N,
            "K": K,
            "world_size": world_size,
            "dtype": "float16",
            "scheduler": "dynamic",
            "label": f"tp{world_size}_m{M}_n{N}_k{K}_fp16_dynamic",
        }
        for _model, M, N, K in model_shapes
        for world_size in SUPPORTED_WORLD_SIZES
    ]


def shape_set(
    model_shapes: tuple[tuple[str, int, int, int], ...],
) -> frozenset[tuple[int, int, int]]:
    """Return the dimension triples accepted by one GemmComm direction."""

    return frozenset((M, N, K) for _model, M, N, K in model_shapes)


__all__ = [
    "ALLGATHER_GEMM_MODEL_SHAPES",
    "GEMM_RS_MODEL_SHAPES",
    "SUPPORTED_WORLD_SIZES",
    "make_configs",
    "shape_set",
]
