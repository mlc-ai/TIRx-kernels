# Copyright (c) 2026 The TIRX Authors
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
from __future__ import annotations

from tirx_kernels.registry import discover_categories, discover_kernels, load_kernel
from tvm import tirx


def test_discover_categories_includes_kernel_dirs() -> None:
    categories = discover_categories()
    assert "basic" in categories
    assert "flashattention" in categories
    assert "flashmla" in categories
    assert "bench" not in categories
    assert "bench_suite" not in categories


def test_discover_kernels_finds_known_gemm() -> None:
    kernels = discover_kernels(category="basic")
    assert "fp16_bf16_gemm" in kernels
    assert "nvfp4_gemm" in kernels


def test_load_kernel_finds_single_module() -> None:
    mod = load_kernel("nvfp4_gemm")
    assert mod.KERNEL_META["name"] == "nvfp4_gemm"


def test_load_kernel_finds_flashmla_unified_entry() -> None:
    mod = load_kernel("flash_mla_sparse_fwd")
    assert mod.KERNEL_META["category"] == "flashmla"


def test_load_kernel_finds_flash_attention_backward() -> None:
    mod = load_kernel("flash_attention_backward_sm100", strict=True)

    assert mod.KERNEL_META == {
        "name": "flash_attention_backward_sm100",
        "category": "flashattention",
        "compute_capability": 10,
    }
    assert {config["is_causal"] for config in mod.CONFIGS} == {False, True}
    kernel = mod.get_kernel(batch_size=1, seq_len=256, num_heads=1, head_dim=128, is_causal=False)
    assert sum(tirx.is_buffer_var(param) for param in kernel.params) == 9
