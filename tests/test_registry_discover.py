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

import importlib
import sys

import pytest

import tirx_kernels
from tirx_kernels.registry import discover_categories, discover_kernels, load_kernel
from tvm import tirx


def test_discover_categories_includes_kernel_dirs() -> None:
    categories = discover_categories()
    assert "gemm" in categories
    assert "attention" in categories
    assert "flashmla" in categories
    assert "bench" not in categories
    assert "bench_suite" not in categories


def test_discover_kernels_finds_known_gemm() -> None:
    kernels = discover_kernels(category="gemm")
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
        "category": "attention",
        "compute_capability": 10,
    }
    assert {config["is_causal"] for config in mod.CONFIGS} == {False, True}
    kernel = mod.get_kernel(batch_size=1, seq_len=256, num_heads=1, head_dim=128, is_causal=False)
    assert sum(tirx.is_buffer_var(param) for param in kernel.params) == 9


def _use_temporary_kernel_package(monkeypatch, root) -> None:
    (root / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(tirx_kernels, "__file__", str(root / "__init__.py"))
    monkeypatch.setattr(tirx_kernels, "__path__", [str(root)])
    importlib.invalidate_caches()


def test_strict_discovery_rejects_malformed_declared_metadata(monkeypatch, tmp_path) -> None:
    category = tmp_path / "fixturecat"
    category.mkdir()
    (category / "__init__.py").write_text("", encoding="utf-8")
    (category / "broken.py").write_text("KERNEL_META = {'name': 'broken'}\n", encoding="utf-8")
    _use_temporary_kernel_package(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="invalid KERNEL_META"):
        discover_kernels(strict=True)

    sys.modules.pop("tirx_kernels.fixturecat.broken", None)
    sys.modules.pop("tirx_kernels.fixturecat", None)


def test_discovery_rejects_duplicate_public_kernel_names(monkeypatch, tmp_path) -> None:
    category = tmp_path / "fixturecat"
    category.mkdir()
    (category / "__init__.py").write_text("", encoding="utf-8")
    metadata = (
        "KERNEL_META = {'name': 'duplicate', 'category': 'fixturecat', 'compute_capability': 10}\n"
    )
    (category / "first.py").write_text(metadata, encoding="utf-8")
    (category / "second.py").write_text(metadata, encoding="utf-8")
    _use_temporary_kernel_package(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="duplicate kernel registry name 'duplicate'"):
        discover_kernels(strict=True)
    with pytest.raises(ValueError, match="duplicate kernel registry name 'duplicate'"):
        load_kernel("duplicate", strict=True)

    sys.modules.pop("tirx_kernels.fixturecat.first", None)
    sys.modules.pop("tirx_kernels.fixturecat.second", None)
    sys.modules.pop("tirx_kernels.fixturecat", None)
