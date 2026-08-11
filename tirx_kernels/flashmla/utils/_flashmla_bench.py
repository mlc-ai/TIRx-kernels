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

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any


def _import_flash_mla():
    path = os.environ.get("FLASH_MLA_PATH", os.path.expanduser("~/FlashMLA"))
    if path not in sys.path:
        sys.path.insert(0, path)
    import flash_mla

    return flash_mla


def run_flashmla_sparse_prefill(case: dict[str, Any]):
    flash_mla = _import_flash_mla()
    cfg = case["config"]
    out, _, _ = flash_mla.flash_mla_sparse_fwd(
        case["q"],
        case["kv"],
        case["indices"],
        case["sm_scale"],
        d_v=cfg.d_v,
        attn_sink=case["attn_sink"] if cfg.have_attn_sink else None,
        topk_length=case["topk_length"] if cfg.have_topk_length else None,
    )
    return out


def flashmla_reference_builder(case: dict[str, Any]) -> Callable[[], Any]:
    _import_flash_mla()
    return lambda: run_flashmla_sparse_prefill(case)


def run_flashmla_sparse_decode(case: dict[str, Any], sched_meta):
    """Run the exact public sparse-decode dispatch used by the CUDA source."""

    flash_mla = _import_flash_mla()
    cfg = case["config"]
    return flash_mla.flash_mla_with_kvcache(
        case["q"],
        case["kv"],
        None,
        None,
        cfg.d_v,
        sched_meta,
        None,
        case["sm_scale"],
        False,
        True,
        case["indices"],
        case["attn_sink"] if cfg.have_attn_sink else None,
        case["extra_kv"] if cfg.extra_topk else None,
        case["extra_indices"] if cfg.extra_topk else None,
        case["topk_length"] if cfg.have_topk_length else None,
        case["extra_topk_length"] if cfg.have_extra_topk_length else None,
    )


def flashmla_decode_reference_builder(case: dict[str, Any]) -> Callable[[], Any]:
    flash_mla = _import_flash_mla()
    sched_meta, _ = flash_mla.get_mla_metadata()
    # Build and cache FlashMLA's scheduler metadata outside the timed closure.
    run_flashmla_sparse_decode(case, sched_meta)
    return lambda: run_flashmla_sparse_decode(case, sched_meta)
