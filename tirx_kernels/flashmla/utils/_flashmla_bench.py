# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any


def _import_flash_mla():
    from tirx_kernels.reference_requirements import load_reference
    from tirx_kernels.runner import prepare_cuda_arch

    if prepare_cuda_arch() == "sm_110a":
        return load_reference("flash-mla")

    path = os.environ.get("FLASH_MLA_PATH", os.path.expanduser("~/FlashMLA"))
    if path not in sys.path:
        sys.path.insert(0, path)
    import flash_mla

    return flash_mla


def run_flashmla_sparse_prefill_outputs(case: dict[str, Any]):
    flash_mla = _import_flash_mla()
    cfg = case["config"]
    return flash_mla.flash_mla_sparse_fwd(
        case["q"],
        case["kv"],
        case["indices"],
        case["sm_scale"],
        d_v=cfg.d_v,
        attn_sink=case["attn_sink"] if cfg.have_attn_sink else None,
        topk_length=case["topk_length"] if cfg.have_topk_length else None,
    )


def run_flashmla_sparse_prefill(case: dict[str, Any]):
    return run_flashmla_sparse_prefill_outputs(case)[0]


def validate_flashmla_sparse_prefill(case, oracle, *, output_rtol: float) -> None:
    """Check all three public source outputs against the oracle and TIRx."""
    import torch

    source_outputs = run_flashmla_sparse_prefill_outputs(case)
    torch.cuda.synchronize()
    for index, key in enumerate(("out", "max_logits", "lse")):
        rtol = output_rtol if index == 0 else 2.01 / 65536
        atol = 5e-3 if index == 0 else 1e-6
        torch.testing.assert_close(source_outputs[index], oracle[index], rtol=rtol, atol=atol)
        torch.testing.assert_close(case[key], source_outputs[index], rtol=rtol, atol=atol)


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
