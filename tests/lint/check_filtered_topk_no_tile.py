#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every filtered top-k specialization.

Reuses the single-CTA scanner.  This port is a two-kernel pipeline, so the IR
scan walks the unified kernel *and* the finalize kernel for every config.
``get_finalize_kernel`` returns ``None`` for the configs where the dispatcher
issues no second launch; those contribute the unified kernel only.
"""

from __future__ import annotations

import sys
from typing import Any

import check_radix_topk_single_cta_no_tile as no_tile

import tvm
from tirx_kernels.flashinfer.topk import filtered_topk as target

no_tile.target = target
_CANDIDATE_TARGETS = (
    "tirx_kernels/flashinfer/topk/filtered_topk.py",
    "tirx_kernels/flashinfer/utils/topk_radix.py",
    # Added when the BlockRadixSort emitters land with the finalize kernel.
    "tirx_kernels/flashinfer/utils/block_radix_sort.py",
    # Traits, row scan, collectors, refine rounds and both overflow fallbacks.
    "tirx_kernels/flashinfer/utils/filtered_topk_ops.py",
)
no_tile.TARGETS = tuple(
    no_tile.REPO / rel for rel in _CANDIDATE_TARGETS if (no_tile.REPO / rel).exists()
)


def _bodies(kernel: Any) -> list[tuple[str, Any]]:
    """Yield every scannable function body, for a PrimFunc or an IRModule."""
    if kernel is None:
        return []
    if isinstance(kernel, tvm.IRModule):
        return [
            (global_var.name_hint, function.body)
            for global_var, function in kernel.functions.items()
            if hasattr(function, "body")
        ]
    return [("", kernel.body)]


def _ir_findings() -> list[no_tile.Finding]:
    findings: list[no_tile.Finding] = []
    for config in target.CONFIGS:
        label = str(config["label"])
        params = {key: value for key, value in config.items() if key != "label"}
        for entry, build in (("main", target.get_kernel), ("finalize", target.get_finalize_kernel)):
            for name, body in _bodies(build(**params)):
                scope = f"CONFIGS[{label!r}]::{entry}" + (f"::{name}" if name else "")
                scanner = no_tile._IRScanner(scope)
                scanner(body)
                findings.extend(scanner.findings)
    return findings


no_tile._ir_findings = _ir_findings


if __name__ == "__main__":
    sys.exit(no_tile.main())
