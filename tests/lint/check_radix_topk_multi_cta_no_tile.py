#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every radix top-k multi-CTA specialization.

Reuses the single-CTA scanner. The multi-CTA kernel reaches its cross-CTA
acquire poll through a private ``PrimFunc`` in the returned ``IRModule``, so
the IR scan walks every function of the module rather than one kernel body.
"""

from __future__ import annotations

import sys
from typing import Any

import check_radix_topk_single_cta_no_tile as no_tile

import tvm
from tirx_kernels.flashinfer.topk import radix_topk_multi_cta as target

no_tile.target = target
no_tile.TARGETS = (
    no_tile.REPO / "tirx_kernels/flashinfer/topk/radix_topk_multi_cta.py",
    no_tile.REPO / "tirx_kernels/flashinfer/utils/topk_radix.py",
)


def _bodies(kernel: Any) -> list[tuple[str, Any]]:
    """Yield every scannable function body, for a PrimFunc or an IRModule."""
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
        kernel = target.get_kernel(**config)
        for name, body in _bodies(kernel):
            scope = f"CONFIGS[{label!r}]" + (f"::{name}" if name else "")
            scanner = no_tile._IRScanner(scope)
            scanner(body)
            findings.extend(scanner.findings)
    return findings


no_tile._ir_findings = _ir_findings


if __name__ == "__main__":
    sys.exit(no_tile.main())
