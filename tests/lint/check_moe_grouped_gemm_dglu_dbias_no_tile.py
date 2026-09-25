#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every MoE BF16 dGLU specialization.

A discrete-weight or dynamically scheduled specialization builds two entries --
a descriptor pre-kernel and the main kernel -- so the IR scan walks the whole
launch sequence rather than a single body.
"""

from __future__ import annotations

import sys

import check_gdn_decode_bf16_wide_vec_t1_no_tile as no_tile

from tirx_kernels.cudnn.dglu import moe_grouped_gemm_dglu_dbias as target

no_tile.target = target
no_tile.TARGET = no_tile.REPO / "tirx_kernels/cudnn/dglu/_moe_grouped_gemm_dglu_dbias/kernel.py"


def _ir_findings() -> list:
    findings = []
    for config in target.CONFIGS:
        label = str(config["label"])
        for index, func in enumerate(target.get_kernel(**config)):
            scanner = no_tile._IRScanner(f"CONFIGS[{label!r}] entry {index}")
            scanner(func.body)
            findings.extend(scanner.findings)
    return findings


no_tile._ir_findings = _ir_findings


if __name__ == "__main__":
    sys.exit(no_tile.main())
