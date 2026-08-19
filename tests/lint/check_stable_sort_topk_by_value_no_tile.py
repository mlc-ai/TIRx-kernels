#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every stable-sort top-k specialization.

Reuses the single-CTA scanner.  This port is a single kernel, so the IR scan
walks one entry point per config -- but it shares ``utils/block_radix_sort.py``
with the filtered port, so that file is scanned here too rather than being
covered only by the sibling checker.
"""

from __future__ import annotations

import sys

import check_radix_topk_single_cta_no_tile as no_tile

from tirx_kernels.flashinfer.topk import stable_sort_topk_by_value as target

no_tile.target = target

_CANDIDATE_TARGETS = (
    "tirx_kernels/flashinfer/topk/stable_sort_topk_by_value.py",
    "tirx_kernels/flashinfer/utils/topk_radix.py",
    "tirx_kernels/flashinfer/utils/block_radix_sort.py",
)
no_tile.TARGETS = tuple(
    no_tile.REPO / rel for rel in _CANDIDATE_TARGETS if (no_tile.REPO / rel).exists()
)


if __name__ == "__main__":
    sys.exit(no_tile.main())
