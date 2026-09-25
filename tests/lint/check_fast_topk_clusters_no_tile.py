#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every clustered exact top-k specialization.

Reuses the single-CTA scanner.  The three source wrappers share one device
worker and are built from one ``get_kernel``, so the IR scan walks one entry
point per config; the config matrix is what spreads it across the modes, the
cluster widths, and both index widths.

``utils/topk_radix.py`` is scanned here as well as by the sibling checkers: the
monotone key map, the shared-memory intrinsics, and the warp shuffle all come
from it, and a tile primitive introduced there would reach this port too.
"""

from __future__ import annotations

import sys

import check_radix_topk_single_cta_no_tile as no_tile

from tirx_kernels.flashinfer.topk import fast_topk_clusters as target

no_tile.target = target

_CANDIDATE_TARGETS = (
    "tirx_kernels/flashinfer/topk/fast_topk_clusters.py",
    "tirx_kernels/flashinfer/utils/topk_radix.py",
)
no_tile.TARGETS = tuple(
    no_tile.REPO / rel for rel in _CANDIDATE_TARGETS if (no_tile.REPO / rel).exists()
)


if __name__ == "__main__":
    sys.exit(no_tile.main())
