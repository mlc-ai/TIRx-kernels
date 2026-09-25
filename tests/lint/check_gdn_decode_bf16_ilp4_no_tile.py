#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every BF16 ILP4 GDN specialization."""

from __future__ import annotations

import sys

import check_gdn_decode_bf16_wide_vec_t1_no_tile as no_tile

from tirx_kernels.flashinfer.gdn_decode import gdn_decode_bf16_ilp4 as target

no_tile.target = target
no_tile.TARGET = no_tile.REPO / "tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_ilp4.py"


if __name__ == "__main__":
    sys.exit(no_tile.main())
