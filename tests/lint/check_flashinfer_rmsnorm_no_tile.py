#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Reject tile primitives in every FlashInfer RMSNorm specialization."""

import sys

import check_gdn_decode_bf16_wide_vec_t1_no_tile as no_tile

from tirx_kernels.flashinfer.norm import rmsnorm as target

no_tile.target = target
no_tile.TARGET = no_tile.REPO / "tirx_kernels/flashinfer/norm/rmsnorm.py"


if __name__ == "__main__":
    sys.exit(no_tile.main())
