# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Device kernels for the DSA sparse attention backward pass.

Upstream source:
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py``.

The upstream ``FlashAttentionDSABackwardSm100.__call__`` launches four kernels on
one stream, and this module builds the same four in the same order:

1. ``sum_OdO``   -- per ``(head, query)`` delta ``-sum_d(O * dO)`` and the
   sink-folded log2-domain LSE, written to the LSE/OdO workspace;
2. ``bwd``       -- the 20-warp main kernel: one CTA per query token, heads on
   the MMA M axis, top-k KV tiles walked in reverse, dQ stored through TMA and
   dKV accumulated into an FP32 workspace with global atomics;
3. ``convert``   -- FP32 dKV workspace to the element dtype, unscrambling the two
   store fragment layouts;
4. ``sum_dSink`` -- the attention-sink gradient, warp-reduced then accumulated
   atomically.

Shapes, allocation sizes and the compile key follow the upstream host wrapper
``_interface_sm100.py``.
"""

import tirx_kernels.kern as K

from . import spec

# ``dsa_bwd_sm100.py:64-76``: warp roles of the main kernel and the resulting
# 640-thread block.
LOAD_KV_WARPS = (0, 1, 2, 3)
COMPUTE_WARPS = (4, 5, 6, 7)
REDUCE_WARPS = (8, 9, 10, 11, 12, 13, 14, 15)
MMA_WARP = 16
LOAD_WARP = 17
EMPTY_WARP = 18
BWD_WARPS = 20

# ``dsa_bwd_sm100.py:149-154``: declared per-role register budgets.
REGS_LOAD_KV = 40
REGS_COMPUTE = 128
REGS_REDUCE = 128
REGS_MMA = 40
REGS_LOAD = 40
REGS_EMPTY = 40

# ``dsa_bwd_sm100.py:82-131``: named barrier ids and their participant counts.
BAR_CTA_SYNC = (1, 640)
BAR_TMEM_ALLOC = (2, 416)
BAR_COMPUTE_SYNC = (3, 128)
BAR_LOAD_SYNC = (4, 32)
BAR_LOAD_KV_SYNC = (5, 128)
BAR_REDUCE_SYNC = (6, 256)
BAR_T2R_DKV01_DONE = (7, 288)
BAR_T2R_DKV4_DONE = (8, 288)
BAR_TMEM_DEALLOC = (9, 288)
BAR_T2R_DKV23_DONE = (10, 288)

# ``dsa_bwd_sm100.py:133-147`` with ``block_tile = 64``. dKV2/dKV3 alias
# dKV0/dKV1 and dKV4 aliases dKV0; the three ``t2r_dKV*_done`` barriers are what
# keeps those reuses safe.
TMEM_S_OFFSET = 0
TMEM_DP_OFFSET = 0
TMEM_DKV0_OFFSET = 64
TMEM_DKV1_OFFSET = 128
TMEM_DKV2_OFFSET = TMEM_DKV0_OFFSET
TMEM_DKV3_OFFSET = TMEM_DKV1_OFFSET
TMEM_DQ0_OFFSET = 192
TMEM_DQ1_OFFSET = 256
TMEM_DQ2_OFFSET = 320
TMEM_DQ3_OFFSET = 384
TMEM_DQ4_OFFSET = 448
TMEM_DKV4_OFFSET = TMEM_DKV0_OFFSET
TMEM_ALLOC_COLUMNS = spec.TMEM_CAPACITY_COLUMNS


def make_bwd_kernel(*, head_dim, num_head, dtype, max_topk, has_topk_length):
    """Trace the main backward kernel for one static specialization."""
    spec.check_dispatch(
        {
            "head_dim": head_dim,
            "dtype": dtype,
            "topk_mode": "full",
            "sink_mode": "normal",
            "has_topk_length": has_topk_length,
        }
    )

    @K.kernel(warps=BWD_WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=False)
    def bwd():
        # ---- kernel body starts here ----
        pass

    return bwd


def make_sum_odo_kernel(*, head_dim, num_head, dtype, max_topk):
    """Trace the delta / sink-folded-LSE preprocess kernel."""

    @K.kernel(warps=4, arch="sm_100a", min_blocks_per_sm=1, grid=False)
    def sum_odo():
        # ---- kernel body starts here ----
        pass

    return sum_odo


def make_convert_kernel(*, head_dim, dtype, max_topk):
    """Trace the FP32 dKV workspace to element-dtype conversion kernel."""

    @K.kernel(warps=1, arch="sm_100a", grid=False)
    def convert():
        # ---- kernel body starts here ----
        pass

    return convert


def make_sum_dsink_kernel(*, num_head):
    """Trace the attention-sink gradient reduction kernel."""

    @K.kernel(warps=1, arch="sm_100a", min_blocks_per_sm=1, grid=False)
    def sum_dsink():
        # ---- kernel body starts here ----
        pass

    return sum_dsink


def get_kernel(**config):
    """Return the four device functions in the upstream launch order."""
    head_dim = config["head_dim"]
    num_head = config["num_head"]
    dtype = config["dtype"]
    max_topk = config["max_topk"]
    has_topk_length = config["has_topk_length"]
    return [
        make_sum_odo_kernel(
            head_dim=head_dim, num_head=num_head, dtype=dtype, max_topk=max_topk
        ).func,
        make_bwd_kernel(
            head_dim=head_dim,
            num_head=num_head,
            dtype=dtype,
            max_topk=max_topk,
            has_topk_length=has_topk_length,
        ).func,
        make_convert_kernel(head_dim=head_dim, dtype=dtype, max_topk=max_topk).func,
        make_sum_dsink_kernel(num_head=num_head).func,
    ]
