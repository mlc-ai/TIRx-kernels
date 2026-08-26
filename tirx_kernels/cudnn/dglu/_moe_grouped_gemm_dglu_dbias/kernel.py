# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Device program for the BF16 MoE grouped GEMM with a fused dGLU backward.

Scaffold stage: the entries below carry the launch shape and the operand ABI
placeholder only. The device body is written in the kernel-sketch stage against
the approved sketch and the line-info PTX export.
"""

import tirx_kernels.kern as K

from . import spec as _spec


def _make_kernel(config, derived):
    """Build the launch sequence for one static specialization."""
    threads_per_cta = derived["threads_per_cta"]
    warps = threads_per_cta // 32

    @K.kernel(warps=warps, arch="sm_100a", min_blocks_per_sm=1, grid=list(derived["grid"]))
    def kernel():
        # ---- kernel body starts here ----
        #
        # Warp specialization (upstream ``:157-181``): epilogue warps 0-3 (warp 0
        # also owns the TMEM allocation and the D store), MMA warp 4, A/B TMA warp
        # 5, C-load warp 6, persistent scheduler warp 7.
        #
        # Nothing is implemented yet. The kernel-sketch stage fixes the operand
        # ABI, the flat shared-memory byte map, the mbarrier protocol and the
        # instruction selection; the correctness gate realizes them here.
        pass

    return kernel


def get_kernel(**config):
    """Return the launch sequence for one static specialization."""
    config = {key: value for key, value in config.items() if key != "label"}
    derived = _spec.derive(config)
    main = _make_kernel(config, derived)
    return [main.func]
