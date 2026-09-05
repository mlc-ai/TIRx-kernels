# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Loader for the upstream DSA sparse attention backward kernel.

Upstream source:
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm100.py``.

The kernel this port follows resolves from the cuDNN Frontend source install
pinned in ``reference-dependencies.json`` and installed by
``scripts/install_reference_dependencies.py``. Released wheels are not an
acceptable substitute: the 1.26.0 wheel present during this port predated both
fp16 element support and two of the kernel's TMEM ordering barriers, which is
why the install replaces any such wheel with the pinned source tree (whose DSA
kernel is byte-identical to the revision cited above).
"""

from tirx_kernels.cudnn._reference import load_reference_module

_DSA_PACKAGE = "cudnn.deepseek_sparse_attention"
_INTERFACE_MODULE = f"{_DSA_PACKAGE}.sparse_attention_backward._interface_sm100"
_KERNEL_MODULE = f"{_DSA_PACKAGE}.sparse_attention_backward.dsa_bwd_sm100"


def load_interface():
    """Import and return the upstream SM100 host wrapper module."""
    return load_reference_module(_INTERFACE_MODULE)


def load_kernel_module():
    """Import and return the upstream kernel module."""
    return load_reference_module(_KERNEL_MODULE)


def flash_attn_bwd_sm100():
    """Return the upstream ``flash_attn_bwd_sm100`` entry point."""
    return load_interface().flash_attn_bwd_sm100
