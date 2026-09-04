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

import hashlib
from pathlib import Path

from tirx_kernels.cudnn._reference import load_reference_module
from tirx_kernels.target import prepare_cuda_arch

_DSA_PACKAGE = "cudnn.deepseek_sparse_attention"
_INTERFACE_MODULE = f"{_DSA_PACKAGE}.sparse_attention_backward._interface_sm100"
_KERNEL_MODULE = f"{_DSA_PACKAGE}.sparse_attention_backward.dsa_bwd_sm100"
_COMPILER_SHA256 = "d5943a7d9c11663bac98cb6d2a15c2b7226bb34b9dd3c9f0bc983336661ea17c"


def _prepare_reference_target():
    """Extend the frozen host target map for Thor without changing device code."""
    if prepare_cuda_arch() != "sm_110a":
        return
    import torch

    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("the Thor DSA source target requires an actual sm_110 GPU")
    compiler = load_reference_module(f"{_DSA_PACKAGE}.utils.compiler")
    if hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest() != _COMPILER_SHA256:
        raise RuntimeError("the Thor DSA source adapter requires the pinned compiler.py")
    previous = compiler._ARCH_MAP.get((11, 0))
    if previous not in (None, "sm_110a"):
        raise RuntimeError(f"conflicting DSA source target for Thor: {previous!r}")
    if previous is None:
        # compile_options() looks up this map at compile time. All existing
        # device mappings and the source kernel's options remain unchanged.
        compiler._ARCH_MAP[(11, 0)] = "sm_110a"
        compiler.gpu_arch_flag.cache_clear()


def load_interface():
    """Import and return the upstream SM100 host wrapper module."""
    _prepare_reference_target()
    return load_reference_module(_INTERFACE_MODULE)


def load_kernel_module():
    """Import and return the upstream kernel module."""
    return load_reference_module(_KERNEL_MODULE)


def flash_attn_bwd_sm100():
    """Return the upstream ``flash_attn_bwd_sm100`` entry point."""
    return load_interface().flash_attn_bwd_sm100
