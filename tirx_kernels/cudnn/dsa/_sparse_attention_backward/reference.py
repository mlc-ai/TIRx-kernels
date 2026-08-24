# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Loader for the upstream DSA sparse attention backward kernel.

Upstream source:
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm100.py``.

The kernel this port follows lives in a cuDNN Frontend source checkout named by
``CUDNN_FRONTEND_PATH``, the contract the other cuDNN Frontend ports in this
repository use. It is deliberately not the ``cudnn`` wheel that may also be
installed: the released package can carry an older revision of this kernel (the
1.26.0 wheel present during this port predates both fp16 element support and two
of the kernel's TMEM ordering barriers), and benchmarking against it would
compare TIRx to a different implementation than the one cited above.

Only the ``cudnn.deepseek_sparse_attention`` subtree is redirected. The rest of
``cudnn`` still resolves to the installed package, because ``cudnn.api_base``
reaches a compiled extension that a source checkout alone does not provide, and
because nothing this port follows lives outside the DSA subtree.
"""

import importlib
import os
import sys
import types

_DSA_PACKAGE = "cudnn.deepseek_sparse_attention"
_INTERFACE_MODULE = f"{_DSA_PACKAGE}.sparse_attention_backward._interface_sm100"
_KERNEL_MODULE = f"{_DSA_PACKAGE}.sparse_attention_backward.dsa_bwd_sm100"


def checkout_root():
    """Absolute path of the cuDNN Frontend source checkout."""
    root = os.environ.get("CUDNN_FRONTEND_PATH")
    if not root:
        raise RuntimeError("CUDNN_FRONTEND_PATH must point to a cuDNN Frontend source checkout")
    return root


def _dsa_package_dir():
    package_dir = os.path.join(checkout_root(), "python", "cudnn", "deepseek_sparse_attention")
    if not os.path.isdir(package_dir):
        raise RuntimeError(f"CUDNN_FRONTEND_PATH does not contain the DSA package: {package_dir}")
    return package_dir


def _bind_checkout():
    """Redirect ``cudnn.deepseek_sparse_attention`` at the checkout.

    Idempotent: the bench suite retries workloads in the same process, and the
    correctness runner imports this more than once per session.
    """
    package_dir = _dsa_package_dir()

    existing = sys.modules.get(_DSA_PACKAGE)
    if existing is not None:
        bound = getattr(existing, "__path__", None)
        if bound and os.path.realpath(next(iter(bound))) == os.path.realpath(package_dir):
            return
        raise RuntimeError(
            f"{_DSA_PACKAGE} is already imported from {bound}; cannot rebind it to {package_dir}. "
            "Import the reference before anything else pulls in the installed cuDNN DSA package."
        )

    # Importing the real ``cudnn`` first is what makes ``cudnn.api_base`` (and
    # the compiled extension underneath it) resolvable for the checkout's code.
    cudnn = importlib.import_module("cudnn")

    package = types.ModuleType(_DSA_PACKAGE)
    package.__path__ = [package_dir]
    package.__package__ = _DSA_PACKAGE
    sys.modules[_DSA_PACKAGE] = package
    cudnn.deepseek_sparse_attention = package


def load_interface():
    """Import and return the upstream SM100 host wrapper module."""
    _bind_checkout()
    return importlib.import_module(_INTERFACE_MODULE)


def load_kernel_module():
    """Import and return the upstream kernel module."""
    _bind_checkout()
    return importlib.import_module(_KERNEL_MODULE)


def flash_attn_bwd_sm100():
    """Return the upstream ``flash_attn_bwd_sm100`` entry point."""
    return load_interface().flash_attn_bwd_sm100
