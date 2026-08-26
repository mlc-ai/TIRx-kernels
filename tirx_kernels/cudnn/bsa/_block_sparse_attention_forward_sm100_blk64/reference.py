# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Lazy loader for the upstream SM100 blk64 BSA forward pass.

Upstream source: ``python/cudnn/block_sparse_attention/_interface.py``.
"""

import importlib
import os
import sys
import types
from pathlib import Path

import torch

_PACKAGE = "cudnn.block_sparse_attention"
_INTERFACE = f"{_PACKAGE}._interface"


def checkout_root():
    configured = os.environ.get("CUDNN_FRONTEND_PATH")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[4] / ".reference-deps" / "cudnn-frontend"


def _bind_checkout():
    package_dir = checkout_root() / "python" / "cudnn" / "block_sparse_attention"
    if not package_dir.is_dir():
        raise RuntimeError(
            f"cuDNN Frontend checkout does not contain block sparse attention: {package_dir}"
        )

    existing = sys.modules.get(_PACKAGE)
    if existing is not None:
        bound = getattr(existing, "__path__", None)
        if bound and Path(next(iter(bound))).resolve() == package_dir.resolve():
            return
        raise RuntimeError(
            f"{_PACKAGE} was already imported from {bound}; cannot bind it to {package_dir}"
        )

    cudnn = importlib.import_module("cudnn")
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(package_dir)]
    package.__package__ = _PACKAGE
    sys.modules[_PACKAGE] = package
    cudnn.block_sparse_attention = package


def load_interface():
    _bind_checkout()
    return importlib.import_module(_INTERFACE)


def compile_reference(data):
    """Compile lazily and return the no-argument upstream launch closure."""
    interface = load_interface()
    config = data["config"]
    inputs = data["inputs"]
    source_batches = None
    if config["batch"] > 1:
        source_batches = tuple(
            (
                inputs["q_user"][batch_idx : batch_idx + 1].clone(),
                inputs["k_user"][batch_idx : batch_idx + 1].clone(),
                inputs["v_user"][batch_idx : batch_idx + 1].clone(),
                inputs["block_index"][batch_idx : batch_idx + 1].clone(),
                inputs["block_nums"][batch_idx : batch_idx + 1].clone(),
            )
            for batch_idx in range(config["batch"])
        )

    def call_source(q, k, v, block_index, block_nums):
        return interface.bsa_attn_fwd_blk64_cutedsl(
            q,
            k,
            v,
            block_index,
            inputs["block_sizes"] if config["has_block_sizes"] else None,
            q2k_block_nums=(block_nums if config["block_count_mode"] != "fixed" else None),
            softmax_scale=inputs["softmax_scale"],
            layout=config["tensor_layout"],
            block_sparse_num=(0 if config["block_count_mode"] != "fixed" else config["kv_blocks"]),
            allow_empty_block_nums=config["block_count_mode"] == "variable_empty",
            use_clc=config["use_clc"],
            kv_splits=config["kv_splits"],
        )

    def launch():
        # The pinned source's static scheduler launches only batch 0 when B>1.
        # Preserve the source device program and cover the full operation by
        # invoking that same specialization once per batch in the reference.
        if config["batch"] == 1:
            out, lse = call_source(
                inputs["q_user"],
                inputs["k_user"],
                inputs["v_user"],
                inputs["block_index"],
                inputs["block_nums"],
            )
        else:
            outputs = []
            lses = []
            for q, k, v, block_index, block_nums in source_batches:
                out_batch, lse_batch = call_source(q, k, v, block_index, block_nums)
                outputs.append(out_batch)
                lses.append(lse_batch)
            out = torch.cat(outputs, dim=0)
            lse = torch.cat(lses, dim=0)
        if config["tensor_layout"] == "bshd":
            out = out.transpose(1, 2).contiguous()
        data["source"]["out"] = out
        data["source"]["lse"] = lse

    return launch
