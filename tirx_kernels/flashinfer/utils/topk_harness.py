# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Host-side harness helpers shared by the FlashInfer radix top-k ports.

Reference launching, output allocation and result comparison are identical for
the single-CTA and multi-CTA specializations of ``RadixTopKKernel_Unified``:
both are reached through the same raw ``topk`` FFI module, write the same
output tensors, and share the same tie-ambiguity rules.  Only the dispatch
domain and the workspace differ, so those stay in the kernel modules.

Torch is imported lazily inside each function so importing a kernel module
never pulls in a CUDA context.
"""

from __future__ import annotations

import os
from typing import Any

SOURCE_ALGO_ENV = "FLASHINFER_TOPK_ALGO"
SOURCE_ALGO_VALUE = "multi_cta"

WORKSPACE_BYTES = 1024 * 1024


def torch_dtype(dtype: str):
    import torch

    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def source_module():
    """Raw FlashInfer topk FFI module (no allocating Python wrapper)."""
    from flashinfer.jit.topk import gen_topk_module

    return gen_topk_module().build_and_load()


def pin_source_algo() -> None:
    """Force dispatch onto the radix family this port targets.

    ``GetTopKAlgoOverride`` (``topk.cuh``) maps this value to
    ``TopKAlgoOverride::MULTI_CTA``, which makes ``ShouldUseFilteredTopK``
    return false so FilteredTopK is never selected.  Which radix
    specialization then runs is decided by ``ctas_per_group`` inside the
    launcher, i.e. by the shape.
    """
    os.environ[SOURCE_ALGO_ENV] = SOURCE_ALGO_VALUE


def row_states_buffer(nbytes: int = WORKSPACE_BYTES):
    """Zeroed cross-CTA workspace, mirroring the cached buffer in ``topk.py``."""
    import torch

    return torch.zeros(nbytes, dtype=torch.uint8, device="cuda")


def alloc_outputs(config: dict[str, Any], nbytes: int = WORKSPACE_BYTES):
    import torch

    rows = config["num_rows"]
    k = config["k"]
    return {
        "indices": torch.empty(rows, k, dtype=torch.int32, device="cuda"),
        "values": torch.empty(rows, k, dtype=torch_dtype(config["dtype"]), device="cuda"),
        "row_states": row_states_buffer(nbytes),
    }


def assert_device_matches_compile_profile(expected: int, domain: str) -> None:
    """Reject a device whose shared-memory budget changes the dispatch domain."""
    import torch

    optin = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).shared_memory_per_block_optin
    if optin != expected:
        raise AssertionError(
            f"device optin shared memory {optin} != compile profile {expected}; "
            f"the {domain} dispatch domain would differ from the configured matrix"
        )


def row_local_indices(cfg: dict[str, Any], data: dict[str, Any], out_indices):
    """Map one launch's raw output back to row-local score offsets.

    Basic returns the index directly; Ragged adds a per-row offset; PageTable
    returns an injective page id (``batch * length + slot``).  ``-1`` marks the
    padding the trivial branches write when a row is shorter than ``k``.
    """
    import torch

    rows = cfg["num_rows"]
    length = cfg["length"]
    idx = out_indices.to(torch.int64)
    pad = idx < 0
    if cfg["mode"] == "ragged":
        idx = idx - data["offsets"].to(torch.int64).unsqueeze(-1)
    elif cfg["mode"] == "page_table":
        row_to_batch = data.get("row_to_batch")
        if row_to_batch is None:
            batch = torch.arange(rows, dtype=torch.int64, device=idx.device)
        else:
            batch = row_to_batch.to(torch.int64)
        idx = idx - batch.unsqueeze(-1) * length
        pt_starts = data.get("page_table_row_starts")
        if pt_starts is None:
            pt_starts = data.get("row_starts")
        if pt_starts is not None:
            idx = idx - pt_starts.to(torch.int64).unsqueeze(-1)
    return idx, pad


def selected_values(cfg: dict[str, Any], data: dict[str, Any], out_indices):
    """Sorted-descending scores of the selected elements, padding removed."""
    import torch

    idx, pad = row_local_indices(cfg, data, out_indices)
    row_starts = data.get("row_starts")
    if row_starts is not None:
        idx = idx + row_starts.to(torch.int64).unsqueeze(-1)
    scores = data["scores"].to(torch.float32)
    safe = idx.clamp_(0, scores.size(1) - 1)
    vals = torch.gather(scores, 1, safe)
    # Padding slots carry no score; sort them to the bottom deterministically.
    vals = torch.where(pad, torch.full_like(vals, float("-inf")), vals)
    return torch.sort(vals, dim=-1, descending=True).values


def compare_outputs(
    cfg: dict[str, Any], data: dict[str, Any], ref: dict[str, Any], got: dict[str, Any]
) -> None:
    """Compare a TIRx launch against the FlashInfer launch.

    Ties make the selected *index set* ambiguous, so the criterion is the
    multiset of selected score values, which every valid top-k shares.  For
    deterministic configs both sides run the same fixed-order collect, so the
    raw outputs must additionally match element for element.
    """
    import torch

    if cfg["deterministic"]:
        torch.testing.assert_close(got["indices"], ref["indices"], rtol=0, atol=0)
        if cfg["mode"] == "basic":
            torch.testing.assert_close(got["values"], ref["values"], rtol=0, atol=0)
        return

    ref_vals = selected_values(cfg, data, ref["indices"])
    got_vals = selected_values(cfg, data, got["indices"])
    torch.testing.assert_close(got_vals, ref_vals, rtol=0, atol=0)
    if cfg["mode"] == "basic":
        ref_out = torch.sort(ref["values"].to(torch.float32), dim=-1, descending=True).values
        got_out = torch.sort(got["values"].to(torch.float32), dim=-1, descending=True).values
        torch.testing.assert_close(got_out, ref_out, rtol=0, atol=0)
        # The emitted values must be the scores at the emitted indices.
        torch.testing.assert_close(got_out, got_vals, rtol=0, atol=0)
