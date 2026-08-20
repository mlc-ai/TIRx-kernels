# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Import and launch helpers for the MSA sparse-attention reference.

MSA ships as the ``fmha_sm100`` package, but its CuTeDSL sources import their
siblings as ``src.*`` (``from src.common import ...``), so ``python/`` and
``python/fmha_sm100/cute`` both have to be importable -- the same two entries
MSA's own benchmarks add (``benchmarks/bench_sparse_attention_ops.py``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MSA_MODULE = None

# Where the checkout may live, in priority order: an explicit override, the
# path `scripts/install_reference_dependencies.py` clones into, and the
# workspace's own kernel-libs checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANDIDATE_ROOTS = (_REPO_ROOT / ".reference-deps" / "msa", Path.home() / "kernel-libs" / "msa")


def msa_root() -> Path:
    """Return the MSA checkout root, or raise with the paths that were tried."""
    override = os.environ.get("MSA_PATH")
    candidates = (Path(override),) if override else _CANDIDATE_ROOTS
    for root in candidates:
        if (root / "python" / "fmha_sm100").is_dir():
            return root
    tried = ", ".join(str(path) for path in candidates)
    raise ImportError(f"no MSA checkout found (set MSA_PATH); tried: {tried}")


def prepare_scheduler_module():
    """Import ``src.sm100.prepare_scheduler`` from the MSA checkout, once."""
    global _MSA_MODULE
    if _MSA_MODULE is not None:
        return _MSA_MODULE

    root = msa_root()
    for entry in (str(root / "python" / "fmha_sm100" / "cute"), str(root / "python")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import src.sm100.prepare_scheduler as prepare_scheduler

    _MSA_MODULE = prepare_scheduler
    return _MSA_MODULE


def compiled_fwd_split_atomic(case: dict):
    """Compile (or fetch from MSA's AOT cache) the forward split-slot kernel.

    Returns the compiled callable behind ``prepare_sparse_fwd_schedule_and_split``'s
    NVTX range, so the timed closure covers the kernel launch alone -- not the
    host wrapper's shape validation, its ``split_counts.zero_()``, or the
    flat-schedule kernel it runs first.
    """
    module = prepare_scheduler_module()
    return module._get_sparse_prepare_fwd_split_atomic(
        case["k2q_row_ptr"],
        case["k2q_q_indices"],
        case["scheduler_metadata"],
        case["work_count"],
        case["k2q_qsplit_indices"],
        case["split_counts"],
        case["cu_seqlens_q"],
        case["work_capacity"],
        case["max_seqlen_q"],
        case["topk"],
    )


def compiled_flat_schedule(case: dict):
    """Compile (or fetch from MSA's AOT cache) the flat-schedule kernel.

    Returns the compiled callable behind ``prepare_sparse_flat_schedule``'s NVTX
    range, so the timed closure covers the kernel launch alone rather than the
    host wrapper's allocation and schedule sizing.
    """
    module = prepare_scheduler_module()
    return module._get_sparse_prepare_flat_schedule(
        case["k2q_row_ptr"],
        case["cu_seqlens_k"],
        case["scheduler_metadata"],
        case["work_count"],
        case["target_q_per_cta"],
        case["work_capacity"],
        case["head_kv"],
        case["blk_kv"],
    )
