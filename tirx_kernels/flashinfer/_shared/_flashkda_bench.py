# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MoonshotAI/FlashKDA benchmark adapter for the recurrent-KDA port."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch


@dataclass
class FlashKDARawReference:
    launch: Callable[[], None]
    correctness: dict[str, float]


_MAX_REL_L2 = 0.03


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Normalized RMS error ratio RMS(actual-reference)/(RMS(reference)+1e-8)."""

    a, r = actual.float(), reference.float()
    return float(
        (torch.mean((a - r) ** 2).sqrt() / (torch.mean(r**2).sqrt() + 1e-8)).item()
    )


def _fla_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """FLA's Triton chunk_kda: the authoritative bf16 oracle for this operator.

    The raw FlashKDA peer is validated against this rather than against the
    candidate it is timed beside, so a candidate is never the yardstick for its
    own baseline, and the bound matches the KDA task contract.
    """

    import os

    os.environ["FLA_FLASH_KDA"] = "0"
    os.environ["FLA_TILELANG"] = "0"
    from fla.ops.kda import chunk_kda

    cfg = case["config"]
    return chunk_kda(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        scale=case["scale"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        safe_gate=True,
        lower_bound=cfg.lower_bound,
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        initial_state=case["initial_state"],
        output_final_state=cfg.store_final_state,
        cu_seqlens=case["cu_seqlens"] if cfg.packed else None,
    )


def _load_flash_kda_peer() -> Any:
    try:
        return import_module("flash_kda")
    except ImportError as error:
        raise RuntimeError("MoonshotAI/FlashKDA is not installed") from error


def prepare_flashkda_raw_reference(case: dict[str, Any]) -> FlashKDARawReference:
    """Prepare and validate the installed raw FlashKDA peer."""
    flash_kda = _load_flash_kda_peer()
    cfg = case["config"]
    batch = 1 if cfg.packed else cfg.num_seqs
    seq_len = cfg.total_tokens if cfg.packed else cfg.seq_lens[0]

    def reshaped(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, seq_len, cfg.num_heads, -1)

    peer_out = torch.empty_like(reshaped(case["out"]))
    peer_initial_state = case["initial_state"].clone() if cfg.use_initial_state else None
    peer_final_state = torch.empty_like(case["final_state"]) if cfg.store_final_state else None
    workspace_bytes = flash_kda.get_workspace_size(cfg.total_tokens, cfg.num_heads, cfg.num_seqs)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=case["q"].device)

    def launch() -> None:
        flash_kda._fwd_raw(
            reshaped(case["q"]),
            reshaped(case["k"]),
            reshaped(case["v"]),
            reshaped(case["g"]),
            case["beta"].reshape(batch, seq_len, cfg.num_heads),
            case["scale"],
            peer_out,
            workspace,
            case["A_log"],
            case["dt_bias"],
            cfg.lower_bound,
            initial_state=peer_initial_state,
            final_state=peer_final_state,
            cu_seqlens=case["cu_seqlens"] if cfg.packed else None,
        )

    launch()
    torch.cuda.synchronize()
    reference_out, reference_state = _fla_reference(case)
    actual_out = peer_out.reshape_as(reference_out)
    out_rel_l2 = _rel_l2(actual_out, reference_out)
    if out_rel_l2 >= _MAX_REL_L2:
        raise AssertionError(
            f"FlashKDA raw output relative L2 {out_rel_l2:.4g} >= {_MAX_REL_L2}"
        )

    correctness = {"output_rel_l2": out_rel_l2}
    if cfg.store_final_state:
        if peer_final_state is None:
            raise AssertionError("FlashKDA raw final-state buffer was not prepared")
        state_rel_l2 = _rel_l2(peer_final_state, reference_state)
        if state_rel_l2 >= _MAX_REL_L2:
            raise AssertionError(
                f"FlashKDA raw final-state relative L2 {state_rel_l2:.4g} >= {_MAX_REL_L2}"
            )
        correctness["final_state_rel_l2"] = state_rel_l2

    return FlashKDARawReference(launch=launch, correctness=correctness)
