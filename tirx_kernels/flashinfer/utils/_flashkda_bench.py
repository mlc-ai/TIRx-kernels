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
    expected_out = case["out"]
    actual_out = peer_out.reshape_as(expected_out)
    out_max_abs = float((actual_out.float() - expected_out.float()).abs().max())
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-2, atol=1e-2)

    correctness = {"output_max_abs": out_max_abs}
    if cfg.store_final_state:
        if peer_final_state is None:
            raise AssertionError("FlashKDA raw final-state buffer was not prepared")
        state_max_abs = float((peer_final_state.float() - case["final_state"].float()).abs().max())
        torch.testing.assert_close(peer_final_state, case["final_state"], rtol=1e-2, atol=1e-2)
        correctness["final_state_max_abs"] = state_max_abs

    return FlashKDARawReference(launch=launch, correctness=correctness)
