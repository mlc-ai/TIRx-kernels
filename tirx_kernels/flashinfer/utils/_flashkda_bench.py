# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MoonshotAI/FlashKDA benchmark adapter for the recurrent-KDA port."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import torch


def _load_flash_kda_peer() -> Any:
    try:
        flash_kda = import_module("flash_kda")
        import_module("flash_kda_C")
    except ImportError as error:
        raise RuntimeError("MoonshotAI/FlashKDA is not installed") from error
    return flash_kda


def prepare_flashinfer_cake_decode_reference(
    case: dict[str, Any],
    variant: str,
    *,
    direct_sm103: bool = False,
    uses_lower_bound: bool = False,
) -> Callable[[], None]:
    """Build one installed FlashInfer CAKE decode launch lazily."""
    from unittest import SkipTest

    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    device = case["device"]
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        target = "sm100f" if torch.version.cuda and torch.version.cuda >= "12.9" else "sm100a"
    elif (major, minor) == (10, 3):
        target = "sm103a" if direct_sm103 else "sm100f"
    else:
        raise SkipTest(f"FlashKDA CAKE decode has no export for capability {major}.{minor}")

    module = get_flash_kda_decode_module(variant, target)
    reference_state = case["initial_state_raw"].clone()
    reference_out = torch.empty_like(case["tirx_out"])
    dummy_f32 = torch.ones(1, device=device, dtype=torch.float32)
    a_log = case["a_log"] if uses_lower_bound else dummy_f32
    dt_bias = case["dt_bias"] if uses_lower_bound else dummy_f32
    lower_bound = float(case["lower_bound"]) if uses_lower_bound else 0.0

    def launch() -> None:
        module.run(
            case["q"],
            case["k"],
            case["v"],
            case["g"],
            case["beta"],
            a_log,
            dt_bias,
            reference_state,
            reference_out,
            case["cu_seqlens"],
            case["ssm_state_indices"],
            case["num_accepted_tokens"],
            float(case["scale"]),
            lower_bound,
            int(torch.cuda.current_stream(device).cuda_stream),
        )

    return launch


def prepare_flashinfer_cutedsl_reference(case: dict[str, Any]) -> Callable[[], None]:
    """Build the installed FlashInfer recurrent-KDA CuTeDSL launch."""
    fi = import_module("flashinfer.kda_kernels.recurrent_kda")
    spec = case["spec"]
    raw_state = case["initial_state_raw"].clone()
    state_stride = int(spec["STATE_SLOT_STRIDE"])
    num_value_heads = int(spec["NUM_VALUE_HEADS"])
    head_dim = 128
    state_slots = raw_state.numel() // state_stride
    state = raw_state.as_strided(
        (state_slots, num_value_heads, head_dim, head_dim),
        (state_stride, head_dim * head_dim, head_dim, 1),
    )
    tokens = int(spec["NUM_TOKENS"])

    def launch() -> None:
        fi.run_recurrent_kda(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            g=case["g"],
            beta=case["beta"],
            A_log=case["a_log"],
            dt_bias=case["dt_bias"],
            scale=case["scale"],
            initial_state=state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=case["lower_bound"],
            cu_seqlens=case["cu_seqlens"],
            ssm_state_indices=case["ssm_state_indices"],
            num_spec_tokens=(tokens - 1) if tokens > 1 else None,
        )

    return launch


def prepare_flashkda_raw_reference(case: dict[str, Any]) -> Callable[[], None]:
    """Prepare the installed raw FlashKDA peer and return its launch."""
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

    return launch
