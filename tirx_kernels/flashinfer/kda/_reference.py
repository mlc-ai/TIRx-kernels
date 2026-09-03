# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Independent numerical references shared by the FlashKDA decode ports."""

from __future__ import annotations

from typing import Any

import torch


def validation_tolerances(default_rtol: float, default_atol: float) -> tuple[float, float]:
    """Use a one-BF16-step floor for the architecture-independent oracle."""
    from tirx_kernels.target import prepare_cuda_arch

    if prepare_cuda_arch() == "sm_110a":
        return max(default_rtol, 2.0**-7), max(default_atol, 2.0**-7)
    return default_rtol, default_atol


def recurrent_kda_reference(
    case: dict[str, Any], *, num_tokens: int, lower_bound_gate: bool = False, l2_eps: float = 1.0e-6
) -> torch.Tensor:
    """Evaluate the gated delta-rule recurrence in structured FP32 PyTorch.

    The frozen Cake exports are architecture-specific.  This reference is used
    when validating the same TIRx kernels on an architecture for which Cake has
    no binary.  It deliberately follows the public recurrent-KDA contract
    instead of the transcribed kernel's instruction schedule.

    ``reference_state`` is updated in place, including every active speculative
    checkpoint.  The FP32 working state remains live between tokens, matching
    the kernel; only checkpoint stores are rounded to BF16.
    """

    spec = case["spec"]
    num_seqs = int(spec["NUM_SEQS"])
    num_heads = int(spec["NUM_HEADS"])
    num_value_heads = int(spec["NUM_VALUE_HEADS"])
    head_ratio = num_value_heads // num_heads

    q = case["q"].float()
    k = case["k"].float()
    v = case["v"].float()
    g = case["g"].float()
    beta = case["beta"].float()
    slots = case["ssm_state_indices"].reshape(num_seqs, num_tokens)
    state_pool = case["reference_state"]
    output = torch.zeros_like(case["tirx_out"])
    scale = float(case["scale"])

    query_heads = torch.arange(num_value_heads, device=q.device) // head_ratio
    if lower_bound_gate:
        a = case["a_log"].exp()[query_heads]
        dt_bias = case["dt_bias"].reshape(num_heads, -1)[query_heads]
        lower_bound = float(case["lower_bound"])

    for seq in range(num_seqs):
        if num_tokens == 1:
            initial_slot = seq
        else:
            accepted = int(case["num_accepted_tokens"][seq]) - 1
            accepted = min(max(accepted, 0), num_tokens - 1)
            initial_slot = int(slots[seq, accepted])
            if initial_slot < 0:
                initial_slot = 0

        # [HV, V, K].  Keep this state in FP32 across the token recurrence.
        state = state_pool[initial_slot].float().clone()
        for token in range(num_tokens):
            row = seq * num_tokens + token
            q_row = q[0, row, query_heads]
            k_row = k[0, row, query_heads]
            q_row = q_row * torch.rsqrt(q_row.square().sum(dim=-1, keepdim=True) + l2_eps)
            k_row = k_row * torch.rsqrt(k_row.square().sum(dim=-1, keepdim=True) + l2_eps)
            q_row = q_row * scale

            if lower_bound_gate:
                decay = torch.exp(lower_bound * torch.sigmoid(a[:, None] * (g[0, row] + dt_bias)))
            else:
                decay = torch.exp(g[0, row])

            state = state * decay[:, None, :]
            prediction = torch.einsum("hvk,hk->hv", state, k_row)
            delta = (v[0, row] - prediction) * beta[0, row, :, None]
            state = state + delta[:, :, None] * k_row[:, None, :]

            slot = int(slots[seq, token])
            if slot >= 0:
                output[0, row] = torch.einsum("hvk,hk->hv", state, q_row).to(output.dtype)
                state_pool[slot].copy_(state.to(state_pool.dtype))

    return output


def benchmark_tirx_with_oracle(
    executable,
    args: tuple[Any, ...],
    case: dict[str, Any],
    reference,
    *,
    rtol: float,
    atol: float,
    warmup: float | None,
    repeat: float | None,
    timer: str | None,
    rounds: int,
    cooldown_s: float,
) -> dict[str, Any]:
    """Validate once, then benchmark TIRx when no peer binary exists."""
    from tirx_kernels.runner import bench

    executable(*args)
    expected = reference(case)
    torch.cuda.synchronize(case["device"])
    torch.testing.assert_close(case["tirx_out"].float(), expected.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(
        case["tirx_state_raw"].float(), case["reference_state_raw"].float(), rtol=rtol, atol=atol
    )
    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


__all__ = ["benchmark_tirx_with_oracle", "recurrent_kda_reference", "validation_tolerances"]
