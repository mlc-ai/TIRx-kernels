# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Small Torch oracle for GDN decode kernels.

The state is stored as ``[pool, value_head, V, K]``, matching the TIRx ABI.
Only the recurrent time dimension is evaluated serially; batch and heads stay
vectorized so correctness cases remain inexpensive.
"""

from __future__ import annotations

import torch


def _map_qk_heads(tensor: torch.Tensor, num_value_heads: int) -> torch.Tensor:
    num_heads = tensor.shape[2]
    if num_heads < num_value_heads:
        return tensor.repeat_interleave(num_value_heads // num_heads, dim=2)
    if num_heads > num_value_heads:
        return tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            num_value_heads,
            num_heads // num_value_heads,
            tensor.shape[3],
        ).mean(dim=3)
    return tensor


@torch.inference_mode()
def gated_delta_rule_decode(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    intermediate_states: torch.Tensor | None = None,
    accepted_steps: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    disable_output: bool = False,
    recovery_steps: int = 0,
    fused_accepted_steps: bool = False,
    negative_read_write_are_padding: bool = False,
) -> torch.Tensor:
    batch, seq_len, _, _ = q.shape
    num_value_heads = v.shape[2]
    q_f = _map_qk_heads(q.float(), num_value_heads)
    k_f = _map_qk_heads(k.float(), num_value_heads)
    v_f = v.float()
    if use_qk_l2norm:
        q_f = torch.nn.functional.normalize(q_f, p=2.0, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, p=2.0, dim=-1)
    q_f = q_f * scale

    softplus = torch.nn.functional.softplus(a.float() + dt_bias.float())
    decay = torch.exp(-torch.exp(A_log.float()) * softplus)
    beta = torch.sigmoid(b.float())

    valid_reads = read_indices >= 0
    valid_writes = write_indices >= 0
    read_slots = read_indices.clamp_min(0).long()
    write_slots = write_indices.clamp_min(0).long()
    current = state_pool.index_select(0, read_slots).float()

    for row in range(batch):
        if negative_read_write_are_padding and not bool(valid_reads[row]):
            continue
        if fused_accepted_steps and accepted_steps is not None:
            phase_a = int(accepted_steps[row].item()) + 1
            phase_b_end = seq_len
        else:
            phase_a = recovery_steps
            phase_b_end = (
                int(accepted_steps[row].item()) + 1 if accepted_steps is not None else seq_len
            )

        for token in range(phase_b_end):
            state = current[row]
            state = state * decay[row, token, :, None, None]
            residual = v_f[row, token] - torch.einsum("hvk,hk->hv", state, k_f[row, token])
            residual = residual * beta[row, token, :, None]
            state = state + residual.unsqueeze(-1) * k_f[row, token].unsqueeze(-2)
            current[row] = state

            if token < phase_a:
                if (
                    token + 1 == phase_a
                    and not disable_state_update
                    and (not negative_read_write_are_padding or bool(valid_writes[row]))
                ):
                    state_pool[write_slots[row]].copy_(current[row].to(state_pool.dtype))
                continue
            if not disable_output:
                output[row, token].copy_(
                    torch.einsum("hvk,hk->hv", current[row], q_f[row, token]).to(output.dtype)
                )
            if intermediate_states is not None:
                intermediate_states[row, token].copy_(current[row].to(intermediate_states.dtype))
            if ssm_state_indices is not None:
                slot = int(ssm_state_indices[row, token].item())
                state_pool[slot].copy_(current[row].to(state_pool.dtype))

        if (
            not disable_state_update
            and recovery_steps == 0
            and not fused_accepted_steps
            and not (ssm_state_indices is not None and torch.equal(read_indices, write_indices))
            and (not negative_read_write_are_padding or bool(valid_writes[row]))
        ):
            state_pool[write_slots[row]].copy_(current[row].to(state_pool.dtype))

    return output


@torch.inference_mode()
def gated_delta_rule_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged prefill recurrence with state stored as ``[head, V, K]``."""

    num_sequences = cu_seqlens.numel() - 1
    num_state_heads = alpha.shape[1]

    def map_heads(tensor: torch.Tensor) -> torch.Tensor:
        heads = tensor.shape[1]
        if heads == num_state_heads:
            return tensor.float()
        if heads < num_state_heads:
            return tensor.float().repeat_interleave(num_state_heads // heads, dim=1)
        return (
            tensor.float()
            .reshape(tensor.shape[0], num_state_heads, heads // num_state_heads, tensor.shape[2])
            .mean(dim=2)
        )

    q_f = map_heads(q) * scale
    k_f = map_heads(k)
    v_f = map_heads(v)
    for sequence in range(num_sequences):
        begin = int(cu_seqlens[sequence].item())
        end = int(cu_seqlens[sequence + 1].item())
        state_slot = int(state_indices[sequence].item()) if state_indices is not None else sequence
        if initial_state is None:
            state = torch.zeros(
                (num_state_heads, v.shape[-1], k.shape[-1]), dtype=torch.float32, device=q.device
            )
            state_dtype = torch.float32
        else:
            state = initial_state[state_slot].float()
            state_dtype = initial_state.dtype

        for token in range(begin, end):
            state = state * alpha[token].float()[:, None, None]
            residual = v_f[token] - torch.einsum("hvk,hk->hv", state, k_f[token])
            residual = residual * beta[token].float()[:, None]
            state = state + residual.unsqueeze(-1) * k_f[token].unsqueeze(-2)
            output[token].copy_(torch.einsum("hvk,hk->hv", state, q_f[token]).to(output.dtype))
        if final_state is not None:
            final_state[state_slot].copy_(state.to(final_state.dtype))
    return output


__all__ = ["gated_delta_rule_decode", "gated_delta_rule_prefill"]
