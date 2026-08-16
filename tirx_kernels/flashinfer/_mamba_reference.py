# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""In-tree Torch oracle for selective state update."""

from __future__ import annotations

import torch


def _index(indices: torch.Tensor | None, row: int, token: int, default: int) -> int:
    if indices is None:
        return default
    if indices.ndim == 1:
        return int(indices[row].item())
    return int(indices[row, token].item())


def _store_state(
    destination: torch.Tensor, slot: int, value: torch.Tensor, scales: torch.Tensor | None
) -> None:
    if destination.dtype == torch.int16:
        amax = value.abs().amax(dim=-1)
        encode = torch.where(amax == 0, torch.ones_like(amax), 32767.0 / amax)
        destination[slot].copy_(
            (value * encode[..., None]).round().clamp(-32767, 32767).to(torch.int16)
        )
        if scales is None:
            raise ValueError("int16 state storage requires decode scales")
        scales[slot].copy_(1.0 / encode)
    else:
        destination[slot].copy_(value.to(destination.dtype))


@torch.inference_mode()
def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    *,
    z: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    state_batch_indices: torch.Tensor | None,
    dst_state_batch_indices: torch.Tensor | None,
    pad_slot_id: int,
    state_scale: torch.Tensor | None,
    out: torch.Tensor | None,
    disable_state_update: bool,
    intermediate_states_buffer: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    intermediate_state_scales: torch.Tensor | None = None,
    cache_steps: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    **_unused,
) -> torch.Tensor:
    """Evaluate fixed- or variable-length selective state update in FP32."""

    del cache_steps
    original_x_ndim = x.ndim
    if out is None:
        out = torch.empty_like(x)
    fixed = cu_seqlens is None
    if fixed:
        if x.ndim == 3:
            x = x.unsqueeze(1)
            dt = dt.unsqueeze(1)
            B = B.unsqueeze(1)
            C = C.unsqueeze(1)
            if z is not None:
                z = z.unsqueeze(1)
            out_view = out.unsqueeze(1)
        else:
            out_view = out
        batch, tokens = x.shape[:2]
        bounds = [(row, 0, tokens) for row in range(batch)]
    else:
        batch = cu_seqlens.numel() - 1
        bounds = [
            (row, int(cu_seqlens[row].item()), int(cu_seqlens[row + 1].item()))
            for row in range(batch)
        ]
        out_view = out

    nheads = state.shape[1]
    groups = B.shape[-2]
    heads_per_group = nheads // groups
    A_f = A.float()
    D_f = None if D is None else D.float()
    bias_f = None if dt_bias is None else dt_bias.float()

    for row, begin, end in bounds:
        length = end - begin
        if length == 0:
            continue
        accepted = int(num_accepted_tokens[row].item()) if num_accepted_tokens is not None else 1
        initial_token = max(accepted - 1, 0)
        source_slot = _index(state_batch_indices, row, initial_token, row)
        is_pad = source_slot == pad_slot_id
        if is_pad:
            current = torch.zeros_like(state[0], dtype=torch.float32)
        else:
            current = state[source_slot].float()
            if state_scale is not None:
                current = current * state_scale[source_slot].float().unsqueeze(-1)

        for local_token in range(length):
            token = begin + local_token
            if fixed:
                x_t = x[row, local_token].float()
                dt_t = dt[row, local_token].float()
                b_t = B[row, local_token].float()
                c_t = C[row, local_token].float()
                z_t = None if z is None else z[row, local_token].float()
                output_t = out_view[row, local_token]
            else:
                x_t = x[token].float()
                dt_t = dt[token].float()
                b_t = B[token].float()
                c_t = C[token].float()
                z_t = None if z is None else z[token].float()
                output_t = out_view[token]

            if bias_f is not None:
                dt_t = dt_t + bias_f
            if dt_softplus:
                dt_t = torch.nn.functional.softplus(dt_t)
            expanded_b = b_t.repeat_interleave(heads_per_group, dim=0)
            expanded_c = c_t.repeat_interleave(heads_per_group, dim=0)
            current = current * torch.exp(dt_t.unsqueeze(-1) * A_f)
            current = current + dt_t.unsqueeze(-1) * expanded_b.unsqueeze(1) * x_t.unsqueeze(-1)
            output_value = (current * expanded_c.unsqueeze(1)).sum(dim=-1)
            if D_f is not None:
                output_value = output_value + D_f * x_t
            if z_t is not None:
                output_value = output_value * torch.nn.functional.silu(z_t)
            output_t.copy_(output_value.to(output_t.dtype))

            destination_slot = -1
            if dst_state_batch_indices is not None:
                destination_slot = _index(dst_state_batch_indices, row, local_token, -1)
                destination = state
                destination_scales = state_scale
            elif intermediate_states_buffer is not None:
                cache_row = (
                    int(intermediate_state_indices[row].item())
                    if intermediate_state_indices is not None
                    else source_slot
                )
                destination_slot = cache_row * intermediate_states_buffer.shape[1] + local_token
                destination = intermediate_states_buffer.view(
                    -1, *intermediate_states_buffer.shape[2:]
                )
                destination_scales = (
                    None
                    if intermediate_state_scales is None
                    else intermediate_state_scales.view(-1, *intermediate_state_scales.shape[2:])
                )
            elif local_token == length - 1 and not disable_state_update:
                destination_slot = source_slot
                destination = state
                destination_scales = state_scale
            if destination_slot != pad_slot_id and destination_slot >= 0:
                _store_state(destination, destination_slot, current, destination_scales)

    return out if original_x_ndim >= 3 else out.squeeze(1)


__all__ = ["selective_state_update"]
