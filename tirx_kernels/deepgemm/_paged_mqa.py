# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Host-owned ABI metadata for the SM100 paged-MQA TIRx kernels."""

from __future__ import annotations

import bisect

import torch


def make_schedule_metadata(
    context_lens: torch.Tensor, *, num_sms: int, indices: torch.Tensor | None = None
) -> torch.Tensor:
    """Balance ``(q token, 256-KV split)`` work over SMs.

    The returned ``[num_sms + 1, 2]`` tensor contains inclusive range
    boundaries ``(q_token_idx, kv_split_idx)``.  This is kernel ABI, not an
    implementation-specific optimization baseline, so it lives with the port.
    """

    if context_lens.ndim != 2 or context_lens.dtype != torch.int32:
        raise ValueError("context_lens must be a contiguous 2-D int32 tensor")
    if not context_lens.is_contiguous() or num_sms <= 0:
        raise ValueError("context_lens must be contiguous and num_sms must be positive")

    num_requests, next_n = context_lens.shape
    num_q_tokens = num_requests * next_n
    lens = context_lens.reshape(-1).tolist()

    starts: list[int] = []
    token_counts: list[int] = []
    request_lens: list[int] = []
    if indices is None:
        for request in range(num_requests):
            starts.append(request * next_n)
            token_counts.append(next_n)
            request_lens.append(int(lens[(request + 1) * next_n - 1]))
    else:
        if next_n != 1 or indices.shape != (num_requests,) or indices.dtype != torch.int32:
            raise ValueError("varlen metadata requires one int32 index per q token")
        index_values = indices.tolist()
        token = 0
        while token < num_q_tokens:
            start = token
            request_id = index_values[token]
            token += 1
            while token < num_q_tokens and index_values[token] == request_id:
                token += 1
            starts.append(start)
            token_counts.append(token - start)
            request_lens.append(int(lens[token - 1]))

    prefix: list[int] = []
    total = 0
    for count, context_len in zip(token_counts, request_lens):
        total += ((context_len + 255) // 256) * count
        prefix.append(total)

    quotient, remainder = divmod(total, num_sms)
    metadata: list[tuple[int, int]] = []
    for sm_idx in range(num_sms + 1):
        work = sm_idx * quotient + min(sm_idx, remainder)
        request_idx = bisect.bisect_right(prefix, work)
        if request_idx == len(prefix):
            metadata.append((num_q_tokens, 0))
            continue
        work_before = prefix[request_idx - 1] if request_idx else 0
        split = (work - work_before) // token_counts[request_idx]
        metadata.append((starts[request_idx], split))

    return torch.tensor(metadata, dtype=torch.int32, device=context_lens.device)


__all__ = ["make_schedule_metadata"]
