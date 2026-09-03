# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Host replica of DeepGEMM's SM100 paged-MQA scheduling metadata."""

from __future__ import annotations

import bisect


def make_schedule_metadata(context_lens, num_sms: int, indices=None, *, split_kv: int = 256):
    """Build per-SM ``(q_token_start, kv_split_start)`` boundaries.

    This is a direct host transcription of
    ``scheduler/sm100_paged_mqa_logits.cuh::sm100_paged_mqa_logits_metadata``.
    It is used on Thor because the pinned DeepGEMM host API rejects any
    architecture whose major version is not exactly 10 before its metadata
    kernel can run.
    """
    import torch

    if num_sms <= 0 or split_kv <= 0:
        raise ValueError("num_sms and split_kv must be positive")
    lens = context_lens.detach().to(device="cpu", dtype=torch.int64)
    if lens.ndim != 2:
        raise ValueError(f"context_lens must be 2-D, got {tuple(lens.shape)}")

    next_n = int(lens.shape[1])
    total_q = int(lens.numel())
    request_starts: list[int] = []
    request_tokens: list[int] = []
    request_contexts: list[int] = []
    if indices is None:
        for request in range(int(lens.shape[0])):
            request_starts.append(request * next_n)
            request_tokens.append(next_n)
            request_contexts.append(int(lens[request, -1]))
    else:
        index_values = indices.detach().to(device="cpu", dtype=torch.int64).tolist()
        flat_lens = lens.reshape(-1)
        if len(index_values) != total_q:
            raise ValueError(f"indices has {len(index_values)} entries for {total_q} Q tokens")
        token = 0
        while token < total_q:
            start = token
            request_id = index_values[token]
            while token < total_q and index_values[token] == request_id:
                token += 1
            request_starts.append(start)
            request_tokens.append(token - start)
            request_contexts.append(int(flat_lens[token - 1]))

    prefix: list[int] = []
    total_work = 0
    for tokens, context in zip(request_tokens, request_contexts):
        total_work += ((context + split_kv - 1) // split_kv) * tokens
        prefix.append(total_work)

    quotient, remainder = divmod(total_work, num_sms)
    metadata: list[tuple[int, int]] = []
    for sm_idx in range(num_sms + 1):
        work = sm_idx * quotient + min(sm_idx, remainder)
        request = bisect.bisect_right(prefix, work)
        if request == len(prefix):
            metadata.append((total_q, 0))
            continue
        work_before = 0 if request == 0 else prefix[request - 1]
        metadata.append((request_starts[request], (work - work_before) // request_tokens[request]))
    return torch.tensor(metadata, device=context_lens.device, dtype=torch.int32)


__all__ = ["make_schedule_metadata"]
