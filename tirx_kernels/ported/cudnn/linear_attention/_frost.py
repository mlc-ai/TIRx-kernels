# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 1dbd2c09432dfd6cbf13a0ad85a071446398c082), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Host-side glue shared by the FROST linear-attention reference adapters.

cuDNN Frontend 1.29 widened the FROST work-item row from 8 to 10 int32 fields
(``common/split_k.py``: ``final_dst`` and ``dstate_dst`` appended) and made the
launch device / SM count explicit kernel-entry arguments. The TIRx ports keep
their 8-field tables, so only the reference side is widened here.
"""

from __future__ import annotations

WORK_ITEM_FIELDS = 10


def source_work_items(torch, base, b_t):
    """Widen 8-field ``[b, h, wstart, wend, cstart, cend, tok_begin, tok_end]``
    rows to the 1.29 layout: ``final_dst`` is the sequence whose final state the
    item stores (it writes the last chunk) and ``dstate_dst`` the sequence whose
    ``d_initial_state`` it stores (it writes chunk 0); ``-1`` stores nothing."""
    batch = base[:, 0]
    chunks = (base[:, 7] - base[:, 6] + b_t - 1) // b_t
    none = torch.full_like(batch, -1)
    final_dst = torch.where(base[:, 3] == chunks, batch, none)
    dstate_dst = torch.where(base[:, 2] == 0, batch, none)
    return torch.cat([base, final_dst[:, None], dstate_dst[:, None]], dim=1).contiguous()


def launch_device(torch):
    """``(device, num_sm)`` for the reference entry points: the ambient CUDA
    ordinal and its SM count, matching what the 1.27 kernels resolved internally."""
    device = torch.cuda.current_device()
    return device, torch.cuda.get_device_properties(device).multi_processor_count
