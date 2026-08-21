# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Shared validity-mask helper for the sparse FlashMLA phase1 kernels."""

from __future__ import annotations

from typing import Any

import tirx_kernels.kern as K


def pack_valid_mask8(
    lane_indices: Any, abs_pos_start: Any, lane_idx: Any, topk_len: Any, s_kv: Any
) -> Any:
    """Pack the validity of 8 KV rows into an int8 bitmask.

    Bit ``i`` is set when row ``i`` has a valid index (``0 <= idx < s_kv``)
    and its absolute topk position is within ``topk_len``. Reduced via a
    balanced OR tree.
    """
    terms = []
    for i in range(8):
        valid = (
            (lane_indices[i] >= 0)
            & (lane_indices[i] < s_kv)
            & (abs_pos_start + lane_idx * 8 + i < topk_len)
        )
        terms.append(K.Select(valid, K.int32(1 << i), K.int32(0)))
    while len(terms) > 1:
        terms = [K.bitwise_or(terms[i], terms[i + 1]) for i in range(0, len(terms), 2)]
    return K.cast(terms[0], "int8")
