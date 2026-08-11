# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025
# DeepSeek, licensed under the MIT License. The upstream sources carry no
# per-file license header; see licenses/LICENSE.flashmla.txt for the full
# license text.
#
# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# Shared tcgen05 gemm_async configuration for the TIRx ports of FlashMLA's
# sparse phase-1 kernels.
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

"""Shared tcgen05 gemm_async config for the sparse FlashMLA phase1 kernels."""

from __future__ import annotations

from typing import Any


def tcgen05_config(
    *, accum: Any, cta_group: int, smem_desc: Any = None, weight_stationary: bool = False
) -> dict[str, Any]:
    """Return the FlashMLA-standard ``Tx.gemm_async`` tcgen05 config dict.

    - ``accum``: per-site ``uint32`` accumulate flag (0 clears D, 1 adds).
    - ``cta_group``: 1 (head64) or 2 (head128 CTA-pair kernels).
    - ``smem_desc``: SMEM descriptor hoisting mode (``"hoist"`` /
      ``"local_hoist"``).
    - ``weight_stationary``: emit the ``tcgen05.mma.ws`` form (head64's
      folded M=64 layout).

    ``None`` / ``False`` means "leave the knob out of the config" so dispatch
    defaults apply; a misspelled keyword fails loudly instead of passing
    through.
    """
    cfg: dict[str, Any] = {"accum": accum, "dispatch": "tcgen05", "cta_group": cta_group}
    if smem_desc is not None:
        cfg["smem_desc"] = smem_desc
    if weight_stationary:
        cfg["weight_stationary"] = True
    return cfg
