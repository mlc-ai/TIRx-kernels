# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Compatibility name for the single tirx-lite-DSL quantization helper module.

The old parser helper and the tirx-lite helper were instruction-for-instruction
duplicates. Keep one implementation so a helper cannot silently diverge by
being imported under the historical name.
"""

from .fp_quant_tirx_lite import *  # noqa: F403
