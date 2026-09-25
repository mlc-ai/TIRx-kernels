# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Shared TIRx port of DeepGEMM's `sm100_fp8_fp4_gemm_1d1d` implementation.

Submodules: `spec` (descriptors, heuristics, launch), `data` (test/bench data
preparation and references), `kernel` (the `PrimFunc` body).  The five public
kernel modules under :mod:`tirx_kernels.deepgemm` pin one descriptor each and
import everything else from here; `spec`'s public surface is re-exported at the
package root.
"""

from .spec import *
from .spec import __all__ as __all__
