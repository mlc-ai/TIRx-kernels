# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import pytest
import torch

from tirx_kernels.flashinfer.kda import recurrent_kda_decode_grouped as kda


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("bad_side", ["actual", "reference"])
def test_grouped_kda_validation_rejects_nonfinite_values(bad_value, bad_side):
    actual = torch.ones((2, 3), dtype=torch.bfloat16)
    reference = actual.clone()
    kda._assert_close(actual, reference, kda._RTOL, kda._ATOL, "finite output")
    (actual if bad_side == "actual" else reference)[0, 1] = bad_value
    with pytest.raises(AssertionError):
        kda._assert_close(actual, reference, kda._RTOL, kda._ATOL, "invalid output or state")
