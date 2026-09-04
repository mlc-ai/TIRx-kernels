# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Numerical acceptance must reject nonfinite values at valid output positions."""

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("name", ["mqa_logits_fp4", "mqa_logits_fp8", "paged_mqa_logits_fp4"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_mqa_validator_rejects_nonfinite_valid_logits(name, bad_value):
    module = importlib.import_module(f"tirx_kernels.deepgemm.{name}")
    reference = torch.ones((2, 3))
    data = {
        "config": SimpleNamespace(compressed_logits=False, seq_len=2, seq_len_kv=3),
        "reference": reference,
    }
    assert module._assert_correct(data, reference.clone(), name="valid") == 0.0
    observed = reference.clone()
    observed[0, 1] = bad_value
    with pytest.raises(AssertionError):
        module._assert_correct(data, observed, name="corrupted valid logit")

    # The source contract intentionally excludes masked attention locations.
    reference[0, 1] = float("-inf")
    assert module._assert_correct(data, observed, name="masked logit") == 0.0


@pytest.mark.parametrize("bad_output", ["d", "sqr"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_hc_validator_rejects_nonfinite_in_either_output(bad_output, bad_value):
    module = importlib.import_module("tirx_kernels.deepgemm.tf32_hc_prenorm_gemm")
    data = {
        "config": SimpleNamespace(num_splits=1),
        "reference_d": torch.ones((2, 3)),
        "reference_sqr": torch.ones(2),
    }
    case = SimpleNamespace(**data)
    d, sqr = data["reference_d"].clone(), data["reference_sqr"].clone()
    assert module._assert_correct(data, d, sqr, name="valid") == 0.0
    assert module._assert_correct_case(case, d, sqr, name="valid") == 0.0
    (d if bad_output == "d" else sqr).reshape(-1)[0] = bad_value
    with pytest.raises(AssertionError):
        module._assert_correct(data, d, sqr, name="corrupted output")
    with pytest.raises(AssertionError):
        module._assert_correct_case(case, d, sqr, name="corrupted output")
