# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Adversarial FP8 MLA cases independent of the small-amplitude benchmark draw."""

import pytest
import torch

from tirx_kernels.flashinfer.mla_dsv4.b200 import mla_dsv4_multishape as mla


def _case(heads, keys, pattern):
    mla._assert_supported_arch()
    device = torch.device("cuda")
    q = torch.zeros((3, heads, 512), device=device)
    q[..., 0] = 1
    kv = torch.zeros((keys, 512), device=device)
    kv[:, 1:] = 1
    if pattern == "large_partial":
        # Each 128-key split accumulates 512 before normalization.
        kv[:, 1:] = 4
    elif pattern == "small_weights":
        # exp(-7.5) disappears when cast directly to E4M3, but survives *448.
        kv[1:, 0] = -7.5
    elif pattern == "rising_max":
        # Force repeated online rescaling and include positive/negative values.
        kv[:, 0] = (torch.arange(keys, device=device) // 64).float() * 0.5 - 4
        kv[:, 1:] = torch.where((torch.arange(keys, device=device) % 3 == 0)[:, None], -2.0, 1.0)
    elif pattern == "rounding_boundary":
        # Cross the small FP8 max-update threshold after several tiny rises.
        kv[:, 0] = (torch.arange(keys, device=device) // 64).float() * 0.25
    else:
        raise ValueError(pattern)
    dtype = torch.float8_e4m3fn
    indices = (
        torch.cat([torch.arange(128, device=device), torch.arange(keys - 128, device=device)])
        .to(torch.int32)
        .repeat(3, 1)
    )
    # Interior holes, a ragged final block, and one completely empty query.
    indices[1, 7::19] = -1
    sinks = torch.linspace(-2, 2, heads, device=device)
    return {
        "config": {"kv_dtype": "float8_e4m3fn"},
        "query": q.to(dtype),
        "swa_kv_cache": kv[:128].to(dtype).reshape(1, 1, 128, 512),
        "compressed_kv_cache": kv[128:].to(dtype).reshape(1, 1, keys - 128, 512),
        "sparse_indices": indices,
        "sparse_topk_lens": torch.tensor([keys, keys - 3, 0], dtype=torch.int32, device=device),
        "sinks": sinks,
        "bmm1_scale": 0.01 if pattern == "rounding_boundary" else 1.0,
        "bmm2_scale": 1.75,
        "output": torch.empty_like(q, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize("heads", [8, 64, 128])
@pytest.mark.parametrize(
    "pattern,keys,splits",
    [
        ("large_partial", 640, 5),
        ("small_weights", 1152, 1),
        ("small_weights", 1152, 3),
        ("small_weights", 16384, 1),
        ("rising_max", 1152, 1),
        ("rising_max", 1152, 3),
        ("rounding_boundary", 1152, 1),
    ],
)
def test_fp8_mla_numerics(monkeypatch, heads, pattern, keys, splits):
    monkeypatch.setenv("MLA_SPLITS", str(splits))
    case = _case(heads, keys, pattern)
    launch = mla._launch_state(case)
    reference = mla._reference_output(case)
    first = None
    for poison in (float("nan"), 42.0):
        case["output"].fill_(poison)
        launch()
        torch.cuda.synchronize()
        actual = case["output"].clone()
        assert torch.isfinite(actual).all()
        # Mixed E4M3 P/BF16 partials have rounding error; the old saturation
        # and underflow failures exceed this bound by a wide margin.
        torch.testing.assert_close(actual.float(), reference.float(), atol=0.025, rtol=0.025)
        assert torch.count_nonzero(actual[2]) == 0
        if first is not None:
            torch.testing.assert_close(actual, first, atol=0, rtol=0)
        first = actual
