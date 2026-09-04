# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TF32 input semantics must distinguish rounding ties from oracle tolerance."""

import pytest
import torch

from tirx_kernels.deepgemm import tf32_hc_prenorm_gemm as hc


@pytest.mark.parametrize("sign", [0, 0x80000000])
def test_tf32_rounding_cells_and_special_values(sign):
    # Literal IEEE bit patterns cover even/odd ties, both sides of a tie,
    # subnormals, overflow, signed zero, infinity, and NaN payload retention.
    before = [
        0,
        0xFFF,
        0x1000,
        0x1001,
        0x3000,
        0x3F800FFF,
        0x3F801000,
        0x3F801001,
        0x3F803000,
        0x7F7FFFFF,
        0x7F800000,
        0x7FC01234,
    ]
    after = [
        0,
        0,
        0,
        0x2000,
        0x4000,
        0x3F800000,
        0x3F800000,
        0x3F802000,
        0x3F804000,
        0x7F800000,
        0x7F800000,
        0x7FC01234,
    ]
    original = torch.tensor([x | sign for x in before], dtype=torch.uint32)
    expected = torch.tensor([x | sign for x in after], dtype=torch.uint32)
    actual = hc._round_to_tf32_rne(original.view(torch.float32))
    assert torch.equal(actual.view(torch.uint32), expected)
    assert torch.equal(original, torch.tensor([x | sign for x in before], dtype=torch.uint32))


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", "sm_110a"])
@pytest.mark.parametrize("allow_tf32", [False, True])
def test_oracle_operand_precision_and_backend_state(monkeypatch, arch, allow_tf32):
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", arch)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        a = torch.eye(2, dtype=torch.bfloat16)
        b = torch.tensor([[1.00048828125, 1.00146484375]], dtype=torch.float32)
        expected = torch.tensor([[1.0], [1.001953125]]) if arch == "sm_110a" else b.T
        actual = hc._reference_hc_output(a, b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.backends.cuda.matmul.allow_tf32 is allow_tf32
        assert hc._TEST_DIFF_THRESHOLD == 1e-8
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_oracle_restores_backend_state_after_failure(monkeypatch):
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with pytest.raises(RuntimeError):
            hc._reference_hc_output(torch.ones((2, 3)), torch.ones((2, 4)))
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
