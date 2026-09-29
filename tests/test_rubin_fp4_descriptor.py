# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FP4 instruction descriptors must encode SM107 v1 even for dense MMA."""

from tirx_kernels.registry import load_kernel


def test_rubin_dense_fp4_descriptors_encode_k128_and_v1():
    dense = load_kernel("dense_blockscaled_gemm_sm107", strict=True)
    grouped = load_kernel("grouped_gemm_masked_rubin", strict=True)
    fused = load_kernel(
        "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin", strict=True
    )
    assert fused._instruction_descriptor() == 0x08201488
    for n, sf, expected in [(128, "float8_e4m3fn", 0x08201488), (64, "float8_e8m0fnu", 0x08901488)]:
        assert dense._instruction_descriptor(128, n, sf) == expected
        assert grouped._instruction_descriptor(128, n, "float4_e2m1fn", sf) == expected
