# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Configuration matrix for the SM100 blk64 BSA forward port.

Upstream sources:
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_sm100.py`` and
``python/cudnn/block_sparse_attention/_interface.py``.
"""


def _config(
    label,
    *,
    batch,
    num_q_heads,
    num_kv_heads=None,
    seqlen_q,
    seqlen_kv,
    kv_blocks,
    tensor_layout="bhsd",
    has_block_sizes=True,
    block_count_mode="fixed",
    block_count_pattern=None,
    use_clc=False,
    kv_splits=1,
    softmax_scale=None,
    use_int64_kv_strides=False,
    seed=0,
):
    # A KV head count of None is the MHA case: one KV head per Q head. Q heads
    # are shared across KV heads in whole groups, which is what lets the kernel
    # map a Q head onto its KV head with a single division.
    if num_kv_heads is None:
        num_kv_heads = num_q_heads
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
    return {
        "label": label,
        "batch": batch,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "kv_blocks": kv_blocks,
        "tensor_layout": tensor_layout,
        "has_block_sizes": has_block_sizes,
        "block_count_mode": block_count_mode,
        "block_count_pattern": block_count_pattern,
        "use_clc": use_clc,
        "kv_splits": kv_splits,
        "softmax_scale": softmax_scale,
        "use_int64_kv_strides": use_int64_kv_strides,
        "seed": seed,
    }


def correctness_configs():
    configs = []
    seed = 6400

    def add(label, **kwargs):
        nonlocal seed
        seed += 1
        configs.append(_config(label, seed=seed, **kwargs))

    for tensor_layout in ("bhsd", "bshd"):
        for kv_splits in (1, 2):
            add(
                f"b1_h2_sq128_skv256_kv2_mask_{tensor_layout}_s{kv_splits}_static",
                batch=1,
                num_q_heads=2,
                seqlen_q=128,
                seqlen_kv=256,
                kv_blocks=2,
                tensor_layout=tensor_layout,
                kv_splits=kv_splits,
            )
    add(
        "b2_h1_sq64_skv255_kv2_mask_bhsd_s1_static",
        batch=2,
        num_q_heads=1,
        seqlen_q=64,
        seqlen_kv=255,
        kv_blocks=2,
    )
    for seqlen_q in (1, 63, 65):
        for kv_splits in (1, 2):
            add(
                f"b1_h1_sq{seqlen_q}_skv256_kv2_mask_bhsd_s{kv_splits}_static",
                batch=1,
                num_q_heads=1,
                seqlen_q=seqlen_q,
                seqlen_kv=256,
                kv_blocks=2,
                kv_splits=kv_splits,
            )

    add(
        "b1_h1_sq129_skv512_kv4_nomask_scale0125_bhsd_s1_static",
        batch=1,
        num_q_heads=1,
        seqlen_q=129,
        seqlen_kv=512,
        kv_blocks=4,
        has_block_sizes=False,
        softmax_scale=0.125,
    )
    for label, mode, pattern, mask, clc, splits in (
        (
            "b1_h2_sq129_skv512_maxkv4_var_1_3_4_mask_bhsd_s1_static",
            "variable",
            "1,3,4",
            True,
            False,
            1,
        ),
        (
            "b1_h2_sq129_skv512_maxkv4_var_1_3_4_mask_bhsd_s1_clc_auto",
            "variable",
            "1,3,4",
            True,
            True,
            1,
        ),
        ("b1_h2_sq129_skv512_kv4_mask_bhsd_s1_clc", "fixed", None, True, True, 1),
        (
            "b1_h2_sq129_skv512_maxkv4_var_0_1_4_mask_bhsd_s1_clc",
            "variable_empty",
            "0,1,4",
            True,
            True,
            1,
        ),
        (
            "b1_h2_sq129_skv512_maxkv4_var_0_1_4_mask_bhsd_s2_static",
            "variable_empty",
            "0,1,4",
            True,
            False,
            2,
        ),
        (
            "b1_h2_sq129_skv512_maxkv4_var_1_3_4_nomask_bhsd_s1_static",
            "variable",
            "1,3,4",
            False,
            False,
            1,
        ),
    ):
        add(
            label,
            batch=1,
            num_q_heads=2,
            seqlen_q=129,
            seqlen_kv=512,
            kv_blocks=4,
            has_block_sizes=mask,
            block_count_mode=mode,
            block_count_pattern=pattern,
            use_clc=clc,
            kv_splits=splits,
        )

    for label, sq, skv, blocks, mask, splits in (
        ("b1_h1_sq65_skv4095_kv48_mask_bhsd_s3_static", 65, 4095, 48, True, 3),
        ("b1_h1_sq65_skv4096_kv64_nomask_bhsd_s8_static", 65, 4096, 64, False, 8),
        ("b1_h1_sq1_skv16384_kv256_mask_bhsd_s256_static", 1, 16384, 256, True, 256),
        ("b1_h1_sq64_skv16384_kv256_mask_bhsd_sauto2_static", 64, 16384, 256, True, "auto"),
    ):
        add(
            label,
            batch=1,
            num_q_heads=1,
            seqlen_q=sq,
            seqlen_kv=skv,
            kv_blocks=blocks,
            has_block_sizes=mask,
            kv_splits=splits,
        )
    add(
        "b1_h1_sq65_skv512_kv4_mask_bhsd_s1_static_i64kv",
        batch=1,
        num_q_heads=1,
        seqlen_q=65,
        seqlen_kv=512,
        kv_blocks=4,
        use_int64_kv_strides=True,
    )

    # Grouped Q-to-KV head maps. These are appended rather than interleaved so
    # every row above keeps its seed, and therefore its exact input data. The
    # ratios cover the three lowerings the head map can take: identity folding
    # is already covered by every row above, a power-of-two ratio takes a single
    # shift, and ratio 3 takes a reciprocal multiply.
    for label, hq, hkv, sq, skv, blocks, kwargs in (
        ("b1_h4kv2_sq128_skv256_kv2_mask_bhsd_s1_static_gqa", 4, 2, 128, 256, 2, {}),
        (
            "b1_h4kv2_sq129_skv512_kv4_mask_bshd_s1_static_gqa",
            4,
            2,
            129,
            512,
            4,
            {"tensor_layout": "bshd"},
        ),
        ("b1_h8kv2_sq129_skv512_kv4_mask_bhsd_s1_clc_gqa", 8, 2, 129, 512, 4, {"use_clc": True}),
        (
            "b1_h8kv2_sq129_skv512_maxkv4_var_1_3_4_mask_bhsd_s1_clc_gqa",
            8,
            2,
            129,
            512,
            4,
            {"use_clc": True, "block_count_mode": "variable", "block_count_pattern": "1,3,4"},
        ),
        (
            "b1_h4kv2_sq129_skv512_maxkv4_var_0_1_4_mask_bhsd_s2_static_gqa",
            4,
            2,
            129,
            512,
            4,
            {"kv_splits": 2, "block_count_mode": "variable_empty", "block_count_pattern": "0,1,4"},
        ),
        ("b1_h8kv1_sq65_skv512_kv4_mask_bhsd_s1_static_mqa", 8, 1, 65, 512, 4, {}),
        ("b1_h3kv1_sq129_skv512_kv4_mask_bhsd_s1_static_mqa", 3, 1, 129, 512, 4, {}),
        ("b1_h8kv1_sq129_skv512_kv4_mask_bhsd_s1_clc_mqa", 8, 1, 129, 512, 4, {"use_clc": True}),
        (
            "b1_h4kv2_sq65_skv4096_kv64_nomask_bhsd_s8_static_gqa",
            4,
            2,
            65,
            4096,
            64,
            {"kv_splits": 8, "has_block_sizes": False},
        ),
        ("b2_h4kv2_sq64_skv255_kv2_mask_bhsd_s1_static_gqa", 4, 2, 64, 255, 2, {"batch": 2}),
        (
            "b1_h4kv2_sq65_skv512_kv4_mask_bhsd_s1_static_i64kv_gqa",
            4,
            2,
            65,
            512,
            4,
            {"use_int64_kv_strides": True},
        ),
        ("b1_h4kv1_sq1_skv256_kv2_mask_bhsd_s2_static_mqa", 4, 1, 1, 256, 2, {"kv_splits": 2}),
    ):
        add(
            label,
            batch=kwargs.pop("batch", 1),
            num_q_heads=hq,
            num_kv_heads=hkv,
            seqlen_q=sq,
            seqlen_kv=skv,
            kv_blocks=blocks,
            **kwargs,
        )
    assert len(configs) == 35
    return configs


def benchmark_configs():
    rows = (
        (
            "p00_b1_h1_sq64_skv4096_kv16_nomask_s1_static",
            1,
            1,
            64,
            4096,
            16,
            False,
            "fixed",
            False,
            1,
            False,
        ),
        (
            "p01_b1_h2_sq4097_skv8191_kv64_mask_s1_static",
            1,
            2,
            4097,
            8191,
            64,
            True,
            "fixed",
            False,
            1,
            False,
        ),
        (
            "p02_b2_h8_sq4096_skv8191_kv32_mask_s1_static",
            2,
            8,
            4096,
            8191,
            32,
            True,
            "fixed",
            False,
            1,
            False,
        ),
        (
            "p03_b1_h4_sq8192_skv4096_kv16_mask_s1_clc",
            1,
            4,
            8192,
            4096,
            16,
            True,
            "fixed",
            True,
            1,
            False,
        ),
        (
            "p04_b1_h8_sq4096_skv8192_maxkv32_var_mask_s1_clc",
            1,
            8,
            4096,
            8192,
            32,
            True,
            "variable",
            True,
            1,
            False,
        ),
        (
            "p05_b2_h4_sq4097_skv8191_maxkv32_empty_mask_s1_clc",
            2,
            4,
            4097,
            8191,
            32,
            True,
            "variable_empty",
            True,
            1,
            False,
        ),
        (
            "p06_b1_h4_sq1024_skv32768_kv256_mask_s2_static",
            1,
            4,
            1024,
            32768,
            256,
            True,
            "fixed",
            False,
            2,
            False,
        ),
        (
            "p07_b1_h4_sq4097_skv16384_maxkv128_var_mask_s2_static",
            1,
            4,
            4097,
            16384,
            128,
            True,
            "variable_empty",
            False,
            2,
            False,
        ),
        (
            "p08_b1_h4_sq2048_skv32768_kv256_mask_s4_static",
            1,
            4,
            2048,
            32768,
            256,
            True,
            "fixed",
            False,
            4,
            False,
        ),
        (
            "p09_b1_h8_sq2048_skv65536_kv512_nomask_s8_static",
            1,
            8,
            2048,
            65536,
            512,
            False,
            "fixed",
            False,
            8,
            False,
        ),
        (
            "p10_b1_h4_sq4096_skv8192_kv32_mask_s1_static_i64kv",
            1,
            4,
            4096,
            8192,
            32,
            True,
            "fixed",
            False,
            1,
            True,
        ),
        # Grouped head maps, mirroring the shape scale of the rows above so a
        # ratio's cost is read against a comparable equal-head row. A trailing
        # element carries the KV head count; rows without one are equal-head.
        (
            "p11_b1_h8kv4_sq4096_skv8192_kv32_mask_s1_static_gqa",
            1,
            8,
            4096,
            8192,
            32,
            True,
            "fixed",
            False,
            1,
            False,
            4,
        ),
        (
            "p12_b2_h8kv2_sq4096_skv8191_kv32_mask_s1_static_gqa",
            2,
            8,
            4096,
            8191,
            32,
            True,
            "fixed",
            False,
            1,
            False,
            2,
        ),
        (
            "p13_b1_h8kv4_sq8192_skv4096_kv16_mask_s1_clc_gqa",
            1,
            8,
            8192,
            4096,
            16,
            True,
            "fixed",
            True,
            1,
            False,
            4,
        ),
        (
            "p14_b1_h8kv2_sq4096_skv8192_maxkv32_var_mask_s1_clc_gqa",
            1,
            8,
            4096,
            8192,
            32,
            True,
            "variable",
            True,
            1,
            False,
            2,
        ),
        (
            "p15_b2_h4kv1_sq4097_skv8191_maxkv32_empty_mask_s1_clc_mqa",
            2,
            4,
            4097,
            8191,
            32,
            True,
            "variable_empty",
            True,
            1,
            False,
            1,
        ),
        (
            "p16_b1_h8kv2_sq1024_skv32768_kv256_mask_s2_static_gqa",
            1,
            8,
            1024,
            32768,
            256,
            True,
            "fixed",
            False,
            2,
            False,
            2,
        ),
        (
            "p17_b1_h8kv4_sq4097_skv16384_maxkv128_var_mask_s2_static_gqa",
            1,
            8,
            4097,
            16384,
            128,
            True,
            "variable_empty",
            False,
            2,
            False,
            4,
        ),
        (
            "p18_b1_h8kv1_sq2048_skv32768_kv256_mask_s4_static_mqa",
            1,
            8,
            2048,
            32768,
            256,
            True,
            "fixed",
            False,
            4,
            False,
            1,
        ),
        (
            "p19_b1_h8kv1_sq2048_skv65536_kv512_nomask_s8_static_mqa",
            1,
            8,
            2048,
            65536,
            512,
            False,
            "fixed",
            False,
            8,
            False,
            1,
        ),
        (
            "p20_b1_h8kv2_sq4096_skv8192_kv32_mask_s1_static_i64kv_gqa",
            1,
            8,
            4096,
            8192,
            32,
            True,
            "fixed",
            False,
            1,
            True,
            2,
        ),
        (
            "p21_b1_h8kv1_sq4097_skv8191_kv64_mask_s1_static_mqa",
            1,
            8,
            4097,
            8191,
            64,
            True,
            "fixed",
            False,
            1,
            False,
            1,
        ),
        (
            "p22_b1_h6kv2_sq2048_skv8192_kv32_mask_s1_static_gqa",
            1,
            6,
            2048,
            8192,
            32,
            True,
            "fixed",
            False,
            1,
            False,
            2,
        ),
    )
    configs = []
    for seed, row in enumerate(rows, start=7400):
        label, batch, heads, sq, skv, blocks, mask, mode, clc, splits, i64kv = row[:11]
        kv_heads = row[11] if len(row) > 11 else heads
        pattern = (
            "0,1,max" if mode == "variable_empty" else "1,mid,max" if mode == "variable" else None
        )
        configs.append(
            _config(
                label,
                batch=batch,
                num_q_heads=heads,
                num_kv_heads=kv_heads,
                seqlen_q=sq,
                seqlen_kv=skv,
                kv_blocks=blocks,
                has_block_sizes=mask,
                block_count_mode=mode,
                block_count_pattern=pattern,
                use_clc=clc,
                kv_splits=splits,
                use_int64_kv_strides=i64kv,
                seed=seed,
            )
        )
    return configs
