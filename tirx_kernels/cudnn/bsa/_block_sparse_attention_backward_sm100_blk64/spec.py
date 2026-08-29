# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Configuration matrix for the SM100 blk64 BSA backward port."""


def _config(label, **overrides):
    config = {
        "label": label,
        "batch": 1,
        "num_heads": 1,
        "seqlen_q": 128,
        "seqlen_kv": 256,
        "head_dim": 128,
        "dtype": "bfloat16",
        "kv_blocks": 2,
        "tensor_layout": "bhsd",
        "has_block_sizes": True,
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "softmax_scale": None,
        "use_int64_kv_strides": False,
        "bucket_size_blocks": None,
        "seed": 16400,
    }
    config.update(overrides)
    return config


def correctness_configs():
    rows = (
        ("c00_b1_h2_sq128_skv256_kv2_mask", dict(num_heads=2)),
        ("c01_b1_h2_sq128_skv256_kv2_nomask", dict(num_heads=2, has_block_sizes=False)),
        ("c02_b1_h2_sq128_skv256_kv2_mask_bshd", dict(num_heads=2, tensor_layout="bshd")),
        ("c03_b2_h3_sq128_skv320_kv3_mask", dict(batch=2, num_heads=3, seqlen_kv=320, kv_blocks=3)),
        ("c04_b1_h1_sq1_skv65_kv2_mask", dict(seqlen_q=1, seqlen_kv=65)),
        ("c05_b1_h1_sq63_skv255_kv2_mask", dict(seqlen_q=63, seqlen_kv=255)),
        ("c06_b1_h1_sq65_skv257_kv3_mask", dict(seqlen_q=65, seqlen_kv=257, kv_blocks=3)),
        (
            "c07_b1_h1_sq192_skv256_maxkv3_var_1_2_3_mask",
            dict(
                seqlen_q=192, kv_blocks=3, block_count_mode="variable", block_count_pattern="1,2,3"
            ),
        ),
        (
            "c08_b1_h2_sq257_skv512_maxkv4_var_1_mid_max_mask",
            dict(
                num_heads=2,
                seqlen_q=257,
                seqlen_kv=512,
                kv_blocks=4,
                block_count_mode="variable",
                block_count_pattern="1,mid,max",
            ),
        ),
        (
            "c09_b1_h2_sq257_skv512_maxkv4_empty_0_1_max_mask",
            dict(
                num_heads=2,
                seqlen_q=257,
                seqlen_kv=512,
                kv_blocks=4,
                block_count_mode="variable_empty",
                block_count_pattern="0,1,max",
            ),
        ),
        (
            "c10_b1_h2_sq129_skv512_maxkv4_all_empty_mask",
            dict(
                num_heads=2,
                seqlen_q=129,
                seqlen_kv=512,
                kv_blocks=4,
                block_count_mode="variable_empty",
                block_count_pattern="0",
            ),
        ),
        (
            "c11_b1_h2_sq129_skv512_kv4_mask_scale0125",
            dict(num_heads=2, seqlen_q=129, seqlen_kv=512, kv_blocks=4, softmax_scale=0.125),
        ),
        (
            "c12_b2_h2_sq129_skv511_kv4_batch_block_sizes",
            dict(batch=2, num_heads=2, seqlen_q=129, seqlen_kv=511, kv_blocks=4),
        ),
        (
            "c13_b1_h1_sq129_skv4096_kv4_nomask",
            dict(seqlen_q=129, seqlen_kv=4096, kv_blocks=4, has_block_sizes=False),
        ),
        ("c14_b1_h1_sq128_skv256_kv2_mask_i64kv", dict(use_int64_kv_strides=True)),
        ("c15_b1_h1_sq69633_skv256_kv2_mask_bucket1088", dict(seqlen_q=69633)),
        ("c16_b1_h1_sq131073_skv256_kv2_mask_bucket1152", dict(seqlen_q=131073)),
        (
            "c17_b1_h1_sq192001_skv256_maxkv2_var_auto1024",
            dict(seqlen_q=192001, block_count_mode="variable", block_count_pattern="1,max"),
        ),
        (
            "c18_b1_h40_sq131072_skv256_kv1_mask_i64_workspace",
            dict(num_heads=40, seqlen_q=131072, kv_blocks=1),
        ),
        (
            "c19_b1_h1_sq695041_skv128_kv1_mask_convert_grid86881",
            dict(seqlen_q=695041, seqlen_kv=128, kv_blocks=1),
        ),
    )
    configs = []
    for index, (label, overrides) in enumerate(rows):
        configs.append(_config(label, seed=16401 + index, **overrides))
    assert len(configs) == 20
    return configs


def benchmark_configs():
    rows = (
        (
            "p00_b1_h1_sq64_skv4096_kv16_nomask",
            dict(seqlen_q=64, seqlen_kv=4096, kv_blocks=16, has_block_sizes=False),
        ),
        (
            "p01_b1_h2_sq4097_skv8191_kv64_mask",
            dict(num_heads=2, seqlen_q=4097, seqlen_kv=8191, kv_blocks=64),
        ),
        (
            "p02_b2_h8_sq4096_skv8191_kv32_mask",
            dict(batch=2, num_heads=8, seqlen_q=4096, seqlen_kv=8191, kv_blocks=32),
        ),
        (
            "p03_b1_h4_sq8192_skv4096_kv16_mask",
            dict(num_heads=4, seqlen_q=8192, seqlen_kv=4096, kv_blocks=16),
        ),
        (
            "p04_b1_h8_sq4096_skv8192_maxkv32_var_mask",
            dict(
                num_heads=8,
                seqlen_q=4096,
                seqlen_kv=8192,
                kv_blocks=32,
                block_count_mode="variable",
                block_count_pattern="1,mid,max",
            ),
        ),
        (
            "p05_b2_h4_sq4097_skv8191_maxkv32_empty_mask",
            dict(
                batch=2,
                num_heads=4,
                seqlen_q=4097,
                seqlen_kv=8191,
                kv_blocks=32,
                block_count_mode="variable_empty",
                block_count_pattern="0,1,max",
            ),
        ),
        (
            "p06_b1_h4_sq1024_skv32768_kv256_mask",
            dict(num_heads=4, seqlen_q=1024, seqlen_kv=32768, kv_blocks=256),
        ),
        (
            "p07_b1_h4_sq4097_skv16384_maxkv128_var_mask",
            dict(
                num_heads=4,
                seqlen_q=4097,
                seqlen_kv=16384,
                kv_blocks=128,
                block_count_mode="variable",
                block_count_pattern="0,1,max",
            ),
        ),
        (
            "p08_b1_h8_sq2048_skv65536_kv512_nomask",
            dict(num_heads=8, seqlen_q=2048, seqlen_kv=65536, kv_blocks=512, has_block_sizes=False),
        ),
        (
            "p09_b1_h2_sq69633_skv8191_kv32_mask_bucket1088",
            dict(num_heads=2, seqlen_q=69633, seqlen_kv=8191, kv_blocks=32),
        ),
        (
            "p10_b1_h1_sq192001_skv8192_maxkv16_var_mask_auto1024_i64kv",
            dict(
                seqlen_q=192001,
                seqlen_kv=8192,
                kv_blocks=16,
                block_count_mode="variable",
                block_count_pattern="1,mid,max",
                use_int64_kv_strides=True,
            ),
        ),
    )
    configs = []
    for index, (label, overrides) in enumerate(rows):
        configs.append(_config(label, seed=16501 + index, **overrides))
    assert len(configs) == 11
    return configs
