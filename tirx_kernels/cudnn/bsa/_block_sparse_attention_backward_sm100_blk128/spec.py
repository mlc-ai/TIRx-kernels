# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Configuration domain for the SM100 blk128 BSA backward program."""


def _config(label, **overrides):
    config = {
        "label": label,
        "batch": 1,
        "num_heads": 1,
        "seqlen_q": 128,
        "seqlen_kv": 512,
        "head_dim": 128,
        "dtype": "bfloat16",
        "kv_blocks": 2,
        "tensor_layout": "bhsd",
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "softmax_scale": None,
        "use_int64_kv_strides": False,
        "bucket_size_blocks": None,
        "preallocate_outputs": False,
        "data_mode": "random",
        "seed": 128000,
    }
    config.update(overrides)
    return config


def q_block_count(config):
    return (int(config["seqlen_q"]) + 127) // 128


def kv_block_count(config):
    return (int(config["seqlen_kv"]) + 127) // 128


def bucket_size_blocks(config):
    explicit = config.get("bucket_size_blocks")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    q_blocks = q_block_count(config)
    if q_blocks >= 4096 and int(config["num_heads"]) <= 1:
        return 256
    if q_blocks >= 2048:
        return 512
    return 384


def q_group_count(config):
    q_blocks = q_block_count(config)
    bucket = bucket_size_blocks(config)
    return (q_blocks + bucket - 1) // bucket


def validate_config(config):
    if config["dtype"] != "bfloat16":
        raise ValueError("SM100 blk128 backward supports bfloat16 only")
    if int(config["head_dim"]) not in (64, 128):
        raise ValueError("SM100 blk128 backward supports head_dim 64 or 128")
    if config["tensor_layout"] not in ("bhsd", "bshd"):
        raise ValueError("tensor_layout must be 'bhsd' or 'bshd'")
    if config["block_count_mode"] not in ("fixed", "variable", "variable_empty"):
        raise ValueError("unsupported block_count_mode")
    if int(config["kv_blocks"]) < 1 or int(config["kv_blocks"]) > kv_block_count(config):
        raise ValueError("kv_blocks must fit the physical KV block count")
    if any(int(config[name]) < 1 for name in ("batch", "num_heads", "seqlen_q", "seqlen_kv")):
        raise ValueError("batch, heads, and sequence lengths must be positive")
    if config["data_mode"] not in ("random", "cancellation", "sharp_softmax"):
        raise ValueError("unsupported data_mode")
    return config


def correctness_configs():
    rows = (
        ("c00_b1_h1_d64_sq128_skv512_kv2", dict(head_dim=64)),
        ("c01_b1_h1_d128_sq512_skv512_kv2_shared", dict(seqlen_q=512)),
        (
            "c02_b1_h2_d128_sq256_skv512_kv2_bshd_prealloc",
            dict(num_heads=2, seqlen_q=256, tensor_layout="bshd", preallocate_outputs=True),
        ),
        (
            "c03_b2_h3_d64_sq257_skv641_kv4_tails",
            dict(batch=2, num_heads=3, seqlen_q=257, seqlen_kv=641, head_dim=64, kv_blocks=4),
        ),
        ("c04_b1_h1_d64_sq1_skv129_kv2", dict(head_dim=64, seqlen_q=1, seqlen_kv=129)),
        ("c05_b1_h1_d128_sq127_skv255_kv2", dict(seqlen_q=127, seqlen_kv=255)),
        ("c06_b1_h1_d128_sq129_skv257_kv2", dict(seqlen_q=129, seqlen_kv=257)),
        (
            "c07_b1_h1_d64_sq513_skv1024_maxkv4_var_1_2_3",
            dict(
                head_dim=64,
                seqlen_q=513,
                seqlen_kv=1024,
                kv_blocks=4,
                block_count_mode="variable",
                block_count_pattern="1,2,3",
            ),
        ),
        (
            "c08_b1_h1_d128_sq513_skv1024_maxkv4_var_0_1_max",
            dict(
                seqlen_q=513,
                seqlen_kv=1024,
                kv_blocks=4,
                block_count_mode="variable_empty",
                block_count_pattern="0,1,max",
            ),
        ),
        (
            "c09_b1_h1_d128_sq513_skv1024_maxkv4_all_empty",
            dict(
                seqlen_q=513,
                seqlen_kv=1024,
                kv_blocks=4,
                block_count_mode="variable_empty",
                block_count_pattern="0",
            ),
        ),
        (
            "c10_b1_h1_d64_sq256_skv512_kv2_scale0125",
            dict(head_dim=64, seqlen_q=256, softmax_scale=0.125),
        ),
        ("c11_b1_h1_d128_sq256_skv512_kv2_i64kv", dict(seqlen_q=256, use_int64_kv_strides=True)),
        ("c12_b1_h1_d64_sq512_skv512_kv2_bucket1", dict(head_dim=64, seqlen_q=512)),
        ("c13_b1_h1_d128_sq513_skv512_kv2_bucket2", dict(seqlen_q=513)),
        ("c14_b1_h1_d128_sq49152_skv512_kv2_qb384_g1", dict(seqlen_q=49152)),
        ("c15_b1_h1_d128_sq49153_skv512_kv2_qb385_g2", dict(seqlen_q=49153)),
        (
            "c16_b1_h1_d64_sq256_skv512_kv2_cancellation",
            dict(head_dim=64, seqlen_q=256, data_mode="cancellation"),
        ),
        (
            "c17_b1_h1_d128_sq256_skv512_kv2_sharp_softmax",
            dict(seqlen_q=256, data_mode="sharp_softmax"),
        ),
    )
    configs = []
    for index, (label, overrides) in enumerate(rows):
        configs.append(_config(label, seed=128001 + index, **overrides))
    assert len(configs) == 18
    return configs


def benchmark_configs():
    rows = (
        ("p00_b1_h1_d64_sq128_skv4096_kv16", dict(head_dim=64, seqlen_kv=4096, kv_blocks=16)),
        ("p01_b1_h1_d128_sq512_skv4096_kv16", dict(seqlen_q=512, seqlen_kv=4096, kv_blocks=16)),
        (
            "p02_b2_h8_d128_sq4096_skv8191_kv32",
            dict(batch=2, num_heads=8, seqlen_q=4096, seqlen_kv=8191, kv_blocks=32),
        ),
        (
            "p03_b1_h8_d64_sq4097_skv8191_maxkv32_var",
            dict(
                num_heads=8,
                head_dim=64,
                seqlen_q=4097,
                seqlen_kv=8191,
                kv_blocks=32,
                block_count_mode="variable",
                block_count_pattern="1,mid,max",
            ),
        ),
        (
            "p04_b2_h4_d128_sq4097_skv8191_maxkv32_empty",
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
            "p05_b1_h4_d64_sq1024_skv32768_kv128",
            dict(num_heads=4, head_dim=64, seqlen_q=1024, seqlen_kv=32768, kv_blocks=128),
        ),
        (
            "p06_b1_h8_d128_sq2048_skv65536_kv256",
            dict(num_heads=8, seqlen_q=2048, seqlen_kv=65536, kv_blocks=256),
        ),
        (
            "p07_b1_h4_d128_sq49152_skv8192_kv32_qb384_g1",
            dict(num_heads=4, seqlen_q=49152, seqlen_kv=8192, kv_blocks=32),
        ),
        (
            "p08_b1_h4_d128_sq49153_skv8191_maxkv32_var_qb385_g2",
            dict(
                num_heads=4,
                seqlen_q=49153,
                seqlen_kv=8191,
                kv_blocks=32,
                block_count_mode="variable",
                block_count_pattern="1,mid,max",
            ),
        ),
        (
            "p09_b1_h2_d64_sq262144_skv8192_kv32_qb2048_g4",
            dict(num_heads=2, head_dim=64, seqlen_q=262144, seqlen_kv=8192, kv_blocks=32),
        ),
        (
            "p10_b1_h1_d128_sq524288_skv8192_maxkv32_var_qb4096_g16_i64kv",
            dict(
                seqlen_q=524288,
                seqlen_kv=8192,
                kv_blocks=32,
                block_count_mode="variable",
                block_count_pattern="1,mid,max",
                use_int64_kv_strides=True,
            ),
        ),
        (
            "p11_b1_h2_d128_sq524288_skv8192_kv32_qb4096_g8",
            dict(num_heads=2, seqlen_q=524288, seqlen_kv=8192, kv_blocks=32),
        ),
        (
            "p12_b1_h4_d128_sq8192_skv4096_kv16_bshd_scale0125",
            dict(
                num_heads=4,
                seqlen_q=8192,
                seqlen_kv=4096,
                kv_blocks=16,
                tensor_layout="bshd",
                softmax_scale=0.125,
            ),
        ),
    )
    configs = []
    for index, (label, overrides) in enumerate(rows):
        configs.append(_config(label, seed=128101 + index, **overrides))
    assert len(configs) == 13
    return configs
