# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Static specialization domain for the DSA sparse attention backward kernel.

Upstream sources:
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py``
(``FlashAttentionDSABackwardSm100.__init__`` tile and role constants), and
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm100.py``
(the ``head_dim`` gate, the fixed ``block_tile``, and the compile key).

The correctness matrix follows the upstream test file
``test/python/fe_api/dsa/test_DSA_sparse_attention_backward.py``; the benchmark
matrix follows ``benchmark/dsa/benchmark_dsa_sparse_attention_backward.py``.
"""

# ``_interface_sm100.py:94``: the SM100 path pins the KV block tile and ignores
# the public API's ``block_tile`` argument.
BLOCK_TILE = 64

# ``_interface_sm100.py:63-64``: any other head_dim indexes shared memory out of
# bounds inside the kernel.
SUPPORTED_HEAD_DIMS = (512, 576)

# ``test/python/fe_api/dsa/dsa_utils.py:34-46``: the (head_dim, head_dim_v,
# num_heads) triples upstream exercises and ships.
SUPPORTED_GEOMETRIES = {512: 64, 576: 32}

SUPPORTED_DTYPES = ("bfloat16", "float16")

# ``utils.get_smem_capacity_in_bytes("sm_100")``; the upstream kernel asserts its
# SharedStorage against a self-imposed 227 KiB bound (``dsa_bwd_sm100.py:446``).
SMEM_CAPACITY = 232_448
SMEM_SELF_IMPOSED_CAP = 227 * 1024

# ``dsa_bwd_sm100.py:79-80``: the kernel claims all of tensor memory.
TMEM_CAPACITY_COLUMNS = 512

# How ``topk_length`` (or the ``-1`` padding of ``topk_idxs``) is generated for a
# config. Each mode targets a distinct control-flow path in the kernel.
TOPK_MODES = (
    "full",  # every query uses all max_topk rows; no ragged tile
    "random",  # per-query length in [1, max_topk]; ragged first-processed tile
    "mixed_zero",  # some queries have length <= 0 and take the early-exit path
    "all_empty",  # every query takes the early-exit path
    "invalid_idx",  # non-compact variant with -1 entries in the index tail
)

# How ``attn_sink`` is filled. ``log_topk`` reproduces the upstream regression
# test for folding the sink into the softmax normalization.
SINK_MODES = ("normal", "log_topk", "disabled")


def head_dim_v_for(head_dim):
    """Mirror ``_interface_sm100.py:64``."""
    return 512 if head_dim == 576 else head_dim


def check_dispatch(config):
    """Reject anything outside the upstream dispatch domain."""
    head_dim = config["head_dim"]
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {SUPPORTED_HEAD_DIMS}, got {head_dim}")
    expected_num_head = SUPPORTED_GEOMETRIES[head_dim]
    if config["num_head"] != expected_num_head:
        raise ValueError(
            f"head_dim {head_dim} requires num_head={expected_num_head}, got {config['num_head']}"
        )
    if config["dtype"] not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {config['dtype']}")
    if config["topk_mode"] not in TOPK_MODES:
        raise ValueError(f"topk_mode must be one of {TOPK_MODES}, got {config['topk_mode']}")
    if config["sink_mode"] not in SINK_MODES:
        raise ValueError(f"sink_mode must be one of {SINK_MODES}, got {config['sink_mode']}")
    if config["topk_mode"] == "invalid_idx" and config["has_topk_length"]:
        raise ValueError(
            "the -1 index padding path is the non-compact variant; it has no topk_length"
        )
    return config


def _config(
    *,
    head_dim,
    num_head,
    seqlen_q,
    seqlen_kv,
    max_topk,
    has_topk_length,
    dtype="bfloat16",
    topk_mode="random",
    sink_mode="normal",
    seed,
    label,
):
    return check_dispatch(
        {
            "head_dim": head_dim,
            "num_head": num_head,
            "seqlen_q": seqlen_q,
            "seqlen_kv": seqlen_kv,
            "max_topk": max_topk,
            "has_topk_length": has_topk_length,
            "dtype": dtype,
            "topk_mode": topk_mode,
            "sink_mode": sink_mode,
            "seed": seed,
            "label": label,
        }
    )


def correctness_configs():
    """The correctness matrix.

    Every row here is a mandatory CI case (``tests/test_correctness.py``
    parametrizes over ``CONFIGS``), so each one must run on a working SM100
    machine without skipping. The base geometry matches the upstream test
    defaults (``s_q=1024``, ``s_kv=4096``, ``topk=512``); the remaining rows each
    pin one additional branch of the kernel.
    """
    configs = []
    seed = 0

    def add(**kwargs):
        nonlocal seed
        seed += 1
        configs.append(_config(seed=seed, **kwargs))

    # The two shipped geometries, both compiled index variants, at the upstream
    # test's base shape.
    for head_dim, num_head in sorted(SUPPORTED_GEOMETRIES.items()):
        for has_len in (True, False):
            add(
                head_dim=head_dim,
                num_head=num_head,
                seqlen_q=1024,
                seqlen_kv=4096,
                max_topk=512,
                has_topk_length=has_len,
                topk_mode="random" if has_len else "full",
                label=f"d{head_dim}_h{num_head}_bf16_sq1024_skv4096_t512_{'len' if has_len else 'nolen'}",
            )

    # fp16 element storage, one row per geometry.
    add(
        head_dim=512,
        num_head=64,
        seqlen_q=1024,
        seqlen_kv=4096,
        max_topk=512,
        has_topk_length=True,
        dtype="float16",
        topk_mode="random",
        label="d512_h64_fp16_sq1024_skv4096_t512_len",
    )
    add(
        head_dim=576,
        num_head=32,
        seqlen_q=1024,
        seqlen_kv=4096,
        max_topk=512,
        has_topk_length=False,
        dtype="float16",
        topk_mode="full",
        label="d576_h32_fp16_sq1024_skv4096_t512_nolen",
    )

    # A single-tile top-k so the ragged, first-processed tile is the only tile.
    add(
        head_dim=512,
        num_head=64,
        seqlen_q=256,
        seqlen_kv=4096,
        max_topk=64,
        has_topk_length=True,
        topk_mode="random",
        label="d512_h64_bf16_sq256_skv4096_t64_len_ragged",
    )

    # Zero and negative lengths interleaved with live rows: the early-exit path
    # must not deadlock the CTAs that still have work.
    add(
        head_dim=512,
        num_head=64,
        seqlen_q=128,
        seqlen_kv=4096,
        max_topk=512,
        has_topk_length=True,
        topk_mode="mixed_zero",
        label="d512_h64_bf16_sq128_skv4096_t512_zeroneg",
    )
    add(
        head_dim=576,
        num_head=32,
        seqlen_q=128,
        seqlen_kv=4096,
        max_topk=512,
        has_topk_length=True,
        topk_mode="all_empty",
        label="d576_h32_bf16_sq128_skv4096_t512_allempty",
    )

    # The non-compact variant's invalid-row marker.
    add(
        head_dim=512,
        num_head=64,
        seqlen_q=128,
        seqlen_kv=4096,
        max_topk=512,
        has_topk_length=False,
        topk_mode="invalid_idx",
        label="d512_h64_bf16_sq128_skv4096_t512_negidx",
    )

    # Sink folded into the normalization, the upstream d576 regression.
    add(
        head_dim=576,
        num_head=32,
        seqlen_q=1024,
        seqlen_kv=4096,
        max_topk=512,
        has_topk_length=True,
        topk_mode="random",
        sink_mode="log_topk",
        label="d576_h32_bf16_sq1024_skv4096_t512_sinkfold",
    )

    # Sink disabled (-inf), so the softmax normalization sees the KV terms alone.
    add(
        head_dim=512,
        num_head=64,
        seqlen_q=256,
        seqlen_kv=4096,
        max_topk=512,
        has_topk_length=True,
        topk_mode="random",
        sink_mode="disabled",
        label="d512_h64_bf16_sq256_skv4096_t512_nosink",
    )

    return configs


def benchmark_configs():
    """The performance matrix: the upstream benchmark sweep on both geometries.

    ``benchmark/dsa/benchmark_dsa_sparse_attention_backward.py`` sweeps
    ``seqlens x topks`` with ``seqlen_kv == seqlen_q``, bf16, the sink enabled and
    a full-length ``topk_length``. The d576 half mirrors that sweep onto the
    second compiled program so it cannot regress unmeasured.
    """
    configs = []
    seed = 100
    for head_dim, num_head in sorted(SUPPORTED_GEOMETRIES.items()):
        for seqlen in (4096, 8192):
            for topk in (128, 512, 1024, 2048):
                seed += 1
                configs.append(
                    _config(
                        head_dim=head_dim,
                        num_head=num_head,
                        seqlen_q=seqlen,
                        seqlen_kv=seqlen,
                        max_topk=topk,
                        has_topk_length=True,
                        topk_mode="full",
                        seed=seed,
                        label=f"d{head_dim}_h{num_head}_bf16_sq{seqlen}_t{topk}_len",
                    )
                )
    return configs
