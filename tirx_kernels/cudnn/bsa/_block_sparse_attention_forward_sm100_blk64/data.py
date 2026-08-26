# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Inputs, launch wiring and validation for the SM100 blk64 BSA forward pass.

Upstream source: ``python/cudnn/block_sparse_attention/_interface.py``.
"""

import math

import torch

_TMA_SWIZZLE_128B = 3
_TMA_L2_256B = 2


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

    def __init__(self):
        import ctypes

        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensormap(tensor, dtype, global_dims, global_strides, box_dims):
    """Encode the exact rank-4/rank-5 descriptors consumed by the producer."""
    import ctypes

    import tvm

    rank = len(global_dims)
    descriptor = _AlignedTensorMap()
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        descriptor.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *global_dims,
        *global_strides,
        *box_dims,
        *((1,) * rank),
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        _TMA_SWIZZLE_128B,
        _TMA_L2_256B,
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def _build_tensor_maps(q, k, v, out, *, seqlen_q, seqlen_kv, heads, batch):
    """Build the Q/K/V/O maps in the source kernel's fused block coordinates."""
    physical_blocks = (seqlen_kv + 63) // 64
    q_strides = (128 * 2, seqlen_q * 128 * 2, heads * seqlen_q * 128 * 2)
    kv_strides = (128 * 2, 64 * 2, 64 * 128 * 2, seqlen_kv * 128 * 2)
    out_element_size = out.element_size()
    out_heads = out.shape[1]
    out_strides = (
        128 * out_element_size,
        seqlen_q * 128 * out_element_size,
        out_heads * seqlen_q * 128 * out_element_size,
    )
    return {
        "q": _encode_tensormap(
            q, "bfloat16", (128, seqlen_q, heads, batch), q_strides, (64, 64, 1, 1)
        ),
        "k": _encode_tensormap(
            k,
            "bfloat16",
            (64, 64, 2, physical_blocks, batch * heads),
            kv_strides,
            (64, 64, 1, 1, 1),
        ),
        "v": _encode_tensormap(
            v,
            "bfloat16",
            (64, 64, 2, physical_blocks, batch * heads),
            kv_strides,
            (64, 64, 2, 1, 1),
        ),
        "o": _encode_tensormap(
            out,
            "float32" if out.dtype == torch.float32 else "bfloat16",
            (128, seqlen_q, out_heads, batch),
            out_strides,
            ((32 if out.dtype == torch.float32 else 64), 64, 1, 1),
        ),
    }


def _resolve_splits(value):
    if value == "auto":
        return 2
    return int(value)


def _counts_from_pattern(config, q_blocks, device):
    batch = config["batch"]
    heads = config["num_heads"]
    maximum = config["kv_blocks"]
    mode = config["block_count_mode"]
    if mode == "fixed":
        return torch.full((batch, heads, q_blocks), maximum, dtype=torch.int32, device=device)

    pattern = config["block_count_pattern"]
    if pattern is None:
        values = (0, 1, maximum) if mode == "variable_empty" else (1, max(1, maximum // 2), maximum)
    else:
        values = []
        for item in pattern.split(","):
            item = item.strip()
            if item == "max":
                values.append(maximum)
            elif item == "mid":
                values.append(max(1, maximum // 2))
            else:
                values.append(int(item))
        values = tuple(values)
    flat = torch.arange(batch * heads * q_blocks, device=device)
    chosen = torch.tensor(values, dtype=torch.int32, device=device)[flat % len(values)]
    return chosen.reshape(batch, heads, q_blocks)


def _build_split_offsets(counts, num_splits):
    split_ids = torch.arange(num_splits + 1, dtype=torch.int64, device=counts.device)
    valid = counts.to(torch.int64).clamp_min(0)
    average = valid // num_splits
    aligned_base = average // 8 * 8
    remainder = valid - aligned_base * num_splits
    even = (valid[..., None] * split_ids + num_splits - 1) // num_splits
    aligned = aligned_base[..., None] * split_ids + torch.minimum(
        remainder[..., None], split_ids * 8
    )
    aligned = torch.minimum(aligned, valid[..., None])
    return torch.where((aligned_base == 0)[..., None], even, aligned).to(torch.int32).contiguous()


def _make_sparse_metadata(config, *, q_blocks, physical_blocks, device):
    maximum = config["kv_blocks"]
    total = config["batch"] * config["num_heads"] * q_blocks
    rows = torch.empty((total, maximum), dtype=torch.int32, device=device)
    base = torch.arange(maximum, dtype=torch.int64, device=device)
    # A coprime-ish stride spreads the selected blocks through the KV sequence;
    # the final slot is pinned to the physical tail so block_sizes coverage is
    # deterministic whenever a row consumes its full roster.
    stride = max(1, physical_blocks // maximum)
    for row in range(total):
        values = (base * stride + row * 7) % physical_blocks
        if maximum > 0:
            values[-1] = physical_blocks - 1
        rows[row] = values.to(torch.int32)
    return rows.reshape(config["batch"], config["num_heads"], q_blocks, maximum)


def _strided_kv(generator, shape, *, use_i64):
    batch, heads, seqlen, dim = shape
    base = torch.randn(
        batch * heads * seqlen * dim, generator=generator, dtype=torch.bfloat16, device="cuda"
    )
    if not use_i64:
        return base.reshape(shape)
    if batch != 1:
        raise ValueError("the Int64-stride probe requires a singleton batch")
    return torch.as_strided(base, shape, (2**31, seqlen * dim, dim, 1))


def prepare_data(**config):
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("CUDA is required")

    batch = config["batch"]
    heads = config["num_heads"]
    seqlen_q = config["seqlen_q"]
    seqlen_kv = config["seqlen_kv"]
    q_blocks = (seqlen_q + 63) // 64
    physical_blocks = (seqlen_kv + 63) // 64
    if config["kv_blocks"] > physical_blocks:
        raise ValueError("kv_blocks exceeds the physical KV block count")

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    q_bhsd = torch.randn(
        batch, heads, seqlen_q, 128, generator=generator, dtype=torch.bfloat16, device="cuda"
    )
    k_bhsd = _strided_kv(
        generator, (batch, heads, seqlen_kv, 128), use_i64=config["use_int64_kv_strides"]
    )
    v_bhsd = _strided_kv(
        generator, (batch, heads, seqlen_kv, 128), use_i64=config["use_int64_kv_strides"]
    )
    if config["tensor_layout"] == "bshd":
        q_user = q_bhsd.transpose(1, 2).contiguous()
        k_user = k_bhsd.transpose(1, 2).contiguous()
        v_user = v_bhsd.transpose(1, 2).contiguous()
        # TIRx consumes the wrapper-normalized native BHSD buffers.
        q_bhsd, k_bhsd, v_bhsd = (
            q_user.transpose(1, 2).contiguous(),
            k_user.transpose(1, 2).contiguous(),
            v_user.transpose(1, 2).contiguous(),
        )
    else:
        q_user, k_user, v_user = q_bhsd, k_bhsd, v_bhsd

    block_index = _make_sparse_metadata(
        config, q_blocks=q_blocks, physical_blocks=physical_blocks, device="cuda"
    )
    block_sizes = torch.full((physical_blocks,), 64, dtype=torch.int32, device="cuda")
    block_sizes[-1] = seqlen_kv - (physical_blocks - 1) * 64
    block_nums = _counts_from_pattern(config, q_blocks, "cuda")
    num_splits = _resolve_splits(config["kv_splits"])
    split_offsets = (
        _build_split_offsets(block_nums, num_splits)
        if num_splits > 1
        else torch.empty(1, dtype=torch.int32, device="cuda")
    )
    softmax_scale = (
        1.0 / math.sqrt(128) if config["softmax_scale"] is None else float(config["softmax_scale"])
    )

    final_out = torch.empty((batch, heads, seqlen_q, 128), dtype=torch.bfloat16, device="cuda")
    final_lse = torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device="cuda")
    if num_splits > 1:
        partial_out = torch.empty(
            (batch, num_splits * heads, seqlen_q, 128), dtype=torch.float32, device="cuda"
        )
        partial_lse = torch.empty(
            (batch, num_splits * heads, seqlen_q), dtype=torch.float32, device="cuda"
        )
    else:
        partial_out = torch.empty(1, dtype=torch.float32, device="cuda")
        partial_lse = final_lse

    producer_out = partial_out if num_splits > 1 else final_out
    tensor_maps = _build_tensor_maps(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        producer_out,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        heads=heads,
        batch=batch,
    )

    return {
        "config": dict(config),
        "inputs": {
            "q": q_bhsd,
            "k": k_bhsd,
            "v": v_bhsd,
            "q_user": q_user,
            "k_user": k_user,
            "v_user": v_user,
            "block_index": block_index,
            "block_sizes": block_sizes,
            "block_nums": block_nums,
            "split_offsets": split_offsets,
            "softmax_scale": softmax_scale,
        },
        "tirx": {
            "out": final_out,
            "lse": final_lse,
            "partial_out": partial_out,
            "partial_lse": partial_lse,
        },
        "source": {"out": None, "lse": None},
        "derived": {"num_splits": num_splits, "q_blocks": q_blocks},
        "tensor_maps": tensor_maps,
    }


def tirx_launch(executables, data):
    inputs = data["inputs"]
    outputs = data["tirx"]
    tensor_maps = data["tensor_maps"]
    forward = executables[0]
    num_splits = data["derived"]["num_splits"]
    producer_lse = outputs["partial_lse"]

    def launch():
        forward(
            tensor_maps["q"].ptr,
            tensor_maps["k"].ptr,
            tensor_maps["v"].ptr,
            tensor_maps["o"].ptr,
            producer_lse.reshape(-1),
            inputs["block_index"].reshape(-1),
            inputs["block_sizes"].reshape(-1),
            inputs["block_nums"].reshape(-1),
            inputs["split_offsets"].reshape(-1),
            inputs["softmax_scale"] * math.log2(math.e),
        )
        if num_splits > 1:
            executables[1](
                outputs["partial_out"].reshape(-1),
                outputs["partial_lse"].reshape(-1),
                outputs["out"].reshape(-1),
                outputs["lse"].reshape(-1),
            )

    return launch


def _oracle(data):
    inputs = data["inputs"]
    config = data["config"]
    q = inputs["q"].float()
    k = inputs["k"].float()
    v = inputs["v"].float()
    block_index = inputs["block_index"]
    block_sizes = inputs["block_sizes"]
    block_nums = inputs["block_nums"]
    batch, heads, seqlen_q, _ = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((batch, heads, seqlen_q), -float("inf"), device=q.device)
    for b in range(batch):
        for h in range(heads):
            for qb in range((seqlen_q + 63) // 64):
                q0, q1 = qb * 64, min(qb * 64 + 64, seqlen_q)
                count = (
                    config["kv_blocks"]
                    if config["block_count_mode"] == "fixed"
                    else int(block_nums[b, h, qb])
                )
                if count <= 0:
                    continue
                tokens = []
                for slot in range(count):
                    sparse_id = int(block_index[b, h, qb, slot])
                    size = int(block_sizes[sparse_id]) if config["has_block_sizes"] else 64
                    tokens.extend(range(sparse_id * 64, min(sparse_id * 64 + size, k.shape[2])))
                token_ids = torch.tensor(tokens, dtype=torch.long, device=q.device)
                gathered_k = k[b, h].index_select(0, token_ids)
                gathered_v = v[b, h].index_select(0, token_ids)
                score = q[b, h, q0:q1] @ gathered_k.T * inputs["softmax_scale"]
                probability = torch.softmax(score, dim=-1)
                out[b, h, q0:q1] = probability @ gathered_v
                lse[b, h, q0:q1] = torch.logsumexp(score, dim=-1)
    return out, lse


def validate_outputs(data, *, sources, with_oracle=True):
    if not with_oracle:
        if "source" in sources and data["source"]["out"] is not None:
            torch.testing.assert_close(
                data["tirx"]["out"].float(), data["source"]["out"].float(), atol=3e-2, rtol=3e-2
            )
        return

    expected_out, expected_lse = _oracle(data)
    for source in sources:
        actual = data[source]
        if actual["out"] is None or actual["lse"] is None:
            raise AssertionError(f"{source} did not produce outputs")
        torch.testing.assert_close(
            actual["out"].float(),
            expected_out,
            atol=3e-2,
            rtol=3e-2,
            msg=lambda message: f"{source}.out: {message}",
        )
        # Empty rows legitimately carry -inf; assert matching finiteness first
        # so close-comparison diagnostics remain local to live rows.
        finite = torch.isfinite(expected_lse)
        if not torch.equal(torch.isfinite(actual["lse"]), finite):
            raise AssertionError(f"{source}.lse finiteness differs from the oracle")
        torch.testing.assert_close(
            actual["lse"][finite],
            expected_lse[finite],
            atol=2e-3,
            rtol=2e-3,
            msg=lambda message: f"{source}.lse: {message}",
        )
