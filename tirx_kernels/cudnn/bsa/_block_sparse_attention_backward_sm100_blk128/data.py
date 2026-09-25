# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Inputs, direct launch wiring, and sparse analytic validation for BSA backward."""

import math

import torch

from tirx_kernels.cudnn._reference import load_reference_module

from . import spec

_REDZONE_ELEMENTS = 256
_REDZONE_F32 = 12345.25
_REDZONE_BF16 = -77.0
_LOG2_E = math.log2(math.e)
_TMA_SWIZZLE_128B = 3
_TMA_L2_256B = 2
_TOLERANCES = {
    "sum_odo": (2**-18, 2**-18),
    "scaled_lse": (2**-15, 2**-15),
    "dq_acc": (3e-2, 3e-2),
    "dk_acc": (2**-7, 2**-7),
    "dv_acc": (3e-2, 3e-2),
    "dq": (2**-9, 2**-9),
    "dk": (2**-9, 2**-9),
    "dv": (3e-2, 3e-2),
}


def _oracle_tolerance(config, name):
    if config["data_mode"] == "sharp_softmax":
        sharp = {
            "dq_acc": (3 / 16, 3 / 16),
            "dq": (2**-5, 2**-5),
            "dk": (2**-4, 2**-4),
            "dv": (2**-6, 2**-6),
        }
        if name in sharp:
            return sharp[name]
    return _TOLERANCES[name]


_SOURCE_PAIR_TOLERANCES = {"dq": (2**-10, 2**-7), "dk": (2**-10, 2**-7), "dv": (2**-10, 2**-7)}


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned 128-byte TensorMap payload."""

    def __init__(self):
        import ctypes

        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensormap(
    tensor, dtype, global_dims, global_strides, box_dims, *, swizzle=_TMA_SWIZZLE_128B
):
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
        0,
        swizzle,
        _TMA_L2_256B,
        0,
    )
    return descriptor


def _as_bhsd(tensor, config):
    """Return a logical BHSD view while preserving the user's physical ABI."""
    return tensor.transpose(1, 2) if config["tensor_layout"] == "bshd" else tensor


def _build_tensor_maps(inputs, output, config):
    """Encode rank-4 BF16 TensorMaps for the selected BHSD/BSHD ABI."""
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])

    seq_axis, head_axis = (1, 2) if config["tensor_layout"] == "bshd" else (2, 1)

    def tensor_map(tensor, seqlen, box_features):
        element_size = tensor.element_size()
        strides = tuple(
            int(tensor.stride(axis) * element_size) for axis in (seq_axis, head_axis, 0)
        )
        return _encode_tensormap(
            tensor,
            "bfloat16",
            (head_dim, seqlen, heads, batch),
            strides,
            (box_features, 128, 1, 1),
            swizzle=3 if box_features == 64 else 2,
        )

    return {
        "q": tensor_map(inputs["q"], seqlen_q, 64),
        "k": tensor_map(inputs["k"], seqlen_kv, 64),
        "v": tensor_map(inputs["v"], seqlen_kv, 64),
        "do": tensor_map(inputs["do"], seqlen_q, 64),
        "dv": tensor_map(output["dv"], seqlen_kv, head_dim // 2),
        "dk": tensor_map(output["dk"], seqlen_kv, head_dim // 2),
    }


def _roundup128(value):
    return (value + 127) // 128 * 128


def _bucket_size(config, q_blocks):
    explicit = config.get("bucket_size_blocks")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    if q_blocks >= 4096 and int(config["num_heads"]) <= 1:
        return 256
    return 512 if q_blocks >= 2048 else 384


def _pattern_values(config):
    maximum = int(config["kv_blocks"])
    mode = config["block_count_mode"]
    if mode == "fixed":
        return (maximum,)
    text = config.get("block_count_pattern")
    if text is None:
        return (0, 1, maximum) if mode == "variable_empty" else (1, max(1, maximum // 2), maximum)
    values = []
    for item in text.split(","):
        item = item.strip()
        if item == "max":
            values.append(maximum)
        elif item == "mid":
            values.append(max(1, maximum // 2))
        else:
            values.append(int(item))
    return tuple(values)


def _make_metadata(config, *, q_blocks, kv_blocks, device):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    maximum = int(config["kv_blocks"])
    if maximum > kv_blocks:
        raise ValueError("kv_blocks exceeds the physical KV block count")

    rows = batch * heads * q_blocks
    row_ids = torch.arange(rows, dtype=torch.int64, device=device)
    slots = torch.arange(maximum, dtype=torch.int64, device=device)
    block_index = (row_ids[:, None] * 7 + slots[None, :]) % kv_blocks
    if maximum and maximum < kv_blocks:
        contains_tail = (block_index == kv_blocks - 1).any(dim=1)
        block_index[~contains_tail, -1] = kv_blocks - 1
    if q_blocks == 3 and maximum == 3 and kv_blocks == 4:
        # c07 deliberately makes the four inverse-CSR task lengths 3/2/1/0.
        block_index[:] = torch.tensor((0, 1, 2), dtype=torch.int64, device=device)
    block_index = block_index.to(torch.int32).reshape(batch, heads, q_blocks, maximum)

    values = _pattern_values(config)
    if config["block_count_mode"] != "variable_empty" and 0 in values:
        raise ValueError("zero block counts require block_count_mode='variable_empty'")
    pattern_shift = (
        1
        if config["block_count_mode"] == "variable_empty" and any(value != 0 for value in values)
        else 0
    )
    counts = torch.tensor(values, dtype=torch.int32, device=device)[
        (row_ids + pattern_shift) % len(values)
    ]
    counts = counts.reshape(batch, heads, q_blocks).contiguous()
    if config["block_count_mode"] == "fixed":
        block_nums = None
        block_sparse_num = maximum
    else:
        block_nums = counts
        block_sparse_num = 0
    return block_index.contiguous(), counts, block_nums, block_sparse_num


def _strided_bhsd(generator, shape, *, use_i64):
    batch, heads, seqlen, dim = shape
    base = torch.randn(
        batch * heads * seqlen * dim, generator=generator, dtype=torch.bfloat16, device="cuda"
    )
    base.mul_(0.125)
    if not use_i64:
        return base.reshape(shape)
    if batch != 1 or heads != 1:
        raise ValueError("the explicit Int64 KV-stride probe requires singleton batch and head")
    # The pinned CuTeDSL source marks batch/head/sequence compact-dynamic and
    # therefore rejects an artificial >INT32 stride on either singleton mode.
    # Keep this case source-comparable; the flag still selects the special
    # large-index configs, while every target byte address is formed in Int64.
    return base.reshape(shape)


def _guarded_tensor(shape, dtype, fill):
    count = math.prod(shape)
    storage = torch.full((count + 2 * _REDZONE_ELEMENTS,), fill, dtype=dtype, device="cuda")
    tensor = storage[_REDZONE_ELEMENTS : _REDZONE_ELEMENTS + count].reshape(shape)
    return tensor, storage


def _workspace_layout(config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    q128 = _roundup128(int(config["seqlen_q"]))
    k128 = _roundup128(int(config["seqlen_kv"]))
    n = batch * heads
    sizes = {
        "sum_odo": n * q128,
        "scaled_lse": n * q128,
        "dq_acc": n * q128 * head_dim,
        "dk_acc": n * k128 * head_dim,
        "dv_acc": n * k128 * head_dim,
    }
    offsets = {}
    cursor = 0
    for name in ("sum_odo", "scaled_lse", "dq_acc", "dk_acc", "dv_acc"):
        offsets[name] = cursor
        cursor += sizes[name]
    return q128, k128, sizes, offsets, cursor


def _new_mutable_state(config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    q128, k128, sizes, offsets, total = _workspace_layout(config)
    workspace, workspace_storage = _guarded_tensor((total,), torch.float32, _REDZONE_F32)
    workspace.zero_()
    q_shape = (
        (batch, seqlen_q, heads, head_dim)
        if config["tensor_layout"] == "bshd"
        else (batch, heads, seqlen_q, head_dim)
    )
    kv_shape = (
        (batch, seqlen_kv, heads, head_dim)
        if config["tensor_layout"] == "bshd"
        else (batch, heads, seqlen_kv, head_dim)
    )
    dq, dq_storage = _guarded_tensor(q_shape, torch.bfloat16, _REDZONE_BF16)
    dk, dk_storage = _guarded_tensor(kv_shape, torch.bfloat16, _REDZONE_BF16)
    dv, dv_storage = _guarded_tensor(kv_shape, torch.bfloat16, _REDZONE_BF16)
    dq.fill_(float("nan"))
    dk.zero_()
    dv.zero_()
    return {
        "workspace": workspace,
        "workspace_storage": workspace_storage,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dq_storage": dq_storage,
        "dk_storage": dk_storage,
        "dv_storage": dv_storage,
        "workspace_sizes": sizes,
        "workspace_offsets": offsets,
        "q8": q128,
        "k8": k128,
    }


def _workspace_fields(state, config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    q128 = state["q8"]
    k128 = state["k8"]
    workspace = state["workspace"]
    offsets = state["workspace_offsets"]
    sizes = state["workspace_sizes"]

    def field(name, shape):
        start = offsets[name]
        return workspace[start : start + sizes[name]].reshape(shape)

    def striped(name, padded):
        physical = field(name, (batch, heads, padded // 128, head_dim // 4, 128, 4))
        return physical.permute(0, 1, 2, 4, 3, 5).reshape(batch, heads, padded, head_dim)

    return {
        "sum_odo": field("sum_odo", (batch, heads, q128)),
        "scaled_lse": field("scaled_lse", (batch, heads, q128)),
        "dq_acc": striped("dq_acc", q128),
        "dv_acc": striped("dv_acc", k128),
        "dk_acc": striped("dk_acc", k128),
    }


def prepare_data(**config):
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("CUDA is required")
    spec.validate_config(config)
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    q_blocks = (seqlen_q + 127) // 128
    physical_kv_blocks = (seqlen_kv + 127) // 128
    bucket_size = _bucket_size(config, q_blocks)
    generator = torch.Generator(device="cuda").manual_seed(int(config["seed"]))

    amplitude = 0.125
    if config["data_mode"] == "cancellation":
        amplitude = 0.5
    elif config["data_mode"] == "sharp_softmax":
        amplitude = 1.5

    def random_bhsd(shape):
        value = torch.randn(*shape, generator=generator, dtype=torch.bfloat16, device="cuda")
        value.mul_(amplitude)
        return value

    q = random_bhsd((batch, heads, seqlen_q, head_dim))
    k = _strided_bhsd(
        generator, (batch, heads, seqlen_kv, head_dim), use_i64=bool(config["use_int64_kv_strides"])
    )
    v = _strided_bhsd(
        generator, (batch, heads, seqlen_kv, head_dim), use_i64=bool(config["use_int64_kv_strides"])
    )
    k.mul_(amplitude / 0.125)
    v.mul_(amplitude / 0.125)
    do = random_bhsd((batch, heads, seqlen_q, head_dim))
    if config["data_mode"] == "cancellation":
        signs = torch.where(
            torch.arange(head_dim, device="cuda") % 2 == 0,
            torch.tensor(1.0, device="cuda"),
            torch.tensor(-1.0, device="cuda"),
        ).to(torch.bfloat16)
        do.mul_(signs)

    block_index, counts, block_nums, block_sparse_num = _make_metadata(
        config, q_blocks=q_blocks, kv_blocks=physical_kv_blocks, device="cuda"
    )
    softmax_scale = (
        1.0 / math.sqrt(head_dim)
        if config["softmax_scale"] is None
        else float(config["softmax_scale"])
    )
    interface = load_reference_module("cudnn.block_sparse_attention._interface")
    use_bshd = config["tensor_layout"] == "bshd"
    q_forward = q.transpose(1, 2).contiguous() if use_bshd else q
    k_forward = k.transpose(1, 2).contiguous() if use_bshd else k
    v_forward = v.transpose(1, 2).contiguous() if use_bshd else v
    do_forward = do.transpose(1, 2).contiguous() if use_bshd else do
    o_user, lse = interface.bsa_attn_fwd(
        q_forward,
        k_forward,
        v_forward,
        block_index,
        block_sparse_num,
        block_sizes=None,
        q2k_block_nums=block_nums,
        allow_empty_block_nums=config["block_count_mode"] == "variable_empty",
        softmax_scale=softmax_scale,
        pack_gqa=False,
        return_lse=True,
        layout=config["tensor_layout"],
        kv_splits=1,
    )
    bucketed_offsets, bucketed_indices, num_q_groups, max_rows = interface._build_bucketed_k2q_csr(
        block_index,
        block_sparse_num,
        physical_kv_blocks,
        bucket_size_blocks=bucket_size,
        q2k_block_nums=block_nums,
    )
    torch.cuda.synchronize()

    tirx = _new_mutable_state(config)
    source = _new_mutable_state(config)
    inputs = {
        "q": q_forward,
        "k": k_forward,
        "v": v_forward,
        "do": do_forward,
        "o": o_user,
        "lse": lse,
        "block_index": block_index,
        "counts": counts,
        "block_nums": block_nums,
        "block_sparse_num": block_sparse_num,
        "bucketed_offsets": bucketed_offsets,
        "bucketed_indices": bucketed_indices,
        "softmax_scale": softmax_scale,
    }
    return {
        "config": dict(config),
        "inputs": inputs,
        "derived": {
            "q_blocks": q_blocks,
            "kv_blocks": physical_kv_blocks,
            "bucket_size_blocks": bucket_size,
            "num_q_groups": int(num_q_groups),
            "max_k2q_rows_per_group": int(max_rows),
            "tasks_per_group": physical_kv_blocks,
        },
        "tirx": tirx,
        "source": source,
        "oracle": None,
    }


def reset_mutable_state(state):
    state["workspace"].zero_()
    state["dq"].fill_(float("nan"))
    state["dk"].zero_()
    state["dv"].zero_()


def tirx_launch(executables, data):
    inputs = data["inputs"]
    output = data["tirx"]
    config = data["config"]
    maps = _build_tensor_maps(inputs, output, config)
    edge_stride = int(inputs["bucketed_indices"].shape[-1])
    direct_dkv = int(data["derived"]["num_q_groups"]) == 1

    def launch():
        if not direct_dkv:
            # Multi-group dK/dV use global FP32 bulk reductions. Host-side zeroing is
            # part of the source launch contract and keeps every closure call
            # deterministic, including repeated benchmark invocations.
            output["workspace"][output["workspace_offsets"]["dk_acc"] :].zero_()
        executables[0](
            inputs["o"].reshape(-1),
            inputs["do"].reshape(-1),
            inputs["lse"].reshape(-1),
            output["workspace"],
        )
        executables[1](
            maps["q"].ptr,
            maps["k"].ptr,
            maps["v"].ptr,
            maps["do"].ptr,
            maps["dv"].ptr,
            maps["dk"].ptr,
            output["dk"].reshape(-1),
            output["dv"].reshape(-1),
            inputs["bucketed_offsets"].reshape(-1),
            inputs["bucketed_indices"].reshape(-1),
            output["workspace"],
            edge_stride,
            inputs["softmax_scale"],
        )
        executables[2](output["workspace"], output["dq"].reshape(-1), inputs["softmax_scale"])
        if not direct_dkv:
            executables[3](output["workspace"], output["dk"].reshape(-1), inputs["softmax_scale"])
            executables[4](output["workspace"], output["dv"].reshape(-1), 1.0)

    launch.tensor_maps = maps
    return launch


def _analytic_oracle(data, row_chunk=32):
    if data["oracle"] is not None:
        return data["oracle"]
    config = data["config"]
    inputs = data["inputs"]
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    q_blocks = data["derived"]["q_blocks"]
    maximum = int(config["kv_blocks"])
    scale = float(inputs["softmax_scale"])
    device = inputs["q"].device

    q64 = _as_bhsd(inputs["q"], config).to(torch.float64)
    k64 = _as_bhsd(inputs["k"], config).to(torch.float64)
    v64 = _as_bhsd(inputs["v"], config).to(torch.float64)
    do64 = _as_bhsd(inputs["do"], config).to(torch.float64)
    o64 = _as_bhsd(inputs["o"], config).to(torch.float64)
    dq = torch.zeros_like(q64)
    dk = torch.zeros_like(k64)
    dv = torch.zeros_like(v64)
    sum_odo = (o64 * do64).sum(dim=-1)
    lse64 = inputs["lse"].to(torch.float64)
    scaled_lse = torch.where(torch.isneginf(lse64), torch.zeros_like(lse64), lse64 * _LOG2_E)

    total_rows = batch * heads * q_blocks
    row_ids_all = torch.arange(total_rows, dtype=torch.int64, device=device)
    token_offsets = torch.arange(128, dtype=torch.int64, device=device)
    q_offsets = torch.arange(128, dtype=torch.int64, device=device)
    for start in range(0, total_rows, row_chunk):
        row_ids = row_ids_all[start : start + row_chunk]
        bh = row_ids // q_blocks
        qb = row_ids - bh * q_blocks
        b = bh // heads
        h = bh - b * heads
        count = inputs["counts"].reshape(-1)[row_ids].to(torch.int64)
        sparse_blocks = inputs["block_index"].reshape(total_rows, maximum)[row_ids].to(torch.int64)
        kv_tokens = sparse_blocks[:, :, None] * 128 + token_offsets[None, None, :]
        slot_valid = torch.arange(maximum, device=device)[None, :] < count[:, None]
        kv_valid = slot_valid[:, :, None] & (kv_tokens < seqlen_kv)
        kv_safe = kv_tokens.clamp(0, seqlen_kv - 1)
        k_sel = k64[b[:, None, None], h[:, None, None], kv_safe].reshape(
            row_ids.numel(), maximum * 128, head_dim
        )
        v_sel = v64[b[:, None, None], h[:, None, None], kv_safe].reshape(
            row_ids.numel(), maximum * 128, head_dim
        )
        kv_valid = kv_valid.reshape(row_ids.numel(), maximum * 128)

        q_tokens = qb[:, None] * 128 + q_offsets[None, :]
        q_valid = q_tokens < seqlen_q
        q_safe = q_tokens.clamp(0, seqlen_q - 1)
        q_sel = q64[b[:, None], h[:, None], q_safe]
        do_sel = do64[b[:, None], h[:, None], q_safe]
        scores = torch.bmm(q_sel, k_sel.transpose(1, 2)) * scale
        scores = scores.masked_fill(~kv_valid[:, None, :], float("-inf"))
        all_empty = ~kv_valid.any(dim=1)
        probs = torch.softmax(scores, dim=-1)
        if bool(all_empty.any()):
            probs[all_empty] = 0.0
        probs = torch.where(q_valid[:, :, None], probs, torch.zeros_like(probs))
        dp = torch.bmm(do_sel, v_sel.transpose(1, 2))
        delta = sum_odo[b[:, None], h[:, None], q_safe]
        ds = probs * (dp - delta[:, :, None])
        ds = torch.where(all_empty[:, None, None], torch.zeros_like(ds), ds)
        dq_sel = torch.bmm(ds, k_sel) * scale
        dk_sel = torch.bmm(ds.transpose(1, 2), q_sel) * scale
        dv_sel = torch.bmm(probs.transpose(1, 2), do_sel)

        global_q = ((b[:, None] * heads + h[:, None]) * seqlen_q + q_tokens).reshape(-1)
        q_valid_flat = q_valid.reshape(-1)
        dq.reshape(-1, head_dim)[global_q[q_valid_flat]] = dq_sel.reshape(-1, head_dim)[
            q_valid_flat
        ]
        global_kv = ((b[:, None, None] * heads + h[:, None, None]) * seqlen_kv + kv_safe).reshape(
            -1
        )
        valid_flat = kv_valid.reshape(-1)
        dk.reshape(-1, head_dim).index_add_(
            0, global_kv[valid_flat], dk_sel.reshape(-1, head_dim)[valid_flat]
        )
        dv.reshape(-1, head_dim).index_add_(
            0, global_kv[valid_flat], dv_sel.reshape(-1, head_dim)[valid_flat]
        )

    q128 = data["source"]["q8"]
    k128 = data["source"]["k8"]
    oracle = {
        "sum_odo": torch.zeros((batch, heads, q128), dtype=torch.float32, device=device),
        "scaled_lse": torch.zeros((batch, heads, q128), dtype=torch.float32, device=device),
        "dq_acc": torch.zeros((batch, heads, q128, head_dim), dtype=torch.float32, device=device),
        "dk_acc": torch.zeros((batch, heads, k128, head_dim), dtype=torch.float32, device=device),
        "dv_acc": torch.zeros((batch, heads, k128, head_dim), dtype=torch.float32, device=device),
        "dq": dq.to(torch.bfloat16),
        "dk": dk.to(torch.bfloat16),
        "dv": dv.to(torch.bfloat16),
    }
    oracle["sum_odo"][:, :, :seqlen_q] = sum_odo.to(torch.float32)
    oracle["scaled_lse"][:, :, :seqlen_q] = scaled_lse.to(torch.float32)
    oracle["dq_acc"][:, :, :seqlen_q] = (dq / scale).to(torch.float32)
    oracle["dk_acc"][:, :, :seqlen_kv] = (dk / scale).to(torch.float32)
    oracle["dv_acc"][:, :, :seqlen_kv] = dv.to(torch.float32)
    data["oracle"] = oracle
    return oracle


def _assert_same_classification(actual, expected, name):
    if not torch.equal(torch.isnan(actual), torch.isnan(expected)):
        raise AssertionError(f"{name}: NaN classification mismatch")
    if not torch.equal(torch.isposinf(actual), torch.isposinf(expected)):
        raise AssertionError(f"{name}: +Inf classification mismatch")
    if not torch.equal(torch.isneginf(actual), torch.isneginf(expected)):
        raise AssertionError(f"{name}: -Inf classification mismatch")


def _assert_close(actual, expected, name, tolerance=None):
    _assert_same_classification(actual, expected, name)
    finite = torch.isfinite(expected)
    if not bool(finite.any()):
        return
    atol, rtol = _TOLERANCES[name] if tolerance is None else tolerance
    torch.testing.assert_close(
        actual[finite].float(), expected[finite].float(), atol=atol, rtol=rtol, equal_nan=False
    )


def _validate_redzones(state, source_name):
    for storage_name, value in (
        ("workspace_storage", _REDZONE_F32),
        ("dq_storage", _REDZONE_BF16),
        ("dk_storage", _REDZONE_BF16),
        ("dv_storage", _REDZONE_BF16),
    ):
        storage = state[storage_name]
        expected = torch.tensor(value, dtype=storage.dtype, device=storage.device)
        if not torch.all(storage[:_REDZONE_ELEMENTS] == expected):
            raise AssertionError(f"{source_name}.{storage_name}: leading redzone modified")
        if not torch.all(storage[-_REDZONE_ELEMENTS:] == expected):
            raise AssertionError(f"{source_name}.{storage_name}: trailing redzone modified")


def validate_outputs(data, *, sources, with_oracle=True, tolerance_overrides=None):
    oracle = _analytic_oracle(data) if with_oracle else None
    config = data["config"]
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    multi_group = spec.q_group_count(config) > 1
    for source_name in sources:
        state = data[source_name]
        _validate_redzones(state, source_name)
        if source_name == "tirx":
            fields = _workspace_fields(state, config)
            if not torch.all(fields["sum_odo"][:, :, seqlen_q:] == 0):
                raise AssertionError("tirx.sum_odo padding is not exactly zero")
            if not torch.all(fields["scaled_lse"][:, :, seqlen_q:] == 0):
                raise AssertionError("tirx.scaled_lse padding is not exactly zero")
            if multi_group:
                if not torch.all(fields["dk_acc"][:, :, seqlen_kv:] == 0):
                    raise AssertionError("tirx.dk_acc padding is not exactly zero")
                if not torch.all(fields["dv_acc"][:, :, seqlen_kv:] == 0):
                    raise AssertionError("tirx.dv_acc padding is not exactly zero")
            if oracle is not None:
                names = ["sum_odo", "scaled_lse", "dq_acc"]
                if multi_group:
                    names.extend(("dk_acc", "dv_acc"))
                for name in names:
                    tolerance = (
                        _oracle_tolerance(config, name)
                        if tolerance_overrides is None
                        else tolerance_overrides.get(name, _oracle_tolerance(config, name))
                    )
                    actual = fields[name]
                    expected = oracle[name]
                    if name == "dq_acc":
                        # Source bulk-reduces complete 128-row tiles; only the
                        # valid SQ rows are semantic, and postprocess predicates
                        # its final BF16 stores for the q tail.
                        actual = actual[:, :, :seqlen_q]
                        expected = expected[:, :, :seqlen_q]
                    _assert_close(actual, expected, name, tolerance)
        for name in ("dq", "dk", "dv"):
            actual = _as_bhsd(state[name], config)
            if oracle is None:
                if not bool(torch.isfinite(actual).all()):
                    raise AssertionError(f"{source_name}.{name} contains non-finite values")
                continue
            tolerance = (
                _oracle_tolerance(config, name)
                if tolerance_overrides is None
                else tolerance_overrides.get(name, _oracle_tolerance(config, name))
            )
            _assert_close(actual, oracle[name], name, tolerance)
    if oracle is not None and "tirx" in sources and "source" in sources:
        for name in ("dq", "dk", "dv"):
            _assert_close(
                _as_bhsd(data["tirx"][name], config),
                _as_bhsd(data["source"][name], config),
                name,
                _SOURCE_PAIR_TOLERANCES[name],
            )


def workspace_fields(data, source_name):
    return _workspace_fields(data[source_name], data["config"])


def tolerance_grid():
    return (
        (0.0, 0.0),
        *((value, value) for value in (2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 1e-2, 2e-2, 3e-2)),
    )


CONFIGS = spec.correctness_configs()
