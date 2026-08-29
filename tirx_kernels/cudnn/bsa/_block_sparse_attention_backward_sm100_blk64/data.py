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
    "sum_odo": (2**-11, 2**-11),
    "scaled_lse": (2**-15, 2**-15),
    "dq_acc": (3e-2, 3e-2),
    "dk_acc": (2**-7, 2**-7),
    "dv_acc": (3e-2, 3e-2),
    "dq": (2**-9, 2**-9),
    "dk": (2**-9, 2**-9),
    "dv": (3e-2, 3e-2),
}


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned 128-byte TensorMap payload."""

    def __init__(self):
        import ctypes

        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensormap(tensor, dtype, global_dims, global_strides, box_dims):
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
        _TMA_SWIZZLE_128B,
        _TMA_L2_256B,
        0,
    )
    return descriptor


def _build_tensor_maps(inputs, output, config):
    """Encode Q/K/V/dO loads and the FP32 dQ reduce-add destination."""
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])

    def load_map(tensor, seqlen):
        element_size = tensor.element_size()
        strides = tuple(int(tensor.stride(axis) * element_size) for axis in (2, 1, 0))
        return _encode_tensormap(
            tensor, "bfloat16", (128, seqlen, heads, batch), strides, (64, 64, 1, 1)
        )

    fields = _workspace_fields(output, config)
    dq_acc = fields["dq_acc"]
    dq_strides = tuple(int(dq_acc.stride(axis) * 4) for axis in (2, 1, 0))
    return {
        "q": load_map(inputs["q"], seqlen_q),
        "k": load_map(inputs["k"], seqlen_kv),
        "v": load_map(inputs["v"], seqlen_kv),
        "do": load_map(inputs["do"], seqlen_q),
        "dq": _encode_tensormap(
            dq_acc, "float32", (128, output["q8"], heads, batch), dq_strides, (32, 64, 1, 1)
        ),
    }


def _roundup8(value):
    return (value + 7) // 8 * 8


def _bucket_size(config, q_blocks):
    explicit = config.get("bucket_size_blocks")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    if q_blocks >= 3000:
        return 1024
    return 1088 if q_blocks < 2048 else 1152


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
    return torch.as_strided(base, shape, (2**31, seqlen * dim, dim, 1))


def _guarded_tensor(shape, dtype, fill):
    count = math.prod(shape)
    storage = torch.full((count + 2 * _REDZONE_ELEMENTS,), fill, dtype=dtype, device="cuda")
    tensor = storage[_REDZONE_ELEMENTS : _REDZONE_ELEMENTS + count].reshape(shape)
    return tensor, storage


def _workspace_layout(config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    q8 = _roundup8(int(config["seqlen_q"]))
    k8 = _roundup8(int(config["seqlen_kv"]))
    n = batch * heads
    sizes = {
        "sum_odo": n * q8,
        "scaled_lse": n * q8,
        "dq_acc": n * q8 * 128,
        "dk_acc": n * k8 * 128,
        "dv_acc": n * k8 * 128,
    }
    offsets = {}
    cursor = 0
    for name in ("sum_odo", "scaled_lse", "dq_acc", "dk_acc", "dv_acc"):
        offsets[name] = cursor
        cursor += sizes[name]
    return q8, k8, sizes, offsets, cursor


def _new_mutable_state(config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    q8, k8, sizes, offsets, total = _workspace_layout(config)
    workspace, workspace_storage = _guarded_tensor((total,), torch.float32, _REDZONE_F32)
    workspace.zero_()
    dq, dq_storage = _guarded_tensor((batch, heads, seqlen_q, 128), torch.bfloat16, _REDZONE_BF16)
    dk, dk_storage = _guarded_tensor((batch, heads, seqlen_kv, 128), torch.bfloat16, _REDZONE_BF16)
    dv, dv_storage = _guarded_tensor((batch, heads, seqlen_kv, 128), torch.bfloat16, _REDZONE_BF16)
    dq.fill_(float("nan"))
    dk.fill_(float("nan"))
    dv.fill_(float("nan"))
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
        "q8": q8,
        "k8": k8,
    }


def _workspace_fields(state, config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    q8 = state["q8"]
    k8 = state["k8"]
    workspace = state["workspace"]
    offsets = state["workspace_offsets"]
    sizes = state["workspace_sizes"]

    def field(name, shape):
        start = offsets[name]
        return workspace[start : start + sizes[name]].reshape(shape)

    return {
        "sum_odo": field("sum_odo", (batch, heads, q8)),
        "scaled_lse": field("scaled_lse", (batch, heads, q8)),
        "dq_acc": field("dq_acc", (batch, heads, q8, 128)),
        "dk_acc": field("dk_acc", (batch, heads, k8, 128)),
        "dv_acc": field("dv_acc", (batch, heads, k8, 128)),
    }


def prepare_data(**config):
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("CUDA is required")
    if config["dtype"] != "bfloat16" or int(config["head_dim"]) != 128:
        raise ValueError("SM100 blk64 backward supports BF16 D=128 only")

    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    q_blocks = (seqlen_q + 63) // 64
    physical_kv_blocks = (seqlen_kv + 63) // 64
    bucket_size = _bucket_size(config, q_blocks)
    generator = torch.Generator(device="cuda").manual_seed(int(config["seed"]))

    q = torch.randn(
        batch, heads, seqlen_q, 128, generator=generator, dtype=torch.bfloat16, device="cuda"
    )
    q.mul_(0.125)
    k = _strided_bhsd(
        generator, (batch, heads, seqlen_kv, 128), use_i64=bool(config["use_int64_kv_strides"])
    )
    v = _strided_bhsd(
        generator, (batch, heads, seqlen_kv, 128), use_i64=bool(config["use_int64_kv_strides"])
    )
    do = torch.randn(
        batch, heads, seqlen_q, 128, generator=generator, dtype=torch.bfloat16, device="cuda"
    )
    do.mul_(0.125)
    block_index, counts, block_nums, block_sparse_num = _make_metadata(
        config, q_blocks=q_blocks, kv_blocks=physical_kv_blocks, device="cuda"
    )
    block_sizes = torch.full((batch, physical_kv_blocks), 64, dtype=torch.int32, device="cuda")
    block_sizes[:, -1] = seqlen_kv - (physical_kv_blocks - 1) * 64
    if batch > 1 and seqlen_q == 129 and seqlen_kv == 511:
        block_sizes[1, -1] = max(1, int(block_sizes[1, -1]) - 16)

    softmax_scale = (
        1.0 / math.sqrt(128) if config["softmax_scale"] is None else float(config["softmax_scale"])
    )
    interface = load_reference_module("cudnn.block_sparse_attention._interface")
    forward_block_sizes = block_sizes if config["has_block_sizes"] else None
    forward_block_sizes_arg = (
        forward_block_sizes[0]
        if forward_block_sizes is not None and batch == 1
        else forward_block_sizes
    )
    use_bshd = config["tensor_layout"] == "bshd"
    q_forward = q.transpose(1, 2).contiguous() if use_bshd else q
    k_forward = k.transpose(1, 2).contiguous() if use_bshd else k
    v_forward = v.transpose(1, 2).contiguous() if use_bshd else v
    o_user, lse = interface.bsa_attn_fwd_blk64_cutedsl(
        q_forward,
        k_forward,
        v_forward,
        block_index,
        forward_block_sizes_arg,
        q2k_block_nums=block_nums,
        softmax_scale=softmax_scale,
        layout=config["tensor_layout"],
        block_sparse_num=block_sparse_num,
        allow_empty_block_nums=config["block_count_mode"] == "variable_empty",
        use_clc=False,
        kv_splits=1,
    )
    o = o_user.transpose(1, 2).contiguous() if use_bshd else o_user
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
        "q": q,
        "k": k,
        "v": v,
        "do": do,
        "o": o,
        "lse": lse,
        "block_index": block_index,
        "counts": counts,
        "block_nums": block_nums,
        "block_sparse_num": block_sparse_num,
        "block_sizes": block_sizes,
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
    state["dk"].fill_(float("nan"))
    state["dv"].fill_(float("nan"))


def tirx_launch(executables, data):
    inputs = data["inputs"]
    output = data["tirx"]
    maps = _build_tensor_maps(inputs, output, data["config"])
    edge_stride = int(inputs["bucketed_indices"].shape[-1])

    def launch():
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
            maps["dq"].ptr,
            inputs["bucketed_offsets"].reshape(-1),
            inputs["bucketed_indices"].reshape(-1),
            inputs["block_sizes"].reshape(-1),
            output["workspace"],
            edge_stride,
            inputs["softmax_scale"],
        )
        executables[2](
            output["workspace"],
            output["dq"].reshape(-1),
            output["dk"].reshape(-1),
            output["dv"].reshape(-1),
            inputs["softmax_scale"],
        )

    return launch


def _analytic_oracle(data, row_chunk=32):
    if data["oracle"] is not None:
        return data["oracle"]
    config = data["config"]
    inputs = data["inputs"]
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    q_blocks = data["derived"]["q_blocks"]
    maximum = int(config["kv_blocks"])
    scale = float(inputs["softmax_scale"])
    device = inputs["q"].device

    q64 = inputs["q"].to(torch.float64)
    k64 = inputs["k"].to(torch.float64)
    v64 = inputs["v"].to(torch.float64)
    do64 = inputs["do"].to(torch.float64)
    o64 = inputs["o"].to(torch.float64)
    dq = torch.zeros_like(q64)
    dk = torch.zeros_like(k64)
    dv = torch.zeros_like(v64)
    sum_odo = -(o64 * do64).sum(dim=-1)
    scaled_lse = inputs["lse"].to(torch.float64) * (-_LOG2_E)

    total_rows = batch * heads * q_blocks
    row_ids_all = torch.arange(total_rows, dtype=torch.int64, device=device)
    token_offsets = torch.arange(64, dtype=torch.int64, device=device)
    q_offsets = torch.arange(64, dtype=torch.int64, device=device)
    for start in range(0, total_rows, row_chunk):
        row_ids = row_ids_all[start : start + row_chunk]
        bh = row_ids // q_blocks
        qb = row_ids - bh * q_blocks
        b = bh // heads
        h = bh - b * heads
        count = inputs["counts"].reshape(-1)[row_ids].to(torch.int64)
        sparse_blocks = inputs["block_index"].reshape(total_rows, maximum)[row_ids].to(torch.int64)
        kv_tokens = sparse_blocks[:, :, None] * 64 + token_offsets[None, None, :]
        slot_valid = torch.arange(maximum, device=device)[None, :] < count[:, None]
        block_size = inputs["block_sizes"][b[:, None], sparse_blocks]
        token_valid = token_offsets[None, None, :] < block_size[:, :, None]
        kv_valid = slot_valid[:, :, None] & token_valid & (kv_tokens < seqlen_kv)
        kv_safe = kv_tokens.clamp(0, seqlen_kv - 1)
        k_sel = k64[b[:, None, None], h[:, None, None], kv_safe].reshape(
            row_ids.numel(), maximum * 64, 128
        )
        v_sel = v64[b[:, None, None], h[:, None, None], kv_safe].reshape(
            row_ids.numel(), maximum * 64, 128
        )
        kv_valid = kv_valid.reshape(row_ids.numel(), maximum * 64)

        q_tokens = qb[:, None] * 64 + q_offsets[None, :]
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
        delta = (o64[b[:, None], h[:, None], q_safe] * do_sel).sum(dim=-1)
        ds = probs * (dp - delta[:, :, None])
        ds = torch.where(all_empty[:, None, None], torch.zeros_like(ds), ds)
        dq_sel = torch.bmm(ds, k_sel) * scale
        dk_sel = torch.bmm(ds.transpose(1, 2), q_sel) * scale
        dv_sel = torch.bmm(probs.transpose(1, 2), do_sel)
        global_q = ((b[:, None] * heads + h[:, None]) * seqlen_q + q_tokens).reshape(-1)
        q_valid_flat = q_valid.reshape(-1)
        dq.reshape(-1, 128)[global_q[q_valid_flat]] = dq_sel.reshape(-1, 128)[q_valid_flat]

        global_kv = ((b[:, None, None] * heads + h[:, None, None]) * seqlen_kv + kv_safe).reshape(
            -1
        )
        valid_flat = kv_valid.reshape(-1)
        dk.reshape(-1, 128).index_add_(
            0, global_kv[valid_flat], dk_sel.reshape(-1, 128)[valid_flat]
        )
        dv.reshape(-1, 128).index_add_(
            0, global_kv[valid_flat], dv_sel.reshape(-1, 128)[valid_flat]
        )

    q8 = data["source"]["q8"]
    k8 = data["source"]["k8"]
    oracle = {
        "sum_odo": torch.zeros((batch, heads, q8), dtype=torch.float32, device=device),
        "scaled_lse": torch.zeros((batch, heads, q8), dtype=torch.float32, device=device),
        "dq_acc": torch.zeros((batch, heads, q8, 128), dtype=torch.float32, device=device),
        "dk_acc": torch.zeros((batch, heads, k8, 128), dtype=torch.float32, device=device),
        "dv_acc": torch.zeros((batch, heads, k8, 128), dtype=torch.float32, device=device),
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
    for source_name in sources:
        state = data[source_name]
        fields = _workspace_fields(state, config)
        _validate_redzones(state, source_name)
        if not torch.all(fields["sum_odo"][:, :, seqlen_q:] == 0):
            raise AssertionError(f"{source_name}.sum_odo padding is not exactly zero")
        if not torch.all(fields["scaled_lse"][:, :, seqlen_q:] == 0):
            raise AssertionError(f"{source_name}.scaled_lse padding is not exactly zero")
        if not torch.all(fields["dq_acc"][:, :, seqlen_q:] == 0):
            raise AssertionError(f"{source_name}.dq_acc padding is not exactly zero")
        if not torch.all(fields["dk_acc"][:, :, seqlen_kv:] == 0):
            raise AssertionError(f"{source_name}.dk_acc padding is not exactly zero")
        if not torch.all(fields["dv_acc"][:, :, seqlen_kv:] == 0):
            raise AssertionError(f"{source_name}.dv_acc padding is not exactly zero")
        if oracle is None:
            continue
        for name in ("sum_odo", "scaled_lse", "dq_acc", "dk_acc", "dv_acc"):
            tolerance = None if tolerance_overrides is None else tolerance_overrides.get(name)
            _assert_close(fields[name], oracle[name], name, tolerance)
        for name in ("dq", "dk", "dv"):
            tolerance = None if tolerance_overrides is None else tolerance_overrides.get(name)
            _assert_close(state[name], oracle[name], name, tolerance)


def workspace_fields(data, source_name):
    return _workspace_fields(data[source_name], data["config"])


def tolerance_grid():
    return (
        (0.0, 0.0),
        *((value, value) for value in (2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 1e-2, 2e-2, 3e-2)),
    )


CONFIGS = spec.correctness_configs()
