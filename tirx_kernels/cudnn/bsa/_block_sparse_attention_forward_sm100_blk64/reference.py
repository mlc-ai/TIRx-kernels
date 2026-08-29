# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Lazy loader for the upstream SM100 blk64 BSA forward pass.

Upstream source: ``python/cudnn/block_sparse_attention/_interface.py``, from the
cuDNN Frontend source install pinned in ``reference-dependencies.json``.

The upstream convenience wrapper ``bsa_attn_fwd_blk64_cutedsl`` accepts only
equal Q and KV head counts: it asserts ``num_head == num_head_kv`` and pins
``qhead_per_kvhead = 1`` before constructing the kernel. The kernel class itself
takes the ratio as a constructor argument and already carries it in the compile
key, and its device program maps a Q head to its KV head in a single expression.
``_call_blk64_grouped`` therefore reaches the same class with the real ratio,
reproducing the wrapper's host work verbatim; equal head counts keep using the
upstream wrapper so that path stays byte-for-byte what upstream runs.
"""

import torch

from tirx_kernels.cudnn._reference import load_reference_module

_INTERFACE = "cudnn.block_sparse_attention._interface"

# Compiled grouped-head programs, keyed exactly like the upstream wrapper keys
# its own cache. Kept local so the upstream wrapper's cache stays untouched.
_GROUPED_COMPILE_CACHE = {}


def load_interface():
    return load_reference_module(_INTERFACE)


def _call_blk64_grouped(
    interface,
    q,
    k,
    v,
    q2k_block_index,
    block_sizes,
    q2k_block_nums=None,
    softmax_scale=None,
    layout="bhsd",
    block_sparse_num=0,
    allow_empty_block_nums=False,
    use_clc=None,
    kv_splits=1,
):
    """Run the upstream blk64 forward class with a grouped Q-to-KV head map.

    Mirrors ``_interface.bsa_attn_fwd_blk64_cutedsl`` host work; the head-count
    equality it asserts is relaxed to divisibility, the ratio it pins to 1 is
    computed, and the compiled programs live in a module-local cache.
    """
    assert q.dtype == torch.bfloat16, "blk64 CuTeDSL requires bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    auto_kv_splits = isinstance(kv_splits, str)
    if auto_kv_splits:
        assert kv_splits == "auto", "kv_splits string value must be 'auto'"
        kv_splits_i = 1
    else:
        kv_splits_i = int(kv_splits)
        assert kv_splits_i >= 1, "kv_splits must be >= 1"

    if layout == "bhsd":
        q_bhsd, k_bhsd, v_bhsd = [interface.maybe_contiguous(t) for t in (q, k, v)]
    else:
        assert layout == "bshd", f"layout must be 'bhsd' or 'bshd', got {layout!r}"
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()

    batch_size, num_head, seqlen_q, head_dim = q_bhsd.shape
    seqlen_k = k_bhsd.shape[2]
    num_head_kv = k_bhsd.shape[1]
    head_dim_v = v_bhsd.shape[-1]

    assert head_dim == 128 and head_dim_v == 128, "blk64 CuTeDSL requires D=DV=128"
    # The upstream wrapper asserts equality here; the kernel class only needs the
    # Q head count to be a whole multiple of the KV head count.
    assert num_head % num_head_kv == 0, "Q heads must be a multiple of KV heads"
    assert k_bhsd.shape == (batch_size, num_head_kv, seqlen_k, head_dim)
    assert v_bhsd.shape == (batch_size, num_head_kv, seqlen_k, head_dim_v)
    assert q2k_block_index.dtype == torch.int32
    q2k_block_index = interface.maybe_contiguous(q2k_block_index)
    has_block_sizes = block_sizes is not None and block_sizes.numel() > 0
    if has_block_sizes:
        block_sizes = interface.maybe_contiguous(block_sizes)
        assert block_sizes.dtype == torch.int32
    else:
        block_sizes = None
    num_q_blocks = (seqlen_q + 63) // 64
    has_variable_block_nums = q2k_block_nums is not None and q2k_block_nums.numel() > 0
    if has_variable_block_nums:
        q2k_block_nums = interface.maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32
        # Sparse metadata stays Q-head-sided under a grouped head map.
        assert q2k_block_nums.shape == (batch_size, num_head, num_q_blocks), (
            f"q2k_block_nums must be shaped (B, H, ceil(S_q/64)); got {tuple(q2k_block_nums.shape)}"
        )
        uniform_block_sparse_num = 0
    else:
        if block_sparse_num <= 0:
            block_sparse_num = int(q2k_block_index.shape[-1])
        assert q2k_block_index.shape[-1] >= block_sparse_num, (
            f"q2k_block_index last dim ({q2k_block_index.shape[-1]}) must be "
            f">= block_sparse_num ({block_sparse_num})"
        )
        uniform_block_sparse_num = int(block_sparse_num)
        q2k_block_nums = None

    interface._validate_sm100_blk64_int32_bounds(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        uniform_block_sparse_num,
        block_sizes,
        q2k_block_nums,
    )
    use_int64_kv_strides = interface._sm100_blk64_requires_int64_kv_strides(k_bhsd, v_bhsd)

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    dtype = interface.torch2cute_dtype_map[q_bhsd.dtype]
    arch = interface._get_device_arch()
    allow_empty_block_nums = has_variable_block_nums and allow_empty_block_nums
    sparse_block_size = 64
    # The wrapper pins this to 1; the kernel maps head -> head // ratio for K/V.
    qhead_per_kvhead = num_head // num_head_kv
    tile_m = 64
    tile_n = 256
    if auto_kv_splits:
        kv_splits_i = interface._sm100_blk64_auto_kv_splits(
            q_bhsd, q2k_block_index, uniform_block_sparse_num
        )
    kv_splits_i = interface._resolve_blk64_split_workspace(
        q_bhsd, head_dim_v, kv_splits_i, allow_fallback=auto_kv_splits
    )
    if use_clc is None:
        if kv_splits_i > 1:
            use_clc_scheduler = False
        else:
            use_clc_scheduler = interface.choose_blk64_cutedsl_use_clc(
                q_bhsd,
                uniform_block_sparse_num,
                q2k_block_nums if has_variable_block_nums else None,
                layout="bhsd",
            )
    else:
        use_clc_scheduler = bool(use_clc)
    assert not (kv_splits_i > 1 and use_clc_scheduler), (
        "blk64 CuTeDSL kv_splits>1 does not support use_clc=True"
    )
    is_persistent = use_clc_scheduler
    pack_gqa = False
    input_layout = "bhsd_native"

    split_offsets = None
    if kv_splits_i > 1:
        split_offsets = interface._build_sm100_blk64_kv_split_offsets(
            q2k_block_nums,
            uniform_block_sparse_num,
            batch_size,
            num_head,
            num_q_blocks,
            kv_splits_i,
            q_bhsd.device,
        )
        out_bhsd = torch.empty(
            (batch_size, kv_splits_i * num_head, seqlen_q, head_dim_v),
            dtype=torch.float32,
            device=q_bhsd.device,
        )
        lse = torch.empty(
            (batch_size, kv_splits_i * num_head, seqlen_q),
            dtype=torch.float32,
            device=q_bhsd.device,
        )
    else:
        out_bhsd = torch.empty(
            (batch_size, num_head, seqlen_q, head_dim_v), dtype=q_bhsd.dtype, device=q_bhsd.device
        )
        lse = torch.empty(
            (batch_size, num_head, seqlen_q), dtype=torch.float32, device=q_bhsd.device
        )

    current_stream = interface.cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = interface._dynamic_tensors_compile_key(
        "sm100_blk64_fwd",
        (
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            pack_gqa,
            tile_m,
            tile_n,
            sparse_block_size,
            arch,
            has_variable_block_nums,
            allow_empty_block_nums,
            has_block_sizes,
            kv_splits_i,
            out_bhsd.dtype,
            is_persistent,
            use_clc_scheduler,
            input_layout,
            use_int64_kv_strides,
        ),
        (
            q_bhsd,
            k_bhsd,
            v_bhsd,
            out_bhsd,
            lse,
            q2k_block_index,
            block_sizes,
            q2k_block_nums,
            split_offsets,
        ),
    )

    if compile_key not in _GROUPED_COMPILE_CACHE:
        cute = interface.cute
        q_tensor, k_tensor, v_tensor, o_tensor = [
            interface._to_cute_tensor(t) for t in (q_bhsd, k_bhsd, v_bhsd, out_bhsd)
        ]
        lse_tensor = interface._to_cute_tensor(lse, assumed_align=4)
        block_index_tensor = interface._to_cute_tensor(q2k_block_index)
        block_sizes_tensor = interface._to_cute_tensor(block_sizes) if has_block_sizes else None
        block_nums_tensor = (
            interface._to_cute_tensor(q2k_block_nums) if has_variable_block_nums else None
        )
        split_offsets_tensor = (
            interface._to_cute_tensor(split_offsets) if split_offsets is not None else None
        )

        bsa_fwd = interface.BlockSparseAttnForwardSm100Blk64(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            sparse_block_size=sparse_block_size,
            is_persistent=is_persistent,
            use_clc_scheduler=use_clc_scheduler,
            allow_empty_block_nums=allow_empty_block_nums,
            has_block_sizes=has_block_sizes,
            num_splits=kv_splits_i,
            use_int64_kv_strides=use_int64_kv_strides,
        )

        _GROUPED_COMPILE_CACHE[compile_key] = cute.compile(
            bsa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            block_index_tensor,
            block_sizes_tensor,
            uniform_block_sparse_num,
            block_nums_tensor,
            split_offsets_tensor,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    with torch.cuda.nvtx.range("bsa_attn_fwd_blk64_cutedsl_kernel"):
        _GROUPED_COMPILE_CACHE[compile_key](
            q_bhsd.detach(),
            k_bhsd.detach(),
            v_bhsd.detach(),
            out_bhsd.detach(),
            lse,
            softmax_scale,
            q2k_block_index.detach(),
            block_sizes.detach() if has_block_sizes else None,
            uniform_block_sparse_num,
            q2k_block_nums.detach() if has_variable_block_nums else None,
            split_offsets.detach() if split_offsets is not None else None,
            current_stream,
        )

    if kv_splits_i > 1:
        out_bhsd, lse = interface._combine_blk64_kv_bucketed_partials(
            q_bhsd, out_bhsd, lse, kv_splits_i
        )
        # Keep split_offsets alive through the combine launch on the same stream.
        _ = split_offsets

    out = out_bhsd if layout == "bhsd" else out_bhsd.transpose(1, 2).contiguous()
    return out, lse


def compile_reference(data):
    """Compile lazily and return the no-argument upstream launch closure."""
    interface = load_interface()
    config = data["config"]
    inputs = data["inputs"]
    # Read the head map off the BHSD buffers rather than the config so the
    # dispatch cannot drift from the tensors actually handed to the source.
    grouped = inputs["q"].shape[1] != inputs["k"].shape[1]
    source_batches = None
    if config["batch"] > 1:
        source_batches = tuple(
            (
                inputs["q_user"][batch_idx : batch_idx + 1].clone(),
                inputs["k_user"][batch_idx : batch_idx + 1].clone(),
                inputs["v_user"][batch_idx : batch_idx + 1].clone(),
                inputs["block_index"][batch_idx : batch_idx + 1].clone(),
                inputs["block_nums"][batch_idx : batch_idx + 1].clone(),
            )
            for batch_idx in range(config["batch"])
        )

    def call_source(q, k, v, block_index, block_nums):
        kwargs = dict(
            q2k_block_nums=(block_nums if config["block_count_mode"] != "fixed" else None),
            softmax_scale=inputs["softmax_scale"],
            layout=config["tensor_layout"],
            block_sparse_num=(0 if config["block_count_mode"] != "fixed" else config["kv_blocks"]),
            allow_empty_block_nums=config["block_count_mode"] == "variable_empty",
            use_clc=config["use_clc"],
            kv_splits=config["kv_splits"],
        )
        block_size_arg = inputs["block_sizes"] if config["has_block_sizes"] else None
        if grouped:
            return _call_blk64_grouped(interface, q, k, v, block_index, block_size_arg, **kwargs)
        return interface.bsa_attn_fwd_blk64_cutedsl(q, k, v, block_index, block_size_arg, **kwargs)

    def launch():
        # The pinned source's static scheduler launches only batch 0 when B>1.
        # Preserve the source device program and cover the full operation by
        # invoking that same specialization once per batch in the reference.
        if config["batch"] == 1:
            out, lse = call_source(
                inputs["q_user"],
                inputs["k_user"],
                inputs["v_user"],
                inputs["block_index"],
                inputs["block_nums"],
            )
        else:
            outputs = []
            lses = []
            for q, k, v, block_index, block_nums in source_batches:
                out_batch, lse_batch = call_source(q, k, v, block_index, block_nums)
                outputs.append(out_batch)
                lses.append(lse_batch)
            out = torch.cat(outputs, dim=0)
            lse = torch.cat(lses, dim=0)
        if config["tensor_layout"] == "bshd":
            out = out.transpose(1, 2).contiguous()
        data["source"]["out"] = out
        data["source"]["lse"] = lse

    return launch
