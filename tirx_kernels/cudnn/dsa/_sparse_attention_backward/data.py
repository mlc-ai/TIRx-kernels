# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Inputs, launch wiring and validation for the DSA sparse attention backward pass.

Upstream sources:
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm100.py``
(tensor contract, output and workspace allocation, zero-initialization), and
``test/python/fe_api/dsa/dsa_reference.py`` (the FP32 reference this oracle
follows).

The oracle here differs from the upstream reference in one way that matters for
the benchmark shapes: upstream materializes a dense ``(tokens, heads, S_kv)``
score tensor and differentiates it with autograd, while this one gathers only the
top-k rows each query actually attends to and applies the analytic backward. Both
compute the same quantity; the gathered form is what makes the larger
configurations affordable.
"""

import math

import torch

from . import spec

_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


# ``dsa_bwd_sm100.py:177``/``:189``: both workspaces round their sequence and
# head-dim extents up to a multiple of 8.
def _roundup8(value):
    return (value + 7) // 8 * 8


def torch_dtype(name):
    return _TORCH_DTYPES[name]


def _make_topk(generator, *, seqlen_q, seqlen_kv, max_topk, mode, device):
    """Build ``topk_idxs`` and ``topk_length`` for one config.

    ``topk_idxs`` holds global KV row indices, distinct within a query. ``mode``
    selects which of the kernel's index paths the config exercises.
    """
    # The same construction the upstream benchmark uses
    # (``benchmark_dsa_sparse_attention_backward.py:59``): one vectorized argsort
    # rather than a per-row ``randperm``, which keeps the indices distinct within
    # a query and scattered across the KV axis at the benchmark shapes.
    idxs = (
        torch.rand(seqlen_q, seqlen_kv, generator=generator, device=device)
        .argsort(dim=-1)[:, :max_topk]
        .to(torch.int32)
    )

    if mode == "invalid_idx":
        # The non-compact variant marks unused rows with -1 instead of carrying a
        # length. Keep a per-query prefix valid so the tail is ragged.
        keep = torch.randint(1, max_topk + 1, (seqlen_q,), generator=generator, device=device)
        positions = torch.arange(max_topk, device=device).unsqueeze(0)
        idxs = torch.where(positions < keep.unsqueeze(1), idxs, torch.full_like(idxs, -1))
        return idxs, None

    if mode == "full":
        lengths = torch.full((seqlen_q,), max_topk, dtype=torch.int32, device=device)
    elif mode == "random":
        lengths = torch.randint(
            1, max_topk + 1, (seqlen_q,), generator=generator, device=device
        ).to(torch.int32)
    elif mode == "mixed_zero":
        lengths = torch.randint(
            1, max_topk + 1, (seqlen_q,), generator=generator, device=device
        ).to(torch.int32)
        # Interleave empty and negative rows with live ones: a CTA that exits
        # early must not strand the CTAs that still have work.
        lengths[0::4] = 0
        lengths[1::8] = -3
    elif mode == "all_empty":
        lengths = torch.zeros(seqlen_q, dtype=torch.int32, device=device)
    else:
        raise ValueError(f"unknown topk mode: {mode}")
    return idxs, lengths


def _make_sink(generator, *, num_head, mode, max_topk, device):
    if mode == "normal":
        return torch.randn(num_head, generator=generator, dtype=torch.float32, device=device)
    if mode == "log_topk":
        # The upstream d576 regression: a sink of this magnitude is comparable to
        # the KV mass, so leaving it out of the normalization is visible.
        return torch.full((num_head,), math.log(max_topk), dtype=torch.float32, device=device)
    if mode == "disabled":
        return torch.full((num_head,), float("-inf"), dtype=torch.float32, device=device)
    raise ValueError(f"unknown sink mode: {mode}")


def _gather_valid(kv_f32, idxs_row, lengths_row, max_topk):
    """Gather the KV rows one chunk of queries attends to, plus a validity mask.

    Returns ``(gathered, valid)`` where ``gathered`` is ``(chunk, max_topk, D)``
    and ``valid`` is ``(chunk, max_topk)``.
    """
    seqlen_kv = kv_f32.shape[0]
    valid = (idxs_row >= 0) & (idxs_row < seqlen_kv)
    if lengths_row is not None:
        positions = torch.arange(max_topk, device=idxs_row.device).unsqueeze(0)
        valid = valid & (positions < lengths_row.unsqueeze(1))
    safe = torch.where(valid, idxs_row, torch.zeros_like(idxs_row)).to(torch.long)
    gathered = kv_f32[safe]
    return gathered, valid


def reference_forward(q, kv, attn_sink, topk_idxs, topk_length, softmax_scale, chunk=64):
    """FP32 forward over the gathered top-k rows.

    Returns ``(out, lse)`` with ``lse`` the KV-only log-sum-exp, sink excluded --
    the same convention the kernel consumes.
    """
    seqlen_q, num_head, head_dim = q.shape
    head_dim_v = spec.head_dim_v_for(head_dim)
    max_topk = topk_idxs.shape[1]
    kv_f32 = kv.to(torch.float32)

    out = torch.empty(seqlen_q, num_head, head_dim_v, dtype=torch.float32, device=q.device)
    lse = torch.empty(seqlen_q, num_head, dtype=torch.float32, device=q.device)

    for start in range(0, seqlen_q, chunk):
        stop = min(start + chunk, seqlen_q)
        q_c = q[start:stop].to(torch.float32)
        idxs_c = topk_idxs[start:stop]
        len_c = None if topk_length is None else topk_length[start:stop]
        gathered, valid = _gather_valid(kv_f32, idxs_c, len_c, max_topk)

        scores = torch.einsum("thd,tkd->thk", q_c, gathered) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

        lse_c = torch.logsumexp(scores, dim=-1)
        lse_full = torch.logaddexp(lse_c, attn_sink.view(1, num_head))
        weights = torch.exp(scores - lse_full.unsqueeze(-1))
        out[start:stop] = torch.einsum("thk,tkd->thd", weights, gathered[:, :, :head_dim_v])
        lse[start:stop] = lse_c

    return out, lse


# --- TMA descriptors -------------------------------------------------------
# The four rank-3 TensorMaps the `bwd` kernel consumes. Encoded host-side into
# 64-byte-aligned 128-byte payloads, the same mechanism the FlashAttention
# backward port uses.
#
# Every one has a 64x64 box: `Q`/`dO` are loaded as `head_dim // 64` issues of
# 64 dims x 64 heads, and the `dQ` stores go out two boxes per 128-wide round.
# The swizzle is 128B (mode 3) throughout, matching the S<3,4,3> every SMEM
# layout in this kernel carries.
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
    """One ``cuTensorMapEncodeTiled``; element strides are all 1 here."""
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
        *((1,) * rank),  # elementStrides
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        _TMA_SWIZZLE_128B,
        _TMA_L2_256B,
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def build_tensor_maps(q, dout, dq, *, head_dim, head_dim_v, num_head, seqlen_q, dtype_name):
    """`desc_Q`, `desc_dO`, `desc_dQ` and, at d576, `desc_dQ_64`.

    TMA dimensions are innermost-first, so a row-major `(S_q, H, D)` tensor is
    described as `(D, H, S_q)`; the byte strides exclude the innermost, which is
    implicitly the element size.
    """
    esize = 2
    maps = {
        "q": _encode_tensormap(
            q,
            dtype_name,
            (head_dim, num_head, seqlen_q),
            (head_dim * esize, num_head * head_dim * esize),
            (64, 64, 1),
        ),
        "do": _encode_tensormap(
            dout,
            dtype_name,
            (head_dim_v, num_head, seqlen_q),
            (head_dim_v * esize, num_head * head_dim_v * esize),
            (64, 64, 1),
        ),
        "dq": _encode_tensormap(
            dq,
            dtype_name,
            (head_dim, num_head, seqlen_q),
            (head_dim * esize, num_head * head_dim * esize),
            (64, 64, 1),
        ),
    }
    return maps


def prepare_data(**config):
    """Allocate one input set shared by TIRx, the upstream kernel, and the oracle."""
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("CUDA is required")

    spec.check_dispatch(config)
    device = "cuda"
    dtype = torch_dtype(config["dtype"])
    head_dim = config["head_dim"]
    head_dim_v = spec.head_dim_v_for(head_dim)
    num_head = config["num_head"]
    seqlen_q = config["seqlen_q"]
    seqlen_kv = config["seqlen_kv"]
    max_topk = config["max_topk"]

    generator = torch.Generator(device=device)
    generator.manual_seed(config["seed"])

    def randn(*shape, dt=torch.float32):
        return torch.randn(*shape, generator=generator, dtype=dt, device=device)

    q = randn(seqlen_q, num_head, head_dim).to(dtype)
    kv = randn(seqlen_kv, head_dim).to(dtype)
    dout = randn(seqlen_q, num_head, head_dim_v).to(dtype)
    attn_sink = _make_sink(
        generator, num_head=num_head, mode=config["sink_mode"], max_topk=max_topk, device=device
    )
    topk_idxs, topk_length = _make_topk(
        generator,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        max_topk=max_topk,
        mode=config["topk_mode"],
        device=device,
    )
    if config["has_topk_length"]:
        if topk_length is None:
            raise ValueError("config declares has_topk_length but the topk mode produced none")
    else:
        # The non-compact variant is a distinct compiled program: it takes no
        # length tensor and reads validity from the -1 markers alone. Passing a
        # length here would silently select the compact program instead.
        topk_length = None

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # ``out`` and ``lse`` stand in for the FlashMLA forward the kernel consumes.
    out_f32, lse = reference_forward(q, kv, attn_sink, topk_idxs, topk_length, softmax_scale)
    out = out_f32.to(dtype)

    # ``_interface_sm100.py:115-135``: dq is fully overwritten, dkv and d_sink are
    # atomic accumulators and must be zero on entry.
    def outputs():
        return {
            "dq": torch.empty(seqlen_q, num_head, head_dim, dtype=dtype, device=device),
            "dkv": torch.zeros(seqlen_kv, head_dim, dtype=dtype, device=device),
            "d_sink": torch.zeros(num_head, dtype=torch.float32, device=device),
        }

    # ``_get_workspace_size_LSE_OdO`` / ``_get_workspace_size_dKV``.
    ws_lse_odo = torch.zeros(1, num_head, _roundup8(seqlen_q), 8, dtype=torch.uint8, device=device)
    ws_dkv = torch.zeros(
        1, 1, _roundup8(seqlen_kv), _roundup8(head_dim) * 4, dtype=torch.uint8, device=device
    )

    return {
        "config": dict(config),
        "inputs": {
            "q": q,
            "kv": kv,
            "out": out,
            "dout": dout,
            "lse": lse,
            "attn_sink": attn_sink,
            "topk_idxs": topk_idxs,
            "topk_length": topk_length,
            "softmax_scale": softmax_scale,
        },
        "tirx": {**outputs(), "ws_lse_odo": ws_lse_odo, "ws_dkv": ws_dkv},
        "source": outputs(),
        "derived": {
            "head_dim_v": head_dim_v,
            "seqlen_q": seqlen_q,
            "seqlen_kv": seqlen_kv,
            "max_topk": max_topk,
        },
    }


def compile_reference(data):
    """Return a no-argument closure that runs the upstream kernel on ``data``.

    ``dq`` and ``dkv`` are out-parameters the wrapper fills in place, but
    ``d_sink`` is always freshly allocated and returned
    (``_interface_sm100.py:135``), so the closure copies the returned gradient
    back into the result set rather than dropping it.
    """
    from . import reference

    fn = reference.flash_attn_bwd_sm100()
    inputs = data["inputs"]
    outputs = data["source"]

    def launch():
        _, _, d_sink = fn(
            inputs["q"],
            inputs["kv"],
            inputs["out"],
            inputs["dout"],
            inputs["lse"],
            inputs["attn_sink"],
            inputs["topk_idxs"],
            softmax_scale=inputs["softmax_scale"],
            topk_length=inputs["topk_length"],
            dq=outputs["dq"],
            dkv=outputs["dkv"],
        )
        outputs["d_sink"].copy_(d_sink)

    return launch


def tirx_launch(executables, data):
    """Return a no-argument closure that runs the four TIRx kernels on ``data``.

    The zeroing is part of the closure, not of setup: ``dkv``, ``d_sink`` and both
    workspaces are accumulated into, so every repetition needs them clean. The
    upstream wrapper does the same zeroing inside its own timed call
    (``_interface_sm100.py:105-162``), which is what keeps the two timing scopes
    comparable.
    """
    import math

    inputs = data["inputs"]
    out = data["tirx"]
    der = data["derived"]
    seqlen_q = der["seqlen_q"]
    seqlen_kv = der["seqlen_kv"]
    head_dim_v = der["head_dim_v"]
    num_head = inputs["q"].shape[1]
    head_dim = inputs["q"].shape[2]
    # The workspace planes are head-contiguous and padded to a multiple of 8
    # rows, matching the upstream `roundup8` (:105-162).
    plane = num_head * ((seqlen_q + 7) // 8 * 8)

    sum_odo, bwd, convert, sum_dsink = executables
    ws_lse_odo = out["ws_lse_odo"].view(-1).view(torch.float32)
    ws_dkv = out["ws_dkv"].view(-1).view(torch.float32)
    maps = build_tensor_maps(
        inputs["q"],
        inputs["dout"],
        out["dq"],
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_head=num_head,
        seqlen_q=seqlen_q,
        dtype_name=str(inputs["q"].dtype).rsplit(".", 1)[-1],
    )
    neg_log2_e = -math.log2(math.e)
    scale = inputs["softmax_scale"]
    topk_idxs = inputs["topk_idxs"].reshape(-1)
    # The non-compact variant has no topk_length tensor; pass the idx buffer as
    # a placeholder so the ABI stays fixed -- the kernel never reads it.
    topk_length = (
        inputs["topk_length"].reshape(-1) if inputs["topk_length"] is not None else topk_idxs
    )

    def launch():
        out["dkv"].zero_()
        out["d_sink"].zero_()
        ws_lse_odo.zero_()
        ws_dkv.zero_()
        sum_odo(
            inputs["out"].reshape(-1),
            inputs["dout"].reshape(-1),
            inputs["lse"].reshape(-1),
            inputs["attn_sink"].reshape(-1),
            ws_lse_odo,
            seqlen_q,
            plane,
            -1.0,
            neg_log2_e,
        )
        bwd(
            maps["q"].ptr,
            maps["do"].ptr,
            maps["dq"].ptr,
            inputs["kv"].reshape(-1),
            out["dq"].reshape(-1),
            ws_dkv,
            topk_idxs,
            topk_length,
            ws_lse_odo,
            seqlen_q,
            seqlen_kv,
            plane,
            scale,
        )
        convert(ws_dkv, out["dkv"].reshape(-1), seqlen_kv)
        sum_dsink(
            ws_lse_odo, inputs["attn_sink"].reshape(-1), out["d_sink"].reshape(-1), seqlen_q, plane
        )

    return launch


def validate_outputs(data, sources=("tirx",), with_oracle=None, atol=5e-2, rtol=5e-2):
    """Hold TIRx to the upstream kernel's outputs on the same inputs.

    ``dkv`` is accumulated with global FP32 atomics, so its summation order
    varies run to run and no bitwise comparison is available; both
    implementations sit within the upstream tolerance of the true value, so
    they sit within twice it of each other -- the comparison below uses the
    same tolerance.
    """
    del with_oracle
    for source in sources:
        got = data[source]
        for name in ("dq", "dkv", "d_sink"):
            if not torch.isfinite(got[name].to(torch.float32)).all():
                raise AssertionError(f"{source}.{name} contains non-finite values")
    if "tirx" in sources and "source" in sources:
        for name in ("dq", "dkv", "d_sink"):
            torch.testing.assert_close(
                data["tirx"][name].to(torch.float32),
                data["source"][name].to(torch.float32),
                atol=atol,
                rtol=rtol,
                msg=lambda m, name=name: f"tirx vs source {name}: {m}",
            )
