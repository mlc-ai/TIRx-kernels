# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Tensors, FP32 oracle and upstream reference for the BF16 dGLU+dBias port.

One input set is shared by TIRx, the upstream kernel and the oracle, with a
separate output set per implementation so the three can be compared directly.
"""

import importlib
import importlib.machinery
import os
import sys
import types
from functools import cache

from . import spec as _spec

_REFERENCE_PACKAGE = "tirx_cudnn_frontend_grouped"

_TORCH_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def _torch_dtype(torch, name):
    return getattr(torch, _TORCH_DTYPES[name])


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def prepare_data(**config):
    """Allocate one input set plus per-implementation output sets."""
    import torch

    derived = _spec.derive(config)
    L, N, K = derived["L"], derived["N"], derived["K"]
    M = derived["tokens_total"]
    device = "cuda"
    torch.manual_seed(20260826 + M + N + K + L)

    ab = torch.bfloat16
    c_dtype = _torch_dtype(torch, config["c_dtype"])
    d_dtype = _torch_dtype(torch, config["d_dtype"])

    a = (torch.randn(M, K, device=device) * 0.125).to(ab)
    b = (torch.randn(L, N, K, device=device) * 0.125).to(ab)
    c = (torch.randn(M, 2 * N, device=device) * 0.5).to(c_dtype)

    # Probe the activation clamps: dGeGLU masks gradients outside +-7 after the
    # beta scale, so seed values that straddle the boundary on both branches.
    beta = torch.empty(L, dtype=torch.float32, device=device).uniform_(-1.5, 1.5)
    beta[beta.abs() < 0.25] = 1.5
    alpha = torch.empty(L, dtype=torch.float32, device=device).uniform_(-1.25, 1.25)
    alpha[alpha.abs() < 0.25] = 0.75
    begin = 0
    for expert, rows in enumerate(config["group_m_list"]):
        if rows == 0:
            continue
        scale = float(beta[expert])
        for column, value in ((0, 8.5), (1, -8.5), (32, 8.25), (33, -8.25)):
            if column < 2 * N:
                c[begin, column] = value / scale
        begin += _spec.align_up(rows, _spec.FIX_PAD_SIZE)

    prob = torch.linspace(-0.75, 0.875, M, device=device, dtype=torch.float32)
    padded_offsets = torch.tensor(derived["padded_offsets"], dtype=torch.int32, device=device)

    def outputs():
        return {
            "d": torch.zeros(M, 2 * N, dtype=d_dtype, device=device),
            "dprob": torch.zeros(M, dtype=torch.float32, device=device),
            "dbias": (
                torch.zeros(L, 2 * N, dtype=torch.bfloat16, device=device)
                if config["with_dbias"]
                else None
            ),
            "workspace": (
                torch.zeros(derived["workspace_bytes"] + 128, dtype=torch.uint8, device=device)
                if derived["workspace_bytes"]
                else None
            ),
        }

    # (N, K, L) views of the same values, one k-major and one n-major, so a
    # specialization can be handed the layout its ``b_major`` promises.
    b_kmajor = b.permute(1, 2, 0)
    b_nmajor = b.permute(0, 2, 1).contiguous().permute(2, 1, 0)

    b_ptrs = None
    b_experts = None
    if config["weight_mode"] == "discrete":
        b_experts = [
            b[i].contiguous() if config["b_major"] == "k" else b[i].t().contiguous().t()
            for i in range(L)
        ]
        b_ptrs = torch.tensor(
            [int(t.data_ptr()) for t in b_experts], dtype=torch.int64, device=device
        )

    return {
        "config": dict(config),
        "derived": derived,
        "a": a,
        "b": b,
        "b_kmajor": b_kmajor,
        "b_nmajor": b_nmajor,
        "b_ptrs": b_ptrs,
        "b_experts": b_experts,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "prob": prob,
        "padded_offsets": padded_offsets,
        "tirx": outputs(),
        "source": outputs(),
    }


# ---------------------------------------------------------------------------
# FP32 oracle
# ---------------------------------------------------------------------------


def reference_outputs(data):
    """FP32 reference for D, dprob and dbias, following the upstream formulas."""
    import torch

    config = data["config"]
    derived = data["derived"]
    L, N = derived["L"], derived["N"]
    two_n = 2 * N
    act = config["act"]
    linear_offset = config["linear_offset"]

    a, b, c = data["a"], data["b"], data["c"]
    alpha, beta, prob = data["alpha"], data["beta"], data["prob"]
    M = a.shape[0]
    device = a.device

    d = torch.zeros(M, two_n, dtype=torch.float32, device=device)
    dprob = torch.zeros(M, dtype=torch.float32, device=device)
    dbias = torch.zeros(L, two_n, dtype=torch.float32, device=device)

    begin = 0
    for expert, rows in enumerate(config["group_m_list"]):
        end = begin + rows
        if rows:
            scale = float(alpha[expert])
            ref = (a[begin:end].float() * scale) @ (b[expert].float() * scale).t()
            scaled = c[begin:end].float() * float(beta[expert])
            blocks = scaled.view(rows, two_n // 32, 32)
            gate_raw = blocks[:, 0::2, :].reshape(rows, N)
            up_raw = blocks[:, 1::2, :].reshape(rows, N)
            p = prob[begin:end].view(-1, 1)

            if act == "dswiglu":
                sigmoid = torch.sigmoid(gate_raw)
                swish = gate_raw * sigmoid
                terms = swish * up_raw * ref
                d_gate = ref * p * up_raw * sigmoid * (1 + gate_raw * (1 - sigmoid))
                d_up = ref * p * swish
            else:
                gate = torch.clamp(gate_raw, max=7.0)
                up = torch.clamp(up_raw, -7.0, 7.0)
                sigmoid = torch.sigmoid(1.702 * gate)
                terms = gate * sigmoid * (up + linear_offset) * ref
                d_gate = (
                    ref * sigmoid * (1 + 1.702 * gate * (1 - sigmoid)) * (up + linear_offset) * p
                )
                d_up = ref * gate * sigmoid * p
                gate_filter = torch.where(gate_raw > 7.0, torch.zeros_like(gate_raw), gate_raw)
                up_filter = torch.where(
                    (up_raw > 7.0) | (up_raw < -7.0), torch.zeros_like(up_raw), up_raw
                )
                d_gate = d_gate * gate_filter
                d_up = d_up * up_filter

            # dprob follows the kernel's 32-column subtile reduction order.
            chunks = terms.split(32, dim=1)
            dprob[begin:end] = torch.cat(
                [chunk.sum(dim=1, keepdim=True) for chunk in chunks], dim=1
            ).sum(dim=1)

            rows_out = torch.zeros(rows, two_n // 32, 32, dtype=torch.float32, device=device)
            rows_out[:, 0::2, :] = d_gate.view(rows, N // 32, 32)
            rows_out[:, 1::2, :] = d_up.view(rows, N // 32, 32)
            block = rows_out.view(rows, two_n)
            d[begin:end] = block
            dbias[expert] = block.sum(dim=0)
        begin += _spec.align_up(rows, _spec.FIX_PAD_SIZE)

    return {"d": d, "dprob": dprob, "dbias": dbias}


# ---------------------------------------------------------------------------
# Upstream kernel as reference
# ---------------------------------------------------------------------------


@cache
def load_reference_source():
    """Import the upstream kernel without executing the cuDNN API package.

    The kernel and its scheduler/util siblings import only cutlass and each
    other, so a synthetic package rooted at ``cutedsl/grouped`` resolves their
    relative imports while ``dglu/__init__.py`` -- which pulls in the compiled
    ``cudnn`` extension -- is never run.
    """
    root = os.environ.get("CUDNN_FRONTEND_PATH")
    if root is None:
        raise RuntimeError("CUDNN_FRONTEND_PATH must point to a cuDNN Frontend source checkout")
    grouped = os.path.join(root, "python/cudnn/gemm/cutedsl/grouped")
    if not os.path.isdir(grouped):
        raise RuntimeError(f"cannot find the grouped GEMM sources under {grouped}")
    for name, path in (
        (_REFERENCE_PACKAGE, grouped),
        (f"{_REFERENCE_PACKAGE}.dglu", os.path.join(grouped, "dglu")),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [path]
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        module.__spec__.submodule_search_locations = [path]
        sys.modules[name] = module
    return importlib.import_module(f"{_REFERENCE_PACKAGE}.dglu.moe_grouped_gemm_dglu_dbias")


def _rebind_submodules(package):
    """Re-attach already-loaded submodules to a freshly re-executed package."""
    prefix = package.__name__ + "."
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix) or module is None:
            continue
        child = name[len(prefix) :]
        if "." in child:
            continue
        if getattr(package, child, None) is not module:
            setattr(package, child, module)
    return package


def import_cutlass_reference():
    """Recover from CuTeDSL's non-idempotent generated builder imports."""
    try:
        return _rebind_submodules(importlib.import_module("cutlass"))
    except RuntimeError as exc:
        message = str(exc)
        if "Attribute builder for '" not in message or "is already registered" not in message:
            raise
        mlir_ir = sys.modules.get("cutlass._mlir.ir")
        if mlir_ir is None:
            raise
        register_attribute_builder = mlir_ir.register_attribute_builder

        def register_replacing_builder(kind, replace=False):
            del replace
            return register_attribute_builder(kind, replace=True)

        mlir_ir.register_attribute_builder = register_replacing_builder
        try:
            return _rebind_submodules(importlib.import_module("cutlass"))
        finally:
            mlir_ir.register_attribute_builder = register_attribute_builder


def compile_reference(data):
    """Compile the upstream kernel over this input set and return a no-arg launch."""
    cutlass = import_cutlass_reference()
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver as cuda
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    if os.environ.get("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0") != "0":
        raise RuntimeError("CUDNNFE_CLUSTER_OVERLAP_MARGIN must be 0")

    module = load_reference_source()
    config = data["config"]
    derived = data["derived"]
    outputs = data["source"]
    discrete = config["weight_mode"] == "discrete"
    N, K, M = derived["N"], derived["K"], derived["tokens_total"]

    kernel = module.MoEGroupedGemmDgluDbiasBf16Kernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=derived["use_2cta"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
        vectorized_f32=config["vectorized_f32"],
        expert_cnt=derived["L"],
        weight_mode=module.MoEWeightMode.DISCRETE if discrete else module.MoEWeightMode.DENSE,
        use_dynamic_sched=config["sched"] == "dynamic",
        act_func=config["act"],
    )

    def dlpack(tensor, *, leading_dim=None, align=16):
        wrapped = from_dlpack(tensor, assumed_align=align)
        if leading_dim is not None:
            wrapped = wrapped.mark_layout_dynamic(leading_dim=leading_dim)
        return wrapped

    b_major = config["b_major"]
    b_major_mode = OperandMajorMode.K if b_major == "k" else OperandMajorMode.MN
    b_stride_size = K if b_major == "k" else N
    workspace = outputs["workspace"]
    workspace_argument = (
        dlpack(workspace, align=128).iterator
        if workspace is not None
        else cute.runtime.nullptr(dtype=cutlass.Uint8, assumed_align=128)
    )
    # ``b`` is (N, K, L) either way; only which extent is contiguous differs, and
    # the leading dimension the DSL is told about must actually carry stride 1.
    b_dense = data["b_kmajor"] if b_major == "k" else data["b_nmajor"]
    b_argument = (
        dlpack(data["b_ptrs"], align=8).iterator
        if discrete
        else dlpack(b_dense, leading_dim=1 if b_major == "k" else 0)
    )
    dbias = outputs["dbias"]

    arguments = {
        "a": dlpack(data["a"].unsqueeze(-1), leading_dim=1),
        "b": b_argument,
        "n": cutlass.Int32(N),
        "k": cutlass.Int32(K),
        "b_stride_size": cutlass.Int64(b_stride_size),
        "b_major_mode": b_major_mode,
        "workspace_ptr": workspace_argument,
        "c": dlpack(data["c"].unsqueeze(-1), leading_dim=1),
        "d": dlpack(outputs["d"].unsqueeze(-1), leading_dim=1),
        "padded_offsets": dlpack(data["padded_offsets"]),
        "alpha": dlpack(data["alpha"]),
        "beta": dlpack(data["beta"]),
        "prob": dlpack(data["prob"].view(M, 1, 1)),
        "dprob": dlpack(outputs["dprob"].view(M, 1, 1)),
        "linear_offset": cutlass.Float32(config["linear_offset"]),
        "dbias_tensor": dlpack(dbias.unsqueeze(-1)) if dbias is not None else None,
        "max_active_clusters": derived["grid"][2],
        "stream": make_fake_stream(use_tvm_ffi_env_stream=False),
    }
    executable = cute.compile(kernel, **arguments, options="--enable-tvm-ffi")

    # ``b`` and ``workspace_ptr`` lower to raw device pointers, so the runtime call
    # takes addresses where the compile-time call took cute iterators.
    runtime = [
        data["a"].unsqueeze(-1),
        int(data["b_ptrs"].data_ptr()) if discrete else b_dense,
        N,
        K,
        b_stride_size,
        int(workspace.data_ptr()) if workspace is not None else 0,
        data["c"].unsqueeze(-1),
        outputs["d"].unsqueeze(-1),
        data["padded_offsets"],
        data["alpha"],
        data["beta"],
        data["prob"].view(M, 1, 1),
        outputs["dprob"].view(M, 1, 1),
        config["linear_offset"],
        dbias.unsqueeze(-1) if dbias is not None else None,
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def launch():
        executable(*runtime, stream)

    launch._keep_alive = (executable, runtime, stream, data.get("b_experts"))
    return launch


def tirx_launch(executables, data):
    """Bind the compiled TIRx entries to this input set.

    The argument order is the kernel ABI, which the kernel-sketch stage fixes.
    """
    raise NotImplementedError(
        "the TIRx launch ABI is defined by the kernel-sketch stage; the kernel body "
        "is still the scaffold placeholder"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Upstream ``assert_grouped_gemm_dglu_close`` tolerances for D and dprob.
_RTOL = 3e-2
_ATOL = 8e-2
# Upstream ``assert_grouped_gemm_dglu_dbias_close``: the bf16 atomic tree makes the
# reduction order unstable, so the bound scales with the number of contributions.
_DBIAS_RTOL = 1e-2
_DBIAS_SCALE = 0.008


def _dbias_atol(expected, tiles):
    peak = float(expected.abs().max()) if expected.numel() else 0.0
    return max(peak * _DBIAS_SCALE * (tiles**0.5), 0.1)


def validate_outputs(data, *, sources=("tirx",), with_oracle=True):
    """Compare the named output sets with each other and with the FP32 oracle."""
    import torch

    derived = data["derived"]
    expected = reference_outputs(data) if with_oracle else None
    tiles = max(derived["tokens_total"] // 128, 1)

    for source in sources:
        outputs = data[source]
        for name in ("d", "dprob", "dbias"):
            got = outputs.get(name)
            if got is None:
                continue
            if not torch.isfinite(got.float()).all():
                raise AssertionError(f"{source}.{name} has non-finite entries")
        if expected is None:
            continue
        for name in ("d", "dprob"):
            got = outputs[name].float()
            want = expected[name].to(got.dtype)
            if not torch.allclose(got, want, rtol=_RTOL, atol=_ATOL):
                worst = (got - want).abs().max()
                raise AssertionError(f"{source}.{name} differs from the oracle: max {worst:.4g}")
        if outputs["dbias"] is not None:
            got = outputs["dbias"].float()
            want = expected["dbias"].to(torch.bfloat16).float()
            atol = _dbias_atol(want, tiles)
            if not torch.allclose(got, want, rtol=_DBIAS_RTOL, atol=atol):
                worst = (got - want).abs().max()
                raise AssertionError(
                    f"{source}.dbias differs from the oracle: max {worst:.4g} > {atol:.4g}"
                )

    if len(sources) > 1:
        first, *rest = sources
        for other in rest:
            for name in ("d", "dprob", "dbias"):
                left = data[first].get(name)
                right = data[other].get(name)
                if left is None or right is None:
                    continue
                atol = _dbias_atol(right.float(), tiles) if name == "dbias" else _ATOL
                rtol = _DBIAS_RTOL if name == "dbias" else _RTOL
                if not torch.allclose(left.float(), right.float(), rtol=rtol, atol=atol):
                    worst = (left.float() - right.float()).abs().max()
                    raise AssertionError(
                        f"{first}.{name} and {other}.{name} disagree: max {worst:.4g}"
                    )
