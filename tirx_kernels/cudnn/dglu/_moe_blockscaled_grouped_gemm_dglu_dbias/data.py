# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Inputs, FP32 oracle, and upstream-kernel reference for the dGLU backward port.

Upstream sources:
``test/python/fe_api/grouped_gemm/test_grouped_gemm_swiglu_utils.py``
(``create_mask``, ``allocate_grouped_gemm_input_tensors``),
``test/python/fe_api/grouped_gemm/test_grouped_gemm_dswiglu_utils.py``
(``allocate_grouped_gemm_dswiglu_tensors``, ``run_grouped_gemm_dswiglu_ref``,
``compute_reference_row_quant``, ``check_ref_grouped_gemm_dswiglu``),
``test/python/fe_api/test_fe_api_utils.py`` (``create_scale_factor_tensor``),
``python/cudnn/gemm/cutedsl/grouped/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py``
(``__call__`` argument contract, ``dswiglu``/``dgeglu``/``dsituglu``).
"""

import os

from tirx_kernels.cudnn._reference import import_cutlass_reference, load_reference_module

from . import spec

# FP4 E2M1 code point values, indexed by the 4-bit code.
_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
# Payload drawn from values every supported A/B format represents exactly, so the
# FP32 oracle sees the same numbers the kernel multiplies.
_PAYLOAD_VALUES = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
# Scale factors stay powers of two so E8M0 and E4M3 storage are both exact.
_SF_VALUES = (1.0, 2.0)


def _torch_dtype(torch, dtype):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e8m0fnu": torch.float8_e8m0fnu,
    }[dtype]


def _generator(torch, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _payload(torch, shape, seed):
    """Deterministic values that survive every supported input quantization."""
    values = torch.tensor(_PAYLOAD_VALUES, dtype=torch.float32, device="cuda")
    index = torch.randint(
        0,
        len(_PAYLOAD_VALUES),
        shape,
        device="cuda",
        generator=_generator(torch, seed),
        dtype=torch.int64,
    )
    return values[index]


def _nonzero_scalars(torch, shape, generator):
    """Per-expert or per-row scalars drawn from ``{-2, -1, 1, 2}``.

    Zero is excluded deliberately. ``alpha`` scales the GEMM and ``beta`` scales
    the C tensor the activation differentiates, so a zero for either silences an
    entire expert's output no matter what the kernel computed there -- and with
    four experts a uniform draw over ``{-2, -1, 0, 1}`` blanks one in four. The
    same holds for a zero ``prob`` and its row. The magnitude range is unchanged.
    """
    values = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float32, device="cuda")
    index = torch.randint(0, 4, shape, device="cuda", generator=generator, dtype=torch.int64)
    return values[index]


def _pack_fp4(torch, values):
    """Encode exact E2M1 values into packed nibble pairs along the last axis."""
    table = torch.tensor(_FP4_VALUES[:8], dtype=torch.float32, device=values.device)
    magnitude = values.abs()
    matches = magnitude.unsqueeze(-1) == table
    if not bool(matches.any(dim=-1).all().item()):
        raise ValueError("payload value is not representable in FP4 E2M1")
    code = matches.to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    code = code | ((values < 0).to(torch.uint8) << 3)
    return code[..., 0::2] | (code[..., 1::2] << 4)


def _unpack_fp4(torch, packed):
    table = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    low = (packed & 0xF).to(torch.int64)
    high = (packed >> 4).to(torch.int64)
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return table[codes]


def _operand(torch, rows, columns, batch, dtype, *, first_major, seed):
    """A/B operand in the upstream stride convention, plus its FP32 value view.

    ``first_major`` selects the ``(rows, columns, batch)`` stride order the upstream
    ``create_and_permute_tensor`` produces for a mode-0-major tensor.
    """
    logical_shape = (batch, columns, rows) if first_major else (batch, rows, columns)
    permute_order = (2, 1, 0) if first_major else (1, 2, 0)
    values = _payload(torch, logical_shape, seed)
    if dtype == "float4_e2m1fn":
        packed = _pack_fp4(torch, values).contiguous()
        raw = packed.reshape(-1)
        tensor = raw.view(*packed.shape).view(torch.float4_e2m1fn_x2).permute(permute_order)
        return {
            "raw": raw,
            "tensor": tensor,
            "values": _unpack_fp4(torch, packed).permute(permute_order),
        }
    storage = values.to(_torch_dtype(torch, dtype)).contiguous()
    tensor = storage.permute(permute_order)
    return {
        "raw": storage.view(torch.uint8).reshape(-1),
        "tensor": tensor,
        "values": storage.float().permute(permute_order),
    }


def _scale_factors(torch, rows, K_dim, batch, sf_dtype, sf_vec_size, seed):
    """Scale factors in the MMA-interleaved layout, plus their logical value view.

    The upstream layout is ``(L, ceil(rows/128), ceil(sf_k/4), 32, 4, 4)`` permuted
    to ``(32, 4, ceil(rows/128), 4, ceil(sf_k/4), L)``; logical row ``m`` maps to
    ``(m % 32, (m // 32) % 4, m // 128)`` and logical ``sf_k`` to ``(k % 4, k // 4)``.
    """
    rest_m = spec.ceil_div(rows, 128)
    sf_k = spec.ceil_div(K_dim, sf_vec_size)
    rest_k = spec.ceil_div(sf_k, 4)
    values = torch.tensor(_SF_VALUES, dtype=torch.float32, device="cuda")
    index = torch.randint(
        0,
        len(_SF_VALUES),
        (batch, rest_m, rest_k, 32, 4, 4),
        device="cuda",
        generator=_generator(torch, seed),
        dtype=torch.int64,
    )
    atoms = values[index]
    storage = atoms.to(_torch_dtype(torch, sf_dtype)).contiguous()
    logical = atoms.permute(0, 1, 4, 3, 2, 5).reshape(batch, rest_m * 128, rest_k * 4)
    return {
        "raw": storage.view(torch.uint8).reshape(-1),
        "tensor": storage.permute(3, 4, 1, 5, 2, 0),
        "values": logical[:, :rows, :sf_k].contiguous(),
    }


def _expand_scale(torch, logical, K_dim, sf_vec_size):
    """``(L, rows, sf_k)`` -> ``(L, rows, K)`` by repeating each scale over its vector."""
    batch, rows, sf_k = logical.shape
    expanded = logical.unsqueeze(-1).expand(batch, rows, sf_k, sf_vec_size)
    return expanded.reshape(batch, rows, sf_k * sf_vec_size)[:, :, :K_dim]


def _epilogue_tensor(torch, rows, columns, batch, dtype, *, seed=None):
    """C/D-style ``(rows, columns, batch)`` tensor with the upstream n-major stride."""
    if seed is None:
        values = torch.zeros((batch, rows, columns), dtype=torch.float32, device="cuda")
    else:
        values = _payload(torch, (batch, rows, columns), seed)
    storage = values.to(_torch_dtype(torch, dtype)).contiguous()
    return {
        "raw": storage.view(torch.uint8).reshape(-1),
        "tensor": storage.permute(1, 2, 0),
        "values": storage.float().permute(1, 2, 0),
    }


def _sfd_buffer(torch, rows, columns, sf_dtype, sf_vec_size):
    rest_m = spec.ceil_div(rows, 128)
    rest_k = spec.ceil_div(spec.ceil_div(columns, sf_vec_size), 4)
    storage = torch.zeros((1, rest_m, rest_k, 32, 4, 4), dtype=torch.float32, device="cuda").to(
        _torch_dtype(torch, sf_dtype)
    )
    return {
        "raw": storage.view(torch.uint8).reshape(-1),
        "tensor": storage.permute(3, 4, 1, 5, 2, 0),
        "storage": storage,
    }


def _padded_offsets(torch, group_m_list):
    offsets = []
    total = 0
    for rows in group_m_list:
        total += spec.align_up(rows, spec.FIX_PAD_SIZE)
        offsets.append(total)
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _output_set(torch, derived, config):
    """One independent set of kernel outputs, zeroed the way the kernel expects."""
    tokens_total = derived["tokens_total"]
    n_out = derived["n_out"]
    L = derived["L"]
    outputs = {
        "d_row": _epilogue_tensor(torch, tokens_total, n_out, 1, config["d_dtype"]),
        "d_col": _epilogue_tensor(torch, tokens_total, n_out, 1, config["d_dtype"]),
        "dprob": torch.zeros((tokens_total, 1, 1), dtype=torch.float32, device="cuda"),
        "dbias": None,
        "amax": None,
        "sfd_row": None,
        "sfd_col": None,
        "workspace": None,
    }
    if derived["generate_dbias"]:
        outputs["dbias"] = torch.zeros((L, n_out, 1), dtype=torch.bfloat16, device="cuda")
    if derived["generate_amax"]:
        outputs["amax"] = torch.full((L, 2, 1), float("-inf"), dtype=torch.float32, device="cuda")
    if derived["generate_sfd"]:
        outputs["sfd_row"] = _sfd_buffer(
            torch, tokens_total, n_out, config["sf_dtype"], config["sf_vec_size"]
        )
        outputs["sfd_col"] = _sfd_buffer(
            torch, n_out, tokens_total, config["sf_dtype"], config["sf_vec_size"]
        )
    if derived["workspace_bytes"]:
        outputs["workspace"] = torch.zeros(
            derived["workspace_bytes"], dtype=torch.uint8, device="cuda"
        )
    return outputs


def _discrete_pointers(torch, data, derived):
    """Per-expert base addresses into the dense B/SFB allocations.

    Every expert slice is contiguous with the uniform stride the upstream discrete
    path assumes, so slicing the dense buffers is equivalent to the separate
    per-expert allocations the upstream test builds.
    """
    L = derived["L"]
    b_raw = data["b"]["raw"]
    sfb_raw = data["sfb"]["raw"]
    b_bytes = b_raw.numel() // L
    sfb_bytes = sfb_raw.numel() // L
    b_ptrs = torch.tensor(
        [b_raw.data_ptr() + index * b_bytes for index in range(L)], dtype=torch.int64, device="cuda"
    )
    sfb_ptrs = torch.tensor(
        [sfb_raw.data_ptr() + index * sfb_bytes for index in range(L)],
        dtype=torch.int64,
        device="cuda",
    )
    return b_ptrs, sfb_ptrs


def prepare_data(**config):
    """Allocate one input set shared by TIRx, the upstream kernel, and the oracle."""
    import torch

    config = {key: value for key, value in config.items() if key != "label"}
    derived = spec.derive_for_config(config)
    tokens_total, N, K_dim, L = (derived[key] for key in ("tokens_total", "N", "K", "L"))
    generator = _generator(torch, 6)
    data = {
        "config": config,
        "derived": derived,
        "a": _operand(torch, tokens_total, K_dim, 1, config["ab_dtype"], first_major=False, seed=1),
        "b": _operand(
            torch, N, K_dim, L, config["ab_dtype"], first_major=config["b_major"] == "n", seed=2
        ),
        "sfa": _scale_factors(
            torch, tokens_total, K_dim, 1, config["sf_dtype"], config["sf_vec_size"], 3
        ),
        "sfb": _scale_factors(torch, N, K_dim, L, config["sf_dtype"], config["sf_vec_size"], 4),
        "c": _epilogue_tensor(torch, tokens_total, derived["n_out"], 1, config["c_dtype"], seed=5),
        "alpha": _nonzero_scalars(torch, (L,), generator),
        "beta": _nonzero_scalars(torch, (L,), generator),
        "prob": _nonzero_scalars(torch, (tokens_total, 1, 1), generator),
        "padded_offsets": _padded_offsets(torch, config["group_m_list"]),
        "norm_const": torch.tensor([0.01], dtype=torch.float32, device="cuda"),
        "tirx": _output_set(torch, derived, config),
        "source": _output_set(torch, derived, config),
    }
    if config["weight_mode"] == "discrete":
        data["b_ptrs"], data["sfb_ptrs"] = _discrete_pointers(torch, data, derived)
    return data


# ---------------------------------------------------------------------------
# FP32 oracle
# ---------------------------------------------------------------------------


def _expert_ranges(config):
    start = 0
    for index, rows in enumerate(config["group_m_list"]):
        end = start + spec.align_up(rows, spec.FIX_PAD_SIZE)
        yield index, start, end
        start = end


def _grouped_gemm(torch, data):
    """``alpha[g]^2 * dequant(A) @ dequant(B[g])^T`` over each expert's row range."""
    derived = data["derived"]
    config = data["config"]
    tokens_total, N, K_dim = derived["tokens_total"], derived["N"], derived["K"]
    sf_vec_size = config["sf_vec_size"]
    a_values = data["a"]["values"][:, :, 0]
    b_values = data["b"]["values"]
    sfa = _expand_scale(torch, data["sfa"]["values"], K_dim, sf_vec_size)[0, :tokens_total, :]
    sfb = _expand_scale(torch, data["sfb"]["values"], K_dim, sf_vec_size)
    result = torch.zeros((tokens_total, N), dtype=torch.float32, device="cuda")
    for index, start, end in _expert_ranges(config):
        if end <= start:
            continue
        alpha = float(data["alpha"][index].item())
        scaled_a = a_values[start:end, :] * sfa[start:end, :] * alpha
        scaled_b = b_values[:, :, index] * sfb[index, :N, :] * alpha
        result[start:end, :] = scaled_a @ scaled_b.transpose(0, 1)
    return result


def _deinterleave(tensor, N):
    """Split the 2N output into gate (even 32-blocks) and up (odd 32-blocks)."""
    blocks = tensor.reshape(tensor.shape[0], (2 * N) // 32, 32)
    return blocks[:, 0::2, :].reshape(tensor.shape[0], N), blocks[:, 1::2, :].reshape(
        tensor.shape[0], N
    )


def _interleave(torch, gate_grad, up_grad, N):
    rows = gate_grad.shape[0]
    stacked = torch.stack(
        (gate_grad.reshape(rows, N // 32, 32), up_grad.reshape(rows, N // 32, 32)), dim=2
    )
    return stacked.reshape(rows, 2 * N)


def _activation_backward(torch, config, gemm, gate, up, prob):
    """The three dGLU derivatives, transcribed from the upstream epilogue."""
    act = config["act"]
    weighted = gemm * prob
    if act == "dswiglu":
        sigmoid = torch.sigmoid(gate)
        swish = gate * sigmoid
        gate_grad = weighted * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        up_grad = weighted * swish
        prob_grad = swish * up * gemm
    elif act == "dsituglu":
        beta1 = float(config["situ_beta1"])
        beta2 = float(config["situ_beta2"])
        sigmoid = torch.sigmoid(gate)
        gate_tanh = torch.tanh(gate / beta1)
        up_tanh = torch.tanh(up / beta2)
        gate_value = beta1 * gate_tanh * sigmoid
        up_value = beta2 * up_tanh
        gate_derivative = (1.0 - gate_tanh.square()) * sigmoid + beta1 * gate_tanh * sigmoid * (
            1.0 - sigmoid
        )
        up_derivative = 1.0 - up_tanh.square()
        gate_grad = weighted * up_value * gate_derivative
        up_grad = weighted * gate_value * up_derivative
        prob_grad = gate_value * up_value * gemm
    elif act == "dgeglu":
        alpha = float(config["geglu_alpha"])
        clamp_max = float(config["glu_clamp_max"])
        clamp_min = float(config["glu_clamp_min"])
        linear_offset = float(config["linear_offset"])
        clamped_gate = torch.clamp(gate, max=clamp_max)
        clamped_up = torch.clamp(up, min=clamp_min, max=clamp_max)
        sigmoid = torch.sigmoid(alpha * clamped_gate)
        offset_up = clamped_up + linear_offset
        gate_grad = weighted * sigmoid * (1.0 + alpha * clamped_gate * (1.0 - sigmoid)) * offset_up
        up_grad = weighted * clamped_gate * sigmoid
        gate_grad = gate_grad * (gate <= clamp_max).to(torch.float32)
        up_grad = up_grad * ((up >= clamp_min) & (up <= clamp_max)).to(torch.float32)
        prob_grad = clamped_gate * sigmoid * offset_up * gemm
    else:
        raise ValueError(f"unknown activation {act}")
    return gate_grad, up_grad, prob_grad


def _rcp_limit(dtype):
    return {"float8_e4m3fn": 1 / 448.0, "float8_e5m2": 1 / 128.0, "float4_e2m1fn": 1 / 6.0}.get(
        dtype, 1.0
    )


def _quantize_scale(torch, scale, sf_dtype):
    """Round a scale the way the epilogue's ``cvt_f32_to_f8_to_f32`` does.

    E8M0 goes through ``cvt.rp.satfinite.ue8m0x2.f32`` -- round toward positive
    infinity, so the stored scale is the next power of two at or above the value.
    E4M3 goes through ``cvt.rn``, which is torch's default rounding.
    """
    if sf_dtype != "float8_e8m0fnu":
        return scale.to(_torch_dtype(torch, sf_dtype)).float()
    mantissa, exponent = torch.frexp(scale)
    # frexp gives scale == mantissa * 2**exponent with mantissa in [0.5, 1), so an
    # exact power of two already sits at the rounded-up exponent.
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    rounded = torch.ldexp(torch.ones_like(scale), exponent.clamp(-127, 127))
    return torch.where(scale > 0, rounded, torch.zeros_like(scale))


def _row_quantize(torch, values, d_dtype, sf_dtype, sf_vec_size, norm_const, folded=False):
    """Upstream ``compute_reference_row_quant``: per-vector amax scale, then quantize."""
    rows, columns = values.shape
    padded_columns = spec.align_up(columns, 128)
    padded_rows = spec.align_up(rows, 128)
    padded = torch.zeros((padded_rows, padded_columns), dtype=torch.float32, device=values.device)
    padded[:rows, :columns] = values
    vectors = padded.reshape(padded_rows, padded_columns // sf_vec_size, sf_vec_size)
    # The multiply order is load-bearing. Upstream's row path evaluates
    # ``amax * rcp_limit * norm_const`` left to right while its column path folds
    # the two constants into one factor first, and E8M0's round-toward-positive-
    # infinity turns a single ULP between the two into a whole binade whenever
    # the exact product lands on a power of two.
    amax = vectors.abs().amax(dim=2)
    limit = _rcp_limit(d_dtype)
    scaled = amax * (limit * norm_const) if folded else amax * limit * norm_const
    scale = _quantize_scale(torch, scaled, sf_dtype)
    reciprocal = torch.where(scale > 0, norm_const / scale, torch.zeros_like(scale))
    expanded = reciprocal.unsqueeze(-1).expand_as(vectors).reshape(padded_rows, padded_columns)
    quantized = (padded * expanded).to(_torch_dtype(torch, d_dtype)).float()
    return quantized[:rows, :columns], scale


def reference_outputs(data):
    """FP32 reference for every output this specialization produces."""
    import torch

    config = data["config"]
    derived = data["derived"]
    N = derived["N"]
    gemm = _grouped_gemm(torch, data)
    scaled_c = data["c"]["values"][:, :, 0].clone()
    # The upstream epilogue passes beta to dswiglu and dsituglu but not to dgeglu,
    # which consumes C unscaled.
    if config["act"] != "dgeglu":
        for index, start, end in _expert_ranges(config):
            scaled_c[start:end, :] *= float(data["beta"][index].item())
    gate, up = _deinterleave(scaled_c, N)
    prob = data["prob"][:, 0, 0].unsqueeze(1)
    prob_values = prob if config["with_prob"] else torch.ones_like(prob)
    gate_grad, up_grad, prob_grad = _activation_backward(torch, config, gemm, gate, up, prob_values)

    outputs = {"d_row": _interleave(torch, gate_grad, up_grad, N)}
    if config["with_prob"]:
        outputs["dprob"] = prob_grad.reshape(prob_grad.shape[0], N // 32, 32).sum(dim=2).sum(dim=1)
    if derived["generate_dbias"]:
        dbias = torch.zeros((derived["L"], derived["n_out"]), dtype=torch.float32, device="cuda")
        for index, start, end in _expert_ranges(config):
            if end > start:
                dbias[index, :] = outputs["d_row"][start:end, :].sum(dim=0)
        outputs["dbias"] = dbias
    if derived["generate_amax"]:
        amax = torch.zeros((derived["L"], 2), dtype=torch.float32, device="cuda")
        for index, start, end in _expert_ranges(config):
            if end > start:
                amax[index, 0] = gate_grad[start:end, :].abs().amax()
                amax[index, 1] = up_grad[start:end, :].abs().amax()
        outputs["amax"] = amax
    if derived["generate_sfd"]:
        norm_const = float(data["norm_const"][0].item())
        unquantized = outputs["d_row"]
        quantized, row_scale = _row_quantize(
            torch,
            unquantized,
            config["d_dtype"],
            config["sf_dtype"],
            config["sf_vec_size"],
            norm_const,
        )
        column_quantized, column_scale = _row_quantize(
            torch,
            unquantized.transpose(0, 1).contiguous(),
            config["d_dtype"],
            config["sf_dtype"],
            config["sf_vec_size"],
            norm_const,
            folded=True,
        )
        outputs["d_row"] = quantized
        outputs["sfd_row_scale"] = row_scale
        outputs["d_col"] = column_quantized.transpose(0, 1)
        outputs["sfd_col_scale"] = column_scale
    return outputs


# ---------------------------------------------------------------------------
# Upstream kernel as reference
# ---------------------------------------------------------------------------


def load_reference_source():
    """Import the upstream kernel from the pinned cuDNN Frontend install.

    The dotted import runs ``grouped/__init__`` and ``dglu/__init__`` on the
    way down, which reach the compiled ``cudnn`` extension -- safe against the
    source install, and cached so the cost is paid once per process.
    """
    return load_reference_module(
        "cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias"
    )


def compile_reference(data):
    """Compile the upstream kernel over this input set and return a no-arg launch."""
    cutlass = import_cutlass_reference()
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver as cuda
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import make_fake_stream

    from tirx_kernels.cudnn._reference import from_dlpack_typed as from_dlpack

    if os.environ.get("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0") != "0":
        raise RuntimeError("CUDNNFE_CLUSTER_OVERLAP_MARGIN must be 0")
    module = load_reference_source()
    config = data["config"]
    derived = data["derived"]
    outputs = data["source"]
    discrete = config["weight_mode"] == "discrete"

    kernel = module.BlockScaledMoEGroupedGemmDgluDbiasKernel(
        sf_vec_size=config["sf_vec_size"],
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=derived["use_2cta"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
        vectorized_f32=config["vectorized_f32"],
        discrete_col_sfd=config["discrete_col_sfd"],
        expert_cnt=derived["L"],
        weight_mode=module.MoEWeightMode.DISCRETE if discrete else module.MoEWeightMode.DENSE,
        use_dynamic_sched=config["sched"] == "dynamic",
        act_func=config["act"],
        situ_beta1=config["situ_beta1"],
    )

    def dlpack(tensor, *, leading_dim=None, align=16):
        wrapped = from_dlpack(tensor, assumed_align=align)
        if leading_dim is not None:
            wrapped = wrapped.mark_layout_dynamic(leading_dim=leading_dim)
        return wrapped

    b_major_mode = OperandMajorMode.K if config["b_major"] == "k" else OperandMajorMode.MN
    b_stride_size = derived["K"] if config["b_major"] == "k" else derived["N"]

    workspace = outputs["workspace"]
    workspace_argument = (
        dlpack(workspace, align=128).iterator
        if workspace is not None
        else cute.runtime.nullptr(dtype=cutlass.Uint8, assumed_align=128)
    )

    if discrete:
        b_argument = dlpack(data["b_ptrs"], align=8).iterator
        sfb_argument = dlpack(data["sfb_ptrs"], align=8).iterator
    else:
        b_argument = dlpack(data["b"]["tensor"], leading_dim=1 if config["b_major"] == "k" else 0)
        sfb_argument = dlpack(data["sfb"]["tensor"])

    arguments = {
        "a": dlpack(data["a"]["tensor"], leading_dim=1),
        "b": b_argument,
        "sfb": sfb_argument,
        "n": cutlass.Int32(derived["N"]),
        "k": cutlass.Int32(derived["K"]),
        "b_stride_size": cutlass.Int64(b_stride_size),
        "b_major_mode": b_major_mode,
        "workspace_ptr": workspace_argument,
        "c": dlpack(data["c"]["tensor"], leading_dim=1),
        "d": dlpack(outputs["d_row"]["tensor"], leading_dim=1),
        "d_col": dlpack(outputs["d_col"]["tensor"], leading_dim=1),
        "sfa": dlpack(data["sfa"]["tensor"]),
        "sfd_row_tensor": dlpack(outputs["sfd_row"]["tensor"]) if outputs["sfd_row"] else None,
        "sfd_col_tensor": dlpack(outputs["sfd_col"]["tensor"]) if outputs["sfd_col"] else None,
        "amax_tensor": dlpack(outputs["amax"]) if outputs["amax"] is not None else None,
        "norm_const_tensor": dlpack(data["norm_const"]) if derived["generate_sfd"] else None,
        "padded_offsets": dlpack(data["padded_offsets"]),
        "alpha": dlpack(data["alpha"]),
        "beta": dlpack(data["beta"]),
        "prob": dlpack(data["prob"]) if config["with_prob"] else None,
        "dprob": dlpack(outputs["dprob"]) if config["with_prob"] else None,
        "dbias_tensor": dlpack(outputs["dbias"]) if outputs["dbias"] is not None else None,
        "max_active_clusters": derived["grid"][2],
        "stream": make_fake_stream(use_tvm_ffi_env_stream=False),
        "linear_offset": config["linear_offset"],
        "geglu_alpha": config["geglu_alpha"],
        "glu_clamp_max": config["glu_clamp_max"],
        "glu_clamp_min": config["glu_clamp_min"],
        "situ_beta1": config["situ_beta1"],
        "situ_beta2": config["situ_beta2"],
    }
    executable = cute.compile(kernel, **arguments, options="--enable-tvm-ffi")

    # ``b``/``sfb``/``workspace_ptr`` lower to raw device pointers, so the runtime
    # call takes addresses where the compile-time call took cute iterators.
    runtime = [
        data["a"]["tensor"],
        int(data["b_ptrs"].data_ptr()) if discrete else data["b"]["tensor"],
        int(data["sfb_ptrs"].data_ptr()) if discrete else data["sfb"]["tensor"],
        derived["N"],
        derived["K"],
        b_stride_size,
        int(workspace.data_ptr()) if workspace is not None else 0,
        data["c"]["tensor"],
        outputs["d_row"]["tensor"],
        outputs["d_col"]["tensor"],
        data["sfa"]["tensor"],
        outputs["sfd_row"]["tensor"] if outputs["sfd_row"] else None,
        outputs["sfd_col"]["tensor"] if outputs["sfd_col"] else None,
        outputs["amax"],
        data["norm_const"] if derived["generate_sfd"] else None,
        data["padded_offsets"],
        data["alpha"],
        data["beta"],
        data["prob"] if config["with_prob"] else None,
        outputs["dprob"] if config["with_prob"] else None,
        outputs["dbias"],
    ]
    # Every parameter is positional in the compiled wrapper, including the ones
    # this specialization compiled away as None.
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def launch():
        executable(*runtime, stream)

    launch._keep_alive = (executable, runtime, stream)
    return launch


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Upstream ``check_ref_grouped_gemm_dswiglu`` tolerances.
_BASE_ATOL = 1e-1
_BASE_RTOL = 1e-2
_FP8_D_TOLERANCE = {"float8_e4m3fn": 0.125, "float8_e5m2": 0.25}


def _assert_close(torch, actual, expected, *, atol, rtol, label):
    actual = actual.float()
    expected = expected.float()
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if bool(torch.all(torch.isclose(actual, expected, atol=atol, rtol=rtol)).item()):
        return
    absolute = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    worst = int(torch.argmax(absolute - allowed).item())
    raise AssertionError(
        f"{label} mismatch: worst absolute error {float(absolute.reshape(-1)[worst].item())} "
        f"exceeds {float(allowed.reshape(-1)[worst].item())}"
    )


def _sfd_logical(buffer, rows, columns):
    """Read an SF buffer back through the interleaved-atom mapping."""
    atoms = buffer["storage"].float()
    batch, rest_m, rest_k = atoms.shape[0], atoms.shape[1], atoms.shape[2]
    logical = atoms.permute(0, 1, 4, 3, 2, 5).reshape(batch, rest_m * 128, rest_k * 4)[0]
    return logical[:rows, :columns]


def _dbias_tolerance(torch, config, derived, reference):
    """Upstream scaled tolerance: BF16 atomics accumulate one partial per tile."""
    tiles_per_expert = max(
        1,
        max(
            spec.ceil_div(spec.align_up(rows, spec.FIX_PAD_SIZE), 128)
            for rows in config["group_m_list"]
        ),
    ) * spec.ceil_div(derived["N"], 256)
    magnitude = float(reference.abs().amax().item())
    return max(magnitude * 0.008 * (tiles_per_expert**0.5), _BASE_ATOL)


def validate_outputs(data, *, sources=("tirx",), with_oracle=True):
    """Compare the named output sets against the FP32 oracle and each other."""
    import torch

    config = data["config"]
    derived = data["derived"]
    reference = reference_outputs(data) if with_oracle else None
    tolerance = _FP8_D_TOLERANCE.get(config["d_dtype"], _BASE_ATOL)
    for name in sources:
        outputs = data[name]
        if with_oracle:
            _assert_close(
                torch,
                outputs["d_row"]["tensor"][:, :, 0],
                reference["d_row"],
                atol=tolerance,
                rtol=max(_BASE_RTOL, tolerance),
                label=f"{name} D_row versus FP32 oracle",
            )
            if config["with_prob"]:
                _assert_close(
                    torch,
                    outputs["dprob"][:, 0, 0],
                    reference["dprob"],
                    atol=_BASE_ATOL,
                    rtol=_BASE_RTOL,
                    label=f"{name} dprob versus FP32 oracle",
                )
            if derived["generate_dbias"]:
                _assert_close(
                    torch,
                    outputs["dbias"][:, :, 0],
                    reference["dbias"],
                    atol=_dbias_tolerance(torch, config, derived, reference["dbias"]),
                    rtol=_BASE_RTOL,
                    label=f"{name} dbias versus FP32 oracle",
                )
            if derived["generate_amax"]:
                _assert_close(
                    torch,
                    outputs["amax"][:, :, 0],
                    reference["amax"],
                    atol=_BASE_ATOL,
                    rtol=_BASE_RTOL,
                    label=f"{name} amax versus FP32 oracle",
                )
            if derived["generate_sfd"]:
                _assert_close(
                    torch,
                    outputs["d_col"]["tensor"][:, :, 0],
                    reference["d_col"],
                    atol=tolerance,
                    rtol=max(_BASE_RTOL, tolerance),
                    label=f"{name} D_col versus FP32 oracle",
                )
                row_scale = reference["sfd_row_scale"]
                _assert_close(
                    torch,
                    _sfd_logical(outputs["sfd_row"], row_scale.shape[0], row_scale.shape[1]),
                    row_scale,
                    atol=_BASE_ATOL,
                    rtol=_BASE_RTOL,
                    label=f"{name} SFD_row versus FP32 oracle",
                )
                if name == "tirx" or not config["discrete_col_sfd"]:
                    # TIRx always writes the column scale factors in the
                    # transposed layout, so the oracle check applies to it
                    # unconditionally. The upstream kernel only lands there
                    # without ``discrete_col_sfd``; see ``_compare_against_source``.
                    column_scale = reference["sfd_col_scale"]
                    _assert_close(
                        torch,
                        _sfd_logical(
                            outputs["sfd_col"], column_scale.shape[0], column_scale.shape[1]
                        ),
                        column_scale,
                        atol=_BASE_ATOL,
                        rtol=_BASE_RTOL,
                        label=f"{name} SFD_col versus FP32 oracle",
                    )
    if "tirx" in sources and "source" in sources:
        _compare_against_source(torch, data, tolerance)


def _compare_against_source(torch, data, tolerance):
    """Every output the specialization writes, TIRx against the upstream kernel.

    The one exception is SFD_col under ``discrete_col_sfd``; the loop below says
    why.
    """
    config = data["config"]
    config = data["config"]
    derived = data["derived"]
    tirx, source = data["tirx"], data["source"]
    _assert_close(
        torch,
        tirx["d_row"]["tensor"][:, :, 0],
        source["d_row"]["tensor"][:, :, 0],
        atol=tolerance,
        rtol=max(_BASE_RTOL, tolerance),
        label="TIRx D_row versus source D_row",
    )
    if config["with_prob"]:
        _assert_close(
            torch,
            tirx["dprob"][:, 0, 0],
            source["dprob"][:, 0, 0],
            atol=_BASE_ATOL,
            rtol=_BASE_RTOL,
            label="TIRx dprob versus source dprob",
        )
    if derived["generate_dbias"]:
        _assert_close(
            torch,
            tirx["dbias"][:, :, 0],
            source["dbias"][:, :, 0],
            atol=_dbias_tolerance(torch, config, derived, source["dbias"][:, :, 0].float()),
            rtol=_BASE_RTOL,
            label="TIRx dbias versus source dbias",
        )
    if derived["generate_amax"]:
        _assert_close(
            torch,
            tirx["amax"][:, :, 0],
            source["amax"][:, :, 0],
            atol=_BASE_ATOL,
            rtol=_BASE_RTOL,
            label="TIRx amax versus source amax",
        )
    if derived["generate_sfd"]:
        _assert_close(
            torch,
            tirx["d_col"]["tensor"][:, :, 0],
            source["d_col"]["tensor"][:, :, 0],
            atol=tolerance,
            rtol=max(_BASE_RTOL, tolerance),
            label="TIRx D_col versus source D_col",
        )
        names = ["sfd_row"]
        if not config["discrete_col_sfd"]:
            # Upstream declares SFD_col with the row-shaped layout under
            # ``discrete_col_sfd`` but stores into it through an MN-chunk
            # partition, so the bytes land in neither layout's positions and its
            # own test leaves that buffer unchecked. TIRx writes the transposed
            # layout the FP32 oracle defines, and the check just above holds it
            # to that exactly. There is nothing well-defined to compare against
            # here, so this comparison covers the shapes where upstream's own
            # declaration and store agree.
            names.append("sfd_col")
        for name in names:
            _assert_close(
                torch,
                tirx[name]["storage"].float(),
                source[name]["storage"].float(),
                atol=_BASE_ATOL,
                rtol=_BASE_RTOL,
                label=f"TIRx {name} versus source {name}",
            )


def tirx_launch(executables, data):
    """Bind the TIRx entries' flat byte-array ABI to this input set.

    ``executables`` is one compiled module per PrimFunc that ``get_kernel``
    returned: the optional per-expert descriptor pre-kernel followed by the main
    kernel, launched in that order on the same stream exactly as the upstream
    ``__call__`` launches them.
    """
    import torch

    config = data["config"]
    derived = data["derived"]
    outputs = data["tirx"]
    if not isinstance(executables, list | tuple):
        executables = [executables]
    helper_arguments = None
    if derived["needs_helper"]:
        helper_arguments = [
            data["b_ptrs"] if config["weight_mode"] == "discrete" else data["padded_offsets"],
            data["sfb_ptrs"] if config["weight_mode"] == "discrete" else data["padded_offsets"],
            outputs["workspace"],
        ]
    main_arguments = [
        data["a"]["raw"],
        data["b_ptrs"] if config["weight_mode"] == "discrete" else data["b"]["raw"],
        data["sfa"]["raw"],
        data["sfb_ptrs"] if config["weight_mode"] == "discrete" else data["sfb"]["raw"],
        data["c"]["raw"],
        outputs["d_row"]["raw"],
    ]
    if derived["generate_sfd"]:
        main_arguments += [
            outputs["d_col"]["raw"],
            outputs["sfd_row"]["raw"],
            outputs["sfd_col"]["raw"],
            data["norm_const"],
        ]
    if derived["generate_amax"]:
        main_arguments.append(outputs["amax"].reshape(-1))
    main_arguments += [data["padded_offsets"], data["alpha"], data["beta"]]
    if config["with_prob"]:
        # The launch ABI takes these flat; the (rows, 1, 1) shape is the upstream
        # tensor contract, not the kernel's.
        main_arguments += [data["prob"].reshape(-1), outputs["dprob"].reshape(-1)]
    if derived["generate_dbias"]:
        main_arguments.append(outputs["dbias"].view(torch.uint8).reshape(-1))
    if outputs["workspace"] is not None:
        main_arguments.append(outputs["workspace"])

    if helper_arguments is None:
        calls = [(executables[-1], main_arguments)]
    else:
        calls = [(executables[0], helper_arguments), (executables[-1], main_arguments)]

    def launch():
        for executable, arguments in calls:
            executable(*arguments)

    launch._keep_alive = (executables, main_arguments, helper_arguments)
    return launch
