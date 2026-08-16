# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""In-tree Torch quantization and scale-layout primitives.

These functions define the input ABI consumed by the TIRx ports.  They are
deliberately ordinary Torch math: correctness and benchmark data preparation
must not import an optimized kernel package merely to construct operands.
"""

from __future__ import annotations

import torch


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 scales up to an exact power of two."""

    return torch.exp2(torch.ceil(torch.log2(x.abs().clamp_min(torch.finfo(torch.float32).tiny))))


def _ue8m0_bytes(scales: torch.Tensor) -> torch.Tensor:
    return (scales.float().contiguous().view(torch.int32) >> 23).to(torch.uint8)


def _scales_from_ue8m0_bytes(exponents: torch.Tensor) -> torch.Tensor:
    return torch.exp2(exponents.to(torch.int32).sub(127).float())


def _float_to_ue8m0(values: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 values with the UE8M0 round-up rule."""

    values = values.float()
    bits = values.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    encoded = (exponent + bump.to(torch.int32)).clamp_max(254)
    return torch.where(values > 0, encoded, 0).to(torch.uint8)


def _ue8m0_inverse(exponents: torch.Tensor) -> torch.Tensor:
    inverse = torch.exp2(127.0 - exponents.float())
    return torch.where(exponents != 0, inverse, 0.0)


def pack_ue8m0_words(scales: torch.Tensor) -> torch.Tensor:
    """Pack the last scale dimension as four UE8M0 bytes per word."""

    if scales.dtype != torch.float32:
        scales = scales.float()
    pad = (-scales.shape[-1]) % 4
    if pad:
        scales = torch.nn.functional.pad(scales, (0, pad), value=1.0)
    return _ue8m0_bytes(scales).contiguous().view(torch.uint32)


def unpack_ue8m0_words(words: torch.Tensor, *, count: int | None = None) -> torch.Tensor:
    """Decode packed UE8M0 words into exact FP32 powers of two."""

    if words.dtype not in (torch.int32, torch.uint32):
        raise ValueError("packed UE8M0 scales must use 32-bit integer words")
    scales = _scales_from_ue8m0_bytes(words.contiguous().view(torch.uint8))
    return scales if count is None else scales[..., :count]


def quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Encode one value per byte using NVIDIA FP4 E2M1 bit assignments."""

    magnitude = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=x.device, dtype=x.dtype
    )
    code = torch.bucketize(magnitude, boundaries).to(torch.uint8)
    # bucketize selects the lower code at every exact midpoint.  E2M1 uses
    # round-to-nearest-even, whose tie direction alternates with that code.
    midpoint = (magnitude.unsqueeze(-1) == boundaries).any(dim=-1)
    code += (midpoint & ((code & 1) != 0)).to(torch.uint8)
    # PTX's E2M1 conversion preserves the sign when a negative value rounds
    # to zero, so the packed ABI distinguishes +0 (0x0) from -0 (0x8).
    return code | ((x < 0).to(torch.uint8) << 3)


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    pairs = codes.view(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return ((pairs[..., 0] & 15) | ((pairs[..., 1] & 15) << 4)).contiguous()


def swizzle_sf(scales: torch.Tensor, layout: str) -> torch.Tensor:
    """Lay out logical ``[row, block]`` scale bytes for a quantization ABI."""

    if scales.dtype != torch.uint8 or scales.ndim != 2:
        raise ValueError("scale swizzle expects a 2-D uint8 tensor")
    rows, columns = scales.shape
    if layout == "linear":
        return scales.contiguous().view(-1)
    if layout not in ("128x4", "8x4"):
        raise ValueError(f"unsupported scale layout: {layout}")

    row_tile = 128 if layout == "128x4" else 8
    padded_rows = align_up(rows, row_tile)
    padded_columns = align_up(columns, 4)
    output = torch.zeros(padded_rows * padded_columns, dtype=torch.uint8, device=scales.device)
    row = torch.arange(rows, device=scales.device, dtype=torch.long)[:, None]
    column = torch.arange(columns, device=scales.device, dtype=torch.long)[None, :]
    if layout == "128x4":
        offset = (
            column % 4
            + (column // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + (row // 128) * (128 * padded_columns)
        )
    else:
        k_tiles = padded_columns // 4
        offset = (row // 8) * (k_tiles * 32) + (column // 4) * 32 + (row % 8) * 4 + column % 4
    output[offset.expand(rows, columns).reshape(-1)] = scales.reshape(-1)
    return output


def quantize_mxfp8(
    values: torch.Tensor, *, sf_layout: str = "linear"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch oracle for 32-value UE8M0-scaled E4M3 quantization."""

    rows, width = values.shape
    blocks = values.float().view(rows, width // 32, 32)
    scale_bytes = _float_to_ue8m0(blocks.abs().amax(dim=-1) / 448.0)
    inverse = _ue8m0_inverse(scale_bytes)
    quantized = (blocks * inverse.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.view(rows, width).view(torch.uint8), swizzle_sf(scale_bytes, sf_layout)


def quantize_mxfp4(
    values: torch.Tensor, *, sf_layout: str = "linear"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch oracle for 32-value UE8M0-scaled E2M1 quantization."""

    rows, width = values.shape
    blocks = values.float().view(rows, width // 32, 32)
    scale_bytes = _float_to_ue8m0(blocks.abs().amax(dim=-1) / 6.0)
    inverse = _ue8m0_inverse(scale_bytes)
    codes = quantize_e2m1(blocks * inverse.unsqueeze(-1)).view(rows, width)
    return _pack_e2m1(codes), swizzle_sf(scale_bytes, sf_layout)


def quantize_nvfp4(
    values: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    sf_layout: str = "linear",
    fuse_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch oracle for 16-value E4M3-scaled E2M1 quantization."""

    if fuse_silu:
        width = values.shape[1] // 2
        values = (
            torch.nn.functional.silu(values[:, :width].float()) * values[:, width:].float()
        ).to(values.dtype)
    rows, width = values.shape
    blocks = values.float().view(rows, width // 16, 16)
    scale = global_scale.float().reshape(-1, 1)
    if scale.shape[0] == 1:
        scale = scale.expand(rows, 1)
    sf = (blocks.abs().amax(dim=-1) * scale / 6.0).to(torch.float8_e4m3fn)
    sf_bytes = sf.view(torch.uint8)
    decoded_sf = sf.float()
    output_scale = torch.where(decoded_sf != 0, scale / decoded_sf, 0.0)
    codes = quantize_e2m1(blocks * output_scale.unsqueeze(-1)).view(rows, width)
    return _pack_e2m1(codes), swizzle_sf(sf_bytes, sf_layout)


def quantize_nvfp4_per_token(
    values: torch.Tensor, global_scale_inverse: torch.Tensor, *, sf_layout: str = "linear"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch oracle for NVFP4 quantization with one additional row scale."""

    scaled, sf_bytes, token_scale = _nvfp4_per_token_state(values, global_scale_inverse)
    codes = quantize_e2m1(scaled).view_as(values)
    return _pack_e2m1(codes), swizzle_sf(sf_bytes, sf_layout), token_scale


def _nvfp4_per_token_state(
    values: torch.Tensor, global_scale_inverse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the logical scaled values, FP8 scales, and token scales."""

    rows, width = values.shape
    row_amax = values.float().abs().amax(dim=1)
    token_scale = row_amax * global_scale_inverse.float().reshape(1)[0]
    encode_scale = torch.where(token_scale != 0, token_scale.reciprocal(), 0.0)
    blocks = values.float().view(rows, width // 16, 16)
    sf = (blocks.abs().amax(dim=-1) * encode_scale.unsqueeze(1) / 6.0).to(torch.float8_e4m3fn)
    sf_bytes = sf.view(torch.uint8)
    decoded_sf = sf.float()
    output_scale = torch.where(decoded_sf != 0, encode_scale.unsqueeze(1) / decoded_sf, 0.0)
    return blocks * output_scale.unsqueeze(-1), sf_bytes, token_scale


def nvfp4_per_token_scaled_values(
    values: torch.Tensor, global_scale_inverse: torch.Tensor
) -> torch.Tensor:
    """Return the exact-math scaled values used by the per-token oracle."""

    scaled, _, _ = _nvfp4_per_token_state(values, global_scale_inverse)
    return scaled.view_as(values)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack two E2M1 nibbles per byte in source element order."""

    codes = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device
    )
    codes[..., 0::2] = packed & 15
    codes[..., 1::2] = packed >> 4
    return codes


def decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=codes.device, dtype=torch.float32
    )
    code = codes.to(torch.long)
    values = magnitudes[code & 7]
    return torch.where((code & 8) != 0, -values, values)


def per_token_cast_to_fp4(
    x: torch.Tensor, *, gran_k: int = 128, packed_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2-D rows to packed E2M1 with one scale per ``gran_k`` values."""

    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError("FP4 per-token quantization requires an even-width 2-D tensor")
    rows, width = x.shape
    padded_width = align_up(width, gran_k)
    padded = torch.zeros((rows, padded_width), dtype=x.dtype, device=x.device)
    padded[:, :width] = x
    groups = padded.view(rows, -1, gran_k)
    scales = ceil_to_ue8m0(groups.abs().float().amax(dim=2).clamp_min(1.0e-4) / 6.0)
    codes = quantize_e2m1(groups / scales.unsqueeze(2)).view(rows, padded_width)
    pairs = codes.view(rows, padded_width // 2, 2)
    packed = (pairs[:, :, 0] & 15) | ((pairs[:, :, 1] & 15) << 4)
    scales_out: torch.Tensor = scales
    if packed_ue8m0:
        if scales.shape[1] % 4:
            raise ValueError("packed UE8M0 requires a whole number of four-byte words")
        scales_out = _ue8m0_bytes(scales).contiguous().view(torch.uint32)
    return packed[:, : width // 2].contiguous(), scales_out


def cast_back_from_fp4(
    packed: torch.Tensor, scales: torch.Tensor, *, gran_k: int = 128, packed_ue8m0: bool = False
) -> torch.Tensor:
    """Decode the exact values consumed by an FP4 MMA operand."""

    rows, half_width = packed.shape
    codes = torch.stack((packed & 15, (packed >> 4) & 15), dim=2).reshape(rows, -1)
    values = decode_e2m1(codes)
    if packed_ue8m0:
        if scales.dtype != torch.uint32:
            raise ValueError("packed UE8M0 scales must use torch.uint32")
        exponents = scales.contiguous().view(torch.uint8).view(rows, -1)
        scale_values = _scales_from_ue8m0_bytes(exponents)
    else:
        scale_values = scales.float()
    if scale_values.shape[1] * gran_k != values.shape[1]:
        raise ValueError("scale groups do not cover the FP4 operand width")
    return (values.view(rows, -1, gran_k) * scale_values.unsqueeze(2)).reshape(rows, half_width * 2)


def per_token_cast_to_fp8(
    x: torch.Tensor, *, gran_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError("FP8 per-token quantization requires a 2-D tensor")
    rows, width = x.shape
    padded_width = align_up(width, gran_k)
    padded = torch.zeros((rows, padded_width), dtype=x.dtype, device=x.device)
    padded[:, :width] = x
    groups = padded.view(rows, -1, gran_k)
    scales = ceil_to_ue8m0(groups.abs().float().amax(dim=2).clamp_min(1.0e-4) / 448.0)
    quantized = (groups / scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    return quantized.view(rows, padded_width)[:, :width].contiguous(), scales


def per_block_cast_to_fp8(
    x: torch.Tensor, *, gran_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError("FP8 per-block quantization requires a 2-D tensor")
    rows, width = x.shape
    padded = torch.zeros(
        (align_up(rows, gran_k), align_up(width, gran_k)), dtype=x.dtype, device=x.device
    )
    padded[:rows, :width] = x
    groups = padded.view(-1, gran_k, padded.shape[1] // gran_k, gran_k)
    scales = ceil_to_ue8m0(
        groups.abs().float().amax(dim=(1, 3), keepdim=True).clamp_min(1.0e-4) / 448.0
    )
    quantized = (groups / scales).to(torch.float8_e4m3fn).view_as(padded)
    return quantized[:rows, :width].contiguous(), scales.view(groups.shape[0], groups.shape[2])


def per_channel_cast_to_fp8(
    x: torch.Tensor, *, gran_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or x.shape[0] % gran_k:
        raise ValueError("FP8 per-channel quantization requires rows divisible by gran_k")
    rows, width = x.shape
    groups = x.view(-1, gran_k, width)
    scales = ceil_to_ue8m0(groups.abs().float().amax(dim=1).clamp_min(1.0e-4) / 448.0)
    return (groups / scales.unsqueeze(1)).to(torch.float8_e4m3fn).view(rows, width), scales


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, *, scale_dims: tuple[int, ...], use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(dim for dim in range(x.ndim) if dim not in set(scale_dims))
    amax = x.abs().float().amax(dim=reduce_dims, keepdim=True).clamp_min(1.0e-4)
    scales = amax / 448.0
    if use_ue8m0:
        scales = ceil_to_ue8m0(scales)
    return (x / scales).to(torch.float8_e4m3fn), scales.squeeze()


def cast_back_from_fp8(
    quantized: torch.Tensor, scales: torch.Tensor, *, gran_k: int = 128
) -> torch.Tensor:
    return (
        quantized.float().view(quantized.shape[0], -1, gran_k) * scales.float().unsqueeze(2)
    ).view_as(quantized)


def pack_ue8m0_mn_major(scales: torch.Tensor) -> torch.Tensor:
    """Pack FP32 power-of-two scales into DeepGEMM's SM100 TMA layout."""

    if scales.dtype != torch.float32 or scales.ndim not in (2, 3):
        raise ValueError("scale packing expects a 2-D or 3-D float32 tensor")
    remove_batch = scales.ndim == 2
    source = scales.unsqueeze(0) if remove_batch else scales
    batch, mn, k_groups = source.shape
    aligned_mn = align_up(mn, 4)
    aligned_k = align_up(k_groups, 4)
    padded = torch.zeros((batch, aligned_mn, aligned_k), dtype=torch.uint8, device=scales.device)
    padded[:, :mn, :k_groups] = _ue8m0_bytes(source)
    words = padded.contiguous().view(torch.int32).view(batch, aligned_mn, aligned_k // 4)
    storage = torch.empty(
        (batch, aligned_k // 4, aligned_mn), dtype=torch.int32, device=scales.device
    )
    transposed = storage.transpose(1, 2)
    transposed.copy_(words)
    result = transposed[:, :mn]
    return result.squeeze(0) if remove_batch else result


def transform_sf(
    scales: torch.Tensor, *, mn: int, gran_mn: int, num_groups: int | None = None
) -> torch.Tensor:
    """Broadcast block-row scales, then pack them into the canonical TMA layout."""

    expected_ndim = 3 if num_groups is not None else 2
    if scales.ndim != expected_ndim:
        raise ValueError(f"expected {expected_ndim}-D scale tensor, got {scales.ndim}-D")
    if gran_mn != 1:
        row_index = torch.arange(mn, device=scales.device) // gran_mn
        scales = scales.index_select(-2, row_index)
    return pack_ue8m0_mn_major(scales.float())


def pack_k_grouped_ue8m0(
    scales: torch.Tensor, ks: list[int], *, mn: int, gran_k: int
) -> torch.Tensor:
    """Pack K-grouped per-channel scales as consecutive K-major group slabs."""

    chunks = []
    offset = 0
    for k in ks:
        rows = ceil_div(k, gran_k)
        chunk = scales[offset : offset + rows]
        offset += rows
        if rows:
            chunks.append(pack_ue8m0_mn_major(chunk.transpose(0, 1)).transpose(0, 1))
    if not chunks:
        return torch.empty((0, align_up(mn, 4)), dtype=torch.int32, device=scales.device)
    return torch.cat(chunks, dim=0)


__all__ = [
    "align_up",
    "cast_back_from_fp4",
    "cast_back_from_fp8",
    "ceil_div",
    "decode_e2m1",
    "nvfp4_per_token_scaled_values",
    "pack_k_grouped_ue8m0",
    "pack_ue8m0_mn_major",
    "pack_ue8m0_words",
    "per_block_cast_to_fp8",
    "per_channel_cast_to_fp8",
    "per_custom_dims_cast_to_fp8",
    "per_token_cast_to_fp4",
    "per_token_cast_to_fp8",
    "quantize_e2m1",
    "quantize_mxfp4",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_nvfp4_per_token",
    "swizzle_sf",
    "transform_sf",
    "unpack_e2m1",
    "unpack_ue8m0_words",
]
