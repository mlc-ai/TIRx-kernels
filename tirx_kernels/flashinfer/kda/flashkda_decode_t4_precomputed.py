# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=4 precomputed-gate decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t4_precomputed_split2.cu``, symbol
``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- the same frozen WY
template as the ported T=2 sibling, re-instantiated at TOKENS=4 and
VALUE_SPLIT=2.

Dispatch reaches this body whenever ``num_tokens == 4`` with
``num_spec_tokens == 3``. The sm100a selector returns value split 2
unconditionally at T=4 (``recurrent_kda.py:1179-1180``), so
``d128_t4_precomputed_split2`` is the only reachable -- and only locally
measurable -- specialization.

Relative to T=2 the geometry doubles (128 threads = 4 warps, 64 value rows per
CTA, a 25856-byte arena) and three things that were dead at T=2 come alive: MMA
accumulator columns 2,3 and 6,7, the ``quad_base + 1`` broadcasts, and the
``u_lo``/``u_hi`` quad broadcasts -- Phase F now runs on ``lane_quad >= 2``,
where lane ``4f+2`` writes tokens 0,1 and lane ``4f+3`` writes tokens 2,3, so
the WY residuals have to cross the quad.

Helper vocabulary is shared with the T=2 module; only the geometry constants,
the frozen digest and the kernel body are per-specialization.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

from . import flashkda_decode_t2_precomputed as _t2

# Shared helper vocabulary (identical PTX forms, identical swizzle -- verified
# against all three bodies).
_ptx_un = _t2._ptx_un
_ptx_bin = _t2._ptx_bin
_ptx_ter = _t2._ptx_ter
_mul = _t2._mul
_add = _t2._add
_sub = _t2._sub
_fma = _t2._fma
_rsqrt = _t2._rsqrt
_expf = _t2._expf
_shfl_bfly = _t2._shfl_bfly
_shfl_idx = _t2._shfl_idx
_load_i32 = _t2._load_i32
_load_bf16_f32 = _t2._load_bf16_f32
_widen_lo = _t2._widen_lo
_widen_hi = _t2._widen_hi
_load_u32x2 = _t2._load_u32x2
_load_u32x4 = _t2._load_u32x4
_store_u32x4 = _t2._store_u32x4
_pack_bf16x2 = _t2._pack_bf16x2
_store_f32_as_bf16 = _t2._store_f32_as_bf16
_swz = _t2._swz
_st_shared_f32 = _t2._st_shared_f32
_ld_shared_f32 = _t2._ld_shared_f32
_st_shared_i32 = _t2._st_shared_i32
_ld_shared_i32 = _t2._ld_shared_i32
_ld_shared_f32x4 = _t2._ld_shared_f32x4
_st_shared_u32x4 = _t2._st_shared_u32x4
_st_shared_b16 = _t2._st_shared_b16
_ldmatrix_x4 = _t2._ldmatrix_x4
_mma_zero = _t2._mma_zero
_mma_acc = _t2._mma_acc

# --- source pinning -------------------------------------------------------
# Raw and normalized digests differ for this export; the hash of the bytes
# between the BEGIN/END markers is the raw one (verified).
FROZEN_FLASHINFER_COMMIT = "f2e04400"
FROZEN_BODY_SHA256 = "6ee50e300a2f20f305252a49e3e84fde957542ccb97b7fcb7ba7f075c7733077"

HEAD_DIM = _t2.HEAD_DIM
NUM_TOKENS = 4
VALUE_SPLIT = 2  # the sm100a selector returns 2 unconditionally at T = 4
L2_EPS = _t2.L2_EPS
LOG2_E = _t2.LOG2_E

THREADS = 128  # 4 warps (flash_kda_decode.py:_variant_metadata)
ROWS_PER_CTA = HEAD_DIM // VALUE_SPLIT  # 64 value rows per CTA
ROWS_PER_THREAD = 8  # each of the 128 threads owns 8 rows x 8 keys
K_PER_THREAD = 8

# Arena offsets copied from the source's #define block (.cu:45-88).
OFF_SSTATE0 = 0
OFF_SSTATE1 = 8192
OFF_SVEC = 16384
OFF_SK = 20480
OFF_SD = 22528
OFF_SBETA = 24576
OFF_SSLOT = 24592
OFF_STOKEN = 24608
OFF_SINIT = 24624
OFF_SL = 24640
OFF_SR = 24704
OFF_SU = 24768
SMEM_TOTAL = 25856
# sGramA0/sGramA1 alias sVec and are t5/t6 machinery -- dead here.

# TIRX_TRANSCRIBE_START flashkda_decode_t4_precomputed


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=4 export bench."""
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 32,
        "pool_slack": 6,
        "padded_seqs": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "accepted": None,
        "scale": None,
        "seed": 20260814,
    }
    config.update(overrides)
    return config


# Every T=4 shape selects split2, so the matrix varies only (HV, H, N).
# hv32h16 mirrors FlashInfer's own export bench exactly.
BENCH_CONFIGS = [
    _case("hv32h16_b8_t4", num_seqs=8),
    _case("hv32h16_b16_t4", num_seqs=16),
    _case("hv32h16_b32_t4", num_seqs=32),
    _case("hv32h16_b64_t4", num_seqs=64),
    _case("hv32h16_b128_t4", num_seqs=128),
    _case("hv16h16_b8_t4", num_seqs=8, num_value_heads=16),
    _case("hv16h16_b64_t4", num_seqs=64, num_value_heads=16),
    _case("hv12h12_b8_t4", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_t4", num_seqs=64, num_heads=12, num_value_heads=12),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # nat selects the initial checkpoint slot as ssm_idx[n*4 + clamp(nat-1, 0, 3)];
    # the upstream test sweeps 0/1/4/11, covering both clamp edges (.cu:283-286).
    _case("hv32h16_b8_t4_nat0", num_seqs=8, accepted="zeros"),
    _case("hv32h16_b8_t4_nat4", num_seqs=8, accepted="fours"),
    _case("hv32h16_b8_t4_nat11", num_seqs=8, accepted="elevens"),
    _case("hv32h16_b8_t4_natmix", num_seqs=8, accepted="mixed"),
    _case("hv32h16_b8_t4_padded", num_seqs=8, padded_seqs=2),
    _case("hv32h16_b8_t4_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv32h16_b8_t4_gstride", num_seqs=8, gate_token_stride_pad=8),
    _case("hv32h16_b8_t4_scale", num_seqs=8, scale=0.05),
    _case("hv64h16_b8_t4", num_seqs=8, num_value_heads=64),
    _case("hv32h16_b1_t4", num_seqs=1),
]

KERNEL_META = {
    "name": "flashkda_decode_t4_precomputed",
    "category": "flashinfer",
    "compute_capability": 10,
}


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    num_seqs = int(kwargs["num_seqs"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    slot_stride_pad = int(kwargs.get("slot_stride_pad", 0))
    gate_token_stride_pad = int(kwargs.get("gate_token_stride_pad", 0))
    pool_slack = int(kwargs.get("pool_slack", 6))

    if num_value_heads % num_heads != 0 or num_value_heads < num_heads:
        raise ValueError("num_value_heads must be a multiple of num_heads and >= it")
    if not 0 < num_seqs <= 65535:
        raise ValueError("num_seqs must fit grid.y (binding_common.cuh:284-287)")
    if slot_stride_pad % 8 != 0:
        raise ValueError("state slot stride padding must stay 8-element aligned")
    if gate_token_stride_pad % 4 != 0:
        raise ValueError("gate token stride padding must stay 4-element aligned")
    if VALUE_SPLIT != 2:
        raise ValueError("only d128_t4_precomputed_split2 is in this port's scope")

    total_tokens = num_seqs * NUM_TOKENS
    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM + slot_stride_pad
    gate_token_stride = num_value_heads * HEAD_DIM + gate_token_stride_pad
    state_slots = total_tokens + pool_slack
    return {
        "NUM_SEQS": num_seqs,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "HEAD_RATIO": num_value_heads // num_heads,
        "STATE_SLOT_STRIDE": slot_stride,
        "GATE_TOKEN_STRIDE": gate_token_stride,
        "Q_ELEMENTS": total_tokens * num_heads * HEAD_DIM,
        "V_ELEMENTS": total_tokens * num_value_heads * HEAD_DIM,
        "GATE_ELEMENTS": total_tokens * gate_token_stride,
        "BETA_ELEMENTS": total_tokens * num_value_heads,
        "STATE_ELEMENTS": state_slots * slot_stride,
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": total_tokens,
        "NAT_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t4_precomputed(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    cu_seqlens_h: T.handle,
    ssm_state_indices_h: T.handle,
    num_accepted_tokens_h: T.handle,
    scale: T.float32,
    *,
    NUM_SEQS: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_VALUE_HEADS: T.constexpr,
    HEAD_RATIO: T.constexpr,
    STATE_SLOT_STRIDE: T.constexpr,
    GATE_TOKEN_STRIDE: T.constexpr,
    Q_ELEMENTS: T.constexpr,
    V_ELEMENTS: T.constexpr,
    GATE_ELEMENTS: T.constexpr,
    BETA_ELEMENTS: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    CU_SEQLENS_ELEMENTS: T.constexpr,
    STATE_INDEX_ELEMENTS: T.constexpr,
    NAT_ELEMENTS: T.constexpr,
):
    """FlashKDA "cake" T=4 precomputed-gate decode -- scaffold body.

    The implementation is written in the correctness gate, after the kernel
    sketch has passed review.
    """
    T.match_buffer(q_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    T.match_buffer(k_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    T.match_buffer(v_h, (V_ELEMENTS,), "bfloat16", scope="global")
    T.match_buffer(g_h, (GATE_ELEMENTS,), "bfloat16", scope="global")
    T.match_buffer(beta_h, (BETA_ELEMENTS,), "bfloat16", scope="global")
    T.match_buffer(state_h, (STATE_ELEMENTS,), "bfloat16", scope="global")
    T.match_buffer(out_h, (V_ELEMENTS,), "bfloat16", scope="global")
    T.match_buffer(cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), "int32", scope="global")
    T.match_buffer(ssm_state_indices_h, (STATE_INDEX_ELEMENTS,), "int32", scope="global")
    T.match_buffer(num_accepted_tokens_h, (NAT_ELEMENTS,), "int32", scope="global")


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=4 decode PrimFunc."""
    return _flashkda_decode_t4_precomputed.specialize(**_specialization(kwargs))


_NAT_PATTERNS = {None: None, "zeros": 0, "fours": 4, "elevens": 11}


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case; the kernel clamps nat-1 into [0, 3]."""
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream test's sweep, tiled to the batch size.
        pattern = torch.tensor([0, 1, 4, 11, 3, 0, 1, 4], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    if kind in _NAT_PATTERNS:
        return torch.full((num_seqs,), _NAT_PATTERNS[kind], device=device, dtype=torch.int32)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Same shape as the T=2 recipe at T=4: packed [1, N*4, ...] bf16, a log-space
    gate, pre-sigmoided beta, flat [N*4] slot indices. Written locally rather
    than parameterizing the already-merged T=2 module.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=4 decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FlashKDA cake decode targets compute capability 10.x, got {capability}")

    spec = _specialization({**kwargs, "device": device})
    num_seqs = spec["NUM_SEQS"]
    num_heads = spec["NUM_HEADS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    gate_token_stride = spec["GATE_TOKEN_STRIDE"]
    total_tokens = num_seqs * NUM_TOKENS
    state_slots = spec["STATE_ELEMENTS"] // slot_stride
    padded_seqs = int(kwargs.get("padded_seqs", 0))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(kwargs["seed"]))

    def randn(*shape: int, dtype=torch.bfloat16, gain: float = 1.0):
        raw = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        return (gain * raw).to(dtype)

    q = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    k = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    v = randn(1, total_tokens, num_value_heads, HEAD_DIM, gain=0.5)
    # GATE_KIND == 0: g holds the per-K log-gate; the kernel applies exp().
    gate_logits = torch.randn(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        device=device,
        dtype=torch.float32,
        generator=gen,
    )
    g_dense = torch.nn.functional.logsigmoid(gate_logits).to(torch.bfloat16)
    g_raw = torch.zeros((total_tokens * gate_token_stride,), device=device, dtype=torch.bfloat16)
    g = g_raw.as_strided(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        (total_tokens * gate_token_stride, gate_token_stride, HEAD_DIM, 1),
    )
    g.copy_(g_dense)
    beta = torch.sigmoid(randn(1, total_tokens, num_value_heads, dtype=torch.float32, gain=0.5)).to(
        torch.bfloat16
    )

    cu_seqlens = torch.arange(0, total_tokens + 1, NUM_TOKENS, device=device, dtype=torch.int32)
    # Slot 0 is deliberately never a checkpoint target (upstream bench recipe).
    slots = torch.arange(1, total_tokens + 1, device=device, dtype=torch.int32).reshape(
        num_seqs, NUM_TOKENS
    )
    if padded_seqs:
        slots[num_seqs - padded_seqs :, :] = -1
    ssm_state_indices = slots.contiguous().view(-1)
    num_accepted_tokens = _accepted_tensor(kwargs.get("accepted"), num_seqs, device)

    def make_state_pool() -> tuple[torch.Tensor, torch.Tensor]:
        raw = randn(state_slots * slot_stride, gain=0.01)
        view = raw.as_strided(
            (state_slots, num_value_heads, HEAD_DIM, HEAD_DIM),
            (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        )
        return raw, view

    tirx_state_raw, tirx_state = make_state_pool()
    reference_state_raw = tirx_state_raw.clone()
    reference_state = reference_state_raw.as_strided(
        (state_slots, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    initial_state_raw = tirx_state_raw.clone()

    tirx_out = torch.empty(
        (1, total_tokens, num_value_heads, HEAD_DIM), device=device, dtype=torch.bfloat16
    )

    scale = kwargs.get("scale")
    return {
        "spec": spec,
        "config": dict(kwargs),
        "device": device,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "g_raw": g_raw,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "ssm_state_indices_2d": slots,
        "num_accepted_tokens": num_accepted_tokens,
        "tirx_state_raw": tirx_state_raw,
        "tirx_state": tirx_state,
        "reference_state_raw": reference_state_raw,
        "reference_state": reference_state,
        "initial_state_raw": initial_state_raw,
        "tirx_out": tirx_out,
        "scale": float(scale) if scale is not None else HEAD_DIM**-0.5,
    }


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["q"].reshape(-1),
        case["k"].reshape(-1),
        case["v"].reshape(-1),
        case["g_raw"],
        case["beta"].reshape(-1),
        case["tirx_state_raw"],
        case["tirx_out"].reshape(-1),
        case["cu_seqlens"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        float(case["scale"]),
    )


def _source_body_path() -> str:
    """Absolute path of the frozen generated body this port transcribes."""
    import flashinfer

    root = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    return os.path.join(root, "csrc", "kda", "flashkda_decode_d128_t4_precomputed_split2.cu")


def assert_frozen_source() -> None:
    """Fail loudly if the upstream generated body was regenerated."""
    path = _source_body_path()
    with open(path, "rb") as handle:
        text = handle.read().decode()
    marker = "// BEGIN FROZEN GENERATED BODY\n"
    start = text.index(marker) + len(marker)
    end = text.index("// END FROZEN GENERATED BODY")
    digest = hashlib.sha256(text[start:end].encode()).hexdigest()
    if digest != FROZEN_BODY_SHA256:
        raise AssertionError(
            f"{path}: frozen body digest {digest} != pinned {FROZEN_BODY_SHA256}; the "
            "upstream export was regenerated and this port must be re-verified"
        )


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the frozen cake export itself on the reference state pool."""
    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    device = case["device"]
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        target = "sm100f" if torch.version.cuda and torch.version.cuda >= "12.9" else "sm100a"
    elif (major, minor) == (10, 3):
        target = "sm100f"  # non-direct variants are never built for sm103a
    else:
        raise SkipTest(f"no FlashKDA cake export for compute capability {major}.{minor}")

    module = get_flash_kda_decode_module("d128_t4_precomputed_split2", target)
    reference_out = torch.empty_like(case["tirx_out"])
    dummy_f32 = torch.ones(1, device=device, dtype=torch.float32)

    module.run(
        case["q"], case["k"], case["v"], case["g"], case["beta"],
        dummy_f32, dummy_f32,
        case["reference_state"], reference_out,
        case["cu_seqlens"], case["ssm_state_indices"], case["num_accepted_tokens"],
        float(case["scale"]), 0.0,
        int(torch.cuda.current_stream(device).cuda_stream),
    )  # fmt: skip
    torch.cuda.synchronize(device)
    return reference_out


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle of the T=4 delta rule (sequential form)."""
    spec = case["spec"]
    num_seqs = spec["NUM_SEQS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    head_ratio = spec["HEAD_RATIO"]
    slot_stride = spec["STATE_SLOT_STRIDE"]

    q = case["q"][0].float()
    k = case["k"][0].float()
    v = case["v"][0].float()
    g = case["g"][0].float()
    beta = case["beta"][0].float()
    slots2d = case["ssm_state_indices_2d"]
    nat = case["num_accepted_tokens"]

    state_raw = case["initial_state_raw"].clone()
    state_slots = state_raw.numel() // slot_stride
    state = state_raw.as_strided(
        (state_slots, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    out = torch.zeros(
        (num_seqs * NUM_TOKENS, num_value_heads, HEAD_DIM),
        device=case["device"],
        dtype=torch.float32,
    )

    scale = float(case["scale"])
    for n in range(num_seqs):
        accepted = min(max(int(nat[n].item()) - 1, 0), NUM_TOKENS - 1)
        init_slot = int(slots2d[n, accepted].item())
        if init_slot < 0:
            init_slot = 0
        for hv in range(num_value_heads):
            h = hv // head_ratio
            # FP32 carry across all tokens; only the checkpoint store rounds.
            s = state[init_slot, hv].float()
            for t in range(NUM_TOKENS):
                row = n * NUM_TOKENS + t
                qn = q[row, h] * (torch.rsqrt(q[row, h].pow(2).sum() + L2_EPS) * scale)
                kn = k[row, h] * torch.rsqrt(k[row, h].pow(2).sum() + L2_EPS)
                gamma = torch.exp(g[row, hv])

                decayed = s * gamma.unsqueeze(0)
                delta = (v[row, hv] - decayed @ kn) * beta[row, hv]
                s = decayed + delta.unsqueeze(1) * kn.unsqueeze(0)
                out[row, hv] = s @ qn

                slot = int(slots2d[n, t].item())
                if slot >= 0:
                    state[slot, hv] = s.to(torch.bfloat16)
                else:
                    out[row, hv] = 0.0

    return out.to(torch.bfloat16).unsqueeze(0), state_raw


def run_test(**kwargs: Any) -> None:
    """Correctness entry point. Implemented in the correctness gate."""
    raise SkipTest(
        "flashkda_decode_t4_precomputed is at the scaffold stage; the kernel body "
        "is written in the correctness gate"
    )


def run_bench(
    *,
    warmup: float | None = None,
    repeat: float | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark entry point. Implemented in the performance gate."""
    raise SkipTest(
        "flashkda_decode_t4_precomputed is at the scaffold stage; benchmarking is "
        "enabled in the performance gate"
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "assert_frozen_source",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
