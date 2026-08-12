# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=1 precomputed-gate decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t1_precomputed_direct_split16.cu`` (and
its ``_split8`` sibling), symbol ``kernel_flashinfer_recurrent_kda_t1_direct``.
Both bodies are frozen machine-generated exports ("Generated from a recurrent-KDA
Loom schedule"); FlashInfer asserts their SHA256 in
``tests/jit/test_flash_kda_decode_jit.py``.

The two exports differ in exactly three lines -- ``value_splits``,
``tile_row_base``'s multiplier, and the ``row_block`` trip count -- so this module
carries a single ``VALUE_SPLIT`` constexpr and derives the rest.

Dispatch (``flashinfer/kda_kernels/recurrent_kda.py:1443-1447``) routes every
``num_tokens == 1`` precomputed call to ``d128_t1_precomputed_direct_split{16,8}``;
the 679-line non-direct ``t1_precomputed_split{1,2,4,8}`` exports are unreachable
through ``run_recurrent_kda`` and are out of this port's scope.

Shape of the kernel: one warp per CTA, grid ``(HV * VALUE_SPLIT, num_seqs)``, zero
shared memory, no barrier, no atomic, no workspace. The value split partitions the
128 value rows into disjoint slabs, so there is no cross-CTA combine of any kind.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

# --- source pinning -------------------------------------------------------
# The generated body carries its own digest at
# flashkda_decode_d128_t1_precomputed_direct_split16.cu:19-20; run_test asserts
# it so a silent upstream regeneration cannot pass unnoticed.
FROZEN_FLASHINFER_COMMIT = "f2e04400"
FROZEN_BODY_SHA256 = {16: "9119163b3b5cb6a8760b6a17a7ce01788a0e1c0078f8c812255225f27e5989e5"}

HEAD_DIM = 128
L2_EPS = 1.0e-6
# 16 lanes cover the 128-wide K dimension, 8 elements each; the remaining lane
# bit selects one of two interleaved value rows (direct_split16.cu:69-70).
K_LANES = 16
V_LANES = 2
ELEMS_PER_LANE = HEAD_DIM // 32  # 4: the pre-redistribution q/k/g slice
K_PER_LANE = HEAD_DIM // K_LANES  # 8: the post-redistribution slice
ROWS_PER_BLOCK = V_LANES * 4  # 8 value rows per row_block iteration

# TIRX_TRANSCRIBE_START flashkda_decode_t1_precomputed


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's cake decode export bench."""
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 16,
        "pool_size": 512,
        "padded_slots": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "scale": None,
        "seed": 20260812,
    }
    config.update(overrides)
    return config


# The split is chosen by `work = num_seqs * num_value_heads` against
# `2 * sm_count` (recurrent_kda.py:1170-1176); on a 148-SM B200 the boundary is
# 296. Every row below names the split its shape actually selects, and
# verify_dispatch.py asserts that against the real selector.
BENCH_CONFIGS = [
    # HV == H == 16, split16 side (work = 16, 64, 128, 256).
    _case("hv16h16_b1_s16", num_seqs=1),
    _case("hv16h16_b4_s16", num_seqs=4),
    _case("hv16h16_b8_s16", num_seqs=8),
    _case("hv16h16_b16_s16", num_seqs=16),
    # HV == H == 16, split8 side (work = 512, 1024, 2048).
    _case("hv16h16_b32_s8", num_seqs=32),
    _case("hv16h16_b64_s8", num_seqs=64),
    _case("hv16h16_b128_s8", num_seqs=128),
    # Kimi K3 TP8 per-rank head count, one row per split side (work = 96, 768).
    _case("hv12h12_b8_s16", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_s8", num_seqs=64, num_heads=12, num_value_heads=12),
    # GQA ratio 2, matching FlashInfer's own export bench (work = 256, 1024).
    _case("hv32h16_b8_s16", num_seqs=8, num_heads=16, num_value_heads=32),
    _case("hv32h16_b32_s8", num_seqs=32, num_heads=16, num_value_heads=32),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # Split knife edge on a 148-SM B200: work = 296 takes split16, 304 takes
    # split8 (tests/kda/test_recurrent_kda_decode_export.py:311-320).
    _case("hv16h16_b18_s16_edge", num_seqs=18),
    _case("hv16h16_b19_s8_edge", num_seqs=19),
    # CUDA-graph padding: negative ssm_state_indices rows must leave the state
    # untouched and zero their output (direct_split16.cu:76-78, :251-266).
    _case("hv16h16_b16_s16_padded", num_seqs=16, padded_slots=5),
    # Envelope-strided state pool: slot stride may exceed HV*V*K while staying
    # 8-element aligned (binding_common.cuh:148-161).
    _case("hv16h16_b16_s16_strided", num_seqs=16, slot_stride_pad=16),
    # Envelope-strided gate: g is the only input whose token stride is free
    # (CheckCompactGate, binding_common.cuh:137-146).
    _case("hv16h16_b16_s16_gstride", num_seqs=16, gate_token_stride_pad=256),
    # Non-default scale (the host passes `scale or 1/sqrt(128)`).
    _case("hv16h16_b8_s16_scale", num_seqs=8, scale=0.05),
    # GQA ratio 4 on the split8 side.
    _case("hv64h16_b16_s8", num_seqs=16, num_heads=16, num_value_heads=64),
]

KERNEL_META = {
    "name": "flashkda_decode_t1_precomputed",
    "category": "flashinfer",
    "compute_capability": 10,
}


def _sm_count(kwargs: dict[str, Any]) -> int:
    """SM count driving the split policy; overridable for deterministic tests."""
    override = kwargs.get("sm_count")
    if override is not None:
        return int(override)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required to resolve the FlashKDA decode value split")
    device = kwargs.get("device", "cuda")
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _select_value_split(work: int, sm_count: int) -> int:
    """Reproduce ``_select_flash_kda_decode_value_split_current`` for T = 1.

    recurrent_kda.py:1170-1176. `work` is `num_seqs * num_value_heads`; the T = 1
    branch is the whole policy on sm100a, and only 16 and 8 are reachable.
    """
    return 16 if work <= 2 * sm_count else 8


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    num_seqs = int(kwargs["num_seqs"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    pool_size = int(kwargs["pool_size"])
    slot_stride_pad = int(kwargs.get("slot_stride_pad", 0))
    gate_token_stride_pad = int(kwargs.get("gate_token_stride_pad", 0))

    if num_value_heads % num_heads != 0 or num_value_heads < num_heads:
        raise ValueError("num_value_heads must be a multiple of num_heads and >= it")
    if not 0 < num_seqs <= 65535:
        raise ValueError("num_seqs must fit grid.y (binding_common.cuh:284-287)")
    if slot_stride_pad % 8 != 0:
        raise ValueError("state slot stride padding must stay 8-element aligned")
    if gate_token_stride_pad % 4 != 0:
        raise ValueError("gate token stride padding must stay 4-element aligned")

    value_split = _select_value_split(num_seqs * num_value_heads, _sm_count(kwargs))
    rows_per_cta = HEAD_DIM // value_split

    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM + slot_stride_pad
    gate_token_stride = num_value_heads * HEAD_DIM + gate_token_stride_pad
    return {
        "NUM_SEQS": num_seqs,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "HEAD_RATIO": num_value_heads // num_heads,
        "VALUE_SPLIT": value_split,
        # direct_split{16,8}.cu:79 -- tile_row_base = value_tile * rows_per_cta.
        "TILE_ROW_STRIDE": rows_per_cta,
        # direct_split{16,8}.cu:182 -- the `#pragma unroll 1` row_block trip count.
        "ROW_BLOCKS": rows_per_cta // ROWS_PER_BLOCK,
        "STATE_SLOT_STRIDE": slot_stride,
        "GATE_TOKEN_STRIDE": gate_token_stride,
        "Q_ELEMENTS": num_seqs * num_heads * HEAD_DIM,
        "V_ELEMENTS": num_seqs * num_value_heads * HEAD_DIM,
        "GATE_ELEMENTS": num_seqs * gate_token_stride,
        "BETA_ELEMENTS": num_seqs * num_value_heads,
        "STATE_ELEMENTS": pool_size * slot_stride,
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t1_precomputed(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    cu_seqlens_h: T.handle,
    ssm_state_indices_h: T.handle,
    scale: T.float32,
    *,
    NUM_SEQS: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_VALUE_HEADS: T.constexpr,
    HEAD_RATIO: T.constexpr,
    VALUE_SPLIT: T.constexpr,
    TILE_ROW_STRIDE: T.constexpr,
    ROW_BLOCKS: T.constexpr,
    STATE_SLOT_STRIDE: T.constexpr,
    GATE_TOKEN_STRIDE: T.constexpr,
    Q_ELEMENTS: T.constexpr,
    V_ELEMENTS: T.constexpr,
    GATE_ELEMENTS: T.constexpr,
    BETA_ELEMENTS: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    CU_SEQLENS_ELEMENTS: T.constexpr,
    STATE_INDEX_ELEMENTS: T.constexpr,
):
    """FlashKDA "cake" T=1 precomputed-gate decode -- scaffold body.

    The implementation is written in the correctness gate, after the kernel
    sketch has passed review.
    """
    T.func_attr({"global_symbol": "flashkda_decode_t1_precomputed", "tir.is_scheduled": 1})
    T.match_buffer(q_h, (Q_ELEMENTS,), "bfloat16")
    T.match_buffer(k_h, (Q_ELEMENTS,), "bfloat16")
    T.match_buffer(v_h, (V_ELEMENTS,), "bfloat16")
    T.match_buffer(g_h, (GATE_ELEMENTS,), "bfloat16")
    T.match_buffer(beta_h, (BETA_ELEMENTS,), "bfloat16")
    T.match_buffer(state_h, (STATE_ELEMENTS,), "bfloat16")
    T.match_buffer(out_h, (V_ELEMENTS,), "bfloat16")
    T.match_buffer(cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), "int32")
    T.match_buffer(ssm_state_indices_h, (STATE_INDEX_ELEMENTS,), "int32")


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=1 decode PrimFunc."""
    return _flashkda_decode_t1_precomputed.specialize(**_specialization(kwargs))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=1 decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FlashKDA cake decode targets compute capability 10.x, got {capability}")

    spec = _specialization({**kwargs, "device": device})
    num_seqs = spec["NUM_SEQS"]
    num_heads = spec["NUM_HEADS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    gate_token_stride = spec["GATE_TOKEN_STRIDE"]
    pool_size = int(kwargs["pool_size"])
    padded_slots = int(kwargs.get("padded_slots", 0))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(kwargs["seed"]))

    def randn(*shape: int, dtype=torch.bfloat16, gain: float = 1.0):
        raw = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        return (gain * raw).to(dtype)

    q = randn(1, num_seqs, num_heads, HEAD_DIM, gain=0.5)
    k = randn(1, num_seqs, num_heads, HEAD_DIM, gain=0.5)
    v = randn(1, num_seqs, num_value_heads, HEAD_DIM, gain=0.5)
    # GATE_KIND == 0: `g` holds the per-K log-gate computed OUTSIDE the kernel,
    # which applies exp() to it. The upstream test builds it exactly this way
    # (tests/kda/test_recurrent_kda_decode_export.py:986-1002).
    gate_logits = torch.randn(
        (1, num_seqs, num_value_heads, HEAD_DIM), device=device, dtype=torch.float32, generator=gen
    )
    g_dense = torch.nn.functional.logsigmoid(gate_logits).to(torch.bfloat16)
    # `g` is the only input whose token stride is free; an envelope-strided case
    # exercises the g_stride_token argument.
    g_raw = torch.zeros((num_seqs * gate_token_stride,), device=device, dtype=torch.bfloat16)
    g = g_raw.as_strided(
        (1, num_seqs, num_value_heads, HEAD_DIM),
        (num_seqs * gate_token_stride, gate_token_stride, HEAD_DIM, 1),
    )
    g.copy_(g_dense)
    # beta arrives pre-sigmoided (kda_decode.py:97-98); the kernel reads it raw.
    beta = torch.sigmoid(randn(1, num_seqs, num_value_heads, dtype=torch.float32, gain=0.5)).to(
        torch.bfloat16
    )

    cu_seqlens = torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
    slots = torch.arange(num_seqs, device=device, dtype=torch.int32)
    if padded_slots:
        slots[num_seqs - padded_slots :] = -1
    num_accepted_tokens = torch.ones(num_seqs, device=device, dtype=torch.int32)

    def make_state_pool() -> tuple[torch.Tensor, torch.Tensor]:
        raw = randn(pool_size * slot_stride, gain=0.01)
        view = raw.as_strided(
            (pool_size, num_value_heads, HEAD_DIM, HEAD_DIM),
            (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        )
        return raw, view

    tirx_state_raw, tirx_state = make_state_pool()
    reference_state_raw = tirx_state_raw.clone()
    reference_state = reference_state_raw.as_strided(
        (pool_size, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    initial_state_raw = tirx_state_raw.clone()

    tirx_out = torch.empty(
        (1, num_seqs, num_value_heads, HEAD_DIM), device=device, dtype=torch.bfloat16
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
        "ssm_state_indices": slots,
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
        float(case["scale"]),
    )


def _source_body_path(value_split: int) -> str:
    """Absolute path of the frozen generated body this port transcribes."""
    import flashinfer  # local: keep kernel discovery free of optional deps

    root = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    return os.path.join(
        root, "csrc", "kda", f"flashkda_decode_d128_t1_precomputed_direct_split{value_split}.cu"
    )


def assert_frozen_source(value_split: int) -> None:
    """Fail loudly if the upstream generated body was regenerated.

    The body declares its own digest at direct_split16.cu:19-20; this checks the
    same bytes the port was transcribed from.
    """
    expected = FROZEN_BODY_SHA256.get(value_split)
    if expected is None:
        return
    path = _source_body_path(value_split)
    with open(path, "rb") as handle:
        text = handle.read().decode()
    marker = "// BEGIN FROZEN GENERATED BODY\n"
    start = text.index(marker) + len(marker)
    end = text.index("// END FROZEN GENERATED BODY")
    digest = hashlib.sha256(text[start:end].encode()).hexdigest()
    if digest != expected:
        raise AssertionError(
            f"{path}: frozen body digest {digest} != pinned {expected}; the upstream "
            "export was regenerated and this port must be re-verified against it"
        )


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the frozen cake export itself on the reference state pool.

    Uses the JIT module's direct ABI rather than
    ``kda_decode.recurrent_kda(backend="cake")`` so the comparison is kernel-only
    and so inactive (negative-slot) rows -- which the public T=1 wrapper cannot
    express, since it synthesizes identity metadata -- are reachable.
    """
    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    spec = case["spec"]
    value_split = spec["VALUE_SPLIT"]
    variant = f"d128_t1_precomputed_direct_split{value_split}"

    device = case["device"]
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        target = "sm100f" if torch.version.cuda and torch.version.cuda >= "12.9" else "sm100a"
    elif (major, minor) == (10, 3):
        target = "sm103a"
    else:
        raise SkipTest(f"FlashKDA cake decode has no export for compute capability {major}.{minor}")

    module = get_flash_kda_decode_module(variant, target)
    reference_out = torch.empty_like(case["tirx_out"])
    dummy_f32 = torch.ones(1, device=device, dtype=torch.float32)

    module.run(
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        dummy_f32,
        dummy_f32,
        case["reference_state"],
        reference_out,
        case["cu_seqlens"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        float(case["scale"]),
        0.0,
        int(torch.cuda.current_stream(device).cuda_stream),
    )
    torch.cuda.synchronize(device)
    return reference_out


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle of the source math (direct_split16.cu:133-267)."""
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
    slots = case["ssm_state_indices"]

    state_raw = case["initial_state_raw"].clone()
    state = state_raw.as_strided(
        (state_raw.numel() // slot_stride, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    out = torch.zeros(
        (num_seqs, num_value_heads, HEAD_DIM), device=case["device"], dtype=torch.float32
    )

    scale = float(case["scale"])
    for n in range(num_seqs):
        slot = int(slots[n].item())
        if slot < 0:
            continue
        for hv in range(num_value_heads):
            h = hv // head_ratio
            qn = q[n, h] * (torch.rsqrt(q[n, h].pow(2).sum() + L2_EPS) * scale)
            kn = k[n, h] * torch.rsqrt(k[n, h].pow(2).sum() + L2_EPS)
            gamma = torch.exp(g[n, hv])
            kq = torch.dot(kn, qn)

            s = state[slot, hv].float()
            decayed = s * gamma.unsqueeze(0)
            pred = decayed @ kn
            base = decayed @ qn
            delta = (v[n, hv] - pred) * beta[n, hv]
            state[slot, hv] = (decayed + delta.unsqueeze(1) * kn.unsqueeze(0)).to(torch.bfloat16)
            out[n, hv] = base + delta * kq

    return out.to(torch.bfloat16).unsqueeze(0), state_raw


def run_test(**kwargs: Any) -> None:
    """Correctness entry point. Implemented in the correctness gate."""
    raise SkipTest(
        "flashkda_decode_t1_precomputed is at the scaffold stage; the kernel body "
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
        "flashkda_decode_t1_precomputed is at the scaffold stage; benchmarking is "
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
