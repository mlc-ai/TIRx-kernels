# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=2 precomputed-gate decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t2_precomputed_split4.cu``, symbol
``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- a frozen machine-generated
export ("Generated from a recurrent-KDA Loom schedule") whose SHA256 it declares
in its own header.

This is the WY-transform schedule: 2 warps, a 14720-byte shared-memory arena,
``ldmatrix`` + ``mma.sync`` tensor-core products, and a per-token checkpoint
contract -- structurally unrelated to the one-warp ``t1_direct`` sibling, which
serves only as a contract reference.

Dispatch reaches this body whenever ``num_tokens == 2`` with
``num_spec_tokens == 1`` (MTP with one draft token). The sm100a split selector
returns 4 unconditionally at T=2 (``recurrent_kda.py:1170-1192``), so
``d128_t2_precomputed_split4`` is the only reachable -- and the only locally
measurable -- specialization; ``_specialization`` raises for anything else.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

# --- source pinning -------------------------------------------------------
# The generated body declares its own digest at
# flashkda_decode_d128_t2_precomputed_split4.cu:19-20. Two digests are printed
# there, "Raw" and "Normalized", and they differ for this export; the hash of
# the bytes between the BEGIN/END markers is the *raw* one.
FROZEN_FLASHINFER_COMMIT = "f2e04400"
FROZEN_BODY_SHA256 = "e0cf2ece1b50e851579df8822fa2f03262a3399e25643a6bd3df03c875cc5ea2"

HEAD_DIM = 128
NUM_TOKENS = 2
VALUE_SPLIT = 4  # the sm100a selector returns 4 unconditionally at T = 2
L2_EPS = 1.0e-6
LOG2_E = 1.4426950408889634

THREADS = 64  # 2 warps (flash_kda_decode.py:_variant_metadata)
ROWS_PER_CTA = HEAD_DIM // VALUE_SPLIT  # 32 value rows per CTA
ROWS_PER_THREAD = 8  # each of the 64 threads owns 8 rows x 8 keys
K_PER_THREAD = 8

# Shared-memory arena, byte offsets copied from the source's #define block
# (.cu:45-87). The source declares it `extern __shared__ __align__(1024)` and
# the launcher passes SMEM_TOTAL dynamically; the port declares the same bytes
# with the same alignment as a static arena, which needs no launch tag.
OFF_SSTATE0 = 0
OFF_SSTATE1 = 4096
OFF_SVEC = 8192
OFF_SK = 12288
OFF_SD = 13312
OFF_SBETA = 14336
OFF_SSLOT = 14344
OFF_STOKEN = 14352
OFF_SINIT = 14360
OFF_SL = 14376
OFF_SR = 14392
OFF_SU = 14408
SMEM_TOTAL = 14720
# sGramA0/sGramA1 (offsets 8192/10240) alias sVec and are dead at T=2/split4 --
# they belong to the t5/t6 coefficient-gram schedules. Not declared here.

# TIRX_TRANSCRIBE_START flashkda_decode_t2_precomputed


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=2 export bench."""
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 32,
        "pool_slack": 6,  # state_slots = num_seqs*T + slack (upstream bench)
        "padded_seqs": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "accepted": None,  # None -> all ones (initial slot = ssm_idx[n, 0])
        "scale": None,
        "seed": 20260813,
    }
    config.update(overrides)
    return config


# Every T=2 shape selects split4, so the matrix varies only (HV, H, N).
# hv32h16 mirrors FlashInfer's own export bench exactly.
BENCH_CONFIGS = [
    _case("hv32h16_b8_t2", num_seqs=8),
    _case("hv32h16_b16_t2", num_seqs=16),
    _case("hv32h16_b32_t2", num_seqs=32),
    _case("hv32h16_b64_t2", num_seqs=64),
    _case("hv32h16_b128_t2", num_seqs=128),
    _case("hv16h16_b8_t2", num_seqs=8, num_heads=16, num_value_heads=16),
    _case("hv16h16_b64_t2", num_seqs=64, num_heads=16, num_value_heads=16),
    _case("hv12h12_b8_t2", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_t2", num_seqs=64, num_heads=12, num_value_heads=12),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # num_accepted_tokens is live at T=2 (unlike t1_direct): it selects the
    # initial checkpoint slot as ssm_idx[n*2 + clamp(nat-1, 0, 1)]. The upstream
    # test sweeps 0/1/2/9, covering both clamp edges (.cu:279-286).
    _case("hv32h16_b8_t2_nat0", num_seqs=8, accepted="zeros"),
    _case("hv32h16_b8_t2_nat2", num_seqs=8, accepted="twos"),
    _case("hv32h16_b8_t2_nat9", num_seqs=8, accepted="nines"),
    _case("hv32h16_b8_t2_natmix", num_seqs=8, accepted="mixed"),
    # CUDA-graph padding: ssm_state_indices == -1 rows must zero their output
    # bit-exactly and leave their state slots untouched (.cu:501-530).
    _case("hv32h16_b8_t2_padded", num_seqs=8, padded_seqs=2),
    # Envelope-strided state pool and gate.
    _case("hv32h16_b8_t2_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv32h16_b8_t2_gstride", num_seqs=8, gate_token_stride_pad=8),
    # Non-default scale; GQA ratio 4; the smallest grid.
    _case("hv32h16_b8_t2_scale", num_seqs=8, scale=0.05),
    _case("hv64h16_b8_t2", num_seqs=8, num_heads=16, num_value_heads=64),
    _case("hv32h16_b1_t2", num_seqs=1),
]

KERNEL_META = {
    "name": "flashkda_decode_t2_precomputed",
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

    # The sm100a policy is `if num_tokens == 2: return 4` with no shape
    # dependence, so there is nothing to select -- but assert it, because a
    # different split would be a different kernel.
    if VALUE_SPLIT != 4:
        raise ValueError("only d128_t2_precomputed_split4 is in this port's scope")

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
        "STATE_INDEX_ELEMENTS": num_seqs * NUM_TOKENS,
        "NAT_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t2_precomputed(
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
    """FlashKDA "cake" T=2 precomputed-gate decode -- scaffold body.

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
    """Return the specialized FlashKDA cake T=2 decode PrimFunc."""
    return _flashkda_decode_t2_precomputed.specialize(**_specialization(kwargs))


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case.

    The kernel clamps `nat - 1` into [0, 1] and uses it to pick the initial
    checkpoint slot, so 0 and 1 both select index 0 while 2 and anything larger
    select index 1 (.cu:279-286).
    """
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "zeros":
        return torch.zeros(num_seqs, device=device, dtype=torch.int32)
    if kind == "twos":
        return torch.full((num_seqs,), 2, device=device, dtype=torch.int32)
    if kind == "nines":
        return torch.full((num_seqs,), 9, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream test's sweep, tiled to the batch size
        # (test_recurrent_kda_decode_export.py:1269-1282).
        pattern = torch.tensor([0, 1, 2, 9, 1, 0, 1, 2], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=2 decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FlashKDA cake decode targets compute capability 10.x, got {capability}")

    spec = _specialization(kwargs)
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

    # Packed [1, N*2, ...] layout -- the form the upstream tests and benches use
    # (recurrent_kda.py:1770-1795).
    q = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    k = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    v = randn(1, total_tokens, num_value_heads, HEAD_DIM, gain=0.5)
    # GATE_KIND == 0: g holds the per-K log-gate computed on the host; the
    # kernel applies exp() to it.
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

    # cu_seqlens steps by exactly NUM_TOKENS per row; the body assumes this
    # stride-T contract and does not bound-check token_base.
    cu_seqlens = torch.arange(0, total_tokens + 1, NUM_TOKENS, device=device, dtype=torch.int32)
    # Slot 0 is deliberately never a checkpoint target, matching the upstream
    # bench (bench_recurrent_kda_decode_export.py:232-236).
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
    import flashinfer  # local: keep kernel discovery free of optional deps

    root = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    return os.path.join(root, "csrc", "kda", "flashkda_decode_d128_t2_precomputed_split4.cu")


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
    """Run the frozen cake export itself on the reference state pool.

    Uses the JIT module's direct ABI rather than
    ``kda_decode.recurrent_kda(backend="cake")`` so the comparison is
    kernel-only and every metadata combination (negative slots, arbitrary
    num_accepted_tokens) is reachable.
    """
    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    device = case["device"]
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        target = "sm100f" if torch.version.cuda and torch.version.cuda >= "12.9" else "sm100a"
    elif (major, minor) == (10, 3):
        # Non-direct variants are not exported for sm103a
        # (flash_kda_decode.py:213-217).
        target = "sm100f"
    else:
        raise SkipTest(f"FlashKDA cake decode has no export for compute capability {major}.{minor}")

    module = get_flash_kda_decode_module("d128_t2_precomputed_split4", target)
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
    """Independent FP32 oracle of the T=2 delta rule.

    Written as the sequential recurrence the WY transform computes in blocked
    form, which is the point: it agrees only if the WY algebra is right. It
    keeps the kernel's two structural choices -- the recurrent state advances in
    FP32 across both tokens (only the stored checkpoint is rounded to bf16), and
    the initial slot is selected by the clamped num_accepted_tokens.
    """
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
            # FP32 carry across both tokens; only the checkpoint store rounds.
            s = state[init_slot, hv].float()
            for t in range(NUM_TOKENS):
                row = n * NUM_TOKENS + t
                qn = q[row, h] * (torch.rsqrt(q[row, h].pow(2).sum() + L2_EPS) * scale)
                kn = k[row, h] * torch.rsqrt(k[row, h].pow(2).sum() + L2_EPS)
                gamma = torch.exp(g[row, hv])

                decayed = s * gamma.unsqueeze(0)
                pred = decayed @ kn
                u = v[row, hv] - pred
                delta = u * beta[row, hv]
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
        "flashkda_decode_t2_precomputed is at the scaffold stage; the kernel body "
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
        "flashkda_decode_t2_precomputed is at the scaffold stage; benchmarking is "
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
