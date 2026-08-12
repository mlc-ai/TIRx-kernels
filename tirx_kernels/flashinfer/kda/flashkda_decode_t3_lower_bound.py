# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=3 lower-bound decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t3_lower_bound_split4.cu``, symbol
``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- the same frozen WY
template as the T=2 and T=4 siblings, re-instantiated at TOKENS=3 with
**GATE_KIND = 1**.

T=3 bypasses the value-split selector entirely: the dispatcher early-returns
``d128_t3_lower_bound_split4`` from the ``is_t3_lower_bound`` predicate
(``recurrent_kda.py:1285-1294``, ``:1437-1438``). Its legal domain is narrow and
enforced host-side -- ``num_spec_tokens == 2``, a finite ``lower_bound < 0``,
``A_log``/``dt_bias`` present, ``N in {1,2,4,8,16}`` and ``H == HV == 16``
(``:1352-1359``, ``:1406-1416``).

Two things make this body different from every cake kernel ported so far:

* **the gate is computed in-kernel** (GATE_KIND 1), so ``A_log``, ``dt_bias``
  and ``lower_bound`` are live arguments rather than the dummies the
  precomputed variants pass. Per element the source computes
  ``lower_bound / (1 + exp(-exp(A_log[h]) * (g + dt_bias[h*128 + k])))`` and
  then exponentiates it -- three transcendental sites against T=2's one, and
  the family's only division.
* **warp 2 is a token-only warp.** At 96 threads with 32 value rows the
  state-side bounds are still 2 warps and 4 groups (they derive from rows/16
  and rows/8, not from T), so warp 2 runs the token preprocess and its share of
  the WY coefficients, hits both barriers, and then does nothing: it never
  gathers state, never issues an MMA, never stores. Both CTA barriers are
  therefore genuine three-warp edges.

Helper vocabulary is shared with the T=2 module; only the geometry constants,
the gate, the frozen digest and the kernel body are per-specialization.
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
FROZEN_BODY_SHA256 = "70c838b717a9eb7765bf9291db786b2b7a0387fbf80a3c337d5c04e7a553fe21"

HEAD_DIM = _t2.HEAD_DIM
NUM_TOKENS = 3
VALUE_SPLIT = 4  # T = 3 bypasses the selector; split4 is the whole surface
L2_EPS = _t2.L2_EPS
LOG2_E = _t2.LOG2_E

THREADS = 96  # 3 warps (flash_kda_decode.py:_variant_metadata)
ROWS_PER_CTA = HEAD_DIM // VALUE_SPLIT  # tile_row_base = value_tile * 32 (.cu:159)
ROWS_PER_THREAD = 8  # groups 0..3 (warps 0,1) own 8 rows x 8 keys each
K_PER_THREAD = 8

# Arena offsets copied from the source's #define block (.cu:45-88). sState and sVec keep the T=2 offsets exactly.
OFF_SSTATE0 = 0
OFF_SSTATE1 = 4096
OFF_SVEC = 8192
OFF_SK = 12288
OFF_SD = 13824
OFF_SBETA = 15360
OFF_SSLOT = 15372
OFF_STOKEN = 15384
OFF_SINIT = 15396
OFF_SL = 15412
OFF_SR = 15448
OFF_SU = 15484
SMEM_TOTAL = 15872
# sGramA0/sGramA1 alias sVec and are t5/t6 machinery -- dead here.

# TIRX_TRANSCRIBE_START flashkda_decode_t3_lower_bound


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=3 export bench.

    H and HV are pinned at 16 and cannot vary: the dispatcher rejects anything
    else (recurrent_kda.py:1352-1359).
    """
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 16,
        "pool_slack": 6,
        "padded_seqs": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "accepted": None,
        "lower_bound": -5.0,
        "scale": None,
        "seed": 20260814,
    }
    config.update(overrides)
    return config


# T=3's legal domain IS the matrix: the dispatcher enforces H == HV == 16 and
# N in {1,2,4,8,16}, so these five rows are every shape the kernel can serve.
# They match FlashInfer's own T=3 export bench exactly.
BENCH_CONFIGS = [
    _case("hv16h16_b1_t3", num_seqs=1),
    _case("hv16h16_b2_t3", num_seqs=2),
    _case("hv16h16_b4_t3", num_seqs=4),
    _case("hv16h16_b8_t3", num_seqs=8),
    _case("hv16h16_b16_t3", num_seqs=16),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # nat selects the initial checkpoint slot as ssm_idx[n*3 + clamp(nat-1, 0, 2)];
    # the upstream test sweeps 0/1/3/10, covering both clamp edges (.cu:287-296).
    _case("hv16h16_b8_t3_nat0", num_seqs=8, accepted="zeros"),
    _case("hv16h16_b8_t3_nat3", num_seqs=8, accepted="threes"),
    _case("hv16h16_b8_t3_nat10", num_seqs=8, accepted="tens"),
    _case("hv16h16_b8_t3_natmix", num_seqs=8, accepted="mixed"),
    _case("hv16h16_b8_t3_padded", num_seqs=8, padded_seqs=2),
    _case("hv16h16_b8_t3_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv16h16_b8_t3_gstride", num_seqs=8, gate_token_stride_pad=8),
    _case("hv16h16_b8_t3_scale", num_seqs=8, scale=0.05),
    # The gate is computed in-kernel, so lower_bound is a real input: a second
    # value must move the output.
    _case("hv16h16_b8_t3_lb1", num_seqs=8, lower_bound=-1.0),
]

KERNEL_META = {
    "name": "flashkda_decode_t3_lower_bound",
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

    # The dispatcher rejects anything outside this box (recurrent_kda.py:1352-1359).
    if num_heads != 16 or num_value_heads != 16:
        raise ValueError("t3 lower-bound requires num_heads == num_value_heads == 16")
    if num_seqs not in (1, 2, 4, 8, 16):
        raise ValueError("t3 lower-bound requires num_seqs in {1, 2, 4, 8, 16}")
    if slot_stride_pad % 8 != 0:
        raise ValueError("state slot stride padding must stay 8-element aligned")
    if gate_token_stride_pad % 4 != 0:
        raise ValueError("gate token stride padding must stay 4-element aligned")
    if VALUE_SPLIT != 4:
        raise ValueError("only d128_t3_lower_bound_split4 is in this port's scope")

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
        "A_LOG_ELEMENTS": num_heads,
        "DT_BIAS_ELEMENTS": num_heads * HEAD_DIM,
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": total_tokens,
        "NAT_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t3_lower_bound(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    a_log_h: T.handle,
    dt_bias_h: T.handle,
    cu_seqlens_h: T.handle,
    ssm_state_indices_h: T.handle,
    num_accepted_tokens_h: T.handle,
    scale: T.float32,
    lower_bound: T.float32,
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
    A_LOG_ELEMENTS: T.constexpr,
    DT_BIAS_ELEMENTS: T.constexpr,
    CU_SEQLENS_ELEMENTS: T.constexpr,
    STATE_INDEX_ELEMENTS: T.constexpr,
    NAT_ELEMENTS: T.constexpr,
):
    """FlashKDA "cake" T=3 lower-bound decode -- scaffold body.

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
    T.match_buffer(a_log_h, (A_LOG_ELEMENTS,), "float32", scope="global")
    T.match_buffer(dt_bias_h, (DT_BIAS_ELEMENTS,), "float32", scope="global")
    T.match_buffer(cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), "int32", scope="global")
    T.match_buffer(ssm_state_indices_h, (STATE_INDEX_ELEMENTS,), "int32", scope="global")
    T.match_buffer(num_accepted_tokens_h, (NAT_ELEMENTS,), "int32", scope="global")


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=3 decode PrimFunc."""
    return _flashkda_decode_t3_lower_bound.specialize(**_specialization(kwargs))


_NAT_PATTERNS = {"zeros": 0, "threes": 3, "tens": 10}


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case; the kernel clamps nat-1 into [0, 2]."""
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream test's sweep, tiled to the batch size.
        pattern = torch.tensor([0, 1, 3, 10, 2, 0, 1, 3], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    if kind in _NAT_PATTERNS:
        return torch.full((num_seqs,), _NAT_PATTERNS[kind], device=device, dtype=torch.int32)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Follows the upstream T=3 recipe: packed [1, N*3, ...] bf16, a RAW
    pre-gate g (NOT log-space -- the kernel computes the gate itself), plus the
    fp32 A_log / dt_bias the lower-bound gate consumes.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=3 decode")
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
    # GATE_KIND == 1: g is the RAW pre-gate input; the kernel derives the gate
    # from it with A_log, dt_bias and lower_bound (upstream recipe
    # test_recurrent_kda_decode_export.py:_make_t3_lower_bound_case).
    g_dense = torch.randn(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        device=device,
        dtype=torch.float32,
        generator=gen,
    ).to(torch.bfloat16)
    g_raw = torch.zeros((total_tokens * gate_token_stride,), device=device, dtype=torch.bfloat16)
    g = g_raw.as_strided(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        (total_tokens * gate_token_stride, gate_token_stride, HEAD_DIM, 1),
    )
    g.copy_(g_dense)
    beta = torch.sigmoid(randn(1, total_tokens, num_value_heads, dtype=torch.float32, gain=0.5)).to(
        torch.bfloat16
    )

    # gate_a = exp(A_log) lands in [1, 2) with this recipe.
    a_log = torch.log(
        torch.rand((num_heads,), device=device, dtype=torch.float32, generator=gen) + 1.0
    )
    dt_bias = torch.randn(
        (num_heads * HEAD_DIM,), device=device, dtype=torch.float32, generator=gen
    )
    lower_bound = float(kwargs.get("lower_bound", -5.0))
    if not lower_bound < 0.0:
        raise ValueError("t3 lower-bound requires a finite negative lower_bound")

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
        "a_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": lower_bound,
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
        case["a_log"],
        case["dt_bias"],
        case["cu_seqlens"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        float(case["scale"]),
        float(case["lower_bound"]),
    )


def _source_body_path() -> str:
    """Absolute path of the frozen generated body this port transcribes."""
    import flashinfer

    root = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    return os.path.join(root, "csrc", "kda", "flashkda_decode_d128_t3_lower_bound_split4.cu")


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

    module = get_flash_kda_decode_module("d128_t3_lower_bound_split4", target)
    reference_out = torch.empty_like(case["tirx_out"])

    # Unlike the precomputed variants, GATE_KIND == 1 dereferences A_log and
    # dt_bias and uses lower_bound.
    module.run(
        case["q"], case["k"], case["v"], case["g"], case["beta"],
        case["a_log"], case["dt_bias"],
        case["reference_state"], reference_out,
        case["cu_seqlens"], case["ssm_state_indices"], case["num_accepted_tokens"],
        float(case["scale"]), float(case["lower_bound"]),
        int(torch.cuda.current_stream(device).cuda_stream),
    )  # fmt: skip
    torch.cuda.synchronize(device)
    return reference_out


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle of the T=3 delta rule (sequential form)."""
    spec = case["spec"]
    num_seqs = spec["NUM_SEQS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    head_ratio = spec["HEAD_RATIO"]
    slot_stride = spec["STATE_SLOT_STRIDE"]

    a_log = case["a_log"].float()
    dt_bias = case["dt_bias"].float().reshape(spec["NUM_HEADS"], HEAD_DIM)
    lower_bound = float(case["lower_bound"])

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
                # GATE_KIND == 1: the gate is derived here, not supplied.
                # log_gate = lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias))
                gate_a = torch.exp(a_log[h])
                biased = g[row, hv] + dt_bias[h]
                log_gate = lower_bound / (1.0 + torch.exp(-gate_a * biased))
                gamma = torch.exp(log_gate)

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
        "flashkda_decode_t3_lower_bound is at the scaffold stage; the kernel body "
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
        "flashkda_decode_t3_lower_bound is at the scaffold stage; benchmarking is "
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
