# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=5 coefficient-gram decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t5_precomputed_gram_split{1,2,4,8}.cu``,
symbol ``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- the same frozen
template family as the ported T=2/T=3/T=4 siblings, re-instantiated at
TOKENS=5 with ``coefficient_gram`` enabled.

Dispatch reaches these bodies whenever ``num_tokens == 5`` with
``num_spec_tokens == 4`` and a precomputed (host-side) log-space gate. Unlike
T=2 (always split4) and T=4 (always split2), the T=5 value split is
**shape-dependent**: with ``work = num_seqs * num_value_heads`` and ``S`` SMs,
``recurrent_kda.py:1181-1191`` returns 8, 2, 4, 2, 1 across five bands, so all
four exports are reachable and all four are in this port's scope. The split is
a constexpr here, exactly as in the two-split T=1 sibling.

What "gram" means: at T<=4 the WY coefficients came from per-warp shuffle
reductions of ``k_t . k_s`` and ``q_t . k_s`` scaled by the gate ratio. At T=5
one designated warp instead forms both 16x16 Gram products on tensor cores
(``ldmatrix`` + ``mma.m16n8k16``) out of a gate-deflated ``k / prefix`` copy in
the new ``sGramA0``/``sGramA1`` shared regions -- regions that existed as dead
aliases of ``sVec`` in the T<=4 bodies. The inter-token butterfly path is gone
entirely, and a partial named barrier (``barrier.sync 1, 160`` /
``barrier.arrive 1, 160``) synchronizes the five token warps around it.

Helper vocabulary is shared with the T=2 module; only the geometry constants,
the frozen digests and the kernel body are per-specialization.
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
# against this body's exports too; see .porting/.../toolchain_notes.md).
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
# Raw and normalized digests differ for every T=5 export; the hash of the bytes
# between the BEGIN/END markers is the raw one (verified for all four).
FROZEN_FLASHINFER_COMMIT = "f2e04400"
FROZEN_BODY_SHA256 = {
    1: "7d44765cc20864dca2fc5f96ed2ae653e4d421f963c20fed5ba825d4989c8b4e",
    2: "d8c24892ac7e456fd04c51fe820ecedfc677c3be0e823cc4919043da7ae025af",
    4: "93020b1e878a584146d7f9c1a4e46c98bd05bcf8fa4f83c41fb6dcd4e616377d",
    8: "2307b896466dd58ff1daba770763b0a7142451e73225e940e9e9461a21bb9452",
}

HEAD_DIM = _t2.HEAD_DIM
NUM_TOKENS = 5
L2_EPS = _t2.L2_EPS
LOG2_E = _t2.LOG2_E

# Per-split geometry, transcribed from the four bodies' `#define` blocks and
# their guard expressions (`.cu:45-88`, `:143-166`, `:180`, `:411`, `:448`,
# `:652`). `threads` is FLASHKDA_DECODE_LAUNCH_THREADS (the `#define THREADS
# 256` in every body is vestigial -- it only feeds a static_assert in
# binding_impl.cuh:40); it is derived upstream by
# `max(tokens, value_rows/16, ((value_rows/rows_per_group)+1)/2) * 32`
# with rows_per_group = 2 only for gram+split8 (flash_kda_decode.py:115-134).
_SPLIT_GEOMETRY: dict[int, dict[str, int]] = {
    1: {
        "threads": 256,  # 8 warps
        "rows_per_cta": 128,
        "mma_warps": 8,  # phases D-G guard `warp_0 < 8`
        "row_groups": 16,  # phases B/H guard `group < 16`
        "rows_per_group": 8,
        "gram_warp": 0,
        "gram_sync_all": 0,  # gram warp syncs, the other token warps arrive
        "su_sync_cta": 0,  # phase G publishes sU behind __syncwarp
        "smem_total": 49024,
        "off_sstate0": 0,
        "off_sstate1": 16384,
        "off_svec": 32768,
        "off_sk": 36864,
        "off_sd": 39424,
        "off_sbeta": 41984,
        "off_sslot": 42004,
        "off_stoken": 42024,
        "off_sinit": 42044,
        "off_sl": 42060,
        "off_sr": 42160,
        "off_su": 42260,
        "off_sgrama0": 44928,
        "off_sgrama1": 46976,
    },
    2: {
        "threads": 160,  # 5 warps
        "rows_per_cta": 64,
        "mma_warps": 4,
        "row_groups": 8,
        "rows_per_group": 8,
        "gram_warp": 0,
        "gram_sync_all": 0,
        "su_sync_cta": 0,
        "smem_total": 31360,
        "off_sstate0": 0,
        "off_sstate1": 8192,
        "off_svec": 16384,
        "off_sk": 20480,
        "off_sd": 23040,
        "off_sbeta": 25600,
        "off_sslot": 25620,
        "off_stoken": 25640,
        "off_sinit": 25660,
        "off_sl": 25676,
        "off_sr": 25776,
        "off_su": 25876,
        "off_sgrama0": 27264,
        "off_sgrama1": 29312,
    },
    4: {
        "threads": 160,
        "rows_per_cta": 32,
        "mma_warps": 2,
        "row_groups": 4,
        "rows_per_group": 8,
        "gram_warp": 4,
        "gram_sync_all": 1,  # all five token warps sync; the arrive arm is dead
        "su_sync_cta": 0,
        "smem_total": 22528,
        "off_sstate0": 0,
        "off_sstate1": 4096,
        "off_svec": 8192,
        "off_sk": 12288,
        "off_sd": 14848,
        "off_sbeta": 17408,
        "off_sslot": 17428,
        "off_stoken": 17448,
        "off_sinit": 17468,
        "off_sl": 17484,
        "off_sr": 17584,
        "off_su": 17684,
        "off_sgrama0": 18432,
        "off_sgrama1": 20480,
    },
    8: {
        "threads": 160,
        "rows_per_cta": 16,
        "mma_warps": 1,
        "row_groups": 8,  # 8 groups x 2 rows
        "rows_per_group": 2,
        "gram_warp": 4,
        "gram_sync_all": 0,
        "su_sync_cta": 1,  # phase G's producer/consumer warps differ -> CTA sync
        "smem_total": 18048,
        "off_sstate0": 0,
        "off_sstate1": 2048,
        "off_svec": 4096,
        "off_sk": 8192,
        "off_sd": 10752,
        "off_sbeta": 13312,
        "off_sslot": 13332,
        "off_stoken": 13352,
        "off_sinit": 13372,
        "off_sl": 13388,
        "off_sr": 13488,
        "off_su": 13588,
        "off_sgrama0": 13952,
        "off_sgrama1": 16000,
    },
}

K_PER_THREAD = 8


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=5 export bench."""
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
        "sm_count": None,
        "seed": 20260815,
    }
    config.update(overrides)
    return config


# The T=5 split moves with shape: with W = num_seqs * num_value_heads and S SMs
# the bands are W <= 3S/8 -> 8, W <= S/2 -> 2, W <= 3S/4 -> 4, W <= 3S/2 -> 2,
# else 1 (recurrent_kda.py:1181-1191). Every label names the split its shape
# actually selects on a 148-SM B200, and verify_dispatch.py asserts that
# against the real selector. The five hv32h16 rows are FlashInfer's own T=5
# export-bench matrix (all split1); the rest exist to cover the other three
# exports, which are equally reachable in production.
BENCH_CONFIGS = [
    # FlashInfer's own T=5 export bench (W = 256..4096, all split1).
    _case("hv32h16_b8_s1", num_seqs=8),
    _case("hv32h16_b16_s1", num_seqs=16),
    _case("hv32h16_b32_s1", num_seqs=32),
    _case("hv32h16_b64_s1", num_seqs=64),
    _case("hv32h16_b128_s1", num_seqs=128),
    # Small batch at HV=32 walks the first three bands (W = 32, 64, 96).
    _case("hv32h16_b1_s8", num_seqs=1),
    _case("hv32h16_b2_s2", num_seqs=2),
    _case("hv32h16_b3_s4", num_seqs=3),
    # HV == H == 16: W = 32, 80, 128, 1024 -- note b8 lands in the *second*
    # split2 band, the one above the split4 island.
    _case("hv16h16_b2_s8", num_seqs=2, num_value_heads=16),
    _case("hv16h16_b5_s4", num_seqs=5, num_value_heads=16),
    _case("hv16h16_b8_s2", num_seqs=8, num_value_heads=16),
    _case("hv16h16_b64_s1", num_seqs=64, num_value_heads=16),
    # Kimi K3 TP8 per-rank head count (W = 96, 768).
    _case("hv12h12_b8_s4", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_s1", num_seqs=64, num_heads=12, num_value_heads=12),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # nat selects the initial checkpoint slot as ssm_idx[n*5 + clamp(nat-1, 0, 4)];
    # the upstream test sweeps 0/1/5/12, covering both clamp arms (.cu:280-286).
    _case("hv32h16_b8_nat0", num_seqs=8, accepted="zeros"),
    _case("hv32h16_b8_nat5", num_seqs=8, accepted="fives"),
    _case("hv32h16_b8_nat12", num_seqs=8, accepted="twelves"),
    _case("hv32h16_b8_natmix", num_seqs=8, accepted="mixed"),
    _case("hv32h16_b8_padded", num_seqs=8, padded_seqs=2),
    _case("hv32h16_b8_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv32h16_b8_gstride", num_seqs=8, gate_token_stride_pad=8),
    _case("hv32h16_b8_scale", num_seqs=8, scale=0.05),
    _case("hv64h16_b8_s1", num_seqs=8, num_value_heads=64),
    # Forced splits at one fixed shape (W = 256), via the SM count the policy
    # reads: the same tensors must give the same answer through all four
    # exports (upstream test_..._forced_t5_splits_match_upstream_cute).
    _case("hv32h16_b8_force_s8", num_seqs=8, sm_count=683),
    _case("hv32h16_b8_force_s2", num_seqs=8, sm_count=512),
    _case("hv32h16_b8_force_s4", num_seqs=8, sm_count=342),
    # Selector knife edges on a 148-SM part: W = 222 -> split2, W = 224 -> split1.
    _case("hv16h16_b13_edge_s2", num_seqs=13, num_value_heads=16, sm_count=148),
    _case("hv16h16_b14_edge_s1", num_seqs=14, num_value_heads=16, sm_count=148),
]

KERNEL_META = {
    "name": "flashkda_decode_t5_gram",
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
    """Reproduce ``_select_flash_kda_decode_value_split_current`` for T = 5.

    recurrent_kda.py:1181-1191. `work` is `num_seqs * num_value_heads`. Note the
    split4 island sits *between* two split2 bands, so this is not monotonic.
    """
    three_wave_ctas = 3 * sm_count
    if 8 * work <= three_wave_ctas:
        return 8
    if 2 * work <= sm_count:
        return 2
    if 4 * work <= three_wave_ctas:
        return 4
    if 2 * work <= three_wave_ctas:
        return 2
    return 1


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

    value_split = _select_value_split(num_seqs * num_value_heads, _sm_count(kwargs))
    geometry = _SPLIT_GEOMETRY[value_split]

    total_tokens = num_seqs * NUM_TOKENS
    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM + slot_stride_pad
    gate_token_stride = num_value_heads * HEAD_DIM + gate_token_stride_pad
    state_slots = total_tokens + pool_slack
    spec = {
        "NUM_SEQS": num_seqs,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "HEAD_RATIO": num_value_heads // num_heads,
        "VALUE_SPLIT": value_split,
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
    spec.update({key.upper(): value for key, value in geometry.items()})
    return spec


# TIRX_TRANSCRIBE_START flashkda_decode_t5_gram


@T.jit
def _flashkda_decode_t5_gram(
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
    VALUE_SPLIT: T.constexpr,
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
    THREADS: T.constexpr,
    ROWS_PER_CTA: T.constexpr,
    MMA_WARPS: T.constexpr,
    ROW_GROUPS: T.constexpr,
    ROWS_PER_GROUP: T.constexpr,
    GRAM_WARP: T.constexpr,
    GRAM_SYNC_ALL: T.constexpr,
    SU_SYNC_CTA: T.constexpr,
    SMEM_TOTAL: T.constexpr,
    OFF_SSTATE0: T.constexpr,
    OFF_SSTATE1: T.constexpr,
    OFF_SVEC: T.constexpr,
    OFF_SK: T.constexpr,
    OFF_SD: T.constexpr,
    OFF_SBETA: T.constexpr,
    OFF_SSLOT: T.constexpr,
    OFF_STOKEN: T.constexpr,
    OFF_SINIT: T.constexpr,
    OFF_SL: T.constexpr,
    OFF_SR: T.constexpr,
    OFF_SU: T.constexpr,
    OFF_SGRAMA0: T.constexpr,
    OFF_SGRAMA1: T.constexpr,
):
    """FlashKDA "cake" T=5 coefficient-gram decode; 5 token warps.

    Scaffold only -- the body is written in the kernel-sketch stage, from the
    reviewer-approved sketch at
    `.agents/sketch/flashinfer/kda/flashkda_decode_t5_gram.md`. Bare `:NNN`
    references in the transcribed body are into
    `flashkda_decode_d128_t5_precomputed_gram_split2.cu` unless a split is named.
    """
    q = T.match_buffer(q_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    k = T.match_buffer(k_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    v = T.match_buffer(v_h, (V_ELEMENTS,), "bfloat16", scope="global")
    g = T.match_buffer(g_h, (GATE_ELEMENTS,), "bfloat16", scope="global")
    beta = T.match_buffer(beta_h, (BETA_ELEMENTS,), "bfloat16", scope="global")
    state = T.match_buffer(state_h, (STATE_ELEMENTS,), "bfloat16", scope="global")
    out = T.match_buffer(out_h, (V_ELEMENTS,), "bfloat16", scope="global")
    cu = T.match_buffer(cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), "int32", scope="global")
    ssm_idx = T.match_buffer(ssm_state_indices_h, (STATE_INDEX_ELEMENTS,), "int32", scope="global")
    nat = T.match_buffer(num_accepted_tokens_h, (NAT_ELEMENTS,), "int32", scope="global")
    T.device_entry()
    # TRANSCRIBE: the frozen body goes here (kernel-sketch stage).


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=5 gram decode PrimFunc."""
    return _flashkda_decode_t5_gram.specialize(**_specialization(kwargs))


_NAT_PATTERNS = {None: None, "zeros": 0, "ones": 1, "fives": 5, "twelves": 12}


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case; the kernel clamps nat-1 into [0, 4]."""
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream T=5 matrix test's sweep, tiled to the batch size.
        pattern = torch.tensor([0, 1, 5, 12, 4, 0, 1, 5], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    if kind in _NAT_PATTERNS:
        return torch.full((num_seqs,), _NAT_PATTERNS[kind], device=device, dtype=torch.int32)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Same recipe as the ported T=2/T=4 siblings at T=5: packed [1, N*5, ...]
    bf16, a log-space gate (GATE_KIND 0 -- the kernel applies exp()),
    pre-sigmoided beta, flat [N*5] slot indices with slot 0 never a target.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=5 decode")
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


def _variant_name(value_split: int) -> str:
    return f"d128_t5_precomputed_gram_split{value_split}"


def _source_body_path(value_split: int) -> str:
    """Absolute path of the frozen generated body this port transcribes."""
    import flashinfer

    root = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    return os.path.join(root, "csrc", "kda", f"flashkda_decode_{_variant_name(value_split)}.cu")


def assert_frozen_source(value_split: int) -> None:
    """Fail loudly if the upstream generated body was regenerated."""
    expected = FROZEN_BODY_SHA256[value_split]
    path = _source_body_path(value_split)
    with open(path, "rb") as handle:
        text = handle.read().decode()
    marker = "// BEGIN FROZEN GENERATED BODY\n"
    start = text.index(marker) + len(marker)
    end = text.index("// END FROZEN GENERATED BODY")
    digest = hashlib.sha256(text[start:end].encode()).hexdigest()
    if digest != expected:
        raise AssertionError(
            f"{path}: frozen body digest {digest} != pinned {expected}; the "
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

    module = get_flash_kda_decode_module(_variant_name(case["spec"]["VALUE_SPLIT"]), target)
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
    """Independent FP32 oracle of the T=5 delta rule (sequential form)."""
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


# The port replicates the source's MMA chain, swizzle and association orders, so
# it should agree with the frozen export to within bf16 rounding.
_RTOL = 2.0**-8
_ATOL = 2.0e-3
# The FP32 oracle is the sequential delta rule; the kernel rounds its MMA
# operands to bf16 -- and at T=5 the WY coefficients themselves come from a
# bf16 Gram product of `k / prefix`, so this band is wider than the T<=4 one and
# is re-measured at the correctness gate rather than inherited.
_ORACLE_RTOL = 2.0**-5
_ORACLE_ATOL = 6.0e-3


def run_test(**kwargs: Any) -> None:
    """Validate one config against the frozen export and an independent oracle."""
    raise NotImplementedError("scaffold stage: the kernel body is not written yet")


def run_bench(
    *,
    warmup: float | None = None,
    repeat: float | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Time the port against the frozen cake export on identical inputs."""
    raise NotImplementedError("scaffold stage: the kernel body is not written yet")


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
