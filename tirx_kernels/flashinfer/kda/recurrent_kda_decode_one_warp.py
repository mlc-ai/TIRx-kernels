# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's one-warp recurrent-KDA decode kernel.

Source CuTe DSL: ``flashinfer/kda_kernels/recurrent_kda.py``
(``recurrent_kda_decode_kernel`` at line 190, helpers ``compute_gate_value``
at 88, ``_dot8_row`` at 154, ``_reduce_k_group`` at 178), launched by
``recurrent_kda_launch`` at line 917 and dispatched from ``run_recurrent_kda``
at line 1525 under ``use_one_warp`` (line 1798).

Target instance: BF16, ``HEAD_DIM = 128``, ``NUM_TOKENS = 1``, packed
``cu_seqlens`` decode, in-kernel gate with ``dt_bias``, in-kernel q/k L2 norm,
pre-sigmoided beta, in-place ``[S, HV, 128, 128]`` state pool, sm_100a.
Launch is ``N * HV * (128 // TILE_ROWS)`` CTAs of **32 threads (one warp)**
with **zero shared memory** and no barrier of any kind; the recurrent state
lives entirely in registers and every cross-lane exchange is a warp shuffle.

This is the specialization SGLang's KDA decode backend reaches
(``sglang/srt/layers/attention/linear/kernels/kda_flashinfer.py::decode``):
``use_qk_l2norm_in_kernel=True``, ``use_gate_in_kernel=True``, ``dt_bias``
always present, beta already sigmoided, ``cu_seqlens = arange(N + 1)``,
``ssm_state_indices`` possibly carrying ``-1`` padding rows.

Two gate modes are covered, both reachable from SGLang:

* ``USE_LOWER_BOUND = 1`` -- Kimi K3, ``lower_bound = -5.0``;
  ``g = exp2(lb * log2e * sigmoid(-exp(A_log) * log2e * (g + dt_bias)))``.
* ``USE_LOWER_BOUND = 0`` -- Kimi Linear, softplus;
  ``g = exp2(-exp(A_log) * log2(1 + exp(g + dt_bias)))``.

Out of scope (they dispatch to a different source implementation or are
unreachable from the production call path): ``NUM_TOKENS > 1`` and the whole
speculative-decode path, ``sequence_heads < 128`` (which routes to
``_grouped_kda_kernel``), ``HEAD_DIM = 64``, ``TILE_ROWS = 32``,
``USE_CU_SEQLENS = 0``, ``BETA_IS_LOGIT = 1``, ``HAS_INITIAL_STATE_SOURCE = 1``,
``HAS_NUM_ACCEPTED_TOKENS = 1``, ``USE_GATE_IN_KERNEL = 0``,
``USE_QK_L2NORM = 0``, and GQA (``HV != H``; Kimi K3 uses ``H == HV``).
"""

from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

# --------------------------------------------------------------------------
# Fixed dimensions of this specialization (recurrent_kda.py:242-244)
# --------------------------------------------------------------------------
HEAD_DIM = 128
THREADS = 32  # recurrent_kda.py:987 -- block = [32, 1, 1]
K_LANES = HEAD_DIM // 8  # 16
V_LANES = 32 // K_LANES  # 2
VALUES_PER_THREAD = HEAD_DIM // 32  # 4
# Single-token decode: this port covers NUM_TOKENS == 1 only
# (recurrent_kda.py:1767; NUM_TOKENS > 1 routes to the grouped-CTA kernel).
NUM_TOKENS = 1

# recurrent_kda.py:74-75
DOT_REDUCTION_TREE = 0
DOT_REDUCTION_DUAL_ACCUM = 1

# recurrent_kda.py:79
ONE_WARP_MIN_SEQUENCE_HEADS = 128

# recurrent_kda.py:2087 -- the host hardcodes the L2-norm epsilon.
L2_EPS = 1.0e-6
# Kimi K3 checkpoint gate bound (sglang models/kimi_k3.py:1556).
DEFAULT_LOWER_BOUND = -5.0


# --------------------------------------------------------------------------
# PTX helpers.
#
# The source's FP32 arithmetic is **non-flush-to-zero** throughout: the exported
# line-info PTX contains zero `.ftz` add/mul/sub/fma.  Only the transcendentals
# are FTZ, and the lower-bound sigmoid reciprocal is the full-precision
# `div.rn.f32` rather than `rcp.approx`.  Reproducing that split is part of the
# port, so these wrappers pin each instruction explicitly.
# --------------------------------------------------------------------------
LOG2_E = 1.4426950408889634


def _ptx_un(chain: str, a, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, a)
    return out


def _ptx_bin(chain: str, a, b, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, a, b)
    return out


def _ptx_ter(chain: str, a, b, c, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, a, b, c)
    return out


def _mul(a, b):
    """``mul.f32`` -- non-FTZ, matching the source."""
    return _ptx_bin("mul.f32", a, b)


def _add(a, b):
    """``add.f32`` -- non-FTZ."""
    return _ptx_bin("add.f32", a, b)


def _sub(a, b):
    """``sub.f32`` -- non-FTZ."""
    return _ptx_bin("sub.f32", a, b)


def _fma(a, b, c):
    """``fma.rn.f32`` -- non-FTZ."""
    return _ptx_ter("fma.rn.f32", a, b, c)


def _exp2(a):
    return _ptx_un("ex2.approx.ftz.f32", a)


def _log2(a):
    return _ptx_un("lg2.approx.ftz.f32", a)


def _rsqrt(a):
    return _ptx_un("rsqrt.approx.ftz.f32", a)


def _div_rn(a, b):
    """``div.rn.f32`` -- the source requests full-precision division here."""
    return _ptx_bin("div.rn.f32", a, b)


def _add_bf16(bf_bits, f):
    """``add.rn.f32.bf16`` -- bf16 operand feeds the FP32 add directly."""
    return _ptx_bin("add.rn.f32.bf16", bf_bits, f)


def _fma_bf16(bf_a, bf_b, acc):
    """``fma.rn.f32.bf16`` -- both multiplicands stay in their bf16 registers."""
    return _ptx_ter("fma.rn.f32.bf16", bf_a, bf_b, acc)


def _bf16_to_f32(bits):
    return _ptx_un("cvt.f32.bf16", K.cast(bits, "uint16"))


def _f32_to_bf16(value):
    return _ptx_un("cvt.rn.bf16.f32", value, dtype="uint16")


def _pack_bf16x2(hi, lo):
    """``cvt.rn.bf16x2.f32 d, hi, lo`` -- d[31:16]=hi, d[15:0]=lo.

    The low half holds the lower-addressed element, so callers pass
    ``(elem[2p + 1], elem[2p])``.
    """
    return _ptx_bin("cvt.rn.bf16x2.f32", hi, lo, dtype="uint32")


def _shfl_bfly_f32(value, lane_xor):
    """``shfl.sync.bfly.b32`` with clamp 31 and the full member mask."""
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), K.uint32(lane_xor), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("float32", out)


def _shfl_idx_f32(value, source_lane):
    """``shfl.sync.idx.b32`` with clamp 31 and the full member mask."""
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.idx.b32(
        out,
        K.reinterpret("uint32", value),
        K.cast(source_lane, "uint32"),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", out)


def _load_f32(buffer, index):
    out = K.local_scalar("uint32")
    K.ptx.ld.global_.b32(out, buffer.ptr_to([index]))
    return K.reinterpret("float32", out)


def _load_i32(buffer, index):
    out = K.local_scalar("uint32")
    K.ptx.ld.global_.b32(out, buffer.ptr_to([index]))
    return K.reinterpret("int32", out)


def _load_bf16_bits(buffer, index):
    out = K.local_scalar("uint16")
    K.ptx.ld.global_.b16(out, buffer.ptr_to([index]))
    return out


def _store_bf16_bits(buffer, index, bits):
    K.ptx.st.global_.b16(buffer.ptr_to([index]), bits)


def _store_bf16_bits_pred(buffer, index, bits, pred):
    """``@p st.global.b16`` -- a genuinely predicated store.

    The approved sketch calls the output store "one predicated scalar store per
    j".  Expressing it as `if cond: store` instead makes ptxas emit a real
    branch plus a BSYNC reconvergence per CTA, which the source does not have.
    """
    K.ptx.st.global_.b16(buffer.ptr_to([index]), bits, pred=pred)


def _load_bf16x4(buffer, index):
    """``ld.global.v4.b16`` -- one 8-byte tile of four bf16."""
    bits = K.alloc_local((4,), "uint16")
    K.ptx.ld.global_.v4.b16(bits[0], bits[1], bits[2], bits[3], buffer.ptr_to([index]))
    return bits


def _load_bf16x8(buffer, index):
    """``ld.global.v4.b32`` -- one 16-byte tile, unpacked to eight bf16."""
    words = K.alloc_local((4,), "uint32")
    K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    bits = K.alloc_local((8,), "uint16")
    for pair in range(4):
        K.buffer_store(
            bits, K.cast(K.bitwise_and(words[pair], K.uint32(0xFFFF)), "uint16"), [2 * pair]
        )
        K.buffer_store(
            bits, K.cast(K.shift_right(words[pair], K.uint32(16)), "uint16"), [2 * pair + 1]
        )
    return bits


def _load_u32x4(buffer, index):
    """``ld.global.v4.b32`` -- one 16-byte tile, left packed."""
    words = K.alloc_local((4,), "uint32")
    K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    return words


def _store_bf16x8_words(buffer, index, words):
    """``st.global.v4.b32`` -- one 16-byte state row."""
    K.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])


def _dot8(state, base, rhs, schedule: int):
    """Eight-element dot with the source's selected dependency graph."""
    if schedule == DOT_REDUCTION_DUAL_ACCUM:
        even = _mul(state[base + 0], rhs[0])
        odd = _mul(state[base + 1], rhs[1])
        for pair in range(1, 4):
            even = _fma(state[base + 2 * pair], rhs[2 * pair], even)
            odd = _fma(state[base + 2 * pair + 1], rhs[2 * pair + 1], odd)
        return _add(even, odd)
    p0 = _fma(state[base + 0], rhs[0], _mul(state[base + 1], rhs[1]))
    p1 = _fma(state[base + 2], rhs[2], _mul(state[base + 3], rhs[3]))
    p2 = _fma(state[base + 4], rhs[4], _mul(state[base + 5], rhs[5]))
    p3 = _fma(state[base + 6], rhs[6], _mul(state[base + 7], rhs[7]))
    return _add(_add(p0, p1), _add(p2, p3))


def _reduce_k_group(value):
    """Reduce a partial dot across the 16 lanes that partition K (HEAD_DIM=128)."""
    for offset in (8, 4, 2, 1):
        value = _add(value, _shfl_bfly_f32(value, offset))
    return value


def _select_kernel_schedule(sequence_heads: int) -> tuple[int, int]:
    """Reproduce ``_select_kernel_schedule`` (recurrent_kda.py:1108-1149).

    Specialized to ``head_dim = 128``, ``num_tokens = 1``, ``use_gate = True``,
    which is the only combination this port serves.
    """
    if (
        sequence_heads <= 176
        or 304 <= sequence_heads <= 368
        or 448 <= sequence_heads <= 560
        or sequence_heads >= 720
    ):
        tile_rows = 8
    elif 224 <= sequence_heads <= 288:
        tile_rows = 32
    else:
        tile_rows = 16

    # recurrent_kda.py:1136-1141 -- the two gated D128 overrides.
    if sequence_heads >= 224:
        tile_rows = 16
    if sequence_heads == 64:
        tile_rows = 16

    reduction = (
        DOT_REDUCTION_DUAL_ACCUM if tile_rows != 8 or sequence_heads >= 304 else DOT_REDUCTION_TREE
    )
    return tile_rows, reduction


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror the SGLang decode benchmark shape."""
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 32,
        "num_heads": 16,
        "num_value_heads": 16,
        "pool_size": 512,
        "lower_bound": DEFAULT_LOWER_BOUND,
        "padded_slots": 0,
        "slot_stride_pad": 0,
        "seed": 20260811,
    }
    config.update(overrides)
    return config


# Every row uses the Kimi K3 gate (lower_bound = -5.0), matches the SGLang
# decode sweep, and stays inside the one-warp dispatch domain
# (num_seqs * num_value_heads >= 128).
BENCH_CONFIGS = [
    # sequence_heads = 128 -> TILE_ROWS = 8, DOT_REDUCTION_TREE
    _case("hv16_b8_tr8_lb", num_seqs=8),
    # sequence_heads = 256/512/1024/2048 -> TILE_ROWS = 16, DUAL_ACCUM
    _case("hv16_b16_tr16_lb", num_seqs=16),
    _case("hv16_b32_tr16_lb", num_seqs=32),
    _case("hv16_b64_tr16_lb", num_seqs=64),
    _case("hv16_b128_tr16_lb", num_seqs=128),
    # Kimi K3 TP8 per-rank head count (12 heads): sequence_heads = 192 / 768
    _case("hv12_b16_tr16_lb", num_seqs=16, num_heads=12, num_value_heads=12),
    _case("hv12_b64_tr16_lb", num_seqs=64, num_heads=12, num_value_heads=12),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # Schedule-band boundaries: 176 is the largest TILE_ROWS = 8 shape and 192
    # the smallest that falls through to the TILE_ROWS = 16 else-branch.
    _case("hv16_b11_tr8_lb", num_seqs=11),
    _case("hv16_b12_tr16_lb", num_seqs=12),
    # Softplus gate (Kimi Linear): USE_LOWER_BOUND = 0, both tile schedules.
    _case("hv16_b8_tr8_sp", num_seqs=8, lower_bound=None),
    _case("hv16_b32_tr16_sp", num_seqs=32, lower_bound=None),
    _case("hv12_b16_tr16_sp", num_seqs=16, num_heads=12, num_value_heads=12, lower_bound=None),
    # CUDA-graph padding rows: negative ssm_state_indices mark inactive
    # sequences whose output must be zeroed and whose state must not change.
    _case("hv16_b16_tr16_lb_padded", num_seqs=16, padded_slots=5),
    # Envelope-strided state pool: the slot stride may exceed HV*V*K as long as
    # it stays 16-element aligned (kda_flashinfer.py:110-123).
    _case("hv16_b16_tr16_lb_strided", num_seqs=16, slot_stride_pad=16),
]

KERNEL_META = {
    "name": "recurrent_kda_decode_one_warp",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    num_seqs = int(kwargs["num_seqs"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    pool_size = int(kwargs["pool_size"])
    slot_stride_pad = int(kwargs.get("slot_stride_pad", 0))
    lower_bound = kwargs.get("lower_bound", DEFAULT_LOWER_BOUND)

    if num_value_heads % num_heads != 0:
        raise ValueError("num_value_heads must be a multiple of num_heads")
    if num_value_heads != num_heads:
        raise ValueError("GQA (HV != H) is outside this port's scope; Kimi K3 uses H == HV")
    if slot_stride_pad % 16 != 0:
        raise ValueError("state slot stride padding must stay 16-element aligned")

    sequence_heads = num_seqs * num_value_heads
    if sequence_heads < ONE_WARP_MIN_SEQUENCE_HEADS:
        raise ValueError(
            f"sequence_heads={sequence_heads} dispatches to the grouped-CTA kernel; "
            f"the one-warp port requires >= {ONE_WARP_MIN_SEQUENCE_HEADS} "
            "(recurrent_kda.py:1798)"
        )

    tile_rows, reduction = _select_kernel_schedule(sequence_heads)
    if tile_rows not in (8, 16):
        raise ValueError(
            f"sequence_heads={sequence_heads} selects TILE_ROWS={tile_rows}, "
            "which is outside this port's scope"
        )

    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM + slot_stride_pad
    return {
        "NUM_V_TILES": HEAD_DIM // tile_rows,
        "ROWS": tile_rows // V_LANES,
        "HEAD_ELEMENTS": HEAD_DIM * HEAD_DIM,
        "GQA_RATIO": num_value_heads // num_heads,
        "NUM_SEQS": num_seqs,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "TILE_ROWS": tile_rows,
        "DOT_REDUCTION_SCHEDULE": reduction,
        "USE_LOWER_BOUND": lower_bound is not None,
        "Q_ELEMENTS": num_seqs * num_heads * HEAD_DIM,
        "V_ELEMENTS": num_seqs * num_value_heads * HEAD_DIM,
        "BETA_ELEMENTS": num_seqs * num_value_heads,
        "STATE_ELEMENTS": pool_size * slot_stride,
        "STATE_SLOT_STRIDE": slot_stride,
        "A_LOG_ELEMENTS": num_heads,
        "DT_BIAS_ELEMENTS": num_heads * HEAD_DIM,
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": num_seqs,
    }


def _make_recurrent_kda_decode_one_warp(spec: dict[str, Any]):
    NUM_SEQS = spec["NUM_SEQS"]
    NUM_HEADS = spec["NUM_HEADS"]
    NUM_VALUE_HEADS = spec["NUM_VALUE_HEADS"]
    TILE_ROWS = spec["TILE_ROWS"]
    NUM_V_TILES = spec["NUM_V_TILES"]
    ROWS = spec["ROWS"]
    HEAD_ELEMENTS = spec["HEAD_ELEMENTS"]
    GQA_RATIO = spec["GQA_RATIO"]
    DOT_REDUCTION_SCHEDULE = spec["DOT_REDUCTION_SCHEDULE"]
    USE_LOWER_BOUND = spec["USE_LOWER_BOUND"]
    STATE_SLOT_STRIDE = spec["STATE_SLOT_STRIDE"]

    @K.kernel(warps=1, arch="sm_100a", grid=NUM_SEQS * NUM_VALUE_HEADS * NUM_V_TILES)
    def _recurrent_kda_decode_one_warp(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        g: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        state: K.gptr[K.bf16],
        out: K.gptr[K.bf16],
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        cu_seqlens: K.gptr[K.i32],
        ssm_state_indices: K.gptr[K.i32],
        scale: K.f32,
        eps: K.f32,
        lower_bound: K.f32,
    ):
        # TIRX_TRANSCRIBE_START recurrent_kda_decode_one_warp
        # --- lane and CTA coordinates (recurrent_kda.py:229-246) ----------------
        bidx = K.cta_id()
        tidx_axis = K.thread_id()
        tidx = K.local_scalar("int32")
        v_tile_idx = K.local_scalar("int32")
        bh = K.local_scalar("int32")
        value_head_idx = K.local_scalar("int32")
        batch_idx = K.local_scalar("int32")
        query_head_idx = K.local_scalar("int32")
        v_offset = K.local_scalar("int32")
        k_lane = K.local_scalar("int32")
        v_lane = K.local_scalar("int32")
        K.assign(tidx, K.cast(K.bitwise_and(K.cast(tidx_axis, "uint32"), K.uint32(31)), "int32"))
        K.assign(v_tile_idx, bidx % NUM_V_TILES)
        K.assign(bh, bidx // NUM_V_TILES)
        K.assign(value_head_idx, bh % NUM_VALUE_HEADS)
        K.assign(batch_idx, bh // NUM_VALUE_HEADS)
        K.assign(query_head_idx, value_head_idx // GQA_RATIO)
        K.assign(v_offset, v_tile_idx * TILE_ROWS)
        K.assign(k_lane, tidx % K_LANES)
        K.assign(v_lane, tidx // K_LANES)

        global_offset = K.local_scalar("int32")

        def gidx(offset):
            K.assign(global_offset, offset)
            return K.Cast("int64", global_offset)

        # --- sequence metadata (recurrent_kda.py:248-252) -----------------------
        token_base_offset = K.local_scalar("int32")
        seq_len = K.local_scalar("int32")
        K.assign(token_base_offset, _load_i32(cu_seqlens, gidx(batch_idx)))
        K.assign(seq_len, _load_i32(cu_seqlens, gidx(batch_idx + 1)) - token_base_offset)

        # --- zero-padded output prefill (recurrent_kda.py:253-255) --------------
        # Runs before the state load so inactive rows are defined even when the
        # token loop stores nothing.  Valid only because NUM_TOKENS == 1 makes the
        # token-axis index equal to batch_idx.
        with K.If(tidx < TILE_ROWS), K.Then():
            _store_bf16_bits(
                out,
                gidx((batch_idx * NUM_VALUE_HEADS + value_head_idx) * HEAD_DIM + v_offset + tidx),
                K.uint16(0),
            )
        K.cuda.warp_sync()

        # --- initial-state slot (recurrent_kda.py:257-272) ----------------------
        init_raw_slot = K.local_scalar("int32")
        init_seq_idx = K.local_scalar("int32")
        K.assign(init_raw_slot, _load_i32(ssm_state_indices, gidx(batch_idx * NUM_TOKENS)))
        K.assign(init_seq_idx, K.max(init_raw_slot, 0))

        # --- register storage (recurrent_kda.py:280-293); no SMEM, no mbarrier --
        h_reg = K.alloc_local((ROWS * 8,), "float32")
        q_src = K.alloc_local((VALUES_PER_THREAD,), "float32")
        k_src = K.alloc_local((VALUES_PER_THREAD,), "float32")
        gate_src = K.alloc_local((VALUES_PER_THREAD,), "float32")
        q_reg = K.alloc_local((8,), "float32")
        k_reg = K.alloc_local((8,), "float32")
        gate_reg = K.alloc_local((8,), "float32")

        # --- state load (recurrent_kda.py:295-300) ------------------------------
        state_head_base = K.local_scalar("int32")
        read_base = K.local_scalar("int64")
        K.assign(state_head_base, value_head_idx * HEAD_ELEMENTS + 8 * k_lane)
        K.assign(
            read_base,
            K.cast(init_seq_idx, "int64") * K.cast(STATE_SLOT_STRIDE, "int64")
            + K.cast(state_head_base, "int64"),
        )
        # Issue every row's load before widening any of them.  bench_suite times a
        # cold L2, so the figure of merit here is how many DRAM misses are in
        # flight; interleaving the widening with the loads lets each cvt chain sit
        # on the critical path of the next load.
        h_words = K.alloc_local((4 * ROWS,), "uint32")
        for j in range(ROWS):
            v_idx_l = K.local_scalar("int32", init=v_offset + v_lane + V_LANES * j)
            words = _load_u32x4(state, read_base + K.cast(v_idx_l * HEAD_DIM, "int64"))
            for pr in range(4):
                K.ptx.mov.b32(h_words[j * 4 + pr], words[pr])
        for j in range(ROWS):
            for pr in range(4):
                w = K.local_scalar("uint32", init=h_words[j * 4 + pr])
                K.ptx.mov.b32(
                    h_reg[j * 8 + 2 * pr],
                    _bf16_to_f32(K.cast(K.bitwise_and(w, K.uint32(0xFFFF)), "uint16")),
                )
                K.ptx.mov.b32(
                    h_reg[j * 8 + 2 * pr + 1],
                    _bf16_to_f32(K.cast(K.shift_right(w, K.uint32(16)), "uint16")),
                )

        # --- per-head gate constants (recurrent_kda.py:302-305) -----------------
        h_K_offset = K.local_scalar("int32")
        A_log_val = K.local_scalar("float32")
        K.assign(h_K_offset, query_head_idx * HEAD_DIM)
        K.assign(A_log_val, _exp2(_mul(_load_f32(a_log, gidx(query_head_idx)), K.float32(LOG2_E))))

        # --- loop-invariant lower-bound gate constants (recurrent_kda.py:124-127)
        # Both live inside compute_gate_value's per-i loop in the source and are
        # loop-invariant, so they are hoisted here.  The neg is folded into the
        # negated log2(e) immediate and emits no neg.f32.
        neg_A_log2e = K.local_scalar("float32")
        lb_log2e = K.local_scalar("float32")
        neg_A_log_val = K.local_scalar("float32")
        K.assign(neg_A_log2e, K.float32(0.0))
        K.assign(lb_log2e, K.float32(0.0))
        K.assign(neg_A_log_val, K.float32(0.0))
        if USE_LOWER_BOUND:
            K.assign(neg_A_log2e, _mul(A_log_val, K.float32(-LOG2_E)))
            K.assign(lb_log2e, _mul(lower_bound, K.float32(LOG2_E)))
        else:
            K.assign(neg_A_log_val, _mul(A_log_val, K.float32(-1.0)))

        # =======================================================================
        # Token loop.  NUM_TOKENS == 1, so this runs exactly once; the source keeps
        # it a loop and the port keeps the same structure.
        # =======================================================================
        for token_t in range(NUM_TOKENS):
            # --- slot / activity resolution (recurrent_kda.py:321-332) ---------
            # The source reloads gSsmStateIndices here (:323) and re-clamps at :329.
            # At NUM_TOKENS == 1 both index the same element as the prologue, and the
            # CuTe compiler CSEs them into a single ld.global.b32 + max.s32.  Our PTX
            # helpers are opaque inline asm, so CSE cannot fire across them; reusing
            # the prologue values reproduces the source's *emitted* code exactly.
            raw_slot = K.local_scalar("int32")
            has_token = K.local_scalar("int32")
            is_active = K.local_scalar("int32")
            token_offset = K.local_scalar("int32")
            seq_idx = K.local_scalar("int32")
            K.assign(raw_slot, init_raw_slot)
            K.assign(has_token, K.cast(token_t < seq_len, "int32"))
            K.assign(is_active, K.cast(K.cast(raw_slot >= 0, "int32") * has_token, "int32"))
            K.assign(
                token_offset,
                K.if_then_else(token_t < seq_len, token_base_offset + token_t, K.int32(0)),
            )
            K.assign(seq_idx, init_seq_idx)

            # --- per-head views and beta (recurrent_kda.py:333-346) ------------
            q_base = K.local_scalar("int32")
            v_base = K.local_scalar("int32")
            beta_val = K.local_scalar("float32")
            K.assign(q_base, (token_offset * NUM_HEADS + query_head_idx) * HEAD_DIM)
            K.assign(v_base, (token_offset * NUM_VALUE_HEADS + value_head_idx) * HEAD_DIM)
            K.assign(
                beta_val,
                _bf16_to_f32(
                    _load_bf16_bits(beta, gidx(token_offset * NUM_VALUE_HEADS + value_head_idx))
                ),
            )

            # --- q/k/gate vector loads (recurrent_kda.py:353-358) --------------
            q_bits = _load_bf16x4(q, gidx(q_base + VALUES_PER_THREAD * tidx))
            k_bits = _load_bf16x4(k, gidx(q_base + VALUES_PER_THREAD * tidx))
            gate_bits = _load_bf16x4(g, gidx(v_base + VALUES_PER_THREAD * tidx))

            # --- V load, hoisted only at TILE_ROWS == 16 (recurrent_kda.py:359-364)
            v_loaded = K.local_scalar("float32", init=K.float32(0.0))
            if TILE_ROWS == 16:
                with K.If(tidx < TILE_ROWS), K.Then():
                    K.assign(
                        v_loaded, _bf16_to_f32(_load_bf16_bits(v, gidx(v_base + v_offset + tidx)))
                    )

            # --- gate + q/k conversion (recurrent_kda.py:366-380, :88-148) -----
            for i in range(VALUES_PER_THREAD):
                k_idx = K.local_scalar("int32")
                g_val = K.local_scalar("float32")
                K.assign(k_idx, tidx * VALUES_PER_THREAD + i)
                K.ptx.mov.b32(q_src[i], _bf16_to_f32(q_bits[i]))
                K.ptx.mov.b32(k_src[i], _bf16_to_f32(k_bits[i]))
                K.assign(
                    g_val, _add_bf16(gate_bits[i], _load_f32(dt_bias, gidx(h_K_offset + k_idx)))
                )
                if USE_LOWER_BOUND:
                    denom = K.local_scalar("float32")
                    K.assign(denom, _add(K.float32(1.0), _exp2(_mul(neg_A_log2e, g_val))))
                    K.ptx.mov.b32(gate_src[i], _exp2(_div_rn(lb_log2e, denom)))
                else:
                    exp_g = K.local_scalar("float32")
                    log2_v = K.local_scalar("float32")
                    K.assign(exp_g, _exp2(_mul(g_val, K.float32(LOG2_E))))
                    K.assign(log2_v, _log2(_add(K.float32(1.0), exp_g)))
                    K.ptx.mov.b32(gate_src[i], _exp2(_mul(neg_A_log_val, log2_v)))

            # --- q/k sum of squares (recurrent_kda.py:382-404) -----------------
            q_sum_sq = K.local_scalar("float32")
            k_sum_sq = K.local_scalar("float32")
            K.assign(q_sum_sq, K.float32(0.0))
            K.assign(k_sum_sq, K.float32(0.0))
            if DOT_REDUCTION_SCHEDULE == DOT_REDUCTION_DUAL_ACCUM:
                K.assign(
                    q_sum_sq,
                    _add(
                        _fma_bf16(q_bits[2], q_bits[2], _mul(q_src[0], q_src[0])),
                        _fma_bf16(q_bits[3], q_bits[3], _mul(q_src[1], q_src[1])),
                    ),
                )
                K.assign(
                    k_sum_sq,
                    _add(
                        _fma_bf16(k_bits[2], k_bits[2], _mul(k_src[0], k_src[0])),
                        _fma_bf16(k_bits[3], k_bits[3], _mul(k_src[1], k_src[1])),
                    ),
                )
            else:
                K.assign(
                    q_sum_sq,
                    _add(
                        _fma_bf16(q_bits[0], q_bits[0], _mul(q_src[1], q_src[1])),
                        _fma_bf16(q_bits[2], q_bits[2], _mul(q_src[3], q_src[3])),
                    ),
                )
                K.assign(
                    k_sum_sq,
                    _add(
                        _fma_bf16(k_bits[0], k_bits[0], _mul(k_src[1], k_src[1])),
                        _fma_bf16(k_bits[2], k_bits[2], _mul(k_src[3], k_src[3])),
                    ),
                )

            # --- full-warp butterfly (recurrent_kda.py:405-411) ----------------
            for off_i in range(5):
                shift = K.local_scalar("int32", init=K.shift_right(K.int32(16), off_i))
                K.assign(q_sum_sq, _add(q_sum_sq, _shfl_bfly_f32(q_sum_sq, shift)))
                K.assign(k_sum_sq, _add(k_sum_sq, _shfl_bfly_f32(k_sum_sq, shift)))

            # --- scale factors (recurrent_kda.py:413-417) ----------------------
            q_scale_factor = K.local_scalar("float32")
            k_scale_factor = K.local_scalar("float32")
            K.assign(q_scale_factor, _mul(_rsqrt(_add(q_sum_sq, eps)), scale))
            K.assign(k_scale_factor, _rsqrt(_add(k_sum_sq, eps)))

            # --- broadcast the load view into the k_lane view (:418-436) -------
            for i in range(8):
                source_lane = K.local_scalar(
                    "int32", init=V_LANES * k_lane + i // VALUES_PER_THREAD
                )
                K.ptx.mov.b32(
                    q_reg[i],
                    _mul(_shfl_idx_f32(q_src[i % VALUES_PER_THREAD], source_lane), q_scale_factor),
                )
                K.ptx.mov.b32(
                    k_reg[i],
                    _mul(_shfl_idx_f32(k_src[i % VALUES_PER_THREAD], source_lane), k_scale_factor),
                )
                K.ptx.mov.b32(
                    gate_reg[i], _shfl_idx_f32(gate_src[i % VALUES_PER_THREAD], source_lane)
                )

            # --- late V load for the other schedules (recurrent_kda.py:437-439)
            if TILE_ROWS != 16:
                with K.If(tidx < TILE_ROWS), K.Then():
                    K.assign(
                        v_loaded, _bf16_to_f32(_load_bf16_bits(v, gidx(v_base + v_offset + tidx)))
                    )

            # --- sequential rank-1 recurrence (recurrent_kda.py:440-460) -------
            for j in range(ROWS):
                for i in range(8):
                    K.ptx.mov.b32(h_reg[j * 8 + i], _mul(h_reg[j * 8 + i], gate_reg[i]))

                pred = K.local_scalar("float32")
                v_idx = K.local_scalar("int32")
                v_val = K.local_scalar("float32")
                delta = K.local_scalar("float32")
                out_val = K.local_scalar("float32")
                K.assign(pred, _reduce_k_group(_dot8(h_reg, j * 8, k_reg, DOT_REDUCTION_SCHEDULE)))
                K.assign(v_idx, v_offset + v_lane + V_LANES * j)
                K.assign(v_val, _shfl_idx_f32(v_loaded, v_lane + V_LANES * j))
                K.assign(delta, _mul(_sub(v_val, pred), beta_val))

                for i in range(8):
                    K.ptx.mov.b32(h_reg[j * 8 + i], _fma(k_reg[i], delta, h_reg[j * 8 + i]))

                K.assign(
                    out_val, _reduce_k_group(_dot8(h_reg, j * 8, q_reg, DOT_REDUCTION_SCHEDULE))
                )

                _store_bf16_bits_pred(
                    out,
                    gidx(v_base + v_idx),
                    _f32_to_bf16(out_val),
                    K.And(is_active != 0, k_lane == j),
                )

            # --- state writeback (recurrent_kda.py:462-469) --------------------
            # The source rebinds h_out to seq_idx at :463.  At NUM_TOKENS == 1 that
            # is the same slot the prologue loaded from, and a negative slot cannot
            # reach here because it clears is_active (:327).
            with K.If(is_active != 0), K.Then():
                write_base = K.local_scalar(
                    "int64",
                    init=K.cast(seq_idx, "int64") * K.cast(STATE_SLOT_STRIDE, "int64")
                    + K.cast(state_head_base, "int64"),
                )
                for j in range(ROWS):
                    v_idx_w = K.local_scalar("int32", init=v_offset + v_lane + V_LANES * j)
                    words = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        K.ptx.mov.b32(
                            words[pair],
                            _pack_bf16x2(h_reg[j * 8 + 2 * pair + 1], h_reg[j * 8 + 2 * pair]),
                        )
                    _store_bf16x8_words(
                        state, write_base + K.cast(v_idx_w * HEAD_DIM, "int64"), words
                    )

    return _recurrent_kda_decode_one_warp.func


def get_kernel(**kwargs: Any):
    """Return the specialized one-warp recurrent-KDA decode PrimFunc."""
    return _make_recurrent_kda_decode_one_warp(_specialization(kwargs))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state."""
    from tirx_kernels.target import supports_sm100_kernel

    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for recurrent-KDA one-warp decode")
    capability = torch.cuda.get_device_capability(device)
    if not supports_sm100_kernel(capability):
        raise SkipTest(
            f"recurrent-KDA one-warp decode requires SM100 or prepared Thor, got {capability}"
        )

    spec = _specialization(kwargs)
    num_seqs = spec["NUM_SEQS"]
    num_heads = spec["NUM_HEADS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    pool_size = int(kwargs["pool_size"])
    slot_stride = spec["STATE_SLOT_STRIDE"]
    padded_slots = int(kwargs.get("padded_slots", 0))
    lower_bound = kwargs.get("lower_bound", DEFAULT_LOWER_BOUND)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(kwargs["seed"]))

    def randn(*shape: int, dtype=torch.bfloat16, gain: float = 1.0):
        raw = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        return (gain * raw).to(dtype)

    # Shapes and magnitudes follow sglang's make_decode_inputs
    # (benchmark/bench_linear_attention/bench_kda_flashinfer_mtp.py:50-78).
    q = randn(1, num_seqs, num_heads, HEAD_DIM, gain=0.5)
    k = randn(1, num_seqs, num_heads, HEAD_DIM, gain=0.5)
    v = randn(1, num_seqs, num_value_heads, HEAD_DIM, gain=0.5)
    # Raw per-K gate: the kernel activates it, so keep the pre-activation range.
    g = (
        0.5
        * torch.randn(
            (1, num_seqs, num_value_heads, HEAD_DIM),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        - 1.0
    ).to(torch.bfloat16)
    # sglang passes beta already sigmoided (kda_flashinfer.py:142-147).
    beta_logit = randn(1, num_seqs, num_value_heads, dtype=torch.float32, gain=0.5)
    beta = torch.sigmoid(beta_logit).to(torch.bfloat16)
    a_log = randn(num_heads, dtype=torch.float32, gain=0.2)
    dt_bias = randn(num_heads * HEAD_DIM, dtype=torch.float32, gain=0.1)

    cu_seqlens = torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
    slots = torch.arange(num_seqs, device=device, dtype=torch.int32)
    if padded_slots:
        # CUDA-graph padding poisons the tail rows with -1
        # (sglang hybrid_linear_attn_backend.py:112-115).
        slots[num_seqs - padded_slots :] = -1

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

    return {
        "spec": spec,
        "config": dict(kwargs),
        "device": device,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": slots,
        "tirx_state_raw": tirx_state_raw,
        "tirx_state": tirx_state,
        "reference_state_raw": reference_state_raw,
        "reference_state": reference_state,
        "initial_state_raw": initial_state_raw,
        "tirx_out": tirx_out,
        "scale": HEAD_DIM**-0.5,
        "eps": L2_EPS,
        "lower_bound": lower_bound,
    }


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    lower_bound = case["lower_bound"]
    return (
        case["q"].reshape(-1),
        case["k"].reshape(-1),
        case["v"].reshape(-1),
        case["g"].reshape(-1),
        case["beta"].reshape(-1),
        case["tirx_state_raw"],
        case["tirx_out"].reshape(-1),
        case["a_log"],
        case["dt_bias"],
        case["cu_seqlens"],
        case["ssm_state_indices"],
        float(case["scale"]),
        float(case["eps"]),
        float(lower_bound if lower_bound is not None else 0.0),
    )


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the FlashInfer CuTe DSL source on the reference state pool."""
    import importlib

    # flashinfer.kda_kernels.__init__ rebinds ``recurrent_kda`` to the run
    # function, so the submodule must be imported explicitly.
    fi = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
    out, _ = fi.run_recurrent_kda(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        scale=case["scale"],
        initial_state=case["reference_state"],
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=case["lower_bound"],
        cu_seqlens=case["cu_seqlens"],
        ssm_state_indices=case["ssm_state_indices"],
    )
    return out


# Two bfloat16 ULP.  The port is within one ULP of the source on every config
# measured; this leaves headroom without hiding a real divergence.
_RTOL = 2.0**-7
_ATOL = 1.0e-4
# The FP32 oracle reassociates freely, so it only needs to agree to bf16 noise.


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize()

    spec = case["spec"]
    shape = (1, spec["NUM_SEQS"], spec["NUM_VALUE_HEADS"], HEAD_DIM)

    # Primary: the ported kernel against the source implementation it transcribes.
    flashinfer_out = _flashinfer_reference(case).reshape(shape)
    torch.testing.assert_close(case["tirx_out"], flashinfer_out, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(
        case["tirx_state_raw"], case["reference_state_raw"], rtol=_RTOL, atol=_ATOL
    )

    # Inactive (negative-slot) rows must be zeroed and must not touch state.
    slots = case["ssm_state_indices"]
    inactive = (slots < 0).nonzero().flatten()
    if inactive.numel():
        padded = case["tirx_out"][0, inactive]
        if padded.abs().max().item() != 0.0:
            raise AssertionError("inactive output rows are not zero")
        touched = set(slots[slots >= 0].tolist())
        stride = spec["STATE_SLOT_STRIDE"]
        changed = (case["tirx_state_raw"] != case["initial_state_raw"]).nonzero().flatten()
        stray = set((changed // stride).tolist()) - touched
        if stray:
            raise AssertionError(f"state slots mutated without an active sequence: {sorted(stray)}")

    # Envelope-strided pools must leave their inter-slot padding untouched.
    payload = spec["NUM_VALUE_HEADS"] * HEAD_DIM * HEAD_DIM
    stride = spec["STATE_SLOT_STRIDE"]
    if stride > payload:
        pool = case["tirx_state_raw"].numel() // stride
        now = case["tirx_state_raw"].as_strided((pool, stride), (stride, 1))[:, payload:]
        before = case["initial_state_raw"].as_strided((pool, stride), (stride, 1))[:, payload:]
        if not torch.equal(now, before):
            raise AssertionError("inter-slot state padding was overwritten")


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = dict(prepared["config"])
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    args = _tirx_args(case)

    def flashinfer_builder():
        # Validate once before the reference enters the timed region.
        executable(*args)
        spec = case["spec"]
        shape = (1, spec["NUM_SEQS"], spec["NUM_VALUE_HEADS"], HEAD_DIM)
        flashinfer_out = _flashinfer_reference(case).reshape(shape)
        torch.cuda.synchronize()
        torch.testing.assert_close(case["tirx_out"], flashinfer_out, rtol=_RTOL, atol=_ATOL)
        torch.testing.assert_close(
            case["tirx_state_raw"], case["reference_state_raw"], rtol=_RTOL, atol=_ATOL
        )
        # Heavy import, CuTe JIT and warmup all happen here, outside the timing.
        for _ in range(2):
            _flashinfer_reference(case)
        torch.cuda.synchronize()

        def launch():
            _flashinfer_reference(case)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_cutedsl": flashinfer_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
