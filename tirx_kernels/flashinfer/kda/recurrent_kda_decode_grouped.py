# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# This file is a TIRx port of FlashInfer's
# flashinfer/kda_kernels/recurrent_kda.py::_grouped_kda_kernel
# (flashinfer-ai/flashinfer @ f2e04400, v0.6.18).
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

"""TIRx port of FlashInfer's grouped-CTA recurrent-KDA decode kernel.

Source CuTe DSL: ``flashinfer/kda_kernels/recurrent_kda.py``
(``_grouped_kda_kernel`` at line 482, host ``_grouped_kda_host`` at 729,
compile cache ``_get_grouped_compiled`` at 836, dispatch at 2090-2198).

This is the sibling of :mod:`recurrent_kda_decode_one_warp`.  FlashInfer's host
routes here whenever the one-warp predicate fails:

* ``NUM_TOKENS == 1`` with ``sequence_heads = N * HV < 128`` -- small-batch decode;
* **every** ``NUM_TOKENS > 1`` call -- SGLang ``target_verify`` (MTP/speculative),
  regardless of N and HV.

Unlike the one-warp kernel, this one is a genuine multi-warp CTA: it allocates
four shared-memory buffers and has exactly **one** ``__syncthreads()`` splitting
a token-preprocessing phase from a barrier-free sequential recurrence.  The
barrier exists because the two phases use *different* thread-to-element
mappings: phase A walks ``d = tid % D`` over ``EPT`` elements, phase B walks the
``part``-granule view of the state tile.

The host picks the tiling by a hard rule (``recurrent_kda.py:859-860``):
``KS = 4 if T == 1 else 2`` and ``VSPLIT = 4``.  That yields exactly two shapes:

===========  ====  ====  ====  ===  ===  ===
mode          T     KS    NT    G   EPT   SW
===========  ====  ====  ====  ===  ===  ===
decode        1     4     128   4    1    4
verify       >1     2     64    8    2    2
===========  ====  ====  ====  ===  ===  ===

Both gate modes SGLang can select are covered: ``GATE_MODE=2``
(``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, Kimi K3, ``-5.0``) and
``GATE_MODE=1`` (softplus with the ``x > 20`` linear guard, Kimi Linear).

Out of scope, because unreachable from the production call path or handled by
the sibling: ``GATE_MODE=0`` (pre-computed gate), ``USE_SRC=1``
(``initial_state_source``), ``BETA_LOGIT=1``, ``USE_L2=0``, ``HEAD_DIM != 128``,
GQA (``HV != H``), and single-token decode with ``N * HV >= 128`` (one-warp).
"""

from __future__ import annotations

from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

# Sibling import: the one-warp port is the canonical home for the shared PTX,
# bf16-conversion and shuffle helpers (mamba's convention, e.g.
# selective_state_update_stp_vertical.py:34 importing _simple).
from . import recurrent_kda_decode_one_warp as _one_warp

HEAD_DIM = _one_warp.HEAD_DIM  # 128
VSPLIT = 4  # recurrent_kda.py:860 -- fixed by the host
ONE_WARP_MIN_SEQUENCE_HEADS = _one_warp.ONE_WARP_MIN_SEQUENCE_HEADS  # 128
L2_EPS = _one_warp.L2_EPS  # 1e-6, hardcoded in the source at :663-664
DEFAULT_LOWER_BOUND = _one_warp.DEFAULT_LOWER_BOUND  # -5.0 (Kimi K3)

# recurrent_kda.py:479
LOG2_E = 1.4426950408889634
# recurrent_kda.py:620 -- softplus switches to the linear branch above this
SOFTPLUS_LINEAR_THRESHOLD = 20.0

GATE_MODE_PRECOMPUTED = 0
GATE_MODE_SOFTPLUS = 1
GATE_MODE_LOWER_BOUND = 2


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _row_lengths(num_seqs: int, num_tokens: int, empty_rows: int) -> list[int]:
    """Per-row token counts.

    Spec mode requires ``cu_seqlens`` to step by exactly ``T`` for *every* row,
    padded rows included (recurrent_kda.py:1658-1661); padding is signalled by
    ``ssm_state_indices == -1``, never by a short row.  That contract is load
    bearing rather than cosmetic: a row with ``seq_len < T`` leaves output
    positions ``token_base + seq_len .. token_base + T - 1`` written by nobody
    -- phase B's zero-fill is guarded by ``in_row`` (:714-717) and the orphan
    loop only covers ``pos >= cu[n_seq]`` (:719-725).  The source therefore has
    no defined result for a short spec row, so it cannot be a correctness case.

    The one in-contract way to make ``in_row`` false is a **zero-length row**
    in ``T = 1`` decode mode: every remaining output position is still owned by
    some row, so the result stays fully defined.
    """
    if empty_rows and num_tokens != 1:
        raise ValueError(
            "empty rows are only in contract for T=1 decode; spec mode requires "
            "cu_seqlens to step by T for every row (recurrent_kda.py:1658-1661)"
        )
    lens = [num_tokens] * num_seqs
    for i in range(min(empty_rows, num_seqs)):
        lens[i] = 0
    return lens


# ---------------------------------------------------------------------------
# PTX helpers specific to the grouped kernel
#
# The one-warp sibling supplies the shared scalar math, BF16 conversion and
# shuffle helpers; imported below.  What this kernel needs on top of those is
# the packed-FP32 granule arithmetic, three non-FTZ scalar forms the sibling
# does not use, and shared-memory accessors -- ``low_level_ir.py:26`` forbids
# BufferLoad/BufferStore on ``shared`` exactly as it does on ``global``, so
# every SMEM touch goes through ``ptr_to`` and raw PTX.
# ---------------------------------------------------------------------------

_ptx_un = _one_warp._ptx_un
_ptx_bin = _one_warp._ptx_bin
_ptx_ter = _one_warp._ptx_ter
_mul = _one_warp._mul
_add = _one_warp._add
_sub = _one_warp._sub
_fma = _one_warp._fma
_exp2 = _one_warp._exp2
_add_bf16 = _one_warp._add_bf16
_fma_bf16 = _one_warp._fma_bf16
_bf16_to_f32 = _one_warp._bf16_to_f32
_f32_to_bf16 = _one_warp._f32_to_bf16
_pack_bf16x2 = _one_warp._pack_bf16x2
_shfl_bfly_f32 = _one_warp._shfl_bfly_f32
_load_f32 = _one_warp._load_f32
_load_i32 = _one_warp._load_i32
_load_bf16_bits = _one_warp._load_bf16_bits
_store_bf16_bits = _one_warp._store_bf16_bits
_store_bf16_bits_pred = _one_warp._store_bf16_bits_pred


def _neg(a):
    """``neg.f32``.

    The GATE_MODE=2 sigmoid negates its argument with a real instruction; the
    sign is not folded into a negative LOG2_E constant (PTX ``.loc 625`` emits
    ``neg.f32`` then two ``mul.f32``).
    """
    return _ptx_un("neg.f32", a)


def _rsqrt_no_ftz(a):
    """``rsqrt.approx.f32`` -- approximate but NOT flush-to-zero.

    The sibling's ``_rsqrt`` is the ``.ftz`` form; this kernel's source keeps
    denormals here, and ``ex2`` is its only ``.ftz`` instruction.
    """
    return _ptx_un("rsqrt.approx.f32", a)


def _rcp_rn(a):
    """``rcp.rn.f32`` -- correctly rounded, not ``rcp.approx`` and not a divide."""
    return _ptx_un("rcp.rn.f32", a)


def _sub_bf16(bf_bits, f):
    """``sub.rn.f32.bf16`` -- the BF16 minuend never widens to FP32 first."""
    return _ptx_bin("sub.rn.f32.bf16", bf_bits, f)


def _pack_f32x2(lo, hi):
    """One ``.f32x2`` operand: ``lo`` is the low half, i.e. the lower address."""
    return T.cuda.make_float2(lo, hi)


def _vmul(a, b):
    """``mul.f32x2`` -- two packed FP32 lanes."""
    out = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mul.f32x2(out[0], a, b))
    return out[0]


def _vadd(a, b):
    """``add.f32x2`` -- two packed FP32 lanes.

    The source writes ``acc + x * y`` but the compiler emits a separate
    multiply and add; there is no ``fma.*.f32x2`` anywhere in its PTX, so the
    port must not fuse them either.
    """
    out = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.add.f32x2(out[0], a, b))
    return out[0]


def _ld_shared_b32(buffer, index):
    """``ld.shared.b32``."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(out[0], buffer.ptr_to([index])))
    return T.reinterpret("float32", out[0])


def _st_shared_f32(buffer, index, value):
    """``st.shared.b32``."""
    T.evaluate(T.ptx.st.shared.b32(buffer.ptr_to([index]), T.reinterpret("uint32", value)))


def _st_shared_f32_pred(buffer, index, value, pred):
    """``@p st.shared.b32`` -- the ``lane == 0 and wid < SW`` publication."""
    T.evaluate(
        T.ptx.st.shared.b32(buffer.ptr_to([index]), T.reinterpret("uint32", value), pred=pred)
    )


def _ld_shared_granule(buffer, index):
    """One 8-element FP32 granule as four packed pairs: 2x ``ld.shared.v2.b64``.

    A granule is 8 contiguous FP32 (32 B), which the source reads as two 16 B
    vectors; ``index`` is an FP32 element index and must be 16 B aligned.
    """
    pairs = T.alloc_local((4,), "uint64")
    T.evaluate(T.ptx.ld.shared.v2.b64(pairs[0], pairs[1], buffer.ptr_to([index])))
    T.evaluate(T.ptx.ld.shared.v2.b64(pairs[2], pairs[3], buffer.ptr_to([index + 4])))
    return pairs


def _ld_global_granule_no_alloc(buffer, index):
    """``ld.global.L1::no_allocate.v4.b32`` -- one 16 B BF16 state granule.

    Mirrors the source's ``autovec_copy(..., CacheEvictionPriority.NO_ALLOCATE)``
    on the state load (recurrent_kda.py:571).
    """
    words = T.alloc_local((4,), "uint32")
    T.evaluate(
        T.ptx["ld.global.L1::no_allocate.v4.b32"](
            words[0], words[1], words[2], words[3], buffer.ptr_to([index])
        )
    )
    return words


def _st_global_granule_no_alloc(buffer, index, words):
    """``st.global.L1::no_allocate.v4.b32`` -- one 16 B BF16 checkpoint granule."""
    T.evaluate(
        T.ptx["st.global.L1::no_allocate.v4.b32"](
            buffer.ptr_to([index]), words[0], words[1], words[2], words[3]
        )
    )


def _warp_reduce_sum(value):
    """``cute.arch.warp_reduction_sum`` -- a full five-round 32-lane butterfly.

    The offsets DESCEND because the source helper halves a group width. FP32
    addition is not associative and the checkpoints must be exact, so the order
    is semantic, not incidental.
    """
    for offset in (16, 8, 4, 2, 1):
        value = _add(value, _shfl_bfly_f32(value, offset))
    return value


def _ks_join(value, ks: int):
    """Join the ``KS`` K-slices of one column: ``(KS-1).bit_length()`` rounds.

    KS = 2 -> offset {1}; KS = 4 -> offsets {1, 2}, ascending (:678-680).
    """
    for off_i in range((ks - 1).bit_length()):
        value = _add(value, _shfl_bfly_f32(value, 1 << off_i))
    return value


def _softplus(x):
    """``log1p(exp(x))`` with the source's ``x > 20 -> x`` linear guard (:618-620).

    The source lowers ``cute.log1p`` to the inlined libdevice ``__nv_log1pf``
    polynomial.  TVM's ``log1p`` intrinsic reaches the same libdevice routine
    through nvcc, so this keeps the accuracy contract without hand-expanding the
    minimax chain.  Only GATE_MODE=1 uses it, and no benchmark row does, so this
    is a correctness obligation rather than a performance-alignment target.
    """
    sp = T.log1p(_exp2(_mul(x, LOG2_E)))
    return T.Select(x > SOFTPLUS_LINEAR_THRESHOLD, x, sp)


def _grouped_tiling(num_tokens: int) -> dict[str, int]:
    """Reproduce the host's tiling rule (recurrent_kda.py:859-860, :516-521)."""
    ks = 4 if num_tokens == 1 else 2
    nt = (HEAD_DIM * ks) // VSPLIT
    return {
        "KS": ks,
        "KC": HEAD_DIM // ks,
        "G": (HEAD_DIM // ks) // 8,
        "CPB": HEAD_DIM // VSPLIT,
        "NT": nt,
        "EPT": max(HEAD_DIM // nt, 1),
        "SW": min(HEAD_DIM, nt) // 32,
    }


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row.  ``mode`` selects the decode or verify input family."""
    config: dict[str, Any] = {
        "label": label,
        "mode": "verify",  # "decode" (T=1) or "verify" (T>1)
        "num_seqs": 8,
        "num_tokens": 8,  # T; must be 1 for mode="decode"
        "num_heads": 16,
        "num_value_heads": 16,
        "scratch_steps": None,  # None -> equals num_tokens (S == T)
        "pool_size": 512,
        "lower_bound": DEFAULT_LOWER_BOUND,  # None -> softplus gate
        "padded_slots": 0,  # rows whose ssm index is -1
        "empty_rows": 0,  # verify rows with seq_len < T
        "orphan_tokens": 0,  # q_total beyond cu[n_seq]
        "seed": 20260812,
    }
    config.update(overrides)
    return config


def _dec(label: str, **kw: Any) -> dict[str, Any]:
    kw.setdefault("mode", "decode")
    kw.setdefault("num_tokens", 1)
    return _case(label, **kw)


# All bench rows use the Kimi K3 gate (lower_bound = -5.0) and sit inside the
# grouped dispatch domain.  Decode rows need N*HV < 128 or the host would pick
# the one-warp kernel; verify rows reach grouped for every N and HV.
BENCH_CONFIGS = [
    # -- decode, T=1 (KS=4, NT=128) -------------------------------------------
    _dec("dec_hv16_b1", num_seqs=1),  # sequence_heads 16
    _dec("dec_hv16_b4", num_seqs=4),  # 64
    _dec("dec_hv12_b4", num_seqs=4, num_heads=12, num_value_heads=12),  # 48
    _dec("dec_hv12_b8", num_seqs=8, num_heads=12, num_value_heads=12),  # 96
    # -- verify, T=8 (KS=2, NT=64) -- production DSPARK gamma=7, sglang parity --
    _case("ver_t8_hv16_b1", num_seqs=1),
    _case("ver_t8_hv16_b4", num_seqs=4),
    _case("ver_t8_hv16_b16", num_seqs=16),
    _case("ver_t8_hv16_b32", num_seqs=32),
    _case("ver_t8_hv16_b64", num_seqs=64),
    _case("ver_t8_hv16_b128", num_seqs=128),
    _case("ver_t8_hv12_b16", num_seqs=16, num_heads=12, num_value_heads=12),
    _case("ver_t8_hv12_b64", num_seqs=64, num_heads=12, num_value_heads=12),
    # -- verify, other draft-window sizes -------------------------------------
    _case("ver_t4_hv16_b32", num_seqs=32, num_tokens=4),  # test parity (32, 3)
    _case("ver_t2_hv16_b8", num_seqs=8, num_tokens=2),  # legal floor
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # Softplus gate (Kimi Linear): GATE_MODE=1, incl. the x>20 linear guard.
    _dec("dec_hv16_b4_sp", num_seqs=4, lower_bound=None),
    _case("ver_t8_hv16_b4_sp", num_seqs=4, lower_bound=None),
    _case("ver_t2_hv16_b8_sp", num_seqs=8, num_tokens=2, lower_bound=None),
    # CUDA-graph padding: negative ssm_state_indices rows must produce zeroed
    # output and leave their state slots untouched.
    _dec("dec_hv16_b7_padded", num_seqs=7, padded_slots=3),  # 112 < 128
    _case("ver_t8_hv16_b4_padded", num_seqs=4, padded_slots=2),
    # Zero-length decode row: the one in-contract way to make in_row False.
    # (A short *spec* row is out of contract -- see _row_lengths.)
    _dec("dec_hv16_b6_empty", num_seqs=6, empty_rows=1),
    # Allocated scratch stride S > T (tests use T+2): ssm index stride is
    # scratch_steps, not T.
    _case("ver_t4_hv16_b8_s6", num_seqs=8, num_tokens=4, scratch_steps=6),
    # Orphan packed suffix: q_total > cu[n_seq] must be zero-filled by the
    # kernel's tail loop (:719-725).  Decode-only -- spec mode sizes out_buf as
    # N*NUM_TOKENS (:1810) while cu must step by T, so q_total == cu[n_seq]
    # there and the tail loop is unreachable (it would write out of bounds).
    _dec("dec_hv16_b4_orphan", num_seqs=4, orphan_tokens=3),
    # Boundary: the largest decode shape still inside the grouped domain
    # (sequence_heads 112 < 128).
    _dec("dec_hv16_b7", num_seqs=7),
]

KERNEL_META = {
    "name": "recurrent_kda_decode_grouped",
    "category": "flashinfer",
    "compute_capability": 10,
}


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    mode = str(kwargs.get("mode", "verify"))
    num_seqs = int(kwargs["num_seqs"])
    num_tokens = int(kwargs["num_tokens"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    pool_size = int(kwargs["pool_size"])
    lower_bound = kwargs.get("lower_bound", DEFAULT_LOWER_BOUND)
    scratch_steps = kwargs.get("scratch_steps") or num_tokens
    orphan_tokens = int(kwargs.get("orphan_tokens", 0))

    if num_value_heads != num_heads:
        raise ValueError("GQA (HV != H) is outside this port's scope")
    if mode == "decode" and num_tokens != 1:
        raise ValueError("mode='decode' requires num_tokens == 1")
    if mode == "verify" and num_tokens < 2:
        raise ValueError("mode='verify' requires num_tokens >= 2")
    if scratch_steps < num_tokens:
        raise ValueError("scratch_steps must be >= num_tokens")

    # Grouped dispatch domain (recurrent_kda.py:1797-1798).  T>1 always lands
    # here; T==1 only when the one-warp predicate fails.
    sequence_heads = num_seqs * num_value_heads
    if num_tokens == 1 and sequence_heads >= ONE_WARP_MIN_SEQUENCE_HEADS:
        raise ValueError(
            f"sequence_heads={sequence_heads} dispatches to the one-warp kernel; "
            f"the grouped port requires < {ONE_WARP_MIN_SEQUENCE_HEADS} at T=1 "
            "(recurrent_kda.py:1798)"
        )

    tiling = _grouped_tiling(num_tokens)
    gate_mode = (
        GATE_MODE_SOFTPLUS if lower_bound is None else GATE_MODE_LOWER_BOUND
    )  # recurrent_kda.py:2140

    empty_rows = int(kwargs.get("empty_rows", 0))
    q_total = sum(_row_lengths(num_seqs, num_tokens, empty_rows)) + orphan_tokens
    # Shared memory: three T*D f32 planes plus the T*16 f32 reduce scratch,
    # each 16-byte aligned (recurrent_kda.py:526-530).
    plane = 4 * num_tokens * HEAD_DIM
    off_eg = 0
    off_kr = _align_up(off_eg + plane, 16)
    off_qr = _align_up(off_kr + plane, 16)
    off_red = _align_up(off_qr + plane, 16)
    shared_bytes = _align_up(off_red + 4 * num_tokens * 16, 16)

    # State pool: committed [pool, HV, D, D] for decode; scratch viewed as
    # [N*S, HV, D, D] for verify (recurrent_kda.py kda_flashinfer.py:307-309).
    state_slots = pool_size if mode == "decode" else num_seqs * scratch_steps
    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM

    return {
        "NUM_SEQS": num_seqs,
        "NUM_TOKENS": num_tokens,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "RATIO": num_value_heads // num_heads,
        "KS": tiling["KS"],
        "G": tiling["G"],
        "CPB": tiling["CPB"],
        "NT": tiling["NT"],
        "EPT": tiling["EPT"],
        "SW": tiling["SW"],
        "NUM_WARPS": tiling["NT"] // 32,
        # Defer the state widening past the barrier only for small T; see the
        # comment at the deferred block for the measured split.
        "VSPLIT": VSPLIT,
        "GATE_MODE": gate_mode,
        "Q_TOTAL": q_total,
        "QK_ELEMENTS": q_total * num_heads * HEAD_DIM,
        "V_ELEMENTS": q_total * num_value_heads * HEAD_DIM,
        "BETA_ELEMENTS": q_total * num_value_heads,
        "STATE_ELEMENTS": state_slots * slot_stride,
        "STATE_SLOT_STRIDE": slot_stride,
        "A_LOG_ELEMENTS": num_heads,
        "DT_BIAS_ELEMENTS": num_heads * HEAD_DIM,
        "CU_ELEMENTS": num_seqs + 1,
        "SSM_IDX_ELEMENTS": num_seqs * num_tokens,
        "NAT_ELEMENTS": num_seqs,
        "OFF_EG": off_eg,
        "OFF_KR": off_kr,
        "OFF_QR": off_qr,
        "OFF_RED": off_red,
        "SHARED_BYTES": shared_bytes,
    }


@T.jit
def _recurrent_kda_decode_grouped(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    a_log_h: T.handle,
    dt_bias_h: T.handle,
    cu_h: T.handle,
    ssm_idx_h: T.handle,
    nat_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    q_total: T.int32,
    g_stride_q: T.int32,
    state_slot_stride: T.int64,
    scale: T.float32,
    lower_bound: T.float32,
    *,
    NUM_SEQS: T.constexpr,
    NUM_TOKENS: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_VALUE_HEADS: T.constexpr,
    RATIO: T.constexpr,
    KS: T.constexpr,
    G: T.constexpr,
    CPB: T.constexpr,
    NT: T.constexpr,
    EPT: T.constexpr,
    SW: T.constexpr,
    NUM_WARPS: T.constexpr,
    VSPLIT: T.constexpr,
    GATE_MODE: T.constexpr,
    Q_TOTAL: T.constexpr,
    QK_ELEMENTS: T.constexpr,
    V_ELEMENTS: T.constexpr,
    BETA_ELEMENTS: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    STATE_SLOT_STRIDE: T.constexpr,
    A_LOG_ELEMENTS: T.constexpr,
    DT_BIAS_ELEMENTS: T.constexpr,
    CU_ELEMENTS: T.constexpr,
    SSM_IDX_ELEMENTS: T.constexpr,
    NAT_ELEMENTS: T.constexpr,
    OFF_EG: T.constexpr,
    OFF_KR: T.constexpr,
    OFF_QR: T.constexpr,
    OFF_RED: T.constexpr,
    SHARED_BYTES: T.constexpr,
):
    q = T.match_buffer(q_h, (QK_ELEMENTS,), "bfloat16", scope="global")
    k = T.match_buffer(k_h, (QK_ELEMENTS,), "bfloat16", scope="global")
    v = T.match_buffer(v_h, (V_ELEMENTS,), "bfloat16", scope="global")
    g = T.match_buffer(g_h, (V_ELEMENTS,), "bfloat16", scope="global")
    beta = T.match_buffer(beta_h, (BETA_ELEMENTS,), "bfloat16", scope="global")
    a_log = T.match_buffer(a_log_h, (A_LOG_ELEMENTS,), "float32", scope="global")
    dt_bias = T.match_buffer(dt_bias_h, (DT_BIAS_ELEMENTS,), "float32", scope="global")
    cu = T.match_buffer(cu_h, (CU_ELEMENTS,), "int32", scope="global")
    ssm_idx = T.match_buffer(ssm_idx_h, (SSM_IDX_ELEMENTS,), "int32", scope="global")
    nat = T.match_buffer(nat_h, (NAT_ELEMENTS,), "int32", scope="global")
    state = T.match_buffer(state_h, (STATE_ELEMENTS,), "bfloat16", scope="global")
    out = T.match_buffer(out_h, (V_ELEMENTS,), "bfloat16", scope="global")
    T.device_entry()
    # TIRX_TRANSCRIBE_START recurrent_kda_decode_grouped
    # --- CTA and thread coordinates (recurrent_kda.py:512-524) -------------
    hv, n, vz = T.cta_id([NUM_VALUE_HEADS, NUM_SEQS, VSPLIT])
    # A flat NT-thread block, matching the source's block=(NT, 1, 1); a 2-D
    # [32, NUM_WARPS] block would make every use of `tid` read two special
    # registers instead of one.
    tid = T.thread_id([NT])
    lane: T.int32 = tid % 32
    wid: T.int32 = tid // 32
    h: T.int32 = hv // RATIO
    v_idx: T.int32 = vz * CPB + tid // KS  # the state column this thread owns
    part: T.int32 = tid % KS  # which K-slice of that column
    d: T.int32 = tid % HEAD_DIM  # phase-A element base

    # --- shared memory (recurrent_kda.py:526-541) --------------------------
    # One 16B-aligned arena carved into the three T-batched staging planes plus
    # the reduction scratch.  Every access below is raw PTX: low_level_ir.py:26
    # forbids BufferLoad/BufferStore on `shared` just as it does on `global`.
    shared_raw = T.alloc_buffer((SHARED_BYTES,), "uint8", scope="shared", align=16)
    s_eg = T.decl_buffer(
        (NUM_TOKENS * HEAD_DIM,),
        "float32",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_EG,
        align=16,
    )
    s_kr = T.decl_buffer(
        (NUM_TOKENS * HEAD_DIM,),
        "float32",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_KR,
        align=16,
    )
    s_qr = T.decl_buffer(
        (NUM_TOKENS * HEAD_DIM,),
        "float32",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_QR,
        align=16,
    )
    s_red = T.decl_buffer(
        (NUM_TOKENS * 16,),
        "float32",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_RED,
        align=16,
    )

    # --- row bounds (recurrent_kda.py:543-544) -----------------------------
    token_base: T.int32 = _load_i32(cu, n)
    seq_len: T.int32 = _load_i32(cu, n + 1) - token_base

    # --- initial checkpoint slot (recurrent_kda.py:551-560) ----------------
    # `nat` is read only for T > 1; SGLang never supplies it, so FlashInfer
    # substitutes a cached ones vector and `ic` collapses to 0.
    ic: T.int32 = 0
    if NUM_TOKENS > 1:
        ic = T.min(T.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1)
    slot0: T.int32 = T.max(_load_i32(ssm_idx, n * NUM_TOKENS + ic), 0)

    # --- state load: one (8, G) granule per thread (recurrent_kda.py:562-574)
    # Element (e, gi) sits at e + gi*KS*8 + part*8, so each granule's eight
    # elements are contiguous: one 16B eviction-hinted vector load.
    head_row: T.int32 = hv * HEAD_DIM * HEAD_DIM + v_idx * HEAD_DIM
    read_base = T.cast(slot0, "int64") * T.cast(STATE_SLOT_STRIDE, "int64") + T.cast(
        head_row, "int64"
    )
    s_pairs = T.alloc_local((4 * G,), "uint64")  # THE recurrent carry
    s_words = T.alloc_local((4 * G,), "uint32")  # BF16 pairs, widened after the barrier
    for gi in range(G):
        words = _ld_global_granule_no_alloc(
            state, read_base + T.cast(gi * KS * 8 + part * 8, "int64")
        )
        for pr in range(4):
            s_words[gi * 4 + pr] = words[pr]

    # --- loop-invariant gate constants (recurrent_kda.py:576-586) ----------
    av: T.float32 = T.float32(1.0)
    if GATE_MODE != GATE_MODE_PRECOMPUTED:
        av = _exp2(_mul(_load_f32(a_log, h), T.float32(LOG2_E)))
    dtb = T.alloc_local((EPT,), "float32")
    for e in range(EPT):
        dtb[e] = _load_f32(dt_bias, h * HEAD_DIM + d + e * NT)

    # =======================================================================
    # Phase A: stage every token's gate/key/query (recurrent_kda.py:588-640)
    # Thread mapping is `d + e*NT` here, NOT the (v_idx, part) mapping below.
    # =======================================================================
    slots = T.alloc_local((NUM_TOKENS,), "int32")
    ves = T.alloc_local((NUM_TOKENS,), "uint16")  # stays BF16 until :681
    bbs = T.alloc_local((NUM_TOKENS,), "float32")

    # Issue every token's global loads before consuming any of them.  With a
    # cold L2 -- which is what bench_suite measures, and what an inference
    # server sees -- the kernel is bound by how many DRAM misses are in flight,
    # not by instruction count.  Loading and consuming one token at a time keeps
    # only EPT*3+2 requests outstanding; hoisting all T tokens keeps T times as
    # many.
    q_bits = T.alloc_local((NUM_TOKENS * EPT,), "uint16")
    k_bits = T.alloc_local((NUM_TOKENS * EPT,), "uint16")
    g_bits = T.alloc_local((NUM_TOKENS * EPT,), "uint16")
    b_bits = T.alloc_local((NUM_TOKENS,), "uint16")
    for t in range(NUM_TOKENS):
        slots[t] = _load_i32(ssm_idx, n * NUM_TOKENS + t)
        # Out-of-row tokens clamp to token 0 so the loads stay in bounds; the
        # value is discarded by the `active` predicate in phase B.
        pidx: T.int32 = T.if_then_else(t < seq_len, token_base + t, 0)
        ves[t] = _load_bf16_bits(v, (pidx * NUM_VALUE_HEADS + hv) * HEAD_DIM + v_idx)
        b_bits[t] = _load_bf16_bits(beta, pidx * NUM_VALUE_HEADS + hv)
        for e in range(EPT):
            de_l: T.int32 = d + e * NT
            q_bits[t * EPT + e] = _load_bf16_bits(q, (pidx * NUM_HEADS + h) * HEAD_DIM + de_l)
            k_bits[t * EPT + e] = _load_bf16_bits(k, (pidx * NUM_HEADS + h) * HEAD_DIM + de_l)
            g_bits[t * EPT + e] = _load_bf16_bits(g, pidx * g_stride_q + hv * HEAD_DIM + de_l)
    for t in range(NUM_TOKENS):
        bbs[t] = _bf16_to_f32(b_bits[t])

        sqp: T.float32 = T.float32(0.0)
        skp: T.float32 = T.float32(0.0)
        for e in range(EPT):
            de: T.int32 = d + e * NT
            qe = q_bits[t * EPT + e]
            ke = k_bits[t * EPT + e]
            ge = g_bits[t * EPT + e]

            # The L2 partials consume the raw BF16 registers.
            sqp = _fma_bf16(qe, qe, sqp)
            skp = _fma_bf16(ke, ke, skp)

            x: T.float32 = _add_bf16(ge, dtb[e])
            gate: T.float32 = T.float32(0.0)
            if GATE_MODE == GATE_MODE_SOFTPLUS:
                gate = _mul(_softplus(x), _neg(av))
            else:
                # The negation is its own instruction; it is not folded into a
                # negative LOG2_E constant (recurrent_kda.py:623-627).
                sig_e = _exp2(_mul(_mul(av, _neg(x)), T.float32(LOG2_E)))
                gate = _mul(lower_bound, _rcp_rn(_add(sig_e, T.float32(1.0))))

            # The staged key/query are the RAW values; L2 normalization is a
            # scalar factor applied in phase B.  They convert here because the
            # staging planes are FP32.
            _st_shared_f32(s_eg, t * HEAD_DIM + de, _exp2(_mul(gate, T.float32(LOG2_E))))
            _st_shared_f32(s_kr, t * HEAD_DIM + de, _bf16_to_f32(ke))
            _st_shared_f32(s_qr, t * HEAD_DIM + de, _bf16_to_f32(qe))

        # Full 32-lane butterfly, five rounds, DESCENDING offsets: the source's
        # warp_reduction_sum halves a group width.  FP32 addition is not
        # associative and the checkpoints must be exact, so the order matters.
        sqp = _warp_reduce_sum(sqp)
        skp = _warp_reduce_sum(skp)
        publish = T.And(lane == 0, wid < SW)
        _st_shared_f32_pred(s_red, t * 16 + wid, sqp, publish)
        _st_shared_f32_pred(s_red, t * 16 + 8 + wid, skp, publish)

    # The ONLY barrier: it separates the two thread-index mappings.
    T.cuda.cta_sync()

    # Nothing in phase A reads the state, so this placement reaches only
    # scheduling: same op, same extent, same instruction, same dependence order
    # as widening at the load site.  It is worth 15-20% on the T = 1 decode
    # shapes (dec_hv16_b4 0.855 -> 1.06-1.09 measured on a clock-verified GPU)
    # and is within run-to-run noise on every T > 1 shape, so it is
    # unconditional.  It used to sit behind a `T <= 2` constexpr because before
    # phase A issued all T tokens' loads up front it cost 2-3% at T = 8; that
    # cost is gone, and with it the boundary.
    for gi in range(G):
        for pr in range(4):
            w = s_words[gi * 4 + pr]
            lo = _bf16_to_f32(T.cast(w, "uint16"))
            hi = _bf16_to_f32(T.cast(T.shift_right(w, T.uint32(16)), "uint16"))
            s_pairs[gi * 4 + pr] = _pack_f32x2(lo, hi)

    # =======================================================================
    # Phase B: sequential recurrence over the tokens (recurrent_kda.py:645-717)
    # =======================================================================
    kreg = T.alloc_local((4 * G,), "uint64")  # keys, loaded in pass 1, reused in pass 2
    pvec = T.alloc_local((4,), "uint64")
    ovec = T.alloc_local((4,), "uint64")
    pf = T.alloc_local((8,), "float32")
    words_w = T.alloc_local((4,), "uint32")

    for t in range(NUM_TOKENS):
        slot: T.int32 = slots[t]
        in_row = t < seq_len
        active = T.And(in_row, slot >= 0)
        if active:
            pidx_b: T.int32 = token_base + t
            base_t: T.int32 = t * HEAD_DIM + part * 8

            # ---- L2 factors (recurrent_kda.py:657-664); eps is hardcoded ----
            sqt: T.float32 = T.float32(0.0)
            skt: T.float32 = T.float32(0.0)
            for w in range(SW):
                sqt = _add(sqt, _ld_shared_b32(s_red, t * 16 + w))
                skt = _add(skt, _ld_shared_b32(s_red, t * 16 + 8 + w))
            rk: T.float32 = _rsqrt_no_ftz(_add(skt, T.float32(L2_EPS)))
            rq: T.float32 = _mul(_rsqrt_no_ftz(_add(sqt, T.float32(L2_EPS))), scale)

            # ---- pass 1: decay the state, accumulate the raw prediction ----
            # gi == 0 is peeled so pvec is initialized by a mul, not fill+add.
            egp = _ld_shared_granule(s_eg, base_t)
            krp = _ld_shared_granule(s_kr, base_t)
            for pr in range(4):
                sv = _vmul(s_pairs[pr], egp[pr])
                s_pairs[pr] = sv
                kreg[pr] = krp[pr]
                pvec[pr] = _vmul(krp[pr], sv)
            for gi in range(1, G):
                egp = _ld_shared_granule(s_eg, base_t + gi * KS * 8)
                krp = _ld_shared_granule(s_kr, base_t + gi * KS * 8)
                for pr in range(4):
                    sv = _vmul(s_pairs[gi * 4 + pr], egp[pr])
                    s_pairs[gi * 4 + pr] = sv
                    kreg[gi * 4 + pr] = krp[pr]
                    # mul then add -- the source emits no fma.*.f32x2.
                    pvec[pr] = _vadd(pvec[pr], _vmul(krp[pr], sv))

            # ---- balanced 8-term tree, then the KS butterfly join ----
            for pr in range(4):
                pf[2 * pr] = T.cuda.float2_x(pvec[pr])
                pf[2 * pr + 1] = T.cuda.float2_y(pvec[pr])
            pred: T.float32 = _add(
                _add(_add(pf[0], pf[1]), _add(pf[2], pf[3])),
                _add(_add(pf[4], pf[5]), _add(pf[6], pf[7])),
            )
            pred = _ks_join(pred, KS)

            # ---- delta rule (:681): rk appears TWICE, by design ----
            deltak: T.float32 = _mul(_mul(rk, bbs[t]), _sub_bf16(ves[t], _mul(rk, pred)))
            dpair = _pack_f32x2(deltak, deltak)

            # ---- pass 2: rank-1 update, accumulate the raw output ----
            qrp = _ld_shared_granule(s_qr, base_t)
            for pr in range(4):
                sv = _vadd(s_pairs[pr], _vmul(kreg[pr], dpair))
                s_pairs[pr] = sv
                ovec[pr] = _vmul(qrp[pr], sv)
            for gi in range(1, G):
                qrp = _ld_shared_granule(s_qr, base_t + gi * KS * 8)
                for pr in range(4):
                    sv = _vadd(s_pairs[gi * 4 + pr], _vmul(kreg[gi * 4 + pr], dpair))
                    s_pairs[gi * 4 + pr] = sv
                    ovec[pr] = _vadd(ovec[pr], _vmul(qrp[pr], sv))

            for pr in range(4):
                pf[2 * pr] = T.cuda.float2_x(ovec[pr])
                pf[2 * pr + 1] = T.cuda.float2_y(ovec[pr])
            o: T.float32 = _add(
                _add(_add(pf[0], pf[1]), _add(pf[2], pf[3])),
                _add(_add(pf[4], pf[5]), _add(pf[6], pf[7])),
            )
            o = _ks_join(o, KS)

            # ---- output store, owned by part == 0 (:698-699) ----
            # Predicated rather than branch-guarded: inline asm is opaque to
            # ptxas, so an `if` around an asm store can never be if-converted.
            _store_bf16_bits_pred(
                out,
                (pidx_b * NUM_VALUE_HEADS + hv) * HEAD_DIM + v_idx,
                _f32_to_bf16(_mul(o, rq)),
                part == 0,
            )

            # ---- BF16 checkpoint write, eviction-hinted (:702-713) ----
            write_base = T.cast(slot, "int64") * T.cast(STATE_SLOT_STRIDE, "int64") + T.cast(
                head_row, "int64"
            )
            for gi in range(G):
                for pr in range(4):
                    words_w[pr] = _pack_bf16x2(
                        T.cuda.float2_y(s_pairs[gi * 4 + pr]), T.cuda.float2_x(s_pairs[gi * 4 + pr])
                    )
                _st_global_granule_no_alloc(
                    state, write_base + T.cast(gi * KS * 8 + part * 8, "int64"), words_w
                )
        else:
            # Pad rows still own their output element: the host allocates `out`
            # uninitialized because the kernel defines every slot.
            _store_bf16_bits_pred(
                out,
                ((token_base + t) * NUM_VALUE_HEADS + hv) * HEAD_DIM + v_idx,
                T.uint16(0),
                T.And(in_row, part == 0),
            )

    # --- orphan packed suffix (recurrent_kda.py:719-725) --------------------
    # Carrier tokens owned by no row.  Reachable only in the T == 1 decode
    # layout: spec mode sizes `out` as N*NUM_TOKENS while cu_seqlens must step
    # by T, so q_total == cu[n_seq] there and this loop is empty.
    if T.And(T.And(n == 0, vz == 0), tid < HEAD_DIM):
        covered: T.int32 = _load_i32(cu, NUM_SEQS)
        for pos in T.serial(covered, q_total):
            for e in range(EPT):
                _store_bf16_bits(
                    out, (pos * NUM_VALUE_HEADS + hv) * HEAD_DIM + tid + e * NT, T.uint16(0)
                )


def get_kernel(**kwargs: Any):
    """Return the specialized grouped-CTA recurrent-KDA decode PrimFunc."""
    return _recurrent_kda_decode_grouped.specialize(**_specialization(kwargs))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Two families, matching the two SGLang call sites:

    * ``mode="decode"`` -- ``kda_flashinfer.py::decode`` with a small batch, so
      ``N * HV < 128`` and the host picks the grouped kernel.  The committed
      pool is updated in place.
    * ``mode="verify"`` -- ``kda_flashinfer.py::target_verify``: packed
      ``[1, N*T, ...]`` inputs, draft-stride ``cu_seqlens``, and the scratch
      ``intermediate_ssm`` pool as ``initial_state`` with step 0 python-seeded
      from the committed pool (``:299-304``).  ``ssm_state_indices`` is the
      ``[N, T]`` matrix ``base_rows * scratch_steps + arange(T)`` (``:293-296``),
      so its stride is the *allocated* step count, which may exceed ``T``.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for grouped recurrent-KDA decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(
            f"grouped recurrent-KDA decode targets compute capability 10.x, got {capability}"
        )

    spec = _specialization(kwargs)
    mode = str(kwargs.get("mode", "verify"))
    n_seq = spec["NUM_SEQS"]
    tokens = spec["NUM_TOKENS"]
    heads = spec["NUM_HEADS"]
    vheads = spec["NUM_VALUE_HEADS"]
    q_total = spec["Q_TOTAL"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    pool_size = int(kwargs["pool_size"])
    scratch_steps = kwargs.get("scratch_steps") or tokens
    lower_bound = kwargs.get("lower_bound", DEFAULT_LOWER_BOUND)
    padded_slots = int(kwargs.get("padded_slots", 0))
    empty_rows = int(kwargs.get("empty_rows", 0))
    orphan_tokens = int(kwargs.get("orphan_tokens", 0))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(kwargs["seed"]))

    def randn(*shape: int, dtype=torch.bfloat16, gain: float = 1.0):
        raw = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        return (gain * raw).to(dtype)

    # Magnitudes follow sglang's make_verify_inputs / make_decode_inputs
    # (bench_kda_flashinfer_mtp.py:50-121).
    q = randn(1, q_total, heads, HEAD_DIM, gain=0.5)
    k = randn(1, q_total, heads, HEAD_DIM, gain=0.5)
    v = randn(1, q_total, vheads, HEAD_DIM, gain=0.5)
    g = (
        0.5
        * torch.randn(
            (1, q_total, vheads, HEAD_DIM), device=device, dtype=torch.float32, generator=gen
        )
        - 1.0
    ).to(torch.bfloat16)
    # Softplus has a linear branch above x > 20 (recurrent_kda.py:620); push a
    # slice of the gate past it so that specialization is actually exercised.
    if lower_bound is None and q_total > 0:
        g_f32 = g.float()
        g_f32[:, :, :, :8] = 25.0
        g = g_f32.to(torch.bfloat16)
    beta_logit = randn(1, q_total, vheads, dtype=torch.float32, gain=0.5)
    beta = torch.sigmoid(beta_logit).to(torch.bfloat16)  # sglang pre-sigmoids
    a_log = randn(heads, dtype=torch.float32, gain=0.2)
    dt_bias = randn(heads * HEAD_DIM, dtype=torch.float32, gain=0.1)

    # cu_seqlens: decode = one token per row (a zero-length row is allowed and
    # is what exercises in_row); verify = draft stride T for every row, padded
    # rows included (recurrent_kda.py:1658-1661).
    if mode == "decode":
        offsets = [0]
        for row_len in _row_lengths(n_seq, tokens, empty_rows):
            offsets.append(offsets[-1] + row_len)
        cu = torch.tensor(offsets, device=device, dtype=torch.int32)
    else:
        lens = _row_lengths(n_seq, tokens, 0)
        offsets = [0]
        for row_len in lens:
            offsets.append(offsets[-1] + row_len)
        cu = torch.tensor(offsets, device=device, dtype=torch.int32)

    committed = randn(pool_size, vheads, HEAD_DIM, HEAD_DIM, gain=0.01)

    if mode == "decode":
        slots = torch.arange(n_seq, device=device, dtype=torch.int32)
        if padded_slots:
            slots[n_seq - padded_slots :] = -1
        ssm_idx = slots.reshape(n_seq, 1)
        tirx_state_raw = committed.reshape(-1).clone()
        reference_state_raw = tirx_state_raw.clone()
        state_slots = pool_size
    else:
        base_rows = torch.arange(n_seq, device=device, dtype=torch.int32)
        ssm_idx = (
            base_rows[:, None] * scratch_steps
            + torch.arange(tokens, device=device, dtype=torch.int32)[None, :]
        ).contiguous()
        if padded_slots:
            ssm_idx[n_seq - padded_slots :, :] = -1
        scratch = torch.zeros(
            n_seq * scratch_steps * slot_stride, device=device, dtype=torch.bfloat16
        )
        # Step-0 seeding, exactly as the sglang wrapper does before the launch.
        seed_src = committed[:n_seq].reshape(n_seq, slot_stride)
        scratch_view = scratch.reshape(n_seq, scratch_steps, slot_stride)
        scratch_view[:, 0, :] = seed_src
        tirx_state_raw = scratch
        reference_state_raw = scratch.clone()
        state_slots = n_seq * scratch_steps

    initial_state_raw = tirx_state_raw.clone()
    nat = torch.ones(n_seq, device=device, dtype=torch.int32)  # sglang's fallback
    tirx_out = torch.empty((1, q_total, vheads, HEAD_DIM), device=device, dtype=torch.bfloat16)

    return {
        "spec": spec,
        "config": dict(kwargs),
        "mode": mode,
        "device": device,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu,
        "ssm_state_indices": ssm_idx,
        "nat": nat,
        "committed": committed,
        "tirx_state_raw": tirx_state_raw,
        "reference_state_raw": reference_state_raw,
        "initial_state_raw": initial_state_raw,
        "state_slots": state_slots,
        "scratch_steps": scratch_steps,
        "tirx_out": tirx_out,
        "scale": HEAD_DIM**-0.5,
        "eps": L2_EPS,
        "lower_bound": lower_bound,
        "q_total": q_total,
        "orphan_tokens": orphan_tokens,
    }


def _state_view(raw: torch.Tensor, case: dict[str, Any]) -> torch.Tensor:
    spec = case["spec"]
    return raw.as_strided(
        (case["state_slots"], spec["NUM_VALUE_HEADS"], HEAD_DIM, HEAD_DIM),
        (spec["STATE_SLOT_STRIDE"], HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    spec = case["spec"]
    lower_bound = case["lower_bound"]
    return (
        case["q"].reshape(-1),
        case["k"].reshape(-1),
        case["v"].reshape(-1),
        case["g"].reshape(-1),
        case["beta"].reshape(-1),
        case["a_log"],
        case["dt_bias"],
        case["cu_seqlens"],
        case["ssm_state_indices"].reshape(-1),
        case["nat"],
        case["tirx_state_raw"],
        case["tirx_out"].reshape(-1),
        int(case["q_total"]),
        int(spec["NUM_VALUE_HEADS"] * HEAD_DIM),  # g token stride (contiguous here)
        int(spec["STATE_SLOT_STRIDE"]),
        float(case["scale"]),
        float(lower_bound if lower_bound is not None else 0.0),
    )


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 oracle: the same recurrence, written as dense torch.

    Two semantics are easy to get wrong and are what this oracle exists to pin:

    * the recurrent state is carried in **FP32** across the tokens of a row and
      is only rounded to BF16 on the way to its checkpoint slot -- token ``t+1``
      consumes the FP32 value, not the round-tripped one (recurrent_kda.py:708
      writes ``sb1`` while ``s`` keeps its precision);
    * ``rk`` appears twice in ``deltak = rk * bb * (ve - rk * pred)`` (:681);
      one factor normalizes the key inside the prediction, the other normalizes
      the key of the rank-1 update.

    Returns ``(out, state_pool)`` with the pool laid out as the kernel sees it.
    """
    spec = case["spec"]
    dev = case["device"]
    tokens, heads = spec["NUM_TOKENS"], spec["NUM_VALUE_HEADS"]
    n_seq, q_total = spec["NUM_SEQS"], spec["Q_TOTAL"]
    ratio = spec["RATIO"]
    mode = spec["GATE_MODE"]
    lower_bound = case["lower_bound"]
    scale = case["scale"]

    f32 = torch.float32
    q = case["q"].reshape(q_total, spec["NUM_HEADS"], HEAD_DIM).to(f32)
    k = case["k"].reshape(q_total, spec["NUM_HEADS"], HEAD_DIM).to(f32)
    v = case["v"].reshape(q_total, heads, HEAD_DIM).to(f32)
    g = case["g"].reshape(q_total, heads, HEAD_DIM).to(f32)
    beta = case["beta"].reshape(q_total, heads).to(f32)
    dt_bias = case["dt_bias"].reshape(spec["NUM_HEADS"], HEAD_DIM).to(f32)
    av = torch.exp(case["a_log"].to(f32))  # [H]
    cu = case["cu_seqlens"].tolist()
    idx = case["ssm_state_indices"].reshape(n_seq, tokens)

    pool = _state_view(case["initial_state_raw"], case).to(f32).clone()
    out = torch.zeros((q_total, heads, HEAD_DIM), device=dev, dtype=f32)

    head_of = torch.arange(heads, device=dev) // ratio  # hv -> h

    for n in range(n_seq):
        token_base = cu[n]
        seq_len = cu[n + 1] - token_base
        # num_accepted_tokens is never supplied by SGLang, so ic collapses to 0.
        slot0 = int(idx[n, 0])
        state = pool[max(slot0, 0)].clone()  # [HV, V, K] f32

        for t in range(tokens):
            slot = int(idx[n, t])
            in_row = t < seq_len
            if not in_row:
                continue  # written by nobody
            if slot < 0:
                out[token_base + t] = 0.0  # pad row
                continue

            p = token_base + t
            qt, kt = q[p][head_of], k[p][head_of]  # [HV, K]
            vt, gt, bt = v[p], g[p], beta[p]

            x = gt + dt_bias[head_of]
            if mode == GATE_MODE_SOFTPLUS:
                sp = torch.log1p(torch.exp(x))
                sp = torch.where(x > SOFTPLUS_LINEAR_THRESHOLD, x, sp)
                gate = -av[head_of].unsqueeze(-1) * sp
            else:
                gate = lower_bound * torch.sigmoid(av[head_of].unsqueeze(-1) * x)
            eg = torch.exp(gate)  # [HV, K]

            rk = torch.rsqrt((kt * kt).sum(-1) + L2_EPS)  # [HV]
            rq = torch.rsqrt((qt * qt).sum(-1) + L2_EPS) * scale

            state = state * eg.unsqueeze(1)  # decay along K
            pred = torch.einsum("hvk,hk->hv", state, kt)
            deltak = rk.unsqueeze(-1) * bt.unsqueeze(-1) * (vt - rk.unsqueeze(-1) * pred)
            state = state + deltak.unsqueeze(-1) * kt.unsqueeze(1)
            out[p] = rq.unsqueeze(-1) * torch.einsum("hvk,hk->hv", state, qt)

            # The checkpoint rounds to BF16; the carried state does not.
            pool[slot] = state.to(torch.bfloat16).to(f32)

    if q_total > cu[n_seq]:
        out[cu[n_seq] :] = 0.0  # orphan suffix
    return out.reshape(1, q_total, heads, HEAD_DIM), pool


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the FlashInfer CuTe DSL source on the reference state pool.

    ``run_recurrent_kda`` derives ``NUM_TOKENS = 1 + num_spec_tokens``
    (recurrent_kda.py:1779), so the verify family must pass it; the decode
    family must not (:1298-1299).  ``num_accepted_tokens`` is deliberately left
    unset, matching SGLang's ``target_verify`` -- FlashInfer then substitutes a
    cached ones vector, which makes ``ic = 0`` and ``slot0 = ssm_idx[n, 0]``.
    """
    import importlib

    # flashinfer.kda_kernels.__init__ rebinds ``recurrent_kda`` to the run
    # function, so the submodule must be imported explicitly.
    fi = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
    spec = case["spec"]
    tokens = spec["NUM_TOKENS"]
    out, _ = fi.run_recurrent_kda(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        scale=case["scale"],
        initial_state=_state_view(case["reference_state_raw"], case),
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=case["lower_bound"],
        cu_seqlens=case["cu_seqlens"],
        ssm_state_indices=case["ssm_state_indices"],
        num_spec_tokens=(tokens - 1) if tokens > 1 else None,
    )
    return out


# Two bfloat16 ULP against the source, matching the one-warp sibling.
_RTOL = 2.0**-7
_ATOL = 1.0e-4
# The FP32 oracle reassociates freely, so it only needs to agree to bf16 noise.
_ORACLE_RTOL = 2.0**-6
_ORACLE_ATOL = 3.0e-4


def _assert_close(got, want, rtol, atol, what: str) -> None:
    got_f, want_f = got.float(), want.float()
    diff = (got_f - want_f).abs()
    tol = atol + rtol * want_f.abs()
    bad = diff > tol
    if bool(bad.any()):
        idx = int(bad.float().argmax())
        raise AssertionError(
            f"{what}: {int(bad.sum())}/{bad.numel()} elements exceed tolerance; "
            f"max |diff| = {float(diff.max()):.3e} (tol {float(tol.flatten()[idx]):.3e}), "
            f"first at flat index {idx}: got {float(got_f.flatten()[idx]):.6e} "
            f"want {float(want_f.flatten()[idx]):.6e}"
        )


def run_test(**kwargs: Any) -> None:
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    spec = case["spec"]
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize()

    tirx_out = case["tirx_out"]
    tirx_state = _state_view(case["tirx_state_raw"], case)

    # Primary oracle: the CuTe DSL source itself, on its own pool clone.
    ref_out = _flashinfer_reference(case)
    ref_state = _state_view(case["reference_state_raw"], case)
    _assert_close(tirx_out, ref_out, _RTOL, _ATOL, "output vs flashinfer")
    _assert_close(tirx_state, ref_state, _RTOL, _ATOL, "state vs flashinfer")

    # Secondary oracle: dense FP32 torch, from the untouched initial pool.
    oracle_out, oracle_state = _torch_reference(case)
    _assert_close(tirx_out, oracle_out, _ORACLE_RTOL, _ORACLE_ATOL, "output vs fp32 oracle")
    _assert_close(
        tirx_state.float(), oracle_state, _ORACLE_RTOL, _ORACLE_ATOL, "state vs fp32 oracle"
    )

    # ---- branch-specific obligations -------------------------------------
    tokens, n_seq = spec["NUM_TOKENS"], spec["NUM_SEQS"]
    stride = spec["STATE_SLOT_STRIDE"]
    cu = case["cu_seqlens"].tolist()
    idx = case["ssm_state_indices"].reshape(n_seq, tokens)
    initial = case["initial_state_raw"]
    final = case["tirx_state_raw"]

    # Pad rows: zeroed output, and not one state slot written.
    for n in range(n_seq):
        for t in range(tokens):
            if t >= cu[n + 1] - cu[n]:
                continue
            if int(idx[n, t]) >= 0:
                continue
            row = tirx_out[0, cu[n] + t]
            assert bool((row == 0).all()), f"pad row {n} token {t} produced nonzero output"
    written = {int(idx[n, t]) for n in range(n_seq) for t in range(min(tokens, cu[n + 1] - cu[n]))}
    for slot in range(case["state_slots"]):
        if slot in written:
            continue
        lo, hi = slot * stride, (slot + 1) * stride
        assert torch.equal(initial[lo:hi], final[lo:hi]), (
            f"state slot {slot} is owned by no active token but was modified"
        )

    # Orphan packed suffix: carrier tokens beyond the last row must be zeroed.
    covered = cu[n_seq]
    if case["q_total"] > covered:
        tail = tirx_out[0, covered:]
        assert bool((tail == 0).all()), (
            f"orphan suffix [{covered}, {case['q_total']}) not zeroed by the kernel"
        )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    spec = case["spec"]
    executable = compile_kernel(get_kernel(**kwargs))
    args = _tirx_args(case)

    # Validate once, outside the timed region.  Both implementations update
    # their own state-pool clone in place, so repeated timed launches let the
    # values drift; the work per launch is identical either way.
    executable(*args)
    shape = (1, spec["Q_TOTAL"], spec["NUM_VALUE_HEADS"], HEAD_DIM)
    flashinfer_out = _flashinfer_reference(case).reshape(shape)
    torch.cuda.synchronize()
    torch.testing.assert_close(case["tirx_out"], flashinfer_out, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(
        case["tirx_state_raw"], case["reference_state_raw"], rtol=_RTOL, atol=_ATOL
    )

    def flashinfer_builder():
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


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
