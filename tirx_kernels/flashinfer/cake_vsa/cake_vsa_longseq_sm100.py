# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ c5365737), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Cake VSA ``longseq`` block-sparse attention forward for SM100a.

Upstream sources (FlashInfer @ c5365737570a2a156d7cae0c4070fa3770ecc670):

- ``csrc/cake_vsa/cake_vsa_longseq_sm_100a.cu`` -- the device kernel
  ``kernel_flashinfer_blackwell_vsa_longseq_warp_specialized_sm100``.
- ``csrc/cake_vsa/cake_vsa_longseq_host.cpp`` -- the tvm-ffi launcher that
  encodes three rank-4 SWIZZLE_128B tensor maps and launches grid
  ``(mb, num_qo_heads, 1)``, 512 threads, and 201344 bytes of dynamic SMEM.
- ``flashinfer/cake_vsa.py`` -- the planner and public dispatch. This profile
  covers BF16, D=128, R=C=128, eight-head MHA, ``N >= 16384``, and a uniform
  selected-block count no larger than 192.

One CTA owns a 128-row query block and one head. Warps 0-7 run two independent
64-row online-softmax instances, warps 8-11 correct and store their output, warp
12 issues all QK/PV tensor-core work, warp 13 loads Q/K and compacts the mask,
warp 14 independently loads V, and warp 15 is idle. K uses a three-stage ring, V
uses a two-stage ring, and scores/probabilities/output use four TMEM regions.
Every shared object lives in one rank-one byte arena addressed by explicit scalar
offsets; no first-class layouts or tile primitives are used.
"""

import math
from functools import lru_cache
from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cake_vsa_longseq_sm100",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "c5365737570a2a156d7cae0c4070fa3770ecc670",
            },
            "import": "flashinfer",
        },
    ),
}
# The upstream kernel is a prebuilt cubin loaded through FlashInfer's own tvm-ffi
# launcher (``flashinfer.cake_vsa._load_module``); it does not use the CuTe DSL, and
# tvm-ffi is a dependency of flashinfer-python itself, so only FlashInfer is pinned.

# ---------------------------------------------------------------------------
# Source contract constants (host.cpp / cake_vsa.py / the device kernel).
# ---------------------------------------------------------------------------
HEAD_DIM = 128
BLOCK = 128  # R == C == 128 query/KV block size of this profile
MAX_SELECTED_BLOCKS = 192  # ``union_blocks`` capacity per instance
THREADS = 512
NUM_WARPS = THREADS // 32
SMEM_TOTAL = 201344
TMEM_COLS = 512
PROFILE = "longseq"
CUDA_ARCH = "sm_100a"
_LN2 = math.log(2.0)

# ``lse`` sentinel used when the caller does not request LSE: upstream passes a
# one-element ``stats`` tensor that the kernel must leave untouched.
_LSE_SENTINEL = 12345.25

# Oracle tolerances (fp32 torch reference), frozen from the errors measured on
# GB200 across CONFIGS with a 1.25-1.6x margin (the TIRx-vs-source comparison is
# bitwise, so both implementations carry the same algorithmic error: bf16 P and
# output rounding, ``ex2.approx``/``rcp.approx``/``lg2.approx``).  The upstream test
# uses atol=rtol=1e-2.  ``max_rel_excess`` in the returned stats is the worst
# error / (atol + rtol*|ref|) ratio.  The bf16 ULP checks apply to |ref| >= 0.125,
# where one ULP is at least 2^-10.
_ORACLE_OUT_ATOL = 1.5e-3
_ORACLE_OUT_RTOL = 6e-3
_ORACLE_LSE_ATOL = 2e-6
_ORACLE_LSE_RTOL = 5e-7
_ORACLE_ULP_MAX = 2
_ORACLE_ULP_LE1_FRAC = 0.9999
_ORACLE_ULP_LE2_FRAC = 1.0
_ORACLE_ULP_MIN_REF = 0.125


def _config(
    label: str,
    *,
    M: int,
    N: int,
    num_heads: int,
    pattern: str,
    selected: int | None = None,
    sel_lo: int | None = None,
    sel_hi: int | None = None,
    return_lse: bool = False,
    q_amp: float = 1.0,
    sm_scale: float | None = None,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    seed: int = 0,
    oracle_atol: float = _ORACLE_OUT_ATOL,
    oracle_rtol: float = _ORACLE_OUT_RTOL,
    ulp_max: int = _ORACLE_ULP_MAX,
    ulp_le1_frac: float = _ORACLE_ULP_LE1_FRAC,
    ulp_le2_frac: float = _ORACLE_ULP_LE2_FRAC,
) -> dict[str, Any]:
    return {
        "label": label,
        "M": M,
        "N": N,
        "num_heads": num_heads,
        "pattern": pattern,
        "selected": selected,
        "sel_lo": sel_lo,
        "sel_hi": sel_hi,
        "return_lse": return_lse,
        "q_amp": q_amp,
        "sm_scale": sm_scale,
        "return_temperature_lse": return_temperature_lse,
        "lse_temperature_scale": lse_temperature_scale,
        "seed": seed,
        "oracle_atol": oracle_atol,
        "oracle_rtol": oracle_rtol,
        "ulp_max": ulp_max,
        "ulp_le1_frac": ulp_le1_frac,
        "ulp_le2_frac": ulp_le2_frac,
    }


# Correctness matrix. Every row is inside the exact longseq dispatch domain.
CONFIGS = [
    _config("m128_n16384_h8_sel8", M=128, N=16384, num_heads=8, pattern="upstream", selected=8),
    _config(
        "m256_n16512_h8_sel1_tail",
        M=256,
        N=16512,
        num_heads=8,
        pattern="random",
        selected=1,
        seed=1,
    ),
    _config(
        "m512_n16384_h8_sel65", M=512, N=16384, num_heads=8, pattern="random", selected=65, seed=2
    ),
    _config(
        "m512_n16384_h8_sel128", M=512, N=16384, num_heads=8, pattern="random", selected=128, seed=3
    ),
    _config(
        "m512_n24576_h8_sel192_lse",
        M=512,
        N=24576,
        num_heads=8,
        pattern="random",
        selected=192,
        return_lse=True,
        seed=4,
    ),
    _config(
        "m512_n32768_h8_sel32_tlse",
        M=512,
        N=32768,
        num_heads=8,
        pattern="random",
        selected=32,
        return_temperature_lse=True,
        lse_temperature_scale=0.7,
        seed=5,
    ),
    _config(
        "m1024_n32768_h8_sel16_qamp",
        M=1024,
        N=32768,
        num_heads=8,
        pattern="random",
        selected=16,
        q_amp=6.0,
        sm_scale=0.05,
        seed=6,
        oracle_atol=6.5e-3,
        oracle_rtol=1e-2,
        ulp_max=8,
        ulp_le1_frac=0.995,
        ulp_le2_frac=0.9998,
    ),
]

# Benchmark matrix: six long-sequence sparsity and streaming regimes.
BENCH_CONFIGS = [
    _config(
        "m4096_n16384_h8_sel1", M=4096, N=16384, num_heads=8, pattern="random", selected=1, seed=100
    ),
    _config(
        "m4096_n16384_h8_sel8", M=4096, N=16384, num_heads=8, pattern="random", selected=8, seed=101
    ),
    _config(
        "m16384_n32768_h8_sel32",
        M=16384,
        N=32768,
        num_heads=8,
        pattern="random",
        selected=32,
        seed=102,
    ),
    _config(
        "m32768_n65536_h8_sel64",
        M=32768,
        N=65536,
        num_heads=8,
        pattern="random",
        selected=64,
        seed=103,
    ),
    _config(
        "m8192_n24576_h8_sel128",
        M=8192,
        N=24576,
        num_heads=8,
        pattern="random",
        selected=128,
        seed=104,
    ),
    _config(
        "m8192_n32768_h8_sel192",
        M=8192,
        N=32768,
        num_heads=8,
        pattern="random",
        selected=192,
        seed=105,
    ),
]


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _validate_shape(M: int, N: int, num_heads: int) -> tuple[int, int]:
    if M <= 0 or M % BLOCK != 0:
        raise ValueError(f"M={M} must be a positive multiple of {BLOCK}")
    if N <= 0 or N % BLOCK != 0:
        raise ValueError(f"N={N} must be a positive multiple of {BLOCK}")
    if num_heads <= 0:
        raise ValueError(f"num_heads={num_heads} must be positive")
    return M // BLOCK, N // BLOCK


def _assert_longseq_route(mask, *, N: int, num_heads: int, mb: int) -> int:
    """Re-evaluate the upstream ``run_cake_vsa`` dispatch chain for this mask.

    BF16, D=128, R=C=128 and MHA are fixed by this module.  Validate the
    remaining public-dispatch predicates and return its runtime top-k argument.
    """
    counts = mask.sum(dim=-1)
    max_selected = int(counts.max())
    uniform = bool((counts == counts.flatten()[0]).all())
    if N < 16384 or num_heads != 8:
        raise ValueError("longseq requires N >= 16384 and exactly eight heads")
    if mb >= 625 and max_selected == 6 and uniform:
        raise ValueError("mask would dispatch to the ultrasparse_bsr profile")
    if not uniform or not 1 <= max_selected <= MAX_SELECTED_BLOCKS:
        raise ValueError("longseq requires a fixed selected-block count in [1, 192]")
    return max_selected


def _make_block_mask(
    *,
    num_heads: int,
    mb: int,
    nb: int,
    pattern: str,
    selected: int | None,
    sel_lo: int | None,
    sel_hi: int | None,
    generator,
):
    """Build the ``(num_heads, mb, nb)`` boolean block mask for one config."""
    import torch

    mask = torch.zeros((num_heads, mb, nb), dtype=torch.bool)
    if pattern == "dense":
        mask[:] = True
        return mask
    if pattern == "upstream":
        if selected is None:
            raise ValueError("pattern 'upstream' needs 'selected'")
        for row in range(mb):
            columns = (torch.arange(selected) * 7 + row) % nb
            mask[:, row, columns] = True
        return mask
    if pattern in {"random", "emptyrow"}:
        if selected is None:
            raise ValueError(f"pattern {pattern!r} needs 'selected'")
        lo = hi = selected
    elif pattern == "random_var":
        if sel_lo is None or sel_hi is None:
            raise ValueError("pattern 'random_var' needs 'sel_lo' and 'sel_hi'")
        lo, hi = sel_lo, sel_hi
    else:
        raise ValueError(f"unknown mask pattern {pattern!r}")
    if lo < 1 or hi > min(nb, MAX_SELECTED_BLOCKS) or lo > hi:
        raise ValueError(f"selected range [{lo}, {hi}] invalid for nb={nb}")
    for head in range(num_heads):
        for row in range(mb):
            count = lo if lo == hi else int(torch.randint(lo, hi + 1, (1,), generator=generator))
            columns = torch.randperm(nb, generator=generator)[:count]
            mask[head, row, columns] = True
    if pattern == "emptyrow":
        mask[min(1, num_heads - 1), min(2, mb - 1), :] = False
    return mask


def prepare_data(**config: Any) -> dict[str, Any]:
    """Allocate logical inputs plus independent TIRx and source output buffers."""
    import torch

    config = _without_label(config)
    M, N, num_heads = config["M"], config["N"], config["num_heads"]
    mb, nb = _validate_shape(M, N, num_heads)
    return_lse = bool(config["return_lse"])
    return_tlse = bool(config["return_temperature_lse"])
    sm_scale = config["sm_scale"]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    device = torch.device("cuda")

    q = torch.randn((M, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32)
    q = (q * float(config["q_amp"])).to(torch.bfloat16).to(device)
    k = torch.randn((N, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32)
    k = k.to(torch.bfloat16).to(device)
    v = torch.randn((N, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32)
    v = v.to(torch.bfloat16).to(device)
    mask = _make_block_mask(
        num_heads=num_heads,
        mb=mb,
        nb=nb,
        pattern=config["pattern"],
        selected=config["selected"],
        sel_lo=config["sel_lo"],
        sel_hi=config["sel_hi"],
        generator=generator,
    )
    selected_blocks = _assert_longseq_route(mask, N=N, num_heads=num_heads, mb=mb)
    block_mask = mask.to(device)

    def outputs() -> dict[str, Any]:
        out = torch.full(
            (M, num_heads, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=device
        )
        if return_lse:
            lse = torch.full((M, num_heads), float("nan"), dtype=torch.float32, device=device)
        else:
            lse = torch.full((1,), _LSE_SENTINEL, dtype=torch.float32, device=device)
        if return_tlse:
            tlse = torch.full((M, num_heads), float("nan"), dtype=torch.float32, device=device)
        else:
            tlse = lse  # upstream passes the same ``stats`` tensor for both slots
        return {"out": out, "lse": lse, "tlse": tlse}

    return {
        "config": config,
        "M": M,
        "N": N,
        "num_heads": num_heads,
        "mb": mb,
        "nb": nb,
        "selected_blocks": selected_blocks,
        "q": q,
        "k": k,
        "v": v,
        "block_mask": block_mask,
        "block_mask_u8": block_mask.view(torch.uint8),
        "sm_scale": float(sm_scale),
        "softmax_scale_log2": float(sm_scale) / _LN2,
        "lse_temperature_scale": float(config["lse_temperature_scale"]),
        "return_lse": return_lse,
        "return_temperature_lse": return_tlse,
        "tirx": outputs(),
        "source": outputs(),
    }


# ---------------------------------------------------------------------------
# TIRx kernel
# ---------------------------------------------------------------------------
# cuTensorMapEncodeTiled enum values (CUtensorMapInterleave / Swizzle /
# L2promotion / FloatOOBfill) as used by the upstream host launcher.
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_128B = 3
_TMA_L2_PROMOTION_NONE = 0
_TMA_OOB_FILL_NONE = 0


def _host_prelude(params):
    """Encode the three upstream tensor maps from the runtime shape scalars.

    Mirrors ``EncodeTma_q`` / ``EncodeTma_k`` / ``EncodeTma_v`` in the upstream
    host launcher: rank-4 bf16 maps, 128-byte swizzle, no L2 promotion, no OOB
    fill.  Q is ``{64, M, Hq, 2}`` with a ``{64, 64, 1, 2}`` box (one 16 KB
    transaction covers both head-dim halves of 64 rows); K and V are
    ``{64, N, 2, Hkv}`` with a ``{64, 64, 1, 1}`` box (8 KB per transaction).
    """
    mb = params["mb"]
    nb = params["nb"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    elem_bytes = 2

    def encode(tensor, dims, strides_bytes, box):
        descriptor = K.stack_alloca("tensormap", 1)
        K.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            "bfloat16",
            4,
            tensor.data,
            *dims,
            *strides_bytes,
            *box,
            1,
            1,
            1,
            1,
            _TMA_INTERLEAVE_NONE,
            _TMA_SWIZZLE_128B,
            _TMA_L2_PROMOTION_NONE,
            _TMA_OOB_FILL_NONE,
        )
        return descriptor

    q_map = encode(
        params["q"],
        (64, mb * BLOCK, num_q_heads, HEAD_DIM // 64),
        (num_q_heads * HEAD_DIM * elem_bytes, HEAD_DIM * elem_bytes, 64 * elem_bytes),
        (64, 64, 1, HEAD_DIM // 64),
    )
    kv_dims = (64, nb * BLOCK, HEAD_DIM // 64, num_kv_heads)
    kv_strides = (num_kv_heads * HEAD_DIM * elem_bytes, 64 * elem_bytes, HEAD_DIM * elem_bytes)
    kv_box = (64, 64, 1, 1)
    k_map = encode(params["k"], kv_dims, kv_strides, kv_box)
    v_map = encode(params["v"], kv_dims, kv_strides, kv_box)
    return (q_map, k_map, v_map)


# Byte offsets inside the 201344-byte dynamic SMEM arena (source macro table).
_MBAR_Q_FULL = 0
_MBAR_UNION_READY = 8
_MBAR_K_FULL = 16  # 3 stages
_MBAR_K_EMPTY = 40  # 3 stages
_MBAR_V_FULL = 64  # 2 stages
_MBAR_V_EMPTY = 80  # 2 stages
_MBAR_S_FULL = 96  # 2 instances
_MBAR_P_FULL = 112  # 2 instances
_MBAR_CORR_SIG = 128  # 2 instances
_MBAR_CORR_DONE = 144  # 2 instances
_MBAR_O_FULL = 160  # 2 instances
_MBAR_TMEM_DEALLOC = 176
_SMEM_TMEM_MAILBOX = 184
_SMEM_Q0 = 1024
_SMEM_Q1 = 17408
_SMEM_K = 33792  # 3 x 32768
_SMEM_V = 132096  # 2 x 32768
_SMEM_STAGE_BYTES = 32768
_SMEM_SCALE = 197632  # 512 f32: acc_scale / row_sum / row_max / temperature_sum
_SMEM_UNION_COUNT = 199680  # 2 i32
_SMEM_UNION_BLOCKS = 199688  # 2 x 192 i32
# (offset, arrival count) in source order: q_full, union_ready, k_full x3, k_empty x3,
# v_full x2, v_empty x2, s_full x2, p_full x2, corr_sig x2, corr_done x2,
# o_full x2, tmem_dealloc.
_MBARRIER_INIT = (
    (_MBAR_Q_FULL, 1),
    (_MBAR_UNION_READY, 32),
    (_MBAR_K_FULL, 1),
    (_MBAR_K_FULL + 8, 1),
    (_MBAR_K_FULL + 16, 1),
    (_MBAR_K_EMPTY, 1),
    (_MBAR_K_EMPTY + 8, 1),
    (_MBAR_K_EMPTY + 16, 1),
    (_MBAR_V_FULL, 1),
    (_MBAR_V_FULL + 8, 1),
    (_MBAR_V_EMPTY, 1),
    (_MBAR_V_EMPTY + 8, 1),
    (_MBAR_S_FULL, 1),
    (_MBAR_S_FULL + 8, 1),
    (_MBAR_P_FULL, 256),
    (_MBAR_P_FULL + 8, 256),
    (_MBAR_CORR_SIG, 128),
    (_MBAR_CORR_SIG + 8, 128),
    (_MBAR_CORR_DONE, 128),
    (_MBAR_CORR_DONE + 8, 128),
    (_MBAR_O_FULL, 1),
    (_MBAR_O_FULL + 8, 1),
    (_MBAR_TMEM_DEALLOC, 128),
)
# TMEM column bases per softmax instance: S (128 columns; P overwrites the upper 64
# in place) and the O accumulator (128 columns).
_TMEM_SCORES = (0, 256)
_TMEM_OUTPUT = (128, 384)
_TMEM_P_OFFSET = 64
# UMMA descriptor words and instruction descriptors (source immediates).
_DESC_HI = 0x40004040  # SBO 1024 B, base-offset bit, 128 B swizzle
_QK_IDESC = 0x04200490  # bf16 x bf16 -> f32, M=64, N=128, K-major A and B
_PV_IDESC = 0x04210490  # same with MN-major B (V is token-major)
_QK_A_STEPS = (2, 2, 2, 506, 2, 2, 2)  # Q low-word walk over K=128 (bytes/16)
_QK_B_STEPS = (2, 2, 2, 1018, 2, 2, 2)  # K low-word walk (second dim half at +16384 B)
_V_LBO_BIT = 0x4000000  # LBO = 16384 B: dim-half stride of the V tile
_PV_B_STEP = 128  # 2048 B = 16 tokens x 128 B per K16 step
_PV_A_STEP = 8  # 16 bf16 P columns = 8 TMEM 32-bit columns per K16 step
_REG_SOFTMAX = 192
_REG_CORRECTION = 80
_REG_OTHER = 48
_WAIT_TICKS = 0x989680

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
_TMEM_ST_X16 = "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
_TMEM_ST_X64 = "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
_LN2_F32 = 0.6931471805599453
_NEG_INF = float("-inf")
_FULL_MASK = 0xFFFFFFFF


def _u32(value):
    return K.uint32(value)


def _i32(value):
    return K.int32(value)


def _f32(value):
    return K.float32(value)


def _mbar_wait(addr, phase):
    """``mbarrier.try_wait.parity.acquire.cta`` retry loop with the source's suspend hint."""
    K.cuda.mbarrier_wait(addr, phase)


def _mbar_arrive(addr):
    K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(addr)


def _mbar_expect_tx(addr, tx_bytes):
    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(addr, _u32(tx_bytes))


def _ld_shared_i32(addr):
    value = K.local_scalar("int32")
    K.ptx.ld.shared.b32(value, addr)
    return value


def _ld_shared_f32(addr):
    value = K.local_scalar("float32")
    K.ptx.ld.shared.b32(value, addr)
    return value


def _ld_shared_counts(addr):
    """Both instance counts in one ``ld.shared.v2.b32`` (the source's vectorized pair)."""
    count0 = K.local_scalar("int32")
    count1 = K.local_scalar("int32")
    K.ptx.ld.shared.v2.b32(count0, count1, addr)
    return count0, count1


def _flip(phase):
    K.assign(phase, phase ^ _i32(1))


def _advance_ring(stage, phase):
    """Three-stage ring cursor: stage += 1, wrap to 0 and flip parity at 3."""
    K.assign(stage, stage + _u32(1))
    with K.If(stage == _u32(3)), K.Then():
        K.assign(stage, _u32(0))
        _flip(phase)


def _advance_v_ring(stage, phase):
    """Two-stage V ring cursor: stage += 1, wrap to 0 and flip parity at 2."""
    K.assign(stage, stage + _u32(1))
    with K.If(stage == _u32(2)), K.Then():
        K.assign(stage, _u32(0))
        _flip(phase)


def _pack2(lo, hi):
    packed = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, lo, hi)
    return packed


def _packed_fma_inplace(values, base, scale_pair, bias_pair):
    """``fma.rn.ftz.f32x2`` on values[base:base+2] with packed scale and bias."""
    result = K.local_scalar("uint64")
    K.ptx.fma.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair, bias_pair)
    K.ptx.mov.b64(values[base], values[base + 1], result)


def _packed_mul_inplace(values, base, scale_pair):
    """``mul.rn.ftz.f32x2`` on values[base:base+2] with a packed scale."""
    result = K.local_scalar("uint64")
    K.ptx.mul.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair)
    K.ptx.mov.b64(values[base], values[base + 1], result)


def _exp2_inplace(values, index):
    K.ptx.ex2.approx.ftz.f32(values[index], values[index])


def _max_f32(a, b):
    out = K.local_scalar("float32")
    K.ptx.max.f32(out, a, b)
    return out


def _shfl_xor_f32(value, lane_xor):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), _u32(lane_xor), _u32(31), _u32(_FULL_MASK)
    )
    return K.reinterpret("float32", out)


def _tmem_load_x32(dst, base, taddr):
    K.ptx[_TMEM_LD_X32](*(dst[base + i] for i in range(32)), taddr, 64)


def _block_sum(values):
    """``softmax_block_sum`` over 64 values: 32 packed ``add.f32x2`` into one accumulator."""
    acc = K.local_scalar("uint64")
    K.ptx.mov.b64(acc, _f32(0.0), _f32(0.0))
    for j in range(32):
        K.ptx.add.f32x2(acc, acc, _pack2(values[2 * j], values[2 * j + 1]))
    acc_x = K.local_scalar("float32")
    acc_y = K.local_scalar("float32")
    K.ptx.mov.b64(acc_x, acc_y, acc)
    half = K.local_scalar("float32")
    K.ptx.add.ftz.f32(half, acc_x, acc_y)
    total = K.local_scalar("float32")
    K.ptx.add.ftz.f32(total, half, _shfl_xor_f32(half, 16))
    return total


def _row_max(values):
    """``row_max_x32_accum`` x2 + ``row_max_reduce``: alternating accumulators, 65 ``max.f32``."""
    acc0 = K.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = K.local_scalar("float32", init=_f32(_NEG_INF))
    for half in range(2):
        for j in range(16):
            pair = _max_f32(values[half * 32 + 2 * j], values[half * 32 + 2 * j + 1])
            if j % 2 == 0:
                K.assign(acc0, _max_f32(acc0, pair))
            else:
                K.assign(acc1, _max_f32(acc1, pair))
    return _max_f32(acc0, acc1)


def _build_kernel():
    @K.kernel(
        warps=NUM_WARPS,
        arch=CUDA_ARCH,
        min_blocks_per_sm=1,
        grid=lambda p: [p["mb"], p["num_q_heads"]],
        host_prelude=_host_prelude,
    )
    def cake_vsa_longseq_sm100(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        temperature_lse: K.gptr[K.f32],
        block_mask: K.gptr[K.u8],
        mb: K.i32,
        nb: K.i32,
        selected_blocks: K.i32,
        num_q_heads: K.i32,
        num_kv_heads: K.i32,
        softmax_scale_log2: K.f32,
        lse_temperature_scale: K.f32,
        return_softmax_lse: K.i32,
        return_temperature_lse: K.i32,
        *,
        host,
    ):
        q_map, k_map, v_map = host
        del q, k, v, num_kv_heads  # reached only through the tensor maps / host prelude
        # >>> kernel_flashinfer_blackwell_vsa_longseq_warp_specialized_sm100 body starts here
        q_tile, q_head = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()

        arena = K.alloc_buffer((SMEM_TOTAL,), K.u8, scope="shared.dyn", align=1024)
        smem = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        def bar(offset):
            return smem + _u32(offset)

        # Mbarrier init (12 groups, 23 barriers) by one elected lane of warp 0.
        with K.If(warp == 0), K.Then():
            init_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(init_leader != _u32(0)), K.Then():
                for offset, count in _MBARRIER_INIT:
                    K.ptx.mbarrier.init.shared__cta.b64(bar(offset), _u32(count))
                K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.warp.sync(_u32(_FULL_MASK))

        # TMEM alloc (512 columns) by warp 0; the address lands in the shared mailbox.
        with K.If(warp == 0), K.Then():
            K.ptx[_TMEM_ALLOC](bar(_SMEM_TMEM_MAILBOX), _u32(TMEM_COLS))
            K.ptx[_TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        K.ptx["tcgen05.fence::after_thread_sync"]()
        taddr = K.local_scalar("uint32")
        K.ptx.ld.volatile.shared.b32(taddr, bar(_SMEM_TMEM_MAILBOX))

        q_local_base = q_tile * 128
        kv_len = nb * 128

        # Sibling role guards, as in the source; folding them into an if/else-if chain
        # measured 1.2-2.3% slower on the two small shapes with no register pressure to relieve.
        roles = K.specialize(chain_dispatch=False)
        other_regs = roles.register_scope("other", warps=range(12, 16), regs=_REG_OTHER)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=_REG_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=_REG_CORRECTION)
        r_mma = roles.role("mma", warps=[12], register_scope=other_regs)
        r_qk_load = roles.role("qk_load", warps=[13], register_scope=other_regs)
        r_v_load = roles.role("v_load", warps=[14], register_scope=other_regs)
        r_idle = roles.role("idle", warps=[15], register_scope=other_regs)

        # Register redistribution: warps 12..15 decrease first so the softmax
        # increase can be granted.
        with K.If(K.And(warp >= 12, warp <= 15)), K.Then():
            other_regs.emit()

        # ---- Role: softmax (warps 0..7, two four-warp instances) ----
        with r_softmax:
            q_valid = K.local_scalar("int32", init=_i32(128))
            with K.If(q_tile >= mb), K.Then():
                K.assign(q_valid, _i32(0))
            union_ready_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase)
            _flip(union_ready_phase)
            instance = K.uniform(warp // 4)
            instance_row_offset = K.uniform(instance * 64)
            instance_token_offset = K.uniform(instance * 64)
            instance_tmem_offset = K.uniform(instance * 256)
            union_count = selected_blocks
            warp_in_instance = warp % 4
            tmem_row_origin = warp_in_instance * 32
            my_row = warp_in_instance * 16 + lane % 16
            col_half = lane // 16
            instance_valid = K.local_scalar("int32", init=q_valid - instance_token_offset)
            with K.If(instance_valid > _i32(64)), K.Then():
                K.assign(instance_valid, _i32(64))
            with K.If(instance_valid < _i32(0)), K.Then():
                K.assign(instance_valid, _i32(0))
            row_valid = K.local_scalar("int32", init=K.cast(my_row < instance_valid, "int32"))
            row_max = K.local_scalar("float32", init=_f32(_NEG_INF))
            row_sum = K.local_scalar("float32", init=_f32(0.0))
            temperature_sum = K.local_scalar("float32", init=_f32(0.0))
            phase_s_full_0 = K.local_scalar("int32", init=_i32(0))
            phase_s_full_1 = K.local_scalar("int32", init=_i32(0))
            phase_corr_done_0 = K.local_scalar("int32", init=_i32(0))
            phase_corr_done_1 = K.local_scalar("int32", init=_i32(0))
            score_addr = K.local_scalar(
                "uint32",
                init=taddr
                + K.cast(instance_tmem_offset, "uint32")
                + K.shift_left(K.cast(tmem_row_origin, "uint32"), _u32(16)),
            )
            p_addr = K.local_scalar("uint32", init=score_addr + _u32(_TMEM_P_OFFSET))
            scale_row_addr = bar(_SMEM_SCALE) + K.cast(
                instance_row_offset + my_row, "uint32"
            ) * _u32(4)
            scores = K.alloc_local((64,), "float32")
            packed_p = K.alloc_local((32,), "uint32")
            block_sum = K.local_scalar("float32", init=_f32(0.0))
            block_temperature_sum = K.local_scalar("float32", init=_f32(0.0))
            acc_scale = K.local_scalar("float32", init=_f32(1.0))
            temperature_acc_scale = K.local_scalar("float32", init=_f32(1.0))
            selected_max = K.local_scalar("float32", init=_f32(_NEG_INF))
            new_max_scaled = K.local_scalar("float32", init=_f32(0.0))
            safe_max = K.local_scalar("float32", init=_f32(0.0))

            def probabilities_from(frag, bias):
                """fma, exp2, block sum, bf16 pack and the two P stores of one branch."""
                scale_pair = _pack2(softmax_scale_log2, softmax_scale_log2)
                bias_pair = _pack2(bias, bias)
                for j in range(32):
                    _packed_fma_inplace(frag, 2 * j, scale_pair, bias_pair)
                for j in range(64):
                    _exp2_inplace(frag, j)
                K.assign(block_sum, _block_sum(frag))
                for j in range(32):
                    K.ptx.cvt.rn.bf16x2.f32(packed_p[j], frag[2 * j + 1], frag[2 * j])
                K.ptx[_TMEM_ST_X16](p_addr, 32, *(packed_p[i] for i in range(16)))
                K.ptx[_TMEM_ST_X16](p_addr + _u32(16), 32, *(packed_p[16 + i] for i in range(16)))

            union_index = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index < union_count):
                n_block = _ld_shared_i32(
                    bar(_SMEM_UNION_BLOCKS)
                    + K.cast(instance * 192 + union_index, "uint32") * _u32(4)
                )
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_wait(bar(_MBAR_S_FULL), phase_s_full_0)
                        _flip(phase_s_full_0)
                    with K.Else():
                        _mbar_wait(bar(_MBAR_S_FULL + 8), phase_s_full_1)
                        _flip(phase_s_full_1)
                valid_cols = K.local_scalar("int32", init=_i32(0))
                with K.If(row_valid != _i32(0)), K.Then():
                    K.assign(valid_cols, kv_len - n_block * 128)
                    with K.If(valid_cols > _i32(128)), K.Then():
                        K.assign(valid_cols, _i32(128))
                    with K.If(valid_cols < _i32(0)), K.Then():
                        K.assign(valid_cols, _i32(0))
                _tmem_load_x32(scores, 0, score_addr)
                _tmem_load_x32(scores, 32, score_addr + _u32(32))
                half_valid = K.local_scalar("int32", init=valid_cols - col_half * 64)
                with K.If(half_valid < _i32(0)), K.Then():
                    K.assign(half_valid, _i32(0))
                with K.If(half_valid > _i32(64)), K.Then():
                    K.assign(half_valid, _i32(64))
                # The source's per-column -inf fill for 0 < half_valid < 64 is unreachable
                # (valid_cols is a multiple of 128) and nvcc emits no code for it.
                tile_max = K.local_scalar("float32", init=_row_max(scores))
                with K.If(half_valid <= _i32(0)), K.Then():
                    K.assign(tile_max, _f32(_NEG_INF))
                K.assign(tile_max, _max_f32(tile_max, _shfl_xor_f32(tile_max, 16)))
                new_max = _max_f32(tile_max, row_max)
                K.assign(safe_max, K.if_then_else(new_max == _f32(_NEG_INF), _f32(0.0), new_max))
                K.ptx.mul.ftz.f32(new_max_scaled, safe_max, softmax_scale_log2)
                neg_new_max_scaled = K.local_scalar("float32")
                K.ptx.neg.ftz.f32(neg_new_max_scaled, new_max_scaled)
                acc_scale_log2 = K.local_scalar("float32")
                K.ptx.fma.rn.ftz.f32(
                    acc_scale_log2, row_max, softmax_scale_log2, neg_new_max_scaled
                )
                with K.If(acc_scale_log2 >= _f32(-8.0)):
                    with K.Then():
                        # The running max moves by less than 2^8: keep the old max, skip the rescale.
                        K.assign(selected_max, row_max)
                        K.assign(
                            safe_max, K.if_then_else(row_max == _f32(_NEG_INF), _f32(0.0), row_max)
                        )
                        K.assign(acc_scale, _f32(1.0))
                        K.assign(temperature_acc_scale, _f32(1.0))
                        K.ptx.mul.ftz.f32(new_max_scaled, safe_max, softmax_scale_log2)
                    with K.Else():
                        K.assign(selected_max, new_max)
                        exp_scale = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(exp_scale, acc_scale_log2)
                        K.assign(
                            acc_scale,
                            K.if_then_else(row_max > _f32(_NEG_INF), exp_scale, _f32(1.0)),
                        )
                        tau_log2 = K.local_scalar("float32")
                        K.ptx.mul.ftz.f32(tau_log2, acc_scale_log2, lse_temperature_scale)
                        exp_tau = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(exp_tau, tau_log2)
                        K.assign(
                            temperature_acc_scale,
                            K.if_then_else(row_max > _f32(_NEG_INF), exp_tau, _f32(1.0)),
                        )
                K.assign(row_max, selected_max)
                with K.If(col_half == 0), K.Then():
                    K.ptx.st.shared.b32(scale_row_addr, acc_scale)
                K.ptx.fence.proxy.async_.shared__cta()
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_CORR_SIG))
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_CORR_SIG + 8))
                # The bias negates the (possibly re-selected) new_max_scaled after the branch.
                neg_bias = K.local_scalar("float32")
                K.ptx.neg.ftz.f32(neg_bias, new_max_scaled)
                score_bias = K.local_scalar("float32")
                K.assign(score_bias, K.if_then_else(valid_cols > _i32(0), neg_bias, _f32(_NEG_INF)))
                K.assign(block_temperature_sum, _f32(0.0))
                K.assign(block_sum, _f32(0.0))
                with K.If(return_temperature_lse != _i32(0)):
                    with K.Then():
                        tau_scale = K.local_scalar("float32")
                        K.ptx.mul.ftz.f32(tau_scale, softmax_scale_log2, lse_temperature_scale)
                        tau_bias = K.local_scalar("float32")
                        K.ptx.mul.ftz.f32(tau_bias, score_bias, lse_temperature_scale)
                        tau_scale_pair = _pack2(tau_scale, tau_scale)
                        tau_bias_pair = _pack2(tau_bias, tau_bias)
                        for j in range(32):
                            _packed_fma_inplace(scores, 2 * j, tau_scale_pair, tau_bias_pair)
                        for j in range(64):
                            _exp2_inplace(scores, j)
                        K.assign(block_temperature_sum, _block_sum(scores))
                        # S is re-read because the temperature sum consumed the first copy.
                        _tmem_load_x32(scores, 0, score_addr)
                        _tmem_load_x32(scores, 32, score_addr + _u32(32))
                        probabilities_from(scores, score_bias)
                    with K.Else():
                        probabilities_from(scores, score_bias)
                K.ptx.tcgen05.wait__st.sync.aligned()
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_P_FULL))
                        _mbar_wait(bar(_MBAR_CORR_DONE), phase_corr_done_0)
                        _flip(phase_corr_done_0)
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_P_FULL + 8))
                        _mbar_wait(bar(_MBAR_CORR_DONE + 8), phase_corr_done_1)
                        _flip(phase_corr_done_1)
                K.ptx.fma.rn.ftz.f32(row_sum, row_sum, acc_scale, block_sum)
                with K.If(return_temperature_lse != _i32(0)), K.Then():
                    K.ptx.fma.rn.ftz.f32(
                        temperature_sum,
                        temperature_sum,
                        temperature_acc_scale,
                        block_temperature_sum,
                    )
                K.assign(union_index, union_index + _i32(1))
            with K.If(col_half == 0), K.Then():
                K.ptx.st.shared.b32(scale_row_addr + _u32(128 * 4), row_sum)
                K.ptx.st.shared.b32(scale_row_addr + _u32(256 * 4), row_max)
                K.ptx.st.shared.b32(scale_row_addr + _u32(384 * 4), temperature_sum)
            K.ptx.fence.proxy.async_.shared__cta()
            with K.If(instance == 0):
                with K.Then():
                    _mbar_arrive(bar(_MBAR_CORR_SIG))
                with K.Else():
                    _mbar_arrive(bar(_MBAR_CORR_SIG + 8))

        # ---- Role: correction and epilogue (warps 8..11) ----
        with r_correction:
            q_valid_c = K.local_scalar("int32", init=_i32(128))
            with K.If(q_tile >= mb), K.Then():
                K.assign(q_valid_c, _i32(0))
            union_ready_phase_c = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_c)
            _flip(union_ready_phase_c)
            max_union_count_c = selected_blocks
            warp_in_role = warp - 8
            tmem_row_origin_c = warp_in_role * 32
            my_row_c = warp_in_role * 16 + lane % 16
            col_half_c = lane // 16
            row_addr = K.local_scalar(
                "uint32", init=K.shift_left(K.cast(tmem_row_origin_c, "uint32"), _u32(16))
            )
            scale_base_c = bar(_SMEM_SCALE) + K.cast(my_row_c, "uint32") * _u32(4)
            _mbar_arrive(bar(_MBAR_P_FULL))
            _mbar_arrive(bar(_MBAR_P_FULL + 8))
            phase_corr_sig_0 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_CORR_SIG), phase_corr_sig_0)
            _flip(phase_corr_sig_0)
            _mbar_arrive(bar(_MBAR_CORR_DONE))
            phase_corr_sig_1 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_CORR_SIG + 8), phase_corr_sig_1)
            _flip(phase_corr_sig_1)
            _mbar_arrive(bar(_MBAR_CORR_DONE + 8))
            o_frag = K.alloc_local((64,), "float32")

            def rescale_instance(instance, phase_corr_sig, union_index_1):
                count = selected_blocks
                with K.If(count > union_index_1), K.Then():
                    _mbar_wait(bar(_MBAR_CORR_SIG + 8 * instance), phase_corr_sig)
                    _flip(phase_corr_sig)
                    acc_scale_c = _ld_shared_f32(scale_base_c + _u32(64 * 4 * instance))
                    any_rescale = K.local_scalar("uint32")
                    K.ptx.vote_sync.any.pred(
                        any_rescale,
                        K.ptx.pred(K.cast(acc_scale_c < _f32(1.0), "bool")),
                        _u32(_FULL_MASK),
                    )
                    with K.If(any_rescale != _u32(0)), K.Then():
                        o_addr = taddr + _u32(_TMEM_OUTPUT[instance]) + row_addr
                        _tmem_load_x32(o_frag, 0, o_addr)
                        _tmem_load_x32(o_frag, 32, o_addr + _u32(32))
                        scale_pair = _pack2(acc_scale_c, acc_scale_c)
                        for j in range(32):
                            _packed_mul_inplace(o_frag, 2 * j, scale_pair)
                        K.ptx[_TMEM_ST_X64](o_addr, 64, *(o_frag[i] for i in range(64)))
                        K.ptx.tcgen05.wait__st.sync.aligned()
                    _mbar_arrive(bar(_MBAR_P_FULL + 8 * instance))
                    _mbar_arrive(bar(_MBAR_CORR_DONE + 8 * instance))

            union_index_1 = K.local_scalar("int32", init=_i32(1))
            with K.While(union_index_1 < max_union_count_c):
                rescale_instance(0, phase_corr_sig_0, union_index_1)
                rescale_instance(1, phase_corr_sig_1, union_index_1)
                K.assign(union_index_1, union_index_1 + _i32(1))
            phase_o_full_0 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_O_FULL), phase_o_full_0)
            _flip(phase_o_full_0)
            phase_o_full_1 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_O_FULL + 8), phase_o_full_1)
            _flip(phase_o_full_1)
            _mbar_wait(bar(_MBAR_CORR_SIG), phase_corr_sig_0)
            _flip(phase_corr_sig_0)
            _mbar_wait(bar(_MBAR_CORR_SIG + 8), phase_corr_sig_1)
            _flip(phase_corr_sig_1)
            K.ptx["tcgen05.fence::after_thread_sync"]()
            for instance in range(2):
                stats_addr = scale_base_c + _u32(64 * 4 * instance)
                final_sum = _ld_shared_f32(stats_addr + _u32(128 * 4))
                final_max = _ld_shared_f32(stats_addr + _u32(256 * 4))
                final_temperature_sum = _ld_shared_f32(stats_addr + _u32(384 * 4))
                rcp_sum = K.local_scalar("float32")
                K.ptx.rcp.approx.ftz.f32(rcp_sum, final_sum)
                sum_positive = K.local_scalar("int32", init=K.cast(final_sum > _f32(0.0), "int32"))
                inv_sum = K.local_scalar(
                    "float32", init=K.if_then_else(sum_positive != _i32(0), rcp_sum, _f32(0.0))
                )
                instance_valid_c = K.local_scalar("int32", init=q_valid_c - _i32(64 * instance))
                with K.If(instance_valid_c > _i32(64)), K.Then():
                    K.assign(instance_valid_c, _i32(64))
                with K.If(instance_valid_c < _i32(0)), K.Then():
                    K.assign(instance_valid_c, _i32(0))
                query = q_local_base + 64 * instance + my_row_c
                output_row = (query * num_q_heads + q_head) * 128
                o_addr = taddr + _u32(_TMEM_OUTPUT[instance]) + row_addr
                _tmem_load_x32(o_frag, 0, o_addr)
                _tmem_load_x32(o_frag, 32, o_addr + _u32(32))
                col_base = col_half_c * 64
                with K.If(my_row_c < instance_valid_c), K.Then():
                    inv_pair = _pack2(inv_sum, inv_sum)
                    for chunk in range(8):
                        base = 8 * chunk
                        for j in range(4):
                            _packed_mul_inplace(o_frag, base + 2 * j, inv_pair)
                        words = K.alloc_local((4,), "uint32")
                        for j in range(4):
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[j], o_frag[base + 2 * j + 1], o_frag[base + 2 * j]
                            )
                        K.ptx.st.global_.v4.b32(
                            out.ptr_to([output_row + col_base + base]),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                    with K.If(col_half_c == 0), K.Then():
                        stat_idx = query * num_q_heads + q_head
                        with K.If(return_softmax_lse != _i32(0)), K.Then():
                            log2_sum = K.local_scalar("float32")
                            K.ptx.lg2.approx.ftz.f32(log2_sum, final_sum)
                            max_scaled = K.local_scalar("float32")
                            K.ptx.mul.ftz.f32(max_scaled, final_max, softmax_scale_log2)
                            log_sum = K.local_scalar("float32")
                            K.ptx.mul.ftz.f32(log_sum, log2_sum, _f32(_LN2_F32))
                            lse_value = K.local_scalar("float32")
                            K.ptx.fma.rn.ftz.f32(lse_value, max_scaled, _f32(_LN2_F32), log_sum)
                            K.ptx.st.global_.b32(
                                lse.ptr_to([stat_idx]),
                                K.if_then_else(sum_positive != _i32(0), lse_value, _f32(_NEG_INF)),
                            )
                        with K.If(return_temperature_lse != _i32(0)), K.Then():
                            log2_tsum = K.local_scalar("float32")
                            K.ptx.lg2.approx.ftz.f32(log2_tsum, final_temperature_sum)
                            t_value = K.local_scalar("float32", init=_f32(_NEG_INF))
                            with K.If(final_temperature_sum > _f32(0.0)), K.Then():
                                max_scaled_t = K.local_scalar("float32")
                                K.ptx.mul.ftz.f32(max_scaled_t, final_max, softmax_scale_log2)
                                max_ln = K.local_scalar("float32")
                                K.ptx.mul.ftz.f32(max_ln, max_scaled_t, _f32(_LN2_F32))
                                log_tsum = K.local_scalar("float32")
                                K.ptx.mul.ftz.f32(log_tsum, log2_tsum, _f32(_LN2_F32))
                                K.ptx.fma.rn.ftz.f32(
                                    t_value, lse_temperature_scale, max_ln, log_tsum
                                )
                            K.ptx.st.global_.b32(temperature_lse.ptr_to([stat_idx]), t_value)
            K.ptx.tcgen05.wait__ld.sync.aligned()
            K.ptx["tcgen05.fence::before_thread_sync"]()
            _mbar_arrive(bar(_MBAR_TMEM_DEALLOC))

        # ---- Role: MMA issuer (warp 12) ----
        with r_mma:
            union_ready_phase_m = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_m)
            _flip(union_ready_phase_m)
            max_union_count_m = selected_blocks
            k_stage_cursor = K.local_scalar("uint32", init=_u32(0))
            k_phase_cursor = K.local_scalar("int32", init=_i32(0))
            v_stage_cursor = K.local_scalar("uint32", init=_u32(0))
            v_phase_cursor = K.local_scalar("int32", init=_i32(0))
            q_full_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_Q_FULL), q_full_phase)
            _flip(q_full_phase)
            first_pv = [K.local_scalar("int32", init=_i32(1)) for _ in range(2)]
            phase_p_full = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            q_addr = (bar(_SMEM_Q0), bar(_SMEM_Q1))
            desc_hi = _u32(_DESC_HI)

            def qk_chain(instance, k_stage):
                a_lo = K.local_scalar(
                    "uint32", init=K.uniform((q_addr[instance] >> 4) & _u32(0x3FFF))
                )
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(((bar(_SMEM_K) >> 4) & _u32(0x3FFF)) + k_stage * _u32(2048)),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                d_tmem = taddr + _u32(_TMEM_SCORES[instance])
                for k16 in range(8):
                    K.ptx[_MMA_F16](
                        d_tmem,
                        _pack2(a_lo, desc_hi),
                        _pack2(b_lo, desc_hi),
                        _u32(_QK_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(_u32(0 if k16 == 0 else 1)),
                        pred=leader,
                    )
                    if k16 < 7:
                        K.assign(a_lo, a_lo + _u32(_QK_A_STEPS[k16]))
                        K.assign(b_lo, b_lo + _u32(_QK_B_STEPS[k16]))
                K.ptx[_TCGEN05_COMMIT](bar(_MBAR_S_FULL + 8 * instance), pred=leader)
                K.ptx[_TCGEN05_COMMIT](bar(_MBAR_K_EMPTY) + k_stage * _u32(8), pred=leader)

            def pv_chain(instance, v_stage):
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        (((bar(_SMEM_V) >> 4) & _u32(0x3FFF)) | _u32(_V_LBO_BIT))
                        + v_stage * _u32(2048)
                    ),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                d_tmem = taddr + _u32(_TMEM_OUTPUT[instance])
                a_tmem = K.local_scalar(
                    "uint32", init=taddr + _u32(_TMEM_SCORES[instance] + _TMEM_P_OFFSET)
                )
                enable_first = K.local_scalar(
                    "uint32", init=K.if_then_else(first_pv[instance] != _i32(0), _u32(0), _u32(1))
                )
                for k16 in range(8):
                    K.ptx[_MMA_F16](
                        d_tmem,
                        a_tmem,
                        _pack2(b_lo, desc_hi),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(enable_first if k16 == 0 else _u32(1)),
                        pred=leader,
                    )
                    if k16 < 7:
                        K.assign(a_tmem, a_tmem + _u32(_PV_A_STEP))
                        K.assign(b_lo, b_lo + _u32(_PV_B_STEP))
                K.assign(first_pv[instance], _i32(0))
                return leader

            # Prime S for both instances before the first PV, exactly as the source.
            for instance in range(2):
                with K.If(selected_blocks > _i32(0)), K.Then():
                    k_stage = K.local_scalar("uint32", init=k_stage_cursor)
                    k_phase = K.local_scalar("int32", init=k_phase_cursor)
                    _advance_ring(k_stage_cursor, k_phase_cursor)
                    _mbar_wait(bar(_MBAR_K_FULL) + k_stage * _u32(8), k_phase)
                    qk_chain(instance, k_stage)

            union_index_2 = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_2 < max_union_count_m):
                for instance in range(2):
                    with K.If(selected_blocks > union_index_2), K.Then():
                        v_stage = K.local_scalar("uint32", init=v_stage_cursor)
                        v_phase = K.local_scalar("int32", init=v_phase_cursor)
                        _advance_v_ring(v_stage_cursor, v_phase_cursor)
                        _mbar_wait(bar(_MBAR_V_FULL) + v_stage * _u32(8), v_phase)
                        _mbar_wait(bar(_MBAR_P_FULL + 8 * instance), phase_p_full[instance])
                        _flip(phase_p_full[instance])
                        leader = pv_chain(instance, v_stage)
                        with K.If(union_index_2 + _i32(1) == selected_blocks), K.Then():
                            K.ptx[_TCGEN05_COMMIT](bar(_MBAR_O_FULL + 8 * instance), pred=leader)
                        K.ptx[_TCGEN05_COMMIT](bar(_MBAR_V_EMPTY) + v_stage * _u32(8), pred=leader)
                    next_union_index = union_index_2 + _i32(1)
                    with K.If(next_union_index < selected_blocks), K.Then():
                        next_k_stage = K.local_scalar("uint32", init=k_stage_cursor)
                        next_k_phase = K.local_scalar("int32", init=k_phase_cursor)
                        _advance_ring(k_stage_cursor, k_phase_cursor)
                        _mbar_wait(bar(_MBAR_K_FULL) + next_k_stage * _u32(8), next_k_phase)
                        qk_chain(instance, next_k_stage)
                K.assign(union_index_2, union_index_2 + _i32(1))
            dealloc_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_TMEM_DEALLOC), dealloc_phase)
            _flip(dealloc_phase)
            taddr_dealloc = K.local_scalar("uint32")
            K.ptx.ld.volatile.shared.b32(taddr_dealloc, bar(_SMEM_TMEM_MAILBOX))
            K.ptx[_TMEM_DEALLOC](taddr_dealloc, _u32(TMEM_COLS))

        # ---- Role: Q/mask/K load warp (warp 13) ----
        with r_qk_load:
            kv_head = q_head
            query_base = q_local_base
            load_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(load_leader != _u32(0)), K.Then():
                _mbar_expect_tx(bar(_MBAR_Q_FULL), 32768)
                K.ptx[_TMA_G2S_4D](
                    bar(_SMEM_Q0),
                    K.address_of(q_map),
                    _i32(0),
                    query_base,
                    q_head,
                    _i32(0),
                    bar(_MBAR_Q_FULL),
                )
                K.ptx[_TMA_G2S_4D](
                    bar(_SMEM_Q1),
                    K.address_of(q_map),
                    _i32(0),
                    query_base + _i32(64),
                    q_head,
                    _i32(0),
                    bar(_MBAR_Q_FULL),
                )
            q_block = query_base // 128
            mask_base = (q_head * mb + q_block) * nb
            selected_count = K.local_scalar("int32", init=_i32(0))
            block_base = K.local_scalar("int32", init=_i32(0))
            with K.While(block_base < nb):
                n_block_1 = block_base + lane
                selected_1 = K.local_scalar("int32", init=_i32(0))
                with K.If(n_block_1 < nb), K.Then():
                    mask_byte = K.local_scalar("uint16")
                    K.ptx.ld.global_.nc.b8(mask_byte, block_mask.ptr_to([mask_base + n_block_1]))
                    K.assign(selected_1, K.cast(mask_byte != K.uint16(0), "int32"))
                vote = K.local_scalar("uint32")
                K.ptx.vote_sync.ballot.b32(
                    vote, K.ptx.pred(K.cast(selected_1 != _i32(0), "bool")), _u32(_FULL_MASK)
                )
                ballot_count = K.local_scalar("uint32")
                K.ptx.popc.b32(ballot_count, vote)
                lower_lanes = K.shift_left(_u32(1), K.cast(lane, "uint32")) - _u32(1)
                lower_count = K.local_scalar("uint32")
                K.ptx.popc.b32(lower_count, vote & lower_lanes)
                selected_slot = selected_count + K.cast(lower_count, "int32")
                with K.If(selected_1 != _i32(0)), K.Then():
                    slot_addr = bar(_SMEM_UNION_BLOCKS) + K.cast(selected_slot, "uint32") * _u32(4)
                    K.ptx.st.shared.b32(slot_addr, n_block_1)
                    K.ptx.st.shared.b32(slot_addr + _u32(192 * 4), n_block_1)
                K.assign(selected_count, selected_count + K.cast(ballot_count, "int32"))
                K.assign(block_base, block_base + _i32(32))
            with K.If(selected_count == _i32(0)), K.Then():
                with K.If(lane == 0), K.Then():
                    K.ptx.st.shared.b32(bar(_SMEM_UNION_BLOCKS), _i32(0))
                    K.ptx.st.shared.b32(bar(_SMEM_UNION_BLOCKS + 192 * 4), _i32(0))
                K.assign(selected_count, _i32(1))
            with K.If(lane < 2), K.Then():
                K.ptx.st.shared.b32(
                    bar(_SMEM_UNION_COUNT) + K.cast(lane, "uint32") * _u32(4), selected_count
                )
            K.ptx.barrier.sync(_u32(8), _u32(32))
            K.ptx.fence.proxy.async_.shared__cta()
            _mbar_arrive(bar(_MBAR_UNION_READY))
            k_stage_load = K.local_scalar("uint32", init=_u32(0))
            phase_k_empty = K.local_scalar("int32", init=_i32(1))
            union_index_3 = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_3 < selected_blocks):
                for instance in range(2):
                    n_block = _ld_shared_i32(
                        bar(_SMEM_UNION_BLOCKS + 192 * 4 * instance)
                        + K.cast(union_index_3, "uint32") * _u32(4)
                    )
                    token_base = n_block * 128
                    _mbar_wait(bar(_MBAR_K_EMPTY) + k_stage_load * _u32(8), phase_k_empty)
                    push_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    with K.If(push_leader != _u32(0)), K.Then():
                        full_bar = bar(_MBAR_K_FULL) + k_stage_load * _u32(8)
                        _mbar_expect_tx(full_bar, 32768)
                        stage_addr = bar(_SMEM_K) + k_stage_load * _u32(_SMEM_STAGE_BYTES)
                        token1 = token_base + _i32(64)
                        for dim_half in range(2):
                            for token_half, token in ((0, token_base), (1, token1)):
                                K.ptx[_TMA_G2S_4D](
                                    stage_addr + _u32(8192 * token_half + 16384 * dim_half),
                                    K.address_of(k_map),
                                    _i32(0),
                                    token,
                                    _i32(dim_half),
                                    kv_head,
                                    full_bar,
                                )
                    _advance_ring(k_stage_load, phase_k_empty)
                K.assign(union_index_3, union_index_3 + _i32(1))

        # ---- Role: V load warp (warp 14) ----
        with r_v_load:
            union_ready_phase_v = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_v)
            _flip(union_ready_phase_v)
            kv_head = q_head
            v_stage_load = K.local_scalar("uint32", init=_u32(0))
            phase_v_empty = K.local_scalar("int32", init=_i32(1))
            union_index_4 = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_4 < selected_blocks):
                for instance in range(2):
                    n_block = _ld_shared_i32(
                        bar(_SMEM_UNION_BLOCKS + 192 * 4 * instance)
                        + K.cast(union_index_4, "uint32") * _u32(4)
                    )
                    token_base = n_block * 128
                    _mbar_wait(bar(_MBAR_V_EMPTY) + v_stage_load * _u32(8), phase_v_empty)
                    push_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    with K.If(push_leader != _u32(0)), K.Then():
                        full_bar = bar(_MBAR_V_FULL) + v_stage_load * _u32(8)
                        _mbar_expect_tx(full_bar, 32768)
                        stage_addr = bar(_SMEM_V) + v_stage_load * _u32(_SMEM_STAGE_BYTES)
                        token1 = token_base + _i32(64)
                        for dim_half in range(2):
                            for token_half, token in ((0, token_base), (1, token1)):
                                K.ptx[_TMA_G2S_4D](
                                    stage_addr + _u32(8192 * token_half + 16384 * dim_half),
                                    K.address_of(v_map),
                                    _i32(0),
                                    token,
                                    _i32(dim_half),
                                    kv_head,
                                    full_bar,
                                )
                    _advance_v_ring(v_stage_load, phase_v_empty)
                K.assign(union_index_4, union_index_4 + _i32(1))

        # ---- Role: idle (warp 15) ----
        with r_idle:
            pass

    return cake_vsa_longseq_sm100


@lru_cache(maxsize=1)
def _kernel():
    return _build_kernel()


def get_kernel(**config: Any):
    """Return the TIRx PrimFunc; the kernel is shape-generic at runtime."""
    if config:
        cfg = _without_label(config)
        _validate_shape(cfg["M"], cfg["N"], cfg["num_heads"])
    return _kernel().func


# Match the upstream cubin's native ptxas ``--register-usage-level=5``.  The
# TIRx nvcc path otherwise defaults to level 10, which leaves this kernel at the
# same 128-register allocation but introduces a 56-byte issuer-role stack and
# dynamic local spill traffic.  A fixed level 4/5/6 bench-suite sweep selected
# level 5 across the short, streaming, and capacity regimes.
_PTXAS_REG_LEVEL = "5"


@lru_cache(maxsize=1)
def _compiled_kernel():
    import os

    from tirx_kernels.runner import compile_kernel

    # nvcc mode keeps the TIRx build on the CUDA toolkit on PATH (13.2, PTX ISA
    # 9.2) -- the same toolchain the upstream ``nvcc -cubin`` build uses -- so
    # both binaries share one PTX ISA for SASS and profiler comparisons.
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REG_LEVEL
    try:
        return compile_kernel(get_kernel(), cuda_compile_mode="nvcc")
    finally:
        if previous is None:
            del os.environ["TVM_CUDA_PTXAS_REG_LEVEL"]
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def _tirx_args(data: dict[str, Any], slot: str = "tirx") -> tuple[Any, ...]:
    buffers = data[slot]
    return (
        data["q"].view(-1),
        data["k"].view(-1),
        data["v"].view(-1),
        buffers["out"].view(-1),
        buffers["lse"].view(-1),
        buffers["tlse"].view(-1),
        data["block_mask_u8"].view(-1),
        data["mb"],
        data["nb"],
        data["selected_blocks"],
        data["num_heads"],
        data["num_heads"],
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_lse"]),
        int(data["return_temperature_lse"]),
    )


def _tirx_launch(executable, data: dict[str, Any]):
    arguments = _tirx_args(data)

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


# ---------------------------------------------------------------------------
# Upstream source kernel (the reference for correctness and benchmarks)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _source_module():
    """Build the upstream cubin + tvm-ffi launcher exactly as FlashInfer does."""
    from flashinfer import cake_vsa

    return cake_vsa._load_module(PROFILE, CUDA_ARCH)


def _source_launch(data: dict[str, Any]):
    """Replicate ``flashinfer.cake_vsa._run_standard`` for ``longseq``."""
    import tvm_ffi

    module = _source_module()
    buffers = data["source"]
    arguments = (
        data["q"],
        data["k"],
        data["v"],
        buffers["out"],
        buffers["lse"],
        buffers["tlse"],
        data["block_mask_u8"],
        data["mb"],
        data["nb"],
        data["selected_blocks"],
        data["num_heads"],
        data["num_heads"],
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_lse"]),
        int(data["return_temperature_lse"]),
        data["mb"],
        data["num_heads"],
        1,
    )

    def launch():
        with tvm_ffi.use_torch_stream():
            module.run(*arguments)

    launch._keep_alive = arguments
    return launch


def _run_public_api(data: dict[str, Any]):
    """Run the upstream public API (``plan_cake_vsa`` + ``run_cake_vsa``)."""
    import torch
    from flashinfer import cake_vsa

    plan = cake_vsa.plan_cake_vsa(
        None,
        None,
        data["block_mask"],
        None,
        None,
        None,
        M=data["M"],
        N=data["N"],
        R=BLOCK,
        C=BLOCK,
        num_qo_heads=data["num_heads"],
        num_kv_heads=data["num_heads"],
        head_dim=HEAD_DIM,
        q_data_type=torch.bfloat16,
        sm_scale=data["sm_scale"],
        device=data["q"].device,
    )
    return cake_vsa.run_cake_vsa(
        plan,
        data["q"],
        data["k"],
        data["v"],
        out=None,
        lse=None,
        return_lse=bool(data["return_lse"]),
        backend="cake",
    )


# ---------------------------------------------------------------------------
# fp32 oracle and validation
# ---------------------------------------------------------------------------
def _oracle(data: dict[str, Any]):
    """Dense fp32 reference: masked softmax attention, one head at a time."""
    import torch

    q = data["q"].float()
    k = data["k"].float()
    v = data["v"].float()
    mask = data["block_mask"]
    M, num_heads = data["M"], data["num_heads"]
    scale = data["sm_scale"]
    tau = data["lse_temperature_scale"]
    out = torch.empty((M, num_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
    lse = torch.empty((M, num_heads), dtype=torch.float32, device=q.device)
    tlse = torch.empty((M, num_heads), dtype=torch.float32, device=q.device)
    for head in range(num_heads):
        scores = torch.matmul(q[:, head], k[:, head].transpose(0, 1)) * scale
        token_mask = mask[head].repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        scores.masked_fill_(~token_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, head] = torch.matmul(probs, v[:, head])
        lse[:, head] = torch.logsumexp(scores, dim=-1)
        tlse[:, head] = torch.logsumexp(scores * tau, dim=-1)
    return out, lse, tlse


def _empty_rows(data: dict[str, Any]):
    """Boolean ``(M, num_heads)`` map of query rows whose block row selects nothing."""
    mask = data["block_mask"]
    empty = ~mask.any(dim=-1)  # (num_heads, mb)
    return empty.repeat_interleave(BLOCK, 1).transpose(0, 1)  # (M, num_heads)


def _bf16_order_key(values):
    """Map bf16 bit patterns to integers ordered like the reals (for ULP distances)."""
    import torch

    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits < 0x8000, bits + 0x8000, 0x8000 - (bits & 0x7FFF))


def _assert_bitwise(name: str, ours, theirs) -> None:
    import torch

    if ours.dtype == torch.bfloat16:
        ours_bits, theirs_bits = ours.view(torch.int16), theirs.view(torch.int16)
    else:
        ours_bits, theirs_bits = ours.view(torch.int32), theirs.view(torch.int32)
    mismatch = ours_bits != theirs_bits
    count = int(mismatch.sum())
    if count:
        index = mismatch.reshape(-1).nonzero()[0].item()
        raise AssertionError(
            f"{name}: {count} of {mismatch.numel()} elements differ bitwise from the source "
            f"kernel (first at flat index {index}: tirx={ours.reshape(-1)[index].item()!r} "
            f"source={theirs.reshape(-1)[index].item()!r})"
        )


def _validate_outputs(data: dict[str, Any], *, with_oracle: bool) -> dict[str, float]:
    import torch

    tirx, source = data["tirx"], data["source"]
    return_lse, return_tlse = data["return_lse"], data["return_temperature_lse"]

    for name, buffers in (("tirx", tirx), ("source", source)):
        if not torch.isfinite(buffers["out"].float()).all():
            raise AssertionError(f"{name}: non-finite values in out")
        if return_lse and not torch.isfinite(buffers["lse"]).all():
            raise AssertionError(f"{name}: non-finite values in lse")
        if return_tlse and not torch.isfinite(buffers["tlse"]).all():
            raise AssertionError(f"{name}: non-finite values in temperature lse")
        if not return_lse and not bool((buffers["lse"] == _LSE_SENTINEL).all()):
            raise AssertionError(f"{name}: lse sentinel was overwritten with return_lse=0")

    _assert_bitwise("out", tirx["out"], source["out"])
    _assert_bitwise("lse", tirx["lse"], source["lse"])
    if return_tlse:
        _assert_bitwise("temperature_lse", tirx["tlse"], source["tlse"])

    stats: dict[str, float] = {}
    if not with_oracle:
        return stats
    ref_out, ref_lse, ref_tlse = _oracle(data)
    valid = ~_empty_rows(data)  # rows with an undefined dense reference are skipped
    cfg = data["config"]
    for name, buffers in (("tirx", tirx), ("source", source)):
        out = buffers["out"].float()[valid]
        expected = ref_out[valid]
        err = (out - expected).abs()
        bound = cfg["oracle_atol"] + cfg["oracle_rtol"] * expected.abs()
        stats[f"{name}_out_max_abs_err"] = float(err.max())
        stats[f"{name}_out_max_rel_excess"] = float((err / bound).max())
        if bool((err > bound).any()):
            raise AssertionError(
                f"{name}: out mismatch vs fp32 oracle (max abs err {float(err.max()):.3e}, "
                f"max err/bound {float((err / bound).max()):.3f})"
            )
        # bf16 ULP accounting against the oracle rounded to bf16 (sign-magnitude
        # bit patterns are mapped to a monotonic integer key first), on the
        # elements large enough for a ULP to be meaningful.
        sizeable = expected.abs() >= _ORACLE_ULP_MIN_REF
        if bool(sizeable.any()):
            ulps = (
                _bf16_order_key(expected.to(torch.bfloat16))
                - _bf16_order_key(buffers["out"][valid])
            ).abs()[sizeable]
            ulp_max = float(ulps.max())
            total = ulps.numel()
            le1 = int((ulps <= 1).sum()) / total
            le2 = int((ulps <= 2).sum()) / total
            stats[f"{name}_out_ulp_max"] = ulp_max
            stats[f"{name}_out_ulp_le1_frac"] = le1
            stats[f"{name}_out_ulp_le2_frac"] = le2
            if ulp_max > cfg["ulp_max"] or le1 < cfg["ulp_le1_frac"] or le2 < cfg["ulp_le2_frac"]:
                raise AssertionError(
                    f"{name}: bf16 ULP distribution vs oracle too wide (max {ulp_max:.0f}, "
                    f"<=1 ULP {le1:.5f}, <=2 ULP {le2:.5f})"
                )
        small = expected.abs() < 0.125
        large = expected.abs() >= 0.5
        stats[f"{name}_out_max_err_small_ref"] = (
            float(err[small].max()) if bool(small.any()) else 0.0
        )
        stats[f"{name}_out_max_rel_err_large_ref"] = (
            float((err[large] / expected[large].abs()).max()) if bool(large.any()) else 0.0
        )
        if return_lse:
            lse = buffers["lse"][valid]
            expected_lse = ref_lse[valid]
            err = (lse - expected_lse).abs()
            bound = _ORACLE_LSE_ATOL + _ORACLE_LSE_RTOL * expected_lse.abs()
            stats[f"{name}_lse_max_abs_err"] = float(err.max())
            if bool((err > bound).any()):
                raise AssertionError(
                    f"{name}: lse mismatch vs oracle (max abs err {float(err.max()):.3e})"
                )
        if return_tlse:
            tlse = buffers["tlse"][valid]
            expected_tlse = ref_tlse[valid]
            err = (tlse - expected_tlse).abs()
            bound = _ORACLE_LSE_ATOL + _ORACLE_LSE_RTOL * expected_tlse.abs()
            stats[f"{name}_tlse_max_abs_err"] = float(err.max())
            if bool((err > bound).any()):
                raise AssertionError(
                    f"{name}: temperature lse mismatch vs oracle (max abs err {float(err.max()):.3e})"
                )
    return stats


def _skip_unless_supported() -> None:
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required")
    if torch.cuda.get_device_capability() != (10, 0):
        raise SkipTest("cake_vsa longseq sm_100a requires compute capability 10.0")


def run_test(**config: Any) -> dict[str, float]:
    """Compile, launch TIRx and the upstream kernel, and validate one config."""
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    executable = _compiled_kernel()
    tirx_launch = _tirx_launch(executable, data)
    tirx_launch()
    torch.cuda.synchronize()
    first_out = data["tirx"]["out"].clone()
    first_lse = data["tirx"]["lse"].clone()

    source_launch = _source_launch(data)
    source_launch()
    torch.cuda.synchronize()
    stats = _validate_outputs(data, with_oracle=True)

    # A second TIRx launch into the same buffers must reproduce the first bit for bit.
    tirx_launch()
    torch.cuda.synchronize()
    _assert_bitwise("out (repeat launch)", data["tirx"]["out"], first_out)
    _assert_bitwise("lse (repeat launch)", data["tirx"]["lse"], first_lse)

    if data["config"]["pattern"] == "upstream":
        # The public API must dispatch this shape to longseq and agree bitwise.
        result = _run_public_api(data)
        api_out, api_lse = result if data["return_lse"] else (result, None)
        _assert_bitwise("out (public API)", data["tirx"]["out"], api_out)
        if api_lse is not None:
            _assert_bitwise("lse (public API)", data["tirx"]["lse"], api_lse)
    return stats


# ---------------------------------------------------------------------------
# Benchmark entry points
# ---------------------------------------------------------------------------
def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = _without_label(config)
    _validate_shape(kernel_config["M"], kernel_config["N"], kernel_config["num_heads"])
    state = {"config": kernel_config, "executable": _compiled_kernel()}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = _without_label({**prepared["config"], **kwargs})
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        data = prepare_data(**config)
        gpu_state = {
            "data": data,
            "tirx_launch": _tirx_launch(prepared["executable"], data),
            "source_launch": None,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    data = gpu_state["data"]
    tirx_launch = gpu_state["tirx_launch"]
    if not gpu_state["validated"]:
        tirx_launch()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                source_launch = _source_launch(data)
                gpu_state["source_launch"] = source_launch
                source_launch()
                torch.cuda.synchronize()
            _validate_outputs(data, with_oracle=False)
        gpu_state["validated"] = True

    source_launch = gpu_state["source_launch"]
    references = {"flashinfer": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config: Any):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
