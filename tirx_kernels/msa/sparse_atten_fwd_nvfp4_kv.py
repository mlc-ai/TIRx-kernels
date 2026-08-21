# This file is a TIRx port of code from MSA
# (https://github.com/MiniMax-AI/MSA @ 80434d7f), Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MSA sparse attention forward over NVFP4 K/V, split-partial form.

Ports ``SparseAttentionForwardNvfp4KvSm100``: the same attention this package's
BF16/FP8 forward computes, but reading K and V as packed NVFP4 -- two E2M1
values per byte, with one E4M3 block scale per sixteen head-dim elements and an
optional FP32 tensor scale.

Nothing here feeds FP4 to the tensor cores. TMA stages the packed bytes into
shared memory, the two softmax warpgroups dequantize them into ordinary BF16 or
FP8 tiles (WG0 takes K, WG1 takes V), and the MMA sees the same operand dtype
the sibling kernel sees. The dequant target is Q's dtype, so one MMA kind
serves both GEMMs.

Where the K tensor scale lands depends on that dtype. Under BF16 Q it folds
into the dequantized values. Under FP8 Q it cannot -- E4M3 would overflow -- so
it multiplies the FP32 S accumulator instead, between the TMEM read and the
mask.

Upstream source: python/fmha_sm100/cute/src/sm100/fwd/atten_fwd_nvfp4_kv.py:58.
Scale layout and the dequant oracle: python/fmha_sm100/cute/quantize.py:58,:265.
"""

from __future__ import annotations

from typing import Any

from tirx_kernels.msa.utils._scalar_ops import (
    ld_global_i32,
    ld_shared_i32,
    st_shared_i32,
    uceil_div_i32,
    udiv_i32,
)
from tvm.script import tirx as T
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.lang.smem_desc import SmemDescriptor

KERNEL_META = {
    "name": "msa_sparse_atten_fwd_nvfp4_kv_sm100",
    "category": "msa",
    "compute_capability": 10,
}

HEAD_DIM = 128
M_BLOCK = 128
N_BLOCK = 128

# Two E2M1 values per byte, so the stored head-dim extent is halved.
PACKED_HEAD_DIM = HEAD_DIM // 2
# One E4M3 scale byte per this many head-dim elements (quantize.py:22).
SCALE_BLOCK = 16
# The scale tensor is padded to whole 128x4 tiles (quantize.py:93-142).
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4

LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

# The config keys the CSR/work-list builder in this package accepts.
_CSR_CONFIG_KEYS = (
    "batch",
    "seqlen_q",
    "seqlen_k",
    "head_kv",
    "qhead_per_kv",
    "topk",
    "blk_kv",
    "seqlen_pattern",
)


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float8_e4m3": torch.float8_e4m3fn,
    }[name]


# The two in-scope dtype modes. Unlike the BF16/FP8 sibling, this kernel takes
# no separate qk/pv choice: `k_dtype = v_dtype = q_dtype` (:333-334) and a
# single `mma_kind` follows from Q's width (:347).
DTYPE_MODES: dict[str, dict[str, str]] = {
    "bf16q": {"q": "bfloat16", "mma": "bfloat16"},
    "fp8q": {"q": "float8_e4m3", "mma": "float8_e4m3"},
}

# `mO_partial.element_type`. fp16 is accepted upstream but never exercised for
# nvfp4, so it stays out of this port's domain.
PARTIAL_DTYPES = ("float32", "bfloat16", "float8_e4m3")


# ---------------------------------------------------------------------------
# Static shape and role layout (:61-195).
# ---------------------------------------------------------------------------
# `k_tile = 64` (:61): the UTCMMA bf16 K-tile, and the bf16 Q sub-tile width.
K_TILE = 64

# `warps_per_group=4`, `total_warps=16` (:130-143). Same sixteen-warp split as
# the BF16/FP8 forward, but warps 0..3 and 4..7 each carry a dequantization pass
# before they enter the softmax loop.
NUM_THREADS = 512
WARP_SIZE = 32
TOTAL_WARPS = 16
WARPS_PER_GROUP = 4
SOFTMAX0_WARP_BASE = 0
SOFTMAX1_WARP_BASE = 4
Q_LOAD_WARP_BASE = 8
MMA_WARP_ID = 12
KV_LOAD_WARP_BASE = 13
NUM_KV_LOAD_WARPS = 2
NUM_Q_LOAD_WARPS = 4
SOFTMAX_THREADS = WARP_SIZE * WARPS_PER_GROUP
# `num_dequant_warps = warps_per_group` (:1791): the SOFTMAX warpgroup's own four
# warps run the dequantization, not the Q-load warpgroup. The same number of
# threads, a different constant; conflating them survives every shape where the
# two happen to agree.
NUM_DEQUANT_WARPS = WARPS_PER_GROUP
DEQUANT_THREADS = WARP_SIZE * NUM_DEQUANT_WARPS

# Scale columns across one 128-element row: one E4M3 byte per 16 elements.
SCALE_COLS = HEAD_DIM // SCALE_BLOCK

# Pipeline depths (:116-125). `kv_stage` is 1: a CTA computes exactly one KV
# block, so K and V are single-shot rather than a ring.
Q_STAGE = 2
S_STAGE = 2
O_STAGE = 2
QIDX_META_STAGES = 16

# `tmem_total` (:156-161), power-of-two rounded.
TMEM_TOTAL = 512

# `split_P_arrive = n_block // 4 * 3`, floored to a multiple of 32 (:165-167).
SPLIT_P_ARRIVE = 96

# Register budgets on the causal path (:173-181).
NUM_REGS_SOFTMAX = 176
NUM_REGS_STORE = 112
NUM_REGS_OTHER = 48
# `ex2_emu_freq = 16 if causal else 0` (:184), `ex2_emu_start_frg = 1` (:185),
# and `ex2_emu_res` keeps softmax.py's default of 4. All three enter the
# emulation predicate; a stride test on the element index picks a different
# sixteen elements and changes the result bitwise.
EX2_EMU_FREQ = 16
EX2_EMU_RES = 4
EX2_EMU_START_FRG = 1
# `frg_tile` / `frg_cnt` (softmax.py:370-372): the predicate is two-dimensional
# over (fragment, position-in-fragment), so both extents are part of it.
EX2_FRG_TILE = 32
EX2_FRG_CNT = N_BLOCK // EX2_FRG_TILE

# cuTensorMapEncodeTiled enum values.
_SWIZZLE_128B = 3
_L2_PROMOTION_256B = 3

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3": 1}

# `scheduler_metadata` columns; the forward reads all six.
WORK_FIELDS = 6

# `q_idx | ((split_slot & 0xFF) << 24)`, decoded at :256-262.
SLOT_SHIFT = 24
SLOT_MASK = 0xFF
Q_IDX_MASK = (1 << SLOT_SHIFT) - 1


def _tirx_dtype(name: str) -> str:
    """The dtype TIRx declares a buffer with.

    CUDA codegen has no fp8 scalar type, and MSA does not want one either: it
    recasts its fp8 tiles to 32-bit words and converts them with a byte-permute
    plus a packed FMA. So an fp8 buffer is declared as raw ``uint8`` and every
    conversion goes through PTX on the bit pattern.
    """
    return "uint8" if name == "float8_e4m3" else name


def _swizzle_elems(elem_bytes: int) -> int:
    """Elements spanned by one 128-byte swizzle atom, i.e. the TMA box width."""
    return 128 // elem_bytes


def USE_GATHER4(qheadperkv: int) -> bool:
    """`use_q_gather4` (:84-89): 1, 2 and 4 take the raw gather4 Q path."""
    return qheadperkv in (1, 2, 4)


def Q_SUBTILES(elem_bytes: int) -> int:
    """Q sub-tiles per token: 1 for fp8 Q, `k_stages` = 2 for bf16 (:2122-2126)."""
    return HEAD_DIM // _swizzle_elems(elem_bytes)


def TOKENS_PER_WARP(qheadperkv: int) -> int:
    """`tokens_per_warp` (:2106-2109) on the TMA-Q path."""
    q_tokens_per_group = M_BLOCK // qheadperkv
    return (q_tokens_per_group + NUM_Q_LOAD_WARPS - 1) // NUM_Q_LOAD_WARPS


# Named-barrier ids. MSA emits `NamedBarrierFwdSm100`'s raw enum values with no
# user-barrier bias, which collides -- `StoreEpilogue + stage` reaches `KvLoad`'s
# id with a different participant count. The port keeps the synchronization
# structure and numbers from 8 upward, the convention this repository documents
# at `flashmla/sparse_decode_head64.py:36-39`.
#
# The source also declares `SoftmaxStatsW0..W7`, and this kernel uses none of
# them: `_wg_softmax` passes `signal_stats_barrier=False` (:2784-2786,
# :2795-2797) and `_epilogue_step` gets `use_stats_barrier=False` (:2853), so
# both guarded sites compile out. Dropping the eight dead ids is what lets this
# renumbering fit -- `BAR_EPILOGUE + stage` occupies 13 and 14, seven ids in all.
BAR_TMEM_ALLOC = 8
BAR_LOAD_WG = 9
BAR_KV_LOAD = 10
BAR_KV_DEQUANT_K = 11
BAR_KV_DEQUANT_V = 12
BAR_EPILOGUE = 13  # and 14, indexed by the softmax stage

_TMA_GATHER4_2D_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_GATHER4_PREFETCH = "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint"
_TMA_G2S_2D_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_G2S_3D_CACHE = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
)
# L2 eviction policy operands. The `.L2::cache_hint` qualifier is present on
# every bulk-tensor copy here, but the OPERAND is zero on all of them except the
# gather4 Q path: the K/V loads and the TMA-Q loads go through quack's
# `tma_get_copy_fn`, which passes no policy, while only the raw-descriptor
# gather4 issue sets one (tma_utils.py:24, :204). The export is unambiguous --
# six copies with `mov.u64 %rd, 0` in a TMA-Q build, and the EVICT_LAST constant
# appearing eight times in a gather4 build and zero times otherwise. The sibling
# port assigns policies to both tensors as a measured performance change; that
# is not this reference.
_TMA_CACHE_EVICT_FIRST = T.uint64(0x12F0000000000000)
_TMA_CACHE_EVICT_LAST = T.uint64(0x14F0000000000000)
_GATHER4_Q_CACHE_HINT = _TMA_CACHE_EVICT_LAST
_TMA_NO_POLICY = T.uint64(0)


_MMA_KIND = {"bfloat16": "f16", "float16": "f16", "float8_e4m3": "f8f6f4"}
_MMA_CHAIN = {kind: f"tcgen05.mma.cta_group::1.kind::{kind}" for kind in ("f16", "f8f6f4")}
_MMA_K = {"f16": 16, "f8f6f4": 32}
_MMA_ELEM_BYTES = {"f16": 2, "f8f6f4": 1}
_SWIZZLE_BYTES = 128
_SUBTILE_16B = 1024
_MMA_K_16B = 2
_PV_K_16B = {"f16": 128, "f8f6f4": 256}
_PV_A_COL_STEP = {"f16": 8, "f8f6f4": 8}
_P_STORE_REP = {"bfloat16": 16, "float16": 16, "float8_e4m3": 8}


def _instr_desc(operand_dtype: str, trans_b: bool = False) -> int:
    """The tcgen05.mma instruction descriptor for a 128x128 f32-accumulating tile.

    Computed, not copied: the immediate encodes the operand dtype, so bf16 and
    f16 differ even though both are ``kind::f16`` chains. ``trans_b`` is the B
    operand's major-ness -- False for QK, where both operands are K-major, and
    True for PV, whose B tile is MN-major.
    """
    from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32

    torch_name = {"bfloat16": "bfloat16", "float16": "float16", "float8_e4m3": "float8_e4m3fn"}[
        operand_dtype
    ]
    k = 32 if operand_dtype == "float8_e4m3" else 16
    return encode_instr_descriptor_dense_uint32(
        M_BLOCK, N_BLOCK, k, "float32", torch_name, torch_name, False, trans_b, cta_group=1
    )


def _stage_16b(operand_dtype: str) -> int:
    """One Q pipeline stage, in 16-byte units: 2048 for bf16, 1024 for fp8."""
    return M_BLOCK * HEAD_DIM * _MMA_ELEM_BYTES[_MMA_KIND[operand_dtype]] // 16


# `POLY_EX2[3]` (utils.py:24-40): the degree-3 minimax polynomial for 2**frac on
# [0, 1), evaluated by Horner in packed f32x2.
_POLY_EX2_3 = (
    1.0,
    0.695146143436431884765625,
    0.227564394474029541015625,
    0.077119089663028717041015625,
)
# `fp32_round_int = 2**23 + 2**22` (utils.py:991).
_FP32_ROUND_INT = float(2**23 + 2**22)
NEG_INF = float("-inf")
LN_2 = 0.6931471805599453
# `MASK_R2P_CHUNK_SIZE` (mask.py:15).
MASK_R2P_CHUNK = 32

# FP8 dequant staging depth. The pair path runs exactly `N_BLOCK * (SCALE_COLS
# // 2) // DEQUANT_THREADS` = 4 iterations per thread, so batching all four
# issues every shared load before the first conversion chain depends on one.
# Module scope, because a bare assignment inside a traced body binds a TIR
# variable and the staging buffer stops being a constant-size allocation.
DEQUANT_FP8_BATCH = 4
DEQUANT_FP8_STAGED = 4 * DEQUANT_FP8_BATCH


# ---------------------------------------------------------------------------
# Register-level primitives.
# ---------------------------------------------------------------------------
def _scale_gather(dst, src, cols, scale):
    """Gather one output group's columns from the O fragment and scale them.

    A plain Python function, so its loop runs at trace time and ``cols`` -- a
    Python list -- can index the fragment directly.
    """
    for j in range(len(cols) // 2):
        _packed_f32x2(
            "mul.rn.f32x2",
            dst,
            j * 2,
            j * 2 + 1,
            src[cols[j * 2]],
            src[cols[j * 2 + 1]],
            scale,
            scale,
        )


@T.inline
def _packed_f32x2(op, dst, di, dj, a0, a1, b0, b1, c0=None, c1=None):
    """One packed two-lane f32 operation.

    The ``.f32x2`` instructions take 64-bit packed operands, so each pair is
    assembled with ``mov.b64`` and split again afterwards. This is one
    instruction with two ordered results, not two scalar operations.
    """
    pa = T.alloc_local((1,), "uint64")
    pb = T.alloc_local((1,), "uint64")
    pd = T.alloc_local((1,), "uint64")
    T.ptx.mov.b64(pa[0], a0, a1)
    T.ptx.mov.b64(pb[0], b0, b1)
    if c0 is None:
        T.ptx[op](pd[0], pa[0], pb[0])
    else:
        pc = T.alloc_local((1,), "uint64")
        T.ptx.mov.b64(pc[0], c0, c1)
        T.ptx[op](pd[0], pa[0], pb[0], pc[0])
    T.ptx.mov.b64(dst[di], dst[dj], pd[0])


@T.inline
def _max3_at(dst, idx, a, b, c):
    """Three-input ``max.f32``, an SM100 form, writing into ``dst[idx]``."""
    T.ptx.max.f32(dst[idx], a, b, c)


@T.inline
def _row_max_128(regs, out, out_idx):
    """`_compute_row_max` for arch 100 (utils.py:258-276).

    Four accumulators SEEDED from the first eight elements with the two-source
    form, each absorbing two more per step through a three-input max, then
    folded in two steps. The seed is the initialization -- there is no -inf
    fill, and the strided loop therefore starts at 8. Dropping the
    ``local_max[0] = fmax(local_max[0], local_max[1])`` fold leaves a quarter of
    the row out of the maximum without changing the instruction shape.
    """
    acc = T.alloc_local((4,), "float32")
    T.evaluate(T.ptx.max.f32(acc[0], regs[0], regs[1]))
    T.evaluate(T.ptx.max.f32(acc[1], regs[2], regs[3]))
    T.evaluate(T.ptx.max.f32(acc[2], regs[4], regs[5]))
    T.evaluate(T.ptx.max.f32(acc[3], regs[6], regs[7]))
    for it in T.unroll(1, N_BLOCK // 8):
        i: T.int32 = it * 8
        _max3_at(acc, 0, acc[0], regs[i + 0], regs[i + 1])
        _max3_at(acc, 1, acc[1], regs[i + 2], regs[i + 3])
        _max3_at(acc, 2, acc[2], regs[i + 4], regs[i + 5])
        _max3_at(acc, 3, acc[3], regs[i + 6], regs[i + 7])
    T.evaluate(T.ptx.max.f32(acc[0], acc[0], acc[1]))
    _max3_at(out, out_idx, acc[0], acc[2], acc[3])


@T.inline
def _scaled_exp2_row_sum_128(regs, scale, out):
    """`fadd_exp2_scaled_reduce` for arch 100 (utils.py:308-350).

    A second pass over the same 128 elements at a different scale, for the
    temperature LSE. It carries NO ex2 emulation -- every one of the 128 is a
    real ``ex2`` -- which is why a temperature build's ex2 census is 480 rather
    than 448. It reads the fragment without writing it back, so the P
    population below still sees the scale-subtracted values.
    """
    acc = T.alloc_local((8,), "float32")
    for j in T.unroll(8):
        acc[j] = T.float32(0.0)
    tmp = T.alloc_local((8,), "float32")
    for it in T.unroll(N_BLOCK // 8):
        i: T.int32 = it * 8
        for j in T.unroll(4):
            _packed_f32x2(
                "mul.rn.f32x2",
                tmp,
                j * 2,
                j * 2 + 1,
                regs[i + j * 2],
                regs[i + j * 2 + 1],
                scale,
                scale,
            )
        for j in T.unroll(8):
            T.evaluate(T.ptx.ex2.approx.ftz.f32(tmp[j], tmp[j]))
        for j in T.unroll(4):
            _packed_f32x2(
                "add.rn.f32x2",
                acc,
                j * 2,
                j * 2 + 1,
                acc[j * 2],
                acc[j * 2 + 1],
                tmp[j * 2],
                tmp[j * 2 + 1],
            )
    _packed_f32x2("add.rn.f32x2", acc, 0, 1, acc[0], acc[1], acc[2], acc[3])
    _packed_f32x2("add.rn.f32x2", acc, 4, 5, acc[4], acc[5], acc[6], acc[7])
    _packed_f32x2("add.rn.f32x2", acc, 0, 1, acc[0], acc[1], acc[4], acc[5])
    out[0] = acc[0] + acc[1]


@T.inline
def _row_sum_128(regs, out):
    """`_compute_row_sum` for arch 100 (utils.py:288-304): four packed
    accumulators seeded from the first eight elements, then three folds."""
    acc = T.alloc_local((8,), "float32")
    for j in T.unroll(8):
        acc[j] = regs[j]
    for it in T.unroll(1, N_BLOCK // 8):
        i: T.int32 = it * 8
        for j in T.unroll(4):
            _packed_f32x2(
                "add.rn.f32x2",
                acc,
                j * 2,
                j * 2 + 1,
                acc[j * 2],
                acc[j * 2 + 1],
                regs[i + j * 2],
                regs[i + j * 2 + 1],
            )
    _packed_f32x2("add.rn.f32x2", acc, 0, 1, acc[0], acc[1], acc[2], acc[3])
    _packed_f32x2("add.rn.f32x2", acc, 4, 5, acc[4], acc[5], acc[6], acc[7])
    _packed_f32x2("add.rn.f32x2", acc, 0, 1, acc[0], acc[1], acc[4], acc[5])
    out[0] = acc[0] + acc[1]


def _combine_int_frac_ex2(x_rounded, frac_ex2):
    """`combine_int_frac_ex2` (utils.py:1008-1030): shift the integer part into
    the exponent field and add it to the polynomial's bits.

    ``add.s32`` deliberately -- it lowers to LEA on the ALU pipe, where
    ``add.u32`` would lower to IMAD and contend with the FMA pipe the polynomial
    is already using.
    """
    xi = T.alloc_local((1,), "int32")
    fi = T.alloc_local((1,), "int32")
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.mov.b32(xi[0], x_rounded))
    T.evaluate(T.ptx.mov.b32(fi[0], frac_ex2))
    T.evaluate(T.ptx.shl.b32(xi[0], xi[0], T.uint32(23)))
    T.evaluate(T.ptx.add.s32(out[0], xi[0], fi[0]))
    return T.reinterpret("float32", out[0])


@T.inline
def _ex2_emulation_2(regs, i, j):
    """`ex2_emulation_2` (utils.py:987-1005): 2**x for a PAIR, without MUFU.

    Clamp to -127, split off the integer part by a round-down add of
    2**23 + 2**22, evaluate the fractional part with a degree-3 packed Horner,
    and reassemble by shifting the integer into the exponent field. The clamp is
    two ``max.f32`` per pair, which is where 32 of the export's 164 ``max.f32``
    live -- they are not part of the row-max tree.
    """
    cl = T.alloc_local((2,), "float32")
    T.evaluate(T.ptx.max.f32(cl[0], regs[i], T.float32(-127.0)))
    T.evaluate(T.ptx.max.f32(cl[1], regs[j], T.float32(-127.0)))
    rounded = T.alloc_local((2,), "float32")
    _packed_f32x2(
        "add.rm.f32x2",
        rounded,
        0,
        1,
        cl[0],
        cl[1],
        T.float32(_FP32_ROUND_INT),
        T.float32(_FP32_ROUND_INT),
    )
    back = T.alloc_local((2,), "float32")
    _packed_f32x2(
        "sub.rn.f32x2",
        back,
        0,
        1,
        rounded[0],
        rounded[1],
        T.float32(_FP32_ROUND_INT),
        T.float32(_FP32_ROUND_INT),
    )
    frac = T.alloc_local((2,), "float32")
    _packed_f32x2("sub.rn.f32x2", frac, 0, 1, cl[0], cl[1], back[0], back[1])
    poly = T.alloc_local((2,), "float32")
    poly[0] = T.float32(_POLY_EX2_3[3])
    poly[1] = T.float32(_POLY_EX2_3[3])
    # Horner, unrolled at trace time: the coefficients are Python floats, so a
    # traced loop over them would bind them as TIR variables.
    _packed_f32x2(
        "fma.rn.f32x2",
        poly,
        0,
        1,
        poly[0],
        poly[1],
        frac[0],
        frac[1],
        T.float32(_POLY_EX2_3[2]),
        T.float32(_POLY_EX2_3[2]),
    )
    _packed_f32x2(
        "fma.rn.f32x2",
        poly,
        0,
        1,
        poly[0],
        poly[1],
        frac[0],
        frac[1],
        T.float32(_POLY_EX2_3[1]),
        T.float32(_POLY_EX2_3[1]),
    )
    _packed_f32x2(
        "fma.rn.f32x2",
        poly,
        0,
        1,
        poly[0],
        poly[1],
        frac[0],
        frac[1],
        T.float32(_POLY_EX2_3[0]),
        T.float32(_POLY_EX2_3[0]),
    )
    regs[i] = _combine_int_frac_ex2(rounded[0], poly[0])
    regs[j] = _combine_int_frac_ex2(rounded[1], poly[1])


# ---------------------------------------------------------------------------
# NVFP4 scale addressing and dequantization -- the code with no counterpart in
# the BF16/FP8 forward.
# ---------------------------------------------------------------------------
def _scale_128x4_offset(row, col, scale_cols: int):
    """`_scale_128x4_offset` (:1197-1213): byte offset into the cuBLAS 128x4
    tiled block-scale layout.

    Pure address arithmetic, and the highest-risk expression in this port: every
    way of getting it wrong yields a plausible number rather than a fault. It is
    checked against ``quantize.nvfp4_scale_128x4_offset`` over a multi-tile grid
    before the kernel runs.

    ``row`` and ``col`` are runtime values; ``scale_cols`` is compile-time.
    """
    tiles_n = (scale_cols + SCALE_TILE_COLS - 1) // SCALE_TILE_COLS
    tile_m = udiv_i32(row, SCALE_TILE_ROWS)
    tile_n = udiv_i32(col, SCALE_TILE_COLS)
    outer = row - tile_m * SCALE_TILE_ROWS
    inner = col - tile_n * SCALE_TILE_COLS
    return (tile_m * tiles_n + tile_n) * 512 + (outer % 32) * 16 + udiv_i32(outer, 32) * 4 + inner


def _flat_kv_scale_row(token_idx, head_kv_idx, num_heads_kv):
    """`_flat_kv_scale_row` (:1320-1326). The token index arrives already offset
    by the batch's ``cu_seqlens_k`` base (:1392, :1458)."""
    return token_idx * num_heads_kv + head_kv_idx


def _load_scale_e4m3_u8(scale, offset):
    """`_load_scale_e4m3_u8` (:1236-1255): the raw E4M3 byte, for the FP8 path.

    ``ld.global.s8``, which is what the export carries even though the source's
    pointer is ``Uint8``: only the low byte survives the ``prmt`` replication
    that consumes it, so the sign extension is inert.
    """
    out = T.alloc_local((1,), "int8")
    T.evaluate(T.ptx.ld.global_.s8(out[0], scale.ptr_to([offset])))
    return out[0]


@T.inline
def _load_scale_bf16x2(scale, offset, out):
    """`_load_scale_bf16x2` (:1216-1234) -> `cvt_fp8_e4m3_to_bf16x2_replicated`
    (utils.py:830) -> `cvt_fp8x4_e4m3_bf16x4` (utils.py:540-567).

    A pure bit-manipulation chain with no float conversion instruction in it:
    mask to a byte, replicate it across a word with one ``mul.lo.s32`` by
    0x01010101, ``prmt`` the two low bytes into the high halves of a bf16x2,
    reassemble sign and mantissa by hand, and let one ``fma.rn.bf16x2`` against
    the constant 0x7b807b80 apply the E4M3 exponent bias.
    ``cvt.rn.f16x2.e4m3x2`` appears zero times in any BF16-Q build; naming it
    here would name an instruction this specialization never emits.
    """
    byte = T.alloc_local((1,), "uint8")
    T.evaluate(T.ptx.ld.global_.u8(byte[0], scale.ptr_to([offset])))
    masked = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.and_.b32(masked[0], T.cast(byte[0], "uint32"), T.uint32(0xFF)))
    packed_i = T.alloc_local((1,), "int32")
    T.evaluate(
        T.ptx.mul.lo.s32(packed_i[0], T.reinterpret("int32", masked[0]), T.int32(0x01010101))
    )
    q = T.alloc_local((1,), "uint32")
    mant = T.alloc_local((1,), "uint32")
    acc = T.alloc_local((1,), "uint32")
    packed: T.uint32 = T.reinterpret("uint32", packed_i[0])
    T.evaluate(T.ptx.prmt.b32(q[0], packed, packed, T.uint32(0x1302)))
    T.evaluate(T.ptx.and_.b32(acc[0], q[0], T.uint32(0x80008000)))
    T.evaluate(T.ptx.and_.b32(mant[0], q[0], T.uint32(0x7F007F00)))
    T.evaluate(T.ptx.shr.u32(mant[0], mant[0], T.uint32(4)))
    T.evaluate(T.ptx.or_.b32(acc[0], acc[0], mant[0]))
    T.evaluate(T.ptx.fma.rn.bf16x2(out[0], acc[0], T.uint32(0x7B807B80), T.uint32(0)))


@T.inline
def _fp4_byte(word, b, out):
    """One packed-E2M1 byte out of a 32-bit word.

    The source spells this ``mov.b32 {byte0, byte1, byte2, byte3}, $2`` and
    feeds a ``.reg .b8``. Inline asm has no 8-bit constraint, so the TIRx PTX
    table stages that register itself and takes a ``uint8`` operand; the byte is
    selected here with a shift and a mask instead of the 4-way move, which is
    the same extraction with a spelling the C boundary can carry.
    """
    out[0] = T.cast(T.bitwise_and(T.shift_right(word, T.uint32(8 * b)), T.uint32(0xFF)), "uint8")


@T.inline
def _dequant_fp4x16_to_bf16(src_words, combined_bf16x2, out_words):
    """`_dequant_fp4x16_to_bf16` (:1258-1277): sixteen E2M1 values, one block
    scale, sixteen BF16 out.

    The pin reports CUDA 12.9, so ``cvt_fp4x8_e2m1_bf16x8`` takes its FALLBACK
    (utils.py:687-692): fp4 -> f16 through ``cvt.rn.f16x2.e2m1x2``, then f16 ->
    bf16 through a ``cvt.f32.f16`` pair and one ``cvt.rn.bf16x2.f32``, then one
    ``mul.rn.bf16x2`` against the replicated block scale. The
    single-instruction ``cvt.rn.bf16x2.e2m1x2`` needs 13.2 and appears zero
    times in every export taken through this pin.
    """
    for w in T.unroll(2):
        for b in T.unroll(4):
            byte = T.alloc_local((1,), "uint8")
            _fp4_byte(src_words[w], b, byte)
            f16_pair = T.alloc_local((1,), "uint32")
            T.evaluate(T.ptx.cvt.rn.f16x2.e2m1x2(f16_pair[0], byte[0]))
            halves = T.alloc_local((2,), "uint16")
            T.evaluate(T.ptx.mov.b32(halves[0], halves[1], f16_pair[0]))
            lo = T.alloc_local((1,), "float32")
            hi = T.alloc_local((1,), "float32")
            T.evaluate(T.ptx.cvt.f32.f16(lo[0], halves[0]))
            T.evaluate(T.ptx.cvt.f32.f16(hi[0], halves[1]))
            bf16_pair = T.alloc_local((1,), "uint32")
            T.evaluate(T.ptx.cvt.rn.bf16x2.f32(bf16_pair[0], hi[0], lo[0]))
            T.evaluate(T.ptx.mul.rn.bf16x2(out_words[w * 4 + b], bf16_pair[0], combined_bf16x2))


@T.inline
def _dequant_fp4x8_scaled_e4m3(src_word, scale_e4m3, out, lo_idx, hi_idx):
    """`cvt_fp4x8_e2m1_scaled_e4m3x8` (utils.py:734-763), the fallback body.

    Eight E2M1 values scaled by one E4M3 byte and converted to E4M3. The pin's
    CUDA 12.9 rules out ``mul.e4m3x4.e2m1x4.e4m3x4.satfinite`` -- which is also
    the one instruction in this family the TIRx PTX table does not carry -- so
    the transcribed chain is the f16 one: replicate the scale byte, decode it to
    f16x2, decode four fp4 byte-pairs to f16x2, multiply, convert back to E4M3
    pairs and repack.
    """
    sf_bytes = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.prmt.b32(sf_bytes[0], T.cast(scale_e4m3, "uint32"), T.uint32(0), T.uint32(0)))
    sf_pair = T.alloc_local((2,), "uint16")
    T.evaluate(T.ptx.mov.b32(sf_pair[0], sf_pair[1], sf_bytes[0]))
    sf_f16x2 = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.cvt.rn.f16x2.e4m3x2(sf_f16x2[0], sf_pair[0]))
    e = T.alloc_local((4,), "uint16")
    for b in T.unroll(4):
        byte = T.alloc_local((1,), "uint8")
        _fp4_byte(src_word, b, byte)
        h = T.alloc_local((1,), "uint32")
        T.evaluate(T.ptx.cvt.rn.f16x2.e2m1x2(h[0], byte[0]))
        T.evaluate(T.ptx.mul.rn.f16x2(h[0], h[0], sf_f16x2[0]))
        T.evaluate(T.ptx.cvt.rn.satfinite.e4m3x2.f16x2(e[b], h[0]))
    T.evaluate(T.ptx.mov.b32(out[lo_idx], e[0], e[1]))
    T.evaluate(T.ptx.mov.b32(out[hi_idx], e[2], e[3]))


@T.inline
def bar_sync_named(bar_id, count):
    """``bar.sync <id>, <count>`` -- arrive AND wait on a named barrier."""
    T.ptx.bar.sync(T.uint32(bar_id), T.uint32(count))


@T.inline
def bar_arrive_named(bar_id, count):
    """``bar.arrive <id>, <count>`` -- arrive WITHOUT waiting.

    Which of the two a site uses is load-bearing. The softmax warpgroups leave
    the TMEM-allocator barrier arrive-only (:1124, :1191); only the MMA warp,
    which owns the deallocation, arrives and waits (:1055). Making the softmax
    exits wait parks both warpgroups behind a TMEM teardown they have no
    ordering dependence on.
    """
    T.ptx.bar.arrive(T.uint32(bar_id), T.uint32(count))


@T.inline
def _mbar_expect_tx(bar, stage, tx_bytes):
    """``mbarrier.expect_tx``: promise bytes without arriving.

    The single-shot K/V barriers take their transaction count in the prologue
    from thread 0 (:868-873) and are arrived on later by the load warp, so this
    stays separate from the arrive -- the barrier's arrival count is 1.
    """
    T.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(bar.ptr_to([stage]), T.uint32(tx_bytes))


@T.inline
def _dequant_kv_fp4(
    src,
    dst,
    scale,
    bar_tma,
    bar_ready,
    bar_id,
    count_raw,
    group_tidx,
    token_base,
    num_heads_kv,
    head_kv_idx,
    k_global,
    mma_dtype,
    fold_global,
):
    """Turn one staged NVFP4 tile into the BF16 or FP8 tile the MMA reads.

    ``_load_k_fp4_to_smem`` / ``_load_v_fp4_to_smem`` (:1338-1674). The two are
    one program: they differ only in which staging buffer, destination and scale
    tensor they name, and -- in the source -- in a destination view whose
    majorness this port carries on the PV descriptor instead (see ``s_v``).

    A strided task loop, ``unroll=1``, over the tile's scale blocks. Sixteen FP4
    values arrive per task on the BF16 path and thirty-two on the FP8 pair path,
    because pairing two scale columns lets one task cover a whole 16-byte
    shared load.

    ``fold_global`` is the BF16-only ``has_k_global_scale`` hook: the tensor
    scale multiplies the block scale before either touches a value, which the
    export counts as one extra ``mul.rn.bf16x2`` (17 against 16). Under FP8 Q it
    is absent here and lands on the S accumulator instead.
    """
    if count_raw > 0:
        bar_tma.wait(0, 0)
        if mma_dtype == "float8_e4m3":
            # Pair form: two scale columns, 32 values, one 16-byte load.
            pairs_per_row = T.meta_var(SCALE_COLS // 2)
            total_pairs = T.meta_var(N_BLOCK * pairs_per_row)
            # BATCHED, against the source's rolled `unroll=1` (:1377).
            #
            # The codegen database's prefetch-depth entry keys the depth to how
            # many warpgroups contend on the same shared memory, and both
            # warpgroups dequantize here -- which argued for keeping this
            # rolled, and it was. But that measurement came from a different
            # kernel's fp8->bf16 pass at eight iterations per thread; this is
            # fp4->e4m3 at four, with a different instruction mix. The entry's
            # own boundary says depth transfers between neither kernels nor
            # passes and that only a sweep separates the axes, so it is measured
            # here rather than inherited.
            #
            # Issuing all four shared loads first means the conversion chain --
            # eight dependent ops before the first multiply -- stops sitting on
            # the critical path of the next load.
            staged = T.alloc_local((DEQUANT_FP8_STAGED,), "uint32")
            for pre in range(DEQUANT_FP8_BATCH):
                task_pre: T.int32 = pre * DEQUANT_THREADS + group_tidx
                row_pre: T.int32 = udiv_i32(task_pre, pairs_per_row)
                pair_pre: T.int32 = task_pre - row_pre * pairs_per_row
                T.evaluate(
                    T.ptx.ld.shared.v4.b32(
                        staged[pre * 4],
                        staged[pre * 4 + 1],
                        staged[pre * 4 + 2],
                        staged[pre * 4 + 3],
                        src.ptr_to([row_pre, pair_pre * 16]),
                    )
                )
            for it in range(DEQUANT_FP8_BATCH):
                task: T.int32 = it * DEQUANT_THREADS + group_tidx
                row: T.int32 = udiv_i32(task, pairs_per_row)
                pair_col: T.int32 = task - row * pairs_per_row
                token: T.int32 = token_base + row
                scale_row: T.int32 = _flat_kv_scale_row(token, head_kv_idx, num_heads_kv)
                src_words = T.alloc_local((4,), "uint32")
                for w in range(4):
                    src_words[w] = staged[it * 4 + w]
                scale_lo = _load_scale_e4m3_u8(
                    scale, _scale_128x4_offset(scale_row, pair_col * 2, SCALE_COLS)
                )
                scale_hi = _load_scale_e4m3_u8(
                    scale, _scale_128x4_offset(scale_row, pair_col * 2 + 1, SCALE_COLS)
                )
                out_words = T.alloc_local((8,), "uint32")
                # `_dequant_fp4x32_to_fp8` (:1296-1317): the first two words take
                # the low scale column, the second two the high one.
                for w in T.unroll(2):
                    _dequant_fp4x8_scaled_e4m3(src_words[w], scale_lo, out_words, w * 2, w * 2 + 1)
                for w in T.unroll(2):
                    _dequant_fp4x8_scaled_e4m3(
                        src_words[w + 2], scale_hi, out_words, w * 2 + 4, w * 2 + 5
                    )
                for half in T.unroll(2):
                    T.evaluate(
                        T.ptx.st.shared.v4.b32(
                            dst.ptr_to([row, pair_col * 32 + half * 16]),
                            out_words[half * 4],
                            out_words[half * 4 + 1],
                            out_words[half * 4 + 2],
                            out_words[half * 4 + 3],
                        )
                    )
        else:
            total_tasks = T.meta_var(N_BLOCK * SCALE_COLS)
            # ROLLED (`unroll=1`, :1444), for the same reason as the pair arm.
            for it in T.serial(0, total_tasks // DEQUANT_THREADS, unroll=False):
                task: T.int32 = it * DEQUANT_THREADS + group_tidx
                row: T.int32 = udiv_i32(task, SCALE_COLS)
                scale_col: T.int32 = task - row * SCALE_COLS
                token: T.int32 = token_base + row
                scale_row: T.int32 = _flat_kv_scale_row(token, head_kv_idx, num_heads_kv)
                src_words = T.alloc_local((2,), "uint32")
                T.evaluate(
                    T.ptx.ld.shared.v2.b32(
                        src_words[0], src_words[1], src.ptr_to([row, scale_col * 8])
                    )
                )
                combined = T.alloc_local((1,), "uint32")
                _load_scale_bf16x2(
                    scale, _scale_128x4_offset(scale_row, scale_col, SCALE_COLS), combined
                )
                if fold_global:
                    # `cvt_f16x2_f32(g, g, BFloat16)` then `mul_bf16x2`
                    # (:1477-1483): the FP32 tensor scale is broadcast into both
                    # bf16 lanes and folded into the block scale, INSIDE the task
                    # loop -- the load is not hoisted, and the export shows one
                    # ld.global.f32 per task.
                    g = T.alloc_local((1,), "float32")
                    T.evaluate(T.ptx.ld.global_.f32(g[0], k_global.ptr_to([0])))
                    g_bf16x2 = T.alloc_local((1,), "uint32")
                    T.evaluate(T.ptx.cvt.rn.bf16x2.f32(g_bf16x2[0], g[0], g[0]))
                    T.evaluate(T.ptx.mul.rn.bf16x2(combined[0], combined[0], g_bf16x2[0]))
                out_words = T.alloc_local((8,), "uint32")
                _dequant_fp4x16_to_bf16(src_words, combined[0], out_words)
                for half in T.unroll(2):
                    T.evaluate(
                        T.ptx.st.shared.v4.b32(
                            dst.ptr_to([row, scale_col * 16 + half * 8]),
                            out_words[half * 4],
                            out_words[half * 4 + 1],
                            out_words[half * 4 + 2],
                            out_words[half * 4 + 3],
                        )
                    )
        T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
        bar_sync_named(bar_id, DEQUANT_THREADS)
        if group_tidx == 0:
            bar_ready.arrive(0)


@T.inline
def _issue_qk(s_slot, q_slot, q_desc, k_desc, kind, operand_dtype, s_col, stage_stride):
    """One QK tile: a chain of same-family ``tcgen05.mma`` over the K extent.

    The A descriptor is advanced by a compile-time 16-byte offset per Q stage
    rather than rebuilt, which is the register-resident `wrap`/`advance` walk
    the source pre-binds in PTX (:1990-1995). Only the first instruction of the
    chain clears the accumulator; the rest accumulate into it.
    """
    mma_k = T.meta_var(_MMA_K[kind])
    steps = T.meta_var(HEAD_DIM // mma_k)
    per_subtile = T.meta_var(_SWIZZLE_BYTES // (mma_k * _MMA_ELEM_BYTES[kind]))
    for ki in T.unroll(steps):
        sub = T.meta_var(ki // per_subtile)
        within = T.meta_var(ki % per_subtile)
        off = T.meta_var(sub * _SUBTILE_16B + within * _MMA_K_16B)
        if T.cuda.elect_sync():
            T.ptx[_MMA_CHAIN[kind]](
                T.cast(s_col + s_slot * stage_stride, "uint32"),
                q_desc.add_16B_offset(q_slot * _stage_16b(operand_dtype) + off),
                k_desc.add_16B_offset(off),
                T.uint32(_instr_desc(operand_dtype)),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                ki != 0,
            )


@T.inline
def _issue_pv(
    pv_slot,
    v_desc,
    kind,
    operand_dtype,
    o_col,
    o_stage_stride,
    p_col,
    p_stage_stride,
    bar_last,
    phase,
):
    """One PV tile, split around the late-P wait.

    ``split_P_arrive = 96`` means the first three quarters of the K extent are
    issued against the early-P barrier; the sequence then blocks on the
    last-split barrier before issuing the final quarter (:2044-2067). The A
    operand lives in TMEM, so its column address is passed explicitly instead of
    being read from a tensor.
    """
    mma_k = T.meta_var(_MMA_K[kind])
    steps = T.meta_var(N_BLOCK // mma_k)
    split_step = T.meta_var(SPLIT_P_ARRIVE // mma_k)
    for ki in T.unroll(steps):
        if ki == split_step:
            bar_last.wait(pv_slot, phase)
        if T.cuda.elect_sync():
            T.ptx[_MMA_CHAIN[kind]](
                T.cast(o_col + pv_slot * o_stage_stride, "uint32"),
                T.cast(p_col + pv_slot * p_stage_stride + ki * _PV_A_COL_STEP[kind], "uint32"),
                v_desc.add_16B_offset(ki * _PV_K_16B[kind]),
                T.uint32(_instr_desc(operand_dtype, trans_b=True)),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                ki != 0,
            )


@T.inline
def _tcgen05_commit(bar, stage):
    """Publish an MMA result, or release a stage the MMA warp consumed.

    Any pipe half whose arrival the MMA warp issues is signalled with
    ``tcgen05.commit`` rather than ``mbarrier.arrive``, so the arrival is
    ordered behind the MMA -- including the softmax-produced P pipes, whose
    consumer release the export also shows as a commit (:2129-2130).
    """
    if T.cuda.elect_sync():
        bar.arrive(stage)


@T.inline
def _pack_p_words(words, regs, j, pv_dtype):
    """Convert one 32-element fragment of P into packed words.

    bf16 packs two values per word with ``cvt.rn.bf16x2.f32``; fp8 packs four,
    two at a time through ``cvt.rn.satfinite.e4m3x2.f32`` and then combined.
    Element assignment needs a traced body, so this is ``@T.inline`` with
    ``T.unroll``; every index is arithmetic, so nothing here needs a Python
    loop variable.
    """
    if pv_dtype == "float8_e4m3":
        for w in T.unroll(8):
            lo = T.alloc_local((1,), "uint16")
            hi = T.alloc_local((1,), "uint16")
            T.evaluate(
                T.ptx.cvt.rn.satfinite.e4m3x2.f32(
                    lo[0], regs[j * 32 + w * 4 + 1], regs[j * 32 + w * 4]
                )
            )
            T.evaluate(
                T.ptx.cvt.rn.satfinite.e4m3x2.f32(
                    hi[0], regs[j * 32 + w * 4 + 3], regs[j * 32 + w * 4 + 2]
                )
            )
            words[j * 8 + w] = T.bitwise_or(
                T.cast(lo[0], "uint32"), T.shift_left(T.cast(hi[0], "uint32"), T.uint32(16))
            )
    else:
        for w in T.unroll(16):
            T.evaluate(
                T.ptx.cvt.rn.bf16x2.f32(
                    words[j * 16 + w], regs[j * 32 + w * 2 + 1], regs[j * 32 + w * 2]
                )
            )


_TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD_16 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
_TMEM_ST = {
    16: "tcgen05.st.sync.aligned.32x32b.x16.b32",
    8: "tcgen05.st.sync.aligned.32x32b.x8.b32",
}

# Values per 128-bit store, per partial dtype (:2772-2984).
_STORE_LANES = {"float32": 4, "bfloat16": 8, "float16": 8, "float8_e4m3": 16}


def _fake_col(partial_dtype: str, col):
    """``real_col_to_stg128{,_half,_fp8}_fake_col`` (copy_utils.py:852, :869, :888).

    ``O_partial`` is laid out in "fake column" order: the reference stores
    through this map and the combine kernel reads it back that way, so the
    permutation is part of the ABI. It exists to make the epilogue's 128-bit
    stores contiguous -- it undoes the column permutation the 16x256b fragment
    already carries, so a warp's stores land in whole sectors.

    Pure index math; ``col`` may be a runtime value, as it is in the reference.
    """
    if partial_dtype == "float32":
        nt, c16 = col // 16, col % 16
        pair = c16 // 2
        return nt * 16 + (pair % 4) * 4 + (pair // 4) * 2 + (c16 % 2)
    if partial_dtype in ("bfloat16", "float16"):
        nt, c32 = col // 32, col % 32
        return nt * 32 + ((c32 % 8) // 2) * 8 + (c32 // 8) * 2 + (c32 % 2)
    nt, c64 = col // 64, col % 64
    return nt * 64 + ((c64 % 8) // 2) * 16 + (c64 // 8) * 2 + (c64 % 2)


_KS_PER_STORE = {name: lanes // 2 for name, lanes in _STORE_LANES.items()}
_GROUPS_PER_PARITY = {name: 8 // ks for name, ks in _KS_PER_STORE.items()}


def _tmem16_store_groups(partial_dtype: str) -> list[tuple[int, int, list[int]]]:
    """Register groups of one ``16x256b.x8`` instruction, one per 128-bit store.

    That instruction hands a thread 32 registers whose tile coordinates are

        row = warp*32 + lane_base + (lane%32)//4 + 8*((r//2) % 2)
        col = col_base + (lane%4)*2 + (r%2) + 8*(r//4)

    so registers split into two row groups by ``(r//2) % 2`` -- call it the
    parity ``p``, which selects row or row+8 -- and inside a parity the columns
    run ``8k + j`` for ``k`` in 0..7 and ``j`` in 0..1. Consecutive ``k`` pairs
    map to consecutive fake columns, so one store takes ``lanes//2`` values of
    ``k`` starting at ``kb``: four registers for fp32, eight for bf16/f16,
    all sixteen for fp8.

    Returns ``(parity, kb, registers)`` per store.
    """
    ks_per_store = _STORE_LANES[partial_dtype] // 2
    return [
        (p, kb, [4 * k + 2 * p + j for k in range(kb, kb + ks_per_store) for j in (0, 1)])
        for p in (0, 1)
        for kb in range(0, 8, ks_per_store)
    ]


def ld_shared_f32(buffer, index):
    """``ld.shared.f32``."""
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ld.shared.f32(out[0], buffer.ptr_to([index])))
    return out[0]


def st_shared_f32(buffer, index, value):
    """``st.shared.f32``."""
    T.evaluate(T.ptx.st.shared.f32(buffer.ptr_to([index]), value))


@T.inline
def _store_o_partial(buf, elem_offset, vals, partial_dtype):
    """One 128-bit ``st.global.cs`` of the partial output.

    fp32 stores four lanes directly; bf16/f16 pack eight into four words; fp8
    packs sixteen into four (:2589-2656).
    """
    if partial_dtype == "float32":
        T.ptx.st.global_.cs.v4.f32(buf.ptr_to([elem_offset]), vals[0], vals[1], vals[2], vals[3])
    elif partial_dtype in ("bfloat16", "float16"):
        words = T.alloc_local((4,), "uint32")
        for w in T.unroll(4):
            if partial_dtype == "bfloat16":
                T.ptx.cvt.rn.bf16x2.f32(words[w], vals[w * 2 + 1], vals[w * 2])
            else:
                T.ptx.cvt.rn.f16x2.f32(words[w], vals[w * 2 + 1], vals[w * 2])
        T.ptx.st.global_.cs.v4.b32(
            buf.ptr_to([elem_offset]), words[0], words[1], words[2], words[3]
        )
    else:
        words = T.alloc_local((4,), "uint32")
        for w in T.unroll(4):
            lo = T.alloc_local((1,), "uint16")
            hi = T.alloc_local((1,), "uint16")
            T.ptx.cvt.rn.satfinite.e4m3x2.f32(lo[0], vals[w * 4 + 1], vals[w * 4])
            T.ptx.cvt.rn.satfinite.e4m3x2.f32(hi[0], vals[w * 4 + 3], vals[w * 4 + 2])
            words[w] = T.bitwise_or(
                T.cast(lo[0], "uint32"), T.shift_left(T.cast(hi[0], "uint32"), T.uint32(16))
            )
        T.ptx.st.global_.cs.v4.b32(
            buf.ptr_to([elem_offset]), words[0], words[1], words[2], words[3]
        )


@T.inline
def _resolve_gather4_rows(
    rows,
    meta,
    meta_slot,
    tok_base,
    qi_base,
    count_raw,
    q_batch_off,
    num_heads_kv,
    head_kv_idx,
    q_oob_m_idx,
    qheadperkv,
):
    """The four GMEM row coordinates one gather4 pulls (:1702-1763).

    The three head-group sizes differ only in how many distinct tokens the four
    rows come from: one head row each from four tokens at ``qheadperkv == 1``,
    two consecutive head rows from each of two tokens at 2, and four
    consecutive head rows from a single token at 4. A token past ``count_raw``
    resolves to an out-of-range row so the gather takes the descriptor's OOB
    fill rather than being predicated off.
    """
    if qheadperkv == 1:
        for j in range(4):
            qi = qi_base + tok_base + j
            rows[j] = q_oob_m_idx
            if qi < count_raw:
                q_idx = T.bitwise_and(ld_shared_i32(meta, meta_slot + tok_base + j), Q_IDX_MASK)
                rows[j] = (q_batch_off + q_idx) * num_heads_kv + head_kv_idx
    elif qheadperkv == 2:
        for t in range(2):
            qi = qi_base + tok_base + t
            base = q_oob_m_idx * qheadperkv
            if qi < count_raw:
                q_idx = T.bitwise_and(ld_shared_i32(meta, meta_slot + tok_base + t), Q_IDX_MASK)
                base = ((q_batch_off + q_idx) * num_heads_kv + head_kv_idx) * qheadperkv
            rows[t * 2] = base
            rows[t * 2 + 1] = base + 1
    else:
        qi = qi_base + tok_base
        base = q_oob_m_idx * qheadperkv
        if qi < count_raw:
            q_idx = T.bitwise_and(ld_shared_i32(meta, meta_slot + tok_base), Q_IDX_MASK)
            base = ((q_batch_off + q_idx) * num_heads_kv + head_kv_idx) * qheadperkv
        for j in range(4):
            rows[j] = base + j


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
@T.jit
def _kernel(
    k_h: T.handle,
    v_h: T.handle,
    k_scale_h: T.handle,
    v_scale_h: T.handle,
    k_global_scale_h: T.Optional(T.handle),
    k2q_q_indices_h: T.handle,
    k2q_qsplit_indices_h: T.handle,
    k2q_row_ptr_h: T.handle,
    scheduler_metadata_h: T.handle,
    work_count_h: T.handle,
    o_partial_h: T.handle,
    lse_partial_h: T.handle,
    lse_temperature_partial_h: T.Optional(T.handle),
    q_flat_h: T.handle,
    cu_seqlens_q_h: T.handle,
    cu_seqlens_k_h: T.handle,
    softmax_scale_log2: T.float32,
    lse_temperature_scale_log2: T.float32,
    lse_temperature_inv_scale: T.float32,
    num_kv_blocks: T.int32,
    num_heads_kv: T.int32,
    seq_len_q: T.int32,
    work_capacity: T.int32,
    total_k: T.int32,
    total_q: T.int32,
    head_q: T.int32,
    nnz: T.int32,
    total_rows: T.int32,
    num_batches: T.int32,
    topk: T.int32,
    scale_numel: T.int32,
    *,
    qheadperkv: T.constexpr,
    causal: T.constexpr,
    dtype_mode: T.constexpr,
    partial_dtype: T.constexpr,
):
    mode = T.meta_var(DTYPE_MODES[dtype_mode])
    q_dtype = T.meta_var(mode["q"])
    # `k_dtype = v_dtype = q_dtype` (:333-334): the dequantization target is Q's
    # dtype, so unlike the BF16/FP8 forward there is no separate qk/pv choice
    # and ONE `mma_kind` covers both GEMMs (:347).
    mma_dtype = T.meta_var(mode["mma"])
    qk_dtype = T.meta_var(mma_dtype)
    pv_dtype = T.meta_var(mma_dtype)
    q_ty = T.meta_var(_tirx_dtype(q_dtype))
    qk_ty = T.meta_var(_tirx_dtype(qk_dtype))
    pv_ty = T.meta_var(_tirx_dtype(pv_dtype))
    partial_ty = T.meta_var(_tirx_dtype(partial_dtype))
    qk_mma_kind = T.meta_var(_MMA_KIND[qk_dtype])
    pv_mma_kind = T.meta_var(_MMA_KIND[pv_dtype])
    q_bytes = T.meta_var(_DTYPE_BYTES[q_dtype])
    mma_bytes = T.meta_var(_DTYPE_BYTES[mma_dtype])
    # The FP8-Q-only hook: the K tensor scale cannot fold into E4M3 values, so
    # it multiplies the FP32 S accumulator instead (:2482-2491). Under BF16 Q it
    # folds into the dequantized K and this is False.
    s_hook_k_global = T.meta_var(q_dtype == "float8_e4m3" and k_global_scale_h is not None)
    fold_k_global = T.meta_var(q_dtype != "float8_e4m3" and k_global_scale_h is not None)
    # `q_load_tile` (:427-429): fp8 Q loads a full 128-wide row per token, bf16
    # Q loads two 64-wide k-subtiles.
    q_load_tile = T.meta_var(HEAD_DIM if q_bytes == 1 else K_TILE)
    q_tokens_per_group = T.meta_var(M_BLOCK // qheadperkv)

    # Two E2M1 values per byte, so the stored head-dim extent is halved.
    k = T.match_buffer(k_h, (total_k, num_heads_kv, PACKED_HEAD_DIM), "uint8", scope="global")
    v = T.match_buffer(v_h, (total_k, num_heads_kv, PACKED_HEAD_DIM), "uint8", scope="global")
    # One E4M3 byte per sixteen head-dim elements, in cuBLAS 128x4 tiled order.
    # Addressed as a flat byte array: `_scale_128x4_offset` computes the whole
    # offset, so a 2-D view would only invite indexing it the logical way.
    k_scale = T.match_buffer(k_scale_h, (scale_numel,), "uint8", scope="global")
    v_scale = T.match_buffer(v_scale_h, (scale_numel,), "uint8", scope="global")
    if k_global_scale_h is not None:
        k_global_scale = T.match_buffer(k_global_scale_h, (1,), "float32", scope="global")
    k2q_q_indices = T.match_buffer(k2q_q_indices_h, (num_heads_kv * nnz,), "int32", scope="global")
    k2q_qsplit_indices = T.match_buffer(
        k2q_qsplit_indices_h, (num_heads_kv * nnz,), "int32", scope="global"
    )
    k2q_row_ptr = T.match_buffer(
        k2q_row_ptr_h, (num_heads_kv * (total_rows + 1),), "int32", scope="global"
    )
    scheduler_metadata = T.match_buffer(
        scheduler_metadata_h, (work_capacity * WORK_FIELDS,), "int32", scope="global"
    )
    work_count = T.match_buffer(work_count_h, (1,), "int32", scope="global")
    o_partial = T.match_buffer(
        o_partial_h, (topk * total_q * head_q * HEAD_DIM,), partial_ty, scope="global"
    )
    lse_partial = T.match_buffer(
        lse_partial_h, (topk * total_q * head_q,), "float32", scope="global"
    )
    if lse_temperature_partial_h is not None:
        lse_temperature_partial = T.match_buffer(
            lse_temperature_partial_h, (topk * total_q * head_q,), "float32", scope="global"
        )
    q_flat = T.match_buffer(q_flat_h, (total_q * head_q, HEAD_DIM), q_ty, scope="global")
    cu_seqlens_q = T.match_buffer(cu_seqlens_q_h, (num_batches + 1,), "int32", scope="global")
    cu_seqlens_k = T.match_buffer(cu_seqlens_k_h, (num_batches + 1,), "int32", scope="global")

    # -----------------------------------------------------------------------
    # TMA descriptors, encoded in the launcher prologue.
    #
    # MSA hands the gather4 Q descriptor in as a `uint8[128]` kernel argument
    # built on the host (`interface.py:1682-1688`). Every TIRx kernel in this
    # repository encodes its own instead, and the descriptor is a plain rank-2
    # tiled map either way -- the gather is in the instruction, not the map --
    # so the port drops that argument and encodes all of them here.
    #
    # K and V keep the KV-head axis as a descriptor mode, which is what makes
    # their loads `cp.async.bulk.tensor.3d` rather than 2-D over a pre-permuted
    # tensor. Both boxes are one 128-byte swizzle atom wide, so a 128-wide tile
    # takes two issues.
    # -----------------------------------------------------------------------
    # K and V are packed FP4 bytes and always land in a PLAIN staging tile, so
    # neither descriptor carries a swizzle -- the dequantization reads the tile
    # back in logical order, and a swizzled map would scramble every row while
    # still passing an isolated probe. Both keep the KV-head axis as a
    # descriptor mode, which is what makes the loads
    # `cp.async.bulk.tensor.3d`. The box spans the whole 64-byte packed row, so
    # one issue covers a tile: 8192 transaction bytes, half the sibling's.
    k_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        k_map,
        "uint8",
        3,
        k.data,
        PACKED_HEAD_DIM,
        num_heads_kv,
        total_k,
        PACKED_HEAD_DIM,
        num_heads_kv * PACKED_HEAD_DIM,
        PACKED_HEAD_DIM,
        1,
        N_BLOCK,
        1,
        1,
        1,
        0,
        0,
        _L2_PROMOTION_256B,
        0,
    )
    v_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        v_map,
        "uint8",
        3,
        v.data,
        PACKED_HEAD_DIM,
        num_heads_kv,
        total_k,
        PACKED_HEAD_DIM,
        num_heads_kv * PACKED_HEAD_DIM,
        PACKED_HEAD_DIM,
        1,
        N_BLOCK,
        1,
        1,
        1,
        0,
        0,
        _L2_PROMOTION_256B,
        0,
    )

    # The TMA-Q box is one token group of `qheadperkv` rows; the gather4 box is
    # a single row, because the instruction supplies four row coordinates.
    q_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        q_map,
        q_ty,
        2,
        q_flat.data,
        HEAD_DIM,
        total_q * head_q,
        HEAD_DIM * q_bytes,
        _swizzle_elems(q_bytes),
        qheadperkv if not USE_GATHER4(qheadperkv) else 1,
        1,
        1,
        0,
        _SWIZZLE_128B,
        _L2_PROMOTION_256B,
        0,
    )

    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})

    # CUDA TRANSCRIPTION START
    block = T.cta_id([work_capacity])
    tidx = T.thread_id([NUM_THREADS])
    # `make_warp_uniform(warp_idx())` (:657) lowers to a lane-0 shfl broadcast.
    # `T.warp_id` is warp-uniform by construction, so the broadcast is redundant
    # here rather than load-bearing.
    warp_idx = T.warp_id([TOTAL_WARPS])
    lane_idx = T.lane_id([WARP_SIZE])

    # Work-item decode and the CTA-level early-out (:660-688).
    # The grid is sized by the work list's CAPACITY, so the tail CTAs retire.
    work_count_val = ld_global_i32(work_count, 0)
    cta_valid_work: T.int32 = T.cast(block < work_count_val, "int32")
    head_kv_idx = T.alloc_local((1,), "int32")
    row_linear = T.alloc_local((1,), "int32")
    work_q_begin = T.alloc_local((1,), "int32")
    work_q_count = T.alloc_local((1,), "int32")
    batch_idx = T.alloc_local((1,), "int32")
    kv_block_idx = T.alloc_local((1,), "int32")
    head_kv_idx[0] = 0
    row_linear[0] = 0
    work_q_begin[0] = 0
    work_q_count[0] = 0
    batch_idx[0] = 0
    kv_block_idx[0] = 0
    if cta_valid_work != 0:
        head_kv_idx[0] = ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 0)
        row_linear[0] = ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 1)
        work_q_begin[0] = ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 2)
        work_q_count[0] = ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 3)
        batch_idx[0] = ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 4)
        kv_block_idx[0] = ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 5)

    # -----------------------------------------------------------------------
    # Shared memory. The source's `@cute.struct SharedStorage` reads static but
    # is handed out by `SmemAllocator` from the dynamic pool; the export carries
    # `.extern .shared .align 1024 .b8 __dynamic_shmem__0[]`, so the matching
    # TIRx form is a pool allocation under `tirx.use_dyn_shared_memory`.
    # -----------------------------------------------------------------------
    pool = T.SMEMPool()
    s_k = pool.alloc_tcgen05_mma_AB((N_BLOCK, HEAD_DIM), qk_ty, swizzle_mode="auto")
    # Same shape as sK: the TMA box's fastest dim is head_dim, so the tile
    # lands (kv rows, head_dim cols) with head_dim contiguous. For the PV
    # B operand that means N = head_dim is the contiguous axis, which is what
    # "MN-major" names (:386-387).
    s_v = pool.alloc_tcgen05_mma_AB((N_BLOCK, HEAD_DIM), pv_ty, swizzle_mode="auto")
    s_q = pool.alloc_tcgen05_mma_AB((Q_STAGE * M_BLOCK, HEAD_DIM), q_ty, swizzle_mode="auto")
    # The packed FP4 staging tiles, UNCONDITIONAL here -- every specialization of
    # this kernel dequantizes (:511-512), where the sibling gates its equivalents
    # on an fp8-staging predicate. Plain buffers, matching their unswizzled
    # descriptors.
    #
    # Both are declared (N_BLOCK, PACKED_HEAD_DIM) and indexed [token, byte].
    # The source declares sVFp4 column-major, (PACKED_HEAD_DIM, N_BLOCK) stride
    # (1, PACKED_HEAD_DIM) (:425-431), but its four dequant sites all compute the
    # same linear byte `row * (head_dim // 2) + byte_col` (:1398, :1464, :1572,
    # :1639) -- column-major V puts that byte at element (byte_col, row), i.e.
    # the same token-major packed bytes K reads from (row, byte_col). The two
    # tiles differ in declared layout, not in addressing.
    s_k_fp4 = pool.alloc((N_BLOCK, PACKED_HEAD_DIM), "uint8", align=1024)
    s_v_fp4 = pool.alloc((N_BLOCK, PACKED_HEAD_DIM), "uint8", align=1024)

    s_scale = pool.alloc((O_STAGE * M_BLOCK * 2,), "float32", align=16)
    if lse_temperature_partial_h is not None:
        s_scale_temp = pool.alloc((O_STAGE * M_BLOCK,), "float32", align=16)
    s_split_idx = pool.alloc((O_STAGE * q_tokens_per_group,), "int32", align=16)
    s_q_idx = pool.alloc((O_STAGE * q_tokens_per_group,), "int32", align=16)
    s_row_meta = pool.alloc((8,), "int32", align=16)
    s_diag_q_count = pool.alloc((1,), "int32", align=16)
    s_q_load_m_idx = pool.alloc((Q_STAGE * q_tokens_per_group,), "int32", align=16)
    s_qidx_meta = pool.alloc((QIDX_META_STAGES * q_tokens_per_group,), "int32", align=16)

    bar_k = MBarrier(pool, 1)
    bar_v = MBarrier(pool, 1)
    # Unconditional, for the same reason the staging tiles are.
    bar_k_tma = MBarrier(pool, 1)
    bar_v_tma = MBarrier(pool, 1)
    bar_q_full = TMABar(pool, Q_STAGE)
    bar_q_empty = TCGen05Bar(pool, Q_STAGE)
    bar_s_full = TCGen05Bar(pool, S_STAGE)
    bar_s_empty = MBarrier(pool, S_STAGE)
    bar_p_full = MBarrier(pool, S_STAGE)
    bar_p_empty = TCGen05Bar(pool, S_STAGE)
    bar_p_last_full = MBarrier(pool, S_STAGE)
    bar_p_last_empty = TCGen05Bar(pool, S_STAGE)
    bar_o_full = TCGen05Bar(pool, O_STAGE)
    bar_o_empty = MBarrier(pool, O_STAGE)
    # Only the empty half of the stats pipe is live upstream: it is a credit on
    # `s_scale`, and nothing ever commits or waits on its full half (:2331,
    # :3019). The full half is still allocated so the word layout matches.
    bar_stats_full = MBarrier(pool, O_STAGE)
    bar_stats_empty = MBarrier(pool, O_STAGE)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    pool.commit()

    # The 128-row datapath-D accumulator takes a 2-D (m, N) shape only, so each
    # stage is its own allocation. The resulting column map is the source's:
    # S0 [0,128), S1 [128,256), O0 [256,384), O1 [384,512) (:145-161).
    tmem_pool = T.TMEMPool(pool, total_cols=TMEM_TOTAL, cta_group=1, tmem_addr=tmem_start_addr)
    tmem_s_col = T.meta_var(tmem_pool.offset)
    _tmem_s0 = tmem_pool.alloc_tcgen05_mma_D((M_BLOCK, N_BLOCK), "float32", M=M_BLOCK, cta_group=1)
    _tmem_s1 = tmem_pool.alloc_tcgen05_mma_D((M_BLOCK, N_BLOCK), "float32", M=M_BLOCK, cta_group=1)
    tmem_o_col = T.meta_var(tmem_pool.offset)
    _tmem_o0 = tmem_pool.alloc_tcgen05_mma_D((M_BLOCK, HEAD_DIM), "float32", M=M_BLOCK, cta_group=1)
    _tmem_o1 = tmem_pool.alloc_tcgen05_mma_D((M_BLOCK, HEAD_DIM), "float32", M=M_BLOCK, cta_group=1)
    tmem_stage_stride = T.meta_var(N_BLOCK)
    tmem_o_stage_stride = T.meta_var(HEAD_DIM)
    # P overlays the upper columns of each S tile (:369-371): the P store
    # destroys the tail of S, which is safe only because row_max and row_sum are
    # already out of it by then.
    p_width = T.meta_var(_DTYPE_BYTES[pv_dtype] * 8)
    tmem_s_to_p = T.meta_var(N_BLOCK - N_BLOCK * p_width // 32)
    tmem_p_col = T.meta_var(tmem_s_col + tmem_s_to_p)

    # -----------------------------------------------------------------------
    # Prologue: descriptor prefetch, thread-0 metadata publish, barrier init.
    # -----------------------------------------------------------------------
    if warp_idx == 0:
        # ONLY the Q descriptor (:690-696). K and V are never prefetched -- the
        # export carries exactly one `prefetch.tensormap`. The two Q programs
        # also differ in election: the CUTE-atom (TMA-Q) arm issues it
        # unelected, the raw-descriptor gather4 arm elects one lane.
        if USE_GATHER4(qheadperkv):
            if T.cuda.elect_sync():
                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_map)))
        else:
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_map)))

    # WARP 0, every lane, NO elect, and BEFORE the thread-0 metadata block
    # (:751-806). `mbarrier.init` is idempotent across the warp's 32 lanes, so
    # the reference does not elect, and the export shows the 24 inits inside a
    # plain `@%p bra` on the warp index with no `elect.sync`.
    if warp_idx == 0:
        for stage in T.unroll(Q_STAGE):
            T.ptx.mbarrier.init.shared.b64(bar_q_full.ptr_to([stage]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(bar_q_empty.ptr_to([stage]), T.uint32(1))
        for stage in T.unroll(S_STAGE):
            T.ptx.mbarrier.init.shared.b64(bar_s_full.ptr_to([stage]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(bar_s_empty.ptr_to([stage]), T.uint32(SOFTMAX_THREADS))
            T.ptx.mbarrier.init.shared.b64(bar_p_full.ptr_to([stage]), T.uint32(SOFTMAX_THREADS))
            T.ptx.mbarrier.init.shared.b64(bar_p_empty.ptr_to([stage]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(
                bar_p_last_full.ptr_to([stage]), T.uint32(SOFTMAX_THREADS)
            )
            T.ptx.mbarrier.init.shared.b64(bar_p_last_empty.ptr_to([stage]), T.uint32(1))
        for stage in T.unroll(O_STAGE):
            T.ptx.mbarrier.init.shared.b64(bar_o_full.ptr_to([stage]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(bar_o_empty.ptr_to([stage]), T.uint32(SOFTMAX_THREADS))
            T.ptx.mbarrier.init.shared.b64(
                bar_stats_full.ptr_to([stage]), T.uint32(SOFTMAX_THREADS)
            )
            T.ptx.mbarrier.init.shared.b64(
                bar_stats_empty.ptr_to([stage]), T.uint32(SOFTMAX_THREADS)
            )

    # The pipeline init has its OWN fence and CTA barrier (:808-809); the
    # thread-0 metadata block below closes with a second pair (:891-892). Two
    # pairs, not one: the export carries `fence.mbarrier_init` twice.
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if tidx == 0:
        base_row_start = ld_global_i32(
            k2q_row_ptr, head_kv_idx[0] * (total_rows + 1) + row_linear[0]
        )
        row_start: T.int32 = base_row_start + work_q_begin[0]
        count_raw: T.int32 = work_q_count[0]
        seqlen_k: T.int32 = ld_global_i32(cu_seqlens_k, batch_idx[0] + 1) - ld_global_i32(
            cu_seqlens_k, batch_idx[0]
        )
        kv_valid_cols: T.int32 = T.min(T.max(seqlen_k - kv_block_idx[0] * N_BLOCK, 0), N_BLOCK)
        q_batch_offset: T.int32 = ld_global_i32(cu_seqlens_q, batch_idx[0])
        k_batch_offset: T.int32 = ld_global_i32(cu_seqlens_k, batch_idx[0])
        # `seqlen_q` and therefore `causal_q_offset` are needed BEFORE the
        # second store, because the eight words go out as TWO four-word vector
        # stores (:849, :863), not eight scalars.
        seqlen_q: T.int32 = ld_global_i32(cu_seqlens_q, batch_idx[0] + 1) - q_batch_offset
        causal_q_offset: T.int32 = seqlen_k - seqlen_q
        T.evaluate(
            T.ptx.st.shared.v4.b32(
                s_row_meta.ptr_to([0]),
                T.reinterpret("uint32", batch_idx[0]),
                T.reinterpret("uint32", kv_block_idx[0]),
                T.reinterpret("uint32", row_start),
                T.reinterpret("uint32", count_raw),
            )
        )
        T.evaluate(
            T.ptx.st.shared.v4.b32(
                s_row_meta.ptr_to([4]),
                T.reinterpret("uint32", kv_valid_cols),
                T.reinterpret("uint32", q_batch_offset),
                T.reinterpret("uint32", k_batch_offset),
                T.reinterpret("uint32", causal_q_offset),
            )
        )

        # The causal diagonal split point: a 32-step binary search over the CSR
        # row, which is sorted by q_idx (:259-285, :919-935). The export keeps
        # the loop rolled with exactly one probe load in the body.
        diag_q_count = T.alloc_local((1,), "int32")
        diag_q_count[0] = 0
        if count_raw > 0 and kv_valid_cols > 0:
            q_threshold: T.int32 = (kv_block_idx[0] * N_BLOCK + kv_valid_cols) - causal_q_offset
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            lo[0] = 0
            hi[0] = count_raw
            # `_lower_bound_q_idx` (:264-286): a FIXED 32-trip loop with
            # `unroll=1` and a predicated body, not a data-dependent `while`.
            # 32 is an upper bound on any int32-sized row rather than a trip
            # count the search runs to convergence, and the export keeps it
            # rolled with exactly one probe load in the body.
            for _ in T.serial(0, 32, unroll=False):
                if lo[0] < hi[0]:
                    mid: T.int32 = udiv_i32(lo[0] + hi[0], 2)
                    probe: T.int32 = ld_global_i32(
                        k2q_q_indices, head_kv_idx[0] * nnz + row_start + mid
                    )
                    if probe < q_threshold:
                        lo[0] = mid + 1
                    else:
                        hi[0] = mid
            diag_q_count[0] = lo[0]
        st_shared_i32(s_diag_q_count, 0, diag_q_count[0])

        bar_k.init(1)
        bar_v.init(1)
        bar_k_tma.init(1)
        bar_v_tma.init(1)
        # 8192 bytes per tile: 128 tokens x 64 packed bytes, half what the same
        # tile costs unpacked.
        _mbar_expect_tx(bar_k_tma, 0, N_BLOCK * PACKED_HEAD_DIM)
        _mbar_expect_tx(bar_v_tma, 0, N_BLOCK * PACKED_HEAD_DIM)

    # The metadata block's own fence and CTA barrier (:891-892) -- the second
    # of the two pairs. It orders the sRowMeta / sDiagQCount publish and the
    # K/V transaction counts against every consumer.
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    # -----------------------------------------------------------------------
    # Role dispatch. A flat sequence of independent `if` blocks, each ANDed
    # with `cta_valid_work` -- not an if/elif chain (:943-1191).
    # -----------------------------------------------------------------------
    if warp_idx == TOTAL_WARPS - 1:
        # Warp 15 is idle and is NOT gated on cta_valid_work (:991-992).
        T.evaluate(T.ptx.setmaxnreg.dec.sync.aligned.u32(T.uint32(NUM_REGS_OTHER)))

    # -----------------------------------------------------------------------
    # ROLE: Q-load warpgroup, warps 8..11 (:1014-1047).
    # -----------------------------------------------------------------------
    if tidx >= Q_LOAD_WARP_BASE * WARP_SIZE and tidx < MMA_WARP_ID * WARP_SIZE:
        if cta_valid_work != 0:
            T.evaluate(T.ptx.setmaxnreg.dec.sync.aligned.u32(T.uint32(NUM_REGS_STORE)))
            q_row_start: T.int32 = ld_shared_i32(s_row_meta, 2)
            q_count_raw: T.int32 = ld_shared_i32(s_row_meta, 3)
            q_batch_off: T.int32 = ld_shared_i32(s_row_meta, 5)
            # Deliberately NOT gated on KV validity: a sparse entry past the
            # sequence still runs the all-masked path so its partial is neutral
            # (:1007-1009).
            if q_count_raw > 0:
                num_q_groups_load: T.int32 = uceil_div_i32(q_count_raw, q_tokens_per_group)
                warp_in_wg: T.int32 = warp_idx - Q_LOAD_WARP_BASE
                # `q_oob_m_idx = mQ_2d.shape[0] // qheadperkv` (:1855) -- one
                # past the last Q *tile*, so an absent token gathers out of
                # bounds and takes the descriptor's OOB fill. `total_q` alone
                # would be an in-range row whenever num_heads_kv > 1.
                q_oob_m_idx: T.int32 = total_q * num_heads_kv

                if not USE_GATHER4(qheadperkv):
                    for qi_group in T.serial(0, num_q_groups_load, unroll=False):
                        slot: T.int32 = qi_group % Q_STAGE
                        phase: T.int32 = udiv_i32(qi_group, Q_STAGE) & 1
                        if warp_in_wg == 0:
                            bar_q_empty.wait(slot, phase ^ 1)
                            if T.cuda.elect_sync():
                                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                    bar_q_full.ptr_to([slot]),
                                    T.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                )
                        bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                        load_meta_slot: T.int32 = slot * q_tokens_per_group
                        qidx_meta_slot: T.int32 = (
                            qi_group & (QIDX_META_STAGES - 1)
                        ) * q_tokens_per_group
                        # One warp's low lanes publish the whole group: with
                        # qheadperkv >= 8 a group is at most 16 tokens
                        # (:1883-1898).
                        if warp_in_wg == 0 and lane_idx < q_tokens_per_group:
                            qi: T.int32 = qi_group * q_tokens_per_group + lane_idx
                            if qi < q_count_raw:
                                word: T.int32 = ld_global_i32(
                                    k2q_qsplit_indices, head_kv_idx[0] * nnz + q_row_start + qi
                                )
                                st_shared_i32(s_qidx_meta, qidx_meta_slot + lane_idx, word)
                                st_shared_i32(
                                    s_q_load_m_idx,
                                    load_meta_slot + lane_idx,
                                    (q_batch_off + T.bitwise_and(word, Q_IDX_MASK)) * num_heads_kv
                                    + head_kv_idx[0],
                                )
                            else:
                                st_shared_i32(s_qidx_meta, qidx_meta_slot + lane_idx, 0)
                                st_shared_i32(
                                    s_q_load_m_idx, load_meta_slot + lane_idx, q_oob_m_idx
                                )
                        bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                        for qi_slot in T.unroll(TOKENS_PER_WARP(qheadperkv)):
                            tok: T.int32 = warp_in_wg * TOKENS_PER_WARP(qheadperkv) + qi_slot
                            if tok < q_tokens_per_group:
                                m_tile: T.int32 = ld_shared_i32(
                                    s_q_load_m_idx, load_meta_slot + tok
                                )
                                if T.cuda.elect_sync():
                                    for ks in T.unroll(Q_SUBTILES(q_bytes)):
                                        T.evaluate(
                                            T.ptx[_TMA_G2S_2D_CACHE](
                                                T.ptr_byte_offset(
                                                    s_q.ptr_to([0, 0]),
                                                    slot * M_BLOCK * HEAD_DIM * q_bytes
                                                    + ks * M_BLOCK * q_load_tile * q_bytes
                                                    + tok * qheadperkv * q_load_tile * q_bytes,
                                                    q_ty,
                                                ),
                                                T.address_of(q_map),
                                                T.int32(ks * q_load_tile),
                                                m_tile * qheadperkv,
                                                T.cuda.cvta_generic_to_shared(
                                                    bar_q_full.ptr_to([slot])
                                                ),
                                                _TMA_NO_POLICY,
                                            )
                                        )

                    if warp_in_wg == 0:
                        # One acquire past the end leaves the ring's empty half
                        # in the state the next work item expects (:1930-1935).
                        bar_q_empty.wait(
                            num_q_groups_load % Q_STAGE,
                            (udiv_i32(num_q_groups_load, Q_STAGE) & 1) ^ 1,
                        )
                        if T.cuda.elect_sync():
                            T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                bar_q_full.ptr_to([num_q_groups_load % Q_STAGE]),
                                T.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                            )
                else:
                    # ---- gather4 Q path, qheadperkv in {1, 2, 4} (:1624-1817) ----
                    # Each gather4 pulls four GMEM rows into one box, so a
                    # 128-row Q tile takes 8 gathers per warp.
                    gathers_per_warp = T.meta_var(M_BLOCK // (NUM_Q_LOAD_WARPS * 4))
                    tokens_per_gather4 = T.meta_var(4 // qheadperkv)
                    meta_iters = T.meta_var(
                        (q_tokens_per_group + NUM_Q_LOAD_WARPS * WARP_SIZE - 1)
                        // (NUM_Q_LOAD_WARPS * WARP_SIZE)
                    )
                    for qi_group in T.serial(0, num_q_groups_load, unroll=False):
                        slot: T.int32 = qi_group % Q_STAGE
                        phase: T.int32 = udiv_i32(qi_group, Q_STAGE) & 1
                        if warp_in_wg == 0:
                            bar_q_empty.wait(slot, phase ^ 1)
                            if T.cuda.elect_sync():
                                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                    bar_q_full.ptr_to([slot]),
                                    T.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                )
                        bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                        qidx_meta_slot: T.int32 = (
                            T.bitwise_and(qi_group, QIDX_META_STAGES - 1) * q_tokens_per_group
                        )
                        # This path's groups hold 32, 64 or 128 tokens, so the
                        # publish takes several sweeps of the whole warpgroup
                        # (:1671-1690).
                        for meta_iter in T.unroll(meta_iters):
                            tok_g4: T.int32 = (
                                meta_iter * NUM_Q_LOAD_WARPS + warp_in_wg
                            ) * WARP_SIZE + lane_idx
                            if tok_g4 < q_tokens_per_group:
                                qi_g4: T.int32 = qi_group * q_tokens_per_group + tok_g4
                                if qi_g4 < q_count_raw:
                                    st_shared_i32(
                                        s_qidx_meta,
                                        qidx_meta_slot + tok_g4,
                                        ld_global_i32(
                                            k2q_qsplit_indices,
                                            head_kv_idx[0] * nnz + q_row_start + qi_g4,
                                        ),
                                    )
                                else:
                                    st_shared_i32(s_qidx_meta, qidx_meta_slot + tok_g4, 0)
                        bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                        if T.cuda.elect_sync():
                            for gather_slot in T.unroll(gathers_per_warp):
                                gather_idx: T.int32 = gather_slot * NUM_Q_LOAD_WARPS + warp_in_wg
                                tok_base: T.int32 = gather_idx * tokens_per_gather4
                                rows = T.alloc_local((4,), "int32")
                                _resolve_gather4_rows(
                                    rows,
                                    s_qidx_meta,
                                    qidx_meta_slot,
                                    tok_base,
                                    qi_base=qi_group * q_tokens_per_group,
                                    count_raw=q_count_raw,
                                    q_batch_off=q_batch_off,
                                    num_heads_kv=num_heads_kv,
                                    head_kv_idx=head_kv_idx[0],
                                    q_oob_m_idx=q_oob_m_idx,
                                    qheadperkv=qheadperkv,
                                )
                                for ks in T.unroll(Q_SUBTILES(q_bytes)):
                                    if ks + 1 < Q_SUBTILES(q_bytes):
                                        T.evaluate(
                                            T.ptx[_TMA_GATHER4_PREFETCH](
                                                T.address_of(q_map),
                                                T.int32((ks + 1) * q_load_tile),
                                                rows[0],
                                                rows[1],
                                                rows[2],
                                                rows[3],
                                                _GATHER4_Q_CACHE_HINT,
                                            )
                                        )
                                    T.evaluate(
                                        T.ptx[_TMA_GATHER4_2D_CACHE](
                                            T.ptr_byte_offset(
                                                s_q.ptr_to([0, 0]),
                                                slot * M_BLOCK * HEAD_DIM * q_bytes
                                                + ks * M_BLOCK * q_load_tile * q_bytes
                                                + gather_idx * 4 * q_load_tile * q_bytes,
                                                q_ty,
                                            ),
                                            T.address_of(q_map),
                                            T.int32(ks * q_load_tile),
                                            rows[0],
                                            rows[1],
                                            rows[2],
                                            rows[3],
                                            T.cuda.cvta_generic_to_shared(
                                                bar_q_full.ptr_to([slot])
                                            ),
                                            _GATHER4_Q_CACHE_HINT,
                                        )
                                    )
                        bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                    if warp_in_wg == 0:
                        bar_q_empty.wait(
                            num_q_groups_load % Q_STAGE,
                            (udiv_i32(num_q_groups_load, Q_STAGE) & 1) ^ 1,
                        )
                        if T.cuda.elect_sync():
                            T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                bar_q_full.ptr_to([num_q_groups_load % Q_STAGE]),
                                T.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                            )

    # -----------------------------------------------------------------------
    # ROLE: KV load, warps 13 and 14. One TMA each; there is no KV ring
    # (:1064-1086, :1676-1777).
    # -----------------------------------------------------------------------
    if warp_idx >= KV_LOAD_WARP_BASE and warp_idx < KV_LOAD_WARP_BASE + NUM_KV_LOAD_WARPS:
        if cta_valid_work != 0:
            T.evaluate(T.ptx.setmaxnreg.dec.sync.aligned.u32(T.uint32(NUM_REGS_OTHER)))
            kv_block_load: T.int32 = ld_shared_i32(s_row_meta, 1)
            k_batch_off: T.int32 = ld_shared_i32(s_row_meta, 6)
            kv_has_work: T.int32 = T.cast(ld_shared_i32(s_row_meta, 3) > 0, "int32")
            if kv_has_work != 0:
                kv_row_start: T.int32 = k_batch_off + kv_block_load * N_BLOCK
                if warp_idx == KV_LOAD_WARP_BASE:
                    # ONE issue per tile, not two: the packed row is 64 bytes,
                    # inside a single 128-byte box. The copy and the arrive sit
                    # in two separate `elect.sync` regions, as the export shows
                    # (:1734-1740).
                    if T.cuda.elect_sync():
                        T.evaluate(
                            T.ptx[_TMA_G2S_3D_CACHE](
                                s_k_fp4.ptr_to([0, 0]),
                                T.address_of(k_map),
                                T.int32(0),
                                head_kv_idx[0],
                                kv_row_start,
                                T.cuda.cvta_generic_to_shared(bar_k_tma.ptr_to([0])),
                                _TMA_NO_POLICY,
                            )
                        )
                    if T.cuda.elect_sync():
                        T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                            bar_k_tma.ptr_to([0]), T.uint32(1)
                        )
                if warp_idx == KV_LOAD_WARP_BASE + 1:
                    if T.cuda.elect_sync():
                        T.evaluate(
                            T.ptx[_TMA_G2S_3D_CACHE](
                                s_v_fp4.ptr_to([0, 0]),
                                T.address_of(v_map),
                                T.int32(0),
                                head_kv_idx[0],
                                kv_row_start,
                                T.cuda.cvta_generic_to_shared(bar_v_tma.ptr_to([0])),
                                _TMA_NO_POLICY,
                            )
                        )
                    if T.cuda.elect_sync():
                        T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                            bar_v_tma.ptr_to([0]), T.uint32(1)
                        )
                # Unconditional here: both tiles always need dequantizing, so
                # the two load warps always close on this barrier (:1777).
                bar_sync_named(BAR_KV_LOAD, WARP_SIZE * NUM_KV_LOAD_WARPS)

    # -----------------------------------------------------------------------
    # ROLE: the single MMA-issue warp, warp 12; also the TMEM allocator warp
    # (:1035-1057, :2189-2436).
    # -----------------------------------------------------------------------
    if warp_idx == MMA_WARP_ID:
        if cta_valid_work != 0:
            T.evaluate(T.ptx.setmaxnreg.dec.sync.aligned.u32(T.uint32(NUM_REGS_OTHER)))
            T.evaluate(
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_start_addr[0]), T.uint32(TMEM_TOTAL)
                )
            )
            # The retrieve barrier spans both softmax warpgroups plus this warp
            # (:1041-1042). On the fp8 paths that also makes the first QK wait for
            # the shared-memory dequantization, without a separate edge.
            bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

            mma_count_raw: T.int32 = ld_shared_i32(s_row_meta, 3)
            if mma_count_raw > 0:
                num_q_groups_mma: T.int32 = uceil_div_i32(mma_count_raw, q_tokens_per_group)

                # Operand descriptors. The Q ring is walked by adding a
                # compile-time 16-byte offset to the A descriptor rather than
                # rebuilding it, which is what the source's `wrap`/`advance`
                # pre-bound partials do in PTX registers (:1990-1995).
                q_desc = SmemDescriptor()
                q_desc.init(s_q.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                q_desc.make_lo_uniform()
                k_desc = SmemDescriptor()
                k_desc.init(s_k.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                k_desc.make_lo_uniform()
                v_desc = SmemDescriptor()
                v_desc.init(s_v.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                v_desc.make_lo_uniform()

                # The DEQUANT-ready barrier, not the TMA one: the MMA reads
                # `s_k`, which a softmax warpgroup fills (:2321).
                bar_k.wait(0, 0)

                # Issue order (:2352-2402): Q0K, Q1K, P0V, Q2K, P1V, ...
                # QK(qi) consumes S slot qi&1; PV(qi-2) frees that slot before
                # QK(qi) reuses it, so a 2-slot S ring is safe.
                bar_q_full.wait(0, 0)
                bar_s_empty.wait(0, 1)
                _issue_qk(
                    0, 0, q_desc, k_desc, qk_mma_kind, qk_dtype, tmem_s_col, tmem_stage_stride
                )
                _tcgen05_commit(bar_s_full, 0)
                _tcgen05_commit(bar_q_empty, 0)

                if num_q_groups_mma > 1:
                    bar_q_full.wait(1, 0)
                    bar_s_empty.wait(1, 1)
                    _issue_qk(
                        1, 1, q_desc, k_desc, qk_mma_kind, qk_dtype, tmem_s_col, tmem_stage_stride
                    )
                    _tcgen05_commit(bar_s_full, 1)
                    _tcgen05_commit(bar_q_empty, 1)

                # V is waited only after the two prologue QKs, so its TMA
                # overlaps them (:2094).
                bar_v.wait(0, 0)

                for qi in T.serial(2, num_q_groups_mma, unroll=False):
                    pv_qi: T.int32 = qi - 2
                    pv_slot: T.int32 = T.bitwise_and(pv_qi, 1)
                    pv_phase: T.int32 = udiv_i32(pv_qi, 2) & 1
                    bar_p_full.wait(pv_slot, pv_phase)
                    bar_o_empty.wait(pv_slot, pv_phase ^ 1)
                    _issue_pv(
                        pv_slot,
                        v_desc,
                        pv_mma_kind,
                        pv_dtype,
                        tmem_o_col,
                        tmem_o_stage_stride,
                        tmem_p_col,
                        tmem_stage_stride,
                        bar_p_last_full,
                        pv_phase,
                    )
                    _tcgen05_commit(bar_o_full, pv_slot)
                    _tcgen05_commit(bar_p_last_empty, pv_slot)
                    _tcgen05_commit(bar_p_empty, pv_slot)

                    q_slot: T.int32 = qi % Q_STAGE
                    q_phase: T.int32 = udiv_i32(qi, Q_STAGE) & 1
                    s_slot: T.int32 = T.bitwise_and(qi, 1)
                    s_phase: T.int32 = udiv_i32(qi, 2) & 1
                    bar_q_full.wait(q_slot, q_phase)
                    bar_s_empty.wait(s_slot, s_phase ^ 1)
                    # The S-slot test is a runtime branch that duplicates the
                    # whole 8-instruction chain in the emitted code; the Q-slot
                    # choice collapses into an address select (export: two
                    # chains behind `@%p bra`, one PV chain).
                    if s_slot == 0:
                        _issue_qk(
                            0,
                            q_slot,
                            q_desc,
                            k_desc,
                            qk_mma_kind,
                            qk_dtype,
                            tmem_s_col,
                            tmem_stage_stride,
                        )
                    else:
                        _issue_qk(
                            1,
                            q_slot,
                            q_desc,
                            k_desc,
                            qk_mma_kind,
                            qk_dtype,
                            tmem_s_col,
                            tmem_stage_stride,
                        )
                    _tcgen05_commit(bar_s_full, s_slot)
                    _tcgen05_commit(bar_q_empty, q_slot)

                # Drain the last one or two PV tiles (:2152-2183).
                drain_begin: T.int32 = T.if_then_else(
                    num_q_groups_mma == 1, 0, num_q_groups_mma - 2
                )
                for pv_qi2 in T.serial(drain_begin, num_q_groups_mma, unroll=False):
                    pv_slot2: T.int32 = T.bitwise_and(pv_qi2, 1)
                    pv_phase2: T.int32 = udiv_i32(pv_qi2, 2) & 1
                    bar_p_full.wait(pv_slot2, pv_phase2)
                    bar_o_empty.wait(pv_slot2, pv_phase2 ^ 1)
                    _issue_pv(
                        pv_slot2,
                        v_desc,
                        pv_mma_kind,
                        pv_dtype,
                        tmem_o_col,
                        tmem_o_stage_stride,
                        tmem_p_col,
                        tmem_stage_stride,
                        bar_p_last_full,
                        pv_phase2,
                    )
                    _tcgen05_commit(bar_o_full, pv_slot2)
                    _tcgen05_commit(bar_p_last_empty, pv_slot2)
                    _tcgen05_commit(bar_p_empty, pv_slot2)

            T.evaluate(T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned())
            bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))
            # The contract forbids a native load on a shared buffer, so the
            # allocated base comes back through an explicit ld.shared.
            tmem_base = T.alloc_local((1,), "uint32")
            T.evaluate(T.ptx.ld.shared.u32(tmem_base[0], tmem_start_addr.ptr_to([0])))
            T.evaluate(
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_base[0], T.uint32(TMEM_TOTAL)
                )
            )
            # No matching griddepcontrol.wait anywhere in this kernel, so no
            # launch attribute is needed -- only a waiting kernel needs one.
            T.evaluate(T.ptx.griddepcontrol.launch_dependents())

    # -----------------------------------------------------------------------
    # ROLE: softmax warpgroups 0 and 1, with the epilogue fused in
    # (:1113-1191, :2437-2616, :2927-3305).
    #
    # `stage` is a compile-time constant, so this body is emitted once per
    # warpgroup -- which is what the export shows, every single-site operation
    # inside it appearing exactly twice.
    # -----------------------------------------------------------------------
    @T.inline
    def epilogue_step(
        stage, qi_group, qidx_meta_slot, group_tidx, count_raw, q_batch_off, scale_log2
    ):
        """Scale O by 1/row_sum and store it, then store LSE (:2927-3305).

        Two rolled column passes over the accumulator, each read with
        ``16x256b.x8`` and stored before the next pass runs. That read shape is
        what makes the stores contiguous: lanes 0-3 of a warp hold four columns
        of ONE row, so a store covers eight rows in whole 64-byte runs. The
        address goes through ``_fake_col`` -- ``O_partial`` is written in
        fake-column order, the combine kernel reads it back that way, and the
        permutation undoes the one the fragment already carries.
        """
        slot: T.int32 = T.bitwise_and(qi_group, 1)
        phase: T.int32 = udiv_i32(qi_group, 2) & 1
        bar_o_full.wait(slot, phase)

        # Decode the packed qsplit once per group into the 2-deep caches; every
        # per-store read below comes out of these, never out of s_qidx_meta.
        if group_tidx < q_tokens_per_group:
            word: T.int32 = ld_shared_i32(s_qidx_meta, qidx_meta_slot + group_tidx)
            st_shared_i32(
                s_q_idx, slot * q_tokens_per_group + group_tidx, T.bitwise_and(word, Q_IDX_MASK)
            )
            st_shared_i32(
                s_split_idx,
                slot * q_tokens_per_group + group_tidx,
                T.bitwise_and(T.shift_right(word, SLOT_SHIFT), SLOT_MASK),
            )
        bar_sync_named(BAR_EPILOGUE + stage, SOFTMAX_THREADS)

        # Four `16x256b.x8` reads tile the 128x128 accumulator: the lane half
        # picks rows `lane_base + 0..15` of each warp's 32, the column half
        # picks 64 of the 128 columns. Each hands the thread 32 registers.
        #
        # This shape, not `32x32b`, is what makes the stores coalesce. Here
        # lanes 0-3 of a warp hold four columns of ONE row, so a store
        # instruction covers eight rows in contiguous 64-byte runs -- 16
        # sectors for 512 bytes, the minimum. Under `32x32b` the thread IS the
        # row, so every lane of a store lands on a different row 512 bytes
        # away: the same bytes at twice the sectors.
        warp_in_wg: T.int32 = group_tidx // WARP_SIZE
        lane_in_warp: T.int32 = group_tidx - warp_in_wg * WARP_SIZE
        row_of_lane: T.int32 = warp_in_wg * 32 + lane_in_warp // 4
        col_of_lane: T.int32 = (lane_in_warp % 4) * 2

        # One reciprocal per (lane_base, parity) for the whole epilogue -- four
        # per thread -- instead of one per 128-bit store.
        row_scale_cache = T.alloc_local((4,), "float32")
        for lb in range(2):
            for par in range(2):
                rs_row: T.int32 = row_of_lane + lb * 16 + 8 * par
                rs_sum: T.float32 = ld_shared_f32(s_scale, slot * M_BLOCK * 2 + rs_row)
                rs_safe: T.float32 = T.if_then_else(
                    T.Or(rs_sum == T.float32(0.0), rs_sum != rs_sum), T.float32(1.0), rs_sum
                )
                T.evaluate(T.ptx.rcp.approx.ftz.f32(row_scale_cache[lb * 2 + par], rs_safe))

        # `lane_base` stays a Python int through the inline call, keeping the
        # TMEM address's lane field constant; `col_base` is the rolled pass's
        # runtime column half.
        @T.inline
        def load_o_pass(o_regs, lane_base, col_base):
            T.evaluate(
                T.ptx[_TMEM_LD_16](
                    *[o_regs[i] for i in range(32)],
                    T.cast(tmem_o_col + slot * tmem_o_stage_stride + col_base, "uint32")
                    + T.uint32(lane_base << 16),
                )
            )

        @T.inline
        def store_o_pass(o_regs, lane_base, col_base):
            # Derived arithmetically rather than looked up: the loop variable is
            # a TIR var here, and the unroller folds these to constants. Indexing
            # a Python list with it would fail while tracing, before that.
            for grp in range(2 * _GROUPS_PER_PARITY[partial_dtype]):
                parity = grp // _GROUPS_PER_PARITY[partial_dtype]
                kb = (grp % _GROUPS_PER_PARITY[partial_dtype]) * _KS_PER_STORE[partial_dtype]
                regs = [
                    4 * (kb + t) + 2 * parity + j
                    for t in range(_KS_PER_STORE[partial_dtype])
                    for j in (0, 1)
                ]
                # `tok` is the token within the group, `row_in_tok` the head
                # inside that token.
                row: T.int32 = row_of_lane + lane_base + 8 * parity
                tok: T.int32 = udiv_i32(row, qheadperkv)
                row_in_tok: T.int32 = row - tok * qheadperkv
                qi: T.int32 = qi_group * q_tokens_per_group + tok
                if qi < count_raw:
                    # Re-read per store, as the reference does: nothing here is
                    # hoisted out of the column loop, the reciprocal included
                    # (:2785-2800, measured one of each per 128-bit store).
                    q_idx_e: T.int32 = ld_shared_i32(s_q_idx, slot * q_tokens_per_group + tok)
                    split_e: T.int32 = ld_shared_i32(s_split_idx, slot * q_tokens_per_group + tok)
                    # The row scale comes from a register, not from shared.
                    #
                    # `row` depends only on `lane_base` and `parity`, never on
                    # the column group or the column pass, so each thread needs
                    # just FOUR of these for the whole epilogue. Read per store
                    # they were the kernel's worst shared access by a wide
                    # margin: ablating them removed 493,694 shared-load
                    # instructions and 1,129,247 bank conflicts -- 2.29 conflicts
                    # apiece, against a total excess over the source of 763,220.
                    # Hoisting inside a pass does not help; nvcc already CSEs
                    # that far. They have to be lifted clear of all four passes.
                    row_scale = T.alloc_local((1,), "float32")
                    row_scale[0] = row_scale_cache[(lane_base // 16) * 2 + parity]
                    q_abs_e: T.int32 = q_batch_off + q_idx_e
                    flat_row: T.int64 = (
                        T.cast(split_e, "int64")
                        * T.cast(total_q, "int64")
                        * T.cast(head_q, "int64")
                        + T.cast(q_abs_e, "int64") * T.cast(head_q, "int64")
                        + T.cast(head_kv_idx[0] * qheadperkv + row_in_tok, "int64")
                    )
                    # The group's first register sits at column
                    # `col_base + (lane%4)*2 + 8*kb`; the fake-column map turns
                    # that into the contiguous output address.
                    store_col: T.int32 = _fake_col(partial_dtype, col_base + col_of_lane + 8 * kb)
                    scaled = T.alloc_local((_STORE_LANES[partial_dtype],), "float32")
                    _scale_gather(scaled, o_regs, regs, row_scale[0])
                    _store_o_partial(
                        o_partial,
                        flat_row * T.int64(HEAD_DIM) + T.cast(store_col, "int64"),
                        scaled,
                        partial_dtype,
                    )

        # TWO column passes, ROLLED (`unroll=1`, :3004): each pass reads its own
        # half of the accumulator and stores it before the next pass runs. The
        # loop variable is a TIR value, so the two `16x256b.x8` reads inside a
        # pass are the two row halves, reached through the LANE field of the
        # TMEM address.
        #
        # The TMEM-load fence sits AFTER the whole store loop (:3270), not
        # between the reads and the stores: it releases the O TMEM slot, and it
        # is the only TMEM-load wait in the kernel.
        # Two forms, chosen by a compile-time predicate rather than globally.
        #
        # Issuing all four TMEM reads before the single `wait::ld` lets them
        # overlap instead of draining one column pass at a time, and it is worth
        # a lot where it fits: measured against the rolled form, +0.075 on
        # varlen_b3_s4096_qh8, +0.026 on varlen_b3_s8192_qh4, +0.025 on
        # ring48k_bf16q, +0.010 on qh1_s8192.
        #
        # It does not fit everywhere. Holding four 32-register fragments live at
        # once instead of two crosses a ceiling on the FP8-Q TMA-Q dispatches,
        # where the same change measured -0.034 on ring48k_fp8q and -0.134 on
        # fp8q_partial_fp8. So the depth is bound to the dispatch, not to the
        # pass: FP8 Q reaching the epilogue through the TMA-Q program keeps the
        # source's rolled two-pass form (:3004); everything else overlaps.
        #
        # The mechanism behind the FP8-Q/TMA-Q boundary specifically is not
        # established -- the split is measured, not derived, and the full
        # required matrix is what separates the two groups.
        overlap_o = T.meta_var(USE_GATHER4(qheadperkv) or q_dtype != "float8_e4m3")
        if overlap_o:
            o_frag_0 = T.alloc_tcgen05_ldst_frag("16x256b", (M_BLOCK // 2, 64), "float32")
            o_frag_1 = T.alloc_tcgen05_ldst_frag("16x256b", (M_BLOCK // 2, 64), "float32")
            o_frag_2 = T.alloc_tcgen05_ldst_frag("16x256b", (M_BLOCK // 2, 64), "float32")
            o_frag_3 = T.alloc_tcgen05_ldst_frag("16x256b", (M_BLOCK // 2, 64), "float32")
            o_regs_0 = o_frag_0.local()
            o_regs_1 = o_frag_1.local()
            o_regs_2 = o_frag_2.local()
            o_regs_3 = o_frag_3.local()
            load_o_pass(o_regs_0, 0, 0)
            load_o_pass(o_regs_1, 0, 64)
            load_o_pass(o_regs_2, 16, 0)
            load_o_pass(o_regs_3, 16, 64)
            T.evaluate(T.ptx.tcgen05.wait__ld.sync.aligned())
            store_o_pass(o_regs_0, 0, 0)
            store_o_pass(o_regs_1, 0, 64)
            store_o_pass(o_regs_2, 16, 0)
            store_o_pass(o_regs_3, 16, 64)
        else:
            o_frag_a = T.alloc_tcgen05_ldst_frag("16x256b", (M_BLOCK // 2, 64), "float32")
            o_frag_b = T.alloc_tcgen05_ldst_frag("16x256b", (M_BLOCK // 2, 64), "float32")
            o_regs_a = o_frag_a.local()
            o_regs_b = o_frag_b.local()
            for col_pass in T.serial(0, 2, unroll=False):
                col_base_rt: T.int32 = col_pass * 64
                load_o_pass(o_regs_a, 0, col_base_rt)
                load_o_pass(o_regs_b, 16, col_base_rt)
                store_o_pass(o_regs_a, 0, col_base_rt)
                store_o_pass(o_regs_b, 16, col_base_rt)
            T.evaluate(T.ptx.tcgen05.wait__ld.sync.aligned())

        # LSE: one row per thread (:2987-3016).
        tok_l: T.int32 = udiv_i32(group_tidx, qheadperkv)
        h_local: T.int32 = group_tidx - tok_l * qheadperkv
        if qi_group * q_tokens_per_group + tok_l < count_raw:
            row_sum_l: T.float32 = ld_shared_f32(s_scale, slot * M_BLOCK * 2 + group_tidx)
            row_max_l: T.float32 = ld_shared_f32(s_scale, slot * M_BLOCK * 2 + M_BLOCK + group_tidx)
            lg = T.alloc_local((1,), "float32")
            T.evaluate(T.ptx.lg2.approx.ftz.f32(lg[0], row_sum_l))
            lse_val: T.float32 = T.if_then_else(
                T.Or(row_sum_l == T.float32(0.0), row_sum_l != row_sum_l),
                -T.infinity("float32"),
                (row_max_l * scale_log2 + lg[0]) * T.float32(LN_2),
            )
            q_idx_l: T.int32 = ld_shared_i32(s_q_idx, slot * q_tokens_per_group + tok_l)
            split_l: T.int32 = ld_shared_i32(s_split_idx, slot * q_tokens_per_group + tok_l)
            h_abs: T.int32 = head_kv_idx[0] * qheadperkv + h_local
            lse_flat: T.int64 = (
                T.cast(split_l, "int64") * T.cast(total_q, "int64") * T.cast(head_q, "int64")
                + T.cast(q_batch_off + q_idx_l, "int64") * T.cast(head_q, "int64")
                + T.cast(h_abs, "int64")
            )
            T.evaluate(T.ptx.st.global_.f32(lse_partial.ptr_to([lse_flat]), lse_val))
            if lse_temperature_partial_h is not None:
                temp_sum: T.float32 = ld_shared_f32(s_scale_temp, slot * M_BLOCK + group_tidx)
                lgt = T.alloc_local((1,), "float32")
                T.evaluate(T.ptx.lg2.approx.ftz.f32(lgt[0], temp_sum))
                lse_t: T.float32 = T.if_then_else(
                    T.Or(temp_sum == T.float32(0.0), temp_sum != temp_sum),
                    NEG_INF,
                    (row_max_l * lse_temperature_scale_log2 + lgt[0]) * T.float32(LN_2),
                )
                T.evaluate(T.ptx.st.global_.f32(lse_temperature_partial.ptr_to([lse_flat]), lse_t))

        bar_sync_named(BAR_EPILOGUE + stage, SOFTMAX_THREADS)
        bar_stats_empty.arrive(slot)
        bar_o_empty.arrive(slot)

    @T.inline
    def softmax_warpgroup(stage):
        group_tidx: T.int32 = tidx - stage * SOFTMAX_THREADS
        kv_block_sm: T.int32 = ld_shared_i32(s_row_meta, 1)
        count_raw_sm: T.int32 = ld_shared_i32(s_row_meta, 3)
        kv_valid_cols: T.int32 = ld_shared_i32(s_row_meta, 4)
        q_batch_off_sm: T.int32 = ld_shared_i32(s_row_meta, 5)
        causal_q_off: T.int32 = ld_shared_i32(s_row_meta, 7)
        diag_q_count_sm: T.int32 = ld_shared_i32(s_diag_q_count, 0)

        # NVFP4 dequantization (:1779-1865). Unconditional and not in the load
        # warps: the TMA lands packed FP4 in the staging tile and a whole
        # softmax warpgroup converts it into the tile the MMA reads, before
        # entering its own loop. WG0 takes K, WG1 takes V.
        k_batch_off_sm: T.int32 = ld_shared_i32(s_row_meta, 6)
        token_base_sm: T.int32 = k_batch_off_sm + kv_block_sm * N_BLOCK
        if stage == 0:
            _dequant_kv_fp4(
                s_k_fp4,
                s_k,
                k_scale,
                bar_k_tma,
                bar_k,
                BAR_KV_DEQUANT_K,
                count_raw_sm,
                group_tidx,
                token_base_sm,
                num_heads_kv,
                head_kv_idx[0],
                k_global_scale if k_global_scale_h is not None else None,
                mma_dtype,
                fold_k_global,
            )
        if stage == 1:
            # V never carries a tensor scale into this kernel: the interface
            # pins `has_v_global_scale` off and applies it in the combine
            # kernel, so the fold is False on both Q dtypes.
            _dequant_kv_fp4(
                s_v_fp4,
                s_v,
                v_scale,
                bar_v_tma,
                bar_v,
                BAR_KV_DEQUANT_V,
                count_raw_sm,
                group_tidx,
                token_base_sm,
                num_heads_kv,
                head_kv_idx[0],
                None,
                mma_dtype,
                False,
            )

        bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

        if count_raw_sm > 0:
            num_q_groups_sm: T.int32 = uceil_div_i32(count_raw_sm, q_tokens_per_group)
            kv_block_col_start: T.int32 = kv_block_sm * N_BLOCK
            # WG0 takes the even Q groups, WG1 the odd ones (:2465-2468).
            num_stage_groups: T.int32 = udiv_i32(num_q_groups_sm + (1 - stage), 2)

            for qi_iter in T.serial(0, num_stage_groups, unroll=False):
                qi_group: T.int32 = qi_iter * 2 + stage
                phase: T.int32 = T.bitwise_and(qi_iter, 1)
                producer_phase: T.int32 = phase ^ 1
                qidx_meta_slot: T.int32 = (
                    T.bitwise_and(qi_group, QIDX_META_STAGES - 1) * q_tokens_per_group
                )
                # How many of this group's tokens still sit on the causal
                # diagonal and therefore need per-column masking (:2478-2486).
                qi_group_start: T.int32 = qi_group * q_tokens_per_group
                masked_tok_count: T.int32 = T.max(
                    0, T.min(q_tokens_per_group, diag_q_count_sm - qi_group_start)
                )

                # ---------------- softmax step (:2437-2616) ----------------
                bar_s_full.wait(stage, phase)
                s_frag = T.alloc_tcgen05_ldst_frag("32x32b", (M_BLOCK, N_BLOCK), "float32")
                s_regs = s_frag.local()
                for chunk in T.unroll(4):
                    T.evaluate(
                        T.ptx[_TMEM_LD_32](
                            *[s_regs[chunk * 32 + i] for i in range(32)],
                            T.cuda.get_tmem_addr(
                                tmem_s_col + stage * tmem_stage_stride + chunk * 32, 0, 0
                            ),
                        )
                    )
                # NOTE: no `tcgen05.wait::ld` here. The export's only two are in
                # the epilogue; the S read is consumed without a TMEM-load fence
                # (:2475-2480).

                if s_hook_k_global:
                    # THE FP8-Q-ONLY HOOK (:2482-2491). Under FP8 Q the K tensor
                    # scale cannot fold into E4M3 values, so it multiplies the
                    # FP32 S accumulator instead -- emitted between the
                    # tcgen05.ld and the mask, so it feeds row-max and every
                    # exp2. Its placement is bitwise-load-bearing, not a rescale
                    # that commutes. On the BF16 path this multiply does not
                    # exist and the fold happens in the dequantization.
                    kg = T.alloc_local((1,), "float32")
                    T.evaluate(T.ptx.ld.global_.f32(kg[0], k_global_scale.ptr_to([0])))
                    for ii in T.unroll(N_BLOCK // 2):
                        i: T.int32 = ii * 2
                        _packed_f32x2(
                            "mul.rn.f32x2", s_regs, i, i + 1, s_regs[i], s_regs[i + 1], kg[0], kg[0]
                        )

                # Column-limit masking, and causal masking for the tokens on
                # the diagonal. Both arms feed one r2p bit-test body, which is
                # how the reference emits it too (mask.py:36-46, :71-121).
                col_limit = T.alloc_local((1,), "int32")
                col_limit[0] = kv_valid_cols
                if masked_tok_count > 0:
                    tok_of_row: T.int32 = udiv_i32(group_tidx, qheadperkv)
                    q_idx_mask: T.int32 = T.bitwise_and(
                        ld_shared_i32(s_qidx_meta, qidx_meta_slot + tok_of_row), Q_IDX_MASK
                    )
                    causal_col_limit: T.int32 = q_idx_mask + causal_q_off - kv_block_col_start + 1
                    col_limit[0] = T.min(kv_valid_cols, causal_col_limit)
                if col_limit[0] < N_BLOCK:
                    for chunk in T.unroll(N_BLOCK // MASK_R2P_CHUNK):
                        shift: T.int32 = T.max((chunk + 1) * MASK_R2P_CHUNK - col_limit[0], 0)
                        # `shr.u32` clamps a shift of 32 or more to zero, which
                        # is exactly what `r2p_bitmask_below` relies on for the
                        # chunks entirely past the column limit. A TIR-level
                        # shift is undefined there and leaves the chunk
                        # unmasked, which shows up as too-large row sums on the
                        # early query rows.
                        bits_reg = T.alloc_local((1,), "uint32")
                        T.evaluate(
                            T.ptx.shr.u32(
                                bits_reg[0], T.uint32(0xFFFFFFFF), T.cast(shift, "uint32")
                            )
                        )
                        bits: T.uint32 = bits_reg[0]
                        # `T.unroll`, not `range`: a `range` here lowers to a
                        # runtime 32-trip loop, and this body runs once per
                        # element, per softmax step, per Q group, in both
                        # warpgroups. Unrolled, `1 << i` and the element index
                        # both fold to constants, so each element costs one
                        # `and` against an immediate and a predicated write --
                        # which is what `mask_r2p_lambda` (mask.py:36-46) emits.
                        #
                        # The bit is tested UNSIGNED. Testing it signed needs a
                        # special case at i == 31, where `1 << 31` overflows an
                        # int32 literal, and that special case is what forced a
                        # per-element select.
                        for i in T.unroll(MASK_R2P_CHUNK):
                            if T.bitwise_and(
                                bits, T.shift_left(T.uint32(1), T.cast(i, "uint32"))
                            ) == T.uint32(0):
                                s_regs[chunk * MASK_R2P_CHUNK + i] = NEG_INF

                # One KV block per Q group, so this is always the first and
                # only online-softmax step: no rescale of a running accumulator
                # (:2284-2287).
                row_max = T.alloc_local((1,), "float32")
                _row_max_128(s_regs, row_max, 0)
                # `row_max_safe` (:246-247): a fully masked row has row_max
                # -inf, and subtracting it would make every element NaN. The
                # reference substitutes 0.0 before the scale-subtract; the
                # row's sum then comes out 0 and the epilogue's zero guard
                # turns it into a neutral partial.
                row_max[0] = T.if_then_else(row_max[0] != NEG_INF, row_max[0], T.float32(0.0))
                # Taken as a scalar, not recomputed: the reference folds
                # `softmax_scale * log2(e)` on the host in double precision
                # (:514), and redoing it in f32 here differs by one ULP, which
                # propagates straight into every LSE.
                scale_log2: T.float32 = softmax_scale_log2
                neg_max_scaled: T.float32 = -(row_max[0] * scale_log2)
                for ii in T.unroll(N_BLOCK // 2):
                    i: T.int32 = ii * 2
                    _packed_f32x2(
                        "fma.rn.f32x2",
                        s_regs,
                        i,
                        i + 1,
                        s_regs[i],
                        s_regs[i + 1],
                        scale_log2,
                        scale_log2,
                        neg_max_scaled,
                        neg_max_scaled,
                    )

                if lse_temperature_partial_h is not None:
                    temp_row_sum = T.alloc_local((1,), "float32")
                    _scaled_exp2_row_sum_128(s_regs, lse_temperature_inv_scale, temp_row_sum)

                bar_p_last_empty.wait(stage, producer_phase)
                bar_p_empty.wait(stage, producer_phase)

                # exp2 with the reference's MUFU / polynomial mix, then the
                # packed conversion into the P operand dtype (:2307-2312).
                # 128 P values pack into 64 words as bf16, 32 as fp8; the
                # store repetition follows (:2429-2439).
                p_words = T.alloc_local((N_BLOCK * _DTYPE_BYTES[pv_dtype] * 8 // 32,), "uint32")
                # Python loops: the MUFU/polynomial choice is a trace-time
                # decision on (j, k), so both indices have to be Python ints.
                for j in range(EX2_FRG_CNT):
                    for k in range(0, EX2_FRG_TILE, 2):
                        use_mufu = T.meta_var(
                            (k % EX2_EMU_FREQ) < (EX2_EMU_FREQ - EX2_EMU_RES)
                            or j >= EX2_FRG_CNT - 1
                            or j < EX2_EMU_START_FRG
                        )
                        if use_mufu:
                            T.evaluate(
                                T.ptx.ex2.approx.ftz.f32(s_regs[j * 32 + k], s_regs[j * 32 + k])
                            )
                            T.evaluate(
                                T.ptx.ex2.approx.ftz.f32(
                                    s_regs[j * 32 + k + 1], s_regs[j * 32 + k + 1]
                                )
                            )
                        else:
                            _ex2_emulation_2(s_regs, j * 32 + k, j * 32 + k + 1)
                    _pack_p_words(p_words, s_regs, j, pv_dtype)

                # Publish P in two pieces: the first three quarters early, so
                # the MMA warp can start PV, and the last quarter on the
                # separate barrier its instruction sequence blocks on.
                split_idx = T.meta_var(4 * SPLIT_P_ARRIVE // N_BLOCK)
                rep = T.meta_var(_P_STORE_REP[pv_dtype])
                st_chain = T.meta_var(_TMEM_ST[rep])
                for k in range(4):
                    T.evaluate(
                        T.ptx[st_chain](
                            T.cuda.get_tmem_addr(
                                tmem_p_col + stage * tmem_stage_stride + k * rep, 0, 0
                            ),
                            *[p_words[k * rep + i] for i in range(rep)],
                        )
                    )
                    if k + 1 == split_idx:
                        T.evaluate(T.ptx.tcgen05.wait__st.sync.aligned())
                        bar_p_full.arrive(stage)
                T.evaluate(T.ptx.tcgen05.wait__st.sync.aligned())
                bar_p_last_full.arrive(stage)

                # The stats pipe's empty half is a credit on s_scale: it stops
                # this group from overwriting the slot the epilogue two groups
                # back has not drained (:2331).
                bar_stats_empty.wait(stage, producer_phase)
                row_sum = T.alloc_local((1,), "float32")
                _row_sum_128(s_regs, row_sum)
                st_shared_f32(s_scale, stage * M_BLOCK * 2 + group_tidx, row_sum[0])
                st_shared_f32(s_scale, stage * M_BLOCK * 2 + M_BLOCK + group_tidx, row_max[0])
                if lse_temperature_partial_h is not None:
                    st_shared_f32(s_scale_temp, stage * M_BLOCK + group_tidx, temp_row_sum[0])
                T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                bar_s_empty.arrive(stage)

                # ---------------- epilogue (:2927-3305) ----------------
                bar_sync_named(BAR_EPILOGUE + stage, SOFTMAX_THREADS)
                epilogue_step(
                    stage,
                    qi_group,
                    qidx_meta_slot,
                    group_tidx,
                    count_raw_sm,
                    q_batch_off_sm,
                    scale_log2,
                )

        bar_arrive_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

    if warp_idx < SOFTMAX1_WARP_BASE:
        if cta_valid_work != 0:
            T.evaluate(T.ptx.setmaxnreg.inc.sync.aligned.u32(T.uint32(NUM_REGS_SOFTMAX)))
            softmax_warpgroup(0)

    if warp_idx >= SOFTMAX1_WARP_BASE and warp_idx < Q_LOAD_WARP_BASE:
        if cta_valid_work != 0:
            T.evaluate(T.ptx.setmaxnreg.inc.sync.aligned.u32(T.uint32(NUM_REGS_SOFTMAX)))
            softmax_warpgroup(1)


def get_kernel(**config):
    """Return the TIRx specialization for one compile key."""
    config.pop("label", None)
    kernel = _kernel.specialize(
        qheadperkv=int(config["qhead_per_kv"]),
        causal=bool(config.get("causal", True)),
        dtype_mode=str(config.get("dtype", "bf16q")),
        partial_dtype=str(config.get("partial_dtype", "float32")),
        **({} if config.get("temperature") else {"lse_temperature_partial_h": None}),
        **({} if config.get("k_global", True) else {"k_global_scale_h": None}),
    )
    return kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


# ---------------------------------------------------------------------------
# Config matrix.
#
# `qhead_per_kv` picks the Q-load program: 1, 2 and 4 use the raw gather4
# descriptor path, 8 and 16 the plain TMA path (:84-89). `k_global` picks where
# the K tensor scale is applied, which is a different device path per Q dtype
# rather than a scalar difference.
# ---------------------------------------------------------------------------
def _case(
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    head_kv: int,
    qhead_per_kv: int,
    topk: int,
    label: str,
    dtype: str = "bf16q",
    partial_dtype: str = "float32",
    temperature: float | None = None,
    k_global: bool = True,
    causal: bool = True,
    blk_kv: int = N_BLOCK,
    seqlen_pattern: str = "uniform",
) -> dict:
    return {
        "label": label,
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "head_kv": head_kv,
        "qhead_per_kv": qhead_per_kv,
        "topk": topk,
        "dtype": dtype,
        "partial_dtype": partial_dtype,
        "temperature": temperature,
        "k_global": k_global,
        "causal": causal,
        "blk_kv": blk_kv,
        "seqlen_pattern": seqlen_pattern,
    }


BENCH_CONFIGS = [
    # MSA's ring-attention benchmark shape, verbatim.
    _case(
        label="ring48k_bf16q_qh16_t16",
        batch=1,
        seqlen_q=49152,
        seqlen_k=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
        partial_dtype="bfloat16",
    ),
    # MSA's ulysses shape at its own sweep's lowest topk: the full 384K sequence
    # at topk=16 needs several GB for O_partial alone, and the geometry -- not
    # the slot count -- is what makes this shape interesting.
    _case(
        label="ulysses384k_bf16q_qh2_t4",
        batch=1,
        seqlen_q=393216,
        seqlen_k=393216,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
        partial_dtype="bfloat16",
    ),
    _case(
        label="long96k_bf16q_qh2_t16",
        batch=1,
        seqlen_q=98304,
        seqlen_k=98304,
        head_kv=1,
        qhead_per_kv=2,
        topk=16,
        partial_dtype="bfloat16",
    ),
    _case(
        label="varlen_b3_s8192_bf16q_qh4_t16",
        batch=3,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        seqlen_pattern="varlen",
    ),
    _case(
        label="varlen_b3_s4096_bf16q_qh8_t8",
        batch=3,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        seqlen_pattern="varlen",
    ),
    # qhead_per_kv=1 is the gather4 path's extreme: one token per gather.
    _case(
        label="qh1_s8192_bf16q_t16",
        batch=1,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=1,
        topk=16,
        partial_dtype="bfloat16",
    ),
    # The FP8-Q family at marquee scale: pair dequant, the S-accumulator tensor
    # scale, the bf16 partial store and the temperature LSE all at once.
    _case(
        label="ring48k_fp8q_qh16_t16",
        batch=1,
        seqlen_q=49152,
        seqlen_k=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
        dtype="fp8q",
        partial_dtype="bfloat16",
        temperature=1.0,
    ),
    _case(
        label="fp8q_gather4_s16384_qh4_t16",
        batch=1,
        seqlen_q=16384,
        seqlen_k=16384,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        dtype="fp8q",
        partial_dtype="bfloat16",
    ),
    _case(
        label="fp8q_partial_fp8_s8192_qh8_t8",
        batch=1,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        dtype="fp8q",
        partial_dtype="float8_e4m3",
        temperature=2.0,
    ),
    _case(
        label="edge_b1_s1024_bf16q_t4",
        batch=1,
        seqlen_q=1024,
        seqlen_k=1024,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
    ),
]


CONFIGS = [
    *[case for case in BENCH_CONFIGS if case["seqlen_k"] <= 16384],
    # FP8 Q on the TMA-Q path, with temperature.
    _case(
        label="corr_fp8q_tma_s4096_qh16_t8",
        batch=1,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=1,
        qhead_per_kv=16,
        topk=8,
        dtype="fp8q",
        partial_dtype="bfloat16",
        temperature=1.0,
    ),
    _case(
        label="corr_bf16q_partial_bf16_s2048_qh16_t4",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=1,
        qhead_per_kv=16,
        topk=4,
        partial_dtype="bfloat16",
    ),
    # No K tensor scale. Under BF16 Q this drops the fold inside the dequant;
    # under FP8 Q it drops the S-accumulator multiply instead, so both dtypes
    # need their own case.
    _case(
        label="corr_bf16q_kgs0_s2048_qh4_t8",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=4,
        topk=8,
        k_global=False,
    ),
    _case(
        label="corr_fp8q_kgs0_s2048_qh2_t8",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=2,
        topk=8,
        dtype="fp8q",
        partial_dtype="bfloat16",
        k_global=False,
    ),
    _case(
        label="corr_fp8q_partial_fp8_s2048_qh8_t8",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        dtype="fp8q",
        partial_dtype="float8_e4m3",
        temperature=2.0,
    ),
    # Degenerate-Q boundary: many KV blocks, almost no query rows.
    _case(
        label="corr_decode_b8_qh16_t8",
        batch=8,
        seqlen_q=8,
        seqlen_k=4096,
        head_kv=1,
        qhead_per_kv=16,
        topk=8,
    ),
    _case(
        label="corr_tiny_b2_s512_qh1_t2",
        batch=2,
        seqlen_q=512,
        seqlen_k=512,
        head_kv=1,
        qhead_per_kv=1,
        topk=2,
        seqlen_pattern="varlen",
    ),
    _case(
        label="corr_bf16q_qh2_varlen_b2",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=2,
        topk=8,
        seqlen_pattern="varlen",
    ),
]


# ---------------------------------------------------------------------------
# Data.
# ---------------------------------------------------------------------------
def _dequant_nvfp4_to_bf16(packed, scale_128x4, global_scale, *, rows: int, cols: int):
    """Reference dequantization, transcribing quantize.py:265-343.

    ``x = e2m1_value * E4M3_block_scale * FP32_global_scale``. Used for the
    BF16 twins that feed the secondary oracle, so it must agree with MSA's own
    function rather than merely being a plausible inverse.
    """
    import torch

    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    flat = packed.reshape(rows, cols // 2)
    values = torch.empty(rows, cols, dtype=torch.float32, device=packed.device)
    values[:, 0::2] = lut[(flat & 0x0F).long()]
    values[:, 1::2] = lut[(flat >> 4).long()]

    scale_cols = cols // SCALE_BLOCK
    row_idx = torch.arange(rows, device=packed.device).view(-1, 1)
    col_idx = torch.arange(scale_cols, device=packed.device).view(1, -1)
    tiles_n = (scale_cols + SCALE_TILE_COLS - 1) // SCALE_TILE_COLS
    offset = (
        (row_idx // SCALE_TILE_ROWS * tiles_n + col_idx // SCALE_TILE_COLS) * 512
        + (row_idx % SCALE_TILE_ROWS % 32) * 16
        + (row_idx % SCALE_TILE_ROWS // 32) * 4
        + col_idx % SCALE_TILE_COLS
    )
    block_scale = scale_128x4.reshape(-1)[offset.reshape(-1)].reshape(rows, scale_cols)
    block_scale = block_scale.view(torch.float8_e4m3fn).float()
    scaled = values * block_scale.repeat_interleave(SCALE_BLOCK, dim=1)
    if global_scale is not None:
        scaled = scaled * global_scale.float().item()
    return scaled.to(torch.bfloat16)


def prepare_data(*, seed: int = 0, **config) -> dict[str, Any]:
    """Build Q, packed NVFP4 K/V with their scales, the CSR payload and the schedule.

    The K/V bytes are drawn uniformly rather than quantized from a BF16 draw:
    the gate compares this port against the source kernel on identical frozen
    inputs, so what the bytes mean matters less than that every code path sees
    a representative spread of them. Transformer Engine, which MSA's own
    quantizer needs, is not a dependency of this repo.

    The scale bytes are drawn from a positive finite E4M3 band. A NaN byte
    would poison a whole softmax row, and upstream's own synthetic helper pins
    every scale to 1.0, which would leave the 128x4 offset arithmetic -- the
    riskiest expression in the port -- exercised by nothing.
    """
    import torch

    from tirx_kernels.msa.sparse_atten_fwd import _frozen_qsplit
    from tirx_kernels.msa.sparse_prepare_fwd_split_atomic import prepare_data as prepare_csr

    config.pop("label", None)
    csr_config = {key: config[key] for key in _CSR_CONFIG_KEYS if key in config}
    csr = prepare_csr(seed=seed, **csr_config)

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(9871 + seed)
    head_kv = config["head_kv"]
    qhead_per_kv = config["qhead_per_kv"]
    head_q = head_kv * qhead_per_kv
    total_q = csr["total_q"]
    total_k = int(sum(csr["seqlens_k"]))
    mode = DTYPE_MODES[config.get("dtype", "bf16q")]
    q_dtype = _torch_dtype(mode["q"])

    q_bf16 = torch.randn(
        (total_q, head_q, HEAD_DIM), dtype=torch.bfloat16, device=device, generator=generator
    )
    q = q_bf16.to(q_dtype).contiguous()

    kv_shape = (total_k, head_kv, PACKED_HEAD_DIM)
    k = torch.randint(0, 256, kv_shape, dtype=torch.uint8, device=device, generator=generator)
    v = torch.randint(0, 256, kv_shape, dtype=torch.uint8, device=device, generator=generator)

    rows = total_k * head_kv
    scale_rows = -(-rows // SCALE_TILE_ROWS) * SCALE_TILE_ROWS
    scale_cols = -(-(HEAD_DIM // SCALE_BLOCK) // SCALE_TILE_COLS) * SCALE_TILE_COLS
    scale_shape = (scale_rows, scale_cols)
    k_scale = torch.randint(
        0x30, 0x41, scale_shape, dtype=torch.uint8, device=device, generator=generator
    )
    v_scale = torch.randint(
        0x30, 0x41, scale_shape, dtype=torch.uint8, device=device, generator=generator
    )

    # Upstream's own benchmark values.
    k_global = (
        torch.tensor([0.75], dtype=torch.float32, device=device)
        if config.get("k_global", True)
        else None
    )
    v_global = torch.tensor([1.25], dtype=torch.float32, device=device)

    cu_seqlens_k = torch.zeros(len(csr["seqlens_k"]) + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.tensor(csr["seqlens_k"], dtype=torch.int32, device=device).cumsum(0)

    qsplit, q_abs_of_edge = _frozen_qsplit(csr)

    def twin(packed, scale, global_scale):
        """The BF16 the source kernel's dequant produces, for the twin oracle.

        The scale row is ``token * head_kv + head_kv_idx``
        (``_flat_kv_scale_row``, :1320-1326), which is exactly the row-major
        flattening of the ``(total_k, head_kv, ...)`` tensor -- so no permute.
        Grouping by head instead (``head_kv_idx * total_k + token``) agrees with
        it whenever ``head_kv == 1`` and pairs every token with another head's
        block scale as soon as it is not.
        """
        flat = packed.reshape(-1, PACKED_HEAD_DIM)
        out = _dequant_nvfp4_to_bf16(flat, scale, global_scale, rows=rows, cols=HEAD_DIM)
        return out.reshape(total_k, head_kv, HEAD_DIM).contiguous()

    return {
        "config": dict(config),
        "csr": csr,
        "q": q,
        "q_flat": q.reshape(-1, HEAD_DIM),
        "k": k,
        "v": v,
        "k_scale_128x4": k_scale,
        "v_scale_128x4": v_scale,
        "k_global_scale": k_global,
        # Accepted by the host entry but never reaching this kernel: the
        # interface pins has_v_global_scale off and applies V's tensor scale in
        # the combine kernel. Kept so the contract is visible in one place.
        "v_global_scale": v_global,
        "k_dequantized": twin(k, k_scale, k_global),
        "v_dequantized": twin(v, v_scale, None),
        "k2q_row_ptr": csr["k2q_row_ptr"].view(head_kv, -1),
        "k2q_q_indices": csr["k2q_q_indices"].view(head_kv, -1),
        "k2q_qsplit_indices": qsplit,
        "q_abs_of_edge": q_abs_of_edge,
        "scheduler_metadata": csr["scheduler_metadata"],
        "work_count": csr["work_count"],
        "degrees": csr["degrees"].to(torch.int32),
        "cu_seqlens_q": csr["cu_seqlens_q"],
        "cu_seqlens_k": cu_seqlens_k,
        "seqlens_q": csr["seqlens_q"],
        "seqlens_k": csr["seqlens_k"],
        "softmax_scale": HEAD_DIM**-0.5,
        "lse_temperature_scale": config.get("temperature"),
        "head_dim": HEAD_DIM,
        "blk_kv": config["blk_kv"],
        "head_kv": head_kv,
        "head_q": head_q,
        "qhead_per_kv": qhead_per_kv,
        "topk": config["topk"],
        "total_q": total_q,
        "total_k": total_k,
        "total_rows": csr["total_rows"],
        "nnz": csr["nnz_capacity"],
        "num_batches": len(csr["seqlens_k"]),
        "max_seqlen_q": csr["max_seqlen_q"],
        "work_capacity": csr["work_capacity"],
        "causal": config.get("causal", True),
        "partial_dtype": config.get("partial_dtype", "float32"),
        "q_dtype": mode["q"],
        "mma_dtype": mode["mma"],
    }


def make_outputs(data: dict[str, Any]) -> dict[str, Any]:
    """Fresh partial buffers, uninitialized exactly as the host allocates them."""
    import torch

    shape = (data["topk"], data["total_q"], data["head_q"])
    device = data["q"].device
    outputs = {
        "o_partial": torch.empty(
            (*shape, HEAD_DIM), dtype=_torch_dtype(data["partial_dtype"]), device=device
        ),
        "lse_partial": torch.empty(shape, dtype=torch.float32, device=device),
    }
    if data["lse_temperature_scale"] is not None:
        outputs["lse_temperature_partial"] = torch.empty(shape, dtype=torch.float32, device=device)
    return outputs


def tirx_args(data: dict[str, Any], outputs: dict[str, Any]) -> tuple:
    """The launch ABI, bound once outside any timed region.

    The three scale scalars are taken from the host exactly as the source
    computes them (:487-488): ``softmax_scale_log2 = softmax_scale * log2(e)``
    and ``lse_temperature_scale_log2 = softmax_scale_log2 * inv_temperature``.
    Recomputing either on the device is an ABI change, and the second chains off
    the already-folded first, so it cannot be re-derived from the raw scales.
    """
    import math

    import torch

    def as_bits(t):
        """fp8 reaches the kernel as raw ``uint8``; see ``_tirx_dtype``."""
        return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t

    scale_log2 = data["softmax_scale"] * math.log2(math.e)
    inv_temperature = 1.0 / (data["lse_temperature_scale"] or 1.0)

    args = [
        data["k"],
        data["v"],
        data["k_scale_128x4"].reshape(-1),
        data["v_scale_128x4"].reshape(-1),
    ]
    if data["k_global_scale"] is not None:
        args.append(data["k_global_scale"])
    args += [
        data["k2q_q_indices"].reshape(-1),
        data["k2q_qsplit_indices"].reshape(-1),
        data["k2q_row_ptr"].reshape(-1),
        data["scheduler_metadata"].reshape(-1),
        data["work_count"],
        as_bits(outputs["o_partial"].reshape(-1)),
        outputs["lse_partial"].reshape(-1),
    ]
    if data["lse_temperature_scale"] is not None:
        args.append(outputs["lse_temperature_partial"].reshape(-1))
    args += [
        as_bits(data["q_flat"]),
        data["cu_seqlens_q"],
        data["cu_seqlens_k"],
        scale_log2,
        scale_log2 * inv_temperature,
        inv_temperature,
        data["total_rows"],
        data["head_kv"],
        data["max_seqlen_q"],
        data["work_capacity"],
        data["total_k"],
        data["total_q"],
        data["head_q"],
        data["nnz"],
        data["total_rows"],
        data["num_batches"],
        data["topk"],
        data["k_scale_128x4"].numel(),
    ]
    return tuple(args)


def reference_case(data: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """Argument bundle for MSA's compiled NVFP4 forward.

    ``o_partial_flat`` is a VIEW of the buffer the caller reads back. The host
    entry would pass ``reshape(-1, head_dim).contiguous()``, which silently
    hands the kernel a detached copy when the reshape is not already
    contiguous.
    """
    temperature = data["lse_temperature_scale"]
    return {
        "head_dim": data["head_dim"],
        "blk_kv": data["blk_kv"],
        "qhead_per_kv": data["qhead_per_kv"],
        "causal": data["causal"],
        "k": data["k"],
        "v": data["v"],
        "k_scale_128x4": data["k_scale_128x4"],
        "v_scale_128x4": data["v_scale_128x4"],
        "k_global_scale": data["k_global_scale"],
        "k2q_q_indices": data["k2q_q_indices"],
        "k2q_qsplit_indices": data["k2q_qsplit_indices"],
        "k2q_row_ptr": data["k2q_row_ptr"],
        "scheduler_metadata": data["scheduler_metadata"],
        "work_count": data["work_count"],
        "q_flat": data["q_flat"],
        "o_partial_flat": outputs["o_partial"].reshape(-1, HEAD_DIM),
        "lse_partial": outputs["lse_partial"],
        "lse_temperature_partial": outputs.get("lse_temperature_partial"),
        "cu_seqlens_q": data["cu_seqlens_q"],
        "cu_seqlens_k": data["cu_seqlens_k"],
        "softmax_scale": data["softmax_scale"],
        "lse_temperature_inv_scale": 1.0 / (temperature or 1.0),
        "num_kv_blocks": data["total_rows"],
        "head_kv": data["head_kv"],
        "max_seqlen_q": data["max_seqlen_q"],
        "work_capacity": data["work_capacity"],
    }


def run_test(**config) -> None:
    """Compile, launch and validate one config against the MSA source kernel.

    Two oracles. The gate is bitwise against the compiled NVFP4 source on
    identical frozen inputs -- valid because slot ownership is deterministic and
    this kernel contains no atomics. On the BF16-Q configs a second check runs
    the already-ported BF16 sibling on the dequantized twins of the same K/V,
    which is the only thing that can catch a data-plumbing error: a transposed
    scale or a wrong nibble order would make both sides of the bitwise
    comparison consume the same wrong bytes and agree.
    """
    import unittest

    import torch

    from tirx_kernels.msa.sparse_atten_fwd import assert_partials_match
    from tirx_kernels.runner import compile_kernel

    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    config.pop("label", None)
    data = prepare_data(**config)

    try:
        from tirx_kernels.msa.utils._msa_bench import compiled_sparse_atten_nvfp4_kv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc

    expected = make_outputs(data)
    try:
        compiled_sparse_atten_nvfp4_kv(reference_case(data, expected))()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    torch.cuda.synchronize()

    executable = compile_kernel(get_kernel(**config))
    outputs = make_outputs(data)
    executable(*tirx_args(data, outputs))
    torch.cuda.synchronize()
    assert_partials_match(data, outputs, expected)

    if data["q_dtype"] == "bfloat16":
        _assert_matches_dequantized_twin(data, outputs)


def _twin_case(data: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """The BF16 sibling's argument bundle over the dequantized K/V twins.

    V carries no tensor scale, because this kernel is never given one: the
    interface pins ``has_v_global_scale`` off and the combine kernel applies it.
    The twin has to see V scaled exactly the way this kernel sees it.
    """
    return {
        "head_dim": data["head_dim"],
        "blk_kv": data["blk_kv"],
        "qhead_per_kv": data["qhead_per_kv"],
        "causal": data["causal"],
        "k": data["k_dequantized"],
        "v": data["v_dequantized"],
        "qk_dtype": _torch_dtype("bfloat16"),
        "pv_dtype": _torch_dtype("bfloat16"),
        "k2q_q_indices": data["k2q_q_indices"],
        "k2q_qsplit_indices": data["k2q_qsplit_indices"],
        "k2q_row_ptr": data["k2q_row_ptr"],
        "scheduler_metadata": data["scheduler_metadata"],
        "work_count": data["work_count"],
        "q_flat": data["q_flat"],
        "o_partial_flat": outputs["o_partial"].reshape(-1, HEAD_DIM),
        "lse_partial": outputs["lse_partial"],
        "lse_temperature_partial": outputs.get("lse_temperature_partial"),
        "cu_seqlens_q": data["cu_seqlens_q"],
        "cu_seqlens_k": data["cu_seqlens_k"],
        "softmax_scale": data["softmax_scale"],
        "lse_temperature_inv_scale": 1.0 / (data["lse_temperature_scale"] or 1.0),
        "num_kv_blocks": data["total_rows"],
        "head_kv": data["head_kv"],
        "max_seqlen_q": data["max_seqlen_q"],
        "work_capacity": data["work_capacity"],
    }


def _assert_matches_dequantized_twin(data: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Second oracle: the BF16 sibling on the dequantized twins of this K/V.

    Mirrors upstream's ``test_sparse_atten_nvfp4_kv_matches_dequantized_bf16``
    and runs at its tolerance. The point is not numerical -- the primary gate is
    already bitwise -- but structural: it is an independent implementation
    reading independently-produced inputs, so a wiring error in the scale layout
    or the nibble order shows up here and nowhere else. On this data the two
    agree exactly, because the FP4 value set and the E4M3 scales carry few
    enough significant bits that both rounding paths land on the same BF16.
    """
    import unittest

    import torch

    from tirx_kernels.msa.sparse_atten_fwd import live_partial_mask

    try:
        from tirx_kernels.msa.utils._msa_bench import compiled_sparse_atten_fwd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc

    twin = make_outputs(data)
    compiled_sparse_atten_fwd(_twin_case(data, twin))()
    torch.cuda.synchronize()

    mask = live_partial_mask(data)
    o_mask = mask.unsqueeze(-1).expand_as(twin["o_partial"])
    torch.testing.assert_close(
        outputs["o_partial"][o_mask].float(),
        twin["o_partial"][o_mask].float(),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        outputs["lse_partial"][mask], twin["lse_partial"][mask], rtol=2e-2, atol=2e-2
    )


# ---------------------------------------------------------------------------
# Benchmark entry points.
#
# Like the sibling, this kernel needs no rotation: it reads its inputs without
# touching them and overwrites -- never accumulates into -- the partial slots it
# owns, so the hundredth launch does exactly the work the first one did.
# ---------------------------------------------------------------------------
def prepare_bench(**config):
    """Compile the TIRx specialization without initializing CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config.pop("label", None)
    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    """Kernel-only comparison against MSA's compiled NVFP4 forward launch."""
    from tirx_kernels.runner import bench

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    executable = prepared["executable"]

    tirx_outputs = make_outputs(data)
    tirx_bound = tirx_args(data, tirx_outputs)

    def tirx_launch():
        executable(*tirx_bound)

    def build_reference():
        from tirx_kernels.msa.utils._msa_bench import compiled_sparse_atten_nvfp4_kv

        launch = compiled_sparse_atten_nvfp4_kv(reference_case(data, make_outputs(data)))
        launch()  # pay the CuTeDSL compile and first-launch cost outside timing
        return launch

    return bench(
        {"tirx": tirx_launch},
        references={"msa": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
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
    "run_test",
]
