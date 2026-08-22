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

import inspect
from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.msa.utils._scalar_ops import (
    ld_global_i32,
    ld_shared_i32,
    st_shared_i32,
    uceil_div_i32,
    udiv_i32,
)

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

# `mO_partial.element_type`.
PARTIAL_DTYPES = ("float32", "bfloat16", "float16", "float8_e4m3")


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
NUM_REGS_SOFTMAX_CAUSAL = 176
NUM_REGS_SOFTMAX_NONCAUSAL = 192
NUM_REGS_STORE_CAUSAL = 112
NUM_REGS_STORE_NONCAUSAL = 80
NUM_REGS_OTHER = 48
# `ex2_emu_freq = 16 if causal else 0` (:184), `ex2_emu_start_frg = 1` (:185),
# and `ex2_emu_res` keeps softmax.py's default of 4. All three enter the
# emulation predicate; a stride test on the element index picks a different
# sixteen elements and changes the result bitwise.
EX2_EMU_FREQ_CAUSAL = 16
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
_TMA_G2S_4D_CACHE = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
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
_TMA_CACHE_EVICT_FIRST = K.uint64(0x12F0000000000000)
_TMA_CACHE_EVICT_LAST = K.uint64(0x14F0000000000000)
_GATHER4_Q_CACHE_HINT = _TMA_CACHE_EVICT_LAST
_TMA_NO_POLICY = K.uint64(0)


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


def _packed_f32x2(op, dst, di, dj, a0, a1, b0, b1, c0=None, c1=None):
    """One packed two-lane f32 operation.

    The ``.f32x2`` instructions take 64-bit packed operands, so each pair is
    assembled with ``mov.b64`` and split again afterwards. This is one
    instruction with two ordered results, not two scalar operations.
    """
    pa = K.alloc_local((1,), "uint64")
    pb = K.alloc_local((1,), "uint64")
    pd = K.alloc_local((1,), "uint64")
    K.ptx.mov.b64(pa[0], a0, a1)
    K.ptx.mov.b64(pb[0], b0, b1)
    if c0 is None:
        K.ptx[op](pd[0], pa[0], pb[0])
    else:
        pc = K.alloc_local((1,), "uint64")
        K.ptx.mov.b64(pc[0], c0, c1)
        K.ptx[op](pd[0], pa[0], pb[0], pc[0])
    K.ptx.mov.b64(dst[di], dst[dj], pd[0])


def _max3_at(dst, idx, a, b, c):
    """Three-input ``max.f32``, an SM100 form, writing into ``dst[idx]``."""
    K.ptx.max.f32(dst[idx], a, b, c)


def _row_max_128(regs, out, out_idx):
    """`_compute_row_max` for arch 100 (utils.py:258-276).

    Four accumulators SEEDED from the first eight elements with the two-source
    form, each absorbing two more per step through a three-input max, then
    folded in two steps. The seed is the initialization -- there is no -inf
    fill, and the strided loop therefore starts at 8. Dropping the
    ``local_max[0] = fmax(local_max[0], local_max[1])`` fold leaves a quarter of
    the row out of the maximum without changing the instruction shape.
    """
    acc = K.alloc_local((4,), "float32")
    K.ptx.max.f32(acc[0], regs[0], regs[1])
    K.ptx.max.f32(acc[1], regs[2], regs[3])
    K.ptx.max.f32(acc[2], regs[4], regs[5])
    K.ptx.max.f32(acc[3], regs[6], regs[7])
    with K.unroll(1, N_BLOCK // 8) as it:
        i = it * 8
        _max3_at(acc, 0, acc[0], regs[i + 0], regs[i + 1])
        _max3_at(acc, 1, acc[1], regs[i + 2], regs[i + 3])
        _max3_at(acc, 2, acc[2], regs[i + 4], regs[i + 5])
        _max3_at(acc, 3, acc[3], regs[i + 6], regs[i + 7])
    K.ptx.max.f32(acc[0], acc[0], acc[1])
    _max3_at(out, out_idx, acc[0], acc[2], acc[3])


def _scaled_exp2_row_sum_128(regs, scale, out):
    """`fadd_exp2_scaled_reduce` for arch 100 (utils.py:308-350).

    A second pass over the same 128 elements at a different scale, for the
    temperature LSE. It carries NO ex2 emulation -- every one of the 128 is a
    real ``ex2`` -- which is why a temperature build's ex2 census is 480 rather
    than 448. It reads the fragment without writing it back, so the P
    population below still sees the scale-subtracted values.
    """
    acc = K.alloc_local((8,), "float32")
    with K.unroll(8) as j:
        K.assign(acc[j], K.float32(0.0))
    tmp = K.alloc_local((8,), "float32")
    with K.unroll(N_BLOCK // 8) as it:
        i = it * 8
        with K.unroll(4) as j:
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
        with K.unroll(8) as j:
            K.ptx.ex2.approx.ftz.f32(tmp[j], tmp[j])
        with K.unroll(4) as j:
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
    K.assign(out[0], acc[0] + acc[1])


def _row_sum_128(regs, out):
    """`_compute_row_sum` for arch 100 (utils.py:288-304): four packed
    accumulators seeded from the first eight elements, then three folds."""
    acc = K.alloc_local((8,), "float32")
    with K.unroll(8) as j:
        K.assign(acc[j], regs[j])
    with K.unroll(1, N_BLOCK // 8) as it:
        i = it * 8
        with K.unroll(4) as j:
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
    K.assign(out[0], acc[0] + acc[1])


def _combine_int_frac_ex2(x_rounded, frac_ex2):
    """`combine_int_frac_ex2` (utils.py:1008-1030): shift the integer part into
    the exponent field and add it to the polynomial's bits.

    ``add.s32`` deliberately -- it lowers to LEA on the ALU pipe, where
    ``add.u32`` would lower to IMAD and contend with the FMA pipe the polynomial
    is already using.
    """
    xi = K.alloc_local((1,), "int32")
    fi = K.alloc_local((1,), "int32")
    out = K.alloc_local((1,), "int32")
    K.ptx.mov.b32(xi[0], x_rounded)
    K.ptx.mov.b32(fi[0], frac_ex2)
    K.ptx.shl.b32(xi[0], xi[0], K.uint32(23))
    K.ptx.add.s32(out[0], xi[0], fi[0])
    return K.reinterpret("float32", out[0])


def _ex2_emulation_2(regs, i, j):
    """`ex2_emulation_2` (utils.py:987-1005): 2**x for a PAIR, without MUFU.

    Clamp to -127, split off the integer part by a round-down add of
    2**23 + 2**22, evaluate the fractional part with a degree-3 packed Horner,
    and reassemble by shifting the integer into the exponent field. The clamp is
    two ``max.f32`` per pair, which is where 32 of the export's 164 ``max.f32``
    live -- they are not part of the row-max tree.
    """
    cl = K.alloc_local((2,), "float32")
    K.ptx.max.f32(cl[0], regs[i], K.float32(-127.0))
    K.ptx.max.f32(cl[1], regs[j], K.float32(-127.0))
    rounded = K.alloc_local((2,), "float32")
    _packed_f32x2(
        "add.rm.f32x2",
        rounded,
        0,
        1,
        cl[0],
        cl[1],
        K.float32(_FP32_ROUND_INT),
        K.float32(_FP32_ROUND_INT),
    )
    back = K.alloc_local((2,), "float32")
    _packed_f32x2(
        "sub.rn.f32x2",
        back,
        0,
        1,
        rounded[0],
        rounded[1],
        K.float32(_FP32_ROUND_INT),
        K.float32(_FP32_ROUND_INT),
    )
    frac = K.alloc_local((2,), "float32")
    _packed_f32x2("sub.rn.f32x2", frac, 0, 1, cl[0], cl[1], back[0], back[1])
    poly = K.alloc_local((2,), "float32")
    K.assign(poly[0], K.float32(_POLY_EX2_3[3]))
    K.assign(poly[1], K.float32(_POLY_EX2_3[3]))
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
        K.float32(_POLY_EX2_3[2]),
        K.float32(_POLY_EX2_3[2]),
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
        K.float32(_POLY_EX2_3[1]),
        K.float32(_POLY_EX2_3[1]),
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
        K.float32(_POLY_EX2_3[0]),
        K.float32(_POLY_EX2_3[0]),
    )
    K.assign(regs[i], _combine_int_frac_ex2(rounded[0], poly[0]))
    K.assign(regs[j], _combine_int_frac_ex2(rounded[1], poly[1]))


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


def _paged_kv_scale_row(row, head_kv_idx, page_idx, num_heads_kv):
    """`_paged_kv_scale_row` (:1328-1336).

    THE one genuine NVFP4 paged difference. `_scale_128x4_offset` is layout-only
    and identical either way; what moves is the logical row fed into it.

    ``row`` is the INTRA-PAGE row, 0..N_BLOCK-1 -- not the batch-offset-adjusted
    absolute token the flat form takes. Handing this the absolute token yields
    ``(page*H + h)*N_BLOCK + k_batch_offset + kv_block*N_BLOCK + row``: a
    different, still in-range E4M3 byte for every task. It does not fault, the
    values look plausible, and only the bitwise gate catches it.
    """
    return (page_idx * num_heads_kv + head_kv_idx) * N_BLOCK + row


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
    out = K.alloc_local((1,), "int8")
    K.ptx.ld.global_.s8(out[0], scale.ptr_to([offset]))
    return out[0]


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
    byte = K.alloc_local((1,), "uint8")
    K.ptx.ld.global_.u8(byte[0], scale.ptr_to([offset]))
    masked = K.alloc_local((1,), "uint32")
    K.ptx.and_.b32(masked[0], K.cast(byte[0], "uint32"), K.uint32(0xFF))
    packed_i = K.alloc_local((1,), "int32")
    K.ptx.mul.lo.s32(packed_i[0], K.reinterpret("int32", masked[0]), K.int32(0x01010101))
    q = K.alloc_local((1,), "uint32")
    mant = K.alloc_local((1,), "uint32")
    acc = K.alloc_local((1,), "uint32")
    packed = K.local_scalar(K.u32, init=K.reinterpret("uint32", packed_i[0]), name="packed")
    K.ptx.prmt.b32(q[0], packed, packed, K.uint32(0x1302))
    K.ptx.and_.b32(acc[0], q[0], K.uint32(0x80008000))
    K.ptx.and_.b32(mant[0], q[0], K.uint32(0x7F007F00))
    K.ptx.shr.u32(mant[0], mant[0], K.uint32(4))
    K.ptx.or_.b32(acc[0], acc[0], mant[0])
    K.ptx.fma.rn.bf16x2(out[0], acc[0], K.uint32(0x7B807B80), K.uint32(0))


def _fp4_byte(word, b, out):
    """One packed-E2M1 byte out of a 32-bit word.

    The source spells this ``mov.b32 {byte0, byte1, byte2, byte3}, $2`` and
    feeds a ``.reg .b8``. Inline asm has no 8-bit constraint, so the TIRx PTX
    table stages that register itself and takes a ``uint8`` operand; the byte is
    selected here with a shift and a mask instead of the 4-way move, which is
    the same extraction with a spelling the C boundary can carry.
    """
    K.assign(
        out[0], K.cast(K.bitwise_and(K.shift_right(word, K.uint32(8 * b)), K.uint32(0xFF)), "uint8")
    )


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
    with K.unroll(2) as w:
        with K.unroll(4) as b:
            byte = K.alloc_local((1,), "uint8")
            _fp4_byte(src_words[w], b, byte)
            f16_pair = K.alloc_local((1,), "uint32")
            K.ptx.cvt.rn.f16x2.e2m1x2(f16_pair[0], byte[0])
            halves = K.alloc_local((2,), "uint16")
            K.ptx.mov.b32(halves[0], halves[1], f16_pair[0])
            lo = K.alloc_local((1,), "float32")
            hi = K.alloc_local((1,), "float32")
            K.ptx.cvt.f32.f16(lo[0], halves[0])
            K.ptx.cvt.f32.f16(hi[0], halves[1])
            bf16_pair = K.alloc_local((1,), "uint32")
            K.ptx.cvt.rn.bf16x2.f32(bf16_pair[0], hi[0], lo[0])
            K.ptx.mul.rn.bf16x2(out_words[w * 4 + b], bf16_pair[0], combined_bf16x2)


def _dequant_fp4x8_scaled_e4m3(src_word, scale_e4m3, out, lo_idx, hi_idx):
    """`cvt_fp4x8_e2m1_scaled_e4m3x8` (utils.py:734-763), the fallback body.

    Eight E2M1 values scaled by one E4M3 byte and converted to E4M3. The pin's
    CUDA 12.9 rules out ``mul.e4m3x4.e2m1x4.e4m3x4.satfinite`` -- which is also
    the one instruction in this family the TIRx PTX table does not carry -- so
    the transcribed chain is the f16 one: replicate the scale byte, decode it to
    f16x2, decode four fp4 byte-pairs to f16x2, multiply, convert back to E4M3
    pairs and repack.
    """
    sf_bytes = K.alloc_local((1,), "uint32")
    K.ptx.prmt.b32(sf_bytes[0], K.cast(scale_e4m3, "uint32"), K.uint32(0), K.uint32(0))
    sf_pair = K.alloc_local((2,), "uint16")
    K.ptx.mov.b32(sf_pair[0], sf_pair[1], sf_bytes[0])
    sf_f16x2 = K.alloc_local((1,), "uint32")
    K.ptx.cvt.rn.f16x2.e4m3x2(sf_f16x2[0], sf_pair[0])
    e = K.alloc_local((4,), "uint16")
    with K.unroll(4) as b:
        byte = K.alloc_local((1,), "uint8")
        _fp4_byte(src_word, b, byte)
        h = K.alloc_local((1,), "uint32")
        K.ptx.cvt.rn.f16x2.e2m1x2(h[0], byte[0])
        K.ptx.mul.rn.f16x2(h[0], h[0], sf_f16x2[0])
        K.ptx.cvt.rn.satfinite.e4m3x2.f16x2(e[b], h[0])
    K.ptx.mov.b32(out[lo_idx], e[0], e[1])
    K.ptx.mov.b32(out[hi_idx], e[2], e[3])


def bar_sync_named(bar_id, count):
    """``bar.sync <id>, <count>`` -- arrive AND wait on a named barrier."""
    K.ptx.bar.sync(K.uint32(bar_id), K.uint32(count))


def bar_arrive_named(bar_id, count):
    """``bar.arrive <id>, <count>`` -- arrive WITHOUT waiting.

    Which of the two a site uses is load-bearing. The softmax warpgroups leave
    the TMEM-allocator barrier arrive-only (:1124, :1191); only the MMA warp,
    which owns the deallocation, arrives and waits (:1055). Making the softmax
    exits wait parks both warpgroups behind a TMEM teardown they have no
    ordering dependence on.
    """
    K.ptx.bar.arrive(K.uint32(bar_id), K.uint32(count))


def _mbar_expect_tx(bar, stage, tx_bytes):
    """``mbarrier.expect_tx``: promise bytes without arriving.

    The single-shot K/V barriers take their transaction count in the prologue
    from thread 0 (:868-873) and are arrived on later by the load warp, so this
    stays separate from the arrive -- the barrier's arrival count is 1.
    """
    K.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(bar.ptr_to([stage]), K.uint32(tx_bytes))


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
    paged,
    s_paged_kv_idx,
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
    with K.If(count_raw > 0), K.Then():
        bar_tma.wait(0, 0)
        if mma_dtype == "float8_e4m3":
            # Pair form: two scale columns, 32 values, one 16-byte load.
            pairs_per_row = SCALE_COLS // 2
            total_pairs = N_BLOCK * pairs_per_row
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
            staged = K.alloc_local((DEQUANT_FP8_STAGED,), "uint32")
            for pre in range(DEQUANT_FP8_BATCH):
                task_pre = K.local_scalar(
                    K.i32, init=pre * DEQUANT_THREADS + group_tidx, name="task_pre"
                )
                row_pre = K.local_scalar(
                    K.i32, init=udiv_i32(task_pre, pairs_per_row), name="row_pre"
                )
                pair_pre = K.local_scalar(
                    K.i32, init=task_pre - row_pre * pairs_per_row, name="pair_pre"
                )
                K.ptx.ld.shared.v4.b32(
                    staged[pre * 4],
                    staged[pre * 4 + 1],
                    staged[pre * 4 + 2],
                    staged[pre * 4 + 3],
                    src.ptr_to([row_pre, pair_pre * 16]),
                )
            for it in range(DEQUANT_FP8_BATCH):
                task = K.local_scalar(K.i32, init=it * DEQUANT_THREADS + group_tidx, name="task")
                row = K.local_scalar(K.i32, init=udiv_i32(task, pairs_per_row), name="row")
                pair_col = K.local_scalar(K.i32, init=task - row * pairs_per_row, name="pair_col")
                if paged:
                    # Re-read EVERY ITERATION: the backend does not hoist this
                    # out of the rolled task loop (.loc 1 1384 K / :1558 V,
                    # inside the `.pragma "nounroll"` body), and lifting it
                    # changes the hottest loop's instruction count.
                    page_idx_d = K.local_scalar(
                        K.i32, init=ld_shared_i32(s_paged_kv_idx, 0), name="page_idx_d"
                    )
                    scale_row = K.local_scalar(
                        K.i32,
                        init=_paged_kv_scale_row(row, head_kv_idx, page_idx_d, num_heads_kv),
                        name="scale_row",
                    )
                else:
                    token = K.local_scalar(K.i32, init=token_base + row, name="token")
                    scale_row = K.local_scalar(
                        K.i32,
                        init=_flat_kv_scale_row(token, head_kv_idx, num_heads_kv),
                        name="scale_row",
                    )
                src_words = K.alloc_local((4,), "uint32")
                for w in range(4):
                    K.assign(src_words[w], staged[it * 4 + w])
                scale_lo = _load_scale_e4m3_u8(
                    scale, _scale_128x4_offset(scale_row, pair_col * 2, SCALE_COLS)
                )
                scale_hi = _load_scale_e4m3_u8(
                    scale, _scale_128x4_offset(scale_row, pair_col * 2 + 1, SCALE_COLS)
                )
                out_words = K.alloc_local((8,), "uint32")
                # `_dequant_fp4x32_to_fp8` (:1296-1317): the first two words take
                # the low scale column, the second two the high one.
                with K.unroll(2) as w:
                    _dequant_fp4x8_scaled_e4m3(src_words[w], scale_lo, out_words, w * 2, w * 2 + 1)
                with K.unroll(2) as w:
                    _dequant_fp4x8_scaled_e4m3(
                        src_words[w + 2], scale_hi, out_words, w * 2 + 4, w * 2 + 5
                    )
                with K.unroll(2) as half:
                    K.ptx.st.shared.v4.b32(
                        dst.ptr_to([row, pair_col * 32 + half * 16]),
                        out_words[half * 4],
                        out_words[half * 4 + 1],
                        out_words[half * 4 + 2],
                        out_words[half * 4 + 3],
                    )
        else:
            total_tasks = N_BLOCK * SCALE_COLS
            # ROLLED (`unroll=1`, :1444), for the same reason as the pair arm.
            with K.serial(0, total_tasks // DEQUANT_THREADS, unroll=False) as it:
                task = K.local_scalar(K.i32, init=it * DEQUANT_THREADS + group_tidx, name="task")
                row = K.local_scalar(K.i32, init=udiv_i32(task, SCALE_COLS), name="row")
                scale_col = K.local_scalar(K.i32, init=task - row * SCALE_COLS, name="scale_col")
                if paged:
                    # Same per-iteration re-read as the pair arm (.loc 1 1450 K
                    # / :1625 V, inside $L__BB0_89 / $L__BB0_120).
                    page_idx_s = K.local_scalar(
                        K.i32, init=ld_shared_i32(s_paged_kv_idx, 0), name="page_idx_s"
                    )
                    scale_row = K.local_scalar(
                        K.i32,
                        init=_paged_kv_scale_row(row, head_kv_idx, page_idx_s, num_heads_kv),
                        name="scale_row",
                    )
                else:
                    token = K.local_scalar(K.i32, init=token_base + row, name="token")
                    scale_row = K.local_scalar(
                        K.i32,
                        init=_flat_kv_scale_row(token, head_kv_idx, num_heads_kv),
                        name="scale_row",
                    )
                src_words = K.alloc_local((2,), "uint32")
                K.ptx.ld.shared.v2.b32(src_words[0], src_words[1], src.ptr_to([row, scale_col * 8]))
                combined = K.alloc_local((1,), "uint32")
                _load_scale_bf16x2(
                    scale, _scale_128x4_offset(scale_row, scale_col, SCALE_COLS), combined
                )
                if fold_global:
                    # `cvt_f16x2_f32(g, g, BFloat16)` then `mul_bf16x2`
                    # (:1477-1483): the FP32 tensor scale is broadcast into both
                    # bf16 lanes and folded into the block scale, INSIDE the task
                    # loop -- the load is not hoisted, and the export shows one
                    # ld.global.f32 per task.
                    g = K.alloc_local((1,), "float32")
                    K.ptx.ld.global_.f32(g[0], k_global.ptr_to([0]))
                    g_bf16x2 = K.alloc_local((1,), "uint32")
                    K.ptx.cvt.rn.bf16x2.f32(g_bf16x2[0], g[0], g[0])
                    K.ptx.mul.rn.bf16x2(combined[0], combined[0], g_bf16x2[0])
                out_words = K.alloc_local((8,), "uint32")
                _dequant_fp4x16_to_bf16(src_words, combined[0], out_words)
                with K.unroll(2) as half:
                    K.ptx.st.shared.v4.b32(
                        dst.ptr_to([row, scale_col * 16 + half * 8]),
                        out_words[half * 4],
                        out_words[half * 4 + 1],
                        out_words[half * 4 + 2],
                        out_words[half * 4 + 3],
                    )
        K.ptx.fence.proxy.async_.shared__cta()
        bar_sync_named(bar_id, DEQUANT_THREADS)
        with K.If(group_tidx == 0), K.Then():
            bar_ready.arrive(0)


def _issue_qk(s_slot, q_slot, q_desc, k_desc, kind, operand_dtype, s_col, stage_stride):
    """One QK tile: a chain of same-family ``tcgen05.mma`` over the K extent.

    The A descriptor is advanced by a compile-time 16-byte offset per Q stage
    rather than rebuilt, which is the register-resident `wrap`/`advance` walk
    the source pre-binds in PTX (:1990-1995). Only the first instruction of the
    chain clears the accumulator; the rest accumulate into it.
    """
    mma_k = _MMA_K[kind]
    steps = HEAD_DIM // mma_k
    per_subtile = _SWIZZLE_BYTES // (mma_k * _MMA_ELEM_BYTES[kind])
    with K.unroll(steps) as ki:
        sub = ki // per_subtile
        within = ki % per_subtile
        off = sub * _SUBTILE_16B + within * _MMA_K_16B
        with K.If(K.cuda.elect_sync()), K.Then():
            K.ptx[_MMA_CHAIN[kind]](
                K.cast(s_col + s_slot * stage_stride, "uint32"),
                q_desc.add_16B_offset(q_slot * _stage_16b(operand_dtype) + off),
                k_desc.add_16B_offset(off),
                K.uint32(_instr_desc(operand_dtype)),
                K.uint32(0),
                K.uint32(0),
                K.uint32(0),
                K.uint32(0),
                K.ptx.pred(K.cast(ki != 0, "uint32")),
            )


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
    mma_k = _MMA_K[kind]
    steps = N_BLOCK // mma_k
    split_step = SPLIT_P_ARRIVE // mma_k
    with K.unroll(steps) as ki:
        with K.If(ki == split_step), K.Then():
            bar_last.wait(pv_slot, phase)
        with K.If(K.cuda.elect_sync()), K.Then():
            K.ptx[_MMA_CHAIN[kind]](
                K.cast(o_col + pv_slot * o_stage_stride, "uint32"),
                K.cast(p_col + pv_slot * p_stage_stride + ki * _PV_A_COL_STEP[kind], "uint32"),
                v_desc.add_16B_offset(ki * _PV_K_16B[kind]),
                K.uint32(_instr_desc(operand_dtype, trans_b=True)),
                K.uint32(0),
                K.uint32(0),
                K.uint32(0),
                K.uint32(0),
                K.ptx.pred(K.cast(ki != 0, "uint32")),
            )


def _tcgen05_commit(bar, stage):
    """Publish an MMA result, or release a stage the MMA warp consumed.

    Any pipe half whose arrival the MMA warp issues is signalled with
    ``tcgen05.commit`` rather than ``mbarrier.arrive``, so the arrival is
    ordered behind the MMA -- including the softmax-produced P pipes, whose
    consumer release the export also shows as a commit (:2129-2130).
    """
    with K.If(K.cuda.elect_sync()), K.Then():
        bar.arrive(stage)


def _pack_p_words(words, regs, j, pv_dtype):
    """Convert one 32-element fragment of P into packed words.

    bf16 packs two values per word with ``cvt.rn.bf16x2.f32``; fp8 packs four,
    two at a time through ``cvt.rn.satfinite.e4m3x2.f32`` and then combined.
    Element assignment needs a traced body, so this is ``@K.inline`` with
    ``K.unroll``; every index is arithmetic, so nothing here needs a Python
    loop variable.
    """
    if pv_dtype == "float8_e4m3":
        with K.unroll(8) as w:
            lo = K.alloc_local((1,), "uint16")
            hi = K.alloc_local((1,), "uint16")
            K.ptx.cvt.rn.satfinite.e4m3x2.f32(lo[0], regs[j * 32 + w * 4 + 1], regs[j * 32 + w * 4])
            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                hi[0], regs[j * 32 + w * 4 + 3], regs[j * 32 + w * 4 + 2]
            )
            K.assign(
                words[j * 8 + w],
                K.bitwise_or(
                    K.cast(lo[0], "uint32"), K.shift_left(K.cast(hi[0], "uint32"), K.uint32(16))
                ),
            )
    else:
        with K.unroll(16) as w:
            K.ptx.cvt.rn.bf16x2.f32(
                words[j * 16 + w], regs[j * 32 + w * 2 + 1], regs[j * 32 + w * 2]
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
    out = K.alloc_local((1,), "float32")
    K.ptx.ld.shared.f32(out[0], buffer.ptr_to([index]))
    return out[0]


def st_shared_f32(buffer, index, value):
    """``st.shared.f32``."""
    K.ptx.st.shared.f32(buffer.ptr_to([index]), value)


def _store_o_partial(buf, elem_offset, vals, partial_dtype):
    """One 128-bit ``st.global.cs`` of the partial output.

    fp32 stores four lanes directly; bf16/f16 pack eight into four words; fp8
    packs sixteen into four (:2589-2656).
    """
    if partial_dtype == "float32":
        K.ptx.st.global_.cs.v4.f32(buf.ptr_to([elem_offset]), vals[0], vals[1], vals[2], vals[3])
    elif partial_dtype in ("bfloat16", "float16"):
        words = K.alloc_local((4,), "uint32")
        with K.unroll(4) as w:
            if partial_dtype == "bfloat16":
                K.ptx.cvt.rn.bf16x2.f32(words[w], vals[w * 2 + 1], vals[w * 2])
            else:
                K.ptx.cvt.rn.f16x2.f32(words[w], vals[w * 2 + 1], vals[w * 2])
        K.ptx.st.global_.cs.v4.b32(
            buf.ptr_to([elem_offset]), words[0], words[1], words[2], words[3]
        )
    else:
        words = K.alloc_local((4,), "uint32")
        with K.unroll(4) as w:
            lo = K.alloc_local((1,), "uint16")
            hi = K.alloc_local((1,), "uint16")
            K.ptx.cvt.rn.satfinite.e4m3x2.f32(lo[0], vals[w * 4 + 1], vals[w * 4])
            K.ptx.cvt.rn.satfinite.e4m3x2.f32(hi[0], vals[w * 4 + 3], vals[w * 4 + 2])
            K.assign(
                words[w],
                K.bitwise_or(
                    K.cast(lo[0], "uint32"), K.shift_left(K.cast(hi[0], "uint32"), K.uint32(16))
                ),
            )
        K.ptx.st.global_.cs.v4.b32(
            buf.ptr_to([elem_offset]), words[0], words[1], words[2], words[3]
        )


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
            K.assign(rows[j], q_oob_m_idx)
            with K.If(qi < count_raw), K.Then():
                q_idx = K.bitwise_and(ld_shared_i32(meta, meta_slot + tok_base + j), Q_IDX_MASK)
                K.assign(rows[j], (q_batch_off + q_idx) * num_heads_kv + head_kv_idx)
    elif qheadperkv == 2:
        for t in range(2):
            qi = qi_base + tok_base + t
            base = K.local_scalar("int32", init=q_oob_m_idx * qheadperkv)
            with K.If(qi < count_raw), K.Then():
                q_idx = K.bitwise_and(ld_shared_i32(meta, meta_slot + tok_base + t), Q_IDX_MASK)
                K.assign(base, ((q_batch_off + q_idx) * num_heads_kv + head_kv_idx) * qheadperkv)
            K.assign(rows[t * 2], base)
            K.assign(rows[t * 2 + 1], base + 1)
    else:
        qi = qi_base + tok_base
        base = K.local_scalar("int32", init=q_oob_m_idx * qheadperkv)
        with K.If(qi < count_raw), K.Then():
            q_idx = K.bitwise_and(ld_shared_i32(meta, meta_slot + tok_base), Q_IDX_MASK)
            K.assign(base, ((q_batch_off + q_idx) * num_heads_kv + head_kv_idx) * qheadperkv)
        for j in range(4):
            K.assign(rows[j], base + j)


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
def _make_kernel(**config):
    """Trace one native Kern specialization and its exact launch ABI."""
    qheadperkv = int(config["qhead_per_kv"])
    causal = bool(config.get("causal", True))
    dtype_mode = str(config.get("dtype", "bf16q"))
    partial_dtype = str(config.get("partial_dtype", "float32"))
    temperature = bool(config.get("temperature"))
    k_global = bool(config.get("k_global", True))
    paged = bool(config.get("paged", False))
    seqused = bool(config.get("seqused", False))
    if seqused and not paged:
        raise ValueError("seqused_k is only supported together with page_table")

    num_regs_softmax = NUM_REGS_SOFTMAX_CAUSAL if causal else NUM_REGS_SOFTMAX_NONCAUSAL
    num_regs_store = NUM_REGS_STORE_CAUSAL if causal else NUM_REGS_STORE_NONCAUSAL
    ex2_emu_freq = EX2_EMU_FREQ_CAUSAL if causal else 0

    mode = DTYPE_MODES[dtype_mode]
    q_dtype = mode["q"]
    # `k_dtype = v_dtype = q_dtype` (:333-334): the dequantization target is Q's
    # dtype, so unlike the BF16/FP8 forward there is no separate qk/pv choice
    # and ONE `mma_kind` covers both GEMMs (:347).
    mma_dtype = mode["mma"]
    qk_dtype = mma_dtype
    pv_dtype = mma_dtype
    q_ty = _tirx_dtype(q_dtype)
    qk_ty = _tirx_dtype(qk_dtype)
    pv_ty = _tirx_dtype(pv_dtype)
    partial_ty = _tirx_dtype(partial_dtype)
    qk_mma_kind = _MMA_KIND[qk_dtype]
    pv_mma_kind = _MMA_KIND[pv_dtype]
    q_bytes = _DTYPE_BYTES[q_dtype]
    mma_bytes = _DTYPE_BYTES[mma_dtype]
    # The FP8-Q-only hook: the K tensor scale cannot fold into E4M3 values, so
    # it multiplies the FP32 S accumulator instead (:2482-2491). Under BF16 Q it
    # folds into the dequantized K and this is False.
    s_hook_k_global = q_dtype == "float8_e4m3" and k_global
    fold_k_global = q_dtype != "float8_e4m3" and k_global
    # `q_load_tile` (:427-429): fp8 Q loads a full 128-wide row per token, bf16
    # Q loads two 64-wide k-subtiles.
    q_load_tile = HEAD_DIM if q_bytes == 1 else K_TILE
    q_tokens_per_group = M_BLOCK // qheadperkv

    def host_prelude(params):
        k = params["k"]
        v = params["v"]
        q_flat = params["q_flat"]
        num_heads_kv = params["num_heads_kv"]
        total_k = params["total_k"]
        total_q = params["total_q"]
        head_q = params["head_q"]

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
        k_map = K.stack_alloca("tensormap", 1)
        if paged:
            # Rank 4: the page becomes the outermost mode and the token coordinate
            # is pinned to 0. Load mode, completion, cache hint and the 8192-byte
            # transaction are unchanged from the flat form.
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                k_map,
                "uint8",
                4,
                k.data,
                PACKED_HEAD_DIM,
                N_BLOCK,
                num_heads_kv,
                total_k // N_BLOCK,
                PACKED_HEAD_DIM,
                N_BLOCK * PACKED_HEAD_DIM,
                num_heads_kv * N_BLOCK * PACKED_HEAD_DIM,
                PACKED_HEAD_DIM,
                N_BLOCK,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                _L2_PROMOTION_256B,
                0,
            )
        else:
            K.call_packed(
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
        v_map = K.stack_alloca("tensormap", 1)
        if paged:
            # Rank 4: the page becomes the outermost mode and the token coordinate
            # is pinned to 0. Load mode, completion, cache hint and the 8192-byte
            # transaction are unchanged from the flat form.
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                v_map,
                "uint8",
                4,
                v.data,
                PACKED_HEAD_DIM,
                N_BLOCK,
                num_heads_kv,
                total_k // N_BLOCK,
                PACKED_HEAD_DIM,
                N_BLOCK * PACKED_HEAD_DIM,
                num_heads_kv * N_BLOCK * PACKED_HEAD_DIM,
                PACKED_HEAD_DIM,
                N_BLOCK,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                _L2_PROMOTION_256B,
                0,
            )
        else:
            K.call_packed(
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
        q_map = K.stack_alloca("tensormap", 1)
        K.call_packed(
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

        return k_map, v_map, q_map

    def trace(values, host):
        k_scale = values["k_scale"]
        v_scale = values["v_scale"]
        k_global_scale = values.get("k_global_scale")
        k2q_q_indices = values["k2q_q_indices"]
        k2q_qsplit_indices = values["k2q_qsplit_indices"]
        k2q_row_ptr = values["k2q_row_ptr"]
        scheduler_metadata = values["scheduler_metadata"]
        work_count = values["work_count"]
        o_partial = values["o_partial"]
        lse_partial = values["lse_partial"]
        lse_temperature_partial = values.get("lse_temperature_partial")
        page_table = values.get("page_table")
        seqused_k = values.get("seqused_k")
        cu_seqlens_q = values["cu_seqlens_q"]
        cu_seqlens_k = values["cu_seqlens_k"]
        softmax_scale_log2 = values["softmax_scale_log2"]
        lse_temperature_scale_log2 = values["lse_temperature_scale_log2"]
        lse_temperature_inv_scale = values["lse_temperature_inv_scale"]
        num_kv_blocks = values["num_kv_blocks"]
        num_heads_kv = values["num_heads_kv"]
        seq_len_q = values["seq_len_q"]
        work_capacity = values["work_capacity"]
        total_k = values["total_k"]
        total_q = values["total_q"]
        head_q = values["head_q"]
        nnz = values["nnz"]
        total_rows = values["total_rows"]
        num_batches = values["num_batches"]
        topk = values["topk"]
        scale_numel = values["scale_numel"]
        k_map, v_map, q_map = host

        # CUDA TRANSCRIPTION START
        block = K.cta_id()
        tidx = K.thread_id()
        # `make_warp_uniform(warp_idx())` (:657) lowers to a lane-0 shfl broadcast.
        # `K.warp_id` is warp-uniform by construction, so the broadcast is redundant
        # here rather than load-bearing.
        warp_idx = K.warp_id()
        lane_idx = K.lane_id()

        # Work-item decode and the CTA-level early-out (:660-688).
        # The grid is sized by the work list's CAPACITY, so the tail CTAs retire.
        work_count_val = K.local_scalar(
            K.i32, init=ld_global_i32(work_count, 0), name="work_count_val"
        )
        cta_valid_work = K.local_scalar(
            K.i32, init=K.cast(block < work_count_val, "int32"), name="cta_valid_work"
        )
        head_kv_idx = K.alloc_local((1,), "int32")
        row_linear = K.alloc_local((1,), "int32")
        work_q_begin = K.alloc_local((1,), "int32")
        work_q_count = K.alloc_local((1,), "int32")
        batch_idx = K.alloc_local((1,), "int32")
        kv_block_idx = K.alloc_local((1,), "int32")
        K.assign(head_kv_idx[0], 0)
        K.assign(row_linear[0], 0)
        K.assign(work_q_begin[0], 0)
        K.assign(work_q_count[0], 0)
        K.assign(batch_idx[0], 0)
        K.assign(kv_block_idx[0], 0)
        with K.If(cta_valid_work != 0), K.Then():
            K.assign(head_kv_idx[0], ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 0))
            K.assign(row_linear[0], ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 1))
            K.assign(work_q_begin[0], ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 2))
            K.assign(work_q_count[0], ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 3))
            K.assign(batch_idx[0], ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 4))
            K.assign(kv_block_idx[0], ld_global_i32(scheduler_metadata, block * WORK_FIELDS + 5))

        # -----------------------------------------------------------------------
        # Shared memory. The source's `@cute.struct SharedStorage` reads static but
        # is handed out by `SmemAllocator` from the dynamic pool; the export carries
        # `.extern .shared .align 1024 .b8 __dynamic_shmem__0[]`, so the matching
        # TIRx form is a pool allocation under `tirx.use_dyn_shared_memory`.
        # -----------------------------------------------------------------------
        smem = K.smem_pool()
        pool = smem.pool
        s_k = smem.alloc((N_BLOCK, HEAD_DIM), qk_ty, swizzle=K.SW128B).buf
        # Same shape as sK: the TMA box's fastest dim is head_dim, so the tile
        # lands (kv rows, head_dim cols) with head_dim contiguous. For the PV
        # B operand that means N = head_dim is the contiguous axis, which is what
        # "MN-major" names (:386-387).
        s_v = smem.alloc((N_BLOCK, HEAD_DIM), pv_ty, swizzle=K.SW128B).buf
        s_q = smem.alloc((Q_STAGE * M_BLOCK, HEAD_DIM), q_ty, swizzle=K.SW128B).buf
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
        if temperature:
            s_scale_temp = pool.alloc((O_STAGE * M_BLOCK,), "float32", align=16)
        s_split_idx = pool.alloc((O_STAGE * q_tokens_per_group,), "int32", align=16)
        s_q_idx = pool.alloc((O_STAGE * q_tokens_per_group,), "int32", align=16)
        s_row_meta = pool.alloc((8,), "int32", align=16)
        s_diag_q_count = pool.alloc((1,), "int32", align=16)
        # `sPagedKvIdx` (:552): the physical page for this CTA's KV block. Allocated
        # unconditionally, as the source does, so flat and paged share every later
        # offset. Written by thread 0; read by both KV-load warps AND, on the paged
        # path, once per iteration inside each dequant task loop.
        s_paged_kv_idx = pool.alloc((1,), "int32", align=16)
        s_q_load_m_idx = pool.alloc((Q_STAGE * q_tokens_per_group,), "int32", align=16)
        s_qidx_meta = pool.alloc((QIDX_META_STAGES * q_tokens_per_group,), "int32", align=16)

        bar_k = K.MBarrier(pool, 1)
        bar_v = K.MBarrier(pool, 1)
        # Unconditional, for the same reason the staging tiles are.
        bar_k_tma = K.MBarrier(pool, 1)
        bar_v_tma = K.MBarrier(pool, 1)
        bar_q_full = K.TMABar(pool, Q_STAGE)
        bar_q_empty = K.TCGen05Bar(pool, Q_STAGE)
        bar_s_full = K.TCGen05Bar(pool, S_STAGE)
        bar_s_empty = K.MBarrier(pool, S_STAGE)
        bar_p_full = K.MBarrier(pool, S_STAGE)
        bar_p_empty = K.TCGen05Bar(pool, S_STAGE)
        bar_p_last_full = K.MBarrier(pool, S_STAGE)
        bar_p_last_empty = K.TCGen05Bar(pool, S_STAGE)
        bar_o_full = K.TCGen05Bar(pool, O_STAGE)
        bar_o_empty = K.MBarrier(pool, O_STAGE)
        # Only the empty half of the stats pipe is live upstream: it is a credit on
        # `s_scale`, and nothing ever commits or waits on its full half (:2331,
        # :3019). The full half is still allocated so the word layout matches.
        bar_stats_full = K.MBarrier(pool, O_STAGE)
        bar_stats_empty = K.MBarrier(pool, O_STAGE)
        tmem_start_addr = pool.alloc((1,), "uint32", align=4)

        # TMEM is represented by raw tcgen05 column operands in Kern. The source
        # allocator's map is S0/S1 [0, 2*N_BLOCK), then O0/O1.
        tmem_s_col = 0
        tmem_o_col = 2 * N_BLOCK
        tmem_stage_stride = N_BLOCK
        tmem_o_stage_stride = HEAD_DIM
        # P overlays the upper columns of each S tile (:369-371): the P store
        # destroys the tail of S, which is safe only because row_max and row_sum are
        # already out of it by then.
        p_width = _DTYPE_BYTES[pv_dtype] * 8
        tmem_s_to_p = N_BLOCK - N_BLOCK * p_width // 32
        tmem_p_col = tmem_s_col + tmem_s_to_p

        # -----------------------------------------------------------------------
        # Prologue: descriptor prefetch, thread-0 metadata publish, barrier init.
        # -----------------------------------------------------------------------
        with K.If(warp_idx == 0), K.Then():
            # ONLY the Q descriptor (:690-696). K and V are never prefetched -- the
            # export carries exactly one `prefetch.tensormap`. The two Q programs
            # also differ in election: the CUTE-atom (TMA-Q) arm issues it
            # unelected, the raw-descriptor gather4 arm elects one lane.
            if USE_GATHER4(qheadperkv):
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(q_map))
            else:
                K.ptx.prefetch.tensormap(K.address_of(q_map))

        # WARP 0, every lane, NO elect, and BEFORE the thread-0 metadata block
        # (:751-806). `mbarrier.init` is idempotent across the warp's 32 lanes, so
        # the reference does not elect, and the export shows the 24 inits inside a
        # plain `@%p bra` on the warp index with no `elect.sync`.
        with K.If(warp_idx == 0), K.Then():
            with K.unroll(Q_STAGE) as stage:
                K.ptx.mbarrier.init.shared.b64(bar_q_full.ptr_to([stage]), K.uint32(1))
                K.ptx.mbarrier.init.shared.b64(bar_q_empty.ptr_to([stage]), K.uint32(1))
            with K.unroll(S_STAGE) as stage:
                K.ptx.mbarrier.init.shared.b64(bar_s_full.ptr_to([stage]), K.uint32(1))
                K.ptx.mbarrier.init.shared.b64(
                    bar_s_empty.ptr_to([stage]), K.uint32(SOFTMAX_THREADS)
                )
                K.ptx.mbarrier.init.shared.b64(
                    bar_p_full.ptr_to([stage]), K.uint32(SOFTMAX_THREADS)
                )
                K.ptx.mbarrier.init.shared.b64(bar_p_empty.ptr_to([stage]), K.uint32(1))
                K.ptx.mbarrier.init.shared.b64(
                    bar_p_last_full.ptr_to([stage]), K.uint32(SOFTMAX_THREADS)
                )
                K.ptx.mbarrier.init.shared.b64(bar_p_last_empty.ptr_to([stage]), K.uint32(1))
            with K.unroll(O_STAGE) as stage:
                K.ptx.mbarrier.init.shared.b64(bar_o_full.ptr_to([stage]), K.uint32(1))
                K.ptx.mbarrier.init.shared.b64(
                    bar_o_empty.ptr_to([stage]), K.uint32(SOFTMAX_THREADS)
                )
                K.ptx.mbarrier.init.shared.b64(
                    bar_stats_full.ptr_to([stage]), K.uint32(SOFTMAX_THREADS)
                )
                K.ptx.mbarrier.init.shared.b64(
                    bar_stats_empty.ptr_to([stage]), K.uint32(SOFTMAX_THREADS)
                )

        # The pipeline init has its OWN fence and CTA barrier (:808-809); the
        # thread-0 metadata block below closes with a second pair (:891-892). Two
        # pairs, not one: the export carries `fence.mbarrier_init` twice.
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(tidx == 0), K.Then():
            base_row_start = K.local_scalar(
                K.i32,
                init=ld_global_i32(k2q_row_ptr, head_kv_idx[0] * (total_rows + 1) + row_linear[0]),
                name="base_row_start",
            )
            row_start = K.local_scalar(
                K.i32, init=base_row_start + work_q_begin[0], name="row_start"
            )
            count_raw = K.local_scalar(K.i32, init=work_q_count[0], name="count_raw")
            # Upstream reads this as `mPageTable.shape[1]` (:211). The table is
            # rectangular because `page_size == blk_kv`, so `num_batches *
            # pages_per_seq == num_pages == total_k / page_size`, and the quotient
            # of two ABI scalars recovers it without a new argument.
            if paged:
                pages_per_seq = K.local_scalar(
                    K.i32,
                    init=udiv_i32(udiv_i32(total_k, N_BLOCK), num_batches),
                    name="pages_per_seq",
                )
            # `_logical_seqlen_k` (:205-216) in the source's priority order:
            # `seqused` FIRST, then the paged capacity, then the cu_seqlens
            # difference. Testing paged first would substitute the zero-padded
            # capacity for a shorter supplied length.
            if seqused:
                seqlen_k = K.local_scalar(
                    K.i32, init=ld_global_i32(seqused_k, batch_idx[0]), name="seqlen_k"
                )
            elif paged:
                seqlen_k = K.local_scalar(K.i32, init=pages_per_seq * N_BLOCK, name="seqlen_k")
            else:
                seqlen_k = K.local_scalar(
                    K.i32,
                    init=ld_global_i32(cu_seqlens_k, batch_idx[0] + 1)
                    - ld_global_i32(cu_seqlens_k, batch_idx[0]),
                    name="seqlen_k",
                )
            kv_valid_cols = K.local_scalar(
                K.i32,
                init=K.min(K.max(seqlen_k - kv_block_idx[0] * N_BLOCK, 0), N_BLOCK),
                name="kv_valid_cols",
            )
            q_batch_offset = K.local_scalar(
                K.i32, init=ld_global_i32(cu_seqlens_q, batch_idx[0]), name="q_batch_offset"
            )
            k_batch_offset = K.local_scalar(K.i32, init=K.int32(0), name="k_batch_offset")
            if not paged:
                K.assign(k_batch_offset, ld_global_i32(cu_seqlens_k, batch_idx[0]))
            # `seqlen_q` and therefore `causal_q_offset` are needed BEFORE the
            # second store, because the eight words go out as TWO four-word vector
            # stores (:849, :863), not eight scalars.
            # Computed only under `causal` (:893-901 analogue). Deliberately
            # unclamped: with `seqused_k` shorter than the Q length the offset goes
            # negative and the leading queries store the neutral partial, O = 0 with
            # LSE = -inf. Clamping would diverge from the source.
            causal_q_offset_l = K.alloc_local((1,), "int32")
            K.assign(causal_q_offset_l[0], 0)
            if causal:
                seqlen_q = K.local_scalar(
                    K.i32,
                    init=ld_global_i32(cu_seqlens_q, batch_idx[0] + 1) - q_batch_offset,
                    name="seqlen_q",
                )
                K.assign(causal_q_offset_l[0], seqlen_k - seqlen_q)
            causal_q_offset = K.local_scalar(
                K.i32, init=causal_q_offset_l[0], name="causal_q_offset"
            )
            K.ptx.st.shared.v4.b32(
                s_row_meta.ptr_to([0]),
                K.reinterpret("uint32", batch_idx[0]),
                K.reinterpret("uint32", kv_block_idx[0]),
                K.reinterpret("uint32", row_start),
                K.reinterpret("uint32", count_raw),
            )
            K.ptx.st.shared.v4.b32(
                s_row_meta.ptr_to([4]),
                K.reinterpret("uint32", kv_valid_cols),
                K.reinterpret("uint32", q_batch_offset),
                K.reinterpret("uint32", k_batch_offset),
                K.reinterpret("uint32", causal_q_offset),
            )

            # The causal diagonal split point: a 32-step binary search over the CSR
            # row, which is sorted by q_idx (:259-285, :919-935). The export keeps
            # the loop rolled with exactly one probe load in the body.
            # Computed ONLY under `const_expr(self.causal)` (:919-934): a non-causal
            # build emits no search at all and leaves the count at zero, which is
            # also the only value its consumer would accept -- a search result there
            # would drive the diagonal mask over rows that have no diagonal.
            diag_q_count = K.alloc_local((1,), "int32")
            K.assign(diag_q_count[0], 0)
            with K.If(K.And(K.And(causal, count_raw > 0), kv_valid_cols > 0)), K.Then():
                q_threshold = K.local_scalar(
                    K.i32,
                    init=(kv_block_idx[0] * N_BLOCK + kv_valid_cols) - causal_q_offset,
                    name="q_threshold",
                )
                lo = K.alloc_local((1,), "int32")
                hi = K.alloc_local((1,), "int32")
                K.assign(lo[0], 0)
                K.assign(hi[0], count_raw)
                # `_lower_bound_q_idx` (:264-286): a FIXED 32-trip loop with
                # `unroll=1` and a predicated body, not a data-dependent `while`.
                # 32 is an upper bound on any int32-sized row rather than a trip
                # count the search runs to convergence, and the export keeps it
                # rolled with exactly one probe load in the body.
                with K.serial(0, 32, unroll=False) as _:
                    with K.If(lo[0] < hi[0]), K.Then():
                        mid = K.local_scalar(K.i32, init=udiv_i32(lo[0] + hi[0], 2), name="mid")
                        probe = K.local_scalar(
                            K.i32,
                            init=ld_global_i32(
                                k2q_q_indices, head_kv_idx[0] * nnz + row_start + mid
                            ),
                            name="probe",
                        )
                        with K.If(probe < q_threshold):
                            with K.Then():
                                K.assign(lo[0], mid + 1)
                            with K.Else():
                                K.assign(hi[0], mid)
                K.assign(diag_q_count[0], lo[0])
            st_shared_i32(s_diag_q_count, 0, diag_q_count[0])

            bar_k.init(1)
            bar_v.init(1)
            bar_k_tma.init(1)
            bar_v_tma.init(1)
            # 8192 bytes per tile: 128 tokens x 64 packed bytes, half what the same
            # tile costs unpacked.
            _mbar_expect_tx(bar_k_tma, 0, N_BLOCK * PACKED_HEAD_DIM)
            _mbar_expect_tx(bar_v_tma, 0, N_BLOCK * PACKED_HEAD_DIM)

        # The physical page for this CTA's KV block (:864-867). Warp 1, not thread 0:
        # it depends on nothing the metadata chain produces -- `batch_idx` and
        # `kv_block_idx` are read from the scheduler record by every thread, and
        # `pages_per_seq` is two divisions of ABI scalars -- while thread 0 is busy
        # with a 32-step binary search of dependent global loads. Behind that chain
        # its own global load lands on the critical path of every CTA, which the
        # long shapes amortize and a 10 us one does not. The publish still rides the
        # same fence and CTA barrier below, so visibility is unchanged and paging
        # still adds no barrier of its own.
        if paged:
            with K.If(K.And(warp_idx == 1, cta_valid_work != 0)), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    pages_per_seq_w1 = K.local_scalar(
                        K.i32,
                        init=udiv_i32(udiv_i32(total_k, N_BLOCK), num_batches),
                        name="pages_per_seq_w1",
                    )
                    st_shared_i32(
                        s_paged_kv_idx,
                        0,
                        ld_global_i32(
                            page_table, batch_idx[0] * pages_per_seq_w1 + kv_block_idx[0]
                        ),
                    )

        # The metadata block's own fence and CTA barrier (:891-892) -- the second
        # of the two pairs. It orders the sRowMeta / sDiagQCount publish and the
        # K/V transaction counts against every consumer.
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        # -----------------------------------------------------------------------
        # Role dispatch. A flat sequence of independent `if` blocks, each ANDed
        # with `cta_valid_work` -- not an if/elif chain (:943-1191).
        # -----------------------------------------------------------------------
        with K.If(warp_idx == TOTAL_WARPS - 1), K.Then():
            # Warp 15 is idle and is NOT gated on cta_valid_work (:991-992).
            K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(NUM_REGS_OTHER))

        # -----------------------------------------------------------------------
        # ROLE: Q-load warpgroup, warps 8..11 (:1014-1047).
        # -----------------------------------------------------------------------
        with (
            K.If(K.And(tidx >= Q_LOAD_WARP_BASE * WARP_SIZE, tidx < MMA_WARP_ID * WARP_SIZE)),
            K.Then(),
        ):
            with K.If(cta_valid_work != 0), K.Then():
                K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(num_regs_store))
                q_row_start = K.local_scalar(
                    K.i32, init=ld_shared_i32(s_row_meta, 2), name="q_row_start"
                )
                q_count_raw = K.local_scalar(
                    K.i32, init=ld_shared_i32(s_row_meta, 3), name="q_count_raw"
                )
                q_batch_off = K.local_scalar(
                    K.i32, init=ld_shared_i32(s_row_meta, 5), name="q_batch_off"
                )
                # Deliberately NOT gated on KV validity: a sparse entry past the
                # sequence still runs the all-masked path so its partial is neutral
                # (:1007-1009).
                with K.If(q_count_raw > 0), K.Then():
                    num_q_groups_load = K.local_scalar(
                        K.i32,
                        init=uceil_div_i32(q_count_raw, q_tokens_per_group),
                        name="num_q_groups_load",
                    )
                    warp_in_wg = K.local_scalar(
                        K.i32, init=warp_idx - Q_LOAD_WARP_BASE, name="warp_in_wg"
                    )
                    # `q_oob_m_idx = mQ_2d.shape[0] // qheadperkv` (:1855) -- one
                    # past the last Q *tile*, so an absent token gathers out of
                    # bounds and takes the descriptor's OOB fill. `total_q` alone
                    # would be an in-range row whenever num_heads_kv > 1.
                    q_oob_m_idx = K.local_scalar(
                        K.i32, init=total_q * num_heads_kv, name="q_oob_m_idx"
                    )

                    if not USE_GATHER4(qheadperkv):
                        with K.serial(0, num_q_groups_load, unroll=False) as qi_group:
                            slot = K.local_scalar(K.i32, init=qi_group % Q_STAGE, name="slot")
                            phase = K.local_scalar(
                                K.i32, init=udiv_i32(qi_group, Q_STAGE) & 1, name="phase"
                            )
                            with K.If(warp_in_wg == 0), K.Then():
                                bar_q_empty.wait(slot, phase ^ 1)
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        bar_q_full.ptr_to([slot]),
                                        K.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                    )
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                            load_meta_slot = K.local_scalar(
                                K.i32, init=slot * q_tokens_per_group, name="load_meta_slot"
                            )
                            qidx_meta_slot = K.local_scalar(
                                K.i32,
                                init=(qi_group & (QIDX_META_STAGES - 1)) * q_tokens_per_group,
                                name="qidx_meta_slot",
                            )
                            # One warp's low lanes publish the whole group: with
                            # qheadperkv >= 8 a group is at most 16 tokens
                            # (:1883-1898).
                            with (
                                K.If(K.And(warp_in_wg == 0, lane_idx < q_tokens_per_group)),
                                K.Then(),
                            ):
                                qi = K.local_scalar(
                                    K.i32, init=qi_group * q_tokens_per_group + lane_idx, name="qi"
                                )
                                with K.If(qi < q_count_raw):
                                    with K.Then():
                                        word = K.local_scalar(
                                            K.i32,
                                            init=ld_global_i32(
                                                k2q_qsplit_indices,
                                                head_kv_idx[0] * nnz + q_row_start + qi,
                                            ),
                                            name="word",
                                        )
                                        st_shared_i32(s_qidx_meta, qidx_meta_slot + lane_idx, word)
                                        st_shared_i32(
                                            s_q_load_m_idx,
                                            load_meta_slot + lane_idx,
                                            (q_batch_off + K.bitwise_and(word, Q_IDX_MASK))
                                            * num_heads_kv
                                            + head_kv_idx[0],
                                        )
                                    with K.Else():
                                        st_shared_i32(s_qidx_meta, qidx_meta_slot + lane_idx, 0)
                                        st_shared_i32(
                                            s_q_load_m_idx, load_meta_slot + lane_idx, q_oob_m_idx
                                        )
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                            with K.unroll(TOKENS_PER_WARP(qheadperkv)) as qi_slot:
                                tok = K.local_scalar(
                                    K.i32,
                                    init=warp_in_wg * TOKENS_PER_WARP(qheadperkv) + qi_slot,
                                    name="tok",
                                )
                                with K.If(tok < q_tokens_per_group), K.Then():
                                    m_tile = K.local_scalar(
                                        K.i32,
                                        init=ld_shared_i32(s_q_load_m_idx, load_meta_slot + tok),
                                        name="m_tile",
                                    )
                                    with K.If(K.cuda.elect_sync()), K.Then():
                                        with K.unroll(Q_SUBTILES(q_bytes)) as ks:
                                            K.ptx[_TMA_G2S_2D_CACHE](
                                                K.ptr_byte_offset(
                                                    s_q.ptr_to([0, 0]),
                                                    slot * M_BLOCK * HEAD_DIM * q_bytes
                                                    + ks * M_BLOCK * q_load_tile * q_bytes
                                                    + tok * qheadperkv * q_load_tile * q_bytes,
                                                    q_ty,
                                                ),
                                                K.address_of(q_map),
                                                K.int32(ks * q_load_tile),
                                                m_tile * qheadperkv,
                                                K.cuda.cvta_generic_to_shared(
                                                    bar_q_full.ptr_to([slot])
                                                ),
                                                _TMA_NO_POLICY,
                                            )

                        with K.If(warp_in_wg == 0), K.Then():
                            # One acquire past the end leaves the ring's empty half
                            # in the state the next work item expects (:1930-1935).
                            bar_q_empty.wait(
                                num_q_groups_load % Q_STAGE,
                                (udiv_i32(num_q_groups_load, Q_STAGE) & 1) ^ 1,
                            )
                            with K.If(K.cuda.elect_sync()), K.Then():
                                K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                    bar_q_full.ptr_to([num_q_groups_load % Q_STAGE]),
                                    K.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                )
                    else:
                        # ---- gather4 Q path, qheadperkv in {1, 2, 4} (:1624-1817) ----
                        # Each gather4 pulls four GMEM rows into one box, so a
                        # 128-row Q tile takes 8 gathers per warp.
                        gathers_per_warp = M_BLOCK // (NUM_Q_LOAD_WARPS * 4)
                        tokens_per_gather4 = 4 // qheadperkv
                        meta_iters = (q_tokens_per_group + NUM_Q_LOAD_WARPS * WARP_SIZE - 1) // (
                            NUM_Q_LOAD_WARPS * WARP_SIZE
                        )
                        with K.serial(0, num_q_groups_load, unroll=False) as qi_group:
                            slot = K.local_scalar(K.i32, init=qi_group % Q_STAGE, name="slot")
                            phase = K.local_scalar(
                                K.i32, init=udiv_i32(qi_group, Q_STAGE) & 1, name="phase"
                            )
                            with K.If(warp_in_wg == 0), K.Then():
                                bar_q_empty.wait(slot, phase ^ 1)
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        bar_q_full.ptr_to([slot]),
                                        K.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                    )
                            # Issue the qsplit reads BEFORE the barrier and store
                            # them after it. The barrier only orders the shared
                            # writes against last iteration's readers; the reads
                            # themselves come from a read-only global input and have
                            # no ordering requirement, so their latency can be spent
                            # inside the wait instead of in front of the store.
                            # Leaving them where the reference puts them costs a
                            # dependent global load three instructions ahead of the
                            # store, and that store carries 517 stall samples -- the
                            # heaviest single STS site in this kernel.
                            qsplit_word = K.alloc_local((meta_iters,), "int32")
                            with K.unroll(meta_iters) as meta_iter:
                                tok_pre = K.local_scalar(
                                    K.i32,
                                    init=(meta_iter * NUM_Q_LOAD_WARPS + warp_in_wg) * WARP_SIZE
                                    + lane_idx,
                                    name="tok_pre",
                                )
                                K.assign(qsplit_word[meta_iter], 0)
                                with K.If(tok_pre < q_tokens_per_group), K.Then():
                                    qi_pre = K.local_scalar(
                                        K.i32,
                                        init=qi_group * q_tokens_per_group + tok_pre,
                                        name="qi_pre",
                                    )
                                    with K.If(qi_pre < q_count_raw), K.Then():
                                        K.assign(
                                            qsplit_word[meta_iter],
                                            ld_global_i32(
                                                k2q_qsplit_indices,
                                                head_kv_idx[0] * nnz + q_row_start + qi_pre,
                                            ),
                                        )
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                            qidx_meta_slot = K.local_scalar(
                                K.i32,
                                init=K.bitwise_and(qi_group, QIDX_META_STAGES - 1)
                                * q_tokens_per_group,
                                name="qidx_meta_slot",
                            )
                            # This path's groups hold 32, 64 or 128 tokens, so the
                            # publish takes several sweeps of the whole warpgroup
                            # (:1671-1690).
                            with K.unroll(meta_iters) as meta_iter:
                                tok_g4 = K.local_scalar(
                                    K.i32,
                                    init=(meta_iter * NUM_Q_LOAD_WARPS + warp_in_wg) * WARP_SIZE
                                    + lane_idx,
                                    name="tok_g4",
                                )
                                with K.If(tok_g4 < q_tokens_per_group), K.Then():
                                    st_shared_i32(
                                        s_qidx_meta, qidx_meta_slot + tok_g4, qsplit_word[meta_iter]
                                    )
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                            with K.If(K.cuda.elect_sync()), K.Then():
                                with K.unroll(gathers_per_warp) as gather_slot:
                                    gather_idx = K.local_scalar(
                                        K.i32,
                                        init=gather_slot * NUM_Q_LOAD_WARPS + warp_in_wg,
                                        name="gather_idx",
                                    )
                                    tok_base = K.local_scalar(
                                        K.i32, init=gather_idx * tokens_per_gather4, name="tok_base"
                                    )
                                    rows = K.alloc_local((4,), "int32")
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
                                    with K.unroll(Q_SUBTILES(q_bytes)) as ks:
                                        with K.If(ks + 1 < Q_SUBTILES(q_bytes)), K.Then():
                                            K.ptx[_TMA_GATHER4_PREFETCH](
                                                K.address_of(q_map),
                                                K.int32((ks + 1) * q_load_tile),
                                                rows[0],
                                                rows[1],
                                                rows[2],
                                                rows[3],
                                                _GATHER4_Q_CACHE_HINT,
                                            )
                                        K.ptx[_TMA_GATHER4_2D_CACHE](
                                            K.ptr_byte_offset(
                                                s_q.ptr_to([0, 0]),
                                                slot * M_BLOCK * HEAD_DIM * q_bytes
                                                + ks * M_BLOCK * q_load_tile * q_bytes
                                                + gather_idx * 4 * q_load_tile * q_bytes,
                                                q_ty,
                                            ),
                                            K.address_of(q_map),
                                            K.int32(ks * q_load_tile),
                                            rows[0],
                                            rows[1],
                                            rows[2],
                                            rows[3],
                                            K.cuda.cvta_generic_to_shared(
                                                bar_q_full.ptr_to([slot])
                                            ),
                                            _GATHER4_Q_CACHE_HINT,
                                        )
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                        with K.If(warp_in_wg == 0), K.Then():
                            bar_q_empty.wait(
                                num_q_groups_load % Q_STAGE,
                                (udiv_i32(num_q_groups_load, Q_STAGE) & 1) ^ 1,
                            )
                            with K.If(K.cuda.elect_sync()), K.Then():
                                K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                    bar_q_full.ptr_to([num_q_groups_load % Q_STAGE]),
                                    K.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                )

        # -----------------------------------------------------------------------
        # ROLE: KV load, warps 13 and 14. One TMA each; there is no KV ring
        # (:1064-1086, :1676-1777).
        # -----------------------------------------------------------------------
        with (
            K.If(
                K.And(
                    warp_idx >= KV_LOAD_WARP_BASE, warp_idx < KV_LOAD_WARP_BASE + NUM_KV_LOAD_WARPS
                )
            ),
            K.Then(),
        ):
            with K.If(cta_valid_work != 0), K.Then():
                K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(NUM_REGS_OTHER))
                kv_block_load = K.local_scalar(
                    K.i32, init=ld_shared_i32(s_row_meta, 1), name="kv_block_load"
                )
                k_batch_off = K.local_scalar(
                    K.i32, init=ld_shared_i32(s_row_meta, 6), name="k_batch_off"
                )
                kv_has_work = K.local_scalar(
                    K.i32,
                    init=K.cast(ld_shared_i32(s_row_meta, 3) > 0, "int32"),
                    name="kv_has_work",
                )
                with K.If(kv_has_work != 0), K.Then():
                    kv_row_start = K.local_scalar(
                        K.i32, init=k_batch_off + kv_block_load * N_BLOCK, name="kv_row_start"
                    )
                    with K.If(warp_idx == KV_LOAD_WARP_BASE), K.Then():
                        # ONE issue per tile, not two: the packed row is 64 bytes,
                        # inside a single 128-byte box. The copy and the arrive sit
                        # in two separate `elect.sync` regions, as the export shows
                        # (:1734-1740).
                        # Read warp-wide in this warp's own arm, before the
                        # copy's elect.sync -- the export shows all 32 lanes
                        # executing it (:1709 for K, :1744 for V).
                        page_idx_k = K.alloc_local((1,), "int32")
                        if paged:
                            K.assign(page_idx_k[0], ld_shared_i32(s_paged_kv_idx, 0))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            if paged:
                                K.ptx[_TMA_G2S_4D_CACHE](
                                    s_k_fp4.ptr_to([0, 0]),
                                    K.address_of(k_map),
                                    K.int32(0),
                                    K.int32(0),
                                    head_kv_idx[0],
                                    page_idx_k[0],
                                    K.cuda.cvta_generic_to_shared(bar_k_tma.ptr_to([0])),
                                    _TMA_NO_POLICY,
                                )
                            else:
                                K.ptx[_TMA_G2S_3D_CACHE](
                                    s_k_fp4.ptr_to([0, 0]),
                                    K.address_of(k_map),
                                    K.int32(0),
                                    head_kv_idx[0],
                                    kv_row_start,
                                    K.cuda.cvta_generic_to_shared(bar_k_tma.ptr_to([0])),
                                    _TMA_NO_POLICY,
                                )
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                                bar_k_tma.ptr_to([0]), K.uint32(1)
                            )
                    with K.If(warp_idx == KV_LOAD_WARP_BASE + 1), K.Then():
                        # Read warp-wide in this warp's own arm, before the
                        # copy's elect.sync -- the export shows all 32 lanes
                        # executing it (:1709 for K, :1744 for V).
                        page_idx_v = K.alloc_local((1,), "int32")
                        if paged:
                            K.assign(page_idx_v[0], ld_shared_i32(s_paged_kv_idx, 0))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            if paged:
                                K.ptx[_TMA_G2S_4D_CACHE](
                                    s_v_fp4.ptr_to([0, 0]),
                                    K.address_of(v_map),
                                    K.int32(0),
                                    K.int32(0),
                                    head_kv_idx[0],
                                    page_idx_v[0],
                                    K.cuda.cvta_generic_to_shared(bar_v_tma.ptr_to([0])),
                                    _TMA_NO_POLICY,
                                )
                            else:
                                K.ptx[_TMA_G2S_3D_CACHE](
                                    s_v_fp4.ptr_to([0, 0]),
                                    K.address_of(v_map),
                                    K.int32(0),
                                    head_kv_idx[0],
                                    kv_row_start,
                                    K.cuda.cvta_generic_to_shared(bar_v_tma.ptr_to([0])),
                                    _TMA_NO_POLICY,
                                )
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                                bar_v_tma.ptr_to([0]), K.uint32(1)
                            )
                    # Unconditional here: both tiles always need dequantizing, so
                    # the two load warps always close on this barrier (:1777).
                    bar_sync_named(BAR_KV_LOAD, WARP_SIZE * NUM_KV_LOAD_WARPS)

        # -----------------------------------------------------------------------
        # ROLE: the single MMA-issue warp, warp 12; also the TMEM allocator warp
        # (:1035-1057, :2189-2436).
        # -----------------------------------------------------------------------
        with K.If(warp_idx == MMA_WARP_ID), K.Then():
            with K.If(cta_valid_work != 0), K.Then():
                K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(NUM_REGS_OTHER))
                K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    K.address_of(tmem_start_addr[0]), K.uint32(TMEM_TOTAL)
                )
                # The retrieve barrier spans both softmax warpgroups plus this warp
                # (:1041-1042). On the fp8 paths that also makes the first QK wait for
                # the shared-memory dequantization, without a separate edge.
                bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

                mma_count_raw = K.local_scalar(
                    K.i32, init=ld_shared_i32(s_row_meta, 3), name="mma_count_raw"
                )
                with K.If(mma_count_raw > 0), K.Then():
                    num_q_groups_mma = K.local_scalar(
                        K.i32,
                        init=uceil_div_i32(mma_count_raw, q_tokens_per_group),
                        name="num_q_groups_mma",
                    )

                    # Operand descriptors. The Q ring is walked by adding a
                    # compile-time 16-byte offset to the A descriptor rather than
                    # rebuilding it, which is what the source's `wrap`/`advance`
                    # pre-bound partials do in PTX registers (:1990-1995).
                    q_desc = K.SmemDescriptor()
                    q_desc.init(s_q.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                    q_desc.make_lo_uniform()
                    k_desc = K.SmemDescriptor()
                    k_desc.init(s_k.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                    k_desc.make_lo_uniform()
                    v_desc = K.SmemDescriptor()
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

                    with K.If(num_q_groups_mma > 1), K.Then():
                        bar_q_full.wait(1, 0)
                        bar_s_empty.wait(1, 1)
                        _issue_qk(
                            1,
                            1,
                            q_desc,
                            k_desc,
                            qk_mma_kind,
                            qk_dtype,
                            tmem_s_col,
                            tmem_stage_stride,
                        )
                        _tcgen05_commit(bar_s_full, 1)
                        _tcgen05_commit(bar_q_empty, 1)

                    # V is waited only after the two prologue QKs, so its TMA
                    # overlaps them (:2094).
                    bar_v.wait(0, 0)

                    with K.serial(2, num_q_groups_mma, unroll=False) as qi:
                        pv_qi = K.local_scalar(K.i32, init=qi - 2, name="pv_qi")
                        pv_slot = K.local_scalar(
                            K.i32, init=K.bitwise_and(pv_qi, 1), name="pv_slot"
                        )
                        pv_phase = K.local_scalar(
                            K.i32, init=udiv_i32(pv_qi, 2) & 1, name="pv_phase"
                        )
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

                        q_slot = K.local_scalar(K.i32, init=qi % Q_STAGE, name="q_slot")
                        q_phase = K.local_scalar(
                            K.i32, init=udiv_i32(qi, Q_STAGE) & 1, name="q_phase"
                        )
                        s_slot = K.local_scalar(K.i32, init=K.bitwise_and(qi, 1), name="s_slot")
                        s_phase = K.local_scalar(K.i32, init=udiv_i32(qi, 2) & 1, name="s_phase")
                        bar_q_full.wait(q_slot, q_phase)
                        bar_s_empty.wait(s_slot, s_phase ^ 1)
                        # The S-slot test is a runtime branch that duplicates the
                        # whole 8-instruction chain in the emitted code; the Q-slot
                        # choice collapses into an address select (export: two
                        # chains behind `@%p bra`, one PV chain).
                        with K.If(s_slot == 0):
                            with K.Then():
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
                            with K.Else():
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
                    drain_begin = K.local_scalar(
                        K.i32,
                        init=K.if_then_else(num_q_groups_mma == 1, 0, num_q_groups_mma - 2),
                        name="drain_begin",
                    )
                    with K.serial(drain_begin, num_q_groups_mma, unroll=False) as pv_qi2:
                        pv_slot2 = K.local_scalar(
                            K.i32, init=K.bitwise_and(pv_qi2, 1), name="pv_slot2"
                        )
                        pv_phase2 = K.local_scalar(
                            K.i32, init=udiv_i32(pv_qi2, 2) & 1, name="pv_phase2"
                        )
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

                K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))
                # The contract forbids a native load on a shared buffer, so the
                # allocated base comes back through an explicit ld.shared.
                tmem_base = K.alloc_local((1,), "uint32")
                K.ptx.ld.shared.u32(tmem_base[0], tmem_start_addr.ptr_to([0]))
                K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_base[0], K.uint32(TMEM_TOTAL)
                )
                # No matching griddepcontrol.wait anywhere in this kernel, so no
                # launch attribute is needed -- only a waiting kernel needs one.
                K.ptx.griddepcontrol.launch_dependents()

        # -----------------------------------------------------------------------
        # ROLE: softmax warpgroups 0 and 1, with the epilogue fused in
        # (:1113-1191, :2437-2616, :2927-3305).
        #
        # `stage` is a compile-time constant, so this body is emitted once per
        # warpgroup -- which is what the export shows, every single-site operation
        # inside it appearing exactly twice.
        # -----------------------------------------------------------------------
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
            slot = K.local_scalar(K.i32, init=K.bitwise_and(qi_group, 1), name="slot")
            phase = K.local_scalar(K.i32, init=udiv_i32(qi_group, 2) & 1, name="phase")
            bar_o_full.wait(slot, phase)

            # Decode the packed qsplit once per group into the 2-deep caches; every
            # per-store read below comes out of these, never out of s_qidx_meta.
            with K.If(group_tidx < q_tokens_per_group), K.Then():
                word = K.local_scalar(
                    K.i32, init=ld_shared_i32(s_qidx_meta, qidx_meta_slot + group_tidx), name="word"
                )
                st_shared_i32(
                    s_q_idx, slot * q_tokens_per_group + group_tidx, K.bitwise_and(word, Q_IDX_MASK)
                )
                st_shared_i32(
                    s_split_idx,
                    slot * q_tokens_per_group + group_tidx,
                    K.bitwise_and(K.shift_right(word, SLOT_SHIFT), SLOT_MASK),
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
            warp_in_wg = K.local_scalar(K.i32, init=group_tidx // WARP_SIZE, name="warp_in_wg")
            lane_in_warp = K.local_scalar(
                K.i32, init=group_tidx - warp_in_wg * WARP_SIZE, name="lane_in_warp"
            )
            row_of_lane = K.local_scalar(
                K.i32, init=warp_in_wg * 32 + lane_in_warp // 4, name="row_of_lane"
            )
            col_of_lane = K.local_scalar(K.i32, init=(lane_in_warp % 4) * 2, name="col_of_lane")

            # One reciprocal per (lane_base, parity) for the whole epilogue -- four
            # per thread -- instead of one per 128-bit store.
            row_scale_cache = K.alloc_local((4,), "float32")
            for lb in range(2):
                for par in range(2):
                    rs_row = K.local_scalar(
                        K.i32, init=row_of_lane + lb * 16 + 8 * par, name="rs_row"
                    )
                    rs_sum = K.local_scalar(
                        K.f32,
                        init=ld_shared_f32(s_scale, slot * M_BLOCK * 2 + rs_row),
                        name="rs_sum",
                    )
                    rs_safe = K.local_scalar(
                        K.f32,
                        init=K.if_then_else(
                            K.Or(rs_sum == K.float32(0.0), rs_sum != rs_sum), K.float32(1.0), rs_sum
                        ),
                        name="rs_safe",
                    )
                    K.ptx.rcp.approx.ftz.f32(row_scale_cache[lb * 2 + par], rs_safe)

            # `lane_base` stays a Python int through the inline call, keeping the
            # TMEM address's lane field constant; `col_base` is the rolled pass's
            # runtime column half.
            def load_o_pass(o_regs, lane_base, col_base):
                K.ptx[_TMEM_LD_16](
                    *[o_regs[i] for i in range(32)],
                    K.cast(tmem_o_col + slot * tmem_o_stage_stride + col_base, "uint32")
                    + K.uint32(lane_base << 16),
                )

            def store_o_pass(o_regs, lane_base, col_base):
                # Keep this as trace-time expansion: `grp` selects Python-list
                # registers and fixes the parity/store width for each emitted op.
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
                    row = K.local_scalar(
                        K.i32, init=row_of_lane + lane_base + 8 * parity, name="row"
                    )
                    tok = K.local_scalar(K.i32, init=udiv_i32(row, qheadperkv), name="tok")
                    row_in_tok = K.local_scalar(
                        K.i32, init=row - tok * qheadperkv, name="row_in_tok"
                    )
                    qi = K.local_scalar(K.i32, init=qi_group * q_tokens_per_group + tok, name="qi")
                    with K.If(qi < count_raw), K.Then():
                        # Re-read per store, as the reference does: nothing here is
                        # hoisted out of the column loop, the reciprocal included
                        # (:2785-2800, measured one of each per 128-bit store).
                        q_idx_e = K.local_scalar(
                            K.i32,
                            init=ld_shared_i32(s_q_idx, slot * q_tokens_per_group + tok),
                            name="q_idx_e",
                        )
                        split_e = K.local_scalar(
                            K.i32,
                            init=ld_shared_i32(s_split_idx, slot * q_tokens_per_group + tok),
                            name="split_e",
                        )
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
                        row_scale = K.alloc_local((1,), "float32")
                        K.assign(row_scale[0], row_scale_cache[(lane_base // 16) * 2 + parity])
                        q_abs_e = K.local_scalar(K.i32, init=q_batch_off + q_idx_e, name="q_abs_e")
                        flat_row = K.local_scalar(
                            K.i64,
                            init=(
                                K.cast(split_e, "int64")
                                * K.cast(total_q, "int64")
                                * K.cast(head_q, "int64")
                                + K.cast(q_abs_e, "int64") * K.cast(head_q, "int64")
                                + K.cast(head_kv_idx[0] * qheadperkv + row_in_tok, "int64")
                            ),
                            name="flat_row",
                        )
                        # The group's first register sits at column
                        # `col_base + (lane%4)*2 + 8*kb`; the fake-column map turns
                        # that into the contiguous output address.
                        store_col = K.local_scalar(
                            K.i32,
                            init=_fake_col(partial_dtype, col_base + col_of_lane + 8 * kb),
                            name="store_col",
                        )
                        scaled = K.alloc_local((_STORE_LANES[partial_dtype],), "float32")
                        _scale_gather(scaled, o_regs, regs, row_scale[0])
                        _store_o_partial(
                            o_partial,
                            flat_row * K.int64(HEAD_DIM) + K.cast(store_col, "int64"),
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
            overlap_o = USE_GATHER4(qheadperkv) or q_dtype != "float8_e4m3"
            if overlap_o:
                o_regs_0 = K.alloc_local((32,), "float32")
                o_regs_1 = K.alloc_local((32,), "float32")
                o_regs_2 = K.alloc_local((32,), "float32")
                o_regs_3 = K.alloc_local((32,), "float32")
                load_o_pass(o_regs_0, 0, 0)
                load_o_pass(o_regs_1, 0, 64)
                load_o_pass(o_regs_2, 16, 0)
                load_o_pass(o_regs_3, 16, 64)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                store_o_pass(o_regs_0, 0, 0)
                store_o_pass(o_regs_1, 0, 64)
                store_o_pass(o_regs_2, 16, 0)
                store_o_pass(o_regs_3, 16, 64)
            else:
                o_regs_a = K.alloc_local((32,), "float32")
                o_regs_b = K.alloc_local((32,), "float32")
                with K.serial(0, 2, unroll=False) as col_pass:
                    col_base_rt = K.local_scalar(K.i32, init=col_pass * 64, name="col_base_rt")
                    load_o_pass(o_regs_a, 0, col_base_rt)
                    load_o_pass(o_regs_b, 16, col_base_rt)
                    store_o_pass(o_regs_a, 0, col_base_rt)
                    store_o_pass(o_regs_b, 16, col_base_rt)
                K.ptx.tcgen05.wait__ld.sync.aligned()

            # LSE: one row per thread (:2987-3016).
            tok_l = K.local_scalar(K.i32, init=udiv_i32(group_tidx, qheadperkv), name="tok_l")
            h_local = K.local_scalar(K.i32, init=group_tidx - tok_l * qheadperkv, name="h_local")
            with K.If(qi_group * q_tokens_per_group + tok_l < count_raw), K.Then():
                row_sum_l = K.local_scalar(
                    K.f32,
                    init=ld_shared_f32(s_scale, slot * M_BLOCK * 2 + group_tidx),
                    name="row_sum_l",
                )
                row_max_l = K.local_scalar(
                    K.f32,
                    init=ld_shared_f32(s_scale, slot * M_BLOCK * 2 + M_BLOCK + group_tidx),
                    name="row_max_l",
                )
                lg = K.alloc_local((1,), "float32")
                K.ptx.lg2.approx.ftz.f32(lg[0], row_sum_l)
                lse_val = K.local_scalar(
                    K.f32,
                    init=K.if_then_else(
                        K.Or(row_sum_l == K.float32(0.0), row_sum_l != row_sum_l),
                        -K.infinity("float32"),
                        (row_max_l * scale_log2 + lg[0]) * K.float32(LN_2),
                    ),
                    name="lse_val",
                )
                q_idx_l = K.local_scalar(
                    K.i32,
                    init=ld_shared_i32(s_q_idx, slot * q_tokens_per_group + tok_l),
                    name="q_idx_l",
                )
                split_l = K.local_scalar(
                    K.i32,
                    init=ld_shared_i32(s_split_idx, slot * q_tokens_per_group + tok_l),
                    name="split_l",
                )
                h_abs = K.local_scalar(
                    K.i32, init=head_kv_idx[0] * qheadperkv + h_local, name="h_abs"
                )
                lse_flat = K.local_scalar(
                    K.i64,
                    init=(
                        K.cast(split_l, "int64")
                        * K.cast(total_q, "int64")
                        * K.cast(head_q, "int64")
                        + K.cast(q_batch_off + q_idx_l, "int64") * K.cast(head_q, "int64")
                        + K.cast(h_abs, "int64")
                    ),
                    name="lse_flat",
                )
                K.ptx.st.global_.f32(lse_partial.ptr_to([lse_flat]), lse_val)
                if temperature:
                    temp_sum = K.local_scalar(
                        K.f32,
                        init=ld_shared_f32(s_scale_temp, slot * M_BLOCK + group_tidx),
                        name="temp_sum",
                    )
                    lgt = K.alloc_local((1,), "float32")
                    K.ptx.lg2.approx.ftz.f32(lgt[0], temp_sum)
                    lse_t = K.local_scalar(
                        K.f32,
                        init=K.if_then_else(
                            K.Or(temp_sum == K.float32(0.0), temp_sum != temp_sum),
                            NEG_INF,
                            (row_max_l * lse_temperature_scale_log2 + lgt[0]) * K.float32(LN_2),
                        ),
                        name="lse_t",
                    )
                    K.ptx.st.global_.f32(lse_temperature_partial.ptr_to([lse_flat]), lse_t)

            bar_sync_named(BAR_EPILOGUE + stage, SOFTMAX_THREADS)
            bar_stats_empty.arrive(slot)
            bar_o_empty.arrive(slot)

        def softmax_warpgroup(stage):
            group_tidx = K.local_scalar(
                K.i32, init=tidx - stage * SOFTMAX_THREADS, name="group_tidx"
            )
            kv_block_sm = K.local_scalar(
                K.i32, init=ld_shared_i32(s_row_meta, 1), name="kv_block_sm"
            )
            count_raw_sm = K.local_scalar(
                K.i32, init=ld_shared_i32(s_row_meta, 3), name="count_raw_sm"
            )
            kv_valid_cols = K.local_scalar(
                K.i32, init=ld_shared_i32(s_row_meta, 4), name="kv_valid_cols"
            )
            q_batch_off_sm = K.local_scalar(
                K.i32, init=ld_shared_i32(s_row_meta, 5), name="q_batch_off_sm"
            )
            # Read unconditionally, as the source does (:1119-1127): both cells only
            # ever feed the diagonal mask, so a non-causal build leaves two dead
            # scalar loads behind rather than gating them away.
            causal_q_off = K.local_scalar(
                K.i32, init=ld_shared_i32(s_row_meta, 7), name="causal_q_off"
            )
            diag_q_count_sm = K.local_scalar(
                K.i32, init=ld_shared_i32(s_diag_q_count, 0), name="diag_q_count_sm"
            )

            # NVFP4 dequantization (:1779-1865). Unconditional and not in the load
            # warps: the TMA lands packed FP4 in the staging tile and a whole
            # softmax warpgroup converts it into the tile the MMA reads, before
            # entering its own loop. WG0 takes K, WG1 takes V.
            k_batch_off_sm = K.local_scalar(
                K.i32, init=ld_shared_i32(s_row_meta, 6), name="k_batch_off_sm"
            )
            token_base_sm = K.local_scalar(
                K.i32, init=k_batch_off_sm + kv_block_sm * N_BLOCK, name="token_base_sm"
            )
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
                    k_global_scale if k_global else None,
                    mma_dtype,
                    fold_k_global,
                    paged,
                    s_paged_kv_idx,
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
                    paged,
                    s_paged_kv_idx,
                )

            bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

            with K.If(count_raw_sm > 0), K.Then():
                num_q_groups_sm = K.local_scalar(
                    K.i32,
                    init=uceil_div_i32(count_raw_sm, q_tokens_per_group),
                    name="num_q_groups_sm",
                )
                # Zero unless causal (:2461-2463); it only ever offsets the
                # diagonal column limit, which a non-causal build never computes.
                if causal:
                    kv_block_col_start = K.local_scalar(
                        K.i32, init=kv_block_sm * N_BLOCK, name="kv_block_col_start"
                    )
                else:
                    kv_block_col_start = K.local_scalar(
                        K.i32, init=K.int32(0), name="kv_block_col_start"
                    )
                # WG0 takes the even Q groups, WG1 the odd ones (:2465-2468).
                num_stage_groups = K.local_scalar(
                    K.i32, init=udiv_i32(num_q_groups_sm + (1 - stage), 2), name="num_stage_groups"
                )

                with K.serial(0, num_stage_groups, unroll=False) as qi_iter:
                    qi_group = K.local_scalar(K.i32, init=qi_iter * 2 + stage, name="qi_group")
                    phase = K.local_scalar(K.i32, init=K.bitwise_and(qi_iter, 1), name="phase")
                    producer_phase = K.local_scalar(K.i32, init=phase ^ 1, name="producer_phase")
                    qidx_meta_slot = K.local_scalar(
                        K.i32,
                        init=K.bitwise_and(qi_group, QIDX_META_STAGES - 1) * q_tokens_per_group,
                        name="qidx_meta_slot",
                    )

                    # ---------------- softmax step (:2437-2616) ----------------
                    bar_s_full.wait(stage, phase)
                    s_regs = K.alloc_local((128,), "float32")
                    with K.unroll(4) as chunk:
                        K.ptx[_TMEM_LD_32](
                            *[s_regs[chunk * 32 + i] for i in range(32)],
                            K.cuda.get_tmem_addr(
                                tmem_s_col + stage * tmem_stage_stride + chunk * 32, 0, 0
                            ),
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
                        kg = K.alloc_local((1,), "float32")
                        K.ptx.ld.global_.f32(kg[0], k_global_scale.ptr_to([0]))
                        with K.unroll(N_BLOCK // 2) as ii:
                            i = ii * 2
                            _packed_f32x2(
                                "mul.rn.f32x2",
                                s_regs,
                                i,
                                i + 1,
                                s_regs[i],
                                s_regs[i + 1],
                                kg[0],
                                kg[0],
                            )

                    # Column-limit masking, and causal masking for the tokens on
                    # the diagonal. Both arms feed one r2p bit-test body, which is
                    # how the reference emits it too (mask.py:36-46, :71-121).
                    col_limit = K.alloc_local((1,), "int32")
                    K.assign(col_limit[0], kv_valid_cols)
                    if causal:
                        # How many of this group's tokens still sit on the causal
                        # diagonal and therefore need per-column masking
                        # (:2478-2486). A non-causal build passes a literal zero
                        # here and takes the column-limit-only arm (:2544-2548), so
                        # none of this is emitted.
                        qi_group_start = K.local_scalar(
                            K.i32, init=qi_group * q_tokens_per_group, name="qi_group_start"
                        )
                        masked_tok_count = K.local_scalar(
                            K.i32,
                            init=K.max(
                                0, K.min(q_tokens_per_group, diag_q_count_sm - qi_group_start)
                            ),
                            name="masked_tok_count",
                        )
                        with K.If(masked_tok_count > 0), K.Then():
                            tok_of_row = K.local_scalar(
                                K.i32, init=udiv_i32(group_tidx, qheadperkv), name="tok_of_row"
                            )
                            q_idx_mask = K.local_scalar(
                                K.i32,
                                init=K.bitwise_and(
                                    ld_shared_i32(s_qidx_meta, qidx_meta_slot + tok_of_row),
                                    Q_IDX_MASK,
                                ),
                                name="q_idx_mask",
                            )
                            causal_col_limit = K.local_scalar(
                                K.i32,
                                init=q_idx_mask + causal_q_off - kv_block_col_start + 1,
                                name="causal_col_limit",
                            )
                            K.assign(col_limit[0], K.min(kv_valid_cols, causal_col_limit))
                    with K.If(col_limit[0] < N_BLOCK), K.Then():
                        with K.unroll(N_BLOCK // MASK_R2P_CHUNK) as chunk:
                            shift = K.local_scalar(
                                K.i32,
                                init=K.max((chunk + 1) * MASK_R2P_CHUNK - col_limit[0], 0),
                                name="shift",
                            )
                            # `shr.u32` clamps a shift of 32 or more to zero, which
                            # is exactly what `r2p_bitmask_below` relies on for the
                            # chunks entirely past the column limit. A TIR-level
                            # shift is undefined there and leaves the chunk
                            # unmasked, which shows up as too-large row sums on the
                            # early query rows.
                            bits_reg = K.alloc_local((1,), "uint32")
                            K.ptx.shr.u32(
                                bits_reg[0], K.uint32(0xFFFFFFFF), K.cast(shift, "uint32")
                            )
                            bits = K.local_scalar(K.u32, init=bits_reg[0], name="bits")
                            # `K.unroll`, not `range`: a `range` here lowers to a
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
                            with K.unroll(MASK_R2P_CHUNK) as i:
                                with (
                                    K.If(
                                        K.bitwise_and(
                                            bits, K.shift_left(K.uint32(1), K.cast(i, "uint32"))
                                        )
                                        == K.uint32(0)
                                    ),
                                    K.Then(),
                                ):
                                    K.assign(s_regs[chunk * MASK_R2P_CHUNK + i], NEG_INF)

                    # One KV block per Q group, so this is always the first and
                    # only online-softmax step: no rescale of a running accumulator
                    # (:2284-2287).
                    row_max = K.alloc_local((1,), "float32")
                    _row_max_128(s_regs, row_max, 0)
                    # `row_max_safe` (:246-247): a fully masked row has row_max
                    # -inf, and subtracting it would make every element NaN. The
                    # reference substitutes 0.0 before the scale-subtract; the
                    # row's sum then comes out 0 and the epilogue's zero guard
                    # turns it into a neutral partial.
                    K.assign(
                        row_max[0],
                        K.if_then_else(row_max[0] != NEG_INF, row_max[0], K.float32(0.0)),
                    )
                    # Taken as a scalar, not recomputed: the reference folds
                    # `softmax_scale * log2(e)` on the host in double precision
                    # (:514), and redoing it in f32 here differs by one ULP, which
                    # propagates straight into every LSE.
                    scale_log2 = K.local_scalar(K.f32, init=softmax_scale_log2, name="scale_log2")
                    neg_max_scaled = K.local_scalar(
                        K.f32, init=-(row_max[0] * scale_log2), name="neg_max_scaled"
                    )
                    with K.unroll(N_BLOCK // 2) as ii:
                        i = ii * 2
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

                    if temperature:
                        temp_row_sum = K.alloc_local((1,), "float32")
                        _scaled_exp2_row_sum_128(s_regs, lse_temperature_inv_scale, temp_row_sum)

                    bar_p_last_empty.wait(stage, producer_phase)
                    bar_p_empty.wait(stage, producer_phase)

                    # exp2 with the reference's MUFU / polynomial mix, then the
                    # packed conversion into the P operand dtype (:2307-2312).
                    # 128 P values pack into 64 words as bf16, 32 as fp8; the
                    # store repetition follows (:2429-2439).
                    p_words = K.alloc_local((N_BLOCK * _DTYPE_BYTES[pv_dtype] * 8 // 32,), "uint32")
                    # Preserve the parser kernel's trace-time expansion. The
                    # zero-frequency specialization is also decided while tracing,
                    # so the modulo-by-zero expression is never constructed.
                    for j in range(EX2_FRG_CNT):
                        for k in range(0, EX2_FRG_TILE, 2):
                            if ex2_emu_freq == 0:
                                K.ptx.ex2.approx.ftz.f32(s_regs[j * 32 + k], s_regs[j * 32 + k])
                                K.ptx.ex2.approx.ftz.f32(
                                    s_regs[j * 32 + k + 1], s_regs[j * 32 + k + 1]
                                )
                            else:
                                use_mufu = K.Or(
                                    (k % ex2_emu_freq) < (ex2_emu_freq - EX2_EMU_RES),
                                    K.Or(j >= EX2_FRG_CNT - 1, j < EX2_EMU_START_FRG),
                                )
                                with K.If(use_mufu):
                                    with K.Then():
                                        K.ptx.ex2.approx.ftz.f32(
                                            s_regs[j * 32 + k], s_regs[j * 32 + k]
                                        )
                                        K.ptx.ex2.approx.ftz.f32(
                                            s_regs[j * 32 + k + 1], s_regs[j * 32 + k + 1]
                                        )
                                    with K.Else():
                                        _ex2_emulation_2(s_regs, j * 32 + k, j * 32 + k + 1)
                        _pack_p_words(p_words, s_regs, j, pv_dtype)

                    # Publish P in two pieces: the first three quarters early, so
                    # the MMA warp can start PV, and the last quarter on the
                    # separate barrier its instruction sequence blocks on.
                    split_idx = 4 * SPLIT_P_ARRIVE // N_BLOCK
                    rep = _P_STORE_REP[pv_dtype]
                    st_chain = _TMEM_ST[rep]
                    for k in range(4):
                        K.ptx[st_chain](
                            K.cuda.get_tmem_addr(
                                tmem_p_col + stage * tmem_stage_stride + k * rep, 0, 0
                            ),
                            *[p_words[k * rep + i] for i in range(rep)],
                        )
                        with K.If(k + 1 == split_idx), K.Then():
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            bar_p_full.arrive(stage)
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    bar_p_last_full.arrive(stage)

                    # The stats pipe's empty half is a credit on s_scale: it stops
                    # this group from overwriting the slot the epilogue two groups
                    # back has not drained (:2331).
                    bar_stats_empty.wait(stage, producer_phase)
                    row_sum = K.alloc_local((1,), "float32")
                    _row_sum_128(s_regs, row_sum)
                    st_shared_f32(s_scale, stage * M_BLOCK * 2 + group_tidx, row_sum[0])
                    st_shared_f32(s_scale, stage * M_BLOCK * 2 + M_BLOCK + group_tidx, row_max[0])
                    if temperature:
                        st_shared_f32(s_scale_temp, stage * M_BLOCK + group_tidx, temp_row_sum[0])
                    K.ptx.fence.proxy.async_.shared__cta()
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

        with K.If(warp_idx < SOFTMAX1_WARP_BASE), K.Then():
            with K.If(cta_valid_work != 0), K.Then():
                K.ptx.setmaxnreg.inc.sync.aligned.u32(K.uint32(num_regs_softmax))
                softmax_warpgroup(0)

        with K.If(K.And(warp_idx >= SOFTMAX1_WARP_BASE, warp_idx < Q_LOAD_WARP_BASE)), K.Then():
            with K.If(cta_valid_work != 0), K.Then():
                K.ptx.setmaxnreg.inc.sync.aligned.u32(K.uint32(num_regs_softmax))
                softmax_warpgroup(1)

    parameters = [
        # K/V keep their three logical axes because the host-side TensorMap
        # encoding consumes that tensor ABI.  Paged storage is reshaped to the
        # same contiguous rank-3 view by tirx_args; no match_buffer is needed.
        ("k", K.gptr(K.u8, shape=lambda p: (p["total_k"], p["num_heads_kv"], PACKED_HEAD_DIM))),
        ("v", K.gptr(K.u8, shape=lambda p: (p["total_k"], p["num_heads_kv"], PACKED_HEAD_DIM))),
        ("k_scale", K.gptr(K.u8, shape=lambda p: (p["scale_numel"],))),
        ("v_scale", K.gptr(K.u8, shape=lambda p: (p["scale_numel"],))),
    ]
    if k_global:
        parameters.append(("k_global_scale", K.gptr[K.f32, (1,)]))
    parameters.extend(
        [
            ("k2q_q_indices", K.gptr(K.i32, shape=lambda p: (p["num_heads_kv"] * p["nnz"],))),
            ("k2q_qsplit_indices", K.gptr(K.i32, shape=lambda p: (p["num_heads_kv"] * p["nnz"],))),
            (
                "k2q_row_ptr",
                K.gptr(K.i32, shape=lambda p: (p["num_heads_kv"] * (p["total_rows"] + 1),)),
            ),
            (
                "scheduler_metadata",
                K.gptr(K.i32, shape=lambda p: (p["work_capacity"] * WORK_FIELDS,)),
            ),
            ("work_count", K.gptr[K.i32, (1,)]),
            (
                "o_partial",
                K.gptr(
                    partial_ty, shape=lambda p: (p["topk"] * p["total_q"] * p["head_q"] * HEAD_DIM,)
                ),
            ),
            (
                "lse_partial",
                K.gptr(K.f32, shape=lambda p: (p["topk"] * p["total_q"] * p["head_q"],)),
            ),
        ]
    )
    if temperature:
        parameters.append(
            (
                "lse_temperature_partial",
                K.gptr(K.f32, shape=lambda p: (p["topk"] * p["total_q"] * p["head_q"],)),
            )
        )
    # Q is flattened only across token/head; HEAD_DIM remains the descriptor's
    # contiguous axis, so the launch argument is a rank-2 tensor.
    parameters.append(
        ("q_flat", K.gptr(q_ty, shape=lambda p: (p["total_q"] * p["head_q"], HEAD_DIM)))
    )
    if paged:
        parameters.append(("page_table", K.gptr(K.i32, shape=lambda p: (p["total_k"] // N_BLOCK,))))
    if seqused:
        parameters.append(("seqused_k", K.gptr(K.i32, shape=lambda p: (p["num_batches"],))))
    parameters.extend(
        [
            ("cu_seqlens_q", K.gptr(K.i32, shape=lambda p: (p["num_batches"] + 1,))),
            ("cu_seqlens_k", K.gptr(K.i32, shape=lambda p: (p["num_batches"] + 1,))),
            ("softmax_scale_log2", K.f32),
            ("lse_temperature_scale_log2", K.f32),
            ("lse_temperature_inv_scale", K.f32),
            ("num_kv_blocks", K.i32),
            ("num_heads_kv", K.i32),
            ("seq_len_q", K.i32),
            ("work_capacity", K.i32),
            ("total_k", K.i32),
            ("total_q", K.i32),
            ("head_q", K.i32),
            ("nnz", K.i32),
            ("total_rows", K.i32),
            ("num_batches", K.i32),
            ("topk", K.i32),
            ("scale_numel", K.i32),
        ]
    )
    names = tuple(name for name, _ in parameters)

    def entry(*args, host):
        trace(dict(zip(names, args, strict=True)), host)

    entry.__name__ = KERNEL_META["name"]
    entry.__signature__ = inspect.Signature(
        [
            *[
                inspect.Parameter(
                    name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation
                )
                for name, annotation in parameters
            ],
            inspect.Parameter("host", inspect.Parameter.KEYWORD_ONLY),
        ]
    )
    kernel = K.kernel(
        warps=TOTAL_WARPS,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid="work_capacity",
        host_prelude=host_prelude,
    )(entry)
    return kernel.func.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


def get_kernel(**config):
    """Return the native Kern specialization for one compile key."""
    config.pop("label", None)
    return _make_kernel(**config)


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
    paged: bool = False,
    seqused: bool = False,
) -> dict:
    return {
        "label": label,
        "paged": paged,
        "seqused": seqused,
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
    # Paged. The first row mirrors the flat marquee shape so the pair reads as
    # the cost of the page indirection alone.
    _case(
        label="paged_ring48k_bf16q_qh16_t16",
        batch=1,
        seqlen_q=49152,
        seqlen_k=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
        partial_dtype="bfloat16",
        paged=True,
    ),
    _case(
        label="paged_ring48k_fp8q_qh16_t16",
        batch=1,
        seqlen_q=49152,
        seqlen_k=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
        dtype="fp8q",
        partial_dtype="bfloat16",
        temperature=1.0,
        paged=True,
    ),
    _case(
        label="paged_seqused_bf16q_s16384_qh4_t16",
        batch=2,
        seqlen_q=16384,
        seqlen_k=16384,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        partial_dtype="bfloat16",
        paged=True,
        seqused=True,
    ),
    _case(
        label="paged_edge_b1_s2048_fp8q_t4",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
        dtype="fp8q",
        partial_dtype="bfloat16",
        paged=True,
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
    # Paged KV. Every paged row runs `head_kv=2`: the paged scale row is
    # `(page * head_kv + head_kv_idx) * page_size + token`, and at `head_kv == 1`
    # it collapses to a form that a wrong head factor would still satisfy.
    _case(
        label="paged_bf16q_s4096_qh4_t16",
        batch=2,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        paged=True,
    ),
    _case(
        label="paged_fp8q_s4096_qh16_t8",
        batch=2,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=16,
        topk=8,
        dtype="fp8q",
        partial_dtype="bfloat16",
        paged=True,
    ),
    _case(
        label="paged_seqused_bf16q_s2048_qh8_t8",
        batch=3,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        paged=True,
        seqused=True,
    ),
    _case(
        label="paged_seqused_fp8q_s2048_qh2_t8",
        batch=3,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=2,
        topk=8,
        dtype="fp8q",
        partial_dtype="bfloat16",
        paged=True,
        seqused=True,
    ),
    # K global scale off: the fold disappears from the dequant, so a paged
    # scale-row error can no longer hide behind a wrong tensor scale.
    _case(
        label="paged_bf16q_kgs0_s2048_qh4_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=4,
        topk=8,
        k_global=False,
        paged=True,
    ),
    _case(
        label="paged_seqused_varlen_b2_s2048_bf16q_qh2_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=2,
        topk=8,
        paged=True,
        seqused=True,
        seqlen_pattern="varlen",
    ),
    _case(
        label="paged_fp8q_partial_fp8_s2048_qh8_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        dtype="fp8q",
        partial_dtype="float8_e4m3",
        temperature=2.0,
        paged=True,
    ),
    # `causal=False`: upstream's only non-causal NVFP4 coverage is at this
    # scale, and the axis changes the register budget, `ex2_emu_freq`, and the
    # diagonal search (:177-184).
    _case(
        label="corr_noncausal_bf16q_s512_qh4_t4",
        batch=1,
        seqlen_q=512,
        seqlen_k=512,
        head_kv=2,
        qhead_per_kv=4,
        topk=4,
        causal=False,
    ),
    _case(
        label="corr_noncausal_paged_fp8q_s2048_qh4_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=4,
        topk=8,
        dtype="fp8q",
        partial_dtype="bfloat16",
        causal=False,
        paged=True,
    ),
    _case(
        label="corr_partial_fp16_bf16q_s2048_qh8_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        partial_dtype="float16",
    ),
    _case(
        label="corr_topk32_bf16q_s4096_qh4",
        batch=1,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=4,
        topk=32,
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

    from tirx_kernels.msa.sparse_atten_fwd import (
        _build_page_table,
        _frozen_qsplit,
        _pack_paged_kv,
        _seqused_trim,
    )
    from tirx_kernels.msa.sparse_prepare_fwd_split_atomic import prepare_data as prepare_csr

    config.pop("label", None)
    paged = bool(config.get("paged", False))
    seqused = bool(config.get("seqused", False))
    if seqused and not paged:
        raise ValueError("seqused_k is only supported together with page_table")
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

    # Paged KV, shared with the sibling port: `page_size == blk_kv`, a shuffled
    # rectangular table, zero-filled trailing pages.
    page_table = seqused_k = None
    k_paged = v_paged = None
    if paged:
        blk_kv = config["blk_kv"]
        pages_per_seq = max((int(length) + blk_kv - 1) // blk_kv for length in csr["seqlens_k"])
        page_table = _build_page_table(len(csr["seqlens_k"]), pages_per_seq, generator)
        k_paged = _pack_paged_kv(k, csr["seqlens_k"], blk_kv, page_table)
        v_paged = _pack_paged_kv(v, csr["seqlens_k"], blk_kv, page_table)
        if seqused:
            trims = _seqused_trim(len(csr["seqlens_k"]), blk_kv)
            seqused_k = torch.tensor(
                [int(length) - trim for length, trim in zip(csr["seqlens_k"], trims, strict=True)],
                dtype=torch.int32,
                device=device,
            )
        else:
            assert all(
                (int(length) + blk_kv - 1) // blk_kv == pages_per_seq and int(length) % blk_kv == 0
                for length in csr["seqlens_k"]
            ), "paged without seqused_k requires uniform lengths: capacity is the logical length"

    # One block scale per 16 head-dim elements, indexed by the logical KV row.
    # Paged moves that row from `token * head_kv + head_kv_idx` to
    # `(page * head_kv + head_kv_idx) * page_size + token` (:1328-1336) -- which
    # is still exactly the row-major flattening of the tensor the kernel reads,
    # so the scale array simply grows to cover the page array.
    rows = int(k_paged.numel() // PACKED_HEAD_DIM) if paged else total_k * head_kv
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

        Under paging the row becomes ``(page * head_kv + head_kv_idx) *
        page_size + token`` (``_paged_kv_scale_row``, :1328-1336), which is the
        row-major flattening of the paged tensor -- so the same reshape holds
        and only the shape restored at the end differs. The twins are produced
        in whichever layout the kernel consumes, because the BF16 sibling they
        feed runs in the same mode.
        """
        flat = packed.reshape(-1, PACKED_HEAD_DIM)
        out = _dequant_nvfp4_to_bf16(flat, scale, global_scale, rows=rows, cols=HEAD_DIM)
        return out.reshape(*packed.shape[:-1], HEAD_DIM).contiguous()

    return {
        "config": dict(config),
        "csr": csr,
        "q": q,
        "q_flat": q.reshape(-1, HEAD_DIM),
        "k": k_paged if paged else k,
        "v": v_paged if paged else v,
        "k_flat": k,
        "v_flat": v,
        "k_scale_128x4": k_scale,
        "v_scale_128x4": v_scale,
        "k_global_scale": k_global,
        # Accepted by the host entry but never reaching this kernel: the
        # interface pins has_v_global_scale off and applies V's tensor scale in
        # the combine kernel. Kept so the contract is visible in one place.
        "v_global_scale": v_global,
        "k_dequantized": twin(k_paged if paged else k, k_scale, k_global),
        "v_dequantized": twin(v_paged if paged else v, v_scale, None),
        "paged": paged,
        "page_table": page_table,
        "seqused_k": seqused_k,
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
        # Under paging the K/V buffers span the page array, not the packed
        # sequences, and the kernel shapes its match_buffer from this value.
        "total_k": (int(page_table.numel()) * config["blk_kv"]) if paged else total_k,
        "total_k_flat": total_k,
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
        # 3-D view; see the sibling. The tensormap carries the paged geometry.
        data["k"].reshape(-1, data["head_kv"], PACKED_HEAD_DIM),
        data["v"].reshape(-1, data["head_kv"], PACKED_HEAD_DIM),
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
    args.append(as_bits(data["q_flat"]))
    # Optional handles, in the reference's own argument order: both are absent
    # from the signature entirely when the specialization pins them to None.
    if data.get("page_table") is not None:
        args.append(data["page_table"].reshape(-1))
    if data.get("seqused_k") is not None:
        args.append(data["seqused_k"])
    args += [
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
        # Present iff their axis is on. Their PRESENCE is what makes the adapter
        # compile a paged / seqused specialization at all, so dropping them here
        # silently reuses the flat binary and feeds it a rank-4 tensor.
        "page_table": data.get("page_table"),
        "seqused_k": data.get("seqused_k"),
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

    Under paging the twins are already page-major and the sibling runs paged
    too, on the same table and the same ``seqused_k``. Handing a paged NVFP4 run
    a flat twin would compare two different traversals of K/V and report the
    difference as a numerical error.
    """
    return {
        "head_dim": data["head_dim"],
        "blk_kv": data["blk_kv"],
        "qhead_per_kv": data["qhead_per_kv"],
        "causal": data["causal"],
        "k": data["k_dequantized"],
        "v": data["v_dequantized"],
        "page_table": data.get("page_table"),
        "seqused_k": data.get("seqused_k"),
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
