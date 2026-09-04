# This file is a TIRx port of code from MSA
# (https://github.com/MiniMax-AI/MSA @ 80434d7f), Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MSA sparse attention forward, split-partial form.

Ports ``SparseAttentionForwardSm100``: the attention kernel itself, consuming
the work list and the packed split slots that the two preparation kernels in
this package produce.

One CTA owns one work item, which is one KV block crossed with a run of the
query rows that attend it. The CTA loads that KV block once, walks its query
rows in groups, and writes each group's result to the split slot the
preparation stage reserved for it -- so the partials of a query never collide
and no atomics appear anywhere in this kernel. A separate combine kernel,
outside this port, reduces the partials of each query.

Upstream source: python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py:56.
"""

import inspect
import math
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
    "name": "msa_sparse_atten_fwd_sm100",
    "category": "msa",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
    "reference_requirements": (
        {
            "package": "msa",
            "git": {
                "url": "https://github.com/MiniMax-AI/MSA.git",
                "commit": "80434d7f67877c6570ca19cac444b84bc9855dac",
            },
            "import": "fmha_sm100",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.5.3", "import": "cutlass"},
        {"package": "quack-kernels", "specifier": "==0.5.0", "import": "quack"},
    ),
}
# `__init__` restricts both to a single supported value (:75-79, :99-107).
HEAD_DIM = 128
BLK_KV = 128
M_BLOCK = 128
N_BLOCK = 128
# `k_tile = 64` (:59): the UTCMMA bf16 K-tile, and the bf16 Q sub-tile width.
K_TILE = 64

# `warps_per_group=4`, `total_warps=16` (:130-143).
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

# Chunks each dequantizing thread owns: sixteen FP8 per 128-bit shared load.
# Module scope, because a bare assignment inside a traced body binds a TIR
# variable and the staging buffer would stop being a constant-size allocation.
DEQUANT_ITERS = N_BLOCK * (HEAD_DIM // 16) // SOFTMAX_THREADS
DEQUANT_BATCH = 4

# Pipeline depths (:116-125).
Q_STAGE = 2
S_STAGE = 2
O_STAGE = 2
K_STAGES = 2
QIDX_META_STAGES = 16

# `tmem_total` (:156-161), power-of-two rounded.
TMEM_TOTAL = 512

# `split_P_arrive = n_block // 4 * 3`, floored to a multiple of 32 (:165-167).
SPLIT_P_ARRIVE = 96

# Register budgets (:173-181). Softmax gains what store gives up, so the two
# move together or the launch exceeds its 512-register budget; `num_regs_other`
# is DERIVED as `512 - softmax*2 - store` and lands on 48 either way, which is
# why the export shows `setmaxnreg.dec 48` three times in both parities.
NUM_REGS_SOFTMAX_CAUSAL = 176
NUM_REGS_SOFTMAX_NONCAUSAL = 192
NUM_REGS_STORE_CAUSAL = 112
NUM_REGS_STORE_NONCAUSAL = 80
NUM_REGS_OTHER = 48
# `ex2_emu_freq = 16 if causal else 0` (:180-184). Zero does not mean "every
# sixteenth" with a different period -- it selects `apply_exp2_convert`'s own
# `const_expr(ex2_emu_freq == 0)` arm (softmax.py:381-383), which sends every
# element through real MUFU with no polynomial and no `fmax(x, -127)` clamp.
EX2_EMU_FREQ_CAUSAL = 16
EX2_EMU_START_FRG = 1

# cuTensorMapEncodeTiled enum values.
_SWIZZLE_128B = 3
_L2_PROMOTION_256B = 3

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3": 1}

# `scheduler_metadata` columns; the forward reads all six.
WORK_FIELDS = 6

# `q_idx | ((split_slot & 0xFF) << 24)`, decoded at :242-257.
SLOT_SHIFT = 24
SLOT_MASK = 0xFF
Q_IDX_MASK = (1 << SLOT_SHIFT) - 1


def _tirx_dtype(name: str) -> str:
    """The dtype TIRx declares a buffer with.

    CUDA codegen has no fp8 scalar type, and MSA does not want one either: it
    recasts its fp8 tiles to 32-bit words and converts them with a byte-permute
    plus a packed FMA. So an fp8 buffer is declared as raw ``uint8`` and every
    conversion goes through PTX on the bit pattern, which is what
    ``flashmla/sparse_decode_head64.py`` does too. ``cuTensorMapEncodeTiled``
    maps both spellings onto ``CU_TENSOR_MAP_DATA_TYPE_UINT8`` anyway.
    """
    return "uint8" if name == "float8_e4m3" else name


def _swizzle_elems(elem_bytes: int) -> int:
    """Elements spanned by one 128-byte swizzle atom, i.e. the TMA box width."""
    return 128 // elem_bytes


def USE_GATHER4(qheadperkv: int) -> bool:
    """`use_q_gather4` (:81): 1, 2 and 4 take the raw gather4 Q path."""
    return qheadperkv in (1, 2, 4)


def KV_SUBTILES(elem_bytes: int) -> int:
    """128-byte swizzle atoms across a 128-wide KV tile, i.e. TMA issues per tile."""
    return HEAD_DIM // _swizzle_elems(elem_bytes)


def Q_SUBTILES(elem_bytes: int) -> int:
    """Q sub-tiles per token: 1 for fp8 Q, `k_stages` = 2 for bf16 (:1872-1875)."""
    return HEAD_DIM // _swizzle_elems(elem_bytes)


def TOKENS_PER_WARP(qheadperkv: int) -> int:
    """`tokens_per_warp` (:1856-1859) on the TMA-Q path."""
    q_tokens_per_group = M_BLOCK // qheadperkv
    return (q_tokens_per_group + NUM_Q_LOAD_WARPS - 1) // NUM_Q_LOAD_WARPS


# Named-barrier ids. MSA emits `NamedBarrierFwdSm100`'s raw enum values with no
# user-barrier bias, which collides: `StoreEpilogue + stage` is 13 at stage 1
# and `KvLoad` is also 13, with 128 and 64 participants respectively. The port
# keeps the synchronization structure and numbers from 8 upward, the convention
# this repository documents at `flashmla/sparse_decode_head64.py:36-39`.
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
# Paging appends a page mode to the KV descriptors, so the same copy is issued
# at rank 4. Every other modifier -- load mode, completion, cache hint -- and
# the transaction byte count are unchanged from the 3-D form.
_TMA_G2S_4D_CACHE = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
)
# L2 eviction policies, per the reference's own defaults (tma_utils.py:23-24,
# :204, :251). Which tensor gets which is load-bearing, not cosmetic: a KV block
# belongs to exactly one CTA and is never read again, while a Q row is read by
# every one of the topK CTAs that carry it. Marking the streamed KV evict-last
# keeps it resident and pushes the Q rows out, so those CTAs re-fetch Q from
# DRAM instead of hitting L2.
_TMA_CACHE_EVICT_FIRST = K.uint64(0x12F0000000000000)
_TMA_CACHE_EVICT_LAST = K.uint64(0x14F0000000000000)
_Q_TMA_CACHE_HINT = _TMA_CACHE_EVICT_LAST
_KV_TMA_CACHE_HINT = _TMA_CACHE_EVICT_FIRST


_MMA_KIND = {"bfloat16": "f16", "float16": "f16", "float8_e4m3": "f8f6f4"}
# `mma_kind` (:363-364) is "f8f6f4" whenever the operand width is 8.
_MMA_CHAIN = {kind: f"tcgen05.mma.cta_group::1.kind::{kind}" for kind in ("f16", "f8f6f4")}
# K elements consumed by one tcgen05.mma instruction, so 128 // this is the
# chain length: 8 for f16, 4 for f8f6f4 -- which is the 48 = 6 x 8 and
# 24 = 6 x 4 census in the export.
_MMA_K = {"f16": 16, "f8f6f4": 32}


def _instr_desc(operand_dtype: str, trans_b: bool = False) -> int:
    """The tcgen05.mma instruction descriptor for a 128x128 f32-accumulating tile.

    Computed, not copied: the immediate encodes the operand dtype, so bf16 and
    f16 differ (0x08200490 against 0x08200010) even though both are
    ``kind::f16`` chains. Taking a neighbouring kernel's magic number is how a
    bf16 port ends up issuing an f16 descriptor.

    ``trans_b`` is the operand's major-ness: False for QK, where both operands
    are K-major, and True for PV, whose B tile is MN-major. Both were pinned
    against torch -- a wrong value here does not fault, it silently returns the
    wrong product.
    """
    from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32

    torch_name = {"bfloat16": "bfloat16", "float16": "float16", "float8_e4m3": "float8_e4m3fn"}[
        operand_dtype
    ]
    k = 32 if operand_dtype == "float8_e4m3" else 16
    return encode_instr_descriptor_dense_uint32(
        M_BLOCK, N_BLOCK, k, "float32", torch_name, torch_name, False, trans_b, cta_group=1
    )


_MMA_ELEM_BYTES = {"f16": 2, "f8f6f4": 1}

# Descriptor offsets, all in 16-byte units, and all identical between the bf16
# and fp8 geometries because a 128-byte swizzle atom holds twice as many fp8
# elements as bf16 ones.
_SWIZZLE_BYTES = 128


def _stage_16b(operand_dtype: str) -> int:
    """One Q pipeline stage, in 16-byte units: 2048 for bf16, 1024 for fp8.

    Element-width dependent, so it cannot be a constant. Hard-coding the bf16
    value points stage 1's A descriptor past the end of the Q tile, which shows
    up as an out-of-range shared address from the MMA warp -- and only once a
    work item has more than one Q group, so small shapes pass.
    """
    return M_BLOCK * HEAD_DIM * _MMA_ELEM_BYTES[_MMA_KIND[operand_dtype]] // 16


_SUBTILE_16B = 1024  # one 128-byte swizzle atom column band
_MMA_K_16B = 2  # one tcgen05.mma K step
# The A operand's TMEM column step: 8 columns per K chunk in both widths.
# The PV B operand advances 128 sixteen-byte units per 16-element K chunk of
# a bf16 MN-major V tile, and 256 per 32-element fp8 chunk -- both measured
# against torch, not derived.
_PV_K_16B = {"f16": 128, "f8f6f4": 256}
_PV_A_COL_STEP = {"f16": 8, "f8f6f4": 8}

# The P operand's tcgen05.st repetition (:2429-2439): 16 for a bf16 P, 8 for
# fp8, chosen so the 3/4 publish boundary falls on an instruction edge.
_P_STORE_REP = {"bfloat16": 16, "float16": 16, "float8_e4m3": 8}


# `POLY_EX2[3]` (utils.py:24-40): the degree-3 minimax polynomial for
# 2**frac on [0, 1), evaluated by Horner in packed f32x2.
_POLY_EX2_3 = (
    1.0,
    0.695146143436431884765625,
    0.227564394474029541015625,
    0.077119089663028717041015625,
)
# `fp32_round_int = 2**23 + 2**22` (utils.py:991): adding it round-down puts the
# integer floor in the low bits of the mantissa.
_FP32_ROUND_INT = float(2**23 + 2**22)
LOG2_E = 1.4426950408889634
NEG_INF = float("-inf")
LN_2 = 0.6931471805599453
# `MASK_R2P_CHUNK_SIZE` (mask.py:15).
MASK_R2P_CHUNK = 32


def _scale_gather(dst, src, cols, scale):
    """Gather one output group's columns from the O fragment and scale them.

    A plain Python function, so its loop runs at trace time and ``cols`` -- a
    Python list -- can index the fragment directly. A ``for`` inside a traced
    body would become a TIR loop and the list index would fail.
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
    assembled with ``mov.b64`` and split again afterwards -- the same shape
    ``flashmla/sparse_prefill_head64_phase1.py`` uses. This is one instruction
    with two ordered results, not two scalar operations.
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
    """Three-input ``max.f32``, an SM100 form, writing into ``dst[idx]``.

    The reference's reduction tree is built on it (``utils.fmax`` passes NVVM a
    third operand), which halves the tree: 66 instructions per 128-element row
    instead of the 127 a two-input tree would need.
    """
    K.ptx.max.f32(dst[idx], a, b, c)


def _row_max_128(regs, out, out_idx):
    """`fmax_reduce` for arch 100 with a size divisible by 8 (utils.py:258-278).

    Four accumulators seeded from the first eight elements, each absorbing two
    more per step through a three-input max, then folded pairwise.
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
    """`fadd_exp2_scaled_reduce` for arch 100: sum of 2**(scale * x).

    A second pass over the same 128 elements, not a rescaling of the main row
    sum: the temperature LSE needs the exponentials at a different scale, so
    every element is multiplied, exponentiated and accumulated again (:2288-2292).
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
    """`fadd_reduce` for arch 100 (utils.py:290-303): four packed accumulators."""
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

    The reference uses ``add.s32`` deliberately -- it lowers to LEA on the ALU
    pipe, where ``add.u32`` would lower to IMAD and contend with the FMA pipe
    the polynomial itself is using.
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
    """`ex2_emulation_2` (utils.py:987-1005): 2**x for a pair, without MUFU.

    Clamp, split off the integer part by a round-down add of 2**23 + 2**22,
    evaluate the fractional part with a degree-3 packed Horner, and reassemble
    by shifting the integer into the exponent field. The reference mixes one of
    these in per sixteen elements on the causal path, which is what keeps the
    MUFU pipe from becoming the softmax bottleneck.
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
    # Horner, unrolled at trace time: the coefficients are Python floats.
    _packed_f32x2(
        "fma.rn.f32x2",
        poly,
        0,
        1,
        poly[0],
        poly[1],
        frac[0],
        frac[1],
        K.float32(0.22756439447402954),
        K.float32(0.22756439447402954),
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
        K.float32(0.6951461434364319),
        K.float32(0.6951461434364319),
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
        K.float32(1.0),
        K.float32(1.0),
    )
    K.assign(regs[i], _combine_int_frac_ex2(rounded[0], poly[0]))
    K.assign(regs[j], _combine_int_frac_ex2(rounded[1], poly[1]))


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


def bar_sync_named(bar_id, count):
    """``bar.sync <id>, <count>`` -- a named barrier over a thread subset."""
    K.ptx.bar.sync(K.uint32(bar_id), K.uint32(count))


def _dequant_kv(src, dst, bar_tma, bar_ready, bar_id, count_raw, group_tidx, batch):
    """Convert one staged fp8 KV tile into the bf16 tile the MMA reads.

    A strided task loop over ``n_block * (head_dim / 16)`` chunks: sixteen fp8
    in through one 128-bit shared load, sixteen bf16 out through two, with the
    reference's byte-permute conversion between them. Both tiles carry the same
    logical (kv, head_dim) indexing in this port, so K and V differ only in
    which buffers they name.

    The two forms below differ only in how the loads are scheduled, and
    ``batch`` picks between them at trace time.
    """
    if batch == 1:
        # Both warpgroups are dequantizing -- BF16 Q over FP8 KV, WG0 on K and
        # WG1 on V. Batching the loads only makes 256 threads burst into the
        # same shared memory at once, and the rolled loop measured faster. A
        # `While`, because a counted loop gets no no-unroll pragma from TVM.
        with K.If(count_raw > 0), K.Then():
            bar_tma.wait(0, 0)
            chunks_per_row = HEAD_DIM // 16
            it = K.alloc_local((1,), "int32")
            K.assign(it[0], 0)
            with K.While(it[0] < DEQUANT_ITERS):
                task = it[0] * SOFTMAX_THREADS + group_tidx
                row = udiv_i32(task, chunks_per_row)
                chunk = task - row * chunks_per_row
                src_words = K.alloc_local((4,), "uint32")
                K.ptx.ld.shared.v4.b32(
                    src_words[0],
                    src_words[1],
                    src_words[2],
                    src_words[3],
                    src.ptr_to([row, chunk * 16]),
                )
                out_words = K.alloc_local((8,), "uint32")
                # Inlined rather than factored into a helper: nesting `@K.inline` calls
                # turns the trace-time `w`/`half` into runtime variables and the register
                # file into indexed local memory.
                #
                # Four packed e4m3 bytes become two packed bf16x2 words. Not the hardware
                # `cvt.rn.bf16x2.e4m3x2`: the reference reassembles sign and mantissa by
                # hand and lets one `fma.rn.bf16x2` against the constant 0x7b807b80 apply
                # the exponent bias, keeping the conversion on the FMA pipe (utils.py:540).
                for w in range(4):
                    for half in range(2):
                        q = K.alloc_local((1,), "uint32")
                        mant = K.alloc_local((1,), "uint32")
                        acc = K.alloc_local((1,), "uint32")
                        K.ptx.prmt.b32(q[0], src_words[w], src_words[w], K.uint32(0x1302))
                        with K.If(half == 1), K.Then():
                            K.ptx.shl.b32(q[0], q[0], K.uint32(8))
                        K.ptx.and_.b32(acc[0], q[0], K.uint32(0x80008000))
                        K.ptx.and_.b32(mant[0], q[0], K.uint32(0x7F007F00))
                        K.ptx.shr.u32(mant[0], mant[0], K.uint32(4))
                        K.ptx.or_.b32(acc[0], acc[0], mant[0])
                        K.ptx.fma.rn.bf16x2(
                            out_words[w * 2 + half], acc[0], K.uint32(0x7B807B80), K.uint32(0)
                        )
                for half in range(2):
                    K.ptx.st.shared.v4.b32(
                        dst.ptr_to([row, chunk * 16 + half * 8]),
                        out_words[half * 4],
                        out_words[half * 4 + 1],
                        out_words[half * 4 + 2],
                        out_words[half * 4 + 3],
                    )
                K.assign(it[0], it[0] + 1)
            K.ptx.fence.proxy.async_.shared__cta()
            bar_sync_named(bar_id, SOFTMAX_THREADS)
            with K.If(group_tidx == 0), K.Then():
                bar_ready.arrive(0)
    else:
        # One warpgroup at work: issue a batch of loads before converting any of
        # them, so the conversion chain -- eight dependent integer ops before
        # the FMA -- stops sitting on the critical path of the next load.
        staged = K.alloc_local((4 * batch,), "uint32")
        with K.If(count_raw > 0), K.Then():
            bar_tma.wait(0, 0)
            chunks_per_row = HEAD_DIM // 16
            for chunk_group in range(DEQUANT_ITERS // batch):
                for bi in range(batch):
                    task_l = (chunk_group * batch + bi) * SOFTMAX_THREADS + group_tidx
                    row_l = udiv_i32(task_l, chunks_per_row)
                    chunk_l = task_l - row_l * chunks_per_row
                    K.ptx.ld.shared.v4.b32(
                        staged[bi * 4],
                        staged[bi * 4 + 1],
                        staged[bi * 4 + 2],
                        staged[bi * 4 + 3],
                        src.ptr_to([row_l, chunk_l * 16]),
                    )
                for bi in range(batch):
                    task = (chunk_group * batch + bi) * SOFTMAX_THREADS + group_tidx
                    row = udiv_i32(task, chunks_per_row)
                    chunk = task - row * chunks_per_row
                    out_words = K.alloc_local((8,), "uint32")
                    # Inlined rather than factored into a helper: nesting `@K.inline` calls
                    # turns the trace-time `w`/`half` into runtime variables and the register
                    # file into indexed local memory.
                    #
                    # Four packed e4m3 bytes become two packed bf16x2 words. Not the hardware
                    # `cvt.rn.bf16x2.e4m3x2`: the reference reassembles sign and mantissa by
                    # hand and lets one `fma.rn.bf16x2` against the constant 0x7b807b80 apply
                    # the exponent bias, keeping the conversion on the FMA pipe (utils.py:540).
                    for w in range(4):
                        for half in range(2):
                            q = K.alloc_local((1,), "uint32")
                            mant = K.alloc_local((1,), "uint32")
                            acc = K.alloc_local((1,), "uint32")
                            K.ptx.prmt.b32(
                                q[0], staged[bi * 4 + w], staged[bi * 4 + w], K.uint32(0x1302)
                            )
                            with K.If(half == 1), K.Then():
                                K.ptx.shl.b32(q[0], q[0], K.uint32(8))
                            K.ptx.and_.b32(acc[0], q[0], K.uint32(0x80008000))
                            K.ptx.and_.b32(mant[0], q[0], K.uint32(0x7F007F00))
                            K.ptx.shr.u32(mant[0], mant[0], K.uint32(4))
                            K.ptx.or_.b32(acc[0], acc[0], mant[0])
                            K.ptx.fma.rn.bf16x2(
                                out_words[w * 2 + half], acc[0], K.uint32(0x7B807B80), K.uint32(0)
                            )
                    for half in range(2):
                        K.ptx.st.shared.v4.b32(
                            dst.ptr_to([row, chunk * 16 + half * 8]),
                            out_words[half * 4],
                            out_words[half * 4 + 1],
                            out_words[half * 4 + 2],
                            out_words[half * 4 + 3],
                        )
            K.ptx.fence.proxy.async_.shared__cta()
            bar_sync_named(bar_id, SOFTMAX_THREADS)
            with K.If(group_tidx == 0), K.Then():
                bar_ready.arrive(0)


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


def _mbar_expect_tx(bar, stage, tx_bytes):
    """``mbarrier.expect_tx``: promise bytes without arriving.

    The single-shot K/V barriers take their transaction count in the prologue
    from thread 0 (:911-918) and are arrived on later by the load warp, so this
    has to stay separate from the arrive -- the barrier's arrival count is 1.
    """
    K.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(bar.ptr_to([stage]), K.uint32(tx_bytes))


LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

# The four in-scope dtype combinations, named by the storage dtypes plus the
# MMA operand dtypes they resolve to at :318-364. `qk`/`pv` narrower than the
# storage dtype is not a thing; `qk`/`pv` *wider* is the fp8 -> bf16 staging
# path, which dequantizes K or V in shared memory before the MMA sees it.
DTYPE_MODES = {
    "bf16": {"q": "bfloat16", "k": "bfloat16", "v": "bfloat16", "qk": "bfloat16", "pv": "bfloat16"},
    "fp8": {
        "q": "float8_e4m3",
        "k": "float8_e4m3",
        "v": "float8_e4m3",
        "qk": "float8_e4m3",
        "pv": "float8_e4m3",
    },
    "bf16q_fp8kv": {
        "q": "bfloat16",
        "k": "float8_e4m3",
        "v": "float8_e4m3",
        "qk": "bfloat16",
        "pv": "bfloat16",
    },
    "fp8_pvbf16": {
        "q": "float8_e4m3",
        "k": "float8_e4m3",
        "v": "float8_e4m3",
        "qk": "float8_e4m3",
        "pv": "bfloat16",
    },
}

# `mO_partial.element_type`, validated at :372-377.
PARTIAL_DTYPES = ("float32", "bfloat16", "float16", "float8_e4m3")

_TORCH_DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float8_e4m3": "float8_e4m3fn",
}


def _torch_dtype(name: str):
    import torch

    return getattr(torch, _TORCH_DTYPES[name])


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
def _make_kernel(**config):
    """Trace one native Kern specialization and its exact launch ABI."""
    qheadperkv = int(config["qhead_per_kv"])
    causal = bool(config.get("causal", True))
    dtype_mode = str(config.get("dtype", "bf16"))
    partial_dtype = str(config.get("partial_dtype", "float32"))
    temperature = bool(config.get("temperature"))
    paged = bool(config.get("paged", False))
    seqused = bool(config.get("seqused", False))
    if seqused and not paged:
        raise ValueError("seqused_k is only supported together with page_table")

    # Trace-time register split and exp2 mix; see the constants block.
    num_regs_softmax = NUM_REGS_SOFTMAX_CAUSAL if causal else NUM_REGS_SOFTMAX_NONCAUSAL
    num_regs_store = NUM_REGS_STORE_CAUSAL if causal else NUM_REGS_STORE_NONCAUSAL
    ex2_emu_freq = EX2_EMU_FREQ_CAUSAL if causal else 0

    mode = DTYPE_MODES[dtype_mode]
    q_dtype = mode["q"]
    k_dtype = mode["k"]
    v_dtype = mode["v"]
    qk_dtype = mode["qk"]
    pv_dtype = mode["pv"]
    # `k_fp8_to_bf16` / `v_fp8_to_bf16` (:354-361): an fp8 tile that the MMA has
    # to see as bf16 is staged through a second shared buffer and dequantized by
    # a softmax warpgroup.
    k_stage_fp8 = k_dtype != qk_dtype
    v_stage_fp8 = v_dtype != pv_dtype
    q_ty = _tirx_dtype(q_dtype)
    k_ty = _tirx_dtype(k_dtype)
    v_ty = _tirx_dtype(v_dtype)
    qk_ty = _tirx_dtype(qk_dtype)
    pv_ty = _tirx_dtype(pv_dtype)
    partial_ty = _tirx_dtype(partial_dtype)
    qk_mma_kind = _MMA_KIND[qk_dtype]
    pv_mma_kind = _MMA_KIND[pv_dtype]
    q_bytes = _DTYPE_BYTES[q_dtype]
    k_bytes = _DTYPE_BYTES[k_dtype]
    v_bytes = _DTYPE_BYTES[v_dtype]
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

        # -------------------------------------------------------------------
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
        # Paging appends a page mode and pins the token coordinate to 0; it changes
        # nothing about the first two dims. Dims are fastest-first, so the flat form
        # is (head_dim, total_k, head_kv) and the paged one
        # (head_dim, page_size, head_kv, num_pages) -- the `layout_t = [2,3,1,0]`
        # host permute (:380-397). A rank-3 encode against a rank-4 tensor is silent
        # garbage, which is why the rank follows the tensor and not the caller.
        k_map = K.stack_alloca("tensormap", 1)
        if paged:
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                k_map,
                k_ty,
                4,
                k.data,
                HEAD_DIM,
                N_BLOCK,
                num_heads_kv,
                total_k // N_BLOCK,
                HEAD_DIM * k_bytes,
                N_BLOCK * HEAD_DIM * k_bytes,
                num_heads_kv * N_BLOCK * HEAD_DIM * k_bytes,
                _swizzle_elems(k_bytes),
                N_BLOCK,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                _SWIZZLE_128B,
                _L2_PROMOTION_256B,
                0,
            )
        else:
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                k_map,
                k_ty,
                3,
                k.data,
                HEAD_DIM,
                num_heads_kv,
                total_k,
                HEAD_DIM * k_bytes,
                num_heads_kv * HEAD_DIM * k_bytes,
                _swizzle_elems(k_bytes),
                1,
                N_BLOCK,
                1,
                1,
                1,
                0,
                _SWIZZLE_128B,
                _L2_PROMOTION_256B,
                0,
            )
        v_map = K.stack_alloca("tensormap", 1)
        if paged:
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                v_map,
                v_ty,
                4,
                v.data,
                HEAD_DIM,
                N_BLOCK,
                num_heads_kv,
                total_k // N_BLOCK,
                HEAD_DIM * v_bytes,
                N_BLOCK * HEAD_DIM * v_bytes,
                num_heads_kv * N_BLOCK * HEAD_DIM * v_bytes,
                _swizzle_elems(v_bytes),
                N_BLOCK,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                _SWIZZLE_128B,
                _L2_PROMOTION_256B,
                0,
            )
        else:
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                v_map,
                v_ty,
                3,
                v.data,
                HEAD_DIM,
                num_heads_kv,
                total_k,
                HEAD_DIM * v_bytes,
                num_heads_kv * HEAD_DIM * v_bytes,
                _swizzle_elems(v_bytes),
                1,
                N_BLOCK,
                1,
                1,
                1,
                0,
                _SWIZZLE_128B,
                _L2_PROMOTION_256B,
                0,
            )
        # The fp8 staging path lands in a PLAIN buffer, so its descriptor must carry
        # no swizzle. The reference builds a separate `cpasync.make_tiled_tma_atom`
        # over a row-major box for exactly this reason (:474-479); reusing the
        # swizzled MMA descriptor writes permuted bytes that the dequantization then
        # reads back in logical order, which silently scrambles every row.
        k_stage_map = v_stage_map = None
        if k_stage_fp8 or v_stage_fp8:
            # The staging descriptor follows the same rank as the MMA one: paging is
            # a property of the tensor, not of who reads it. Leaving this rank-3
            # while the load arm issues the 4-D copy is an illegal instruction, and
            # the two are selected by different flags -- rank by `paged`, staging by
            # `k_stage_fp8` -- so they have to be kept in step by hand.
            k_stage_map = K.stack_alloca("tensormap", 1)
            if paged:
                K.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    k_stage_map,
                    k_ty,
                    4,
                    k.data,
                    HEAD_DIM,
                    N_BLOCK,
                    num_heads_kv,
                    total_k // N_BLOCK,
                    HEAD_DIM * k_bytes,
                    N_BLOCK * HEAD_DIM * k_bytes,
                    num_heads_kv * N_BLOCK * HEAD_DIM * k_bytes,
                    HEAD_DIM,
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
                    k_stage_map,
                    k_ty,
                    3,
                    k.data,
                    HEAD_DIM,
                    num_heads_kv,
                    total_k,
                    HEAD_DIM * k_bytes,
                    num_heads_kv * HEAD_DIM * k_bytes,
                    HEAD_DIM,
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
            v_stage_map = K.stack_alloca("tensormap", 1)
            if paged:
                K.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    v_stage_map,
                    v_ty,
                    4,
                    v.data,
                    HEAD_DIM,
                    N_BLOCK,
                    num_heads_kv,
                    total_k // N_BLOCK,
                    HEAD_DIM * v_bytes,
                    N_BLOCK * HEAD_DIM * v_bytes,
                    num_heads_kv * N_BLOCK * HEAD_DIM * v_bytes,
                    HEAD_DIM,
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
                    v_stage_map,
                    v_ty,
                    3,
                    v.data,
                    HEAD_DIM,
                    num_heads_kv,
                    total_k,
                    HEAD_DIM * v_bytes,
                    num_heads_kv * HEAD_DIM * v_bytes,
                    HEAD_DIM,
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

        return k_map, v_map, k_stage_map, v_stage_map, q_map

    def trace(values, host):
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
        k_map, v_map, k_stage_map, v_stage_map, q_map = host

        # CUDA TRANSCRIPTION START
        block = K.cta_id()
        tidx = K.thread_id()
        # `make_warp_uniform(warp_idx())` (:706) lowers to a lane-0 shfl broadcast.
        # `K.warp_id` is warp-uniform by construction, so the broadcast is redundant
        # here rather than load-bearing.
        warp_idx = K.warp_id()
        lane_idx = K.lane_id()

        # Work-item decode and the CTA-level early-out (:689-705).
        # The grid is sized by the work list's CAPACITY, so the tail CTAs retire.
        work_count_val = ld_global_i32(work_count, 0)
        cta_valid_work = K.cast(block < work_count_val, "int32")
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
        if k_stage_fp8:
            s_k_fp8 = pool.alloc((N_BLOCK, HEAD_DIM), k_ty, align=1024)
        if v_stage_fp8:
            s_v_fp8 = pool.alloc((N_BLOCK, HEAD_DIM), v_ty, align=1024)

        s_scale = pool.alloc((O_STAGE * M_BLOCK * 2,), "float32", align=16)
        if temperature:
            s_scale_temp = pool.alloc((O_STAGE * M_BLOCK,), "float32", align=16)
        s_split_idx = pool.alloc((O_STAGE * q_tokens_per_group,), "int32", align=16)
        s_q_idx = pool.alloc((O_STAGE * q_tokens_per_group,), "int32", align=16)
        s_row_meta = pool.alloc((8,), "int32", align=16)
        s_diag_q_count = pool.alloc((1,), "int32", align=16)
        # `sPagedKvIdx` (:581): the physical page for this CTA's KV block. Allocated
        # unconditionally, exactly as the source does, so flat and paged share every
        # later offset in the pool; written by thread 0 and read by both KV-load
        # warps under `paged`.
        s_paged_kv_idx = pool.alloc((1,), "int32", align=16)
        s_q_load_m_idx = pool.alloc((Q_STAGE * q_tokens_per_group,), "int32", align=16)
        s_qidx_meta = pool.alloc((QIDX_META_STAGES * q_tokens_per_group,), "int32", align=16)

        bar_k = K.MBarrier(pool, 1)
        bar_v = K.MBarrier(pool, 1)
        if k_stage_fp8:
            bar_k_tma = K.MBarrier(pool, 1)
        if v_stage_fp8:
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
        # TMEM is an instruction-level column space in Kern. Keep the source's
        # allocation map as raw columns instead of creating a TIR tmem buffer:
        # S0/S1 occupy [0, 2*N_BLOCK), O0/O1 follow them.
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
            # One issue, not thirty-two: the prefetch is warp-uniform and rides the
            # memory pipeline, so letting the whole warp issue it costs L1TEX slots
            # for nothing. The reference elects a single lane (:663-669).
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(k_map))
                K.ptx.prefetch.tensormap(K.address_of(v_map))
                K.ptx.prefetch.tensormap(K.address_of(q_map))

        with K.If(tidx == 0), K.Then():
            base_row_start = ld_global_i32(
                k2q_row_ptr, head_kv_idx[0] * (total_rows + 1) + row_linear[0]
            )
            row_start = base_row_start + work_q_begin[0]
            count_raw = work_q_count[0]
            # Upstream reads this as `mPageTable.shape[1]` (:211). The table is
            # rectangular because `page_size == blk_kv`, so `num_batches *
            # pages_per_seq == num_pages == total_k / page_size`, and the quotient
            # of two ABI scalars recovers it without a new argument.
            if paged:
                pages_per_seq = udiv_i32(udiv_i32(total_k, N_BLOCK), num_batches)
            # `_logical_seqlen_k` (:204-212), in the source's own priority order.
            # `seqused` is tested FIRST: checking paged first would silently
            # substitute the zero-padded paged capacity for a shorter supplied
            # length.
            if seqused:
                seqlen_k = ld_global_i32(seqused_k, batch_idx[0])
            elif paged:
                seqlen_k = pages_per_seq * N_BLOCK
            else:
                seqlen_k = ld_global_i32(cu_seqlens_k, batch_idx[0] + 1) - ld_global_i32(
                    cu_seqlens_k, batch_idx[0]
                )
            kv_valid_cols = K.min(K.max(seqlen_k - kv_block_idx[0] * N_BLOCK, 0), N_BLOCK)
            q_batch_offset = ld_global_i32(cu_seqlens_q, batch_idx[0])
            # Paging offsets by page, not by batch (:880-884).
            if paged:
                k_batch_offset = 0
            else:
                k_batch_offset = ld_global_i32(cu_seqlens_k, batch_idx[0])
            st_shared_i32(s_row_meta, 0, batch_idx[0])
            st_shared_i32(s_row_meta, 1, kv_block_idx[0])
            st_shared_i32(s_row_meta, 2, row_start)
            st_shared_i32(s_row_meta, 3, count_raw)
            st_shared_i32(s_row_meta, 4, kv_valid_cols)
            st_shared_i32(s_row_meta, 5, q_batch_offset)
            st_shared_i32(s_row_meta, 6, k_batch_offset)
            # Computed ONLY under `const_expr(self.causal)` (:892-901); a non-causal
            # build never loads `cu_seqlens_q[batch+1]` at all. Deliberately
            # unclamped: with `seqused_k` shorter than the Q length the offset goes
            # negative, the leading `seqlen_q - seqused_k[b]` queries have no legal
            # key, and their neutral partial is O = 0 with LSE = -inf. Clamping it
            # to a finite value would diverge from the source.
            causal_q_offset = K.alloc_local((1,), "int32")
            K.assign(causal_q_offset[0], 0)
            if causal:
                seqlen_q = ld_global_i32(cu_seqlens_q, batch_idx[0] + 1) - q_batch_offset
                K.assign(causal_q_offset[0], seqlen_k - seqlen_q)
            st_shared_i32(s_row_meta, 7, causal_q_offset[0])

            # The physical page for this CTA's KV block, resolved once by thread 0
            # (:903-906, paged_kv.py:60-65) and published under the same fence as
            # the row metadata below -- paging adds no barrier of its own.
            if paged:
                st_shared_i32(
                    s_paged_kv_idx,
                    0,
                    ld_global_i32(page_table, batch_idx[0] * pages_per_seq + kv_block_idx[0]),
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
                q_threshold = (kv_block_idx[0] * N_BLOCK + kv_valid_cols) - causal_q_offset[0]
                lo = K.alloc_local((1,), "int32")
                hi = K.alloc_local((1,), "int32")
                K.assign(lo[0], 0)
                K.assign(hi[0], count_raw)
                # A `While`, not a counted loop: TVM's C codegen consults no loop
                # annotation, so `K.serial(..., unroll=False)` still leaves nvcc free
                # to unroll, and several copies of a body that is one dependent
                # global load cost more than the loop overhead they remove. The
                # condition also retires the search as soon as it converges instead
                # of predicating off the remaining fixed steps, which is most of them
                # whenever the CSR row is short.
                with K.While(lo[0] < hi[0]):
                    mid = udiv_i32(lo[0] + hi[0], 2)
                    probe = ld_global_i32(k2q_q_indices, head_kv_idx[0] * nnz + row_start + mid)
                    with K.If(probe < q_threshold):
                        with K.Then():
                            K.assign(lo[0], mid + 1)
                        with K.Else():
                            K.assign(hi[0], mid)
                K.assign(diag_q_count[0], lo[0])
            st_shared_i32(s_diag_q_count, 0, diag_q_count[0])

            bar_k.init(1)
            bar_v.init(1)
            if k_stage_fp8:
                bar_k_tma.init(1)
                _mbar_expect_tx(bar_k_tma, 0, N_BLOCK * HEAD_DIM * k_bytes)
            else:
                _mbar_expect_tx(bar_k, 0, N_BLOCK * HEAD_DIM * k_bytes)
            if v_stage_fp8:
                bar_v_tma.init(1)
                _mbar_expect_tx(bar_v_tma, 0, N_BLOCK * HEAD_DIM * v_bytes)
            else:
                _mbar_expect_tx(bar_v, 0, N_BLOCK * HEAD_DIM * v_bytes)

        # Warp 1, not warp 0: thread 0 is already serialising the whole CTA behind
        # its metadata chain, whose binary search is 32 dependent global loads, and
        # every other thread is waiting at the barrier below. The pipe mbarriers do
        # not depend on any of that, so initialising them from another warp puts the
        # two on top of each other. The fence and CTA barrier that follow order both
        # against every consumer, so visibility is unchanged.
        with K.If(warp_idx == 1), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
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

        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        # -----------------------------------------------------------------------
        # Role dispatch. A flat sequence of independent `if` blocks, each ANDed
        # with `cta_valid_work` -- not an if/elif chain (:991-1235).
        # -----------------------------------------------------------------------
        with K.If(warp_idx == TOTAL_WARPS - 1), K.Then():
            # Warp 15 is idle and is NOT gated on cta_valid_work (:991-992).
            K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(NUM_REGS_OTHER))

        # -----------------------------------------------------------------------
        # ROLE: Q-load warpgroup, warps 8..11 (:994-1047).
        # -----------------------------------------------------------------------
        with (
            K.If(K.And(tidx >= Q_LOAD_WARP_BASE * WARP_SIZE, tidx < MMA_WARP_ID * WARP_SIZE)),
            K.Then(),
        ):
            with K.If(cta_valid_work != 0), K.Then():
                K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(num_regs_store))
                q_row_start = ld_shared_i32(s_row_meta, 2)
                q_count_raw = ld_shared_i32(s_row_meta, 3)
                q_batch_off = ld_shared_i32(s_row_meta, 5)
                # Deliberately NOT gated on KV validity: a sparse entry past the
                # sequence still runs the all-masked path so its partial is neutral
                # (:1007-1009).
                with K.If(q_count_raw > 0), K.Then():
                    num_q_groups_load = uceil_div_i32(q_count_raw, q_tokens_per_group)
                    warp_in_wg = warp_idx - Q_LOAD_WARP_BASE
                    # `q_oob_m_idx = mQ_2d.shape[0] // qheadperkv` (:1855) -- one
                    # past the last Q *tile*, so an absent token gathers out of
                    # bounds and takes the descriptor's OOB fill. `total_q` alone
                    # would be an in-range row whenever num_heads_kv > 1.
                    q_oob_m_idx = total_q * num_heads_kv

                    if not USE_GATHER4(qheadperkv):
                        with K.serial(0, num_q_groups_load, unroll=False) as qi_group:
                            slot = qi_group % Q_STAGE
                            phase = udiv_i32(qi_group, Q_STAGE) & 1
                            with K.If(warp_in_wg == 0), K.Then():
                                bar_q_empty.wait(slot, phase ^ 1)
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        bar_q_full.ptr_to([slot]),
                                        K.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                    )
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                            load_meta_slot = slot * q_tokens_per_group
                            qidx_meta_slot = (
                                qi_group & (QIDX_META_STAGES - 1)
                            ) * q_tokens_per_group
                            # One warp's low lanes publish the whole group: with
                            # qheadperkv >= 8 a group is at most 16 tokens
                            # (:1883-1898).
                            with (
                                K.If(K.And(warp_in_wg == 0, lane_idx < q_tokens_per_group)),
                                K.Then(),
                            ):
                                qi = qi_group * q_tokens_per_group + lane_idx
                                with K.If(qi < q_count_raw):
                                    with K.Then():
                                        word = ld_global_i32(
                                            k2q_qsplit_indices,
                                            head_kv_idx[0] * nnz + q_row_start + qi,
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
                                tok = warp_in_wg * TOKENS_PER_WARP(qheadperkv) + qi_slot
                                with K.If(tok < q_tokens_per_group), K.Then():
                                    m_tile = ld_shared_i32(s_q_load_m_idx, load_meta_slot + tok)
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
                                                _Q_TMA_CACHE_HINT,
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
                            slot = qi_group % Q_STAGE
                            phase = udiv_i32(qi_group, Q_STAGE) & 1
                            with K.If(warp_in_wg == 0), K.Then():
                                bar_q_empty.wait(slot, phase ^ 1)
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        bar_q_full.ptr_to([slot]),
                                        K.uint32(M_BLOCK * HEAD_DIM * q_bytes),
                                    )
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                            qidx_meta_slot = (
                                K.bitwise_and(qi_group, QIDX_META_STAGES - 1) * q_tokens_per_group
                            )
                            # This path's groups hold 32, 64 or 128 tokens, so the
                            # publish takes several sweeps of the whole warpgroup
                            # (:1671-1690).
                            with K.unroll(meta_iters) as meta_iter:
                                tok_g4 = (
                                    meta_iter * NUM_Q_LOAD_WARPS + warp_in_wg
                                ) * WARP_SIZE + lane_idx
                                with K.If(tok_g4 < q_tokens_per_group), K.Then():
                                    qi_g4 = qi_group * q_tokens_per_group + tok_g4
                                    with K.If(qi_g4 < q_count_raw):
                                        with K.Then():
                                            st_shared_i32(
                                                s_qidx_meta,
                                                qidx_meta_slot + tok_g4,
                                                ld_global_i32(
                                                    k2q_qsplit_indices,
                                                    head_kv_idx[0] * nnz + q_row_start + qi_g4,
                                                ),
                                            )
                                        with K.Else():
                                            st_shared_i32(s_qidx_meta, qidx_meta_slot + tok_g4, 0)
                            bar_sync_named(BAR_LOAD_WG, SOFTMAX_THREADS)

                            with K.If(K.cuda.elect_sync()), K.Then():
                                with K.unroll(gathers_per_warp) as gather_slot:
                                    gather_idx = gather_slot * NUM_Q_LOAD_WARPS + warp_in_wg
                                    tok_base = gather_idx * tokens_per_gather4
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
                                                _Q_TMA_CACHE_HINT,
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
                                            _Q_TMA_CACHE_HINT,
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
        # (:1049-1086, :1519-1621).
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
                kv_block_load = ld_shared_i32(s_row_meta, 1)
                k_batch_off = ld_shared_i32(s_row_meta, 6)
                kv_has_work = K.cast(ld_shared_i32(s_row_meta, 3) > 0, "int32")
                with K.If(kv_has_work != 0), K.Then():
                    kv_row_start = k_batch_off + kv_block_load * N_BLOCK
                    # Read warp-wide, in this warp's own arm, AFTER the has-work
                    # early-out (:1569 for K, :1610 for V -- two static loads in the
                    # module, not one hoisted read). Within an arm the value feeds
                    # every subtile issue.
                    with K.If(warp_idx == KV_LOAD_WARP_BASE), K.Then():
                        page_idx_k = K.alloc_local((1,), "int32")
                        if paged:
                            K.assign(page_idx_k[0], ld_shared_i32(s_paged_kv_idx, 0))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            with K.unroll(KV_SUBTILES(k_bytes)) as sub:
                                # PAGED_KV selects the rank and the coordinate tuple
                                # and nothing else; the staging destination is an
                                # independent axis (:1332-1340), so the two must not
                                # be composed with an `elif`.
                                if paged:
                                    K.ptx[_TMA_G2S_4D_CACHE](
                                        K.ptr_byte_offset(
                                            (s_k_fp8 if k_stage_fp8 else s_k).ptr_to([0, 0]),
                                            sub * N_BLOCK * _swizzle_elems(k_bytes) * k_bytes,
                                            k_ty,
                                        ),
                                        K.address_of(k_stage_map if k_stage_fp8 else k_map),
                                        K.int32(sub * _swizzle_elems(k_bytes)),
                                        K.int32(0),
                                        head_kv_idx[0],
                                        page_idx_k[0],
                                        K.cuda.cvta_generic_to_shared(
                                            (bar_k_tma if k_stage_fp8 else bar_k).ptr_to([0])
                                        ),
                                        _KV_TMA_CACHE_HINT,
                                    )
                                else:
                                    K.ptx[_TMA_G2S_3D_CACHE](
                                        K.ptr_byte_offset(
                                            (s_k_fp8 if k_stage_fp8 else s_k).ptr_to([0, 0]),
                                            sub * N_BLOCK * _swizzle_elems(k_bytes) * k_bytes,
                                            k_ty,
                                        ),
                                        K.address_of(k_stage_map if k_stage_fp8 else k_map),
                                        K.int32(sub * _swizzle_elems(k_bytes)),
                                        head_kv_idx[0],
                                        kv_row_start,
                                        K.cuda.cvta_generic_to_shared(
                                            (bar_k_tma if k_stage_fp8 else bar_k).ptr_to([0])
                                        ),
                                        _KV_TMA_CACHE_HINT,
                                    )
                            # OUTSIDE the paged/flat conditional, as in the source
                            # (:1579-1580): an arm that omitted it would never signal
                            # the barrier and the MMA warp's wait would hang.
                            K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                                (bar_k_tma if k_stage_fp8 else bar_k).ptr_to([0]), K.uint32(1)
                            )
                    with K.If(warp_idx == KV_LOAD_WARP_BASE + 1), K.Then():
                        page_idx_v = K.alloc_local((1,), "int32")
                        if paged:
                            K.assign(page_idx_v[0], ld_shared_i32(s_paged_kv_idx, 0))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            with K.unroll(KV_SUBTILES(v_bytes)) as sub:
                                if paged:
                                    K.ptx[_TMA_G2S_4D_CACHE](
                                        K.ptr_byte_offset(
                                            (s_v_fp8 if v_stage_fp8 else s_v).ptr_to([0, 0]),
                                            sub * N_BLOCK * _swizzle_elems(v_bytes) * v_bytes,
                                            v_ty,
                                        ),
                                        K.address_of(v_stage_map if v_stage_fp8 else v_map),
                                        K.int32(sub * _swizzle_elems(v_bytes)),
                                        K.int32(0),
                                        head_kv_idx[0],
                                        page_idx_v[0],
                                        K.cuda.cvta_generic_to_shared(
                                            (bar_v_tma if v_stage_fp8 else bar_v).ptr_to([0])
                                        ),
                                        _KV_TMA_CACHE_HINT,
                                    )
                                else:
                                    K.ptx[_TMA_G2S_3D_CACHE](
                                        K.ptr_byte_offset(
                                            (s_v_fp8 if v_stage_fp8 else s_v).ptr_to([0, 0]),
                                            sub * N_BLOCK * _swizzle_elems(v_bytes) * v_bytes,
                                            v_ty,
                                        ),
                                        K.address_of(v_stage_map if v_stage_fp8 else v_map),
                                        K.int32(sub * _swizzle_elems(v_bytes)),
                                        head_kv_idx[0],
                                        kv_row_start,
                                        K.cuda.cvta_generic_to_shared(
                                            (bar_v_tma if v_stage_fp8 else bar_v).ptr_to([0])
                                        ),
                                        _KV_TMA_CACHE_HINT,
                                    )
                            K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                                (bar_v_tma if v_stage_fp8 else bar_v).ptr_to([0]), K.uint32(1)
                            )
                    if k_stage_fp8 or v_stage_fp8:
                        bar_sync_named(BAR_KV_LOAD, WARP_SIZE * NUM_KV_LOAD_WARPS)

        # -----------------------------------------------------------------------
        # ROLE: the single MMA-issue warp, warp 12; also the TMEM allocator warp
        # (:1088-1111, :1938-2183).
        # -----------------------------------------------------------------------
        with K.If(warp_idx == MMA_WARP_ID), K.Then():
            with K.If(cta_valid_work != 0), K.Then():
                K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(NUM_REGS_OTHER))
                K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    K.address_of(tmem_start_addr[0]), K.uint32(TMEM_TOTAL)
                )
                # The retrieve barrier spans both softmax warpgroups plus this warp
                # (:767-772). On the fp8 paths that also makes the first QK wait for
                # the shared-memory dequantization, without a separate edge.
                bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

                mma_count_raw = ld_shared_i32(s_row_meta, 3)
                with K.If(mma_count_raw > 0), K.Then():
                    num_q_groups_mma = uceil_div_i32(mma_count_raw, q_tokens_per_group)

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

                    bar_k.wait(0, 0)

                    # Issue order (:2070-2077): Q0K, Q1K, P0V, Q2K, P1V, ...
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
                        pv_qi = qi - 2
                        pv_slot = K.bitwise_and(pv_qi, 1)
                        pv_phase = udiv_i32(pv_qi, 2) & 1
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

                        q_slot = qi % Q_STAGE
                        q_phase = udiv_i32(qi, Q_STAGE) & 1
                        s_slot = K.bitwise_and(qi, 1)
                        s_phase = udiv_i32(qi, 2) & 1
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
                    drain_begin = K.if_then_else(num_q_groups_mma == 1, 0, num_q_groups_mma - 2)
                    with K.serial(drain_begin, num_q_groups_mma, unroll=False) as pv_qi2:
                        pv_slot2 = K.bitwise_and(pv_qi2, 1)
                        pv_phase2 = udiv_i32(pv_qi2, 2) & 1
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
        # (:1113-1235, :2186-2352, :2659-3020).
        #
        # `stage` is a compile-time constant, so this body is emitted once per
        # warpgroup -- which is what the export shows, every single-site operation
        # inside it appearing exactly twice.
        # -----------------------------------------------------------------------
        def epilogue_step(
            stage, qi_group, qidx_meta_slot, group_tidx, count_raw, q_batch_off, scale_log2
        ):
            """Scale O by 1/row_sum and store it, then store LSE (:2659-3020).

            This departs from the reference, which reads O with
            ``Ld16x256bOp(Repetition(8))``. That shape's register map on SM100 is
            ``row = warp*32 + lane//4 + 8*((r//2)%2)``,
            ``col = pass*64 + (lane%4)*2 + (r//4)*8 + r%2`` -- one instruction
            covering a strided 64-row subset, whose 128-row bookkeeping has no
            counterpart in the reference to transcribe. This port reads with
            ``32x32b.x32`` instead, whose map is the identity (thread = M row,
            register = column). That is the shape all
            three merged flashmla epilogues use. The stored values are unchanged;
            the load instruction count per warpgroup goes from 2 to 4, and the
            swizzle-inverse column remap drops out, because it exists only to undo
            the 16x256b fragment's permutation.
            """
            slot = K.bitwise_and(qi_group, 1)
            phase = udiv_i32(qi_group, 2) & 1
            bar_o_full.wait(slot, phase)

            # Decode the packed qsplit once per group into the 2-deep caches; every
            # per-store read below comes out of these, never out of s_qidx_meta.
            with K.If(group_tidx < q_tokens_per_group), K.Then():
                word = ld_shared_i32(s_qidx_meta, qidx_meta_slot + group_tidx)
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
            warp_in_wg = group_tidx // WARP_SIZE
            lane_in_warp = group_tidx - warp_in_wg * WARP_SIZE
            row_of_lane = warp_in_wg * 32 + lane_in_warp // 4
            col_of_lane = (lane_in_warp % 4) * 2

            # `lane_base` and `col_base` stay Python ints through the inline call,
            # which keeps the TMEM address and the group's column base constant.
            def load_o_instruction(o_regs, lane_base, col_base):
                K.ptx[_TMEM_LD_16](
                    *[o_regs[i] for i in range(32)],
                    K.cast(tmem_o_col + slot * tmem_o_stage_stride + col_base, "uint32")
                    + K.uint32(lane_base << 16),
                )

            def store_o_instruction(o_regs, lane_base, col_base):
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
                    row = row_of_lane + lane_base + 8 * parity
                    tok = udiv_i32(row, qheadperkv)
                    row_in_tok = row - tok * qheadperkv
                    qi = qi_group * q_tokens_per_group + tok
                    with K.If(qi < count_raw), K.Then():
                        # Re-read per store, as the reference does: nothing here is
                        # hoisted out of the column loop, the reciprocal included
                        # (:2785-2800, measured one of each per 128-bit store).
                        q_idx_e = ld_shared_i32(s_q_idx, slot * q_tokens_per_group + tok)
                        split_e = ld_shared_i32(s_split_idx, slot * q_tokens_per_group + tok)
                        row_sum_e = ld_shared_f32(s_scale, slot * M_BLOCK * 2 + row)
                        safe_sum = K.if_then_else(
                            K.Or(row_sum_e == K.float32(0.0), row_sum_e != row_sum_e),
                            K.float32(1.0),
                            row_sum_e,
                        )
                        row_scale = K.alloc_local((1,), "float32")
                        K.ptx.rcp.approx.ftz.f32(row_scale[0], safe_sum)
                        q_abs_e = q_batch_off + q_idx_e
                        flat_row = (
                            K.cast(split_e, "int64")
                            * K.cast(total_q, "int64")
                            * K.cast(head_q, "int64")
                            + K.cast(q_abs_e, "int64") * K.cast(head_q, "int64")
                            + K.cast(head_kv_idx[0] * qheadperkv + row_in_tok, "int64")
                        )
                        # The group's first register sits at column
                        # `col_base + (lane%4)*2 + 8*kb`; the fake-column map turns
                        # that into the contiguous output address.
                        store_col = _fake_col(partial_dtype, col_base + col_of_lane + 8 * kb)
                        scaled = K.alloc_local((_STORE_LANES[partial_dtype],), "float32")
                        _scale_gather(scaled, o_regs, regs, row_scale[0])
                        _store_o_partial(
                            o_partial,
                            flat_row * K.int64(HEAD_DIM) + K.cast(store_col, "int64"),
                            scaled,
                            partial_dtype,
                        )

            # All four reads are issued before the single wait. `wait::ld` drains
            # every TMEM load this thread has outstanding, so one after the last
            # issue still orders all four ahead of the first store that reads them
            # -- and the four now overlap instead of draining one at a time.
            o_regs_0 = K.alloc_local((32,), "float32")
            o_regs_1 = K.alloc_local((32,), "float32")
            o_regs_2 = K.alloc_local((32,), "float32")
            o_regs_3 = K.alloc_local((32,), "float32")
            load_o_instruction(o_regs_0, 0, 0)
            load_o_instruction(o_regs_1, 0, 64)
            load_o_instruction(o_regs_2, 16, 0)
            load_o_instruction(o_regs_3, 16, 64)
            K.ptx.tcgen05.wait__ld.sync.aligned()
            store_o_instruction(o_regs_0, 0, 0)
            store_o_instruction(o_regs_1, 0, 64)
            store_o_instruction(o_regs_2, 16, 0)
            store_o_instruction(o_regs_3, 16, 64)

            # LSE: one row per thread (:2987-3016).
            tok_l = udiv_i32(group_tidx, qheadperkv)
            h_local = group_tidx - tok_l * qheadperkv
            with K.If(qi_group * q_tokens_per_group + tok_l < count_raw), K.Then():
                row_sum_l = ld_shared_f32(s_scale, slot * M_BLOCK * 2 + group_tidx)
                row_max_l = ld_shared_f32(s_scale, slot * M_BLOCK * 2 + M_BLOCK + group_tidx)
                lg = K.alloc_local((1,), "float32")
                K.ptx.lg2.approx.ftz.f32(lg[0], row_sum_l)
                lse_val = K.if_then_else(
                    K.Or(row_sum_l == K.float32(0.0), row_sum_l != row_sum_l),
                    -K.infinity("float32"),
                    (row_max_l * scale_log2 + lg[0]) * K.float32(LN_2),
                )
                q_idx_l = ld_shared_i32(s_q_idx, slot * q_tokens_per_group + tok_l)
                split_l = ld_shared_i32(s_split_idx, slot * q_tokens_per_group + tok_l)
                h_abs = head_kv_idx[0] * qheadperkv + h_local
                lse_flat = (
                    K.cast(split_l, "int64") * K.cast(total_q, "int64") * K.cast(head_q, "int64")
                    + K.cast(q_batch_off + q_idx_l, "int64") * K.cast(head_q, "int64")
                    + K.cast(h_abs, "int64")
                )
                K.ptx.st.global_.f32(lse_partial.ptr_to([lse_flat]), lse_val)
                if temperature:
                    temp_sum = ld_shared_f32(s_scale_temp, slot * M_BLOCK + group_tidx)
                    lgt = K.alloc_local((1,), "float32")
                    K.ptx.lg2.approx.ftz.f32(lgt[0], temp_sum)
                    lse_t = K.if_then_else(
                        K.Or(temp_sum == K.float32(0.0), temp_sum != temp_sum),
                        NEG_INF,
                        (row_max_l * lse_temperature_scale_log2 + lgt[0]) * K.float32(LN_2),
                    )
                    K.ptx.st.global_.f32(lse_temperature_partial.ptr_to([lse_flat]), lse_t)

            bar_sync_named(BAR_EPILOGUE + stage, SOFTMAX_THREADS)
            bar_stats_empty.arrive(slot)
            bar_o_empty.arrive(slot)

        def softmax_warpgroup(stage):
            # Both softmax roles are consecutive 128-thread warpgroups, so the
            # low seven thread-rank bits are their exact role-local id.  This is
            # the native Kern pattern used by FlashAttention-4 as well; unlike
            # ``tidx - stage * 128`` it also makes the 0..127 range explicit to
            # unsigned quotient lowering without relying on a separate warp-id
            # predicate to constrain ``tidx``.
            group_tidx = K.local_scalar("int32", init=K.bitwise_and(tidx, SOFTMAX_THREADS - 1))
            kv_block_sm = ld_shared_i32(s_row_meta, 1)
            count_raw_sm = ld_shared_i32(s_row_meta, 3)
            kv_valid_cols = ld_shared_i32(s_row_meta, 4)
            q_batch_off_sm = ld_shared_i32(s_row_meta, 5)
            # Read unconditionally, as the source does (:1119-1127): both cells only
            # ever feed the diagonal mask, so a non-causal build leaves two dead
            # scalar loads behind rather than gating them away.
            causal_q_off = ld_shared_i32(s_row_meta, 7)
            diag_q_count_sm = ld_shared_i32(s_diag_q_count, 0)

            # FP8 staging (:1255-1303, :1488-1516). This is NOT in the load warps:
            # the TMA lands raw fp8 in the staging tile and a whole softmax
            # warpgroup converts it into the tile the MMA reads, before entering its
            # own loop. WG0 takes K, WG1 takes V.
            if stage == 0 and k_stage_fp8:
                _dequant_kv(
                    s_k_fp8,
                    s_k,
                    bar_k_tma,
                    bar_k,
                    BAR_KV_DEQUANT_K,
                    count_raw_sm,
                    group_tidx,
                    1 if v_stage_fp8 else DEQUANT_BATCH,
                )
            if stage == 1 and v_stage_fp8:
                _dequant_kv(
                    s_v_fp8,
                    s_v,
                    bar_v_tma,
                    bar_v,
                    BAR_KV_DEQUANT_V,
                    count_raw_sm,
                    group_tidx,
                    1 if k_stage_fp8 else DEQUANT_BATCH,
                )

            bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

            with K.If(count_raw_sm > 0), K.Then():
                num_q_groups_sm = uceil_div_i32(count_raw_sm, q_tokens_per_group)
                # Zero unless causal (:2461-2463); it only ever offsets the
                # diagonal column limit, which a non-causal build never computes.
                if causal:
                    kv_block_col_start = kv_block_sm * N_BLOCK
                else:
                    kv_block_col_start = 0
                # WG0 takes the even Q groups, WG1 the odd ones (:2465-2468).
                num_stage_groups = udiv_i32(num_q_groups_sm + (1 - stage), 2)

                with K.serial(0, num_stage_groups, unroll=False) as qi_iter:
                    qi_group = qi_iter * 2 + stage
                    phase = K.bitwise_and(qi_iter, 1)
                    producer_phase = phase ^ 1
                    qidx_meta_slot = (
                        K.bitwise_and(qi_group, QIDX_META_STAGES - 1) * q_tokens_per_group
                    )

                    # ---------------- softmax step (:2186-2352) ----------------
                    bar_s_full.wait(stage, phase)
                    s_regs = K.alloc_local((128,), "float32")
                    with K.unroll(4) as chunk:
                        K.ptx[_TMEM_LD_32](
                            *[s_regs[chunk * 32 + i] for i in range(32)],
                            K.cuda.get_tmem_addr(
                                tmem_s_col + stage * tmem_stage_stride + chunk * 32, 0, 0
                            ),
                        )
                    K.ptx.tcgen05.wait__ld.sync.aligned()

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
                        qi_group_start = qi_group * q_tokens_per_group
                        masked_tok_count = K.max(
                            0, K.min(q_tokens_per_group, diag_q_count_sm - qi_group_start)
                        )
                        with K.If(masked_tok_count > 0), K.Then():
                            tok_of_row = udiv_i32(group_tidx, qheadperkv)
                            q_idx_mask = K.bitwise_and(
                                ld_shared_i32(s_qidx_meta, qidx_meta_slot + tok_of_row), Q_IDX_MASK
                            )
                            causal_col_limit = q_idx_mask + causal_q_off - kv_block_col_start + 1
                            K.assign(col_limit[0], K.min(kv_valid_cols, causal_col_limit))
                    with K.If(col_limit[0] < N_BLOCK), K.Then():
                        with K.unroll(N_BLOCK // MASK_R2P_CHUNK) as chunk:
                            shift = K.max((chunk + 1) * MASK_R2P_CHUNK - col_limit[0], 0)
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
                            bits = bits_reg[0]
                            signed_bits = K.reinterpret("int32", bits)
                            for i in range(MASK_R2P_CHUNK):
                                # The same bit pattern as the reference's
                                # `mask & (1 << i)`, spelled signed so bit 31 fits
                                # an int32 literal and stays one `and.b32` with an
                                # immediate rather than a shift plus a test.
                                imm = K.local_scalar(
                                    "int32", init=K.int32((1 << i) if i < 31 else -(1 << 31))
                                )
                                with K.If(K.bitwise_and(signed_bits, imm) == K.int32(0)), K.Then():
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
                    scale_log2 = softmax_scale_log2
                    neg_max_scaled = -(row_max[0] * scale_log2)
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
                    # Preserve the parser kernel's trace-time expansion.  The
                    # zero-frequency specialization is also decided while tracing,
                    # so the modulo-by-zero arm remains unspellable.
                    for j in range(4):
                        for k in range(0, 32, 2):
                            # The zero-frequency arm is a SHORT-CIRCUIT, not a
                            # period of zero: `apply_exp2_convert` has its own
                            # `const_expr(ex2_emu_freq == 0)` branch
                            # (softmax.py:381-383) that takes real exp2 for both
                            # elements of every pair, with no polynomial and no
                            # `fmax` clamp. Reaching the modulo with a zero
                            # frequency would divide by zero at trace time.
                            if ex2_emu_freq == 0:
                                K.ptx.ex2.approx.ftz.f32(s_regs[j * 32 + k], s_regs[j * 32 + k])
                                K.ptx.ex2.approx.ftz.f32(
                                    s_regs[j * 32 + k + 1], s_regs[j * 32 + k + 1]
                                )
                            else:
                                use_mufu = K.Or(
                                    (k % ex2_emu_freq) < (ex2_emu_freq - 4),
                                    K.Or(j >= 3, j < EX2_EMU_START_FRG),
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

                    # ---------------- epilogue (:2659-3020) ----------------
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

            bar_sync_named(BAR_TMEM_ALLOC, WARP_SIZE * (WARPS_PER_GROUP * 2 + 1))

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
        # encoding consumes that tensor ABI.  The body still addresses them
        # through the descriptor/base pointer; this rank is not a buffer view.
        ("k", K.gptr[k_ty, 3]),
        ("v", K.gptr[v_ty, 3]),
        ("k2q_q_indices", K.gptr[K.i32]),
        ("k2q_qsplit_indices", K.gptr[K.i32]),
        ("k2q_row_ptr", K.gptr[K.i32]),
        ("scheduler_metadata", K.gptr[K.i32]),
        ("work_count", K.gptr[K.i32]),
        ("o_partial", K.gptr[partial_ty]),
        ("lse_partial", K.gptr[K.f32]),
    ]
    if temperature:
        parameters.append(("lse_temperature_partial", K.gptr[K.f32]))
    # Q is flattened only across token/head; HEAD_DIM remains the descriptor's
    # contiguous axis, so the launch argument is a rank-2 tensor.
    parameters.append(("q_flat", K.gptr[q_ty, 2]))
    if paged:
        parameters.append(("page_table", K.gptr[K.i32]))
    if seqused:
        parameters.append(("seqused_k", K.gptr[K.i32]))
    parameters.extend(
        [
            ("cu_seqlens_q", K.gptr[K.i32]),
            ("cu_seqlens_k", K.gptr[K.i32]),
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
# `qhead_per_kv` is the axis that picks the Q-load program: 1, 2 and 4 use the
# raw gather4 descriptor path, 8 and 16 the plain TMA path (:81-87), so both
# appear in the benchmark matrix rather than only in correctness.
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
    dtype: str = "bf16",
    partial_dtype: str = "float32",
    temperature: float | None = None,
    causal: bool = True,
    blk_kv: int = BLK_KV,
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
        "causal": causal,
        "blk_kv": blk_kv,
        "seqlen_pattern": seqlen_pattern,
    }


BENCH_CONFIGS = [
    # MSA's own ring-attention benchmark shape, verbatim.
    _case(
        label="ring48k_bf16_qh16_t16",
        batch=1,
        seqlen_q=49152,
        seqlen_k=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
    ),
    # MSA's ulysses shape at its own sweep's lowest topk: the full 384K sequence
    # with topk=16 needs 6.4 GB for O_partial alone, and the geometry -- not the
    # slot count -- is what makes this shape interesting.
    _case(
        label="ulysses384k_bf16_qh2_t4",
        batch=1,
        seqlen_q=393216,
        seqlen_k=393216,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
    ),
    # The long-sequence x high-topk regime, at a length whose partials fit.
    _case(
        label="long96k_bf16_qh2_t16",
        batch=1,
        seqlen_q=98304,
        seqlen_k=98304,
        head_kv=1,
        qhead_per_kv=2,
        topk=16,
    ),
    _case(
        label="varlen_b3_s8192_qh4_t16",
        batch=3,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        seqlen_pattern="varlen",
    ),
    _case(
        label="varlen_b3_s4096_qh8_t8",
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
        label="qh1_s8192_bf16_t16",
        batch=1,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=1,
        topk=16,
    ),
    _case(
        label="ring48k_fp8_qh16_t16",
        batch=1,
        seqlen_q=49152,
        seqlen_k=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
        dtype="fp8",
        partial_dtype="bfloat16",
        temperature=1.0,
    ),
    _case(
        label="fp8kv_s16384_qh4_t16",
        batch=1,
        seqlen_q=16384,
        seqlen_k=16384,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        dtype="bf16q_fp8kv",
    ),
    _case(
        label="fp8_pvbf16_s8192_qh8_t8",
        batch=1,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        dtype="fp8_pvbf16",
    ),
    _case(
        label="edge_b1_s1024_bf16_t4",
        batch=1,
        seqlen_q=1024,
        seqlen_k=1024,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
    ),
    # Paged. `paged_ring48k_bf16_qh16_t16` deliberately mirrors the flat
    # marquee row shape for shape, so the pair reads directly as the cost of
    # the page indirection rather than of a different workload.
    _case(
        label="paged_ring48k_bf16_qh16_t16",
        batch=1,
        seqlen_q=49152,
        seqlen_k=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
        paged=True,
    ),
    _case(
        label="paged_seqused_bf16_s16384_qh4_t16",
        batch=2,
        seqlen_q=16384,
        seqlen_k=16384,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        paged=True,
        seqused=True,
    ),
    _case(
        label="paged_fp8_s16384_qh16_t16",
        batch=1,
        seqlen_q=16384,
        seqlen_k=16384,
        head_kv=2,
        qhead_per_kv=16,
        topk=16,
        dtype="fp8",
        partial_dtype="bfloat16",
        temperature=1.0,
        paged=True,
    ),
    _case(
        label="paged_edge_b1_s2048_bf16_t4",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
        paged=True,
    ),
]

# Correctness runs at test scale: a full reference for the marquee shapes would
# need tens of GB live, and the axes those shapes carry -- both Q paths, all
# four dtype modes, all three partial dtypes -- are covered here at sizes that
# fit alongside their reference.
CONFIGS = [
    *[case for case in BENCH_CONFIGS if case["seqlen_k"] <= 16384 and case["dtype"] != "fp8"],
    _case(
        label="corr_fp8_s4096_qh16_t8",
        batch=1,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=1,
        qhead_per_kv=16,
        topk=8,
        dtype="fp8",
        partial_dtype="bfloat16",
        temperature=1.0,
    ),
    _case(
        label="corr_fp8_partial_s2048_qh8_t8",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        dtype="fp8",
        partial_dtype="float8_e4m3",
        temperature=2.0,
    ),
    # FP8 Q through the gather4 descriptor. The gather4 box width is derived
    # from the Q element size (:1699), so an FP8 Q is a different descriptor and
    # a different per-instruction byte count than the BF16 Q the other gather4
    # cases load; the FP8 cases above all sit on the TMA-Q side of :81-87.
    _case(
        label="corr_fp8_gather4_qh2_s2048",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=2,
        topk=8,
        dtype="fp8",
        partial_dtype="bfloat16",
        temperature=1.0,
    ),
    # The staged FP8-KV dequant (:1255/:1488) runs in the softmax warpgroups and
    # the Q program runs in its own, so the two compose independently; the other
    # staged case is gather4, this one pairs the same dequant with TMA Q.
    _case(
        label="corr_fp8kv_tma_qh16_s2048",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=1,
        qhead_per_kv=16,
        topk=8,
        dtype="bf16q_fp8kv",
    ),
    # PV in BF16 while QK stays FP8 selects a different MMA kind for the second
    # GEMM (:1938 issue order); the other pv=bf16 case is TMA Q.
    _case(
        label="corr_fp8_pvbf16_gather4_qh4_s2048",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=4,
        topk=8,
        dtype="fp8_pvbf16",
    ),
    _case(
        label="corr_bf16_qh2_varlen_b2",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=2,
        topk=8,
        seqlen_pattern="varlen",
    ),
    _case(
        label="corr_bf16_qh16_s2048_t4",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=16,
        topk=4,
    ),
    _case(
        label="corr_decode_b8_qh16_t8",
        batch=8,
        seqlen_q=8,
        seqlen_k=4096,
        head_kv=2,
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
    # Paged KV. `page_size == blk_kv`, so a CTA's block is one page and the only
    # kernel-visible change is the TMA coordinate; these cover both Q paths,
    # both KV staging forms, and the two `seqused_k` states.
    _case(
        label="paged_bf16_s4096_qh4_t16",
        batch=2,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        paged=True,
    ),
    _case(
        label="paged_bf16_s4096_qh16_t8",
        batch=2,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=16,
        topk=8,
        paged=True,
    ),
    _case(
        label="paged_seqused_bf16_s4096_qh4_t16",
        batch=3,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        paged=True,
        seqused=True,
    ),
    # Varlen under `seqused_k`: the per-batch trims land at different offsets
    # inside the last page, so the column limit and the causal diagonal move
    # independently per batch.
    _case(
        label="paged_seqused_varlen_b3_s2048_qh8_t8",
        batch=3,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        paged=True,
        seqused=True,
        seqlen_pattern="varlen",
    ),
    _case(
        label="paged_fp8_s2048_qh16_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=16,
        topk=8,
        dtype="fp8",
        partial_dtype="bfloat16",
        temperature=1.0,
        paged=True,
    ),
    # FP8 K/V staged to BF16: the staging descriptor is a second rank-4 map.
    _case(
        label="paged_fp8kv_s2048_qh4_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=4,
        topk=8,
        dtype="bf16q_fp8kv",
        paged=True,
    ),
    _case(
        label="paged_seqused_fp8_pvbf16_s2048_qh8_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        dtype="fp8_pvbf16",
        paged=True,
        seqused=True,
    ),
    # `causal=False`: a different register budget, `ex2_emu_freq = 0`, no
    # diagonal binary search, and a column-limit-only mask (:173-184).
    _case(
        label="corr_noncausal_bf16_s2048_qh4_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=4,
        topk=8,
        causal=False,
    ),
    _case(
        label="corr_noncausal_paged_fp8_s2048_qh16_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=16,
        topk=8,
        dtype="fp8",
        partial_dtype="bfloat16",
        causal=False,
        paged=True,
    ),
    # FP16 partials share the BF16 v8 epilogue store width (:372-377).
    _case(
        label="corr_partial_fp16_s2048_qh8_t8",
        batch=2,
        seqlen_q=2048,
        seqlen_k=2048,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        partial_dtype="float16",
    ),
    # topk=32 is the top of upstream's supported set; it changes the split-slot
    # count and the work list, not the compiled code.
    _case(
        label="corr_topk32_s4096_qh4",
        batch=1,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=2,
        qhead_per_kv=4,
        topk=32,
    ),
]


# ---------------------------------------------------------------------------
# Data preparation.
#
# The CSR payload, the work list and the schedule sizing come from the
# split-atomic module, which builds exactly the inputs this kernel consumes.
# What that module cannot supply is the packed split slots: it produces them
# with a device atomic, in arrival order, so re-running it would hand this
# kernel a different slot permutation on every call. The assignment below is a
# valid instance of the same contract -- per `(q_abs, head_kv)` group the slots
# are exactly `{0 .. degree-1}` -- fixed to CSR order so the forward becomes
# reproducible run to run, which is what makes an element-wise oracle possible.
# ---------------------------------------------------------------------------
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


def _seqused_trim(batch: int, blk_kv: int) -> list[int]:
    """Per-batch shortfall between the logical K length and the paged capacity.

    Every trim stays **inside one page**, which is what keeps the CSR valid
    without a pruning pass: `ceil(effective / blk_kv)` still equals the capacity
    row count, so no edge can reference a block past `seqused_k`. Upstream
    reaches the same state from the other side, selecting against the full
    length and rewriting out-of-range entries to `-1`
    (test_sparse_atten.py:956-971); a trim of a whole page or more would need
    that rewrite to stay a legal input, and an input upstream would never
    produce is not worth constructing.

    Deterministic and deliberately mixed, so the partial-block column limit and
    the shifted causal diagonal both get exercised. The first entry is nonzero
    and the second is the full page rather than the other way round: a
    single-batch config would otherwise trim nothing, making `seqused_k` equal
    to the paged capacity and testing the axis only in name.
    """
    offsets = (37, 0, blk_kv - 1, 149, blk_kv // 2)
    return [offsets[i % len(offsets)] % blk_kv for i in range(batch)]


def _pack_paged_kv(flat, seqlens_k, page_size: int, page_table):
    """Scatter `[total_k, head_kv, D]` into `[num_pages, head_kv, page_size, D]`.

    Physical pages come from `page_table`, which is shuffled, so a kernel that
    ignored the table and walked pages in order would read another sequence's
    tokens rather than merely stale ones. Trailing partial pages stay zero, as
    upstream's builder leaves them (test_sparse_atten.py:401).
    """
    import torch

    head_kv, dim = flat.shape[1], flat.shape[2]
    num_pages = int(page_table.numel())
    paged = torch.zeros((num_pages, head_kv, page_size, dim), dtype=flat.dtype, device=flat.device)
    table = page_table.tolist()
    offset = 0
    for batch_idx, length in enumerate(seqlens_k):
        length = int(length)
        for page in range(len(table[batch_idx])):
            lo = page * page_size
            hi = min(lo + page_size, length)
            if hi > lo:
                paged[int(table[batch_idx][page]), :, : hi - lo] = flat[
                    offset + lo : offset + hi
                ].transpose(0, 1)
        offset += length
    return paged


def _build_page_table(batch: int, pages_per_seq: int, generator):
    """A shuffled, rectangular `[batch, pages_per_seq]` int32 table.

    16-byte aligned because the production adapter asserts it and over-allocates
    to guarantee it (sparse_fmha_adapter.py:239-242).
    """
    import torch

    num_pages = batch * pages_per_seq
    padded = torch.empty(num_pages + 4, dtype=torch.int32, device="cuda")
    shift = (-padded.data_ptr()) % 16 // padded.element_size()
    table = padded[shift : shift + num_pages]
    table.copy_(torch.randperm(num_pages, device="cuda", generator=generator).to(torch.int32))
    table = table.view(batch, pages_per_seq)
    assert table.data_ptr() % 16 == 0
    return table


def _frozen_qsplit(csr: dict[str, Any]):
    """Assign each CSR edge the slot the split-atomic kernel would reserve for it."""
    import torch

    from tirx_kernels.msa.sparse_prepare_flat_schedule import row_coords

    head_kv = csr["head_kv"]
    total_rows = csr["total_rows"]
    row_ptr = csr["k2q_row_ptr"].view(head_kv, total_rows + 1)
    q_indices = csr["k2q_q_indices"].view(head_kv, -1)
    device = q_indices.device

    batch_of_row, _ = row_coords(csr["seqlens_k"], csr["config"]["blk_kv"])
    batch_of_row = torch.tensor(batch_of_row, dtype=torch.int64, device=device)
    q_offset = torch.zeros(len(csr["seqlens_q"]) + 1, dtype=torch.int64, device=device)
    q_offset[1:] = torch.tensor(csr["seqlens_q"], dtype=torch.int64, device=device).cumsum(0)

    qsplit = torch.full_like(q_indices, -1)
    q_abs_all = torch.zeros_like(q_indices, dtype=torch.int64)
    rows = torch.arange(total_rows, device=device)
    for h in range(head_kv):
        counts = (row_ptr[h, 1:] - row_ptr[h, :-1]).to(torch.int64)
        live = int(row_ptr[h, total_rows].item())
        row_of_edge = torch.repeat_interleave(rows, counts)
        q_idx = q_indices[h, :live].to(torch.int64)
        q_abs = q_idx + q_offset[batch_of_row[row_of_edge]]
        q_abs_all[h, :live] = q_abs

        # Slots in CSR order within each `q_abs` group: sort the edges by group
        # (stably, so CSR order survives inside a group), number each group
        # 0..degree-1, and scatter the numbering back to edge positions.
        order = q_abs.argsort(stable=True)
        degrees = torch.bincount(q_abs, minlength=csr["total_q"])
        starts = torch.cumsum(degrees, 0) - degrees
        ranks = torch.arange(live, device=device) - starts[q_abs[order]]
        slots = torch.empty(live, dtype=torch.int64, device=device)
        slots[order] = ranks
        qsplit[h, :live] = (q_indices[h, :live] | (slots.to(torch.int32) << SLOT_SHIFT)).to(
            torch.int32
        )
    return qsplit.contiguous(), q_abs_all


def prepare_data(*, seed: int = 0, **config) -> dict[str, Any]:
    """Build Q/K/V, the CSR payload, the work list and the frozen split slots."""
    import torch

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
    topk = config["topk"]
    total_q = csr["total_q"]
    total_k = int(sum(csr["seqlens_k"]))
    mode = DTYPE_MODES[config.get("dtype", "bf16")]
    partial_dtype = config.get("partial_dtype", "float32")

    def draw(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)

    q_bf16 = draw((total_q, head_q, HEAD_DIM))
    k_bf16 = draw((total_k, head_kv, HEAD_DIM))
    v_bf16 = draw((total_k, head_kv, HEAD_DIM))
    q = q_bf16.to(_torch_dtype(mode["q"])).contiguous()
    k = k_bf16.to(_torch_dtype(mode["k"])).contiguous()
    v = v_bf16.to(_torch_dtype(mode["v"])).contiguous()

    # Paged KV. `page_size == blk_kv` is mandatory upstream (:99-107), so a CTA's
    # KV block is exactly one page and the pages-per-sequence count is uniform;
    # that rectangularity is what lets the kernel recover `pages_per_seq` from
    # the scalars it already takes instead of a new argument.
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
            # Without `seqused_k` the kernel takes the full paged capacity as the
            # logical length (:210-211), so every batch must fill its pages
            # exactly or the tail of zero-padding would be attended to. The CSR's
            # lengths are already blk_kv-aligned, which is that condition.
            assert all(int(length) % blk_kv == 0 for length in csr["seqlens_k"])
            assert all(
                (int(length) + blk_kv - 1) // blk_kv == pages_per_seq for length in csr["seqlens_k"]
            ), "paged without seqused_k requires uniform lengths: capacity is the logical length"

    cu_seqlens_k = torch.zeros(len(csr["seqlens_k"]) + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.tensor(csr["seqlens_k"], dtype=torch.int32, device=device).cumsum(0)

    qsplit, q_abs_of_edge = _frozen_qsplit(csr)

    return {
        "config": dict(config),
        "csr": csr,
        "q": q,
        "q_flat": q.reshape(-1, HEAD_DIM),
        # Paged runs hand the kernel and the reference the page-major tensors;
        # the flat ones stay for the dequantized twins and for host-side checks.
        "k": k_paged if paged else k,
        "v": v_paged if paged else v,
        "k_flat": k,
        "v_flat": v,
        "paged": paged,
        "page_table": page_table,
        "seqused_k": seqused_k,
        # The bf16 twins the fp8-KV path's exact-match oracle re-runs against.
        "k_dequantized": k.to(torch.bfloat16).contiguous(),
        "v_dequantized": v.to(torch.bfloat16).contiguous(),
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
        "softmax_scale": 1.0 / math.sqrt(HEAD_DIM),
        "lse_temperature_scale": config.get("temperature"),
        "head_dim": HEAD_DIM,
        "blk_kv": config["blk_kv"],
        "head_kv": head_kv,
        "head_q": head_q,
        "qhead_per_kv": qhead_per_kv,
        "topk": topk,
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
        "partial_dtype": partial_dtype,
        "qk_dtype": mode["qk"],
        "pv_dtype": mode["pv"],
    }


def make_outputs(data: dict[str, Any]) -> dict[str, Any]:
    """Fresh partial buffers, uninitialized exactly as the host allocates them."""
    import torch

    shape = (data["topk"], data["total_q"], data["head_q"])
    outputs = {
        "o_partial": torch.empty(
            (*shape, HEAD_DIM), dtype=_torch_dtype(data["partial_dtype"]), device="cuda"
        ),
        "lse_partial": torch.empty(shape, dtype=torch.float32, device="cuda"),
    }
    if data["lse_temperature_scale"] is not None:
        outputs["lse_temperature_partial"] = torch.empty(shape, dtype=torch.float32, device="cuda")
    return outputs


def tirx_args(data: dict[str, Any], outputs: dict[str, Any]) -> tuple:
    """The launch ABI, bound once outside any timed region."""
    import torch

    def as_bits(t):
        """fp8 reaches the kernel as raw ``uint8``; see ``_tirx_dtype``."""
        return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t

    # The kernel declares the index and output buffers flat; only K, V and Q
    # keep their real shape, and only because their base pointer reaches a
    # tensor map.
    args = [
        # The kernel binds K/V with one 3-D shape on both axes and reads only
        # their base pointer; the tensormap carries the paged geometry. A paged
        # tensor is contiguous, so this view is free and the bytes are the same.
        as_bits(data["k"].reshape(-1, data["head_kv"], HEAD_DIM)),
        as_bits(data["v"].reshape(-1, data["head_kv"], HEAD_DIM)),
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
        data["softmax_scale"] * math.log2(math.e),
        data["softmax_scale"] * math.log2(math.e) / (data["lse_temperature_scale"] or 1.0),
        1.0 / (data["lse_temperature_scale"] or 1.0),
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
    ]
    return tuple(args)


def reference_case(data: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """The argument bundle MSA's own compiled forward takes."""
    return {
        "k": data["k"],
        "v": data["v"],
        "k2q_q_indices": data["k2q_q_indices"],
        "k2q_qsplit_indices": data["k2q_qsplit_indices"],
        "k2q_row_ptr": data["k2q_row_ptr"],
        "scheduler_metadata": data["scheduler_metadata"],
        "work_count": data["work_count"],
        "o_partial_flat": outputs["o_partial"].reshape(-1, HEAD_DIM),
        "lse_partial": outputs["lse_partial"],
        "lse_temperature_partial": outputs.get("lse_temperature_partial"),
        "q_flat": data["q_flat"],
        "page_table": data.get("page_table"),
        "seqused_k": data.get("seqused_k"),
        "cu_seqlens_q": data["cu_seqlens_q"],
        "cu_seqlens_k": data["cu_seqlens_k"],
        "softmax_scale": data["softmax_scale"],
        "lse_temperature_inv_scale": 1.0 / (data["lse_temperature_scale"] or 1.0),
        "num_kv_blocks": data["total_rows"],
        "head_kv": data["head_kv"],
        "max_seqlen_q": data["max_seqlen_q"],
        "work_capacity": data["work_capacity"],
        "head_dim": HEAD_DIM,
        "blk_kv": data["blk_kv"],
        "qhead_per_kv": data["qhead_per_kv"],
        "causal": data["causal"],
        "qk_dtype": _torch_dtype(data["qk_dtype"]),
        "pv_dtype": _torch_dtype(data["pv_dtype"]),
    }


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------
def live_partial_mask(data: dict[str, Any]):
    """`split < split_counts[q_abs, head_kv]`, broadcast over the partial shape.

    Both partial buffers arrive uninitialized, and a query with degree `d`
    leaves slots `d .. topk-1` untouched, so every comparison has to be masked
    to the slots the schedule actually assigns.
    """
    import torch

    degrees = data["degrees"]  # [total_q, head_kv]
    per_head_q = degrees.repeat_interleave(data["qhead_per_kv"], dim=1)
    splits = torch.arange(data["topk"], device=degrees.device).view(-1, 1, 1)
    return splits < per_head_q.unsqueeze(0)


def torch_reference_partials(data: dict[str, Any]) -> dict[str, Any]:
    """Compute every CSR edge independently with PyTorch.

    The implementation follows MSA's public ``sparse_attention_ref`` math but
    works row-by-row over K-to-Q CSR edges.  This avoids constructing the dense
    Q-by-K mask while still checking the frozen split-slot and fake-column ABI.
    """
    import torch

    from tirx_kernels.msa.sparse_prepare_flat_schedule import row_coords

    expected = make_outputs(data)
    batch_of_row, level_of_row = row_coords(data["seqlens_k"], data["blk_kv"])
    cu_k = data["cu_seqlens_k"].to(torch.int64)
    qh = data["qhead_per_kv"]
    qk_fp8 = data["qk_dtype"] == "float8_e4m3"
    pv_fp8 = data["pv_dtype"] == "float8_e4m3"
    k_values = data.get("k_reference_flat", data["k_flat"] if qk_fp8 else data["k_dequantized"])
    v_values = data.get("v_reference_flat", data["v_flat"] if pv_fp8 else data["v_dequantized"])
    fake_columns = torch.tensor(
        [_fake_col(data["partial_dtype"], column) for column in range(HEAD_DIM)],
        dtype=torch.long,
        device="cuda",
    )

    for head in range(data["head_kv"]):
        head_start = head * qh
        head_end = head_start + qh
        row_ptr = data["k2q_row_ptr"][head]
        for row, (batch, level) in enumerate(zip(batch_of_row, level_of_row, strict=True)):
            edge_start = int(row_ptr[row])
            edge_end = int(row_ptr[row + 1])
            if edge_end == edge_start:
                continue
            q_local = data["k2q_q_indices"][head, edge_start:edge_end].to(torch.int64)
            q_abs = data["q_abs_of_edge"][head, edge_start:edge_end].to(torch.int64)
            packed = data["k2q_qsplit_indices"][head, edge_start:edge_end]
            slots = torch.bitwise_right_shift(packed, SLOT_SHIFT).bitwise_and(SLOT_MASK).long()

            token_start = int(cu_k[batch]) + level * data["blk_kv"]
            token_end = min(token_start + data["blk_kv"], int(cu_k[batch + 1]))
            token_count = token_end - token_start
            k_tile = k_values[token_start:token_end, head].float()
            v_tile = v_values[token_start:token_end, head].float()
            query = data["q"][q_abs, head_start:head_end].float()
            scores = torch.einsum("ehd,td->eht", query, k_tile) * data.get(
                "reference_softmax_scale", data["softmax_scale"]
            )

            token_pos = level * data["blk_kv"] + torch.arange(
                token_count, device="cuda", dtype=torch.int64
            )
            used_k = (
                int(data["seqused_k"][batch])
                if data["seqused_k"] is not None
                else int(data["seqlens_k"][batch])
            )
            valid = token_pos.unsqueeze(0) < used_k
            if data["causal"]:
                causal_limit = q_local + used_k - int(data["seqlens_q"][batch])
                valid = valid & (token_pos.unsqueeze(0) <= causal_limit.unsqueeze(1))
            valid = valid.unsqueeze(1)
            scores = scores.masked_fill(~valid, float("-inf"))
            has_value = valid.any(dim=-1)
            safe_scores = torch.where(has_value.unsqueeze(-1), scores, torch.zeros_like(scores))
            row_max = safe_scores.max(dim=-1, keepdim=True).values
            exponent = torch.exp(safe_scores - row_max)
            exponent = torch.where(
                valid & has_value.unsqueeze(-1), exponent, torch.zeros_like(exponent)
            )
            row_sum = exponent.sum(dim=-1, keepdim=True)
            lse = torch.where(
                has_value,
                row_max.squeeze(-1) + torch.log(row_sum.squeeze(-1)),
                torch.full_like(row_max.squeeze(-1), float("-inf")),
            )

            if pv_fp8:
                probability = exponent.to(torch.float8_e4m3fn).float()
                logical_out = torch.einsum("eht,td->ehd", probability, v_tile)
                logical_out = logical_out / torch.where(
                    row_sum > 0, row_sum, torch.ones_like(row_sum)
                )
            else:
                probability = exponent / torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
                probability = probability.to(torch.bfloat16).float()
                logical_out = torch.einsum("eht,td->ehd", probability, v_tile)
            logical_out = torch.where(
                has_value.unsqueeze(-1), logical_out, torch.zeros_like(logical_out)
            )

            physical_out = torch.empty_like(logical_out)
            physical_out.index_copy_(-1, fake_columns, logical_out)
            expected["o_partial"][slots, q_abs, head_start:head_end] = physical_out.to(
                expected["o_partial"].dtype
            )
            expected["lse_partial"][slots, q_abs, head_start:head_end] = lse
            if data["lse_temperature_scale"] is not None:
                temperature_lse = torch.logsumexp(
                    scores / float(data["lse_temperature_scale"]), dim=-1
                )
                expected["lse_temperature_partial"][slots, q_abs, head_start:head_end] = (
                    temperature_lse
                )
    return expected


def assert_partials_match(
    data: dict[str, Any],
    outputs: dict[str, Any],
    expected: dict[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    lse_rtol: float | None = None,
    lse_atol: float | None = None,
) -> None:
    """Compare the live slots of both partial buffers."""
    import torch

    mask = live_partial_mask(data)
    o_mask = mask.unsqueeze(-1).expand_as(outputs["o_partial"])
    torch.testing.assert_close(
        outputs["o_partial"][o_mask].float(),
        expected["o_partial"][o_mask].float(),
        rtol=rtol,
        atol=atol,
    )
    lse_rtol = rtol if lse_rtol is None else lse_rtol
    lse_atol = atol if lse_atol is None else lse_atol
    torch.testing.assert_close(
        outputs["lse_partial"][mask], expected["lse_partial"][mask], rtol=lse_rtol, atol=lse_atol
    )
    if "lse_temperature_partial" in outputs:
        torch.testing.assert_close(
            outputs["lse_temperature_partial"][mask],
            expected["lse_temperature_partial"][mask],
            rtol=lse_rtol,
            atol=lse_atol,
        )


def run_test(**config):
    """Compile, launch, and validate one config against MSA's own kernel."""
    import unittest

    import torch

    from tirx_kernels.runner import compile_kernel

    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    config.pop("label", None)
    data = prepare_data(**config)

    try:
        from tirx_kernels.msa.utils._msa_bench import compiled_sparse_atten_fwd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    expected = make_outputs(data)
    try:
        compiled_sparse_atten_fwd(reference_case(data, expected))()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    torch.cuda.synchronize()

    executable = compile_kernel(get_kernel(**config))
    outputs = make_outputs(data)
    executable(*tirx_args(data, outputs))
    torch.cuda.synchronize()
    assert_partials_match(data, outputs, expected)


def prepare_bench(**config):
    """Compile the TIRx specialization without initializing CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config.pop("label", None)
    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


# ---------------------------------------------------------------------------
# Benchmark entry points.
#
# Unlike the two preparation kernels, this one needs no rotation: it reads its
# inputs without touching them and overwrites -- never accumulates into -- the
# partial slots it owns, so the hundredth launch does exactly the work the
# first one did.
# ---------------------------------------------------------------------------
def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    """Kernel-only comparison against MSA's compiled forward launch."""
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
        from tirx_kernels.msa.utils._msa_bench import compiled_sparse_atten_fwd

        launch = compiled_sparse_atten_fwd(reference_case(data, make_outputs(data)))
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
