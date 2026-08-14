# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""The kernel body: eight warps, five barrier families, one persistent walk.

Plain TIRx only -- explicit loops, hand-carved shared/tensor memory and `T.ptx`
intrinsics.  No tile primitive may appear here: no `tirx.tile.*` call and no op
carrying ``TIRxOpCategory == "tile_primitive"``, in any specialization.

Role map (source `:206`, `:281`, `:432`, `:470`):

===== ===============================================================
warp  role
===== ===============================================================
0     TMA descriptor prefetch, then the TMA load loop, then TMEM free
1     barrier init, then UMMA issue + UTCCP (leader CTA only)
2     TMEM allocation, then the scale-factor SMEM transpose
3     idle -- no branch selects it, deliberately
4..   epilogue, bounded by ``kNumUMMAStoreThreads / 32``
===== ===============================================================

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh,
scheduler/gemm.cuh.
"""

from __future__ import annotations

import tvm
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)
from tvm.ir.type import PointerType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IterVar, Layout, is_buffer_var
from tvm.tirx.script.builder.ir import name_meta_class_value

from .spec import (
    BLOCK_K,
    LAYOUT_AD_M,
    NUM_EPILOGUE_STAGES,
    NUM_MAX_STAGES,
    NUM_TMA_STORE_STAGES,
    NUM_UTCCP_ALIGNED_ELEMS,
    UMMA_K,
    UMMA_STEP_N,
    GemmSpec,
    GemmType,
    Major,
)

__all__ = ["build_kernel"]

#: Named-barrier id used by the epilogue.  CUTLASS `NamedBarrier::sync(n, 0)`
#: offsets user ids by `ReservedNamedBarrierCount = 8`; id 0 belongs to
#: `__syncthreads()`, so reusing it with a partial thread count would deadlock.
EPILOGUE_NAMED_BARRIER = 8

#: `cute::TMA::CacheHintSm100::EVICT_NORMAL`.
EVICT_NORMAL = 1152921504606846976

#: `timeHint` for `mbarrier.try_wait.parity`, a nanosecond budget after which
#: the instruction reports "not yet" rather than keep waiting.  Same value the
#: `T.cuda.mbarrier_wait` helper bakes into the spin loop it emits.
TRY_WAIT_TICKS = 0x989680

_TORCH_SMEM_DTYPE = {
    "fp8": "float8_e4m3fn",
    "fp4": "float8_e4m3fn",
    "bf16": "bfloat16",
    "fp32": "float32",
}
_UMMA_DTYPE = {"fp8": "float8_e4m3fn", "fp4": "float4_e2m1fn"}

_BUILDER_MISSING = object()


def _builder_runtime_condition(value):
    return value


def _builder_enter(frame):
    frames = frame.frames if hasattr(frame, "frames") else [frame]
    prim_func = next(
        candidate
        for candidate in reversed(IRBuilder.current().frames)
        if type(candidate).__name__ == "PrimFuncFrame"
    )
    for item in frames:
        prim_func.add_callback(lambda item=item: item.__exit__(None, None, None))
        item.__enter__()


def _builder_emit(value):
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if isinstance(value, IRBuilderFrame) or (
        hasattr(value, "frames") and hasattr(value, "__enter__")
    ):
        _builder_enter(value)
    elif tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)
    elif isinstance(value, int | bool):
        T.evaluate(tvm.tirx.const(value))


def _builder_alloc_scalar(name, dtype):
    scalar = T.local_scalar(dtype)
    IRBuilder.name(name, scalar.scalar.buffer)
    return scalar.scalar


def _builder_scalar(name, value, dtype):
    scalar = _builder_alloc_scalar(name, dtype)
    T.buffer_store(scalar.buffer, value, scalar.indices)
    return scalar


def _builder_buffer(name, shape, dtype):
    buffer = T.alloc_local(shape, dtype)
    IRBuilder.name(name, buffer)
    return buffer


def _builder_bind(name, value, type_annotation=None):
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_assign(name, value, previous=_BUILDER_MISSING):
    if isinstance(value, I.meta_var):
        return value.value
    if previous is not _BUILDER_MISSING:
        if isinstance(previous, T.scalar_wrapper | tvm.tirx.expr.BufferLoad):
            target = previous.scalar if isinstance(previous, T.scalar_wrapper) else previous
            T.buffer_store(target.buffer, value, target.indices)
            return target
        if (
            is_buffer_var(previous)
            and len(previous.ty.shape) == 1
            and bool(previous.ty.shape[0] == 1)
        ):
            try:
                T.buffer_store(previous, value, [0])
                return previous
            except TypeError:
                pass
    if getattr(type(value), "_is_meta_class", False):
        name_meta_class_value(name, value)
        return value
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _builder_assign(f"{name}_{index}", item)
        return value
    if is_buffer_var(value) or isinstance(value, IterVar | Layout):
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Var):
        if isinstance(value.ty, PointerType):
            return _builder_bind(name, value, value.ty)
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Expr) and isinstance(getattr(value, "ty", None), PointerType):
        return _builder_bind(name, value, value.ty)
    if isinstance(value, tvm.ir.Expr) and tvm.ir.is_prim_expr(value):
        return _builder_scalar(name, value, str(value.ty.dtype))
    if isinstance(value, tvm.tirx.expr.ExprOp):
        return _builder_scalar(name, value, "bool")
    return value


def _builder_assign_many(names, values, previous):
    return tuple(
        _builder_assign(name, value, old) for name, value, old in zip(names, values, previous)
    )


def _validate(spec: GemmSpec) -> None:
    if spec.block_k != BLOCK_K:
        raise ValueError(f"invalid block K: {spec.block_k}")
    if spec.block_k % UMMA_K != 0:
        raise ValueError("block K must be divisible by UMMA K")
    if spec.num_multicast not in (1, 2):
        raise ValueError("only 1/2 multicast is supported")
    if spec.swap_ab:
        if spec.block_n != LAYOUT_AD_M:
            raise ValueError("swap-AB requires block N == 128")
    elif spec.block_m not in (32, 64, LAYOUT_AD_M):
        raise ValueError(f"invalid block M: {spec.block_m}")
    if spec.gran_k_a not in (32, 128) or spec.gran_k_b not in (32, 128):
        raise ValueError("invalid K granularity")
    if spec.gemm_type.is_k_grouped_contiguous:
        if spec.gran_k_a != spec.gran_k_b:
            raise ValueError("k-grouped SF requires gran_k_a == gran_k_b")
        if spec.k_alignment % UMMA_K != 0:
            raise ValueError("K alignment must be divisible by UMMA K")
    if spec.num_stages > NUM_MAX_STAGES:
        raise ValueError("too many stages")
    if spec.num_umma_store_threads % 32 != 0:
        raise ValueError("invalid store block M")
    if not 32 <= spec.num_tmem_cols <= 512:
        raise ValueError("invalid tensor memory columns")
    # `UMMA_A_SIZE_PER_STAGE` may read past one A slab into the A/B tail; the
    # source asserts that stays in bounds rather than shortening the read.
    from .spec import align_up

    umma_a = align_up(spec.load_block_m, LAYOUT_AD_M) * spec.block_k
    if umma_a > spec.smem_a_size_per_stage + spec.smem_b_size_per_stage * spec.num_stages:
        raise ValueError("UMMA A padding would read out of bounds")


#: `#pragma unroll 4` on the UMMA warp's K-block loop.
MMA_K_UNROLL = 4


def _uceil(x, d):
    """`math::ceil_div` over a value that is known non-negative.

    The scheduler counters come from `grouped_layout`, an `int32` global whose
    sign TVM cannot prove, so a plain `//` lowers to the signed floordiv sequence
    (`IABS` + `ISETP.GE` + sign fixup) where the source -- `uint32_t` throughout --
    gets a single shift.
    """
    return T.cast((T.cast(x, "uint32") + T.uint32(d - 1)) // T.uint32(d), "int32")


def _udiv(x, d):
    """Exact division of a known non-negative value; see `_uceil`."""
    return T.cast(T.cast(x, "uint32") // T.cast(d, "uint32"), "int32")


def _umod(x, d):
    """Remainder of a known non-negative value; see `_uceil`."""
    return T.cast(T.cast(x, "uint32") % T.cast(d, "uint32"), "int32")


def build_kernel(spec: GemmSpec):
    """Build the TIRx `PrimFunc` for one `sm100_fp8_fp4_gemm_1d1d_impl` instantiation."""
    from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
    from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.ir.type import PointerType, PrimType
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    _validate(spec)

    def _u64_const(value):
        """A `uint64` literal whose top bit may be set.

        `T.uint64` routes a Python int through an int64 conversion, so a value
        at or above 2**63 does not survive it -- and a descriptor whose swizzle
        puts `layout_type_` at bits [61,64) is exactly that. Assembling it from
        two 32-bit halves does; the compiler folds it straight back.
        """
        return T.bitwise_or(
            T.shift_left(T.uint64((value >> 32) & 0xFFFFFFFF), T.uint64(32)),
            T.uint64(value & 0xFFFFFFFF),
        )

    def _smem_addr_field(addr_u32):
        """Bits [13:0] of a matrix descriptor: the shared address over 16."""
        return T.cast(
            T.bitwise_and(T.shift_right(addr_u32, T.uint32(4)), T.uint32(0x3FFF)), "uint64"
        )

    def _with_smem_addr(base, addr_u32):
        """Put an address into a descriptor base whose address field is zero."""
        return T.bitwise_or(base, _smem_addr_field(addr_u32))

    def _rebase(desc, addr_u32):
        """`replace_smem_desc_addr`: rewrite bits [13:0] with `smem_addr >> 4`."""
        return T.bitwise_or(
            T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), _smem_addr_field(addr_u32)
        )

    def _with_sf_id(desc, sfa_id, sfb_id):
        """`make_runtime_instr_desc_with_sf_id`: fields [31:29] and [6:4]."""
        out = T.bitwise_and(desc, T.uint32(0x9FFFFFCF))
        out = T.bitwise_or(out, T.shift_left(T.cast(sfa_id, "uint32"), T.uint32(29)))
        return T.bitwise_or(out, T.shift_left(T.cast(sfb_id, "uint32"), T.uint32(4)))

    def _advance_lo(desc, base_lo, units):
        """`advance_umma_desc_lo`: replace the low word with `base_lo + units`."""
        return T.bitwise_or(
            T.bitwise_and(desc, T.shift_left(T.uint64(0xFFFFFFFF), T.uint64(32))),
            T.cast(base_lo + T.uint32(units), "uint64"),
        )

    # ---- compile-time constants (all Python ints; nothing is emitted) --------
    cta_group = spec.num_multicast
    stages = spec.num_stages
    umma_n = spec.umma_n
    umma_m = spec.umma_m
    block_m, block_n, block_k = spec.block_m, spec.block_n, spec.block_k
    load_block_m, load_block_n = spec.load_block_m, spec.load_block_n
    sf_block_m, sf_block_n = spec.sf_block_m, spec.sf_block_n
    store_block_m, store_block_n = spec.store_block_m, spec.store_block_n
    num_store_threads = spec.num_umma_store_threads
    num_store_warps = num_store_threads // 32
    first_epilogue_warp = spec.num_non_epilogue_threads // 32
    a_smem_dtype = _TORCH_SMEM_DTYPE[spec.a_dtype]
    b_smem_dtype = _TORCH_SMEM_DTYPE[spec.b_dtype]
    cd_dtype = _TORCH_SMEM_DTYPE[spec.cd_dtype]
    sfa_stages_per_load = spec.num_sfa_stages_per_load
    sfb_stages_per_load = spec.num_sfb_stages_per_load
    num_sfa_chunks = sf_block_m // NUM_UTCCP_ALIGNED_ELEMS
    num_sfb_chunks = sf_block_n // NUM_UTCCP_ALIGNED_ELEMS
    umma_k_steps = block_k // UMMA_K
    # `kMayHaveTailKBlock` (sm100_fp8_fp4_gemm_1d1d.cuh:338).  When set, the final K
    # block of a group may be partial and only its leading UMMA-K steps are valid.
    may_have_tail_k = spec.may_have_tail_k_block
    num_sms = spec.num_sms
    swap_ab = spec.swap_ab
    with_accumulation = spec.with_accumulation
    cd_is_fp32 = spec.cd_dtype == "fp32"
    cd_elem = 4 if cd_is_fp32 else 2
    swizzle_cd = spec.swizzle_cd_mode
    major_a_is_k = spec.major_a is Major.K
    major_b_is_k = spec.major_b is Major.K
    is_multicast_on_a = spec.is_multicast_on_a
    is_m_grouped_contiguous = spec.gemm_type is GemmType.M_GROUPED_CONTIGUOUS
    is_m_grouped_masked = spec.gemm_type is GemmType.M_GROUPED_MASKED
    is_batched = spec.gemm_type is GemmType.BATCHED
    is_k_grouped = spec.gemm_type.is_k_grouped_contiguous
    is_k_grouped_psum = spec.gemm_type is GemmType.K_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT
    k_alignment = spec.k_alignment
    sf_k_span = spec.gran_k_a * 4
    is_m_grouped_psum = spec.gemm_type is GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT
    # `get_aligned_effective_m_in_block` (scheduler/gemm.cuh:189, sketch `:559`).
    # On the last block of a psum group without zero padding three places narrow
    # together: the loader's per-CTA M offset (`:211`, `:236`), the instruction
    # descriptor's `n_dim_` (`:332`) and the swap-AB store count
    # (`sm100_store_cd_swap_ab.cuh:50`).  Narrowing fewer than all three leaves
    # part of the block unwritten.
    use_effective_m = spec.swap_ab and is_m_grouped_psum and not spec.ensure_zero_padding
    # `n_dim_` is bits [17,23) of the instruction descriptor.
    UMMA_N_FIELD_MASK = 0xFF81FFFF
    num_groups = spec.num_groups
    blocks_per_group = _blocks_per_group(spec)
    swizzle_a_enum = _swizzle_enum(spec.swizzle_a_mode)
    swizzle_b_enum = _swizzle_enum(spec.swizzle_b_mode)
    umma_a_dtype = _UMMA_DTYPE[spec.b_dtype if spec.swap_ab else spec.a_dtype]
    umma_b_dtype = _UMMA_DTYPE[spec.a_dtype if spec.swap_ab else spec.b_dtype]
    umma_trans_a = (spec.major_b if spec.swap_ab else spec.major_a) is Major.MN
    umma_trans_b = (spec.major_a if spec.swap_ab else spec.major_b) is Major.MN
    a_bytes_per_stage = spec.smem_a_size_per_stage
    b_bytes_per_stage = spec.smem_b_size_per_stage
    # `advance_umma_desc_lo`: base + ((offset + k_idx * stride_k) * sizeof) >> 4.
    # `stride_k` is 1 for a K-major operand and the swizzle atom for MN-major.
    a_stride_k = 1 if major_a_is_k else (spec.swizzle_a_mode or load_block_m)
    b_stride_k = 1 if major_b_is_k else (spec.swizzle_b_mode or load_block_n)
    a_k_step_units = UMMA_K * a_stride_k // 16
    b_k_step_units = UMMA_K * b_stride_k // 16

    # Non-swap epilogue geometry (`sm100_store_cd.cuh:30-49`).
    elems_per_bank_group = 16 // cd_elem
    num_m_waves = block_m // store_block_m
    num_stores = block_n // store_block_n
    elems_per_store = store_block_n // elems_per_bank_group
    has_shortcut = (swizzle_cd // 16) == 8
    cd_stage_bytes = store_block_m * store_block_n * cd_elem

    # Swap-AB epilogue geometry.  The accumulator is transposed relative to the
    # output, so a full warpgroup reads all 128 TMEM rows and `stmatrix.trans`
    # does the transpose on the way to shared memory.
    store_block_n_atom = swizzle_cd // cd_elem
    warps_per_atom = store_block_n_atom // 32 if store_block_n_atom >= 32 else 1
    num_atom_rows = store_block_m // 8
    num_n_atoms = store_block_n // store_block_n_atom
    num_swap_stores = block_m // store_block_m
    if swap_ab:
        if store_block_n != 128:
            raise ValueError("swap-AB needs STORE_BLOCK_N == 128")
        if swizzle_cd != 128:
            raise ValueError("swap-AB needs a 128 B C/D swizzle")
        if store_block_m % 8 != 0 or store_block_n_atom % 32 != 0:
            raise ValueError("invalid swap-AB store block")

    # TMA atom counts: the box is split along its inner dimension.
    a_inner = block_k if spec.major_a is Major.K else load_block_m
    b_inner = block_k if spec.major_b is Major.K else load_block_n
    a_atom = a_inner if spec.swizzle_a_mode == 0 else spec.swizzle_a_mode
    b_atom = b_inner if spec.swizzle_b_mode == 0 else spec.swizzle_b_mode
    num_a_atoms = a_inner // a_atom
    num_b_atoms = b_inner // b_atom

    # FP4 transfers half a byte per element while occupying one SMEM byte.
    arrival_bytes_ab = spec.smem_a_size_per_stage // (1 if spec.a_dtype == "fp8" else 2) + (
        spec.smem_b_size_per_stage // (1 if spec.b_dtype == "fp8" else 2)
    )

    swizzle_a = SwizzleMode(_swizzle_enum(spec.swizzle_a_mode))
    swizzle_b = SwizzleMode(_swizzle_enum(spec.swizzle_b_mode))

    def _operand_layout(is_k_major, load_mn, swizzle_mode, smem_dtype, swizzle_enum, elem):
        """SMEM layout and descriptor stride offset for one operand (`mma/sm100.cuh:107`).

        For a K-major operand the swizzle atom runs along K, so the layout is
        built over `(stage, MN, K)`; for an MN-major one it runs along MN and
        the layout is built over `(stage, K, MN)`.  Only `sdo` reaches the
        descriptor -- every in-scope config leaves the leading offset at zero
        (see the single-atom checks below) -- but `ldo` is still computed
        because the 16-byte swizzle swaps the two.
        """
        if is_k_major:
            shape = (stages, load_mn, block_k)
            # `kSwizzleMode * pack == BLOCK_K * sizeof` is asserted upstream, so
            # each block holds exactly one swizzle atom along K.
            sdo = 8 * block_k * elem // 16
        else:
            atom = swizzle_mode // elem if swizzle_mode else load_mn
            shape = (stages, block_k, load_mn)
            sdo = 8 * atom * elem // 16
            ldo = block_k * atom * elem // 16
            if swizzle_mode == 16:
                sdo = ldo
        return mma_shared_layout(smem_dtype, swizzle_enum, shape), sdo

    a_layout, a_desc_sdo = _operand_layout(
        major_a_is_k, load_block_m, spec.swizzle_a_mode, a_smem_dtype, swizzle_a, 1
    )
    b_layout, b_desc_sdo = _operand_layout(
        major_b_is_k, load_block_n, spec.swizzle_b_mode, b_smem_dtype, swizzle_b, 1
    )
    # The MN-major TMA destination for atom `i` sits at `i * BLOCK_K * atom`
    # elements, which the `(stage, K, MN)` view cannot express with a plain
    # index; every in-scope MN-major config has exactly one atom.
    if not major_a_is_k and num_a_atoms != 1:
        raise ValueError("MN-major A with multiple swizzle atoms is not supported")
    if not major_b_is_k and num_b_atoms != 1:
        raise ValueError("MN-major B with multiple swizzle atoms is not supported")
    sf_desc_sdo = 8 * 4 * 4 // 16

    # Matrix / instruction descriptors, folded here rather than built by the
    # runtime encoders (`encode_matrix_descriptor` /
    # `encode_instr_descriptor_block_scaled`), which are pure-C bitfield fills
    # and become opaque helpers in the generated CUDA.  Every field but the
    # shared address is a constant at this point; the address is ORed in at
    # run time by `_with_smem_addr`.
    A_DESC_BASE = encode_smem_descriptor_base_uint64(0, a_desc_sdo, swizzle_a_enum)
    B_DESC_BASE = encode_smem_descriptor_base_uint64(0, b_desc_sdo, swizzle_b_enum)
    SF_DESC_BASE = encode_smem_descriptor_base_uint64(0, sf_desc_sdo, 0)
    INSTR_DESC = encode_instr_descriptor_block_scaled_uint32(
        M=umma_m,
        N=umma_n,
        K=UMMA_K,
        d_dtype="float32",
        a_dtype=umma_a_dtype,
        b_dtype=umma_b_dtype,
        sf_dtype="float8_e8m0fnu",
        trans_a=umma_trans_a,
        trans_b=umma_trans_b,
        cta_group=cta_group,
    )

    # Barrier family bases, in 8-byte slots after the SF region.
    full_base, empty_base = 0, stages
    with_sf_base = stages * 2
    tmem_full_base = stages * 3
    tmem_empty_base = stages * 3 + NUM_EPILOGUE_STAGES
    num_barriers = stages * 3 + NUM_EPILOGUE_STAGES * 2

    # Batched is the only type whose A/B/C-D descriptors are rank 3.
    rank = 3 if is_batched else 2
    mma_chain = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.scale_vec::1X"
    utccp_chain = f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
    commit_chain = (
        f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
    )
    commit_mc_chain = (
        f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster"
        ".multicast::cluster.b64"
    )
    # The source always passes `num_tma_multicast = 1` to `tma::copy` in this kernel
    # (`sm100_fp8_fp4_gemm_1d1d.cuh:240`): each CTA loads its own `LOAD_BLOCK_M`
    # slice and the 2-CTA behaviour lives in the UMMA, not in the TMA.  No
    # `.cta_group::N` here, matching the PTX DeepGEMM emits for the same config.
    load_chain = (
        f"cp.async.bulk.tensor.{rank}d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint"
    )
    # The scale-factor descriptors are rank 2 for every type: the batch is folded
    # into their outer (K) extent, not into a third dimension.
    sf_load_chain = (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    store_chain = f"cp.async.bulk.tensor.{rank}d.global.shared::cta.tile.bulk_group"
    reduce_chain = f"cp.reduce.async.bulk.tensor.{rank}d.global.shared::cta.add.tile.bulk_group"

    def _cd_word(stage, ep_warp, row, col, j):
        """Word index of the `j`-th of four `u32` in one 16-byte bank group.

        Mirrors `sm100_store_cd.cuh:80-82`: stage base, warp offset, then the
        swizzled in-atom row and column, all in bytes.
        """
        # Every term is a multiple of 16 bytes, so fold the `/ 4` into the
        # compile-time factors instead of dividing the assembled byte offset.
        return (
            stage * (cd_stage_bytes // 4)
            + ep_warp * (32 * swizzle_cd // 4)
            + row * ((16 * 8) // 4)
            + col * (16 // 4)
            + j
        )

    def _swizzled(block_idx, num_m_blocks, num_n_blocks):
        """`get_swizzled_block_idx`: group blocks along the multicast axis.

        Returns `(m_block_idx, n_block_idx)` as expressions.  Each caller binds
        both to locals once and reuses them, as the source does.
        """
        if is_batched:
            if is_multicast_on_a:
                return _udiv(block_idx, num_n_blocks), _umod(block_idx, num_n_blocks)
            return _umod(block_idx, num_m_blocks), _udiv(block_idx, num_m_blocks)
        primary = num_n_blocks if is_multicast_on_a else num_m_blocks
        secondary = num_m_blocks if is_multicast_on_a else num_n_blocks
        per_group = secondary * blocks_per_group
        # All four quantities are non-negative block counts; keeping the division
        # unsigned avoids the signed-floordiv fixup (see `_uceil`), which matters
        # here because `per_group` is a runtime value whenever the group sizes are.
        first_block = _udiv(block_idx, per_group) * blocks_per_group
        in_group = _umod(block_idx, per_group)
        in_group_blocks = T.min(blocks_per_group, primary - first_block)
        if is_multicast_on_a:
            return _udiv(in_group, in_group_blocks), first_block + _umod(in_group, in_group_blocks)
        return first_block + _umod(in_group, in_group_blocks), _udiv(in_group, in_group_blocks)

    def _tma_coords(coords, batch):
        """TMA tensor coordinates, batch index appended for a 3-D descriptor.

        `is_batched` is the only thing that changes the arity of every
        `cp.async.bulk.tensor` in this kernel, so it is resolved here once
        rather than by duplicating each call site.
        """
        out = [T.cast(c, "int32") for c in coords]
        if is_batched:
            out.append(T.cast(batch, "int32"))
        return out

    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("sm100_fp8_fp4_gemm_1d1d")
            grouped_layout_ptr = T.arg("grouped_layout_ptr", T.handle())
            grouped_len = T.arg("grouped_len", T.int32())
            shape_m = T.arg("shape_m", T.int32())
            shape_n = T.arg("shape_n", T.int32())
            shape_k = T.arg("shape_k", T.int32())
            tensor_map_a = T.arg("tensor_map_a", T.TensorMap())
            tensor_map_b = T.arg("tensor_map_b", T.TensorMap())
            tensor_map_sfa = T.arg("tensor_map_sfa", T.TensorMap())
            tensor_map_sfb = T.arg("tensor_map_sfb", T.TensorMap())
            tensor_map_cd = T.arg("tensor_map_cd", T.TensorMap())
            grouped_layout = _builder_assign(
                "grouped_layout",
                T.match_buffer(grouped_layout_ptr, (grouped_len,), "int32"),
                locals().get("grouped_layout", _BUILDER_MISSING),
            )
            _builder_emit(T.device_entry())
            _builder_emit(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            sm_idx = _builder_assign(
                "sm_idx", T.cta_id([spec.num_sms]), locals().get("sm_idx", _BUILDER_MISSING)
            )
            if cta_group > 1:
                cta_in_cluster = _builder_assign(
                    "cta_in_cluster",
                    T.cta_id_in_cluster([cta_group]),
                    locals().get("cta_in_cluster", _BUILDER_MISSING),
                )
                is_leader_cta = _builder_assign(
                    "is_leader_cta",
                    cta_in_cluster == 0,
                    locals().get("is_leader_cta", _BUILDER_MISSING),
                )
            else:
                is_leader_cta = _builder_assign(
                    "is_leader_cta", True, locals().get("is_leader_cta", _BUILDER_MISSING)
                )
            thread_idx = _builder_assign(
                "thread_idx",
                T.thread_id([spec.num_non_epilogue_threads + spec.num_epilogue_threads]),
                locals().get("thread_idx", _BUILDER_MISSING),
            )
            lane_idx = _builder_assign(
                "lane_idx", T.lane_id([32]), locals().get("lane_idx", _BUILDER_MISSING)
            )
            warp = _builder_alloc_scalar("warp", "int32")
            _builder_emit(
                T.ptx.shfl_sync.idx.b32(
                    warp, thread_idx // 32, T.uint32(0), T.uint32(31), T.uint32(4294967295)
                )
            )
            smem = _builder_assign(
                "smem",
                T.alloc_buffer([spec.smem_size], "uint8", scope="shared.dyn", align=1024),
                locals().get("smem", _BUILDER_MISSING),
            )
            _builder_emit(T.attr({"tirx.dyn_smem_bytes": spec.smem_size}))
            smem_cd_data = _builder_bind(
                "smem_cd_data",
                T.reinterpret(PointerType(PrimType(cd_dtype)), smem.ptr_to([0])),
                None,
            )
            smem_a_data = _builder_bind(
                "smem_a_data",
                T.reinterpret(
                    PointerType(PrimType(a_smem_dtype)), smem.ptr_to([spec.smem_a_offset])
                ),
                None,
            )
            smem_b_data = _builder_bind(
                "smem_b_data",
                T.reinterpret(
                    PointerType(PrimType(b_smem_dtype)), smem.ptr_to([spec.smem_b_offset])
                ),
                None,
            )
            smem_sfa_data = _builder_bind(
                "smem_sfa_data",
                T.reinterpret(PointerType(PrimType("uint32")), smem.ptr_to([spec.smem_sfa_offset])),
                None,
            )
            smem_sfb_data = _builder_bind(
                "smem_sfb_data",
                T.reinterpret(PointerType(PrimType("uint32")), smem.ptr_to([spec.smem_sfb_offset])),
                None,
            )
            smem_bar_data = _builder_bind(
                "smem_bar_data",
                T.reinterpret(
                    PointerType(PrimType("uint64")), smem.ptr_to([spec.smem_barrier_offset])
                ),
                None,
            )
            smem_tmem_data = _builder_bind(
                "smem_tmem_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")), smem.ptr_to([spec.smem_tmem_ptr_offset])
                ),
                None,
            )
            smem_cd = _builder_assign(
                "smem_cd",
                T.decl_buffer(
                    (NUM_TMA_STORE_STAGES, store_block_m, store_block_n),
                    cd_dtype,
                    data=smem_cd_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=1024,
                ),
                locals().get("smem_cd", _BUILDER_MISSING),
            )
            smem_a = _builder_assign(
                "smem_a",
                T.decl_buffer(
                    (stages, load_block_m, block_k),
                    a_smem_dtype,
                    data=smem_a_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=1024,
                    layout=a_layout,
                ),
                locals().get("smem_a", _BUILDER_MISSING),
            )
            smem_b = _builder_assign(
                "smem_b",
                T.decl_buffer(
                    (stages, load_block_n, block_k),
                    b_smem_dtype,
                    data=smem_b_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=1024,
                    layout=b_layout,
                ),
                locals().get("smem_b", _BUILDER_MISSING),
            )
            smem_sfa = _builder_assign(
                "smem_sfa",
                T.decl_buffer(
                    (stages, sf_block_m),
                    "uint32",
                    data=smem_sfa_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_sfa", _BUILDER_MISSING),
            )
            smem_sfb = _builder_assign(
                "smem_sfb",
                T.decl_buffer(
                    (stages, sf_block_n),
                    "uint32",
                    data=smem_sfb_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_sfb", _BUILDER_MISSING),
            )
            barriers = _builder_assign(
                "barriers",
                T.decl_buffer(
                    (num_barriers,),
                    "uint64",
                    data=smem_bar_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=8,
                ),
                locals().get("barriers", _BUILDER_MISSING),
            )
            tmem_slot = _builder_assign(
                "tmem_slot",
                T.decl_buffer(
                    (1,), "uint32", data=smem_tmem_data, scope="shared.dyn", elem_offset=0, align=4
                ),
                locals().get("tmem_slot", _BUILDER_MISSING),
            )
            smem_cd_word_data = _builder_bind(
                "smem_cd_word_data",
                T.reinterpret(PointerType(PrimType("uint32")), smem.ptr_to([0])),
                None,
            )
            smem_cd_u32 = _builder_assign(
                "smem_cd_u32",
                T.decl_buffer(
                    (NUM_TMA_STORE_STAGES * cd_stage_bytes // 4,),
                    "uint32",
                    data=smem_cd_word_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=1024,
                ),
                locals().get("smem_cd_u32", _BUILDER_MISSING),
            )
            tmem = _builder_assign(
                "tmem",
                T.decl_buffer(
                    (128, spec.num_tmem_cols),
                    "float32",
                    scope="tmem",
                    allocated_addr=tmem_slot[0],
                    layout=TileLayout(S[(128, spec.num_tmem_cols) : (1 @ TLane, 1 @ TCol)]),
                ),
                locals().get("tmem", _BUILDER_MISSING),
            )
            sfa_tmem = _builder_assign(
                "sfa_tmem",
                T.decl_buffer(
                    (128, sf_block_m // 32),
                    "float8_e8m0fnu",
                    scope="tmem",
                    allocated_addr=spec.tmem_start_col_of_sfa,
                    layout=sf_tmem_layout(128, SF_K=sf_block_m // 32, sf_per_mma=1),
                ),
                locals().get("sfa_tmem", _BUILDER_MISSING),
            )
            sfb_tmem = _builder_assign(
                "sfb_tmem",
                T.decl_buffer(
                    (128, sf_block_n // 32),
                    "float8_e8m0fnu",
                    scope="tmem",
                    allocated_addr=spec.tmem_start_col_of_sfb,
                    layout=sf_tmem_layout(128, SF_K=sf_block_n // 32, sf_per_mma=1),
                ),
                locals().get("sfb_tmem", _BUILDER_MISSING),
            )
            if cta_group > 1:
                _builder_emit(T.ptx.barrier.cluster.arrive.relaxed.aligned())
                _builder_emit(T.ptx.barrier.cluster.wait.acquire.aligned())
            with T.If(warp == 0):
                with T.Then():
                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_a))))
                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_b))))
                    _builder_emit(
                        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_sfa)))
                    )
                    _builder_emit(
                        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_sfb)))
                    )
                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_cd))))
            with T.If(warp == 1):
                with T.Then():
                    init_elected = _builder_alloc_scalar("init_elected", "uint32")
                    init_elected_lane = _builder_alloc_scalar("init_elected_lane", "uint32")
                    _builder_emit(
                        T.ptx.elect_sync(init_elected_lane, init_elected, T.uint32(4294967295))
                    )
                    with T.If(init_elected == T.uint32(1)):
                        with T.Then():
                            with T.unroll(0, stages) as s:
                                IRBuilder.name("s", s)
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        barriers.ptr_to([full_base + s]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        barriers.ptr_to([empty_base + s]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        barriers.ptr_to([with_sf_base + s]),
                                        T.uint32(cta_group * 32),
                                    )
                                )
                            with T.unroll(0, NUM_EPILOGUE_STAGES) as e:
                                IRBuilder.name("e", e)
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        barriers.ptr_to([tmem_full_base + e]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        barriers.ptr_to([tmem_empty_base + e]),
                                        T.uint32(cta_group * num_store_threads),
                                    )
                                )
                            _builder_emit(T.evaluate(T.ptx.fence.mbarrier_init.release.cluster()))
                with T.Else():
                    with T.If(warp == 2):
                        with T.Then():
                            _builder_emit(
                                T.ptx[
                                    f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"
                                ](T.address_of(tmem_slot[0]), T.uint32(spec.num_tmem_cols))
                            )
            if cta_group > 1:
                _builder_emit(T.ptx.barrier.cluster.arrive.relaxed.aligned())
                _builder_emit(T.ptx.barrier.cluster.wait.acquire.aligned())
            else:
                _builder_emit(T.ptx.bar.sync(T.uint32(0)))
            _builder_emit(T.evaluate(T.ptx.griddepcontrol.wait()))
            eff_m = _builder_scalar("eff_m", spec.shape_m if spec.shape_m > 0 else shape_m, "int32")
            eff_n = _builder_scalar("eff_n", spec.shape_n if spec.shape_n > 0 else shape_n, "int32")
            eff_k = _builder_scalar("eff_k", spec.shape_k if spec.shape_k > 0 else shape_k, "int32")
            num_m_blocks = _builder_alloc_scalar("num_m_blocks", "int32")
            num_n_blocks = _builder_alloc_scalar("num_n_blocks", "int32")
            num_blocks = _builder_alloc_scalar("num_blocks", "int32")
            num_k_blocks = _builder_alloc_scalar("num_k_blocks", "int32")
            num_m_blocks = _builder_assign(
                "num_m_blocks",
                _uceil(eff_m, block_m),
                locals().get("num_m_blocks", _BUILDER_MISSING),
            )
            num_n_blocks = _builder_assign(
                "num_n_blocks",
                (eff_n + (block_n - 1)) // block_n,
                locals().get("num_n_blocks", _BUILDER_MISSING),
            )
            num_blocks = _builder_assign(
                "num_blocks",
                num_m_blocks * num_n_blocks,
                locals().get("num_blocks", _BUILDER_MISSING),
            )
            num_k_blocks = _builder_assign(
                "num_k_blocks",
                _uceil(eff_k, block_k),
                locals().get("num_k_blocks", _BUILDER_MISSING),
            )
            shape_sfa_k = _builder_alloc_scalar("shape_sfa_k", "int32")
            shape_sfb_k = _builder_alloc_scalar("shape_sfb_k", "int32")
            shape_sfa_k = _builder_assign(
                "shape_sfa_k",
                (eff_k + (spec.gran_k_a * 4 - 1)) // (spec.gran_k_a * 4),
                locals().get("shape_sfa_k", _BUILDER_MISSING),
            )
            shape_sfb_k = _builder_assign(
                "shape_sfb_k",
                (eff_k + (spec.gran_k_b * 4 - 1)) // (spec.gran_k_b * 4),
                locals().get("shape_sfb_k", _BUILDER_MISSING),
            )
            with T.If(warp == 0):
                with T.Then():
                    ld_elected = _builder_alloc_scalar("ld_elected", "uint32")
                    ld_elected_lane = _builder_alloc_scalar("ld_elected_lane", "uint32")
                    _builder_emit(
                        T.ptx.elect_sync(ld_elected_lane, ld_elected, T.uint32(4294967295))
                    )
                    with T.If(ld_elected == T.uint32(1)):
                        with T.Then():
                            ld_stage = _builder_alloc_scalar("ld_stage", "int32")
                            ld_phase = _builder_alloc_scalar("ld_phase", "int32")
                            ld_stage = _builder_assign(
                                "ld_stage", 0, locals().get("ld_stage", _BUILDER_MISSING)
                            )
                            ld_phase = _builder_assign(
                                "ld_phase", 0, locals().get("ld_phase", _BUILDER_MISSING)
                            )
                            ld_it = _builder_alloc_scalar("ld_it", "int32")
                            ld_valid = _builder_alloc_scalar("ld_valid", "int32")
                            ld_grp = _builder_alloc_scalar("ld_grp", "int32")
                            ld_cum = _builder_alloc_scalar("ld_cum", "int32")
                            ld_nmb = _builder_alloc_scalar("ld_nmb", "int32")
                            ld_last = _builder_alloc_scalar("ld_last", "int32")
                            ld_psum = _builder_alloc_scalar("ld_psum", "int32")
                            ld_nb = _builder_alloc_scalar("ld_nb", "int32")
                            ld_nxt = _builder_alloc_scalar("ld_nxt", "int32")
                            ld_nxtk = _builder_alloc_scalar("ld_nxtk", "int32")
                            ld_sfk = _builder_alloc_scalar("ld_sfk", "int32")
                            ld_kblocks = _builder_alloc_scalar("ld_kblocks", "int32")
                            ld_vgrp = _builder_alloc_scalar("ld_vgrp", "int32")
                            ld_kend = _builder_alloc_scalar("ld_kend", "int32")
                            ld_it = _builder_assign(
                                "ld_it", 0, locals().get("ld_it", _BUILDER_MISSING)
                            )
                            ld_valid = _builder_assign(
                                "ld_valid", 1, locals().get("ld_valid", _BUILDER_MISSING)
                            )
                            ld_grp = _builder_assign(
                                "ld_grp", 0, locals().get("ld_grp", _BUILDER_MISSING)
                            )
                            ld_cum = _builder_assign(
                                "ld_cum", 0, locals().get("ld_cum", _BUILDER_MISSING)
                            )
                            ld_last = _builder_assign(
                                "ld_last", 0, locals().get("ld_last", _BUILDER_MISSING)
                            )
                            ld_sfk = _builder_assign(
                                "ld_sfk", 0, locals().get("ld_sfk", _BUILDER_MISSING)
                            )
                            ld_vgrp = _builder_assign(
                                "ld_vgrp", 0, locals().get("ld_vgrp", _BUILDER_MISSING)
                            )
                            ld_kend = _builder_assign(
                                "ld_kend", 0, locals().get("ld_kend", _BUILDER_MISSING)
                            )
                            ld_nxt = _builder_assign(
                                "ld_nxt", 0, locals().get("ld_nxt", _BUILDER_MISSING)
                            )
                            ld_nxtk = _builder_assign(
                                "ld_nxtk", 0, locals().get("ld_nxtk", _BUILDER_MISSING)
                            )
                            ld_kblocks = _builder_assign(
                                "ld_kblocks", 0, locals().get("ld_kblocks", _BUILDER_MISSING)
                            )
                            if is_k_grouped:
                                ld_psum = _builder_assign(
                                    "ld_psum", 0, locals().get("ld_psum", _BUILDER_MISSING)
                                )
                                ld_nmb = _builder_assign(
                                    "ld_nmb", num_m_blocks, locals().get("ld_nmb", _BUILDER_MISSING)
                                )
                                if is_k_grouped_psum:
                                    with T.While(ld_grp < num_groups):
                                        ld_nxtk = _builder_assign(
                                            "ld_nxtk",
                                            grouped_layout[ld_grp],
                                            locals().get("ld_nxtk", _BUILDER_MISSING),
                                        )
                                        ld_last = _builder_assign(
                                            "ld_last",
                                            _uceil(ld_kend, k_alignment) * k_alignment,
                                            locals().get("ld_last", _BUILDER_MISSING),
                                        )
                                        ld_psum = _builder_assign(
                                            "ld_psum",
                                            ld_nxtk - ld_last,
                                            locals().get("ld_psum", _BUILDER_MISSING),
                                        )
                                        ld_kend = _builder_assign(
                                            "ld_kend",
                                            ld_nxtk,
                                            locals().get("ld_kend", _BUILDER_MISSING),
                                        )
                                        with T.If(ld_psum > 0):
                                            with T.Then():
                                                T.evaluate(T.break_loop())
                                        ld_grp = _builder_assign(
                                            "ld_grp",
                                            ld_grp + 1,
                                            locals().get("ld_grp", _BUILDER_MISSING),
                                        )
                                else:
                                    with T.While(ld_grp < num_groups):
                                        ld_psum = _builder_assign(
                                            "ld_psum",
                                            grouped_layout[ld_grp],
                                            locals().get("ld_psum", _BUILDER_MISSING),
                                        )
                                        with T.If(ld_psum > 0):
                                            with T.Then():
                                                T.evaluate(T.break_loop())
                                        ld_grp = _builder_assign(
                                            "ld_grp",
                                            ld_grp + 1,
                                            locals().get("ld_grp", _BUILDER_MISSING),
                                        )
                                    ld_nxt = _builder_assign(
                                        "ld_nxt",
                                        ld_grp + 1,
                                        locals().get("ld_nxt", _BUILDER_MISSING),
                                    )
                                    with T.While(ld_nxt < num_groups):
                                        ld_nxtk = _builder_assign(
                                            "ld_nxtk",
                                            grouped_layout[ld_nxt],
                                            locals().get("ld_nxtk", _BUILDER_MISSING),
                                        )
                                        with T.If(ld_nxtk > 0):
                                            with T.Then():
                                                T.evaluate(T.break_loop())
                                        ld_nxt = _builder_assign(
                                            "ld_nxt",
                                            ld_nxt + 1,
                                            locals().get("ld_nxt", _BUILDER_MISSING),
                                        )
                            elif is_m_grouped_psum:
                                ld_psum = _builder_assign(
                                    "ld_psum",
                                    grouped_layout[0],
                                    locals().get("ld_psum", _BUILDER_MISSING),
                                )
                                ld_nmb = _builder_assign(
                                    "ld_nmb",
                                    _uceil(ld_psum, block_m),
                                    locals().get("ld_nmb", _BUILDER_MISSING),
                                )
                            else:
                                ld_psum = _builder_assign(
                                    "ld_psum", 0, locals().get("ld_psum", _BUILDER_MISSING)
                                )
                                ld_nmb = _builder_assign(
                                    "ld_nmb", num_m_blocks, locals().get("ld_nmb", _BUILDER_MISSING)
                                )
                            with T.While(ld_valid == 1):
                                ld_nb = _builder_assign(
                                    "ld_nb",
                                    ld_it * num_sms + sm_idx,
                                    locals().get("ld_nb", _BUILDER_MISSING),
                                )
                                ld_done = _builder_alloc_scalar("ld_done", "int32")
                                ld_done = _builder_assign(
                                    "ld_done", 0, locals().get("ld_done", _BUILDER_MISSING)
                                )
                                if is_m_grouped_masked:
                                    with T.While(ld_done == 0):
                                        with T.If(ld_grp == num_groups):
                                            with T.Then():
                                                ld_valid = _builder_assign(
                                                    "ld_valid",
                                                    0,
                                                    locals().get("ld_valid", _BUILDER_MISSING),
                                                )
                                                ld_done = _builder_assign(
                                                    "ld_done",
                                                    1,
                                                    locals().get("ld_done", _BUILDER_MISSING),
                                                )
                                            with T.Else():
                                                ld_nmb = _builder_assign(
                                                    "ld_nmb",
                                                    _uceil(grouped_layout[ld_grp], block_m),
                                                    locals().get("ld_nmb", _BUILDER_MISSING),
                                                )
                                                with T.If(ld_nb < (ld_cum + ld_nmb) * num_n_blocks):
                                                    with T.Then():
                                                        ld_done = _builder_assign(
                                                            "ld_done",
                                                            1,
                                                            locals().get(
                                                                "ld_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        ld_cum = _builder_assign(
                                                            "ld_cum",
                                                            ld_cum + ld_nmb,
                                                            locals().get(
                                                                "ld_cum", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ld_grp = _builder_assign(
                                                            "ld_grp",
                                                            ld_grp + 1,
                                                            locals().get(
                                                                "ld_grp", _BUILDER_MISSING
                                                            ),
                                                        )
                                elif is_m_grouped_psum:
                                    with T.While(ld_done == 0):
                                        with T.If(ld_nb < (ld_cum + ld_nmb) * num_n_blocks):
                                            with T.Then():
                                                ld_done = _builder_assign(
                                                    "ld_done",
                                                    1,
                                                    locals().get("ld_done", _BUILDER_MISSING),
                                                )
                                            with T.Else():
                                                ld_grp = _builder_assign(
                                                    "ld_grp",
                                                    ld_grp + 1,
                                                    locals().get("ld_grp", _BUILDER_MISSING),
                                                )
                                                with T.If(ld_grp == num_groups):
                                                    with T.Then():
                                                        ld_valid = _builder_assign(
                                                            "ld_valid",
                                                            0,
                                                            locals().get(
                                                                "ld_valid", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ld_done = _builder_assign(
                                                            "ld_done",
                                                            1,
                                                            locals().get(
                                                                "ld_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        ld_last = _builder_assign(
                                                            "ld_last",
                                                            _uceil(ld_psum, block_m) * block_m,
                                                            locals().get(
                                                                "ld_last", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ld_psum = _builder_assign(
                                                            "ld_psum",
                                                            grouped_layout[ld_grp],
                                                            locals().get(
                                                                "ld_psum", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ld_cum = _builder_assign(
                                                            "ld_cum",
                                                            ld_cum + ld_nmb,
                                                            locals().get(
                                                                "ld_cum", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ld_nmb = _builder_assign(
                                                            "ld_nmb",
                                                            _uceil(ld_psum - ld_last, block_m),
                                                            locals().get(
                                                                "ld_nmb", _BUILDER_MISSING
                                                            ),
                                                        )
                                elif is_k_grouped:
                                    with T.While(ld_done == 0):
                                        with T.If(ld_grp == num_groups):
                                            with T.Then():
                                                ld_valid = _builder_assign(
                                                    "ld_valid",
                                                    0,
                                                    locals().get("ld_valid", _BUILDER_MISSING),
                                                )
                                                ld_done = _builder_assign(
                                                    "ld_done",
                                                    1,
                                                    locals().get("ld_done", _BUILDER_MISSING),
                                                )
                                            with T.Else():
                                                with T.If(ld_nb < (ld_vgrp + 1) * num_blocks):
                                                    with T.Then():
                                                        ld_done = _builder_assign(
                                                            "ld_done",
                                                            1,
                                                            locals().get(
                                                                "ld_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        ld_sfk = _builder_assign(
                                                            "ld_sfk",
                                                            ld_sfk + _uceil(ld_psum, sf_k_span),
                                                            locals().get(
                                                                "ld_sfk", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ld_vgrp = _builder_assign(
                                                            "ld_vgrp",
                                                            ld_vgrp + 1,
                                                            locals().get(
                                                                "ld_vgrp", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        if is_k_grouped_psum:
                                                            ld_grp = _builder_assign(
                                                                "ld_grp",
                                                                ld_grp + 1,
                                                                locals().get(
                                                                    "ld_grp", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            with T.While(ld_grp < num_groups):
                                                                ld_nxtk = _builder_assign(
                                                                    "ld_nxtk",
                                                                    grouped_layout[ld_grp],
                                                                    locals().get(
                                                                        "ld_nxtk", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                ld_last = _builder_assign(
                                                                    "ld_last",
                                                                    _uceil(ld_kend, k_alignment)
                                                                    * k_alignment,
                                                                    locals().get(
                                                                        "ld_last", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                ld_psum = _builder_assign(
                                                                    "ld_psum",
                                                                    ld_nxtk - ld_last,
                                                                    locals().get(
                                                                        "ld_psum", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                ld_kend = _builder_assign(
                                                                    "ld_kend",
                                                                    ld_nxtk,
                                                                    locals().get(
                                                                        "ld_kend", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                with T.If(ld_psum > 0):
                                                                    with T.Then():
                                                                        T.evaluate(T.break_loop())
                                                                ld_grp = _builder_assign(
                                                                    "ld_grp",
                                                                    ld_grp + 1,
                                                                    locals().get(
                                                                        "ld_grp", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                        else:
                                                            ld_last = _builder_assign(
                                                                "ld_last",
                                                                ld_last + ld_psum,
                                                                locals().get(
                                                                    "ld_last", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            ld_grp = _builder_assign(
                                                                "ld_grp",
                                                                ld_nxt,
                                                                locals().get(
                                                                    "ld_grp", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            ld_nxt = _builder_assign(
                                                                "ld_nxt",
                                                                ld_nxt + 1,
                                                                locals().get(
                                                                    "ld_nxt", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            ld_psum = _builder_assign(
                                                                "ld_psum",
                                                                ld_nxtk,
                                                                locals().get(
                                                                    "ld_psum", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            with T.While(ld_nxt < num_groups):
                                                                ld_nxtk = _builder_assign(
                                                                    "ld_nxtk",
                                                                    grouped_layout[ld_nxt],
                                                                    locals().get(
                                                                        "ld_nxtk", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                with T.If(ld_nxtk > 0):
                                                                    with T.Then():
                                                                        T.evaluate(T.break_loop())
                                                                ld_nxt = _builder_assign(
                                                                    "ld_nxt",
                                                                    ld_nxt + 1,
                                                                    locals().get(
                                                                        "ld_nxt", _BUILDER_MISSING
                                                                    ),
                                                                )
                                    ld_cum = _builder_assign(
                                        "ld_cum",
                                        ld_vgrp * num_m_blocks,
                                        locals().get("ld_cum", _BUILDER_MISSING),
                                    )
                                elif is_batched:
                                    with T.If(ld_nb >= num_blocks * num_groups):
                                        with T.Then():
                                            ld_valid = _builder_assign(
                                                "ld_valid",
                                                0,
                                                locals().get("ld_valid", _BUILDER_MISSING),
                                            )
                                        with T.Else():
                                            ld_grp = _builder_assign(
                                                "ld_grp",
                                                ld_nb // num_blocks,
                                                locals().get("ld_grp", _BUILDER_MISSING),
                                            )
                                            ld_cum = _builder_assign(
                                                "ld_cum",
                                                ld_grp * num_m_blocks,
                                                locals().get("ld_cum", _BUILDER_MISSING),
                                            )
                                            ld_nmb = _builder_assign(
                                                "ld_nmb",
                                                num_m_blocks,
                                                locals().get("ld_nmb", _BUILDER_MISSING),
                                            )
                                if not (
                                    is_m_grouped_masked
                                    or is_m_grouped_psum
                                    or is_k_grouped
                                    or is_batched
                                ):
                                    with T.If(ld_nb >= num_blocks):
                                        with T.Then():
                                            ld_valid = _builder_assign(
                                                "ld_valid",
                                                0,
                                                locals().get("ld_valid", _BUILDER_MISSING),
                                            )
                                with T.If(ld_valid == 1):
                                    with T.Then():
                                        ld_kblocks = _builder_assign(
                                            "ld_kblocks",
                                            _uceil(ld_psum, block_k)
                                            if is_k_grouped
                                            else num_k_blocks,
                                            locals().get("ld_kblocks", _BUILDER_MISSING),
                                        )
                                        ld_m_local, ld_n_local = _builder_assign_many(
                                            ("ld_m_local", "ld_n_local"),
                                            _swizzled(
                                                ld_nb - ld_cum * num_n_blocks, ld_nmb, num_n_blocks
                                            ),
                                            (
                                                locals().get("ld_m_local", _BUILDER_MISSING),
                                                locals().get("ld_n_local", _BUILDER_MISSING),
                                            ),
                                        )
                                        m_idx = _builder_alloc_scalar("m_idx", "int32")
                                        n_idx = _builder_alloc_scalar("n_idx", "int32")
                                        m_idx = _builder_assign(
                                            "m_idx",
                                            (
                                                ld_m_local
                                                + (
                                                    _udiv(ld_last, block_m)
                                                    if is_m_grouped_psum
                                                    else 0
                                                )
                                            )
                                            * block_m,
                                            locals().get("m_idx", _BUILDER_MISSING),
                                        )
                                        n_idx = _builder_assign(
                                            "n_idx",
                                            ld_n_local * block_n,
                                            locals().get("n_idx", _BUILDER_MISSING),
                                        )
                                        sfa_mn = _builder_alloc_scalar("sfa_mn", "int32")
                                        sfb_mn = _builder_alloc_scalar("sfb_mn", "int32")
                                        sfa_mn = _builder_assign(
                                            "sfa_mn",
                                            m_idx,
                                            locals().get("sfa_mn", _BUILDER_MISSING),
                                        )
                                        sfb_mn = _builder_assign(
                                            "sfb_mn",
                                            n_idx,
                                            locals().get("sfb_mn", _BUILDER_MISSING),
                                        )
                                        k_b_idx = _builder_alloc_scalar("k_b_idx", "int32")
                                        k_a_offset = _builder_alloc_scalar("k_a_offset", "int32")
                                        k_b_idx = _builder_assign(
                                            "k_b_idx", 0, locals().get("k_b_idx", _BUILDER_MISSING)
                                        )
                                        k_a_offset = _builder_assign(
                                            "k_a_offset",
                                            0,
                                            locals().get("k_a_offset", _BUILDER_MISSING),
                                        )
                                        sfa_k_offset = _builder_alloc_scalar(
                                            "sfa_k_offset", "int32"
                                        )
                                        sfa_k_offset = _builder_assign(
                                            "sfa_k_offset",
                                            0,
                                            locals().get("sfa_k_offset", _BUILDER_MISSING),
                                        )
                                        if is_m_grouped_contiguous:
                                            expert = _builder_alloc_scalar("expert", "int32")
                                            expert = _builder_assign(
                                                "expert",
                                                T.max(0, grouped_layout[m_idx]),
                                                locals().get("expert", _BUILDER_MISSING),
                                            )
                                            if major_b_is_k:
                                                n_idx = _builder_assign(
                                                    "n_idx",
                                                    expert * eff_n + n_idx,
                                                    locals().get("n_idx", _BUILDER_MISSING),
                                                )
                                            else:
                                                k_b_idx = _builder_assign(
                                                    "k_b_idx",
                                                    expert * eff_k + k_b_idx,
                                                    locals().get("k_b_idx", _BUILDER_MISSING),
                                                )
                                            sfb_k_offset = _builder_assign(
                                                "sfb_k_offset",
                                                expert * shape_sfb_k,
                                                locals().get("sfb_k_offset", _BUILDER_MISSING),
                                            )
                                        elif is_m_grouped_masked:
                                            m_idx = _builder_assign(
                                                "m_idx",
                                                ld_grp * eff_m + m_idx,
                                                locals().get("m_idx", _BUILDER_MISSING),
                                            )
                                            if major_b_is_k:
                                                n_idx = _builder_assign(
                                                    "n_idx",
                                                    ld_grp * eff_n + n_idx,
                                                    locals().get("n_idx", _BUILDER_MISSING),
                                                )
                                            else:
                                                k_b_idx = _builder_assign(
                                                    "k_b_idx",
                                                    ld_grp * eff_k + k_b_idx,
                                                    locals().get("k_b_idx", _BUILDER_MISSING),
                                                )
                                            sfa_k_offset = _builder_assign(
                                                "sfa_k_offset",
                                                ld_grp * shape_sfa_k,
                                                locals().get("sfa_k_offset", _BUILDER_MISSING),
                                            )
                                            sfb_k_offset = _builder_assign(
                                                "sfb_k_offset",
                                                ld_grp * shape_sfb_k,
                                                locals().get("sfb_k_offset", _BUILDER_MISSING),
                                            )
                                        elif is_m_grouped_psum:
                                            if major_b_is_k:
                                                n_idx = _builder_assign(
                                                    "n_idx",
                                                    ld_grp * eff_n + n_idx,
                                                    locals().get("n_idx", _BUILDER_MISSING),
                                                )
                                            else:
                                                k_b_idx = _builder_assign(
                                                    "k_b_idx",
                                                    ld_grp * eff_k + k_b_idx,
                                                    locals().get("k_b_idx", _BUILDER_MISSING),
                                                )
                                            sfb_k_offset = _builder_assign(
                                                "sfb_k_offset",
                                                ld_grp * shape_sfb_k,
                                                locals().get("sfb_k_offset", _BUILDER_MISSING),
                                            )
                                        elif is_k_grouped:
                                            k_a_offset = _builder_assign(
                                                "k_a_offset",
                                                ld_last,
                                                locals().get("k_a_offset", _BUILDER_MISSING),
                                            )
                                            k_b_idx = _builder_assign(
                                                "k_b_idx",
                                                ld_last,
                                                locals().get("k_b_idx", _BUILDER_MISSING),
                                            )
                                            sfa_k_offset = _builder_assign(
                                                "sfa_k_offset",
                                                ld_sfk,
                                                locals().get("sfa_k_offset", _BUILDER_MISSING),
                                            )
                                            sfb_k_offset = _builder_assign(
                                                "sfb_k_offset",
                                                ld_sfk,
                                                locals().get("sfb_k_offset", _BUILDER_MISSING),
                                            )
                                        elif is_batched:
                                            sfa_k_offset = _builder_assign(
                                                "sfa_k_offset",
                                                ld_grp * shape_sfa_k,
                                                locals().get("sfa_k_offset", _BUILDER_MISSING),
                                            )
                                            sfb_k_offset = _builder_alloc_scalar(
                                                "sfb_k_offset", "int32"
                                            )
                                            sfb_k_offset = _builder_assign(
                                                "sfb_k_offset",
                                                ld_grp * shape_sfb_k,
                                                locals().get("sfb_k_offset", _BUILDER_MISSING),
                                            )
                                        else:
                                            sfb_k_offset = _builder_alloc_scalar(
                                                "sfb_k_offset", "int32"
                                            )
                                            sfb_k_offset = _builder_assign(
                                                "sfb_k_offset",
                                                0,
                                                locals().get("sfb_k_offset", _BUILDER_MISSING),
                                            )
                                        if use_effective_m:
                                            ld_eff_m = _builder_alloc_scalar("ld_eff_m", "int32")
                                            ld_eff_m = _builder_assign(
                                                "ld_eff_m",
                                                block_m,
                                                locals().get("ld_eff_m", _BUILDER_MISSING),
                                            )
                                            with T.If(ld_m_local == ld_nmb - 1), T.Then():
                                                ld_eff_m = _builder_assign(
                                                    "ld_eff_m",
                                                    _uceil(
                                                        ld_psum
                                                        - (ld_m_local + _udiv(ld_last, block_m))
                                                        * block_m,
                                                        UMMA_STEP_N,
                                                    )
                                                    * UMMA_STEP_N,
                                                    locals().get("ld_eff_m", _BUILDER_MISSING),
                                                )
                                        if cta_group > 1:
                                            if is_multicast_on_a:
                                                m_idx = _builder_assign(
                                                    "m_idx",
                                                    m_idx
                                                    + cta_in_cluster
                                                    * (
                                                        _udiv(ld_eff_m, cta_group)
                                                        if use_effective_m
                                                        else load_block_m
                                                    ),
                                                    locals().get("m_idx", _BUILDER_MISSING),
                                                )
                                            else:
                                                n_idx = _builder_assign(
                                                    "n_idx",
                                                    n_idx + cta_in_cluster * load_block_n,
                                                    locals().get("n_idx", _BUILDER_MISSING),
                                                )
                                        ld_k = _builder_alloc_scalar("ld_k", "int32")
                                        ld_k = _builder_assign(
                                            "ld_k", 0, locals().get("ld_k", _BUILDER_MISSING)
                                        )
                                        with T.While(ld_k < ld_kblocks):
                                            ld_wait = _builder_alloc_scalar("ld_wait", "uint32")
                                            ld_wait = _builder_assign(
                                                "ld_wait",
                                                T.uint32(0),
                                                locals().get("ld_wait", _BUILDER_MISSING),
                                            )
                                            with T.While(ld_wait == T.uint32(0)):
                                                _builder_emit(
                                                    T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                                        ld_wait,
                                                        barriers.ptr_to([empty_base + ld_stage]),
                                                        T.cast(
                                                            T.bitwise_xor(ld_phase, 1), "uint32"
                                                        ),
                                                        T.uint32(TRY_WAIT_TICKS),
                                                    )
                                                )
                                            k_b = _builder_alloc_scalar("k_b", "int32")
                                            k_a = _builder_alloc_scalar("k_a", "int32")
                                            k_b = _builder_assign(
                                                "k_b",
                                                k_b_idx + ld_k * block_k,
                                                locals().get("k_b", _BUILDER_MISSING),
                                            )
                                            k_a = _builder_assign(
                                                "k_a",
                                                k_a_offset + ld_k * block_k,
                                                locals().get("k_a", _BUILDER_MISSING),
                                            )
                                            with T.unroll(0, num_a_atoms) as i:
                                                IRBuilder.name("i", i)
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx[load_chain](
                                                            smem_a.ptr_to([ld_stage, 0, i * a_atom])
                                                            if major_a_is_k
                                                            else smem_a.ptr_to([ld_stage, 0, 0]),
                                                            T.address_of(tensor_map_a),
                                                            *_tma_coords(
                                                                (k_a + i * a_atom, m_idx)
                                                                if major_a_is_k
                                                                else (m_idx + i * a_atom, k_a),
                                                                ld_grp,
                                                            ),
                                                            barriers.ptr_to([full_base + ld_stage]),
                                                            T.uint64(EVICT_NORMAL),
                                                        )
                                                    )
                                                )
                                            with T.unroll(0, num_b_atoms) as i:
                                                IRBuilder.name("i", i)
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx[load_chain](
                                                            smem_b.ptr_to([ld_stage, 0, i * b_atom])
                                                            if major_b_is_k
                                                            else smem_b.ptr_to([ld_stage, 0, 0]),
                                                            T.address_of(tensor_map_b),
                                                            *_tma_coords(
                                                                (k_b + i * b_atom, n_idx)
                                                                if major_b_is_k
                                                                else (n_idx + i * b_atom, k_b),
                                                                ld_grp,
                                                            ),
                                                            barriers.ptr_to([full_base + ld_stage]),
                                                            T.uint64(EVICT_NORMAL),
                                                        )
                                                    )
                                                )
                                            arrival = _builder_alloc_scalar("arrival", "int32")
                                            arrival = _builder_assign(
                                                "arrival",
                                                arrival_bytes_ab,
                                                locals().get("arrival", _BUILDER_MISSING),
                                            )
                                            with T.If(T.EQ(ld_k % sfa_stages_per_load, 0)):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.evaluate(
                                                            T.ptx[sf_load_chain](
                                                                smem_sfa.ptr_to([ld_stage, 0]),
                                                                T.address_of(tensor_map_sfa),
                                                                T.cast(sfa_mn, "int32"),
                                                                T.cast(
                                                                    sfa_k_offset
                                                                    + ld_k // sfa_stages_per_load,
                                                                    "int32",
                                                                ),
                                                                barriers.ptr_to(
                                                                    [full_base + ld_stage]
                                                                ),
                                                                T.uint64(EVICT_NORMAL),
                                                            )
                                                        )
                                                    )
                                                    arrival = _builder_assign(
                                                        "arrival",
                                                        arrival + block_m * 4,
                                                        locals().get("arrival", _BUILDER_MISSING),
                                                    )
                                            with T.If(T.EQ(ld_k % sfb_stages_per_load, 0)):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.evaluate(
                                                            T.ptx[sf_load_chain](
                                                                smem_sfb.ptr_to([ld_stage, 0]),
                                                                T.address_of(tensor_map_sfb),
                                                                T.cast(sfb_mn, "int32"),
                                                                T.cast(
                                                                    sfb_k_offset
                                                                    + ld_k // sfb_stages_per_load,
                                                                    "int32",
                                                                ),
                                                                barriers.ptr_to(
                                                                    [full_base + ld_stage]
                                                                ),
                                                                T.uint64(EVICT_NORMAL),
                                                            )
                                                        )
                                                    )
                                                    arrival = _builder_assign(
                                                        "arrival",
                                                        arrival + block_n * 4,
                                                        locals().get("arrival", _BUILDER_MISSING),
                                                    )
                                            _builder_emit(
                                                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                                    barriers.ptr_to([full_base + ld_stage]),
                                                    T.cast(arrival, "uint32"),
                                                )
                                            )
                                            ld_k = _builder_assign(
                                                "ld_k",
                                                ld_k + 1,
                                                locals().get("ld_k", _BUILDER_MISSING),
                                            )
                                            ld_stage = _builder_assign(
                                                "ld_stage",
                                                T.Select(ld_stage == stages - 1, 0, ld_stage + 1),
                                                locals().get("ld_stage", _BUILDER_MISSING),
                                            )
                                            ld_phase = _builder_assign(
                                                "ld_phase",
                                                T.bitwise_xor(
                                                    ld_phase, T.cast(ld_stage == 0, "int32")
                                                ),
                                                locals().get("ld_phase", _BUILDER_MISSING),
                                            )
                                ld_it = _builder_assign(
                                    "ld_it", ld_it + 1, locals().get("ld_it", _BUILDER_MISSING)
                                )
                with T.Else():
                    with T.If(warp == 1):
                        with T.Then():
                            with T.If(is_leader_cta):
                                with T.Then():
                                    desc_a = _builder_alloc_scalar("desc_a", "uint64")
                                    desc_b = _builder_alloc_scalar("desc_b", "uint64")
                                    desc_sf = _builder_alloc_scalar("desc_sf", "uint64")
                                    desc_i = _builder_alloc_scalar("desc_i", "uint32")
                                    a_smem_u32 = _builder_alloc_scalar("a_smem_u32", "uint32")
                                    b_smem_u32 = _builder_alloc_scalar("b_smem_u32", "uint32")
                                    sfa_smem_u32 = _builder_alloc_scalar("sfa_smem_u32", "uint32")
                                    sfb_smem_u32 = _builder_alloc_scalar("sfb_smem_u32", "uint32")
                                    a_smem_u32 = _builder_assign(
                                        "a_smem_u32",
                                        T.cuda.cvta_generic_to_shared(smem_a.ptr_to([0, 0, 0])),
                                        locals().get("a_smem_u32", _BUILDER_MISSING),
                                    )
                                    b_smem_u32 = _builder_assign(
                                        "b_smem_u32",
                                        T.cuda.cvta_generic_to_shared(smem_b.ptr_to([0, 0, 0])),
                                        locals().get("b_smem_u32", _BUILDER_MISSING),
                                    )
                                    sfa_smem_u32 = _builder_assign(
                                        "sfa_smem_u32",
                                        T.cuda.cvta_generic_to_shared(smem_sfa.ptr_to([0, 0])),
                                        locals().get("sfa_smem_u32", _BUILDER_MISSING),
                                    )
                                    sfb_smem_u32 = _builder_assign(
                                        "sfb_smem_u32",
                                        T.cuda.cvta_generic_to_shared(smem_sfb.ptr_to([0, 0])),
                                        locals().get("sfb_smem_u32", _BUILDER_MISSING),
                                    )
                                    desc_a = _builder_assign(
                                        "desc_a",
                                        _with_smem_addr(_u64_const(A_DESC_BASE), a_smem_u32),
                                        locals().get("desc_a", _BUILDER_MISSING),
                                    )
                                    desc_b = _builder_assign(
                                        "desc_b",
                                        _with_smem_addr(_u64_const(B_DESC_BASE), b_smem_u32),
                                        locals().get("desc_b", _BUILDER_MISSING),
                                    )
                                    desc_sf = _builder_assign(
                                        "desc_sf",
                                        _with_smem_addr(_u64_const(SF_DESC_BASE), sfa_smem_u32),
                                        locals().get("desc_sf", _BUILDER_MISSING),
                                    )
                                    desc_i = _builder_assign(
                                        "desc_i",
                                        T.uint32(INSTR_DESC),
                                        locals().get("desc_i", _BUILDER_MISSING),
                                    )
                                    a_desc_lo = _builder_alloc_scalar("a_desc_lo", "uint32")
                                    b_desc_lo = _builder_alloc_scalar("b_desc_lo", "uint32")
                                    a_desc_lo = _builder_assign(
                                        "a_desc_lo",
                                        T.Select(
                                            lane_idx < stages,
                                            T.cast(
                                                T.bitwise_and(desc_a, T.uint64(4294967295)),
                                                "uint32",
                                            )
                                            + T.cast(
                                                lane_idx * (a_bytes_per_stage // 16), "uint32"
                                            ),
                                            T.uint32(0),
                                        ),
                                        locals().get("a_desc_lo", _BUILDER_MISSING),
                                    )
                                    b_desc_lo = _builder_assign(
                                        "b_desc_lo",
                                        T.Select(
                                            lane_idx < stages,
                                            T.cast(
                                                T.bitwise_and(desc_b, T.uint64(4294967295)),
                                                "uint32",
                                            )
                                            + T.cast(
                                                lane_idx * (b_bytes_per_stage // 16), "uint32"
                                            ),
                                            T.uint32(0),
                                        ),
                                        locals().get("b_desc_lo", _BUILDER_MISSING),
                                    )
                                    mma_stage = _builder_alloc_scalar("mma_stage", "int32")
                                    mma_phase = _builder_alloc_scalar("mma_phase", "int32")
                                    mma_iter = _builder_alloc_scalar("mma_iter", "int32")
                                    mma_stage = _builder_assign(
                                        "mma_stage", 0, locals().get("mma_stage", _BUILDER_MISSING)
                                    )
                                    mma_phase = _builder_assign(
                                        "mma_phase", 0, locals().get("mma_phase", _BUILDER_MISSING)
                                    )
                                    mma_it = _builder_alloc_scalar("mma_it", "int32")
                                    mma_valid = _builder_alloc_scalar("mma_valid", "int32")
                                    mma_grp = _builder_alloc_scalar("mma_grp", "int32")
                                    mma_cum = _builder_alloc_scalar("mma_cum", "int32")
                                    mma_nmb = _builder_alloc_scalar("mma_nmb", "int32")
                                    mma_last = _builder_alloc_scalar("mma_last", "int32")
                                    mma_psum = _builder_alloc_scalar("mma_psum", "int32")
                                    mma_nb = _builder_alloc_scalar("mma_nb", "int32")
                                    mma_nxt = _builder_alloc_scalar("mma_nxt", "int32")
                                    mma_nxtk = _builder_alloc_scalar("mma_nxtk", "int32")
                                    mma_kblocks = _builder_alloc_scalar("mma_kblocks", "int32")
                                    mma_vgrp = _builder_alloc_scalar("mma_vgrp", "int32")
                                    mma_kend = _builder_alloc_scalar("mma_kend", "int32")
                                    mma_elected = _builder_alloc_scalar("mma_elected", "uint32")
                                    mma_elected_lane = _builder_alloc_scalar(
                                        "mma_elected_lane", "uint32"
                                    )
                                    _builder_emit(
                                        T.ptx.elect_sync(
                                            mma_elected_lane, mma_elected, T.uint32(4294967295)
                                        )
                                    )
                                    mma_it = _builder_assign(
                                        "mma_it", 0, locals().get("mma_it", _BUILDER_MISSING)
                                    )
                                    mma_valid = _builder_assign(
                                        "mma_valid", 1, locals().get("mma_valid", _BUILDER_MISSING)
                                    )
                                    mma_grp = _builder_assign(
                                        "mma_grp", 0, locals().get("mma_grp", _BUILDER_MISSING)
                                    )
                                    mma_cum = _builder_assign(
                                        "mma_cum", 0, locals().get("mma_cum", _BUILDER_MISSING)
                                    )
                                    mma_last = _builder_assign(
                                        "mma_last", 0, locals().get("mma_last", _BUILDER_MISSING)
                                    )
                                    mma_vgrp = _builder_assign(
                                        "mma_vgrp", 0, locals().get("mma_vgrp", _BUILDER_MISSING)
                                    )
                                    mma_kend = _builder_assign(
                                        "mma_kend", 0, locals().get("mma_kend", _BUILDER_MISSING)
                                    )
                                    mma_nxt = _builder_assign(
                                        "mma_nxt", 0, locals().get("mma_nxt", _BUILDER_MISSING)
                                    )
                                    mma_nxtk = _builder_assign(
                                        "mma_nxtk", 0, locals().get("mma_nxtk", _BUILDER_MISSING)
                                    )
                                    mma_kblocks = _builder_assign(
                                        "mma_kblocks",
                                        0,
                                        locals().get("mma_kblocks", _BUILDER_MISSING),
                                    )
                                    if is_k_grouped:
                                        mma_psum = _builder_assign(
                                            "mma_psum",
                                            0,
                                            locals().get("mma_psum", _BUILDER_MISSING),
                                        )
                                        mma_nmb = _builder_assign(
                                            "mma_nmb",
                                            num_m_blocks,
                                            locals().get("mma_nmb", _BUILDER_MISSING),
                                        )
                                        if is_k_grouped_psum:
                                            with T.While(mma_grp < num_groups):
                                                mma_nxtk = _builder_assign(
                                                    "mma_nxtk",
                                                    grouped_layout[mma_grp],
                                                    locals().get("mma_nxtk", _BUILDER_MISSING),
                                                )
                                                mma_last = _builder_assign(
                                                    "mma_last",
                                                    _uceil(mma_kend, k_alignment) * k_alignment,
                                                    locals().get("mma_last", _BUILDER_MISSING),
                                                )
                                                mma_psum = _builder_assign(
                                                    "mma_psum",
                                                    mma_nxtk - mma_last,
                                                    locals().get("mma_psum", _BUILDER_MISSING),
                                                )
                                                mma_kend = _builder_assign(
                                                    "mma_kend",
                                                    mma_nxtk,
                                                    locals().get("mma_kend", _BUILDER_MISSING),
                                                )
                                                with T.If(mma_psum > 0):
                                                    with T.Then():
                                                        T.evaluate(T.break_loop())
                                                mma_grp = _builder_assign(
                                                    "mma_grp",
                                                    mma_grp + 1,
                                                    locals().get("mma_grp", _BUILDER_MISSING),
                                                )
                                        else:
                                            with T.While(mma_grp < num_groups):
                                                mma_psum = _builder_assign(
                                                    "mma_psum",
                                                    grouped_layout[mma_grp],
                                                    locals().get("mma_psum", _BUILDER_MISSING),
                                                )
                                                with T.If(mma_psum > 0):
                                                    with T.Then():
                                                        T.evaluate(T.break_loop())
                                                mma_grp = _builder_assign(
                                                    "mma_grp",
                                                    mma_grp + 1,
                                                    locals().get("mma_grp", _BUILDER_MISSING),
                                                )
                                            mma_nxt = _builder_assign(
                                                "mma_nxt",
                                                mma_grp + 1,
                                                locals().get("mma_nxt", _BUILDER_MISSING),
                                            )
                                            with T.While(mma_nxt < num_groups):
                                                mma_nxtk = _builder_assign(
                                                    "mma_nxtk",
                                                    grouped_layout[mma_nxt],
                                                    locals().get("mma_nxtk", _BUILDER_MISSING),
                                                )
                                                with T.If(mma_nxtk > 0):
                                                    with T.Then():
                                                        T.evaluate(T.break_loop())
                                                mma_nxt = _builder_assign(
                                                    "mma_nxt",
                                                    mma_nxt + 1,
                                                    locals().get("mma_nxt", _BUILDER_MISSING),
                                                )
                                    elif is_m_grouped_psum:
                                        mma_psum = _builder_assign(
                                            "mma_psum",
                                            grouped_layout[0],
                                            locals().get("mma_psum", _BUILDER_MISSING),
                                        )
                                        mma_nmb = _builder_assign(
                                            "mma_nmb",
                                            _uceil(mma_psum, block_m),
                                            locals().get("mma_nmb", _BUILDER_MISSING),
                                        )
                                    else:
                                        mma_psum = _builder_assign(
                                            "mma_psum",
                                            0,
                                            locals().get("mma_psum", _BUILDER_MISSING),
                                        )
                                        mma_nmb = _builder_assign(
                                            "mma_nmb",
                                            num_m_blocks,
                                            locals().get("mma_nmb", _BUILDER_MISSING),
                                        )
                                    mma_iter = _builder_assign(
                                        "mma_iter", 0, locals().get("mma_iter", _BUILDER_MISSING)
                                    )
                                    with T.While(mma_valid == 1):
                                        mma_nb = _builder_assign(
                                            "mma_nb",
                                            mma_it * num_sms + sm_idx,
                                            locals().get("mma_nb", _BUILDER_MISSING),
                                        )
                                        mma_done = _builder_alloc_scalar("mma_done", "int32")
                                        mma_done = _builder_assign(
                                            "mma_done",
                                            0,
                                            locals().get("mma_done", _BUILDER_MISSING),
                                        )
                                        if is_m_grouped_masked:
                                            with T.While(mma_done == 0):
                                                with T.If(mma_grp == num_groups):
                                                    with T.Then():
                                                        mma_valid = _builder_assign(
                                                            "mma_valid",
                                                            0,
                                                            locals().get(
                                                                "mma_valid", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        mma_done = _builder_assign(
                                                            "mma_done",
                                                            1,
                                                            locals().get(
                                                                "mma_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        mma_nmb = _builder_assign(
                                                            "mma_nmb",
                                                            _uceil(
                                                                grouped_layout[mma_grp], block_m
                                                            ),
                                                            locals().get(
                                                                "mma_nmb", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(
                                                            mma_nb
                                                            < (mma_cum + mma_nmb) * num_n_blocks
                                                        ):
                                                            with T.Then():
                                                                mma_done = _builder_assign(
                                                                    "mma_done",
                                                                    1,
                                                                    locals().get(
                                                                        "mma_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                mma_cum = _builder_assign(
                                                                    "mma_cum",
                                                                    mma_cum + mma_nmb,
                                                                    locals().get(
                                                                        "mma_cum", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                mma_grp = _builder_assign(
                                                                    "mma_grp",
                                                                    mma_grp + 1,
                                                                    locals().get(
                                                                        "mma_grp", _BUILDER_MISSING
                                                                    ),
                                                                )
                                        elif is_m_grouped_psum:
                                            with T.While(mma_done == 0):
                                                with T.If(
                                                    mma_nb < (mma_cum + mma_nmb) * num_n_blocks
                                                ):
                                                    with T.Then():
                                                        mma_done = _builder_assign(
                                                            "mma_done",
                                                            1,
                                                            locals().get(
                                                                "mma_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        mma_grp = _builder_assign(
                                                            "mma_grp",
                                                            mma_grp + 1,
                                                            locals().get(
                                                                "mma_grp", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(mma_grp == num_groups):
                                                            with T.Then():
                                                                mma_valid = _builder_assign(
                                                                    "mma_valid",
                                                                    0,
                                                                    locals().get(
                                                                        "mma_valid",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                                mma_done = _builder_assign(
                                                                    "mma_done",
                                                                    1,
                                                                    locals().get(
                                                                        "mma_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                mma_last = _builder_assign(
                                                                    "mma_last",
                                                                    _uceil(mma_psum, block_m)
                                                                    * block_m,
                                                                    locals().get(
                                                                        "mma_last", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                mma_psum = _builder_assign(
                                                                    "mma_psum",
                                                                    grouped_layout[mma_grp],
                                                                    locals().get(
                                                                        "mma_psum", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                mma_cum = _builder_assign(
                                                                    "mma_cum",
                                                                    mma_cum + mma_nmb,
                                                                    locals().get(
                                                                        "mma_cum", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                mma_nmb = _builder_assign(
                                                                    "mma_nmb",
                                                                    _uceil(
                                                                        mma_psum - mma_last, block_m
                                                                    ),
                                                                    locals().get(
                                                                        "mma_nmb", _BUILDER_MISSING
                                                                    ),
                                                                )
                                        elif is_k_grouped:
                                            with T.While(mma_done == 0):
                                                with T.If(mma_grp == num_groups):
                                                    with T.Then():
                                                        mma_valid = _builder_assign(
                                                            "mma_valid",
                                                            0,
                                                            locals().get(
                                                                "mma_valid", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        mma_done = _builder_assign(
                                                            "mma_done",
                                                            1,
                                                            locals().get(
                                                                "mma_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        with T.If(
                                                            mma_nb < (mma_vgrp + 1) * num_blocks
                                                        ):
                                                            with T.Then():
                                                                mma_done = _builder_assign(
                                                                    "mma_done",
                                                                    1,
                                                                    locals().get(
                                                                        "mma_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                mma_vgrp = _builder_assign(
                                                                    "mma_vgrp",
                                                                    mma_vgrp + 1,
                                                                    locals().get(
                                                                        "mma_vgrp", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                if is_k_grouped_psum:
                                                                    mma_grp = _builder_assign(
                                                                        "mma_grp",
                                                                        mma_grp + 1,
                                                                        locals().get(
                                                                            "mma_grp",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    with T.While(
                                                                        mma_grp < num_groups
                                                                    ):
                                                                        mma_nxtk = _builder_assign(
                                                                            "mma_nxtk",
                                                                            grouped_layout[mma_grp],
                                                                            locals().get(
                                                                                "mma_nxtk",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        mma_last = _builder_assign(
                                                                            "mma_last",
                                                                            _uceil(
                                                                                mma_kend,
                                                                                k_alignment,
                                                                            )
                                                                            * k_alignment,
                                                                            locals().get(
                                                                                "mma_last",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        mma_psum = _builder_assign(
                                                                            "mma_psum",
                                                                            mma_nxtk - mma_last,
                                                                            locals().get(
                                                                                "mma_psum",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        mma_kend = _builder_assign(
                                                                            "mma_kend",
                                                                            mma_nxtk,
                                                                            locals().get(
                                                                                "mma_kend",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        with T.If(mma_psum > 0):
                                                                            with T.Then():
                                                                                T.evaluate(
                                                                                    T.break_loop()
                                                                                )
                                                                        mma_grp = _builder_assign(
                                                                            "mma_grp",
                                                                            mma_grp + 1,
                                                                            locals().get(
                                                                                "mma_grp",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                else:
                                                                    mma_last = _builder_assign(
                                                                        "mma_last",
                                                                        mma_last + mma_psum,
                                                                        locals().get(
                                                                            "mma_last",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    mma_grp = _builder_assign(
                                                                        "mma_grp",
                                                                        mma_nxt,
                                                                        locals().get(
                                                                            "mma_grp",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    mma_nxt = _builder_assign(
                                                                        "mma_nxt",
                                                                        mma_nxt + 1,
                                                                        locals().get(
                                                                            "mma_nxt",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    mma_psum = _builder_assign(
                                                                        "mma_psum",
                                                                        mma_nxtk,
                                                                        locals().get(
                                                                            "mma_psum",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    with T.While(
                                                                        mma_nxt < num_groups
                                                                    ):
                                                                        mma_nxtk = _builder_assign(
                                                                            "mma_nxtk",
                                                                            grouped_layout[mma_nxt],
                                                                            locals().get(
                                                                                "mma_nxtk",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        with T.If(mma_nxtk > 0):
                                                                            with T.Then():
                                                                                T.evaluate(
                                                                                    T.break_loop()
                                                                                )
                                                                        mma_nxt = _builder_assign(
                                                                            "mma_nxt",
                                                                            mma_nxt + 1,
                                                                            locals().get(
                                                                                "mma_nxt",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                            mma_cum = _builder_assign(
                                                "mma_cum",
                                                mma_vgrp * num_m_blocks,
                                                locals().get("mma_cum", _BUILDER_MISSING),
                                            )
                                        elif is_batched:
                                            with T.If(mma_nb >= num_blocks * num_groups):
                                                with T.Then():
                                                    mma_valid = _builder_assign(
                                                        "mma_valid",
                                                        0,
                                                        locals().get("mma_valid", _BUILDER_MISSING),
                                                    )
                                                with T.Else():
                                                    mma_grp = _builder_assign(
                                                        "mma_grp",
                                                        mma_nb // num_blocks,
                                                        locals().get("mma_grp", _BUILDER_MISSING),
                                                    )
                                                    mma_cum = _builder_assign(
                                                        "mma_cum",
                                                        mma_grp * num_m_blocks,
                                                        locals().get("mma_cum", _BUILDER_MISSING),
                                                    )
                                                    mma_nmb = _builder_assign(
                                                        "mma_nmb",
                                                        num_m_blocks,
                                                        locals().get("mma_nmb", _BUILDER_MISSING),
                                                    )
                                        if not (
                                            is_m_grouped_masked
                                            or is_m_grouped_psum
                                            or is_k_grouped
                                            or is_batched
                                        ):
                                            with T.If(mma_nb >= num_blocks):
                                                with T.Then():
                                                    mma_valid = _builder_assign(
                                                        "mma_valid",
                                                        0,
                                                        locals().get("mma_valid", _BUILDER_MISSING),
                                                    )
                                        with T.If(mma_valid == 1):
                                            with T.Then():
                                                if use_effective_m:
                                                    mma_eff_m = _builder_alloc_scalar(
                                                        "mma_eff_m", "int32"
                                                    )
                                                    mma_m_local = _builder_alloc_scalar(
                                                        "mma_m_local", "int32"
                                                    )
                                                    mma_m_local = _builder_assign(
                                                        "mma_m_local",
                                                        _swizzled(
                                                            mma_nb - mma_cum * num_n_blocks,
                                                            mma_nmb,
                                                            num_n_blocks,
                                                        )[0],
                                                        locals().get(
                                                            "mma_m_local", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    mma_eff_m = _builder_assign(
                                                        "mma_eff_m",
                                                        block_m,
                                                        locals().get("mma_eff_m", _BUILDER_MISSING),
                                                    )
                                                    with T.If(mma_m_local == mma_nmb - 1), T.Then():
                                                        mma_eff_m = _builder_assign(
                                                            "mma_eff_m",
                                                            _uceil(
                                                                mma_psum
                                                                - (
                                                                    mma_m_local
                                                                    + _udiv(mma_last, block_m)
                                                                )
                                                                * block_m,
                                                                UMMA_STEP_N,
                                                            )
                                                            * UMMA_STEP_N,
                                                            locals().get(
                                                                "mma_eff_m", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    desc_i = _builder_assign(
                                                        "desc_i",
                                                        T.bitwise_or(
                                                            T.bitwise_and(
                                                                desc_i, T.uint32(UMMA_N_FIELD_MASK)
                                                            ),
                                                            T.shift_left(
                                                                T.cast(mma_eff_m // 8, "uint32"),
                                                                T.uint32(17),
                                                            ),
                                                        ),
                                                        locals().get("desc_i", _BUILDER_MISSING),
                                                    )
                                                mma_kblocks = _builder_assign(
                                                    "mma_kblocks",
                                                    _uceil(mma_psum, block_k)
                                                    if is_k_grouped
                                                    else num_k_blocks,
                                                    locals().get("mma_kblocks", _BUILDER_MISSING),
                                                )
                                                accum_stage = _builder_alloc_scalar(
                                                    "accum_stage", "int32"
                                                )
                                                accum_phase = _builder_alloc_scalar(
                                                    "accum_phase", "int32"
                                                )
                                                accum_stage = _builder_assign(
                                                    "accum_stage",
                                                    mma_iter % NUM_EPILOGUE_STAGES,
                                                    locals().get("accum_stage", _BUILDER_MISSING),
                                                )
                                                accum_phase = _builder_assign(
                                                    "accum_phase",
                                                    T.bitwise_and(
                                                        mma_iter // NUM_EPILOGUE_STAGES, 1
                                                    ),
                                                    locals().get("accum_phase", _BUILDER_MISSING),
                                                )
                                                mma_tmem_wait = _builder_alloc_scalar(
                                                    "mma_tmem_wait", "uint32"
                                                )
                                                mma_tmem_wait = _builder_assign(
                                                    "mma_tmem_wait",
                                                    T.uint32(0),
                                                    locals().get("mma_tmem_wait", _BUILDER_MISSING),
                                                )
                                                with T.While(mma_tmem_wait == T.uint32(0)):
                                                    _builder_emit(
                                                        T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                                            mma_tmem_wait,
                                                            barriers.ptr_to(
                                                                [tmem_empty_base + accum_stage]
                                                            ),
                                                            T.cast(
                                                                T.bitwise_xor(accum_phase, 1),
                                                                "uint32",
                                                            ),
                                                            T.uint32(TRY_WAIT_TICKS),
                                                        )
                                                    )
                                                _builder_emit(
                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                )
                                                mma_k = _builder_alloc_scalar("mma_k", "int32")
                                                mma_k = _builder_assign(
                                                    "mma_k",
                                                    0,
                                                    locals().get("mma_k", _BUILDER_MISSING),
                                                )
                                                mma_k_rounded = _builder_alloc_scalar(
                                                    "mma_k_rounded", "int32"
                                                )
                                                mma_k_rounded = _builder_assign(
                                                    "mma_k_rounded",
                                                    (mma_kblocks + (MMA_K_UNROLL - 1))
                                                    // MMA_K_UNROLL
                                                    * MMA_K_UNROLL,
                                                    locals().get("mma_k_rounded", _BUILDER_MISSING),
                                                )
                                                with T.While(mma_k < mma_k_rounded):
                                                    with T.unroll(0, MMA_K_UNROLL) as u:
                                                        IRBuilder.name("u", u)
                                                        with T.If(mma_k < mma_kblocks), T.Then():
                                                            mma_sf_wait = _builder_alloc_scalar(
                                                                "mma_sf_wait", "uint32"
                                                            )
                                                            mma_sf_wait = _builder_assign(
                                                                "mma_sf_wait",
                                                                T.uint32(0),
                                                                locals().get(
                                                                    "mma_sf_wait", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            with T.While(
                                                                mma_sf_wait == T.uint32(0)
                                                            ):
                                                                _builder_emit(
                                                                    T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                                                        mma_sf_wait,
                                                                        barriers.ptr_to(
                                                                            [
                                                                                with_sf_base
                                                                                + mma_stage
                                                                            ]
                                                                        ),
                                                                        T.cast(mma_phase, "uint32"),
                                                                        T.uint32(TRY_WAIT_TICKS),
                                                                    )
                                                                )
                                                            _builder_emit(
                                                                T.ptx.tcgen05.fence__after_thread_sync()
                                                            )
                                                            a_base_lo = _builder_alloc_scalar(
                                                                "a_base_lo", "uint32"
                                                            )
                                                            b_base_lo = _builder_alloc_scalar(
                                                                "b_base_lo", "uint32"
                                                            )
                                                            _builder_emit(
                                                                T.ptx.shfl_sync.idx.b32(
                                                                    a_base_lo,
                                                                    a_desc_lo,
                                                                    T.cast(mma_stage, "uint32"),
                                                                    T.uint32(31),
                                                                    T.uint32(4294967295),
                                                                )
                                                            )
                                                            _builder_emit(
                                                                T.ptx.shfl_sync.idx.b32(
                                                                    b_base_lo,
                                                                    b_desc_lo,
                                                                    T.cast(mma_stage, "uint32"),
                                                                    T.uint32(31),
                                                                    T.uint32(4294967295),
                                                                )
                                                            )
                                                            with T.If(
                                                                T.EQ(u % sfa_stages_per_load, 0)
                                                            ):
                                                                with T.Then():
                                                                    with T.unroll(
                                                                        0, num_sfa_chunks
                                                                    ) as c:
                                                                        IRBuilder.name("c", c)
                                                                        desc_sf = _builder_assign(
                                                                            "desc_sf",
                                                                            _rebase(
                                                                                desc_sf,
                                                                                sfa_smem_u32
                                                                                + T.cast(
                                                                                    mma_stage
                                                                                    * (
                                                                                        sf_block_m
                                                                                        * 4
                                                                                    ),
                                                                                    "uint32",
                                                                                )
                                                                                + T.uint32(
                                                                                    c
                                                                                    * NUM_UTCCP_ALIGNED_ELEMS
                                                                                    * 4
                                                                                ),
                                                                            ),
                                                                            locals().get(
                                                                                "desc_sf",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        _builder_emit(
                                                                            T.evaluate(
                                                                                T.ptx[utccp_chain](
                                                                                    T.cast(
                                                                                        sfa_tmem.allocated_addr[
                                                                                            0
                                                                                        ]
                                                                                        + c * 4,
                                                                                        "uint32",
                                                                                    ),
                                                                                    desc_sf,
                                                                                    pred=T.cast(
                                                                                        mma_elected
                                                                                        == T.uint32(
                                                                                            1
                                                                                        ),
                                                                                        "bool",
                                                                                    ),
                                                                                )
                                                                            )
                                                                        )
                                                            with T.If(
                                                                T.EQ(u % sfb_stages_per_load, 0)
                                                            ):
                                                                with T.Then():
                                                                    with T.unroll(
                                                                        0, num_sfb_chunks
                                                                    ) as c:
                                                                        IRBuilder.name("c", c)
                                                                        desc_sf = _builder_assign(
                                                                            "desc_sf",
                                                                            _rebase(
                                                                                desc_sf,
                                                                                sfb_smem_u32
                                                                                + T.cast(
                                                                                    mma_stage
                                                                                    * (
                                                                                        sf_block_n
                                                                                        * 4
                                                                                    ),
                                                                                    "uint32",
                                                                                )
                                                                                + T.uint32(
                                                                                    c
                                                                                    * NUM_UTCCP_ALIGNED_ELEMS
                                                                                    * 4
                                                                                ),
                                                                            ),
                                                                            locals().get(
                                                                                "desc_sf",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        _builder_emit(
                                                                            T.evaluate(
                                                                                T.ptx[utccp_chain](
                                                                                    T.cast(
                                                                                        sfb_tmem.allocated_addr[
                                                                                            0
                                                                                        ]
                                                                                        + c * 4,
                                                                                        "uint32",
                                                                                    ),
                                                                                    desc_sf,
                                                                                    pred=T.cast(
                                                                                        mma_elected
                                                                                        == T.uint32(
                                                                                            1
                                                                                        ),
                                                                                        "bool",
                                                                                    ),
                                                                                )
                                                                            )
                                                                        )
                                                            with T.unroll(0, umma_k_steps) as ki:
                                                                IRBuilder.name("ki", ki)
                                                                with (
                                                                    T.If(
                                                                        T.Or(
                                                                            mma_k < mma_kblocks - 1,
                                                                            ki * UMMA_K
                                                                            < (
                                                                                mma_psum
                                                                                if is_k_grouped
                                                                                else eff_k
                                                                            )
                                                                            - mma_k * block_k,
                                                                        )
                                                                        if may_have_tail_k
                                                                        else ki < umma_k_steps
                                                                    ),
                                                                    T.Then(),
                                                                ):
                                                                    sfa_id = _builder_alloc_scalar(
                                                                        "sfa_id", "uint32"
                                                                    )
                                                                    sfb_id = _builder_alloc_scalar(
                                                                        "sfb_id", "uint32"
                                                                    )
                                                                    sfa_id = _builder_assign(
                                                                        "sfa_id",
                                                                        T.cast(ki, "uint32")
                                                                        if sfa_stages_per_load == 1
                                                                        else T.cast(
                                                                            u % sfa_stages_per_load,
                                                                            "uint32",
                                                                        ),
                                                                        locals().get(
                                                                            "sfa_id",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    sfb_id = _builder_assign(
                                                                        "sfb_id",
                                                                        T.cast(ki, "uint32")
                                                                        if sfb_stages_per_load == 1
                                                                        else T.cast(
                                                                            u % sfb_stages_per_load,
                                                                            "uint32",
                                                                        ),
                                                                        locals().get(
                                                                            "sfb_id",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    rt_desc = _builder_alloc_scalar(
                                                                        "rt_desc", "uint32"
                                                                    )
                                                                    rt_desc = _builder_assign(
                                                                        "rt_desc",
                                                                        _with_sf_id(
                                                                            desc_i,
                                                                            sfb_id
                                                                            if swap_ab
                                                                            else sfa_id,
                                                                            sfa_id
                                                                            if swap_ab
                                                                            else sfb_id,
                                                                        ),
                                                                        locals().get(
                                                                            "rt_desc",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    adv_a = _builder_alloc_scalar(
                                                                        "adv_a", "uint64"
                                                                    )
                                                                    adv_b = _builder_alloc_scalar(
                                                                        "adv_b", "uint64"
                                                                    )
                                                                    adv_a = _builder_assign(
                                                                        "adv_a",
                                                                        _advance_lo(
                                                                            desc_a,
                                                                            a_base_lo,
                                                                            ki * a_k_step_units,
                                                                        ),
                                                                        locals().get(
                                                                            "adv_a",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    adv_b = _builder_assign(
                                                                        "adv_b",
                                                                        _advance_lo(
                                                                            desc_b,
                                                                            b_base_lo,
                                                                            ki * b_k_step_units,
                                                                        ),
                                                                        locals().get(
                                                                            "adv_b",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx[mma_chain](
                                                                                T.cast(
                                                                                    accum_stage
                                                                                    * umma_n,
                                                                                    "uint32",
                                                                                ),
                                                                                adv_b
                                                                                if swap_ab
                                                                                else adv_a,
                                                                                adv_a
                                                                                if swap_ab
                                                                                else adv_b,
                                                                                rt_desc,
                                                                                T.cast(
                                                                                    (
                                                                                        sfb_tmem
                                                                                        if swap_ab
                                                                                        else sfa_tmem
                                                                                    ).allocated_addr[
                                                                                        0
                                                                                    ],
                                                                                    "uint32",
                                                                                ),
                                                                                T.cast(
                                                                                    (
                                                                                        sfa_tmem
                                                                                        if swap_ab
                                                                                        else sfb_tmem
                                                                                    ).allocated_addr[
                                                                                        0
                                                                                    ],
                                                                                    "uint32",
                                                                                ),
                                                                                T.Or(
                                                                                    ki > 0,
                                                                                    mma_k > 0,
                                                                                ),
                                                                                pred=T.cast(
                                                                                    mma_elected
                                                                                    == T.uint32(1),
                                                                                    "bool",
                                                                                ),
                                                                            )
                                                                        )
                                                                    )
                                                            _builder_emit(
                                                                T.ptx.bar.warp.sync(
                                                                    T.uint32(4294967295)
                                                                )
                                                            )
                                                            if cta_group > 1:
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx[commit_mc_chain](
                                                                            barriers.ptr_to(
                                                                                [
                                                                                    empty_base
                                                                                    + mma_stage
                                                                                ]
                                                                            ),
                                                                            T.uint16(
                                                                                (1 << cta_group) - 1
                                                                            ),
                                                                            pred=T.cast(
                                                                                mma_elected
                                                                                == T.uint32(1),
                                                                                "bool",
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                            else:
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx[commit_chain](
                                                                            barriers.ptr_to(
                                                                                [
                                                                                    empty_base
                                                                                    + mma_stage
                                                                                ]
                                                                            ),
                                                                            pred=T.cast(
                                                                                mma_elected
                                                                                == T.uint32(1),
                                                                                "bool",
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                            if cta_group > 1:
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx[commit_mc_chain](
                                                                            barriers.ptr_to(
                                                                                [
                                                                                    tmem_full_base
                                                                                    + accum_stage
                                                                                ]
                                                                            ),
                                                                            T.uint16(
                                                                                (1 << cta_group) - 1
                                                                            ),
                                                                            pred=T.cast(
                                                                                T.And(
                                                                                    mma_elected
                                                                                    == T.uint32(1),
                                                                                    mma_k
                                                                                    == mma_kblocks
                                                                                    - 1,
                                                                                ),
                                                                                "bool",
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                            else:
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx[commit_chain](
                                                                            barriers.ptr_to(
                                                                                [
                                                                                    tmem_full_base
                                                                                    + accum_stage
                                                                                ]
                                                                            ),
                                                                            pred=T.cast(
                                                                                T.And(
                                                                                    mma_elected
                                                                                    == T.uint32(1),
                                                                                    mma_k
                                                                                    == mma_kblocks
                                                                                    - 1,
                                                                                ),
                                                                                "bool",
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                            _builder_emit(
                                                                T.ptx.bar.warp.sync(
                                                                    T.uint32(4294967295)
                                                                )
                                                            )
                                                            mma_stage = _builder_assign(
                                                                "mma_stage",
                                                                T.Select(
                                                                    mma_stage == stages - 1,
                                                                    0,
                                                                    mma_stage + 1,
                                                                ),
                                                                locals().get(
                                                                    "mma_stage", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            mma_phase = _builder_assign(
                                                                "mma_phase",
                                                                T.bitwise_xor(
                                                                    mma_phase,
                                                                    T.cast(mma_stage == 0, "int32"),
                                                                ),
                                                                locals().get(
                                                                    "mma_phase", _BUILDER_MISSING
                                                                ),
                                                            )
                                                        mma_k = _builder_assign(
                                                            "mma_k",
                                                            mma_k + 1,
                                                            locals().get("mma_k", _BUILDER_MISSING),
                                                        )
                                                mma_iter = _builder_assign(
                                                    "mma_iter",
                                                    mma_iter + 1,
                                                    locals().get("mma_iter", _BUILDER_MISSING),
                                                )
                                        mma_it = _builder_assign(
                                            "mma_it",
                                            mma_it + 1,
                                            locals().get("mma_it", _BUILDER_MISSING),
                                        )
                                    if cta_group > 1:
                                        with T.If(mma_iter > 0):
                                            with T.Then():
                                                mma_end_wait = _builder_alloc_scalar(
                                                    "mma_end_wait", "uint32"
                                                )
                                                mma_end_wait = _builder_assign(
                                                    "mma_end_wait",
                                                    T.uint32(0),
                                                    locals().get("mma_end_wait", _BUILDER_MISSING),
                                                )
                                                with T.While(mma_end_wait == T.uint32(0)):
                                                    _builder_emit(
                                                        T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                                            mma_end_wait,
                                                            barriers.ptr_to(
                                                                [
                                                                    tmem_empty_base
                                                                    + (mma_iter - 1)
                                                                    % NUM_EPILOGUE_STAGES
                                                                ]
                                                            ),
                                                            T.cast(
                                                                T.bitwise_and(
                                                                    (mma_iter - 1)
                                                                    // NUM_EPILOGUE_STAGES,
                                                                    1,
                                                                ),
                                                                "uint32",
                                                            ),
                                                            T.uint32(TRY_WAIT_TICKS),
                                                        )
                                                    )
                        with T.Else():
                            with T.If(warp == 2):
                                with T.Then():
                                    tr_stage = _builder_alloc_scalar("tr_stage", "int32")
                                    tr_phase = _builder_alloc_scalar("tr_phase", "int32")
                                    tr_stage = _builder_assign(
                                        "tr_stage", 0, locals().get("tr_stage", _BUILDER_MISSING)
                                    )
                                    tr_phase = _builder_assign(
                                        "tr_phase", 0, locals().get("tr_phase", _BUILDER_MISSING)
                                    )
                                    tr_it = _builder_alloc_scalar("tr_it", "int32")
                                    tr_valid = _builder_alloc_scalar("tr_valid", "int32")
                                    tr_grp = _builder_alloc_scalar("tr_grp", "int32")
                                    tr_cum = _builder_alloc_scalar("tr_cum", "int32")
                                    tr_nmb = _builder_alloc_scalar("tr_nmb", "int32")
                                    tr_last = _builder_alloc_scalar("tr_last", "int32")
                                    tr_psum = _builder_alloc_scalar("tr_psum", "int32")
                                    tr_nb = _builder_alloc_scalar("tr_nb", "int32")
                                    tr_nxt = _builder_alloc_scalar("tr_nxt", "int32")
                                    tr_nxtk = _builder_alloc_scalar("tr_nxtk", "int32")
                                    tr_kblocks = _builder_alloc_scalar("tr_kblocks", "int32")
                                    tr_vgrp = _builder_alloc_scalar("tr_vgrp", "int32")
                                    tr_kend = _builder_alloc_scalar("tr_kend", "int32")
                                    tr_it = _builder_assign(
                                        "tr_it", 0, locals().get("tr_it", _BUILDER_MISSING)
                                    )
                                    tr_valid = _builder_assign(
                                        "tr_valid", 1, locals().get("tr_valid", _BUILDER_MISSING)
                                    )
                                    tr_grp = _builder_assign(
                                        "tr_grp", 0, locals().get("tr_grp", _BUILDER_MISSING)
                                    )
                                    tr_cum = _builder_assign(
                                        "tr_cum", 0, locals().get("tr_cum", _BUILDER_MISSING)
                                    )
                                    tr_last = _builder_assign(
                                        "tr_last", 0, locals().get("tr_last", _BUILDER_MISSING)
                                    )
                                    tr_vgrp = _builder_assign(
                                        "tr_vgrp", 0, locals().get("tr_vgrp", _BUILDER_MISSING)
                                    )
                                    tr_kend = _builder_assign(
                                        "tr_kend", 0, locals().get("tr_kend", _BUILDER_MISSING)
                                    )
                                    tr_nxt = _builder_assign(
                                        "tr_nxt", 0, locals().get("tr_nxt", _BUILDER_MISSING)
                                    )
                                    tr_nxtk = _builder_assign(
                                        "tr_nxtk", 0, locals().get("tr_nxtk", _BUILDER_MISSING)
                                    )
                                    tr_kblocks = _builder_assign(
                                        "tr_kblocks",
                                        0,
                                        locals().get("tr_kblocks", _BUILDER_MISSING),
                                    )
                                    if is_k_grouped:
                                        tr_psum = _builder_assign(
                                            "tr_psum", 0, locals().get("tr_psum", _BUILDER_MISSING)
                                        )
                                        tr_nmb = _builder_assign(
                                            "tr_nmb",
                                            num_m_blocks,
                                            locals().get("tr_nmb", _BUILDER_MISSING),
                                        )
                                        if is_k_grouped_psum:
                                            with T.While(tr_grp < num_groups):
                                                tr_nxtk = _builder_assign(
                                                    "tr_nxtk",
                                                    grouped_layout[tr_grp],
                                                    locals().get("tr_nxtk", _BUILDER_MISSING),
                                                )
                                                tr_last = _builder_assign(
                                                    "tr_last",
                                                    _uceil(tr_kend, k_alignment) * k_alignment,
                                                    locals().get("tr_last", _BUILDER_MISSING),
                                                )
                                                tr_psum = _builder_assign(
                                                    "tr_psum",
                                                    tr_nxtk - tr_last,
                                                    locals().get("tr_psum", _BUILDER_MISSING),
                                                )
                                                tr_kend = _builder_assign(
                                                    "tr_kend",
                                                    tr_nxtk,
                                                    locals().get("tr_kend", _BUILDER_MISSING),
                                                )
                                                with T.If(tr_psum > 0):
                                                    with T.Then():
                                                        T.evaluate(T.break_loop())
                                                tr_grp = _builder_assign(
                                                    "tr_grp",
                                                    tr_grp + 1,
                                                    locals().get("tr_grp", _BUILDER_MISSING),
                                                )
                                        else:
                                            with T.While(tr_grp < num_groups):
                                                tr_psum = _builder_assign(
                                                    "tr_psum",
                                                    grouped_layout[tr_grp],
                                                    locals().get("tr_psum", _BUILDER_MISSING),
                                                )
                                                with T.If(tr_psum > 0):
                                                    with T.Then():
                                                        T.evaluate(T.break_loop())
                                                tr_grp = _builder_assign(
                                                    "tr_grp",
                                                    tr_grp + 1,
                                                    locals().get("tr_grp", _BUILDER_MISSING),
                                                )
                                            tr_nxt = _builder_assign(
                                                "tr_nxt",
                                                tr_grp + 1,
                                                locals().get("tr_nxt", _BUILDER_MISSING),
                                            )
                                            with T.While(tr_nxt < num_groups):
                                                tr_nxtk = _builder_assign(
                                                    "tr_nxtk",
                                                    grouped_layout[tr_nxt],
                                                    locals().get("tr_nxtk", _BUILDER_MISSING),
                                                )
                                                with T.If(tr_nxtk > 0):
                                                    with T.Then():
                                                        T.evaluate(T.break_loop())
                                                tr_nxt = _builder_assign(
                                                    "tr_nxt",
                                                    tr_nxt + 1,
                                                    locals().get("tr_nxt", _BUILDER_MISSING),
                                                )
                                    elif is_m_grouped_psum:
                                        tr_psum = _builder_assign(
                                            "tr_psum",
                                            grouped_layout[0],
                                            locals().get("tr_psum", _BUILDER_MISSING),
                                        )
                                        tr_nmb = _builder_assign(
                                            "tr_nmb",
                                            _uceil(tr_psum, block_m),
                                            locals().get("tr_nmb", _BUILDER_MISSING),
                                        )
                                    else:
                                        tr_psum = _builder_assign(
                                            "tr_psum", 0, locals().get("tr_psum", _BUILDER_MISSING)
                                        )
                                        tr_nmb = _builder_assign(
                                            "tr_nmb",
                                            num_m_blocks,
                                            locals().get("tr_nmb", _BUILDER_MISSING),
                                        )
                                    sf_vals = _builder_assign(
                                        "sf_vals",
                                        T.alloc_local((4,), "uint32"),
                                        locals().get("sf_vals", _BUILDER_MISSING),
                                    )
                                    with T.While(tr_valid == 1):
                                        tr_nb = _builder_assign(
                                            "tr_nb",
                                            tr_it * num_sms + sm_idx,
                                            locals().get("tr_nb", _BUILDER_MISSING),
                                        )
                                        tr_done = _builder_alloc_scalar("tr_done", "int32")
                                        tr_done = _builder_assign(
                                            "tr_done", 0, locals().get("tr_done", _BUILDER_MISSING)
                                        )
                                        if is_m_grouped_masked:
                                            with T.While(tr_done == 0):
                                                with T.If(tr_grp == num_groups):
                                                    with T.Then():
                                                        tr_valid = _builder_assign(
                                                            "tr_valid",
                                                            0,
                                                            locals().get(
                                                                "tr_valid", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        tr_done = _builder_assign(
                                                            "tr_done",
                                                            1,
                                                            locals().get(
                                                                "tr_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        tr_nmb = _builder_assign(
                                                            "tr_nmb",
                                                            _uceil(grouped_layout[tr_grp], block_m),
                                                            locals().get(
                                                                "tr_nmb", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(
                                                            tr_nb < (tr_cum + tr_nmb) * num_n_blocks
                                                        ):
                                                            with T.Then():
                                                                tr_done = _builder_assign(
                                                                    "tr_done",
                                                                    1,
                                                                    locals().get(
                                                                        "tr_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                tr_cum = _builder_assign(
                                                                    "tr_cum",
                                                                    tr_cum + tr_nmb,
                                                                    locals().get(
                                                                        "tr_cum", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                tr_grp = _builder_assign(
                                                                    "tr_grp",
                                                                    tr_grp + 1,
                                                                    locals().get(
                                                                        "tr_grp", _BUILDER_MISSING
                                                                    ),
                                                                )
                                        elif is_m_grouped_psum:
                                            with T.While(tr_done == 0):
                                                with T.If(tr_nb < (tr_cum + tr_nmb) * num_n_blocks):
                                                    with T.Then():
                                                        tr_done = _builder_assign(
                                                            "tr_done",
                                                            1,
                                                            locals().get(
                                                                "tr_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        tr_grp = _builder_assign(
                                                            "tr_grp",
                                                            tr_grp + 1,
                                                            locals().get(
                                                                "tr_grp", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(tr_grp == num_groups):
                                                            with T.Then():
                                                                tr_valid = _builder_assign(
                                                                    "tr_valid",
                                                                    0,
                                                                    locals().get(
                                                                        "tr_valid", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                tr_done = _builder_assign(
                                                                    "tr_done",
                                                                    1,
                                                                    locals().get(
                                                                        "tr_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                tr_last = _builder_assign(
                                                                    "tr_last",
                                                                    _uceil(tr_psum, block_m)
                                                                    * block_m,
                                                                    locals().get(
                                                                        "tr_last", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                tr_psum = _builder_assign(
                                                                    "tr_psum",
                                                                    grouped_layout[tr_grp],
                                                                    locals().get(
                                                                        "tr_psum", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                tr_cum = _builder_assign(
                                                                    "tr_cum",
                                                                    tr_cum + tr_nmb,
                                                                    locals().get(
                                                                        "tr_cum", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                tr_nmb = _builder_assign(
                                                                    "tr_nmb",
                                                                    _uceil(
                                                                        tr_psum - tr_last, block_m
                                                                    ),
                                                                    locals().get(
                                                                        "tr_nmb", _BUILDER_MISSING
                                                                    ),
                                                                )
                                        elif is_k_grouped:
                                            with T.While(tr_done == 0):
                                                with T.If(tr_grp == num_groups):
                                                    with T.Then():
                                                        tr_valid = _builder_assign(
                                                            "tr_valid",
                                                            0,
                                                            locals().get(
                                                                "tr_valid", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        tr_done = _builder_assign(
                                                            "tr_done",
                                                            1,
                                                            locals().get(
                                                                "tr_done", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    with T.Else():
                                                        with T.If(
                                                            tr_nb < (tr_vgrp + 1) * num_blocks
                                                        ):
                                                            with T.Then():
                                                                tr_done = _builder_assign(
                                                                    "tr_done",
                                                                    1,
                                                                    locals().get(
                                                                        "tr_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                tr_vgrp = _builder_assign(
                                                                    "tr_vgrp",
                                                                    tr_vgrp + 1,
                                                                    locals().get(
                                                                        "tr_vgrp", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                if is_k_grouped_psum:
                                                                    tr_grp = _builder_assign(
                                                                        "tr_grp",
                                                                        tr_grp + 1,
                                                                        locals().get(
                                                                            "tr_grp",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    with T.While(
                                                                        tr_grp < num_groups
                                                                    ):
                                                                        tr_nxtk = _builder_assign(
                                                                            "tr_nxtk",
                                                                            grouped_layout[tr_grp],
                                                                            locals().get(
                                                                                "tr_nxtk",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        tr_last = _builder_assign(
                                                                            "tr_last",
                                                                            _uceil(
                                                                                tr_kend, k_alignment
                                                                            )
                                                                            * k_alignment,
                                                                            locals().get(
                                                                                "tr_last",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        tr_psum = _builder_assign(
                                                                            "tr_psum",
                                                                            tr_nxtk - tr_last,
                                                                            locals().get(
                                                                                "tr_psum",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        tr_kend = _builder_assign(
                                                                            "tr_kend",
                                                                            tr_nxtk,
                                                                            locals().get(
                                                                                "tr_kend",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        with T.If(tr_psum > 0):
                                                                            with T.Then():
                                                                                T.evaluate(
                                                                                    T.break_loop()
                                                                                )
                                                                        tr_grp = _builder_assign(
                                                                            "tr_grp",
                                                                            tr_grp + 1,
                                                                            locals().get(
                                                                                "tr_grp",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                else:
                                                                    tr_last = _builder_assign(
                                                                        "tr_last",
                                                                        tr_last + tr_psum,
                                                                        locals().get(
                                                                            "tr_last",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    tr_grp = _builder_assign(
                                                                        "tr_grp",
                                                                        tr_nxt,
                                                                        locals().get(
                                                                            "tr_grp",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    tr_nxt = _builder_assign(
                                                                        "tr_nxt",
                                                                        tr_nxt + 1,
                                                                        locals().get(
                                                                            "tr_nxt",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    tr_psum = _builder_assign(
                                                                        "tr_psum",
                                                                        tr_nxtk,
                                                                        locals().get(
                                                                            "tr_psum",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    with T.While(
                                                                        tr_nxt < num_groups
                                                                    ):
                                                                        tr_nxtk = _builder_assign(
                                                                            "tr_nxtk",
                                                                            grouped_layout[tr_nxt],
                                                                            locals().get(
                                                                                "tr_nxtk",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        with T.If(tr_nxtk > 0):
                                                                            with T.Then():
                                                                                T.evaluate(
                                                                                    T.break_loop()
                                                                                )
                                                                        tr_nxt = _builder_assign(
                                                                            "tr_nxt",
                                                                            tr_nxt + 1,
                                                                            locals().get(
                                                                                "tr_nxt",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                            tr_cum = _builder_assign(
                                                "tr_cum",
                                                tr_vgrp * num_m_blocks,
                                                locals().get("tr_cum", _BUILDER_MISSING),
                                            )
                                        elif is_batched:
                                            with T.If(tr_nb >= num_blocks * num_groups):
                                                with T.Then():
                                                    tr_valid = _builder_assign(
                                                        "tr_valid",
                                                        0,
                                                        locals().get("tr_valid", _BUILDER_MISSING),
                                                    )
                                                with T.Else():
                                                    tr_grp = _builder_assign(
                                                        "tr_grp",
                                                        tr_nb // num_blocks,
                                                        locals().get("tr_grp", _BUILDER_MISSING),
                                                    )
                                                    tr_cum = _builder_assign(
                                                        "tr_cum",
                                                        tr_grp * num_m_blocks,
                                                        locals().get("tr_cum", _BUILDER_MISSING),
                                                    )
                                                    tr_nmb = _builder_assign(
                                                        "tr_nmb",
                                                        num_m_blocks,
                                                        locals().get("tr_nmb", _BUILDER_MISSING),
                                                    )
                                        if not (
                                            is_m_grouped_masked
                                            or is_m_grouped_psum
                                            or is_k_grouped
                                            or is_batched
                                        ):
                                            with T.If(tr_nb >= num_blocks):
                                                with T.Then():
                                                    tr_valid = _builder_assign(
                                                        "tr_valid",
                                                        0,
                                                        locals().get("tr_valid", _BUILDER_MISSING),
                                                    )
                                        with T.If(tr_valid == 1):
                                            with T.Then():
                                                tr_kblocks = _builder_assign(
                                                    "tr_kblocks",
                                                    _uceil(tr_psum, block_k)
                                                    if is_k_grouped
                                                    else num_k_blocks,
                                                    locals().get("tr_kblocks", _BUILDER_MISSING),
                                                )
                                                tr_k = _builder_alloc_scalar("tr_k", "int32")
                                                tr_k = _builder_assign(
                                                    "tr_k",
                                                    0,
                                                    locals().get("tr_k", _BUILDER_MISSING),
                                                )
                                                with T.While(tr_k < tr_kblocks):
                                                    tr_wait = _builder_alloc_scalar(
                                                        "tr_wait", "uint32"
                                                    )
                                                    tr_wait = _builder_assign(
                                                        "tr_wait",
                                                        T.uint32(0),
                                                        locals().get("tr_wait", _BUILDER_MISSING),
                                                    )
                                                    with T.While(tr_wait == T.uint32(0)):
                                                        _builder_emit(
                                                            T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                                                tr_wait,
                                                                barriers.ptr_to(
                                                                    [full_base + tr_stage]
                                                                ),
                                                                T.cast(tr_phase, "uint32"),
                                                                T.uint32(TRY_WAIT_TICKS),
                                                            )
                                                        )
                                                    # The prior logical task may still read this
                                                    # stage through tcgen05's async proxy. Complete
                                                    # that handoff before generic-proxy transpose
                                                    # stores reuse the same shared bytes.
                                                    _builder_emit(
                                                        T.ptx.fence.proxy.async_.shared__cta()
                                                    )
                                                    with T.If(T.EQ(tr_k % sfa_stages_per_load, 0)):
                                                        with T.Then():
                                                            with T.unroll(0, num_sfa_chunks) as c:
                                                                IRBuilder.name("c", c)
                                                                base = _builder_alloc_scalar(
                                                                    "base", "int32"
                                                                )
                                                                base = _builder_assign(
                                                                    "base",
                                                                    c * NUM_UTCCP_ALIGNED_ELEMS,
                                                                    locals().get(
                                                                        "base", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                with T.unroll(0, 4) as i:
                                                                    IRBuilder.name("i", i)
                                                                    T.buffer_store(
                                                                        sf_vals,
                                                                        smem_sfa[
                                                                            tr_stage,
                                                                            base
                                                                            + i * 32
                                                                            + lane_idx,
                                                                        ],
                                                                        [i],
                                                                    )
                                                                _builder_emit(
                                                                    T.ptx.bar.warp.sync(
                                                                        T.uint32(4294967295)
                                                                    )
                                                                )
                                                                with T.unroll(0, 4) as i:
                                                                    IRBuilder.name("i", i)
                                                                    T.buffer_store(
                                                                        smem_sfa,
                                                                        sf_vals[i],
                                                                        [
                                                                            tr_stage,
                                                                            base
                                                                            + lane_idx * 4
                                                                            + i,
                                                                        ],
                                                                    )
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx.fence.proxy.async_.shared__cta()
                                                                )
                                                            )
                                                    with T.If(T.EQ(tr_k % sfb_stages_per_load, 0)):
                                                        with T.Then():
                                                            with T.unroll(0, num_sfb_chunks) as c:
                                                                IRBuilder.name("c", c)
                                                                base = _builder_alloc_scalar(
                                                                    "base", "int32"
                                                                )
                                                                base = _builder_assign(
                                                                    "base",
                                                                    c * NUM_UTCCP_ALIGNED_ELEMS,
                                                                    locals().get(
                                                                        "base", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                with T.unroll(0, 4) as i:
                                                                    IRBuilder.name("i", i)
                                                                    T.buffer_store(
                                                                        sf_vals,
                                                                        smem_sfb[
                                                                            tr_stage,
                                                                            base
                                                                            + i * 32
                                                                            + lane_idx,
                                                                        ],
                                                                        [i],
                                                                    )
                                                                _builder_emit(
                                                                    T.ptx.bar.warp.sync(
                                                                        T.uint32(4294967295)
                                                                    )
                                                                )
                                                                with T.unroll(0, 4) as i:
                                                                    IRBuilder.name("i", i)
                                                                    T.buffer_store(
                                                                        smem_sfb,
                                                                        sf_vals[i],
                                                                        [
                                                                            tr_stage,
                                                                            base
                                                                            + lane_idx * 4
                                                                            + i,
                                                                        ],
                                                                    )
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx.fence.proxy.async_.shared__cta()
                                                                )
                                                            )
                                                    rem = _builder_assign(
                                                        "rem",
                                                        T.alloc_local((1,), "uint64"),
                                                        locals().get("rem", _BUILDER_MISSING),
                                                    )
                                                    _builder_emit(
                                                        T.ptx.mapa.shared__cluster.u64(
                                                            rem[0],
                                                            barriers.ptr_to(
                                                                [with_sf_base + tr_stage]
                                                            ),
                                                            T.uint32(0),
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.ptx.mbarrier.arrive.b64(
                                                            rem[0], T.uint32(1), pred=T.bool(True)
                                                        )
                                                    )
                                                    tr_k = _builder_assign(
                                                        "tr_k",
                                                        tr_k + 1,
                                                        locals().get("tr_k", _BUILDER_MISSING),
                                                    )
                                                    tr_stage = _builder_assign(
                                                        "tr_stage",
                                                        T.Select(
                                                            tr_stage == stages - 1, 0, tr_stage + 1
                                                        ),
                                                        locals().get("tr_stage", _BUILDER_MISSING),
                                                    )
                                                    tr_phase = _builder_assign(
                                                        "tr_phase",
                                                        T.bitwise_xor(
                                                            tr_phase, T.cast(tr_stage == 0, "int32")
                                                        ),
                                                        locals().get("tr_phase", _BUILDER_MISSING),
                                                    )
                                        tr_it = _builder_assign(
                                            "tr_it",
                                            tr_it + 1,
                                            locals().get("tr_it", _BUILDER_MISSING),
                                        )
                                with T.Else():
                                    with T.If(
                                        T.And(
                                            warp >= first_epilogue_warp,
                                            warp < first_epilogue_warp + num_store_warps,
                                        )
                                    ):
                                        with T.Then():
                                            ep_warp = _builder_alloc_scalar("ep_warp", "int32")
                                            ep_warp = _builder_assign(
                                                "ep_warp",
                                                warp - first_epilogue_warp,
                                                locals().get("ep_warp", _BUILDER_MISSING),
                                            )
                                            tma_stage = _builder_alloc_scalar("tma_stage", "int32")
                                            ep_it = _builder_alloc_scalar("ep_it", "int32")
                                            ep_valid = _builder_alloc_scalar("ep_valid", "int32")
                                            ep_grp = _builder_alloc_scalar("ep_grp", "int32")
                                            ep_cum = _builder_alloc_scalar("ep_cum", "int32")
                                            ep_nmb = _builder_alloc_scalar("ep_nmb", "int32")
                                            ep_last = _builder_alloc_scalar("ep_last", "int32")
                                            ep_psum = _builder_alloc_scalar("ep_psum", "int32")
                                            ep_nb = _builder_alloc_scalar("ep_nb", "int32")
                                            ep_nxt = _builder_alloc_scalar("ep_nxt", "int32")
                                            ep_nxtk = _builder_alloc_scalar("ep_nxtk", "int32")
                                            ep_vgrp = _builder_alloc_scalar("ep_vgrp", "int32")
                                            ep_kend = _builder_alloc_scalar("ep_kend", "int32")
                                            ep_it = _builder_assign(
                                                "ep_it", 0, locals().get("ep_it", _BUILDER_MISSING)
                                            )
                                            ep_valid = _builder_assign(
                                                "ep_valid",
                                                1,
                                                locals().get("ep_valid", _BUILDER_MISSING),
                                            )
                                            ep_grp = _builder_assign(
                                                "ep_grp",
                                                0,
                                                locals().get("ep_grp", _BUILDER_MISSING),
                                            )
                                            ep_cum = _builder_assign(
                                                "ep_cum",
                                                0,
                                                locals().get("ep_cum", _BUILDER_MISSING),
                                            )
                                            ep_last = _builder_assign(
                                                "ep_last",
                                                0,
                                                locals().get("ep_last", _BUILDER_MISSING),
                                            )
                                            ep_vgrp = _builder_assign(
                                                "ep_vgrp",
                                                0,
                                                locals().get("ep_vgrp", _BUILDER_MISSING),
                                            )
                                            ep_kend = _builder_assign(
                                                "ep_kend",
                                                0,
                                                locals().get("ep_kend", _BUILDER_MISSING),
                                            )
                                            ep_nxt = _builder_assign(
                                                "ep_nxt",
                                                0,
                                                locals().get("ep_nxt", _BUILDER_MISSING),
                                            )
                                            ep_nxtk = _builder_assign(
                                                "ep_nxtk",
                                                0,
                                                locals().get("ep_nxtk", _BUILDER_MISSING),
                                            )
                                            if is_k_grouped:
                                                ep_psum = _builder_assign(
                                                    "ep_psum",
                                                    0,
                                                    locals().get("ep_psum", _BUILDER_MISSING),
                                                )
                                                ep_nmb = _builder_assign(
                                                    "ep_nmb",
                                                    num_m_blocks,
                                                    locals().get("ep_nmb", _BUILDER_MISSING),
                                                )
                                                if is_k_grouped_psum:
                                                    with T.While(ep_grp < num_groups):
                                                        ep_nxtk = _builder_assign(
                                                            "ep_nxtk",
                                                            grouped_layout[ep_grp],
                                                            locals().get(
                                                                "ep_nxtk", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ep_last = _builder_assign(
                                                            "ep_last",
                                                            _uceil(ep_kend, k_alignment)
                                                            * k_alignment,
                                                            locals().get(
                                                                "ep_last", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ep_psum = _builder_assign(
                                                            "ep_psum",
                                                            ep_nxtk - ep_last,
                                                            locals().get(
                                                                "ep_psum", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ep_kend = _builder_assign(
                                                            "ep_kend",
                                                            ep_nxtk,
                                                            locals().get(
                                                                "ep_kend", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(ep_psum > 0):
                                                            with T.Then():
                                                                T.evaluate(T.break_loop())
                                                        ep_grp = _builder_assign(
                                                            "ep_grp",
                                                            ep_grp + 1,
                                                            locals().get(
                                                                "ep_grp", _BUILDER_MISSING
                                                            ),
                                                        )
                                                else:
                                                    with T.While(ep_grp < num_groups):
                                                        ep_psum = _builder_assign(
                                                            "ep_psum",
                                                            grouped_layout[ep_grp],
                                                            locals().get(
                                                                "ep_psum", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(ep_psum > 0):
                                                            with T.Then():
                                                                T.evaluate(T.break_loop())
                                                        ep_grp = _builder_assign(
                                                            "ep_grp",
                                                            ep_grp + 1,
                                                            locals().get(
                                                                "ep_grp", _BUILDER_MISSING
                                                            ),
                                                        )
                                                    ep_nxt = _builder_assign(
                                                        "ep_nxt",
                                                        ep_grp + 1,
                                                        locals().get("ep_nxt", _BUILDER_MISSING),
                                                    )
                                                    with T.While(ep_nxt < num_groups):
                                                        ep_nxtk = _builder_assign(
                                                            "ep_nxtk",
                                                            grouped_layout[ep_nxt],
                                                            locals().get(
                                                                "ep_nxtk", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(ep_nxtk > 0):
                                                            with T.Then():
                                                                T.evaluate(T.break_loop())
                                                        ep_nxt = _builder_assign(
                                                            "ep_nxt",
                                                            ep_nxt + 1,
                                                            locals().get(
                                                                "ep_nxt", _BUILDER_MISSING
                                                            ),
                                                        )
                                            elif is_m_grouped_psum:
                                                ep_psum = _builder_assign(
                                                    "ep_psum",
                                                    grouped_layout[0],
                                                    locals().get("ep_psum", _BUILDER_MISSING),
                                                )
                                                ep_nmb = _builder_assign(
                                                    "ep_nmb",
                                                    _uceil(ep_psum, block_m),
                                                    locals().get("ep_nmb", _BUILDER_MISSING),
                                                )
                                            else:
                                                ep_psum = _builder_assign(
                                                    "ep_psum",
                                                    0,
                                                    locals().get("ep_psum", _BUILDER_MISSING),
                                                )
                                                ep_nmb = _builder_assign(
                                                    "ep_nmb",
                                                    num_m_blocks,
                                                    locals().get("ep_nmb", _BUILDER_MISSING),
                                                )
                                            tma_stage = _builder_assign(
                                                "tma_stage",
                                                0,
                                                locals().get("tma_stage", _BUILDER_MISSING),
                                            )
                                            values = _builder_assign(
                                                "values",
                                                T.alloc_local((8,), "uint32"),
                                                locals().get("values", _BUILDER_MISSING),
                                            )
                                            packed = _builder_assign(
                                                "packed",
                                                T.alloc_local((4,), "uint32"),
                                                locals().get("packed", _BUILDER_MISSING),
                                            )
                                            with T.While(ep_valid == 1):
                                                ep_nb = _builder_assign(
                                                    "ep_nb",
                                                    ep_it * num_sms + sm_idx,
                                                    locals().get("ep_nb", _BUILDER_MISSING),
                                                )
                                                ep_done = _builder_alloc_scalar("ep_done", "int32")
                                                ep_done = _builder_assign(
                                                    "ep_done",
                                                    0,
                                                    locals().get("ep_done", _BUILDER_MISSING),
                                                )
                                                if is_m_grouped_masked:
                                                    with T.While(ep_done == 0):
                                                        with T.If(ep_grp == num_groups):
                                                            with T.Then():
                                                                ep_valid = _builder_assign(
                                                                    "ep_valid",
                                                                    0,
                                                                    locals().get(
                                                                        "ep_valid", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                ep_done = _builder_assign(
                                                                    "ep_done",
                                                                    1,
                                                                    locals().get(
                                                                        "ep_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                ep_nmb = _builder_assign(
                                                                    "ep_nmb",
                                                                    _uceil(
                                                                        grouped_layout[ep_grp],
                                                                        block_m,
                                                                    ),
                                                                    locals().get(
                                                                        "ep_nmb", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                with T.If(
                                                                    ep_nb
                                                                    < (ep_cum + ep_nmb)
                                                                    * num_n_blocks
                                                                ):
                                                                    with T.Then():
                                                                        ep_done = _builder_assign(
                                                                            "ep_done",
                                                                            1,
                                                                            locals().get(
                                                                                "ep_done",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                    with T.Else():
                                                                        ep_cum = _builder_assign(
                                                                            "ep_cum",
                                                                            ep_cum + ep_nmb,
                                                                            locals().get(
                                                                                "ep_cum",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        ep_grp = _builder_assign(
                                                                            "ep_grp",
                                                                            ep_grp + 1,
                                                                            locals().get(
                                                                                "ep_grp",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                elif is_m_grouped_psum:
                                                    with T.While(ep_done == 0):
                                                        with T.If(
                                                            ep_nb < (ep_cum + ep_nmb) * num_n_blocks
                                                        ):
                                                            with T.Then():
                                                                ep_done = _builder_assign(
                                                                    "ep_done",
                                                                    1,
                                                                    locals().get(
                                                                        "ep_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                ep_grp = _builder_assign(
                                                                    "ep_grp",
                                                                    ep_grp + 1,
                                                                    locals().get(
                                                                        "ep_grp", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                with T.If(ep_grp == num_groups):
                                                                    with T.Then():
                                                                        ep_valid = _builder_assign(
                                                                            "ep_valid",
                                                                            0,
                                                                            locals().get(
                                                                                "ep_valid",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        ep_done = _builder_assign(
                                                                            "ep_done",
                                                                            1,
                                                                            locals().get(
                                                                                "ep_done",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                    with T.Else():
                                                                        ep_last = _builder_assign(
                                                                            "ep_last",
                                                                            _uceil(ep_psum, block_m)
                                                                            * block_m,
                                                                            locals().get(
                                                                                "ep_last",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        ep_psum = _builder_assign(
                                                                            "ep_psum",
                                                                            grouped_layout[ep_grp],
                                                                            locals().get(
                                                                                "ep_psum",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        ep_cum = _builder_assign(
                                                                            "ep_cum",
                                                                            ep_cum + ep_nmb,
                                                                            locals().get(
                                                                                "ep_cum",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        ep_nmb = _builder_assign(
                                                                            "ep_nmb",
                                                                            _uceil(
                                                                                ep_psum - ep_last,
                                                                                block_m,
                                                                            ),
                                                                            locals().get(
                                                                                "ep_nmb",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                elif is_k_grouped:
                                                    with T.While(ep_done == 0):
                                                        with T.If(ep_grp == num_groups):
                                                            with T.Then():
                                                                ep_valid = _builder_assign(
                                                                    "ep_valid",
                                                                    0,
                                                                    locals().get(
                                                                        "ep_valid", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                ep_done = _builder_assign(
                                                                    "ep_done",
                                                                    1,
                                                                    locals().get(
                                                                        "ep_done", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            with T.Else():
                                                                with T.If(
                                                                    ep_nb
                                                                    < (ep_vgrp + 1) * num_blocks
                                                                ):
                                                                    with T.Then():
                                                                        ep_done = _builder_assign(
                                                                            "ep_done",
                                                                            1,
                                                                            locals().get(
                                                                                "ep_done",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                    with T.Else():
                                                                        ep_vgrp = _builder_assign(
                                                                            "ep_vgrp",
                                                                            ep_vgrp + 1,
                                                                            locals().get(
                                                                                "ep_vgrp",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        if is_k_grouped_psum:
                                                                            ep_grp = _builder_assign(
                                                                                "ep_grp",
                                                                                ep_grp + 1,
                                                                                locals().get(
                                                                                    "ep_grp",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            with T.While(
                                                                                ep_grp < num_groups
                                                                            ):
                                                                                ep_nxtk = _builder_assign(
                                                                                    "ep_nxtk",
                                                                                    grouped_layout[
                                                                                        ep_grp
                                                                                    ],
                                                                                    locals().get(
                                                                                        "ep_nxtk",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                ep_last = _builder_assign(
                                                                                    "ep_last",
                                                                                    _uceil(
                                                                                        ep_kend,
                                                                                        k_alignment,
                                                                                    )
                                                                                    * k_alignment,
                                                                                    locals().get(
                                                                                        "ep_last",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                ep_psum = _builder_assign(
                                                                                    "ep_psum",
                                                                                    ep_nxtk
                                                                                    - ep_last,
                                                                                    locals().get(
                                                                                        "ep_psum",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                ep_kend = _builder_assign(
                                                                                    "ep_kend",
                                                                                    ep_nxtk,
                                                                                    locals().get(
                                                                                        "ep_kend",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                with T.If(
                                                                                    ep_psum > 0
                                                                                ):
                                                                                    with T.Then():
                                                                                        T.evaluate(
                                                                                            T.break_loop()
                                                                                        )
                                                                                ep_grp = _builder_assign(
                                                                                    "ep_grp",
                                                                                    ep_grp + 1,
                                                                                    locals().get(
                                                                                        "ep_grp",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                        else:
                                                                            ep_last = _builder_assign(
                                                                                "ep_last",
                                                                                ep_last + ep_psum,
                                                                                locals().get(
                                                                                    "ep_last",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            ep_grp = _builder_assign(
                                                                                "ep_grp",
                                                                                ep_nxt,
                                                                                locals().get(
                                                                                    "ep_grp",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            ep_nxt = _builder_assign(
                                                                                "ep_nxt",
                                                                                ep_nxt + 1,
                                                                                locals().get(
                                                                                    "ep_nxt",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            ep_psum = _builder_assign(
                                                                                "ep_psum",
                                                                                ep_nxtk,
                                                                                locals().get(
                                                                                    "ep_psum",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            with T.While(
                                                                                ep_nxt < num_groups
                                                                            ):
                                                                                ep_nxtk = _builder_assign(
                                                                                    "ep_nxtk",
                                                                                    grouped_layout[
                                                                                        ep_nxt
                                                                                    ],
                                                                                    locals().get(
                                                                                        "ep_nxtk",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                with T.If(
                                                                                    ep_nxtk > 0
                                                                                ):
                                                                                    with T.Then():
                                                                                        T.evaluate(
                                                                                            T.break_loop()
                                                                                        )
                                                                                ep_nxt = _builder_assign(
                                                                                    "ep_nxt",
                                                                                    ep_nxt + 1,
                                                                                    locals().get(
                                                                                        "ep_nxt",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                    ep_cum = _builder_assign(
                                                        "ep_cum",
                                                        ep_vgrp * num_m_blocks,
                                                        locals().get("ep_cum", _BUILDER_MISSING),
                                                    )
                                                elif is_batched:
                                                    with T.If(ep_nb >= num_blocks * num_groups):
                                                        with T.Then():
                                                            ep_valid = _builder_assign(
                                                                "ep_valid",
                                                                0,
                                                                locals().get(
                                                                    "ep_valid", _BUILDER_MISSING
                                                                ),
                                                            )
                                                        with T.Else():
                                                            ep_grp = _builder_assign(
                                                                "ep_grp",
                                                                ep_nb // num_blocks,
                                                                locals().get(
                                                                    "ep_grp", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            ep_cum = _builder_assign(
                                                                "ep_cum",
                                                                ep_grp * num_m_blocks,
                                                                locals().get(
                                                                    "ep_cum", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            ep_nmb = _builder_assign(
                                                                "ep_nmb",
                                                                num_m_blocks,
                                                                locals().get(
                                                                    "ep_nmb", _BUILDER_MISSING
                                                                ),
                                                            )
                                                if not (
                                                    is_m_grouped_masked
                                                    or is_m_grouped_psum
                                                    or is_k_grouped
                                                    or is_batched
                                                ):
                                                    with T.If(ep_nb >= num_blocks):
                                                        with T.Then():
                                                            ep_valid = _builder_assign(
                                                                "ep_valid",
                                                                0,
                                                                locals().get(
                                                                    "ep_valid", _BUILDER_MISSING
                                                                ),
                                                            )
                                                with T.If(ep_valid == 1):
                                                    with T.Then():
                                                        accum_stage_e = _builder_alloc_scalar(
                                                            "accum_stage_e", "int32"
                                                        )
                                                        accum_phase_e = _builder_alloc_scalar(
                                                            "accum_phase_e", "int32"
                                                        )
                                                        accum_stage_e = _builder_assign(
                                                            "accum_stage_e",
                                                            ep_it % NUM_EPILOGUE_STAGES,
                                                            locals().get(
                                                                "accum_stage_e", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        accum_phase_e = _builder_assign(
                                                            "accum_phase_e",
                                                            T.bitwise_and(
                                                                ep_it // NUM_EPILOGUE_STAGES, 1
                                                            ),
                                                            locals().get(
                                                                "accum_phase_e", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ep_wait = _builder_alloc_scalar(
                                                            "ep_wait", "uint32"
                                                        )
                                                        ep_wait = _builder_assign(
                                                            "ep_wait",
                                                            T.uint32(0),
                                                            locals().get(
                                                                "ep_wait", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.While(ep_wait == T.uint32(0)):
                                                            _builder_emit(
                                                                T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                                                    ep_wait,
                                                                    barriers.ptr_to(
                                                                        [
                                                                            tmem_full_base
                                                                            + accum_stage_e
                                                                        ]
                                                                    ),
                                                                    T.cast(accum_phase_e, "uint32"),
                                                                    T.uint32(TRY_WAIT_TICKS),
                                                                )
                                                            )
                                                        _builder_emit(
                                                            T.ptx.tcgen05.fence__after_thread_sync()
                                                        )
                                                        ep_m_local, ep_n_local = (
                                                            _builder_assign_many(
                                                                ("ep_m_local", "ep_n_local"),
                                                                _swizzled(
                                                                    ep_nb - ep_cum * num_n_blocks,
                                                                    ep_nmb,
                                                                    num_n_blocks,
                                                                ),
                                                                (
                                                                    locals().get(
                                                                        "ep_m_local",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                    locals().get(
                                                                        "ep_n_local",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                ),
                                                            )
                                                        )
                                                        base_m = _builder_alloc_scalar(
                                                            "base_m", "int32"
                                                        )
                                                        base_n = _builder_alloc_scalar(
                                                            "base_n", "int32"
                                                        )
                                                        base_m = _builder_assign(
                                                            "base_m",
                                                            (
                                                                ep_m_local
                                                                + (
                                                                    _udiv(ep_last, block_m)
                                                                    if is_m_grouped_psum
                                                                    else 0
                                                                )
                                                            )
                                                            * block_m,
                                                            locals().get(
                                                                "base_m", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        base_n = _builder_assign(
                                                            "base_n",
                                                            ep_n_local * block_n,
                                                            locals().get(
                                                                "base_n", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        if is_m_grouped_masked or is_k_grouped:
                                                            base_m = _builder_assign(
                                                                "base_m",
                                                                ep_grp * eff_m + base_m,
                                                                locals().get(
                                                                    "base_m", _BUILDER_MISSING
                                                                ),
                                                            )
                                                        tmem_base = _builder_alloc_scalar(
                                                            "tmem_base", "int32"
                                                        )
                                                        tmem_base = _builder_assign(
                                                            "tmem_base",
                                                            accum_stage_e * umma_n,
                                                            locals().get(
                                                                "tmem_base", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ep_stores = _builder_alloc_scalar(
                                                            "ep_stores", "int32"
                                                        )
                                                        if use_effective_m:
                                                            ep_eff_m = _builder_alloc_scalar(
                                                                "ep_eff_m", "int32"
                                                            )
                                                            ep_eff_m = _builder_assign(
                                                                "ep_eff_m",
                                                                block_m,
                                                                locals().get(
                                                                    "ep_eff_m", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            with (
                                                                T.If(ep_m_local == ep_nmb - 1),
                                                                T.Then(),
                                                            ):
                                                                ep_eff_m = _builder_assign(
                                                                    "ep_eff_m",
                                                                    _uceil(
                                                                        ep_psum
                                                                        - (
                                                                            ep_m_local
                                                                            + _udiv(
                                                                                ep_last, block_m
                                                                            )
                                                                        )
                                                                        * block_m,
                                                                        UMMA_STEP_N,
                                                                    )
                                                                    * UMMA_STEP_N,
                                                                    locals().get(
                                                                        "ep_eff_m", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                            ep_stores = _builder_assign(
                                                                "ep_stores",
                                                                _udiv(ep_eff_m, store_block_m),
                                                                locals().get(
                                                                    "ep_stores", _BUILDER_MISSING
                                                                ),
                                                            )
                                                        else:
                                                            ep_stores = _builder_assign(
                                                                "ep_stores",
                                                                num_swap_stores,
                                                                locals().get(
                                                                    "ep_stores", _BUILDER_MISSING
                                                                ),
                                                            )
                                                        if swap_ab:
                                                            with T.unroll(0, num_swap_stores) as st:
                                                                IRBuilder.name("st", st)
                                                                with T.If(st < ep_stores), T.Then():
                                                                    with T.If(ep_warp == 0):
                                                                        with T.Then():
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx.cp.async_.bulk.wait_group(
                                                                                        NUM_TMA_STORE_STAGES
                                                                                        - 1
                                                                                    )
                                                                                )
                                                                            )
                                                                    _builder_emit(
                                                                        T.ptx.bar.sync(
                                                                            T.uint32(
                                                                                EPILOGUE_NAMED_BARRIER
                                                                            ),
                                                                            T.uint32(
                                                                                num_store_threads
                                                                            ),
                                                                        )
                                                                    )
                                                                    with T.unroll(
                                                                        0, num_atom_rows
                                                                    ) as i:
                                                                        IRBuilder.name("i", i)
                                                                        taddr_s = (
                                                                            _builder_alloc_scalar(
                                                                                "taddr_s", "uint32"
                                                                            )
                                                                        )
                                                                        taddr_s = _builder_assign(
                                                                            "taddr_s",
                                                                            T.cast(
                                                                                tmem_base
                                                                                + st * store_block_m
                                                                                + i * 8,
                                                                                "uint32",
                                                                            ),
                                                                            locals().get(
                                                                                "taddr_s",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        atom_byte = _builder_assign(
                                                                            "atom_byte",
                                                                            ep_warp
                                                                            // warps_per_atom
                                                                            * store_block_m
                                                                            * swizzle_cd
                                                                            + i * 8 * swizzle_cd,
                                                                            locals().get(
                                                                                "atom_byte",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        if cd_is_fp32:
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx[
                                                                                        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
                                                                                    ](
                                                                                        *[
                                                                                            values[
                                                                                                j
                                                                                            ]
                                                                                            for j in range(
                                                                                                8
                                                                                            )
                                                                                        ],
                                                                                        taddr_s,
                                                                                    )
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.tcgen05.wait__ld.sync.aligned()
                                                                            )
                                                                            col_f = _builder_alloc_scalar(
                                                                                "col_f", "int32"
                                                                            )
                                                                            col_f = _builder_assign(
                                                                                "col_f",
                                                                                lane_idx // 4,
                                                                                locals().get(
                                                                                    "col_f",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            with T.unroll(
                                                                                0, 8
                                                                            ) as row:
                                                                                IRBuilder.name(
                                                                                    "row", row
                                                                                )
                                                                                T.buffer_store(
                                                                                    smem_cd_u32,
                                                                                    values[row],
                                                                                    [
                                                                                        (
                                                                                            tma_stage
                                                                                            * cd_stage_bytes
                                                                                            + atom_byte
                                                                                            + row
                                                                                            * (
                                                                                                16
                                                                                                * 8
                                                                                            )
                                                                                            + T.bitwise_xor(
                                                                                                col_f,
                                                                                                row,
                                                                                            )
                                                                                            * 16
                                                                                            + lane_idx
                                                                                            % 4
                                                                                            * 4
                                                                                        )
                                                                                        // 4
                                                                                    ],
                                                                                )
                                                                        else:
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx[
                                                                                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                                                                    ](
                                                                                        values[0],
                                                                                        values[1],
                                                                                        values[2],
                                                                                        values[3],
                                                                                        taddr_s,
                                                                                    )
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx[
                                                                                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                                                                    ](
                                                                                        values[4],
                                                                                        values[5],
                                                                                        values[6],
                                                                                        values[7],
                                                                                        T.bitwise_or(
                                                                                            taddr_s,
                                                                                            T.uint32(
                                                                                                1048576
                                                                                            ),
                                                                                        ),
                                                                                    )
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.tcgen05.wait__ld.sync.aligned()
                                                                            )
                                                                            with T.unroll(
                                                                                0, 4
                                                                            ) as j:
                                                                                IRBuilder.name(
                                                                                    "j", j
                                                                                )
                                                                                _builder_emit(
                                                                                    T.ptx.cvt.rn.bf16x2.f32(
                                                                                        packed[j],
                                                                                        T.reinterpret(
                                                                                            "float32",
                                                                                            values[
                                                                                                2
                                                                                                * j
                                                                                                + 1
                                                                                            ],
                                                                                        ),
                                                                                        T.reinterpret(
                                                                                            "float32",
                                                                                            values[
                                                                                                2
                                                                                                * j
                                                                                            ],
                                                                                        ),
                                                                                    )
                                                                                )
                                                                            row_s = _builder_alloc_scalar(
                                                                                "row_s", "int32"
                                                                            )
                                                                            col_s = _builder_alloc_scalar(
                                                                                "col_s", "int32"
                                                                            )
                                                                            row_s = _builder_assign(
                                                                                "row_s",
                                                                                lane_idx % 8,
                                                                                locals().get(
                                                                                    "row_s",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            col_s = _builder_assign(
                                                                                "col_s",
                                                                                ep_warp % 2 * 4
                                                                                + lane_idx // 8,
                                                                                locals().get(
                                                                                    "col_s",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                                                                    smem_cd_u32.ptr_to(
                                                                                        [
                                                                                            (
                                                                                                tma_stage
                                                                                                * cd_stage_bytes
                                                                                                + atom_byte
                                                                                                + row_s
                                                                                                * (
                                                                                                    16
                                                                                                    * 8
                                                                                                )
                                                                                                + T.bitwise_xor(
                                                                                                    col_s,
                                                                                                    row_s,
                                                                                                )
                                                                                                * 16
                                                                                            )
                                                                                            // 4
                                                                                        ]
                                                                                    ),
                                                                                    packed[0],
                                                                                    packed[1],
                                                                                    packed[2],
                                                                                    packed[3],
                                                                                )
                                                                            )
                                                                    with T.If(st == ep_stores - 1):
                                                                        with T.Then():
                                                                            _builder_emit(
                                                                                T.ptx.tcgen05.fence__before_thread_sync()
                                                                            )
                                                                            rem_s = _builder_assign(
                                                                                "rem_s",
                                                                                T.alloc_local(
                                                                                    (1,), "uint64"
                                                                                ),
                                                                                locals().get(
                                                                                    "rem_s",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.mapa.shared__cluster.u64(
                                                                                    rem_s[0],
                                                                                    barriers.ptr_to(
                                                                                        [
                                                                                            tmem_empty_base
                                                                                            + accum_stage_e
                                                                                        ]
                                                                                    ),
                                                                                    T.uint32(0),
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.mbarrier.arrive.b64(
                                                                                    rem_s[0],
                                                                                    T.uint32(1),
                                                                                    pred=T.bool(
                                                                                        True
                                                                                    ),
                                                                                )
                                                                            )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx.fence.proxy.async_.shared__cta()
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        T.ptx.bar.sync(
                                                                            T.uint32(
                                                                                EPILOGUE_NAMED_BARRIER
                                                                            ),
                                                                            T.uint32(
                                                                                num_store_threads
                                                                            ),
                                                                        )
                                                                    )
                                                                    with T.If(ep_warp == 0):
                                                                        with T.Then():
                                                                            ep_elected = _builder_alloc_scalar(
                                                                                "ep_elected",
                                                                                "uint32",
                                                                            )
                                                                            ep_elected_lane = _builder_alloc_scalar(
                                                                                "ep_elected_lane",
                                                                                "uint32",
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.elect_sync(
                                                                                    ep_elected_lane,
                                                                                    ep_elected,
                                                                                    T.uint32(
                                                                                        4294967295
                                                                                    ),
                                                                                )
                                                                            )
                                                                            with T.unroll(
                                                                                0, num_n_atoms
                                                                            ) as i:
                                                                                IRBuilder.name(
                                                                                    "i", i
                                                                                )
                                                                                _builder_emit(
                                                                                    T.evaluate(
                                                                                        T.ptx[
                                                                                            reduce_chain
                                                                                            if with_accumulation
                                                                                            else store_chain
                                                                                        ](
                                                                                            T.address_of(
                                                                                                tensor_map_cd
                                                                                            ),
                                                                                            *_tma_coords(
                                                                                                (
                                                                                                    base_n
                                                                                                    + i
                                                                                                    * store_block_n_atom,
                                                                                                    base_m
                                                                                                    + st
                                                                                                    * store_block_m,
                                                                                                ),
                                                                                                ep_grp,
                                                                                            ),
                                                                                            smem_cd_u32.ptr_to(
                                                                                                [
                                                                                                    (
                                                                                                        tma_stage
                                                                                                        * cd_stage_bytes
                                                                                                        + i
                                                                                                        * store_block_m
                                                                                                        * swizzle_cd
                                                                                                    )
                                                                                                    // 4
                                                                                                ]
                                                                                            ),
                                                                                            pred=T.cast(
                                                                                                ep_elected
                                                                                                == T.uint32(
                                                                                                    1
                                                                                                ),
                                                                                                "bool",
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx.cp.async_.bulk.commit_group()
                                                                                )
                                                                            )
                                                                    _builder_emit(
                                                                        T.ptx.bar.warp.sync(
                                                                            T.uint32(4294967295)
                                                                        )
                                                                    )
                                                                    tma_stage = _builder_assign(
                                                                        "tma_stage",
                                                                        T.Select(
                                                                            tma_stage
                                                                            == NUM_TMA_STORE_STAGES
                                                                            - 1,
                                                                            0,
                                                                            tma_stage + 1,
                                                                        ),
                                                                        locals().get(
                                                                            "tma_stage",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                        else:
                                                            with T.unroll(0, num_m_waves) as w:
                                                                IRBuilder.name("w", w)
                                                                with T.unroll(0, num_stores) as st:
                                                                    IRBuilder.name("st", st)
                                                                    with T.If(ep_warp == 0):
                                                                        with T.Then():
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx.cp.async_.bulk.wait_group(
                                                                                        NUM_TMA_STORE_STAGES
                                                                                        - 1
                                                                                    )
                                                                                )
                                                                            )
                                                                    _builder_emit(
                                                                        T.ptx.bar.sync(
                                                                            T.uint32(
                                                                                EPILOGUE_NAMED_BARRIER
                                                                            ),
                                                                            T.uint32(
                                                                                num_store_threads
                                                                            ),
                                                                        )
                                                                    )
                                                                    with T.unroll(
                                                                        0, elems_per_store
                                                                    ) as i:
                                                                        IRBuilder.name("i", i)
                                                                        bank_group = (
                                                                            _builder_alloc_scalar(
                                                                                "bank_group",
                                                                                "int32",
                                                                            )
                                                                        )
                                                                        bank_group = _builder_assign(
                                                                            "bank_group",
                                                                            i
                                                                            + lane_idx
                                                                            * (swizzle_cd // 16),
                                                                            locals().get(
                                                                                "bank_group",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        row = _builder_alloc_scalar(
                                                                            "row", "int32"
                                                                        )
                                                                        row = _builder_assign(
                                                                            "row",
                                                                            i // 8 + lane_idx
                                                                            if has_shortcut
                                                                            else bank_group // 8,
                                                                            locals().get(
                                                                                "row",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        col = _builder_alloc_scalar(
                                                                            "col", "int32"
                                                                        )
                                                                        col = _builder_assign(
                                                                            "col",
                                                                            i
                                                                            if has_shortcut
                                                                            else bank_group % 8,
                                                                            locals().get(
                                                                                "col",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        col = _builder_assign(
                                                                            "col",
                                                                            T.bitwise_xor(
                                                                                col,
                                                                                row
                                                                                % (
                                                                                    swizzle_cd // 16
                                                                                ),
                                                                            ),
                                                                            locals().get(
                                                                                "col",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        cd_word = (
                                                                            _builder_alloc_scalar(
                                                                                "cd_word", "int32"
                                                                            )
                                                                        )
                                                                        cd_word = _builder_assign(
                                                                            "cd_word",
                                                                            _cd_word(
                                                                                tma_stage,
                                                                                ep_warp,
                                                                                row,
                                                                                col,
                                                                                0,
                                                                            ),
                                                                            locals().get(
                                                                                "cd_word",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        taddr = (
                                                                            _builder_alloc_scalar(
                                                                                "taddr", "uint32"
                                                                            )
                                                                        )
                                                                        taddr = _builder_assign(
                                                                            "taddr",
                                                                            T.cast(
                                                                                tmem_base
                                                                                + w * block_n
                                                                                + st * store_block_n
                                                                                + i
                                                                                * elems_per_bank_group,
                                                                                "uint32",
                                                                            ),
                                                                            locals().get(
                                                                                "taddr",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        if cd_is_fp32:
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx[
                                                                                        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
                                                                                    ](
                                                                                        values[0],
                                                                                        values[1],
                                                                                        values[2],
                                                                                        values[3],
                                                                                        taddr,
                                                                                    )
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.tcgen05.wait__ld.sync.aligned()
                                                                            )
                                                                            with T.unroll(
                                                                                0, 4
                                                                            ) as j:
                                                                                IRBuilder.name(
                                                                                    "j", j
                                                                                )
                                                                                T.buffer_store(
                                                                                    smem_cd_u32,
                                                                                    values[j],
                                                                                    [cd_word + j],
                                                                                )
                                                                        else:
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx[
                                                                                        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
                                                                                    ](
                                                                                        *[
                                                                                            values[
                                                                                                j
                                                                                            ]
                                                                                            for j in range(
                                                                                                8
                                                                                            )
                                                                                        ],
                                                                                        taddr,
                                                                                    )
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.tcgen05.wait__ld.sync.aligned()
                                                                            )
                                                                            with T.unroll(
                                                                                0, 4
                                                                            ) as j:
                                                                                IRBuilder.name(
                                                                                    "j", j
                                                                                )
                                                                                _builder_emit(
                                                                                    T.ptx.cvt.rn.bf16x2.f32(
                                                                                        packed[j],
                                                                                        T.reinterpret(
                                                                                            "float32",
                                                                                            values[
                                                                                                2
                                                                                                * j
                                                                                                + 1
                                                                                            ],
                                                                                        ),
                                                                                        T.reinterpret(
                                                                                            "float32",
                                                                                            values[
                                                                                                2
                                                                                                * j
                                                                                            ],
                                                                                        ),
                                                                                    )
                                                                                )
                                                                            with T.unroll(
                                                                                0, 4
                                                                            ) as j:
                                                                                IRBuilder.name(
                                                                                    "j", j
                                                                                )
                                                                                T.buffer_store(
                                                                                    smem_cd_u32,
                                                                                    packed[j],
                                                                                    [cd_word + j],
                                                                                )
                                                                    with T.If(
                                                                        T.And(
                                                                            w == num_m_waves - 1,
                                                                            st == num_stores - 1,
                                                                        )
                                                                    ):
                                                                        with T.Then():
                                                                            _builder_emit(
                                                                                T.ptx.tcgen05.fence__before_thread_sync()
                                                                            )
                                                                            rem_e = _builder_assign(
                                                                                "rem_e",
                                                                                T.alloc_local(
                                                                                    (1,), "uint64"
                                                                                ),
                                                                                locals().get(
                                                                                    "rem_e",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.mapa.shared__cluster.u64(
                                                                                    rem_e[0],
                                                                                    barriers.ptr_to(
                                                                                        [
                                                                                            tmem_empty_base
                                                                                            + accum_stage_e
                                                                                        ]
                                                                                    ),
                                                                                    T.uint32(0),
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.mbarrier.arrive.b64(
                                                                                    rem_e[0],
                                                                                    T.uint32(1),
                                                                                    pred=T.bool(
                                                                                        True
                                                                                    ),
                                                                                )
                                                                            )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx.fence.proxy.async_.shared__cta()
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        T.ptx.bar.sync(
                                                                            T.uint32(
                                                                                EPILOGUE_NAMED_BARRIER
                                                                            ),
                                                                            T.uint32(
                                                                                num_store_threads
                                                                            ),
                                                                        )
                                                                    )
                                                                    with T.If(ep_warp == 0):
                                                                        with T.Then():
                                                                            ep_elected = _builder_alloc_scalar(
                                                                                "ep_elected",
                                                                                "uint32",
                                                                            )
                                                                            ep_elected_lane = _builder_alloc_scalar(
                                                                                "ep_elected_lane",
                                                                                "uint32",
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.elect_sync(
                                                                                    ep_elected_lane,
                                                                                    ep_elected,
                                                                                    T.uint32(
                                                                                        4294967295
                                                                                    ),
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx[
                                                                                        reduce_chain
                                                                                        if with_accumulation
                                                                                        else store_chain
                                                                                    ](
                                                                                        T.address_of(
                                                                                            tensor_map_cd
                                                                                        ),
                                                                                        *_tma_coords(
                                                                                            (
                                                                                                base_n
                                                                                                + st
                                                                                                * store_block_n,
                                                                                                base_m
                                                                                                + w
                                                                                                * store_block_m,
                                                                                            ),
                                                                                            ep_grp,
                                                                                        ),
                                                                                        smem_cd.ptr_to(
                                                                                            [
                                                                                                tma_stage,
                                                                                                0,
                                                                                                0,
                                                                                            ]
                                                                                        ),
                                                                                        pred=T.cast(
                                                                                            ep_elected
                                                                                            == T.uint32(
                                                                                                1
                                                                                            ),
                                                                                            "bool",
                                                                                        ),
                                                                                    )
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    T.ptx.cp.async_.bulk.commit_group()
                                                                                )
                                                                            )
                                                                    _builder_emit(
                                                                        T.ptx.bar.warp.sync(
                                                                            T.uint32(4294967295)
                                                                        )
                                                                    )
                                                                    tma_stage = _builder_assign(
                                                                        "tma_stage",
                                                                        T.Select(
                                                                            tma_stage
                                                                            == NUM_TMA_STORE_STAGES
                                                                            - 1,
                                                                            0,
                                                                            tma_stage + 1,
                                                                        ),
                                                                        locals().get(
                                                                            "tma_stage",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                ep_it = _builder_assign(
                                                    "ep_it",
                                                    ep_it + 1,
                                                    locals().get("ep_it", _BUILDER_MISSING),
                                                )
            if cta_group > 1:
                _builder_emit(T.ptx.barrier.cluster.arrive.relaxed.aligned())
                _builder_emit(T.ptx.barrier.cluster.wait.acquire.aligned())
            else:
                _builder_emit(T.ptx.bar.sync(T.uint32(0)))
            with T.If(warp == 0):
                with T.Then():
                    _builder_emit(
                        T.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                            T.uint32(0), T.uint32(spec.num_tmem_cols)
                        )
                    )

    return builder.get()


def _blocks_per_group(spec: GemmSpec) -> int:
    """`get_num_1d_blocks_per_group` (scheduler/gemm.cuh:14).

    Pick the group width in {8, 16} that minimises the bytes one wave touches.
    """
    best, best_usage = 0, None
    for candidate in (8, 16):
        if spec.is_multicast_on_a:
            usage = candidate * spec.block_n + -(-spec.num_sms // candidate) * spec.block_m
        else:
            usage = candidate * spec.block_m + -(-spec.num_sms // candidate) * spec.block_n
        if best_usage is None or usage < best_usage:
            best, best_usage = candidate, usage
    if best % spec.num_multicast != 0:
        raise ValueError("invalid L2 group size")
    return best


def _swizzle_enum(mode: int) -> int:
    return {0: 0, 16: 0, 32: 1, 64: 2, 128: 3}[mode]
