# Copyright (c) 2025 DeepSeek
# Copyright (c) 2026 The TIRX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# TIRx transcription of DeepGEMM's `sm100_fp8_fp4_gemm_1d1d_impl`
# (deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh) together with
# `scheduler/gemm.cuh` and `epilogue/sm100_store_cd{,_swap_ab}.cuh`.
# The structure follows `.agents/sketch/sm100_fp8_fp4_gemm_1d1d.md`.
# See NOTICE and THIRD_PARTY_LICENSES.md for upstream attribution.

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
"""

from __future__ import annotations

from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)
from tvm.script import tirx as T

from ._sm100_fp8_fp4_gemm_1d1d import (
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
    from ._sm100_fp8_fp4_gemm_1d1d import align_up

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

    @T.prim_func
    def sm100_fp8_fp4_gemm_1d1d(
        grouped_layout_ptr: T.handle,
        grouped_len: T.int32,
        shape_m: T.int32,
        shape_n: T.int32,
        shape_k: T.int32,
        tensor_map_a: T.TensorMap(),
        tensor_map_b: T.TensorMap(),
        tensor_map_sfa: T.TensorMap(),
        tensor_map_sfb: T.TensorMap(),
        tensor_map_cd: T.TensorMap(),
    ):
        grouped_layout = T.match_buffer(grouped_layout_ptr, (grouped_len,), "int32")
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})

        # ---- role ids -------------------------------------------------------
        sm_idx = T.cta_id([spec.num_sms])
        if cta_group > 1:
            cta_in_cluster = T.cta_id_in_cluster([cta_group])
            is_leader_cta = cta_in_cluster == 0
        else:
            is_leader_cta = True
        thread_idx = T.thread_id([spec.num_non_epilogue_threads + spec.num_epilogue_threads])
        lane_idx = T.lane_id([32])
        warp: T.int32
        T.ptx.shfl_sync.idx.b32(
            warp, thread_idx // 32, T.uint32(0), T.uint32(0x1F), T.uint32(0xFFFFFFFF)
        )

        # ---- shared memory --------------------------------------------------
        smem = T.alloc_buffer([spec.smem_size], "uint8", scope="shared.dyn", align=1024)
        T.attr({"tirx.dyn_smem_bytes": spec.smem_size})

        smem_cd_data: T.let = T.reinterpret(PointerType(PrimType(cd_dtype)), smem.ptr_to([0]))
        smem_a_data: T.let = T.reinterpret(
            PointerType(PrimType(a_smem_dtype)), smem.ptr_to([spec.smem_a_offset])
        )
        smem_b_data: T.let = T.reinterpret(
            PointerType(PrimType(b_smem_dtype)), smem.ptr_to([spec.smem_b_offset])
        )
        smem_sfa_data: T.let = T.reinterpret(
            PointerType(PrimType("uint32")), smem.ptr_to([spec.smem_sfa_offset])
        )
        smem_sfb_data: T.let = T.reinterpret(
            PointerType(PrimType("uint32")), smem.ptr_to([spec.smem_sfb_offset])
        )
        smem_bar_data: T.let = T.reinterpret(
            PointerType(PrimType("uint64")), smem.ptr_to([spec.smem_barrier_offset])
        )
        smem_tmem_data: T.let = T.reinterpret(
            PointerType(PrimType("uint32")), smem.ptr_to([spec.smem_tmem_ptr_offset])
        )

        # C/D is a linear byte view: the swizzle lives in the CD TensorMap and in
        # the epilogue's own `col ^= row % (swizzle / 16)` arithmetic below.
        smem_cd = T.decl_buffer(
            (NUM_TMA_STORE_STAGES, store_block_m, store_block_n),
            cd_dtype,
            data=smem_cd_data,
            scope="shared.dyn",
            elem_offset=0,
            align=1024,
        )
        smem_a = T.decl_buffer(
            (stages, load_block_m, block_k),
            a_smem_dtype,
            data=smem_a_data,
            scope="shared.dyn",
            elem_offset=0,
            align=1024,
            layout=a_layout,
        )
        smem_b = T.decl_buffer(
            (stages, load_block_n, block_k),
            b_smem_dtype,
            data=smem_b_data,
            scope="shared.dyn",
            elem_offset=0,
            align=1024,
            layout=b_layout,
        )
        smem_sfa = T.decl_buffer(
            (stages, sf_block_m),
            "uint32",
            data=smem_sfa_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_sfb = T.decl_buffer(
            (stages, sf_block_n),
            "uint32",
            data=smem_sfb_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        barriers = T.decl_buffer(
            (num_barriers,),
            "uint64",
            data=smem_bar_data,
            scope="shared.dyn",
            elem_offset=0,
            align=8,
        )
        tmem_slot = T.decl_buffer(
            (1,), "uint32", data=smem_tmem_data, scope="shared.dyn", elem_offset=0, align=4
        )
        smem_cd_word_data: T.let = T.reinterpret(PointerType(PrimType("uint32")), smem.ptr_to([0]))
        smem_cd_u32 = T.decl_buffer(
            (NUM_TMA_STORE_STAGES * cd_stage_bytes // 4,),
            "uint32",
            data=smem_cd_word_data,
            scope="shared.dyn",
            elem_offset=0,
            align=1024,
        )

        # ---- tensor memory --------------------------------------------------
        tmem = T.decl_buffer(
            (128, spec.num_tmem_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_slot[0],
            layout=TileLayout(S[(128, spec.num_tmem_cols) : (1 @ TLane, 1 @ TCol)]),
        )
        sfa_tmem = T.decl_buffer(
            (128, sf_block_m // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=spec.tmem_start_col_of_sfa,
            layout=sf_tmem_layout(128, SF_K=sf_block_m // 32, sf_per_mma=1),
        )
        sfb_tmem = T.decl_buffer(
            (128, sf_block_n // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=spec.tmem_start_col_of_sfb,
            layout=sf_tmem_layout(128, SF_K=sf_block_n // 32, sf_per_mma=1),
        )

        # ---- cluster rendezvous before the 2-CTA TMEM allocation -------------
        if cta_group > 1:
            T.ptx.barrier.cluster.arrive.relaxed.aligned()
            T.ptx.barrier.cluster.wait.acquire.aligned()

        if warp == 0:
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_a)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_b)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_sfa)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_sfb)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_cd)))

        # ---- barrier init and TMEM allocation -------------------------------
        if warp == 1:
            init_elected: T.uint32
            init_elected_lane: T.uint32
            T.ptx.elect_sync(init_elected_lane, init_elected, T.uint32(0xFFFFFFFF))
            if init_elected == T.uint32(1):
                for s in T.unroll(0, stages):
                    T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([full_base + s]), T.uint32(1))
                    T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([empty_base + s]), T.uint32(1))
                    T.ptx.mbarrier.init.shared.b64(
                        barriers.ptr_to([with_sf_base + s]), T.uint32(cta_group * 32)
                    )
                for e in T.unroll(0, NUM_EPILOGUE_STAGES):
                    T.ptx.mbarrier.init.shared.b64(
                        barriers.ptr_to([tmem_full_base + e]), T.uint32(1)
                    )
                    T.ptx.mbarrier.init.shared.b64(
                        barriers.ptr_to([tmem_empty_base + e]),
                        T.uint32(cta_group * num_store_threads),
                    )
                T.evaluate(T.ptx.fence.mbarrier_init.release.cluster())
        elif warp == 2:
            T.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                T.address_of(tmem_slot[0]), T.uint32(spec.num_tmem_cols)
            )

        if cta_group > 1:
            T.ptx.barrier.cluster.arrive.relaxed.aligned()
            T.ptx.barrier.cluster.wait.acquire.aligned()
        else:
            T.ptx.bar.sync(T.uint32(0))

        T.evaluate(T.ptx.griddepcontrol.wait())

        # ===============================================================
        # Persistent scheduler (source `scheduler/gemm.cuh`)
        # ===============================================================
        # Each scheduler-driving role keeps its own copy of this state and walks
        # the identical block sequence; that is what keeps the four roles'
        # `stage_idx`/`phase` cursors in lockstep without any extra barrier.

        # `compiled_dims` bakes a dimension in; thechoice happens in Python, so a
        # baked dimension becomes a literal and its runtime argument goes dead
        # (source `:123-125`).  The parameter cannot be reassigned -- TVMScript
        # forbids shadowing a parameter name.
        eff_m = spec.shape_m if spec.shape_m > 0 else shape_m
        eff_n = spec.shape_n if spec.shape_n > 0 else shape_n
        eff_k = spec.shape_k if spec.shape_k > 0 else shape_k

        num_m_blocks: T.int32
        num_n_blocks: T.int32
        num_blocks: T.int32
        num_k_blocks: T.int32
        num_m_blocks = _uceil(eff_m, block_m)
        num_n_blocks = (eff_n + (block_n - 1)) // block_n
        num_blocks = num_m_blocks * num_n_blocks
        num_k_blocks = _uceil(eff_k, block_k)
        shape_sfa_k: T.int32
        shape_sfb_k: T.int32
        shape_sfa_k = (eff_k + (spec.gran_k_a * 4 - 1)) // (spec.gran_k_a * 4)
        shape_sfb_k = (eff_k + (spec.gran_k_b * 4 - 1)) // (spec.gran_k_b * 4)

        # ===============================================================
        # Role 0: TMA load warp, one elected lane (source `:206`)
        # ===============================================================

        if warp == 0:
            ld_elected: T.uint32
            ld_elected_lane: T.uint32
            T.ptx.elect_sync(ld_elected_lane, ld_elected, T.uint32(0xFFFFFFFF))
            if ld_elected == T.uint32(1):
                ld_stage: T.int32
                ld_phase: T.int32
                ld_stage = 0
                ld_phase = 0
                ld_it: T.int32
                ld_valid: T.int32
                ld_grp: T.int32
                ld_cum: T.int32
                ld_nmb: T.int32
                ld_last: T.int32
                ld_psum: T.int32
                ld_nb: T.int32
                ld_nxt: T.int32
                ld_nxtk: T.int32
                ld_sfk: T.int32
                ld_kblocks: T.int32
                ld_vgrp: T.int32
                ld_kend: T.int32
                ld_it = 0
                ld_valid = 1
                ld_grp = 0
                ld_cum = 0
                ld_last = 0
                # Scheduler cursor init (`scheduler/gemm.cuh:88`).
                ld_sfk = 0
                ld_vgrp = 0
                ld_kend = 0
                ld_nxt = 0
                ld_nxtk = 0
                ld_kblocks = 0
                if is_k_grouped:
                    # Start on the first non-empty group (`get_next_k_group`, `:66`).
                    ld_psum = 0
                    ld_nmb = num_m_blocks
                    if is_k_grouped_psum:
                        while ld_grp < num_groups:
                            ld_nxtk = grouped_layout[ld_grp]
                            ld_last = (_uceil(ld_kend, k_alignment)) * k_alignment
                            ld_psum = ld_nxtk - ld_last
                            ld_kend = ld_nxtk
                            if ld_psum > 0:
                                break
                            ld_grp = ld_grp + 1
                    else:
                        while ld_grp < num_groups:
                            ld_psum = grouped_layout[ld_grp]
                            if ld_psum > 0:
                                break
                            ld_grp = ld_grp + 1
                        ld_nxt = ld_grp + 1
                        while ld_nxt < num_groups:
                            ld_nxtk = grouped_layout[ld_nxt]
                            if ld_nxtk > 0:
                                break
                            ld_nxt = ld_nxt + 1
                elif is_m_grouped_psum:
                    ld_psum = grouped_layout[0]
                    ld_nmb = _uceil(ld_psum, block_m)
                else:
                    ld_psum = 0
                    ld_nmb = num_m_blocks
                while ld_valid == 1:
                    ld_nb = ld_it * num_sms + sm_idx
                    # `get_next_block`: advance the group cursor for this linear block index.
                    ld_done: T.int32
                    ld_done = 0
                    if is_m_grouped_masked:
                        while ld_done == 0:
                            if ld_grp == num_groups:
                                ld_valid = 0
                                ld_done = 1
                            else:
                                ld_nmb = _uceil(grouped_layout[ld_grp], block_m)
                                if ld_nb < (ld_cum + ld_nmb) * num_n_blocks:
                                    ld_done = 1
                                else:
                                    ld_cum = ld_cum + ld_nmb
                                    ld_grp = ld_grp + 1
                    elif is_m_grouped_psum:
                        while ld_done == 0:
                            if ld_nb < (ld_cum + ld_nmb) * num_n_blocks:
                                ld_done = 1
                            else:
                                ld_grp = ld_grp + 1
                                if ld_grp == num_groups:
                                    ld_valid = 0
                                    ld_done = 1
                                else:
                                    ld_last = (_uceil(ld_psum, block_m)) * block_m
                                    ld_psum = grouped_layout[ld_grp]
                                    ld_cum = ld_cum + ld_nmb
                                    ld_nmb = _uceil(ld_psum - ld_last, block_m)
                    elif is_k_grouped:
                        # Walk the valid (non-empty) groups; each contributes `num_blocks`
                        # tiles and its own K extent (`scheduler/gemm.cuh:238`).
                        while ld_done == 0:
                            if ld_grp == num_groups:
                                ld_valid = 0
                                ld_done = 1
                            elif ld_nb < (ld_vgrp + 1) * num_blocks:
                                ld_done = 1
                            else:
                                ld_sfk = ld_sfk + (_uceil(ld_psum, sf_k_span))
                                ld_vgrp = ld_vgrp + 1
                                if is_k_grouped_psum:
                                    ld_grp = ld_grp + 1
                                    while ld_grp < num_groups:
                                        ld_nxtk = grouped_layout[ld_grp]
                                        ld_last = (_uceil(ld_kend, k_alignment)) * k_alignment
                                        ld_psum = ld_nxtk - ld_last
                                        ld_kend = ld_nxtk
                                        if ld_psum > 0:
                                            break
                                        ld_grp = ld_grp + 1
                                else:
                                    ld_last = ld_last + ld_psum
                                    ld_grp = ld_nxt
                                    ld_nxt = ld_nxt + 1
                                    ld_psum = ld_nxtk
                                    while ld_nxt < num_groups:
                                        ld_nxtk = grouped_layout[ld_nxt]
                                        if ld_nxtk > 0:
                                            break
                                        ld_nxt = ld_nxt + 1
                        ld_cum = ld_vgrp * num_m_blocks
                    elif is_batched:
                        if ld_nb >= num_blocks * num_groups:
                            ld_valid = 0
                        else:
                            ld_grp = ld_nb // num_blocks
                            ld_cum = ld_grp * num_m_blocks
                            ld_nmb = num_m_blocks
                    else:
                        if ld_nb >= num_blocks:
                            ld_valid = 0
                    if ld_valid == 1:
                        ld_kblocks = _uceil(ld_psum, block_k) if is_k_grouped else num_k_blocks
                        # One swizzle walk feeds `m_idx`, `n_idx` and the tail-block
                        # test below, as the source's single `get_swizzled_block_idx`
                        # call does (`scheduler/gemm.cuh:197`).
                        ld_m_local, ld_n_local = _swizzled(
                            ld_nb - ld_cum * num_n_blocks, ld_nmb, num_n_blocks
                        )
                        m_idx: T.int32
                        n_idx: T.int32
                        m_idx = (
                            ld_m_local + (_udiv(ld_last, block_m) if is_m_grouped_psum else 0)
                        ) * block_m
                        n_idx = ld_n_local * block_n
                        sfa_mn: T.int32
                        sfb_mn: T.int32
                        # The scale factors are indexed by the *whole* block, before
                        # the cluster split below.
                        sfa_mn = m_idx
                        sfb_mn = n_idx
                        # `get_global_idx` carries the expert offset on whichever of
                        # B's axes the group was folded into: N for a K-major B,
                        # K for an MN-major one (source `:223`, `:233`, `:271`).
                        k_b_idx: T.int32
                        k_a_offset: T.int32
                        k_b_idx = 0
                        k_a_offset = 0
                        sfa_k_offset: T.int32
                        sfa_k_offset = 0
                        if is_m_grouped_contiguous:
                            # Non-psum contiguous: the expert is a per-row id.
                            expert: T.int32
                            expert = T.max(0, grouped_layout[m_idx])
                            if major_b_is_k:
                                n_idx = expert * eff_n + n_idx
                            else:
                                k_b_idx = expert * eff_k + k_b_idx
                            sfb_k_offset = expert * shape_sfb_k
                        elif is_m_grouped_masked:
                            # Masked: A, B, SFA and SFB are all `[G, ...]` slabs, so every axis
                            # carries the group offset (source `:221-234`, `:264-272`).
                            m_idx = ld_grp * eff_m + m_idx
                            if major_b_is_k:
                                n_idx = ld_grp * eff_n + n_idx
                            else:
                                k_b_idx = ld_grp * eff_k + k_b_idx
                            sfa_k_offset = ld_grp * shape_sfa_k
                            sfb_k_offset = ld_grp * shape_sfb_k
                        elif is_m_grouped_psum:
                            # PSUM: A is one flat `[M, K]`; only B and SFB are grouped.
                            if major_b_is_k:
                                n_idx = ld_grp * eff_n + n_idx
                            else:
                                k_b_idx = ld_grp * eff_k + k_b_idx
                            sfb_k_offset = ld_grp * shape_sfb_k
                        elif is_k_grouped:
                            # Groups are concatenated along K; both operands are MN-major, so the
                            # K index carries the running offset and the SF index its own
                            # cumulative row count (source `:166-180`).
                            k_a_offset = ld_last
                            k_b_idx = ld_last
                            sfa_k_offset = ld_sfk
                            sfb_k_offset = ld_sfk
                        elif is_batched:
                            # Batched: the batch index rides the SF outer extent (source `:183`).
                            sfa_k_offset = ld_grp * shape_sfa_k
                            sfb_k_offset = ld_grp * shape_sfb_k
                        else:
                            sfb_k_offset = 0
                        # Each CTA of a cluster loads its own slice; the 2-CTA
                        # behaviour lives here and in the UMMA, not in a multicast
                        # TMA (source `:237-240`).
                        if use_effective_m:
                            # `get_aligned_effective_m_in_block` for this block (sketch `:559`).
                            ld_eff_m: T.int32
                            ld_eff_m = block_m
                            with T.If(ld_m_local == ld_nmb - 1), T.Then():
                                ld_eff_m = (
                                    _uceil(
                                        ld_psum - (ld_m_local + _udiv(ld_last, block_m)) * block_m,
                                        UMMA_STEP_N,
                                    )
                                    * UMMA_STEP_N
                                )
                        if cta_group > 1:
                            if is_multicast_on_a:
                                m_idx = m_idx + cta_in_cluster * (
                                    _udiv(ld_eff_m, cta_group) if use_effective_m else load_block_m
                                )
                            else:
                                n_idx = n_idx + cta_in_cluster * load_block_n
                        ld_k: T.int32
                        ld_k = 0
                        while ld_k < ld_kblocks:
                            ld_wait: T.uint32
                            ld_wait = T.uint32(0)
                            while ld_wait == T.uint32(0):
                                T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                    ld_wait,
                                    barriers.ptr_to([empty_base + ld_stage]),
                                    T.cast(T.bitwise_xor(ld_phase, 1), "uint32"),
                                    T.uint32(TRY_WAIT_TICKS),
                                )
                            k_b: T.int32
                            k_a: T.int32
                            k_b = k_b_idx + ld_k * block_k
                            k_a = k_a_offset + ld_k * block_k
                            for i in T.unroll(0, num_a_atoms):
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
                            for i in T.unroll(0, num_b_atoms):
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
                            arrival: T.int32
                            arrival = arrival_bytes_ab
                            if ld_k % sfa_stages_per_load == 0:
                                T.evaluate(
                                    T.ptx[sf_load_chain](
                                        smem_sfa.ptr_to([ld_stage, 0]),
                                        T.address_of(tensor_map_sfa),
                                        T.cast(sfa_mn, "int32"),
                                        T.cast(sfa_k_offset + ld_k // sfa_stages_per_load, "int32"),
                                        barriers.ptr_to([full_base + ld_stage]),
                                        T.uint64(EVICT_NORMAL),
                                    )
                                )
                                arrival = arrival + block_m * 4
                            if ld_k % sfb_stages_per_load == 0:
                                T.evaluate(
                                    T.ptx[sf_load_chain](
                                        smem_sfb.ptr_to([ld_stage, 0]),
                                        T.address_of(tensor_map_sfb),
                                        T.cast(sfb_mn, "int32"),
                                        T.cast(sfb_k_offset + ld_k // sfb_stages_per_load, "int32"),
                                        barriers.ptr_to([full_base + ld_stage]),
                                        T.uint64(EVICT_NORMAL),
                                    )
                                )
                                arrival = arrival + block_n * 4
                            T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                barriers.ptr_to([full_base + ld_stage]), T.cast(arrival, "uint32")
                            )
                            ld_k = ld_k + 1
                            ld_stage = T.Select(ld_stage == stages - 1, 0, ld_stage + 1)
                            ld_phase = T.bitwise_xor(ld_phase, T.cast(ld_stage == 0, "int32"))
                    ld_it = ld_it + 1

        # ===============================================================
        # Role 1: UMMA issue warp, leader CTA only (source `:281`)
        # ===============================================================

        elif warp == 1:
            if is_leader_cta:
                desc_a: T.uint64
                desc_b: T.uint64
                desc_sf: T.uint64
                desc_i: T.uint32
                # Every descriptor field but the shared address is a build-time
                # constant, so the bases are folded in Python and only
                # `addr >> 4` is computed here.  The `*_DESC_BASE` constants
                # come from the same bit layout the runtime C encoders fill in.
                a_smem_u32: T.uint32
                b_smem_u32: T.uint32
                sfa_smem_u32: T.uint32
                sfb_smem_u32: T.uint32
                a_smem_u32 = T.cuda.cvta_generic_to_shared(smem_a.ptr_to([0, 0, 0]))
                b_smem_u32 = T.cuda.cvta_generic_to_shared(smem_b.ptr_to([0, 0, 0]))
                sfa_smem_u32 = T.cuda.cvta_generic_to_shared(smem_sfa.ptr_to([0, 0]))
                sfb_smem_u32 = T.cuda.cvta_generic_to_shared(smem_sfb.ptr_to([0, 0]))
                desc_a = _with_smem_addr(_u64_const(A_DESC_BASE), a_smem_u32)
                desc_b = _with_smem_addr(_u64_const(B_DESC_BASE), b_smem_u32)
                # `make_sf_desc`: unswizzled, stride offset 8*16, leading offset 0.
                desc_sf = _with_smem_addr(_u64_const(SF_DESC_BASE), sfa_smem_u32)
                # Stays a mutable local: the scale-factor ids and, under
                # `use_effective_m`, the N field are patched per MMA below.
                desc_i = T.uint32(INSTR_DESC)
                # The per-stage descriptor low words live one stage per lane; a warp
                # shuffle indexes the table instead of recomputing the descriptor.
                a_desc_lo: T.uint32
                b_desc_lo: T.uint32
                a_desc_lo = T.Select(
                    lane_idx < stages,
                    T.cast(T.bitwise_and(desc_a, T.uint64(0xFFFFFFFF)), "uint32")
                    + T.cast(lane_idx * (a_bytes_per_stage // 16), "uint32"),
                    T.uint32(0),
                )
                b_desc_lo = T.Select(
                    lane_idx < stages,
                    T.cast(T.bitwise_and(desc_b, T.uint64(0xFFFFFFFF)), "uint32")
                    + T.cast(lane_idx * (b_bytes_per_stage // 16), "uint32"),
                    T.uint32(0),
                )

                mma_stage: T.int32
                mma_phase: T.int32
                mma_iter: T.int32
                mma_stage = 0
                mma_phase = 0
                mma_it: T.int32
                mma_valid: T.int32
                mma_grp: T.int32
                mma_cum: T.int32
                mma_nmb: T.int32
                mma_last: T.int32
                mma_psum: T.int32
                mma_nb: T.int32
                mma_nxt: T.int32
                mma_nxtk: T.int32
                mma_kblocks: T.int32
                mma_vgrp: T.int32
                mma_kend: T.int32
                # `cute::elect_one_sync()` is loop-invariant for this fully-active warp;
                # nvcc hoists it into a uniform predicate, so hoist it here too rather
                # than re-executing `elect.sync` twice per K block.
                mma_elected: T.uint32
                mma_elected_lane: T.uint32
                T.ptx.elect_sync(mma_elected_lane, mma_elected, T.uint32(0xFFFFFFFF))
                mma_it = 0
                mma_valid = 1
                mma_grp = 0
                mma_cum = 0
                mma_last = 0
                # Scheduler cursor init (`scheduler/gemm.cuh:88`).
                mma_vgrp = 0
                mma_kend = 0
                mma_nxt = 0
                mma_nxtk = 0
                mma_kblocks = 0
                if is_k_grouped:
                    # Start on the first non-empty group (`get_next_k_group`, `:66`).
                    mma_psum = 0
                    mma_nmb = num_m_blocks
                    if is_k_grouped_psum:
                        while mma_grp < num_groups:
                            mma_nxtk = grouped_layout[mma_grp]
                            mma_last = (_uceil(mma_kend, k_alignment)) * k_alignment
                            mma_psum = mma_nxtk - mma_last
                            mma_kend = mma_nxtk
                            if mma_psum > 0:
                                break
                            mma_grp = mma_grp + 1
                    else:
                        while mma_grp < num_groups:
                            mma_psum = grouped_layout[mma_grp]
                            if mma_psum > 0:
                                break
                            mma_grp = mma_grp + 1
                        mma_nxt = mma_grp + 1
                        while mma_nxt < num_groups:
                            mma_nxtk = grouped_layout[mma_nxt]
                            if mma_nxtk > 0:
                                break
                            mma_nxt = mma_nxt + 1
                elif is_m_grouped_psum:
                    mma_psum = grouped_layout[0]
                    mma_nmb = _uceil(mma_psum, block_m)
                else:
                    mma_psum = 0
                    mma_nmb = num_m_blocks
                mma_iter = 0
                while mma_valid == 1:
                    mma_nb = mma_it * num_sms + sm_idx
                    # `get_next_block`: advance the group cursor for this linear block index.
                    mma_done: T.int32
                    mma_done = 0
                    if is_m_grouped_masked:
                        while mma_done == 0:
                            if mma_grp == num_groups:
                                mma_valid = 0
                                mma_done = 1
                            else:
                                mma_nmb = _uceil(grouped_layout[mma_grp], block_m)
                                if mma_nb < (mma_cum + mma_nmb) * num_n_blocks:
                                    mma_done = 1
                                else:
                                    mma_cum = mma_cum + mma_nmb
                                    mma_grp = mma_grp + 1
                    elif is_m_grouped_psum:
                        while mma_done == 0:
                            if mma_nb < (mma_cum + mma_nmb) * num_n_blocks:
                                mma_done = 1
                            else:
                                mma_grp = mma_grp + 1
                                if mma_grp == num_groups:
                                    mma_valid = 0
                                    mma_done = 1
                                else:
                                    mma_last = (_uceil(mma_psum, block_m)) * block_m
                                    mma_psum = grouped_layout[mma_grp]
                                    mma_cum = mma_cum + mma_nmb
                                    mma_nmb = _uceil(mma_psum - mma_last, block_m)
                    elif is_k_grouped:
                        # Walk the valid (non-empty) groups; each contributes `num_blocks`
                        # tiles and its own K extent (`scheduler/gemm.cuh:238`).
                        while mma_done == 0:
                            if mma_grp == num_groups:
                                mma_valid = 0
                                mma_done = 1
                            elif mma_nb < (mma_vgrp + 1) * num_blocks:
                                mma_done = 1
                            else:
                                mma_vgrp = mma_vgrp + 1
                                if is_k_grouped_psum:
                                    mma_grp = mma_grp + 1
                                    while mma_grp < num_groups:
                                        mma_nxtk = grouped_layout[mma_grp]
                                        mma_last = (_uceil(mma_kend, k_alignment)) * k_alignment
                                        mma_psum = mma_nxtk - mma_last
                                        mma_kend = mma_nxtk
                                        if mma_psum > 0:
                                            break
                                        mma_grp = mma_grp + 1
                                else:
                                    mma_last = mma_last + mma_psum
                                    mma_grp = mma_nxt
                                    mma_nxt = mma_nxt + 1
                                    mma_psum = mma_nxtk
                                    while mma_nxt < num_groups:
                                        mma_nxtk = grouped_layout[mma_nxt]
                                        if mma_nxtk > 0:
                                            break
                                        mma_nxt = mma_nxt + 1
                        mma_cum = mma_vgrp * num_m_blocks
                    elif is_batched:
                        if mma_nb >= num_blocks * num_groups:
                            mma_valid = 0
                        else:
                            mma_grp = mma_nb // num_blocks
                            mma_cum = mma_grp * num_m_blocks
                            mma_nmb = num_m_blocks
                    else:
                        if mma_nb >= num_blocks:
                            mma_valid = 0
                    if mma_valid == 1:
                        if use_effective_m:
                            # `get_aligned_effective_m_in_block` for this block (sketch `:559`).
                            mma_eff_m: T.int32
                            mma_m_local: T.int32
                            mma_m_local = _swizzled(
                                mma_nb - mma_cum * num_n_blocks, mma_nmb, num_n_blocks
                            )[0]
                            mma_eff_m = block_m
                            with T.If(mma_m_local == mma_nmb - 1), T.Then():
                                mma_eff_m = (
                                    _uceil(
                                        mma_psum
                                        - (mma_m_local + _udiv(mma_last, block_m)) * block_m,
                                        UMMA_STEP_N,
                                    )
                                    * UMMA_STEP_N
                                )
                            desc_i = T.bitwise_or(
                                T.bitwise_and(desc_i, T.uint32(UMMA_N_FIELD_MASK)),
                                T.shift_left(T.cast(mma_eff_m // 8, "uint32"), T.uint32(17)),
                            )
                        mma_kblocks = _uceil(mma_psum, block_k) if is_k_grouped else num_k_blocks
                        accum_stage: T.int32
                        accum_phase: T.int32
                        accum_stage = mma_iter % NUM_EPILOGUE_STAGES
                        accum_phase = T.bitwise_and(mma_iter // NUM_EPILOGUE_STAGES, 1)
                        mma_tmem_wait: T.uint32
                        mma_tmem_wait = T.uint32(0)
                        while mma_tmem_wait == T.uint32(0):
                            T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                mma_tmem_wait,
                                barriers.ptr_to([tmem_empty_base + accum_stage]),
                                T.cast(T.bitwise_xor(accum_phase, 1), "uint32"),
                                T.uint32(TRY_WAIT_TICKS),
                            )
                        T.ptx.tcgen05.fence__after_thread_sync()

                        mma_k: T.int32
                        mma_k = 0
                        # `#pragma unroll 4` (source `:339`).  Four bodies per back-edge with
                        # `mma_k == 4 * j + u` makes `mma_k % kNumSF?StagesPerLoad` the constant
                        # `u`, which is what removes the scale-factor id arithmetic and the UTCCP
                        # branch from three of every four K blocks.
                        mma_k_rounded: T.int32
                        mma_k_rounded = (
                            (mma_kblocks + (MMA_K_UNROLL - 1)) // MMA_K_UNROLL
                        ) * MMA_K_UNROLL
                        while mma_k < mma_k_rounded:
                            for u in T.unroll(0, MMA_K_UNROLL):
                                with T.If(mma_k < mma_kblocks), T.Then():
                                    mma_sf_wait: T.uint32
                                    mma_sf_wait = T.uint32(0)
                                    while mma_sf_wait == T.uint32(0):
                                        T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                            mma_sf_wait,
                                            barriers.ptr_to([with_sf_base + mma_stage]),
                                            T.cast(mma_phase, "uint32"),
                                            T.uint32(TRY_WAIT_TICKS),
                                        )
                                    T.ptx.tcgen05.fence__after_thread_sync()
                                    a_base_lo: T.uint32
                                    b_base_lo: T.uint32
                                    T.ptx.shfl_sync.idx.b32(
                                        a_base_lo,
                                        a_desc_lo,
                                        T.cast(mma_stage, "uint32"),
                                        T.uint32(0x1F),
                                        T.uint32(0xFFFFFFFF),
                                    )
                                    T.ptx.shfl_sync.idx.b32(
                                        b_base_lo,
                                        b_desc_lo,
                                        T.cast(mma_stage, "uint32"),
                                        T.uint32(0x1F),
                                        T.uint32(0xFFFFFFFF),
                                    )
                                    # One elected lane owns the UTCCP and UMMA issues.  Predicating the
                                    # instructions keeps the warp converged; branching on the elected
                                    # lane instead costs a BSSY/BSYNC pair per K block.
                                    if u % sfa_stages_per_load == 0:
                                        for c in T.unroll(0, num_sfa_chunks):
                                            desc_sf = _rebase(
                                                desc_sf,
                                                sfa_smem_u32
                                                + T.cast(mma_stage * (sf_block_m * 4), "uint32")
                                                + T.uint32(c * NUM_UTCCP_ALIGNED_ELEMS * 4),
                                            )
                                            T.evaluate(
                                                T.ptx[utccp_chain](
                                                    T.cast(
                                                        sfa_tmem.allocated_addr[0] + c * 4, "uint32"
                                                    ),
                                                    desc_sf,
                                                    pred=mma_elected == T.uint32(1),
                                                )
                                            )
                                    if u % sfb_stages_per_load == 0:
                                        for c in T.unroll(0, num_sfb_chunks):
                                            desc_sf = _rebase(
                                                desc_sf,
                                                sfb_smem_u32
                                                + T.cast(mma_stage * (sf_block_n * 4), "uint32")
                                                + T.uint32(c * NUM_UTCCP_ALIGNED_ELEMS * 4),
                                            )
                                            T.evaluate(
                                                T.ptx[utccp_chain](
                                                    T.cast(
                                                        sfb_tmem.allocated_addr[0] + c * 4, "uint32"
                                                    ),
                                                    desc_sf,
                                                    pred=mma_elected == T.uint32(1),
                                                )
                                            )
                                    for ki in T.unroll(0, umma_k_steps):
                                        # `issue_full_k_block` / `issue_tail_k_block`:
                                        # only the leading `ceil_div(remaining_k, UMMA_K)`
                                        # steps of a partial final K block are valid.
                                        with (
                                            T.If(
                                                T.Or(
                                                    mma_k < mma_kblocks - 1,
                                                    ki * UMMA_K
                                                    < (mma_psum if is_k_grouped else eff_k)
                                                    - mma_k * block_k,
                                                )
                                                if may_have_tail_k
                                                else ki < umma_k_steps
                                            ),
                                            T.Then(),
                                        ):
                                            sfa_id: T.uint32
                                            sfb_id: T.uint32
                                            sfa_id = (
                                                T.cast(ki, "uint32")
                                                if sfa_stages_per_load == 1
                                                else T.cast(u % sfa_stages_per_load, "uint32")
                                            )
                                            sfb_id = (
                                                T.cast(ki, "uint32")
                                                if sfb_stages_per_load == 1
                                                else T.cast(u % sfb_stages_per_load, "uint32")
                                            )
                                            rt_desc: T.uint32
                                            rt_desc = _with_sf_id(
                                                desc_i,
                                                sfb_id if swap_ab else sfa_id,
                                                sfa_id if swap_ab else sfb_id,
                                            )
                                            adv_a: T.uint64
                                            adv_b: T.uint64
                                            adv_a = _advance_lo(
                                                desc_a, a_base_lo, ki * a_k_step_units
                                            )
                                            adv_b = _advance_lo(
                                                desc_b, b_base_lo, ki * b_k_step_units
                                            )
                                            T.evaluate(
                                                T.ptx[mma_chain](
                                                    T.cast(accum_stage * umma_n, "uint32"),
                                                    adv_b if swap_ab else adv_a,
                                                    adv_a if swap_ab else adv_b,
                                                    rt_desc,
                                                    T.cast(
                                                        (
                                                            sfb_tmem if swap_ab else sfa_tmem
                                                        ).allocated_addr[0],
                                                        "uint32",
                                                    ),
                                                    T.cast(
                                                        (
                                                            sfa_tmem if swap_ab else sfb_tmem
                                                        ).allocated_addr[0],
                                                        "uint32",
                                                    ),
                                                    T.Or(ki > 0, mma_k > 0),
                                                    pred=mma_elected == T.uint32(1),
                                                )
                                            )
                                    T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
                                    # `tcgen05.commit` implies `fence::before_thread_sync`.
                                    # Same here: predicate the commits on the elected lane rather than
                                    # branching, so the warp never diverges inside the K loop.
                                    if cta_group > 1:
                                        T.evaluate(
                                            T.ptx[commit_mc_chain](
                                                barriers.ptr_to([empty_base + mma_stage]),
                                                T.uint16((1 << cta_group) - 1),
                                                pred=mma_elected == T.uint32(1),
                                            )
                                        )
                                    else:
                                        T.evaluate(
                                            T.ptx[commit_chain](
                                                barriers.ptr_to([empty_base + mma_stage]),
                                                pred=mma_elected == T.uint32(1),
                                            )
                                        )
                                    if cta_group > 1:
                                        T.evaluate(
                                            T.ptx[commit_mc_chain](
                                                barriers.ptr_to([tmem_full_base + accum_stage]),
                                                T.uint16((1 << cta_group) - 1),
                                                pred=T.And(
                                                    mma_elected == T.uint32(1),
                                                    mma_k == mma_kblocks - 1,
                                                ),
                                            )
                                        )
                                    else:
                                        T.evaluate(
                                            T.ptx[commit_chain](
                                                barriers.ptr_to([tmem_full_base + accum_stage]),
                                                pred=T.And(
                                                    mma_elected == T.uint32(1),
                                                    mma_k == mma_kblocks - 1,
                                                ),
                                            )
                                        )
                                    T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
                                    mma_stage = T.Select(mma_stage == stages - 1, 0, mma_stage + 1)
                                    mma_phase = T.bitwise_xor(
                                        mma_phase, T.cast(mma_stage == 0, "int32")
                                    )
                                # Advance unconditionally: the guard only masks the rounded-up tail,
                                # and the loop still has to reach `mma_k_rounded`.
                                mma_k = mma_k + 1
                        mma_iter = mma_iter + 1
                    mma_it = mma_it + 1

                # A 2-CTA cluster needs one more accumulator wait before the
                # barriers can be safely destroyed (source `:426`).
                if cta_group > 1:
                    if mma_iter > 0:
                        mma_end_wait: T.uint32
                        mma_end_wait = T.uint32(0)
                        while mma_end_wait == T.uint32(0):
                            T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                mma_end_wait,
                                barriers.ptr_to(
                                    [tmem_empty_base + (mma_iter - 1) % NUM_EPILOGUE_STAGES]
                                ),
                                T.cast(
                                    T.bitwise_and((mma_iter - 1) // NUM_EPILOGUE_STAGES, 1),
                                    "uint32",
                                ),
                                T.uint32(TRY_WAIT_TICKS),
                            )

        # ===============================================================
        # Role 2: scale-factor transposer warp (source `:432`)
        # ===============================================================

        elif warp == 2:
            tr_stage: T.int32
            tr_phase: T.int32
            tr_stage = 0
            tr_phase = 0
            tr_it: T.int32
            tr_valid: T.int32
            tr_grp: T.int32
            tr_cum: T.int32
            tr_nmb: T.int32
            tr_last: T.int32
            tr_psum: T.int32
            tr_nb: T.int32
            tr_nxt: T.int32
            tr_nxtk: T.int32
            tr_kblocks: T.int32
            tr_vgrp: T.int32
            tr_kend: T.int32
            tr_it = 0
            tr_valid = 1
            tr_grp = 0
            tr_cum = 0
            tr_last = 0
            # Scheduler cursor init (`scheduler/gemm.cuh:88`).
            tr_vgrp = 0
            tr_kend = 0
            tr_nxt = 0
            tr_nxtk = 0
            tr_kblocks = 0
            if is_k_grouped:
                # Start on the first non-empty group (`get_next_k_group`, `:66`).
                tr_psum = 0
                tr_nmb = num_m_blocks
                if is_k_grouped_psum:
                    while tr_grp < num_groups:
                        tr_nxtk = grouped_layout[tr_grp]
                        tr_last = (_uceil(tr_kend, k_alignment)) * k_alignment
                        tr_psum = tr_nxtk - tr_last
                        tr_kend = tr_nxtk
                        if tr_psum > 0:
                            break
                        tr_grp = tr_grp + 1
                else:
                    while tr_grp < num_groups:
                        tr_psum = grouped_layout[tr_grp]
                        if tr_psum > 0:
                            break
                        tr_grp = tr_grp + 1
                    tr_nxt = tr_grp + 1
                    while tr_nxt < num_groups:
                        tr_nxtk = grouped_layout[tr_nxt]
                        if tr_nxtk > 0:
                            break
                        tr_nxt = tr_nxt + 1
            elif is_m_grouped_psum:
                tr_psum = grouped_layout[0]
                tr_nmb = _uceil(tr_psum, block_m)
            else:
                tr_psum = 0
                tr_nmb = num_m_blocks
            sf_vals = T.alloc_local((4,), "uint32")
            while tr_valid == 1:
                tr_nb = tr_it * num_sms + sm_idx
                # `get_next_block`: advance the group cursor for this linear block index.
                tr_done: T.int32
                tr_done = 0
                if is_m_grouped_masked:
                    while tr_done == 0:
                        if tr_grp == num_groups:
                            tr_valid = 0
                            tr_done = 1
                        else:
                            tr_nmb = _uceil(grouped_layout[tr_grp], block_m)
                            if tr_nb < (tr_cum + tr_nmb) * num_n_blocks:
                                tr_done = 1
                            else:
                                tr_cum = tr_cum + tr_nmb
                                tr_grp = tr_grp + 1
                elif is_m_grouped_psum:
                    while tr_done == 0:
                        if tr_nb < (tr_cum + tr_nmb) * num_n_blocks:
                            tr_done = 1
                        else:
                            tr_grp = tr_grp + 1
                            if tr_grp == num_groups:
                                tr_valid = 0
                                tr_done = 1
                            else:
                                tr_last = (_uceil(tr_psum, block_m)) * block_m
                                tr_psum = grouped_layout[tr_grp]
                                tr_cum = tr_cum + tr_nmb
                                tr_nmb = _uceil(tr_psum - tr_last, block_m)
                elif is_k_grouped:
                    # Walk the valid (non-empty) groups; each contributes `num_blocks`
                    # tiles and its own K extent (`scheduler/gemm.cuh:238`).
                    while tr_done == 0:
                        if tr_grp == num_groups:
                            tr_valid = 0
                            tr_done = 1
                        elif tr_nb < (tr_vgrp + 1) * num_blocks:
                            tr_done = 1
                        else:
                            tr_vgrp = tr_vgrp + 1
                            if is_k_grouped_psum:
                                tr_grp = tr_grp + 1
                                while tr_grp < num_groups:
                                    tr_nxtk = grouped_layout[tr_grp]
                                    tr_last = (_uceil(tr_kend, k_alignment)) * k_alignment
                                    tr_psum = tr_nxtk - tr_last
                                    tr_kend = tr_nxtk
                                    if tr_psum > 0:
                                        break
                                    tr_grp = tr_grp + 1
                            else:
                                tr_last = tr_last + tr_psum
                                tr_grp = tr_nxt
                                tr_nxt = tr_nxt + 1
                                tr_psum = tr_nxtk
                                while tr_nxt < num_groups:
                                    tr_nxtk = grouped_layout[tr_nxt]
                                    if tr_nxtk > 0:
                                        break
                                    tr_nxt = tr_nxt + 1
                    tr_cum = tr_vgrp * num_m_blocks
                elif is_batched:
                    if tr_nb >= num_blocks * num_groups:
                        tr_valid = 0
                    else:
                        tr_grp = tr_nb // num_blocks
                        tr_cum = tr_grp * num_m_blocks
                        tr_nmb = num_m_blocks
                else:
                    if tr_nb >= num_blocks:
                        tr_valid = 0
                if tr_valid == 1:
                    tr_kblocks = _uceil(tr_psum, block_k) if is_k_grouped else num_k_blocks
                    tr_k: T.int32
                    tr_k = 0
                    while tr_k < tr_kblocks:
                        tr_wait: T.uint32
                        tr_wait = T.uint32(0)
                        while tr_wait == T.uint32(0):
                            T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                                tr_wait,
                                barriers.ptr_to([full_base + tr_stage]),
                                T.cast(tr_phase, "uint32"),
                                T.uint32(TRY_WAIT_TICKS),
                            )
                        if tr_k % sfa_stages_per_load == 0:
                            for c in T.unroll(0, num_sfa_chunks):
                                base = c * NUM_UTCCP_ALIGNED_ELEMS
                                for i in T.unroll(0, 4):
                                    sf_vals[i] = smem_sfa[tr_stage, base + i * 32 + lane_idx]
                                T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
                                for i in T.unroll(0, 4):
                                    smem_sfa[tr_stage, base + lane_idx * 4 + i] = sf_vals[i]
                            T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                        if tr_k % sfb_stages_per_load == 0:
                            for c in T.unroll(0, num_sfb_chunks):
                                base = c * NUM_UTCCP_ALIGNED_ELEMS
                                for i in T.unroll(0, 4):
                                    sf_vals[i] = smem_sfb[tr_stage, base + i * 32 + lane_idx]
                                T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
                                for i in T.unroll(0, 4):
                                    smem_sfb[tr_stage, base + lane_idx * 4 + i] = sf_vals[i]
                            T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                        # `arrive(0u)` passes a destination CTA rank, not a count:
                        # every thread arrives on the leader CTA's barrier copy.
                        rem = T.alloc_local((1,), "uint64")
                        T.ptx.mapa.shared__cluster.u64(
                            rem[0], barriers.ptr_to([with_sf_base + tr_stage]), T.uint32(0)
                        )
                        T.ptx.mbarrier.arrive.b64(rem[0], T.uint32(1), pred=T.bool(True))
                        tr_k = tr_k + 1
                        tr_stage = T.Select(tr_stage == stages - 1, 0, tr_stage + 1)
                        tr_phase = T.bitwise_xor(tr_phase, T.cast(tr_stage == 0, "int32"))
                tr_it = tr_it + 1

        # ===============================================================
        # Role 3: epilogue warps (source `:470`)
        # ===============================================================

        elif warp >= first_epilogue_warp and warp < first_epilogue_warp + num_store_warps:
            ep_warp: T.int32
            ep_warp = warp - first_epilogue_warp
            tma_stage: T.int32
            ep_it: T.int32
            ep_valid: T.int32
            ep_grp: T.int32
            ep_cum: T.int32
            ep_nmb: T.int32
            ep_last: T.int32
            ep_psum: T.int32
            ep_nb: T.int32
            ep_nxt: T.int32
            ep_nxtk: T.int32
            ep_vgrp: T.int32
            ep_kend: T.int32
            ep_it = 0
            ep_valid = 1
            ep_grp = 0
            ep_cum = 0
            ep_last = 0
            # Scheduler cursor init (`scheduler/gemm.cuh:88`).
            ep_vgrp = 0
            ep_kend = 0
            ep_nxt = 0
            ep_nxtk = 0
            if is_k_grouped:
                # Start on the first non-empty group (`get_next_k_group`, `:66`).
                ep_psum = 0
                ep_nmb = num_m_blocks
                if is_k_grouped_psum:
                    while ep_grp < num_groups:
                        ep_nxtk = grouped_layout[ep_grp]
                        ep_last = (_uceil(ep_kend, k_alignment)) * k_alignment
                        ep_psum = ep_nxtk - ep_last
                        ep_kend = ep_nxtk
                        if ep_psum > 0:
                            break
                        ep_grp = ep_grp + 1
                else:
                    while ep_grp < num_groups:
                        ep_psum = grouped_layout[ep_grp]
                        if ep_psum > 0:
                            break
                        ep_grp = ep_grp + 1
                    ep_nxt = ep_grp + 1
                    while ep_nxt < num_groups:
                        ep_nxtk = grouped_layout[ep_nxt]
                        if ep_nxtk > 0:
                            break
                        ep_nxt = ep_nxt + 1
            elif is_m_grouped_psum:
                ep_psum = grouped_layout[0]
                ep_nmb = _uceil(ep_psum, block_m)
            else:
                ep_psum = 0
                ep_nmb = num_m_blocks
            tma_stage = 0
            values = T.alloc_local((8,), "uint32")
            packed = T.alloc_local((4,), "uint32")
            while ep_valid == 1:
                ep_nb = ep_it * num_sms + sm_idx
                # `get_next_block`: advance the group cursor for this linear block index.
                ep_done: T.int32
                ep_done = 0
                if is_m_grouped_masked:
                    while ep_done == 0:
                        if ep_grp == num_groups:
                            ep_valid = 0
                            ep_done = 1
                        else:
                            ep_nmb = _uceil(grouped_layout[ep_grp], block_m)
                            if ep_nb < (ep_cum + ep_nmb) * num_n_blocks:
                                ep_done = 1
                            else:
                                ep_cum = ep_cum + ep_nmb
                                ep_grp = ep_grp + 1
                elif is_m_grouped_psum:
                    while ep_done == 0:
                        if ep_nb < (ep_cum + ep_nmb) * num_n_blocks:
                            ep_done = 1
                        else:
                            ep_grp = ep_grp + 1
                            if ep_grp == num_groups:
                                ep_valid = 0
                                ep_done = 1
                            else:
                                ep_last = (_uceil(ep_psum, block_m)) * block_m
                                ep_psum = grouped_layout[ep_grp]
                                ep_cum = ep_cum + ep_nmb
                                ep_nmb = _uceil(ep_psum - ep_last, block_m)
                elif is_k_grouped:
                    # Walk the valid (non-empty) groups; each contributes `num_blocks`
                    # tiles and its own K extent (`scheduler/gemm.cuh:238`).
                    while ep_done == 0:
                        if ep_grp == num_groups:
                            ep_valid = 0
                            ep_done = 1
                        elif ep_nb < (ep_vgrp + 1) * num_blocks:
                            ep_done = 1
                        else:
                            ep_vgrp = ep_vgrp + 1
                            if is_k_grouped_psum:
                                ep_grp = ep_grp + 1
                                while ep_grp < num_groups:
                                    ep_nxtk = grouped_layout[ep_grp]
                                    ep_last = (_uceil(ep_kend, k_alignment)) * k_alignment
                                    ep_psum = ep_nxtk - ep_last
                                    ep_kend = ep_nxtk
                                    if ep_psum > 0:
                                        break
                                    ep_grp = ep_grp + 1
                            else:
                                ep_last = ep_last + ep_psum
                                ep_grp = ep_nxt
                                ep_nxt = ep_nxt + 1
                                ep_psum = ep_nxtk
                                while ep_nxt < num_groups:
                                    ep_nxtk = grouped_layout[ep_nxt]
                                    if ep_nxtk > 0:
                                        break
                                    ep_nxt = ep_nxt + 1
                    ep_cum = ep_vgrp * num_m_blocks
                elif is_batched:
                    if ep_nb >= num_blocks * num_groups:
                        ep_valid = 0
                    else:
                        ep_grp = ep_nb // num_blocks
                        ep_cum = ep_grp * num_m_blocks
                        ep_nmb = num_m_blocks
                else:
                    if ep_nb >= num_blocks:
                        ep_valid = 0
                if ep_valid == 1:
                    accum_stage_e: T.int32
                    accum_phase_e: T.int32
                    accum_stage_e = ep_it % NUM_EPILOGUE_STAGES
                    accum_phase_e = T.bitwise_and(ep_it // NUM_EPILOGUE_STAGES, 1)
                    ep_wait: T.uint32
                    ep_wait = T.uint32(0)
                    while ep_wait == T.uint32(0):
                        T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                            ep_wait,
                            barriers.ptr_to([tmem_full_base + accum_stage_e]),
                            T.cast(accum_phase_e, "uint32"),
                            T.uint32(TRY_WAIT_TICKS),
                        )
                    T.ptx.tcgen05.fence__after_thread_sync()
                    # One swizzle walk feeds `base_m`, `base_n` and the tail-block test.
                    ep_m_local, ep_n_local = _swizzled(
                        ep_nb - ep_cum * num_n_blocks, ep_nmb, num_n_blocks
                    )
                    base_m: T.int32
                    base_n: T.int32
                    base_m = (
                        ep_m_local + (_udiv(ep_last, block_m) if is_m_grouped_psum else 0)
                    ) * block_m
                    base_n = ep_n_local * block_n
                    if is_m_grouped_masked or is_k_grouped:
                        base_m = ep_grp * eff_m + base_m
                    tmem_base: T.int32
                    tmem_base = accum_stage_e * umma_n

                    # `num_stores = effective_m / STORE_BLOCK_M` (sketch `:1276`).
                    ep_stores: T.int32
                    if use_effective_m:
                        # `get_aligned_effective_m_in_block` for this block (sketch `:559`).
                        ep_eff_m: T.int32
                        ep_eff_m = block_m
                        with T.If(ep_m_local == ep_nmb - 1), T.Then():
                            ep_eff_m = (
                                _uceil(
                                    ep_psum - (ep_m_local + _udiv(ep_last, block_m)) * block_m,
                                    UMMA_STEP_N,
                                )
                                * UMMA_STEP_N
                            )
                        ep_stores = _udiv(ep_eff_m, store_block_m)
                    else:
                        ep_stores = num_swap_stores

                    if swap_ab:
                        for st in T.unroll(0, num_swap_stores):
                            with T.If(st < ep_stores), T.Then():
                                if ep_warp == 0:
                                    T.evaluate(
                                        T.ptx.cp.async_.bulk.wait_group(NUM_TMA_STORE_STAGES - 1)
                                    )
                                T.ptx.bar.sync(
                                    T.uint32(EPILOGUE_NAMED_BARRIER), T.uint32(num_store_threads)
                                )
                                for i in T.unroll(0, num_atom_rows):
                                    taddr_s: T.uint32
                                    taddr_s = T.cast(
                                        tmem_base + st * store_block_m + i * 8, "uint32"
                                    )
                                    atom_byte = (
                                        ep_warp // warps_per_atom
                                    ) * store_block_m * swizzle_cd + i * 8 * swizzle_cd
                                    if cd_is_fp32:
                                        T.evaluate(
                                            T.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                                                *[values[j] for j in range(8)], taddr_s
                                            )
                                        )
                                        T.ptx.tcgen05.wait__ld.sync.aligned()
                                        col_f: T.int32
                                        col_f = lane_idx // 4
                                        for row in T.unroll(0, 8):
                                            smem_cd_u32[
                                                (
                                                    tma_stage * cd_stage_bytes
                                                    + atom_byte
                                                    + row * (16 * 8)
                                                    + T.bitwise_xor(col_f, row) * 16
                                                    + (lane_idx % 4) * 4
                                                )
                                                // 4
                                            ] = values[row]
                                    else:
                                        # Two `16x256b` slices: the second takes the upper
                                        # 16 rows via bit 20 of the TMEM address.
                                        T.evaluate(
                                            T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                                values[0], values[1], values[2], values[3], taddr_s
                                            )
                                        )
                                        T.evaluate(
                                            T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                                values[4],
                                                values[5],
                                                values[6],
                                                values[7],
                                                T.bitwise_or(taddr_s, T.uint32(0x00100000)),
                                            )
                                        )
                                        T.ptx.tcgen05.wait__ld.sync.aligned()
                                        for j in T.unroll(0, 4):
                                            # `cvt.rn.bf16x2.f32 d, a, b` packs a
                                            # into the UPPER half and b into the
                                            # lower, the reverse of the
                                            # `make_float2(lo, hi)` helper this
                                            # replaces -- hence the swap.
                                            T.ptx.cvt.rn.bf16x2.f32(
                                                packed[j],
                                                T.reinterpret("float32", values[2 * j + 1]),
                                                T.reinterpret("float32", values[2 * j]),
                                            )
                                        row_s: T.int32
                                        col_s: T.int32
                                        row_s = lane_idx % 8
                                        col_s = (ep_warp % 2) * 4 + lane_idx // 8
                                        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                            smem_cd_u32.ptr_to(
                                                [
                                                    (
                                                        tma_stage * cd_stage_bytes
                                                        + atom_byte
                                                        + row_s * (16 * 8)
                                                        + T.bitwise_xor(col_s, row_s) * 16
                                                    )
                                                    // 4
                                                ]
                                            ),
                                            packed[0],
                                            packed[1],
                                            packed[2],
                                            packed[3],
                                        )

                                if st == ep_stores - 1:
                                    T.ptx.tcgen05.fence__before_thread_sync()
                                    rem_s = T.alloc_local((1,), "uint64")
                                    T.ptx.mapa.shared__cluster.u64(
                                        rem_s[0],
                                        barriers.ptr_to([tmem_empty_base + accum_stage_e]),
                                        T.uint32(0),
                                    )
                                    T.ptx.mbarrier.arrive.b64(
                                        rem_s[0], T.uint32(1), pred=T.bool(True)
                                    )

                                T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                                T.ptx.bar.sync(
                                    T.uint32(EPILOGUE_NAMED_BARRIER), T.uint32(num_store_threads)
                                )
                                if ep_warp == 0:
                                    # The store is issued by one elected lane; predicate rather than
                                    # branch so the epilogue warp stays converged.
                                    ep_elected: T.uint32
                                    ep_elected_lane: T.uint32
                                    T.ptx.elect_sync(
                                        ep_elected_lane, ep_elected, T.uint32(0xFFFFFFFF)
                                    )
                                    for i in T.unroll(0, num_n_atoms):
                                        T.evaluate(
                                            T.ptx[
                                                reduce_chain if with_accumulation else store_chain
                                            ](
                                                T.address_of(tensor_map_cd),
                                                *_tma_coords(
                                                    (
                                                        base_n + i * store_block_n_atom,
                                                        base_m + st * store_block_m,
                                                    ),
                                                    ep_grp,
                                                ),
                                                smem_cd_u32.ptr_to(
                                                    [
                                                        (
                                                            tma_stage * cd_stage_bytes
                                                            + i * store_block_m * swizzle_cd
                                                        )
                                                        // 4
                                                    ]
                                                ),
                                                pred=ep_elected == T.uint32(1),
                                            )
                                        )
                                    T.evaluate(T.ptx.cp.async_.bulk.commit_group())
                                T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
                                tma_stage = T.Select(
                                    tma_stage == NUM_TMA_STORE_STAGES - 1, 0, tma_stage + 1
                                )
                    else:
                        for w in T.unroll(0, num_m_waves):
                            for st in T.unroll(0, num_stores):
                                if ep_warp == 0:
                                    T.evaluate(
                                        T.ptx.cp.async_.bulk.wait_group(NUM_TMA_STORE_STAGES - 1)
                                    )
                                T.ptx.bar.sync(
                                    T.uint32(EPILOGUE_NAMED_BARRIER), T.uint32(num_store_threads)
                                )
                                for i in T.unroll(0, elems_per_store):
                                    bank_group = i + lane_idx * (swizzle_cd // 16)
                                    row = (i // 8 + lane_idx) if has_shortcut else (bank_group // 8)
                                    col = i if has_shortcut else (bank_group % 8)
                                    col = T.bitwise_xor(col, row % (swizzle_cd // 16))
                                    # `smem_ptr` in the source: one address per bank
                                    # group, four registers stored at it.
                                    cd_word: T.int32
                                    cd_word = _cd_word(tma_stage, ep_warp, row, col, 0)
                                    taddr: T.uint32
                                    taddr = T.cast(
                                        tmem_base
                                        + w * block_n
                                        + st * store_block_n
                                        + i * elems_per_bank_group,
                                        "uint32",
                                    )
                                    if cd_is_fp32:
                                        T.evaluate(
                                            T.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
                                                values[0], values[1], values[2], values[3], taddr
                                            )
                                        )
                                        T.ptx.tcgen05.wait__ld.sync.aligned()
                                        for j in T.unroll(0, 4):
                                            smem_cd_u32[cd_word + j] = values[j]
                                    else:
                                        T.evaluate(
                                            T.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                                                *[values[j] for j in range(8)], taddr
                                            )
                                        )
                                        T.ptx.tcgen05.wait__ld.sync.aligned()
                                        for j in T.unroll(0, 4):
                                            # `cvt.rn.bf16x2.f32 d, a, b` packs a
                                            # into the UPPER half and b into the
                                            # lower, the reverse of the
                                            # `make_float2(lo, hi)` helper this
                                            # replaces -- hence the swap.
                                            T.ptx.cvt.rn.bf16x2.f32(
                                                packed[j],
                                                T.reinterpret("float32", values[2 * j + 1]),
                                                T.reinterpret("float32", values[2 * j]),
                                            )
                                        for j in T.unroll(0, 4):
                                            smem_cd_u32[cd_word + j] = packed[j]

                                if w == num_m_waves - 1 and st == num_stores - 1:
                                    T.ptx.tcgen05.fence__before_thread_sync()
                                    rem_e = T.alloc_local((1,), "uint64")
                                    T.ptx.mapa.shared__cluster.u64(
                                        rem_e[0],
                                        barriers.ptr_to([tmem_empty_base + accum_stage_e]),
                                        T.uint32(0),
                                    )
                                    T.ptx.mbarrier.arrive.b64(
                                        rem_e[0], T.uint32(1), pred=T.bool(True)
                                    )

                                T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                                T.ptx.bar.sync(
                                    T.uint32(EPILOGUE_NAMED_BARRIER), T.uint32(num_store_threads)
                                )
                                if ep_warp == 0:
                                    # The store is issued by one elected lane; predicate rather than
                                    # branch so the epilogue warp stays converged.
                                    ep_elected: T.uint32
                                    ep_elected_lane: T.uint32
                                    T.ptx.elect_sync(
                                        ep_elected_lane, ep_elected, T.uint32(0xFFFFFFFF)
                                    )
                                    T.evaluate(
                                        T.ptx[reduce_chain if with_accumulation else store_chain](
                                            T.address_of(tensor_map_cd),
                                            *_tma_coords(
                                                (
                                                    base_n + st * store_block_n,
                                                    base_m + w * store_block_m,
                                                ),
                                                ep_grp,
                                            ),
                                            smem_cd.ptr_to([tma_stage, 0, 0]),
                                            pred=ep_elected == T.uint32(1),
                                        )
                                    )
                                    T.evaluate(T.ptx.cp.async_.bulk.commit_group())
                                T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
                                tma_stage = T.Select(
                                    tma_stage == NUM_TMA_STORE_STAGES - 1, 0, tma_stage + 1
                                )
                ep_it = ep_it + 1

        # ===============================================================
        # Teardown (source `:524`)
        # ===============================================================

        if cta_group > 1:
            T.ptx.barrier.cluster.arrive.relaxed.aligned()
            T.ptx.barrier.cluster.wait.acquire.aligned()
        else:
            T.ptx.bar.sync(T.uint32(0))

        # The allocating warp (2) and the freeing warp (0) deliberately differ.
        if warp == 0:
            T.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                T.uint32(0), T.uint32(spec.num_tmem_cols)
            )

    return sm100_fp8_fp4_gemm_1d1d


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
