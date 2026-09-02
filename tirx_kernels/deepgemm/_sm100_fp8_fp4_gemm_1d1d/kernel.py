# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""The kernel body: eight warps, five barrier families, one persistent walk.

K owns the launch and role partition, typed shared-memory and barrier storage,
the reusable persistent scheduler, and each role's pipeline cursors.  Kernel-
specific descriptor construction, instruction order, cluster signalling and
tensor-memory operations remain explicit low-level TIRx/PTX.

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

import tirx_kernels.kern as K
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)

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

#: Source-matching time hint for the shared barrier retry loop.
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
    return K.cast((K.cast(x, "uint32") + K.uint32(d - 1)) // K.uint32(d), "int32")


def _load_grouped_layout(dst, grouped_layout, index):
    return K.ptx.ld.global_.s32(dst, grouped_layout.ptr_to([index]))


def _udiv(x, d):
    """Exact division of a known non-negative value; see `_uceil`."""
    return K.cast(K.cast(x, "uint32") // K.cast(d, "uint32"), "int32")


def _umod(x, d):
    """Remainder of a known non-negative value; see `_uceil`."""
    return K.cast(K.cast(x, "uint32") % K.cast(d, "uint32"), "int32")


def _wait_barrier(barrier, phase):
    complete = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(complete == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            complete, barrier, K.cast(phase, "uint32"), K.uint32(TRY_WAIT_TICKS)
        )


class _PersistentScheduler:
    """One role-local walk over DeepGEMM's persistent block sequence.

    Load, MMA, transpose, and epilogue roles intentionally own independent
    cursors, but they execute the same scheduler contract. Keeping that
    contract here makes the cursor state and transitions one source of truth;
    no communication or additional synchronization is introduced.
    """

    def __init__(
        self, spec, grouped_layout, sm_idx, num_m_blocks, num_n_blocks, num_blocks, *, track_sfk
    ):
        self.grouped_layout = grouped_layout
        self.sm_idx = sm_idx
        self.num_m_blocks = num_m_blocks
        self.num_n_blocks = num_n_blocks
        self.num_blocks = num_blocks
        self.num_sms = spec.num_sms
        self.num_groups = spec.num_groups
        self.block_m = spec.block_m
        self.k_alignment = spec.k_alignment
        self.sf_k_span = spec.gran_k_a * 4
        self.is_m_grouped_masked = spec.gemm_type is GemmType.M_GROUPED_MASKED
        self.is_m_grouped_psum = spec.gemm_type is GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT
        self.is_k_grouped = spec.gemm_type.is_k_grouped_contiguous
        self.is_k_grouped_psum = spec.gemm_type is GemmType.K_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT
        self.is_batched = spec.gemm_type is GemmType.BATCHED
        self.track_sfk = track_sfk

        self.it = K.local_scalar("int32")
        self.valid = K.local_scalar("int32")
        self.grp = K.local_scalar("int32")
        self.cum = K.local_scalar("int32")
        self.nmb = K.local_scalar("int32")
        self.last = K.local_scalar("int32")
        self.psum = K.local_scalar("int32")
        self.nb = K.local_scalar("int32")
        self.nxt = K.local_scalar("int32")
        self.nxtk = K.local_scalar("int32")
        self.sfk = K.local_scalar("int32")
        self.vgrp = K.local_scalar("int32")
        self.kend = K.local_scalar("int32")
        self.init()

    def init(self):
        K.assign(self.it, 0)
        K.assign(self.valid, 1)
        K.assign(self.grp, 0)
        K.assign(self.cum, 0)
        K.assign(self.last, 0)
        K.assign(self.sfk, 0)
        K.assign(self.vgrp, 0)
        K.assign(self.kend, 0)
        K.assign(self.nxt, 0)
        K.assign(self.nxtk, 0)
        if self.is_k_grouped:
            K.assign(self.psum, 0)
            K.assign(self.nmb, self.num_m_blocks)
            if self.is_k_grouped_psum:
                with K.While(self.grp < self.num_groups):
                    _load_grouped_layout(self.nxtk, self.grouped_layout, self.grp)
                    K.assign(self.last, _uceil(self.kend, self.k_alignment) * self.k_alignment)
                    K.assign(self.psum, self.nxtk - self.last)
                    K.assign(self.kend, self.nxtk)
                    with K.If(self.psum > 0):
                        with K.Then():
                            K.Break()
                    K.assign(self.grp, self.grp + 1)
            else:
                with K.While(self.grp < self.num_groups):
                    _load_grouped_layout(self.psum, self.grouped_layout, self.grp)
                    with K.If(self.psum > 0):
                        with K.Then():
                            K.Break()
                    K.assign(self.grp, self.grp + 1)
                K.assign(self.nxt, self.grp + 1)
                with K.While(self.nxt < self.num_groups):
                    _load_grouped_layout(self.nxtk, self.grouped_layout, self.nxt)
                    with K.If(self.nxtk > 0):
                        with K.Then():
                            K.Break()
                    K.assign(self.nxt, self.nxt + 1)
        else:
            if self.is_m_grouped_psum:
                _load_grouped_layout(self.psum, self.grouped_layout, 0)
                K.assign(self.nmb, _uceil(self.psum, self.block_m))
            else:
                K.assign(self.psum, 0)
                K.assign(self.nmb, self.num_m_blocks)

    def next(self):
        K.assign(self.nb, self.it * self.num_sms + self.sm_idx)
        done = K.local_scalar("int32", init=0)
        if self.is_m_grouped_masked:
            with K.While(done == 0):
                with K.If(self.grp == self.num_groups):
                    with K.Then():
                        K.assign(self.valid, 0)
                        K.assign(done, 1)
                    with K.Else():
                        _load_grouped_layout(self.nmb, self.grouped_layout, self.grp)
                        K.assign(self.nmb, _uceil(self.nmb, self.block_m))
                        with K.If(self.nb < (self.cum + self.nmb) * self.num_n_blocks):
                            with K.Then():
                                K.assign(done, 1)
                            with K.Else():
                                K.assign(self.cum, self.cum + self.nmb)
                                K.assign(self.grp, self.grp + 1)
        else:
            if self.is_m_grouped_psum:
                with K.While(done == 0):
                    with K.If(self.nb < (self.cum + self.nmb) * self.num_n_blocks):
                        with K.Then():
                            K.assign(done, 1)
                        with K.Else():
                            K.assign(self.grp, self.grp + 1)
                            with K.If(self.grp == self.num_groups):
                                with K.Then():
                                    K.assign(self.valid, 0)
                                    K.assign(done, 1)
                                with K.Else():
                                    K.assign(
                                        self.last, (_uceil(self.psum, self.block_m) * self.block_m)
                                    )
                                    _load_grouped_layout(self.psum, self.grouped_layout, self.grp)
                                    K.assign(self.cum, self.cum + self.nmb)
                                    K.assign(self.nmb, _uceil(self.psum - self.last, self.block_m))
            else:
                if self.is_k_grouped:
                    with K.While(done == 0):
                        with K.If(self.grp == self.num_groups):
                            with K.Then():
                                K.assign(self.valid, 0)
                                K.assign(done, 1)
                            with K.Else():
                                with K.If(self.nb < (self.vgrp + 1) * self.num_blocks):
                                    with K.Then():
                                        K.assign(done, 1)
                                    with K.Else():
                                        if self.track_sfk:
                                            K.assign(
                                                self.sfk,
                                                self.sfk + _uceil(self.psum, self.sf_k_span),
                                            )
                                        K.assign(self.vgrp, self.vgrp + 1)
                                        if self.is_k_grouped_psum:
                                            K.assign(self.grp, self.grp + 1)
                                            with K.While(self.grp < self.num_groups):
                                                _load_grouped_layout(
                                                    self.nxtk, self.grouped_layout, self.grp
                                                )
                                                K.assign(
                                                    self.last,
                                                    (
                                                        _uceil(self.kend, self.k_alignment)
                                                        * self.k_alignment
                                                    ),
                                                )
                                                K.assign(self.psum, self.nxtk - self.last)
                                                K.assign(self.kend, self.nxtk)
                                                with K.If(self.psum > 0):
                                                    with K.Then():
                                                        K.Break()
                                                K.assign(self.grp, self.grp + 1)
                                        else:
                                            K.assign(self.last, self.last + self.psum)
                                            K.assign(self.grp, self.nxt)
                                            K.assign(self.nxt, self.nxt + 1)
                                            K.assign(self.psum, self.nxtk)
                                            with K.While(self.nxt < self.num_groups):
                                                _load_grouped_layout(
                                                    self.nxtk, self.grouped_layout, self.nxt
                                                )
                                                with K.If(self.nxtk > 0):
                                                    with K.Then():
                                                        K.Break()
                                                K.assign(self.nxt, self.nxt + 1)
                    K.assign(self.cum, self.vgrp * self.num_m_blocks)
                else:
                    if self.is_batched:
                        with K.If(self.nb >= self.num_blocks * self.num_groups):
                            with K.Then():
                                K.assign(self.valid, 0)
                            with K.Else():
                                K.assign(self.grp, self.nb // self.num_blocks)
                                K.assign(self.cum, self.grp * self.num_m_blocks)
                                K.assign(self.nmb, self.num_m_blocks)
                    else:
                        with K.If(self.nb >= self.num_blocks):
                            with K.Then():
                                K.assign(self.valid, 0)

    def advance(self):
        K.assign(self.it, self.it + 1)


def build_kernel(spec: GemmSpec):
    """Build the TIRx `PrimFunc` for one `sm100_fp8_fp4_gemm_1d1d_impl` instantiation."""
    from tvm.ir.type import PointerType, PrimType

    _validate(spec)

    def _u64_const(value):
        """A `uint64` literal whose top bit may be set.

        `K.uint64` routes a Python int through an int64 conversion, so a value
        at or above 2**63 does not survive it -- and a descriptor whose swizzle
        puts `layout_type_` at bits [61,64) is exactly that. Assembling it from
        two 32-bit halves does; the compiler folds it straight back.
        """
        return K.bitwise_or(
            K.shift_left(K.uint64((value >> 32) & 0xFFFFFFFF), K.uint64(32)),
            K.uint64(value & 0xFFFFFFFF),
        )

    def _smem_addr_field(addr_u32):
        """Bits [13:0] of a matrix descriptor: the shared address over 16."""
        return K.cast(
            K.bitwise_and(K.shift_right(addr_u32, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
        )

    def _with_smem_addr(base, addr_u32):
        """Put an address into a descriptor base whose address field is zero."""
        return K.bitwise_or(base, _smem_addr_field(addr_u32))

    def _rebase(desc, addr_u32):
        """`replace_smem_desc_addr`: rewrite bits [13:0] with `smem_addr >> 4`."""
        return K.bitwise_or(
            K.bitwise_and(desc, K.bitwise_not(K.uint64(0x3FFF))), _smem_addr_field(addr_u32)
        )

    def _with_sf_id(desc, sfa_id, sfb_id):
        """`make_runtime_instr_desc_with_sf_id`: fields [31:29] and [6:4]."""
        out = K.bitwise_and(desc, K.uint32(0x9FFFFFCF))
        out = K.bitwise_or(out, K.shift_left(K.cast(sfa_id, "uint32"), K.uint32(29)))
        return K.bitwise_or(out, K.shift_left(K.cast(sfb_id, "uint32"), K.uint32(4)))

    def _advance_lo(desc, base_lo, units):
        """`advance_umma_desc_lo`: replace the low word with `base_lo + units`."""
        return K.bitwise_or(
            K.bitwise_and(desc, K.shift_left(K.uint64(0xFFFFFFFF), K.uint64(32))),
            K.cast(base_lo + K.uint32(units), "uint64"),
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

    swizzle_a = K.SwizzleMode(_swizzle_enum(spec.swizzle_a_mode))
    swizzle_b = K.SwizzleMode(_swizzle_enum(spec.swizzle_b_mode))

    def _operand_desc_sdo(is_k_major, load_mn, swizzle_mode, elem):
        """Descriptor stride offset for one K-owned operand tile (`mma/sm100.cuh:107`).

        K allocates `(stage, MN, K)` for K-major and `(stage, K, MN)` for
        MN-major. Only `sdo` reaches the hand-folded descriptor; every in-scope
        MN-major tile has one swizzle atom, so its leading offset remains zero.
        """
        if is_k_major:
            # `kSwizzleMode * pack == BLOCK_K * sizeof` is asserted upstream, so
            # each block holds exactly one swizzle atom along K.
            sdo = 8 * block_k * elem // 16
        else:
            atom = swizzle_mode // elem if swizzle_mode else load_mn
            sdo = 8 * atom * elem // 16
            ldo = block_k * atom * elem // 16
            if swizzle_mode == 16:
                sdo = ldo
        return sdo

    a_desc_sdo = _operand_desc_sdo(major_a_is_k, load_block_m, spec.swizzle_a_mode, 1)
    b_desc_sdo = _operand_desc_sdo(major_b_is_k, load_block_n, spec.swizzle_b_mode, 1)
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

    # Batched is the only type whose A/B/C-D descriptors are rank 3.
    rank = 3 if is_batched else 2
    mma_chain = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.scale_vec::1X"
    utccp_chain = f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
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

    def _next_tma_store_stage(stage):
        """Advance the phase-free TMA store-group ring without extra state."""
        return K.Select(stage == NUM_TMA_STORE_STAGES - 1, 0, stage + 1)

    def _advance_pipeline(state, depth):
        """Advance with the source kernel's branchless stage/phase lowering."""
        K.assign(state.stage, K.Select(state.stage == depth - 1, 0, state.stage + 1))
        K.assign(state.phase, K.bitwise_xor(state.phase, K.cast(state.stage == 0, "int32")))

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
        in_group_blocks = K.min(blocks_per_group, primary - first_block)
        if is_multicast_on_a:
            return _udiv(in_group, in_group_blocks), first_block + _umod(in_group, in_group_blocks)
        return first_block + _umod(in_group, in_group_blocks), _udiv(in_group, in_group_blocks)

    def _tma_coords(coords, batch):
        """TMA tensor coordinates, batch index appended for a 3-D descriptor.

        `is_batched` is the only thing that changes the arity of every
        `cp.async.bulk.tensor` in this kernel, so it is resolved here once
        rather than by duplicating each call site.
        """
        out = [K.cast(c, "int32") for c in coords]
        if is_batched:
            out.append(K.cast(batch, "int32"))
        return out

    total_warps = (spec.num_non_epilogue_threads + spec.num_epilogue_threads) // 32

    @K.kernel(warps=total_warps, arch="sm_100a", min_blocks_per_sm=1, grid=spec.num_sms)
    def sm100_fp8_fp4_gemm_1d1d(
        grouped_layout: K.gptr[K.i32],
        grouped_len: K.i32,
        shape_m: K.i32,
        shape_n: K.i32,
        shape_k: K.i32,
        tensor_map_a: K.TensorMap,
        tensor_map_b: K.TensorMap,
        tensor_map_sfa: K.TensorMap,
        tensor_map_sfb: K.TensorMap,
        tensor_map_cd: K.TensorMap,
    ):
        # ---- role ids -------------------------------------------------------
        sm_idx = K.cta_id()
        if cta_group > 1:
            cta_in_cluster = K.cta_id_in_cluster([cta_group])
            is_leader_cta = cta_in_cluster == 0
        else:
            is_leader_cta = True
        thread_idx = K.thread_id()
        lane_idx = K.lane_id()

        roles = K.specialize(chain_dispatch=True)
        load_role = roles.role("load", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        transpose_role = roles.role("transpose", warps=[2])
        roles.role("idle", warps=[3])
        epilogue_role = roles.role(
            "epilogue", warps=range(first_epilogue_warp, first_epilogue_warp + num_store_warps)
        )

        # ---- shared memory --------------------------------------------------
        smem = K.smem_pool()
        # C/D is a linear view: the swizzle lives in the TensorMap and in the
        # epilogue's explicit bank-group arithmetic.
        smem_cd = smem.alloc(
            (NUM_TMA_STORE_STAGES, store_block_m, store_block_n), cd_dtype, align=1024
        )
        smem_a_tile = smem.alloc(
            (stages, load_block_m, block_k) if major_a_is_k else (stages, block_k, load_block_m),
            a_smem_dtype,
            align=1024,
            swizzle=swizzle_a,
        )
        smem_b_tile = smem.alloc(
            (stages, load_block_n, block_k) if major_b_is_k else (stages, block_k, load_block_n),
            b_smem_dtype,
            align=1024,
            swizzle=swizzle_b,
        )
        smem_sfa = smem.alloc((stages, sf_block_m), K.u32, align=16)
        smem_sfb = smem.alloc((stages, sf_block_n), K.u32, align=16)
        full_barriers = K.TMABar(smem, stages)
        empty_barriers = K.TCGen05Bar(smem, stages)
        with_sf_barriers = K.MBarrier(smem, stages)
        tmem_full_barriers = K.TCGen05Bar(smem, NUM_EPILOGUE_STAGES)
        tmem_empty_barriers = K.MBarrier(smem, NUM_EPILOGUE_STAGES)
        tmem_slot = smem.alloc((1,), K.u32, align=4)
        if smem.bytes != spec.smem_tmem_ptr_offset + 4:
            raise ValueError(
                f"K SMEM layout ends at {smem.bytes}, expected {spec.smem_tmem_ptr_offset + 4}"
            )
        smem.commit(spec.smem_size)

        smem_cd_word_data = K.reinterpret(
            PointerType(PrimType("uint32")), smem_cd.ptr_to([0, 0, 0])
        )
        smem_cd_u32 = K.decl_buffer(
            (NUM_TMA_STORE_STAGES * cd_stage_bytes // 4,),
            "uint32",
            data=smem_cd_word_data,
            scope="shared.dyn",
            elem_offset=0,
            align=1024,
        )

        # ---- tensor memory --------------------------------------------------
        # TMEM has no ordinary buffer view in Kern.  D's base is the runtime
        # allocation mailbox; the scale-factor operands use fixed columns from
        # the specialization.
        tmem_col = tmem_slot[0]
        sfa_tmem_col = spec.tmem_start_col_of_sfa
        sfb_tmem_col = spec.tmem_start_col_of_sfb

        # ---- cluster rendezvous before the 2-CTA TMEM allocation -------------
        if cta_group > 1:
            K.ptx.barrier.cluster.arrive.relaxed.aligned()
            K.ptx.barrier.cluster.wait.acquire.aligned()

        with load_role:
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_a))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_b))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_sfa))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_sfb))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_cd))

        # ---- barrier init and TMEM allocation -------------------------------
        def init_barriers():
            init_elected = K.local_scalar("uint32")
            init_elected_lane = K.local_scalar("uint32")
            K.ptx.elect_sync(init_elected_lane, init_elected, K.uint32(0xFFFFFFFF))
            with K.If(init_elected == K.uint32(1)):
                with K.Then():
                    with K.unroll(0, stages) as s:
                        K.ptx.mbarrier.init.shared.b64(full_barriers.ptr_to([s]), K.uint32(1))
                        K.ptx.mbarrier.init.shared.b64(empty_barriers.ptr_to([s]), K.uint32(1))
                        K.ptx.mbarrier.init.shared.b64(
                            with_sf_barriers.ptr_to([s]), K.uint32(cta_group * 32)
                        )
                    with K.unroll(0, NUM_EPILOGUE_STAGES) as e:
                        K.ptx.mbarrier.init.shared.b64(tmem_full_barriers.ptr_to([e]), K.uint32(1))
                        K.ptx.mbarrier.init.shared.b64(
                            tmem_empty_barriers.ptr_to([e]), K.uint32(cta_group * num_store_threads)
                        )
                    K.ptx.fence.mbarrier_init.release.cluster()

        with mma_role:
            init_barriers()
        with transpose_role:
            K.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                K.address_of(tmem_slot[0]), K.uint32(spec.num_tmem_cols)
            )

        if cta_group > 1:
            K.ptx.barrier.cluster.arrive.relaxed.aligned()
            K.ptx.barrier.cluster.wait.acquire.aligned()
        else:
            K.ptx.bar.sync(K.uint32(0))

        K.ptx.griddepcontrol.wait()

        # ===============================================================
        # Persistent scheduler (source `scheduler/gemm.cuh`)
        # ===============================================================
        # Each scheduler-driving role keeps its own copy of this state and walks
        # the identical block sequence; that is what keeps the four roles'
        # `stage_idx`/`phase` cursors in lockstep without any extra barrier.

        # `compiled_dims` bakes a dimension in; the choice happens in Python, so a
        # baked dimension becomes a literal and its runtime argument goes dead
        # (source `:123-125`).  The parameter cannot be reassigned -- TVMScript
        # forbids shadowing a parameter name.
        eff_m = spec.shape_m if spec.shape_m > 0 else shape_m
        eff_n = spec.shape_n if spec.shape_n > 0 else shape_n
        eff_k = spec.shape_k if spec.shape_k > 0 else shape_k

        num_m_blocks = K.local_scalar("int32")
        num_n_blocks = K.local_scalar("int32")
        num_blocks = K.local_scalar("int32")
        num_k_blocks = K.local_scalar("int32")
        K.assign(num_m_blocks, _uceil(eff_m, block_m))
        K.assign(num_n_blocks, (eff_n + (block_n - 1)) // block_n)
        K.assign(num_blocks, num_m_blocks * num_n_blocks)
        K.assign(num_k_blocks, _uceil(eff_k, block_k))
        shape_sfa_k = K.local_scalar("int32")
        shape_sfb_k = K.local_scalar("int32")
        K.assign(shape_sfa_k, (eff_k + (spec.gran_k_a * 4 - 1)) // (spec.gran_k_a * 4))
        K.assign(shape_sfb_k, (eff_k + (spec.gran_k_b * 4 - 1)) // (spec.gran_k_b * 4))

        # ===============================================================
        # Role 0: TMA load warp, one elected lane (source `:206`)
        # ===============================================================

        def load_role_body():
            ld_elected = K.local_scalar("uint32")
            ld_elected_lane = K.local_scalar("uint32")
            K.ptx.elect_sync(ld_elected_lane, ld_elected, K.uint32(0xFFFFFFFF))
            with K.If(ld_elected == K.uint32(1)):
                with K.Then():
                    ld_pipe = K.PipelineState(stages, phase=0)
                    ld_sched = _PersistentScheduler(
                        spec,
                        grouped_layout,
                        sm_idx,
                        num_m_blocks,
                        num_n_blocks,
                        num_blocks,
                        track_sfk=True,
                    )
                    ld_kblocks = K.local_scalar("int32", init=0)
                    with K.While(ld_sched.valid == 1):
                        ld_sched.next()
                        with K.If(ld_sched.valid == 1):
                            with K.Then():
                                K.assign(
                                    ld_kblocks,
                                    (
                                        _uceil(ld_sched.psum, block_k)
                                        if is_k_grouped
                                        else num_k_blocks
                                    ),
                                )
                                # One swizzle walk feeds `m_idx`, `n_idx` and the tail-block
                                # test below, as the source's single `get_swizzled_block_idx`
                                # call does (`scheduler/gemm.cuh:197`).
                                ld_m_local, ld_n_local = _swizzled(
                                    ld_sched.nb - ld_sched.cum * num_n_blocks,
                                    ld_sched.nmb,
                                    num_n_blocks,
                                )
                                m_idx = K.local_scalar("int32")
                                n_idx = K.local_scalar("int32")
                                K.assign(
                                    m_idx,
                                    (
                                        ld_m_local
                                        + (
                                            _udiv(ld_sched.last, block_m)
                                            if is_m_grouped_psum
                                            else 0
                                        )
                                    )
                                    * block_m,
                                )
                                K.assign(n_idx, ld_n_local * block_n)
                                sfa_mn = K.local_scalar("int32")
                                sfb_mn = K.local_scalar("int32")
                                # The scale factors are indexed by the *whole* block, before
                                # the cluster split below.
                                K.assign(sfa_mn, m_idx)
                                K.assign(sfb_mn, n_idx)
                                # `get_global_idx` carries the expert offset on whichever of
                                # B's axes the group was folded into: N for a K-major B,
                                # K for an MN-major one (source `:223`, `:233`, `:271`).
                                k_b_idx = K.local_scalar("int32")
                                k_a_offset = K.local_scalar("int32")
                                K.assign(k_b_idx, 0)
                                K.assign(k_a_offset, 0)
                                sfa_k_offset = K.local_scalar("int32", init=0)
                                if is_m_grouped_contiguous:
                                    # Non-psum contiguous: the expert is a per-row id.
                                    expert = K.local_scalar("int32")
                                    _load_grouped_layout(expert, grouped_layout, m_idx)
                                    K.assign(expert, K.max(0, expert))
                                    if major_b_is_k:
                                        K.assign(n_idx, expert * eff_n + n_idx)
                                    else:
                                        K.assign(k_b_idx, expert * eff_k + k_b_idx)
                                    sfb_k_offset = expert * shape_sfb_k
                                else:
                                    if is_m_grouped_masked:
                                        # Masked: A, B, SFA and SFB are all `[G, ...]` slabs, so every axis
                                        # carries the group offset (source `:221-234`, `:264-272`).
                                        K.assign(m_idx, ld_sched.grp * eff_m + m_idx)
                                        if major_b_is_k:
                                            K.assign(n_idx, ld_sched.grp * eff_n + n_idx)
                                        else:
                                            K.assign(k_b_idx, ld_sched.grp * eff_k + k_b_idx)
                                        K.assign(sfa_k_offset, ld_sched.grp * shape_sfa_k)
                                        sfb_k_offset = ld_sched.grp * shape_sfb_k
                                    else:
                                        if is_m_grouped_psum:
                                            # PSUM: A is one flat `[M, K]`; only B and SFB are grouped.
                                            if major_b_is_k:
                                                K.assign(n_idx, ld_sched.grp * eff_n + n_idx)
                                            else:
                                                K.assign(k_b_idx, (ld_sched.grp * eff_k + k_b_idx))
                                            sfb_k_offset = ld_sched.grp * shape_sfb_k
                                        else:
                                            if is_k_grouped:
                                                # Groups are concatenated along K; both operands are MN-major, so the
                                                # K index carries the running offset and the SF index its own
                                                # cumulative row count (source `:166-180`).
                                                K.assign(k_a_offset, ld_sched.last)
                                                K.assign(k_b_idx, ld_sched.last)
                                                K.assign(sfa_k_offset, ld_sched.sfk)
                                                sfb_k_offset = ld_sched.sfk
                                            else:
                                                if is_batched:
                                                    # Batched: the batch index rides the SF outer extent (source `:183`).
                                                    K.assign(
                                                        sfa_k_offset, (ld_sched.grp * shape_sfa_k)
                                                    )
                                                    sfb_k_offset = ld_sched.grp * shape_sfb_k
                                                else:
                                                    sfb_k_offset = 0
                                # Each CTA of a cluster loads its own slice; the 2-CTA
                                # behaviour lives here and in the UMMA, not in a multicast
                                # TMA (source `:237-240`).
                                if use_effective_m:
                                    # `get_aligned_effective_m_in_block` for this block (sketch `:559`).
                                    ld_eff_m = K.local_scalar("int32", init=block_m)
                                    with K.If(ld_m_local == ld_sched.nmb - 1), K.Then():
                                        K.assign(
                                            ld_eff_m,
                                            (
                                                _uceil(
                                                    ld_sched.psum
                                                    - (ld_m_local + _udiv(ld_sched.last, block_m))
                                                    * block_m,
                                                    UMMA_STEP_N,
                                                )
                                                * UMMA_STEP_N
                                            ),
                                        )
                                if cta_group > 1:
                                    if is_multicast_on_a:
                                        K.assign(
                                            m_idx,
                                            m_idx
                                            + cta_in_cluster
                                            * (
                                                _udiv(ld_eff_m, cta_group)
                                                if use_effective_m
                                                else load_block_m
                                            ),
                                        )
                                    else:
                                        K.assign(n_idx, n_idx + cta_in_cluster * load_block_n)
                                ld_k = K.local_scalar("int32", init=0)
                                with K.While(ld_k < ld_kblocks):
                                    _wait_barrier(
                                        empty_barriers.ptr_to([ld_pipe.stage]),
                                        K.bitwise_xor(ld_pipe.phase, 1),
                                    )
                                    k_b = K.local_scalar("int32")
                                    k_a = K.local_scalar("int32")
                                    K.assign(k_b, k_b_idx + ld_k * block_k)
                                    K.assign(k_a, k_a_offset + ld_k * block_k)
                                    with K.unroll(0, num_a_atoms) as i:
                                        K.ptx[load_chain](
                                            (
                                                smem_a_tile[ld_pipe.stage].ptr_to(0, i * a_atom)
                                                if major_a_is_k
                                                else smem_a_tile[ld_pipe.stage].ptr_to(0, 0)
                                            ),
                                            K.address_of(tensor_map_a),
                                            *_tma_coords(
                                                (
                                                    (k_a + i * a_atom, m_idx)
                                                    if major_a_is_k
                                                    else (m_idx + i * a_atom, k_a)
                                                ),
                                                ld_sched.grp,
                                            ),
                                            full_barriers.ptr_to([ld_pipe.stage]),
                                            K.uint64(EVICT_NORMAL),
                                        )
                                    with K.unroll(0, num_b_atoms) as i:
                                        K.ptx[load_chain](
                                            (
                                                smem_b_tile[ld_pipe.stage].ptr_to(0, i * b_atom)
                                                if major_b_is_k
                                                else smem_b_tile[ld_pipe.stage].ptr_to(0, 0)
                                            ),
                                            K.address_of(tensor_map_b),
                                            *_tma_coords(
                                                (
                                                    (k_b + i * b_atom, n_idx)
                                                    if major_b_is_k
                                                    else (n_idx + i * b_atom, k_b)
                                                ),
                                                ld_sched.grp,
                                            ),
                                            full_barriers.ptr_to([ld_pipe.stage]),
                                            K.uint64(EVICT_NORMAL),
                                        )
                                    arrival = K.local_scalar("int32", init=arrival_bytes_ab)
                                    with K.If(ld_k % sfa_stages_per_load == 0):
                                        with K.Then():
                                            K.ptx[sf_load_chain](
                                                smem_sfa.ptr_to([ld_pipe.stage, 0]),
                                                K.address_of(tensor_map_sfa),
                                                K.cast(sfa_mn, "int32"),
                                                K.cast(
                                                    sfa_k_offset + ld_k // sfa_stages_per_load,
                                                    "int32",
                                                ),
                                                full_barriers.ptr_to([ld_pipe.stage]),
                                                K.uint64(EVICT_NORMAL),
                                            )
                                            K.assign(arrival, arrival + block_m * 4)
                                    with K.If(ld_k % sfb_stages_per_load == 0):
                                        with K.Then():
                                            K.ptx[sf_load_chain](
                                                smem_sfb.ptr_to([ld_pipe.stage, 0]),
                                                K.address_of(tensor_map_sfb),
                                                K.cast(sfb_mn, "int32"),
                                                K.cast(
                                                    sfb_k_offset + ld_k // sfb_stages_per_load,
                                                    "int32",
                                                ),
                                                full_barriers.ptr_to([ld_pipe.stage]),
                                                K.uint64(EVICT_NORMAL),
                                            )
                                            K.assign(arrival, arrival + block_n * 4)
                                    full_barriers.arrive(ld_pipe.stage, K.cast(arrival, "uint32"))
                                    K.assign(ld_k, ld_k + 1)
                                    _advance_pipeline(ld_pipe, stages)
                        ld_sched.advance()

        # ===============================================================
        # Role 1: UMMA issue warp, leader CTA only (source `:281`)
        # ===============================================================

        def mma_role_body():
            with K.If(is_leader_cta), K.Then():
                desc_a = K.local_scalar("uint64")
                desc_b = K.local_scalar("uint64")
                desc_sf = K.local_scalar("uint64")
                desc_i = K.local_scalar("uint32")
                # Every descriptor field but the shared address is a build-time
                # constant, so the bases are folded in Python and only
                # `addr >> 4` is computed here.  The `*_DESC_BASE` constants
                # come from the same bit layout the runtime C encoders fill in.
                a_smem_u32 = K.local_scalar("uint32")
                b_smem_u32 = K.local_scalar("uint32")
                sfa_smem_u32 = K.local_scalar("uint32")
                sfb_smem_u32 = K.local_scalar("uint32")
                K.assign(a_smem_u32, K.cuda.cvta_generic_to_shared(smem_a_tile[0].ptr_to(0, 0)))
                K.assign(b_smem_u32, K.cuda.cvta_generic_to_shared(smem_b_tile[0].ptr_to(0, 0)))
                K.assign(sfa_smem_u32, K.cuda.cvta_generic_to_shared(smem_sfa.ptr_to([0, 0])))
                K.assign(sfb_smem_u32, K.cuda.cvta_generic_to_shared(smem_sfb.ptr_to([0, 0])))
                K.assign(desc_a, _with_smem_addr(_u64_const(A_DESC_BASE), a_smem_u32))
                K.assign(desc_b, _with_smem_addr(_u64_const(B_DESC_BASE), b_smem_u32))
                # `make_sf_desc`: unswizzled, stride offset 8*16, leading offset 0.
                K.assign(desc_sf, _with_smem_addr(_u64_const(SF_DESC_BASE), sfa_smem_u32))
                # Stays a mutable local: the scale-factor ids and, under
                # `use_effective_m`, the N field are patched per MMA below.
                K.assign(desc_i, K.uint32(INSTR_DESC))
                # The per-stage descriptor low words live one stage per lane; a warp
                # shuffle indexes the table instead of recomputing the descriptor.
                a_desc_lo = K.local_scalar("uint32")
                b_desc_lo = K.local_scalar("uint32")
                K.assign(
                    a_desc_lo,
                    K.Select(
                        lane_idx < stages,
                        K.cast(K.bitwise_and(desc_a, K.uint64(0xFFFFFFFF)), "uint32")
                        + K.cast(lane_idx * (a_bytes_per_stage // 16), "uint32"),
                        K.uint32(0),
                    ),
                )
                K.assign(
                    b_desc_lo,
                    K.Select(
                        lane_idx < stages,
                        K.cast(K.bitwise_and(desc_b, K.uint64(0xFFFFFFFF)), "uint32")
                        + K.cast(lane_idx * (b_bytes_per_stage // 16), "uint32"),
                        K.uint32(0),
                    ),
                )

                mma_pipe = K.PipelineState(stages, phase=0)
                accum_pipe = K.PipelineState(NUM_EPILOGUE_STAGES, phase=0)
                mma_sched = _PersistentScheduler(
                    spec,
                    grouped_layout,
                    sm_idx,
                    num_m_blocks,
                    num_n_blocks,
                    num_blocks,
                    track_sfk=False,
                )
                mma_iter = K.local_scalar("int32")
                mma_kblocks = K.local_scalar("int32")
                # `cute::elect_one_sync()` is loop-invariant for this fully-active warp;
                # nvcc hoists it into a uniform predicate, so hoist it here too rather
                # than re-executing `elect.sync` twice per K block.
                mma_elected = K.local_scalar("uint32")
                mma_elected_lane = K.local_scalar("uint32")
                K.ptx.elect_sync(mma_elected_lane, mma_elected, K.uint32(0xFFFFFFFF))
                K.assign(mma_kblocks, 0)
                K.assign(mma_iter, 0)
                with K.While(mma_sched.valid == 1):
                    mma_sched.next()
                    with K.If(mma_sched.valid == 1):
                        with K.Then():
                            if use_effective_m:
                                # `get_aligned_effective_m_in_block` for this block (sketch `:559`).
                                mma_eff_m = K.local_scalar("int32")
                                mma_m_local = K.local_scalar(
                                    "int32",
                                    init=_swizzled(
                                        mma_sched.nb - mma_sched.cum * num_n_blocks,
                                        mma_sched.nmb,
                                        num_n_blocks,
                                    )[0],
                                )
                                K.assign(mma_eff_m, block_m)
                                with K.If(mma_m_local == mma_sched.nmb - 1), K.Then():
                                    K.assign(
                                        mma_eff_m,
                                        (
                                            _uceil(
                                                mma_sched.psum
                                                - (mma_m_local + _udiv(mma_sched.last, block_m))
                                                * block_m,
                                                UMMA_STEP_N,
                                            )
                                            * UMMA_STEP_N
                                        ),
                                    )
                                K.assign(
                                    desc_i,
                                    K.bitwise_or(
                                        K.bitwise_and(desc_i, K.uint32(UMMA_N_FIELD_MASK)),
                                        K.shift_left(
                                            K.cast(mma_eff_m // 8, "uint32"), K.uint32(17)
                                        ),
                                    ),
                                )
                            K.assign(
                                mma_kblocks,
                                (_uceil(mma_sched.psum, block_k) if is_k_grouped else num_k_blocks),
                            )
                            _wait_barrier(
                                tmem_empty_barriers.ptr_to([accum_pipe.stage]),
                                K.bitwise_xor(accum_pipe.phase, 1),
                            )
                            K.ptx.tcgen05.fence__after_thread_sync()

                            mma_k = K.local_scalar("int32", init=0)
                            # `#pragma unroll 4` (source `:339`).  Four bodies per back-edge with
                            # `mma_k == 4 * j + u` makes `mma_k % kNumSF?StagesPerLoad` the constant
                            # `u`, which is what removes the scale-factor id arithmetic and the UTCCP
                            # branch from three of every four K blocks.
                            mma_k_rounded = K.local_scalar(
                                "int32",
                                init=((mma_kblocks + (MMA_K_UNROLL - 1)) // MMA_K_UNROLL)
                                * MMA_K_UNROLL,
                            )
                            with K.While(mma_k < mma_k_rounded):
                                with K.unroll(0, MMA_K_UNROLL) as u:
                                    with K.If(mma_k < mma_kblocks), K.Then():
                                        _wait_barrier(
                                            with_sf_barriers.ptr_to([mma_pipe.stage]),
                                            mma_pipe.phase,
                                        )
                                        K.ptx.tcgen05.fence__after_thread_sync()
                                        a_base_lo = K.local_scalar("uint32")
                                        b_base_lo = K.local_scalar("uint32")
                                        K.ptx.shfl_sync.idx.b32(
                                            a_base_lo,
                                            a_desc_lo,
                                            K.cast(mma_pipe.stage, "uint32"),
                                            K.uint32(0x1F),
                                            K.uint32(0xFFFFFFFF),
                                        )
                                        K.ptx.shfl_sync.idx.b32(
                                            b_base_lo,
                                            b_desc_lo,
                                            K.cast(mma_pipe.stage, "uint32"),
                                            K.uint32(0x1F),
                                            K.uint32(0xFFFFFFFF),
                                        )
                                        # One elected lane owns the UTCCP and UMMA issues.  Predicating the
                                        # instructions keeps the warp converged; branching on the elected
                                        # lane instead costs a BSSY/BSYNC pair per K block.
                                        with K.If(u % sfa_stages_per_load == 0):
                                            with K.Then():
                                                with K.unroll(0, num_sfa_chunks) as c:
                                                    K.assign(
                                                        desc_sf,
                                                        _rebase(
                                                            desc_sf,
                                                            sfa_smem_u32
                                                            + K.cast(
                                                                mma_pipe.stage * (sf_block_m * 4),
                                                                "uint32",
                                                            )
                                                            + K.uint32(
                                                                c * NUM_UTCCP_ALIGNED_ELEMS * 4
                                                            ),
                                                        ),
                                                    )
                                                    K.ptx[utccp_chain](
                                                        K.cast(sfa_tmem_col + c * 4, "uint32"),
                                                        desc_sf,
                                                        pred=mma_elected,
                                                    )
                                        with K.If(u % sfb_stages_per_load == 0):
                                            with K.Then():
                                                with K.unroll(0, num_sfb_chunks) as c:
                                                    K.assign(
                                                        desc_sf,
                                                        _rebase(
                                                            desc_sf,
                                                            sfb_smem_u32
                                                            + K.cast(
                                                                mma_pipe.stage * (sf_block_n * 4),
                                                                "uint32",
                                                            )
                                                            + K.uint32(
                                                                c * NUM_UTCCP_ALIGNED_ELEMS * 4
                                                            ),
                                                        ),
                                                    )
                                                    K.ptx[utccp_chain](
                                                        K.cast(sfb_tmem_col + c * 4, "uint32"),
                                                        desc_sf,
                                                        pred=mma_elected,
                                                    )
                                        with K.unroll(0, umma_k_steps) as ki:
                                            # `issue_full_k_block` / `issue_tail_k_block`:
                                            # only the leading `ceil_div(remaining_k, UMMA_K)`
                                            # steps of a partial final K block are valid.
                                            with (
                                                K.If(
                                                    K.Or(
                                                        mma_k < mma_kblocks - 1,
                                                        ki * UMMA_K
                                                        < (
                                                            mma_sched.psum
                                                            if is_k_grouped
                                                            else eff_k
                                                        )
                                                        - mma_k * block_k,
                                                    )
                                                    if may_have_tail_k
                                                    else ki < umma_k_steps
                                                ),
                                                K.Then(),
                                            ):
                                                sfa_id = K.local_scalar("uint32")
                                                sfb_id = K.local_scalar("uint32")
                                                K.assign(
                                                    sfa_id,
                                                    (
                                                        K.cast(ki, "uint32")
                                                        if sfa_stages_per_load == 1
                                                        else K.cast(
                                                            u % sfa_stages_per_load, "uint32"
                                                        )
                                                    ),
                                                )
                                                K.assign(
                                                    sfb_id,
                                                    (
                                                        K.cast(ki, "uint32")
                                                        if sfb_stages_per_load == 1
                                                        else K.cast(
                                                            u % sfb_stages_per_load, "uint32"
                                                        )
                                                    ),
                                                )
                                                rt_desc = K.local_scalar(
                                                    "uint32",
                                                    init=_with_sf_id(
                                                        desc_i,
                                                        (sfb_id if swap_ab else sfa_id),
                                                        (sfa_id if swap_ab else sfb_id),
                                                    ),
                                                )
                                                adv_a = K.local_scalar("uint64")
                                                adv_b = K.local_scalar("uint64")
                                                K.assign(
                                                    adv_a,
                                                    _advance_lo(
                                                        desc_a, a_base_lo, ki * a_k_step_units
                                                    ),
                                                )
                                                K.assign(
                                                    adv_b,
                                                    _advance_lo(
                                                        desc_b, b_base_lo, ki * b_k_step_units
                                                    ),
                                                )
                                                K.ptx[mma_chain](
                                                    K.cast(accum_pipe.stage * umma_n, "uint32"),
                                                    (adv_b if swap_ab else adv_a),
                                                    (adv_a if swap_ab else adv_b),
                                                    rt_desc,
                                                    K.cast(
                                                        sfb_tmem_col if swap_ab else sfa_tmem_col,
                                                        "uint32",
                                                    ),
                                                    K.cast(
                                                        sfa_tmem_col if swap_ab else sfb_tmem_col,
                                                        "uint32",
                                                    ),
                                                    K.Or(ki > 0, mma_k > 0),
                                                    pred=mma_elected,
                                                )
                                        K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                                        # `tcgen05.commit` implies `fence::before_thread_sync`.
                                        # Same here: predicate the commits on the elected lane rather than
                                        # branching, so the warp never diverges inside the K loop.
                                        if cta_group > 1:
                                            empty_barriers.arrive(
                                                mma_pipe.stage,
                                                cta_group=cta_group,
                                                cta_mask=(1 << cta_group) - 1,
                                                pred=mma_elected,
                                            )
                                        else:
                                            empty_barriers.arrive(
                                                mma_pipe.stage,
                                                cta_group=cta_group,
                                                pred=mma_elected,
                                            )
                                        if cta_group > 1:
                                            tmem_full_barriers.arrive(
                                                accum_pipe.stage,
                                                cta_group=cta_group,
                                                cta_mask=(1 << cta_group) - 1,
                                                pred=K.And(
                                                    mma_elected == K.uint32(1),
                                                    mma_k == mma_kblocks - 1,
                                                ),
                                            )
                                        else:
                                            tmem_full_barriers.arrive(
                                                accum_pipe.stage,
                                                cta_group=cta_group,
                                                pred=K.And(
                                                    mma_elected == K.uint32(1),
                                                    mma_k == mma_kblocks - 1,
                                                ),
                                            )
                                        K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                                        _advance_pipeline(mma_pipe, stages)
                                    # Advance unconditionally: the guard only masks the rounded-up tail,
                                    # and the loop still has to reach `mma_k_rounded`.
                                    K.assign(mma_k, mma_k + 1)
                            _advance_pipeline(accum_pipe, NUM_EPILOGUE_STAGES)
                            K.assign(mma_iter, mma_iter + 1)
                    mma_sched.advance()

                # A 2-CTA cluster needs one more accumulator wait before the
                # barriers can be safely destroyed (source `:426`).
                if cta_group > 1:
                    with K.If(mma_iter > 0):
                        with K.Then():
                            _wait_barrier(
                                tmem_empty_barriers.ptr_to([(mma_iter - 1) % NUM_EPILOGUE_STAGES]),
                                K.bitwise_and((mma_iter - 1) // NUM_EPILOGUE_STAGES, 1),
                            )

        # ===============================================================
        # Role 2: scale-factor transposer warp (source `:432`)
        # ===============================================================

        def transpose_role_body():
            tr_pipe = K.PipelineState(stages, phase=0)
            tr_sched = _PersistentScheduler(
                spec,
                grouped_layout,
                sm_idx,
                num_m_blocks,
                num_n_blocks,
                num_blocks,
                track_sfk=False,
            )
            tr_kblocks = K.local_scalar("int32", init=0)
            sf_vals = K.alloc_local((4,), "uint32")
            with K.While(tr_sched.valid == 1):
                tr_sched.next()
                with K.If(tr_sched.valid == 1):
                    with K.Then():
                        K.assign(
                            tr_kblocks,
                            (_uceil(tr_sched.psum, block_k) if is_k_grouped else num_k_blocks),
                        )
                        tr_k = K.local_scalar("int32", init=0)
                        with K.While(tr_k < tr_kblocks):
                            _wait_barrier(full_barriers.ptr_to([tr_pipe.stage]), tr_pipe.phase)
                            # The prior logical task may still read this stage through tcgen05's
                            # async proxy.  Complete that handoff before generic-proxy transpose
                            # stores reuse the same shared bytes.
                            K.ptx.fence.proxy.async_.shared__cta()
                            with K.If(tr_k % sfa_stages_per_load == 0):
                                with K.Then():
                                    with K.unroll(0, num_sfa_chunks) as c:
                                        base = c * NUM_UTCCP_ALIGNED_ELEMS
                                        with K.unroll(0, 4) as i:
                                            K.ptx.ld.shared.u32(
                                                sf_vals[i],
                                                smem_sfa.ptr_to(
                                                    [tr_pipe.stage, base + i * 32 + lane_idx]
                                                ),
                                            )
                                        K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                                        K.ptx.st.shared.v4.u32(
                                            smem_sfa.ptr_to([tr_pipe.stage, base + lane_idx * 4]),
                                            sf_vals[0],
                                            sf_vals[1],
                                            sf_vals[2],
                                            sf_vals[3],
                                        )
                                    K.ptx.fence.proxy.async_.shared__cta()
                            with K.If(tr_k % sfb_stages_per_load == 0):
                                with K.Then():
                                    with K.unroll(0, num_sfb_chunks) as c:
                                        base = c * NUM_UTCCP_ALIGNED_ELEMS
                                        with K.unroll(0, 4) as i:
                                            K.ptx.ld.shared.u32(
                                                sf_vals[i],
                                                smem_sfb.ptr_to(
                                                    [tr_pipe.stage, base + i * 32 + lane_idx]
                                                ),
                                            )
                                        K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                                        K.ptx.st.shared.v4.u32(
                                            smem_sfb.ptr_to([tr_pipe.stage, base + lane_idx * 4]),
                                            sf_vals[0],
                                            sf_vals[1],
                                            sf_vals[2],
                                            sf_vals[3],
                                        )
                                    K.ptx.fence.proxy.async_.shared__cta()
                            # `arrive(0u)` passes a destination CTA rank, not a count:
                            # every thread arrives on the leader CTA's barrier copy.
                            rem = K.local_scalar("uint64")
                            K.ptx.mapa.shared__cluster.u64(
                                rem, with_sf_barriers.ptr_to([tr_pipe.stage]), K.uint32(0)
                            )
                            K.ptx.mbarrier.arrive.b64(rem, K.uint32(1), pred=K.bool(True))
                            K.assign(tr_k, tr_k + 1)
                            _advance_pipeline(tr_pipe, stages)
                tr_sched.advance()

        # ===============================================================
        # Role 3: epilogue warps (source `:470`)
        # ===============================================================

        def epilogue_role_body(ep_warp):
            accum_pipe_e = K.PipelineState(NUM_EPILOGUE_STAGES, phase=0)
            tma_stage = K.local_scalar("int32", init=0)
            ep_sched = _PersistentScheduler(
                spec,
                grouped_layout,
                sm_idx,
                num_m_blocks,
                num_n_blocks,
                num_blocks,
                track_sfk=False,
            )
            values = K.alloc_local((8,), "uint32")
            packed = K.alloc_local((4,), "uint32")
            with K.While(ep_sched.valid == 1):
                ep_sched.next()
                with K.If(ep_sched.valid == 1):
                    with K.Then():
                        _wait_barrier(
                            tmem_full_barriers.ptr_to([accum_pipe_e.stage]), accum_pipe_e.phase
                        )
                        K.ptx.tcgen05.fence__after_thread_sync()
                        # One swizzle walk feeds `base_m`, `base_n` and the tail-block test.
                        ep_m_local, ep_n_local = _swizzled(
                            ep_sched.nb - ep_sched.cum * num_n_blocks, ep_sched.nmb, num_n_blocks
                        )
                        base_m = K.local_scalar("int32")
                        base_n = K.local_scalar("int32")
                        K.assign(
                            base_m,
                            (
                                ep_m_local
                                + (_udiv(ep_sched.last, block_m) if is_m_grouped_psum else 0)
                            )
                            * block_m,
                        )
                        K.assign(base_n, ep_n_local * block_n)
                        if is_m_grouped_masked or is_k_grouped:
                            K.assign(base_m, ep_sched.grp * eff_m + base_m)
                        tmem_base = K.local_scalar("int32", init=accum_pipe_e.stage * umma_n)

                        # `num_stores = effective_m / STORE_BLOCK_M` (sketch `:1276`).
                        ep_stores = K.local_scalar("int32")
                        if use_effective_m:
                            # `get_aligned_effective_m_in_block` for this block (sketch `:559`).
                            ep_eff_m = K.local_scalar("int32", init=block_m)
                            with K.If(ep_m_local == ep_sched.nmb - 1), K.Then():
                                K.assign(
                                    ep_eff_m,
                                    (
                                        _uceil(
                                            ep_sched.psum
                                            - (ep_m_local + _udiv(ep_sched.last, block_m))
                                            * block_m,
                                            UMMA_STEP_N,
                                        )
                                        * UMMA_STEP_N
                                    ),
                                )
                            K.assign(ep_stores, _udiv(ep_eff_m, store_block_m))
                        else:
                            K.assign(ep_stores, num_swap_stores)

                        if swap_ab:
                            with K.unroll(0, num_swap_stores) as st:
                                with K.If(st < ep_stores), K.Then():
                                    with K.If(ep_warp == 0):
                                        with K.Then():
                                            K.ptx.cp.async_.bulk.wait_group(
                                                NUM_TMA_STORE_STAGES - 1
                                            )
                                    K.ptx.bar.sync(
                                        K.uint32(EPILOGUE_NAMED_BARRIER),
                                        K.uint32(num_store_threads),
                                    )
                                    with K.unroll(0, num_atom_rows) as i:
                                        taddr_s = K.local_scalar(
                                            "uint32",
                                            init=K.cast(
                                                tmem_base + st * store_block_m + i * 8, "uint32"
                                            ),
                                        )
                                        atom_byte = (
                                            ep_warp // warps_per_atom
                                        ) * store_block_m * swizzle_cd + i * 8 * swizzle_cd
                                        if cd_is_fp32:
                                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                                                *[values[j] for j in range(8)], taddr_s
                                            )
                                            K.ptx.tcgen05.wait__ld.sync.aligned()
                                            col_f = K.local_scalar("int32", init=lane_idx // 4)
                                            with K.unroll(0, 8) as row:
                                                K.ptx.st.shared.u32(
                                                    smem_cd_u32.ptr_to(
                                                        [
                                                            (
                                                                tma_stage * cd_stage_bytes
                                                                + atom_byte
                                                                + row * (16 * 8)
                                                                + K.bitwise_xor(col_f, row) * 16
                                                                + (lane_idx % 4) * 4
                                                            )
                                                            // 4
                                                        ]
                                                    ),
                                                    values[row],
                                                )
                                        else:
                                            # Two `16x256b` slices: the second takes the upper
                                            # 16 rows via bit 20 of the TMEM address.
                                            K.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                                values[0], values[1], values[2], values[3], taddr_s
                                            )
                                            K.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                                values[4],
                                                values[5],
                                                values[6],
                                                values[7],
                                                K.bitwise_or(taddr_s, K.uint32(0x00100000)),
                                            )
                                            K.ptx.tcgen05.wait__ld.sync.aligned()
                                            with K.unroll(0, 4) as j:
                                                # `cvt.rn.bf16x2.f32 d, a, b` packs a
                                                # into the UPPER half and b into the
                                                # lower, the reverse of the
                                                # `make_float2(lo, hi)` helper this
                                                # replaces -- hence the swap.
                                                K.ptx.cvt.rn.bf16x2.f32(
                                                    packed[j],
                                                    K.reinterpret("float32", values[2 * j + 1]),
                                                    K.reinterpret("float32", values[2 * j]),
                                                )
                                            row_s = K.local_scalar("int32")
                                            col_s = K.local_scalar("int32")
                                            K.assign(row_s, lane_idx % 8)
                                            K.assign(col_s, (ep_warp % 2) * 4 + lane_idx // 8)
                                            K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                                smem_cd_u32.ptr_to(
                                                    [
                                                        (
                                                            tma_stage * cd_stage_bytes
                                                            + atom_byte
                                                            + row_s * (16 * 8)
                                                            + K.bitwise_xor(col_s, row_s) * 16
                                                        )
                                                        // 4
                                                    ]
                                                ),
                                                packed[0],
                                                packed[1],
                                                packed[2],
                                                packed[3],
                                            )

                                    with K.If(st == ep_stores - 1):
                                        with K.Then():
                                            K.ptx.tcgen05.fence__before_thread_sync()
                                            rem_s = K.local_scalar("uint64")
                                            K.ptx.mapa.shared__cluster.u64(
                                                rem_s,
                                                tmem_empty_barriers.ptr_to([accum_pipe_e.stage]),
                                                K.uint32(0),
                                            )
                                            K.ptx.mbarrier.arrive.b64(
                                                rem_s, K.uint32(1), pred=K.bool(True)
                                            )

                                    K.ptx.fence.proxy.async_.shared__cta()
                                    K.ptx.bar.sync(
                                        K.uint32(EPILOGUE_NAMED_BARRIER),
                                        K.uint32(num_store_threads),
                                    )
                                    with K.If(ep_warp == 0):
                                        with K.Then():
                                            # The store is issued by one elected lane; predicate rather than
                                            # branch so the epilogue warp stays converged.
                                            ep_elected = K.local_scalar("uint32")
                                            ep_elected_lane = K.local_scalar("uint32")
                                            K.ptx.elect_sync(
                                                ep_elected_lane, ep_elected, K.uint32(0xFFFFFFFF)
                                            )
                                            with K.unroll(0, num_n_atoms) as i:
                                                K.ptx[
                                                    (
                                                        reduce_chain
                                                        if with_accumulation
                                                        else store_chain
                                                    )
                                                ](
                                                    K.address_of(tensor_map_cd),
                                                    *_tma_coords(
                                                        (
                                                            base_n + i * store_block_n_atom,
                                                            base_m + st * store_block_m,
                                                        ),
                                                        ep_sched.grp,
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
                                                    pred=ep_elected,
                                                )
                                            K.ptx.cp.async_.bulk.commit_group()
                                    K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                                    K.assign(tma_stage, _next_tma_store_stage(tma_stage))
                        else:
                            with K.unroll(0, num_m_waves) as w:
                                with K.unroll(0, num_stores) as st:
                                    with K.If(ep_warp == 0):
                                        with K.Then():
                                            K.ptx.cp.async_.bulk.wait_group(
                                                NUM_TMA_STORE_STAGES - 1
                                            )
                                    K.ptx.bar.sync(
                                        K.uint32(EPILOGUE_NAMED_BARRIER),
                                        K.uint32(num_store_threads),
                                    )
                                    with K.unroll(0, elems_per_store) as i:
                                        bank_group = i + lane_idx * (swizzle_cd // 16)
                                        row = i // 8 + lane_idx if has_shortcut else bank_group // 8
                                        col = i if has_shortcut else bank_group % 8
                                        col = K.bitwise_xor(col, row % (swizzle_cd // 16))
                                        # `smem_ptr` in the source: one address per bank
                                        # group, four registers stored at it.
                                        cd_word = K.local_scalar(
                                            "int32", init=_cd_word(tma_stage, ep_warp, row, col, 0)
                                        )
                                        taddr = K.local_scalar(
                                            "uint32",
                                            init=K.cast(
                                                tmem_base
                                                + w * block_n
                                                + st * store_block_n
                                                + i * elems_per_bank_group,
                                                "uint32",
                                            ),
                                        )
                                        if cd_is_fp32:
                                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
                                                values[0], values[1], values[2], values[3], taddr
                                            )
                                            K.ptx.tcgen05.wait__ld.sync.aligned()
                                            K.ptx.st.shared.v4.u32(
                                                smem_cd_u32.ptr_to([cd_word]),
                                                values[0],
                                                values[1],
                                                values[2],
                                                values[3],
                                            )
                                        else:
                                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                                                *[values[j] for j in range(8)], taddr
                                            )
                                            K.ptx.tcgen05.wait__ld.sync.aligned()
                                            with K.unroll(0, 4) as j:
                                                # `cvt.rn.bf16x2.f32 d, a, b` packs a
                                                # into the UPPER half and b into the
                                                # lower, the reverse of the
                                                # `make_float2(lo, hi)` helper this
                                                # replaces -- hence the swap.
                                                K.ptx.cvt.rn.bf16x2.f32(
                                                    packed[j],
                                                    K.reinterpret("float32", values[2 * j + 1]),
                                                    K.reinterpret("float32", values[2 * j]),
                                                )
                                            K.ptx.st.shared.v4.u32(
                                                smem_cd_u32.ptr_to([cd_word]),
                                                packed[0],
                                                packed[1],
                                                packed[2],
                                                packed[3],
                                            )

                                    with K.If(K.And(w == num_m_waves - 1, st == num_stores - 1)):
                                        with K.Then():
                                            K.ptx.tcgen05.fence__before_thread_sync()
                                            rem_e = K.local_scalar("uint64")
                                            K.ptx.mapa.shared__cluster.u64(
                                                rem_e,
                                                tmem_empty_barriers.ptr_to([accum_pipe_e.stage]),
                                                K.uint32(0),
                                            )
                                            K.ptx.mbarrier.arrive.b64(
                                                rem_e, K.uint32(1), pred=K.bool(True)
                                            )

                                    K.ptx.fence.proxy.async_.shared__cta()
                                    K.ptx.bar.sync(
                                        K.uint32(EPILOGUE_NAMED_BARRIER),
                                        K.uint32(num_store_threads),
                                    )
                                    with K.If(ep_warp == 0):
                                        with K.Then():
                                            # The store is issued by one elected lane; predicate rather than
                                            # branch so the epilogue warp stays converged.
                                            ep_elected = K.local_scalar("uint32")
                                            ep_elected_lane = K.local_scalar("uint32")
                                            K.ptx.elect_sync(
                                                ep_elected_lane, ep_elected, K.uint32(0xFFFFFFFF)
                                            )
                                            K.ptx[
                                                (reduce_chain if with_accumulation else store_chain)
                                            ](
                                                K.address_of(tensor_map_cd),
                                                *_tma_coords(
                                                    (
                                                        base_n + st * store_block_n,
                                                        base_m + w * store_block_m,
                                                    ),
                                                    ep_sched.grp,
                                                ),
                                                smem_cd.ptr_to([tma_stage, 0, 0]),
                                                pred=ep_elected,
                                            )
                                            K.ptx.cp.async_.bulk.commit_group()
                                    K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                                    K.assign(tma_stage, _next_tma_store_stage(tma_stage))
                        _advance_pipeline(accum_pipe_e, NUM_EPILOGUE_STAGES)
                ep_sched.advance()

        with load_role:
            load_role_body()
        with mma_role:
            mma_role_body()
        with transpose_role:
            transpose_role_body()
        with epilogue_role:
            epilogue_role_body(K.warp_id_in_role())

        # ===============================================================
        # Teardown (source `:524`)
        # ===============================================================

        if cta_group > 1:
            K.ptx.barrier.cluster.arrive.relaxed.aligned()
            K.ptx.barrier.cluster.wait.acquire.aligned()
        else:
            K.ptx.bar.sync(K.uint32(0))

        # The allocating warp (2) and the freeing warp (0) deliberately differ.
        with load_role:
            K.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                K.uint32(0), K.uint32(spec.num_tmem_cols)
            )

    return sm100_fp8_fp4_gemm_1d1d.func


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
