<!--
Copyright (c) 2025 DeepSeek
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.

This design sketch documents a TIRx port of DeepGEMM's
deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh together with the
scheduler and epilogue helpers it instantiates. See NOTICE and
licenses/ for upstream attribution.
-->

# SM100 FP8/FP4 1D1D GEMM: coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new
IR, a builder API, or a mathematical reference implementation. Its purpose is to
show the TIRx kernel as:

- an explicit runtime ABI and launch;
- explicit GMEM, SMEM, TMEM, and register tiles, including byte and column
  offsets;
- the persistent scheduler and the eight-warp role control flow;
- primitive directional copies and primitive computation inside every reachable
  helper;
- explicit stage/index/phase changes and publication/reuse edges;
- hardware instruction selection derived only after placement, shape, layout,
  and schedule have been stated.

The implementation represented by this sketch is maintained in
[`tirx_kernels/deepgemm/_sm100_fp8_fp4_gemm_1d1d.py`](../../tirx_kernels/deepgemm/_sm100_fp8_fp4_gemm_1d1d.py),
with five thin entry modules (`fp8_gemm_1d1d.py`,
`m_grouped_fp8_gemm_contiguous.py`, `m_grouped_fp8_gemm_masked.py`,
`k_grouped_fp8_gemm_contiguous.py`, `fp8_bmm.py`) that pin one descriptor each.
Those modules are the source of truth.

DeepGEMM has exactly one SM100 FP8/FP4 dense-GEMM device template; its five host
entries differ only in template arguments. This sketch therefore covers the
whole template and marks each compile-time branch where the source does.

**In scope.** All seven `kGemmType` values (`Normal`, `MGroupedContiguous`,
`MGroupedContiguousWithPsumLayout`, `MGroupedMasked`, `KGroupedContiguous`,
`KGroupedContiguousWithPsumLayout`, `Batched`); `a_dtype` FP8 e4m3 and `b_dtype`
FP8 e4m3 or unpacked-SMEM FP4 e2m1; `cd_dtype` BF16 and FP32; majors K/K, K/MN
and MN/MN; `swap_ab` both ways; `with_accumulation` both ways; `gran_k` 32 and
128; `kNumMulticast` 1 and 2; `ensure_zero_padding` both ways.

**Out of scope.** `KernelType::Kernel1D2D` and every SM90 implementation;
`epilogue_type_t` other than `EpilogueIdentity`; FP4 on the A operand and
FP4xFP4; the BF16 MMA kind; `fp8_gemm_nt_skip_head_mid`; tile (`Tx`) primitives.

## Pipeline at a glance

| Warp | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0 | prefetch all five TensorMaps; then, from one elected lane, drive the persistent scheduler and issue A/B/SFA/SFB TMA loads per K block; after the role chain, free TMEM | wait `empty[stage]`; publish `full[stage]` with an expect-tx byte count |
| 1 | initialize all five barrier families; then, on the leader CTA only, drive the same scheduler, issue UTCCP scale-factor copies and the UMMA chain, and arrive on the reuse edges | wait `tmem_empty[accum]` and `with_sf_full[stage]`; arrive `empty[stage]` and (last K block) `tmem_full[accum]` |
| 2 | allocate TMEM; then drive the same scheduler and transpose each freshly arrived SFA/SFB tile in place so UTCCP can read it | wait `full[stage]`; publish `with_sf_full[stage]` |
| 3 | **idle** -- no branch of the role chain selects it | none |
| 4..(4 + kNumUMMAStoreThreads/32 - 1) | drive the same scheduler and run the epilogue: TMEM load, cast, swizzled shared store, TMA store | wait `tmem_full[accum]`; arrive `tmem_empty[accum]`; TMA-store ring shared across scheduler blocks |

With `swap_ab` the epilogue is the full warpgroup (warps 4..7); without it only
`STORE_BLOCK_M / 32` warps take the branch, so warps beyond that are idle too.
All four scheduler-driving roles instantiate the scheduler independently and
walk the identical block sequence; `stage_idx`/`phase` are per-warp local state
kept in lockstep by that shared walk.

The role chain is a single `if / elif / elif / elif` on a warp-uniform warp
index. Warp 0's and warp 1's branches carry an extra predicate
(`elect_one_sync()` and `is_leader_cta` respectively), and both the barrier-init
and the TMEM-allocation work happen *before* the chain in a separate
`if warp == 1 ... elif warp == 2` pair. That exact shape remains visible below.

## Primitive vocabulary

Structural operations do not compute values:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
tile(...)             # declare storage, dtype, logical shape, and placement
view(...)             # change logical indexing without moving values
slice(...)            # select a logical interval
reg_tile(...)         # declare a role-local register tile
desc(...)             # build an SMEM matrix / instruction / scale-factor descriptor
```

Copies always state their storage direction:

```python
copy_g2s(src, dst, completion=None)   # global -> shared, TMA, mbarrier completion
copy_s2g(src, dst, reduce=None)       # shared -> global, TMA, optional add-reduce
copy_s2r(src, dst)                    # shared -> register
copy_r2s(src, dst, transpose=False)   # register -> shared
copy_s2t(src, dst)                    # shared -> tensor memory (UTCCP)
copy_t2r(src, dst)                    # tensor memory -> register
```

The complete computational vocabulary used below is:

```python
fill(dst, value)
cast(dst, src, rounding=None, pack=False)
add(dst, lhs, rhs)
mul(dst, lhs, rhs)
div_ceil(dst, lhs, rhs)
align_up(dst, value, granularity)
bitwise_and(dst, lhs, rhs)
bitwise_xor(dst, lhs, rhs)
min_u32(dst, lhs, rhs)
max_i32(dst, lhs, rhs)
move(dst, src)
select(dst, predicate, true_value, false_value)
shuffle_index(dst, src, source_lane, mask, clamp)
gemm(dst, lhs, rhs, accumulate, scale_a, scale_b, instr)
elect_predicate(active_mask)
```

`prefetch`, `init`, `wait`, `expect_bytes`, `arrive`, `remote_barrier_addr`,
`umma_arrive`, `commit`,
`fence`, `cta_sync`, `cluster_sync`, `barrier`, `tmem_alloc`, `tmem_free`,
`tmem_relinquish`, `pdl_wait`, `store_wait`, `store_arrive`, and cursor updates
are schedule operations. Address expressions, stage/phase expressions, and
guards are shown directly; they do not hide copies, computation, role changes,
or synchronization.

There are deliberately no computational primitives named `TMA`, `UTCCP`,
`TCGEN05`, `mma`, `stmatrix`, `TensorMap`, `scheduler`, `epilogue`, or `gemm_1d1d`.

`gemm(...)` below is one `tcgen05.mma` instruction covering `UMMA_M x UMMA_N x
UMMA_K`. It is not shorthand for a K loop; the K loop is written out.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    kMajorA=(K, MN), kMajorB=(K, MN),
    kGranKA=(32, 128), kGranKB=(32, 128), kKAlignment=(32, 128, 160, 224),
    SHAPE_M=..., SHAPE_N=..., SHAPE_K=...,     # 0 means "stays runtime"
    BLOCK_M=..., BLOCK_N=..., BLOCK_K=128,
    kNumGroups=...,
    kSwizzleAMode=(16, 32, 64, 128), kSwizzleBMode=(16, 32, 64, 128),
    kSwizzleCDMode=(16, 32, 64, 128),
    kNumStages=1..32,
    kNumNonEpilogueThreads=128, kNumEpilogueThreads=128,
    kNumMulticast=(1, 2), kIsMulticastOnA=(False, True),
    kNumSMs=...,
    kSwapAB=(False, True), kEnsureZeroPadding=(False, True),
    kGemmType=(Normal, MGroupedContiguous, MGroupedContiguousWithPsumLayout,
               MGroupedMasked, KGroupedContiguous,
               KGroupedContiguousWithPsumLayout, Batched),
    kWithAccumulation=(False, True),
    a_dtype=(e4m3,), b_dtype=(e4m3, e2m1_unpacksmem),
    cd_dtype=(bf16, f32),
    epilogue_type=(Identity,),
    target="sm_100f",
)
# instruction_selection: none; extent: the 24 template arguments of
#   `sm100_fp8_fp4_gemm_1d1d_impl`, in the order the host emits them

launch_config = launch(
    grid=(kNumSMs, 1, 1),               # persistent: one block per SM
    cluster=(kNumMulticast, 1, 1),
    block=(kNumNonEpilogueThreads + kNumEpilogueThreads, 1, 1),   # 256
    min_blocks_per_sm=1,
    dynamic_smem_bytes=smem_size,       # from the host pipeline config
    programmatic_dependent_launch=True,
)
# instruction_selection: none; extent: static launch metadata; `__launch_bounds__(256, 1)`

def sm100_fp8_fp4_gemm_1d1d(
    grouped_layout,   # int32* ; meaning depends on kGemmType, may be a dummy
    shape_m,          # runtime M
    shape_n,          # runtime N
    shape_k,          # runtime K (summed K for the k-grouped types)
    tmap_a,           # by-value 128-byte TensorMap, grid-constant
    tmap_b,           # by-value 128-byte TensorMap, grid-constant
    tmap_sfa,         # by-value 128-byte TensorMap, grid-constant
    tmap_sfb,         # by-value 128-byte TensorMap, grid-constant
    tmap_cd,          # by-value 128-byte TensorMap, grid-constant
):

    # =======================================================================
    # Compile-time derived constants (no emitted code)
    # =======================================================================

    LAYOUT_AD_M          = 128
    UMMA_M               = LAYOUT_AD_M * kNumMulticast
    UMMA_N               = BLOCK_M if kSwapAB else BLOCK_N
    UMMA_K               = 32
    LOAD_BLOCK_M         = BLOCK_M // (kNumMulticast if kIsMulticastOnA else 1)
    LOAD_BLOCK_N         = BLOCK_N // (1 if kIsMulticastOnA else kNumMulticast)
    # host_assert: BLOCK_K == 128; BLOCK_K % UMMA_K == 0; kNumMulticast in (1, 2)
    # host_assert (k-grouped only): kGranKA == kGranKB; kKAlignment % UMMA_K == 0
    # host_assert: (kSwapAB and BLOCK_N == 128) or
    #              (not kSwapAB and BLOCK_M in (32, 64, 128))
    # host_assert: (UMMA_M, UMMA_N) satisfies the tcgen05 shape table

    kNumUTCCPAlignedElems = 128
    SF_BLOCK_M            = align_up(BLOCK_M, kNumUTCCPAlignedElems)
    SF_BLOCK_N            = align_up(BLOCK_N, kNumUTCCPAlignedElems)
    kNumSFAStagesPerLoad  = 1 if kGranKA == 32 else 4
    kNumSFBStagesPerLoad  = 1 if kGranKB == 32 else 4

    kNumEpilogueStages    = 2
    kNumTMAStoreStages    = 2
    STORE_BLOCK_M         = 16 if kSwapAB else min(BLOCK_M, LAYOUT_AD_M)
    STORE_BLOCK_N         = BLOCK_N if kSwapAB else kSwizzleCDMode // sizeof(cd_dtype)
    kNumUMMAStoreThreads  = kNumEpilogueThreads if kSwapAB else STORE_BLOCK_M

    SMEM_CD_PER_STAGE     = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype)
    SMEM_CD_SIZE          = SMEM_CD_PER_STAGE * kNumTMAStoreStages
    SMEM_A_PER_STAGE      = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype)   # FP4 counts 1 B/elem
    SMEM_B_PER_STAGE      = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype)
    SMEM_SFA_PER_STAGE    = SF_BLOCK_M * 4
    SMEM_SFB_PER_STAGE    = SF_BLOCK_N * 4

    # UMMA addresses the A operand padded up to LAYOUT_AD_M rows, so its descriptor
    # legally reads past one A stage slab and into the following A/B stages. The
    # source guarantees that stays in bounds rather than shrinking the read.
    UMMA_A_SIZE_PER_STAGE = align_up(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype)
    # host_assert: UMMA_A_SIZE_PER_STAGE <= SMEM_A_PER_STAGE + SMEM_B_PER_STAGE * kNumStages

    # TMA splits each box into swizzle atoms along the *inner* dimension, which is
    # BLOCK_K for a K-major operand and the load block for an MN-major one. With an
    # unswizzled descriptor the atom is the whole inner block, which is why one
    # scale-factor load (`tma::copy<BLOCK_M, 1, 0>`) is a single instruction.
    inner_a = BLOCK_K if kMajorA == K else LOAD_BLOCK_M
    inner_b = BLOCK_K if kMajorB == K else LOAD_BLOCK_N
    atom_a  = inner_a if kSwizzleAMode == 0 else kSwizzleAMode // sizeof(a_dtype)
    atom_b  = inner_b if kSwizzleBMode == 0 else kSwizzleBMode // sizeof(b_dtype)

    kNumAccumTmemCols     = UMMA_N * kNumEpilogueStages
    kTmemStartColOfSFA    = kNumAccumTmemCols
    kTmemStartColOfSFB    = kNumAccumTmemCols + SF_BLOCK_M // 32
    kNumTmemCols          = round_up_to(kNumAccumTmemCols + SF_BLOCK_M // 32
                                        + SF_BLOCK_N // 32, {32, 64, 128, 256, 512})
    # instruction_selection: none; extent: compile-time constants only

    # =======================================================================
    # Cluster rendezvous before the 2-CTA TMEM allocation
    # =======================================================================

    if kNumMulticast > 1:
        cluster_sync(arrive="relaxed")
        # instruction_selection: barrier.cluster.arrive.relaxed.aligned then
        #   barrier.cluster.wait.aligned; extent: all 256 threads of both CTAs

    # =======================================================================
    # Role identifiers
    # =======================================================================

    is_leader_cta = (cta_id_in_cluster() == 0)
    # instruction_selection: mov.u32 from %cluster_ctarank plus setp.eq.u32;
    #   extent: scalar per thread; folded to constant true when kNumMulticast == 1
    warp = shuffle_index(thread_id() // 32, source_lane=0, mask=0xFFFFFFFF, clamp=0x1F)
    # instruction_selection: mov.u32 %tid.x, shr.u32, shfl.sync.idx.b32; extent: warp-uniform scalar
    lane = thread_id() % 32
    # instruction_selection: mov.u32 %laneid; extent: scalar per thread

    # =======================================================================
    # TensorMap prefetch
    # =======================================================================

    if warp == 0:
        prefetch(tmap_a)
        # instruction_selection: prefetch.tensormap; extent: one descriptor
        prefetch(tmap_b)
        # instruction_selection: prefetch.tensormap; extent: one descriptor
        prefetch(tmap_sfa)
        # instruction_selection: prefetch.tensormap; extent: one descriptor
        prefetch(tmap_sfb)
        # instruction_selection: prefetch.tensormap; extent: one descriptor
        prefetch(tmap_cd)
        # instruction_selection: prefetch.tensormap; extent: one descriptor

    # =======================================================================
    # Compiled-dimension override and scale-factor K extents
    # =======================================================================

    shape_m = SHAPE_M if SHAPE_M != 0 else shape_m
    shape_n = SHAPE_N if SHAPE_N != 0 else shape_n
    shape_k = SHAPE_K if SHAPE_K != 0 else shape_k
    # instruction_selection: none; extent: compile-time select; a baked dimension
    #   removes its runtime register entirely
    shape_sfa_k = div_ceil(shape_k, kGranKA * 4)
    # instruction_selection: integer add/shr, or none when SHAPE_K is baked; extent: scalar
    shape_sfb_k = div_ceil(shape_k, kGranKB * 4)
    # instruction_selection: integer add/shr, or none when SHAPE_K is baked; extent: scalar

    # =======================================================================
    # Exact dynamic shared-memory layout and lifetimes
    # =======================================================================

    smem_raw = tile("shared", "u8", [smem_size], byte_offset=0,
                    requested_alignment=1024)
    # instruction_selection: none; extent: one dynamic-SMEM allocation; 1024 B is
    #   required by the 128 B swizzle patterns

    smem_cd  = view(smem_raw, cd_dtype,
                    [kNumTMAStoreStages, STORE_BLOCK_M, STORE_BLOCK_N],
                    byte_offset=0, layout="linear_no_swizzle")
    # instruction_selection: none; extent: SMEM_CD_SIZE bytes, epilogue lifetime.
    #   Unlike A and B this view is *not* swizzled: kSwizzleCDMode lives in the CD
    #   TensorMap and in the epilogue's own `col ^= row % (kSwizzleCDMode / 16)`
    #   arithmetic below. Attaching a swizzled layout here would apply it twice.
    smem_a   = view(smem_raw, a_dtype, [kNumStages, LOAD_BLOCK_M, BLOCK_K],
                    byte_offset=SMEM_CD_SIZE,
                    layout="xor_swizzle_%dB" % kSwizzleAMode)
    # instruction_selection: none; extent: kNumStages * SMEM_A_PER_STAGE bytes
    smem_b   = view(smem_raw, b_dtype, [kNumStages, LOAD_BLOCK_N, BLOCK_K],
                    byte_offset=SMEM_CD_SIZE + kNumStages * SMEM_A_PER_STAGE,
                    layout="xor_swizzle_%dB" % kSwizzleBMode)
    # instruction_selection: none; extent: kNumStages * SMEM_B_PER_STAGE bytes
    sf_base  = SMEM_CD_SIZE + kNumStages * (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE)
    smem_sfa = view(smem_raw, "u32", [kNumStages, SF_BLOCK_M], byte_offset=sf_base,
                    layout="linear_no_swizzle")
    # instruction_selection: none; extent: kNumStages * SMEM_SFA_PER_STAGE bytes;
    #   each u32 packs four ue8m0 bytes
    smem_sfb = view(smem_raw, "u32", [kNumStages, SF_BLOCK_N],
                    byte_offset=sf_base + kNumStages * SMEM_SFA_PER_STAGE,
                    layout="linear_no_swizzle")
    # instruction_selection: none; extent: kNumStages * SMEM_SFB_PER_STAGE bytes

    bar_base      = sf_base + kNumStages * (SMEM_SFA_PER_STAGE + SMEM_SFB_PER_STAGE)
    full          = view(smem_raw, "mbarrier.b64", [kNumStages], byte_offset=bar_base)
    # instruction_selection: none; extent: kNumStages barrier slots
    empty         = view(smem_raw, "mbarrier.b64", [kNumStages],
                         byte_offset=bar_base + 8 * kNumStages)
    # instruction_selection: none; extent: kNumStages barrier slots
    with_sf_full  = view(smem_raw, "mbarrier.b64", [kNumStages],
                         byte_offset=bar_base + 16 * kNumStages)
    # instruction_selection: none; extent: kNumStages barrier slots
    tmem_full     = view(smem_raw, "mbarrier.b64", [kNumEpilogueStages],
                         byte_offset=bar_base + 24 * kNumStages)
    # instruction_selection: none; extent: two barrier slots
    tmem_empty    = view(smem_raw, "mbarrier.b64", [kNumEpilogueStages],
                         byte_offset=bar_base + 24 * kNumStages + 16)
    # instruction_selection: none; extent: two barrier slots
    tmem_ptr_slot = view(smem_raw, "u32", [1],
                         byte_offset=bar_base + 24 * kNumStages + 32)
    # instruction_selection: none; extent: four bytes holding the TMEM base column

    # =======================================================================
    # Barrier initialization and tensor-memory allocation
    # =======================================================================

    if warp == 1:
        elected = elect_predicate(active_mask=0xFFFFFFFF)
        # instruction_selection: elect.sync; extent: one predicate in warp 1
        if elected:
            for s in static_range(kNumStages):
                init(full[s], arrival_count=1)
                # instruction_selection: mbarrier.init.shared.b64; extent: one slot
                init(empty[s], arrival_count=1)
                # instruction_selection: mbarrier.init.shared.b64; extent: one slot
                init(with_sf_full[s], arrival_count=kNumMulticast * 32)
                # instruction_selection: mbarrier.init.shared.b64; extent: one slot;
                #   the transposer arrives once per *thread*, hence 32 per CTA
            for e in static_range(kNumEpilogueStages):
                init(tmem_full[e], arrival_count=1)
                # instruction_selection: mbarrier.init.shared.b64; extent: one slot
                init(tmem_empty[e], arrival_count=kNumMulticast * kNumUMMAStoreThreads)
                # instruction_selection: mbarrier.init.shared.b64; extent: one slot
            fence(scope="mbarrier_init", order="release", visibility="cluster")
            # instruction_selection: fence.mbarrier_init.release.cluster; extent: warp 1
    elif warp == 2:
        tmem_alloc(tmem_ptr_slot, columns=kNumTmemCols, cta_group=kNumMulticast)
        # instruction_selection: tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32;
        #   extent: one allocation of kNumTmemCols columns, base written to SMEM

    if kNumMulticast > 1:
        cluster_sync(arrive="relaxed")
        # instruction_selection: barrier.cluster.arrive.relaxed.aligned +
        #   barrier.cluster.wait.aligned; extent: both CTAs
    else:
        cta_sync()
        # instruction_selection: bar.sync 0; extent: all 256 threads

    pdl_wait()
    # instruction_selection: griddepcontrol.wait; extent: all 256 threads; waits for
    #   the primary kernel of a programmatic dependent launch
```

### Persistent scheduler and pipeline cursor

The scheduler is constructed identically by warps 0, 1, 2 and the epilogue
warps; each keeps its own copy of the state below. Nothing here is shared
memory, and no synchronization happens inside it.

```python
    # `get_num_1d_blocks_per_group`: pick the L2 group width that minimises the
    # bytes touched per wave.
    kNum1DBlocksPerGroup = argmin_over({8, 16}, lambda c:
        c * BLOCK_N + div_ceil(kNumSMs, c) * BLOCK_M if kIsMulticastOnA
        else c * BLOCK_M + div_ceil(kNumSMs, c) * BLOCK_N)
    # instruction_selection: none; extent: compile-time constant in {8, 16}
    # host_assert: kNum1DBlocksPerGroup % kNumMulticast == 0

    sched = reg_tile("i32", [17])   # role-local scheduler state, named below
    # instruction_selection: none; extent: registers only
    sched.current_iter            = -1
    sched.num_m_blocks            = div_ceil(shape_m, BLOCK_M)
    sched.num_n_blocks            = div_ceil(shape_n, BLOCK_N)
    sched.current_shape_k         = shape_k
    sched.current_group_idx       = 0
    sched.current_m_cumsum        = 0        # MGroupedMasked only
    sched.last_psum_m             = 0        # psum contiguous only
    sched.current_m_block_cumsum  = 0        # psum contiguous only
    sched.current_num_valid_groups = 0       # k-grouped only
    sched.current_k_cumsum        = 0        # k-grouped only
    sched.current_sf_k_cumsum     = 0        # k-grouped only
    sched.current_k_start         = 0        # k-grouped psum only
    sched.current_k_end           = 0        # k-grouped psum only
    # instruction_selection: mov.u32 immediates; extent: one register each; the
    #   fields a specialization never reads are dead and disappear

    if kGemmType in (Normal, Batched, MGroupedContiguous):
        sched.num_blocks = sched.num_m_blocks * sched.num_n_blocks
        # instruction_selection: mul.lo.s32; extent: scalar
    elif kGemmType == MGroupedContiguousWithPsumLayout:
        sched.current_psum_m = copy_g2r(grouped_layout[0])
        # instruction_selection: ld.global.b32; extent: one scalar
        sched.num_m_blocks = div_ceil(sched.current_psum_m, BLOCK_M)
        # instruction_selection: integer add/shr; extent: scalar
    elif kGemmType in (KGroupedContiguous, KGroupedContiguousWithPsumLayout):
        sched.num_blocks = sched.num_m_blocks * sched.num_n_blocks
        # instruction_selection: mul.lo.s32; extent: scalar
        if kGemmType == KGroupedContiguousWithPsumLayout:
            get_next_psum_k_group(sched)
        else:
            get_next_k_group(sched, into="current")
            sched.next_group_idx = sched.current_group_idx + 1
            get_next_k_group(sched, into="next")

    # -----------------------------------------------------------------------
    # Group walks used by the k-grouped types
    # -----------------------------------------------------------------------

    def get_next_k_group(sched, into):
        # Advance past empty groups; `grouped_layout[g]` is that group's K length.
        while sched[into].group_idx < kNumGroups:
            sched[into].shape_k = copy_g2r(grouped_layout[sched[into].group_idx])
            # instruction_selection: ld.global.b32; extent: one scalar per probe
            if sched[into].shape_k > 0:
                break
            # instruction_selection: setp.gt.s32 + bra; extent: one loop-exit branch
            sched[into].group_idx = sched[into].group_idx + 1
            # instruction_selection: add.s32; extent: scalar

    def get_next_psum_k_group(sched):
        # `grouped_layout[g]` is the *end* offset in K; each group starts at
        # `align(previous_end, kKAlignment)`.
        while sched.current_group_idx < kNumGroups:
            next_k_end = copy_g2r(grouped_layout[sched.current_group_idx])
            # instruction_selection: ld.global.b32; extent: one scalar per probe
            sched.current_k_start = align_up(sched.current_k_end, kKAlignment)
            # instruction_selection: integer add/and (kKAlignment is a constant); extent: scalar
            sched.current_shape_k = next_k_end - sched.current_k_start
            # instruction_selection: sub.s32; extent: scalar
            sched.current_k_end = next_k_end
            # instruction_selection: mov.b32; extent: scalar
            if sched.current_shape_k > 0:
                break
            # instruction_selection: setp.gt.s32 + bra; extent: one loop-exit branch
            sched.current_group_idx = sched.current_group_idx + 1
            # instruction_selection: add.s32; extent: scalar

    # -----------------------------------------------------------------------
    # L2 swizzle: group `kNum1DBlocksPerGroup` blocks along the multicast axis
    # -----------------------------------------------------------------------

    def get_swizzled_block_idx(sched, block_idx):
        primary   = sched.num_n_blocks if kIsMulticastOnA else sched.num_m_blocks
        secondary = sched.num_m_blocks if kIsMulticastOnA else sched.num_n_blocks
        num_per_group = secondary * kNum1DBlocksPerGroup
        # instruction_selection: mul.lo.s32; extent: scalar
        group_idx = block_idx // num_per_group
        # instruction_selection: integer division sequence; extent: scalar
        first_block_idx = group_idx * kNum1DBlocksPerGroup
        # instruction_selection: mul.lo.s32 (shift when the constant is a power of two); extent: scalar
        in_group_idx = block_idx % num_per_group
        # instruction_selection: integer remainder sequence; extent: scalar
        sched.num_blocks_in_group = min_u32(kNum1DBlocksPerGroup,
                                            primary - first_block_idx)
        # instruction_selection: sub.s32 + min.u32; extent: scalar
        # The SM90 unaligned-multicast fixup is `#if __CUDA_ARCH__ < 1000` and is
        # therefore absent from every specialization in scope.
        if kIsMulticastOnA:
            m_block_idx = in_group_idx // sched.num_blocks_in_group
            n_block_idx = first_block_idx + in_group_idx % sched.num_blocks_in_group
        else:
            m_block_idx = first_block_idx + in_group_idx % sched.num_blocks_in_group
            n_block_idx = in_group_idx // sched.num_blocks_in_group
        # instruction_selection: integer division/remainder sequence + add.s32;
        #   extent: two scalars
        return m_block_idx, n_block_idx

    # -----------------------------------------------------------------------
    # Global index of one block along MN, K, or SF-K
    # -----------------------------------------------------------------------

    def get_global_idx(sched, with_group_offset, index_type,
                       shape_dim, block_size, block_idx, m_block_idx=0):
        if kGemmType == Normal:
            return block_idx * block_size
            # instruction_selection: mul.lo.s32 or shl.b32; extent: scalar
        if kGemmType == MGroupedContiguous:
            # `grouped_layout` holds a per-row expert id; -1 marks padding rows.
            expert = copy_g2r(grouped_layout[m_block_idx * BLOCK_M]) if with_group_offset else 0
            # instruction_selection: ld.global.b32 (elided when the offset is unused); extent: one scalar
            offset = max_i32(0, expert) if with_group_offset else 0
            # instruction_selection: max.s32; extent: scalar
            return offset * shape_dim + block_idx * block_size
            # instruction_selection: mad.lo.s32; extent: scalar
        if kGemmType in (MGroupedMasked, MGroupedContiguousWithPsumLayout):
            offset = sched.current_group_idx if with_group_offset else 0
            return offset * shape_dim + block_idx * block_size
            # instruction_selection: mad.lo.s32; extent: scalar
        if kGemmType in (KGroupedContiguous, KGroupedContiguousWithPsumLayout):
            if not with_group_offset:
                offset = 0
            elif index_type == MN:
                offset = sched.current_group_idx * shape_dim
                # instruction_selection: mul.lo.s32; extent: scalar
            elif index_type == K:
                offset = (sched.current_k_start
                          if kGemmType == KGroupedContiguousWithPsumLayout
                          else sched.current_k_cumsum)
                # instruction_selection: none (register select); extent: scalar
            else:   # SF_K
                offset = sched.current_sf_k_cumsum
            return offset + block_idx * block_size
            # instruction_selection: mad.lo.s32; extent: scalar
        if kGemmType == Batched:
            # `with_group_offset` is ignored; only SF_K carries a batch offset.
            offset = sched.current_group_idx if index_type == SF_K else 0
            return offset * shape_dim + block_idx * block_size
            # instruction_selection: mad.lo.s32; extent: scalar

    # -----------------------------------------------------------------------
    # Effective M inside one block; only the psum contiguous type shortens it
    # -----------------------------------------------------------------------

    def get_aligned_effective_m_in_block(sched, m_block_idx):
        UMMA_STEP_N = 16
        # host_assert: BLOCK_M % UMMA_STEP_N == 0
        if kGemmType == MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding:
            is_last = (m_block_idx == sched.last_psum_m // BLOCK_M
                                      + sched.num_m_blocks - 1)
            # instruction_selection: integer shr + add + setp.eq.s32; extent: predicate
            tail = sched.current_psum_m - m_block_idx * BLOCK_M
            # instruction_selection: mad.lo.s32; extent: scalar
            return align_up(select(is_last, tail, BLOCK_M), UMMA_STEP_N)
            # instruction_selection: selp.b32 + integer add/and; extent: scalar
        return BLOCK_M
        # instruction_selection: none; extent: compile-time constant

    # -----------------------------------------------------------------------
    # The persistent block walk
    # -----------------------------------------------------------------------

    def get_next_block(sched):
        sched.current_iter = sched.current_iter + 1
        # instruction_selection: add.s32; extent: scalar
        next_block_idx = sched.current_iter * kNumSMs + cta_id(axis="x")
        # instruction_selection: mad.lo.s32 with %ctaid.x; extent: scalar

        if kGemmType == MGroupedMasked:
            while True:
                if sched.current_group_idx == kNumGroups:
                    return False
                # instruction_selection: setp.eq.s32 + bra; extent: one exit branch
                sched.num_m_blocks = div_ceil(
                    copy_g2r(grouped_layout[sched.current_group_idx]), BLOCK_M)
                # instruction_selection: ld.global.b32 + integer add/shr; extent: scalar
                cumsum = sched.current_m_cumsum + sched.num_m_blocks
                # instruction_selection: add.s32; extent: scalar
                if next_block_idx < cumsum * sched.num_n_blocks:
                    break
                # instruction_selection: mad.lo.s32 + setp.lt.s32 + bra; extent: one loop branch
                sched.current_group_idx = sched.current_group_idx + 1
                sched.current_m_cumsum = cumsum
                # instruction_selection: add.s32, mov.b32; extent: two scalars
            m_block_idx, n_block_idx = get_swizzled_block_idx(
                sched, next_block_idx - sched.current_m_cumsum * sched.num_n_blocks)

        elif kGemmType == MGroupedContiguousWithPsumLayout:
            while True:
                if next_block_idx < ((sched.current_m_block_cumsum + sched.num_m_blocks)
                                     * sched.num_n_blocks):
                    break
                # instruction_selection: add.s32 + mad.lo.s32 + setp.lt.s32 + bra; extent: one loop branch
                sched.current_group_idx = sched.current_group_idx + 1
                if sched.current_group_idx == kNumGroups:
                    return False
                # instruction_selection: add.s32 + setp.eq.s32 + bra; extent: one exit branch
                sched.last_psum_m = align_up(sched.current_psum_m, BLOCK_M)
                # instruction_selection: integer add/and; extent: scalar
                sched.current_psum_m = copy_g2r(grouped_layout[sched.current_group_idx])
                # instruction_selection: ld.global.b32; extent: one scalar
                sched.current_m_block_cumsum += sched.num_m_blocks
                # instruction_selection: add.s32; extent: scalar
                sched.num_m_blocks = div_ceil(
                    sched.current_psum_m - sched.last_psum_m, BLOCK_M)
                # instruction_selection: sub.s32 + integer add/shr; extent: scalar
            m_block_idx, n_block_idx = get_swizzled_block_idx(
                sched,
                next_block_idx - sched.current_m_block_cumsum * sched.num_n_blocks)
            m_block_idx = m_block_idx + sched.last_psum_m // BLOCK_M
            # instruction_selection: shr.u32 + add.s32; extent: scalar; `last_psum_m`
            #   is block-M aligned by construction

        elif kGemmType in (KGroupedContiguous, KGroupedContiguousWithPsumLayout):
            while True:
                if sched.current_group_idx == kNumGroups:
                    return False
                # instruction_selection: setp.eq.s32 + bra; extent: one exit branch
                if next_block_idx < (sched.current_num_valid_groups + 1) * sched.num_blocks:
                    break
                # instruction_selection: mad.lo.s32 + setp.lt.s32 + bra; extent: one loop branch
                sched.current_sf_k_cumsum += div_ceil(sched.current_shape_k, kGranKA * 4)
                # instruction_selection: integer add/shr + add.s32; extent: scalar
                sched.current_num_valid_groups += 1
                # instruction_selection: add.s32; extent: scalar
                if kGemmType == KGroupedContiguousWithPsumLayout:
                    sched.current_group_idx += 1
                    get_next_psum_k_group(sched)
                else:
                    sched.current_k_cumsum += sched.current_shape_k
                    # instruction_selection: add.s32; extent: scalar
                    sched.current_group_idx = sched.next_group_idx
                    sched.next_group_idx += 1
                    sched.current_shape_k = sched.next_shape_k
                    # instruction_selection: mov.b32 x3, add.s32; extent: four scalars
                    get_next_k_group(sched, into="next")
            m_block_idx, n_block_idx = get_swizzled_block_idx(
                sched,
                next_block_idx - sched.current_num_valid_groups * sched.num_blocks)

        elif kGemmType == Batched:
            if next_block_idx >= sched.num_blocks * kNumGroups:
                return False
            # instruction_selection: setp.ge.s32 + bra; extent: one exit branch
            sched.current_group_idx = next_block_idx // sched.num_blocks
            # instruction_selection: integer division sequence; extent: scalar
            block_idx = next_block_idx - sched.current_group_idx * sched.num_blocks
            # instruction_selection: mad.lo.s32; extent: scalar
            # The batched walk deliberately skips the L2 swizzle.
            if kIsMulticastOnA:
                m_block_idx = block_idx // sched.num_n_blocks
                n_block_idx = block_idx % sched.num_n_blocks
            else:
                m_block_idx = block_idx % sched.num_m_blocks
                n_block_idx = block_idx // sched.num_m_blocks
            # instruction_selection: integer division/remainder sequence; extent: two scalars

        else:   # Normal, MGroupedContiguous
            if next_block_idx >= sched.num_blocks:
                return False
            # instruction_selection: setp.ge.s32 + bra; extent: one exit branch
            # `is_peer_cta_alive` is SM90-only and is dead in every specialization here.
            m_block_idx, n_block_idx = get_swizzled_block_idx(sched, next_block_idx)

        return True

    # -----------------------------------------------------------------------
    # Shared pipeline cursor
    # -----------------------------------------------------------------------

    stage_idx = 0
    phase = 0
    # instruction_selection: mov.b32 zero x2; extent: two role-local registers

    def advance_pipeline(k_block_idx):
        k_block_idx = k_block_idx + 1
        # instruction_selection: add.s32; extent: scalar
        stage_idx = 0 if stage_idx == kNumStages - 1 else stage_idx + 1
        # instruction_selection: setp.eq.s32 + selp.b32 (or add+and for a power-of-two
        #   stage count); extent: scalar
        phase = phase ^ (stage_idx == 0)
        # instruction_selection: setp.eq.s32 + xor.b32; extent: scalar; the phase flips
        #   only when the ring wraps
```

### Role 1: TMA load warp (warp 0, one elected lane)

Every TMA load in this kernel is issued with a multicast count of one, even when
`kNumMulticast == 2`: the two CTAs of a cluster each load their own
`LOAD_BLOCK_M`/`LOAD_BLOCK_N` slice, selected by adding the cluster rank to the
global index. The 2-CTA behaviour lives entirely in the UMMA and the barrier
arrivals, not in the copies.

```python
    if warp == 0 and elect_predicate(active_mask=0xFFFFFFFF):
        # instruction_selection: elect.sync + bra; extent: one lane of warp 0
        while get_next_block(sched):
            load_block_m = (get_aligned_effective_m_in_block(sched, m_block_idx)
                            // kNumMulticast) if kSwapAB else LOAD_BLOCK_M
            # instruction_selection: integer shr (compile-time constant when not
            #   swap-AB or not psum); extent: scalar per block
            num_total_k_blocks = div_ceil(sched.current_shape_k, BLOCK_K)
            # instruction_selection: integer add/shr; extent: scalar per block

            for k_block_idx in runtime_range(num_total_k_blocks, advance=advance_pipeline):
                wait(empty[stage_idx], parity=phase ^ 1)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64
                #   spin loop; extent: one selected stage

                m_idx = get_global_idx(sched, kGemmType == MGroupedMasked, MN,
                                       shape_m, BLOCK_M, m_block_idx)
                # instruction_selection: see `get_global_idx`; extent: scalar
                n_idx = get_global_idx(sched, kMajorB == K, MN,
                                       shape_n, BLOCK_N, n_block_idx, m_block_idx)
                # instruction_selection: see `get_global_idx`; extent: scalar
                # host_assert: kMajorA == K unless kGemmType is Normal, k-grouped, or Batched
                k_idx = k_block_idx * BLOCK_K
                # instruction_selection: shl.b32; extent: scalar
                k_a_idx = get_global_idx(sched, kMajorA == MN, K,
                                         shape_k, BLOCK_K, k_block_idx, m_block_idx)
                # instruction_selection: see `get_global_idx`; extent: scalar
                k_b_idx = get_global_idx(sched, kMajorB == MN, K,
                                         shape_k, BLOCK_K, k_block_idx, m_block_idx)
                # instruction_selection: see `get_global_idx`; extent: scalar

                if kNumMulticast > 1:
                    m_idx = m_idx + (cta_id_in_cluster() * load_block_m
                                     if kIsMulticastOnA else 0)
                    n_idx = n_idx + (0 if kIsMulticastOnA
                                     else cta_id_in_cluster() * LOAD_BLOCK_N)
                    # instruction_selection: mad.lo.s32; extent: one scalar

                batch_idx = sched.current_group_idx if kGemmType == Batched else 0
                # instruction_selection: none (register select); extent: scalar

                # --- A ---------------------------------------------------------
                if kMajorA == K:
                    # The trip count and the SMEM stride come from the *template*
                    # arguments <BLOCK_K, LOAD_BLOCK_M>; the runtime `load_block_m`
                    # above feeds only the cluster-rank offset on `m_idx`.
                    for i in static_range(BLOCK_K // atom_a):
                        copy_g2s(
                            tmap_a.coord(k_a_idx + i * atom_a, m_idx, batch_idx),
                            smem_a[stage_idx].byte(i * LOAD_BLOCK_M * atom_a),
                            completion=full[stage_idx],
                        )
                        # instruction_selection:
                        #   cp.async.bulk.tensor.{2,3}d.shared::cluster.global
                        #   .mbarrier::complete_tx::bytes.L2::cache_hint (EVICT_NORMAL);
                        #   extent: one swizzle atom; 3d only when kGemmType == Batched
                else:   # kMajorA == MN
                    for i in static_range(LOAD_BLOCK_M // atom_a):
                        copy_g2s(
                            tmap_a.coord(m_idx + i * atom_a, k_a_idx, batch_idx),
                            smem_a[stage_idx].byte(i * BLOCK_K * atom_a),
                            completion=full[stage_idx],
                        )
                        # instruction_selection: same TMA family, MN-major coordinates;
                        #   extent: one swizzle atom

                # --- B ---------------------------------------------------------
                if kMajorB == K:
                    for i in static_range(BLOCK_K // atom_b):
                        copy_g2s(
                            tmap_b.coord(k_b_idx + i * atom_b, n_idx, batch_idx),
                            smem_b[stage_idx].byte(i * LOAD_BLOCK_N * atom_b),
                            completion=full[stage_idx],
                        )
                        # instruction_selection: same TMA family; extent: one swizzle atom
                else:   # kMajorB == MN
                    for i in static_range(LOAD_BLOCK_N // atom_b):
                        copy_g2s(
                            tmap_b.coord(n_idx + i * atom_b, k_b_idx, batch_idx),
                            smem_b[stage_idx].byte(i * BLOCK_K * atom_b),
                            completion=full[stage_idx],
                        )
                        # instruction_selection: same TMA family; extent: one swizzle atom

                # FP4 operands transfer half a byte per element while occupying one
                # byte in shared memory, so their expected-bytes contribution halves.
                num_arrival_bytes = (SMEM_A_PER_STAGE // (1 if a_dtype == e4m3 else 2)
                                     + SMEM_B_PER_STAGE // (1 if b_dtype == e4m3 else 2))
                # instruction_selection: none; extent: compile-time constant

                # --- SFA -------------------------------------------------------
                if k_block_idx % kNumSFAStagesPerLoad == 0:
                    sfa_m_idx = m_block_idx * BLOCK_M
                    # instruction_selection: mul.lo.s32; extent: scalar
                    sfa_k_idx = get_global_idx(
                        sched, not kGemmType.is_m_grouped_contiguous, SF_K,
                        shape_sfa_k, 1, div_ceil(k_idx, BLOCK_K * kNumSFAStagesPerLoad))
                    # instruction_selection: integer shr + see `get_global_idx`; extent: scalar
                    copy_g2s(tmap_sfa.coord(sfa_m_idx, sfa_k_idx),
                             smem_sfa[stage_idx], completion=full[stage_idx])
                    # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
                    #   .mbarrier::complete_tx::bytes.L2::cache_hint; extent: exactly one
                    #   instruction -- the SF descriptor is unswizzled, so the whole
                    #   BLOCK_M x 1 box is a single atom
                    num_arrival_bytes += BLOCK_M * 4
                    # instruction_selection: add.s32; extent: scalar

                # --- SFB -------------------------------------------------------
                if k_block_idx % kNumSFBStagesPerLoad == 0:
                    sfb_n_idx = n_block_idx * BLOCK_N
                    # instruction_selection: mul.lo.s32; extent: scalar
                    sfb_k_idx = get_global_idx(
                        sched, True, SF_K,
                        shape_sfb_k, 1, div_ceil(k_idx, BLOCK_K * kNumSFBStagesPerLoad),
                        m_block_idx)
                    # instruction_selection: integer shr + see `get_global_idx`; extent: scalar
                    copy_g2s(tmap_sfb.coord(sfb_n_idx, sfb_k_idx),
                             smem_sfb[stage_idx], completion=full[stage_idx])
                    # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
                    #   .mbarrier::complete_tx::bytes.L2::cache_hint; extent: one instruction
                    num_arrival_bytes += BLOCK_N * 4
                    # instruction_selection: add.s32; extent: scalar

                expect_bytes(full[stage_idx], bytes=num_arrival_bytes)
                # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64;
                #   extent: one producer transaction group covering every copy above
```

### Role 2: UMMA issue warp (warp 1, leader CTA only)

```python
    elif warp == 1 and is_leader_cta:
        instr_desc = desc("instr_block_scaled",
                          a=b_dtype if kSwapAB else a_dtype,
                          b=a_dtype if kSwapAB else b_dtype,
                          acc="f32", sf="ue8m0",
                          M=UMMA_M, N=UMMA_N,
                          major_a=kMajorB if kSwapAB else kMajorA,
                          major_b=kMajorA if kSwapAB else kMajorB)
        # instruction_selection: none; extent: one compile-time instruction descriptor
        #   word; swap-AB exchanges both the operand types and their majors
        sf_desc = desc("smem_matrix", base=None, sbo=8 * 16, lbo=0,
                       layout="swizzle_none")
        # instruction_selection: none; extent: one unswizzled scale-factor descriptor.
        #   `make_sf_desc` sets the *stride* byte offset to 8 * 16 and the *leading*
        #   byte offset to 0 (one atom along K). Both fields are stored shifted right
        #   by 4, and `replace_desc_addr` below rewrites only `start_address = addr >> 4`.
        a_desc = desc("smem_matrix", base=smem_a[0], major=kMajorA,
                      rows=LOAD_BLOCK_M, cols=BLOCK_K, swizzle=kSwizzleAMode)
        # instruction_selection: none; extent: one 64-bit descriptor built from
        #   compile-time strides plus the stage-0 shared address
        b_desc = desc("smem_matrix", base=smem_b[0], major=kMajorB,
                      rows=LOAD_BLOCK_N, cols=BLOCK_K, swizzle=kSwizzleBMode)
        # instruction_selection: none; extent: one 64-bit descriptor

        # The per-stage descriptor low words are held *distributed across lanes*:
        # lane `l` owns stage `l`. This trades kNumStages registers per lane for a
        # single-register table that a warp shuffle can index.
        a_desc_lo = select(lane < kNumStages,
                           a_desc.lo + lane * (SMEM_A_PER_STAGE // 16), 0)
        # instruction_selection: setp.lt.u32 + mad.lo.s32 + selp.b32; extent: one register per lane
        b_desc_lo = select(lane < kNumStages,
                           b_desc.lo + lane * (SMEM_B_PER_STAGE // 16), 0)
        # instruction_selection: setp.lt.u32 + mad.lo.s32 + selp.b32; extent: one register per lane
        # host_assert: kNumStages <= 32

        while get_next_block(sched):
            accum_stage_idx = sched.current_iter % kNumEpilogueStages
            # instruction_selection: and.b32 with 1; extent: scalar
            accum_phase_idx = (sched.current_iter // kNumEpilogueStages) & 1
            # instruction_selection: shr.u32 + and.b32; extent: scalar
            wait(tmem_empty[accum_stage_idx], parity=accum_phase_idx ^ 1)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 LAB_WAIT/DONE spin (ticks=0x989680); extent: one accumulator stage
            fence(scope="tcgen05", position="after_thread_sync")
            # instruction_selection: tcgen05.fence::after_thread_sync; extent: warp 1

            if kSwapAB:
                umma_n = get_aligned_effective_m_in_block(sched, m_block_idx)
                # instruction_selection: see helper; extent: scalar
                instr_desc = update_instr_desc_with_umma_n(instr_desc, umma_n)
                # instruction_selection: integer and/or on the descriptor word; extent: scalar

            num_total_k_blocks = div_ceil(sched.current_shape_k, BLOCK_K)
            # instruction_selection: integer add/shr; extent: scalar
            kMayHaveTailKBlock = (kKAlignment % BLOCK_K != 0
                                  if kGemmType.is_k_grouped_contiguous
                                  else (SHAPE_K == 0 or SHAPE_K % BLOCK_K != 0))
            # instruction_selection: none; extent: compile-time predicate

            for k_block_idx in runtime_range(num_total_k_blocks, unroll_hint=4,
                                             advance=advance_pipeline):
                wait(with_sf_full[stage_idx], parity=phase)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 LAB_WAIT/DONE spin (ticks=0x989680); extent: one stage
                fence(scope="tcgen05", position="after_thread_sync")
                # instruction_selection: tcgen05.fence::after_thread_sync; extent: warp 1

                a_desc_base_lo = shuffle_index(a_desc_lo, source_lane=stage_idx,
                                               mask=0xFFFFFFFF, clamp=0x1F)
                # instruction_selection: shfl.sync.idx.b32; extent: one warp shuffle
                b_desc_base_lo = shuffle_index(b_desc_lo, source_lane=stage_idx,
                                               mask=0xFFFFFFFF, clamp=0x1F)
                # instruction_selection: shfl.sync.idx.b32; extent: one warp shuffle

                if elect_predicate(active_mask=0xFFFFFFFF):
                    # instruction_selection: elect.sync + bra; extent: one lane of warp 1
                    sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad
                    # instruction_selection: and.b32 (or none when the modulus is 1); extent: scalar
                    if sfa_stage_in_group_idx == 0:
                        for i in static_range(SF_BLOCK_M // kNumUTCCPAlignedElems):
                            sf_desc = replace_desc_addr(
                                sf_desc, smem_sfa[stage_idx].elem(i * kNumUTCCPAlignedElems))
                            # instruction_selection: integer and/or on the descriptor word; extent: scalar
                            copy_s2t(sf_desc, tmem[kTmemStartColOfSFA + i * 4])
                            # instruction_selection:
                            #   tcgen05.cp.cta_group::{1,2}.32x128b.warpx4;
                            #   extent: one 128-element scale-factor chunk (4 TMEM columns)
                    sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad
                    # instruction_selection: and.b32 (or none); extent: scalar
                    if sfb_stage_in_group_idx == 0:
                        for i in static_range(SF_BLOCK_N // kNumUTCCPAlignedElems):
                            sf_desc = replace_desc_addr(
                                sf_desc, smem_sfb[stage_idx].elem(i * kNumUTCCPAlignedElems))
                            # instruction_selection: integer and/or; extent: scalar
                            copy_s2t(sf_desc, tmem[kTmemStartColOfSFB + i * 4])
                            # instruction_selection: tcgen05.cp.cta_group::{1,2}.32x128b.warpx4;
                            #   extent: one 128-element chunk

                    def issue_umma(kUMMAKIdx):
                        offset = kUMMAKIdx * UMMA_K
                        # With gran_k == 32 the four packed ue8m0 bytes map to the four
                        # UMMA-K substeps of one BLOCK_K; with gran_k == 128 they map to
                        # four consecutive BLOCK_K iterations.
                        sfa_id = kUMMAKIdx if kGranKA == 32 else sfa_stage_in_group_idx
                        sfb_id = kUMMAKIdx if kGranKB == 32 else sfb_stage_in_group_idx
                        # instruction_selection: none (compile-time or existing register); extent: two scalars
                        runtime_instr_desc = make_instr_desc_with_sf_id(
                            instr_desc,
                            sfb_id if kSwapAB else sfa_id,
                            sfa_id if kSwapAB else sfb_id)
                        # instruction_selection: integer shl/or on the descriptor word; extent: scalar
                        a_desc.lo = advance_desc_lo(a_desc_base_lo, kMajorA, LOAD_BLOCK_M,
                                                    kSwizzleAMode, a_dtype, k_offset=offset)
                        # instruction_selection: add.s32 (constant stride); extent: scalar
                        b_desc.lo = advance_desc_lo(b_desc_base_lo, kMajorB, LOAD_BLOCK_N,
                                                    kSwizzleBMode, b_dtype, k_offset=offset)
                        # instruction_selection: add.s32; extent: scalar
                        gemm(
                            dst=tmem[accum_stage_idx * UMMA_N],
                            lhs=b_desc if kSwapAB else a_desc,
                            rhs=a_desc if kSwapAB else b_desc,
                            accumulate=(kUMMAKIdx > 0 or k_block_idx > 0),
                            scale_a=tmem[kTmemStartColOfSFB if kSwapAB else kTmemStartColOfSFA],
                            scale_b=tmem[kTmemStartColOfSFA if kSwapAB else kTmemStartColOfSFB],
                            instr=runtime_instr_desc,
                        )
                        # instruction_selection:
                        #   tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale (kNumMulticast == 1)
                        #   or tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale (kNumMulticast == 2);
                        #   extent: one UMMA_M x UMMA_N x 32 instruction, SMEM-SMEM operands.
                        #   The same opcode serves FP8 and FP4 operands; the operand format
                        #   travels in the instruction descriptor, not the opcode.

                    if kMayHaveTailKBlock and k_block_idx == num_total_k_blocks - 1:
                        remaining_k = sched.current_shape_k - k_block_idx * BLOCK_K
                        # instruction_selection: mad.lo.s32; extent: scalar
                        if remaining_k < BLOCK_K:
                            num_valid_umma_k = div_ceil(remaining_k, UMMA_K)
                            # instruction_selection: integer add/shr; extent: scalar
                            for kUMMAKIdx in static_prefix(BLOCK_K // UMMA_K,
                                                           num_valid_umma_k):
                                issue_umma(kUMMAKIdx)
                            # instruction_selection: a switch over the BLOCK_K/UMMA_K
                            #   prefix lengths, each arm a straight-line run of
                            #   tcgen05.mma; extent: up to four instructions
                        else:
                            for kUMMAKIdx in static_range(BLOCK_K // UMMA_K):
                                issue_umma(kUMMAKIdx)
                            # instruction_selection: four tcgen05.mma; extent: one full K block
                    else:
                        for kUMMAKIdx in static_range(BLOCK_K // UMMA_K):
                            issue_umma(kUMMAKIdx)
                        # instruction_selection: four tcgen05.mma; extent: one full K block

                barrier(scope="warp")
                # instruction_selection: bar.warp.sync 0xffffffff; extent: warp 1

                # No explicit `tcgen05.fence::before_thread_sync` is needed here:
                # `tcgen05.commit` performs it implicitly.
                umma_arrive(empty[stage_idx], multicast=(kNumMulticast > 1))
                # instruction_selection: elect.sync, then
                #   tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64,
                #   or ...cta_group::2...multicast::cluster.b64 with mask (1<<kNumMulticast)-1;
                #   extent: reached by all 32 lanes of warp 1 -- this call sits *outside*
                #   the elected block above -- but the helper elects internally, so exactly
                #   one commit is issued. That is what makes the count-1 barrier correct.
                if k_block_idx == num_total_k_blocks - 1:
                    umma_arrive(tmem_full[accum_stage_idx], multicast=(kNumMulticast > 1))
                    # instruction_selection: elect.sync then the same tcgen05.commit family;
                    #   extent: one arrival, again elected inside the helper. The accumulator
                    #   pipeline is not multicast-dependent, but the arrival still uses the
                    #   cluster-scoped form when kNumMulticast > 1.
                barrier(scope="warp")
                # instruction_selection: bar.warp.sync 0xffffffff; extent: warp 1

        # Drain: a 2-CTA cluster needs one more accumulator wait before the
        # barriers can be safely destroyed.
        if kNumMulticast > 1 and sched.current_iter - 1 >= 0:
            iter_idx = sched.current_iter - 1
            wait(tmem_empty[iter_idx % kNumEpilogueStages],
                 parity=(iter_idx // kNumEpilogueStages) & 1)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 LAB_WAIT/DONE spin (ticks=0x989680); extent: one drain wait
```

### Role 3: scale-factor transposer warp (warp 2)

`tcgen05.cp ... 32x128b.warpx4` requires each lane's four source words to be
contiguous. The scale factors arrive from TMA in MN-major order with stride 32,
so warp 2 rewrites each 128-element chunk in place before publishing it.

```python
    elif warp == 2:
        def utccp_required_smem_warp_transpose(smem_ptr):
            # host_assert: kNumUTCCPAlignedElems == 128
            values = reg_tile("u32", [4])
            # instruction_selection: none; extent: four registers per lane
            for i in static_range(4):
                copy_s2r(smem_ptr.elem(i * 32 + lane), values[i])
                # instruction_selection: ld.shared.u32; extent: one word per lane
            barrier(scope="warp")
            # instruction_selection: bar.warp.sync 0xffffffff; extent: warp 2;
            #   required because the store below overwrites what other lanes just read
            copy_r2s(values, smem_ptr.elem(lane * 4), vector="v4.u32")
            # instruction_selection: st.shared.v4.u32; extent: 16 bytes per lane

        while get_next_block(sched):
            num_total_k_blocks = div_ceil(sched.current_shape_k, BLOCK_K)
            # instruction_selection: integer add/shr; extent: scalar
            for k_block_idx in runtime_range(num_total_k_blocks, advance=advance_pipeline):
                wait(full[stage_idx], parity=phase)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 LAB_WAIT/DONE spin (ticks=0x989680); extent: one stage

                if k_block_idx % kNumSFAStagesPerLoad == 0:
                    for i in static_range(SF_BLOCK_M // kNumUTCCPAlignedElems):
                        utccp_required_smem_warp_transpose(
                            smem_sfa[stage_idx].elem(i * kNumUTCCPAlignedElems))
                        # instruction_selection: four ld.shared.u32, one bar.warp.sync,
                        #   one st.shared.v4.u32; extent: one 128-element chunk
                    fence(scope="view_async_shared")
                    # instruction_selection: fence.proxy.async.shared::cta; extent: warp 2

                if k_block_idx % kNumSFBStagesPerLoad == 0:
                    for i in static_range(SF_BLOCK_N // kNumUTCCPAlignedElems):
                        utccp_required_smem_warp_transpose(
                            smem_sfb[stage_idx].elem(i * kNumUTCCPAlignedElems))
                        # instruction_selection: four ld.shared.u32, one bar.warp.sync,
                        #   one st.shared.v4.u32; extent: one 128-element chunk
                    fence(scope="view_async_shared")
                    # instruction_selection: fence.proxy.async.shared::cta; extent: warp 2

                remote = remote_barrier_addr(with_sf_full[stage_idx], dst_cta=0)
                # instruction_selection: mapa.shared::cluster.u32; extent: one address
                #   remap per thread. The `0u` argument of the source's `arrive(0u)` is a
                #   *destination CTA rank*, not a count: every thread arrives on the
                #   leader CTA's copy of the barrier.
                arrive(remote)
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one
                #   arrival *per thread*, i.e. 32 per CTA. Both CTAs of a cluster land on
                #   CTA 0's barrier, which is why it was initialized with kNumMulticast * 32.
                #   The cluster-scoped form is emitted even when kNumMulticast == 1.
```

### Role 4: epilogue warps (warps 4 upward)

```python
    elif (kNumNonEpilogueThreads // 32) <= warp \
            < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) // 32:
        epilogue_warp_idx = warp - kNumNonEpilogueThreads // 32
        # instruction_selection: sub.s32; extent: scalar
        # TMEM addresses ignore the warp-index bits in hardware, so no
        # `tmem_ptr |= (epilogue_warp_idx * 32) << 16` is needed. Two CTAs must
        # not share one SM's tensor memory; the source asserts the base is 0.
        device_assert(copy_s2r(tmem_ptr_slot[0]) == 0)
        # instruction_selection: ld.shared.u32 + setp.ne.s32 + trap; extent: one check

        tma_stage_idx = 0
        # instruction_selection: mov.b32 zero; extent: one register carried *across*
        #   scheduler blocks, so the store ring survives block boundaries

        while get_next_block(sched):
            accum_stage_idx = sched.current_iter % kNumEpilogueStages
            # instruction_selection: and.b32 with 1; extent: scalar
            accum_phase_idx = (sched.current_iter // kNumEpilogueStages) & 1
            # instruction_selection: shr.u32 + and.b32; extent: scalar
            wait(tmem_full[accum_stage_idx], parity=accum_phase_idx)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 LAB_WAIT/DONE spin (ticks=0x989680); extent: one accumulator stage
            fence(scope="tcgen05", position="after_thread_sync")
            # instruction_selection: tcgen05.fence::after_thread_sync; extent: epilogue warps

            tmem_base_addr = accum_stage_idx * UMMA_N
            # instruction_selection: mul.lo.s32 (or shl.b32); extent: scalar
            base_m_idx = get_global_idx(sched, not kGemmType.is_m_grouped_contiguous,
                                        MN, shape_m, BLOCK_M, m_block_idx)
            # instruction_selection: see `get_global_idx`; extent: scalar
            base_n_idx = n_block_idx * BLOCK_N
            # instruction_selection: mul.lo.s32; extent: scalar

            if kSwapAB:
                effective_m = get_aligned_effective_m_in_block(sched, m_block_idx)
                store_cd_swap_ab(...)     # below
            else:
                store_cd(...)             # below

    # =======================================================================
    # Teardown
    # =======================================================================

    if kNumMulticast > 1:
        cluster_sync(arrive="relaxed")
        # instruction_selection: barrier.cluster.arrive.relaxed.aligned +
        #   barrier.cluster.wait.aligned; extent: both CTAs
    else:
        cta_sync()
        # instruction_selection: bar.sync 0; extent: all 256 threads

    if warp == 0:
        tmem_free(base=0, columns=kNumTmemCols, cta_group=kNumMulticast)
        # instruction_selection: tcgen05.dealloc.cta_group::N.sync.aligned.b32;
        #   extent: one deallocation. Note the allocating warp (2) and the freeing
        #   warp (0) deliberately differ.
```

#### 4a. Non-swap store path (`sm100_store_cd`)

```python
    def store_cd(smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, batch_idx, epilogue_warp_idx, lane,
                 tmem_empty_barrier, tmap_cd):
        kNumBankGroupBytes    = 16
        kNumElemsPerBankGroup = kNumBankGroupBytes // sizeof(cd_dtype)   # 4 f32 / 8 bf16
        kNumMWaves            = BLOCK_M // STORE_BLOCK_M
        kNumStores            = BLOCK_N // STORE_BLOCK_N
        kHasShortcut          = (kSwizzleCDMode // kNumBankGroupBytes) == 8
        # host_assert: kSwizzleCDMode > 0; STORE_BLOCK_N % kNumElemsPerBankGroup == 0
        # host_assert: BLOCK_M % STORE_BLOCK_M == 0; BLOCK_N % STORE_BLOCK_N == 0

        for w in static_range(kNumMWaves):
            for s in static_range(kNumStores):
                if epilogue_warp_idx == 0:
                    store_wait(pending=kNumTMAStoreStages - 1)
                    # instruction_selection: cp.async.bulk.wait_group.read 1; extent: one warp
                barrier(barrier_id=8, arrival_count=kNumUMMAStoreThreads)
                # instruction_selection: bar.sync 8, kNumUMMAStoreThreads; extent: the store warps.
                #   CUTLASS `NamedBarrier::sync(n, 0)` adds `ReservedNamedBarrierCount = 8`,
                #   so the emitted id is 8. Id 0 belongs to `__syncthreads()`; reusing it
                #   with a partial thread count would deadlock.

                m_idx = base_m_idx + w * STORE_BLOCK_M
                # instruction_selection: add.s32; extent: scalar
                n_idx = base_n_idx + s * STORE_BLOCK_N       # EpilogueIdentity
                # instruction_selection: add.s32; extent: scalar; `apply_index_n` is the
                #   identity for the only epilogue type in scope

                for i in static_range(STORE_BLOCK_N // kNumElemsPerBankGroup):
                    bank_group_index = i + lane * (kSwizzleCDMode // kNumBankGroupBytes)
                    # instruction_selection: mad.lo.s32; extent: scalar
                    row = (i // 8 + lane) if kHasShortcut else (bank_group_index // 8)
                    col = i if kHasShortcut else (bank_group_index % 8)
                    # instruction_selection: integer shr/and (constant divisors); extent: two scalars
                    col = bitwise_xor(col, row % (kSwizzleCDMode // 16))
                    # instruction_selection: and.b32 + xor.b32; extent: scalar

                    tmem_addr = (tmem_base_addr + w * BLOCK_N
                                 + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup)
                    # instruction_selection: add.s32 chain; extent: scalar
                    smem_ptr = (smem_cd[tma_stage_idx].byte(0)
                                + epilogue_warp_idx * 32 * kSwizzleCDMode
                                + row * (kNumBankGroupBytes * 8)
                                + col * kNumBankGroupBytes)
                    # instruction_selection: mad.lo.s32 chain; extent: scalar

                    if cd_dtype == f32:
                        values = reg_tile("u32", [4])
                        copy_t2r(tmem[tmem_addr], values)
                        # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x4.b32;
                        #   extent: four f32 lanes per thread
                        fence(scope="view_async_tmem_load")
                        # instruction_selection: tcgen05.wait::ld.sync.aligned; extent: one wait
                        copy_r2s(values, smem_ptr, vector="v4.u32")
                        # instruction_selection: st.shared.v4.u32; extent: 16 bytes per lane
                    else:   # bf16
                        values = reg_tile("u32", [8])
                        copy_t2r(tmem[tmem_addr], values)
                        # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32;
                        #   extent: eight f32 lanes per thread
                        fence(scope="view_async_tmem_load")
                        # instruction_selection: tcgen05.wait::ld.sync.aligned; extent: one wait
                        packed = reg_tile("u32", [4])
                        for p in static_range(4):
                            cast(packed[p], (values[2 * p], values[2 * p + 1]),
                                 rounding="rn", pack=True)
                            # instruction_selection: cvt.rn.bf16x2.f32; extent: one packed pair
                        copy_r2s(packed, smem_ptr, vector="v4.u32")
                        # instruction_selection: st.shared.v4.u32; extent: 16 bytes per lane

                if w == kNumMWaves - 1 and s == kNumStores - 1:
                    fence(scope="tcgen05", position="before_thread_sync")
                    # instruction_selection: tcgen05.fence::before_thread_sync; extent: store warps
                    remote = remote_barrier_addr(tmem_empty_barrier, dst_cta=0)
                    # instruction_selection: mapa.shared::cluster.u32; extent: one address remap
                    arrive(remote)
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one
                    #   arrival per store thread, on the leader CTA's copy -- which is why the
                    #   barrier was initialized with kNumMulticast * kNumUMMAStoreThreads

                fence(scope="tma_store")
                # instruction_selection: fence.proxy.async.shared::cta; extent: store warps
                barrier(barrier_id=8, arrival_count=kNumUMMAStoreThreads)
                # instruction_selection: bar.sync 8, kNumUMMAStoreThreads; extent: the store warps.
                #   CUTLASS `NamedBarrier::sync(n, 0)` adds `ReservedNamedBarrierCount = 8`,
                #   so the emitted id is 8. Id 0 belongs to `__syncthreads()`; reusing it
                #   with a partial thread count would deadlock.
                if epilogue_warp_idx == 0 and elect_predicate(active_mask=0xFFFFFFFF):
                    # instruction_selection: elect.sync + bra; extent: one lane
                    if kGemmType == Batched:
                        copy_s2g(smem_cd[tma_stage_idx],
                                 tmap_cd.coord(n_idx, m_idx, batch_idx),
                                 reduce="add" if kWithAccumulation else None)
                        # instruction_selection:
                        #   cp.async.bulk.tensor.3d.global.shared::cta.bulk_group, or
                        #   cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group when
                        #   accumulating; extent: one 3-D store
                    else:
                        copy_s2g(smem_cd[tma_stage_idx],
                                 tmap_cd.coord(n_idx, m_idx),
                                 reduce="add" if kWithAccumulation else None)
                        # instruction_selection:
                        #   cp.async.bulk.tensor.2d.global.shared::cta.bulk_group, or
                        #   cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group when
                        #   accumulating; extent: one 2-D store
                    store_arrive()
                    # instruction_selection: cp.async.bulk.commit_group; extent: one group
                barrier(scope="warp")
                # instruction_selection: bar.warp.sync 0xffffffff; extent: one store warp
                tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages
                # instruction_selection: add.s32 + and.b32; extent: scalar
```

#### 4b. Swap-AB store path (`sm100_store_cd_swap_ab`)

With swap-AB the accumulator is transposed relative to the output, so the BF16
path loads two `16x256b` TMEM slices and lets `stmatrix.trans` do the
transpose. `num_stores` is a *runtime* bound here, because the psum layout can
shorten the last M block.

```python
    def store_cd_swap_ab(smem_cd, tma_stage_idx, tmem_base_addr,
                         base_m_idx, base_n_idx, batch_idx, effective_m,
                         epilogue_warp_idx, lane, tmem_empty_barrier, tmap_cd):
        STORE_BLOCK_N_ATOM  = kSwizzleCDMode // sizeof(cd_dtype)
        kNumBankGroupBytes  = 16
        kNumSwizzleAtomRows = 8
        kNumWarpsPerAtom    = STORE_BLOCK_N_ATOM // 32
        # host_assert: STORE_BLOCK_N == 128 (a full warpgroup reads all 128 TMEM rows)
        # host_assert: kSwizzleCDMode == 128
        # host_assert: STORE_BLOCK_M % kNumSwizzleAtomRows == 0
        # host_assert: STORE_BLOCK_N_ATOM % 32 == 0

        num_stores = effective_m // STORE_BLOCK_M
        # instruction_selection: shr.u32; extent: runtime scalar

        for s in runtime_range(num_stores):
            if epilogue_warp_idx == 0:
                store_wait(pending=kNumTMAStoreStages - 1)
                # instruction_selection: cp.async.bulk.wait_group.read 1; extent: one warp
            barrier(barrier_id=8, arrival_count=kNumUMMAStoreThreads)
            # instruction_selection: bar.sync 8, kNumUMMAStoreThreads; extent: the warpgroup
            #   (`NamedBarrier::sync(n, 0)` offsets the id by ReservedNamedBarrierCount = 8)

            for i in static_range(STORE_BLOCK_M // kNumSwizzleAtomRows):
                tmem_addr = tmem_base_addr + s * STORE_BLOCK_M + i * kNumSwizzleAtomRows
                # instruction_selection: mad.lo.s32; extent: scalar
                outer_atom_offset = ((epilogue_warp_idx // kNumWarpsPerAtom)
                                     * STORE_BLOCK_M * kSwizzleCDMode)
                inner_atom_offset = i * kNumSwizzleAtomRows * kSwizzleCDMode
                smem_base_ptr = (smem_cd[tma_stage_idx].byte(0)
                                 + outer_atom_offset + inner_atom_offset)
                # instruction_selection: integer shr + mad.lo.s32; extent: scalar
                values = reg_tile("u32", [8])
                # instruction_selection: none; extent: eight registers per lane

                if cd_dtype == f32:
                    copy_t2r(tmem[tmem_addr], values)
                    # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32;
                    #   extent: eight f32 lanes per thread
                    col = lane // 4
                    # instruction_selection: shr.u32; extent: scalar
                    for row in static_range(kNumSwizzleAtomRows):
                        smem_ptr = (smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                    + bitwise_xor(col, row) * kNumBankGroupBytes
                                    + (lane % 4) * sizeof(f32))
                        # instruction_selection: xor.b32 + mad.lo.s32; extent: scalar
                        copy_r2s(values[row], smem_ptr)
                        # instruction_selection: st.shared.u32; extent: four bytes per lane
                    # Swizzling is unnecessary for this layout but is kept for
                    # consistency with the other store paths.
                else:   # bf16
                    copy_t2r(tmem[tmem_addr], values[0:4])
                    # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x1.b32;
                    #   extent: four registers, lanes starting at 0
                    copy_t2r(tmem[tmem_addr | 0x00100000], values[4:8])
                    # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x1.b32
                    #   with the row-half bit set; extent: four registers, lanes starting at 16
                    fence(scope="view_async_tmem_load")
                    # instruction_selection: tcgen05.wait::ld.sync.aligned; extent: one wait
                    row = lane % 8
                    col = (epilogue_warp_idx % 2) * 4 + lane // 8
                    # instruction_selection: and.b32, shr.u32, mad.lo.s32; extent: two scalars
                    smem_ptr = (smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                + bitwise_xor(col, row) * kNumBankGroupBytes)
                    # instruction_selection: xor.b32 + mad.lo.s32; extent: scalar
                    packed = reg_tile("u32", [4])
                    for p in static_range(4):
                        cast(packed[p], (values[2 * p], values[2 * p + 1]),
                             rounding="rn", pack=True)
                        # instruction_selection: cvt.rn.bf16x2.f32; extent: one packed pair
                    copy_r2s(packed, smem_ptr, transpose=True, vector="x4")
                    # instruction_selection:
                    #   stmatrix.sync.aligned.x4.m8n8.shared.b16.trans;
                    #   extent: one transposed 8x8x4 matrix store per lane group

            if s == num_stores - 1:
                fence(scope="tcgen05", position="before_thread_sync")
                # instruction_selection: tcgen05.fence::before_thread_sync; extent: the warpgroup
                remote = remote_barrier_addr(tmem_empty_barrier, dst_cta=0)
                # instruction_selection: mapa.shared::cluster.u32; extent: one address remap
                arrive(remote)
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one
                #   arrival per store thread, on the leader CTA's copy

            fence(scope="tma_store")
            # instruction_selection: fence.proxy.async.shared::cta; extent: the warpgroup
            barrier(barrier_id=8, arrival_count=kNumUMMAStoreThreads)
            # instruction_selection: bar.sync 8, kNumUMMAStoreThreads; extent: the warpgroup
            #   (`NamedBarrier::sync(n, 0)` offsets the id by ReservedNamedBarrierCount = 8)
            if epilogue_warp_idx == 0 and elect_predicate(active_mask=0xFFFFFFFF):
                # instruction_selection: elect.sync + bra; extent: one lane
                for i in static_range(STORE_BLOCK_N // STORE_BLOCK_N_ATOM):
                    smem_ptr = smem_cd[tma_stage_idx].elem(
                        i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM)
                    m_idx = base_m_idx + s * STORE_BLOCK_M
                    # instruction_selection: mad.lo.s32; extent: scalar
                    n_idx = base_n_idx + i * STORE_BLOCK_N_ATOM      # EpilogueIdentity
                    # instruction_selection: add.s32; extent: scalar
                    if kGemmType == Batched:
                        copy_s2g(smem_ptr, tmap_cd.coord(n_idx, m_idx, batch_idx),
                                 reduce="add" if kWithAccumulation else None)
                        # instruction_selection: cp.async.bulk.tensor.3d.global.shared::cta
                        #   .bulk_group, or cp.reduce.async.bulk.tensor.3d.global.shared::cta
                        #   .add.bulk_group; extent: one store
                    else:
                        copy_s2g(smem_ptr, tmap_cd.coord(n_idx, m_idx),
                                 reduce="add" if kWithAccumulation else None)
                        # instruction_selection: cp.async.bulk.tensor.2d.global.shared::cta
                        #   .bulk_group, or cp.reduce.async.bulk.tensor.2d.global.shared::cta
                        #   .add.bulk_group; extent: one store
                store_arrive()
                # instruction_selection: cp.async.bulk.commit_group; extent: one group
            barrier(scope="warp")
            # instruction_selection: bar.warp.sync 0xffffffff; extent: one store warp
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages
            # instruction_selection: add.s32 + and.b32; extent: scalar
```

## Scheduler walk by GemmType

| `kGemmType` | `grouped_layout` contents | block walk | MN offset | K offset | SF-K offset | L2 swizzle |
| --- | --- | --- | --- | --- | --- | --- |
| `Normal` | unused | `idx >= num_blocks` ends | `block * size` | `block * size` | `block * size` | yes |
| `MGroupedContiguous` | per-row expert id, `-1` = padding | same as `Normal` | `max(0, layout[m_block*BLOCK_M]) * dim + block*size` | `block * size` | no group offset | yes |
| `MGroupedContiguousWithPsumLayout` | per-group psum end offsets in M | walk group psum ends, add `last_psum_m / BLOCK_M` | `group * dim + block*size` | `block * size` | `group * dim + ...` | yes |
| `MGroupedMasked` | per-group valid M counts | walk group M-block cumsum | `group * dim + block*size` | `block * size` | `group * dim + ...` | yes |
| `KGroupedContiguous` | per-group K lengths (skip zeros) | walk valid groups, `num_blocks` each | `group * dim + block*size` | `current_k_cumsum + block*size` | `current_sf_k_cumsum + ...` | yes |
| `KGroupedContiguousWithPsumLayout` | per-group psum end offsets in K | same, group start is `align(prev_end, kKAlignment)` | `group * dim + block*size` | `current_k_start + block*size` | `current_sf_k_cumsum + ...` | yes |
| `Batched` | unused | `idx >= num_blocks * kNumGroups` ends; `group = idx / num_blocks` | `block * size` | `block * size` | `group * dim + block*size` | **no** |

`get_aligned_effective_m_in_block` returns `BLOCK_M` for every type except
`MGroupedContiguousWithPsumLayout` with `kEnsureZeroPadding == False`, where the
last block of each group is shortened and re-aligned to 16.

## TensorMap fields

| ABI parameter | rank | global dims | box | element type | swizzle | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `tmap_a` | 2, or 3 when `kGemmType == Batched` | `(k, m)` for K-major, `(m, k)` for MN-major | `(BLOCK_K, LOAD_BLOCK_M)` or `(LOAD_BLOCK_M, BLOCK_K)` | `e4m3`; FP4 would use `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` | `kSwizzleAMode` | m-grouped and batched fold the group into the outer extent |
| `tmap_b` | 2, or 3 for `Batched` | `(k, n)` or `(n, k)` | `(BLOCK_K, LOAD_BLOCK_N)` or `(LOAD_BLOCK_N, BLOCK_K)` | `e4m3` or `16U4_ALIGN16B` | `kSwizzleBMode` | FP4 requires the global inner dim to be a multiple of 128 |
| `tmap_sfa` | 2 | `(m, ceil_div(k, gran_k_a * 4))` | `(BLOCK_M, 1)` | `int32` (four packed `ue8m0`) | **0** | MN-major, TMA-aligned; one instruction per load |
| `tmap_sfb` | 2 | `(n, ceil_div(k, gran_k_b * 4))` | `(BLOCK_N, 1)` | `int32` | **0** | same |
| `tmap_cd` | 2, or 3 for `Batched` | `(n, m)` | `(STORE_BLOCK_N, STORE_BLOCK_M)` | `bf16` or `f32` | `kSwizzleCDMode` | the store becomes `cp.reduce...add` when `kWithAccumulation` |

Coordinates are always `(inner, outer[, batch])`, matching the source's
`tma::copy(inner_idx, outer_idx, 1, batch_idx)` argument order. Every load
passes a multicast count of one.

## Storage lifetimes and TMEM columns

| Object | Producer | Consumer | Released by |
| --- | --- | --- | --- |
| `smem_a[s]`, `smem_b[s]` | warp 0 TMA, publishes `full[s]` | warp 1 UMMA (through SMEM descriptors) | `umma_arrive(empty[s])` from warp 1 |
| `smem_sfa[s]`, `smem_sfb[s]` | warp 0 TMA, publishes `full[s]` | warp 2 transposes in place, publishes `with_sf_full[s]`; warp 1 UTCCPs into TMEM | same `empty[s]` edge (the SF words live in the same stage slot) |
| TMEM accumulator columns `[e*UMMA_N, (e+1)*UMMA_N)` | warp 1 UMMA, publishes `tmem_full[e]` | epilogue warps | `arrive(tmem_empty[e])` from every store thread |
| TMEM SF columns `[kTmemStartColOfSFA, ...)` | warp 1 UTCCP | warp 1 UMMA (same instruction) | overwritten on the next refresh; no barrier |
| `smem_cd[t]` | epilogue warps | TMA store | `cp.async.bulk.wait_group.read kNumTMAStoreStages-1` |

`with_sf_full` and `tmem_empty` arrivals are **cluster-scoped and always land on
the leader CTA's copy** of the barrier (`mapa.shared::cluster.u32` with destination
rank 0, then `mbarrier.arrive.shared::cluster.b64`), which is why their initial
counts carry the `kNumMulticast *` factor. `full`, `empty` and `tmem_full` are
published by TMA expect-tx and by `tcgen05.commit`, not by a plain arrive.

TMEM column map: `[0, UMMA_N)` accumulator stage 0, `[UMMA_N, 2*UMMA_N)`
accumulator stage 1, then `SF_BLOCK_M/32` SFA columns, then `SF_BLOCK_N/32` SFB
columns, rounded up to one of `{32, 64, 128, 256, 512}`.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `kGemmType` | static, 7 values | selects the scheduler walk, the index offsets, the tail-K rule, and 2-D vs 3-D stores |
| `kSwapAB` | static bool | swaps the UMMA operands and their majors, sets `UMMA_N`, `STORE_BLOCK_M/N` and `kNumUMMAStoreThreads`, and picks the epilogue helper |
| `a_dtype` / `b_dtype` | static | halves the TMA expected-bytes for an FP4 operand and changes the instruction descriptor; the MMA opcode is unchanged |
| `cd_dtype` | static | selects the TMEM-load atom and whether a BF16 pack happens |
| `kGranKA` / `kGranKB` | static, 32 or 128 | sets the SF load cadence and whether `sf_id` tracks the UMMA-K substep or the BLOCK_K group |
| `kNumMulticast` | static, 1 or 2 | cluster dim, `cta_group::N` on the MMA/UTCCP/commit, cluster syncs, barrier arrival counts, and the extra drain wait |
| `kKAlignment` | static | for k-grouped types, decides `kMayHaveTailKBlock` and the psum group start alignment |
| `SHAPE_M/N/K` | static when compiled in, else runtime | a baked `SHAPE_K` can remove the tail-K prefix switch entirely |
| `kEnsureZeroPadding` | static bool | only reachable through `get_aligned_effective_m_in_block` for the psum contiguous type |
| `shape_m/n/k`, `grouped_layout` values | runtime | the persistent walk, group boundaries and `num_stores` stay dynamic |
| 256 threads, `BLOCK_K = 128`, `UMMA_K = 32`, `LAYOUT_AD_M = 128` | static | fixes the launch bounds, the MMA shape table and the SMEM stage sizes |

## TIRx module and benchmark contract

- One shared builder, `_sm100_fp8_fp4_gemm_1d1d.build_kernel(spec)`, with `spec`
  mirroring the 24 C++ template arguments; five registry modules pin one
  descriptor each and expose `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`,
  `get_kernel`, `prepare_data`, `run_test` and `run_bench`.
- `get_best_config` is a transcription of `SM100ArchSpec` and must keep
  reproducing DeepGEMM's layout choice exactly; `.porting/sm100_fp8_fp4_gemm_1d1d/config_parity.py`
  is the regression harness for that claim.
- The kernel is written in plain TIRx: `T.ptx` / `T.cuda` intrinsics, explicit
  loops, and hand-carved shared/tensor-memory buffers. No `Tx` tile primitive may
  appear in any pre-dispatch specialization.
- Correctness compares against a dequantized torch matmul using DeepGEMM's own
  `calc_diff` metric and thresholds (FP8xFP8 below 1e-3, FP8xFP4 below 1e-2), plus
  a cross-check against the corresponding `deep_gemm` entry when it is installed.
- The timed implementation is named `tirx`; quantization, scale packing, TensorMap
  encoding, compilation and validation stay outside the timed closure. The
  reference is supplied as a lazy `references={"deepgemm": ...}` builder, and both
  sides set the same `mk_alignment` before their launch.

## Instruction selection is a lowering consequence

| Sketch construct | PTX | SASS family |
| --- | --- | --- |
| `copy_g2s` on A/B/SFA/SFB | `cp.async.bulk.tensor.{2,3}d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` | `UTMALDG` |
| `copy_s2g` without accumulation | `cp.async.bulk.tensor.{2,3}d.global.shared::cta.bulk_group` | `UTMASTG` |
| `copy_s2g` with `reduce="add"` | `cp.reduce.async.bulk.tensor.{2,3}d.global.shared::cta.add.bulk_group` | `UTMASTG` (reduce variant) |
| `copy_s2t` (scale factors) | `tcgen05.cp.cta_group::{1,2}.32x128b.warpx4` | UTCCP sequence |
| `gemm(...)` | `tcgen05.mma.cta_group::{1,2}.kind::mxf8f6f4.block_scale` | `UTCHMMA` |
| `copy_t2r` f32 / bf16 non-swap | `tcgen05.ld.sync.aligned.32x32b.x{4,8}.b32` | `LDTM` |
| `copy_t2r` bf16 swap-AB | two `tcgen05.ld.sync.aligned.16x256b.x1.b32` | `LDTM` |
| `copy_r2s(..., transpose=True)` | `stmatrix.sync.aligned.x4.m8n8.shared.b16.trans` | `STS` (transposing) |
| `cast(..., pack=True)` | `cvt.rn.bf16x2.f32` | `F2FP` |
| `umma_arrive` | `elect.sync` + `tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one` (or `multicast::cluster`) | matrix completion sequence |
| barrier init / wait / arrive | `mbarrier.init`, `mbarrier.try_wait.parity`, `mbarrier.arrive[.expect_tx]` | `BSSY`/`SYNC` plus `NANOSLEEP` inside the spin loop |
| `tmem_alloc` / `tmem_free` | `tcgen05.alloc` / `tcgen05.dealloc` | TMEM-control sequence |
| `barrier(barrier_id=8, count=N)` | `bar.sync 8, N` | `BAR` |
| `cluster_sync` | `barrier.cluster.arrive.relaxed.aligned` + `barrier.cluster.wait.aligned` | `BAR` (cluster) |
| `remote_barrier_addr(..., dst_cta=0)` | `mapa.shared::cluster.u32` | address remap |
| the lane-distributed descriptor table | `shfl.sync.idx.b32` | `SHFL` |

Per scheduler block and K block, the leader CTA's warp 1 issues at most
`BLOCK_K / UMMA_K = 4` `tcgen05.mma` instructions, plus
`SF_BLOCK_M / 128` and `SF_BLOCK_N / 128` UTCCP copies on the stages where the
scale-factor cadence fires. Warp 0 issues `BLOCK_K / atom_a` plus
`BLOCK_K / atom_b` TMA loads for A and B (or the MN-major equivalents), plus at
most two single-instruction scale-factor loads, all covered by one
`expect_tx` arrival. Warp 2 performs `4` `ld.shared.u32` and one
`st.shared.v4.u32` per 128-element scale-factor chunk. The epilogue performs
`BLOCK_M / STORE_BLOCK_M * BLOCK_N / STORE_BLOCK_N` store stages without swap-AB
and `effective_m / 16` with it.

`SYNC` and `NANOSLEEP` counts come from the `mbarrier` spin loops and are not
directly controllable; they are not an alignment target.
