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
deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh together with the
scheduler, layout and comm helpers it instantiates. See NOTICE and licenses/
for upstream attribution.
-->

# SM100 FP8/FP4 MegaMoE: coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new
IR, a builder API, or a mathematical reference implementation. Its purpose is to
show the TIRx kernel as:

- an explicit runtime ABI and launch;
- explicit GMEM, SMEM, TMEM, and register tiles, including byte and column
  offsets;
- the persistent task scheduler and the warp-role control flow;
- primitive directional copies and primitive computation inside every reachable
  helper;
- explicit stage/index/phase changes and publication/reuse edges;
- hardware instruction selection derived only after placement, shape, layout,
  and schedule have been stated.

The implementation represented by this sketch is maintained in
[`tirx_kernels/deepgemm/_sm100_fp8_fp4_mega_moe/`](../../tirx_kernels/deepgemm/_sm100_fp8_fp4_mega_moe/__init__.py)
with the thin entry module
[`tirx_kernels/deepgemm/sm100_fp8_fp4_mega_moe.py`](../../tirx_kernels/deepgemm/sm100_fp8_fp4_mega_moe.py).
Those modules are the source of truth.

DeepGEMM has one SM100 FP8/FP4 MegaMoE device template. This sketch fixes the
`kNumSharedExperts > 0` specialization, because that is the one this port is
being extended to cover; every `kHasShared` branch is written out, and the
`kNumSharedExperts == 0` degenerate is marked at each such branch (it folds
`is_shared()` to a compile-time `false` and deletes the shared regions).

**In scope.** One fused MegaMoE megakernel: EP dispatch (token routing plus
NVLink pull), a persistent three-phase task schedule, the L1 (gate/up + SwiGLU)
and L2 (down-projection) GEMMs for routed experts, the SharedLinear1 and
SharedLinear2 GEMMs for `S = kNumSharedExperts > 0` fused shared experts, the
NVLink combine write-back, and the top-k + shared reduction. FP8 e4m3
activations with UE8M0 1x32 block scale factors; routed weights packed FP4
(e2m1); shared weights FP8 e4m3. `swiglu` activation with optional clamp;
`kFastMath` both ways. `kNumRanks` 1..72.

**Out of scope.** `sm100_bf16_mega_moe` (the BF16 sibling template); activations
other than `swiglu`; recipes other than `(1, 1, 32)`; tile (`Tx`) primitives,
which may not appear in any specialization.

## Pipeline at a glance

`kNumMMANonEpilogueWarps == 4`, so the four warps after the dispatch warps are
the GEMM producer/consumer roles and everything from
`kNumDispatchWarps + 4` upward is the epilogue.

| Warp | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| `0 .. kNumDispatchWarps-1` | count expert tokens, claim remote slots, write source indices over NVLink, pull token+SF+weight into the routed L1 ring, then clean the workspace for the next launch | publish `l1_full_count[ring]`; wait `l1_empty_count[ring]`; grid sync + NVLink barriers |
| `kNumDispatchWarps` | drive the scheduler stream; per task select one of four A/SFA TensorMaps, wait the phase-specific producer counter, then issue A and SFA TMA loads per K block | wait `empty[stage]`; publish `full[stage]` with an expect-tx byte count |
| `kNumDispatchWarps + 1` | drive the same stream; select one of four B/SFB TensorMaps and issue B and SFB TMA loads, with a dtype-split copy (FP4 routed / FP8 shared) | wait `empty[stage]`; publish `full[stage]`, expect-tx differs by operand dtype |
| `kNumDispatchWarps + 2` | leader CTA only: UTCCP the scale factors into TMEM and issue the UMMA chain, choosing between the routed (e2m1) and shared (e4m3) instruction descriptors | wait `full[stage]` and `tmem_empty[accum]`; arrive `empty[stage]`, and `tmem_full[accum]` on the last K block |
| `kNumDispatchWarps + 3` | leader CTA only: the task scheduler mainloop — SharedLinear1 tasks, then routed L1/L2 tasks, then SharedLinear2 tasks, then a sentinel | wait `task_info_empty[s]`; publish `task_info_full[s]` by `st_async` into the peer CTA |
| `kNumDispatchWarps + 4 ..` | epilogue: L1/SharedL1 does SwiGLU + FP8 cast + TMA store and signals the L2 producer counter; L2/SharedL2 does BF16 cast + NVLink combine write. Afterwards every epilogue warp runs the combine reduction | wait `tmem_full[accum]`; arrive `tmem_empty[accum]`; publish `l2_full_count` / `shared_l2_full_count` |

All GEMM roles consume the *same* published `task_info` stream, so the schedule
is decided once by warp `kNumDispatchWarps + 3` and broadcast; the consumers
never re-derive it. Shared-expert tasks are not interleaved with routed tasks —
they are two extra runs of the same consumer machinery, bracketing the routed
stream.

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
peer(ptr, rank)       # map a symmetric-buffer pointer onto another rank
```

Copies always state their storage direction:

```python
copy_g2s(src, dst, completion=None)   # global -> shared, TMA
copy_s2g(src, dst)                    # shared -> global, TMA
copy_g2r(src, dst)                    # global -> register
copy_r2g(src, dst)                    # register -> global (incl. peer ranks)
copy_s2r(src, dst)                    # shared -> register
copy_r2s(src, dst, transpose=False)   # register -> shared
copy_s2t(src, dst)                    # shared -> tensor memory (UTCCP)
copy_t2r(src, dst)                    # tensor memory -> register
```

The complete computational vocabulary used below is:

```python
fill(dst, value)
cast(dst, src, rounding=None, pack=False)
add(dst, lhs, rhs);  sub(dst, lhs, rhs);  mul(dst, lhs, rhs);  div(dst, lhs, rhs)
exp(dst, src);  rcp(dst, src);  abs(dst, src);  max(dst, lhs, rhs);  min(dst, lhs, rhs)
div_ceil(dst, lhs, rhs);  align_up(dst, value, granularity)
bitwise_and / bitwise_or / bitwise_xor / shift_left / shift_right
move(dst, src);  select(dst, predicate, true_value, false_value)
shuffle_index(dst, src, source_lane, mask, clamp)
ballot(dst, predicate);  popc(dst, mask);  ffs(dst, mask);  fns(dst, mask, base, n)
warp_reduce_add(dst, src);  warp_reduce_min(dst, src);  warp_reduce_max(dst, src, width)
atomic_add(dst, address, value);  red_add(address, value);  red_add_rel(address, value)
gemm(dst, lhs, rhs, accumulate, scale_a, scale_b, instr)
elect_predicate(active_mask)
```

`prefetch`, `init`, `wait`, `expect_bytes`, `arrive`, `st_async`,
`umma_arrive`, `commit`, `fence`, `cta_sync`, `cluster_sync`, `barrier`,
`warp_sync`, `tmem_alloc`, `tmem_free`, `store_wait`, `store_arrive`,
`grid_sync`, `nvlink_barrier`, `reg_dealloc`, `reg_alloc` and cursor updates are
schedule operations. Address expressions, stage/phase expressions and guards are
shown directly; they do not hide copies, computation, role changes or
synchronization.

There are deliberately no computational primitives named `TMA`, `UTCCP`,
`TCGEN05`, `mma`, `stmatrix`, `TensorMap`, `dispatch`, `combine`, `scheduler`,
`swiglu`, or `mega_moe`.

`gemm(...)` below is one `tcgen05.mma` instruction covering
`UMMA_M x UMMA_N x UMMA_K`. It is not shorthand for a K loop; the K loop is
written out.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    kNumMaxTokensPerRank=T, kHidden=H, kIntermediateHidden=I,
    kNumExperts=E, kNumSharedExperts=S,          # S > 0 in this sketch
    kNumTopk=K,
    BLOCK_M=(16, 32, 64, 96, 128, 192), BLOCK_N=128, BLOCK_K=(128, 256),
    STORE_BLOCK_M=..., SF_BLOCK_M=align_up(BLOCK_M, 128), SF_BLOCK_N=BLOCK_N,
    kNumRingTokens=..., kNumSFRingTokens=...,
    kNumStages=..., kNumBytesPerPull=...,
    kNumDispatchThreads=..., kNumNonEpilogueThreads=128, kNumEpilogueThreads=...,
    kNumSMs=..., kNumRanks=1..72,
    kActivationClamp=..., kFastMath=(False, True),
    target="sm_100f",
)
# instruction_selection: none; extent: the 23 template arguments of
#   `sm100_fp8_fp4_mega_moe_impl`, in the order the host emits them;
#   `kNumSharedExperts` is the 5th

# Derived compile-time shapes. `S` is a *width multiplier* only: there is no
# per-shared-expert index anywhere in the kernel. `S` shared experts are one
# fused FFN of intermediate width `S * I`.
kHasShared          = S > 0
L1_SHAPE_N, L1_SHAPE_K = I * 2, H
L2_SHAPE_N, L2_SHAPE_K = H, I
SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K = L1_SHAPE_N * S, L1_SHAPE_K
SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K = L2_SHAPE_N, L2_SHAPE_K * S
kNumSharedSFTokens  = div_ceil(T, 8) * 128          # worst case over all BLOCK_M
kNumExpertsPerRank  = E // kNumRanks
kNumRingBlocks      = kNumRingTokens // BLOCK_M
kNumTokensPerWarp   = 32 // K
# host_assert (scheduler): L1_SHAPE_N % (BLOCK_N*2) == 0; L2_SHAPE_N % (BLOCK_N*2) == 0
# host_assert (scheduler): L1_SHAPE_K % BLOCK_K == 0; L2_SHAPE_K % BLOCK_K == 0
# host_assert (scheduler, shared): SHARED_L1_SHAPE_N % (BLOCK_N*2) == 0
# host_assert (scheduler, shared): SHARED_L2_SHAPE_N % (BLOCK_N*2) == 0
# host_assert (scheduler, shared): SHARED_L1_SHAPE_K % BLOCK_K == 0
# host_assert (scheduler, shared): SHARED_L2_SHAPE_K % BLOCK_K == 0
# instruction_selection: none; extent: compile-time constants only

launch_config = launch(
    grid=(kNumSMs, 1, 1),                     # persistent: one block per SM
    cluster=(2, 1, 1),                        # 2-CTA UMMA
    block=(kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1, 1),
    min_blocks_per_sm=1,
    dynamic_smem_bytes=smem_size,
)
# instruction_selection: none; extent: static launch metadata;
#   `__launch_bounds__(kNumThreads, 1)`

def sm100_fp8_fp4_mega_moe(
    y,                       # bf16 [num_tokens, H], final output
    cumulative_local_expert_recv_stats,   # int32 [kNumExpertsPerRank] or null
    num_tokens,              # runtime local token count; also the shared-FFN M extent
    sym_buffer,              # by-value SymBuffer<kNumRanks>: base + 72 peer offsets + rank
    tmap_l1_acts, tmap_l1_acts_sf, tmap_l1_weights, tmap_l1_weights_sf,
    tmap_l1_output,
    tmap_l2_acts, tmap_l2_acts_sf, tmap_l2_weights, tmap_l2_weights_sf,
    tmap_shared_l1_acts, tmap_shared_l1_acts_sf,
    tmap_shared_l1_weights, tmap_shared_l1_weights_sf, tmap_shared_l1_output,
    tmap_shared_l2_acts, tmap_shared_l2_acts_sf,
    tmap_shared_l2_weights, tmap_shared_l2_weights_sf,
):
    # 18 by-value 128-byte TensorMaps, grid-constant. The nine shared ones are
    # *always* in the signature; when S == 0 the host binds the routed
    # descriptors into those slots as dummies, so the ABI never changes.

    # =======================================================================
    # Roles and descriptor prefetch
    # =======================================================================

    is_leader_cta = (cta_id_in_cluster() == 0)
    # instruction_selection: mov.u32 %cluster_ctarank + setp.ne.b32 <rank>, 0;
    #   extent: scalar -- nvcc materializes the NEGATED predicate
    sm_idx   = cta_id(axis="x")
    warp     = shuffle_index(thread_id() // 32, source_lane=0, mask=0xFFFFFFFF, clamp=0x1F)
    # instruction_selection: mov.u32 %tid.x, shr.u32, shfl.sync.idx.b32; extent: warp-uniform scalar
    lane     = thread_id() % 32
    # instruction_selection: mov.u32 %laneid; extent: scalar per thread

    if warp == 0:
        for tmap in (tmap_l1_acts, tmap_l1_acts_sf, tmap_l1_weights,
                     tmap_l1_weights_sf, tmap_l1_output,
                     tmap_l2_acts, tmap_l2_acts_sf, tmap_l2_weights,
                     tmap_l2_weights_sf,
                     tmap_shared_l1_acts, tmap_shared_l1_acts_sf,
                     tmap_shared_l1_weights, tmap_shared_l1_weights_sf,
                     tmap_shared_l1_output,
                     tmap_shared_l2_acts, tmap_shared_l2_acts_sf,
                     tmap_shared_l2_weights, tmap_shared_l2_weights_sf):
            prefetch(tmap)
            # instruction_selection: prefetch.tensormap; extent: one descriptor,
            #   18 in source order. The nine shared prefetches are unconditional
            #   upstream -- they are issued even when S == 0.

    # =======================================================================
    # Symmetric-buffer regions (GMEM). Offsets are host/device-identical.
    # =======================================================================

    # Workspace header: barrier/grid-sync counters, per-expert send/recv counts,
    # ring full/empty counters, the shared-L2 full counters, dispatch metadata.
    workspace = view(sym_buffer.base, "u8", [workspace_bytes], byte_offset=0)
    # instruction_selection: none; extent: one base pointer

    input_tokens   = view(sym_buffer.base, "e4m3", [T, H],  byte_offset=input_token_offset)
    input_sf       = view(sym_buffer.base, "u32",  [T, H // 128], byte_offset=input_sf_offset)
    input_topk_idx = view(sym_buffer.base, "i64",  [T, K],  byte_offset=input_topk_idx_offset)
    input_topk_w   = view(sym_buffer.base, "f32",  [T, K],  byte_offset=input_topk_weights_offset)
    # instruction_selection: none; extent: four per-rank input regions

    # Shared-expert regions exist only when S > 0. `shared_l1_tokens` *aliases*
    # `input_tokens`: SharedLinear1 reads x in place, so there is no dispatch,
    # no pull and no ring for it.
    shared_l1_tokens = input_tokens
    shared_l1_sf     = view(sym_buffer.base, "u32", [H // 128, kNumSharedSFTokens],
                            byte_offset=shared_l1_sf_offset)
    # instruction_selection: none; extent: HOST-written region -- the kernel only
    #   reads it. The caller must lay it out with the same UTCCP transpose the
    #   L1 epilogue applies to the routed SFs, which depends on the runtime
    #   BLOCK_M (see "Host-side contract").
    shared_l2_tokens = view(sym_buffer.base, "e4m3", [T, S * I],
                            byte_offset=shared_l2_token_offset)
    shared_l2_sf     = view(sym_buffer.base, "u32", [S * I // 128, kNumSharedSFTokens],
                            byte_offset=shared_l2_sf_offset)
    # instruction_selection: none; extent: full-size (non-ring) shared
    #   intermediate buffers, written by the SharedL1 epilogue

    # Routed ring buffers. When S > 0 they start *after* the shared regions.
    l1_tokens    = view(sym_buffer.base, "e4m3", [kNumRingTokens, H], byte_offset=l1_token_offset)
    l1_sf        = view(sym_buffer.base, "u32", [H // 128, kNumSFRingTokens],
                        byte_offset=l1_sf_offset)
    l1_topk_w    = view(sym_buffer.base, "f32", [kNumRingTokens], byte_offset=l1_topk_weights_offset)
    l2_tokens    = view(sym_buffer.base, "e4m3", [kNumRingTokens, I], byte_offset=l2_token_offset)
    l2_sf        = view(sym_buffer.base, "u32", [I // 128, kNumSFRingTokens],
                        byte_offset=l2_sf_offset)
    # instruction_selection: none; extent: ALL FOUR scale-factor planes share one
    #   physical layout -- SF column slow, token FAST (`sf[col * num_tokens + tok]`,
    #   impl:566-568 and impl:1149). The host allocates the shared pair as
    #   `empty_strided((tokens, sf_cols), (1, tokens))`, which is the same bytes
    #   written with the axes swapped; the shape order used here is uniform.

    def transform_sf_token_idx(token_idx_in_expert):
        idx = token_idx_in_expert % BLOCK_M
        return (token_idx_in_expert // BLOCK_M * SF_BLOCK_M
                + (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3))
    # instruction_selection: and.b32 + shl.b32 + add.s32; extent: scalar. This is the
    #   UTCCP 4x32 in-group transpose and it is THE layout contract for all four SF
    #   planes: the routed L1 write (dispatch), the routed and shared L2 writes (L1
    #   epilogue), and the HOST-written shared L1 plane. Both in-kernel call sites
    #   pass an argument already < BLOCK_M, so the leading term is zero there.
    combine      = view(sym_buffer.base, "bf16", [K + (1 if kHasShared else 0), T, H],
                        byte_offset=combine_token_offset)
    # instruction_selection: none; extent: the combine buffer gains ONE extra
    #   rank-slot when S > 0; slot index `K` carries the shared-expert output

    # =======================================================================
    # Exact dynamic shared-memory layout and lifetimes
    # =======================================================================

    L1_OUT_BLOCK_N   = BLOCK_N // 2          # gate/up halves collapse in SwiGLU
    LOAD_BLOCK_M     = BLOCK_M // 2          # multicast on A
    LOAD_BLOCK_N     = BLOCK_N
    UMMA_M, UMMA_N   = 256, BLOCK_M          # always swap A/B
    UMMA_BLOCK_K, UMMA_K = 128, 32
    kNumEpilogueStages, kNumTMAStoreStages, kNumScheduleStages = 2, 2, 2
    kNumDispatchWarps       = kNumDispatchThreads // 32
    kNumMMANonEpilogueWarps = kNumNonEpilogueThreads // 32     # == 4
    kNumEpilogueWarps      = kNumEpilogueThreads // 32
    kNumEpilogueWarpgroups = kNumEpilogueWarps // 4
    WG_BLOCK_M       = BLOCK_M // kNumEpilogueWarpgroups
    ATOM_M           = 8
    kNumAtomsPerStore = STORE_BLOCK_M // ATOM_M
    kNumBankGroupBytes = 16
    kSwizzleAMode = kSwizzleBMode = kSwizzleCDMode = 128
    kNumRanksPerLane   = div_ceil(kNumRanks, 32)
    kNumExpertsPerLane = div_ceil(kNumExpertsPerRank, 32)
    kNumL1Clusters     = L1_SHAPE_N // BLOCK_N // 2
    kNumL2Clusters     = L2_SHAPE_N // BLOCK_N // 2
    # Register budgets: more experts per rank costs the scheduler more registers.
    kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64
    kNumDispatchRegisters     = 48  if kUseMoreEpilogueRegisters else 96
    kNumNonEpilogueRegisters  = 40  if kUseMoreEpilogueRegisters else 88
    kNumEpilogueRegisters     = 208 if kUseMoreEpilogueRegisters else 160
    # host_assert: the three budgets times their thread counts must be <= 64512
    AMAX_BUF         = STORE_BLOCK_M // 2
    # host_assert: BLOCK_M % kNumEpilogueWarpgroups == 0;
    #   WG_BLOCK_M % STORE_BLOCK_M == 0; STORE_BLOCK_M % ATOM_M == 0; BLOCK_N == 128
    # instruction_selection: none; extent: compile-time constants

    smem_raw = tile("shared", "u8", [smem_size], byte_offset=0, requested_alignment=1024)
    # instruction_selection: none; extent: one dynamic-SMEM allocation

    expert_token_count = view(smem_raw, "u32", [E], byte_offset=0)
    send_buf   = view(smem_raw, "u8", [kNumDispatchWarps, kNumBytesPerPull],
                      byte_offset=smem_send_buffer_offset)
    # `smem_d` is a union: the L1 path sees FP8 store stages, the L2 path sees
    # one BF16 stage. Both start at the same byte.
    smem_d_l1  = view(smem_raw, "e4m3", [kNumEpilogueWarpgroups, kNumTMAStoreStages,
                                         STORE_BLOCK_M * L1_OUT_BLOCK_N],
                      byte_offset=smem_cd_offset)
    smem_d_l2  = view(smem_raw, "bf16", [kNumEpilogueWarpgroups, STORE_BLOCK_M * BLOCK_N],
                      byte_offset=smem_cd_offset)
    smem_a     = view(smem_raw, "e4m3", [kNumStages, LOAD_BLOCK_M, BLOCK_K],
                      byte_offset=smem_a_offset, layout="xor_swizzle_128B")
    # The B slab is typed by the *task*: routed tasks view it as unpacked-SMEM
    # FP4, shared tasks as FP8 e4m3. Both are one byte per element in SMEM, so
    # the slab, its stage stride and its UMMA descriptor low words coincide.
    smem_b     = view(smem_raw, "e2m1_unpacksmem | e4m3", [kNumStages, LOAD_BLOCK_N, BLOCK_K],
                      byte_offset=smem_b_offset, layout="xor_swizzle_128B")
    smem_sfa   = view(smem_raw, "u32", [kNumStages, SF_BLOCK_M * (BLOCK_K // 128)],
                      byte_offset=smem_sfa_offset)
    smem_sfb   = view(smem_raw, "u32", [kNumStages, SF_BLOCK_N * (BLOCK_K // 128)],
                      byte_offset=smem_sfb_offset)
    amax_red   = view(smem_raw, "f32x2", [kNumEpilogueWarps, AMAX_BUF],
                      byte_offset=smem_amax_reduction_offset)
    task_infos = view(smem_raw, "u32x8", [kNumScheduleStages], byte_offset=smem_task_info_offset)
    # instruction_selection: none; extent: `TaskInfo` is eight u32 registers,
    #   16-byte aligned; `sizeof(TaskInfo<true>) == sizeof(TaskInfo<false>)`, so
    #   enabling shared experts does not move any SMEM offset

    dispatch_bar   = view(smem_raw, "mbarrier.b64", [kNumDispatchWarps],
                          byte_offset=dispatch_barrier_base)
    full           = view(smem_raw, "mbarrier.b64", [kNumStages],  byte_offset=full_barrier_base)
    empty          = view(smem_raw, "mbarrier.b64", [kNumStages],  byte_offset=empty_barrier_base)
    tmem_full      = view(smem_raw, "mbarrier.b64", [kNumEpilogueStages], byte_offset=tmem_full_base)
    tmem_empty     = view(smem_raw, "mbarrier.b64", [kNumEpilogueStages], byte_offset=tmem_empty_base)
    combine_bar    = view(smem_raw, "mbarrier.b64", [kNumEpilogueWarps * 2], byte_offset=combine_base)
    task_full      = view(smem_raw, "mbarrier.b64", [kNumScheduleStages], byte_offset=task_full_base)
    task_empty     = view(smem_raw, "mbarrier.b64", [kNumScheduleStages], byte_offset=task_empty_base)
    tmem_ptr_slot  = view(smem_raw, "u32", [1], byte_offset=smem_tmem_ptr_offset)
    # The combine phase reuses every byte before `dispatch_bar` as its staging
    # ring; everything from `dispatch_bar` onward must survive it.
    kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers)
    # instruction_selection: none; extent: compile-time byte count -- the combine
    #   staging ring may reuse every byte below this offset

    kNumAccumTmemCols = UMMA_N * kNumEpilogueStages
    kTmemStartColOfSFA = kNumAccumTmemCols
    kTmemStartColOfSFB = kNumAccumTmemCols + SF_BLOCK_M // 32
    kNumTmemCols = round_up_to(kNumAccumTmemCols + SF_BLOCK_M // 32 + SF_BLOCK_N // 32,
                               {32, 64, 128, 256, 512})
    # instruction_selection: none; extent: TMEM column map, unchanged by S

    # =======================================================================
    # Cluster rendezvous, barrier init, TMEM allocation
    # =======================================================================

    cluster_sync(arrive="relaxed")
    # instruction_selection: barrier.cluster.arrive.relaxed.aligned then
    #   barrier.cluster.wait.aligned; extent: all threads of both CTAs

    if warp == 0:
        if elect_predicate(active_mask=0xFFFFFFFF):
            fill(expert_token_count, 0, bytes=align_up(E * 4, 1024))
            # instruction_selection: st.bulk.weak.shared::cta; extent: one bulk
            #   shared-memory zero fill, 8-byte aligned
    elif warp == 1:
        for i in strided_range(lane, kNumDispatchWarps, step=32):
            init(dispatch_bar[i], arrival_count=1)
            # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot
        fence(scope="mbarrier_init", order="release", visibility="cluster")
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: warp 1
    elif warp == 2:
        if elect_predicate(active_mask=0xFFFFFFFF):
            for s in static_range(kNumStages):
                init(full[s],  arrival_count=2 * 2)    # 2 CTAs x (A-warp, B-warp)
                # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot
                init(empty[s], arrival_count=1)
                # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot
            for e in static_range(kNumEpilogueStages):
                init(tmem_full[e],  arrival_count=1)
                # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot
                init(tmem_empty[e], arrival_count=2 * kNumEpilogueThreads)
                # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot;
                #   every epilogue thread of both CTAs arrives on the leader's copy
            for i in static_range(kNumEpilogueWarps * 2):
                init(combine_bar[i], arrival_count=1)
                # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot
            for s in static_range(kNumScheduleStages):
                init(task_full[s],  arrival_count=1)
                # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot
                init(task_empty[s], arrival_count=2 * kNumEpilogueThreads)
                # instruction_selection: mbarrier.init.shared::cta.b64; extent: one slot
        fence(scope="mbarrier_init", order="release", visibility="cluster")
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: warp 2
    elif warp == 3:
        tmem_alloc(tmem_ptr_slot, columns=kNumTmemCols, cta_group=2)
        # instruction_selection: tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32;
        #   extent: one allocation, base written to SMEM

    cluster_sync(arrive="relaxed")
    # instruction_selection: barrier.cluster.arrive.relaxed.aligned +
    #   barrier.cluster.wait.aligned; extent: both CTAs

    # =======================================================================
    # Shared pipeline cursor (per-role copies, kept in lockstep by the stream)
    # =======================================================================

    stage_idx, phase = 0, 0
    def advance_pipeline(k_block_idx):
        k_block_idx += 1
        stage_idx = 0 if stage_idx == kNumStages - 1 else stage_idx + 1
        phase ^= (stage_idx == 0)
        # instruction_selection: add.s32, setp.eq.b32 + selp.b32, xor.b32; extent: scalars
```

### The task stream: `TaskInfo` and its three phases

`TaskInfo` is eight `u32` registers published by one producer (the scheduler
warp on the leader CTA) into `task_infos[stage]` of **both** CTAs, and consumed
by the four GEMM roles. `block_phase` is what makes a task shared:

```python
    BlockPhase = {None: 0, Linear1: 1, Linear2: 2, SharedLinear1: 3, SharedLinear2: 4}

    task_info = reg_tile("u32", [8])   # block_phase, local_expert_idx, m_block_idx,
                                       # n_cluster_idx, pool_block_idx, valid_m,
                                       # shape_n, shape_k
    def is_shared(task_info):
        return task_info.block_phase > BlockPhase.Linear2 if kHasShared else False
        # instruction_selection: BOTH polarities are emitted, by consumer kind --
        #   `setp.lt.u32 <block_phase>, 3` (the COMPLEMENT, with the consuming `selp`
        #   arms swapped) where a select consumes it, and the direct
        #   `setp.gt.u32 <block_phase>, 2` at the two `if (not is_shared())` branch
        #   guards (the L1 ring handshake and the L2-empty release). This is the
        #   central shared-expert predicate. With S == 0
        #   this folds to a compile-time false and every guard below collapses.
    def get_umma_aligned_valid_m(task_info):
        return align_up(task_info.valid_m, 16)
        # instruction_selection: add.s32 + and.b32; extent: scalar

    def get_next_task(task_info):          # consumer side, all four GEMM roles
        wait(task_full[sched_stage_idx], parity=sched_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin; extent: one slot
        move(task_info, task_infos[sched_stage_idx])
        # instruction_selection: role-dependent ld.shared.{v4,v2,b32,b16} covering
        #   only the fields that role consumes (the A-load warp reads v4+b32, the
        #   B-load warp v4+v2, the UMMA warp b32 x2 + b16, the epilogue b32 x4)
        sched_stage_idx ^= 1; sched_phase ^= (sched_stage_idx == 0)
        return task_info.block_phase != BlockPhase.None

    def release_task_info():                # consumer side, epilogue only
        arrive(task_empty[sched_stage_idx ^ 1], dst_cta=0)
        # instruction_selection: mapa.shared::cluster.u32 +
        #   mbarrier.arrive.shared::cluster.b64; extent: one arrival
        #   per epilogue thread; the slot was initialized with 2 * kNumEpilogueThreads

    def publish_task(task_info):            # producer side, scheduler warp
        if lane < 2:
            expect_bytes(task_full[sched_stage_idx], bytes=32, dst_cta=lane)
            # instruction_selection: mbarrier.arrive.expect_tx.shared::cluster.b64;
            #   extent: one transaction per destination CTA
            st_async(task_infos[sched_stage_idx], task_info,
                     dst_cta=lane, completion=task_full[sched_stage_idx])
            # instruction_selection: st.async.shared::cluster.mbarrier::complete_tx::bytes
            #   .u32.v4 x2; extent: 32 bytes to each CTA of the cluster
        warp_sync()
        # instruction_selection: bar.warp.sync 0xffffffff; extent: the scheduler warp
        sched_stage_idx ^= 1; sched_phase ^= (sched_stage_idx == 0)
```

### Role 1: dispatch warps (`warp < kNumDispatchWarps`)

Shared experts never enter dispatch: they read `x` in place, so no routing, no
NVLink pull and no per-rank accounting is done for them. The only shared work
here is zeroing their counters for the *next* launch.

```python
    if warp < kNumDispatchWarps:
        reg_dealloc(kNumDispatchRegisters)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: the dispatch warpgroup

        kNumActivateLanes = kNumTokensPerWarp * K
        def read_topk_idx(process):
            for i in strided_range(start=(sm_idx * kNumDispatchWarps + warp) * kNumTokensPerWarp,
                                   bound=num_tokens,
                                   step=kNumSMs * kNumDispatchWarps * kNumTokensPerWarp):
                expert_idx = -1
                if i + lane // K < num_tokens and lane < kNumActivateLanes:
                    copy_g2r(input_topk_idx[i * K + lane], expert_idx)
                    # instruction_selection: ld.global.nc.s64; extent: one i64 per lane
                    if expert_idx >= 0:
                        process(i * K + lane, expert_idx)
                warp_sync()
                # instruction_selection: bar.warp.sync 0xffffffff; extent: one dispatch warp

        # (a) count this rank's tokens per expert
        read_topk_idx(lambda token_topk_idx, expert_idx:
                      atomic_add(expert_token_count[expert_idx], 1))
        # instruction_selection: atom.shared.cta.add.u32; extent: one per active lane
        barrier(barrier_id=0, arrival_count=kNumDispatchThreads)
        # instruction_selection: bar.sync 0, kNumDispatchThreads; extent: dispatch threads

        # (b) claim a contiguous slot range on the owning rank
        for i in strided_range(thread_id(), E, step=kNumDispatchThreads):
            send_value = (1 << 32) | expert_token_count[i]
            atomic_add(expert_token_count[i], workspace.expert_send_count[i], send_value)
            # instruction_selection: atom.global.add.u64; extent: one per expert;
            #   the high half counts arrivals, the low half is the slot base
        barrier(barrier_id=0, arrival_count=kNumDispatchThreads)
        # instruction_selection: bar.sync 0, kNumDispatchThreads; extent: dispatch threads

        # (c) publish each (token, topk) source index into the owner's workspace
        def write_src(token_topk_idx, expert_idx):
            dst_rank = expert_idx // kNumExpertsPerRank
            slot = atomic_add(expert_token_count[expert_idx], 1)
            # instruction_selection: atom.shared.cta.add.u32; extent: one per active lane
            copy_r2g(token_topk_idx,
                     peer(workspace.src_token_topk_idx[expert_idx % kNumExpertsPerRank,
                                                       sym_buffer.rank_idx, slot], dst_rank))
            # instruction_selection: st.global.b32; extent: one word to a peer rank
        read_topk_idx(write_src)

        grid_sync(index=0, count=kNumSMs)
        # instruction_selection: atom.release.gpu.global.add.u32 + ld.acquire.gpu.global.b32 spin;
        #   extent: one grid rendezvous
        if sm_idx == 0:
            for i in strided_range(thread_id(), E, step=kNumDispatchThreads):
                copy_g2r(workspace.expert_send_count[i], status)
                # instruction_selection: ld.global.b64; extent: one per expert
                copy_r2g(status & 0xffffffff,
                         peer(workspace.expert_recv_count[sym_buffer.rank_idx,
                                                          i % kNumExpertsPerRank],
                              i // kNumExpertsPerRank))
                # instruction_selection: st.global.b64; extent: one 64-bit slot per
                #   expert (the mask only zeroes the high half)
                red_add(peer(workspace.expert_recv_count_sum[i % kNumExpertsPerRank],
                             i // kNumExpertsPerRank), status)
                # instruction_selection: atom.sys.global.add.u64; extent: one per expert
        barrier(barrier_id=0, arrival_count=kNumDispatchThreads)
        # instruction_selection: bar.sync 0, kNumDispatchThreads; extent: dispatch threads

        nvlink_barrier(tag=1, index=0, sync_prologue=False, sync_epilogue=True)
        # instruction_selection: one cross-rank rendezvous before pulling, with only
        #   the TRAILING grid sync -- the caller already grid-synced above.
        #   Each ENABLED grid sync is `bar.sync 0, kNumDispatchThreads` +
        #   `atom.release.gpu.global.add.u32` + `ld.acquire.gpu.global.b32` spin +
        #   `bar.sync 0, kNumDispatchThreads`, and the mid-signal `sync_scope()` contributes a
        #   THIRD `bar.sync` plus a `red.gpu.global.add.u32` counter bump -- so one
        #   enabled grid sync emits three `bar.sync`, not two. The cross-rank signal
        #   itself is `red.release.sys.global.add.s32` to every peer, spun on with
        #   `ld.acquire.sys.global.s32`.
        barrier(barrier_id=1, arrival_count=kNumDispatchThreads + kNumEpilogueThreads)
        # instruction_selection: barrier.sync 1, N (unaligned); extent: dispatch + epilogue,
        #   so the combine NVLink barrier can never overlap the pull barrier

        # (d) pull loop: walk this warp's stride of the local pool
        pull_phase = 0
        # This warp's PRIVATE mbarrier phase for `dispatch_bar[warp]`, flipped after
        # every wait below. (Cursor bookkeeping: the zeroing is sunk into an
        # unrelated block entry, so there is nothing to attribute here.)
        fetch_expert_recv_count()          # see "Scheduler" below; blocks on dispatch
        for token_idx in strided_range(start=sm_idx * kNumDispatchWarps + warp,
                                       step=kNumSMs * kNumDispatchWarps):
            # advance the expert cursor until token_idx is inside [start, end)
            while token_idx >= expert_end_idx:
                current_expert_idx += 1
                if current_expert_idx >= kNumExpertsPerRank: break
                expert_pool_block_offset += div_ceil(expert_end_idx - expert_start_idx, BLOCK_M)
                expert_start_idx = expert_end_idx
                expert_end_idx  += get_num_tokens(current_expert_idx)
                # instruction_selection: add/sub/shr + shfl.sync.idx.b32; extent: scalars
            if current_expert_idx >= kNumExpertsPerRank: break

            expert_changed = (old_expert_idx != current_expert_idx)
            # instruction_selection: setp.eq.b32 + or.pred + @p bra; extent: one
            #   predicate, emitted with INVERTED polarity and merged into the guard
            if expert_changed:
                for i in static_range(kNumRanksPerLane):
                    copy_g2r(workspace.expert_recv_count[i * 32 + lane, current_expert_idx],
                             stored_rank_count[i])
                    # instruction_selection: ld.global.b32; extent: the low 32 bits of one
                    #   uint64 counter per lane-held rank (the cast narrows it)

            # round-robin rank selection by iterative min-peeling
            while True:
                warp_reduce_add(num_active_ranks, count_of(remaining > 0))
                # instruction_selection: redux.sync.add.u32; extent: one warp reduction
                warp_reduce_min(length, min_of(remaining > 0))
                # instruction_selection: redux.sync.min.u32; extent: one warp reduction
                if slot_idx < length * num_active_ranks:
                    ballot(mask, remaining[i] > 0)
                    # instruction_selection: vote.sync.ballot.b32; extent: one per lane-group
                    popc(num_active_lanes, mask)
                    # instruction_selection: popc.b32; extent: scalar
                    fns(current_rank_in_expert_idx, mask, 0, slot_idx_in_round - seen + 1)
                    # instruction_selection: fns.b32; extent: scalar -- find the n-th set bit
                    token_idx_in_rank = offset + slot_idx // num_active_ranks
                    break
                slot_idx -= length * num_active_ranks; offset += length
                sub(remaining, remaining, min(remaining, length))

            copy_g2r(workspace.src_token_topk_idx[current_expert_idx,
                                                  current_rank_in_expert_idx,
                                                  token_idx_in_rank], src_token_topk_idx)
            # instruction_selection: ld.global.b32; extent: one word
            src_token_idx, src_topk_idx = src_token_topk_idx // K, src_token_topk_idx % K

            pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert
            pool_block_idx = pool_token_idx // BLOCK_M
            # Ring reuse: wait until the previous wave's consumers released this slot
            if (pool_block_idx // kNumRingBlocks) * (L1_SHAPE_N // BLOCK_N) > 0:
                while load_acquire(workspace.l1_empty_count[pool_block_idx % kNumRingBlocks]) \
                        < l1_empty_count_target:
                    pass
                # instruction_selection: ld.acquire.gpu.global.b32 spin; extent: one ring slot

            # (e) chunked pull of one token, remote -> shared -> local ring
            if elect_predicate(active_mask=0xFFFFFFFF):
                for i in static_range(H // kNumBytesPerPull):
                    copy_g2s(peer(input_tokens[src_token_idx].byte(i * kNumBytesPerPull),
                                  current_rank_in_expert_idx),
                             send_buf[warp], completion=dispatch_bar[warp])
                    # instruction_selection: cp.async.bulk.shared::cluster.global
                    #   .mbarrier::complete_tx::bytes.L2::cache_hint (EVICT_FIRST);
                    #   extent: one kNumBytesPerPull chunk from a peer rank
                    expect_bytes(dispatch_bar[warp], bytes=kNumBytesPerPull)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64; extent: one
                    if i != H // kNumBytesPerPull - 1:
                        wait(dispatch_bar[warp], parity=pull_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin
                        pull_phase ^= 1
                        # instruction_selection: xor.b32; extent: scalar
                        copy_s2g(send_buf[warp],
                                 l1_tokens[pool_token_idx % kNumRingTokens].byte(i * kNumBytesPerPull))
                        # instruction_selection: cp.async.bulk.global.shared::cta
                        #   .bulk_group.L2::cache_hint; extent: one chunk
                        store_arrive(); store_wait(pending=0)
                        # instruction_selection: cp.async.bulk.commit_group +
                        #   cp.async.bulk.wait_group 0; extent: one group
            warp_sync()

            # (f) scale factors follow the same route, transposed into the ring
            for i in static_range(div_ceil(H // 128, 32)):
                j = i * 32 + lane
                if j < H // 128:
                    copy_g2r(peer(input_sf[src_token_idx, j], current_rank_in_expert_idx), sf_word)
                    # instruction_selection: ld.global.b32; extent: one word per lane
                    copy_r2g(sf_word, l1_sf[j, ring_block_idx * SF_BLOCK_M
                                              + transform_sf_token_idx(token_idx_in_block)])
                    # instruction_selection: st.global.b32; extent: one word per lane;
                    #   `transform_sf_token_idx` is the UTCCP 4x32 in-group transpose
            warp_sync()

            if elect_predicate(active_mask=0xFFFFFFFF):
                copy_g2r(peer(input_topk_w[src_token_topk_idx], current_rank_in_expert_idx), weight)
                # instruction_selection: ld.global.b32; extent: one float
                copy_r2g(weight, l1_topk_w[pool_token_idx % kNumRingTokens])
                # instruction_selection: st.global.b32; extent: one float
                copy_r2g((current_rank_in_expert_idx, src_token_idx, src_topk_idx),
                         workspace.token_src_metadata[pool_token_idx])
                # instruction_selection: st.global.b32 x3; extent: 12 bytes;
                #   the combine write-back reads this back per row
                # Complete the LAST chunk's store here, after the metadata writes:
                # the loop above deliberately skipped it so the final remote load
                # overlaps the SF and weight traffic.
                wait(dispatch_bar[warp], parity=pull_phase)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                #   extent: the final chunk's arrival
                pull_phase ^= 1
                # instruction_selection: xor.b32; extent: scalar
                copy_s2g(send_buf[warp],
                         l1_tokens[pool_token_idx % kNumRingTokens]
                             .byte((H // kNumBytesPerPull - 1) * kNumBytesPerPull))
                # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group
                #   .L2::cache_hint; extent: one kNumBytesPerPull chunk
                store_arrive()
                # instruction_selection: cp.async.bulk.commit_group; extent: one group
                store_wait(pending=0)
                # instruction_selection: cp.async.bulk.wait_group 0; extent: one group
                red_add_rel(workspace.l1_full_count[pool_block_idx % kNumRingBlocks],
                            BLOCK_M - token_idx_in_block if is_last_token else 1)
                # instruction_selection: red.release.gpu.global.add.u32; extent: one counter;
                #   the tail token bumps the counter to the full BLOCK_M so a
                #   partially-filled block still satisfies the consumer's equality wait
            warp_sync()

        # (g) clean the workspace for the next launch, overlapped with combine
        barrier(barrier_id=1, arrival_count=kNumDispatchThreads + kNumEpilogueThreads)
        # instruction_selection: barrier.sync 1, N (unaligned); extent: dispatch + epilogue
        if sm_idx == 0:
            for i in strided_range(thread_id(), E, step=kNumDispatchThreads):
                fill(workspace.expert_send_count[i], 0)
                # instruction_selection: st.global.b64; extent: one per expert
            if warp == 0 and elect_predicate(active_mask=0xFFFFFFFF):
                fill(workspace.l1_task_count, 0)
                fill(workspace.l2_task_count, 0)
                fill(workspace.shared_l1_task_count, 0)
                fill(workspace.shared_l2_task_count, 0)
                # instruction_selection: st.global.b32 x4; extent: the four schedule
                #   counters. The two shared ones are cleared unconditionally,
                #   exactly as upstream does, even when S == 0.
            warp_sync()
            for i in strided_range(thread_id(), workspace.num_shared_l2_pool_blocks,
                                   step=kNumDispatchThreads):
                fill(workspace.shared_l2_full_count[i], 0)
                # instruction_selection: st.global.b32; extent: one per shared pool block;
                #   the array is sized `div_ceil(T, 8)` (worst case over all BLOCK_M)
                #   and is allocated regardless of S
            warp_sync()
        else:
            for i in strided_range(sm_idx - 1, kNumExpertsPerRank, step=kNumSMs - 1):
                copy_g2r(workspace.expert_recv_count_sum[i], num_recv_tokens)
                # instruction_selection: ld.global.b32; extent: the low 32 bits (cast)
                expert_pool_block_offset = get_pool_block_offset(i)
                barrier(barrier_id=0, arrival_count=kNumDispatchThreads)
                if warp == 0:
                    fill(workspace.expert_recv_count_sum[i], 0)
                    # instruction_selection: st.global.b64; extent: one counter
                elif warp == 1 and elect_predicate(active_mask=0xFFFFFFFF) \
                        and cumulative_local_expert_recv_stats is not None:
                    red_add(cumulative_local_expert_recv_stats[i], num_recv_tokens)
                    # instruction_selection: red.gpu.global.add.s32; extent: one counter
                    warp_sync()
                    # instruction_selection: bar.warp.sync 0xffffffff; extent: this warp
                for j in strided_range(thread_id(), kNumRanks, step=kNumDispatchThreads):
                    fill(workspace.expert_recv_count[j, i], 0)
                    # instruction_selection: st.global.b64; extent: one per rank
                warp_sync()
                # instruction_selection: bar.warp.sync 0xffffffff; extent: this dispatch warp
                for j in strided_range(thread_id(), div_ceil(num_recv_tokens, BLOCK_M),
                                       step=kNumDispatchThreads):
                    fill(workspace.l1_full_count[(expert_pool_block_offset + j) % kNumRingBlocks], 0)
                    fill(workspace.l1_empty_count[...], 0)
                    fill(workspace.l2_full_count[...], 0)
                    fill(workspace.l2_empty_count[...], 0)
                    # instruction_selection: st.global.b32 x4; extent: one ring slot
                warp_sync()
                # instruction_selection: bar.warp.sync 0xffffffff; extent: this dispatch warp
        nvlink_barrier(tag=3, index=0, sync_prologue=True, sync_epilogue=False)
        # instruction_selection: one cross-rank rendezvous after the clean, with only
        #   the LEADING grid sync -- the kernel ends here, so no trailing sync.
        #   Each ENABLED grid sync is `bar.sync 0, kNumDispatchThreads` +
        #   `atom.release.gpu.global.add.u32` + `ld.acquire.gpu.global.b32` spin +
        #   `bar.sync 0, kNumDispatchThreads`, and the mid-signal `sync_scope()` contributes a
        #   THIRD `bar.sync` plus a `red.gpu.global.add.u32` counter bump -- so one
        #   enabled grid sync emits three `bar.sync`, not two. The cross-rank signal
        #   itself is `red.release.sys.global.add.s32` to every peer, spun on with
        #   `ld.acquire.sys.global.s32`.
```

### Role 2: A/SFA TMA load warp (`warp == kNumDispatchWarps`)

The four-way descriptor select and the phase-specific producer wait are the
whole shared-expert delta here.

```python
    elif warp == kNumDispatchWarps:
        reg_dealloc(kNumNonEpilogueRegisters)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: this warp

        while get_next_task(task_info):
            tmap_a   = select4(task_info.block_phase,
                               tmap_l1_acts, tmap_l2_acts,
                               tmap_shared_l1_acts, tmap_shared_l2_acts)
            tmap_sfa = select4(task_info.block_phase,
                               tmap_l1_acts_sf, tmap_l2_acts_sf,
                               tmap_shared_l1_acts_sf, tmap_shared_l2_acts_sf)
            # instruction_selection: a `block_phase` branch chain for the Linear1 and
            #   Linear2 arms (setp.eq.b32 + @p bra, with mov.b64 writing those two),
            #   terminating in one `setp.eq.b32 <phase>, 3` plus exactly TWO
            #   `selp.b64` covering the SharedLinear1/SharedLinear2 pair; extent: two
            #   selects per role. With S == 0 the shared arms are unreachable and the
            #   whole tail collapses to the routed pair.

            pool_block_idx = task_info.pool_block_idx
            ring_block_idx = pool_block_idx % kNumRingBlocks
            block_idx = select(is_shared(task_info), pool_block_idx, ring_block_idx)
            # instruction_selection: mul.hi.u32 + shr.u32 + mul.lo.s32 + sub.s32
            #   (magic-number `% kNumRingBlocks`; degenerates to and.b32 when the ring
            #   size is a power of two -- there is no `rem` opcode in the module);
            #   extent: scalar. Here the `is_shared` select FOLDS into the four-way
            #   phase branch above, so no selp is emitted -- unlike the epilogue copy,
            #   which does emit one. Shared tasks
            #   index the *absolute* pool block: their activation buffers are
            #   full-size, not a ring, so there is no wrap.

            # Producer wait, one arm per phase
            if task_info.block_phase == BlockPhase.Linear1:
                while load_acquire(workspace.l1_full_count[block_idx]) \
                        != BLOCK_M * (pool_block_idx // kNumRingBlocks + 1):
                    pass
                # instruction_selection: ld.acquire.gpu.global.b32 spin; extent: one counter;
                #   the wave multiplier makes the equality test survive ring reuse
            elif task_info.block_phase == BlockPhase.Linear2:
                while load_acquire(workspace.l2_full_count[block_idx]) \
                        != (L2_SHAPE_K // BLOCK_N) * 2 * (pool_block_idx // kNumRingBlocks + 1):
                    pass
                # instruction_selection: ld.acquire.gpu.global.b32 spin; extent: one counter
            elif task_info.block_phase == BlockPhase.SharedLinear2:
                while load_acquire(workspace.shared_l2_full_count[block_idx]) \
                        != (SHARED_L2_SHAPE_K // BLOCK_N) * 2:
                    pass
                # instruction_selection: ld.acquire.gpu.global.b32 spin; extent: one counter;
                #   NO wave multiplier -- the shared intermediate buffer is never reused,
                #   so the counter is written exactly once per block
            # SharedLinear1 has no wait at all: its input is `x`, written by the
            # host before the launch. That is what lets the scheduler issue these
            # tasks before dispatch has finished.

            for k_block_idx in runtime_range(div_ceil(task_info.shape_k, BLOCK_K),
                                             advance=advance_pipeline):
                wait(empty[stage_idx], parity=phase ^ 1)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                #   extent: one stage

                m_idx     = block_idx * BLOCK_M
                sfa_m_idx = block_idx * SF_BLOCK_M
                k_idx     = k_block_idx * BLOCK_K
                sfa_k_idx = k_block_idx * (BLOCK_K // 128)
                if not is_leader_cta:
                    m_idx += get_umma_aligned_valid_m(task_info) // 2
                    # instruction_selection: add.s32 + shr.u32; extent: scalar;
                    #   A is M-multicast-split across the two CTAs

                if elect_predicate(active_mask=0xFFFFFFFF):
                    copy_g2s(tmap_a.coord(k_idx, m_idx), smem_a[stage_idx],
                             completion=full[stage_idx], cta_group=2)
                    # instruction_selection: cp.async.bulk.tensor.2d.cta_group::2
                    #   .shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint;
                    #   issued independently by BOTH CTAs with no CTA mask. The SM90
                    #   `.multicast::cluster` form is the other branch of `tma::copy`
                    #   (leader-CTA-only, takes a mask) and is NOT taken here;
                    #   extent: BLOCK_K / 128 boxes of <128, LOAD_BLOCK_M>, swizzle
                    #   128B, e4m3 (one box at BLOCK_K == 128, two at 256)
                    copy_g2s(tmap_sfa.coord(sfa_m_idx, sfa_k_idx), smem_sfa[stage_idx],
                             completion=full[stage_idx], cta_group=2)
                    # instruction_selection: same TMA family, unswizzled; extent: one
                    #   <SF_BLOCK_M, 1> box -- a single instruction, since an
                    #   unswizzled descriptor makes the whole inner block one atom
                    if is_leader_cta:
                        expect_bytes(full[stage_idx],
                                     bytes=sizeof(smem_a[0]) * 2 + sizeof(smem_sfa[0]) * 2)
                        # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64;
                        #   extent: one transaction group covering both CTAs' boxes
                    else:
                        arrive(full[stage_idx], dst_cta=0)
                        # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64;
                        #   extent: one arrival on the leader's copy
                warp_sync()
                # instruction_selection: bar.warp.sync 0xffffffff; extent: this warp
```

### Role 3: B/SFB TMA load warp (`warp == kNumDispatchWarps + 1`)

This is where the routed/shared operand dtype split lives.

```python
    elif warp == kNumDispatchWarps + 1:
        reg_dealloc(kNumNonEpilogueRegisters)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: this warp

        while get_next_task(task_info):
            tmap_b   = select4(task_info.block_phase,
                               tmap_l1_weights, tmap_l2_weights,
                               tmap_shared_l1_weights, tmap_shared_l2_weights)
            tmap_sfb = select4(task_info.block_phase,
                               tmap_l1_weights_sf, tmap_l2_weights_sf,
                               tmap_shared_l1_weights_sf, tmap_shared_l2_weights_sf)
            # instruction_selection: same shape as the A/SFA select above -- branch
            #   chain for Linear1/Linear2, then setp.eq.b32 <phase>, 3 + two selp.b64

            n_block_idx = task_info.n_cluster_idx * 2 + (0 if is_leader_cta else 1)
            # (the two CTAs of a cluster take adjacent N blocks of the same M block)
            shape_sfb_k = div_ceil(task_info.shape_k, 32 * 4)

            for k_block_idx in runtime_range(div_ceil(task_info.shape_k, BLOCK_K),
                                             advance=advance_pipeline):
                wait(empty[stage_idx], parity=phase ^ 1)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                #   extent: one stage

                # Shared weights are ONE 2-D matrix, so they carry no expert-group
                # offset; routed weights are `[E_local, N, K]` and do.
                n_idx = select(is_shared(task_info),
                               n_block_idx * BLOCK_N,
                               task_info.local_expert_idx * task_info.shape_n
                               + n_block_idx * BLOCK_N)
                # instruction_selection: mad.lo.s32 + selp.b32; extent: scalar
                sfb_k_idx = select(is_shared(task_info),
                                   k_block_idx * (BLOCK_K // 128),
                                   task_info.local_expert_idx * shape_sfb_k
                                   + k_block_idx * (BLOCK_K // 128))
                k_idx, sfb_n_idx = k_block_idx * BLOCK_K, n_block_idx * BLOCK_N

                if elect_predicate(active_mask=0xFFFFFFFF):
                    if is_shared(task_info):
                        copy_g2s(tmap_b.coord(k_idx, n_idx), smem_b[stage_idx].as_("e4m3"),
                                 completion=full[stage_idx], cta_group=2)
                        # instruction_selection: cp.async.bulk.tensor.2d.cta_group::2
                        #   .shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint;
                        #   issued independently by BOTH CTAs with no CTA mask. The SM90
                        #   `.multicast::cluster` form is the other branch of `tma::copy`
                        #   (leader-CTA-only, takes a mask) and is NOT taken here;
                        #   extent: BLOCK_K / 128 boxes of <128, LOAD_BLOCK_N>, swizzle
                        #   128B, e4m3 (one box at BLOCK_K == 128, two at 256)
                        copy_g2s(tmap_sfb.coord(sfb_n_idx, sfb_k_idx), smem_sfb[stage_idx],
                                 completion=full[stage_idx], cta_group=2)
                        # instruction_selection: same TMA family, unswizzled; extent: one
                        #   <BLOCK_N, 1> box
                        if is_leader_cta:
                            expect_bytes(full[stage_idx],
                                         bytes=sizeof(smem_b[0]) * 2 + sizeof(smem_sfb[0]) * 2)
                            # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64;
                            #   extent: FP8 transfers one byte per element, so the B
                            #   contribution is the full SMEM slab of both CTAs
                        else:
                            arrive(full[stage_idx], dst_cta=0)
                            # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64; extent: one
                    else:
                        copy_g2s(tmap_b.coord(k_idx, n_idx), smem_b[stage_idx].as_("e2m1_unpacksmem"),
                                 completion=full[stage_idx], cta_group=2)
                        # instruction_selection: same TMA family with a
                        #   CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B descriptor;
                        #   extent: BLOCK_K / 128 boxes of <128, LOAD_BLOCK_N> of
                        #   packed FP4 (one box at BLOCK_K == 128, two at 256)
                        copy_g2s(tmap_sfb.coord(sfb_n_idx, sfb_k_idx), smem_sfb[stage_idx],
                                 completion=full[stage_idx], cta_group=2)
                        # instruction_selection: same TMA family, unswizzled; extent: one box
                        if is_leader_cta:
                            expect_bytes(full[stage_idx],
                                         bytes=sizeof(smem_b[0]) + sizeof(smem_sfb[0]) * 2)
                            # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64;
                            #   extent: packed FP4 moves half a byte per element while
                            #   occupying one byte in SMEM, so the B term is HALVED
                            #   relative to the shared branch above
                        else:
                            arrive(full[stage_idx], dst_cta=0)
                            # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64; extent: one
                warp_sync()
                # instruction_selection: bar.warp.sync 0xffffffff; extent: this warp
```

### Role 4: UMMA issue warp (`warp == kNumDispatchWarps + 2`, leader CTA only)

```python
    elif warp == kNumDispatchWarps + 2:
        reg_dealloc(kNumNonEpilogueRegisters)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: this warp
        if is_leader_cta:
            # Two instruction descriptors: the operand *format* travels in the
            # descriptor, not in the opcode, so both feed the same MMA below.
            routed_instr_desc = desc("instr_block_scaled", a="e2m1_unpacksmem", b="e4m3",
                                     acc="f32", sf="ue8m0", M=UMMA_M, N=UMMA_N,
                                     major_a="K", major_b="K")
            # instruction_selection: none; extent: one compile-time descriptor word.
            #   A/B are swapped: upstream's B operand occupies the instruction's A slot.
            shared_instr_desc = desc("instr_block_scaled", a="e4m3", b="e4m3",
                                     acc="f32", sf="ue8m0", M=UMMA_M, N=UMMA_N,
                                     major_a="K", major_b="K")
            # instruction_selection: none; extent: one compile-time descriptor word;
            #   only reachable when S > 0
            sf_desc = desc("smem_matrix", base=None, sbo=8 * 16, lbo=0, layout="swizzle_none")
            # instruction_selection: none; extent: one unswizzled SF descriptor
            a_desc  = desc("smem_matrix", base=smem_a[0], major="K",
                           rows=LOAD_BLOCK_M, cols=UMMA_BLOCK_K, swizzle=128)
            b_desc  = desc("smem_matrix", base=smem_b[0], major="K",
                           rows=LOAD_BLOCK_N, cols=UMMA_BLOCK_K, swizzle=128)
            shared_b_desc = desc("smem_matrix", base=smem_b[0].as_("e4m3"), major="K",
                                 rows=LOAD_BLOCK_N, cols=UMMA_BLOCK_K, swizzle=128)
            # instruction_selection: none; extent: three 64-bit descriptors. An SMEM
            #   matrix descriptor carries only {layout_type, address, SBO, LBO} -- it
            #   has NO operand-format field. With K-major, swizzle 128 and two 1-byte
            #   element types, all four fields coincide, so `shared_b_desc` is
            #   BIT-IDENTICAL to `b_desc` here. The entire routed/shared operand-type
            #   distinction is carried by the instruction descriptor selected below.
            #   Both descriptors and the runtime select are kept for source fidelity.

            # Per-stage descriptor low words held distributed across lanes:
            # lane `l` owns stage `l`.
            a_desc_lo        = select(lane < kNumStages, a_desc.lo + lane * sizeof(smem_a[0]) // 16, 0)
            b_desc_lo        = select(lane < kNumStages, b_desc.lo + lane * sizeof(smem_b[0]) // 16, 0)
            shared_b_desc_lo = select(lane < kNumStages,
                                      shared_b_desc.lo + lane * sizeof(smem_b[0]) // 16, 0)
            # Only TWO registers per lane are materialized (`a_desc_lo`, `b_desc_lo`) --
            #   `shared_b_desc_lo` is CSE'd onto `b_desc_lo`, because the two SMEM
            #   matrix descriptors are bit-identical here (see the note above).
            #   host_assert: kNumStages <= 32

            current_iter_idx = 0
            while get_next_task(task_info):
                instr_desc = select(is_shared(task_info), shared_instr_desc, routed_instr_desc)
                update_instr_desc_with_umma_n(instr_desc, get_umma_aligned_valid_m(task_info))
                # instruction_selection: integer and/or on the descriptor word; extent:
                #   scalar; the *selected* descriptor is mutated in place per task

                accum_stage_idx = current_iter_idx % kNumEpilogueStages
                accum_phase     = (current_iter_idx // kNumEpilogueStages) & 1
                current_iter_idx += 1
                wait(tmem_empty[accum_stage_idx], parity=accum_phase ^ 1)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                #   extent: one accumulator stage
                fence(scope="tcgen05", position="after_thread_sync")
                # instruction_selection: tcgen05.fence::after_thread_sync; extent: this warp

                for k_block_idx in runtime_range(task_info.shape_k // BLOCK_K, unroll_hint=2,
                                                 advance=advance_pipeline):
                    wait(full[stage_idx], parity=phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                    #   extent: one stage
                    fence(scope="tcgen05", position="after_thread_sync")
                    # instruction_selection: tcgen05.fence::after_thread_sync; extent: this warp

                    a_desc_base_lo = shuffle_index(a_desc_lo, source_lane=stage_idx,
                                                   mask=0xFFFFFFFF, clamp=0x1F)
                    # instruction_selection: shfl.sync.idx.b32; extent: one warp shuffle
                    b_desc_base_lo = shuffle_index(select(is_shared(task_info),
                                                          shared_b_desc_lo, b_desc_lo),
                                                   source_lane=stage_idx,
                                                   mask=0xFFFFFFFF, clamp=0x1F)
                    # instruction_selection: shfl.sync.idx.b32; extent: one warp shuffle.
                    #   The `is_shared` select FOLDS AWAY -- `shared_b_desc_lo` and
                    #   `b_desc_lo` are the same value, so all three unrolled copies
                    #   shuffle the one register directly, with no selp.

                    if elect_predicate(active_mask=0xFFFFFFFF):
                        for umma_k_block_idx in static_range(BLOCK_K // UMMA_BLOCK_K):
                            for i in static_range(SF_BLOCK_M // 128):
                                replace_desc_addr(sf_desc,
                                    smem_sfa[stage_idx].elem(umma_k_block_idx * SF_BLOCK_M
                                                             + i * 128))
                                # instruction_selection: integer and/or; extent: scalar
                                copy_s2t(sf_desc, tmem[kTmemStartColOfSFA + i * 4])
                                # instruction_selection: tcgen05.cp.cta_group::2.32x128b.warpx4;
                                #   extent: one 128-element SFA chunk (4 TMEM columns)
                            for i in static_range(SF_BLOCK_N // 128):
                                replace_desc_addr(sf_desc,
                                    smem_sfb[stage_idx].elem(umma_k_block_idx * SF_BLOCK_N
                                                             + i * 128))
                                # instruction_selection: integer and/or; extent: scalar
                                copy_s2t(sf_desc, tmem[kTmemStartColOfSFB + i * 4])
                                # instruction_selection: tcgen05.cp.cta_group::2.32x128b.warpx4;
                                #   extent: one 128-element SFB chunk

                            for k in static_range(UMMA_BLOCK_K // UMMA_K):
                                runtime_instr_desc = make_instr_desc_with_sf_id(instr_desc, k, k)
                                a_desc.lo = advance_desc_lo(a_desc_base_lo, major="K",
                                                            rows=LOAD_BLOCK_M, swizzle=128,
                                                            elem_bytes=1,
                                                            stage_byte=umma_k_block_idx
                                                                       * UMMA_BLOCK_K * LOAD_BLOCK_M,
                                                            k_offset=k * UMMA_K)
                                # instruction_selection: add.s32; extent: scalar
                                b_desc.lo = advance_desc_lo(b_desc_base_lo, major="K",
                                                            rows=LOAD_BLOCK_N, swizzle=128,
                                                            elem_bytes=1,
                                                            stage_byte=umma_k_block_idx
                                                                       * UMMA_BLOCK_K * LOAD_BLOCK_N,
                                                            k_offset=k * UMMA_K)
                                # instruction_selection: add.s32; extent: scalar. Upstream
                                #   selects the advance by dtype, but e4m3 and
                                #   e2m1_unpacksmem are both 1 B/elem, so the byte
                                #   arithmetic is identical for shared and routed tasks.
                                gemm(dst=tmem[accum_stage_idx * UMMA_N],
                                     lhs=b_desc, rhs=a_desc,
                                     accumulate=(k_block_idx > 0 or umma_k_block_idx > 0 or k > 0),
                                     scale_a=tmem[kTmemStartColOfSFB],
                                     scale_b=tmem[kTmemStartColOfSFA],
                                     instr=runtime_instr_desc)
                                # instruction_selection:
                                #   tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale;
                                #   extent: one UMMA_M x UMMA_N x 32 instruction,
                                #   SMEM-SMEM operands. The SAME opcode covers FP8xFP4
                                #   (routed) and FP8xFP8 (shared).
                    warp_sync()
                    # instruction_selection: bar.warp.sync 0xffffffff; extent: this warp

                    umma_arrive(empty[stage_idx], multicast=True)
                    # instruction_selection: elect.sync then
                    #   tcgen05.commit.cta_group::2.mbarrier::arrive::one
                    #   .shared::cluster.multicast::cluster.b64 with mask 0b11; extent: one commit.
                    #   No explicit `tcgen05.fence::before_thread_sync` -- the commit
                    #   performs it implicitly.
                    if k_block_idx == num_k_blocks - 1:
                        umma_arrive(tmem_full[accum_stage_idx], multicast=True)
                        # instruction_selection: same tcgen05.commit family; extent: one arrival
                    warp_sync()
                    # instruction_selection: bar.warp.sync 0xffffffff; extent: this warp

            # Drain so the barriers can be destroyed safely.
            if current_iter_idx > 0:
                wait(tmem_empty[(current_iter_idx - 1) % kNumEpilogueStages],
                     parity=((current_iter_idx - 1) // kNumEpilogueStages) & 1)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                #   extent: one drain wait
```

### Role 5: scheduler warp (`warp == kNumDispatchWarps + 3`, leader CTA only)

Three sequential phases. Shared and routed tasks are never interleaved with each
other; SharedLinear1 goes first precisely because it does **not** depend on
dispatch, so it fills the EP-dispatch latency bubble.

```python
    elif warp == kNumDispatchWarps + 3:
        reg_dealloc(kNumNonEpilogueRegisters)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: this warp
        if is_leader_cta:
            def fetch_expert_recv_count():
                for i in static_range(kNumExpertsPerLane):
                    expert_idx = i * 32 + lane
                    if expert_idx < kNumExpertsPerRank:
                        while True:
                            copy_g2r(workspace.expert_recv_count_sum[expert_idx], value)
                            # instruction_selection: ld.volatile.global.b64 spin; extent: one
                            #   counter per lane-held expert
                            if (value >> 32) == kNumSMs * kNumRanks: break
                    stored_num_tokens_per_expert[i] = value & 0xffffffff
                warp_sync()
                num_total_m_blocks = get_pool_block_offset(kNumExpertsPerRank)
                # instruction_selection: integer add/shr + redux.sync.add.u32; extent: one
                #   warp reduction over the per-lane expert block counts
                num_sched_l1_waves = min(get_num_l1_warmup_waves(...),
                                         div_ceil(num_total_m_blocks * kNumL1Clusters,
                                                  kNumSMs // 2))
                # instruction_selection: integer add/shr/min; extent: scalars

            def shared_mainloop(block_phase, shape_n, shape_k, task_count_ptr):
                kNumNClusters = shape_n // BLOCK_N // 2
                num_tasks = div_ceil(num_tokens, BLOCK_M) * kNumNClusters
                # instruction_selection: integer add/shr + mul.lo.s32; extent: scalars
                while True:
                    wait(task_empty[sched_stage_idx], parity=sched_phase ^ 1)
                    # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                    #   extent: one schedule slot
                    task_idx = get_next_task_idx(task_count_ptr)
                    # instruction_selection: elect.sync + atom.global.add.u32 +
                    #   shfl.sync.idx.b32; extent: one atomic claim broadcast to the warp.
                    #   Dynamic (atomic) scheduling spreads shared tiles over all CTA pairs.
                    if task_idx >= num_tasks: break
                    m_block_idx   = task_idx // kNumNClusters
                    n_cluster_idx = task_idx % kNumNClusters
                    valid_m = min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M)
                    # instruction_selection: integer div/rem sequence + sub + min.u32;
                    #   extent: three scalars
                    publish_task(TaskInfo(block_phase, local_expert_idx=0,
                                          m_block_idx=m_block_idx,
                                          n_cluster_idx=n_cluster_idx,
                                          pool_block_idx=m_block_idx,   # == m_block_idx
                                          valid_m=valid_m,
                                          shape_n=shape_n, shape_k=shape_k))
                    # `pool_block_idx == m_block_idx` and `local_expert_idx == 0` are the
                    # two facts the consumers rely on for shared tasks.

            # --- phase 1: SharedLinear1, before dispatch results are needed ---
            if kHasShared:
                shared_mainloop(BlockPhase.SharedLinear1,
                                SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K,
                                workspace.shared_l1_task_count)

            # --- phase 2: routed L1/L2, the existing interleaved schedule ---
            fetch_expert_recv_count()
            while True:
                wait(task_empty[sched_stage_idx], parity=sched_phase ^ 1)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                #   extent: one schedule slot
                task_info = get_next_routed_task()      # L1 warmup waves, then L1<->L2
                if task_info.block_phase == BlockPhase.None: break
                publish_task(task_info)

            # --- phase 3: SharedLinear2, after the routed stream drains ---
            if kHasShared:
                shared_mainloop(BlockPhase.SharedLinear2,
                                SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K,
                                workspace.shared_l2_task_count)
                # Running last means the SharedLinear2 producer wait in role 2 is
                # essentially always already satisfied.

            wait(task_empty[sched_stage_idx], parity=sched_phase ^ 1)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin; extent: one slot
            publish_task(TaskInfo(BlockPhase.None, 0, 0, 0, 0, 0, 0, 0))
            # sentinel: ends every consumer's `get_next_task` loop
```

The routed task walk itself is unchanged by shared experts:

```python
    def get_next_routed_task():
        while True:
            if num_sched_l1_waves not in (0, DONE):
                num_sched_l1_waves -= 1
                l1_task_idx = get_next_task_idx(workspace.l1_task_count)
                # instruction_selection: elect.sync + atom.global.add.u32 + shfl.sync.idx.b32
                if l1_task_idx >= num_total_m_blocks * kNumL1Clusters:
                    num_sched_l1_waves = DONE; continue
                return create_routed_task(BlockPhase.Linear1, l1_task_idx,
                                          kNumL1Clusters, L1_SHAPE_N, L1_SHAPE_K)
            l2_task_idx = get_next_task_idx(workspace.l2_task_count)
            # instruction_selection: elect.sync + atom.global.add.u32 + shfl.sync.idx.b32
            if l2_task_idx >= num_total_m_blocks * kNumL2Clusters: break
            if num_sched_l1_waves != DONE: num_sched_l1_waves = 1
            task_info = create_routed_task(BlockPhase.Linear2, l2_task_idx,
                                           kNumL2Clusters, L2_SHAPE_N, L2_SHAPE_K)
            while load_volatile(workspace.l1_task_count) \
                    < (task_info.pool_block_idx + 1) * kNumL1Clusters:
                pass
            # instruction_selection: ld.volatile.global.b32 spin; extent: one counter;
            #   an L2 task may not run before its block's L1 tasks were all issued
            return task_info
        return TaskInfo(BlockPhase.None, ...)

    def create_routed_task(block_phase, task_idx, num_clusters, shape_n, shape_k):
        # Locate which expert owns `m_block_idx` by a warp-wide inclusive scan over
        # the per-lane expert block counts, then broadcast that expert's fields.
        m_block_idx, n_cluster_idx = task_idx // num_clusters, task_idx % num_clusters
        for i in static_range(kNumExpertsPerLane):
            num_m_blocks = div_ceil(stored_num_tokens_per_expert[i], BLOCK_M)
            inclusive = warp_inclusive_sum(num_m_blocks, lane)
            # instruction_selection: shfl.sync.up.b32 log2(32) times + add.s32;
            #   extent: one warp scan
            ballot(owner_mask, is_owner)
            # instruction_selection: vote.sync.ballot.b32; extent: one mask
            if owner_mask:
                ffs(owner_lane, owner_mask)
                # instruction_selection: bfind/ffs sequence; extent: scalar
                local_expert_idx = shuffle_index(expert_idx, owner_lane, 0xFFFFFFFF, 0x1F)
                m_block_idx      = shuffle_index(owner_m_block_idx, owner_lane, 0xFFFFFFFF, 0x1F)
                valid_m          = shuffle_index(owner_valid_m, owner_lane, 0xFFFFFFFF, 0x1F)
                # instruction_selection: shfl.sync.idx.b32 x3; extent: three broadcasts
        return TaskInfo(block_phase, local_expert_idx, m_block_idx, n_cluster_idx,
                        pool_block_idx=task_idx // num_clusters, valid_m=valid_m,
                        shape_n=shape_n, shape_k=shape_k)
```

### Role 6a: epilogue, L1 and SharedLinear1 (`warp >= kNumDispatchWarps + 4`)

```python
    elif warp >= kNumDispatchWarps + kNumMMANonEpilogueWarps:
        reg_alloc(kNumEpilogueRegisters)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: the epilogue warps
        device_assert(copy_s2r(tmem_ptr_slot[0]) == 0)
        # instruction_selection: ld.shared.u32 + setp.eq.b32 + @p bra <skip> + trap;
        #   extent: one check, branch inverted around the trap

        epilogue_warp_idx = warp - (kNumDispatchWarps + kNumMMANonEpilogueWarps)
        epilogue_wg_idx   = epilogue_warp_idx // 4
        warp_idx_in_wg    = epilogue_warp_idx % 4
        barrier(barrier_id=1, arrival_count=kNumDispatchThreads + kNumEpilogueThreads)
        # instruction_selection: barrier.sync 1, N (unaligned); extent: dispatch + epilogue

        current_iter_idx = 0
        while get_next_task(task_info):
            accum_stage_idx = current_iter_idx % kNumEpilogueStages
            accum_phase     = (current_iter_idx // kNumEpilogueStages) & 1
            current_iter_idx += 1
            wait(tmem_full[accum_stage_idx], parity=accum_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
            #   extent: one accumulator stage
            fence(scope="tcgen05", position="after_thread_sync")
            # instruction_selection: tcgen05.fence::after_thread_sync; extent: epilogue warps
            release_task_info()

            valid_m        = shuffle_index(task_info.valid_m, 0, 0xFFFFFFFF, 0x1F)
            # instruction_selection: shfl.sync.idx.b32; extent: one broadcast so NVCC
            #   treats the valid-row early exit as warp-uniform
            pool_block_idx = task_info.pool_block_idx
            ring_block_idx = pool_block_idx % kNumRingBlocks
            block_idx      = select(is_shared(task_info), pool_block_idx, ring_block_idx)
            ring_m_idx     = ring_block_idx * BLOCK_M      # ring data buffers
            m_idx          = block_idx * BLOCK_M           # TMA store coordinate
            pool_m_idx     = pool_block_idx * BLOCK_M      # non-ring metadata
            n_block_idx    = task_info.n_cluster_idx * 2 + (0 if is_leader_cta else 1)
            n_idx          = n_block_idx * BLOCK_N
            # For a shared task `pool_block_idx == m_block_idx`, so `pool_m_idx` is
            # the plain local token offset.

            if task_info.block_phase in (BlockPhase.Linear1, BlockPhase.SharedLinear1):
                if not is_shared(task_info):
                    while load_acquire(workspace.l2_empty_count[ring_block_idx]) \
                            != (L2_SHAPE_N // BLOCK_N) * (pool_block_idx // kNumRingBlocks):
                        pass
                    # instruction_selection: ld.acquire.gpu.global.b32 spin; extent: one
                    #   counter. Shared tasks skip this: their L2 buffer is not a ring,
                    #   so there is nothing to wait to be released.

                stored_cached_weight = 1.0
                # instruction_selection: mov.b32 <reg>, 1065353216 (1.0f), sunk into the
                #   guard at impl:1017 -- impl:997 carries no instruction of its own.
                #   This default
                #   IS the shared-expert path: the shared output is UNWEIGHTED.

                for s in static_range(WG_BLOCK_M // STORE_BLOCK_M):
                    if epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m:
                        fence(scope="tcgen05", position="before_thread_sync")
                        # instruction_selection: tcgen05.fence::before_thread_sync; extent: warp
                        arrive(tmem_empty[accum_stage_idx], dst_cta=0)
                        # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64; extent: one
                        break

                    for i in static_range(kNumAtomsPerStore):
                        j = s * kNumAtomsPerStore + i
                        if not is_shared(task_info) and (j * ATOM_M) % 32 == 0 \
                                and (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane < WG_BLOCK_M):
                            copy_g2r(l1_topk_w[ring_m_idx + epilogue_wg_idx * WG_BLOCK_M
                                               + j * ATOM_M + lane], stored_cached_weight)
                            # instruction_selection: ld.global.b32; extent: one float per lane,
                            #   refreshed once per 32 tokens. Guarded off for shared tasks,
                            #   which keep the 1.0 default.
                        weights = (shuffle_index(stored_cached_weight,
                                                 (j * ATOM_M) % 32 + (lane % 4) * 2 + 0, ...),
                                   shuffle_index(stored_cached_weight,
                                                 (j * ATOM_M) % 32 + (lane % 4) * 2 + 1, ...))
                        # instruction_selection: shfl.sync.idx.b32 x2; extent: two broadcasts

                        tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M \
                                    + j * ATOM_M
                        copy_t2r(tmem[tmem_addr], raw_values[0:4])
                        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x1.b32;
                        #   extent: four registers, lanes from 0
                        copy_t2r(tmem[tmem_addr | 0x00100000], raw_values[4:8])
                        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x1.b32
                        #   with the row-half bit; extent: four registers, lanes from 16
                        fence(scope="view_async_tmem_load")
                        # instruction_selection: tcgen05.wait::ld.sync.aligned; extent: one wait
                        if j == WG_BLOCK_M // ATOM_M - 1:
                            fence(scope="tcgen05", position="before_thread_sync")
                            # instruction_selection: tcgen05.fence::before_thread_sync; extent: warp
                            arrive(tmem_empty[accum_stage_idx], dst_cta=0)
                            # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64; extent: one

                        for k in static_range(2):
                            cast(bf16_gate, fp32_values[k * 2 + 0], rounding="rn", pack=True)
                            # instruction_selection: cvt.rn.bf16x2.f32; extent: one packed pair
                            cast(bf16_up,   fp32_values[k * 2 + 1], rounding="rn", pack=True)
                            # instruction_selection: cvt.rn.bf16x2.f32; extent: one packed pair
                            if kActivationClamp != inf:
                                min(bf16_gate, bf16_gate, +kActivationClamp)
                                # instruction_selection: min.bf16x2; extent: one packed pair
                                max(bf16_up, bf16_up, -kActivationClamp)
                                # instruction_selection: max.bf16x2; extent: one packed pair
                                min(bf16_up, bf16_up, +kActivationClamp)
                                # instruction_selection: min.bf16x2; extent: one packed pair
                            cast(gate, bf16_gate)
                            # instruction_selection: cvt.f32.bf16 x2; extent: two values
                            exp(neg_gate_exp, -gate)
                            # instruction_selection: ex2.approx.f32 x2 (kFastMath) or the
                            #   libdevice expf sequence; extent: two values
                            add(denom, 1.0, neg_gate_exp)
                            # instruction_selection: add.rn.f32x2; extent: one packed pair
                            if kFastMath:
                                rcp(inv_denom, denom)
                                # instruction_selection: rcp.approx.ftz.f32 x2; extent: two values
                                mul(gate, gate, inv_denom)
                                # instruction_selection: mul.rn.f32x2; extent: one packed pair
                            else:
                                div(gate, gate, denom)
                                # instruction_selection: div.rn.f32 x2; extent: two values
                            cast(up, bf16_up)
                            # instruction_selection: cvt.f32.bf16 x2; extent: two values;
                            #   the up half converts back AFTER the div/rcp, immediately
                            #   before the product below (impl:1070)
                            mul(activation_values[i][k], mul(gate, up), weights)
                            # instruction_selection: mul.rn.f32x2 x2; extent: two packed pairs;
                            #   `weights` is 1.0 for shared tasks, so this is the identity there

                        max(thread_local_amax, abs(activation_values[i][:]))
                        # instruction_selection: setp.lt.f32 + neg.f32 + selp.f32 (abs) +
                        #   max.f32; extent: per-thread
                        warp_reduce_max(amax_values[i], thread_local_amax, width=4)
                        # instruction_selection: shfl.sync.bfly.b32 x3 (xor 4, 8, 16) +
                        #   setp.gt.f32 + selp.f32; extent: per component, 6 shuffles for
                        #   the float2. This reduces ACROSS the width-4 lane groups: every
                        #   lane ends holding the max over the 8 lanes sharing `lane % 4`,
                        #   which is what makes the warp-pair step below correct.
                        if lane < 4:
                            copy_r2s(amax_values[i], amax_red[epilogue_warp_idx][i * (ATOM_M // 2) + lane])
                            # instruction_selection: st.shared.v2.b32; extent: 8 bytes per lane
                        warp_sync()
                        # instruction_selection: bar.warp.sync 0xffffffff; extent: this warp

                    tma_stage_idx = s % kNumTMAStoreStages
                    store_wait(pending=kNumTMAStoreStages - 1)
                    # instruction_selection: cp.async.bulk.wait_group 1; extent: one warp
                    barrier(barrier_id=3 + epilogue_wg_idx, arrival_count=128)
                    # instruction_selection: bar.sync 3+wg, 128; extent: one epilogue warpgroup;
                    #   also fences `amax_red`

                    for i in static_range(kNumAtomsPerStore):
                        copy_s2r(amax_red[epilogue_warp_idx ^ 1][i * (ATOM_M // 2) + lane % 4], wp_amax)
                        # instruction_selection: ld.shared.v2.b32; extent: 8 bytes per lane
                        max(amax_values[i], amax_values[i], wp_amax)
                        # instruction_selection: setp.lt.f32 + selp.f32 x2; extent: the
                        #   warp-pair reduction, one predicate/select per float2 component
                        get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv)
                        # instruction_selection: mul.rn.f32x2 by 1/448, then
                        #   `fast_log2_ceil` (shr.u32 + and.b32 + setp.ne.b32 +
                        #   selp.b32 + add.s32) per component, then ONE `fast_pow2`
                        #   bit construction per component -- `shl.b32 <exp>, 23` +
                        #   `sub.s32 1065353216, ...` -- producing ONLY `sf_inv`.
                        #   `sf` is never materialized: its sole use `sf >> 23` folds
                        #   into the `add.s32 <exp>, -385` at the two st.global.b8 rows
                        #   below. There is no reciprocal anywhere here.
                        mul(upper, activation_values[i][0], sf_inv)
                        # instruction_selection: mul.rn.f32x2; extent: one packed pair
                        mul(lower, activation_values[i][1], sf_inv)
                        # instruction_selection: mul.rn.f32x2; extent: one packed pair
                        cast(fp8x4_values, (upper, lower), rounding="rn", pack=True)
                        # instruction_selection: cvt.rn.satfinite.e4m3x2.f32 x2; extent: 4 bytes
                        copy_r2s(fp8x4_values,
                                 smem_d_l1[epilogue_wg_idx][tma_stage_idx]
                                     .byte(i * ATOM_M * L1_OUT_BLOCK_N + lane * L1_OUT_BLOCK_N
                                           + (warp_idx_in_wg ^ (lane // 2)) * 16),
                                 transpose=True)
                        # instruction_selection: stmatrix.sync.aligned.m16n8.x1.trans.shared.b8;
                        #   extent: one transposed 4-byte group per lane; the 64B swizzle
                        #   is the `col ^ (row / 2)` term

                        if warp_idx_in_wg % 2 == 0 and lane < 4:
                            mn_stride = select(is_shared(task_info),
                                               kNumSharedSFTokens, kNumSFRingTokens) * 4
                            sf_base   = select(is_shared(task_info), shared_l2_sf, l2_sf)
                            # instruction_selection: selp.b32 + selp.b64; extent: two selects.
                            #   The shared SF plane is `kNumSharedSFTokens` rows tall and
                            #   is indexed by the ABSOLUTE pool block.
                            k_idx = n_block_idx * 2 + warp_idx_in_wg // 2
                            # instruction_selection: the whole SF-address block below is
                            #   HOISTED into the inlined `sync_aligned` at impl:1098 and
                            #   emits exactly one mad.lo.s32 plus shl/or/shr/and/add and
                            #   the two selects above; extent: one hoisted block, not one
                            #   sequence per row. Selects which SF column this exponent
                            #   belongs to.
                            k_uint_idx, byte_idx = k_idx // 4, k_idx % 4
                            token_base_idx = (epilogue_wg_idx * WG_BLOCK_M
                                              + s * STORE_BLOCK_M + i * ATOM_M)
                            # (always < BLOCK_M, which is why `lane*2` can be hoisted
                            #  out of `transform_sf_token_idx` below)
                            sf_token_idx = block_idx * SF_BLOCK_M \
                                           + transform_sf_token_idx(token_base_idx) + (lane * 2) * 4
                            copy_r2g(sf.x >> 23, sf_base.byte(k_uint_idx * mn_stride
                                                              + sf_token_idx * 4 + byte_idx))
                            # instruction_selection: st.global.b8; extent: one UE8M0 exponent
                            copy_r2g(sf.y >> 23, sf_base.byte(... + 4 * 4))
                            # instruction_selection: st.global.b8; extent: one UE8M0 exponent
                        warp_sync()
                        # instruction_selection: bar.warp.sync 0xffffffff; extent: this warp
                    barrier(barrier_id=3 + epilogue_wg_idx, arrival_count=128)
                    # instruction_selection: bar.sync 3+wg, 128; extent: one epilogue warpgroup

                    if warp_idx_in_wg == 0 and elect_predicate(active_mask=0xFFFFFFFF):
                        tmap_out = select(is_shared(task_info),
                                          tmap_shared_l1_output, tmap_l1_output)
                        # instruction_selection: selp.b64; extent: one descriptor select
                        fence(scope="tma_store")
                        # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                        copy_s2g(smem_d_l1[epilogue_wg_idx][tma_stage_idx],
                                 tmap_out.coord(n_block_idx * L1_OUT_BLOCK_N,
                                                m_idx + epilogue_wg_idx * WG_BLOCK_M
                                                + s * STORE_BLOCK_M))
                        # instruction_selection: cp.async.bulk.tensor.2d.global.shared::cta
                        #   .bulk_group; extent: one <L1_OUT_BLOCK_N, STORE_BLOCK_M> box
                        store_arrive()
                        # instruction_selection: cp.async.bulk.commit_group; extent: one group
                    warp_sync()

                store_wait(pending=0)
                # instruction_selection: cp.async.bulk.wait_group 0; extent: one warp
                barrier(barrier_id=2, arrival_count=kNumEpilogueThreads)
                # instruction_selection: bar.sync 2, kNumEpilogueThreads; extent: epilogue
                if epilogue_warp_idx == 0 and elect_predicate(active_mask=0xFFFFFFFF):
                    if is_shared(task_info):
                        red_add_rel(workspace.shared_l2_full_count[pool_block_idx], 1)
                        # instruction_selection: red.release.gpu.global.add.u32; extent: one
                        #   counter. Shared has NO l1_empty bookkeeping: nothing recycles.
                    else:
                        red_add_rel(workspace.l2_full_count[ring_block_idx], 1)
                        # instruction_selection: red.release.gpu.global.add.u32; extent: one counter
                        red_add(workspace.l1_empty_count[ring_block_idx], 1)
                        # instruction_selection: red.gpu.global.add.u32; extent: one ring slot release
                warp_sync()
```

### Role 6b: epilogue, L2 and SharedLinear2

```python
            else:
                if not is_shared(task_info):
                    if epilogue_warp_idx == 0 and elect_predicate(active_mask=0xFFFFFFFF):
                        red_add(workspace.l2_empty_count[ring_block_idx], 1)
                        # instruction_selection: red.gpu.global.add.u32; extent: one ring slot release
                    warp_sync()
                    # Shared tasks skip this entirely -- no ring, nothing to release.

                for s in static_range(WG_BLOCK_M // STORE_BLOCK_M):
                    if epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m:
                        fence(scope="tcgen05", position="before_thread_sync")
                        # instruction_selection: tcgen05.fence::before_thread_sync; extent: warp
                        arrive(tmem_empty[accum_stage_idx], dst_cta=0)
                        # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64; extent: one
                        break

                    for i in static_range(STORE_BLOCK_M // ATOM_M):
                        tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M \
                                    + s * STORE_BLOCK_M + i * ATOM_M
                        copy_t2r(tmem[tmem_addr], values[0:4])
                        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x1.b32;
                        #   extent: four registers, lanes from 0
                        copy_t2r(tmem[tmem_addr | 0x00100000], values[4:8])
                        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x1.b32
                        #   with the row-half bit; extent: four registers, lanes from 16
                        fence(scope="view_async_tmem_load")
                        # instruction_selection: tcgen05.wait::ld.sync.aligned; extent: one wait
                        if i == 0 and s > 0:
                            barrier(barrier_id=3 + epilogue_wg_idx, arrival_count=128)
                            # instruction_selection: bar.sync 3+wg, 128; extent: one warpgroup
                        if s == WG_BLOCK_M // STORE_BLOCK_M - 1 and i == STORE_BLOCK_M // ATOM_M - 1:
                            fence(scope="tcgen05", position="before_thread_sync")
                            # instruction_selection: tcgen05.fence::before_thread_sync; extent: warp
                            arrive(tmem_empty[accum_stage_idx], dst_cta=0)
                            # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64; extent: one
                        for p in static_range(4):
                            cast(packed[p], (values[2 * p], values[2 * p + 1]),
                                 rounding="rn", pack=True)
                            # instruction_selection: cvt.rn.bf16x2.f32; extent: one packed pair
                        copy_r2s(packed,
                                 smem_d_l2[epilogue_wg_idx]
                                     .byte((warp_idx_in_wg // 2) * STORE_BLOCK_M * 128
                                           + i * ATOM_M * 128
                                           + (lane % 8) * (16 * 8)
                                           + (((epilogue_warp_idx % 2) * 4 + lane // 8)
                                              ^ (lane % 8)) * 16),
                                 transpose=True)
                        # instruction_selection: stmatrix.sync.aligned.x4.m8n8.shared.b16.trans;
                        #   extent: one transposed 8x8x4 matrix store per lane group

                    barrier(barrier_id=3 + epilogue_wg_idx, arrival_count=128)
                    # instruction_selection: bar.sync 3+wg, 128; extent: one warpgroup

                    row_in_atom    = (warp_idx_in_wg * 2 + lane // 16) % ATOM_M
                    bank_group_idx = lane % 8
                    # (address arithmetic for the swizzled SMEM row read back below;
                    #  emits nothing of its own -- it folds into the ld.shared.v4.f32)
                    for j in static_range(STORE_BLOCK_M // 8):
                        row_in_store  = j * 8 + warp_idx_in_wg * 2 + lane // 16
                        m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M \
                                         + row_in_store
                        if m_idx_in_block >= valid_m: break
                        # instruction_selection: setp.ge.u32 + bra; extent: one loop exit

                        if is_shared(task_info):
                            dst_rank_idx, dst_token_idx, dst_topk_idx = \
                                sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, K
                            # instruction_selection: mov/add.s32; extent: three scalars.
                            #   The shared result goes to THIS rank's own combine buffer,
                            #   slot `K` (the extra slot), at the local token index.
                            #   No NVLink traffic and no metadata read.
                        else:
                            copy_g2r(workspace.token_src_metadata[pool_m_idx + m_idx_in_block],
                                     (dst_rank_idx, dst_token_idx, dst_topk_idx))
                            # instruction_selection: ld.global.b32 x3; extent: up to
                            #   12 bytes -- where this row came from; the rank field
                            #   folds away when kNumRanks == 1

                        copy_s2r(smem_d_l2[epilogue_wg_idx]
                                     .byte((lane % 16 // 8) * STORE_BLOCK_M * kSwizzleCDMode
                                           + row_in_store * kSwizzleCDMode
                                           + (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes),
                                 packed_out)
                        # instruction_selection: ld.shared.v4.f32; extent: 16 bytes per lane
                        copy_r2g(packed_out,
                                 peer(combine[dst_topk_idx][dst_token_idx]
                                          .byte(n_idx * 2 + (lane % 16) * 16), dst_rank_idx))
                        # instruction_selection: st.global.v4.b32; extent: 16 bytes to the
                        #   destination rank's combine buffer
                barrier(barrier_id=2, arrival_count=kNumEpilogueThreads)
                # instruction_selection: bar.sync 2, kNumEpilogueThreads; extent: epilogue
```

### Role 6c: combine reduction (every epilogue warp)

```python
        if epilogue_warp_idx == 0:
            tmem_free(base=0, columns=kNumTmemCols)
            # instruction_selection: tcgen05.dealloc.cta_group::2.sync.aligned.b32;
            #   extent: one deallocation, from the same logical warp on both CTAs
        nvlink_barrier(tag=2, index=1, sync_prologue=True, sync_epilogue=True)
        # instruction_selection: one cross-rank rendezvous with BOTH grid syncs
        #   (the defaults); ~4 us. NOTE the bracketing barrier is the EPILOGUE's:
        #   `bar.sync 2, kNumEpilogueThreads`, not the dispatch path's `0, ...` --
        #   reusing barrier 0 here would collide with the dispatch warps.
        #   Each ENABLED grid sync is `bar.sync 2, kNumEpilogueThreads` +
        #   `atom.release.gpu.global.add.u32` + `ld.acquire.gpu.global.b32` spin +
        #   `bar.sync 2, kNumEpilogueThreads` (BOTH brackets, and the mid-signal
        #   `sync_scope` too); the cross-rank signal itself is
        #   `red.release.sys.global.add.s32` to every peer, spun on with
        #   `ld.acquire.sys.global.s32`.
        barrier(barrier_id=1, arrival_count=kNumDispatchThreads + kNumEpilogueThreads)
        # instruction_selection: barrier.sync 1, N (unaligned); extent: dispatch + epilogue,
        #   releasing the dispatch warps to clean the workspace

        # Reuse everything before `dispatch_bar` as the staging ring.
        combine_load  = view(smem_raw, "u4x4", [2], byte_offset=lambda i:
                             (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes)
        combine_store = view(smem_raw, "u4x4", [1], byte_offset=
                             (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes)
        combine_bar_w = view(combine_bar, [2], base=epilogue_warp_idx * 2)
        # instruction_selection: none; extent: this warp's PRIVATE pair of combine
        #   barriers. The array is kNumEpilogueWarps*2 long with arrival_count 1
        #   precisely so no two epilogue warps share a slot.
        combine_phase, load_stage_idx = 0, 0
        # Two warp-local cursors carried ACROSS tokens and chunks, not reset per
        # iteration. (Cursor bookkeeping; the zeroing is sunk elsewhere.)
        kNumHiddenBytes    = H * 2
        kNumChunkSlots     = 3          # two load stages + one store buffer
        kNumMaxRegsForBuf  = 128
        # One chunk when the whole row fits both the reusable SMEM budget AND the
        # per-lane register budget; otherwise two. This trip count is load-bearing:
        # it is the number of 1-D TMA loads and stores per token below.
        kNumChunks = 1 if (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes
                           <= kNumReusableSmemBytes and H <= 32 * kNumMaxRegsForBuf) else 2
        kNumChunkBytes     = kNumHiddenBytes // kNumChunks
        kNumChunkUint4     = kNumChunkBytes // 16
        kNumUint4PerLane   = kNumChunkUint4 // 32
        kNumElemsPerUint4  = 16 // 4            # uint4 -> four bf16x2 pairs
        # instruction_selection: none; extent: compile-time constants
        # host_assert: H % kNumChunks == 0; kNumChunkBytes % 16 == 0
        # host_assert: kNumChunkUint4 % 32 == 0 (one 16-byte element per lane)
        device_assert(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes
                      <= kNumReusableSmemBytes)
        # instruction_selection: none -- every operand is a compile-time constant, so
        #   this DG_DEVICE_ASSERT (impl:1348) folds away and emits no setp/trap in any
        #   valid instantiation. It uses the runtime-assert macro but is not a runtime
        #   check; the PTX contains no trap attributed to that line.
        # host_assert: kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks
        #   <= kNumReusableSmemBytes (impl:1341, the genuinely static budget assert)
        # host_assert: kNumTopk + (1 if S > 0 else 0) <= 32

        for token_idx in strided_range(start=sm_idx * kNumEpilogueWarps + epilogue_warp_idx,
                                       bound=num_tokens,
                                       step=kNumSMs * kNumEpilogueWarps):
            if lane < K:
                copy_g2r(input_topk_idx[token_idx * K + lane], stored_topk_slot_idx)
                # instruction_selection: ld.global.nc.s64; extent: one i64 per lane
            elif kHasShared and lane == K:
                move(stored_topk_slot_idx, K)
                # instruction_selection: mov.pred; extent: NO value is materialized --
                #   `stored_topk_slot_idx` feeds only the `>= 0` ballot below, so the
                #   lane == K arm collapses into that predicate and the slot index is
                #   recovered downstream from the mask by `ffs`. Lane `K` still
                #   synthesizes the shared slot UNCONDITIONALLY, so a token whose
                #   topk_idx are all -1 still receives the shared contribution.
            else:
                move(stored_topk_slot_idx, -1)
            ballot(total_mask, stored_topk_slot_idx >= 0)
            # instruction_selection: vote.sync.ballot.b32; extent: one mask of up to K+1 bits

            for chunk in static_range(kNumChunks):
                def move_mask_and_load(i):
                    if mask == 0: return False
                    ffs(slot_idx, mask); mask ^= 1 << slot_idx
                    # instruction_selection: bfind/ffs + xor.b32; extent: two scalars
                    if elect_predicate(active_mask=0xFFFFFFFF):
                        copy_g2s(combine[slot_idx][token_idx].byte(chunk * kNumChunkBytes),
                                 combine_load[i], completion=combine_bar_w[i])
                        # instruction_selection: cp.async.bulk.shared::cluster.global
                        #   .mbarrier::complete_tx::bytes.L2::cache_hint; extent: one chunk
                        expect_bytes(combine_bar_w[i], bytes=kNumChunkBytes)
                        # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64; extent: one
                    warp_sync()
                    return True

                do_reduce = move_mask_and_load(load_stage_idx)
                fill(reduced, 0.0)
                # instruction_selection: one `mov.b32 <reg>, 0f00000000` zero constant
                #   (materialized twice) feeding 2N `mov.b32` register copies at
                #   impl:1405; extent: all 2 * kNumUint4PerLane * kNumElemsPerUint4
                #   accumulator registers
                while do_reduce:
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1)   # prefetch next slot
                    wait(combine_bar_w[load_stage_idx], parity=combine_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
                    #   extent: one load slot
                    for j in static_range(kNumUint4PerLane):
                        copy_s2r(combine_load[load_stage_idx][j * 32 + lane], uint4_values)
                        # instruction_selection: ld.shared.v4.b32; extent: 16 bytes per lane
                        for l in static_range(kNumElemsPerUint4):
                            add(reduced[j * kNumElemsPerUint4 + l], reduced[...], bf16_values[l])
                            # instruction_selection: add.rn.f32.bf16 x2; extent: two
                            #   scalar FUSED cast+adds per bf16x2 -- there is no separate
                            #   unpack step (2 * kNumUint4PerLane * kNumElemsPerUint4 per
                            #   chunk). The shared slot is one more addend, unweighted.
                    combine_phase ^= load_stage_idx; load_stage_idx ^= 1

                for j in static_range(kNumUint4PerLane):
                    for l in static_range(kNumElemsPerUint4):
                        cast(casted_bf16[l], reduced[j * kNumElemsPerUint4 + l],
                             rounding="rn", pack=True)
                        # instruction_selection: cvt.rn.bf16x2.f32; extent: one packed pair
                    if j == 0:
                        store_wait(pending=0)
                        # instruction_selection: cp.async.bulk.wait_group 0; extent: one warp
                        warp_sync()
                    copy_r2s(casted, combine_store + j * 32 + lane)
                    # instruction_selection: st.shared.v4.u32; extent: 16 bytes per lane
                warp_sync()

                if elect_predicate(active_mask=0xFFFFFFFF):
                    fence(scope="tma_store")
                    # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                    copy_s2g(combine_store, y.byte(token_idx * H * 2 + chunk * kNumChunkBytes))
                    # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group
                    #   .L2::cache_hint; extent: one chunk of the final output row
                    store_arrive()
                    # instruction_selection: cp.async.bulk.commit_group; extent: one group
                warp_sync()
```

## TensorMap fields

The nine shared descriptors mirror the routed ones with `num_groups = 1` and
`I -> S*I`; the host binds routed descriptors into those slots when `S == 0`.

| ABI parameter | global dims (inner, outer) | box | element type | swizzle |
| --- | --- | --- | --- | --- |
| `tmap_l1_acts` | `(H, kNumRingTokens)` | `(128 = swizzle/elem, LOAD_BLOCK_M)` | `e4m3` | 128 |
| `tmap_l1_acts_sf` | `(kNumSFRingTokens, H/128)` | `(SF_BLOCK_M, BLOCK_K/128)` | `int32` | 0 |
| `tmap_l1_weights` | `(H, E_local * 2I)` | `(128 = swizzle/elem, LOAD_BLOCK_N)` | `16U4_ALIGN16B` | 128 |
| `tmap_l1_weights_sf` | `(2I, ceil(H/128))`, groups `E_local` | `(BLOCK_N, BLOCK_K/128)` | `int32` | 0 |
| `tmap_l1_output` | `(I, kNumRingTokens)` | `(BLOCK_N/2, STORE_BLOCK_M)` | `e4m3` | 64 |
| `tmap_l2_acts` | `(I, kNumRingTokens)` | `(128 = swizzle/elem, LOAD_BLOCK_M)` | `e4m3` | 128 |
| `tmap_l2_acts_sf` | `(kNumSFRingTokens, I/128)` | `(SF_BLOCK_M, BLOCK_K/128)` | `int32` | 0 |
| `tmap_l2_weights` | `(I, E_local * H)` | `(128 = swizzle/elem, LOAD_BLOCK_N)` | `16U4_ALIGN16B` | 128 |
| `tmap_l2_weights_sf` | `(H, ceil(I/128))`, groups `E_local` | `(BLOCK_N, BLOCK_K/128)` | `int32` | 0 |
| `tmap_shared_l1_acts` | `(H, T)` — aliases `x` | `(128 = swizzle/elem, LOAD_BLOCK_M)` | `e4m3` | 128 |
| `tmap_shared_l1_acts_sf` | `(kNumSharedSFTokens, H/128)` | `(SF_BLOCK_M, BLOCK_K/128)` | `int32` | 0 |
| `tmap_shared_l1_weights` | `(H, 2*S*I)` | `(128 = swizzle/elem, LOAD_BLOCK_N)` | **`e4m3`** | 128 |
| `tmap_shared_l1_weights_sf` | `(2SI, ceil(H/128))`, groups `1` | `(BLOCK_N, BLOCK_K/128)` | `int32` | 0 |
| `tmap_shared_l1_output` | `(S*I, T)` | `(BLOCK_N/2, STORE_BLOCK_M)` | `e4m3` | 64 |
| `tmap_shared_l2_acts` | `(S*I, T)` | `(128 = swizzle/elem, LOAD_BLOCK_M)` | `e4m3` | 128 |
| `tmap_shared_l2_acts_sf` | `(kNumSharedSFTokens, SI/128)` | `(SF_BLOCK_M, BLOCK_K/128)` | `int32` | 0 |
| `tmap_shared_l2_weights` | `(S*I, H)` | `(128 = swizzle/elem, LOAD_BLOCK_N)` | **`e4m3`** | 128 |
| `tmap_shared_l2_weights_sf` | `(H, ceil(SI/128))`, groups `1` | `(BLOCK_N, BLOCK_K/128)` | `int32` | 0 |

The encoded SMEM box inner dim is `kSwizzleMode / elem_size` (128 for every 1-byte
A/B type here), NOT `BLOCK_K`; `tma::copy` then issues `BLOCK_K / 128` such boxes
per call -- one when `BLOCK_K == 128`, two in the `BLOCK_K == 256` (bm16) arm.
Only the unswizzled SF descriptors keep the literal box the caller passes.

Coordinates are always `(inner, outer)`. Every load passes a multicast count of
two; the two CTAs of a cluster split A along M and take adjacent N blocks.

## Storage lifetimes

| Object | Producer | Consumer | Released by |
| --- | --- | --- | --- |
| `l1_tokens[ring]`, `l1_sf` | dispatch warps, publish `l1_full_count[ring]` | A-load warp (Linear1), via `tmap_l1_acts` / `tmap_l1_acts_sf` | `l1_empty_count[ring]` from the L1 epilogue |
| `l1_topk_w` | dispatch warps, same `l1_full_count[ring]` edge | L1 **epilogue** warps (`copy_g2r`, impl:1019) -- no TensorMap ever references it | `l1_empty_count[ring]` from the L1 epilogue |
| `l2_tokens[ring]`, `l2_sf` | L1 epilogue TMA store, publish `l2_full_count[ring]` | A-load warp (Linear2) | `l2_empty_count[ring]` from the L2 epilogue |
| `shared_l1_sf` | **the host**, before the launch | A-load warp (SharedLinear1) | never — read-only for the kernel |
| `shared_l2_tokens`, `shared_l2_sf` | SharedL1 epilogue, publish `shared_l2_full_count[pool]` | A-load warp (SharedLinear2) | never — full-size, one write per block |
| `smem_a[s]`, `smem_b[s]`, `smem_sfa[s]`, `smem_sfb[s]` | A/B load warps, publish `full[s]` | UMMA warp | `umma_arrive(empty[s])` |
| TMEM accum `[e*UMMA_N, (e+1)*UMMA_N)` | UMMA warp, publish `tmem_full[e]` | epilogue warps | `arrive(tmem_empty[e])` per epilogue thread |
| `smem_d.l1[wg][t]` | L1 epilogue | TMA store | `cp.async.bulk.wait_group 1` |
| `smem_d.l2[wg]` | L2 epilogue | NVLink combine write | `bar.sync 3+wg` |
| `combine[slot][token]` | L2/SharedL2 epilogue across ranks | combine reduction | end of kernel |
| `task_infos[s]` | scheduler warp `st_async` | four GEMM roles | `task_empty[s]` from the epilogue |

The shared rows are what make the shared path non-ring: `shared_l2_full_count`
is written exactly once per `(pool block, N block)` pair and never reset inside
the launch, so its consumer wait carries no wave multiplier.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `kNumSharedExperts` (`S`) | static | adds the two shared schedule phases, the four-way descriptor selects, the second instruction descriptor, the shared symmetric-buffer regions, and the extra combine slot; `S == 0` folds `is_shared()` to false and deletes all of it |
| `kHasShared` | static, derived | selects `TaskInfo<true>` — same size as `TaskInfo<false>`, so no SMEM offset moves |
| `SHARED_L2_SHAPE_K = I*S` | static | only the SharedLinear2 producer-wait count |
| `kNumSharedSFTokens` | static | only the shared SF plane's MN stride |
| shared weight dtype `e4m3` | static | second instruction descriptor and the doubled B expect-tx; the MMA opcode is unchanged |
| `BLOCK_M/N/K`, stages, thread split, ring sizes | static, from the heuristic | **independent of `S`** — the block-config heuristic has no shared-expert term |
| `num_tokens` | runtime | the shared-FFN M extent, and the shared task count |
| `task_info.shape_n/shape_k` | runtime | carries the shared shapes without new registers |
| `kActivationClamp`, `kFastMath` | static | SwiGLU clamp instructions and `ex2.approx`/`rcp.approx` selection |

## Host-side contract

Two obligations do not appear in the kernel and must be met by the caller:

1. `shared_l1_acts_sf` is **written by the host** before every launch, in the
   UTCCP-transposed layout `block * align_up(BLOCK_M,128) + transform(m)` with
   `BLOCK_M` from the runtime heuristic (`get_block_m_for_mega_moe`). The plane is
   **zero-filled first** and only rows `[0, num_tokens)` are written, so every
   padding row -- including the tail of the last partial `BLOCK_M` block -- stays
   zero. (`_copy_fp8_sf`'s replicate-last-row branch is dead at this call site:
   the plane is built with `num_max_sf_tokens == shared_l1_acts_sf.shape[0]`, so
   `dst.shape == src.shape` and the plain copy path is taken.) Those rows feed only
   UMMA lanes that `valid_m` discards, so the padding value is not load-bearing.
   The kernel only reads this plane.
2. The symmetric buffer must be allocated with the same `S` used at launch: the
   routed ring buffers are rebased after the shared regions, and the combine
   buffer gains a slot, so a buffer allocated with `S == 0` is not layout
   compatible with a call that passes shared weights.

## TIRx module and benchmark contract

- One builder, `_sm100_fp8_fp4_mega_moe.kernel.get_kernel(...)`, with
  `num_shared_experts` as a compile-time keyword feeding the `@cache`d
  `_compile_tirx_mega_moe_for_config` key; the prim_func keeps a constant
  18-TensorMap ABI so the `S == 0` and `S > 0` specializations share it.
- The kernel is plain TIRx: `T.ptx` / `T.cuda` intrinsics, explicit loops, and
  hand-carved shared/tensor-memory buffers. No `Tx` tile primitive may appear.
- Correctness compares against `deep_gemm.fp8_fp4_mega_moe` with the same shared
  weights, requiring **bitwise** equality of `y` and of the cumulative expert
  recv stats.
- The timed implementation is named `tirx`; weight transformation, SF layout
  conversion, TensorMap encoding, compilation and validation stay outside the
  timed closure. The reference is `deepgemm`, timed by the `megamoe` protocol
  (DeepGEMM `bench_kineto`), which fixes its own schedule.

## Instruction selection is a lowering consequence

| Sketch construct | PTX | SASS family |
| --- | --- | --- |
| `copy_g2s` on A/B/SFA/SFB | `cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` | `UTMALDG` |
| `copy_g2s` / `copy_s2g` in the pull and combine paths | `cp.async.bulk.{shared::cluster.global,global.shared::cta}` (1-D) | `UTMALDG` / `UTMASTG` |
| `copy_s2g` on the L1 output | `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` | `UTMASTG` |
| `copy_s2t` (scale factors) | `tcgen05.cp.cta_group::2.32x128b.warpx4` | UTCCP sequence |
| `gemm(...)` routed **and** shared | `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale` | `UTCHMMA` |
| `copy_t2r` | two `tcgen05.ld.sync.aligned.16x256b.x1.b32` | `LDTM` |
| `copy_r2s(..., transpose=True)` FP8 / BF16 | `stmatrix.sync.aligned.m16n8.x1.trans.shared.b8` / `stmatrix.sync.aligned.x4.m8n8.shared.b16.trans` | `STS` (transposing) |
| `cast(..., pack=True)` | `cvt.rn.bf16x2.f32`, `cvt.rn.satfinite.e4m3x2.f32` | `F2FP` |
| `exp` / `rcp` under `kFastMath` | `ex2.approx.f32`, `rcp.approx.ftz.f32` | `MUFU` |
| `umma_arrive` | `elect.sync` + `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64` | matrix completion |
| `st_async` (task publish) | `st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4` | `STAS` |
| barrier init / wait / arrive | `mbarrier.init.shared::cta.b64`, `mbarrier.try_wait.parity.shared::cta.b64`, `mapa.shared::cluster.u32` + `mbarrier.arrive[.expect_tx].shared::cluster.b64` | `BSSY`/`SYNC` plus `NANOSLEEP` in the spin |
| `nvlink_barrier` signal / spin | `red.release.sys.global.add.s32` + `ld.acquire.sys.global.s32` spin |
| `red_add_rel` / `red_add` / `atomic_add` | `red.release.gpu.global.add.u32`, `red.gpu.global.add.{u32,s32}`, `atom.global.add.{u32,u64}`, `atom.sys.global.add.u64`, `atom.shared.cta.add.u32` | `RED` / `ATOM` / `ATOMS` |
| `grid_sync` | `atom.release.gpu.global.add.u32` + `ld.acquire.gpu.global.b32` spin, bracketed by the caller's `bar.sync <id>, N` — `0, kNumDispatchThreads` on the dispatch path, `2, kNumEpilogueThreads` on the epilogue path | `RED` + `LDG` spin |
| `peer(ptr, rank)` | `add.s64` on the by-value offset table | address arithmetic |
| `tmem_alloc` / `tmem_free` | `tcgen05.alloc` / `tcgen05.dealloc` | TMEM control |
| `barrier(barrier_id=n, count)` | `bar.sync n, N` / `barrier.sync n, N` | `BAR` |
| `cluster_sync` | `barrier.cluster.arrive.relaxed.aligned` + `barrier.cluster.wait.aligned` | `BAR` (cluster) |
| lane-distributed descriptor tables | `shfl.sync.idx.b32` | `SHFL` |
| `warp_reduce_add` / `warp_reduce_min` | `redux.sync.{add,min}.u32` | `REDUX` |
| `ballot` / `popc` / `ffs` / `fns` | `vote.sync.ballot.b32`, `popc.b32`, `bfind`, `fns.b32` | `VOTE` / `FLO` |

Per task and K block, the leader CTA's UMMA warp issues
`BLOCK_K / UMMA_K` `tcgen05.mma` instructions plus `SF_BLOCK_M / 128` and
`SF_BLOCK_N / 128` UTCCP copies per `UMMA_BLOCK_K`. The A-load warp issues one A
box and one SFA box per K block; the B-load warp issues one B box and one SFB
box, whose expect-tx byte counts differ between the routed (FP4, halved) and
shared (FP8, full) branches. The L1 epilogue performs
`WG_BLOCK_M / STORE_BLOCK_M` store stages of `kNumAtomsPerStore` atoms each; the
L2 epilogue performs the same stage count with `STORE_BLOCK_M / 8` NVLink row
writes per stage. The combine reduction performs one 1-D TMA load per active
slot per chunk — `K` slots for a routed-only token and `K + 1` when `S > 0`.

`SYNC` and `NANOSLEEP` counts come from the `mbarrier` spin loops and are not
directly controllable; they are not an alignment target.
