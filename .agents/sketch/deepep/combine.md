<!--
This file is a design sketch for a TIRx port of code from DeepEP
(https://github.com/deepseek-ai/DeepEP @ 01dc3aa), Copyright (c) 2025 DeepSeek.
It documents deep_ep/include/deep_ep/impls/combine.cuh and
   deep_ep/include/deep_ep/impls/combine_reduce_epilogue.cuh; see NOTICE and licenses/ for upstream attribution.
SPDX-License-Identifier: Apache-2.0 AND MIT
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# DeepEP elastic combine (single-domain NVLink): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new
IR, a builder API, or a mathematical reference implementation. Its purpose is
to show the two CUDA kernels as:

- an explicit runtime ABI and launch;
- explicit GMEM, SMEM, and register tiles, including byte offsets;
- the uniform copy-warp (kernel 1) and reduce-warp (kernel 2) role structure
  and the source-order control flow;
- primitive directional copies and primitive computation inside every role;
- explicit synchronization edges: mbarriers, NVLink barriers, grid-wide
  barriers, and the PDL chaining edge;
- hardware instruction selection stated per key operation.

The implementation represented by this sketch is maintained in
[`tirx_kernels/deepep/combine.py`](../../tirx_kernels/deepep/combine.py).
That module is the source of truth.

**In scope.** `combine_impl` instantiated with `kIsScaleupNVLink=true`
(`team_t = ncclTeamTagLsa`), `kUseExpandedLayout=false`,
`kAllowMultipleReduction=true`, `kNumScaleoutRanks=1` — the backward partner of
the non-expand dispatch port; and `combine_reduce_epilogue_impl` instantiated
with the same booleans and `kNumScaleoutRanks=1`. Derived:
`kUseRankLayout=false` (8 ranks > 6 top-k), `kNumTokensInLayout=kNumTopk=6`,
`kDoExpandedSend=false`. The two kernels form one PDL-chained pair, ported
together.

**Out of scope.** `hybrid_combine.cuh` (multi-node); every RDMA/GIN `gin.put`
branch (dead in this specialization: `nvlink_bypass` is always true inside the
LSA domain); `kUseExpandedLayout=true` (the kernel-1 local-reduction branch at
combine.cuh:144-176 and the expanded-send loop at combine.cuh:177-213 require
it; both are dead here); `bias_0`/`bias_1` (nullptr; the `add_bias` prologue in
`combine_reduce` is dead); FP8/SF variants; `do_cpu_sync` host-count paths;
linked-list/channel metadata (scaleout-only); tile (`Tx`) primitives.

## Multiple-reduction semantics (why dedup-by-rank is correct)

The dispatch kernel dedups by destination rank (dispatch.cuh:344: only the
master lane of each rank group allocates a sender-counter slot; the torch
reference does the same, refs.py:67 `.any(dim=1)`), so each (token, rank)
group produces exactly ONE received token, no matter how many of the token's
experts live on that rank. `src_metadata[i][1]` written by the dispatch
epilogue is `src_rank * kNumTopk + m`, where `m` is that group's MASTER
(highest) top-k lane — the group's unique identity. Every remote rank-buffer
slot is therefore written exactly once; there is no same-slot race. The
caller-side expert computation is expected to produce rank-group pre-reduced
outputs: `x[i]` already holds the sum of this rank group's expert outputs
(tests build it as the masked in-order accumulate over this rank's lanes).
Kernel 2 then dedups by dst rank (each group contributes once, through its
master lane's rank buffer) and sums the groups. The `no_local_reduce` path is
the only reachable path in this specialization (combine.cuh:126-127:
`not kUseExpandedLayout` is constant-true).

## Sanctioned substitutions (ABI adaptations, not algorithm changes)

Same three substitutions as the dispatch port (see
[dispatch.md](dispatch.md) "Sanctioned substitutions"), restated where used:

1. **Peer pointer table for `NCCLGin` device-side translation.** Here
   `NCCLGin` is used only for `get_sym_ptr` (LSA translation) and
   `red_add_rel_sys` (barrier signals); `is_nvlink_accessible<Lsa>` is
   constant-true and `put` is dead. The port replaces
   `ncclGetLsaPointer(window, offset, rank)` with host-precomputed tables
   `peer_ws_ptrs[8]` and `peer_buf_ptrs[8]` (int64):
   `remote(addr, dst) = peer_table[dst] + (addr - local_base)`.
2. **Software grid barrier for `this_grid().sync()`.** Every grid-sync site in
   `comm::gpu_barrier` becomes the dispatch port's monotonic per-site u64
   counter on dedicated port-scratch workspace slots (arrive via
   `red.release.gpu`, spin via `ld.acquire.gpu`, no reset). Arrive-before-
   proceed semantics and call sites are unchanged.
3. **PDL via TIRx launch tags.** `cudaGridDependencySynchronize` in kernel 2
   maps to `T.ptx.griddepcontrol.wait()` plus the
   `tirx.use_programtic_dependent_launch` kernel-launch tag. Kernel 1 contains
   **no** `cudaTriggerProgrammaticLaunchCompletion` (unlike dispatch kernel 1;
   zero `griddepcontrol` instructions in the kernel-1 PTX): the PDL wait
   releases at kernel-1 completion (implicit trigger), so kernel 1 emits no
   `griddepcontrol.launch_dependents`.

Everything else — warp rotation, contiguous token chunking, TMA load/store
dataflow, the direct NVLink weights store, barrier protocols, dedup/slot
computation, and both reduce dtype paths — is a faithful transcription of the
source.

## Pipeline at a glance

### Kernel 1: `combine_impl`

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0..15 (uniform copy role, rotated by `rank_idx`) | contiguous token chunks (`ceil_div(num_reduced, 64*16)` per warp): read `src_metadata[i][0..1]` (`__ldg`); elect-one TMA-load `x[i]` hidden into the warp's SMEM slot; mbarrier wait; TMA-store hidden into the source rank's rank buffer `src_topk_idx`, slot `src_token_idx` (remote, via peer table); commit; write this token's 6 top-k weights into the same remote slot via direct `st.b32` | per-warp mbarrier for TMA load completion; per-token `tma_store_wait` before SMEM slot reuse; entry NVLink barrier (tag0, kCombineTag0=4) before everything |
| all | exit: TMA flush (commit + wait), grid barrier, SM 0 runs NVLink barrier (tag1, kCombineTag1=5); no PDL trigger | software grid barrier (substitution 2), NVLink barrier |

### Kernel 2: `combine_reduce_epilogue_impl`

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0..15 x 148 SMs (uniform reduce role) | `griddepcontrol.wait`; strided loop over local output tokens: read `combined_topk_idx` (plain ld, post-PDL visibility), compute dst ranks, dedup by rank, `compute_topk_slots`; `combine_reduce` over <= 6 rank buffers: 7 hidden chunks x (4 x int4 `.nc` gez-predicated loads per valid slot; bf16 `hadd` bypass when <= 2 valid slots, else fp32 `add.rn.f32.bf16` accumulate + `cvt.rn.bf16x2.f32`); first chunk drains prior TMA store; `tma_store_fence`; elect-one TMA-store the reduced token to `combined_x`; write `combined_topk_weights` from the master lane's rank-buffer weights slot | PDL edge from kernel 1; per-token `tma_store_wait` before SMEM slot reuse; no mbarrier (kernel 2 only TMA-stores FROM smem) |

## Primitive vocabulary

Structural operations and copy directions are the same as in
[dispatch.md](dispatch.md) ("Primitive vocabulary"):

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
tile(...) / view(...) / slice(...) / reg_tile(...)
copy_g2s(src, dst, completion=mb)   # global -> shared, TMA, mbarrier completion
copy_s2g(src, dst)                  # shared -> global, TMA (dst may be remote)
copy_g2r(src, dst)                  # global -> register (may carry .nc / acquire semantics)
copy_s2r(src, dst)                  # shared -> register
copy_r2s(src, dst)                  # register -> shared
copy_r2g(src, dst)                  # register -> global (may be remote)
fill / add / sub / mul / div_floor / align_up / select
bitwise_and / shift_right
shuffle_index(dst, src, source_lane)
ballot(dst, predicate)
match_any(dst, value)
bfind(dst, mask)                    # highest set bit; 31 - clz
ffs(dst, mask)                      # lowest set bit index + 1; 0 when mask == 0
```

Computation additions for the reduce body:

```python
vec_g2r_16B(src, predicate)         # one gez-predicated int4 (16 B) vector load per lane
add_bf162(dst, lhs, rhs)            # pairwise bf16x2 adds on int4-views (hadd path)
accum_bf16_f32(dst_f32x2, src_bf16) # fused bf16 -> f32 convert-and-add, per scalar
cast_f32_bf162(dst, src)            # float2 -> bf162 round-to-nearest pair cast
```

`mbarrier_init`, `expect_tx`, `mbarrier_wait`, `tma_store_commit`,
`tma_store_wait`, `tma_store_fence`, `grid_barrier`, `nvlink_barrier`,
`pdl_wait`, `syncwarp`, `named_barrier`, `red_global`, `clock`, `timeout_trap`
are schedule operations, as in the dispatch sketch. Scalar address arithmetic,
pointer offsets, and guards are shown directly; they do not hide copies,
computation, role changes, or synchronization.

There are deliberately no computational primitives named `dispatch`, `combine`,
`reduce_all`, `barrier_all`, `gin`, or `nccl`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

k1_variant = specialize(
    kIsScaleupNVLink=True,              # team_t = ncclTeamTagLsa; RDMA branches dead
    kUseExpandedLayout=False,           # non-expand dispatch handle (do_expand=false)
    kAllowMultipleReduction=True,       # test/default recipe (reduce_in_local)
    kNumSMs=...,                        # auto: get_theoretical_num_sms; 64 for e256/k6
    kNumWarps=16,                       # min(kNumSmemBytes // 14432, 32)
    kNumRanks=8,                        # = kNumScaleupRanks; kNumScaleoutRanks = 1
    kHidden=7168,                       # bf16; kNumHiddenBytes = 14336
    kNumMaxTokensPerRank=...,           # config: 128 / 1024 / 4096
    kNumExperts=256, kNumTopk=6,
    target="sm_100a",
)
k2_variant = specialize(
    kUseExpandedLayout=False, kAllowMultipleReduction=True,
    kNumSMs=...,                        # FULL device SMs: 148 on B200
                                        # (buffer.hpp: launch_combine_reduce_epilogue
                                        #  passes device_runtime->get_num_sms())
    kNumWarps=16,                       # min(kNumSmemBytes // 14336, 32)
    kNumScaleoutRanks=1, kNumScaleupRanks=8,
    kHidden=7168, kNumMaxTokensPerRank=...,
    kNumExperts=256, kNumTopk=6,
    target="sm_100a",
)
# instruction_selection: none; extent: the template argument lists of
#   combine_impl / combine_reduce_epilogue_impl for the direct path

# Compile-time derived constants (no emitted code)
kNumExpertsPerRank   = 32               # kNumExperts / kNumRanks
kUseRankLayout       = False            # kAllowMultipleReduction and kNumRanks <= kNumTopk -> 8 <= 6 is False
kNumTokensInLayout   = 6                # = kNumTopk when not kUseRankLayout
kCombineTokenBytes   = 14400            # align(14336,32) + align(0,32) + align(6*4+6*4,32)
kCombineTokenSmem    = 14432            # + align(8,32) mbarrier area (kernel 1 only)
kOutputTokenBytes    = 14336            # TokenLayout(14336, 0, 0, false): flat hidden
kHiddenVec           = 896              # 14336 / sizeof(int4)
kUnrollFactor        = 4                # get_max_unroll_factor<896, 4>: 896 % (32*4) == 0
kHiddenChunks        = 7                # 896 / (4*32)
kNumSmemBytes        = 232448           # max opt-in dynamic smem per block on SM100
kRecvRegionBytes     = kNumTokensInLayout * kNumMaxTokensPerRank * kCombineTokenBytes
# Workspace byte offsets (subset of dispatch.md's table; barrier-only usage)
WS_BARRIER_COUNTER   = 0                # u64; phase bit 0, sign bit 1
WS_BARRIER_SIGNAL    = 8                # i32[2]; slot phase at (2+phase)*4
WS_PORT_SCRATCH      = 12820528         # substitution 2: software grid-barrier slots
# Barrier tags (comm.cuh:18-21)
kCombineTag0         = 4                # entry barrier
kCombineTag1         = 5                # exit barrier
# instruction_selection: none; extent: compile-time constants only

k1_launch = launch(
    grid=(kNumSMs, 1, 1),
    cluster=(2 - kNumSMs % 2, 1, 1),    # 2 for even kNumSMs (combine.hpp:173-174)
    block=(kNumWarps * 32, 1, 1),       # 512
    min_blocks_per_sm=1,
    dynamic_smem_bytes=kNumSmemBytes,
    cooperative=True,                   # source only, for this_grid().sync();
                                        # realized via substitution 2
    programmatic_dependent_launch=False,
)
# instruction_selection: none; extent: static launch metadata;
#   `__launch_bounds__(512, 1)`; cudaLaunchKernelEx cluster attribute

k2_launch = launch(
    grid=(kNumSMs_k2, 1, 1),            # kNumSMs_k2 = full device SM count (148)
    cluster=(1, 1, 1),
    block=(kNumWarps * 32, 1, 1),       # 512
    min_blocks_per_sm=1,
    dynamic_smem_bytes=kNumSmemBytes,
    programmatic_dependent_launch=True, # waits on kernel 1 completion (implicit trigger)
)
# instruction_selection: none; extent: static launch metadata;
#   `__launch_bounds__(512, 1)`

# ---------------------------------------------------------------------------
# Shared barrier helpers (identical to dispatch.md; restated for tags 4/5)
# ---------------------------------------------------------------------------

def nvlink_barrier(workspace, peer_ws_ptrs, rank_idx, sm_idx, thread_idx, tag):
    # comm.cuh:88-129 nvlink_barrier_wo_local_sync; SM 0 only
    if sm_idx == 0:
        status = copy_g2r(workspace + WS_BARRIER_COUNTER, reg_tile([], "u64")) & 3
        # instruction_selection: ld.global.b32 (plain C++ deref of the counter's
        #   low word; comm.cuh:98) + bitwise; extent: SM 0
        phase, sign = status & 1, status >> 1
        if thread_idx < kNumRanks:
            red_global(remote(peer_ws_ptrs, workspace, thread_idx)
                           + WS_BARRIER_SIGNAL + phase * 4,
                       select(sign == 0, 1, -1), scope="sys", type="s32")
            # instruction_selection: red.release.sys.global.add.s32;
            #   extent: 8 lanes of SM 0
        named_barrier(idx=0, threads=kNumWarps * 32)
        # instruction_selection: bar.sync 0, 512 (__syncthreads)
        if thread_idx == 0:
            atomic_add_gmem(workspace + WS_BARRIER_COUNTER, 1)
            # instruction_selection: atom.add.u64; extent: scalar
            target = select(sign == 0, kNumRanks, 0)
            timeout_loop():
                if copy_g2r(workspace + WS_BARRIER_SIGNAL + phase * 4,
                            reg_tile([], "i32"), semantic="acquire_sys") == target:
                    break
                # instruction_selection: ld.acquire.sys.L1::no_allocate.global.u32
                #   spin (ptx::ld_acquire_sys); extent: SM 0 thread 0;
                #   timeout_trap on kNumTimeoutCycles

def grid_barrier(workspace, site):
    # substitution 2 for this_grid().sync(): monotonic per-site u64 counter
    counter = workspace + WS_PORT_SCRATCH + site * 8
    if thread_idx == 0:
        c0 = copy_g2r(counter, reg_tile([], "u64"), semantic="acquire_gpu")
        target = (c0 // kNumSMs_grid + 1) * kNumSMs_grid
        red_global(counter, 1, scope="gpu", type="u64")
        # instruction_selection: red.release.gpu.global.add.u64; extent: one per CTA
        while copy_g2r(counter, reg_tile([], "u64"), semantic="acquire_gpu") < target:
            pass
        # instruction_selection: ld.acquire.gpu.u64 spin; extent: CTA thread 0
    named_barrier(idx=0, threads=kNumWarps * 32)
    # instruction_selection: bar.sync 0, 512

# ===========================================================================
# Kernel 1: combine_impl
# ===========================================================================

def combine_impl(
    x,                  # bf16* [num_reduced_tokens, kHidden]     (expert outputs)
    topk_weights,       # f32*  [num_reduced_tokens, kNumTopk]
    src_metadata,       # i32*  [num_reduced_tokens, 2 + kNumTopk]
    psum_rank,          # i32*  [kNumRanks] (dispatch's inclusive prefix)
    peer_ws_ptrs,       # i64*  [kNumRanks]  (substitution 1)
    peer_buf_ptrs,      # i64*  [kNumRanks]  (substitution 1)
    buffer,             # u8*   local combine recv-region base (6 rank buffers)
    workspace,          # u8*   local window base
    num_reduced_tokens, # i32   host value: kNumMaxTokensPerRank * kNumRanks (worst case)
    rank_idx,           # i32   0..7
):
    sm_idx = block_id()
    thread_idx = thread_id()
    # Rotated warp index (combine.cuh:39): staggers per-rank channel usage
    warp = (shuffle_index(thread_idx // 32, source_lane=0) + rank_idx) % kNumWarps
    # instruction_selection: mov.u32 %tid.x, shr.u32, shfl.sync.idx.b32,
    #   add + rem (by 16 -> and); extent: warp-uniform scalar
    lane = lane_id()
    global_warp_idx = warp * kNumSMs + sm_idx

    # Real received-token count from the GPU prefix (combine.cuh:45-46);
    # kernel 1 has no PDL edge, so `__ldg` is legal here
    if num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks:
        num_reduced_tokens = copy_g2r(psum_rank[kNumRanks - 1], reg_tile([], "i32"),
                                      semantic="nc")
        # instruction_selection: ld.global.nc.s32 (__ldg); extent: warp-uniform

    smem = tile("shared", "u8", [kNumSmemBytes], byte_offset=0, alignment=1024)
    tma_buffer = view(smem, "u8", [kCombineTokenSmem],
                      byte_offset=warp * kCombineTokenSmem)
    tma_hidden = view(tma_buffer, "u8", [14336], byte_offset=0)
    tma_mbar   = view(tma_buffer, "u64", [1], byte_offset=kCombineTokenBytes)  # @14400

    phase = fill(reg_tile([], "u32"), 0)
    if elect_one():
        mbarrier_init(tma_mbar, arrive_count=1)
        # instruction_selection: mbarrier.init.shared::cta.b64 +
        #   fence.mbarrier_init.release.cluster; extent: one elected lane
    syncwarp()

    # (kDoExpandedSend=false: topk_weights==nullptr assert at combine.cuh:68-69 dead)

    # Entry barrier (combine.cuh:77-80):
    # gpu_barrier<kCombineTag0, kFlushStores=false, kSyncAtStart=false, kSyncAtEnd=true>
    nvlink_barrier(workspace, peer_ws_ptrs, rank_idx, sm_idx, thread_idx,
                   tag=kCombineTag0)
    grid_barrier(workspace, site=0)     # kSyncAtEnd=true (substitution 2)

    # Contiguous token chunks per warp (combine.cuh:83-85: NOT a strided loop)
    num_tokens_per_warp = ceil_div(num_reduced_tokens, kNumSMs * kNumWarps)
    token_start = num_tokens_per_warp * global_warp_idx
    token_end = min(token_start + num_tokens_per_warp, num_reduced_tokens)
    for i in range(token_start, token_end):
        # Source routing indices (combine.cuh:88-92)
        src_token_idx = copy_g2r(src_metadata[i * 8 + 0], reg_tile([], "i32"),
                                 semantic="nc") % kNumMaxTokensPerRank
        # instruction_selection: ld.global.nc.s32 (__ldg) + rem; extent: per thread
        src_rank_topk = copy_g2r(src_metadata[i * 8 + 1], reg_tile([], "i32"),
                                 semantic="nc")
        # instruction_selection: ld.global.nc.s32 (__ldg); extent: per thread
        src_rank_idx = div_floor(src_rank_topk, kNumTopk)
        src_topk_idx = src_rank_topk % kNumTopk
        # instruction_selection: s32 div/mod by 6; extent: scalar

        # nvlink_bypass = is_nvlink_accessible<Lsa>(src_rank_idx): constant-true here;
        # master slot is on the SOURCE rank's combine recv region (combine.cuh:96-106)
        master_slot = remote(peer_buf_ptrs, buffer, src_rank_idx) \
                      + (src_topk_idx * kNumMaxTokensPerRank + src_token_idx) \
                        * kCombineTokenBytes

        # (kUseExpandedLayout=false: stored_topk_slot_idx reads at
        #  combine.cuh:115-119 dead; reduce_valid_mask empty;
        #  no_local_reduce = true (combine.cuh:125-127), always the simple path)
        if elect_one():
            # Drain the previous token's TMA store before reusing the SMEM slot
            # (combine.cuh:136, inside the elected lane in source order)
            tma_store_wait(remaining=0)
            # instruction_selection: cp.async.bulk.wait_group 0;
            #   extent: one elected lane
            copy_g2s(x + i * 14336, tma_hidden, completion=tma_mbar, num_bytes=14336)
            # instruction_selection: cp.async.bulk.shared::cluster.global.
            #   mbarrier::complete_tx::bytes.L2::cache_hint (evict-first);
            #   extent: one 14336-byte 1-D bulk copy
            expect_tx(tma_mbar, 14336)
            # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64
            mbarrier_wait(tma_mbar, phase)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
            #   extent: one elected lane, phase flips
            copy_s2g(tma_hidden, master_slot, num_bytes=14336)
            # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group.
            #   L2::cache_hint (evict-normal); extent: one 14336-byte 1-D bulk copy
            #   to a PEER (NVLink) address
            tma_store_commit()
            # instruction_selection: cp.async.bulk.commit_group;
            #   extent: one elected lane
        syncwarp()

        # (kAllowMultipleReduction local-reduce branch combine.cuh:144-176: dead,
        #  requires kUseExpandedLayout; expanded-send loop combine.cuh:177-213: dead)

        # Write this token's top-k weights into the same remote slot (combine.cuh:216-226)
        if lane < kNumTopk:
            w = copy_g2r(topk_weights[i * kNumTopk + lane], reg_tile([], "f32"),
                         semantic="nc")
            # instruction_selection: ld.global.nc.f32 (__ldg) predicated lane < 6;
            #   extent: scalar per lane
            copy_r2g(w, master_slot + 14360 + lane * 4)
            # instruction_selection: st.b32 (generic space; the peer address is
            #   runtime-computed, so NVCC cannot prove .global; f32 payload
            #   carried as b32; TIRx's handle-typed peer pointer lowers to
            #   st.global.b32, a same-address refinement); extent: <= 6 lanes
        syncwarp()

        # (nvlink_bypass constant-true: RDMA send block combine.cuh:230-236 dead)

    # Exit barrier (combine.cuh:239-242):
    # gpu_barrier<kCombineTag1, kFlushStores=true, kSyncAtStart=true, kSyncAtEnd=false>
    tma_store_commit()
    # instruction_selection: cp.async.bulk.commit_group; extent: per thread (kFlushStores)
    tma_store_wait(remaining=0)
    # instruction_selection: cp.async.bulk.wait_group 0; extent: per thread
    syncwarp()
    grid_barrier(workspace, site=1)     # kSyncAtStart=true (substitution 2)
    nvlink_barrier(workspace, peer_ws_ptrs, rank_idx, sm_idx, thread_idx,
                   tag=kCombineTag1)
    # kSyncAtEnd=false: non-SM-0 CTAs proceed; kernel 1 carries no PDL trigger,
    # so kernel 2's griddepcontrol.wait releases at kernel-1 completion

# ===========================================================================
# Kernel 2: combine_reduce_epilogue_impl
# ===========================================================================

def combine_reduce_epilogue_impl(
    combined_x,             # bf16* [num_combined_tokens, kHidden]  (out)
    combined_topk_weights,  # f32*  [num_combined_tokens, kNumTopk] (out)
    combined_topk_idx,      # i64*  [num_combined_tokens, kNumTopk] (original top-k)
    buffer,                 # u8*   local combine recv-region base (6 rank buffers)
    num_combined_tokens,    # i32   host-known local token count
    rank_idx,               # i32   0..7 (scaleup_rank_idx; scaleout is 0)
):
    sm_idx = block_id()
    warp = shuffle_index(thread_id() // 32, source_lane=0)
    lane = lane_id()
    global_warp_idx = warp * kNumSMs_k2 + sm_idx
    # (epilogue.cuh:39 comment: warp-major index concentrates the last wave on SMs)

    smem = tile("shared", "u8", [kNumSmemBytes], byte_offset=0, alignment=1024)
    tma_buffer = view(smem, "u8", [kOutputTokenBytes],
                      byte_offset=warp * kOutputTokenBytes)
    # (BufferLayout<false>: no mbarrier — kernel 2 never TMA-loads into smem)

    # Block until kernel 1 finished and all peer data are visible (epilogue.cuh:59)
    pdl_wait()
    # instruction_selection: griddepcontrol.wait; extent: every CTA;
    #   PDL edge: completes at kernel-1 completion (no explicit trigger)

    for token_idx in range(global_warp_idx, num_combined_tokens,
                           kNumWarps * kNumSMs_k2):
        # Dst expert/rank per top-k lane (epilogue.cuh:66-71; plain ld, NOT __ldg:
        # "PDL is used, please do not use __ldg"; the i64 -> int truncation reads
        # only the little-endian low word)
        dst_expert = select(lane < kNumTopk,
                            i32(copy_g2r(combined_topk_idx[token_idx * kNumTopk + lane],
                                         reg_tile([], "i32"))), -1)
        # instruction_selection: ld.global.b32 (plain; low word of the i64 entry,
        #   no explicit conversion emitted) predicated lane < 6;
        #   extent: scalar per lane
        dst_rank = select(dst_expert >= 0, div_floor(dst_expert, kNumExpertsPerRank), -1)
        # instruction_selection: s32 div by 32 -> shr.s32 5; extent: scalar
        syncwarp()

        # Dedup on dst rank (epilogue.cuh:82-84 else-branch, scaleout == 1):
        # each rank group contributes once, through its master lane
        is_group_master = (bfind(match_any(dst_rank)) == lane)
        # instruction_selection: match.any.sync.b32 + bfind.u32 + setp;
        #   extent: warp-wide (ptx::deduplicate)
        reduce_valid_mask = ballot(is_group_master and dst_rank >= 0)
        # instruction_selection: vote.sync.ballot.b32; extent: warp-wide

        # Sort valid top-k slots to front (combine_utils.cuh:43-53; fetch = identity
        # because kUseRankLayout=false: the slot index IS the rank-buffer index)
        topk_slot_idx = reg_tile([kNumTokensInLayout], "i32")
        mask = reduce_valid_mask
        for k in range(kNumTokensInLayout):         # #pragma unroll
            lowest = ffs(mask) - 1
            topk_slot_idx[k] = select(lowest >= 0, lowest, -1)
            # instruction_selection: ffs.b32 + select; extent: warp-uniform;
            #   exchange is identity here (epilogue.cuh:93, kUseRankLayout=false)
            mask = bitwise_and(mask, mask - 1)

        # ------------------------------------------------------------------
        # combine_reduce<896, 4, 6, 6> (combine_utils.cuh:57-170):
        # sum the valid rank buffers' token slots into the smem staging buffer
        # ------------------------------------------------------------------
        # enable_hadd_bypass (utils.cuh:68-70): no bias in this specialization, so
        # the path selector is exactly (topk_slot_idx[2] < 0), i.e. <= 2 valid slots
        hadd_bypass = topk_slot_idx[2] < 0

        if hadd_bypass:
            # --- bf16 hadd path (utils.cuh:73-110) ---
            for c in range(kHiddenChunks):          # #pragma unroll 1 (serial)
                # 4 x int4 gez-predicated reads per slot (slots 0 and 1)
                values_0 = reg_tile([kUnrollFactor], "int4")
                values_1 = reg_tile([kUnrollFactor], "int4")
                base_0 = buffer + (topk_slot_idx[0] * kNumMaxTokensPerRank + token_idx) \
                                  * kCombineTokenBytes
                base_1 = buffer + (topk_slot_idx[1] * kNumMaxTokensPerRank + token_idx) \
                                  * kCombineTokenBytes
                for j in range(kUnrollFactor):      # #pragma unroll
                    values_0[j] = vec_g2r_16B(
                        base_0 + (c * 128 + j * 32 + lane) * 16, predicate=topk_slot_idx[0])
                    values_1[j] = vec_g2r_16B(
                        base_1 + (c * 128 + j * 32 + lane) * 16, predicate=topk_slot_idx[1])
                # instruction_selection: setp.ge.s32 + @p
                #   ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32 (evict-first),
                #   zero-filled when the predicate is false (ptx::ldg_with_gez_pred);
                #   extent: 4 x 16 B per slot per lane per chunk, LOCAL window
                #   addresses; slot -1 computes a wraparound address never loaded
                if c == 0:
                    # Drain the previous token's TMA store before reusing smem
                    # (utils.cuh:96-98 wait_buffer_func)
                    tma_store_wait(remaining=0)
                    # instruction_selection: cp.async.bulk.wait_group 0;
                    #   extent: warp-wide
                    syncwarp()
                for j in range(kUnrollFactor):      # #pragma unroll
                    values_0[j] = add_bf162(values_0[j], values_1[j])
                    # instruction_selection: add.bf16x2 x4 per int4 (PTX omits the
                    #   default .rn), 16 per chunk; extent: 4 x int4 per lane
                    copy_r2s(values_0[j], tma_buffer + (c * 128 + j * 32 + lane) * 16)
                    # instruction_selection: st.shared.v4.b32; extent: 16 B per lane
        else:
            # --- fp32 accumulate path (utils.cuh:111-168) ---
            for c in range(kHiddenChunks):          # #pragma unroll 1 (serial)
                reduced = fill(reg_tile([kUnrollFactor * 4], "f32x2"), 0.0)
                # (bias_0/bias_1 == nullptr: add_bias prologue utils.cuh:129-130 dead)
                for k in range(kNumTokensInLayout): # #pragma unroll
                    # (k >= kNumExpectedTopk(6) never true: no early break, utils.cuh:134)
                    base_k = buffer + (topk_slot_idx[k] * kNumMaxTokensPerRank
                                       + token_idx) * kCombineTokenBytes
                    values = reg_tile([kUnrollFactor], "int4")
                    for j in range(kUnrollFactor):  # #pragma unroll
                        values[j] = vec_g2r_16B(
                            base_k + (c * 128 + j * 32 + lane) * 16,
                            predicate=topk_slot_idx[k])
                    # instruction_selection: setp.ge.s32 + @p
                    #   ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32
                    #   (evict-first); extent: 4 x 16 B per slot per lane per chunk
                    for j in range(kUnrollFactor * 4):  # #pragma unroll
                        reduced[j] = accum_bf16_f32(reduced[j], values bf16 pair j)
                    # instruction_selection: add.rn.f32.bf16 x8 per int4
                    #   (ptx::accumulate, SM100 fused cast+add); extent: 32 scalars/lane
                if c == 0:
                    tma_store_wait(remaining=0)
                    # instruction_selection: cp.async.bulk.wait_group 0;
                    #   extent: warp-wide
                    syncwarp()
                for j in range(kUnrollFactor):      # #pragma unroll
                    copy_r2s(cast_f32_bf162(reduced[j*4:j*4+4]),
                             tma_buffer + (c * 128 + j * 32 + lane) * 16)
                    # instruction_selection: cvt.rn.bf16x2.f32 x4
                    #   (__float22bfloat162_rn) + st.shared.v4.b32; extent: 16 B per lane

        # Async-proxy fence so the TMA engine sees the smem reduce output
        # (epilogue.cuh:116)
        tma_store_fence()
        # instruction_selection: fence.proxy.async.shared::cta; extent: warp-wide
        syncwarp()

        # TMA-store the reduced token (epilogue.cuh:120-123)
        if elect_one():
            copy_s2g(tma_buffer, combined_x + token_idx * kOutputTokenBytes,
                     num_bytes=kOutputTokenBytes)
            # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group.
            #   L2::cache_hint (evict-normal); extent: one 14336-byte 1-D bulk copy
            tma_store_commit()
            # instruction_selection: cp.async.bulk.commit_group
        syncwarp()

        # Write combined top-k weights from the master lane's rank buffer
        # (epilogue.cuh:128-141); every written slot carries the full 6-weight
        # array, so any valid group buffer serves
        weight_master_lane = bfind(match_any(dst_rank))
        # instruction_selection: match.any.sync.b32 + bfind.u32; extent: warp-wide
        if lane < kNumTopk:
            w = select(dst_rank >= 0,
                       copy_g2r(buffer + (weight_master_lane * kNumMaxTokensPerRank
                                          + token_idx) * kCombineTokenBytes
                                + 14360 + lane * 4, reg_tile([], "f32")),
                       0.0)
            # instruction_selection: ld.global.b32 (plain; post-PDL visibility)
            #   predicated on dst_rank >= 0; extent: scalar per lane
            copy_r2g(w, combined_topk_weights[token_idx * kNumTopk + lane])
            # instruction_selection: st.global.b32; extent: scalar per lane
        syncwarp()
```

## Kernel-specific tables

### Combine token slot layout (`TokenLayout(14336, 0, 6, false)`)

| region | byte offset | dtype | extent | writer -> consumer |
| --- | --- | --- | --- | --- |
| hidden | 0 | bf16 | 14336 B | peer K1 TMA store -> K2 reduce reads |
| topk_idx (dead) | 14336 | i32 | 24 B | never written/read in this specialization |
| topk_weights | 14360 | f32 | 24 B | peer K1 `st.b32` -> K2 weights read |
| padding | 14384 | - | 16 B | align(48, 32) = 64 |
| (smem only) mbarrier | 14400 | u64 | 8 B | K1 TMA load completion (smem slot 14432 B) |

### Reduce dtype paths (`combine_reduce`, combine_utils.cuh:57-170)

| selector | path | loads | arithmetic | store |
| --- | --- | --- | --- | --- |
| `topk_slot_idx[2] < 0` (<= 2 valid) | hadd bypass | 2 slots x 4 x int4 `.nc` gez-pred per chunk | `add.bf16x2` x4 per int4 (16/chunk) | `st.shared.v4.b32` |
| else | fp32 accumulate | 6 slots x 4 x int4 `.nc` gez-pred per chunk | `add.rn.f32.bf16` x8 per int4 into f32x2 | `cvt.rn.bf16x2.f32` + `st.shared.v4.b32` |

Both paths: 7 serial hidden chunks (`#pragma unroll 1`), first chunk drains the
prior TMA store (`cp.async.bulk.wait_group 0` + `bar.warp.sync`), identical
chunk/lane iteration order, so the fp32 path's accumulation order matches the
reference's in-order accumulate bit-for-bit; the 2-operand hadd path rounds the
exact sum once, identically to fp32-accumulate-then-cast.

### Workspace regions used by this specialization

| region | byte offset | dtype | writer -> consumer |
| --- | --- | --- | --- |
| nvl_barrier_counter | 0 | u64 | SM-0 thread-0 `atom.add.u64` -> phase/sign reader |
| nvl_barrier_signal[phase] | 8 + phase*4 | i32 | peers' `red.release.sys` -> SM-0 `ld.acquire.sys` spin |
| port grid-barrier sites 0/1 | 12820528 + site*8 | u64 | every CTA `red.release.gpu` -> all CTAs `ld.acquire.gpu` spin (substitution 2) |

### TIRx module and benchmark contract

- Module: `tirx_kernels/deepep/combine.py`, registry name `deepep_combine`.
- Correctness: `python -m tirx_kernels.test --kernel deepep_combine`, 8 ranks,
  4 configs (t128 / t4096 / t1024 masked 0.3 / t1024 align128), reference
  `ElasticBuffer.combine` on identical group-sum inputs; compare `combined_x`
  and `combined_topk_weights` bitwise.
- Benchmark: `tirx_kernels/bench_suite/config/deepep/deepep_combine.yaml`
  (`num_gpus: 8`, `timer: kineto`, `default: false`), config
  `t4096_h7168_e256_k6`; gate `source_time / tirx_time > 0.99`.

## Instruction-selection summary

- Placement selects the loop shape: kernel 1 chunks tokens contiguously across
  64 x 16 rank-rotated warps; kernel 2 strides tokens across 148 x 16 warps.
- All bulk hidden movement is 1-D `cp.async.bulk` with mbarrier completion
  (loads, evict-first) or bulk-group commit (stores, evict-normal); peer stores
  target NVLink-translated addresses from the sanctioned pointer table.
- The only non-TMA cross-rank traffic is the 6 x generic-space `st.b32` weights
  write per token and the barrier's `red.release.sys` signals.
- The reduce body is register-vectorized `.nc` int4 loads with gez predication
  (invalid slots load nothing and read as zero), bf16x2 adds on the <= 2-slot
  path, fused `add.rn.f32.bf16` accumulation + `cvt.rn.bf16x2.f32` on the
  wider path, and a single `fence.proxy.async` before the output TMA store.
- Synchronization is mbarrier parity spin (kernel 1 only), bulk-group
  waits/commits, `bar.warp.sync` between phases, one `bar.sync 0` per CTA at
  each grid-barrier site, and the PDL edge (`griddepcontrol.wait`) that gates
  kernel 2 on kernel-1 completion.
