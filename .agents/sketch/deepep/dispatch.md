<!--
This file is a design sketch for a TIRx port of code from DeepEP
(https://github.com/deepseek-ai/DeepEP @ 01dc3aa), Copyright (c) 2025 DeepSeek.
It documents deep_ep/include/deep_ep/impls/dispatch.cuh and
   deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh; see NOTICE and licenses/ for upstream attribution.
SPDX-License-Identifier: Apache-2.0 AND MIT
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# DeepEP elastic dispatch (single-domain NVLink): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new
IR, a builder API, or a mathematical reference implementation. Its purpose is
to show the two CUDA kernels as:

- an explicit runtime ABI and launch;
- explicit GMEM, SMEM, and register tiles, including byte offsets;
- the notify-warp / dispatch-warp role split and the source-order control flow;
- primitive directional copies and primitive computation inside every role;
- explicit synchronization edges: mbarriers, named barriers, NVLink barriers,
  grid-wide barriers, and the PDL chaining edge;
- hardware instruction selection stated per key operation.

The implementation represented by this sketch is maintained in
[`tirx_kernels/deepep/dispatch.py`](../../tirx_kernels/deepep/dispatch.py).
That module is the source of truth.

**In scope.** `dispatch_impl` instantiated with `kIsScaleupNVLink=true`
(`team_t = ncclTeamTagLsa`), `kDoCPUSync=false`, `kReuseSlotIndices=false`,
`kNumSFPacks=0` (bf16 tokens), `kNumScaleoutRanks=1`, non-expand mode; and
`dispatch_copy_epilogue_impl` instantiated with `kDoExpand=false`,
`kCachedMode=false`, `kDoZeroPadding=false`, `kNumScaleoutRanks=1`
(`kDoCreateLinkedList=false`), `kNumSFPacks=0`. The two kernels form one
PDL-chained pair, ported together.

**Out of scope.** `hybrid_dispatch.cuh` / `hybrid_combine.cuh` (multi-node);
every `not kIsScaleupNVLink` RDMA/GIN branch (compiled out in this
specialization); FP8+SF input (`kNumSFPacks > 0`); cached-handle mode
(`kReuseSlotIndices=true`, zero notify warps); `do_expand` / `do_zero_padding`
layouts; `kDoCPUSync=true` host-workspace writes; `deterministic` sorting;
`cumulative_local_expert_recv_stats` (nullptr in this port); combine kernels;
tile (`Tx`) primitives.

## Sanctioned substitutions (ABI adaptations, not algorithm changes)

TIRx cannot express three source mechanisms literally. Each substitution below
preserves the dataflow and synchronization semantics and is called out inline
at every use site.

1. **Peer pointer table for `NCCLGin` device-side translation.** In this
   specialization `NCCLGin` is used only for `get_sym_ptr` (LSA translation),
   `put_value` (`st.relaxed.sys`), and `red_add_rel_sys`. GIN/RDMA paths are
   compiled out. The port replaces the device-side
   `ncclGetLsaPointer(window, offset, rank)` with host-precomputed tables
   `peer_ws_ptrs[8]` and `peer_buf_ptrs[8]` (int64, from
   `ncclGetLsaDevicePointer`, csrc/kernels/backend/nccl.cu:141-152):
   `remote(addr, dst) = peer_table[dst] + (addr - local_base)`. The
   `nccl_dev_comm` / `nccl_window` / QP-mode plumbing (`comm::get_qp_mode`)
   disappears with it.
2. **Software grid barrier for `this_grid().sync()`.** TIRx has no
   `%envreg`/cooperative-grid-sync PTX. Every `this_grid().sync()` site
   becomes an arrive-and-spin software grid barrier on dedicated port-scratch
   slots inside the aligned workspace tail (unused slack in
   `WorkspaceLayout`'s 2 MB-aligned reservation): last-arriver releases via
   `st.release.gpu`, others spin with `ld.acquire.gpu`. Arrive-before-proceed
   semantics and the exact call sites are unchanged.
3. **PDL via TIRx launch tags.** `cudaTriggerProgrammaticLaunchCompletion` /
   `cudaGridDependencySynchronize` map to
   `T.ptx.griddepcontrol.launch_dependents()` / `.wait()` plus the
   `tirx.use_programtic_dependent_launch` kernel-launch tag (precedent:
   `tirx_kernels/deepgemm/paged_mqa_logits_fp4.py`).

Everything else — warp roles, counting and prefix-sum math, dedup and slot
allocation, TMA load/store dataflow, barrier protocols, timeout policy — is a
faithful transcription of the source.

## Pipeline at a glance

### Kernel 1: `dispatch_impl`

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0..3 (notify) | zero SMEM counts; per-token expert/rank counting with SMEM atomics and rank dedup; full-grid count reduction into workspace; SM 0: wait reduction, exchange counts with all peers (`st.relaxed.sys`), wait peer counts, per-expert reduce + align, warp 0/1 prefix sums | entry NVLink barrier (tag0) before everything; `red.gpu` grid reduction; peer count exchange through workspace recv area |
| 4..(4+kNumDispatchWarps-1) (dispatch) | per-warp channel: strided loop over tokens; TMA-load token hidden into the warp's SMEM slot; write topk metadata; dedup ranks; global-atomic slot allocation; TMA-store the token slot into the destination rank's symmetric recv region | per-warp mbarrier for TMA load completion; `red.release.sys` slot counters; exit NVLink barrier (tag1) after TMA flush |
| all | exit: TMA store flush, grid barrier, SM 0 runs NVLink barrier (tag1), PDL trigger, SM 0 cleans sender counters | grid barrier (software), NVLink barrier, `griddepcontrol.launch_dependents` |

### Kernel 2: `dispatch_copy_epilogue_impl`

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0..(kNumWarps-1) | resolve `num_recv_tokens` from the on-GPU prefix sum; strided loop over received tokens: locate source rank via prefix walk, TMA-load the token slot from the local symmetric recv region, read/localize topk indices (dedup assert, master lane), store recv_topk_idx, TMA-store hidden to recv_x, store topk weights, write recv_src_metadata | `griddepcontrol.wait` at kernel entry (PDL edge); per-warp mbarrier for TMA load completion |

## Primitive vocabulary

Structural operations do not compute values:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
tile(...)             # declare storage, dtype, logical shape, and placement
view(...)             # change logical indexing without moving values
slice(...)            # select a logical interval
reg_tile(...)         # declare a role-local register tile
```

Copies always state their storage direction. `remote` marks a peer-GPU GMEM
address formed through the sanctioned pointer-table substitution:

```python
copy_g2s(src, dst, completion=mb)   # global -> shared, TMA, mbarrier completion
copy_s2g(src, dst)                  # shared -> global, TMA (dst may be remote)
copy_g2r(src, dst)                  # global -> register (may carry .nc / volatile / acquire semantics)
copy_s2r(src, dst)                  # shared -> register
copy_r2s(src, dst)                  # register -> shared
copy_r2g(src, dst)                  # register -> global (may be remote, may carry .relaxed.sys)
```

The complete computational vocabulary used below is:

```python
fill(dst, value)
add(dst, lhs, rhs)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs)
div_floor(dst, lhs, rhs)
align_up(dst, value, granularity)
encode_positive(dst, value)        # -value - 1 (involution; 0 decodes to "not ready")
is_ready(predicate, encoded)       # encode_positive(encoded) >= 0
select(dst, predicate, true_value, false_value)
bitwise_and / bitwise_or / bitwise_xor / shift_right
atomic_add_block(dst, value)       # shared-memory atomic add
atomic_add_gmem(dst, value)        # global atomic add, returns old
shuffle_index(dst, src, source_lane)
shuffle_up(dst, src, delta)
ballot(dst, predicate)
match_any(dst, value)
bfind(dst, mask)                   # highest set bit; 31 - clz
reduce_add(dst, value)             # warp reduce-add
```

`mbarrier_init`, `expect_tx`, `mbarrier_wait`, `named_barrier`,
`tma_store_commit`, `tma_store_wait`, `tma_store_fence`, `grid_barrier`,
`nvlink_barrier`, `pdl_trigger`, `pdl_wait`, `fence_proxy_async`,
`red_global` (fire-and-forget global reduction, gpu/sys scope), `store_sys`
(relaxed sys store), `load_sys` (acquire/volatile sys load), `clock`,
`timeout_trap`, and stage/phase updates are schedule operations. Scalar
address arithmetic, pointer offsets, and guards are shown directly; they do
not hide copies, computation, role changes, or synchronization.

There are deliberately no computational primitives named `dispatch`,
`combine`, `notify`, `barrier_all`, `gin`, or `nccl`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    kIsScaleupNVLink=True,            # team_t = ncclTeamTagLsa; RDMA branches dead
    kDoCPUSync=False,
    kReuseSlotIndices=False,          # non-cached: notify warps present
    kNumSMs=...,                      # auto: get_theoretical_num_sms; 64 for e256/k6
    kNumNotifyWarps=4,
    kNumDispatchWarps=...,            # min((smem - 1536) // 14464, 28); 15 @ 232448 B
    kNumRanks=8,                      # = kNumScaleupRanks; kNumScaleoutRanks = 1
    kNumHiddenBytes=14336,            # hidden 7168 * sizeof(bf16)
    kNumSFPacks=0,                    # bf16 mode: no scale factors
    kNumMaxTokensPerRank=...,         # config: 128 / 1024 / 4096
    kNumExperts=256, kNumTopk=6, kExpertAlignment=(1, 128),
    kDoExpand=False, kCachedMode=False, kDoZeroPadding=False,
    target="sm_100a",
)
# instruction_selection: none; extent: the template argument list of
#   `dispatch_impl` / `dispatch_copy_epilogue_impl` for the direct path

# Compile-time derived constants (no emitted code)
kNumExpertsPerRank   = 32                       # kNumExperts / kNumRanks
kNumNotifyThreads    = 128                      # kNumNotifyWarps * 32
kNumNotifySmemBytes  = 1536                     # align(8 + 256, 128) * 4
kTokenMetadataBytes  = 76                       # 6*4 topk idx + 6*4 weights + 4 src + 6*4 linked-list
kTokenBytesGmem      = 14432                    # align(14336,32) + align(0,32) + align(76,32)=96
kTokenBytesSmem      = 14464                    # kTokenBytesGmem + align(8,32) mbarrier
kNumThreads          = (kNumNotifyWarps + kNumDispatchWarps) * 32   # 608 @ 15 warps
kRecvRegionBytes     = kNumRanks * kNumMaxTokensPerRank * kTokenBytesGmem
# Workspace byte offsets (layout.cuh; kNumMaxRanks=1024, kNumMaxExperts=2048)
WS_BARRIER_COUNTER   = 0                        # u64; phase bit 0, sign bit 1
WS_BARRIER_SIGNAL    = 8                        # i32[2]; slot phase at (2+phase)*4
WS_NOTIFY_REDUCTION  = 16                       # i64[3072]
WS_COUNT_SEND        = 24592                    # i64[3072]: rank counts [0:8), expert counts [8:264)
WS_COUNT_RECV        = 49168                    # i64[3072]: peer-written, decoded-consumed
WS_SENDER_COUNTER    = 73744                    # i32[1024]: per-dst-rank slot atomics
WS_PORT_SCRATCH      = 12820528                 # port substitution 2: software grid-barrier slots,
                                                # inside the 2 MB-aligned workspace reservation
# instruction_selection: none; extent: compile-time constants only

launch_config = launch(
    grid=(kNumSMs, 1, 1),
    cluster=(2 - kNumSMs % 2, 1, 1),            # 2 for even kNumSMs
    block=(kNumThreads, 1, 1),
    min_blocks_per_sm=1,
    dynamic_smem_bytes=kNumSmemBytes,           # jit::device_runtime->get_num_smem_bytes()
    programmatic_dependent_launch=True,
)
# instruction_selection: none; extent: static launch metadata;
#   `__launch_bounds__(kNumThreads, 1)`; cudaLaunchKernelEx cluster + PDL attributes

def dispatch_impl(
    x,                  # bf16* [num_tokens, kNumHiddenBytes/2]
    topk_idx,           # i64*  [num_tokens, kNumTopk]
    topk_weights,       # f32*  [num_tokens, kNumTopk]
    copied_topk_idx,    # i64*  [num_tokens, kNumTopk] (do_handle_copy)
    psum_rank,          # i32*  [kNumRanks]             (out: inclusive prefix)
    psum_expert,        # i32*  [kNumExpertsPerRank+1]  (out: exclusive prefix)
    num_unaligned,      # i32*  [kNumExpertsPerRank]    (out: raw counts)
    dst_slot_idx,       # i32*  [num_tokens, kNumTopk]  (out)
    peer_ws_ptrs,       # i64*  [kNumRanks]  (substitution 1: peer workspace bases)
    peer_buf_ptrs,      # i64*  [kNumRanks]  (substitution 1: peer recv-region bases)
    buffer,             # u8*   local recv-region base (window_base + aligned workspace bytes)
    workspace,          # u8*   local window base
    num_tokens,         # i32   rank-local token count
    rank_idx,           # i32   0..7
):
    sm_idx  = block_id()
    thread_idx = thread_id()
    warp = shuffle_index(thread_idx // 32, source_lane=0)
    # instruction_selection: mov.u32 %tid.x, shr.u32, shfl.sync.idx.b32;
    #   extent: warp-uniform scalar
    lane = lane_id()
    # instruction_selection: mov.s32 from %laneid; extent: scalar per thread

    smem = tile("shared", "u8", [kNumSmemBytes], byte_offset=0, alignment=1024)

    # -----------------------------------------------------------------------
    # Entry NVLink barrier (tag0): no TMA flush, no start grid sync, end grid sync
    # (comm.cuh: gpu_barrier<..., kDispatchTag0, false, false, true>)
    # -----------------------------------------------------------------------
    nvlink_barrier(workspace, peer_ws_ptrs, rank_idx, sm_idx, thread_idx,
                   tag="dispatch0", flush=False)
    # instruction_selection: red.release.sys.global.add.s32 (8 lanes of SM 0) +
    #   atom.add.u64 counter + ld.acquire.sys spin on i32 signal;
    #   extent: SM 0 only, one phase toggle
    grid_barrier(workspace, WS_PORT_SCRATCH, site=0)
    # instruction_selection (substitution 2): atom.add.s32 + st.release.gpu /
    #   ld.acquire.gpu spin; extent: one arrive per SM; replaces this_grid().sync()

    if warp < kNumNotifyWarps:
        # ===================================================================
        # NOTIFY ROLE: warps 0..3
        # ===================================================================

        rank_expert_count = view(smem, "i32", [384], byte_offset=0)
        rank_count   = slice(rank_expert_count, 0, 8)
        expert_count = slice(rank_expert_count, 8, 264)

        # Clean initial counts (dispatch.cuh:87-89)
        for i in range(384 // kNumNotifyThreads):
            fill(rank_expert_count[i * kNumNotifyThreads + thread_idx], 0)
        # instruction_selection: st.shared.s32; extent: 3-element vector loop per thread
        named_barrier(idx=1, threads=kNumNotifyThreads)
        # instruction_selection: bar.sync 1, 128; extent: all 128 notify threads

        # Per-token counting (dispatch.cuh:94-107)
        global_warp_idx = warp * kNumSMs + sm_idx
        for i in range(global_warp_idx, num_tokens, kNumNotifyWarps * kNumSMs):
            dst_expert = select(lane < kNumTopk,
                                copy_g2r(topk_idx[i * kNumTopk + lane], reg_tile([], "i64")), -1)
            # instruction_selection: ld.global.nc.s64 (__ldg) predicated on lane < 6;
            #   extent: scalar per lane
            if dst_expert >= 0:
                atomic_add_block(expert_count[dst_expert], 1)
                # instruction_selection: atom.shared.add.s32; extent: scalar, <= kNumTopk lanes
            dst_rank = select(dst_expert >= 0, div_floor(dst_expert, kNumExpertsPerRank), -1)
            # instruction_selection: s32 div by 32 -> shr.s32 5; extent: scalar
            dup_mask = match_any(dst_rank)
            # instruction_selection: match.any.sync.b32; extent: warp-wide
            is_master = bfind(dup_mask) == lane
            # instruction_selection: bfind.u32 + setp; extent: warp-wide deduplicate
            if is_master and dst_rank >= 0:
                atomic_add_block(rank_count[dst_rank], 1)
                # instruction_selection: atom.shared.add.s32; extent: scalar, one lane per distinct rank
        named_barrier(idx=1, threads=kNumNotifyThreads)
        # instruction_selection: bar.sync 1, 128

        # Full-grid reduction into workspace (dispatch.cuh:111-115)
        for i in range(thread_idx, 8 + 256, kNumNotifyThreads):
            red_global(workspace + WS_NOTIFY_REDUCTION + i * 8,
                       (1 << 32) | rank_expert_count[i], scope="gpu", type="u64")
            # instruction_selection: red.gpu.global.add.u64; extent: 2-3 per thread

        # SM 0 completes the exchange (dispatch.cuh:118-257)
        if sm_idx == 0:
            # Wait all SMs' arrivals; decode; clean reduction slots (dispatch.cuh:121-147)
            for i in range(thread_idx, 8 + 256, kNumNotifyThreads):
                timeout_loop():
                    status = copy_g2r(workspace + WS_NOTIFY_REDUCTION + i * 8,
                                      reg_tile([], "i64"), semantic="volatile")
                    # instruction_selection: ld.volatile.global.u64; extent: spin per slot
                    if shift_right(status, 32) == kNumSMs:
                        encoded = encode_positive(bitwise_and(status, 0xFFFFFFFF))
                        # instruction_selection: s32 neg + sub (encode: -v-1); extent: scalar
                        copy_r2s(encoded, rank_expert_count[i])
                        copy_r2g(0, workspace + WS_NOTIFY_REDUCTION + i * 8, semantic="relaxed")
                        # instruction_selection: st.global.u64 (plain); extent: scalar
                        break
                    on_timeout(): printf_and_trap(tag="notify-reduction")
                    # instruction_selection: clock64 loop + trap; extent: 100 s budget
            named_barrier(idx=1, threads=kNumNotifyThreads)
            # instruction_selection: bar.sync 1, 128

            # Publish rank counters to every peer (dispatch.cuh:152-158)
            for i in range(thread_idx, kNumRanks, kNumNotifyThreads):
                store_sys(remote(workspace + WS_COUNT_RECV + rank_idx * 8, dst=i, table=peer_ws_ptrs),
                          rank_count[i])
                # instruction_selection: st.relaxed.sys.global.u64; extent: 8 lanes, one per peer;
                #   remote address = peer_ws_ptrs[i] + offset (substitution 1)
            syncwarp()
            # Publish per-expert counters, NVLink per-element form (dispatch.cuh:162-170)
            for i in range(thread_idx, kNumExperts, kNumNotifyThreads):
                idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank)
                store_sys(remote(workspace + WS_COUNT_RECV + 8 * 8 + idx * 8,
                                 dst=i // kNumExpertsPerRank, table=peer_ws_ptrs),
                          expert_count[i])
                # instruction_selection: st.relaxed.sys.global.u64; extent: 2 per thread
            named_barrier(idx=1, threads=kNumNotifyThreads)
            # instruction_selection: bar.sync 1, 128 (results rewrite smem below)

            # Wait for every peer's counts; consume and clean (dispatch.cuh:184-201)
            for i in range(thread_idx, 8 + 256, kNumNotifyThreads):
                timeout_loop(start_clock=shared_clock):
                    count = copy_g2r(workspace + WS_COUNT_RECV + i * 8,
                                     reg_tile([], "i64"), semantic="volatile")
                    # instruction_selection: ld.volatile.global.u64; extent: spin per slot
                    decoded = encode_positive(count)
                    if is_ready(decoded):
                        copy_r2g(0, workspace + WS_COUNT_RECV + i * 8, semantic="relaxed")
                        copy_r2s(decoded, rank_expert_count[i])
                        break
                    on_timeout(): printf_and_trap(tag="notify")
            named_barrier(idx=1, threads=kNumNotifyThreads)
            # instruction_selection: bar.sync 1, 128

            # Per-expert reduce across source ranks + align (dispatch.cuh:205-220)
            for i in range(thread_idx, kNumExpertsPerRank, kNumNotifyThreads):
                total = fill(reg_tile([], "i32"), 0)
                for j in range(kNumRanks):
                    total = add(total, expert_count[j * kNumExpertsPerRank + i])
                # instruction_selection: ld.shared.s32 + add.s32; extent: 8-term serial sum
                copy_r2g(total, num_unaligned[i])
                # instruction_selection: st.global.s32; extent: scalar
                copy_r2s(align_up(total, kExpertAlignment), expert_count[i])
                # instruction_selection: s32 round-up to kExpertAlignment; extent: scalar
            named_barrier(idx=1, threads=kNumNotifyThreads)
            # instruction_selection: bar.sync 1, 128

            # (kDoCPUSync=false: the host-workspace write block at dispatch.cuh:224-230
            #  is compiled out in this specialization)

            # Prefix sums, one warp each (dispatch.cuh:234-257)
            if warp == 0:
                # Inclusive prefix over 8 rank counts -> psum_rank[0:8)
                # (source do_psum: ceil_div(8 + 0, 32) = 1 iteration; warps 2..3 skip)
                psum = fill(reg_tile([], "i32"), 0)
                idx = lane
                value = select(idx < kNumRanks, rank_count[idx], 0)
                scan = add(psum, warp_inclusive_sum(value))
                # instruction_selection: 5x (shfl.sync.up.b32 + predicated add.s32);
                #   extent: 32-lane Hillis-Steele
                if idx < kNumRanks:
                    copy_r2g(scan, psum_rank[idx])
                    # instruction_selection: st.global.s32; extent: 8 lanes
            elif warp == 1:
                # Exclusive prefix over 32 expert counts -> psum_expert[0:33)
                # (source do_psum: ceil_div(32 + 1, 32) = 2 iterations)
                psum = fill(reg_tile([], "i32"), 0)
                for it in range(2):
                    idx = it * 32 + lane
                    value = select(0 <= idx - 1 < kNumExpertsPerRank, expert_count[idx - 1], 0)
                    scan = add(psum, warp_inclusive_sum(value))
                    # instruction_selection: 5x (shfl.sync.up.b32 + predicated add.s32)
                    #   per iteration; extent: 32-lane Hillis-Steele, 2 iterations
                    if idx < kNumExpertsPerRank + 1:
                        copy_r2g(scan, psum_expert[idx])
                        # instruction_selection: st.global.s32; extent: 33 lanes over 2 iterations
                    psum = shuffle_index(scan, source_lane=31)
                    # instruction_selection: shfl.sync.idx.b32 from lane 31; extent: warp-uniform

    else:
        # ===================================================================
        # DISPATCH ROLE: warps 4..(4+kNumDispatchWarps-1), one channel each
        # ===================================================================
        dispatch_warp_idx = warp - kNumNotifyWarps

        # Per-warp SMEM token buffer with trailing mbarrier (layout.cuh)
        tma_buffer = view(smem, "u8", [kTokenBytesSmem],
                          byte_offset=kNumNotifySmemBytes + dispatch_warp_idx * kTokenBytesSmem)
        tma_hidden    = view(tma_buffer, "u8",  [14336], byte_offset=0)
        tma_topk_idx  = view(tma_buffer, "i32", [6],     byte_offset=14336)
        tma_topk_w    = view(tma_buffer, "f32", [6],     byte_offset=14336 + 24)
        tma_src_idx   = view(tma_buffer, "i32", [1],     byte_offset=14336 + 48)
        # (linked-list i32[6] @ +52 unused in this specialization)
        tma_mbar      = view(tma_buffer, "u64", [1],     byte_offset=14432)

        # Local recv region: [kNumRanks][kNumMaxTokensPerRank][kTokenBytesGmem]
        # (dispatch.cuh:266-268; the RDMA send region at recv end is dead here)
        recv_region = tile("gmem", "u8", [kRecvRegionBytes], base=buffer)

        phase = fill(reg_tile([], "u32"), 0)
        if elect_one():
            mbarrier_init(tma_mbar, arrive_count=1)
            # instruction_selection: mbarrier.init.shared::cta.b64 +
            #   fence.mbarrier_init.release.cluster; extent: one elected lane
        syncwarp()

        token_start = dispatch_warp_idx * kNumSMs + sm_idx
        token_stride = kNumDispatchWarps * kNumSMs
        for token_idx in range(token_start, num_tokens, token_stride):

            # Wait prior TMA store group fully drained (dispatch.cuh:284)
            tma_store_wait(remaining=0)
            # instruction_selection: cp.async.bulk.wait_group 0; extent: warp-wide
            syncwarp()

            # TMA-load the token's hidden bytes into SMEM (dispatch.cuh:288-291)
            if elect_one():
                copy_g2s(x + token_idx * kNumHiddenBytes, tma_hidden,
                         completion=tma_mbar, num_bytes=kNumHiddenBytes)
                # instruction_selection: cp.async.bulk.shared::cluster.global.
                #   mbarrier::complete_tx::bytes.L2::cache_hint (evict-first);
                #   extent: one 14336-byte 1-D bulk copy
            syncwarp()

            # (kNumSFPacks == 0: the SF cp.async block at dispatch.cuh:295-312
            #  is compiled out)

            # Load top-k into registers and SMEM metadata (dispatch.cuh:317-326)
            stored_dst_rank = fill(reg_tile([], "i32"), -1)
            if lane < kNumTopk:
                raw_expert = copy_g2r(topk_idx[token_idx * kNumTopk + lane], reg_tile([], "i64"))
                # instruction_selection: ld.global.nc.s64 (__ldg); extent: scalar per lane
                dst_expert = i32(raw_expert)
                stored_dst_rank = select(dst_expert >= 0,
                                         div_floor(dst_expert, kNumExpertsPerRank), -1)
                copy_r2s(dst_expert, tma_topk_idx[lane])
                # instruction_selection: st.shared.s32; extent: scalar per lane
                copy_r2s(copy_g2r(topk_weights[token_idx * kNumTopk + lane], reg_tile([], "f32")),
                         tma_topk_w[lane])
                # instruction_selection: ld.global.nc.f32 + st.shared.f32; extent: scalar per lane
                copy_r2g(raw_expert, copied_topk_idx[token_idx * kNumTopk + lane])
                # instruction_selection: st.global.s64; extent: scalar per lane
            syncwarp()

            # Source metadata; must be the last SMEM write before the fence (dispatch.cuh:331-333)
            if elect_one():
                copy_r2s(rank_idx * kNumMaxTokensPerRank + token_idx, tma_src_idx[0])
                # instruction_selection: st.shared.s32; extent: one elected lane
            fence_proxy_async()
            # instruction_selection: fence.proxy.async.shared::cta; extent: warp-wide
            syncwarp()

            # Deduplicate destination ranks and allocate slots (dispatch.cuh:337-351,
            # kReuseSlotIndices=false branch)
            stored_slot = fill(reg_tile([], "i32"), -1)
            dup_mask = match_any(stored_dst_rank)
            # instruction_selection: match.any.sync.b32; extent: warp-wide
            if bfind(dup_mask) == lane and stored_dst_rank >= 0:
                stored_slot = atomic_add_gmem(workspace + WS_SENDER_COUNTER + stored_dst_rank * 4, 1)
                # instruction_selection: atom.global.add.s32 (returns old);
                #   extent: one lane per distinct destination rank
            if lane < kNumTopk:
                copy_r2g(select(stored_slot >= 0,
                                rank_idx * kNumMaxTokensPerRank + stored_slot, -1),
                         dst_slot_idx[token_idx * kNumTopk + lane])
                # instruction_selection: st.global.s32; extent: scalar per lane
            syncwarp()

            # Publish expected bytes and wait TMA load arrival (dispatch.cuh:356-359)
            if elect_one():
                expect_tx(tma_mbar, kNumHiddenBytes)
                # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64;
                #   extent: one elected lane
                mbarrier_wait(tma_mbar, phase)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin loop
                #   (0x989680 suspend hint); extent: one elected lane, phase flips
            syncwarp()

            # (not kIsScaleupNVLink: the send-buffer TMA store at dispatch.cuh:364-369
            #  is compiled out)

            # TMA-store the whole token slot to the destination rank (dispatch.cuh:372-379)
            dst_ptr = select(stored_slot >= 0,
                             remote(buffer + rank_idx * (kNumMaxTokensPerRank * kTokenBytesGmem)
                                    + stored_slot * kTokenBytesGmem,
                                    dst=stored_dst_rank, table=peer_buf_ptrs),
                             nullptr)
            # instruction_selection (substitution 1): peer_buf_ptrs[dst] + offset;
            #   extent: scalar address form
            if dst_ptr != nullptr:
                copy_s2g(tma_buffer, dst_ptr, num_bytes=kTokenBytesGmem)
                # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group.
                #   L2::cache_hint (evict-normal); extent: one 14432-byte 1-D bulk copy,
                #   issued by the lane holding a valid slot (source: any lane with
                #   dst_ptr != nullptr issues it; per distinct dst rank one store)
            tma_store_commit()
            # instruction_selection: cp.async.bulk.commit_group; extent: warp-wide
            syncwarp()

            # (not kIsScaleupNVLink: the GIN put at dispatch.cuh:382-393 is compiled out)

    # -----------------------------------------------------------------------
    # Exit barrier (tag1): TMA flush + start grid sync, no end grid sync
    # (comm.cuh: gpu_barrier<..., kDispatchTag1, true, true, false>)
    # -----------------------------------------------------------------------
    tma_store_commit()
    tma_store_wait(remaining=0)
    # instruction_selection: cp.async.bulk.commit_group + cp.async.bulk.wait_group 0;
    #   extent: warp-wide, all warps
    syncwarp()
    grid_barrier(workspace, WS_PORT_SCRATCH, site=1)
    # instruction_selection (substitution 2): replaces this_grid().sync()
    nvlink_barrier(workspace, peer_ws_ptrs, rank_idx, sm_idx, thread_idx,
                   tag="dispatch1", flush=True)
    # instruction_selection: red.release.sys.global.add.s32 + ld.acquire.sys spin;
    #   extent: SM 0 only, second phase toggle

    # Chain the copy epilogue (dispatch.cuh:403)
    pdl_trigger()
    # instruction_selection: griddepcontrol.launch_dependents; extent: every CTA
    #   (PDL completes when all CTAs trigger; SM 0 triggers after the barrier)

    # Clean atomic sender counters (dispatch.cuh:406-408)
    if sm_idx == 0 and thread_idx < kNumRanks:
        copy_r2g(0, workspace + WS_SENDER_COUNTER + thread_idx * 4, semantic="relaxed")
        # instruction_selection: st.global.s32 (plain); extent: 8 threads of SM 0


# ===========================================================================
# Kernel 2: dispatch_copy_epilogue_impl
# ===========================================================================

epilogue_variant = specialize(
    kDoExpand=False, kCachedMode=False, kDoZeroPadding=False,
    kNumSMs=...,                        # same as kernel 1
    kNumChannels=1,                     # scaleout-only concept; fixed 1 here
    kNumWarps=...,                      # min(smem // 14464, 32); 16 @ 232448 B
    kNumScaleoutRanks=1, kNumScaleupRanks=8,
    kNumHiddenBytes=14336, kNumSFPacks=0,
    kNumMaxTokensPerRank=..., kNumExperts=256, kNumTopk=6, kExpertAlignment=(1, 128),
)
# instruction_selection: none; extent: template argument list

epilogue_launch = launch(
    grid=(kNumSMs, 1, 1),
    cluster=(1, 1, 1),
    block=(kNumWarps * 32, 1, 1),       # 512 @ 16 warps
    min_blocks_per_sm=1,
    dynamic_smem_bytes=kNumSmemBytes,
    programmatic_dependent_launch=True, # waits on kernel 1's trigger
)
# instruction_selection: none; extent: static launch metadata;
#   `__launch_bounds__(kNumWarps * 32, 1)`

def dispatch_copy_epilogue_impl(
    buffer,             # u8* local recv-region base
    psum_rank,          # i32* [kNumRanks] inclusive prefix (from kernel 1)
    psum_expert,        # i32* sliced [1:] (unused in non-expand; kept for ABI)
    recv_x,             # bf16* [num_recv_alloc, kNumHiddenBytes/2]
    recv_topk_idx,      # i64*  [num_recv_alloc, kNumTopk]
    recv_topk_weights,  # f32*  [num_recv_alloc, kNumTopk]
    recv_src_metadata,  # i32*  [num_recv_alloc, 2 + kNumTopk]
    num_unaligned,      # i32*  (unused in non-expand; kept for ABI)
    num_recv_tokens,    # i32   host value: kNumMaxTokensPerRank * kNumRanks (worst case)
    rank_idx,           # i32   0..7
):
    sm_idx = block_id()
    warp = shuffle_index(thread_id() // 32, source_lane=0)
    lane = lane_id()
    global_warp_idx = warp * kNumSMs + sm_idx

    smem = tile("shared", "u8", [kNumSmemBytes], byte_offset=0, alignment=1024)
    tma_buffer = view(smem, "u8", [kTokenBytesSmem],
                      byte_offset=warp * kTokenBytesSmem)
    tma_hidden    = view(tma_buffer, "u8",  [14336], byte_offset=0)
    tma_topk_idx  = view(tma_buffer, "i32", [6],     byte_offset=14336)
    tma_topk_w    = view(tma_buffer, "f32", [6],     byte_offset=14336 + 24)
    tma_src_idx   = view(tma_buffer, "i32", [1],     byte_offset=14336 + 48)
    tma_mbar      = view(tma_buffer, "u64", [1],     byte_offset=14432)

    phase = fill(reg_tile([], "u32"), 0)
    if elect_one():
        mbarrier_init(tma_mbar, arrive_count=1)
        # instruction_selection: mbarrier.init.shared::cta.b64 +
        #   fence.mbarrier_init.release.cluster; extent: one elected lane
    syncwarp()

    # Block until kernel 1 finished and all data is visible (epilogue.cuh:60)
    pdl_wait()
    # instruction_selection: griddepcontrol.wait; extent: every CTA;
    #   PDL edge: completes after all kernel-1 CTAs triggered (SM 0 triggers
    #   post-barrier, so peer data is visible here)

    # Worst-case host count -> read the real count from the GPU prefix (epilogue.cuh:63-64)
    if num_recv_tokens == kNumMaxTokensPerRank * kNumRanks:
        num_recv_tokens = copy_g2r(psum_rank[kNumRanks - 1], reg_tile([], "i32"))
        # instruction_selection: ld.global.s32 (plain; post-PDL visibility);
        #   extent: scalar per thread, warp-uniform value

    # Per-warp strided loop over received tokens (epilogue.cuh:67-80)
    current_rank = fill(reg_tile([], "i32"), -1)
    rank_start = fill(reg_tile([], "i32"), 0)
    rank_end = fill(reg_tile([], "i32"), 0)
    for i in range(global_warp_idx, num_recv_tokens, kNumWarps * kNumSMs):
        # Locate the source rank of received token i via the inclusive prefix
        while i >= rank_end:
            current_rank = add(current_rank, 1)
            stored_lane = current_rank % 32
            stored_psum = select(stored_lane == 0 and current_rank + lane < kNumRanks,
                                 copy_g2r(psum_rank[current_rank + lane], reg_tile([], "i32")),
                                 stored_psum)
            # instruction_selection: ld.global.s32 predicated; extent: 32-lane
            #   cooperative prefetch of the prefix array
            rank_start = rank_end
            rank_end = shuffle_index(stored_psum, source_lane=stored_lane)
            # instruction_selection: shfl.sync.idx.b32; extent: warp-uniform scalar

        token_addr = buffer + current_rank * (kNumMaxTokensPerRank * kTokenBytesGmem) \
                     + (i - rank_start) * kTokenBytesGmem

        # Drain prior TMA stores before reusing the SMEM slot (epilogue.cuh:84)
        tma_store_wait(remaining=0)
        # instruction_selection: cp.async.bulk.wait_group 0; extent: warp-wide
        syncwarp()

        # TMA-load the full token slot (hidden + metadata) (epilogue.cuh:89-93)
        if elect_one():
            copy_g2s(token_addr, tma_buffer, completion=tma_mbar, num_bytes=kTokenBytesGmem)
            # instruction_selection: cp.async.bulk.shared::cluster.global.
            #   mbarrier::complete_tx::bytes.L2::cache_hint (evict-first);
            #   extent: one 14432-byte 1-D bulk copy
            expect_tx(tma_mbar, kTokenBytesGmem)
            # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64
        syncwarp()

        # Read target expert indices early, DIRECTLY FROM THE GMEM token slot,
        # to tolerate TMA latency (epilogue.cuh:96-100: buffer_token is the GMEM
        # token view, not the SMEM staging buffer; the SMEM copy is not yet
        # guaranteed to have arrived)
        dst_expert = select(lane < kNumTopk,
                            copy_g2r(token_addr + 14336 + lane * 4, reg_tile([], "i32")), -1)
        # instruction_selection: ld.global.s32 (plain; NO .nc — PDL visibility,
        #   source comment "PDL is used, please do not use __ldg") predicated on
        #   lane < 6; extent: scalar per lane
        syncwarp()

        # Validate, localize, and check per-token rank uniqueness (epilogue.cuh:104-109)
        expert_start = kNumExpertsPerRank * rank_idx
        expert_end = expert_start + kNumExpertsPerRank
        in_range = expert_start <= dst_expert < expert_end
        master_lane = bfind(ballot(in_range))
        # instruction_selection: vote.sync.ballot.b32 + bfind.u32;
        #   extent: warp-wide; highest set lane = master source top-k slot
        dst_expert = select(in_range, dst_expert - expert_start, -1)
        # device_assert: match_any(dst_expert) dedup holds (or dst_expert == -1)
        # instruction_selection: match.any.sync.b32 + bfind.u32 (assert only)
        if lane < kNumTopk:
            copy_r2g(i64(dst_expert), recv_topk_idx[i * kNumTopk + lane])
            # instruction_selection: st.global.s64; extent: scalar per lane
        syncwarp()

        # Non-expand destination index is the token ordinal (epilogue.cuh:114-116)
        dst_tensor_idx = select(elect_one(), i, -1)
        # instruction_selection: elect.sync; extent: one elected lane
        syncwarp()

        # Wait TMA arrival (epilogue.cuh:126-127)
        if elect_one():
            mbarrier_wait(tma_mbar, phase)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 spin;
            #   extent: one elected lane, phase flips
        syncwarp()

        # (kDoCreateLinkedList=false: linked-list block at epilogue.cuh:131-135 dead)

        # TMA-store hidden to the output tensor (epilogue.cuh:138-142)
        if elect_one():
            copy_s2g(tma_hidden, recv_x + i * kNumHiddenBytes, num_bytes=kNumHiddenBytes)
            # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group.
            #   L2::cache_hint (evict-normal); extent: one 14336-byte 1-D bulk copy
            tma_store_commit()
            # instruction_selection: cp.async.bulk.commit_group
        syncwarp()

        # (kNumSFPacks == 0: SF store block at epilogue.cuh:146-177 compiled out)

        # Store top-k weights (epilogue.cuh:182-184)
        if lane < kNumTopk:
            copy_r2g(copy_s2r(tma_topk_w[lane], reg_tile([], "f32")),
                     recv_topk_weights[i * kNumTopk + lane])
            # instruction_selection: ld.shared.f32 + st.global.f32; extent: scalar per lane
        syncwarp()

        # Write source metadata, non-cached mode (epilogue.cuh:192-201)
        if elect_one():
            src_global = copy_s2r(tma_src_idx[0], reg_tile([], "i32"))
            # instruction_selection: ld.shared.s32; extent: one elected lane
            copy_r2g(src_global, recv_src_metadata[i * 8 + 0])
            # instruction_selection: st.global.s32; extent: scalar
            copy_r2g(current_rank * kNumTopk + master_lane, recv_src_metadata[i * 8 + 1])
            # instruction_selection: st.global.s32; extent: scalar
        syncwarp()

    # (kDoCreateLinkedList=false: tail block at epilogue.cuh:212-229 dead)
    # (kDoZeroPadding=false: zero-padding block at epilogue.cuh:232-322 dead)
```

## Kernel-specific tables

### Workspace regions used by this specialization

| region | byte offset | dtype | extent | writer -> consumer |
| --- | --- | --- | --- | --- |
| nvl_barrier_counter | 0 | u64 | 1 | every SM-0 thread-0 (atomicAdd) -> phase/sign reader |
| nvl_barrier_signal[phase] | 8 + phase*4 | i32 | 2 | peers' `red.release.sys` -> local `ld.acquire.sys` spin |
| notify_reduction | 16 | i64 | 8 + 256 used of 3072 | every SM's notify warps (`red.gpu`) -> SM 0 wait/decode/clean |
| scaleup count recv area | 49168 | i64 | 8 rank + 256 expert | peers' `st.relaxed.sys` -> SM 0 wait/decode/clean |
| sender counters | 73744 | i32 | 8 used of 1024 | dispatch warps `atom.add` -> SM 0 post-barrier cleanup |
| port scratch (grid barrier) | 12820528 | i32 | 2 sites x (counter + flag) | substitution 2; inside the 2 MB-aligned reservation slack |
| count send area | 24592 | i64 | unused | only written when `not kIsScaleupNVLink` |

### Token slot layout (SMEM 14464 B / GMEM 14432 B)

| field | offset | dtype | extent |
| --- | --- | --- | --- |
| hidden | 0 | bf16 | 7168 elems / 14336 B |
| topk_idx | 14336 | i32 | 6 |
| topk_weights | 14360 | f32 | 6 |
| src_token_global_idx | 14384 | i32 | 1 |
| linked_list_idx (dead) | 14388 | i32 | 6 |
| pad | 14412..14431 | - | to 32 B |
| mbarrier (SMEM only) | 14432 | u64 | 1 (+ 32 B pad to 14464) |

### NVLink barrier state machine (comm.cuh:88-129)

`status = counter & 3`; `phase = status & 1`; `sign = status >> 1`. SM 0's
threads `red.release.sys.add` (`+1` when `sign==0`, `-1` when `sign==1`) into
each peer's `signal[phase]`; then a CTA-local `__syncthreads()`
(comm.cuh:107: `bar.sync 0` — orders the per-lane signal writes and the
phase read before the counter increment) fires before thread 0 `atomicAdd`s
the local counter and spins with `ld.acquire.sys` until
`signal[phase] == (sign ? 0 : kNumRanks)`.
Two toggling slots make the barrier self-cleaning across calls. The dispatch
kernel uses it twice (tag0 entry, tag1 exit), so both phases are exercised per
launch.

### Software grid barrier (substitution 2)

Each site owns `{counter: i32, flag: i32}` in the port-scratch area, zero at
allocation. Every SM's thread 0 arrives with `atom.add(counter, 1)`; the
arriver observing `old == kNumSMs - 1` resets the counter to 0 and issues
`st.release.gpu(flag, generation)`; all other SMs spin
`ld.acquire.gpu(flag) != generation`. `generation` is a per-site value that
toggles once per completed barrier (tracked in the slot pair via a third word
or derived from the flag's previous value). Semantics equal
`this_grid().sync()` at the two source call sites (entry-barrier end,
exit-barrier start).

### TIRx module and benchmark contract

- Module: `tirx_kernels/deepep/dispatch.py`; `get_kernel` returns the PrimFunc
  pair; per-rank worker launches kernel 1 then kernel 2 on one stream with the
  PDL launch attribute on both.
- Rank-local inputs from `prepare_data`; symmetric window allocated per rank
  via ctypes `ncclMemAlloc` + `ncclCommWindowRegister` +
  `ncclGetLsaDevicePointer` on a raw `ncclComm_t` from
  `deep_ep.utils.comm.get_nccl_comm_handle`.
- Correctness: outputs and handle metadata compared rank-local against the
  source `ElasticBuffer.dispatch` on identical inputs (bf16, do_cpu_sync=false,
  non-cached, non-expand).
- Benchmark: `tirx_kernels.bench_suite`, `num_gpus: 8`, `timer: kineto`; timed
  scope is the two-kernel sequence on both sides (source:
  `dispatch_impl` + `dispatch_copy_epilogue_impl`).

### Static specialization boundary

Fixed per config: `kNumSMs`, `kNumDispatchWarps`, `kNumWarps`,
`kNumMaxTokensPerRank`, `kExpertAlignment`. Fixed for the whole port:
`kNumRanks=8`, `kNumExperts=256`, `kNumTopk=6`, `kNumHiddenBytes=14336`,
`kNumSFPacks=0`, `kNumNotifyWarps=4`, `kIsScaleupNVLink=true`,
`kDoCPUSync=false`, `kReuseSlotIndices=false`, `kDoExpand=false`,
`kCachedMode=false`, `kDoZeroPadding=false`, `kNumScaleoutRanks=1`.
Everything outside this boundary is out of scope (see header).

## Instruction-selection summary

Placement, shape, and schedule select the instructions; the inline
`instruction_selection` annotations carry the evidence. Expected families:

| Primitive pattern | PTX family |
| --- | --- |
| full-token `copy_g2s(..., completion=mbar)` | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` |
| full-token `copy_s2g(smem, gmem/remote)` | `cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint` |
| scalar `copy_g2r` of topk/weights/counts | `ld.global.nc.*` / `ld.volatile.global.*` / `ld.acquire.sys.*` |
| `store_sys` to peer workspace | `st.relaxed.sys.global.u64` |
| `red_global` count reduction | `red.gpu.global.add.u64` |
| NVLink barrier arrive | `red.release.sys.global.add.s32` + `ld.acquire.sys` spin |
| slot allocation | `atom.global.add.s32` (returns old) |
| SMEM counting | `atom.shared.add.s32` |
| dedup / master lane | `match.any.sync.b32` + `bfind.u32`; `vote.sync.ballot.b32` |
| warp prefix sums | `shfl.sync.up.b32` chain (5 steps) + predicated adds |
| mbarrier lifecycle | `mbarrier.init` / `mbarrier.arrive.expect_tx` / `mbarrier.try_wait.parity` |
| TMA store ordering | `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group` / `fence.proxy.async.shared::cta` |
| named barrier | `bar.sync 1, 128` |
| PDL chaining | `griddepcontrol.launch_dependents` / `griddepcontrol.wait` |
| software grid barrier (substitution) | `atom.add.s32` + `st.release.gpu` / `ld.acquire.gpu` |
| timeout policy | `clock64` deltas, `printf`, `trap` |
