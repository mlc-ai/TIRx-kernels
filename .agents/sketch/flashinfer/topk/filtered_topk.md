<!--
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

This design sketch documents a TIRx port of FlashInfer's
include/flashinfer/topk.cuh FilteredTopK pipeline: FilteredTopKUnifiedKernel
and its companion FinalizeTopKIndicesKernel.
-->

# filtered_topk SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/flashinfer/topk/filtered_topk.py`](../../../../tirx_kernels/flashinfer/topk/filtered_topk.py).
That TIRx module is the authoritative implementation.

The port covers **two kernels**, because the dispatchers launch them as one
pipeline. `FilteredTopKUnifiedKernel<DType, int32_t, VEC_SIZE, DETERMINISTIC,
MODE, TIE_BREAK>` (:2362) selects the top-k; `FinalizeTopKIndicesKernel<SORT,
MODE, BT, IPT, DType, int32_t>` (:2958) then sorts each row's indices ascending
and applies the deferred PageTable/Ragged transform whenever the first kernel
deferred it. Omitting the second would leave "deterministic" output with a
deterministic *set* in a nondeterministic *order*.

Instantiations: `DType in {f32, f16, bf16}` x `MODE in {Plain, PageTable,
Ragged}` x `TIE_BREAK in {None, Small, Large}` x `DETERMINISTIC` x `VEC_SIZE in
{4,2,1}` (f32) / `{8,4,2,1}` (16-bit). `DETERMINISTIC` is not the user's
`deterministic` flag: it is `deterministic || tie_break != None` (:3181), so the
four reachable template combinations are `(F,None)`, `(T,None)`, `(T,Small)`,
`(T,Large)`. `num_rows`, `max_len`, `top_k` and the derived `VEC_SIZE`,
`end_bit`, finalize `(BT, IPT)` are static per config, exactly like the values
`LaunchFilteredTopKUnified` (:3170) and `LaunchFinalizeTopKIndices` (:3032)
compute per call.

Accepted target SM100/B200. Out of scope, with the dispatch predicate that
excludes each: `top_k > FILTERED_TOPK_MAX_K == 2048` (radix-only, :3333);
`sorted_output` (`StableSortTopKByValueKernel` :3090, which both radix ports also
leave out); fp8 (never instantiated, :2277-2337); and the SM100 cluster kernel,
which the `FLASHINFER_TOPK_ALGO=filtered` pin keeps out of the reference path.
Tile (`Tx`) primitives are out of scope everywhere.

## Pipeline at a glance

| Kernel / role | Program | Publication / reuse edges |
| --- | --- | --- |
| **Unified**, all 32 warps of the single CTA (uniform, no warp specialization) | trivial early-out; coarse 256-bin histogram over `ToCoarseKey`; 8-step ping-pong suffix scan; threshold-bin pick; **filter** pass compacting only threshold-bin candidates into a 16K shared index buffer; 1 or 4 refine rounds over that buffer; tie collection; mode epilogue | Everything is intra-CTA: `bar.sync 0` around `s_histogram`, `s_indices`, `s_input_idx`, and the scalars. `atom.shared.add.u32` publishes histogram bins and output slots. There are **no global atomics, no workspace, no clusters, and no cross-CTA edges at all** — one CTA owns one row. |
| **Unified**, thread 0 / one predicated thread | `s_refine_overflow` init (:2451); per-round pivot bytes under DETERMINISTIC (:2683-2687); and — note — the counter resets at :2519-2523 are done by *the single thread that finds the threshold bin*, not by tx0 | published through `bar.sync 0` |
| **Finalize**, all BT threads | load <= k row indices as blocked per-thread `uint32` keys with `~0u` padding; `SORT` ? cub BlockRadixSort over `[0, end_bit)` : nothing; deferred transform; writeback with `~0u -> -1` | rank counters and the exchange buffer alias one shared union; `bar.sync 0` per rank/exchange phase |

The two kernels communicate only through `output_indices` (and `output_values`
in Plain) in global memory — the finalize kernel re-reads what the unified kernel
wrote.

## Primitive vocabulary

```python
copy_g2r_v(dst_reg, gmem, n)      # vectorized row read, VEC_SIZE lanes
copy_r2s(smem, reg) / copy_s2r    # shared traffic, always through T.ptx
to_coarse(v)                      # Traits::ToCoarseKey  (:2282-2289 / :2306-2311 / :2326-2331)
to_ordered(v)                     # Traits::ToOrdered     (:2291-2294 / :2313-2316 / :2333-2336)
# Both are written in source as `(bits & sign) ? ~bits : (bits | sign)`, i.e. the OR/NOT form.
# RadixTopKTraits::ToOrdered (topk_common.cuh:35-49) is the XOR form instead; the two must NOT be
# unified, because StableSortTopKByValueKernel (:3094) needs the XOR form's FromOrdered inverse.
# NOTE this is a SOURCE-level distinction only -- see the instruction-selection table for what
# each one actually lowers to, which is not the same thing.
atomic_add_shared(smem, i, v)     # atom.shared.add.u32, returns old
atomic_or_shared(smem, i, v)      # atom.shared.or.b32, result discarded
barrier()                         # bar.sync 0
warp_incl_sum(v, lane)            # 5-step shfl.sync.up.b32 inclusive scan
block_exclusive_sum(v)            # cub BlockScan, RAKING_MEMOIZE geometry
rank_keys(keys) / scatter_blocked(items, ranks)   # finalize only, per digit pass
```

`for_each_score(fn)` is the row-scan helper (:2467-2481): a `#pragma unroll 2`
vector loop over `aligned_length = (length / VEC_SIZE) * VEC_SIZE` followed by a
scalar tail.  The vector load is `ld.global.nc.v4.b32` for **both** f32 VEC_SIZE=4
and 16-bit VEC_SIZE=8 (16 bytes either way), `ld.global.nc.v2.b32` for f32
VEC_SIZE=2, `ld.global.nc.v2.b16` for 16-bit VEC_SIZE=2, and a scalar
`ld.global.nc.b32`/`b16` for the tail. It is shown expanded once and folded thereafter; it is re-run in
full on every phase that rescans the row.

## Complete sketch — unified kernel

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
variant = specialize(DTYPE=("f32","f16","bf16"), MODE=("Plain","PageTable","Ragged"),
                     TIE_BREAK=("None","Small","Large"), DETERMINISTIC=(False,True),
                     VEC_SIZE=(8,4,2,1), target="sm_100a")
# instruction_selection: none; extent: compile-time instantiations

BLOCK        = 1024                    # FILTERED_TOPK_BLOCK_THREADS (:2341)
RADIX        = 256
SMEM_INPUT   = 16384                   # FILTERED_TOPK_SMEM_INPUT_SIZE (:2342)
DYN_SMEM     = 4 * 2 * SMEM_INPUT      # 131072 B, compile-time constant (:2343-2344)
NUM_ROUNDS   = 4 if DTYPE == "f32" else 1     # Traits::NUM_REFINE_ROUNDS
FIRST_SHIFT  = 24 if DTYPE == "f32" else 0    # Traits::FIRST_REFINE_SHIFT
# VEC_SIZE = gcd(max_len, 16/sizeof) (:2941), forced to 1 by dsa_graph_safe (:2936-2938)
# and by a non-null row_starts on a transform mode (:3190-3192).

launch_config = launch(grid=(NUM_ROWS,1,1), block=(BLOCK,1,1),
                       dynamic_smem_bytes=DYN_SMEM, launch_bounds=BLOCK)
# instruction_selection: none; extent: static launch metadata.  cudaFuncSetAttribute
#   (MaxDynamicSharedMemorySize, 131072) is re-issued on every launch (:3198); in TIRx the
#   pool high-water mark drives the request via `tirx.use_dyn_shared_memory`.

# ===========================================================================
# Storage  (:2431-2446)
# ===========================================================================
s_hist2   = static_shared((2, RADIX + 128), "int32", align=128)   # (:2432) ping-pong scan
s_hist    = view(s_hist2[0])                                      # (:2443) buffer 0 is live
s_counter          = static_shared((1,), "int32")                 # (:2433)
s_threshold_bin_id = static_shared((1,), "int32")                 # (:2434)
s_refine_thresholds= static_shared((4,), "int32")                 # (:2436) DETERMINISTIC only
s_num_input        = static_shared((2,), "int32")                 # (:2437)
s_indices          = static_shared((2048,), "int32", align=128)   # (:2438) FILTERED_TOPK_MAX_K
s_refine_overflow  = static_shared((1,), "int32")                 # (:2440)
s_last_remain      = static_shared((1,), "int32")                 # (:2441)
s_input_idx        = dyn_shared((2, SMEM_INPUT), "int32")         # (:2446) 131072 B
if DETERMINISTIC:
    s_scan_temp = static_shared(cub_blockscan_raking_bytes)       # (:2584-2585)
if TIE_BREAK in (Small, Large):
    s_emitted    = static_shared((1,), "uint32")                  # (:313) chunk-walk state,
    s_chunk_base = static_shared((1,), "uint32")                  # (:314) 12 B total, live only
    s_chunk_take = static_shared((1,), "uint32")                  # (:315) across the collector
# instruction_selection: none; extent: static shared layout.
#   INVARIANT: `s_hist[RADIX]` must stay 0.  It is the exclusive-suffix sentinel read at
#   :2508/:2513/:2519/:2527/:2688 and again in both fallbacks at :2842/:2843/:2850; it is
#   zeroed by the `tx < RADIX + 1` clears (:2457, :2562, :2650, :2712, :2816) and never
#   written by the scan (which only ever writes indices < RADIX).

def filtered_topk(input, out_idx, out_val, aux, lengths, row_starts,
                  page_table_row_starts, row_to_batch, aux_stride):
    row = cta_id(axis="x", extent=NUM_ROWS)     # instruction_selection: mov.u32 %ctaid.x
    tx  = thread_id(extent=BLOCK)               # instruction_selection: mov.u32 %tid.x
    if row >= NUM_ROWS: return                  # (:2382)

    # ---- per-row header (:2384-2406) --------------------------------------
    length     = load_global(lengths[row]) if lengths else MAX_LEN     # (:2384)
    row_start  = load_global(row_starts[row]) if (row_starts and MODE != Plain) else 0
    page_start = load_global(page_table_row_starts[row]) if (pts and MODE == PageTable) else row_start
    score      = input + row * MAX_LEN + row_start                     # (:2391)
    dst        = out_idx + row * top_k                                 # (:2392)
    # instruction_selection: ld.global.nc.b32 per optional pointer (the inputs are
    #   `__restrict__ const`, so every unified-kernel global read is non-coherent);
    #   extent: scalar per row
    if MODE == PageTable:
        batch = load_global(row_to_batch[row]) if row_to_batch else row
        src_page_entry = aux + batch * aux_stride                      # (:2400-2402)
    elif MODE == Ragged:
        offset_val = load_global(aux[row])                             # (:2403)
    else:
        dst_values = out_val + row * top_k                             # (:2404)

    # ---- trivial early-out, length <= top_k (:2409-2429) ------------------
    if length <= top_k:
        i = tx
        while i < top_k:                                               # (:2410)
            # The source is an if/else with FOUR independent store sites.  nvcc splits them
            # by shape: the two arms that must GUARD A LOAD (:2414 Plain value, :2423 PageTable
            # gather) become `@p bra`, while the two that merely SELECT A VALUE (:2421, :2425)
            # become `selp.b32`.  Same rule as the finalize writeback below.
            if MODE == Plain:
                if i < length:                                          # (:2412-2415)
                    store_global(dst[i], i)
                    store_global(dst_values[i], load_global(score[i]))
                    # instruction_selection: st.global.b32 (.loc 2 2413) +
                    #   ld.global.nc.b32|b16 (.loc 2 2414) + st.global.b32|b16; extent: one pair
                else:                                                   # (:2416-2417)
                    store_global(dst[i], -1); store_global(dst_values[i], 0)
                    # instruction_selection: 2x st.global.b32|b16 of immediates
            elif DETERMINISTIC:
                store_global(dst[i], i if i < length else -1)            # (:2419-2421)
                # NOTE local index, transform DEFERRED to the finalize kernel
                # instruction_selection: selp.b32 (.loc 2 2421) + st.global.b32; extent: one
                #   scalar store.  Pure value select, no load to guard -> no branch.
            elif MODE == PageTable:
                if i < length:                                          # (:2422-2423)
                    store_global(dst[i], load_global(src_page_entry[page_start + i]))
                else:
                    store_global(dst[i], -1)
                # instruction_selection: @p bra over ld.global.nc.b32 (.loc 2 2423) +
                #   st.global.b32; extent: one gather pair.  No selp is emitted here.
            else:
                store_global(dst[i], i + offset_val if i < length else -1)   # (:2424-2426)
                # instruction_selection: add.s32 + selp.b32 (.loc 2 2425) + st.global.b32;
                #   extent: one scalar store.  Value select, not a branch.
            i = i + BLOCK
        return                                                          # (:2428)

    # ---- init (:2450-2458) ------------------------------------------------
    topk = top_k                                                        # register (:2450)
    if tx == 0: store_shared(s_refine_overflow, 0)                      # (:2451)
    if DETERMINISTIC and tx < 4: store_shared(s_refine_thresholds[tx], 0xFF)
    if tx < RADIX + 1: store_shared(s_hist[tx], 0)                      # 257 entries (:2457)
    barrier()
    # instruction_selection: st.shared.b32 + bar.sync 0; extent: CTA-wide

    # ===================================================================
    # Stage 1: coarse histogram over the whole row (:2482-2487)
    # ===================================================================
    for v, i in for_each_score():                                       # (:2486)
        atomic_add_shared(s_hist, to_coarse(v), 1)
        # instruction_selection: cvt.rn.f16.f32 (f32 only) + setp/selp/or/not for the
        #   monotone flip + shr + atom.shared.add.u32; extent: one per element.
        #   f32's ToCoarseKey rounds through fp16 (:2284) and is therefore LOSSY, but it is
        #   re-derived identically in every phase, so the partition stays consistent.
    barrier()

    # ---- run_cumsum: Hillis-Steele inclusive SUFFIX scan (:2489-2504) -----
    def run_cumsum():
        for i in unroll(8):                                             # (:2490-2491)
            if tx < RADIX:
                j = 1 << i; k = i & 1
                value = load_shared(s_hist2[k][tx])
                if tx < RADIX - j:
                    value = value + load_shared(s_hist2[k][tx + j])
                store_shared(s_hist2[k ^ 1][tx], value)
                # instruction_selection: ld.shared.b32 x2 + add.s32 + st.shared.b32;
                #   extent: one per active lane per step
            barrier()                                                   # (:2502)
        # INVARIANT: 8 is even, so the result lands back in buffer 0 == s_hist.
        # Changing RADIX or the step count silently breaks this.

    # ---- first threshold pick (:2517-2524) --------------------------------
    run_cumsum()
    if tx < RADIX and load_shared(s_hist[tx]) > topk and load_shared(s_hist[tx+1]) <= topk:
        store_shared(s_threshold_bin_id, tx)                            # (:2519-2523)
        store_shared(s_num_input[0], 0)
        store_shared(s_counter, 0)
        # instruction_selection: three short-circuit guards (`@%p bra` over setp.gt.u32
        #   (tx > 255) / setp.le.s32 / setp.gt.s32 -- these are the NEGATED source predicates,
        #   emitted in that order; the export shows and.pred = 0) + 3x st.shared.b32;
        #   extent: exactly ONE thread satisfies this.  The counter resets ride on that same
        #   predicate -- they are NOT a tx0 block.  Uniqueness follows from the suffix scan's
        #   monotonicity plus `length > top_k`.
    barrier()
    threshold_bin = load_shared(s_threshold_bin_id)                     # (:2526)
    topk = topk - load_shared(s_hist[threshold_bin + 1])                # (:2527) now = how
                                                                        # many are still owed
                                                                        # from INSIDE the bin
    topk_after_coarse = topk                                            # (:2528)

    # ---- fast exit: the coarse pass already resolved k (:2549-2559) -------
    if topk == 0:
        for v, i in for_each_score():
            if to_coarse(v) > threshold_bin:
                pos = atomic_add_shared(s_counter, 1)
                store_shared(s_indices[pos], i)
                # instruction_selection: atom.shared.add.u32 + st.shared.b32
        barrier()
        goto OUTPUT

    # ===================================================================
    # Stage 2: FILTER -- the step the algorithm is named for (:2611-2629)
    # Only the threshold bin's candidates are compacted into s_input_idx, so
    # every later pass re-reads <= 16384 scattered elements instead of the row.
    # ===================================================================
    barrier()                                                           # (:2561) opens the
                                                                        # else branch
    if tx < RADIX + 1: store_shared(s_hist[tx], 0)                      # (:2562)
    barrier()                                                           # (:2563)
    for v, i in for_each_score():                                       # (:2628)
        bin = to_coarse(v)
        if bin > threshold_bin:                                         # strict winner
            pos = atomic_add_shared(s_counter, 1)
            store_shared(s_indices[pos], i)
        elif bin == threshold_bin:                                      # candidate
            pos = atomic_add_shared(s_num_input[0], 1)
            if likely(pos < SMEM_INPUT):                                # __builtin_expect (:2618)
                store_shared(s_input_idx[0][pos], i)
                sub = (to_ordered(v) >> FIRST_SHIFT) & 0xFF
                atomic_add_shared(s_hist, sub, 1)
                # instruction_selection: atom.shared.add.u32 x2 + st.shared.b32 + shr + and
            else:
                atomic_or_shared(s_refine_overflow, 1)                  # (:2624)
                # instruction_selection: atom.shared.or.b32 (result discarded -- the export
                #   shows atom, NOT the red.shared.or.b32 reduction form)
    barrier()

    # ===================================================================
    # Stage 3: refine rounds over the candidate buffer (:2675-2709)
    # ===================================================================
    def run_refine_round(r_idx, offset, is_last):
        # r_idx is the PING-PONG BUFFER index (round % 2, :2775), never the round number --
        # s_input_idx / s_num_input are 2 deep, so indexing them by the round would run off
        # the end from round 2 onwards.
        num_input = min(load_shared(s_num_input[r_idx]), SMEM_INPUT)     # (:2677-2678)
        update_refine_threshold(next=r_idx ^ 1, reset=True)              # (:2680) = run_cumsum
                                                                         # (:2507) + the same
                                                                         # predicated pick, also
                                                                         # writing s_last_remain
                                                                         # (:2513).  It does NOT
                                                                         # touch s_counter --
                                                                         # only the first pick
                                                                         # (:2522) resets that.
        threshold = load_shared(s_threshold_bin_id)
        if DETERMINISTIC and tx == 0:
            store_shared(s_refine_thresholds[(FIRST_SHIFT - offset) // 8], threshold)  # (:2685)
        topk = topk - load_shared(s_hist[threshold + 1])                 # (:2688)
        if topk == 0:                                                    # (:2689-2701)
            i = tx
            while i < num_input:
                idx = load_shared(s_input_idx[r_idx][i])
                # instruction_selection: ld.shared.b32 (.loc 2 2692); extent: one per candidate
                if ((to_ordered(load_global(score[idx])) >> offset) & 0xFF) > threshold:
                    # instruction_selection: ld.global.nc.b32|b16 (.loc 2 2693) + the ToOrdered
                    #   sequence + shr + and + setp.gt.s32; extent: one per candidate
                    pos = atomic_add_shared(s_counter, 1)
                    store_shared(s_indices[pos], idx)
                    # instruction_selection: atom.shared.add.u32 + st.shared.b32 (.loc 2 2696)
                i = i + BLOCK
            barrier()                                                    # (:2699)
            return True                                                  # pivot resolved
        if is_last:
            collect_with_threshold_last_round(r_idx, num_input, offset, threshold)  # (:2635-2645)
            # one barrier (:2644)
        else:
            collect_with_threshold_non_last_round(r_idx, num_input, offset, threshold)  # (:2646-2672)
            # THREE barriers (:2649, :2651, :2671): clear the histogram, then ping-pong the
            # survivors into s_input_idx[r_idx ^ 1] with the NEXT byte's histogram, or set
            # s_refine_overflow when that buffer would overflow too.
        return False

    if NUM_ROUNDS == 1:                                                  # 16-bit (:2710-2766)
        if load_shared(s_refine_overflow):                               # (:2711) FALLBACK
            # A coarse bin held more than 16384 candidates, so the compacted buffer is
            # truncated and unusable.  Rebuild the selection from the row instead.
            if tx < RADIX + 1: store_shared(s_hist[tx], 0)               # (:2712)
            barrier()                                                    # (:2713)
            for v, i in for_each_score():                                # (:2715-2724)
                if to_coarse(v) == threshold_bin:
                    atomic_add_shared(s_hist, to_ordered(v) & 0xFF, 1)
                    # instruction_selection: ld.global.nc.b16 + the ToOrdered pair + and +
                    #   atom.shared.add.u32; extent: one per row element in the bin
            barrier()                                                    # (:2725)
            if tx == 0:                                                  # (:2727-2730)
                store_shared(s_threshold_bin_id, 0); store_shared(s_last_remain, 0)
            barrier()                                                    # (:2731)
            update_refine_threshold(next=0, reset=False)                 # (:2733) the ONLY
                                                                         # RESET_NEXT_INPUT=false
                                                                         # call in the kernel
            for v, i in for_each_score():                                # (:2740-2748)
                if to_coarse(v) != threshold_bin: continue               # (:2741-2744) ONLY
                                                                         # threshold-bin elements
                                                                         # are re-examined by low
                                                                         # byte; without this the
                                                                         # pass would re-collect
                                                                         # the strict winners the
                                                                         # filter stage already
                                                                         # appended
                collect_gt_and_nondet_eq_threshold(to_ordered(v) & 0xFF,  # (:2745-2747)
                                                   load_shared(s_threshold_bin_id), i, True)
                # appends AFTER the s_counter prefix the coarse pass already wrote (:2737-2739)
            barrier()                                                    # (:2751)
            if DETERMINISTIC:
                collect_det_eq_pivot(pivot=(threshold_bin << 8) | threshold,
                                     eq_needed=load_shared(s_last_remain))   # (:2752-2757)
        else:
            run_refine_round(r_idx=0, offset=FIRST_SHIFT, is_last=True)  # (:2759-2762)
            if DETERMINISTIC:
                collect_det_eq_pivot(pivot=build_det_pivot(0), eq_needed=topk)  # (:2763-2765)
    else:                                                                # f32 (:2767-2902)
        det_stop_round = NUM_ROUNDS - 1                                  # (:2771)
        if not load_shared(s_refine_overflow):                           # (:2772) the whole
                                                                         # round loop is GUARDED
            for round in unroll(4):                                      # (:2774)
                r_idx  = round % 2                                       # (:2775) buffer index
                offset = FIRST_SHIFT - round * 8                         # (:2776)
                if run_refine_round(r_idx, offset, is_last=(round == 3)):
                    det_stop_round = round; break                        # (:2778-2787)
                if load_shared(s_refine_overflow): break                 # (:2788-2790)
        # (:2792) the guarded round loop CLOSES here.  The deterministic collect below is at
        # the OUTER level and carries its OWN re-check, because run_refine_round can set
        # s_refine_overflow mid-loop via the atomicOr at :2667; the source comments exactly this
        # at :2798-2799 ("intentionally separate from the first if (!s_refine_overflow)").
        # Nesting it inside the pre-loop guard would let a stale collect write tie fillers that
        # the fallback then re-collects at a different offset.
        if DETERMINISTIC:                                                # (:2793)
            if not load_shared(s_refine_overflow):                       # (:2794) SEPARATE check
                collect_det_eq_pivot(pivot=build_det_pivot(det_stop_round),
                                     eq_needed=topk)                     # (:2795)
        if load_shared(s_refine_overflow):                               # (:2800) FALLBACK
            # 32-bit pivot rebuild.  static_assert(sizeof(OrderedType) == 4) at :2801-2802.
            topk_remain = topk_after_coarse                              # (:2804)
            threshold_bytes = [0xFF, 0xFF, 0xFF, 0xFF]                   # (:2805-2810)
            stop_round = NUM_ROUNDS - 1
            for round in unroll(4):                                      # (:2812)
                if tx < RADIX + 1: store_shared(s_hist[tx], 0)           # (:2816)
                barrier()                                                # (:2817)
                for v, i in for_each_score():                            # (:2819-2837)
                    ordered = to_ordered(v)
                    if to_coarse(v) == threshold_bin and prefix_matches(ordered, threshold_bytes, round):
                        atomic_add_shared(s_hist, (ordered >> (24 - round*8)) & 0xFF, 1)
                        # instruction_selection: ld.global.nc.b32 + ToOrdered + shr + and +
                        #   setp chain + atom.shared.add.u32; extent: one per row element
                barrier()                                                # (:2839)
                run_cumsum()                                             # (:2841)
                if tx < RADIX and crossing_predicate:                    # (:2842-2846)
                    store_shared(s_threshold_bin_id, tx)                 # writes ONLY the bin id
                barrier()                                                # (:2846)
                threshold_bytes[round] = load_shared(s_threshold_bin_id)
                topk_remain -= load_shared(s_hist[threshold_bytes[round] + 1])   # (:2850)
                barrier()                                                # (:2852)
                if topk_remain == 0: stop_round = round; break           # (:2853-2856)
            pivot = assemble(threshold_bytes, stop_round)                # (:2859-2868) byte[round]
                                                                         # is forced to 0xFF when
                                                                         # topk_remain == 0 and
                                                                         # round > stop_round
                                                                         # (:2864-2866)
            if tx == 0:                                                  # (:2873-2876)
                store_shared(s_counter, 0); store_shared(s_last_remain, topk_remain)
            barrier()                                                    # (:2877)
            for v, i in for_each_score():                                # (:2883-2895)
                # three-way dispatch, NOT a single collector call:
                if to_coarse(v) > threshold_bin:                         # (:2885-2889)
                    collect_gt_and_nondet_eq_threshold(to_coarse(v), threshold_bin, i,
                                                       allow_eq_claim=False)
                    continue                                             # compares COARSE keys
                elif to_coarse(v) != threshold_bin:                      # (:2890-2892)
                    continue                                             # dropped
                collect_gt_and_nondet_eq_threshold(to_ordered(v), pivot, i,   # (:2893-2894)
                                                   allow_eq_claim=(eq_needed > 0))
                # compares the FULL 32-bit ordered key against the rebuilt pivot
            barrier()                                                    # (:2897)
            if DETERMINISTIC:
                collect_det_eq_pivot(pivot=pivot, eq_needed=topk_remain) # (:2898-2900)

    # ===================================================================
    # Stage 4: tie handling at the pivot -- SHARED HELPER DESCRIPTION.
    # The four call sites are shown above at their source positions inside the
    # Stage-3 branch structure (:2752-2757, :2763-2765, :2793-2797, :2898-2900),
    # each with its own pivot source and eq_needed source.
    # ===================================================================
    # NON-DETERMINISTIC (:2571-2578) -- inside collect_gt_and_nondet_eq_threshold:
    #   equal-to-pivot elements race for the remaining slots, filling s_indices from the BACK:
    #       pos = atomic_add_shared(s_last_remain, -1)      # atom.shared.add.u32 (negative)
    #       if pos > 0: store_shared(s_indices[top_k - pos], idx)
    #   So s_indices[0:s_counter) are strict winners and the tail holds tie fillers; WHICH ties
    #   win is racy.  This branch is `else if constexpr (!DETERMINISTIC)` -- it does not exist
    #   in the deterministic instantiations.
    #
    # DETERMINISTIC (:2582-2608) -- collect_det_eq_pivot(pivot, eq_needed):
    def collect_det_eq_pivot(pivot, eq_needed):   # (:2582-2608), DETERMINISTIC only
      if eq_needed > 0:
        if TIE_BREAK == Small:
            deterministic_contiguous_collect(REVERSE=False)   # (:2591-2595) -> smallest indices
        elif TIE_BREAK == Large:
            deterministic_contiguous_collect(REVERSE=True)    # (:2596-2600) -> largest indices
        else:
            deterministic_thread_strided_collect()            # (:2601-2606)
        # predicate is always `to_ordered(score[idx]) == pivot`; emit target is
        # s_indices[top_k - eq_needed + local_pos] (:2587-2590).
        # instruction_selection: cub BlockScan ExclusiveSum -> 5x shfl.sync.up.b32 per warp
        #   scan + bar.sync 0; extent: the export shows shfl.sync.up.b32 = 10 on every
        #   DETERMINISTIC instantiation and 0 on every non-deterministic one.
        # contiguous variant (:298-377): ITEMS_PER_THREAD=4, CHUNK=4096, a quota walk with
        #   shared s_emitted/s_chunk_base/s_chunk_take and an early break at the limit.
        # instruction_selection: the predicate re-reads the row: ld.global.nc.b32|b16
        #   (.loc 2 2594/2599/2604) + the ToOrdered sequence + setp.eq; extent: one per
        #   scanned index.
        # build_det_pivot(stop_round) (:2534-2547) reassembles the pivot from
        #   s_refine_thresholds: 16-bit -> (threshold_bin << 8) | s_refine_thresholds[0];
        #   32-bit -> OR of s_refine_thresholds[round] << (24 - round*8) for round <=
        #   stop_round, 0xFF for later rounds.

    # ===================================================================
    # Stage 5: output (:2905-2918).  All top_k slots are filled on this path
    # (strict winners + tie fillers sum to exactly top_k), so there is NO -1
    # padding here -- only the trivial path pads.
    # ===================================================================
    OUTPUT:
    for base in serial(tx, top_k, step=BLOCK, unroll=2):                 # (:2906)
        idx = load_shared(s_indices[base])
        if MODE == Plain:
            store_global(dst[base], idx)                                 # (:2910)
            store_global(dst_values[base], load_global(score[idx]))      # (:2911)
            # instruction_selection: st.global.b32 (.loc 2 2910) +
            #   ld.global.nc.b32|b16 (.loc 2 2911) + st.global.b32|b16
        elif DETERMINISTIC:
            store_global(dst[base], idx)                                 # (:2913) transform
                                                                         # DEFERRED to finalize
            # instruction_selection: st.global.b32; extent: one scalar store
        elif MODE == PageTable:
            store_global(dst[base], load_global(src_page_entry[page_start + idx]))   # (:2915)
            # instruction_selection: ld.global.nc.b32 (.loc 2 2915) + st.global.b32;
            #   extent: one gather pair
        else:
            store_global(dst[base], idx + offset_val)                    # (:2917)
            # instruction_selection: add.s32 + st.global.b32; extent: one scalar store
```

## Complete sketch — finalize kernel

```python
# Launched only when `finalize_plan` says so; see the matrix below.
launch_config = launch(grid=(NUM_ROWS,1,1), block=(BT,1,1), dynamic_smem_bytes=0,
                       launch_bounds=BT)
# BT, IPT from the k ladder (:3061-3079); end_bit = bit_length(max_len), static here
# (the reference computes it at runtime -- the export shows clz.b32 = 1 per instantiation).

def filtered_topk_finalize(out_idx, out_val, aux, page_table_row_starts, row_to_batch, aux_stride):
    row = cta_id(axis="x", extent=NUM_ROWS); tx = thread_id(extent=BT)
    row_output = out_idx + row * top_k                                   # (:2971)
    keys   = alloc_local(IPT, "uint32")
    values = alloc_local(IPT, DTYPE)          # Plain only

    # ---- load, blocked arrangement, ~0u padding (:2976-2991) -------------
    for i in unroll(IPT):
        pos = tx * IPT + i
        if pos < top_k:
            idx = load_global(row_output[pos])
            keys[i] = select(idx >= 0, idx, 0xFFFFFFFF)                  # (:2980)
            # instruction_selection: ld.global.b32 + **max.s32 (imm -1)** (.loc 2 2981) --
            #   nvcc folds the `(idx >= 0) ? idx : ~0u` clamp into a single max; there is no
            #   setp/selp pair.  Note the finalize kernel's pointers are NOT __restrict__, so
            #   its loads are plain `ld.global.b32`, unlike the unified kernel's `.nc` forms.
            if MODE == Plain: values[i] = load_global(out_val[row*top_k + pos])
        else:
            keys[i] = 0xFFFFFFFF; values[i] = 0                          # (:2986-2990)
    # PADDING INVARIANT: the sort covers [0, end_bit) and `~0u` truncated to end_bit bits is
    # 2^end_bit - 1, while every valid index is <= max_len - 1 < 2^end_bit - 1.  So padding is
    # strictly maximal in EVERY digit pass and sorts last.  The full 0xFFFFFFFF survives in the
    # untouched high bits, which is what the `idx != ~0u` guards below still test.

    # ---- block radix sort, ascending over [0, end_bit) (:2993-2999) ------
    if SORT:
        for (begin_bit, pass_bits) in static_passes(end_bit):   # 2..5 passes, RADIX_BITS=4
            rank_keys(keys, ranks, begin_bit, pass_bits)
            # rank = reset 9-lane counter grid; per-item u16 counter RMW (each thread owns its
            #   tid column, NO atomics); memoized 9-word raking upsweep; BlockScan ExclusiveSum
            #   with the `block_prefix = aggregate << 16` packing trick; exclusive downsweep;
            #   ranks[i] = thread_prefix[i] + reloaded counter.
            # instruction_selection: ld/st.shared.b32 + ld/st.shared.b16 + 5x shfl.sync.up.b32
            #   + bar.sync 0 x2 inside the scan; extent: one per digit pass.
            #   Digit extraction is shr + and (cub's bitfield_extract falls through to
            #   shift+mask on SM>=70), NOT bfe.  No match.any, no ballot, no PRMT anywhere.
            barrier()
            scatter_to_blocked(keys, ranks)                     # blocked -> blocked
            # instruction_selection: st.shared.b32 to pad(rank) + bar.sync 0 + ld.shared.b32
            #   from pad(tx*IPT+i); pad(x) = x + (x>>5) iff IPT == 8.  The export shows
            #   ld.shared.v4.b32 = 2 on (256,8) as well as (32,4), i.e. nvcc vectorizes part
            #   of the gather at IPT=8 too.
            if MODE == Plain:
                barrier(); scatter_to_blocked(values, ranks)
            if not last_pass: barrier()
    # sync accounting: keys-only 7P-1, key-value 9P-1 (P = pass count).
    # When SORT is False the whole block above vanishes: the export's keys-only PageTable
    # SORT=false instantiation has ZERO shared traffic and ZERO bar.sync -- it is a
    # transform-only kernel, not a sort with a flag off.

    # ---- deferred transform + writeback (:3002-3029) ---------------------
    if MODE == PageTable:
        batch = load_global(row_to_batch[row]) if row_to_batch else row
        src_page_entry = aux + batch * aux_stride
        page_start = load_global(page_table_row_starts[row]) if pts else 0
    elif MODE == Ragged:
        offset = load_global(aux[row])
    for i in unroll(IPT):
        pos = tx * IPT + i
        if pos < top_k:
            idx = keys[i]
            if MODE == Plain:
                store_global(row_output[pos], idx)          # ~0u reinterprets to -1 for free
                store_global(out_val[row*top_k + pos], values[i])
            elif MODE == PageTable:
                store_global(row_output[pos],
                             select(idx != 0xFFFFFFFF, load_global(src_page_entry[page_start+idx]), -1))
                # instruction_selection: setp.eq.b32 + `@p bra` over ld.global.b32 +
                #   st.global.b32 (.loc 2 3023) -- a branch, not a select
            else:
                store_global(row_output[pos], select(idx != 0xFFFFFFFF, idx + offset, -1))
                # instruction_selection: setp.eq.b32 + add.s32 + selp.b32 + st.global.b32
                #   (.loc 2 3026) -- this one IS a select, because there is no load to guard
            # Plain writes both stores unguarded; `~0u` reinterprets to -1 for free.
```

## Finalize launch matrix (verified in all three dispatchers)

| runtime flags | Plain (:3460-3464) | PageTable (:3392-3402) / Ragged (:3428-3437) |
| --- | --- | --- |
| det=F, tie=None | main only | main only |
| det=T (any tie) | + finalize SORT=true, keys+values | + finalize SORT=true, keys-only |
| det=F, tie=Small/Large | **main only** | + finalize SORT=false (transform only) |

Early-outs: `k == 0` -> none; Plain and `k <= 1` -> none (:3043-3047); `k > 2048`
-> `cudaErrorInvalidValue` (:3040). The asymmetry in the last row is real: a
tie-break-only Plain config gets no second launch, because Plain has nothing to
defer.

## Shared-memory budget

| region | bytes | source |
| --- | ---: | --- |
| `s_input_idx[2][16384]` (dynamic) | 131072 | :2343-2344 |
| `s_histogram_buf[2][384]` | 3072 | :2432 |
| `s_indices[2048]` | 8192 | :2438 |
| scalars (`s_counter`, `s_threshold_bin_id`, `s_refine_thresholds[4]`, `s_num_input[2]`, `s_refine_overflow`, `s_last_remain`) | 40 | :2433-2441 |
| cub BlockScan TempStorage (DETERMINISTIC only) | ~4256 | :2584-2585 |
| `s_emitted` / `s_chunk_base` / `s_chunk_take` (TIE_BREAK Small\|Large only) | 12 | :313-315 |
| **total** | **~146644** vs 232448 optin on B200 | |

The 128 KiB arena is a compile-time constant, never scaled by `max_len` or `k`.
That is why `CanImplementFilteredTopK` (:3285-3294) gates on
`maxSharedMemoryPerMultiprocessor >= 131072` rather than anything shape-dependent,
and why SM89/SM120 cannot run this path at all.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DETERMINISTIC = deterministic \|\| tie_break != None` | static | selects the tie collector, enables `s_refine_thresholds`, defers the transform |
| `TIE_BREAK` | static | contiguous ascending / descending / thread-strided collector |
| `MODE` | static | header, trivial branch, epilogue, and whether finalize transforms |
| `VEC_SIZE` | static per config | staging load width; forced to 1 by `dsa_graph_safe` or by `row_starts` on a transform mode |
| `NUM_ROUNDS`, `FIRST_SHIFT` | static from DType | 4/24 for f32, 1/0 for 16-bit |
| dynamic smem = 131072 | static constant | independent of shape |
| `end_bit`, finalize `(BT, IPT)` | static per config here, runtime in the source | the digit-pass loop unrolls; `clz.b32` disappears |
| `s_refine_overflow` | **runtime, data-dependent** | the only branch this port cannot reach by shape alone — needs tie-heavy inputs |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "filtered_topk", "category": "flashinfer",
  "compute_capability": 10}`; device symbols `filtered_topk` and
  `filtered_topk_finalize`.
- Plain TIRx only: a `T.SMEMPool` arena at the source's byte offsets, explicit
  loops, and native `T.ptx.*` forms for every key operation. Global and shared
  memory are reached exclusively through `T.ptx.ld/st.*` on `buffer.ptr_to([...])`
  — never a native `BufferLoad`/`BufferStore`, which the repository's low-level IR
  contract forbids. No `T.cuda.func_call` (this kernel has no allowlist budget).
- The finalize kernel's rank counter grid is addressed as **both** `uint32`
  (packed raking adds) and `uint16` (per-digit RMW). There is no buffer re-view
  API in this tree; the union is built by allocating both dtypes at the same pool
  offset via `move_base_to`, verified on device (see
  `.porting/filtered_topk/probe_findings.md`). Pool offsets must be **Python
  literals** — capturing `pool.offset` inside the traced body makes it a TIR
  variable and `move_base_to` then fails in `__bool__`.
- No `Tx` tile primitives anywhere.
- Correctness compares against the same FlashInfer pipeline launched through the
  raw `topk` FFI module with `FLASHINFER_TOPK_ALGO=filtered`, which
  `GetTopKAlgoOverride` (:3299-3305) maps to `TopKAlgoOverride::FILTERED`. That
  pin skips the `num_rows`/`max_len` heuristics (:3346-3368) but not the hard
  preconditions, and it also keeps the SM100 cluster kernel out of the reference
  path. Comparison per flag combination: exact positional equality where the
  reference finalize-sorts; exact selected-index **sets** where a tie-break makes
  the set unique but the order racy; selected-value multisets otherwise.
- Both sides launch the same pipeline: one kernel where `finalize_plan` returns
  `None`, two where it does not. The canonical timer aggregates every kernel in a
  closure (probe: 58.3 / 92.7 / 152.1 us for 1 / 2 / 3 launches).

## Instruction selection is a lowering consequence

| Primitive / pattern | PTX family (fresh sm_100a export) |
| --- | --- |
| `ToCoarseKey`, f32 (`.loc 2 2284/2287`) | `cvt.rn.f16.f32`, then `not.b16` + `or.b16` + `setp.lt.s16` + `selp.b16` + `shr.u16` |
| `ToCoarseKey`, 16-bit (`.loc 2 2309`) | same 16-bit sequence, no `cvt` |
| `ToOrdered`, f32 (`.loc 2 2293`) | `not.b32` + `abs.ftz.f32` + `neg.ftz.f32` + `setp.lt.s32` + `selp.b32` |
| `ToOrdered`, 16-bit (`.loc 2 2315`) | `shr.s16` (sign broadcast) + `xor.b16` |
| histogram / counter bump | `atom.shared.add.u32` |
| non-deterministic tie back-fill | `atom.shared.add.u32` with `-1`, present only when DETERMINISTIC is false |
| overflow flag | `atom.shared.or.b32` (result discarded; not `red.shared.or.b32`) |
| ping-pong suffix scan | `ld.shared.b32` x2 + `add.s32` + `st.shared.b32` + `bar.sync 0`, 8 steps |
| threshold pick | three short-circuit `@p bra` guards (no `and.pred`) + `st.shared.b32` |
| tie collection (DETERMINISTIC) | cub BlockScan -> `shfl.sync.up.b32` x5 per warp scan |
| row scan (unified) | `ld.global.nc.v4.b32` (f32 v4 and 16-bit v8) / `nc.v2.b32` / `nc.v2.b16` / scalar `nc` tail |
| all unified global reads | `.nc` qualified — the inputs are `__restrict__ const` |
| finalize global reads | plain `ld.global.b32` — those pointers are not `__restrict__` |
| finalize pad clamp | `max.s32` with immediate `-1` (not `setp`+`selp`) |
| finalize rank | `ld/st.shared.b16` + `ld/st.shared.b32` + `shfl.sync.up.b32` |
| finalize digit extraction | `shr` + `and` (shift+mask on SM>=70, not `bfe`) |
| finalize exchange | `st.shared.b32` / `ld.shared.b32`, partly `ld.shared.v4.b32` |
| finalize writeback | PageTable: `setp.eq.b32` + `@p bra` + `ld.global.b32`; Ragged: `setp.eq.b32` + `add.s32` + `selp.b32` |

The two `ToOrdered` forms differ in *lowering* even though the source writes one
expression: on 16 bits nvcc recognises `(bits & 0x8000) ? ~bits : (bits | 0x8000)`
as a sign-broadcast XOR, and on 32 bits it routes the same shape through
`abs`/`neg`. The `or.b16`/`not.b16` counts in the table below belong to
**`ToCoarseKey`**, not to `ToOrdered`.

Static opcode counts per exported instantiation (full table in
`.porting/filtered_topk/ptx/opcode_table.md`):

| op | f32 v4 Plain nondet | f32 v4 Plain det | f32 v4 Plain tieSmall | f32 v1 Plain nondet | f16 v8 Plain nondet | f16 v8 Plain det |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `atom.shared.add.u32` | 237 | 219 | 219 | 51 | 255 | 225 |
| `atom.shared.or.b32` | 20 | 20 | 20 | 6 | 29 | 29 |
| `bar.sync` | 115 | 121 | 129 | 115 | 39 | 45 |
| `shfl.sync.up.b32` | 0 | 10 | 10 | 0 | 0 | 10 |
| `cvt.rn.f16.f32` | 154 | 154 | 154 | 24 | 0 | 0 |
| `xor.b16` | 0 | 0 | 0 | 0 | 99 | 99 |
| `or.b16` | 154 | 154 | 154 | 24 | 165 | 225 |

The `bar.sync` drop from 115 (f32, four refine rounds) to 39 (16-bit, one round)
is the refine structure showing through; tie-break adds a further 8 for the
chunked contiguous collector. `shfl.sync.up.b32` appearing only on DETERMINISTIC
instantiations is the cub BlockScan, which the non-deterministic tie race does not
use.
