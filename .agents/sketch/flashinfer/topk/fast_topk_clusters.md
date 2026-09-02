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
include/flashinfer/fast_topk_clusters_exact.cuh fast_topk_clusters_exact family.
-->

# fast_topk_clusters SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/flashinfer/topk/fast_topk_clusters.py`](../../../../tirx_kernels/flashinfer/topk/fast_topk_clusters.py).
That TIRx module is the authoritative implementation.

The port covers **three kernels**, because they are one program: a single device
worker `fast_topk_cuda_v4` (`:80-405`) behind three `__global__` wrappers that
differ only in where the row length comes from and what the epilogue does to each
index -- `fast_topk_clusters_exact_kernel` (`:407-449`),
`fast_topk_clusters_exact_page_table_transform_kernel` (`:451-494`), and
`fast_topk_clusters_exact_ragged_transform_kernel` (`:496-538`). All three carry
`__launch_bounds__(1024)` and `__cluster_dims__(NClusters, 1, 1)`. This is the
kernel [`filtered_topk.md`](filtered_topk.md) and
[`radix_topk_single_cta.md`](radix_topk_single_cta.md) name as out of scope --
`radix_topk_multi_cta.md` inherits that scope by deferring to its single-CTA
sibling rather than naming this kernel itself: the SM100 cluster algorithm, which the Python layer selects
*before* the FFI (`flashinfer/topk.py:502-507`, requiring `not deterministic`,
`tie_break == NONE`, `not dsa_graph_safe`, and compute capability major 10), so
the C++ `TopKDispatch` those three ports go through never reaches it.

Instantiations: `DType in {f32, f16, bf16}` x `NClusters in {1, 2, 4, 8}` x
`{plain int32, plain int64, page_table, ragged}` = **48 reachable
specializations**, all exported in `.porting/fast_topk_clusters/ptx/`. `TopK`,
`num_cached`, `NClusters`, the mode, and the index width are static per config
here, exactly the values the launchers (`:584-693`) and their Python caller pick
per call.

Accepted target SM100/B200. Out of scope, with the predicate that excludes each:
`PDL_ENABLED == true` -- `pdl` is `False` at every Python call site and no caller
passes `True` (`topk.py:404, 443, 473`), so `cudaGridDependencySynchronize`
(`:156`) is unreachable; the pre-computed histogram -- `pre_hist` is always
`None` (`topk.py:432, 463, 492`), so `:159-163` and the `sum_hist == false`
threshold path (`:187`) are dead; `NClusters` outside `{1, 2, 4, 8}` -- the
launcher rewrites it to `1` and re-dispatches before any kernel is selected
(`:607-611`). Tile (`Tx`) primitives are out of scope everywhere.

## Pipeline at a glance

One CTA owns one `(row, rank)` pair; `NC` CTAs form one cluster and cooperate on
one row of `logits`. The grid is `batch * NC`.

| Kernel / role | Program | Publication / reuse edges |
| --- | --- | --- |
| **All three wrappers**, all 32 warps, trivial branch | when `seq_len <= TopK`, write **this mode's** index for `i < seq_len` (plain: `i`, `:421`; page_table: `page_table[pt_off + i]`, `:471`; ragged: `i + ragged_offset`, `:515`) then `-1` padding, distributed across the cluster by `(blockIdx.x % NC) * 1024`, and return without entering the worker | Nothing is published. This is the **only** path whose output is positionally deterministic, and the only one that never touches shared memory. |
| **Worker**, all 32 warps, histogram | scalar strided read of this CTA's slice of the row, `to_ordered` then `>> LSHIFT_START`, bump `hist[bin]` | `atom.shared.add.u32` into bank 0. Bank 0 is the input to the first threshold, and then becomes this CTA's DSMEM publication when `:122` overwrites it with the local cumsum -- what peers read at `:129-130` is that republished cumsum, not the raw histogram. Protecting bank 0 from being cleared while peers are still reading it is what the split cluster barrier does. |
| **Worker**, lanes `< 256` (`radix_thread`), threshold | 256-bin suffix cumsum of one bank; at `NC > 1` add the same bin from every peer CTA through distributed shared memory; select the bin where the running count crosses `k_remaining` | `cluster.sync()` separates publication from the peer reads; bank 2 receives the cluster-wide sums; `threshold_bin` publishes the choice to the whole CTA through `bar.sync 0`. |
| **Worker**, 8 warps, inside the cumsum | each warp suffix-scans 32 bins by `shfl.sync.down.b32`, lane 0 publishes its warp total (a `.down` scan totals into the low lane), warp 0 scans the 8 totals | `s_cum_reduce_buf[8]` is the only cross-warp edge in the scan; one `bar.warp.sync` inside it. |
| **Worker**, all 32 warps, classification and refinement | split each element on the current byte: above the threshold bin, equal to it, or below | above -> `topk_inds` by `atom.shared.add.u32` compaction; equal -> the double-buffered candidate cache, spilling to this CTA's global overflow ring, and bumps the next round's histogram bank. Below is dropped. |
| **Worker**, all 32 warps, final round only | the equal-bin survivors race a DSMEM atomic on **rank 0's** counter for the remaining slots | `atom.shared::cluster.add.u32`; the first `top_k_remaining` arrivals win and the rest latch out. Present even at `NC == 1`. |
| **Worker**, thread 0 of ranks `> 0`, epilogue | claim a contiguous output range with a DSMEM atomic on rank 0's output cursor | `atom.shared::cluster.add.u32` between two `cluster.sync()` edges; the claimed base is broadcast to the CTA through shared memory. Rank 0 keeps base 0. |
| **All three wrappers**, all 32 warps, writeback | strided store of this CTA's slice of the result, guarded `offs < TopK` | plain re-gathers each value from global; page_table maps the index through the table; ragged adds the row's offset. |

There is no producer/consumer split, no warp specialization, no pipe, no
mbarrier, and no asynchronous copy anywhere in this family. Every warp runs the
same program and the roles above are lane predicates. The only asynchrony is the
**split cluster barrier** -- an `arrive` issued before a classification loop and
the matching `wait` issued at the top of the next round -- which exists so peers
may finish reading a histogram bank before it is cleared (`:218-225`, `:248`).
The candidate and overflow loops stride by `1024`, not by `1024 * NC`: each CTA
owns its own candidate slice and the cluster does **not** re-partition after the
first pass.

## Primitive vocabulary

```python
copy_g2r(reg, gmem)               # ld.global.nc.b32 | ld.global.nc.b16 on the
                                  #   READ-ONLY inputs -- logits, page_table,
                                  #   offsets, seq_lens -- which are __restrict__
                                  #   (:82), unlike stable_sort and like the
                                  #   filtered unified kernel.  ONE EXCEPTION:
                                  #   cached_overflow is written by this kernel,
                                  #   so reads of the ring are plain ld.global.b32
copy_g2r_v(dst, gmem, n)          # ld.global.nc.v4.b32 (f32, 16 B) |
                                  #   ld.global.nc.v2.b32 (16-bit, 8 B); vec_t<T,4>
                                  #   (:228-233), one issue, VEC_SIZE = 4 lanes
copy_r2g(gmem, reg)               # st.global.b32 | st.global.b64 (i64 indices) |
                                  #   st.global.b16 (16-bit values)
copy_s2r(reg, smem) / copy_r2s    # ld.shared.b32 / st.shared.b32, always T.ptx,
                                  #   through the [base+imm] form when the
                                  #   target is `shared_hist` or a tail scalar,
                                  #   and at displacement 0 for the runtime-
                                  #   indexed `topk_inds` / `cached_*` -- see
                                  #   the addressing note below
atomic_add_shared(smem, v)        # atom.shared.add.u32, returns old (:178, :194, ...)
to_ordered(x)                     # RadixTopKTraits::ToOrdered, the XOR form
                                  #   (topk_common.cuh:35-39); already ported as
                                  #   topk_radix.to_ordered_u32/u16
digit(bits, shift)                # shift then mask -- but the emitted form
                                  #   depends on the CALL SITE and on the shift
                                  #   VALUE, not on the dtype alone.  There are
                                  #   three sites (histogram seed, classify,
                                  #   next-bank seed) and they lower differently;
                                  #   each is annotated where it is called.
                                  #   The CLASSIFY site emits nothing when its
                                  #   shift is 0 (nvcc narrows the candidate load
                                  #   to b8 and compares the byte directly).  The
                                  #   two HISTOGRAM-BUMP sites always emit the
                                  #   x4 bin-to-byte scaling, even at shift 0 --
                                  #   but not always a mask.  `and.b32 1020`
                                  #   accompanies it everywhere EXCEPT 16-bit
                                  #   :178, where `mul.wide.u16 ..., 4` consumes
                                  #   the u16 left by `shr.u16 8` at :177, which
                                  #   is already <= 255, so the x4 product cannot
                                  #   exceed 1020 and no mask is needed.
                                  #   NEVER bfe on SM >= 70
warp_suffix_scan_step(v, d)       # ONE shfl.sync.down.b32, mask -1, clamp 31.
                                  #   The caller supplies the explicit d-loop and
                                  #   the `lane < width - d` guard (:24-28), so
                                  #   the 32-wide instance is five steps and the
                                  #   8-wide instance is three.  A SUFFIX scan,
                                  #   hence .down (:21-31), which is why the
                                  #   total lands in lane 0
barrier()                         # bar.sync 0
warp_barrier()                    # bar.warp.sync, once inside cum_sum (:48)
map_peer(smem, rank)              # cvta_generic_to_shared + mapa.shared::cluster.u32
                                  #   -- produces an address, moves nothing
                                  #   (:129, :309, :391).  SPELLING: the port
                                  #   emits the 32-bit shared-window form, which
                                  #   is the form this repo can construct; the
                                  #   reference emits the 64-bit generic form
                                  #   in the OPPOSITE order: mapa.u64 first
                                  #   (generic -> peer generic), then
                                  #   cvta.to.shared::cluster.u64 (generic ->
                                  #   cluster window).  Verified def-use over
                                  #   the whole export: all 468 cvta consume a
                                  #   preceding mapa result, 0 exceptions.
                                  #   Both name the same byte -- one carries the
                                  #   peer address in the shared window, the
                                  #   other in the generic window.  The reference
                                  #   spelling appears below only as PTX evidence
copy_peer_s2r(reg, peer)          # ld.shared::cluster.b32 -- see below
atomic_add_peer(peer, v)          # atom.shared::cluster.add.u32, returns old
cluster_arrive() / cluster_wait() # barrier.cluster.arrive / .wait, the SPLIT form
cluster_sync()                    # barrier.cluster.arrive + .wait, the FUSED form
```

There is deliberately no compound `radix_select`, `histogram`, `threshold`,
`refine`, `cluster_reduce`, `ration_ties`, tile-copy, or tile-compute primitive.
`threshold(...)` appears in the body as a named region for readability, but every
COPY, COMPUTE and SYNC operation inside it is drawn from the list above and is
written out in full at its first use. Plain scalar arithmetic (`add`, `sub`,
`min`, `max`), register declarations (`reg`), compile-time loops (`unroll`) and
the structural views (`ring`, `phase_half`) are ordinary Python and carry no
`instruction_selection` -- they are the incidental plumbing the reviewer bound
excludes.

### Peer shared memory is a different address space

`map_peer` is a separate primitive from the access that follows it, and the two
must agree. `mapa` returns an address in `shared::cluster`; reaching it with an
own-CTA `ld.shared.b32` is an **illegal instruction**, not a wrong value -- an
asynchronous trap with no attribution to the offending access. The reference's
own export pairs them correctly and never mixes them:

The `PTX in the export` column is the **reference's** spelling, and its order is
`mapa.u64` -> `cvta.to.shared::cluster.u64` -> `ld.shared::cluster.b32` /
`atom.shared::cluster.add.u32`. The port emits the two address steps in the
opposite order -- `cvta_generic_to_shared` then `mapa.shared::cluster.u32`, the
32-bit shared-window form, which is what this repo can construct -- for the same
peer byte. The load and atomic opcodes are identical on both sides; only the
address-forming pair, and its order, differ.

| operation | source | PTX in the reference export |
| --- | --- | --- |
| peer histogram read | `map_shared_rank(&hist[hist_idx * 256 + tid], r)[0]` (`:129-130`) -- the PING-PONG bank, not bank 2 | `mapa.u64` -> `cvta.to.shared::cluster.u64` -> `ld.shared::cluster.b32` |
| tie rationing | `atomicAdd(map_shared_rank(s_k_remaining_counter, 0), 1)` (`:309`, `:360`) | `mapa.u64` -> `cvta.to.shared::cluster.u64` -> `atom.shared::cluster.add.u32`. The mapa is CSE'd *within* each tie loop across its unrolled copies, and across the *two* tie sites only at f32 -- f32 NC=1 is 1 mapa / 2 atomics, 16-bit NC=1 is 2 mapa / 10 atomics -- so the atomics always outnumber it |
| output range claim | `atomicAdd(map_shared_rank(shared_final_idx_count, 0), n)` (`:391`) | `mapa.u64` -> `cvta.to.shared::cluster.u64` -> `atom.shared::cluster.add.u32`, with a REGISTER value operand (`topk_num`), unlike the tie sites' immediate `1` |

The atomics are safe by construction because the space is part of the opcode
name; only the load can be spelled wrongly, and `.porting/fast_topk_clusters/probe_findings.md`
records that trap being reproduced and fixed before any kernel code existed.

### Shared accesses carry the carve offset in the addressing mode

The reference holds one base per carve block and lets the offset ride in the
instruction:

```
st.shared.b32        [%r+1024], %r;      # hist bank 1
atom.shared.add.u32  %r, [%r+3072], 1;   # final_idx_count
ld.global.nc.b32     %r, [%rd+-114688];  # the global side does it too
```

`copy_s2r` / `copy_r2s` / `atomic_add_shared` mean that form wherever the target
sits at a constant offset within an already-materialized block base -- the
`shared_hist` banks and the tail scalars. `topk_inds[slot]` and `cached_*[slot]`
get their own base plus a runtime element index, so the reference leaves them at
displacement 0 and the port follows.

`T.ptx.addr(base, byte_offset)` reaches this form: it builds `[base+byte_offset]`
for any operand slot declared `kind="addr"` with `allow_imm_offset=True`, which is
how `atom`, `ld` and `st` declare their address operand. `byte_offset` is a byte
displacement, not an element index, and cannot be nested.

The reference's IMMEDIATE INCREMENT is not reachable: it writes
`atom.shared.add.u32 %r, [%r+3072], 1`, but `atom`'s `value` is a plain register
slot with no `kind="imm"`, so the port emits `..., %r`. Opcode and count match;
only that operand differs, and it is not a port error.

Measured displacement sets, per-entry access counts, the mode/index spread and
the two mechanisms behind it are in
`.porting/fast_topk_clusters/probe_findings.md`, re-derivable with
`check_sketch_counts.py`. They are evidence for this rule, not part of the
design.

### The tie-rationing atomic is not guarded by cluster width

`atom.shared::cluster.add.u32` appears **twice** in the f32 `NClusters == 1`
export (and **ten** times in the 16-bit one), even though those entries have zero
cluster barriers and zero peer loads. The mapping is CSE'd rather than paired
1:1 with the atomics: f32 `NC = 1` is 2 atomics behind **1** `mapa`, and f16
`NC = 1` is 10 atomics behind **2**. The address is CSE'd *within* each tie loop
across its unrolled copies; it is additionally CSE'd *across* the two tie sites
only at f32. That is exactly why the 16-bit entry shows two -- one feeding the
five shared-cache-loop atomics (`:309`) and one feeding the five overflow-loop
atomics (`:360`). The histogram peer-sum (`:121-131`) and the epilogue range claim
(`:379-399`) are both inside `if (NClusters > 1)`; the two tie sites (`:309`,
`:360`) are not -- they call `map_shared_rank(..., 0)` unconditionally, which at
one CTA per cluster maps to the CTA itself. A port that hoists all DSMEM behind
`NC > 1` diverges from the reference on every single-cluster shape, which is the
whole large-batch and short-row half of the matrix.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
@specialize(
    DTYPE=("f32", "f16", "bf16"),          # ROUNDS = sizeof(T), LSHIFT_START = 8*sizeof(T)-8
    MODE=("plain", "page_table", "ragged"),
    IDX=("i32", "i64"),                    # i64 only reachable when MODE == "plain"
    NC=(1, 2, 4, 8),                       # NClusters, a template parameter (:598-605)
    TOPK="positive compile-time integer",
    NUM_CACHED="positive compile-time integer",
    TARGET="sm_100a",
)
# instruction_selection: none; extent: 48 compile-time instantiations
@launch(
    grid=(batch * NC, 1, 1),               # :615
    block=(1024, 1, 1),                    # :571, __launch_bounds__(1024)
    cluster=((NC, 1, 1) if NC > 1 else None),   # :408, :563-565
    dynamic_smem_bytes=16 * NUM_CACHED + 4 * TOPK + 3124,   # :579-582
)
# instruction_selection: none; extent: static launch metadata.  Codegen emits no
#   __cluster_dims__; the cluster dimension reaches the driver as
#   CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, exactly as the source's own
#   cudaLaunchKernelExC does (:563-576).  The source's explicit
#   cudaFuncAttributeMaxDynamicSharedMemorySize / carveout calls (:557-558) have
#   no counterpart: the runtime opts in past 48 KB on its own.
def fast_topk_clusters(
    logits,           # DTYPE [batch, seq_len], __restrict__, row stride logit_stride
    output_indices,   # IDX   [batch, TOPK],  row stride indices_stride
    output_values,    # DTYPE [batch, TOPK] or None -- MODE == "plain" only
    seq_lens,         # i32   [batch]        -- MODE != "plain" only
    page_table,       # i32   [batch, pt_stride]  -- MODE == "page_table" only
    offsets,          # i32   [batch]        -- MODE == "ragged" only
    cached_overflow,  # i32   [batch, 4 * overflow_stride * NC], scratch, uninitialized
    overflow_stride: i32, logit_stride: i32, indices_stride: i32,
    pt_stride: i32, seq_len_scalar: i32,
):
    tid  = thread_id(extent=1024)          # instruction_selection: mov.u32 %tid.x
    cta  = cta_id(axis="x", extent=batch * NC)   # instruction_selection: mov.u32 %ctaid.x
    rank = cta_id_in_cluster(extent=NC)
    # instruction_selection: mov.u32 %cluster_ctarank; extent: one scalar.
    #   MUST stay live even at NC == 1 and even where the value folds: the
    #   resolved launch params carry the cluster dimension only when the body
    #   binds the scope.
    row  = cta // NC                       # :414, cluster_id
    warp, lane = tid >> 5, tid & 31

    logit_base = row * logit_stride        # :415
    ind_base   = row * indices_stride      # :416
    seq_len    = seq_len_scalar if MODE == "plain" else copy_g2r(seq_lens[row])
    # instruction_selection: ld.global.nc.b32 on the transform modes only
    #   (:466, :509); extent: one scalar per CTA
    if MODE == "ragged":
        ragged_offset = copy_g2r(offsets[row])
        # instruction_selection: ld.global.nc.b32; extent: ONE scalar per CTA.
        #   Hoisted above the trivial branch exactly as the source does (:510),
        #   and reused by both the trivial branch and the writeback -- neither
        #   re-loads it, so this is not a per-slot or per-result read.

    # =======================================================================
    # Trivial branch (:417-429 / :467-476 / :511-520)
    # =======================================================================
    # Distributed across the cluster, and the only positionally deterministic
    # output in the family.  Note it does NOT wait on PDL -- the wait lives
    # inside the worker at :156 -- which is one reason PDL is out of scope.
    if seq_len <= TOPK:
        for i in range(tid + (cta % NC) * 1024, TOPK, 1024 * NC):
            if i < seq_len:
                if MODE == "plain":
                    out = i
                elif MODE == "page_table":
                    out = copy_g2r(page_table[row * pt_stride + i])
                    # instruction_selection: ld.global.nc.b32; extent: one per slot
                else:
                    out = i + ragged_offset      # hoisted at CTA entry (:510)
                copy_r2g(output_indices[ind_base + i], out)
                # instruction_selection: st.global.b32 | st.global.b64 by IDX;
                #   extent: one per in-range slot
                if MODE == "plain" and output_values is not None:
                    copy_r2g(output_values[ind_base + i], copy_g2r(logits[logit_base + i]))
                    # instruction_selection: ld.global.nc.{b32,b16} then
                    #   st.global.{b32,b16}; extent: one pair per in-range slot
            else:
                copy_r2g(output_indices[ind_base + i], -1)
                # instruction_selection: st.global.b32 | st.global.b64; extent:
                #   one per pad slot.  NOTE output_values is left UNTOUCHED over
                #   the pad region (:425-427) -- the port must not zero it.
        return

    # =======================================================================
    # Shared carve, in source declaration order (:95-106)
    # =======================================================================
    smem           = raw_shared(16 * NUM_CACHED + 4 * TOPK + 3124, align=128)
    cached_bits    = view(smem, "u32", 0,                       2 * NUM_CACHED)
    cached_indices = view(smem, "i32", 8 * NUM_CACHED,          2 * NUM_CACHED)
    topk_inds      = view(smem, "i32", 16 * NUM_CACHED,         TOPK)
    hist           = view(smem, "i32", 16 * NUM_CACHED + 4*TOPK, 3 * 256)
    final_count    = view(smem, "i32", ..., 1)   # output cursor; rank 0's copy is
                                                 #   also the cluster's range allocator
    num_cached_cnt = view(smem, "i32", ..., 2)   # one cursor per phase
    threshold_bin  = view(smem, "i32", ..., 1)
    cum_reduce_buf = view(smem, "i32", ..., 8)   # one slot per warp
    k_rem_counter  = view(smem, "i32", ..., 1)   # rank 0's copy rations the ties
    # instruction_selection: none; extent: dynamic shared layout.
    #   cached_bits/cached_indices are DOUBLE-BUFFERED: half `ph`, half `ph^1`.
    #   hist has THREE banks: 0/1 ping-pong per round; bank 2 receives this
    #   CTA's cluster-wide cumsum and is read ONLY LOCALLY (written
    #   :132 on the cluster path and :136 on the single-CTA path,
    #   read :142 / :151).  Note :149 reads `shared_threshold_bin`
    #   at +3084, not bank 2.
    #   The bank peers read is the PING-PONG bank `hist_idx`, republished with
    #   the local cumsum at :122 just before the cluster.sync at :124 --
    #   `map_shared_rank(&shared_hist[hist_idx * 256 + threadIdx.x], ...)` at
    #   :129-130.  Bank 2 is not even written until AFTER the peer reads, so
    #   mapping the peer read onto it would read uninitialized memory.

    # This CTA's slice of the global overflow ring (:433).  Per CTA, not per
    # cluster: 4*overflow_stride ints = 2 phases x overflow_stride {bits,index}
    # pairs.  The header comment at :71-74 describes a struct-of-arrays layout;
    # the code implements array-of-structs and the code is authoritative.
    ring_base = cta * overflow_stride * 4
    ring = lambda ph: ring_base + ph * overflow_stride * 2

    # =======================================================================
    # Phase 0: seed the histogram (:158-180)
    # =======================================================================
    if tid < 256:
        copy_r2s(hist[tid], 0)
        copy_r2s(hist[256 + tid], 0)
        # instruction_selection: st.shared.b32; extent: two per radix lane
    if tid == 0:
        copy_r2s(final_count, 0)
        copy_r2s(k_rem_counter, 0)
        copy_r2s(num_cached_cnt[0], 0)
        # instruction_selection: st.shared.b32; extent: three scalars.
        #   HAZARD: threshold_bin is deliberately NOT initialized here, matching
        #   :166-170 exactly.  A round in which no lane satisfies the crossing
        #   test reads the previous round's value.  Reproduce, do not repair.
    barrier()
    # instruction_selection: bar.sync 0; extent: CTA-wide

    for i in range(tid + rank * 1024, seq_len, 1024 * NC):
        x = copy_g2r(logits[logit_base + i])
        # instruction_selection: ld.global.nc.b32 (f32) | ld.global.nc.b16
        #   (16-bit); extent: one scalar per iteration.  SCALAR, not vectorized:
        #   the source's first pass is a plain strided read (:173-179) and only
        #   the classification pass uses vec_t.  The row is therefore read from
        #   global TWICE in total.
        atomic_add_shared(hist[digit(to_ordered(x), LSHIFT_START)], 1)
        # instruction_selection: atom.shared.add.u32; extent: one per element.
        #   The digit extraction HERE is not the classify form: the x4 int
        #   scaling of the bin index folds into it, so f32 emits
        #   `shr.u32 22` + `and.b32 1020` (:178) -- a mask IS present, and the
        #   immediate is 1020, not 0xff -- and 16-bit emits `shr.u16 8`
        #   followed by `mul.wide.u16 ..., 4`, with no `cvt.u32.u16` at all.
```

```python
    # =======================================================================
    # threshold(bank, k_rem, sum_across_cluster) -- called once per round (:116-154)
    # =======================================================================
    # Written out here at first use.  It carries every peer read in the kernel.
    def threshold(bank, k_rem, sum_across_cluster=True):
        barrier()
        # instruction_selection: bar.sync 0; extent: one per call

        # --- 256-bin suffix cumsum, 8 warps x 32 bins (cum_sum, :33-58) -----
        v = reg(0)                     # meaningful only for tid < 256
        if tid < 256:                  # :38 -- warps 8..31 do NOT scan at all
            v = copy_s2r(hist[bank * 256 + tid])
            # instruction_selection: ld.shared.b32; extent: one per radix lane
            for d in (1, 2, 4, 8, 16):
                p = warp_suffix_scan_step(v, d)
                # instruction_selection: shfl.sync.down.b32, mask -1, clamp 31;
                #   extent: five steps at THIS site, and only for warps 0..7.
                #   A threshold() call issues 8 shuffles in total -- these 5
                #   plus the 3 of the warp-0 scan below -- not 8 here and not
                #   8 per warp.  .down
                #   because this is a SUFFIX scan: the count of everything at
                #   or above a bin (:26).
                if lane < 32 - d:
                    v = add(v, p)
            if lane == 0:
                copy_r2s(cum_reduce_buf[warp], v)
                # instruction_selection: st.shared.b32; extent: one per warp.
                #   LANE 0, not lane 31: a .down (suffix) scan accumulates
                #   toward the low lanes, so the warp's total over its 32 bins
                #   lands in lane 0.  Lane 31 holds only bin 255's own count.
        barrier()
        if warp == 0:
            w = copy_s2r(cum_reduce_buf[lane]) if lane < 8 else 0
            # instruction_selection: ld.shared.b32; extent: one per lane of warp 0
            for d in (1, 2, 4):
                p = warp_suffix_scan_step(w, d)
                # instruction_selection: shfl.sync.down.b32; extent: three steps
                if lane < 8 - d:
                    w = add(w, p)
            warp_barrier()
            # instruction_selection: bar.warp.sync; extent: one per call, and
            #   only for warp 0 (:48).  It orders the 3-step scan's shuffles
            #   against the write-back into the SAME buffer, so it must sit
            #   between them -- moving it after the store would leave the
            #   buffer it exists to protect unprotected.
            if lane < 8:
                copy_r2s(cum_reduce_buf[lane], w)
                # instruction_selection: st.shared.b32; extent: one per valid
                #   slot (:49-51).  The guard is load-bearing, not cosmetic:
                #   cum_reduce_buf is 8 ints at +3088 and k_remaining_counter is
                #   at +3120, both measured from `shared_hist` (pool-relative
                #   they are 16*nc + 4*TopK + 3088 / + 3120).  The 32 B between
                #   them is what matters: an unguarded 32-lane store writes
                #   128 B where 32 B are allocated, running 96 B past the buffer
                #   and over k_remaining_counter.
        barrier()
        if warp < 7:
            v = add(v, copy_s2r(cum_reduce_buf[warp + 1]))
            # instruction_selection: ld.shared.b32 + add.s32; extent: one per
            #   lane of warps 0..6 (:54-56)

        # --- cluster-wide sum of the same bin (:120-138) --------------------
        if NC > 1 and sum_across_cluster:
            if tid < 256:
                copy_r2s(hist[bank * 256 + tid], v)
                # instruction_selection: st.shared.b32; extent: one per radix
                #   lane.  This IS the DSMEM publication -- peers read this bank.
            cluster_sync()
            # instruction_selection: barrier.cluster.arrive + barrier.cluster.wait;
            #   extent: one per CTA.  Separates publication from every peer read.
            if tid < 256:
                for c in unroll(NC - 1):                       # #pragma unroll (:127)
                    src = map_peer(hist[bank * 256 + tid], (c + rank + 1) % NC)
                    # instruction_selection: cvta_generic_to_shared then
                    #   mapa.shared::cluster.u32; extent: NC-1 per radix lane.
                    #   The reference builds the same peer byte with the two
                    #   steps in the REVERSE order -- mapa.u64 then
                    #   cvta.to.shared::cluster.u64 -- because it maps a generic
                    #   pointer and then narrows, while the port narrows to the
                    #   shared window first and then maps.  The rank rotation
                    #   is not addressing detail: it staggers which peer each CTA
                    #   hits at each unrolled step.
                    v = add(v, copy_peer_s2r(src))
                    # instruction_selection: ld.shared::cluster.b32; extent:
                    #   NC-1 per radix lane
                copy_r2s(hist[2 * 256 + tid], v)
                # instruction_selection: st.shared.b32; extent: one per radix lane
        else:
            if tid < 256:
                copy_r2s(hist[2 * 256 + tid], v)
                # instruction_selection: st.shared.b32; extent: one per radix lane
        barrier()

        # --- pick the crossing bin (:142-152) -------------------------------
        nxt = copy_s2r(hist[2 * 256 + tid + 1]) if tid < 255 else 0
        # instruction_selection: ld.shared.b32; extent: one per radix lane
        if tid < 256 and v > k_rem and nxt <= k_rem:
            copy_r2s(threshold_bin, tid)
            # instruction_selection: st.shared.b32; extent: at most one lane.
            #   INVARIANT: exactly one lane satisfies this on a well-formed
            #   histogram; zero lanes on a degenerate one, which is what makes
            #   the uninitialized threshold_bin observable.
        barrier()
        bin_ = copy_s2r(threshold_bin)
        # instruction_selection: ld.shared.b32; extent: one per call.  Reads a
        #   location that is never zeroed between rounds (see the hazard note).
        if bin_ < 255:
            k_rem = sub(k_rem, copy_s2r(hist[2 * 256 + bin_ + 1]))
            # instruction_selection: ld.shared.b32 + sub.s32; extent: one scalar
        return bin_, k_rem

    # =======================================================================
    # classify(...) -- the three-way split, shared by every round (:189-217)
    # =======================================================================
    def classify(bits, index, bin_, phase, shift, last_round):
        d = digit(bits, shift)
        # instruction_selection: varies by SITE and by SHIFT VALUE; extent:
        #   one per element.  This annotation covers the CLASSIFY site only
        #   (:191, :276, :328); the two histogram-bump sites are different and
        #   are annotated at their own call sites.
        #     f32 :191 (shift == LSHIFT_START == 24)  `shr.u32` alone
        #     f32 :276/:328, rounds t = 1,2           `shr.u32` + `and.b32 255`
        #     f32 :276/:328, round  t = 3             NOTHING
        #     16-bit :191 (shift == 8)                `shr.u16` + `cvt.u32.u16`
        #     16-bit :276/:328, its only round        NOTHING
        #   The empty cases are SHIFT-driven, not dtype-driven: whenever
        #   `shift == 0` the whole extraction disappears because nvcc narrows
        #   the candidate load to `b8` and compares that byte directly.  f32
        #   reaches shift 0 on its last of three refinement rounds and 16-bit on
        #   its only one, which is why `.loc 1 276` carries `shr.u32` x2 rather
        #   than x3 in the f32 export.  Never `bfe` on SM >= 70 (absent in all
        #   48 entries).
        if d > bin_:
            slot = atomic_add_shared(final_count, 1)
            # instruction_selection: atom.shared.add.u32; extent: one per survivor
            copy_r2s(topk_inds[slot], index)
            # instruction_selection: st.shared.b32; extent: one per survivor.
            #   HAZARD: unguarded.  The source's `if (topk_offset < TopK)` is
            #   commented out at :195-197.  Reproduce, do not repair -- the
            #   writeback's `offs < TOPK` guard is what bounds the global store.
        elif d == bin_ and not last_round:
            slot = atomic_add_shared(num_cached_cnt[phase], 1)
            # instruction_selection: atom.shared.add.u32; extent: one per tie
            if slot < NUM_CACHED:
                copy_r2s(phase_half(cached_indices, phase)[slot], index)
                copy_r2s(phase_half(cached_bits,    phase)[slot], bits)
                # instruction_selection: st.shared.b32 x2; extent: one pair.
                #   `bits` is held in a 16-BIT register end to end for the
                #   16-bit dtypes -- `to_ordered` leaves it in %rs, and it stays
                #   there.  `cvt.u32.u16` appears at three independent sites,
                #   9 issues each in the f16 NC=1 export, because three separate
                #   consumers each widen the SAME u16 register:
                #     :191  the bin extraction shifts in 16 bits (`shr.u16`) and
                #           widens the EXTRACTED BIN afterwards
                #     :205  this shared store widens the unshifted value
                #     :211  the ring spill widens it again, independently
                #   The port must NOT widen `bits` once up front: doing so makes
                #   it a u32 register and collapses all three sites to one.
            elif slot - NUM_CACHED < overflow_stride:
                copy_r2g(ring(phase)[slot - NUM_CACHED].bits,  bits)
                copy_r2g(ring(phase)[slot - NUM_CACHED].index, index)
                # instruction_selection: st.global.b32 x2; extent: one pair.
                #   The spill is the only global write before the epilogue.
            else:
                return          # ring full: candidate silently dropped (:210-214)
            atomic_add_shared(hist[(phase ^ 1) * 256 + digit(bits, shift - 8)], 1)
            # instruction_selection: atom.shared.add.u32; extent: one per cached
            #   tie -- seeds the NEXT round's bank while this round classifies.
            #   The extraction again folds the x4 bin-to-byte scaling, and
            #   this ONE sketch line is reached from pass 1 and from both
            #   refinement rounds, so at f32 it lowers three different ways:
            #     f32 pass 1 (:206/:213), shift-8 == 16   `shr.u32 14` + `and.b32 1020`
            #     f32 t = 1  (:289/:296/:341/:347), == 8  `shr.u32 6`  + `and.b32 1020`
            #     f32 t = 2  (same locs),           == 0  `shl.b32 2`  + `and.b32 1020`
            #     16-bit pass 1 (:206/:213)              `mul.wide.u16 ...,4` + `and.b32 1020`
            #   Note the t = 2 case: shift 0 does NOT make this site vanish the
            #   way it does at classify -- the x4 scaling still has to happen, so
            #   the shift becomes a left shift.  At 16 bits pass 1 is the ONLY
            #   reachable instance: the single refinement round is also the final
            #   round, so `t < NRemainingRounds` is dead and the bump never runs.
            #   The 16-bit widening here is `mul.wide.u16`, not `cvt.u32.u16`.
        elif d == bin_ and last_round:
            ration(index)

    # =======================================================================
    # ration(...) -- final-round tie arbitration (:300-319, dup :351-370)
    # =======================================================================
    # `exceeded` is declared INSIDE the refinement-round body (:268), so it
    # resets at the top of every round and is shared between that round's shared
    # -cache loop (:306) and its overflow loop (:357).  Only the final round
    # reaches ration() today, which makes the lifetime unobservable -- but it is
    # the round's scalar state, and the port must scope it that way.
    def ration(index):
        if k_rem > 0 and not exceeded:
            got = atomic_add_peer(map_peer(k_rem_counter, 0), 1)
            # instruction_selection: cvta_generic_to_shared + mapa.shared::cluster.u32,
            #   then atom.shared::cluster.add.u32; extent: one per tie candidate until
            #   the thread latches out.  NOT guarded by NC > 1 -- at one CTA per
            #   cluster rank 0 is this CTA, and the reference's NC == 1 export
            #   still carries this opcode.
            if got < k_rem:
                slot = atomic_add_shared(final_count, 1)
                # instruction_selection: atom.shared.add.u32; extent: one per winner
                copy_r2s(topk_inds[slot], index)
                # instruction_selection: st.shared.b32; extent: one per winner
            else:
                exceeded = True
    # INVARIANT: every candidate reaching ration() is bit-equal in the final
    # byte, so the winners are arbitrary and vary run to run, while the selected
    # VALUE multiset does not.  This is why the correctness gate compares
    # multisets and never positions.

    # =======================================================================
    # Pass 1: first classification (:186-239)
    # =======================================================================
    k_rem = TOPK
    bin0, k_rem = threshold(bank=0, k_rem=k_rem, sum_across_cluster=True)

    if NC > 1:
        cluster_arrive()
        # instruction_selection: barrier.cluster.arrive; extent: one per CTA.
        #   SPLIT BARRIER: the matching wait is the first statement of round 1.
        #   The arrive straddles the whole classification loop and says "I am
        #   done reading peers' bank 0"; the wait says "everyone is done, bank
        #   may be cleared" (:218-225, :248).  Without it a peer clears bank 0
        #   while this CTA still reads it (:219-224).

    # PASS 1 IS THE `phase == 0` PRODUCER.  `classify(phase=p)` writes cursor
    # `num_cached_cnt[p]`, candidate half `p`, ring `p`, and seeds histogram bank
    # `p ^ 1`.  The source's pass 1 (:200-215) writes count[0], half 0, ring(0)
    # and seeds bank 1, so `p = 0`.  Round `t = 1` then consumes `phase ^ 1 = 0`
    # -- exactly what pass 1 produced -- and thresholds bank `phase = 1`, which
    # pass 1 seeded.  Passing `phase=1` here would invert all four destinations
    # and leave round 1 thresholding an all-zero bank over an empty half.
    base = rank * 1024 + tid
    for i in range(base * 4, (seq_len // 4) * 4, NC * 1024 * 4):
        v4 = copy_g2r_v(logits[logit_base + i], n=4)
        # instruction_selection: ld.global.nc.v4.b32 (f32) | ld.global.nc.v2.b32
        #   (16-bit); extent: one vector per iteration
        for j in unroll(4):
            classify(to_ordered(v4[j]), i + j, bin0, phase=0, shift=LSHIFT_START,
                     last_round=False)
    for i in range((seq_len // 4) * 4 + base, seq_len, NC * 1024):
        classify(to_ordered(copy_g2r(logits[logit_base + i])), i, bin0,
                 phase=0, shift=LSHIFT_START, last_round=False)
        # instruction_selection: ld.global.nc.{b32,b16}; extent: one per tail element

    # =======================================================================
    # Refinement rounds t = 1 .. sizeof(T)-1 (:241-373), fully unrolled
    # =======================================================================
    for t in unroll(range(1, ROUNDS)):                        # #pragma unroll (:241)
        phase = t % 2
        exceeded = reg(False)          # :268 -- per round, per thread; latches
                                       #   once this thread loses a tie race
        if NC > 1:
            cluster_wait()
            # instruction_selection: barrier.cluster.wait; extent: one per CTA.
            #   Pairs with the arrive issued before the previous classification.
        if tid < 256:
            copy_r2s(hist[(phase ^ 1) * 256 + tid], 0)
            # instruction_selection: st.shared.b32; extent: one per radix lane
        if tid == 0:
            copy_r2s(num_cached_cnt[phase], 0)
            # instruction_selection: st.shared.b32; extent: one scalar, thread 0
            #   only (:256)

        bin_t, k_rem = threshold(bank=phase, k_rem=k_rem, sum_across_cluster=True)

        if NC > 1 and t < ROUNDS - 1:
            cluster_arrive()
            # instruction_selection: barrier.cluster.arrive; extent: one per CTA.
            #   Skipped on the final round: no further threshold() means no
            #   further peer read of this bank (:260-263).

        raw   = copy_s2r(num_cached_cnt[phase ^ 1])
        # instruction_selection: ld.shared.b32; extent: one scalar per thread,
        #   once per round (:265)
        n_sh  = min(NUM_CACHED, raw)                          # :266
        n_gl  = min(overflow_stride, max(0, raw - NUM_CACHED))  # :267
        shift = LSHIFT_START - t * 8
        last  = (t == ROUNDS - 1)

        # Stride 1024, NOT 1024*NC: each CTA owns its own candidate slice and the
        # cluster does not re-partition here (:271-273).
        for i in range(tid, n_sh, 1024):
            b  = copy_s2r(phase_half(cached_bits,    phase ^ 1)[i])
            ix = copy_s2r(phase_half(cached_indices, phase ^ 1)[i])
            # instruction_selection: ld.shared.b32 x2; extent: one pair per
            #   candidate.  The f32 export shows `ld.shared.b32` x2 +
            #   `ld.shared.b8` x1 at `.loc 1 274` (at 16 bits that loc is
            #   `ld.shared.b8` x5 and no b32, since its only round has
            #   shift 0): on the final round only the low byte of
            #   `bits` is read, so nvcc narrows that load while the index load
            #   stays b32.  That narrowing is a benign compiler consequence of
            #   `shift == 0`, not a selection the port must reproduce -- the
            #   port emits the uniform b32 form.
            classify(b, ix, bin_t, phase, shift, last)
        for i in range(tid, n_gl, 1024):
            b  = copy_g2r(ring(phase ^ 1)[i].bits)
            ix = copy_g2r(ring(phase ^ 1)[i].index)
            # instruction_selection: ld.global.b32 x2 -- NO `.nc`; extent: one
            #   pair per spill.  The ring is written by this same kernel, so
            #   despite `__restrict__` the read is coherent: `.loc 1 325` carries
            #   zero `ld.global.nc.*` in every entry.  Two separate scalar loads,
            #   never a `v2` pair.
            #   Static shape at that loc, per dtype:
            #     f32     `ld.global.b32` x5 + `ld.global.b8` x1
            #     16-bit  `ld.global.b32` x5 + `ld.global.b8` x5
            #   The b8 is the shift-0 (final-round) narrowing of `bits`, which is
            #   why f32 has one (only t = 3 is final) and 16-bit has five (its
            #   only round is the final one).  That narrowing is a benign
            #   compiler consequence, NOT a selection the port reproduces -- the
            #   port emits the uniform b32 pair, exactly as at the shared twin.
            classify(b, ix, bin_t, phase, shift, last)

    # =======================================================================
    # Epilogue: cluster-wide output range (:374-404)
    # =======================================================================
    barrier()
    # instruction_selection: bar.sync 0; extent: CTA-wide, final_count settled
    if NC > 1:
        my_num, my_start = copy_s2r(final_count), 0
        # instruction_selection: ld.shared.b32; extent: one scalar per thread
        #   (:381).  Load-bearing for the ordering argument below: it must
        #   complete before rank 0's cursor starts absorbing peer counts.
        cluster_sync()
        # instruction_selection: barrier.cluster.arrive + barrier.cluster.wait;
        #   extent: one per CTA.  Every CTA must read its OWN count before rank
        #   0's cursor starts absorbing peers' contributions (:386).
        if rank > 0:
            if tid == 0:
                my_start = atomic_add_peer(map_peer(final_count, 0), my_num)
                # instruction_selection: cvta_generic_to_shared +
                #   mapa.shared::cluster.u32, then atom.shared::cluster.add.u32;
                #   extent: one per CTA with rank > 0
                copy_r2s(final_count, my_start)   # broadcast locally (:392)
                # instruction_selection: st.shared.b32; extent: one scalar,
                #   thread 0 of ranks > 0
            barrier()
            my_start = copy_s2r(final_count)
            # instruction_selection: ld.shared.b32; extent: one scalar per
            #   thread of ranks > 0 (:395)
        cluster_sync()
        # instruction_selection: barrier.cluster.arrive + barrier.cluster.wait;
        #   extent: one per CTA.  No CTA may exit while a peer still reads its
        #   cursor (:398).
        n_out, out_start = min(TOPK, my_num), my_start        # :400
    else:
        n_out, out_start = TOPK, 0                            # :403
        # NOTE the single-cluster path returns TOPK, not the measured count, so
        # the writeback bound is TOPK and an over-count from the unguarded emit
        # cannot walk past the output row.

    # =======================================================================
    # Writeback (:438-447 / :486-492 / :530-536)
    # =======================================================================
    for i in range(tid, n_out, 1024):
        offs = i + out_start
        if offs < TOPK:
            ind = copy_s2r(topk_inds[i])
            # instruction_selection: `ld.shared.b32` for plain-i32 (:441) and
            #   ragged (:533); `ld.shared.s32` wherever the index feeds a
            #   64-bit address computation -- page_table (:489, s32 x5, zero
            #   b32) and, mixed, plain-i64 (:441, s32 x5 + b32 x5).  extent:
            #   one per result
            if MODE == "plain":
                out = ind
            elif MODE == "page_table":
                out = copy_g2r(page_table[row * pt_stride + ind])
                # instruction_selection: ld.global.nc.b32; extent: one per result
            else:
                out = ind + ragged_offset        # hoisted at CTA entry (:510)
            copy_r2g(output_indices[ind_base + offs], out)
            # instruction_selection: st.global.b32 | st.global.b64 by IDX;
            #   extent: one per result
            if MODE == "plain" and output_values is not None:
                copy_r2g(output_values[ind_base + offs],
                         copy_g2r(logits[logit_base + ind]))
                # instruction_selection: ld.global.nc.{b32,b16} then
                #   st.global.{b32,b16}; extent: one pair per result.  The value
                #   is RE-GATHERED from global, never carried through shared.
```

## Cluster barrier and DSMEM accounting

Per launch, per CTA. `P = ROUNDS - 1` refinement rounds (3 for f32, 1 for the
16-bit dtypes), so `threshold()` is called `P + 1` times.

| region | `cluster.sync()` | split arrive / wait | peer loads | peer atomics | source |
| --- | ---: | ---: | ---: | ---: | --- |
| `threshold()`, each call | 1 | 0 | `NC - 1` per radix lane | 0 | :124-131 |
| before pass 1 | 0 | 1 arrive | 0 | 0 | :218-225 |
| each round head | 0 | 1 wait | 0 | 0 | :248 |
| each round, `t < P` | 0 | 1 arrive | 0 | 0 | :261 |
| final round ties | 0 | 0 | 0 | 1 per candidate until latched | :309, :360 |
| epilogue | 2 | 0 | 0 | 1 for ranks > 0 | :386, :391, :398 |

Totals at `NC > 1`, counting arrives and waits separately, for f32 (`P = 3`):

| contributor | arrives | waits |
| --- | ---: | ---: |
| `threshold()` x (P+1) = 4 fused syncs | 4 | 4 |
| split barrier: 1 before pass 1, then 1 per round with `t < P` | 3 | 0 |
| split barrier: 1 wait at each round head | 0 | 3 |
| epilogue, 2 fused syncs | 2 | 2 |
| **total** | **9** | **9** |

which is exactly what the `NC = 4` f32 entry exports. The split barrier
contributes an equal number of arrives and waits only because the arrive issued
before pass 1 is consumed by the first round's wait and the final round skips
its arrive -- the pairing is offset by one region, not nested.

At the 16-bit dtypes (`P = 1`) the same table gives `2 (threshold x 2) + 1 (the
single pass-1 arrive; no round satisfies t < P) + 2 (epilogue) = 5` arrives and
`2 + 1 + 2 = 5` waits, which is what the `NC = 4` f16 and bf16 entries export.
The 16-bit columns of the opcode table below are measured, not carried over from
f32. Only the SCAN rows halve exactly (`shfl.sync.down.b32` 32 -> 16,
`bar.warp.sync` 4 -> 2), because they scale with the `P + 1` = 4 -> 2 threshold
calls. The barrier rows do NOT halve: `bar.sync` goes 22 -> 12 at `NC = 1` and
23 -> 13 at `NC > 1`, and the cluster barriers go 9/9 -> 5/5. The cluster total
is `(P + 1) + P + 2 = 2P + 3`: `threshold()` contributes `P + 1` fused syncs
(4 -> 2), the SPLIT barrier contributes `P` (3 -> 1), and only the epilogue's 2
is fixed. So 9 at `P = 3` and 5 at `P = 1` -- a shrink, but not a halving,
because of that constant epilogue term. The split barrier is NOT a fixed
contributor: treating it as 1 would give `4 + 1 + 2 = 7` at f32, which the export
contradicts.

At `NC == 1` every row above collapses to zero **except** the tie atomics, which
remain (see the vocabulary section).  Those atomics are *more* numerous at 16
bits, not fewer: with `P = 1` the single refinement round is also the final
round, so both tie sites inline into every unrolled copy.

## Shared-memory budget

`num_cached` IS derived from `TOPK`, inversely: the host solves for the cache
size that makes the TOTAL request fill half the device's opt-in shared budget
(`topk.py:375-387`), so `16*num_cached` falls by exactly what `4*TopK` rises
(111040/1024, 107968/4096, 103872/8192 -- all summing to 112064). That is why the
request lands on the same 115188 B at every k in this matrix. The cancellation is
exact only for `TopK` a multiple of 4; see the residue note in the addressing
section.

| region | elements | bytes | source |
| --- | --- | ---: | --- |
| `s_cached_logit_bits[2][num_cached]` | `2 * nc` u32 | `8 * nc` | :97 |
| `s_cached_indices[2][num_cached]` | `2 * nc` i32 | `8 * nc` | :98 |
| `s_topk_inds[TopK]` | `TopK` i32 | `4 * TopK` | :99 |
| `shared_hist[3][256]` | 768 i32 | 3072 | :101 |
| `shared_final_idx_count` | 1 i32 | 4 | :102 |
| `shared_num_cached_count[2]` | 2 i32 | 8 | :103 |
| `shared_threshold_bin` | 1 i32 | 4 | :104 |
| `s_cum_reduce_buf[8]` | 8 i32 | 32 | :105 |
| `s_k_remaining_counter` | 1 i32 | 4 | :106 |
| **total** | | **`16*nc + 4*TopK + 3124`** | :579-582 |

On B200 `shared_memory_per_block_optin` is 232448, so `num_cached` is 6940 at
`k = 256` and the realized request is **115188 bytes** -- confirmed on device in
`.porting/fast_topk_clusters/probe_findings.md`, which also confirms the runtime
opts in past the 48 KB static ceiling without the explicit `cudaFuncSetAttribute`
calls the source's launcher makes (`:557-558`).

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `NClusters` | static, template parameter | the peer-sum loop is `#pragma unroll` (`:127`); the port specializes per width and the launch tag set changes with it |
| `TopK`, `num_cached` | static per config here, kernel args in the source | fix the shared carve offsets at compile time |
| `overflow_stride` | **runtime, and never passed** | derived in the binding as `cached_overflow.stride(0) / (4 * num_clusters)` (binding:47) -- the port must reproduce the derivation, not invent a parameter |
| `pre_hist` | **dead from Python** | always `None` (`topk.py:432`), so `:159-163` and `sum_hist == false` are unreachable and dropped |
| `PDL_ENABLED` | **dead from Python** | `pdl` defaults `False` everywhere; `cudaGridDependencySynchronize` (`:156`) is dropped |
| `seq_len` | scalar arg on plain, `seq_lens[row]` on the transforms | selects the trivial branch per row on the transforms, per launch on plain |
| `shared_threshold_bin` | **runtime, uninitialized** | never zeroed or reset (`:166-170`, `:104`); a degenerate round reads the previous round's value. Reproduced deliberately |
| emit bounds check | **absent in source** | `:195-197` is commented out; the `offs < TOPK` writeback guard is the only bound |
| tie survivors | **race-dependent** | arbitrary among bit-equal candidates; the selected set is exact, the selected indices are not stable |
| output order | **race-dependent** | atomic compaction plus a racing per-CTA base; positions carry no meaning |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "fast_topk_clusters", "category": "flashinfer",
  "runtime_cuda_archs": ["sm_100a"]}`; device symbol `fast_topk_clusters_kernel`.
- Plain TIRx only: a dynamic shared allocation carved by explicit views, explicit
  loops, and native `T.ptx.*` forms for every key operation. Global and shared
  memory are reached exclusively through `T.ptx.ld/st.*` on
  `buffer.ptr_to([...])` -- never a native `BufferLoad`/`BufferStore`, which the
  repository's low-level IR contract forbids. No `T.cuda.func_call`. No `Tx` tile
  primitives anywhere.
- The cluster scope is bound with `T.cta_id_in_cluster([NC])` and kept live even
  where the value folds; `clusterCtaIdx.x` is added to
  `tirx.kernel_launch_params` only when `NC > 1`.
- Reference: the source's own FFI, reached as
  `getattr(source_module(), "fast_topk_clusters_exact")` and the two transform
  siblings. The entries do **not** appear in `dir(module)`; they must be taken by
  name. `num_clusters` and `num_cached` are passed explicitly and identically to
  both sides, never left to the Python heuristic.
- Correctness compares **value multisets per row**, bit-exactly, against both
  `torch.topk` and the reference -- never positions, because both sides are
  nondeterministic in order and in tie choice. Additional per-row checks: indices
  in range, no duplicates, `output_values[i] == logits[row, ind[i]]` bit-exact on
  plain, and the transform inverse applied before comparison. The trivial branch
  is the one path compared positionally.
- Benchmark: one kernel per launch on both sides. The overflow workspace and every
  output tensor are allocated in `prepare_bench`, never inside the timed closure.

## Instruction selection is a lowering consequence

The sketch never requests a hardware instruction. Placement, address space, tile
width, and synchronization scope select the following families:

| Primitive / pattern | PTX family (fresh sm_100a export, 48 entries) |
| --- | --- |
| row read, first pass | `ld.global.nc.b32` \| `ld.global.nc.b16` -- `.nc` because `logits` is `const` AND `__restrict__` (`:82`); both are required, and `cached_overflow` (`:85`) is `__restrict__` WITHOUT `const`, which is exactly why its reads carry no `.nc` |
| row read, classification | `ld.global.nc.v4.b32` (f32) \| `ld.global.nc.v2.b32` (16-bit), one issue per `vec_t<T,4>` |
| `to_ordered` | f32: `setp.gt.s32 %r, -1` + `selp.b32` (of the two masks) + `xor.b32` (`topk_common.cuh:35-39`). 16-bit: `setp.gt.s16` + `selp.b16` + `xor.b16`, staying in 16-bit registers (`topk_common.cuh:61-64` for half, `:87-90` for bf16 -- both emit the identical triple), `bits` then stays in a 16-bit register; `cvt.u32.u16` widens it at **three independent** sites, 9 each in the f16 `NC=1` export -- after the 16-bit shift at `:191`, at the shared store `:205`, and at the ring spill `:211` |
| `digit`, classify site (`:191`, `:276`, `:328`) | f32 first pass `shr.u32` alone; f32 rounds t = 1,2 `shr.u32` + `and.b32 255`; **f32 round t = 3 and the 16-bit refinement round emit NOTHING** (`shift == 0`, candidate load narrowed to `b8`); 16-bit first pass `shr.u16` + `cvt.u32.u16`. The empty case is shift-driven, not dtype-driven -- **never** `bfe` on SM >= 70 |
| `digit`, histogram-bump sites (`:178`, `:206`/`:213`, `:289`/`:296`/`:341`/`:347`) | always emit the x4 bin-to-byte scaling, even at shift 0: f32 `shr.u32 22`/`14`/`6` or `shl.b32 2`, each + `and.b32 1020`; 16-bit, per site: `:178` is `mul.wide.u16 ...,4` with NO mask (the `shr.u16 8` belongs to `.loc 1 177`, and a `u16 >> 8` bin already fits 8 bits), while `:206`/`:213` are `mul.wide.u16 ...,4` + `and.b32 1020` with NO shift (their shift is 0). These sites never vanish |
| cursors (`final_count`, `num_cached_cnt`) | `atom.shared.add.u32`, result used |
| histogram bumps | `atom.shared.add.u32` with the destination DISCARDED -- 55 of 85 per f32 entry, 47 of 85 at 16 bits. nvcc never narrows these to `red.shared.add.u32`: `red.*` is **0** across the whole 48-entry export. A port that emitted `red` where the old value is unused would diverge on most of its shared atomics |
| suffix scan | `shfl.sync.down.b32` x5 (warps 0-7 only) then x3 (warp 0 only), plus one `bar.warp.sync` inside warp 0 |
| peer histogram read | port: `cvta_generic_to_shared` + `mapa.shared::cluster.u32` + `ld.shared::cluster.b32`; reference PTX, in this order: `mapa.u64` -> `cvta.to.shared::cluster.u64` -> `ld.shared::cluster.b32` (468/468 def-use, 0 exceptions) |
| tie rationing, range claim | port: `cvta_generic_to_shared` + `mapa.shared::cluster.u32` + `atom.shared::cluster.add.u32`; reference PTX: `mapa.u64` -> `cvta.to.shared::cluster.u64` -> the same atomic |
| overflow-ring read | `ld.global.b32` x2 -- the one global read with **no** `.nc`, because this kernel writes the ring |
| fused cluster edge | `barrier.cluster.arrive` + `barrier.cluster.wait` |
| split cluster edge | the same two opcodes, issued in different regions |
| block synchronization | `bar.sync 0` |
| absent everywhere | `match.any`, `vote.sync`, `prmt`, `popc`, `clz`, `atom.global.*`, `mbarrier.*`, `cp.async.*`, `st.async.*`, `setmaxnreg`, `elect.sync` |

Static opcode counts per exported instantiation, measured on the **plain-int32**
entries (full table in
`.porting/fast_topk_clusters/ptx/opcode_table.md`):

`total instructions` and `ld.shared.b32` are the two rows that also move with
mode and index width; the rest are invariant across all four modes. At f32
`NC = 1` the totals run 1907 (plain-i32) / 1904 (plain-i64) / 1729 (ragged) /
1703 (page_table).

| op | f32 NC=1 | f32 NC=4 | f16 NC=1 | f16 NC=4 |
| --- | ---: | ---: | ---: | ---: |
| total instructions | 1907 | 2125 | 1790 | 1913 |
| `mapa` | 1 | 14 | 2 | 9 |
| `cvta.to.shared::cluster.u64` | 1 | 14 | 2 | 9 |
| `ld.shared::cluster.b32` | 0 | 12 | 0 | 6 |
| `atom.shared::cluster.add.u32` | **2** | 3 | **10** | 11 |
| `barrier.cluster.arrive` / `.wait` | 0 / 0 | 9 / 9 | 0 / 0 | 5 / 5 |
| `atom.shared.add.u32` | 85 | 85 | 85 | 85 |
| `bar.sync` | 22 | 23 | 12 | 13 |
| `bar.warp.sync` | 4 | 4 | 2 | 2 |
| `shfl.sync.down.b32` | 32 | 32 | 16 | 16 |
| `ld.global.nc.v4.b32` | 1 | 1 | 0 | 0 |
| `ld.global.nc.v2.b32` | 0 | 0 | 1 | 1 |

`bf16` is byte-identical to `f16` at every NC. The 16-bit columns are measured,
not derived from the f32 ones: `ROUNDS = 2` gives `P = 1`, so `threshold()` runs
twice instead of four times and `vec_t<T,4>` is 8 B rather than 16 B.

Two rows are port checks rather than description. `atom.shared::cluster.add.u32`
at **2 with zero cluster barriers** on the f32 `NC = 1` entry (and **10** on the
16-bit one) is the check that the tie-rationing atomic stayed outside the
`NC > 1` guard; a port that hoisted it would show 0. And `ld.shared::cluster.b32`
at 12 in the f32 `NC = 4` entry is the check that peer reads were spelled in the
cluster space -- that count and `ld.shared.b32` must never trade places, because
a peer read spelled `ld.shared.b32` does not produce a wrong number, it produces
an asynchronous illegal-instruction trap with no attribution.

Both tables are the **plain-int32** entries. `ld.shared.b32` and the totals also
move with mode and index width; the spread and its causes are in
`.porting/fast_topk_clusters/probe_findings.md`.

The counts are audit evidence, not operands or issue-count hints in the sketch.
