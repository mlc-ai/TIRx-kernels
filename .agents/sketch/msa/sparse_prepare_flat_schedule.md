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

This design sketch documents a TIRx port of MSA's
python/fmha_sm100/cute/src/sm100/prepare_scheduler.py
SparseAttentionPrepareFlatScheduleSm100.
-->

# msa_sparse_prepare_flat_schedule_sm100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/msa/sparse_prepare_flat_schedule.py`](../../../tirx_kernels/msa/sparse_prepare_flat_schedule.py).
That TIRx module is the authoritative implementation.

The port covers **one kernel**: `SparseAttentionPrepareFlatScheduleSm100.kernel`
(`prepare_scheduler.py:277-345`) with its launcher `__call__` (`:233-275`), the
stage that turns the CSR k2q row fanout into the flat worklist the SM100 sparse
forward kernels consume.

Instantiations: **exactly one**. The compile shim keys its cache on the constant
tuple `("sparse_prepare_flat_schedule_sm100_csr_varlen",)` (`:536-538`) and
`to_cute_tensor` marks every layout dynamic, so one binary serves every shape;
`total_rows`, `num_heads_kv`, `blk_kv`, `target_q_per_cta` and `work_capacity`
are runtime `Int32` kernel arguments. The single compile-time constant is
`num_threads = 128` (`:130`), hence `warps_per_cta = 4` (`:135`).

Accepted target SM100/B200. Out of scope, with the host predicate that excludes
each: `enabled=False` (`:577`) and `total_rows <= 0 or head_kv <= 0` (`:581`)
never launch; the sibling `SparseAttentionPrepareFwdSplitAtomicSm100` (`:348`) is
a separate kernel with its own cache key and its own shared storage. Tile (`Tx`)
primitives are out of scope everywhere.

## Pipeline at a glance

| Kernel / role | Program | Publication / reuse edges |
| --- | --- | --- |
| **warp `w` of 4 per CTA**, lane 0 only (`:306-318`) | load the row's two CSR bounds; decode `row_linear -> (batch_idx, kv_block_idx)` by binary search over kv-block levels, each probe re-scanning all `B` batches of `cu_seqlens_k`; compute `num_chunks` | Produces four scalars in lane 0's registers. Published to the other 31 lanes by four `shfl.sync.idx.b32` (`:319-322`) — the only intra-warp edge in the kernel. |
| **all 32 lanes of warp `w`** (`:324-345`) | stride `chunk_idx` by 32 over `num_chunks`; per chunk reserve one slot with a device-scope atomic on `work_count`, then write the six-field work item under the capacity guard | Publishes to global memory only. Slot order is the atomic arrival order, so the schedule is a **multiset** of work rows plus a final `work_count`; the consuming forward kernel reads `work_idx < work_count[0]` (`atten_fwd.py:698-705`) and does not depend on row order. |

There is **no** producer/consumer split, no pipe, no stage, no phase, no
mbarrier, no async copy, no shared memory, and no `bar.sync`. The launch requests
no dynamic smem at all (`:271-275`); contrast the sibling kernel, which does
(`:428`). Every global access is a plain synchronous `ld.global`/`st.global`, the
only cross-lane traffic is the four `shfl.sync.idx.b32`, and the only
cross-CTA edge is the single relaxed atomic.

**One warp owns one `(row_linear, head_kv_idx)` pair.** The grid is
`ceil_div(total_rows * num_heads_kv, 4)` CTAs of 128 threads, so warp
`row_head_idx = blockIdx.x * 4 + warp_idx` and the tail warps of the last CTA
fall out on the `row_head_idx < total_row_heads` guard.

## Primitive vocabulary

```python
load_global(buf, idx) -> reg          # ld.global.b32   (NOT .nc: no __restrict__ const)
store_global(buf, idx, reg)           # st.global.b32
atom_add_global(buf, idx, v) -> old   # atom.global.add.u32 -- see below
shfl_idx(value, src_lane)             # shfl.sync.idx.b32 d, a, src, 31, -1
ceil_div(a, b) / floor_div(a, b)      # div.s32 + sign-correction (see below)
min(a, b) / max(a, b)                 # min.s32 / max.s32
```

### The atomic is emitted as the short form, not the qualified one

The source spells the reservation with explicit qualifiers (`:326-331`):

```python
work_idx = cute.arch.atomic_add(mWorkCount.iterator.llvm_ptr, Int32(1),
                                sem="relaxed", scope="gpu")
```

but relaxed ordering and device scope are PTX's defaults for `atom.global`, so
the export carries the **short encoding**:

```
.loc 1 232 0    atom.global.add.u32  %r19, [%rd1], 1;
```

One instruction, `u32`, immediate operand `1`, no `.relaxed`, no `.gpu`, and no
surrounding `fence`. The port must emit that exact form — `atom.relaxed.gpu.
global.add.s32` would be semantically equal but a different encoding, and the
repository's `utils/topk_radix.py` keeps both spellings precisely because the
distinction is observable in the export. The `.loc` points at the `__call__`
decorator line rather than `:326` because the emission helper is inlined.

### Division carries a sign-correction tail

CuTeDSL's `//` is Python floor division, so every quotient in the kernel lowers
to `div.s32` plus a six-instruction correction (`mul.lo.s32`, `setp.ne.b32`,
`xor.b32`, `setp.lt.s32`, `and.pred`, `selp.b32`, `add.s32`) that subtracts one
when the operands' signs differ and the division is inexact:

```
.loc 1 304   div.s32   %r31, %r1, %r23;        # row_linear = row_head_idx // num_heads_kv
             mul.lo.s32 %r32, %r31, %r23;
             setp.ne.b32 %p2, %r1, %r32;       # inexact?
             xor.b32   %r33, %r23, %r1;
             setp.lt.s32 %p3, %r33, 0;         # signs differ?
             and.pred  %p4, %p3, %p2;
             selp.b32  %r34, -1, 0, %p4;
             add.s32   %r111, %r31, %r34;
```

Six `div.s32` appear: `row_linear` (`:304`), the four inlined `_rows_in_batch`
sites (`:166`), and `num_chunks` (`:316-318`). The binary-search midpoint
(`:205`) divides by the constant 2 and lowers to `shr.u32`/`shr.s32`/`and.b32`
with the same correction shape instead.

Every dividend here is provably non-negative, so the corrected quotient equals
truncating division; the correction is a lowering artifact of the source
language, not an algorithmic step. The port must produce the **same quotient
values** at the same points — it is not required to reproduce the correction
tail, and doing so would only add instructions. This is the one place where the
port deliberately does not chase the source's instruction count; the key copy,
compute, and sync operations below are matched exactly.

### `_rows_in_batch` is one load per iteration, not two

`_rows_in_batch` (`:158-166`) reads `mCuSeqlensK[b+1] - mCuSeqlensK[b]`, but the
three unconditional scan loops each hoist the first element and rotate the
previous value forward, so the steady-state loop body carries exactly **one**
`ld.global.b32`:

```
        ld.global.b32 %r99, [%rd2];        # .loc 1 165 49 -- cu_seqlens_k[0], hoisted
$L__BB0_4:
        .pragma "nounroll"
        ld.global.b32 %r11, [%rd19+4];     # .loc 1 165 13 -- cu_seqlens_k[b+1]
        ...
        mov.b32 %r99, %r11;                # rotate: this iteration's [b+1] is next's [b]
```

The fourth scan — the batch search at `:222-229` — keeps **both** loads in the
body (`%r76` at col 13, `%r77` at col 49) because its work sits under the
`found == 0` predicate, which breaks the rotation.

The probe scan's base load is hoisted further than the other two: it sits in the
pre-header of the **binary search** (`$L__BB0_8`), not of the probe loop, so lane
0 issues it once for the whole search and every probe re-seeds `prev` from a
register:

```
        ld.global.b32 %r4, [%rd2];         # .loc 1 165 49 -- once for the search
$L__BB0_8:                                 # binary-search probe
        ...
        mov.b32 %r97, %r4;                 # re-seed prev from a register
$L__BB0_9:
        .pragma "nounroll"
        ld.global.b32 %r9, [%rd24];        # .loc 1 165 13 -- the only load per iter
```

Ten `ld.global.b32` total: two CSR bounds, three loop-hoisted (`col 49`, one per
unpredicated scan), and five in-loop (four at `col 13`, plus the `col 49` partner
inside the `found == 0` scan).

### `.file` map of the export

`.file 1` = `prepare_scheduler.py`. Line info comes from a fresh
`CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1 CUTE_DSL_NO_CACHE=1` export, preserved at
`.porting/sparse_prepare_flat_schedule/ptx_lineinfo/source_flat_schedule_lineinfo.sm_100a.ptx`
(95 `.loc` directives, 230 instructions). Every `.loc` cited below uses that file.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
variant = specialize(NUM_THREADS=128, target="sm_100a")
# instruction_selection: none; extent: the only compile-time constant (:130)
WARPS_PER_CTA = NUM_THREADS // 32          # 4 (:135)
WORK_FIELDS   = 6                          # _emit_work writes six fields (:151-156)

# Runtime ABI -- all int32, all global, all `assume_tensor_aligned` (:253-256).
#   k2q_row_ptr        [num_heads_kv, total_rows + 1]   read-only
#   cu_seqlens_k       [num_batches + 1]                read-only
#   scheduler_metadata [work_capacity, WORK_FIELDS]     write-only
#   work_count         [1]                              atomic read-modify-write
# Runtime scalars: total_rows, num_batches, target, work_capacity,
#                  num_heads_kv, blk_kv.

launch_config = launch(grid=(ceil_div(total_rows * num_heads_kv, WARPS_PER_CTA), 1, 1),
                       block=(NUM_THREADS, 1, 1),
                       dynamic_smem_bytes=0)
# instruction_selection: none; extent: static launch metadata.  No smem argument
#   is passed at all (:271-275), and no cudaFuncSetAttribute exists in this path.

# ===========================================================================
# Storage.  Registers only -- no shared, no tensor memory, no mbarrier.
# ===========================================================================
# Warp-uniform after the broadcast; lane-private before it.
row_count = num_chunks = batch_idx = kv_block_idx = reg_i32(0)   # (:297-302)
head_kv_idx = row_linear = reg_i32(0)
# instruction_selection: mov.b32 of the immediate 0; extent: six registers.
#   These are initialized BEFORE the grid-tail guard, so a tail warp that skips
#   the whole body still broadcasts defined zeros (:297-302 precede :303).

def kernel():
    tidx  = thread_id(extent=NUM_THREADS)   # instruction_selection: mov.u32 %tid.x
    block = cta_id(axis="x")                # instruction_selection: mov.u32 %ctaid.x
    lane  = tidx % 32                       # instruction_selection: and.b32 with 31 (:292)
    warp  = tidx // 32                      # instruction_selection: shr.u32 by 5 (:293)
    row_head_idx = block * WARPS_PER_CTA + warp
    # instruction_selection: shl.b32 by 2 + or.b32 (:294); extent: scalar.  The
    #   OR is exact because warp < 4 -- a plain add would be equivalent.
    total_row_heads = total_rows * num_heads_kv
    # instruction_selection: mul.lo.s32 (:295); extent: scalar

    # -----------------------------------------------------------------------
    # Stage 1: grid tail, then the row/head split (:303-305)
    # -----------------------------------------------------------------------
    if row_head_idx < total_row_heads:      # (:303)
        # instruction_selection: setp.ge.s32 + predicated branch to the
        #   broadcast block; extent: warp-uniform branch
        row_linear  = row_head_idx // num_heads_kv          # (:304)
        # instruction_selection: div.s32 + floor-correction tail; extent: scalar
        head_kv_idx = row_head_idx - row_linear * num_heads_kv   # (:305)
        # instruction_selection: mul.lo.s32 + sub.s32; extent: scalar

        # -------------------------------------------------------------------
        # Stage 2: lane 0 owns the whole decode (:306-318)
        # -------------------------------------------------------------------
        if lane == 0:                        # (:306)
            # instruction_selection: setp.ne.b32 + predicated branch; extent:
            #   intra-warp branch -- 31 lanes jump straight to the broadcast
            row_start = load_global(k2q_row_ptr, head_kv_idx, row_linear)      # (:307)
            # instruction_selection: ld.global.b32 (.loc 1 307 24); extent: one
            #   scalar.  Address = base + (head_kv_idx * stride + row_linear)*4,
            #   built with cvt.s64.s32 + mul.lo.s64 + add.s64 + shl.b64.
            row_end = load_global(k2q_row_ptr, head_kv_idx, row_linear + 1)    # (:308)
            # instruction_selection: ld.global.b32 (.loc 1 308 22); extent: one
            #   scalar.  NOT fused with the load above into ld.global.v2.b32 --
            #   the export keeps two independent loads and two address chains.
            row_count = row_end - row_start                                    # (:309)
            # instruction_selection: sub.s32; extent: scalar

            # --- 2a. max_rows_per_batch (:183-193, called from :203) --------
            max_rows = 0
            prev = load_global(cu_seqlens_k, 0)
            # instruction_selection: ld.global.b32 (.loc 1 165 49); extent: one
            #   scalar, hoisted out of the loop below
            for b in range(num_batches):        # `unroll=1` (:190)
                # instruction_selection: .pragma "nounroll" backward branch;
                #   extent: one iteration per batch
                nxt = load_global(cu_seqlens_k, b + 1)
                # instruction_selection: ld.global.b32 (.loc 1 165 13); extent:
                #   one scalar per iteration -- `prev` is rotated from the last
                #   iteration, so the body holds ONE load, not two
                rows = ceil_div(nxt - prev, blk_kv)                    # (:165-166)
                # instruction_selection: add.s32 + sub.s32 + div.s32 + floor
                #   correction; extent: one per iteration
                max_rows = max(max_rows, rows)                         # (:192)
                # instruction_selection: max.s32; extent: one per iteration
                prev = nxt

            # --- 2b. binary search for the level band (:202-215) -----------
            lo, hi = 0, max_rows
            probe_base = load_global(cu_seqlens_k, 0)
            # instruction_selection: ld.global.b32 (.loc 1 165 49); extent: one
            #   scalar, hoisted out of BOTH the binary search and the probe loop
            #   -- it sits in the pre-header of the search loop, so lane 0 issues
            #   it once for the whole decode, not once per probe
            while lo < hi:                                             # (:204)
                # instruction_selection: setp.lt.s32 + backward branch; extent:
                #   ~log2(max_rows) iterations, serial on lane 0
                mid = (lo + hi) // 2                                   # (:205)
                # instruction_selection: shr.u32 + add.s32 + shr.s32 + and.b32 +
                #   setp/selp correction (division by the constant 2); extent: scalar
                rows_before_next = 0                                   # (:206-210)
                prev = probe_base
                # instruction_selection: mov.b32 %r97, %r4; extent: register copy
                #   -- prev is re-seeded per probe from the hoisted register, so
                #   the probe loop body holds exactly one global load
                for b in range(num_batches):    # `unroll=1` (:177)
                    # instruction_selection: .pragma "nounroll" backward branch;
                    #   extent: one iteration per batch, PER PROBE
                    nxt = load_global(cu_seqlens_k, b + 1)
                    # instruction_selection: ld.global.b32 (.loc 1 165 13);
                    #   extent: one scalar per iteration
                    rows_before_next += min(ceil_div(nxt - prev, blk_kv), mid + 1)
                    # instruction_selection: div.s32 + correction + min.s32 +
                    #   add.s32 (.loc 1 179); extent: one per iteration
                    prev = nxt
                if rows_before_next <= row_linear:                     # (:211-214)
                    lo = mid + 1
                else:
                    hi = mid
                # instruction_selection: setp.gt.s32 + two selp.b32; extent:
                #   scalar -- the branch is if-converted, not a real branch
            level = lo                                                 # (:216)

            # --- 2c. offset inside the level band (:217) -------------------
            rows_before = 0
            prev = load_global(cu_seqlens_k, 0)
            # instruction_selection: ld.global.b32 (.loc 1 165 49); extent: one
            #   scalar, hoisted
            for b in range(num_batches):        # `unroll=1` (:177)
                nxt = load_global(cu_seqlens_k, b + 1)
                # instruction_selection: ld.global.b32 (.loc 1 165 13); extent:
                #   one scalar per iteration
                rows_before += min(ceil_div(nxt - prev, blk_kv), level)
                # instruction_selection: div.s32 + correction + min.s32 +
                #   add.s32; extent: one per iteration
                prev = nxt
            offset = row_linear - rows_before
            # instruction_selection: sub.s32; extent: scalar

            # --- 2d. batch scan for the offset-th batch above `level` (:218-229)
            active_idx, batch_idx, found = 0, 0, 0
            for b in range(num_batches):        # `unroll=1` (:222)
                # instruction_selection: .pragma "nounroll" backward branch;
                #   extent: one iteration per batch -- the loop runs to the end
                #   even after `found`, exactly as the source writes it
                if found == 0:                                         # (:223)
                    # instruction_selection: setp.ne.b32 + branch over the body;
                    #   extent: intra-loop branch
                    nxt  = load_global(cu_seqlens_k, b + 1)
                    # instruction_selection: ld.global.b32 (.loc 1 165 13);
                    #   extent: one scalar per iteration
                    prev = load_global(cu_seqlens_k, b)
                    # instruction_selection: ld.global.b32 (.loc 1 165 49);
                    #   extent: one scalar per iteration -- this scan keeps BOTH
                    #   loads because the predicate breaks the rotation
                    rows = ceil_div(nxt - prev, blk_kv)                # (:224)
                    # instruction_selection: div.s32 + correction; extent: one per iteration
                    if rows > level:                                   # (:225)
                        if active_idx == offset:                       # (:226-228)
                            batch_idx, found = b, 1
                        active_idx += 1
                    # instruction_selection: setp.gt.s32 + setp.eq.b32 +
                    #   and.pred + three selp.b32 + add.s32; extent: scalar --
                    #   fully if-converted, no branch
            kv_block_idx = level                                       # (:230)

            # --- 2e. chunk count (:315-318) --------------------------------
            if row_count > 0:
                # instruction_selection: setp.lt.s32 + branch; extent: scalar
                num_chunks = ceil_div(row_count, target)
                # instruction_selection: add.s32 x2 + div.s32 + floor correction;
                #   extent: scalar.  A row with no references keeps num_chunks = 0
                #   and therefore emits nothing.

    # -----------------------------------------------------------------------
    # Stage 3: publish lane 0's four scalars to the warp (:319-322)
    # -----------------------------------------------------------------------
    row_count    = shfl_idx(row_count, 0)
    # instruction_selection: shfl.sync.idx.b32 d, a, 0, 31, -1 (.loc 1 319 16);
    #   extent: one instruction, full-warp mask, clamp 31
    num_chunks   = shfl_idx(num_chunks, 0)
    # instruction_selection: shfl.sync.idx.b32 (.loc 1 320 17); extent: one
    batch_idx    = shfl_idx(batch_idx, 0)
    # instruction_selection: shfl.sync.idx.b32 (.loc 1 321 16); extent: one
    kv_block_idx = shfl_idx(kv_block_idx, 0)
    # instruction_selection: shfl.sync.idx.b32 (.loc 1 322 19); extent: one
    # ORDER IS PART OF THE PROTOCOL: four separate broadcasts, in this order,
    # executed by EVERY warp including the ones the grid-tail guard emptied --
    # they sit after the `row_head_idx < total_row_heads` block, so the
    # convergent shuffles are never executed by a partial warp.

    # -----------------------------------------------------------------------
    # Stage 4: lane-strided chunk emission (:324-345)
    # -----------------------------------------------------------------------
    chunk_idx = lane                                                   # (:324)
    while chunk_idx < num_chunks:                                      # (:325)
        # instruction_selection: setp.lt.s32 + backward branch; extent:
        #   ceil(num_chunks / 32) iterations per lane.  `chunk_idx` reuses the
        #   lane register, so the loop needs no separate induction variable.
        work_idx = atom_add_global(work_count, 0, 1)                   # (:326-331)
        # instruction_selection: atom.global.add.u32 %r, [ptr], 1 (.loc 1 232 0);
        #   extent: one instruction per chunk.  Relaxed/device-scope is the PTX
        #   default -- no .relaxed, no .gpu, no fence.
        q_begin = chunk_idx * target                                   # (:332)
        # instruction_selection: mul.lo.s32; extent: scalar
        q_count = min(target, row_count - q_begin)                     # (:333)
        # instruction_selection: sub.s32 + min.s32; extent: scalar.  Computed
        #   BETWEEN the third and fourth store in the export, not hoisted.
        if work_idx < work_capacity:                                   # (:150)
            # instruction_selection: setp.ge.s32 + branch over the six stores;
            #   extent: scalar.  The comparison is SIGNED even though the atomic
            #   returned u32.
            store_global(scheduler_metadata, work_idx, 0, head_kv_idx) # (:151)
            # instruction_selection: st.global.b32 [addr] (.loc 1 151 8); extent: one
            store_global(scheduler_metadata, work_idx, 1, row_linear)  # (:152)
            # instruction_selection: st.global.b32 [addr+4] (.loc 1 152 8); extent: one
            store_global(scheduler_metadata, work_idx, 2, q_begin)     # (:153)
            # instruction_selection: st.global.b32 [addr+8] (.loc 1 153 8); extent: one
            store_global(scheduler_metadata, work_idx, 3, q_count)     # (:154)
            # instruction_selection: st.global.b32 [addr+12] (.loc 1 154 8); extent: one
            store_global(scheduler_metadata, work_idx, 4, batch_idx)   # (:155)
            # instruction_selection: st.global.b32 [addr+16] (.loc 1 155 8); extent: one
            store_global(scheduler_metadata, work_idx, 5, kv_block_idx)# (:156)
            # instruction_selection: st.global.b32 [addr+20] (.loc 1 156 8); extent: one
            # SIX SEPARATE SCALAR STORES off one base register, at +0/+4/.../+20.
            # The export does NOT vectorize them into st.global.v4 + st.global.v2,
            # even though the row is 24 contiguous bytes and 16-byte aligned only
            # every other row (the row stride is 6 words).
        chunk_idx += 32                                                # (:345)
        # instruction_selection: add.s32 with the immediate 32; extent: scalar
```

## Control-flow map

| Region | Source | Predicate | Who executes |
| --- | --- | --- | --- |
| grid tail | `:303` | `row_head_idx < total_row_heads` | warp-uniform; empty warps fall through to the broadcast |
| decode owner | `:306` | `lane_idx == 0` | one lane per warp; the other 31 idle until the shuffle |
| level search | `:204-215` | `lo < hi` | lane 0, `~log2(max_rows_per_batch)` serial probes |
| batch scans | `:177, :190, :222` | `unroll=1`, full `B` trip count | lane 0, four inlined `O(B)` loops |
| found guard | `:223` | `found == 0` | lane 0; a real taken branch over the scan body once the batch is located — the loop still runs to `B` |
| empty row | `:315` | `row_count > 0` | lane 0; leaves `num_chunks = 0` so the warp emits nothing |
| chunk stride | `:325` | `chunk_idx < num_chunks` | all 32 lanes, stride 32 |
| capacity | `:150` | `work_idx < work_capacity` | per chunk; drops the six stores, keeps the atomic |

The decode cost is `O(B * log(max_rows_per_batch))` scalar loads on **one lane
per warp**, which is where essentially all of this kernel's latency lives; the
emission tail is one atomic plus six stores per work item.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `num_threads = 128` | static (`:130`) | block shape and `WARPS_PER_CTA = 4`; validated `% 32 == 0` on the host (`:132`) |
| `warps_per_cta = 4` | static | `blockIdx.x * 4 + warp` lowers to `shl.b32 2` + `or.b32` |
| `WORK_FIELDS = 6` | static | the six-word row stride and the `+0..+20` store displacements |
| `total_rows`, `num_heads_kv` | **runtime** | grid extent and the row/head split; the source reads `total_rows` from `mK2qCounts.shape[1] - 1` on the host (`:257`) and passes it in |
| `num_batches` | **runtime** | the source reads `mCuSeqlensK.shape[0] - 1` inside the decode helpers (`:176, :189, :221`); the port passes the same value as a scalar |
| `blk_kv`, `target_q_per_cta`, `work_capacity` | **runtime** | divisors and the store guard stay register operands, never immediates |
| output row order | **non-deterministic** | atomic arrival order; only the multiset and the final `work_count` are defined |
| `work_count` initial value | **caller contract** | the host allocates it zeroed per call (`:607`); the counter is monotone and is never reset by the kernel |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "msa_sparse_prepare_flat_schedule_sm100", "category":
  "msa", "runtime_cuda_archs": ["sm_100a"]}`.
- Plain TIRx only: explicit loops, scalar registers via `T.alloc_local`, and
  native `T.ptx.*` forms for every key operation. Global memory is reached
  exclusively through `T.ptx.ld/st.*` on `buffer.ptr_to([...])`, never a native
  `BufferLoad`/`BufferStore`, which the repository's low-level IR contract
  forbids. No `T.cuda.func_call`. No `Tx` tile primitives anywhere.
- The 2-D source tensors are matched as flat 1-D buffers: `k2q_row_ptr` is
  indexed `head_kv_idx * (total_rows + 1) + row_linear` and
  `scheduler_metadata` as `work_idx * 6 + field`, which reproduces the export's
  addresses without a dynamic stride operand.
- Correctness compares against **MSA's own compiled kernel** over identical
  inputs, plus an independent torch oracle that rebuilds the expected work
  multiset from `(counts, cu_seqlens_k, target)`. Both comparisons sort the rows
  first: the output order is not defined.
- The benchmark is **kernel-only** on both sides — the timed closure holds one
  launch of the compiled kernel, with allocation, schedule sizing and the
  CuTeDSL compile outside it. Both sides rotate over the same pre-zeroed
  `work_count` slots, because the counter is monotone: a second launch against a
  spent counter reserves slots past `work_capacity`, silently drops all six
  stores, and eventually wraps int32 negative.

## Instruction selection is a lowering consequence

| Primitive / pattern | PTX family (fresh `sm_100a` lineinfo export) |
| --- | --- |
| CSR bound loads (`:307`, `:308`) | `ld.global.b32` x2, independent address chains — **not** `ld.global.v2.b32` |
| `cu_seqlens_k` scan loads (`:165`) | `ld.global.b32`; 3 loop-hoisted (`col 49`, one per unpredicated scan, the probe one hoisted above the whole binary search) + 5 in-loop (4 at `col 13`, plus the `col 49` partner inside the `found == 0` scan) |
| work-item stores (`:151-156`) | `st.global.b32` x6 at `+0/+4/+8/+12/+16/+20` — **not** vectorized |
| slot reservation (`:326-331`) | `atom.global.add.u32 %r, [ptr], 1` — short form, no `.relaxed`, no `.gpu`, no `fence` |
| warp broadcast (`:319-322`) | `shfl.sync.idx.b32 d, a, 0, 31, -1` x4 |
| `lane` / `warp` split (`:292-293`) | `and.b32 31` / `shr.u32 5` |
| `row_head_idx` (`:294`) | `shl.b32 2` + `or.b32` |
| integer division (`:166, :304, :316`) | `div.s32` + `mul/setp.ne/xor/setp.lt/and.pred/selp/add` correction |
| midpoint `// 2` (`:205`) | `shr.u32 31` + `add.s32` + `shr.s32 1` + `and.b32 -2` + correction |
| `min` / `max` (`:179, :192, :333`) | `min.s32` / `max.s32` |
| scan loops (`:177, :190, :222`) | `.pragma "nounroll"` backward branch |
| absent everywhere | `bar.sync`, `ld.shared`, `st.shared`, `mbarrier`, `cp.async`, `vote.sync`, `match.any`, `red.`, `fence`, `ld.global.nc` |

Static opcode counts of the exported entry (230 instructions,
`.reg .b32 %r<116>`, `.reg .pred %p<47>`, `.reg .b64 %rd<27>`):

| op | count | op | count |
| --- | ---: | --- | ---: |
| `add.s32` | 35 | `sub.s32` | 8 |
| `mov.b32` | 28 | `xor.b32` | 6 |
| `setp.lt.s32` | 17 | `st.global.b32` | 6 |
| `selp.b32` | 15 | `div.s32` | 6 |
| `setp.ne.b32` | 13 | `shfl.sync.idx.b32` | 4 |
| `add.s64` | 11 | `min.s32` | 3 |
| `ld.global.b32` | 10 | `max.s32` | 1 |
| `mul.lo.s32` | 9 | `atom.global.add.u32` | 1 |
| `and.pred` | 9 | `bar.sync` | **0** |

`ld.global.b32 = 10`, `st.global.b32 = 6`, `shfl.sync.idx.b32 = 4` and
`atom.global.add.u32 = 1` are the counts that pin the port's key operations:
two CSR bounds plus eight scan loads, one six-store work item, one four-way
broadcast, and one slot reservation. `xor.b32 = 6` equals `div.s32 = 6`, one
correction per floor division — the count that identifies the source-language
lowering rather than an algorithmic step. `bar.sync = 0` is the proof that the
kernel has no CTA-level synchronization at all.
