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
SparseAttentionPrepareFwdSplitAtomicSm100.
-->

# msa_sparse_prepare_fwd_split_atomic_sm100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/msa/sparse_prepare_fwd_split_atomic.py`](../../../tirx_kernels/msa/sparse_prepare_fwd_split_atomic.py).
That TIRx module is the authoritative implementation.

The port covers **one kernel**: `SparseAttentionPrepareFwdSplitAtomicSm100.kernel`
(`prepare_scheduler.py:432-481`) with its launcher `__call__` (`:366-430`), the
stage that hands every CSR edge a split slot so the forward attention kernel can
write its partial output without a second round of atomics.

Instantiations: **exactly one**. The compile shim keys its cache on the constant
tuple `("sparse_prepare_fwd_split_atomic_sm100_csr_varlen",)` (`:496-498`) and
`to_cute_tensor` marks every layout dynamic, so one binary serves every shape;
all extents, strides and scalars are runtime. The compile-time constants are
`num_threads = 256` (`:354`) and the three-slot shared struct (`:360-364`).

Accepted target SM100/B200. Out of scope, with the predicate that excludes each:
the host rejects `topk > 255` (`:673`) and `max_seqlen_q >= 2^24` (`:675`); the
default adapter path builds its schedule with the fused CUDA builder and skips
this kernel entirely, so it runs only when the caller passes no prebuilt
schedule or limits `usable_SM_count`. The producing sibling
`SparseAttentionPrepareFlatScheduleSm100` (`:124-345`) is a separate kernel with
its own cache key, already ported. Tile (`Tx`) primitives are out of scope
everywhere.

## Pipeline at a glance

| Kernel / role | Program | Publication / reuse edges |
| --- | --- | --- |
| **CTA `b`, all 256 threads** (`:445-455`) | read `work_count`, exit if `b >= work_count`; load `head_kv_idx` from the work item | `head_kv_idx` stays in every thread's registers — it is the one metadata field the emission loop needs per thread, and the export keeps its load unsunk for exactly that reason |
| **CTA `b`, thread 0 only** (`:457-461`) | load `row_linear`, `q_begin`, `q_count`, `batch_idx` from the work item; load the row's CSR base; publish `(row_start, q_count, batch_idx)` into `sRow[0..2]` | Publishes to shared memory. nvcc **sinks** those four metadata loads into this block, so the other 255 threads never issue them — which is what makes all three shared slots load-bearing rather than redundant. |
| **CTA `b`, all 256 threads** (`:462-481`) | one `bar.sync`; read the three fields back from `sRow`; stride `qi += 256` over the row's edges, reserving a slot per valid edge with a device-scope atomic and writing the packed field under the capacity guard | Consumes `sRow`. Publishes to global memory only. Slot order is the atomic arrival order, so the schedule is defined as a **per-`(q_abs, head_kv)` permutation**, not a fixed assignment. |

There is no producer/consumer warp split, no pipe, no stage, no phase, no
mbarrier, and no async copy. The only intra-CTA edge is the single `bar.sync`
separating thread 0's three shared stores from everyone's three shared loads,
and the only cross-CTA edge is the atomic.

**One CTA owns one work item.** The grid is `work_capacity` CTAs of 256 threads
— sized by the work list's *capacity*, not its length, because `work_count`
lives on the device and is never read back to the host. On the decode shapes
that leaves most CTAs doing nothing but the early-out.

## Primitive vocabulary

```python
load_global(buf, idx) -> reg          # ld.global.b32 | ld.global.s32 (address operands)
store_global(buf, idx, reg)           # st.global.b32
atom_add_global(buf, idx) -> old      # atom.global.add.u32 d, [a], 1
store_shared(smem, idx, reg)          # st.shared.b32
load_shared(smem, idx) -> reg         # ld.shared.b32
barrier()                             # bar.sync 0
pack(q_idx, slot)                     # shl.b32 24 + or.b32
```

### The atomic is raw inline PTX with an immediate operand

Unlike the sibling kernel, which asks for `sem="relaxed", scope="gpu"` and lets
the DSL choose the encoding, this kernel calls `copy_utils.atomic_add_i32`
(`src/common/copy_utils.py:159-172`), which *is* the instruction:

```python
llvm.inline_asm(T.i32(), [ptr], "atom.global.add.u32 $0, [$1], 1;\n", "=r,l", ...)
```

The export carries it verbatim inside `// begin inline asm` markers, with the
addend as the **immediate `1`**. TIRx's `T.ptx.atom.global_.add.u32` takes the
addend as a register operand, so the port emits `mov.b32 %r, 1` plus the same
instruction — an operand-form difference ptxas folds, already recorded from the
sibling port. Relaxed ordering and device scope remain the defaults: no
`.relaxed`, no `.gpu`, no surrounding fence.

### The shared storage is dynamic, and its base is broadcast

The source's `@cute.struct SharedStorage` reads like a static `__shared__` array,
but `cutlass.utils.SmemAllocator` allocates from the dynamic pool, and the export
broadcasts the base pointer across the warp:

```
.extern .shared .align 1024 .b8 __dynamic_shmem__0[];
...
.loc 1 432   mov.b32           %r8, __dynamic_shmem__0;
             shfl.sync.idx.b32 %r2, %r8, 0, 31, -1;
```

So the matching TIRx form is a `T.SMEMPool()` allocation plus the
`tirx.use_dyn_shared_memory` launch tag, not a static `T.alloc_shared`. The
declared size is 12 bytes (`3 * int32`, no padding); the `.align 1024` is the
pool's alignment, not the struct's.

### `.file` map of the export

`.file 1` = `prepare_scheduler.py`, `.file 2` = `src/common/utils.py` (only
`elem_pointer`'s `shl.b64` at `:371`). Line info comes from a fresh
`CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1 CUTE_DSL_NO_CACHE=1` export preserved at
`.porting/sparse_prepare_fwd_split_atomic/ptx_lineinfo/source_split_atomic_lineinfo.sm_100a.ptx`
(42 `.loc`, 73 instructions). Every `.loc` cited below uses that file.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
variant = specialize(NUM_THREADS=256, SROW_FIELDS=3, target="sm_100a")
# instruction_selection: none; extent: the only compile-time constants (:354, :360-364)
SLOT_SHIFT, SLOT_MASK = 24, 0xFF          # packing constants (:479)

# Runtime ABI -- all int32, all global, all `assume_tensor_aligned` (:395-414).
#   k2q_row_ptr        [head_kv, total_rows + 1]      read-only
#   k2q_q_indices      [head_kv, total_q * topk]      read-only  (-1 past nnz)
#   scheduler_metadata [work_capacity, 6]             read-only  (cols 0..4; col 5 unused)
#   work_count         [1]                            read-only  (the early-out bound)
#   k2q_qsplit_indices [head_kv, total_q * topk]      write-only, sparse, UNINITIALIZED
#   split_counts       [total_q, head_kv]             read-modify-write, must arrive zeroed
#   cu_seqlens_q       [num_batches + 1]              read-only
# Runtime scalars: max_seqlen_q, topk (+ the extents the port passes explicitly).

launch_config = launch(grid=(work_capacity, 1, 1), block=(256, 1, 1),
                       dynamic_smem_bytes=12)
# instruction_selection: none; extent: static launch metadata (:425-430).
#   grid.x is the work list's CAPACITY, not its length -- `work_count` is device
#   resident, so the launch is deliberately oversized and the tail CTAs exit.

# ===========================================================================
# Storage.  One 12-byte dynamic shared allocation; everything else is registers.
# ===========================================================================
sRow = smem_pool.alloc((3,), "int32")     # SharedStorage.sRow (:360-364, :448-450)
# instruction_selection: mov.b32 of __dynamic_shmem__0 + shfl.sync.idx.b32
#   broadcast of the base (.loc 1 432); extent: one allocation per CTA.

def kernel():
    tidx  = thread_id(extent=NUM_THREADS)   # instruction_selection: mov.u32 %tid.x (.loc 1 445)
    block = cta_id(axis="x")                # instruction_selection: mov.u32 %ctaid.x (.loc 1 446)

    # -----------------------------------------------------------------------
    # Stage 1: the oversized-grid early-out (:447)
    # -----------------------------------------------------------------------
    produced = load_global(work_count, 0)
    # instruction_selection: ld.global.b32 (.loc 1 447 19); extent: one scalar,
    #   issued by every thread of every CTA including the ones about to exit
    if block >= produced:
        return
    # instruction_selection: setp.ge.s32 + predicated branch to the exit
    #   (.loc 1 447 7); extent: CTA-uniform branch. It precedes every shared
    #   access, so the barrier below is never reached by a partial CTA.

    # -----------------------------------------------------------------------
    # Stage 2: the one metadata field every thread needs (:451)
    # -----------------------------------------------------------------------
    head_kv_idx = load_global(scheduler_metadata, block * 6 + 0)
    # instruction_selection: ld.global.s32 (.loc 1 451 22) -- sign-extending,
    #   because the value feeds 64-bit address arithmetic; extent: one scalar
    #   per thread.  This load stays UNSUNK: every thread uses it in the
    #   emission loop.

    # -----------------------------------------------------------------------
    # Stage 3: thread 0 publishes the row through shared memory (:457-461)
    # -----------------------------------------------------------------------
    if tidx == 0:
        # instruction_selection: setp.ne.b32 + predicated branch (.loc 1 457 11);
        #   extent: intra-CTA branch -- 255 threads jump straight to the barrier
        batch_idx = load_global(scheduler_metadata, block * 6 + 4)
        # instruction_selection: ld.global.b32 (.loc 1 455 23); extent: one scalar
        q_count   = load_global(scheduler_metadata, block * 6 + 3)
        # instruction_selection: ld.global.b32 (.loc 1 454 18); extent: one scalar
        q_begin   = load_global(scheduler_metadata, block * 6 + 2)
        # instruction_selection: ld.global.b32 (.loc 1 453 18); extent: one scalar
        row_linear = load_global(scheduler_metadata, block * 6 + 1)
        # instruction_selection: ld.global.s32 (.loc 1 452 21) -- sign-extending,
        #   same reason as head_kv_idx; extent: one scalar.
        # THESE FOUR ARE SUNK HERE BY THE COMPILER.  The source reads all five
        # columns with every thread; only head_kv_idx survives that way, because
        # only it is used outside this block.  The consequence is that all three
        # shared slots below are load-bearing -- the other 255 threads never
        # issue these loads at all.
        row_base = load_global(k2q_row_ptr, head_kv_idx * (total_rows + 1) + row_linear)
        # instruction_selection: mad.lo.s64 + shl.b64 + add.s64 + ld.global.b32
        #   (.loc 1 458 27); extent: one scalar -- the extra load thread 0 exists
        #   to amortize
        store_shared(sRow, 0, row_base + q_begin)                       # (:459)
        # instruction_selection: add.s32 + st.shared.b32 [base] (.loc 1 459 12); extent: one
        store_shared(sRow, 1, q_count)                                  # (:460)
        # instruction_selection: st.shared.b32 [base+4] (.loc 1 460 12); extent: one
        store_shared(sRow, 2, batch_idx)                                # (:461)
        # instruction_selection: st.shared.b32 [base+8] (.loc 1 461 12); extent: one

    barrier()                                                           # (:462)
    # instruction_selection: bar.sync 0 (.loc 1 462 8); extent: CTA-wide.  The
    #   only synchronization in the kernel.

    # -----------------------------------------------------------------------
    # Stage 4: everyone reads the row back (:463-465)
    # -----------------------------------------------------------------------
    row_count = load_shared(sRow, 1)
    # instruction_selection: ld.shared.b32 [base+4] (.loc 1 464 20); extent: one.
    #   Read FIRST in the export, because it is the loop guard below.
    row_start = load_shared(sRow, 0)
    # instruction_selection: ld.shared.b32 [base] (.loc 1 463 20); extent: one
    batch_idx = load_shared(sRow, 2)
    # instruction_selection: ld.shared.b32 [base+8] (.loc 1 465 20); extent: one

    # -----------------------------------------------------------------------
    # Stage 5: lane-strided edge emission (:466-481)
    # -----------------------------------------------------------------------
    qi = tidx                                                           # (:466)
    while qi < row_count:                                               # (:467)
        # instruction_selection: setp.ge.s32 to skip the loop entirely, then
        #   setp.lt.s32 + backward branch for the back edge (.loc 1 467 14);
        #   extent: ceil(row_count / 256) iterations per thread.  The source
        #   carries NO nounroll pragma here and nvcc leaves it rolled anyway;
        #   the trip count is the runtime, data-dependent row_count.
        edge = row_start + qi                                           # (:468)
        # instruction_selection: add.s32 (.loc 1 468 19); extent: scalar
        q_idx = load_global(k2q_q_indices, head_kv_idx * nnz_capacity + edge)
        # instruction_selection: cvt.s64.s32 + add.s64 + shl.b64 + add.s64 +
        #   ld.global.b32 (.loc 1 469 20); extent: one scalar per iteration.
        #   The row base is hoisted above the loop (mul.lo.s64); only the
        #   per-edge displacement is computed inside.
        if q_idx >= 0 and q_idx < max_seqlen_q:                         # (:470)
            # instruction_selection: setp.lt.s32 + setp.ge.s32 + or.pred +
            #   predicated branch (.loc 1 470 15, 470 37); extent: scalar.  A
            #   real branch, not if-conversion.  Well-formed input never takes
            #   it -- the CSR row covers only filled entries -- but the guard
            #   exists because the GPU index builder leaves its tail
            #   uninitialized and the reference builder fills it with -1.
            q_abs = load_global(cu_seqlens_q, batch_idx) + q_idx        # (:471)
            # instruction_selection: ld.global.b32 + add.s32 (.loc 1 471 24);
            #   extent: one scalar PER ITERATION.  The address is hoisted
            #   (mul.wide.s32 + add.s64 above the loop) but the load is not:
            #   the atomic's memory clobber keeps this loop-invariant value
            #   being re-fetched every edge.
            slot = atom_add_global(split_counts, q_abs * num_heads_kv + head_kv_idx)
            # instruction_selection: cvt.s64.s32 + mad.lo.s64 (.loc 1 472 28) +
            #   shl.b64 + add.s64 (.loc 2 371 11, elem_pointer) then
            #   `atom.global.add.u32 %r, [%rd], 1` inside inline-asm markers
            #   (.loc 1 476 29); extent: one instruction per valid edge.
            #   Returns the pre-increment value.  Relaxed, device scope, no fence.
            if slot < topk:                                             # (:477)
                # instruction_selection: setp.ge.s32 + predicated branch
                #   (.loc 1 477 19); extent: scalar.  SIGNED compare against the
                #   u32 the atomic returned.  Also never taken on well-formed
                #   input, where a (q_abs, head) group has at most topk edges.
                store_global(k2q_qsplit_indices,
                             head_kv_idx * nnz_capacity + edge,
                             q_idx | ((slot & SLOT_MASK) << SLOT_SHIFT))
                # instruction_selection: shl.b32 by 24 (.loc 1 479 33) + or.b32
                #   (.loc 1 479 24) + add.s64 + shl.b64 + add.s64 +
                #   st.global.b32 (.loc 1 478 20); extent: one store per written
                #   edge.  The `& 0xFF` folds away -- ptxas knows the shift
                #   discards everything above bit 7 -- so the export carries a
                #   bare shl+or, no `and.b32`.
        qi += 256                                                       # (:481)
        # instruction_selection: add.s32 with the immediate 256 (.loc 1 481 12);
        #   extent: scalar
```

## Control-flow map

| Region | Source | Predicate | Who executes |
| --- | --- | --- | --- |
| oversized-grid early-out | `:447` | `block_idx < work_count[0]` | CTA-uniform; precedes every shared access, so the barrier is never partial |
| row publisher | `:457` | `tidx == 0` | one thread per CTA; the other 255 branch to the barrier |
| emission loop | `:467` | `qi < row_count` | all 256 threads, stride 256, rolled |
| edge validity | `:470` | `0 <= q_idx < max_seqlen_q` | per edge; a real branch, unreachable on well-formed input |
| slot capacity | `:477` | `split_slot < topk` | per valid edge; drops the store, keeps the atomic; unreachable on well-formed input |

Per valid edge the kernel issues two `ld.global.b32`, one
`atom.global.add.u32`, and one predicated `st.global.b32`. That is the whole
cost model: the prefill shapes are edge-bound, and the decode shapes are
dominated instead by CTAs that take the first branch and retire.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `num_threads = 256` | static (`:354`) | block shape and the `qi += 256` stride immediate |
| `SharedStorage` = 3 x int32 | static (`:360-364`) | 12 dynamic-pool bytes and the `+0/+4/+8` displacements |
| metadata columns 0..4, stride 6 | static | the `[%rd2]`, `+4`, `+8`, `+12`, `+16` displacements off one base |
| `0xFF`, `24` | static (`:479`) | `shl.b32` + `or.b32`; the mask folds away |
| `work_capacity` | **runtime**, host-only | becomes `grid.x`; never a kernel argument in the source |
| `work_count[0]` | **runtime**, device-resident | why the grid is oversized in the first place |
| `max_seqlen_q`, `topk` | **runtime** | guard operands stay registers, never immediates |
| every extent and stride | **runtime** | dynamic layouts; one binary for all shapes |
| slot assignment | **non-deterministic** | atomic arrival order; only the per-group permutation is defined |
| `split_counts` initial value | **caller contract** | zeroed by the host before every launch (`:695-696`), never by the kernel |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "msa_sparse_prepare_fwd_split_atomic_sm100", "category":
  "msa", "compute_capability": 10}`.
- Plain TIRx only: a `T.SMEMPool()` arena for `sRow`, explicit loops, scalar
  registers via `T.alloc_local`, and native `T.ptx.*` forms for every key
  operation. Global **and shared** memory are reached exclusively through
  `T.ptx.ld/st.*` on `buffer.ptr_to([...])` — the repository's low-level IR
  contract forbids a native `BufferLoad`/`BufferStore` in either scope. No
  `T.cuda.func_call`. No `Tx` tile primitives anywhere.
- The 2-D source tensors are matched as flat 1-D buffers, with the row strides
  passed as scalars, which reproduces the export's addresses without a dynamic
  stride operand.
- Correctness compares against **MSA's own compiled kernel** over identical
  inputs, plus a host oracle built from the same edge list. Because slot
  assignment is arrival-ordered, the comparison is over invariants, not values:
  `split_counts` element-wise; a written entry's low 24 bits equal that edge's
  `q_idx`; and per `(q_abs, head_kv)` the slots are exactly `{0..degree-1}`,
  gapless and without duplicates. That last one is the consumer's requirement,
  not a testing convenience — the forward kernel writes `split_counts` partials
  into `O_partial[0..count-1]` and combine reduces exactly that many. The oracle
  reads only each head's live CSR range, since `k2q_qsplit_indices` arrives
  uninitialized.
- The benchmark is **kernel-only** on both sides: the timed closure holds one
  launch, with the CSR build, the work list, the allocation and every zeroing
  outside it — the same line MSA's own instrumentation draws by putting
  `split_counts.zero_()` in a separate NVTX range. `split_counts` is
  read-modify-write and the kernel never resets it, so both sides rotate over
  pre-zeroed slices and re-zero once per rotation; without that, the second and
  every later launch finds `split_slot >= topk` everywhere and the store path
  stops firing, leaving a strictly cheaper kernel being timed.

## Instruction selection is a lowering consequence

| Primitive / pattern | PTX family (fresh `sm_100a` lineinfo export) |
| --- | --- |
| `work_count` load (`:447`) | `ld.global.b32` |
| `head_kv_idx`, `row_linear` (`:451`, `:452`) | `ld.global.s32` — sign-extending, they feed 64-bit addressing |
| `q_begin`, `q_count`, `batch_idx` (`:453-455`) | `ld.global.b32`, **sunk into the `tidx == 0` block** |
| CSR row base (`:458`) | `mad.lo.s64` + `shl.b64` + `add.s64` + `ld.global.b32` |
| shared publish (`:459-461`) | `st.shared.b32` x3 at `+0/+4/+8` |
| barrier (`:462`) | `bar.sync 0` |
| shared read-back (`:463-465`) | `ld.shared.b32` x3; `q_count` first, it is the loop guard |
| edge `q_idx` load (`:469`) | `ld.global.b32`, row base hoisted |
| validity guard (`:470`) | `setp.lt.s32` + `setp.ge.s32` + `or.pred` + branch |
| `cu_seqlens_q[batch_idx]` (`:471`) | `ld.global.b32` **per iteration** — address hoisted, load not |
| slot reservation (`:476`) | `atom.global.add.u32 d, [a], 1` inside inline-asm markers, immediate addend |
| capacity guard (`:477`) | `setp.ge.s32` + branch, signed |
| packing (`:479`) | `shl.b32 24` + `or.b32`; the `& 0xFF` folds away, no `and.b32` |
| work-item store (`:478`) | `st.global.b32`, row base hoisted |
| smem base | `mov.b32 __dynamic_shmem__0` + `shfl.sync.idx.b32` broadcast |
| absent everywhere | `mbarrier`, `cp.async`, `fence`, `red.*`, `vote.sync`, `match.any`, `ld.global.nc`, `ld/st.global.v2\|v4`, any `div`, `.relaxed`, `.gpu` |

Static opcode counts of the exported entry (73 instructions,
`.reg .b32 %r<23>`, `.reg .b64 %rd<37>`, `.reg .pred %p<9>`):

| op | count | op | count |
| --- | ---: | --- | ---: |
| `add.s64` | 8 | `mad.lo.s64` | 2 |
| `ld.global.b32` | 7 | `ld.global.s32` | 2 |
| `shl.b64` | 5 | `cvt.s64.s32` | 2 |
| `setp.ge.s32` | 4 | `st.global.b32` | 1 |
| `add.s32` | 4 | `shfl.sync.idx.b32` | 1 |
| `st.shared.b32` | 3 | `atom.global.add.u32` | 1 |
| `mul.lo.s64` | 3 | `bar.sync` | 1 |
| `ld.shared.b32` | 3 | `or.pred` / `or.b32` | 1 / 1 |
| `setp.lt.s32` | 2 | `shl.b32` | 1 |

`st.shared.b32 = ld.shared.b32 = 3` with a single `bar.sync` between them is the
whole synchronization protocol. `ld.global.b32 = 7` decomposes as one
`work_count`, three sunk metadata fields, one CSR base, and the two per-edge
loads; `atom.global.add.u32 = 1` and `st.global.b32 = 1` are the emission tail.
The absence of `and.b32` is the check that the port folded the `& 0xFF` rather
than emitting it, and `div` being absent everywhere is why the sibling port's
unsigned-division lever has nothing to act on here.
