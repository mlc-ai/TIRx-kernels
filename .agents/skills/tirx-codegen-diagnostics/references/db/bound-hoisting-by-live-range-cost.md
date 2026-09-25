# Bound hoisting by live-range cost

**Symptoms:** `register_pressure`, `local_memory_traffic`, `low_occupancy`, `underfilled_pipeline`, `schedule_regression`

## Symptom

Register pressure or dynamic local traffic after hoisting work out of a loop: a
small operation whose results remain live across a recurrent loop, large
fragment, or synchronization chain.

## What to change

Hoist work only when hidden latency outweighs the added lifetime. Tile wide
epilogue fragments so only the next consumed tile remains live: allocate the
narrow fragment inside the tile loop rather than one wide buffer outside it.

```python
# before: every chunk stays live between the loads and the stores.
reg_all_f32 = T.alloc_local((MMA_N,), "float32")
for no in T.unroll(MMA_N // EPI_TILE):
    _load_chunk(reg_all_f32, no * EPI_TILE)
for no in T.unroll(MMA_N // EPI_TILE):
    _cast_and_store(reg_all_f32, no * EPI_TILE)

# after: the wide fragment never exists; one tile is live at a time.
reg_words = T.alloc_local((EPI_TILE // 2,), "uint32", align=16)
for no in T.unroll(MMA_N // EPI_TILE):
    reg_f32 = T.alloc_local((EPI_TILE,), "float32")
    _load_chunk(reg_f32, T.meta_var(no * EPI_TILE))
    _cast_chunk(reg_words, reg_f32)
    _store_chunk(reg_words, T.meta_var(no * EPI_TILE))
```

Apply the same rule when a wide producer fragment feeds both a consumer and a
chunk-local reduction. Publish one chunk after its local work instead of keeping
the whole fragment live until every chunk is ready.

```python
# before: all chunks remain live until publication begins.
for chunk in T.unroll(CHUNKS):
    _compute_fragment(fragment[chunk])
for chunk in T.unroll(CHUNKS):
    _reduce_chunk(fragment[chunk])
    _publish_chunk(fragment[chunk])

# after: finish and publish one chunk at a time.
for chunk in T.unroll(CHUNKS):
    _compute_fragment(fragment[chunk])
    _reduce_chunk(fragment[chunk])
    _publish_chunk(fragment[chunk])
```

Likewise, when one pass produces public metadata and a derived value used by a
single output, consume that derived value at its production point instead of
retaining a second wide array for a later pass.

```python
# before: every derived value remains live until the second pass.
derived = T.alloc_local((ROWS,), "float32")
for row in T.unroll(ROWS):
    metadata = _compute_metadata(row)
    _publish_metadata(row, metadata)
    derived[row] = _derive(metadata)
for row in T.unroll(ROWS):
    _produce_output(row, derived[row])

# after: metadata publication and its dependent output share one lifetime.
for row in T.unroll(ROWS):
    metadata = _compute_metadata(row)
    _publish_metadata(row, metadata)
    _produce_output(row, _derive(metadata))
```

When two same-shaped inputs exist only to form one output, reuse those inputs
instead of allocating a third fragment. Scale or clear the first input in
place, scale the second into its final temporary form, then accumulate it into
the first.

```python
# before: three fragments are simultaneously live.
combined = txl.alloc_local((WIDTH,), "float32")
_scale(values0, scale0, combined)
_scale_add(values1, scale1, combined)

# after: values0 is the combined result and values1 is the scaled temporary.
_scale_in_place(values0, scale0)
_scale_in_place(values1, scale1)
_add_in_place(values0, values1)
```

## Rationale

One measured FP32 bias hoist regressed 6.6%; by contrast, staging a larger load
set won when it created enough outstanding DRAM misses.

Wider TMEM rescaling fragments can cross the same boundary. Increasing a
packed-FP32 online-softmax rescale from x32 to x128 reduced eight load/store
pairs to two, but increased registers from 88 to 168 and introduced a 56-byte
stack. All fifteen targeted numerical cases passed; the large-prefill
after/before latency ratio nevertheless worsened from 1.255 to 1.304 in
three-round paired measurements. Fewer TMEM waits did not repay the spills.

Repeated inline-assembly helper calls in generated CUDA are not sufficient
evidence that another hoist will help. One tiled decay loop emitted four or
eight identical shared beta loads per iteration. Moving the load and its two
scaled operands outside the tile's unrolled body left the SASS instruction,
shared-load, and local-load counts unchanged in both kernel families. Large
strong-decay timings stayed at about 4.80 ms and 513 us. Inspect the assembled
code before treating repeated generated-source expressions as surviving work.

The same tradeoff applies to persistent reductions. Moving a nonnegative amax
from a per-work shared reduction and atomic into a per-lane value carried across
the persistent loop reduced the number of global atomics, but the loop-carried
dependency stayed live across every epilogue. After repairing collective
deallocation ordering, the once-per-CTA form was still about 0.1 us slower on
the FP8 guards and only 2/5 targeted rows passed. Reducing a tail operation
count did not repay the longer recurrent live range.

In another pipeline, chunking a wide producer fragment under the same register
caps and with zero spilling reduced elapsed cycles from 23,433 to 23,294 and
executed instructions from 11,495 to 11,406. The two critical benchmark ratios
moved from 0.981x/0.989x to 0.987x/0.997x; a later role-budget adjustment supplied
the remaining margin without undoing the shorter fragment lifetime.

In a multi-output epilogue, publishing a 32-element metadata array at production
reduced stack use from 72 to 8 bytes and static local-memory instructions from 34
to 2; two production workloads improved by 4.75% and 4.91%. Consuming the
remaining 32-element derived array at production then reduced registers from 128
to 102, eliminated the stack and static local-memory instructions, and improved
the same workloads by another 1.12% and 0.88%. Correctness passed for both output
orientations after each rewrite.

In a two-stream correction epilogue, three 16-float fragments plus a two-float
temporary were live under a 64-register role budget. Reusing the two input
fragments in place removed eighteen live FP32 values. Two representative paths
improved from 87.407 to 80.320 us and from 64.513 to 49.870 us, moving their
reference ratios from 0.953x to 1.039x and from 0.923x to 1.191x. Tight source
and independent-oracle comparisons passed. The shorter lifetime also changed
the best role-register split, so the neighboring budget sweep had to be repeated.

A grouped-head gradient epilogue retained two sixteen-float reciprocal arrays
across query output although only key output consumed them. Computing each
two-float pair at that consumer preserved the reciprocal and masking operations
while reducing two specializations' stacks from 144/152 to 88/96 bytes. Local
load/store sites decreased from 123/82 to 110/70 and from 120/81 to 103/65;
special-function counts were unchanged. Two representative initialization
profiles improved 4.1–4.9%, and all 39 correctness configurations retained all
six output tensors bitwise. This is a lifetime change, not an approximation or
a change to the state representation.

## Boundary

A two-pass gradient epilogue computed a reciprocal in its first pass and then
recomputed either that reciprocal or a bounded state multiplier in the second.
Reusing the existing first-pass registers for the state multiplier removed
32 static special-function sites without adding a fragment. Five cases stayed
bitwise equal; mixed and dense workloads improved 1.6–2.5% with an unchanged
96-byte stack. An independent alternative computed each multiplier at its last
use and removed both 16-element arrays. This reduced local load/store sites
from 55/35 to 43/23 and improved the same workloads another 0.7–3.0% against
the reuse version, again bitwise equal. The register allocation and stack size
did not change, so inspect local instruction sites as well as allocation totals
when comparing hoisting with late evaluation.

Do not shorten a fragment lifetime across an ordering that belongs to the
correctness contract. One vector-at-a-time epilogue reduced registers from 96
to 94 and moved a ratio from 0.979x to 0.981x, but it also stored each vector
before the remaining fragment had completed its multiply, bias, and narrowing
phases. The measured gain was rejected and the full-fragment phase order was
restored.

State hoisted across persistent work also creates a dependency between work
items that were previously independent. Measure both one-work CTAs, where the
hoist can only add lifetime, and high-trip-count CTAs, where removing repeated
tail operations has a chance to repay it.

Lower resource counts do not guarantee a win on short work. The derived-array
rewrite above made a one-work guard 24.0% slower even though it removed the
remaining stack traffic, while the persistent production workloads improved.
Keep short and persistent shapes in the validation matrix when changing phase
boundaries.

Do not publish earlier than the chunk's own correctness and scheduling boundary.
Moving publication and release ahead of the chunk-local reduction reduced one
profile from 23,294 to 22,771 cycles, but its critical benchmark ratio regressed
from 0.987x to 0.986x. The shortest apparent lifetime was not the best accepted
schedule; the final form kept reduction before publication.

In-place reuse is legal only after every use of the overwritten input has been
accounted for, including later scale, normalization, and packed-conversion
branches. Preserve the original arithmetic and conversion order when tight
comparison requires it; this transformation is about storage lifetime, not
reassociation.

Keeping a rounded gate cache packed until its derivative phase was a limited
counterexample. Although the early source representation shrank from32FP32
values to16packed words, the assembled104-byte stack and57/36local load/store
sites were unchanged. Four paired profiles were bitwise identical. Ordinary
latency increased0.62%, while initialization and sparse correction improved
1.74%and1.31%, and dense correction was effectively flat. This did not resolve
the ordinary-path regression; do not infer register relief from source packing
alone when the eventual expanded epilogue still binds allocation.

Sinking two sixteen-element intra-gate reciprocal arrays into their final
two-element consumer was also specialization-dependent. In a compensated
epilogue, the rewrite left the 96-byte stack and 44/26 local load/store sites
unchanged. Three same-workspace comparisons retained all output bits, but two
initialization profiles regressed 0.6–1.2%; the ordinary control was effectively
flat. Removing source arrays does not guarantee a shorter assembled lifetime,
especially when ptxas already schedules across the source phase boundary.
An ungrouped-head specialization of the successful grouped-head rewrite kept
its 32-byte stack: initialization improved 0.6–1.4%, but a dense correction
control regressed 1.1%. Preserve that boundary when reporting the benefit;
smaller local instruction counts alone do not establish a gain on every path.

The ordinary specialization of that epilogue retained its 16-byte stack and
26/36 local load/store sites after the same reciprocal sinking. Assembly
scheduling changed, but two ordinary profiles improved only 0.4–0.6%; two
inactive-path controls changed between -0.06% and +0.12%. Four paired checks
retained all output bytes. This limited gain did not resolve the operation's
remaining 5% latency target, and source-array removal was not spill relief.

Splitting at a semantic boundary can make allocation worse when the compiler
changes its mapping of the remaining fragment. In an independent D128/V128
backward prototype, peeling the final token into its own kernel left the body
with one fewer row but changed it from 255 registers and roughly 650 reported
spills to 128 registers and roughly 780 spills. The complete six-output call
regressed from 15.50x to 16.58x and from 17.48x to 18.67x versus the original
on two workloads. All ten targeted FP64-oracle checks still passed. The
source-level reduction in work did not establish a shorter realized lifetime.

By contrast, publishing the large gradient matrices between two kernels cut
the two stages to roughly 178 and 150 reported spills. The same two complete
calls improved from 15.67x/17.51x to 12.62x/14.00x, with a six-output FP64
smoke check passing. The extra global workspace grew from about 0.61/1.08 GB
to 0.76/1.36 GB. This confirms live matrices as a material cost, but the result
is still far outside a 5% operation-level limit; lower spill totals do not by
themselves justify a phase split or its storage traffic.

Capturing a diagonal before shared-memory reuse showed the same tradeoff at
three representations. Holding 32 FP32 values across the epilogue regressed a
complete long workload by 21.72%; holding 16 raw BF16x2 words regressed it by
11.24%. Reducing storage to one BF16 value per lane and broadcasting each pair
with warp shuffles still regressed 11.17%. Shorter source storage can replace
register lifetime with communication instructions rather than remove the cost.
Measure the full consumer sequence, including broadcasts and conversions, not
only the number of retained source values.

Conditional reuse of a dead unrolled array can be much worse than leaving the
original live range in place. One cold correction overwrote sixteen packed
words only when a uniform risk bit was set; although each word was dead after
its query-side use, the conditional definitions made the assembled stack grow
from 56 to 168 bytes and static local loads/stores from 35/46 to 84/90. Restoring
the original array and re-reading the correction inputs later grew the stack to
216 bytes and local sites to 94/101. Moving all query corrections into one
post-loop branch reduced static branch sites, but extended the intermediate
array across the split and grew the stack to 200 bytes with 125/108 local
loads/stores. A source-level dead range or fewer branches does not imply a cheap
conditional phi across an unrolled array; compile an ablation before timing it.

Packing the same diagonal into shared memory removed the cross-phase FP32 array
and reduced the native epilogue stack from 56 to 48 bytes. Static local
load/store sites also fell under the same counting script, but the short full
operation first measured +4.90% and then +6.34% in a confirmation run; the ten
samples combined were about +5.61%. The long operation measured +4.79%.
Resource relief therefore did not robustly satisfy a 5% operation limit.
Keeping sixteen expanded `uint64` pairs or sixteen raw `uint32` BF16 words
across the query/key split instead produced a 112-byte stack in both cases.
Deleting every array and re-reading scalar values produced a 168-byte stack.
Cross-loop conditional arrays and the phi values created by source-level
scalarization both need assembled-code checks; source storage width alone does
not predict the allocator result.

## Verification

Compare registers and dynamic LDL/STL before instruction count, and sweep the
tightest specialization where one spill can reverse the result. Verify the
compute, reduction, first-publication, and release order in emitted PTX/SASS
before treating a shorter lifetime as a legal candidate. For in-place reuse,
also test zero/nonzero scale branches and every output packing type, then repeat
the role-budget sweep because the prior optimum may no longer apply.
