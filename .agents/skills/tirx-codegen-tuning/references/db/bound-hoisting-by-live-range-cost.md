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

## Rationale

One measured FP32 bias hoist regressed 6.6%; by contrast, staging a larger load
set won when it created enough outstanding DRAM misses.

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

## Boundary

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

## Verification

Compare registers and dynamic LDL/STL before instruction count, and sweep the
tightest specialization where one spill can reverse the result. Verify the
compute, reduction, first-publication, and release order in emitted PTX/SASS
before treating a shorter lifetime as a legal candidate.
