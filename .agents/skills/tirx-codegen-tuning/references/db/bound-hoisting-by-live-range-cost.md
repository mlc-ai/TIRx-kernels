# Bound hoisting by live-range cost

**Symptoms:** `register_pressure`, `local_memory_traffic`, `low_occupancy`

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

## Rationale

One measured FP32 bias hoist regressed 6.6%; by contrast, staging a larger load
set won when it created enough outstanding DRAM misses.

## Boundary

Do not shorten a fragment lifetime across an ordering that belongs to the
correctness contract. One vector-at-a-time epilogue reduced registers from 96
to 94 and moved a ratio from 0.979x to 0.981x, but it also stored each vector
before the remaining fragment had completed its multiply, bias, and narrowing
phases. The measured gain was rejected and the full-fragment phase order was
restored.

## Verification

Compare registers and dynamic LDL/STL before instruction count, and sweep the
tightest specialization where one spill can reverse the result. Verify the
compute and store order in emitted PTX/SASS before treating a shorter lifetime
as a legal candidate.
