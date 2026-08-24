# Run independent work under a store in flight

**Symptoms:** `fence_stall`, `epilogue_serialization`, `bulk_store_latency`, `latency_bound_epilogue`

## Symptom

A reduction or side computation sits between the activation and the bulk store
that ends a subtile, and the kernel is latency-bound rather than issue-bound.

## What to change

Move the independent work after `cp.async.bulk.commit_group`, so it runs while
the store is in flight -- and put it after the *barrier* that precedes the
store, not merely after the shared writes.

```python
K.ptx["fence.proxy.async.shared::cta"]()
K.ptx.bar.sync(K.uint32(2), K.uint32(128))
with K.If(warp == K.int32(0)), K.Then():
    K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group..."](...)
    K.ptx["cp.async.bulk.commit_group"]()
# Here: reductions that read the activation fragments and touch their own
# shared region. Placed before the barrier above, ptxas schedules them back
# on top of the stores and nothing is gained.
_column_sums(...)
```

## Rationale

One column-sum reduction is 64 shared stores, two barriers and 16 vector
loads; run
ahead of the store it delays it by all of that, run behind it it costs nothing.
On a block-scaled MoE grouped GEMM this was 4.8% on the binding row. The
barrier
matters: an earlier attempt that placed the same work after the shared writes
but before the barrier produced no change, because ptxas is free to hoist across
plain stores and is not free to hoist across `bar.sync`.

## Boundary

Only for work with no dependence on the stored data -- reductions over the
activation fragments the packing merely copied, in a separate shared region.
Anything the store reads must stay ahead of it.

## Verification

Static instruction counts will not change at all, so check the stall samples on
the post-store `fence.proxy.async.shared` instead, and keep a shape in the set
that does not run the moved work as a control.
