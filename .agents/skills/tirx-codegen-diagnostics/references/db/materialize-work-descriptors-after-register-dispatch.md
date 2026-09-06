# Materialize work descriptors after register dispatch

**Symptoms:** `pre_dispatch_spill`, `local_memory_traffic`, `long_scoreboard`, `register_budget_mismatch`

## Symptom

A warp-specialized kernel loads persistent-work coordinates before dynamic
register redistribution. The descriptor scalars then remain live across role
dispatch, even though each worker role consumes them independently and a
scheduler-only role does not consume them at all. Exact SASS places those
scalars in local memory around `setmaxnreg` despite adequate role-local budgets.

## What to change

Declare shared control state before dispatch, but materialize each work
descriptor only after entering a consuming role. Advance the work queue with an
option that suppresses descriptor loading in roles that only maintain the
scheduler protocol.

```python
# before: every role carries the descriptor across register redistribution.
load_work_descriptor(work_index)
roles = K.specialize()
with worker_role:
    consume_work()

# after: only consuming roles materialize role-local state.
roles = K.specialize()
with worker_role:
    load_work_descriptor(work_index)
    consume_work()
with scheduler_role:
    advance_work(load_descriptor=False)
```

Apply the same ownership test to secondary tile metadata. A role that only
waits, cancels, arrives, or advances phases should not load coordinates used by
the data path.

## Rationale

In one 16-warp persistent pipeline, exact default-NVRTC SASS showed two work
coordinates loaded before register dispatch and spilled across it. Moving the
loads into the worker roles eliminated all local spilling requests and reduced
the affected benchmark from 26.173 to 26.096 us, a 0.29% improvement, while
tight source and independent-oracle checks passed across aligned and irregular
layouts.

Omitting unused descriptor and tile-metadata loads from a scheduler-only role
was semantically valid, but changed the same path only from 26.096 to 26.092 us.
Treat that latter result as cleanup rather than proof of a latency mechanism;
the reusable signal is the measured pre-dispatch spill removal.

## Boundary

The dispatch decision and queue protocol must not depend on the deferred
descriptor. Keep queue-valid state, phases, cancellation responses, and remote
arrivals in their original common scope. Duplicate a descriptor load across
consumer roles only when the shorter lifetime is cheaper than sharing it.

Do not generalize this into sinking every load. In the same pipeline, moving a
four-word mask payload from before a score wait to its consuming branch
regressed 26.17 to 27.08 us because the earlier position hid global latency.
Role-state loads that spill across register redistribution and independent
payload prefetches have opposite scheduling tradeoffs.

## Verification

Compile through the production path and compare the same role regions in SASS.
Require the pre-dispatch LDL/STL sites and dynamic local traffic to disappear,
then benchmark the affected path plus roles that retain descriptor loading.
Run correctness on aligned and irregular scheduler cases, and repeat the
role-register sweep after changing the live ranges.
