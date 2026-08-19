# Preserve cache policy across launch-local handoffs

**Symptoms:** `cold_cache_regression`, `inter_kernel_handoff`, `global_store_policy`, `dispatch_specific_deficit`

## Symptom

A deficit localized to one dispatch in a multi-launch chain whose producer
stores carry a cache hint the source SASS does not have.

## What to change

When the source SASS uses the default `STG.E.128` form, preserve that default
policy instead of adding a cache hint on general streaming-store intuition.

```python
# before: a cache hint added on streaming-store intuition, where the workspace
# has an immediate cross-launch consumer.
T.evaluate(
    T.ptx["st.global.L1::no_allocate.v4.b32"](
        buffer.ptr_to([index]), w[0], w[1], w[2], w[3]
    )
)

# after: the default policy the source SASS uses.
T.evaluate(
    T.ptx.st.global_.v4.b32(buffer.ptr_to([index]), w[0], w[1], w[2], w[3])
)
```

A modifier containing `::` must use the bracket form; the default form can use
attribute access, where CUDA reserved words take a trailing underscore.

## Rationale

A global store into a workspace that the next launch consumes is not a terminal
streaming store. Its cache operator is part of the producer-consumer schedule:
adding `L1::no_allocate` can change the next launch's cache behavior even when
the address stream and vector width match. In one four-launch recurrent chain,
the revert reduced the weakest 128-row shape from 86.56 to 82.93 us and moved
source/port from 0.962 to 1.005. A 64-row guard stayed effectively flat at
119.65 versus 119.85 us, which localized the gain to the affected dispatch. A
later clean seven-shape matrix retained the result at 82.68 us for the port
versus 82.95 us for the source, or 1.003 source/port.

## Boundary

This does not justify default caching for final outputs or write-only
workspaces; the boundary is an immediate cross-launch consumer.

## Verification

Verify the source and target SASS cache operators and vector width, then time
the producer and consumer in the same cold-cache benchmark scope. Re-run the
affected dispatch, a different-dispatch guard, and the full matrix before
keeping the change.
