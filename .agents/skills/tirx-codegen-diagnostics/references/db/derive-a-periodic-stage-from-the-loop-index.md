# Derive a periodic stage from the loop index

**Symptoms:** `manual_stage_arithmetic`, `redundant_modulo`, `long_scoreboard`, `dispatch_specific_deficit`

## Symptom

A small epilogue ring carries a mutable stage scalar through every persistent
work item. The stage feeds shared-memory addresses, its update remains on the
loop-back dependency chain, and the number of subtiles per work item always
returns the ring to its initial stage.

## What to change

When the stage starts at zero and the subtile count is a multiple of the ring
depth, derive the stage from the subtile index and delete the loop-carried
cursor. Use a mask for a power-of-two depth.

```python
# before: the next work item depends on the previous cursor update.
stage = K.local_scalar("int32", init=0)
with K.While(subtile < SUBTILES):
    _store(fragment, stage)
    K.assign(stage, K.bitwise_and(stage + 1, K.int32(STAGES - 1)))
    K.assign(subtile, subtile + 1)

# after: no mutable stage survives the iteration.
with K.While(subtile < SUBTILES):
    stage_index = K.bitwise_and(subtile, K.int32(STAGES - 1))
    _store(fragment, stage_index)
    K.assign(subtile, subtile + 1)
```

## Rationale

On a high-persistence two-stage FP8 epilogue, the generated CUDA lost the local
stage variable and its update, and every store address used `subtile & 1`
directly. The active E5M2 path moved from 0.9889x to 0.9919x. After applying the
same proven recurrence to the sibling FP8 conversion path, six active,
boundary, and control rows passed on two GPUs, and the retained implementation
passed the complete correctness and 66-row performance matrices.

The same transformation was not automatically useful for a four-stage FP32
path: both long-batch guards remained near 0.988x. Removing a recurrence is a
dependency-graph lever, not a guarantee that the resulting schedule improves.

## Boundary

Prove all three invariants: the stage starts at a known value, `SUBTILES` is a
multiple of `STAGES`, and no barrier phase or cross-work protocol observes the
mutable cursor. Otherwise deriving from the inner index changes the ring phase.

Scope the rewrite to the persistence and dtype/fragment regimes where the
cursor is actually on the critical path. Keep inactive boundary
specializations on the original trace.

## Verification

Check generated CUDA/PTX for removal of the local cursor and loop-back update,
then inspect SASS addresses for the direct index. Test the first wrap, the next
work item, the specialization boundary, and sibling dtypes before running the
complete matrix.
