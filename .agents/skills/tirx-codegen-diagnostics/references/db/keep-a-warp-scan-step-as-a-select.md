# Keep a warp-scan step as a select

**Symptoms:** `excess_control_instructions`, `branch_in_hot_loop`, `schedule_regression`, `instruction_count_gap`

## Symptom

A warp scan carrying one more dependent instruction per step than the reference:
`shfl` then add then select, against the reference's `shfl` then predicated add.
cub takes the condition from the shuffle's own destination predicate
(`shfl.sync.up.b32 r0|p, ...` followed by `@p add.u32`), where the port computes
`lane >= 1 << step` separately.

## What to change

Nothing. Keep the select form.

```python
# keep this: lowers to setp + selp, no control flow.
peer: T.uint32 = shfl_up_u32(incl[0], T.shift_left(T.int32(1), step))
incl[0] = T.Select(lane >= T.shift_left(T.int32(1), step), incl[0] + peer, incl[0])
```

The destination-predicate form is reachable -- the shuffle entry accepts a
writable `uint32` as its predicate output -- but the accumulate that consumes it
has to be written as a guarded statement, and that does not become a predicated
add.

```python
# rejected: the guard is emitted as control flow, not folded into the add.
shfl_up_u32_p(peer, valid, incl[0], T.shift_left(T.int32(1), step))
if valid[0] != T.uint32(0):
    incl[0] = incl[0] + peer[0]
```

## Rationale

nvcc did not if-convert the guarded accumulate: the entry kept all 14 `selp` and
gained no predicated add, while total instructions fell by 8 because the five
lane comparisons disappeared. Those comparisons are loop-invariant and hoisted,
so removing them buys nothing on the critical path, and the shape measured
**worse** -- 0.948 to 0.934, and a second shape 0.977 to 0.970.

## Boundary

This is about consuming a destination predicate through a guarded statement, not
about predication generally. Where a condition already feeds a value expression,
`T.Select` is the right lowering and produces the `selp` the reference has.

## Verification

Count predicated instructions in the PTX before crediting an if-conversion: the
guard either became `@%p` on the operation or it became a branch, and only the
first is the change you intended.
