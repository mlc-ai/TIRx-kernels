# Hoist invariant work out of a runtime-conditional region

**Symptoms:** `excess_address_math`, `instruction_count_bloat`, `dispatch_specific_deficit`, `branch_in_hot_loop`

## Symptom

One specialization trails the reference while its siblings pass, and the paired
per-opcode diff shows a large dynamic `IMAD` surplus (1.5x or more) at matched
protocol and matched tensor-op counts. The surplus sits in a block guarded by a
runtime predicate inside the persistent loop -- a snapshot, a cadence-gated
publish -- that in the trailing specialization fires every iteration.

## What to change

Loop-invariant values inside a runtime-guarded block do not get hoisted: the
compiler will not move them across the branch. Materialize them once in the
role preamble into a small local array indexed only by compile-time constants,
and index that array inside the guarded block.

```python
# before: the swizzled store offsets are iteration-invariant, but the
# guard stops the compiler from hoisting them.
with K.If(do_snapshot), K.Then():
    for sub in range(SUBTILES):
        for group in range(GROUPS):
            column = sub * SUB_COLS + group * 8
            element = (
                (column // ATOM) * ATOM_STRIDE
                + row * ROW_ELEMS
                + _swizzle_xor_128b(row, column % ATOM)
            )
            K.ptx.st.shared.v4.b32(arena.ptr_to([BASE + element * 2]), *words)

# after: materialized once per role; the guarded block only indexes.
snapshot_off = K.alloc_local((SUBTILES * GROUPS,), "int32")
for sub in range(SUBTILES):
    for group in range(GROUPS):
        column = sub * SUB_COLS + group * 8
        element = (
            (column // ATOM) * ATOM_STRIDE
            + row * ROW_ELEMS
            + _swizzle_xor_128b(row, column % ATOM)
        )
        K.assign(snapshot_off[sub * GROUPS + group], BASE + element * 2)
...
with K.If(do_snapshot), K.Then():
    for sub in range(SUBTILES):
        for group in range(GROUPS):
            K.ptx.st.shared.v4.b32(
                arena.ptr_to([snapshot_off[sub * GROUPS + group]]), *words
            )
```

Keep every index into the materialized array a compile-time constant: a
dynamically indexed local array lowers to local memory and inverts the win.

## Rationale

Loop-invariant code motion and CSE stop at runtime branches, so a guarded block
that fires every iteration re-derives its address chains each time. On the
specialization where the guard was always true, the paired profile showed
+1.15M dynamic `IMAD` (1.7x the reference) concentrated in the conditional
region; materializing its sixteen invariant byte offsets moved the focused
shape from 0.968x to 0.977x with the guard shapes unchanged, and the change
survived the complete correctness matrix and the final complete
performance-matrix winner.

## Boundary

Unconditional loop bodies gain nothing: the backend already hoists invariant
addresses executed on every iteration, and the identical hoist applied to
always-executed `ldmatrix`/`stmatrix` offsets measured no change on two shapes
(0.950x/0.971x against 0.953x/0.973x, within run noise). Each materialized
offset also occupies a register for the role's lifetime; on a warpgroup near
its budget, hoist only what the guarded block consumes.

## Verification

Diff `IMAD`/`LEA` dynamic counts in the affected region against the reference
before and after, confirm the guarded block in generated CUDA reads the
materialized locals rather than rebuilding the chain, then measure the affected
specialization and its guard shapes.
