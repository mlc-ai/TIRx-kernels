# Avoid trace-time name capture across runtime branches

**Symptoms:** `poisoned_outputs`, `illegal_memory_access`, `runtime_branch`

## Symptom

Adding a cold device branch changes stores on inputs that never execute it.
An output retains its poison value, or a later store acquires a wrong coordinate
and faults. The numerical values computed before that store still agree with
the reference.

## What to change

Give temporary Python names inside a traced runtime branch their own scope or
distinct names. A DSL conditional controls the emitted device program, not
whether Python executes its body while constructing that program.

```python
# before: tracing the cold branch rebinds the later store's coordinate.
output_row = quad * 8 + (lane >> 2)
with txl.If(needs_correction), txl.Then():
    output_row = row_base + lane
    correct(output_row)
store(output_row, value)

# after: the correction uses its own coordinate.
output_row = quad * 8 + (lane >> 2)
with txl.If(needs_correction), txl.Then():
    correction_row = row_base + lane
    correct(correction_row)
store(output_row, value)
```

## Rationale

A sparse correction experiment reused a short token-row name that the later
beta reduction already used. Even ordinary inputs that skipped correction
left beta-gradient outputs at their poison value; a mixed case subsequently
reported an illegal memory access. Renaming the correction coordinate restored
repeatable outputs and passed 16 ordinary, sparse, dense, and mixed gate checks
across both kernel families under the unchanged tolerances.

## Boundary

This is a tracing ownership bug, not a GPU memory-order fix. Once coordinates
are correct, cooperative writers still need their own synchronization before
a subset of threads signals that a shared buffer can be reused.

## Verification

Inspect uses of the rebound name after the new branch. Test both branch
outcomes, including the supposedly unaffected path, with poisoned outputs and
repeated launches; checking only computed values can miss stores that never
reached their destination.
