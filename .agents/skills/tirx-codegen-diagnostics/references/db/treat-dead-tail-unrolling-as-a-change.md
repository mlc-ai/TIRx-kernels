# Treat a dead tail's unrolling as a change, not a cleanup

**Symptoms:** `instruction_count_bloat`, `unreachable_code_expansion`, `dead_tail_loop`

## Symptom

A loop with a runtime bound that is empty in every dispatched regime is still
unrolled. One orphan tail loop expanded roughly 33x, emitting 70 narrow global
stores where the reference emitted 18.

## What to change

Disable unrolling on the dead tail -- then put the change through the complete
matrix like any other change, not as a free cleanup, and do not land it
alongside a second change.

```python
# The bound is a runtime value that is zero in every dispatched regime,
# so the unrolled copies are unreachable code.
for t_offset in T.serial(0, tail_bound, unroll=False):
    ...

# Equivalent explicit form.
for t_offset in T.serial(0, tail_bound, annotations={"disable_unroll": True}):
    ...
```

## Rationale

Disabling unrolling removed 112 static instructions from the phase. But code
that never executes costs issue bandwidth and instruction cache, not cycles, so
the static win is large while the runtime effect is small and can be negative.

## Boundary

The same change measured a small gain on the contaminated run that motivated it,
then measured 0.924-0.979 on one shape across five later campaigns, reproducibly
the worst variant there; that shape cleared the gate only after the unroll
change was reverted. The change had also been introduced together with a second
one, and attributing their combined delta to the other produced a
specialization axis that had to be withdrawn.

## Verification

Identify the expansion in SASS, then run the complete shape matrix with the
unroll change isolated from any other change.
