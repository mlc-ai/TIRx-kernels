# A PDL wait without a trigger releases at completion

**Symptoms:** `pdl_chain_suspicion`, `missing_trigger`, `launch_config_drift`

## Symptom

A PDL chain that looks broken because the dependent kernel waits but no kernel
triggers.

## What to change

Nothing, unless the reference has a trigger. `griddepcontrol.wait` in the
dependent kernel releases when the primary kernel completes; an explicit
`launch_dependents` is required only where the reference has one.

```python
# In the dependent kernel: releases at primary-kernel completion on its own.
T.evaluate(T.ptx.griddepcontrol.wait())

# Only where the reference triggers, and at the reference's exact position:
T.evaluate(T.ptx.griddepcontrol.launch_dependents())
```

## Rationale

One dispatch kernel triggers right after its data-arrival barrier, but the
sibling combine main kernel contains no trigger at all -- the reduce epilogue's
wait releases at kernel-1 completion, overlapping only its prologue. Adding a
trigger there would start the epilogue early on unguarded data.

## Boundary

A missing trigger is not a bug to fix, and an added one is a semantic
reordering.

## Verification

Check the reference for the trigger's existence and exact position before wiring
the chain.
