# Keep setmaxnreg on the complete warpgroup scope

**Symptoms:** `kernel_deadlock`, `partial_setmaxnreg_participation`, `conditional_mma_role`, `register_budget_mismatch`

## Symptom

A warp-specialized kernel hangs, or realizes a register budget the roles did not
ask for, after the functional roles under one producer warpgroup were split
apart. One role's guard is narrower than a warpgroup or conditional on a
CTA-uniform coordinate.

## What to change

`setmaxnreg.sync` is a warpgroup collective. Keep one instruction with one
operand in the enclosing four-warp scope, then dispatch the loader, scheduler,
MMA, and idle roles beneath it.

```python
# before: the register instruction rides a narrower, conditional role, so only
# the MMA warps execute it.
with mma_role:  # entered only when cbx == 0
    K.ptx.setmaxnreg.dec.sync.aligned.u32(56)
    ...

# after: one instruction where every warp of the warpgroup reaches it, with the
# functional roles beneath.
with producer_warpgroup:
    K.ptx.setmaxnreg.dec.sync.aligned.u32(56)
    if cbx == 0:
        ...  # MMA role
    ...      # loader, scheduler, idle roles
```

The functional role predicate may still include the CTA-uniform condition; the
register scope may not.

## Rationale

Functional roles can be narrower than one warpgroup or conditional on a
CTA-uniform coordinate, while the register instruction cannot: partial
participation leaves the collective unsatisfied.

This appeared while splitting three 2-CTA GEMM producers into their real roles.
Moving the 56-register instruction into the conditional MMA role made a
no-overlap FP16 specialization hang, while the overlap shape happened to
complete. Restoring one common producer instruction passed all ten FP16/BF16
GEMM configurations and all eight TP1 reduce-scatter configurations. A
three-round large-shape A/B then measured 9405.779 us against 9406.143 us for
the prior common-scope implementation, with no performance separation.

## Boundary

This governs where the register instruction sits, not whether the roles are
worth splitting. Splitting the functional roles is a separate change with its
own evidence.

## Verification

Verify in the realized TIR that there is one producer `setmaxnreg` and that
every warp of its warpgroup reaches it before any sub-role guard.
