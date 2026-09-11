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

Scope validity does not prove the compiler honors the hint. On CUDA 13.2/B200,
a three-warpgroup probe with only a one-operand launch bound compiled to eight
registers, warning C7508 and no USETMAXREG. Giving the same body an explicit
minimum-blocks-per-SM bound of one produced 168 registers and USETMAXREG.
Do not add an occupancy constraint merely to satisfy a source-level register
model: it activates a resource contract the original compiled kernel did not
have, and the same requests can then genuinely overdraw the CTA register pool.

Reentering a role that allocates registers also requires explicit
synchronization of every participating warpgroup between successive
`setmaxnreg` instructions,
even when the requested count is unchanged. A barrier inside an optional work
loop does not cover a CTA that skips that loop. Put the convergence outside
the loop, within the owning compute or auxiliary cohort, before its next
register transition.

Do not assume that deleting the repeated instruction preserves compiler
allocation. In a 12-warp persistent kernel, removing the second 208-register
compute and 88-register auxiliary transitions grew the NVRTC stack from 16 to
64 bytes. Retaining those transitions and adding explicit 256-thread compute
and 128-thread auxiliary barriers preserved the 16-byte stack and register
count. The repair passed complete synchronization and race checks on a CTA
that skipped stream work, a GPU/NumSim/reference gradient comparison, and all
17 production GPU correctness configurations. Production GQA compilations
added 12 bytes of static spill stores while spill loads stayed unchanged;
that resource result alone does not establish unchanged execution time.

## Verification

Verify in the realized TIR that there is one producer `setmaxnreg` and that
every warp of its warpgroup reaches it before any sub-role guard.
For repeated role entries, also check the path that skips prior work and
confirm that all warps in the cohort converge before the next register
transition.
