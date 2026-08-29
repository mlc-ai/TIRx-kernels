# Keep persistent grid-stride iteration as a while loop

**Symptoms:** `persistent_grid_regression`, `repeated_divide_fixup`, `mufu_rcp`, `role_code_duplication`

## Symptom

A persistent kernel emits reciprocal-and-fixup division chains that the
reference does not have, once per independently traced role.

## What to change

Keep the grid-stride form: materialize the work index once per role and
increment it by the grid size.

```python
work = K.alloc_local([1], "int32")
K.assign(work[0], K.cta_id())
with K.While(work[0] < total_work):
    ...
    K.assign(work[0], work[0] + grid_x)
```

Do not replace it with a precomputed per-CTA trip count such as
`ceildiv(max(total_work - cta_id, 0), grid_size)` ahead of a role-local serial
loop. A shared source expression is not a shared lowered value when each role
owns a separate device region.

## Rationale

One six-role persistent kernel computed that trip count before each role-local
serial loop, and its largest specialization emitted six absent-in-reference
`I2F.RP -> MUFU.RCP -> F2I.FTZ.U32.TRUNC.NTZ` chains. Restoring the direct
grid-stride while loop removed all six reciprocal chains, reduced static SASS
from 5992 to 5656 instructions against the reference's 5688, and retained 168
registers with no stack or spills. Three correctness shapes passed, and clean
45-round A/B ratios were 0.9852x, 1.0047x, and 1.0038x under a 1.01 gate.

## Boundary

Do not apply this mechanically. It does not follow when the trip count is
compile-time, when it is consumed once, or when the backend already emits an
integer divide with no duplicated fixup chain; inspect the role-local SASS
first.

## Verification

Count the reciprocal-and-fixup chains per role in SASS and compare static
instructions and registers against the reference before measuring.
