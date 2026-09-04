# Sweep ptxas register level by static shape

**Symptoms:** `schedule_regression`, `register_budget_mismatch`, `instruction_parity_with_deficit`, `unstable_benchmark_ratio`

## Symptom

A spill-free port matches the source launch and memory traffic and may even
execute fewer instructions, but benchmark timing remains slightly behind or
unstable. A static register cap or launch bound either regresses or changes
dynamic work while the natural register allocation is already legal.

## What to change

Sweep ptxas `--register-usage-level` independently of hard register caps. Scope
the selected level to the compile-time dimensions and target that were measured,
and retain the established default for siblings.

```python
def _select_ptxas_reg_level(seq_len, num_heads, num_v_heads, tile_v):
    if prepare_cuda_arch() == "sm_110a" and (
        seq_len,
        num_heads,
        num_v_heads,
        tile_v,
    ) == (4, 8, 16, 16):
        return "5"
    return "10"
```

## Rationale

Register-usage level can select a different schedule without changing realized
allocation, occupancy, spilling, or memory traffic. In one 128-thread Thor GDN
decode specialization, both level 10 and level 5 allocated 56 registers, were
limited to nine CTAs by registers, and executed identical global/shared traffic
with zero local traffic. Level 5 nevertheless moved the required 30-round ratio
from `0.9988x` to `1.0060x`; two independent 30-pair counterbalanced diagnostics
measured `1.0098x` and `1.0065x`.

The sweep was non-monotonic. Short screens for levels 0, 2, 4, 5, 6, 7, 8, and
9 ranged from `0.9870x` to `1.0453x`, but level 7 fell to `1.0064x` over 30
pairs, effectively tied with level 5. A hard 48-register cap appeared to win at
`1.1016x` in a short screen and then collapsed to `0.9096x` over 30 pairs. A
ten-block launch bound reached the same 48-register allocation but increased
dynamic warp instructions and did not reproduce the cap schedule.

## Boundary

Do not infer a winning compiler level from a single NCU duration, instruction
count, requested register count, or short timing screen. The retained GDN child
executed slightly more dynamic warp instructions than level 10, and its profiled
single-launch duration did not predict the long-run win. Compiler levels and
static caps are different controls even when they happen to realize the same
allocation.

Keep the selector as narrow as the evidence. Dimensions used in the selector
must already be compile-time specialization keys; a runtime batch size does not
justify a new branch. Restore the family default on every unmeasured target and
shape so one marginal schedule does not leak into unrelated compilation.

## Verification

Run a neighboring-level screen, then repeat promising candidates with the full
required bench-suite rounds. For sub-percent effects, use counterbalanced order
as a diagnostic and inspect both order subsets, but use the repository
bench-suite artifact for the final gate. Record realized registers, occupancy,
local traffic, complete memory traffic, and dynamic instructions for the final
child. Compile every affected specialization, run its full correctness matrix,
and compile an unchanged-target guard before retaining the selector.
