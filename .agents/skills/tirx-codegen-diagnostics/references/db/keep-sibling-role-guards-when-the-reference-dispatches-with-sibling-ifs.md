# Keep sibling role guards when the reference dispatches with sibling ifs

**Symptoms:** `slow_small_shape`, `dispatch_specific_deficit`, `instruction_parity_with_deficit`, `schedule_regression`

## Symptom

A warp-specialized port whose static SASS already matches the reference (same
registers, no spills, identical MMA/TMA/TMEM/barrier and float-pipe counts) sits
slightly below the reference on its smallest, latency-bound shapes, and the
role dispatch was folded into an if/else-if chain while the reference selects
its roles with independent sibling `if` blocks.

## What to change

Let the role guards stay sibling `if`s, matching the reference, when the kernel
has no register pressure for the chain to relieve.

```python
# before: chained dispatch, the K.specialize default.
roles = K.specialize()

# after: sibling guards, the reference's structure.
roles = K.specialize(chain_dispatch=False)
```

## Rationale

The chain exists to tell ptxas that role bodies are mutually exclusive, which
matters at the register cliff. Without that pressure it only changes the
dispatch control flow, and on one 16-warp block-sparse attention kernel that was
a pure cost: with 128 registers and zero spill bytes in both forms, the chained
build measured 19.25 us against 18.82 us for the sibling build on the sparsest
shape (4 KV blocks per 128-row tile, 128 CTAs) and 53.10 us against 52.45 us on
the next shape (256 CTAs), i.e. 2.3% and 1.2% slower, moving the reference/port
ratio from 0.996 to 0.974 and from 1.007 to 0.995. Large streaming shapes were
not measured because the small shapes already decided the direction.

## Boundary

Applies when the reference's dispatch is sibling `if`s and the port assembles
without spills at its register targets. When a role sits at the 255-register
cliff or ptxas reports spills, the chain is still the documented remedy
(`K.specialize` docstring) and its cost here is the price of assembling.

## Verification

Build both forms, confirm identical `ptxas -v` register and spill counts, and
compare the smallest required shapes in the bench suite before choosing; the
difference is invisible in static opcode histograms.
