# Guard configured-shape arithmetic with a runtime fallback

**Symptoms:** `runtime_division`, `excess_integer_math`, `instruction_count_gap`, `slow_small_shape`

## Symptom

A kernel factory specializes for a configured power-of-two shape, but the
generated entry still decomposes each flat row with a runtime quotient and
remainder because the ABI also passes that dimension. The ordinary launch uses
the configured value, while the dynamic operations remain on every thread's
short critical path.

## What to change

Guard constant shape arithmetic with a uniform runtime equality check and keep
the original dynamic expressions as a real fallback.

```python
# before:
outer = flat // runtime_extent
inner = flat % runtime_extent

# after:
with K.If(runtime_extent == K.int64(configured_extent)):
    with K.Then():
        K.assign(outer, flat // K.int64(configured_extent))
        K.assign(inner, flat % K.int64(configured_extent))
    with K.Else():
        K.assign(outer, flat // runtime_extent)
        K.assign(inner, flat % runtime_extent)
```

## Rationale

On an H=128 normalization specialization with configured N=32, this reduced
dynamic warp instructions from 95,744 to 77,824 and predicated-on thread
instructions from 2,917,344 to 2,424,832. Requested registers fell from 28 to
26, matching the reference; long- and short-scoreboard samples fell from 32/9
to 27/7, versus 28/9 for the reference. Memory traffic did not change and no
local or spill traffic appeared.

Three independent counterbalanced 30-pair measurements reported 1.0126x,
1.0186x, and 1.0192x over the source. All six order subsets independently
exceeded parity, and the complete affected correctness matrix passed.

## Boundary

Use this only when the factory's configured extent is an actual specialization
contract for normal launches. Do not discard the runtime value: callers may
reuse the compiled entry with a different extent, so preserve and numerically
test the dynamic fallback. Scope the fast path to the measured target and shape
family; the added branch is not automatically profitable for longer kernels or
non-power-of-two configured extents.

## Verification

Compare dynamic integer and total instruction counts, requested and allocated
registers, scoreboard stalls, memory traffic, and spills. Run the entire
affected configured-shape matrix, then deliberately launch one compiled entry
with a different valid runtime extent to exercise the fallback. Use
counterbalanced long-run timing and require both execution-order subsets to
agree.
