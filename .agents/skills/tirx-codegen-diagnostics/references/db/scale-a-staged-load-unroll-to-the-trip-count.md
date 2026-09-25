# Scale a staged-load unroll to the loop's trip count

**Symptoms:** `long_scoreboard`, `exposed_load_latency`, `insufficient_memory_parallelism`, `slow_small_shape`

## Symptom

Fewer instructions than the reference and more time -- the signature of exposed
load latency rather than a code-size problem. One chunk-staging loop transcribed
with the reference's factor executed 11% fewer instructions yet ran 6% slower,
with the whole gap in `long_scoreboard`: 7.12 warp-cycles per issued instruction
against 3.74.

## What to change

Treat a reference's `#pragma unroll N` as a statement about how many independent
global loads it wants in flight, not as a portable constant, and derive the
factor from the compile-time per-thread trip count.

```python
# The factor is a property of this loop's trip count, not a global constant.
CHUNK_ITERS = T.meta_var(chunk // (threads * vec))
for i in T.serial(0, CHUNK_ITERS, unroll=4 if CHUNK_ITERS >= 28 else 2):
    ...
```

Raising only the staging loop to `unroll=4` moved the stall figure to 2.89 and
the shape from 41.2 us to 33.5 us.

## Rationale

This toolchain can extract less memory parallelism from the same unroll factor
than nvcc does, leaving the load latency exposed while the instruction count
actually drops.

The factor must scale with the per-thread trip count. At 28 iterations per
thread the deeper unroll took four shapes from 0.91-0.98x to 1.18-1.21x; at 6.8
iterations the remainder dominated and the same change took one shape from 1.11x
to 0.95x.

A second kernel reproduced the scaling law with a different threshold, so derive
it per kernel rather than reusing a number. A clustered radix select swept the
factor against its per-thread trip count and found the crossover at 8, not 28:

| trips/thread | unroll 4 | unroll 6 | unroll 8 |
| ---: | --- | --- | --- |
| 4 | **1.083, 1.032** | - | 0.896, 0.955 |
| 8 | 0.977 | 0.959 | **0.995** |
| 16 | 0.987, 1.006 | 0.924, 0.969 | **1.060, 1.011** |
| 32 | 0.963 | 0.970 | **1.009**, and 12 fell back to 0.920 |

`unroll = 8 if trips >= 8 else min(4, trips)` took that kernel's whole matrix
from three failing shapes to zero.

A rolled loop in the reference's source text is not a statement that the loop
should be rolled. One port transcribed a preprocess loop in the reference's own
rolled form for faithfulness, leaving a single body and one load in flight; the
reference's compiler had unrolled the same loop. The kernel is purely
bandwidth-bound, and the rolled form ran at 50.1% of memory throughput against
the reference's 74.9%, 141.3 us against 94.8. Restoring the unroll took it to
83.8% and 84.5 us -- faster than the reference. Read the factor off the
reference's generated code, not its source.

The cost lands unevenly when the loop's grid scales with a problem dimension. That
preprocess kernel launches one block per head, so the 64-head program paid the
deficit twice over relative to the 32-head one, and the shortfall showed up as a
one-program failure pattern that looked like a main-kernel problem.

## Boundary

Do not assume monotonicity: unroll 6 and 8 regressed back to 5.75 and 6.26
warp-cycles on register pressure. Sweep instead. Confirm the two trip-count
regimes are actually separated in the config domain before picking a threshold,
and do not apply the factor globally.

The non-monotonicity is not only at the top of the range. In the sweep above,
unroll 6 was worse than BOTH 4 and 8 on every shape tested, at every trip count.
A three-point sweep that brackets the current factor can therefore point the
wrong way; sweep the endpoints you would actually ship.

The lever also does not transfer between loops in the same kernel. Applying the
same staging to a loop whose bound is a runtime value, rather than an exact
multiple of the stride, regressed three shapes (0.966 -> 0.946, 0.984 -> 0.944,
0.988 -> 0.863): the per-element bounds re-check the runtime bound forces costs
more than the added parallelism returns.

## Verification

Diagnose with paired stall breakdowns, not instruction counts -- fewer
instructions and more time is the signature. Profiling the same kernel on a fast
and a slow shape isolates whether the cost is in staging or elsewhere.
