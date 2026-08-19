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

## Boundary

Do not assume monotonicity: unroll 6 and 8 regressed back to 5.75 and 6.26
warp-cycles on register pressure. Sweep instead. Confirm the two trip-count
regimes are actually separated in the config domain before picking a threshold,
and do not apply the factor globally.

## Verification

Diagnose with paired stall breakdowns, not instruction counts -- fewer
instructions and more time is the signature. Profiling the same kernel on a fast
and a slow shape isolates whether the cost is in staging or elsewhere.
