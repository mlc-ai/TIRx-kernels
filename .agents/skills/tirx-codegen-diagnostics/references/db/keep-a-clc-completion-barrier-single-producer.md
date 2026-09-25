# Keep a CLC completion barrier single-producer

**Symptoms:** `unspecified_launch_failure`, `clc_duplicate_producer`, `syncs_arrive_fault`, `multicast_barrier_overarrival`

## Symptom

A launch failure that surfaces at the barrier instruction rather than at a
memory access. One duplicated-producer port produced 152 Compute Sanitizer
unknown errors, first on the two producer-warp leaders at
`SYNCS.ARRIVE.TRANS64`.

## What to change

A multicast cluster-launch-control response barrier must have exactly one
elected producer warp per CTA-group protocol instance. Guard both the
`try_cancel` loop and its `arrive.expect_tx` with the same producer-warp role.

```python
# Electing one lane independently in two warps still creates two producers.
if warp_id == CLC_PRODUCER_WARP:
    if lane_id == 0:
        ...  # try_cancel and its arrive.expect_tx, together
```

## Rationale

Two producers over-arrive the shared completion barrier and its expected byte
count. Restoring the single-producer role passed all three registered runtime
shapes, the affected instrumentation metadata checks, and a forced SM100
compile.

## Boundary

Do not generalize the guard to unrelated CLC consumers: multiple warps may wait
on or consume the response when the protocol requires it. The invariant is that
only the designated producer issues the asynchronous request and accounts its
completion bytes to that barrier.

## Verification

Run Compute Sanitizer and read which instruction and which threads report first;
an arrive-instruction fault on more than one warp leader is the
duplicate-producer signature.
