# Specialize static register caps by execution path

**Symptoms:** `register_budget_mismatch`, `schedule_regression`, `dispatch_specific_deficit`, `low_occupancy`

## Symptom

A spill-free kernel has a persistent deficit in one compile-time execution path,
or neighboring shape families realize different register allocations and
schedules despite sharing a block size.

## What to change

Place `tirx.max_registers` immediately after `T.device_entry()` and select it by
the compile-time path that changes the live range or scheduling problem. Sweep
neighboring integer caps for each affected path instead of assuming that the
response is monotonic or that one cap fits the whole kernel family.

```python
T.device_entry()
if USE_DEPENDENCY_PROTOCOL:
    T.attr({"tirx.max_registers": 64})
elif LARGE_FRAGMENT:
    T.attr({"tirx.max_registers": 96})
else:
    T.attr({"tirx.max_registers": 93})
```

This is a static whole-kernel ceiling. It is not a replacement for
`setmaxnreg.sync.aligned`, which transfers register budget between warpgroup
roles while a kernel is running.

## Rationale

The static ceiling participates in ptxas scheduling and allocation. It can
therefore change recomputation, instruction placement, and occupancy even when
both candidates report zero stack and local-memory use.

One measured 128-thread dependency-protocol specialization was non-monotonic at
single-register granularity: caps 61, 63, 64, 65, and 68 measured 0.984x,
0.983x, 0.998x, 0.978x, and 0.985x respectively against the same reference.
The selected cap of 64 subsequently cleared a five-shape guard matrix at
0.993-1.026x, with the affected path at 1.001x. The same source family required
caps of 93 and 96 for two non-protocol 128-thread shape regimes, while its
256-thread regime retained a launch bound instead of a static cap.

The wider 128-thread regime had a separate hard boundary: every cap below 92
produced 8-40 bytes of stack, while cap 96 kept both guarded variants spill-free
and measured 0.992x. Cap 104 also passed but with less margin, so the selector
retained 96 rather than applying the dependency-protocol cap globally.

## Boundary

`tirx.max_registers` and `tirx.launch_bounds_*` are alternative static codegen
contracts and must not be declared together. Keep block-size, fragment-width,
and dependency-protocol families separate when their live ranges differ. A cap
that is too low can introduce spills or recomputation; a cap that is too high
can reduce occupancy or produce a worse schedule. Re-sweep after any change to
fragment lifetime, instruction ordering, or launch protocol.

Do not copy the reference's realized register count into the cap. One
source-like cap of 74 moved a wide-fragment path to 4.7-4.9 microseconds, and a
cap matching the reference's 56-register dependency path improved the target
only slightly without clearing the gate. The two instruction streams can need
different budgets to realize comparable schedules.

## Verification

Confirm that CUDA and PTX contain the requested `__maxnreg__` and `.maxnreg`,
then read the realized register allocation, stack size, local-memory traffic,
and scheduled SASS rather than trusting the declaration. Run correctness plus
the affected performance shapes and guard shapes on every retained cap.
