# Measure elided warp synchronization as a scheduling constraint

**Symptoms:** `warp_sync_schedule_shift`, `bar_warp_elided`, `long_scoreboard`, `exposed_load_latency`, `small_shape_regression`

## Symptom

A converged full-mask `bar.warp.sync` remains in PTX while ptxas emits no
`BAR.WARP` instruction, and the change looks free. It is not: the ordering
constraint can still reschedule surrounding instructions differently for each
specialization.

## What to change

When paired SASS shows independent loads hoisted before asynchronous-copy issue
or sunk behind its wait, use full-mask warp synchronization as a scheduling
boundary. Bracket only the prefix of independent loads that should remain in the
wait window; the prefix length and both boundary positions are compile-time
specialization choices.

```python
# before: ptxas is free to move the entire load group across issue and wait.
T.ptx.cp.async_.commit_group()
for i in T.unroll(PREFETCHES):
    _load_independent_fragment(i)
T.ptx.cp.async_.wait_group(0)

# after: keep exactly the intended prefix between async issue and wait.
T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
T.ptx.cp.async_.commit_group()
for i in T.unroll(PREFETCHES):
    _load_independent_fragment(i)
    if i == WAIT_WINDOW_LOADS - 1:
        T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
T.ptx.cp.async_.wait_group(0)
```

Adding or dropping one of these constraints is never automatically safe. The
barrier can become a `NOP` or disappear as a machine barrier while its ordering
constraint still changes the schedule.

## Rationale

One B200 one-warp decode sweep added exactly one PTX `bar.warp.sync`, emitted no
SASS barrier, and kept registers and spills unchanged. Cold-L2 time nevertheless
moved by +2.07%, -0.76%, and -1.17% across its small, medium, and large default
shapes; both benchmark orderings agreed.

In an asynchronous-copy path, moving the independent-load source block alone
produced byte-equivalent scheduled SASS. A single warp boundary did change the
schedule, but placed all eight loads after `DEPBAR` and the first shared loads.
Keeping a boundary at asynchronous issue and adding a second after four of the
eight independent loads placed exactly that prefix between `LDGDEPBAR` and
`DEPBAR`; the only opcode-vector change was one extra `NOP` per warp. Extending
the same two-boundary shape to a neighboring specialization moved its complete-
matrix ratio from 0.9862 to 0.9931, and the resulting eight-shape matrix cleared
the strict gate.

## Boundary

Treat the movement as specialization-specific scheduling, not as a fixed barrier
latency: the sign differs per shape. A boundary immediately before or after the
wrong pipeline operation can sink every independent load behind the wait and be
worse than no boundary. Use a full mask only where every named lane converges;
this idiom is not permission to weaken a synchronization required for
correctness.

## Verification

Inspect PTX, final SASS, and resources for every affected shape. Confirm the
intended load prefix lies between asynchronous issue and wait, then check
registers, spills, the affected workloads, and their guard shapes.
