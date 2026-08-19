# Measure elided warp synchronization as a scheduling constraint

**Symptoms:** `warp_sync_schedule_shift`, `small_shape_regression`, `bar_warp_elided`

## Symptom

A converged full-mask `bar.warp.sync` remains in PTX while ptxas emits no
`BAR.WARP` instruction, and the change looks free. It is not: the ordering
constraint can still reschedule surrounding instructions differently for each
specialization.

## What to change

Nothing is automatically safe to add or drop here. Treat the added or removed
sync as a scheduling change and A/B the complete dispatch matrix even when the
machine barrier is elided.

```python
T.cuda.warp_sync()  # may emit no SASS barrier and still move time
```

## Rationale

One B200 one-warp decode sweep added exactly one PTX `bar.warp.sync`, emitted no
SASS barrier, and kept registers and spills unchanged. Cold-L2 time nevertheless
moved by +2.07%, -0.76%, and -1.17% across its small, medium, and large default
shapes; both benchmark orderings agreed.

## Boundary

Treat the movement as specialization-specific scheduling, not as a fixed barrier
latency: the sign differs per shape.

## Verification

Inspect PTX, SASS, and resources for every affected shape, then A/B the complete
dispatch matrix.
