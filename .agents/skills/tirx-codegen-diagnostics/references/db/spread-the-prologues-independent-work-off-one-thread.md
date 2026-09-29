# Spread the prologue's independent work off one thread

**Symptoms:** `serialized_prologue`, `fixed_overhead`, `slow_small_shape`, `barrier_stall`

## Symptom

A per-CTA cost that the long shapes amortize and the short ones do not, so the
deficit tracks kernel length rather than any tile-level metric. One thread runs
a dependent chain -- work-item decode, a search, a metadata publish -- while the
rest of the CTA waits at the barrier that follows.

## What to change

Work out which prologue blocks share no data dependence and give them different
warps, keeping one fence and one CTA barrier after all of them. Pipeline
mbarrier initialization is the usual candidate: it depends on nothing the
metadata chain produces, yet transcription tends to put both on thread 0
because the source writes them as consecutive `if tid == 0` blocks.

```python
# before: one thread owns the dependent metadata chain AND the barrier init,
# and everyone else waits for the sum of the two.
if tidx == 0:
    ...dependent global loads, search, publish to shared...
if warp_idx == 0:
    if T.cuda.elect_sync():
        ...init the pipeline mbarriers...

# after: the init runs on another warp, under the same fence and barrier.
if tidx == 0:
    ...dependent global loads, search, publish to shared...
if warp_idx == 1:
    if T.cuda.elect_sync():
        ...init the pipeline mbarriers...
T.ptx.fence.mbarrier_init.release.cluster()
T.cuda.cta_sync()
```

## Rationale

The fence and CTA barrier already order both blocks against every consumer, so
splitting them costs nothing and removes one of the two from the critical path.
Measured, moving about thirty mbarrier initializations off the thread that also
owned a dependent-load metadata chain took the shortest shape from 0.9570 to
0.9931 and the next from 0.9822 to 0.9906, while the long-sequence shapes moved
by less than a tenth of that -- the signature of a fixed per-CTA cost.

At one block per SM there is no second block to hide the prologue behind, so
the whole chain is exposed. That is also why the effect is invisible on the
shapes that motivate most tuning.

## Boundary

Only for blocks with no data dependence, and only while the prologue runs
before role dispatch, so the warp taking the work has not yet specialized.
Splitting an mbarrier init from the `expect_tx` that arms the same barrier is
not safe -- keep an init and its transaction count together.

## Verification

Measure across the trip-count range rather than on one shape: the gain
concentrates where the kernel is shortest and disappears where it is longest,
so a single long-shape measurement will report no change.
