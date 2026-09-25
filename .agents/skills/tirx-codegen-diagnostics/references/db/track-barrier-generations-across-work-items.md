# Track barrier generations across work items

**Symptoms:** `kernel_deadlock`, `mbarrier_arrive_before_consumption`, `repeated_async_launch_hang`, `work_item_boundary`

## Symptom

A persistent kernel completes when every launch is synchronized but can hang
after many launches are queued. Synccheck reports that a new barrier generation
arrives before the prior generation was consumed. The reported barrier may be
reused by several logical streams or may have a conditional consumer.

## What to change

Gate a depth-one tile's reuse with a counter whose lifetime matches the shared
tile, rather than an inner-loop index that resets at each work item.

```python
# before: inner_index resets when the next stream starts
if inner_index > 0:
    tile_empty.wait(0, previous_phase)

# after: tile_cycle advances on every physical reuse
if tile_cycle > 0:
    tile_empty.wait(0, previous_phase)
```

When every thread arrives at a fixed-count rendezvous, every thread must also
consume that generation before a later iteration can arrive. Keep only the
work after the wait conditional.

```python
# before: generations with no work are never consumed
barrier.arrive(0)
if has_work:
    barrier.wait(0, phase)
    consume()

# after: all participants close the generation
barrier.arrive(0)
barrier.wait(0, phase)
if has_work:
    consume()
```

## Rationale

A phase bit identifies which generation a wait expects; it does not by itself
prevent a producer from starting the next generation. If a logical loop index
resets while the barrier and shared storage persist, the first use in the next
work item can skip the only empty wait. Likewise, a conditional wait leaves an
arrived generation unconsumed when its condition is false.

Changing the reuse guard to a role-level monotone counter made Synccheck clean
on the realistic 20-CTA topology and allowed 1,000 launches to be queued before
one final device synchronization for every registered benchmark shape. One
shape that first hit an external watchdog completed a 2,000-launch repeat.
Moving the fixed-count wait outside its work predicate made both the persistent
and fused stable specializations clean; their checks explored 2,332 and 664
verifier states respectively.

## Boundary

The counter must advance once per physical buffer reuse in every role that
participates in the protocol. An unconditional wait is appropriate only when
the barrier's expected arrival count includes those same threads and they all
execute the surrounding loop iteration.

Do not judge a globally scheduled persistent kernel from an artificially small
CTA topology. Too few CTAs can deadlock on real cross-CTA dependencies that the
production launch satisfies. Match the launcher's CTA count and schedule table.

## Verification

Run Synccheck with enough loop iterations to cross a logical work-item boundary
and with predicates that produce both empty and nonempty work generations.
Require zero findings on the production CTA topology. Then queue many complete
launches without intermediate host synchronization and require the final
device synchronization to finish; per-launch synchronization can hide the
timing needed to reproduce the failure.
