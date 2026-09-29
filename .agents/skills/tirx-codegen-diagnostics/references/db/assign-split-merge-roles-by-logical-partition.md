# Assign split-merge roles by logical partition

**Symptoms:** `nondeterministic_output`, `split_merge_order`, `one_ulp_drift`

## Symptom

Identical launches differ by one output ULP even though every partial is
deterministic. The first split CTA to increment an atomic counter becomes the
stored operand and the second becomes the live operand, so scheduler timing can
swap the two sides of an asymmetric multiply-add lowering.

## What to change

Encode the logical partition ID in the work metadata. Give one partition the
producer role and the other the consumer role regardless of arrival order.
Publish the stored partial with a release operation and have the consumer wait
with an acquire load before merging it.

```python
# before: arrival order chooses the floating-point operand roles.
arrival = T.ptx.atom.acq_rel.gpu.global_.add.s32(counter, 1)
if arrival == 0:
    store_partial()
    publish_ready()
else:
    wait_ready_acquire()
    merge_live_with_stored()

# after: logical partition 0 always publishes and partition 1 always merges.
if partition == 0:
    store_partial()
    publish_ready_release()
else:
    wait_ready_acquire()
    merge_live_with_stored()
```

## Rationale

A measured two-way attention split differed by one BF16 ULP (6.103516e-5)
between identical launches. Fixing the roles by partition removed the
merge-order atomic and its shared broadcast barrier in generated code. The
finite, oracle, and exact-repeatability gates then passed, while latency moved
from 211.81 to 209.13 microseconds. A subsequent 15-round same-request A/B
measured 210.51 to 207.88 microseconds (-1.25%).

## Boundary

The consumer may wait only when the scheduler guarantees that its producer can
run. A persistent grid can deadlock if every resident CTA waits for producers
that have not been scheduled. Pair or order work IDs so each consumer's
producer is already assigned, or use a nonblocking rendezvous design.

This rule is needed only when swapping logical operands changes generated
floating-point evaluation. If the merge is proved bitwise symmetric, arrival
order may remain a valid scheduling optimization.

## Verification

Poison the output differently and compare at least two identical launches
bitwise. Inspect generated code to confirm the merge-order atomic and shared
order broadcast disappeared while the release/acquire ready protocol remains.
Then measure the split shapes; deterministic role assignment changes both
synchronization and instruction order.
