# Retry only work items published by the producer

**Symptoms:** `rare_numerical_fallback`, `global_retry_amplification`, `fallback_cost`

## Symptom

A persistent kernel can detect a numerical hazard only after producing an
intermediate.  Publishing one launch-wide hazard bit lets a following stable
kernel repair the result, but one risky item makes that kernel recompute every
recurrence stream and every otherwise independent item.

## What to change

Keep the launch-wide bit as the cheap entry predicate, and have the producer
append the global indices of risky independent items to a device queue.  A
same-stream retry kernel can reuse completed recurrence snapshots and their
epoch-stamped ready flags, map each compact queue rank back to the original
work-table index, and overwrite only those item outputs.  Let the retry
kernel's existing last-CTA completion election reset the queue count.

```python
# Producer, after the last reader of the diagnostic intermediate.
if work_item_is_risky:
    slot = atomic_add(retry_count, 1)
    retry_items[slot] = global_item
    atomic_or(mode_bits, RETRY_ITEMS)

# Same-stream consumer.  Recurrence snapshots are already complete.
if mode_bits & RETRY_ITEMS:
    for compact_rank in persistent_schedule(retry_count):
        global_item = retry_items[compact_rank]
        stable_item_epilogue(global_item, native_snapshots, native_epoch)
```

Aggregate across subheads before appending, or deduplicate with a per-item
bitmap.  Duplicate queue entries can race while overwriting the same output.
Allocate the queue in launcher-private workspace and initialize its count
before first use.

## Rationale

In one grouped-head backward kernel, a post-MMA diagonal scan published a
launch-wide retry bit.  Re-running the complete stable persistent kernel made a
sparse-risk full operation 2.64x as expensive as the original.  Reusing the
native recurrence snapshots but retrying all independent items reduced this to
2.35x.  Publishing one compact item index per risky work item reduced the same
operation to 1.46x.  Four amplified cancellation cases returned exact zero for
the affected gradient and were exactly repeatable.

The ordinary path kept its original item epilogue and added one inactive retry
launch.  Five-round complete-operation measurements were +3.05% and +2.21% on
two boundary shapes; a second measurement of the shorter shape was +3.01%.
A four-value-head grouping measured -0.03%.  All ordinary outputs were
elementwise identical and no retry bit was published.

The producer retained 168 registers and an 8-byte stack under the same local
nvcc path, although static local loads increased from 4 to 13.  The retry-only
specialization used 168 registers and an 80-byte stack; those spills execute
only after the entry predicate accepts a nonempty queue.  Full-operation
timing, rather than retry-kernel resources alone, established the ordinary
cost.

## Boundary

This works only when the retried item is independent once the producer's
recurrence snapshots are complete.  The retry must deliberately use the
producer's epoch; assigning a new epoch would make it wait for recurrence work
that the retry does not run.  A separate full stable kernel is still required
when the producer skipped recurrence work, such as an input guard selecting a
stable path before native execution.

Make that precedence launch-wide when any stable-path work item prevents the
native producer from publishing a complete snapshot set.  In a measured mixed
KDA case, one strong-decay value head caused the native launch to exit before
all work, while a different head contained a diagonal-8 retry hazard.  The
correct dispatch retained only the strong-mode bit, left the item-retry bit and
queue count at zero, and let the full stable kernel cover every head.  The same
result held for grouping ratios two and four, with the retry hazard in the mild
head, the strong head, or both.  A mild companion input published the retry bit,
confirming that the diagnostic itself was active.  Do not run an item-only
consumer when its required native snapshots were skipped, even if an
independent item would otherwise meet the retry predicate.

Scope the retry's output writes to the outputs whose error mechanism the
diagnostic actually covers, but determine that set with directed numerical
tests rather than from the diagnostic's name.  In the measured KDA case, a
diagonal-only gate-gradient retry left a key gradient difference as large as
`1.60e33` on the amplified input, even though the existing relative tolerance
still passed.  Retrying query, key, and gate gradients restored the intended
rounded-diagonal formulation.  Retrying value and beta gradients was
unnecessary: under correlated inputs it changed beta-gradient error in both
directions without fixing the conditioning failure.  Keeping the native value
and beta outputs also made the sparse-risk operation 1.26% faster than the
all-output item retry.  The retry body retained 168 registers and an 80-byte
stack, while static global stores fell from 270 to 237 and local loads/stores
fell from 69/57 to 44/54 under the same local toolchain.  Removing output stores
can shorten fallback work and compiler-managed live ranges even when the
high-level computation is otherwise unchanged.  Test outputs one by one;
neither “rewrite only the
reported gradient” nor “rewrite every output produced by the item” is a safe
default.

The measured sparse-risk gain does not bound a dense-risk workload.  As the
queue approaches all items, cost approaches the all-item retry.  The
launch-wide diagnostic threshold also remains part of numerical coverage: a
queue cannot repair hazards that the producer does not publish.

A threshold that missed four initially selected ordinary shapes still fired on
one later official shape.  That call remained correct, but enough items entered
the queue to regress the complete operation by 28.16%.  Raising the rounded
BF16 threshold from 5 to 8 kept the controlled diagonal-8 correction boundary,
left an 18-bit measured margin to the first pointwise-tolerance crossing, and
made all sixteen official instances of that kernel family stay on the native
path.  The formerly triggered shape then measured +1.42%.  Qualify dispatch
thresholds across the complete workload portfolio; a few zero-trigger timing
shapes do not establish a low fallback rate.

Do not infer the missed-hazard margin from the diagnostic magnitude alone when
the native result also depends on rounded secondary factors.  In the same
kernel, a zero-gate diagonal-4 case left a `2.95e-7` analytical-cancellation
residual.  Scanning 2,049 mild-gate points with the same diagonal increased the
worst residual to `5.46e-4`; scanning the largest tested BF16 diagonal below 8
increased it to `1.087e-3`.  All 4,098 dense-scan rows stayed below the retry
threshold, so this measured the native path rather than the repair.  The latter
still had about 92x margin to the pointwise `atol=0.1`, but the apparent
zero-gate amplitude margin substantially overstated the full factorization
margin.  Sweep each independent rounded factor, positions within the tile, and
the threshold's immediate lower edge.  When the analytical reference is zero,
report maximum absolute residual as well as normalized RMS; any nonzero result
can make the latter look unbounded without being a pointwise material error.

Also perturb upstream magnitude around an ordinary input that sits near the
threshold.  Lowering the same rounded threshold from 8 to 6 left the base input
on the native path, but multiplying its upstream gradients by only 1.2 moved
the lower-threshold version into retry while threshold 8 stayed native.  Across
scales 1.2 through 1.5, this made the complete operation 24.9--25.2% slower;
the largest output difference between thresholds was only `2.68e-4`, far below
the `0.1` pointwise tolerance.  At scale 1.6 both thresholds retried and their
times converged.  A zero-trigger default seed is therefore insufficient even
when a lower threshold passes the official input unchanged; sweep plausible
gradient scaling before trading fallback frequency for a smaller absolute
rounding residual.

The queue and completion counters make the workspace invocation-private.  Do
not overlap calls that share it.  CUDA Graph replay still needs correct ready
flag lifecycle; resetting the retry queue does not repair stale recurrence
epochs.

## Verification

Test zero, one, several, and all-item queues; multiple grouped subheads; partial
chunks; and work-table index reconstruction.  Alternate risky and ordinary
inputs through the same launcher and require the mode bit and queue count to
return to zero.  Compare every output, require exact repeatability, and measure
the complete operation on both the inactive path and representative sparse and
dense retry rates.  Include the largest supported grouping ratio on an actual
risk input: a zero-trigger ordinary run only checks the native mapping.  In the
measured KDA case, Hq=2/Hv=8 (four value heads per query/key head) passed four
amplified cancellation inputs with exact zero repaired gradients and bitwise
repeatability.
