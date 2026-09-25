# Aggregate launch-wide mode bits in the producer

**Symptoms:** `dispatch_overhead`, `repeated_metadata_scan`, `unnecessary_cta_reduction`

## Symptom

Every CTA in two mutually exclusive kernel entries scans the same array of
producer flags and reduces it to choose an execution mode. The mode is constant
throughout the subsequent launches, but the scans and CTA reductions repeat.

## What to change

Keep the original per-part flags, and aggregate their mode bits in the producer
with a device-wide atomic OR. A completed same-stream producer launch publishes
one summary word for both consumers. Each consumer reads the summary once and
uses the same original mode predicates.

When the existing completion protocol already elects the last active CTA, it
can reset the summary without another launch, but only after all readers have
finished. Initialize the summary to zero before first use. Skip the atomic when
the produced flag is zero.

```python
# before: duplicated in every consumer CTA.
local_bits = 0
for index in assigned_flag_indices:
    local_bits |= flags[index]
mode = reduce_mode_across_cta(local_bits)

# after: producer writes its original flag, then contributes a nonzero mode.
flags[part] = part_mode
if part_mode != 0:
    atomic_or_device(summary, part_mode)
# Consumer launch is ordered after the complete producer launch.
mode = load(summary)
```

The sketch omits the reset. Its location must follow from the actual readers,
launch order, and completion-counter protocol, not from the last loop iteration
of an arbitrary CTA.

## Rationale

Sixteen paired input transitions preserved all six output byte arrays,
including ordinary, strong, mixed, initialization-like and amplitude-rejected
inputs. On the affected family, the summary matched the OR of current per-part
flags and returned to zero after every complete invocation, including
strong-to-ordinary transitions through the same launcher.

Five-round complete-operation measurements on one shape moved ordinary input
from about 147.42 to 145.76 us, a synthetic initialization profile from 289.00
to 267.72 us, and dense strong decay from 643.66 to 631.67 us. Every paired
output and post-benchmark check remained byte-identical. The inactive entry
fell from about 3.66 to 2.78 us on ordinary input. The active native body also
changed slightly; the improvement must not be attributed only to fewer loads.

The native entry retained its 168-register metadata and 16-byte stack, while
static local load sites changed from 26 to 25 and barrier sites from 13 to 12.
The enhanced entry's assembled stack changed from 144 to 112 bytes, with local
load/store sites changing from 123/82 to 106/72 and barrier sites from 16 to 14.
Removing an entry reduction can change downstream register allocation too.

## Boundary

The tested producer emits only zero, strong-mode, or strong-plus-unsafe bits.
The native entry therefore runs only when the summary is zero and need not
reset it; the enhanced entry resets only after its existing last-CTA completion
election. Every thread must have completed its entry read before its CTA
increments that counter. An active path that exits before contributing to the
completion election invalidates the reset proof.

This is not a general grid barrier and does not permit overlapping invocations
that share the same mutable workspace. Preserve the original epoch protocol;
this change does not repair an existing Graph replay lifecycle problem.
Race-sanitizer and broader shape qualification are still required before
promotion of the measured prototype.

The extra guard and inactive launch still cost time. Direct original-versus-
prototype comparisons on three ordinary shapes remained about 6.95%, 7.27%,
and 9.12% slower, exceeding their 5% budget. A win over the previous guarded
implementation does not prove acceptance against the original operation.

Placing the summary in an unused word of an existing counter allocation let a
later native entry drop its extra pointer/count parameters. The entry used a
fixed-offset load instead of count-based address formation and recovered the
original parameter-region size, but retained the 168-register allocation,
16-byte stack and 25/36 local load/store sites. Sixteen changing-input checks
retained all six output arrays and summary resets. Direct complete-operation
comparisons with the original on three ordinary shapes were still 5.82–6.71%
slower. Restoring the parameter list is not equivalent to restoring the
original generated body or meeting its latency budget; no paired timing with
the previous summary placement was made in that experiment.

Fusing the same scan into a large persistent consumer is not automatically an
improvement. One experiment assigned original guard entries by CTA stride to
the first four warps of a twelve-warp consumer and inserted a native cooperative
grid synchronization before reading the summary. Guard reductions used a
separate named barrier; all threads synchronized before any early exit, and
local and remote lowering retained the cooperative launch flag. Sixteen input
transitions preserved the flags, summary resets, and all six output byte arrays.
Nevertheless, offline native stack grew from 16 to 24 bytes and local
load/store sites from 25/36 to 28/37. In paired complete-operation measurements,
ordinary input slowed from 145.62 to 153.75 us, initialization-like input from
269.19 to 295.62 us, and dense decay from 630.83 to 663.42 us. The scan and
synchronization inside the mostly inactive native entry cost about 42.5–42.8 us
on the latter two profiles, versus about 17.7–17.8 us for the separate guard
plus inactive native entry. Three direct original comparisons remained
11.54–14.54% slower. The experiment changes scan occupancy and adds grid
synchronization as well as removing a launch; do not attribute the regression
to only the extra stack or assume a fused producer retains standalone scan
throughput. Keep the separate producer unless full-operation evidence justifies
the consumer's extra synchronization and resource costs.

Moving the native consumer's launch-wide mode test from an entry return into
its scheduler initialization did not recover the remaining launch cost. The
scheduler seeded its first work item with `total_work` when enhanced mode was
active, so the ordinary path kept the native body and the enhanced path ran no
work after common initialization. Six ordinary, strong, and mixed transitions
kept all six outputs byte-identical to the prior summary-counter version.
Direct complete-operation measurements against the original were still
5.456%, 7.887%, and 7.757% slower on three ordinary shapes; only a packed shape
stayed within budget at 1.472%. Eliminating the early return does not by itself
restore the original entry cost because common scheduler setup and the extra
launch remain in the measured operation.

## Verification

Check the producer summary against the OR of the actual current flags, verify
reset after active and inactive paths, and alternate modes through one launcher.
Compare all output bytes and unchanged numerical checks, including after repeated
timing calls. Validate partial chunks, dispatch/cache-key agreement, flag/reset
sanitizer coverage, and complete-operation timing against both the previous
guarded implementation and the original baseline.
