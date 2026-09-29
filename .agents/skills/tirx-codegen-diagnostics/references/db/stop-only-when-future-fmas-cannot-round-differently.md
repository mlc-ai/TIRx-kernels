# Stop only when future FMAs cannot round differently

**Symptoms:** `excess_instructions`, `loop_overhead`, `long_dependency_chain`, `schedule_regression`

## Symptom

A monotone-product reduction continues long after its addends are too small to
change any live FP32 accumulator. Waiting for the product itself to become zero
can leave many ineffective loads and FMAs.

## What to change

Bound every remaining FMA addend using the actual coefficient maximum and the
largest live monotone product. Compare that bound against a conservative lower
bound on half an ULP of every affected accumulator. Stop only if every later
FMA must round back to the current value. Recompute the accumulator bound at
the proposed exit point; a minimum saved before intervening cancellation is
insufficient. Retain exact-zero termination when the new predicate is disabled.

For a finite nonzero FP32 accumulator, half its distance to the closest neighbor
is at least its magnitude times 2^-25. One measured formulation used 2^-28 to
leave margin for coefficient products and predicate rounding. It scanned two
bf16 coefficient tiles with vector shared loads, reduced their absolute maximum
over the warp before divergence, and used a rounded division once per tile.
Zero accumulators and nonfinite or out-of-range coefficient bounds disabled
the shortcut. Full-mask collectives stayed outside divergent reduction loops.

## Rationale

Six paired cases passed unchanged correctness checks, and all six outputs in
every case were exactly equal to the otherwise identical baseline. Two constant
strong-decay cases improved from 6249 to 5313 us and from 634 to 570 us.
Ordinary controls changed by about 0.1% and 1.5%.

## Boundary

The same experiment slowed mixed decay distributions from 6732 to 7893 us and
from 670 to 825 us. These are 17–23% regressions despite exact output equality.
One specialization's stack grew from 8 to 16 bytes and local-load/store counts
from 32/31 to 38/46; the other kept its allocation and local-traffic counts.
The scan, per-iteration minimum, and exit predicates can cost more than the
skipped work. This is a limited intervention, not a default optimization.

Restricting the coefficient scan to warps where every active channel had a
chunk-end log decay at most -96 passed another six paired cases with exact
output equality. It retained approximately 9% and 15% strong-case improvements.
The mixed-distribution overhead fell to about 1.6% for one specialization, but
remained 16% for the other. Skipping the bound scan alone did not remove the
cost of the larger loop body and its generated schedule.

Moving the enabled/disabled choice outside the entire reduction, with the
original loop in the disabled branch, reduced that specialization's mixed-case
overhead to 3.35% (670 to 693 us). The ordinary control stayed near 142 us and
the strong case improved from 635 to 577 us. All outputs again matched exactly.
Stack and static local-load/store counts were unchanged from the guarded
version, so those resource counts alone did not explain the removed overhead.
The uniform outer choice avoids executing new per-iteration predicates in
warps where no shortcut is possible.

The argument requires monotone products, bounded finite operands, and no
observable side effects in skipped iterations. Include every accumulator and
both scan directions. Do not substitute an arbitrary small absolute threshold:
that changes numerical results rather than proving the skipped FMAs are no-ops.
The proof also needs the actual rounding mode and FTZ behavior of emitted
instructions. Separate rounding safety from a full numerical-accuracy claim
about the enclosing algorithm.

## Verification

Compare every output for exact equality in addition to the established oracle
checks. Exercise cancellation, zeros, sparse and dense masks, partial tiles,
large coefficients, and cases that cannot exit early. Inspect local traffic and
benchmark the complete kernel, including the bound computation.
