# Widen contiguous range scan loads

**Symptoms:** `unsaturated_bandwidth`, `excess_loop_control`, `instruction_count_gap`

## Symptom

A magnitude-check pass has enough independent CTAs and no spills, but each
thread repeatedly loads one word, updates a predicate, and advances a strided
loop. The check costs a substantial fraction of the operation it protects.

## What to change

Assign each thread several adjacent words and issue explicit wide global
loads. Check every component with the original predicate before advancing by
the corresponding larger stride. Use the tensor's actual alignment and extent
to prove full vectors are valid, or handle the final partial vector separately.

## Rationale

On two measured input scans with the grid fixed at 4096 four-warp CTAs, widening
bf16 input loads from one word to eight words, and FP32 state loads from one to
four words, reduced scan latency from 24.73 to 10.85 us and from 228.04 to
94.66 us. Main-operation results passed the existing correctness checks.
SASS changed from scalar LDG.E to LDG.E.ENL2.256 and LDG.E.128; registers grew
from 17 to 28 while stack and local allocation remained zero. The complete
operation improved less, from 180.60 to 168.22 us and from 1111.08 to 978.17 us,
because its main kernel was unchanged.
The vector form also passed six same-launcher checks that changed upstream
gradients from their ordinary magnitude to 2^64 times larger and back. Flags
cleared on ordinary calls and selected the bounded fallback on large calls.

## Boundary

The wider load must preserve coverage and classification, including sign bits,
NaNs, infinities, and threshold equality. This measurement used extents divisible
by the vector width; it does not justify reading outside a tensor for a tail.
Avoid scanning undefined padding or unused matrix entries: a conservative guard
can otherwise reject valid inputs because of bytes the operation never reads.

More registers or a different grid can move the bottleneck. Inspect the emitted
load width, allocation, and realized grid rather than assuming source vector
syntax supplies the same schedule. The remaining pass cost may still be too
large to justify unconditional execution.

A subsequent gate-first scan skipped the magnitude pass for entirely mild
chains, but one CTA per chain left a small shape with only 44 CTAs. Partitioning
each chain's vector scan across 16 CTAs reduced its strong-profile scan from
99.46 to 16.87 us, while its mild gate-only scan changed from about 5.77 to
6.01 us. Both mild and initialization profiles passed the original checker.
On a larger shape, using two partitions per chain changed the strong scan only
from 98.42 to 96.68 us. Extra CTAs help an underfilled grid; they do not remove
the cost of reading all protected inputs. Every partition checked the same
chain gate condition and wrote its own flag, which the consumer reduced across
the complete partition count.

Do not reverse that partitioning just to remove duplicate gate reads. In a
later measured composition, halving 16 partitions to eight preserved all six
output arrays and each chain's combined mode bits on six paired profiles. The
guard kept 32 registers and no stack, and ordinary gate-only scans improved
only about 0.11/0.04 us. Initialization scans grew from 14.53 to 20.21 us and
23.35 to 32.74 us; dense scans grew similarly. Complete-operation changes
ranged from a 0.52% improvement to a 1.30% regression. Reduced duplicate
traffic did not repay the longer magnitude loops and reduced parallelism.

## Verification

Compare the guard result before and after widening, including values just above
and below the threshold at vector boundaries. Check changing inputs on repeated
invocations. Measure the guard independently to identify its cost, and include
all guard kernels in the final operation timing.
