# Isolate unused fallback from the common path

**Symptoms:** `register_pressure`, `cold_branch_overhead`, `local_memory_spills`

## Symptom

Adding a numerically stable fallback increases the common path's register and
stack cost even when its runtime predicate is false. Retuning register budgets
helps that path but regresses inputs that use the fallback.

## What to change

Compile the common arithmetic separately from the extended implementation. A
small device guard selects exactly one active implementation; both entry points
read the same freshly written decision. Preserve the original arithmetic in
the common specialization and keep mutable workspaces and epochs consistent
across the alternatives.

## Rationale

In a measured persistent backward kernel, the extended specialization needed
a104-byte stack with57 local-load and36 local-store sites. A separately compiled
common specialization used an8-byte stack with12 loads and one store. Including
the guard and the inactive launch, a five-round paired ordinary case improved
from978.17 to852.71 microseconds. Three fallback-using controls improved4.1-5.8%
and preserved all six outputs bitwise; the common control passed the unchanged
numerical checker. The29-case registered matrix also passed.

Two register-budget alternatives reduced ordinary latency2.5-3.1% but increased
fallback-using latency3.1-8.8%. Separating the compiled bodies addressed the
unused-path cost without selecting a role budget that penalized active fallback.

## Boundary

The extra launch and decision scan still cost time, and duplicating bodies
increases compilation and maintenance cost. This measured result does not imply
that separate launches outperform one kernel for small workloads or that a
runtime branch creates independently allocated register budgets.

Recompute the decision when inputs change; do not use a one-time host sample.
Every CTA must reach the same entry decision before a persistent dependency
protocol starts. An inactive implementation must leave shared workspaces alone.
Graph replay correctness is an additional requirement when epochs are captured;
ordinary repeated-launch checks do not establish it.

Reducing replicated decision loads in the common entry is not automatically a
win. Reading one partition's gate bit per chain preserved all outputs and flags
on ten controlled pairs, with unchanged register and stack usage in the inspected
bodies. Three ordinary shapes regressed by 0.15–0.53%, while two improved by
0.69% and 1.79%; fallback-using controls were nearly neutral. Keep the complete
compiled body in the measurement rather than assuming fewer entry loads reduce
total time. That broadly applied rewrite was not adopted.

A per-warp copy of the entry decision is also not automatically cheaper than
a CTA reduction. Each warp scanned one representative gate flag per chain,
then performed a full-mask integer OR reduction. The guard guaranteed that the
selected bit agreed across each chain's partitions, so all warps reached the
same decision without the entry CTA barrier. Registers and stack were unchanged;
ordinary latency changed only -0.10% and a fallback control +0.08%, with original
checks and all six bytewise outputs preserved. The extra replicated loads and
address arithmetic erased the expected synchronization benefit. This rewrite
was not adopted; the complete tail matrix was not qualified.

Retaining both complete arithmetic bodies inside one runtime-branched entry
also regressed the common path in a measured prototype. The
mutually exclusive bodies reused one dynamic shared arena and kept their role
budgets, but the merged stack grew from 16 to 88 bytes. Including the unchanged
guard, ordinary latency increased 11.11% despite eliminating a 3.66-microsecond
inactive launch. Initializer inputs were nearly flat (-0.15%) and dense fallback
improved 2.45%; all three cases passed original tolerances and six bytewise output
comparisons. The merged compilation used register optimization level 10 for both
bodies, whereas the separated fallback used 5, so the dense improvement cannot
be attributed solely to the removed launch. A top-level branch does not isolate
compiler register allocation. This composition was not adopted.

Moving an inactive-path return ahead of an unrelated amplitude reduction
saved about 0.19 microseconds in that entry (3.62 to 3.43), with unchanged static
resources and all three original/bytewise checks passing. The complete ordinary
call was flat (+0.04%); initializer and dense controls changed +0.18% and -0.37%.
That isolated entry saving did not establish a worthwhile full-operation gain,
so this rewrite was not adopted either.

## Verification

Inspect both compiled bodies and compare them with the intended common formula.
Measure the complete guard-plus-alternatives launch sequence with ordinary,
mixed, and dense fallback inputs. Test transitions using the same launcher,
original numerical tolerances, and poisoned outputs. Include compile time and
required prepare-stage cache entries in the integration review.
