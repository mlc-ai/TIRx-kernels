# Keep TMA output copies single-issuer

**Symptoms:** `duplicate_tma_store`, `unordered_write_write`, `descriptor_not_acquired`

## Symptom

Racecheck reports overlapping global writes by different lanes at the same
TMA instruction, or a non-elected lane uses an unacquired mutable TensorMap.

## What to change

Predicate each shared-to-global TMA copy on one designated lane. Where the
TensorMap is acquired by an elected lane, use that same election for the copy.
Keep the other lanes' barrier arrivals and stage-state advances intact.

## Rationale

A warp guard alone issues 32 independent copies, not one warp-wide copy.
Restricting the issuing lane removed these ERRORs while preserving numerical
and synchronization checks. Four quantized GEMM epilogues also passed GPU,
NumSim and independent-reference comparison; a dense epilogue passed its
production GPU reference check. No latency equivalence is established.

## Boundary

Do not put a counted barrier inside the single-lane guard. Bulk-group waits
belong to their issuer; the stage-release synchronization must still order the
issuer's completion before any other producer reuses the shared source.

## Verification

Inspect the copy's generated predicate and descriptor acquire, then run the
affected numerical/synchronization cases and GPU comparisons. Benchmark hot
epilogues separately; fewer redundant writes do not prove unchanged timing.
