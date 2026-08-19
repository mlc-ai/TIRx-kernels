# Tune pipeline shape and transfer granularity

**Symptoms:** `barrier_stall`, `exposed_epilogue_tail`, `underfilled_pipeline`, `late_stage_completion`, `tma_issue_overhead`

## Symptom

Barrier stalls, an exposed epilogue tail, an underfilled pipeline, late stage
completion, or TMA issue overhead on short shapes.

## What to change

Choose pipeline depth from wave count and exposed latency, not from the largest
possible ring. Single-wave work may benefit from deeper accumulator buffering;
multi-wave work may need more producer stages. Depth is a constexpr the ring
init and every stage index derive from:

```python
# Depth solved from the shared-memory budget, then capped.
num_stages = min((SMEM_CAPACITY - smem_extra) // smem_per_stage, NUM_MAX_STAGES)

# One init per ring, arrive-count set per barrier family.
for stage_init in T.unroll(STAGES):
    T.ptx.mbarrier.init.shared__cta.b64(
        smem_raw.ptr_to([full_off + stage_init * 8]), T.uint32(1), pred=leader
    )
for stage_init in T.unroll(STAGES):
    T.ptx.mbarrier.init.shared__cta.b64(
        smem_raw.ptr_to([empty_off + stage_init * 8]), T.uint32(32), pred=leader
    )
```

Split a large TMA box only when earlier sub-box completion benefits a real
steady state. Preserve alignment, swizzle atoms, coverage, and exact barrier
byte accounting: one expect-tx covers every sub-box issued against it.

```python
_mbarrier_expect_tx(smem_raw, stage * T.uint32(8), 8192)  # total bytes, all boxes
for box in T.unroll(4):
    _tma_2d_g2s(
        smem_raw,
        T.uint32(WT_OFF) + (stage * T.uint32(4) + T.uint32(box)) * T.uint32(2048),
        a_tmap, k_base + box * 64, mib, stage * T.uint32(8),
    )
```

## Rationale

Ring depth trades buffering against footprint and stage completion timing, and
extra TMA issue instructions often regress short shapes.

## Boundary

Never reduce a protocol ring below its proven safe depth.

## Verification

Benchmark both sides of every dispatch boundary and validate deadlock freedom,
footprints, registers, stage completion timing, and issue counts.
