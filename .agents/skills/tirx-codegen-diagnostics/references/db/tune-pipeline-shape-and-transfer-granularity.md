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

Treat a separate L2-only data-prefetch stream as another compile-time pipeline
parameter. When the staged input ring already supplies enough lookahead, try a
specialization that omits only the `cp.async.bulk.prefetch.tensor.*.L2`
instructions. Keep TensorMap descriptor prefetches, the real global-to-shared
TMA loads, and their barrier protocol unchanged.

```python
# before: every specialization issues an independent L2 data-prefetch stream.
for future_k in range(PREFETCH_DISTANCE):
    _prefetch_input_tiles_to_l2(future_k)

# after: select the stream at trace time from the measured pipeline regime.
if ENABLE_L2_DATA_PREFETCH:
    for future_k in range(PREFETCH_DISTANCE):
        _prefetch_input_tiles_to_l2(future_k)
```

## Rationale

Ring depth trades buffering against footprint and stage completion timing, and
extra TMA issue instructions often regress short shapes.

Two SM107 specializations made that issue cost concrete. Disabling only the
L2 data-prefetch stream removed eight static `UTMAL2CCTL` instructions from
each final SASS while leaving TensorMap prefetches and functional TMA transfers
present. One specialization moved from 17.7983 to 13.2121 us, a 1.3471x
speedup, and its reference/target ratio moved from 0.9788 to 1.3200. A second
moved from 15.7198 to 13.3047 us, a 1.1815x speedup, and its ratio moved from
0.9853 to 1.1641. Both changes passed bitwise source/oracle checks and the
complete correctness and performance matrices.

## Boundary

Never reduce a protocol ring below its proven safe depth. Safe depth is also
not fast depth, and a sibling specialization's tolerance is no proof: an input
ring one specialization runs correctly at depth three (its iterations are
longer) starved a throughput build whose consumer keeps a two-stage lookahead
in flight when the ring was cut from four to three to fund an extra output
stage -- two shapes near 0.99x fell to 0.92x and 0.82x, and the cut was
reverted. Fund new stages from rings whose consumers hold no lookahead.

Do not turn L2 prefetch off globally. In the measurements above, one ring
covered its complete eleven-tile K loop, while the other covered only six of
twelve tiles; full-loop coverage is therefore sufficient motivation to test,
not a necessary or transferable cutoff. A sibling prefetching specialization
that already passed was left unchanged. Cold-cache behavior, reuse distance,
and a different producer schedule can reverse the result. The timing and
binary diff prove the instruction removal and its effect, not which downstream
stall counter changed.

## Verification

Benchmark both sides of every dispatch boundary and validate deadlock freedom,
footprints, registers, stage completion timing, and issue counts. For an
L2-only prefetch change, confirm that `UTMAL2CCTL.PF` disappears while
descriptor prefetches, functional TMA loads, and barrier byte counts remain;
then test both the changed specialization and unchanged prefetching guards.
