# Spend warps before pipeline slots

**Symptoms:** `deeper_pipeline_slower`, `concurrency_capped`, `underfilled_pipeline`

## Symptom

A deeper per-warp pipeline measures slower than the single-buffered form it
replaced, at constant shared memory.

## What to change

For a single-buffered per-warp TMA copy loop, concurrency is warps times one
outstanding bulk copy. Count outstanding bytes -- warps times slot bytes --
before adding per-warp depth, and scale depth only after the SM's warp slots are
full.

```python
# A two-slot per-warp pipeline at constant SMEM halves the warp count.
# Outstanding bytes, not slot count, is the quantity to hold up.
OUTSTANDING = T.meta_var(NUM_WARPS * SLOTS_PER_WARP * SLOT_BYTES)
```

## Rationale

An A/B-paired two-slot variant of one dispatch epilogue (8 warps x 2 slots,
token B's load overlapping token A's wait and store) was correct but measured
126.5 us against the single-slot form's 112.4 us p25, reference 99.2 us. The
source's 16-warps x 1-slot shape was already near the bandwidth ceiling, so the
extra slots bought latency hiding that the warp count was already providing.

## Verification

Pair the A/B at constant shared memory and compare outstanding bytes on both
sides, not slot depth alone.
