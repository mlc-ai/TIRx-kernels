# Preserve a lane-warp block layout when CTA prologue cost is visible

**Symptoms:** `thread_axis_drift`, `short_kernel_regression`, `barrier_init_overhead`, `launch_config_drift`

## Symptom

A short kernel regresses against a source whose CTA is shaped
`threadIdx.x/y = lane/warp`, after the port flattened it to one linear
`threadIdx.x` axis. Registers and instruction counts look reasonable and the
external reference is stable on both sides.

## What to change

Give the entry a lane-warp layout whose second dimension derives from the
canonical warp count, and put all barrier initialization under one explicit CTA
leader scope.

```python
@K.kernel(warps=NUM_WARPS, arch="sm_100a", grid=(...), thread_layout="lane_warp")
```

## Rationale

Flattening is semantically equivalent but not instruction-selection equivalent.
In a five-warp recurrent-state kernel, the flat entry plus three separate
barrier-leader guards compiled to 56 registers and 1,088 SASS instructions, yet
clean 45-round A/B measured after/before at 1.01244 and 1.01309 on the medium
and large production rows. The external reference was stable across both sides,
so these were real crossings rather than noise.

Restoring the lane-warp layout and one CTA leader scope restored the source's
60-register allocation and removed the duplicate leader work while keeping role
and barrier ownership in the entry. The final clean ratios were 1.00182 and
1.00179 at 45 rounds, the small-row guard was 0.99717 at 15 rounds, and all 36
correctness configurations passed.

## Boundary

Do not generalize this into a preference for two-dimensional blocks. Retain the
flat default, and select lane-warp only when the source owns that layout and
measured CTA-scale overhead makes the distinction observable.

## Verification

Compare registers, static SASS, and the duplicated barrier-leader work against
the source, then measure every affected row plus a small-shape guard in a clean
interleaved campaign.
