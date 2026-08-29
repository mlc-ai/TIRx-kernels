# Materialize reused thread-derived indices in short kernels

**Symptoms:** `repeated_index_expression`, `excess_address_math`, `short_kernel_regression`, `sass_instruction_gap`

## Symptom

A rewritten body sits above a frozen SASS instruction count and regresses a
short kernel, with the excess visible as repeated row, column, or lane-unit
arithmetic in generated address code. The source materialized those indices
once.

## What to change

Bind the reused thread-derived indices to local scalars, and only those.

```python
row_in_block = K.local_scalar("int32")
sf_idx_in_row = K.local_scalar("int32")
K.assign(row_in_block, tid // COLS_PER_ROW)
K.assign(sf_idx_in_row, (tid % COLS_PER_ROW) // SF_UNIT)
```

Materialize values whose reuse is visible in generated address arithmetic, not
every one-use expression.

## Rationale

In a 1,024-thread MXFP8 swizzled quantizer, repeated row, column, and lane-unit
expressions produced 320 SASS instructions at 32 registers and measured 1.03863
after/before. Materializing only those reused indices restored the frozen
304-instruction count without changing registers or introducing spill; targeted
correctness passed and the clean 45-round ratio was 1.00465.

The same boundary held for the small-K multi-row MXFP4 and NVFP4 quantizers.
Materializing only `row_in_block` and `sf_idx_in_row` moved MXFP4 from 216 to
the frozen 208 SASS instructions and from 54 to 48 registers; NVFP4 retained 152
instructions while restoring 31 registers from 32. Both targeted correctness
rows passed, and clean 45-round ratios were 1.00261 and 1.00391.

## Boundary

This result covers the measured small-K multi-row branch. Treat other branches
as separate specializations until their SASS and timings demonstrate the same
mechanism.

## Verification

Compare static SASS against the frozen count and check that registers do not
move and no spill appears, then measure the affected rows over a clean
interleaved campaign.
