# Derive fixed TMEM columns from a known-good lowering

**Symptoms:** `manual_tmem_overlap`, `all_nan_output`, `tmem_base_guess`, `pool_removal_regression`

## Symptom

All-NaN output after a TMEM allocator was removed in favour of fixed column
bases, with no fault and no correctness signal from the individual MMA
instructions.

## What to change

Removing the allocator transfers ownership of its column-placement contract to
the kernel. Read the fixed bases from a known-good specialized lowering and
verify the complete allocation footprint. Do not infer a base from the last
visible MMA operand or from a guessed logical extent.

## Rationale

For one sparse head128 prefill, the correct allocation was O columns 0-255, P
columns 256-319, and Q starting at 320. A guessed Q base of 288 overlapped P and
produced all-NaN output. Transcribing 320 from the baseline lowering restored
the representative case; after an independent predicate-snapshot fix, all six
regular configurations passed.

## Boundary

The transcribed bases are valid for the specialization they were read from.
Re-read them whenever the tile shape, operand count, or MMA kind changes.

## Verification

List every TMEM region's base and column count and confirm the footprints are
disjoint and within the allocation, before attributing a NaN result to
arithmetic.
