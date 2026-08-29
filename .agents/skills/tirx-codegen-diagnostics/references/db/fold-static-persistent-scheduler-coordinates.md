# Fold static persistent scheduler coordinates

**Symptoms:** `repeated_divide_fixup`, `excess_integer_math`, `persistent_grid_regression`, `latency_bound_fixup`

## Symptom

A persistent work index is decoded with quotient/remainder operations even when
the batch count is one or a tile count is a compile-time power of two. The
decode repeats for every work item and sits on the address path for the next
TMA or epilogue transaction.

## What to change

Fold the statically known batch coordinate, and use mask/shift only where the
counter is non-negative and the divisor is an exact power of two. Keep the
generic division path as a compile-time alternative for specializations whose
scheduled binary regresses.

```python
# before: generic decode on every persistent work item.
n_index = quotient % N_TILES
batch = quotient // N_TILES

# after: a single batch makes both operations dead.
if BATCHES == 1:
    n_index = quotient
    batch = 0
# after: exact non-negative power-of-two decode.
elif N_TILES & (N_TILES - 1) == 0:
    n_index = K.bitwise_and(quotient, K.int32(N_TILES - 1))
    batch = K.shift_right(quotient, K.uint32(N_TILES.bit_length() - 1))
else:
    n_index = quotient % N_TILES
    batch = quotient // N_TILES
```

## Rationale

Folding the one-batch decode removed a large-shape BF16 failure: the affected
and guard rows both passed at 0.9981-1.0102x. Extending the exact mask/shift
decode to power-of-two tile counts reduced one FP32 target by about 0.925 us;
paired profiling measured 0.9928x even though a separate canonical timing
window remained narrowly below its gate.

The arithmetic identity was not universally a scheduling win. One packed
output path became about 1 us slower at the largest one-batch shape. Restoring
the original divmod only for that compile-time path reduced the target from
about 35.6 to 34.35 us and passed at 1.0004x, while the other output paths kept
the fold. The reusable change is therefore a specialized decode, not a global
source rewrite.

## Boundary

The shift/mask equivalence requires a non-negative counter and an exact
power-of-two divisor. The one-batch fold also requires the quotient to already
be in the N-tile range after the preceding coordinate decode.

Equivalent integer math can still change register lifetimes and ptxas
scheduling around a large epilogue. Retain a generic compile-time path when an
output or fragment family measures worse.

## Verification

Confirm the removed divide/fixup chain in SASS and compare the complete address
dependency path, not only static instruction totals. Validate the scheduler's
work-to-tile mapping and tails, then measure both low- and high-trip-count
shapes for every specialization that selects the fold.
