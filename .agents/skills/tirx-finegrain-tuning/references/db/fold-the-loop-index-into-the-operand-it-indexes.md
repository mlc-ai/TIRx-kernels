# Fold the loop index into the operand it indexes

**Symptoms:** `unroll_no_effect`, `instruction_count_bloat`, `excess_guard_math`, `branch_in_hot_loop`

## Symptom

Unrolling a mainloop changes nothing. One mainloop unrolled four ways with a
per-copy predicate, the shape nvcc emits for a trip count that is not a multiple
of four, measured 813 us against 810 us with dynamic instruction count unmoved,
because the body was still computing its scale-factor index as the loop counter
modulo four.

## What to change

Re-index the loop so the unroll factor turns the runtime value into a per-copy
compile-time constant: with `counter = U * outer + inner`, `inner` becomes that
index and the per-copy constants fold. Round the trip count up and cover the
tail with one predicate rather than duplicating the body, and advance the
counter outside that predicate -- inside it, the rounded loop never reaches its
bound and the kernel hangs.

```python
# before: the body computes its index from the runtime counter, so unrolling
# takes the code growth and none of the folding.
while mma_k < mma_kblocks:
    if mma_k % stages_per_load == 0:
        _copy_scale_factors(...)
    ...
    mma_k = mma_k + 1

# after: `u` is a per-copy literal, so the guard and the descriptor bit-field
# inserts vanish in the copies where it is nonzero.
mma_k: T.int32
mma_k = 0
mma_k_rounded: T.int32
mma_k_rounded = ((mma_kblocks + (UNROLL - 1)) // UNROLL) * UNROLL
while mma_k < mma_k_rounded:
    for u in T.unroll(0, UNROLL):
        with T.If(mma_k < mma_kblocks), T.Then():
            if u % stages_per_load == 0:
                _copy_scale_factors(...)
            ...
        # Advance unconditionally: the guard masks only the rounded-up tail,
        # and the loop still has to reach `mma_k_rounded`.
        mma_k = mma_k + 1
```

## Rationale

Where a reference unrolls, the payoff is usually that the unroll factor turns a
value the body reads at runtime into a per-copy compile-time constant; a
transcription that keeps reading the runtime value takes the code growth and
none of the folding. After re-indexing, the bit-field inserts that place the
index into the instruction descriptor, and the guard around the scale-factor
copy, both disappear in three of every four copies. The mainloop warp fell from
58.1 to 42.0 instructions per K block against the reference's 37.4, and the
kernel from 61.68M to 58.81M instructions.

## Boundary

The boundary is the fold, not the unroll. On the specializations where the
folded quantity was already constant, one scale-factor stage per load instead of
four, the same code measured unchanged, as predicted. A runtime trip count is
not itself the cost either: the same kernel compiled with a dynamic K bound
measured 0.9908x of the static-bound build.

## Verification

Confirm in SASS that the per-copy constants fold -- descriptor inserts and
guards gone in the folded copies -- then run the affected matrix.
