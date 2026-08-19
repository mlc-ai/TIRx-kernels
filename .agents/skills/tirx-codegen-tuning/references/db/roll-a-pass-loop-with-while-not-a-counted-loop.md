# Roll a pass loop with While, not a counted loop

**Symptoms:** `instruction_count_bloat`, `slow_small_shape`, `barrier_stall`, `sass_divergence`

## Symptom

A transcribed loop the reference leaves rolled appears several times over in the
port. One digit-pass loop emitted 837 instructions and 35 static `bar.sync`
against the reference's 258 and 9, for identical dynamic work, and the port was
slowest on exactly the shapes with the fewest passes.

## What to change

A counted loop has a compile-time trip count and nvcc fully unrolls it. Where
the reference's loop is a `while (true)` with a data-dependent break -- anything
bounded by a runtime `end_bit`, or by a break the compiler cannot resolve -- carry
the offset in a scalar and write a `While`, which TVM prints `#pragma unroll 1`
ahead of. Take the trailing barrier only when another iteration follows, so the
`9P - 1` barrier count falls out instead of being constructed.

```python
# before: a compile-time trip count, which nvcc unrolls into P copies.
for p in T.serial(0, num_passes):
    begin: T.int32 = p * RADIX_BITS
    emit_rank_and_exchange(..., begin)
    if p != num_passes - 1:
        bar_sync()

# after: the reference's shape -- advance in the body, barrier only if more follows.
begin = T.alloc_local((1,), "int32")
begin[0] = 0
while begin[0] < num_passes * RADIX_BITS:
    emit_rank_and_exchange(..., begin[0])
    begin[0] = begin[0] + RADIX_BITS
    if begin[0] < num_passes * RADIX_BITS:
        bar_sync()
```

## Rationale

The unrolled body is the same dynamic work, but several copies of it on a grid
too small to hide instruction fetch cost more than the loop overhead they
remove. The effect is concentrated where the pass count is lowest, because that
is where the extra code has the fewest iterations to amortize over: the worst
shape moved 0.886 to 0.948, while shapes with twice the passes barely moved.

| | reference | counted loop | `While` |
| --- | ---: | ---: | ---: |
| `bar.sync` | 9 | 35 | 9 |
| instructions | 258 | 837 | 297 |

## Boundary

Only where the reference's loop is genuinely rolled. If it unrolls its own loop,
match that instead. Rolling is not a general win: it trades code size against
loop overhead, and a loop whose body is small relative to its trip count can
prefer the unrolled form.

## Verification

Count static `bar.sync` in the port against the reference before believing any
loop is rolled. A rolled loop in the generated CUDA proves nothing, because the
unrolling happens in nvcc -- this change was first made, measured, and written
up as a win while the loop was still being fully unrolled.
