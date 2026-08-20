# Roll a loop with While, not a counted loop

**Symptoms:** `instruction_count_bloat`, `unroll_no_effect`, `slow_small_shape`, `barrier_stall`, `sass_divergence`

## Symptom

A transcribed loop the reference leaves rolled appears several times over in the
port. One digit-pass loop emitted 837 instructions and 35 static `bar.sync`
against the reference's 258 and 9, for identical dynamic work, and the port was
slowest on exactly the shapes with the fewest passes. A scheduler's batch scans
carried 32 `ld.global.b32` and 24 `div.s32` against the reference's 10 and 6,
with the extra copies fed by a peeled remainder tail.

## What to change

Carry the offset in a scalar and write a `While`, which TVM prints
`#pragma unroll 1` ahead of. A counted loop gets no unroll pragma at all: TVM's
C codegen consults no loop annotation, so there is no way to ask a `For` to stay
rolled, and nvcc decides on its own.

Take a trailing barrier only when another iteration follows, so a `9P - 1`
barrier count falls out instead of being constructed.

```python
# before: a counted loop, which nvcc unrolls.
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

A runtime trip count needs the same treatment; hoist one cursor if several
loops in a row want it.

```python
# before: extent is a kernel argument, and nvcc still unrolls it 4x plus a tail.
for b in T.serial(0, num_items):
    consume(b)

# after:
cursor = T.alloc_local((1,), "int32")
cursor[0] = 0
while cursor[0] < num_items:
    consume(cursor[0])
    cursor[0] = cursor[0] + 1
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

A runtime extent is not protection. Unrolling needs no compile-time trip count:
nvcc emits four copies plus a remainder tail against a runtime bound just as
readily, which on a scan whose body is one dependent global load tripled the
load and divide traffic of the serial chain the kernel's latency sat on.

| | reference | counted loop | `While` |
| --- | ---: | ---: | ---: |
| `ld.global.b32` | 10 | 32 | 10 |
| `div.s32` | 6 | 24 | 6 |
| `.pragma "nounroll"` | 4, one per scan | 5, none on a scan | 6, all four scans |

## Boundary

Only where the reference's loop is genuinely rolled -- whether it says so with a
`while (true)` and a data-dependent break, or with an explicit no-unroll pragma
on a counted loop. If it unrolls its own loop, match that instead. Rolling is
not a general win: it trades code size against loop overhead, and a loop whose
body is small relative to its trip count can prefer the unrolled form.

## Verification

Count static `bar.sync`, or the body's own dominant opcode, in the port against
the reference before believing any loop is rolled. A rolled loop in the
generated CUDA proves nothing, because the unrolling happens in nvcc -- this
change was first made, measured, and written up as a win while the loop was
still being fully unrolled. Matching totals do not prove it either: read the
loop body, since a port can hit the reference's exact static counts and still
carry a fatter body.
