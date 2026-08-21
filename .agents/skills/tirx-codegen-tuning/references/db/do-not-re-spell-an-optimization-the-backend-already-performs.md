# Do not re-spell an optimization the backend already performs

**Symptoms:** `sass_divergence`, `instruction_count_gap`, `excess_control_instructions`, `unroll_no_effect`

## Symptom

A PTX-level census shows the port carrying far more of some instruction than the
reference, the source is rewritten to close the gap, and the timed path does not
move -- because the machine code never changed.

## What to change

Nothing, on the family of rewrites that only re-express common-subexpression
elimination, uniformity analysis, or loop-invariant predicate hoisting. Both
nvcc and ptxas already perform these, and doing them by hand in TIRx reaches
neither.

- Binding values held in one-element local buffers to SSA scalars so repeated
  reads collapse: six scheduler-metadata fields read 14x and 8x, and every SASS
  counter was unmoved -- 3552 static instructions on both sides, `R2UR` 32,
  `USHF` 21, `UIADD3` 116, `LEA.HI.X` 4, `IMAD.WIDE` 44. nvcc had already
  eliminated the repeated local-array reads.
- Hoisting a predicate out of a chain of predicated tensor-core instructions so
  it is computed once: the PTX gap that motivated it was 68 `elect.sync`
  against the reference's 30, and in SASS both sides carry 14 `VOTEU`, at 3552
  static instructions each. ptxas had already hoisted it.
- Hoisting a repeated shared read out of an epilogue store loop: `LOP3` fell
  327 to 310 and `IADD3` 204 to 188, but `LDS` stayed at 94 on both sides. The
  loads -- the quantity that was actually binding -- had already been
  eliminated; only integer work that was sitting in latency shadow moved.
- Staging several independent index decodes ahead of the asynchronous copies
  they feed, so the copies issue back to back: 4096 static instructions on both
  sides, `LDS` 94, `UTMA` 10, `IADD3` 204 against 205. ptxas already schedules
  independent decodes ahead of the copies.

A PTX-level difference is not evidence that a rewrite is available.

## Rationale

The two backend stages each close part of this space before any hand rewrite
can. nvcc eliminates repeated reads of one-element local buffers and of shared
addresses it can prove unaliased, so a source-level cache of them is dead on
arrival. ptxas re-derives warp uniformity from the whole function and hoists
loop-invariant predicates out of instruction chains, so a local hint restating
either is discarded.

What survives to SASS is the shape of the work, not its spelling. The `LOP3`
and `IADD3` movement in the third case is the clearest form of it: a rewrite can
move real counters and still touch nothing that binds, because the counter it
was aimed at was already at the backend's floor.

## Boundary

This is about spellings the backend normalizes, not about layout or scheduling.
Changes that alter *which* memory an access touches, how deep a staging buffer
is, or how many long-latency results are in flight do reach the machine code,
and those are where the gains on that kernel came from.

It is also distinct from a cut that does change SASS and still does not convert.
That is a stopping rule about instruction count no longer being the binding
resource; this is about a rewrite whose instructions never reach the machine at
all.

The reverse error is real: a rewrite whose static count is unchanged may still
change dynamic execution when it alters loop structure, so read the dynamic
counter rather than the static one before discarding it.

## Verification

Compile both variants to SASS and diff opcode histograms and the static total
before spending a measurement. Treat a one-instruction difference as identical;
two of the cases above moved a single `LOP3` or `IADD3` and neither reached the
clock. Where the two genuinely differ, bench.
