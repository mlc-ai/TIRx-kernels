# Stop cutting instructions once parity is reached

**Symptoms:** `instruction_parity_with_deficit`, `slow_small_shape`, `instruction_count_gap`, `schedule_regression`

## Symptom

A port that is level with or ahead of the reference on every quantity that can
be counted, and still slower on the small shapes. One entry sat at 0.948 while
holding 240 SASS instructions against 256, 2612 dynamic against 2692, 49
registers against 51, 2 kernel parameters against 4, an identical shared-memory
opcode sequence, identical bank-conflict counts, and identical barriers.

## What to change

Nothing further on instruction count. Past that point the remaining cuts do not
convert:

- widening a narrow load to make a sign test a bare `setp.lt.s32`, removing 16
  instructions from the prologue and epilogue, measured **worse**;
- collapsing four contiguous unguarded loads into one `ld.global.v4.b32` --
  available because a compile-time `k` folds the guard the reference must keep --
  moved the shape by 0.000;
- rebasing every item on a per-thread base, cutting 64-bit address arithmetic
  from 17 ops to 12, left the ratio unchanged.

Fewer or wider memory instructions do not relax an ordering constraint, and a
prologue that is already tighter than the reference's does not get faster by
being tighter still.

## Rationale

Instruction count stops being the binding resource once it matches. What
remains is latency the schedule cannot hide, and on a grid of a few single-warp
CTAs there is nothing resident to hide it behind.

The state itself is the signal: **everything countable equal or better, still
slower** means the difference is not in the code. Check the grid first -- a
deficit that disappears once the SM array fills is occupancy -- and then the
measurement, because a benchmark can carry a per-side bias that no amount of
codegen work will move.

A selective-state-update port gives the stronger counterexample. Forcing four
two-trip state loops to a single trip per unrolled body cut PTX instructions
from 1,548 to 1,007, packed FMA copies from 128 to 64, and main-function SASS
from 1,608 to 1,080 -- below the reference's 1,224. The same-GPU production row
nevertheless moved from about 378.5 to 381.1 us, farther from the 351.9 us
reference. A large static code reduction therefore falsified the
instruction-cache explanation for that 7.6% deficit. Treat loop-body size as a
mechanism only after timing moves with it; otherwise inspect the realized
schedule and latency of the executed path instead of continuing to minimize dead
or duplicated code.

## Boundary

This is a stopping rule for the fixed region of a small kernel, not licence to
ignore instruction counts generally. A loop body that runs thousands of
iterations still pays for every instruction in it.

## Verification

Before spending another change, list the countable quantities side by side --
PTX and SASS totals, dynamic instructions, registers, shared bytes, barriers,
bank conflicts, parameters, launch bounds. If none of them is worse, the next
measurement to take is not of the kernel.

Count them correctly. A `grep`-based opcode histogram undercounts a cub-style
reference twice over: predicated instructions begin with `@%p` rather than an
opcode, so a `^\s+[a-z]` pattern drops them, and an inline `asm` block's line
begins with `{`, hiding several instructions behind one unmatched line -- cub's
warp-scan shuffle hides three that way. Both bit here, and an early table
claiming the reference emitted zero `shfl.sync.up` when it emits five nearly
became the basis of a hypothesis.

Keep PTX-level and SASS-level claims separate as well. An absent opcode in PTX
says nothing about ptxas's own selection: one sketch recorded `prmt` as absent,
which was true of the PTX while ptxas still chose `PRMT` for a 16-bit sign
extension. That is not a divergence.
