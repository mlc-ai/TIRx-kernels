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

## Boundary

This is a stopping rule for the fixed region of a small kernel, not licence to
ignore instruction counts generally. A loop body that runs thousands of
iterations still pays for every instruction in it.

## Verification

Before spending another change, list the countable quantities side by side --
PTX and SASS totals, dynamic instructions, registers, shared bytes, barriers,
bank conflicts, parameters, launch bounds. If none of them is worse, the next
measurement to take is not of the kernel.
