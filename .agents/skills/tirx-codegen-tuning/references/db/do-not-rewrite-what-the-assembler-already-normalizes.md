# Do not rewrite what the assembler already normalizes

**Symptoms:** `unroll_no_effect`, `sass_divergence`, `instruction_count_gap`, `excess_control_instructions`

## Symptom

A PTX-level census shows the port carrying far more of some instruction than the
reference, a source rewrite is made to close the gap, and the timed path does
not move — because the machine code never changed.

## What to change

Nothing, in these cases. Each of the following was written, compiled and found
to produce byte-identical SASS against the build it was meant to improve:

- reinstating an identity warp-uniform broadcast on a warp index that the
  backend already proves uniform;
- binding values held in a one-element local buffer to SSA scalars so repeated
  reads collapse — the reads were already common-subexpression-eliminated;
- hoisting a predicate out of a chain of predicated tensor-core instructions so
  it is computed once instead of per instruction;
- staging several independent index decodes ahead of the asynchronous copies
  they feed, so the copies issue back to back.

A PTX-level difference is not evidence that a rewrite is available. One
predicate gap of 68 against the reference's 30 collapsed to 14 against 14 in
SASS.

## Rationale

Five consecutive expansions on one kernel produced no machine-code change at
all; a sixth changed SASS but not the clock. The wasted work was not the edits,
which are cheap, but the measurement runs spent on them on a contended machine.

Adopting a rule of compiling to SASS and diffing before benching turned that
around: every subsequent candidate that passed the SASS screen moved the timed
path, and the two that failed it were dropped for free.

## Boundary

This is about spellings the backend normalizes, not about layout or scheduling.
Changes that alter *which* memory an access touches, how deep a staging buffer
is, or how many long-latency results are in flight do reach the machine code,
and those are where the gains on that kernel came from.

The reverse error is also real: a rewrite whose static count is unchanged may
still change dynamic execution when it alters loop structure, so read the
dynamic counter rather than the static one before discarding it.

## Verification

Compile both variants to SASS and diff opcode histograms and the static total.
Identical output means no measurement is owed. Where the two differ, bench.
