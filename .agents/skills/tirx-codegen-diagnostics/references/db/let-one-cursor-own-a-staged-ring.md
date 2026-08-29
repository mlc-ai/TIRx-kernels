# Let one cursor own a staged ring only when lowering proves the fit

**Symptoms:** `manual_stage_arithmetic`, `redundant_modulo`, `pipeline_state_suspicion`, `instruction_count_bloat`

## Symptom

Hand-rolled `iteration % STAGES` / `iteration // STAGES` stage and phase
arithmetic repeated across roles, or the opposite suspicion: a stage cursor
blamed for instruction-count or register growth.

## What to change

Use a pipeline cursor where stage and phase really are one protocol cursor, then
verify the final lowering rather than assuming the abstraction is free.

```python
state = K.PipelineState(NUM_STAGES, phase=0)
with K.While(index < total):
    full_barriers[state.stage].wait(state.phase)
    ...
    K.assign(index, index + STEP)
    state.advance()
```

Where the schedule's phase is dead protocol state, or the ring is a compile-time
unrolled stage map, a topology-strided ring, or a schedule whose phase is not
the cursor's wrap bit, explicit current/next stage indexing or `RingState` may
be clearer and faster.

## Rationale

Replacing two independent full-empty barrier schedules built on `iteration % 2`
/ `iteration // 2` with role-local two-stage cursors preserved a
1,920-instruction SASS stream exactly, including issue order, and kept 60
registers with no stack or spill; the corresponding 36-case correctness matrix
passed.

A phase-free shared-state double buffer is a separate boundary: its phase is
dead protocol state, yet the same cursor can still be useful if ptxas removes
it. Replacing the current/next modulo expressions in one runtime pass loop
reduced representative SASS from 736 to 728 instructions while retaining 60
registers and zero stack and spill, and all 43 correctness cases passed.

## Boundary

Do not decide this on instruction count alone. In another phase-free two-stage
prefetch loop, the cursor kept the same 936 instructions but raised registers
from 52 to 54 and clean after/before ratios to 1.01365 and 1.01360. Explicit
current/next stage indexing restored 52 registers and measured 0.99999 on both
rows at 45 rounds, with all 43 correctness cases passing.

## Verification

Compare the final SASS stream, including issue order, plus registers, stack, and
spill against the hand-rolled schedule, then measure the affected rows before
keeping the cursor.
