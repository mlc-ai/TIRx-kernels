# Hoist warp-converging instructions out of the loop

**Symptoms:** `excess_control_instructions`, `branch_in_hot_loop`, `vectorized_uniform_math`, `instruction_count_gap`

## Symptom

Far more executed elections, votes, or shuffles than the reference. One mainloop
re-elected twice per K block, executing 495,036 elections against the
reference's 36,284, while the reference executed 1.1M more uniform-register
moves -- the predicate it was holding instead.

## What to change

Elect (or vote, or shuffle) once outside the loop and consume the resulting
predicate on each instruction inside it.

```python
# before: re-executes elect.sync at every use site inside the loop.
for kphase in range(8):
    T.evaluate(T.ptx[_MMA_CHAIN](..., pred=T.cuda.elect_sync()))

# after: the election is loop-invariant, so hoist it above the mainloop and
# carry the predicate, which is what the reference's compiler emits.
mma_elected: T.uint32
mma_elected_lane: T.uint32
T.ptx.elect_sync(mma_elected_lane, mma_elected, T.uint32(0xFFFFFFFF))
while mma_k < mma_k_rounded:
    ...
    T.evaluate(T.ptx[_MMA_CHAIN](..., pred=mma_elected == T.uint32(1)))
```

## Rationale

A lane election, vote or shuffle is a warp-converging instruction with real
issue cost, and a reference that calls one at each use site does not necessarily
execute it there: nvcc hoists a loop-invariant election into a single uniform
predicate carried across the loop. Transcribing the call at every use is
faithful to the text and diverges from the compiled form. The hoist took the
kernel from 809.5 to 806.1 us and from 62.27M to 61.68M instructions.

## Boundary

Hoisting is not deleting. Where a reference's election or broadcast is the only
evidence a later guard has that its condition is warp-uniform, removing it costs
more than the instruction saves; lift it to the enclosing region and keep the
predicate live instead of dropping the proof.

## Verification

Count executed warp-converging instructions against the reference and confirm
the loop body consumes a live predicate.
