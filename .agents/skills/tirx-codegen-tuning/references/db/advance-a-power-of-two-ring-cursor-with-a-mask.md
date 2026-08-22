# Advance a power-of-two ring cursor with a mask

**Symptoms:** `redundant_modulo`, `excess_integer_math`, `branch_in_hot_loop`, `latency_bound_fixup`

## Symptom

A staged pipeline's stage cursor advances as `(stage + 1) % STAGES` on a signed
integer, and the cursor gates the next iteration's addresses. The stall
breakdown shows a fixed-latency execution dependency the reference does not
carry, with eligible warps and warp-cycles-per-instruction both slightly worse.

## What to change

Where the ring depth is a power of two and the cursor is non-negative by
construction, advance it with a mask.

```python
# before: signed modulo, so the lowering adds the sign-correction fixup.
K.assign(stage, K.truncmod(stage + 1, STAGES))

# after: one instruction, no fixup and no control flow.
K.assign(stage, K.bitwise_and(stage + 1, K.int32(STAGES - 1)))
```

A reference that spells the same advance as a conditional reset is describing
the wrap, not prescribing a branch; do not transcribe it as one.

```python
# rejected: matches the reference's spelling and emits a real branch.
K.assign(stage, K.if_then_else(stage == STAGES - 1, 0, stage + 1))
```

## Rationale

The signed modulo pays the floordiv fixup because the dividend's sign cannot be
proven, and it pays it on the chain that produces the next iteration's shared
addresses. Both alternatives are shorter than it, but they are not equivalent to
each other: the conditional form removes the fixup and adds control flow, and
measured net positive overall while regressing the shape with the fewest loop
iterations, where a branch costs more than the fixup it replaces. The mask has
neither cost and dominated both -- taking the shape count above the gate from 3
to 4 and the worst ratio from 0.9512 to 0.9538, against the conditional form's
0.9471.

This is worth spending a lever on only because of where the cursor sits. The
same arithmetic off the critical path is invisible: an earlier change on this
kernel removed a recomputed swizzle chain from a hot loop and measured
neutral-to-negative across four shapes, because that expression fed no
address that anything waited on.

## Boundary

Non-power-of-two depths keep a real modulo; cast the dividend unsigned instead
so at least the fixup goes. The preference for a mask over a select is specific
to a value that is a wrap of a counter -- where a condition genuinely selects
between two different values, the select is the right lowering.

## Verification

Confirm the fixup opcodes and any branch are both gone from the cursor's
advance, then measure the low-trip-count shape specifically: it is the one that
separates the conditional form from the mask, and a guard set without it will
rank them equal.
