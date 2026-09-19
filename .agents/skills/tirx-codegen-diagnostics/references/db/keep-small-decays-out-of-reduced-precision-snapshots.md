# Keep small decays out of reduced precision snapshots

**Symptoms:** `nonfinite_outputs`, `underflow`, `reciprocal_overflow`, `denormal_mismatch`

## Symptom

A backward state contribution is finite algebraically but becomes zero, infinity,
or NaN after storing a small exponential in bf16 and recovering its inverse.
Ordinary short-range inputs pass while stronger monotone decay fails.

## What to change

Preserve the unscaled state gradient when its decay would make the reduced
precision snapshot unsafe. Apply the bounded exponent difference to the other
operand before contraction. Keep the decayed FP32 state separately when the
recurrence itself needs it.

```python
# before: narrowing and inversion happen before the factors can cancel.
snapshot = bf16(state_gradient * exp2(g_end))
operand = bf16(key * reciprocal(bf16(exp2(g_i))))

# after: for monotone nonpositive gates, g_end - g_i is nonpositive.
snapshot = bf16(state_gradient)
operand = bf16(key * exp2(g_end - g_i))
```

Audit every consumer of the snapshot when changing its scale, including gate
gradients, state-gradient publication, and the token epilogue. A real token whose
cached exponential underflows is not padding: mask with the token-valid predicate
instead of replacing every zero exponential with one.

## Rationale

On measured 64-token chunks, a constant base-2 increment of -2 made all six
outputs of one fused path nonfinite. A grouped-head path produced query-gradient
values around 10^36 and nonfinite values in its other five outputs. The reference
outputs were finite. Moving the state scale and also repairing the intra-chunk
ratios passed constant increments through -16 and mixed-channel and abrupt-reset
profiles under the original tolerances. The strongest constant case required
care over a one-token trailing chunk as well as full chunks.

A producer/consumer split also exposed a rounded-predicate mismatch: one half
of a packed pair selected snapshot scaling from the raw log decay, while the
other half and the consumer classified its bf16 exponential. Channels near the
boundary produced a state-gradient RMS ratio of 0.01031 against a 0.008 limit.
Using the same rounded-cache predicate for both pair members and the snapshot
reduced that ratio to 0.00208 on the mixed-boundary case.

## Boundary

This only fixes the state contraction. An intra-chunk product factored as
exp2(g_i) times exp2(-g_j) has the same range problem and needs its own bounded
evaluation. Raising the snapshot precision alone does not repair an overflowing
inverse. A conditional fast path must use a numerical bound valid for both full
and partial chunks, not a full-chunk assumption. Every producer and consumer
must classify the same representation; mathematically equivalent comparisons
can disagree after conversion to bf16.

The bounded-difference argument assumes monotone finite cumulative gates. It
does not justify changing the input contract or suppressing nonfinite input.

## Verification

Exercise both sides of the selected range boundary, mixed channels in one warp,
masked tails including one token, repeated launches with poisoned outputs, and
an abrupt large decay followed by zero increments. Compare every state, token,
and gate-gradient output. Measure normal-range performance separately: the first
correct scalar fallback approximately doubled three normal-range workloads
because it changed code size, register pressure, and memory lifetimes even when
the fallback did not execute.
