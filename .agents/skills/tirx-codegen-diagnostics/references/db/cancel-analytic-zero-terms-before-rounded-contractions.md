# Cancel analytic zero terms before rounded contractions

**Symptoms:** `gradient_error`, `catastrophic_cancellation`, `precision_mismatch`

## Symptom

A derivative is formed by subtracting two rounded contractions. The same large
diagonal term occurs on both sides and should cancel, but its two evaluation
paths leave a residual that dominates a much smaller off-diagonal gradient.
Inputs remain finite and exponent range checks do not explain the failure.

## What to change

Remove the analytically cancelling diagonal before the contractions that feed
the small derivative. Preserve the diagonal separately and add it back only to
outputs whose derivatives genuinely include it.

```python
diagonal = diag(interaction_gradient)
off_diagonal = interaction_gradient - diag_embed(diagonal)
query_off = contract_query(off_diagonal)
key_off = contract_key(off_diagonal)
gate_gradient = query * query_off - key * key_off
query_gradient = query_off + diagonal * key
key_gradient = key_off + diagonal * query
```

The sketch omits other terms and scaling factors. Prove cancellation in the
actual derivative before applying it; two similar-looking terms need not be
mathematically identical.

## Rationale

In two measured 64-token backward families, constant base-2 decay increments
of -1.25 produced gate-gradient normalized RMS errors of 0.02332 and 0.02359,
above an unchanged 0.02 limit. Keeping the original tensor-core contraction
structure but excluding its diagonal reduced those errors to 0.00961 and
0.00953. At -1.5, the errors fell from about 0.023 to 0.00800 and 0.00840.
The corrected tensor-core path passed without switching these inputs to the
scalar range fallback. Both families also passed mild and extreme-decay
controls and repeated poisoned-output launches.

## Boundary

This does not fix underflow or overflow in a separately factored exponential
ratio. State snapshots and intra-chunk ratios can require different range
protection even after diagonal cancellation is repaired.

A more accurate fast path can still be slower. An initial implementation
extracted diagonals with repeated predicated shared stores and added shared
loads to the output epilogue. Its assembled local stack measured 48/104 bytes
across the two families; one ordinary workload moved from about
143 to 156 microseconds. Accuracy alone does not establish the final schedule.

A later schedule extracted the rounded diagonal from the completed shared
tile, scaled it once into an FP32 scratch row, and restored query/key terms
with packed FMA. A forward-only beta buffer became that scratch row after the
phase barrier, so the change needed no new allocation. One grouped-head
workload improved from roughly 155 to 150 us; mild, balanced, and extreme
gate tests retained the same tolerances. This arrangement requires proving
both that the beta readers have finished and that the next chunk cannot
overwrite the diagonal before both epilogues finish.

## Verification

Test the individual output gradients as well as their cancelling combination,
and include uniform strong channels so that a global error norm cannot hide
them behind larger mild-channel gradients. Exercise mixed channels, partial
chunks, and both sides of every range predicate. Inspect the added extraction
and restoration instructions, register lifetimes, and shared-buffer reuse,
then measure ordinary and fallback-heavy inputs separately.
