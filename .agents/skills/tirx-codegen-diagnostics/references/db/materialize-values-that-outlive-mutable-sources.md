# Materialize values that outlive mutable sources

**Symptoms:** `tracing_alias_re_evaluation`, `prefetch_overwrite`, `predicate_changes_after_mutation`, `data_dependent_correctness_failure`

## Symptom

A data-dependent numerical failure where a value read later differs from the
value that was computed earlier: a prefetched word reads as its successor, or a
predicate that was true at its definition is false at its use.

## What to change

In the tracing DSL a plain Python expression alias is not a value snapshot. If a
later use occurs after an operand, local buffer element, or referenced scalar
changes, tracing can reproduce the expression with the new value. Take the
snapshot into a scalar local.

```python
# Scalar computation or predicate: K.assign fixes the value at this point.
needs_rescale = K.local_scalar("bool")
K.assign(needs_rescale, m_new > m_prev)

# Bit-preserving register copy of a word that a later prefetch will overwrite.
saved = K.local_scalar("uint64")
K.ptx.mov.b64(saved, words[0], words[1])
```

## Rationale

Two independent failures exposed the boundary. A sparse-decode prefetch alias
was reread after the source register had received the next FP8 word; an explicit
`mov.b64` snapshot reduced the representative mismatch from 46% / 0.355 max-abs
to 10.3% / 0.00891 before the separate stride fix, and the final 15-case matrix
passed.

In a sparse-prefill online-softmax loop, a warp-wide rescale predicate was first
true, then implicitly reevaluated after `mi` changed and became false, so the
old TMEM accumulator was never scaled. Materializing that boolean once made both
stable failing configurations and all six descriptor-path cases pass.

## Boundary

The hazard is re-evaluation across a mutation, not aliasing itself. An alias
whose operands do not change between definition and last use needs no snapshot.

## Verification

For each alias that crosses a mutation of its operands, check the generated code
at the use site rather than the source, and run the data-dependent
configurations that exercise both sides of the predicate.
