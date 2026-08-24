# Clamp with min/max and keep a filter a predicate

**Symptoms:** `instruction_count_bloat`, `excess_guard_math`, `slow_epilogue`

## Symptom

An activation's clamps lower to a `setp` and a `selp` per bound, and its
gradient filter materialises a 1.0/0.0 mask that is then multiplied in -- two
instructions per element where one would do, in a fully unrolled epilogue loop.

## What to change

Express a bound as `min.f32` / `max.f32`, and keep a filter as a predicate that
selects zero at the point of use instead of a mask that is built and multiplied.

```python
# before: two instructions per bound, and a materialised mask per element.
K.ptx["setp.le.f32"](p, value, K.float32(hi))
K.ptx["selp.f32"](clamped, value, K.float32(hi), p)
mask = _one_or_zero(in_range)
ops["product"](grad, grad, mask)

# after: one instruction per bound, and a select at the consumer.
K.ptx["min.f32"](clamped, value, K.float32(hi))
keep = K.local_scalar("bool")
K.assign(keep, K.cast(in_range, "bool"))
K.ptx["selp.f32"](grad, grad, K.float32(0.0), keep)
```

## Rationale

Both forms compute the same number, and in a hot unrolled loop the payoff is far
superlinear in the instruction count: a 2.7% static reduction from the clamps
alone bought 8.6% on the family that runs them. The filter rewrite removed less
machine code than expected -- ptxas was already folding most of the mask
multiply -- but still dropped registers from 241 to 233 and moved both measured
rows again, because a mask has to be materialised into a register the select
does not need.

## Boundary

`min.f32` and `max.f32` differ from `setp` plus `selp` only on a NaN input, so
this diverges from a reference that writes the compare-and-select form. Justify
that the operand cannot be NaN -- a matrix element that has already survived
quantization cannot be -- before making the substitution, and leave the
reference's spelling alone where it can be.

## Verification

Count the compare and select instructions per element on both sides, and check
registers as well: the filter rewrite shows up there more than in the
instruction total.
