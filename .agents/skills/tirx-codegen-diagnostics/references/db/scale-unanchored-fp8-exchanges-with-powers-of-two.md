# Scale unanchored FP8 exchanges with powers of two

**Symptoms:** `e4m3_underflow_cliff`, `e4m3_saturation_cliff`, `all_zero_output`, `magnitude_sensitive_error`

## Symptom

An internal tensor is converted directly to E4M3 without a scale. Ordinary
inputs pass, but shrinking the producer silently zeros many values, growing it
silently clamps them, or a long softmax loses low-probability numerator mass
while retaining that mass in its denominator.

## What to change

Apply an exact power-of-two scale before the E4M3 conversion and fold its
reciprocal into an existing downstream multiplier.

```python
# before: the E4M3 window is anchored to an accidental input magnitude.
T.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(q_fp8, q_bf16x2)
score = mma(q_fp8, k_fp8) * score_scale

# after: the packed multiply shifts the useful window without adding a hot
# score-path operation; the reciprocal is folded into the launch scalar.
T.ptx.mul.rn.bf16x2(q_scaled, q_bf16x2, packed_power_of_two)
T.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(q_fp8, q_scaled)
score = mma(q_fp8, k_fp8) * (score_scale / power_of_two)
```

For an E4M3 softmax-probability exchange, add the scale in log2 space and lower
the lazy-rescale threshold by the same amount. This preserves the previous
maximum exchanged probability while moving the underflow boundary down:

```python
p = T.ptx.ex2.approx.ftz.f32(logit_minus_max + log2_p_scale)
lazy_rescale_threshold = old_threshold - log2_p_scale
```

## Rationale

One measured attention path used an 8x Q scale folded into its QK multiplier
and a 16x P scale paired with a lazy-rescale threshold reduction from 8 to 4.
Generated code added one packed BF16 multiply to Q preparation, removed the
half-precision exponential forms, and added no per-score scale operation. Six
range-focused GPU rows passed, including Q standard deviations near 0.003 and
21, 2,047 low-probability keys followed by a late maximum, and exponent ranges
that crossed the rescale boundary. In the final 15-round same-request A/B, the
two production FP8 shapes moved by +0.68% and -1.39%; all ten production rows
stayed within the 5% per-shape gate (worst +1.93%, median -0.34%).

## Boundary

A fixed scale moves a finite E4M3 window; it does not make arbitrary BF16 input
ranges representable. Select it from both low- and high-amplitude tests. Use a
dynamic scale when the contract truly permits a wider range and its
reduction/compensation cost passes the performance gate.

The softmax threshold coupling is mandatory. Raising P without lowering the
maximum permitted logit delta can turn an underflow fix into silent E4M3
saturation. The P scale cancels only when every numerator and denominator term
uses the same factor.

## Verification

Inspect generated code for the intended packed multiply and reciprocal-folded
launch scalar. Test the low and high ends independently, plus a constant-V
softmax with more than 1,024 tail keys and a late maximum. Re-run the full
performance matrix because Q preparation and probability exchange are hot even
when the new scale is algebraically exact.
