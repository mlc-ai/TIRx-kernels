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

Balancing both factors by an exact power of two requires an operand-amplitude
bound as well as a gate-range bound. A gate-only selector is insufficient.
In one tested 64-token formulation,
adding 80 to cached gate exponents for chunk-end logs in [-176, -96) moved the
inverse away from underflow without changing the contractions algebraically.
The incoming state operand was scaled down by 2^-80 and the backward snapshot
scaled up by 2^80. The separate FP32 recurrence stayed unscaled; the fused
recurrence explicitly undid its previous chunk's scale before reuse and final
publication. Omitting either transition changes the derivative.
That experiment is not a generally safe replacement for bounded differences:
large upstream gradients overflowed in its amplified contractions.

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

After removing analytically cancelling diagonal terms from the gate derivative,
the balanced tensor-core experiment passed constant increments through -2.75,
mixed channels, partial chunks, and transitions between mild, balanced, and
fallback chunks under unchanged tolerances. On one grouped-head shape, a
synthetic FLA-initialization distribution with Gaussian projection noise of
standard deviation 0.5 took about 158 us rather than the scalar fallback's
683 us. At noise 2, it still took about 499 us: a few strong channels spread
over warps can keep many scalar loops active. These synthetic distributions
do not establish a real training trigger rate.

An amplitude stress test invalidated unconditional use of that balancing rule.
With constant base-2 increments of -2 and upstream gradients multiplied by
2^64, the balanced fused path produced nonfinite key, value, beta, gate, and
initial-state gradients; the grouped-head path produced nonfinite key and gate
gradients. The bounded-difference version and reference kept all six outputs
finite, with normalized RMS differences around 0.0008-0.0020 for the former.
Those RMS values were computed in FP64: squaring such large gradients in FP32
can overflow the error metric itself. The original absolute-error assertion
also failed at this amplitude, so this is finite-range and relative-error
evidence, not a full correctness pass at unchanged absolute tolerances.

An amplitude guard must protect accuracy as well as finite range. A later
experiment admitted inputs with magnitudes up to 8 and used a shift and inverse
bound of 80. With value inputs, upstream gradients, and both endpoint states
set to signed magnitude 8, two fused cases failed individual beta-gradient
comparisons at base-2 increments -1.25 and -2.5. The bounded baseline passed
both; every output of the experiment was finite. The worst failing differences
were 0.8182 and 0.1504. This limits that guard policy even though ordinary gate
sweeps and much larger, correctly rejected upstream amplitudes had passed.

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
It bounds exponent overflow, not the error from underflowing a decay before
multiplying by a large operand. A measured unit-basis query/key case with zero
beta, initial-state and upstream magnitude 2^60, and a first-token base-2
increment of -128 followed by zeros returned all-zero gradients in both fused
and grouped paths. Over 128 tokens the FP64 recurrence's largest gate gradient
was about 0.04419; the existing reduced-precision reference also returned zero.
Changing only that first increment to -64 retained the nonzero derivatives,
with gate-gradient RMS differences from FP64 below 0.00051 on both shapes.
An explicit `ex2.approx.ftz.f32` can therefore discard a contribution whose
final product is representable. Exact-zero recurrence termination preserves
the rounded decay calculation, but does not prove accuracy against the
unfactored expression for unbounded operand amplitudes. This limits the
fallback's range claim; it does not establish a replacement scaling policy.

A later GPU component experiment isolated the exponential from the surrounding
FP32 recurrence. Below a base-2 increment of -126, it evaluated two half-sized
exponents with `ex2.approx.f32`, multiplied the state by one factor and then the
other, and did the same for the adjoint. Both the exponentials and their
consuming multiplies retained subnormals. With unit-basis inputs, first-step
increments -128/-160/-256 and endpoint/upstream magnitudes 2^60/2^80/2^120,
the unsplit control returned zero gate gradients. The split variant recovered
approximately 0.04419/11.31372/0.00017263 over 128 tokens. All six gradients
passed the RMS limits against FP64 on both full and partial-chunk shapes;
valid input-matrix entries were finite and repeated outputs were byte-identical.

The generated forward/backward PTX changed from one FTZ exponential each to two
non-FTZ exponentials each, with no FTZ multiply or FMA. Register counts changed
56->64 and 96->95, without local spills. This is a range-repair component,
not a qualified replacement kernel or a performance result. Removing FTZ from
the exponential alone cannot represent 2^-160 in FP32, and retaining a
subnormal exponential is ineffective if its consumer flushes that operand.
The final stored state, adjoint, partial sums and each split factor still need
range checks; two factors do not provide arbitrary-finite-input safety.

A later whole-kernel experiment confirmed that this limitation also appears
before an explicit zero. Replacing selected FTZ exponentials with non-FTZ forms
and splitting one post-reduction chunk factor recovered query and gate gradients
for a first-token -128 decrement with endpoint and upstream magnitudes 2^60.
It passed 39 registered comparisons. Moving the same decrement inside a chunk,
however, left every beta gradient before the decrement at zero: depending on
position, beta-gradient RMS error against a token FP64 recurrence was about
0.15-0.99. At a chunk boundary, the candidate's beta and gate gradients were
about 6.0% high. The prepared tensor operand `q * exp2(-128) / sqrt(128)` is
about 2.83 bf16 minimum-subnormal units and rounds to three units, explaining
the 6.066% error. Changing the adjacent-decay exponential to a non-FTZ form, or
forming that adjacent decay by squaring a half exponent, did not change the
outputs. The information had already been narrowed before the large state or
adjoint factor could restore a normal result.

Offline GB200 code generation for that incomplete candidate kept the same 168
registers and 88/96-byte stacks as the baseline. Non-FTZ repair sequences grew
the grouped-head specialization by 388 static instructions (including 130 more
FMUL and 129 more FSEL instructions) and the fused specialization by 110 static
instructions (34 more FMUL and 34 more FSEL instructions). These are code-size
observations, not GPU timing evidence. With first-token decrements -160 and
-256 and matching magnitudes 2^80 and 2^120, the whole candidate produced
nonfinite query, key, beta, and gate gradients even though the FP64 recurrence
was finite. Do not promote a recurrence-only non-FTZ change from a component
test. Keep the decay factored until it has met the large operand, or add an
explicitly bounded direct cross term; then test interior positions and chunk
boundaries as separate cases.

Amplitude bounds alone also do not preserve the bounded path's accuracy.
With normalized aligned q/k, beta 0.75, endpoint-state magnitude 1, input and
upstream magnitude 4, and constant base-2 increments -2, guarded balancing
increased gate-gradient RMS error against the existing reference from about
0.005 to 0.090 and 0.055 on two specializations. An independent FP64 token
recurrence confirmed errors of 0.098 and 0.062 versus approximately 0.011 and
0.008 for the bounded path. Both bounded cases passed unchanged checks, while
the balanced ones failed despite finite outputs. This rejects that admission
policy; it does not justify adding a beta threshold without a new accuracy and
acceptance-rate study. Saved intermediate rounding and cancellation matter
separately from exponent range.
Higher-precision accumulation cannot recover information already lost in a
saved triangular inverse. In an isolated aligned-input recurrence with beta
0.75, rounding only the exact inverse to bf16 produced initial-state-gradient
relative errors of about 0.379, 0.102, and 0.132 at base-2 increments -0.03,
-0.125, and -0.25; all subsequent arithmetic used FP64. The mildest case
subtracted terms near 1.09679 and 1.08665 to obtain about 0.01014. Retaining a
bf16 high/low pair for that inverse reduced the isolated errors below 0.0015,
but does not qualify a kernel change: the low part must be recovered from the
original inputs, and other narrowed operands remain separate error sources.
Power-of-two balancing has a finite range too. Check both the largest scaled
operand and the smallest state snapshot; increasing the shift indefinitely
just moves overflow or underflow to another consumer. A shift chosen solely
from the gate range can overflow a later gradient contraction while the
unfactored result and bounded fallback remain finite. Keep such a candidate
experimental until amplitude-aware dispatch or a bounded formulation has been
validated; ordinary-amplitude gate sweeps do not establish this safety property.

The same rule applies even without an exponential. In a measured KDA backward
case with zero gates, the FP32 endpoint states were narrowed for BF16 tensor
operands before meeting an oppositely scaled input. Values just below `2^-134`
rounded to zero; FP32 values from about `3.3962e38` through FP32 max rounded to
BF16 infinity. Unit-basis tests kept the unfactored FP64 gradients finite. The
low initial-state pair lost essentially all of material `dq/db/dg`, and the low
terminal-adjoint pair lost all of `dk`; the high pairs produced nonfinite
outputs. Both persistent specializations reproduced the cliffs, and the
reduced-precision reference shared several of them. A guard that only examines
decay range cannot protect this case. When an FP32 interface value is narrowed
for an MMA, include the eventual counterpart's exponent in the range proof or
delay the narrowing until after their scales have met.

BF16 interface operands can hit the same cliff at the normal/subnormal
boundary. With zero gates and orthogonal unit-basis query/key vectors,
`v=2^-127` and `do=2^127` have a normal, finite product. A fused path lost all
material query and key gradients, while a grouped-head path lost about 74% RMS;
changing `v` to the BF16 minimum normal `2^-126` restored those gradients. A
separate case with `beta=2^-127`, `v=2^127`, and `do=2^-16` lost both gradients
in the fused path and most or all of them in the grouped path; `beta=2^-126`
restored the material terms. These are operand-range failures at zero decay,
so a gate-only guard cannot classify them.

Generated PTX exposed 32 scalar `mul.ftz.f32` instructions for one plain-source
`v * beta` preparation loop. Replacing its four source variants with explicit
`mul.rn.f32x2` removed all 32 FTZ multiplies. That scratch change repaired the
grouped-head beta-boundary case, but the fused path still lost about 70% RMS of
the query gradient after a chunk transition. Its separate `beta * key` value
had already been narrowed to a BF16 subnormal tensor-core operand. Removing FTZ
from one scalar product therefore does not close the range proof; enumerate
every factorization and every reduced-precision handoff before qualifying the
change.

Normal inputs can create the same reduced-precision hazard after a scale. In a
normalized query with one small component, query components `2^-127`, `2^-126`,
and `2^-125` were multiplied by `1/sqrt(128)` before BF16 contraction and paired
with large upstream gradients. The resulting beta-gradient RMS errors against
FP64 were about 6.07%, 2.77%, and 1.65% on one fused path even though the last
two query components are normal BF16 values. Apply range checks to the actual
prepared operand, including fixed scales, rather than only to the public input.

A downscale applied after a contraction does not protect that contraction's
range. With orthogonal normalized query/key vectors, zero gates and endpoint
states, `v=bf16_max`, and `do=0.25`, the exact FP64 query/key gradients stayed
finite at about `8.47e37` and `1.69e38` on two KDA layouts. Both measured GPU
implementations produced infinity or NaN because an unscaled contraction
overflowed before the later `1/sqrt(128)` factor. Apply a known downscale before
the overflow-prone reduction when algebra permits, and retest rounding-sensitive
cases; checking the final mathematical range is insufficient.

Cancellation does not weaken that requirement. In a second measured case, an
initial-state row contained 64 copies of `+2^116` followed by 64 copies of
`-2^116`; a single upstream row contained `2^7`, while normalized query/key
vectors selected independent basis rows. The exact query and gate gradients
were zero and every exact output was finite. Two GPU layouts and their reference
both returned infinity for those gradients. Interleaving the same positive and
negative components returned exact zero, and reducing the state magnitude to
`2^110` also passed. A reduction can therefore overflow a sign-clustered partial
sum even when the full dot product cancels exactly. Include partial-sum bounds
and adversarial component order in range tests; a bound on the final dot product
does not establish accumulator safety.

Analytically cancelling terms are also a range decision. A mild-path KDA
formulation retained two query/key diagonal contributions to the gate gradient
and cancelled them numerically. With exactly unit-norm, orthogonal BF16 vectors,
zero endpoint states, `v=bf16_max`, and `do=2^-8`, the exact and reference gate
gradient was zero while the mild path returned about `4.00e32`; all outputs
were finite. The enhanced formulation removed those diagonal terms from the
gate gradient and restored them only to query/key gradients; on the same input
it returned exact zero. Preserve the removed term for every output where it
does not cancel, and measure the barriers, storage, and live ranges needed to
restore it rather than assuming algebraic simplification is free.

An amplitude sweep on the same exact-zero gate derivative found the residual
scaled almost linearly with `v*do`: products 1, 16, 256, 4096, and 16384 gave
maximum spurious gradients about `9.44e-6`, `1.51e-4`, `0.00242`, `0.0387`, and
`0.155`. Thus an unbounded relative metric reports failure even at unit scale,
while the existing `0.1` elementwise tolerance becomes material only around a
product of `1.06e4`. Report both absolute and relative errors for an analytic
zero, and use the absolute crossover when deciding whether hot-path cost is
justified.

## Verification

Exercise both sides of the selected range boundary, mixed channels in one warp,
masked tails including one token, repeated launches with poisoned outputs, and
an abrupt large decay followed by zero increments. Compare every state, token,
and gate-gradient output. Measure normal-range performance separately: the first
correct scalar fallback approximately doubled three normal-range workloads
because it changed code size, register pressure, and memory lifetimes even when
the fallback did not execute.
