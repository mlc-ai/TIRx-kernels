# Compensate rounded products before sensitive contractions

**Symptoms:** `gradient_error`, `catastrophic_cancellation`, `precision_mismatch`

## Symptom

A tensor-core fast path narrows a product of bf16 inputs and FP32 scaling to
bf16 before contraction. A bounded scalar baseline keeps the product in FP32.
All inputs and outputs are finite, but subtracting related contractions
amplifies the additional operand-rounding error.

## What to change

Identify the narrowed product responsible for the discrepancy. Retain its
original high part and contract a bf16 residual in a second tensor operation:

```python
full = multiply_in_original_fp32_order(inputs)
high = cast_bf16(full)
low = cast_bf16(full - cast_f32(high))
accumulate_tensor_product(accumulator, high, other_operand)
accumulate_tensor_product(accumulator, low, other_operand)
```

Use the actual stored high part when the existing path includes masking or
other preparation. Preserve the operation order used to compute the full
product. Ensure output consumers wait for the residual contraction as well
as the original contraction.

## Rationale

In two measured backward families, aligned normalized query/key inputs with
beta0.75 and base-2 gate increments -2 passed a bounded baseline but failed a
new factored fast path. Compensating beta*key*gate narrowed operands reduced
gate-gradient normalized RMS error from0.08963/0.05476 to0.009636/0.008720.
Compensating query*gate*scale additionally reduced it to0.004957/0.004789,
near the baseline's0.00496/0.00478. Independent FP64 recurrence supported the
improvement.

A noninteger -1.9 gate still failed after those two corrections: the first
family's gate-gradient RMS error was0.04945. Compensating key/gate in both
remaining contractions reduced it to0.005402, near baseline0.00524. The final
variant passed29 registered cases and repeated poisoned-output launches.
The generated code added four contraction chains, or16 static matrix issue
instructions per family, without increasing their40/96-byte local stacks.
A paired six-case comparison with the bounded baseline passed unchanged
checks. Synthetic initialization inputs fell from6277 to1610 us and from683
to199 us, while ordinary controls increased3.3-4.0% and fully strong controls
increased5.6-6.4%. This is evidence for recovering some fast-path performance,
not general acceptance or a measured training distribution.

Moving residual arithmetic under the channel predicate, while still storing
zero residuals for scalar-fallback channels, preserved every output bitwise on
six paired cases. Fully strong timings improved4.45%/2.65% relative to the
unconditional-residual version; mixed initialization improved1.49%/2.83%.
The shared publication and reuse protocol remained unchanged. Avoiding unused
arithmetic was useful even though the tensor consumer still read the zero
residual tile.

## Boundary

Compensating the product of already-rounded inputs does not recover their
earlier precision loss. A later noninteger-gate sweep found a new failure at
aligned normalized query/key, beta0.5, and base-2 increment -0.75: key-gradient
RMS changed from0.007871 to0.008096 against an unchanged0.008 limit. Independent
FP64 error changed from0.010197 to0.010332. Another specialization's beta0.75,
increment-0.75 case already failed, but its FP64 key error worsened from0.02282
to0.03073. Thus the compensated variant's registered successes do not establish
general acceptance. Inspect the cached gate's own narrowing before treating
product residuals as a complete replacement for bounded scalar differences.

This does not repair rounded interaction matrices, pre-existing conditioning
errors, or exponential underflow/overflow. A beta1 cancellation case still
failed the original checker after the optimization's additional error was
largely removed. Do not relabel such failures as passes.

A subsequent FP32 arithmetic model recovered a saved triangular inverse with
one residual update and represented the inverse and downstream operands as bf16
high/low pairs, using three products per matrix contraction. On three correlated
beta0.75 profiles, isolated initial-state-gradient errors fell below0.0018.
The beta1 near-zero-gradient case still had an error ratio of about0.0266,
while full FP32 operators and intermediates reached about0.000053. These are
single-head/channel arithmetic-model results, not full-kernel correctness or
GPU timings. Two-part compensation is not a universal precision guarantee,
and the extra contractions cannot be treated as free on the ordinary path.

A component ablation on those same ten inputs found that adding the omitted
low-times-low product did not improve the beta1 case. Keeping a third bf16
component and summing six products instead reduced its state-gradient error
ratio from about0.0266 to0.000053. This identifies lost operand bits, not just a
missing cross product, as a limiting factor in that case. It is still a CPU
arithmetic model: separate FP32 products do not establish tensor-accumulator
rounding, full-gradient correctness, or GPU cost.

Additional aligned beta1 inputs with weak decay invalidate extrapolating even
that result. One residual inverse update left about0.0008 absolute error in a
state gradient whose FP64 magnitude was about0.00000654; two updates removed
most of the error, but the resulting FP32 chunk formulation still had percent-
level relative error. Avoid promoting a fixed refinement count or extra
mantissa components as a universal conditioning repair. Compare the chunk
form with an independently evaluated token recurrence before adding more
tensor contractions.


A later full-GPU input ablation reconstructed the causal saved operators in
FP64, then rounded them back to the unchanged bf16 input format. On a correlated
weak-decay case, inverse RMS error improved from about0.000698 to0.000214, but
replacing only that inverse worsened the initial-state-gradient error ratio
from0.336 to0.495. The gate-gradient ratio also rose from0.0697 to0.0876. Another
family's state-gradient ratio rose from0.0202 to0.1885. Ten inputs and four
saved-matrix combinations covered every output; intervention repeats were
byte-identical, and restoring the saved inputs restored all original output
bytes. Default controls remained within their original checks, while related
correlation failures remained failures. This does not qualify replacement
saved inputs as a repair: lowering matrix-level error while retaining bf16
storage need not lower derivative error after sensitive downstream
contractions. Preserve residual information through consumers and evaluate the
complete gradients rather than optimizing the saved operator in isolation.

GPU operand captures can separate accumulated operand error from the final
FP32 tensor accumulation. In three native-path inputs, a diagnostic copy
preserved all six output byte arrays while recording pre-narrowed W/dv and
packed operands. FP64 reconstruction from those recorded operands differed
from the actual final contraction by at most about1.1e-7, while a correlated
state-gradient error was about0.0033. Removing just the last W/dv narrowings
in a counterfactual did not repair it: an initial error ratio near0.326 moved
to0.381 with only W retained,0.158 with only dv retained, and0.213 with both.
For the correlated full chunks, an exact chunk operator given the actual,
already-inaccurate incoming adjoint recovered the token reference to below
6e-16 absolute error. Do not attribute such cases primarily to final FP32
accumulation or inter-chunk snapshot storage. Trace the prepared operands and
saved operator through the cancellation; a counterfactual is not a qualified
GPU correction.

After analytically reformulating a state adjoint to avoid its query-update
cancellation, an independent GPU component compared three BF16 products
(`low*high`, `high*low`, then `high*high` in the accumulator) with `tf32x3`.
Both operands were split from FP32, and operators stayed FP32 between kernels.
The recurrence's static `tcgen05.mma` sites fell from120 to60; its two
specializations went from128/136-byte stacks and27/30 or28/31 local load/store
sites to zero stack and no local traffic. A separate operator producer kept
zero or16 stack bytes. Twenty-five bounded-gate inputs passed independent
FP64 state checks, including near-zero cancellation cases: the worst initial
state-gradient RMS ratio was0.0003221, and340 per-chunk/head snapshot checks
had a maximum ratio0.002974. Repeated poisoned launches were byte-identical.
In five alternating same-device rounds, the two-kernel component improved
25.8–26.3% against its FP32 `tf32x3` form. It still cost5.07–5.59 times an
original *complete* backward operation while producing only state adjoints.
This qualifies the contraction substitution on those inputs, not a full
gradient repair, an exponent-range repair, or acceptable full-operation cost.
The changed formula and two residual inverse updates are prerequisites of
that experiment; the result does not remove the cancellation failures of
unmodified chunk formulas discussed above.

The same substitution was subsequently tested in a complete projected-query,
peeled-tail backward prototype. Twenty GPU inputs covered grouped and ungrouped
heads, exact and near correlation, and one-token and chunk-boundary tails.
All six outputs passed an independent token FP64 oracle and repeated poisoned
launches. The maximum RMS ratio was0.001845 for the BF16 value-gradient output;
the other five outputs stayed below0.000323. Forward and reverse state loops
lost their local stacks, while the two gradient specializations' stacks fell
from2000/2008 to1584/1464 bytes. Static `tcgen05.mma` sites halved in all four
matrix kernels. Five same-device alternating rounds measured25.3–25.4% lower
complete-operation latency than the `tf32x3` version, but the result still
cost27.6–30.9 times the original full operation. The gradient phase accounted
for about65% of independently measured phase times. These results extend the
tested numerical scope to complete gradients; they do not qualify the prototype
for production, remove its bounded-exponent requirement, or make residual
contractions free. Preserved FLA comparisons still failed on15 of20 inputs;
the independent-oracle pass must not be reported as passing that comparator.

Replacing just the derivative consumer's state snapshots with exact token
states rounded to the unchanged BF16 format is not equivalent to this repair.
In five grouped-head inputs, a diagnostic read-descriptor override preserved
all baseline output bytes when disabled, every recurrence-produced snapshot
when enabled, and every output byte after restoration. On a correlated
beta0.75 weak-decay input, ideal forward and reverse snapshots left the
beta-gradient RMS ratio near0.132 and moved the gate-gradient ratio from0.0697
to0.0754. Improving the snapshots alone did not fix the downstream contractions.
That experiment retains BF16 storage; it does not assess an FP32 snapshot
consumer or justify repairing only the recurrence while reusing every old
derivative contraction.

Residual storage reused from an earlier phase must account for every alias.
An invalid first implementation overwrote a reduction buffer still being read
by another compute warp, and the loader's next-chunk prefetch could overwrite
the residual before its tensor consumer. Joining the compute readers and
retaining the existing reuse release until final tensor completion removed
those large errors. Merely waiting for the original tensor consumer was
insufficient.

Late derivative preparation must also account for mixed accumulators. A common
exponent shift canceled between the two intra operands, but two accumulators
already contained unshifted state contractions. Applying the shifted final
factor amplified those state terms by2^80: four previously passing cases failed
with errors around1e23 even though both state-dependent outputs stayed bitwise
equal. Scaling the initial FP32 mixed accumulators by the inverse power of two,
after their producers completed and before the intra additions, restored the
24-case baseline pass/fail pattern with no new RMS threshold crossing. All33
registered cases and27 long-sequence/dispatch-boundary cases then passed.

The TMEM read/modify/write used full-warp operations, explicit load/store waits,
and the existing publication fence; recurrence operands and snapshots stayed
unchanged. Twelve mixed-path cases passed Racecheck with zero hazards. On a
paired synthetic initialization sweep, the complete implementation reduced
latency57.7-70.3% in one family and52.2-56.5% in the other versus its bounded
baseline. These results qualify the tested composition, not arbitrary shifts:
underflow of a rescaled accumulator and the range of both operand factors still
need explicit bounds, and the pre-existing conditioning failures remain.

A later native-epilogue experiment exposed a stricter cancellation boundary.
Two gate-gradient terms cancel analytically, but their tensor operands were
separately rounded as `bf16(k/eg)` and `bf16(q*eg*scale)`. Correcting with the
mathematical FP32 factors did not remove the residual. Reconstructing each
term from the exact rounded BF16 diagonal and the exact factorization used by
its own MMA made eight fused/grouped high-amplitude cases return exact zero.
Contracting a multiply-plus-subtract into FMA was not equivalent: it left a
spurious result as large as about1.61e28 on the largest finite case. Preserve
the original rounding boundaries when the goal is cancellation of already
rounded tensor terms; algebraic equivalence and fewer instructions are not
sufficient.

Refining an upstream FP32 reciprocal also did not repair this boundary. One
Newton step after `rcp.approx` was tested across three native gate values and
both fused and grouped kernel families. Of36 FP64 RMS output metrics, only11
changed at all, and the largest relative metric change was0.0003210609%.
The refinement was effectively erased by the later BF16 operand or snapshot
rounding. When a sensitive contraction consumes reduced-precision values,
first repair or reproduce that exact rounding boundary; improving an upstream
approximation alone may add instructions without changing the consumed bits.

A magnitude gate for that exact correction needs its own boundary experiment.
On one analytical zero-gradient family, the rounded BF16 diagonal increased
from 128 to 2,097,152 while the uncorrected residual increased from 9.44e-6 to
0.154663 and crossed an absolute tolerance of 0.1 only at the largest sampled
point. A threshold of 5 corrected a diagonal of 8 to exact zero and deliberately
left a diagonal of 4 unchanged at a 2.95e-7 residual. Four ordinary synthetic
shapes had maxima from 3.89 to 4.91. This establishes a wide measured margin
for that factorization and workload set; it is not a proof for arbitrary finite
operands or a reason to infer the trigger rate of real training data.

Adjacent rounded BF16 diagonal values can be packed into one shared `uint32`
and expanded at each consumer without changing the operands being corrected.
Four directed native-mega cancellation cases, including a BF16-maximum sparse
case, then returned exact-zero gate gradients and repeated exactly. This form
reduced stack traffic but missed the operation-level performance gate: the
short shape measured +4.90% once and +6.34% on confirmation, or about +5.61%
over all ten paired samples. Re-run any result close to the acceptance boundary
before qualifying it. Replacing masks and shifts with `PRMT` kept the same
stack/local traffic and increased the static permutation-plus-logic instruction
count, so byte permutation was not a cheaper expansion for this pair layout.

## Verification

Compare the unmodified baseline, candidate, and an independent high-precision
oracle on correlated as well as random inputs. Include non-power-of-two
gates and partial chunks. Preserve the original absolute and relative checks;
an overall pre-existing failure can hide a new error in a different output.
Validate shared-buffer lifetimes and repeated launches, inspect generated
resource usage, then measure ordinary and correction-heavy cases separately.
