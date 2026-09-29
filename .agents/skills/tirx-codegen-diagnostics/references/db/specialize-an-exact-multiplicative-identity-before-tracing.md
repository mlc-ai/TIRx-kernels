# Specialize an exact multiplicative identity before tracing

**Symptoms:** `redundant_identity_math`, `slow_epilogue`, `dispatch_specific_deficit`, `instruction_count_gap`

## Symptom

A scalar is fixed by the compiled specialization to exactly `1.0`, but the
kernel still receives it through a runtime pointer. Generated code loads and
round-trips the scalar through the output format, then multiplies every output
pair by the widened value. The runtime ABI prevents ordinary constant folding,
so one otherwise identical specialization retains an avoidable epilogue chain.

## What to change

Derive the exact-identity fact from the static specialization inputs before
tracing. Omit the load, conversion, and multiply only in that compiled branch;
retain the pointer argument and the complete generic path.

```python
# before: a statically fixed scale remains opaque behind the runtime pointer.
scale = _load_and_round_scale(scale_ptr, OUTPUT_DTYPE)
for pair in range(NUM_PAIRS):
    values[pair] = _mul_f32x2(values[pair], scale)

# after: tracing emits no scale chain for the exact-identity specialization.
scale_is_one = float(static_scale) == 1.0
if not scale_is_one:
    scale = _load_and_round_scale(scale_ptr, OUTPUT_DTYPE)
    for pair in range(NUM_PAIRS):
        values[pair] = _mul_f32x2(values[pair], scale)
```

## Rationale

On one SM107 BF16 epilogue, the identity specialization removed one PTX
`ld.global.f32`, two scalar conversion instructions, and sixteen
`mul.f32x2`; final SASS removed all sixteen corresponding `FMUL2`
instructions. Target time fell from 26.0168 to 25.3485 us, a 1.0264x speedup,
while the reference/target ratio moved from 0.9893 to 1.0170. The retained
implementation passed an exact identity-scale source/oracle comparison with
zero differing bits, output guards, and a deterministic rerun, then passed the
complete seven-case correctness and five-shape performance matrices.

## Boundary

The scalar must be compile-time fixed for that exact artifact and exactly the
multiplicative identity under the reference's load and rounding semantics.
Do not infer identity merely because a nearby value rounds to one unless that
rounding is itself part of the frozen contract. Keep the generic path for every
other scalar and preserve the public ABI when callers still pass the pointer.

Skipping a floating-point instruction can change NaN quieting, denormal/FTZ
behavior, or exception semantics even for a mathematical identity. Require
bitwise evidence over the supported input domain; use tolerance only when the
kernel's existing correctness contract already permits it.

## Verification

Compare PTX and final SASS for identity and non-identity specializations. The
identity artifact should lose the scalar load, round-trip conversions, and
pairwise multiplies while the generic artifact retains them. Run exact
correctness for both branches, including exceptional values when supported,
then run the affected and complete performance matrices.
