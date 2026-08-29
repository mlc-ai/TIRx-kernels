# Match mixed-precision FMA operand widths to the reference

**Symptoms:** `instruction_variant_mismatch`, `instruction_count_gap`, `sass_divergence`, `performance_regression`

## Symptom

A recurrence widens stored 16-bit operands to FP32 and feeds packed FP32 FMA,
while the reference keeps the operands in their 16-bit register form and uses a
native mixed-precision FMA with an FP32 accumulator. The port then carries
extra conversions and packed arithmetic even though both inputs were already
rounded to the narrower format in memory.

## What to change

Keep the narrow carriers live through the multiply and select the same
mixed-precision FMA form as the reference instead of widening and pairing the
operands first.

```python
# before: widen both halves, then tie two independent results to FP32x2.
a_lo = _bf16_to_f32(_low_half(a_word))
a_hi = _bf16_to_f32(_high_half(a_word))
b_lo = _bf16_to_f32(_low_half(b_word))
b_hi = _bf16_to_f32(_high_half(b_word))
next_lo, next_hi = _fma_f32x2(a_lo, a_hi, b_lo, b_hi, accum_lo, accum_hi)

# after: the inputs remain BF16 carriers and each result accumulates in FP32.
a_lo_bits = K.cast(a_word, "uint16")
a_hi_bits = K.cast(K.shift_right(a_word, K.uint32(16)), "uint16")
b_lo_bits = K.cast(b_word, "uint16")
b_hi_bits = K.cast(K.shift_right(b_word, K.uint32(16)), "uint16")
next_lo = K.local_scalar("float32")
next_hi = K.local_scalar("float32")
K.ptx.fma.rn.f32.bf16(next_lo, a_lo_bits, b_lo_bits, accum_lo)
K.ptx.fma.rn.f32.bf16(next_hi, a_hi_bits, b_hi_bits, accum_hi)
```

## Rationale

In one persistent mixed-precision recurrence, paired profiling attributed
520,192 excess dynamic `FFMA2` executions to the widened path. Selecting the
reference's BF16-input FMA removed 64 static packed-FMA sites and their
conversion chains. All ten correctness specializations passed against both the
source and an independent recurrence oracle.

The four affected ratios moved from 0.9186-0.9286 to 1.0071-1.0590. Across the
complete matrix, the minimum moved from 0.8981 to 0.9708 and the mean from
0.9240 to 1.0327; ten of eleven rows finished above parity. The improvement
tracked the exact arithmetic surplus identified in the profile.

## Boundary

Apply this only when the logical operands have already been rounded to the
narrow format and the reference PTX uses the mixed-precision instruction. Do
not downcast live FP32 values merely to reach the instruction. Packed FP32
arithmetic remains appropriate when the reference operates on FP32 pairs or
when separating the pair introduces more moves than it removes.

The accumulator and rounding modifiers are part of the contract. Match them
from reference PTX and confirm that the target architecture certifies the exact
instruction form.

## Verification

Compare native mixed-precision FMA, packed FP32 FMA, conversion, and move counts
in final SASS. Check registers and local traffic, run every affected numerical
branch, and then run the complete performance matrix because a remaining
specialization-specific deficit can still set the final minimum.
