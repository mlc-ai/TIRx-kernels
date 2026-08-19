# Match conversion and packing idioms

**Symptoms:** `bitwise_mismatch`, `packing_mismatch`, `excess_shift_or`, `store_width_mismatch`

## Symptom

Packed words differ bitwise from the reference, packing lowers to shift/or
chains, or store width diverges from the reference's transaction width.

## What to change

- Check operand order for packed conversions; paired FP formats commonly take
  the high element first. Use asymmetric correctness inputs.

  ```python
  # PTX's packed cvt names the high-half source first.
  T.ptx.cvt.rn.bf16x2.f32(words[pair], reg_f32[pair * 2 + 1], reg_f32[pair * 2])
  ```

- Pack two 16-bit halves with the native two-input 32-bit move and two 32-bit
  halves with the corresponding 64-bit move.

  ```python
  def cvt_f32x2_to_packed(lo, hi, dtype):
      """Two scalar cvt + one two-input ``mov.b32``."""
      h0 = T.alloc_local([1], "uint16")
      h1 = T.alloc_local([1], "uint16")
      cvt = T.ptx.cvt.rn.f16.f32 if dtype == "float16" else T.ptx.cvt.rn.bf16.f32
      T.evaluate(cvt(h0[0], lo))
      T.evaluate(cvt(h1[0], hi))
      out = T.alloc_local([1], "uint32")
      T.evaluate(T.ptx.mov.b32(out[0], h0[0], h1[0]))
      return out[0]


  # Two 32-bit halves into one 64-bit word.
  T.evaluate(T.ptx.mov.b64(out[0], v[0], v[1]))
  ```

- For an unavailable packed form, reuse a native equivalent already validated
  in another kernel instead of inventing a new lowering.
- Match shuffle masks and saturation/rounding modifiers exactly.

The store keeps its own width independent of both:

```python
T.evaluate(T.ptx.st.global_.v4.b32(buf.ptr_to([index]), w[0], w[1], w[2], w[3]))
```

## Rationale

Keep conversion width, packing width, and transaction width independent. A
validated 128-bit state store used eight scalar FP32-to-16-bit conversions, four
two-input `mov.b32` packs, and one `st.global.v4.b32`; ptxas then selected
paired `F2FP.*.F32.PACK_AB` instructions feeding the 128-bit SASS store. A
packed PTX conversion is therefore not required to obtain a packed store.

## Verification

Confirm packed words bitwise before judging instruction count. Trace the
conversion-to-store def-use chain in SASS instead of comparing conversion
mnemonics in isolation.
