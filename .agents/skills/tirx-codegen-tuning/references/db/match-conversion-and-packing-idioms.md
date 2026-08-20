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

Keep the conversion selector independent from the store selector. A fragment
may require scalar narrowing and still use a packed transaction after an
explicit register pack:

```python
if PACKED_NARROW:
    for pair in T.unroll(NUM_PAIRS):
        words[pair] = _cvt_pair(reg_f32[2 * pair + 1], reg_f32[2 * pair])
else:
    for value in T.unroll(NUM_VALUES):
        halves[value] = _cvt_scalar(reg_f32[value])
    if PACKED_STORE:
        for pair in T.unroll(NUM_VALUES // 2):
            T.ptx.mov.b32(words[pair], halves[2 * pair], halves[2 * pair + 1])
```

Derive both selectors from fresh reference PTX for the complete static fragment
family. Do not infer packed narrowing merely from an element vector width above
one.

## Rationale

Keep conversion width, packing width, and transaction width independent. A
validated 128-bit state store used eight scalar FP32-to-16-bit conversions, four
two-input `mov.b32` packs, and one `st.global.v4.b32`; ptxas then selected
paired `F2FP.*.F32.PACK_AB` instructions feeding the 128-bit SASS store. A
packed PTX conversion is therefore not required to obtain a packed store.

One measured fragment family made this independence non-monotonic: two- and
four-element vectors used packed conversions for some fragment extents and
scalar conversions for neighboring extents. A six-value FP16 fragment emitted
six `cvt.rn.f16.f32` instructions rather than three packed conversions, while a
scalar-narrowed eight-element path still repacked its results for a
`st.global.v4.b32`. Matching those two selectors separately passed the complete
correctness matrix and retained a final 1.003-1.010x three-shape benchmark
matrix.

A packed instruction can keep the reference's PTX mnemonic and arithmetic count
and still lose at the register-pair boundary. Against an identical baseline
launch, one port kept every global and shared transaction count and had no
spill, yet executed 339,902,492 instructions instead of 309,821,468 (+9.71%)
and took 398.3 instead of 370.3 microseconds (+7.56%). The opcode delta was
`MOV +19,038,208`, `IMAD +14,155,776`, and `LEA -7,733,248`; source correlation
assigned about 23.59 million of the `MOV`/`IMAD.MOV` instructions to
`fma.rn.f32x2`'s independent 64-bit output constraint, while the explicit
`mov.b64` helper contributed only about 0.79 million. Check dynamic opcodes and
constraint tying before blaming static expansion or the explicit packing calls.
Where the state update is naturally in place, the missing primitive is a native
tied read-write operand; a kernel-local asm wrapper is not an
instruction-selection fix.

## Boundary

The exact packed/scalar selector is a property of the source fragment layout
and compiler version, not a universal power-of-two or vector-width rule. Record
the observed selector as compile-time specialization data and re-probe its
boundary when either changes.

## Verification

Confirm packed words bitwise before judging instruction count. Trace the
conversion-to-store def-use chain in SASS instead of comparing conversion
mnemonics in isolation. Probe adjacent fragment extents and both dtypes, then
count scalar conversions, packed conversions, explicit packs, and final store
transactions independently.
