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

When one source helper converts four pairs and packs the four byte results with
the four-input form of `mov.b32`, keep the conversion results inside one typed
compound intrinsic. Returning each byte through an ordinary scalar helper can
force a shift/or pack even though the final PTX idiom is native.

```python
# before: four separately returned byte carriers are shifted and ORed.
for pair in T.unroll(4):
    byte[pair] = _cvt_pair_to_byte(values[2 * pair], values[2 * pair + 1])
packed = _shift_or_four_bytes(byte)

# after: the helper owns four paired conversions and one four-input mov.b32.
packed = T.cuda.cvt_e2m1x8_f32(
    values[0], values[1], values[2], values[3],
    values[4], values[5], values[6], values[7],
)
```

The store keeps its own width independent of both:

```python
T.evaluate(T.ptx.st.global_.v4.b32(buf.ptr_to([index]), w[0], w[1], w[2], w[3]))
```

If a paired conversion already produces final 16-bit carriers in memory byte
order, do not repack them into 32-bit words only to reach a wide shared store.
Store the halves directly at the same byte addresses:

```python
# before: eight pair carriers are repacked into four words.
for pair in T.unroll(4):
    T.ptx.mov.b32(words[pair], halves[2 * pair], halves[2 * pair + 1])
T.ptx.st.shared.v4.b32(dst, words[0], words[1], words[2], words[3])

# after: the pair carriers already are the final bytes.
T.ptx["st.shared.v4.b16"](
    dst, halves[0], halves[1], halves[2], halves[3]
)
T.ptx["st.shared.v4.b16"](
    dst + 8, halves[4], halves[5], halves[6], halves[7]
)
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

The inverse case matters too. If the reference narrows scalars and performs
scalar halfword stores, keep the narrowed halves live through those stores
instead of using packed conversion merely because two values fit in one word.

```python
# before: packed narrowing is immediately undone for scalar stores.
for pair in K.unroll(NUM_VALUES // 2):
    words[pair] = _cvt_pair(values[2 * pair + 1], values[2 * pair])
for i in K.unroll(NUM_VALUES):
    K.ptx.st.shared.b16(dst[i], _extract_half(words[i // 2], i & 1))

# after: scalar narrowing feeds the reference's scalar transaction directly.
for i in K.unroll(NUM_VALUES):
    halves[i] = _cvt_scalar(values[i])
for i in K.unroll(NUM_VALUES):
    K.ptx.st.shared.b16(dst[i], halves[i])
```

The PTX store's data carrier is independent as well. If the reference feeds the
low byte of a word directly to `st.global.b8`, preserve the word carrier instead
of inserting a TIR-level narrowing operation.

```python
# before: narrowing is introduced only to satisfy the wrapper's source dtype.
byte = T.cast(scale_word, "uint8")
T.ptx.st.global_.b8(address, byte)

# after: the certified overload consumes the low byte of the uint32 carrier.
T.ptx.st.global_.b8(address, scale_word)
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

In a measured FP8 epilogue, sixteen paired conversions already produced the
final halfword sequence. Four direct `st.shared.v4.b16` operations removed
eight PRMT-equivalent repacks while preserving the shared bytes. All five
focused rows passed at a 0.9909x minimum, the direct stores survived the full
correctness matrix, and the retained implementation passed the final 66-row
suite. A broader candidate initially failed elsewhere because of an unrelated
global register policy; separating store width from register selection kept the
packing result usable.

In another measured scalar-store epilogue, replacing sixteen packed
conversions plus halfword extraction with 32 scalar conversions feeding 32
`st.shared.b16` operations matched the reference instruction shape. Two small
guard rows moved from 0.9805x to 0.9827x and from 0.9899x to 0.9923x; the latter
cleared the strict gate, while the former still required an independent
register-schedule fix. This rewrite was retained through the complete
correctness and performance matrices.

## Boundary

The exact packed/scalar selector is a property of the source fragment layout
and compiler version, not a universal power-of-two or vector-width rule. Record
the observed selector as compile-time specialization data and re-probe its
boundary when either changes.

Only add a wider source-carrier overload after ptxas certifies that exact PTX
operand class. A byte store consuming the low byte of a uint32 register does not
justify weakening every b8 operation's dtype contract. Likewise, use a compound
intrinsic for a native multi-output packing idiom, not as a container for an
arbitrary workload-specific instruction sequence.

Direct halfword stores require exact byte-order and alignment equivalence.
Confirm that no later consumer expects the temporary 32-bit packed words, and
do not infer the rewrite from output dtype alone; it applies only when the
conversion result is already the final memory representation.

Scalar halfword stores have the same constraint. They are appropriate when the
reference transaction is scalar and packing would be immediately reversed;
they are not a general replacement for a naturally packed shared transaction.

## Verification

Confirm packed words bitwise before judging instruction count. Trace the
conversion-to-store def-use chain in SASS instead of comparing conversion
mnemonics in isolation. Probe adjacent fragment extents and both dtypes, then
count scalar conversions, packed conversions, explicit packs, and final store
transactions independently. For low-byte stores, also check that final PTX has
no cast or narrowing instruction between the producing word and `st.global.b8`.
