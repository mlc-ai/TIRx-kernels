# Express low-level memory access through raw PTX

**Symptoms:** `low_level_ir_contract`, `buffer_load_violation`, `buffer_store_violation`, `precompile_test_failure`

## Symptom

Contract violations for `BufferLoad` and `BufferStore` in global or shared
scope, even where their eventual CUDA happens to match the reference.

## What to change

Keep buffers for shape and pointer ownership, but perform the access with
`T.ptx.ld.*` and `T.ptx.st.*` through `buffer.ptr_to([index])`. The pointer
operand is recorded as an address-only load and is contract-safe.

```python
# before: a raw global BufferLoad/BufferStore.
value = src[index]
dst[index] = value

# after: buffers still own shape and pointer; the access is typed PTX.
out = T.alloc_local([1], "int32")
T.evaluate(T.ptx.ld.global_.b32(out[0], src.ptr_to([index])))
T.evaluate(T.ptx.st.global_.b32(dst.ptr_to([index]), out[0]))
```

Do not reinterpret an arbitrary rvalue solely to feed a bit-typed store:
reinterpreting a literal such as zero can lower to an invalid address-of-rvalue
expression in CUDA. Cast integer values to the store's bit type, or reinterpret
a real register-backed lvalue where exact floating-point bits are required.

```python
# before: reinterpreting a literal -- invalid address-of-rvalue in CUDA.
T.evaluate(T.ptx.st.global_.b32(p, T.reinterpret("uint32", T.float32(0.0))))

# after: cast the value, or reinterpret a register-backed lvalue.
T.evaluate(T.ptx.st.global_.b32(p, T.cast(0, "uint32")))
```

## Rationale

The public low-level IR contract intentionally rejects those nodes. Migrating
one dispatch port this way reduced 29 violations across its two public functions
to zero (50 address-only pointer operands remained), passed all four correctness
configurations, and retained 0.993x and 1.022x in stable five-round 8-GPU
campaigns.

## Verification

Re-run the contract check over every public function, then the full correctness
matrix.
