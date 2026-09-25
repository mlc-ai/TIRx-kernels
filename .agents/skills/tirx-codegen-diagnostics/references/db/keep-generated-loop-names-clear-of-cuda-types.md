# Keep generated loop names clear of CUDA scalar types

**Symptoms:** `invalid_generated_cuda`, `expected_expression`, `type_name_shadowed`, `address_cast_failure`

## Symptom

Generated CUDA fails to compile at a typed pointer cast with `expected an
expression`, under both NVRTC and nvcc, even though the TIR and the PTX
instruction form are valid.

## What to change

Rename the loop coordinate, not the instruction or the address representation.

```python
# before: the loop variable's name reaches generated CUDA and shadows the
# `half` type, so `(half*)ptr` parses as an expression on the loop index.
for half in T.unroll(2):
    ...

# after: same traced dataflow, different generated C++ identifier.
for half_idx in T.unroll(2):
    ...
```

## Rationale

TIRx preserves a Python loop variable's name in generated CUDA. Naming that
variable `half` shadows CUDA's `half` type inside the loop, so a later shared
buffer address lowers as `(half*)ptr` but parses as an expression using the
integer loop variable. One SM100 matrix-precompute specialization failed at its
first `stmatrix` address before the rename and compiled with zero errors
afterward; the five sibling device variants in the same launch chain also
compiled.

## Boundary

Apply this only to names that collide with CUDA scalar types in a scope that
emits typed pointer casts. It is not a reason to rename unrelated coordinates.

## Verification

Read the generated CUDA at the failing line: a cast whose type name is also a
live loop variable in that scope is the signature.
