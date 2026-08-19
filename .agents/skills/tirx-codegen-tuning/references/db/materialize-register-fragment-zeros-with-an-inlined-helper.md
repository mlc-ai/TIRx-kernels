# Materialize register-fragment zeros with an inlined helper

**Symptoms:** `zero_fill_instruction_gap`, `register_fragment_undefined`, `inline_asm_boundary`, `instruction_count_gap`

## Symptom

A source kernel deliberately materializes a complete register fragment before
predicated loads, but the target emits missing, extra, reordered, or differently
packed zero moves. The numerical result may still pass when every lane loads
valid data, while tail lanes expose undefined values or the emitted instruction
profile remains different from the reference.

## What to change

When ordinary typed assignments do not preserve the required initialization
profile, put only the zero materialization in a typed, force-inlined device
helper. Write directly to the register-backed lvalues with the PTX constraint
that matches their physical width, then call the helper before the first load.

```python
from tvm.backend.cuda.op import cuda_func_call


def zero_u16_fragment(values, count: int):
    # The containing module compiles one specialization of this helper.
    helper = "tvm_builtin_zero_fragment_u16"
    moves = "\n".join(
        f'    asm volatile("mov.b16 %0, 0;" : "=h"(values[{i}]) :: "memory");'
        for i in range(count)
    )
    source = (
        f"\n__forceinline__ __device__ void {helper}(uint16_t* values) {{\n"
        f"{moves}\n"
        "}\n"
    )
    T.evaluate(cuda_func_call(helper, values.ptr_to([0]), source_code=source))
```

Use `=h`, `=r`, `=f`, and `=l` for 16-bit, 32-bit, FP32, and 64-bit
destinations respectively. Match the reference's materialization width rather
than always choosing the widest form: adjacent 16-bit lanes may need one
`mov.b32`, while two FP32 lanes may need one FP32 zero seed followed by a
`mov.b64` pair materialization.

Declare only the exact helper names to the low-level IR checker. Do not exempt a
prefix or arbitrary device calls.

## Rationale

In one measured fragment family, a normal typed path emitted eight scalar
16-bit zero moves for a seven-value fragment. Moving the same initialization
into the typed helper emitted exactly seven, all before the first global load.
The other measured profiles emitted exactly 128 packed 32-bit half-zero moves,
232 FP32 zero moves, one FP32 seed plus 128 64-bit pair moves, and 232 scalar
16-bit moves as selected by their fragment layouts.

Every helper disappeared after inlining: final PTX contained no call, `.local`,
`ld.local`, or `st.local`, and the initialized registers flowed directly into
the predicated loads and arithmetic. The complete numerical matrix passed. This
establishes a reusable generated-code control mechanism; it does not claim an
isolated timing speedup for zero filling.

## Boundary

Use this only when initialization instruction shape or placement is part of the
observed mismatch. A volatile memory clobber can constrain scheduling, so keep
the helper limited to the smallest proven initialization region and retain
ordinary TIRx assignments when their final binary already matches.

Do not collapse zero writes that the source places on different control-flow
edges into one helper call. Preserve their separate CFG positions. Do not use a
source-string helper to hide computation, synchronization, or an arbitrary PTX
bundle.

A benchmark that selects an asynchronous path and never executes the helper
cannot prove its performance impact. Require an affected sync profile before
making a timing claim; otherwise retain the entry only as instruction-fidelity
and correctness evidence.

## Verification

Count the width and extent of zero moves in fresh PTX and prove that they all
precede the first consuming load. Confirm the final binary has no surviving
helper call or local-memory traffic, then check registers, stack, correctness on
full and tail fragments, and the affected plus guard benchmark shapes.
