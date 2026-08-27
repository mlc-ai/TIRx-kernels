# Materialize and forward reused values

**Symptoms:** `repeated_expression`, `excess_address_math`, `excess_unpack_math`, `instruction_count_bloat`, `register_count_gap`

## Symptom

The corresponding PTX/SASS basic block carries a 1.5-2x instruction-count
increase with the same control flow, or the realized register allocation runs
above the reference with no spill. Source or PTX size can remain unchanged.

## What to change

Bind swizzle offsets, unpacked lanes, scale products, and other reused values to
local scalars or buffers exactly once. `T.let` binds an immutable value; a
`name: T.dtype = expr` annotation gives a reassignable local scalar.

```python
# Reused index arithmetic: bound once, not rebuilt at every use.
v_tile: T.let[T.int32] = linear_cta % NUM_V_TILES
cta_head: T.let[T.int32] = linear_cta // NUM_V_TILES

# Values reused across an unrolled body, where the reference's compiler keeps
# them live: inline-PTX shared loads are opaque to nvcc's CSE, so materialize
# the register lifetime explicitly.
g_value: T.float32 = _shared_load_f32(s_gb, 0)
beta_value: T.float32 = _shared_load_f32(s_gb, 1)
```

Helpers used multiple times should return materialized values rather than
rebuild an expression:

```python
def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs))
    return out[0]  # a materialized value, not an expression tree
```

When translating a parser kernel to a tracing DSL, preserve every typed scalar
binding that defined a single-evaluation boundary. A plain Python assignment in
the traced function only names an expression tree.

```python
# before: parser syntax materializes this typed scalar once.
q_begin: T.int32 = chunk_idx * target

# after: native tracing spells the same register and evaluation boundary.
q_begin = K.local_scalar(K.i32, init=chunk_idx * target, name="q_begin")
```

Preserve an untyped parser assignment too when it acted as an inferred-width
or single-evaluation boundary. Materialize compact element offsets at the
parser-inferred 32-bit width and strided byte/store offsets at 64-bit width;
making every offset 64-bit is not equivalent.

```python
# before: tracing aliases and expands both expression trees at each use.
x_offset = row * hidden + lane
store_offset = batch * stride + x_offset

# after: preserve the parser's inferred integer boundaries.
x_offset = K.local_scalar(K.i32, init=row * hidden + lane, name="x_offset")
store_offset = K.local_scalar(
    K.i64, init=batch * stride + x_offset, name="store_offset"
)
```

Where the reference's own compiler CSEd two accesses to the same element across
opaque inline asm, reuse the earlier value explicitly; two excess loads and a
redundant integer max disappeared that way.

For an in/out inline-PTX destination written directly into a dynamically indexed
local buffer whose just-written value the next instruction consumes: bind the
PTX result to a scalar, write that scalar back, and forward the scalar to the
next instruction instead of rereading the buffer expression.

```python
# before: the next instruction rereads the dynamically indexed buffer.
_fma_store(new_states, state_iter * STATE_VECTOR + e, state_value, da_value, db_x)
out_value = _fma(new_states[state_iter * STATE_VECTOR + e], c_value, out_value)

# after: bind to a scalar, write it back, forward the scalar.
new_state: T.float32 = _fma(state_value, da_value, db_x)
new_states[state_iter * STATE_VECTOR + e] = new_state
out_value = _fma(new_state, c_value, out_value)
```

## Rationale

TIRx expressions are trees. Reusing a `PrimExpr` can emit its complete subtree
at every use; ptxas does not reliably recover the intended common subexpression,
and no backend merges across opaque inline asm.

In one measured KDA port, restoring two fixed-loop unrolls reduced
stage-4/stage-8 SASS from 688/792 to 672/776 instructions; materializing the
reused entry, address, and output scalars then restored the reference's 608/704
instructions and 33/46 registers, with one fewer stage-8 special-register read.
The combined change moved all eight benchmark ratios from 0.894-0.996x to
0.995-1.008x.

In one recurrent state update, the scalar-forwarding rewrite changed no PTX line
count but reduced the realized register allocation from 60 to 48 and static SASS
from 1080 to 1000 instructions; `IMAD.MOV.U32` fell from 84 to 31 and `MOV` from
45 to 20, with no spill. The resulting build then cleared all 43 gate workloads.

In one parser-to-tracing scheduler port, leaving typed parser scalars as Python
aliases preserved global-load and branch counts but lowered issue-active from
79.21% to 74.75% and raised average active cycles from 594.1k to 632.6k. Two
15-round paired cases regressed by 6.14% and 7.84%. Explicitly materializing the
parser's scalar bindings, including stable source names, restored byte-identical
SASS; all 12 correctness cases passed and the same paired cases measured
0.999951 and 1.000013 after/before.

In one normalization port, omitting ordinary untyped offset assignments changed
the fused path from 744 instructions / 126 registers to 728 / 124 and slowed it
by 5.8%; the quantized path stayed at 696 instructions but fell from 90 to 82
registers and slowed by 2.0%. Restoring the compact offsets as 32-bit locals and
the strided/store offsets as 64-bit locals made both SASS images byte-identical
to the parser builds. Correctness passed, and 15-round same-GPU A/B measured
1.000239 and 1.000194 after/before on the two affected configurations.

## Boundary

The reverse direction has a limit. Hoisting address invariants the backend
already merges changes almost nothing: one such hoist moved static SASS by five
instructions and left the shift count untouched.

`K.Bind` has its own lowering boundary. A let-bound quotient, index, or offset
that feeds a buffer view or pointer expression can still be substituted into
every address use. In one eight-warp PDL combine, that repeated dynamic division
and address arithmetic produced 296 SASS instructions and 42 registers, and two
clean short-row campaigns measured 1.01773 and 1.01334 after/before.
Materializing only the persistent indices, split bounds, and base offsets in
one-element local slots restored the 280-instruction, 47-register SASS
byte-for-byte across all three default specializations, and all 15 correctness
configurations passed. Keep `K.Bind` for ordinary reused expressions, but
inspect generated CUDA when a value crosses buffer/view lowering and use a local
slot where the let is expanded instead of forwarded.

A smaller generated program is not sufficient evidence to keep this rewrite.
In one dependency-protocol specialization, forwarding a row predicate removed
eight static instructions but raised registers from 96 to 98 and left the gate
unchanged at 0.979-0.980x. A broader address-base materialization removed 16
static instructions and reduced registers from 92 to 88 with no spill, yet the
affected path still regressed to 0.985x. Both changes were reverted.

Shared-memory swizzles can benefit from materializing an arena-relative row
offset rather than repeatedly forming and canceling the absolute shared base.
In one block-scaled epilogue this changed 69 registers / 984 static
instructions into 66 / 960, improved a frozen 136-row targeted minimum from
0.974556 to 0.976191, and raised its geometric mean from 0.998707 to 1.005177.
The subsequent 184-row validation still had five strict-gate failures and a
0.974381 minimum. Keep the relative-offset form only when its alignment proof
holds, and treat the resource reduction as mechanism evidence rather than a
full-matrix performance verdict.

Composing the same relative-offset rewrite with a source-shaped native
accumulator wait preserved the 66-register / 960-instruction program. A
worktree-isolated 136-row run improved the geometric mean from 1.001507 to
1.005664, but raised the minimum by only 0.000999, from 0.973880 to 0.974879,
and left six strict-gate failures in the FP32-output path. Do not infer
composition safety from resource counts or an aggregate: require the minimum
and every affected path to clear the targeted trigger before full validation.

## Verification

Count instructions in the corresponding PTX/SASS basic block and inspect ptxas
resources and SASS; source or PTX size can remain unchanged while registers and
SASS move. Treat those changes as mechanism evidence, then require a measured
gain on the affected and guard workloads before retaining the materialization.
