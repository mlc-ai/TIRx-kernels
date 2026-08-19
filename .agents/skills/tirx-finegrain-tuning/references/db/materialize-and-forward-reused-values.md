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

In one recurrent state update, the scalar-forwarding rewrite changed no PTX line
count but reduced the realized register allocation from 60 to 48 and static SASS
from 1080 to 1000 instructions; `IMAD.MOV.U32` fell from 84 to 31 and `MOV` from
45 to 20, with no spill. The resulting build then cleared all 43 gate workloads.

## Boundary

The reverse direction has a limit. Hoisting address invariants the backend
already merges changes almost nothing: one such hoist moved static SASS by five
instructions and left the shift count untouched.

## Verification

Count instructions in the corresponding PTX/SASS basic block and inspect ptxas
resources and SASS; source or PTX size can remain unchanged while registers and
SASS move.
