# Select address lowering by shape

**Symptoms:** `excess_address_math`, `register_pressure`, `schedule_regression`

## Symptom

Excess integer address ops, register pressure, or scheduling regressions traced
to address lowering, with no single lowering best across the shape matrix.

## What to change

Keep native `[base+imm]` offsets and explicit pointer arithmetic as alternatives
until a full shape picker is measured. Put both lowerings behind one helper flag
so a specialization can choose:

```python
def _global_load_u16_ptr_offset(ptr, byte_offset: int, native_offset: bool = True):
    if not native_offset:
        # Explicit pointer arithmetic.
        return _global_load_u16_ptr(T.ptr_byte_offset(ptr, byte_offset, "bfloat16"))
    # Native [base+imm]: the offset is baked into the instruction operand.
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.b16(out[0], T.ptx.addr(ptr, byte_offset)))
    return out[0]
```

The choice is then a per-shape constexpr passed at specialization, not a global
preference:

```python
USE_NATIVE_OFFSETS=_HAS_NATIVE_PTX_ADDR and (seq_len == 2 or 3 < seq_len < 8),
```

Retain cursor induction only when the emitted address chain and the affected
shapes demonstrate a gain.

## Rationale

Native `[base+imm]` addressing removes explicit pointer arithmetic only when the
byte offset is a compile-time immediate. It can still alter scheduling,
allocation, dependency chains, or pointer lifetimes. One FP32 MTP matrix
selected native offsets for 56 of 97 configurations and explicit arithmetic for
the other 41; neither form was globally best.

Likewise, replacing `step * stride` with a moving cursor is not intrinsically
cheaper. Five live cursors in one recurrent specialization increased registers
from 58 to 60 and static SASS from 1072 to 1080 instructions, with no spill and
no consistent full-matrix gain. ptxas had already strength-reduced much of the
original indexing, while the explicit cursors extended five live ranges and
added loop-back updates.

## Verification

Confirm normalized PTX/SASS addresses, register counts, integer address ops,
spills, and latency per specialization.
