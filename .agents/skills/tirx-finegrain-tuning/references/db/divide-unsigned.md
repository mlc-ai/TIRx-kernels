# Divide unsigned

**Symptoms:** `excess_integer_math`, `excess_address_math`, `instruction_count_gap`, `slow_small_shape`

## Symptom

Absolute-value and signed-compare chains around integer division against a
reference that has none. One masked-layout shape carried 49,920 absolute-value
and 38,460 signed-compare instructions against the reference's zero.

## What to change

Cast each counter to unsigned before dividing, and route related arithmetic --
such as swizzled-block math -- through the same unsigned helpers.

```python
def _udiv(x, d):
    """Exact division of a value known non-negative but not provably so."""
    return T.cast(T.cast(x, "uint32") // T.cast(d, "uint32"), "int32")


def _umod(x, d):
    return T.cast(T.cast(x, "uint32") % T.cast(d, "uint32"), "int32")


def _uceil(x, d):
    return T.cast((T.cast(x, "uint32") + T.uint32(d - 1)) // T.uint32(d), "int32")
```

An index variable can also be born unsigned, which removes the per-site cast:

```python
tid_u32 = T.thread_id([THREADS], dtype="uint32")
lane_div8: T.uint32 = tid_u32 // T.uint32(8)
```

Apply the cast across the whole family: these counters feed every role, so the
sites that matter are usually more numerous than the one the profile
attributed.

## Rationale

A signed `//` or `%` lowers to the full floordiv/floormod fixup sequence
whenever the dividend's sign cannot be proven. A scheduler counter read from a
signed integer global carries no such proof even when every value it can hold is
a count, so each division emits an absolute value, a sign compare and a chain of
moves that a reference written in unsigned arithmetic throughout does not have.
The cast took the masked-layout shape from 3.868M to 3.537M instructions against
the reference's 3.388M, and from 0.976x to above the gate.

## Boundary

What the cast removes is the correction, not the division: these divisors are
runtime values whenever the group sizes are, so both sides issue a real integer
divide.

## Verification

Count the fixup opcodes rather than the divide.
