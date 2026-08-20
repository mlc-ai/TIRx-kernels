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
    """Runtime divisor; a compile-time one folds the subtraction away."""
    du: T.uint32 = T.cast(d, "uint32")
    return T.cast((T.cast(x, "uint32") + du - T.uint32(1)) // du, "int32")
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

The correction is charged per quotient, so what decides the size of the win is
where the quotients sit, not how many there are. A scheduler whose decode divides
inside a serial scan run once per batch, nested in a binary search, paid it on
its whole latency chain: the same seven-site cast took that loop body from
22 instructions to 14 against a reference's 15, and the shapes with the deepest
search from 0.824x and 0.853x to 1.405x and 1.368x, with the worst required
shape moving 0.824x to 1.082x. Static totals had already matched the reference
exactly at that point -- same load, store, divide, atomic and barrier counts --
because the correction is arithmetic the totals do not separate from the work.

## Boundary

What the cast removes is the correction, not the division: these divisors are
runtime values whenever the group sizes are, so both sides issue a real integer
divide.

## Verification

Count the fixup opcodes rather than the divide.
