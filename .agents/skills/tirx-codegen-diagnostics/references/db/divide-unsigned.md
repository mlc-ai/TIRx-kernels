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

The proof travels with the value, not with the variable, so one specialization
of a kernel can pay the correction while another does not for the identical
source expression. An index taken straight from a grid coordinate is provably
non-negative and folds; the same index handed back by a persistent scheduler's
`divmod`, or decoded from a cluster-launch-control response, is an ordinary
signed integer and pays again. One kernel dividing a head index by a
compile-time ratio emitted nothing at ratio one, a single `shr.u32` at a
power-of-two ratio and a four-instruction 16-bit reciprocal at ratio three while
the index came from the grid, then a ten-instruction correction for the same two
power-of-two ratios, and nine with a 32-bit magic multiply for ratio three, once
persistent and split-KV scheduling supplied the index instead. Cast where the
scheduler hands the index over and the specializations agree again. Reading only
the specialization whose index comes from the grid hides the cost entirely.
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

In a 12-warp persistent attention pipeline, making all four role counters
unsigned and using unsigned division/remainder for query/head decomposition cut
static SASS from 3,048 to 2,632 instructions while keeping 151 registers and
zero local traffic. The weakest ratio in a four-shape targeted matrix improved
from 0.955x to 1.015x, and a 17-shape sweep then cleared with a 1.015x minimum
and 1.046x geometric mean. Merely materializing the same signed expressions had
reduced static code by 56 instructions but was benchmark-neutral; carrying the unsigned
proof from the grid-stride counter was the decisive part of the rewrite.

## Boundary

What the cast removes is the correction, not the division: these divisors are
runtime values whenever the group sizes are, so both sides issue a real integer
divide.

## Verification

Count the fixup opcodes rather than the divide.
