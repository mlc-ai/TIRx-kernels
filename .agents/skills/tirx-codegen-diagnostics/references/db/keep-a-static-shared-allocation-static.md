# Keep a static shared allocation static

**Symptoms:** `excess_address_math`, `slow_small_shape`, `zero_dynamic_smem`, `fixed_overhead`

## Symptom

A short kernel is slower than its reference for no visible difference in the
algorithm, and every shared access in the generated code computes its address
from a runtime base rather than a compile-time offset.

## What to change

Where the reference declares its scratch as a plain `__shared__` array, declare
it as a static shared buffer, not from the dynamic pool. The pool hands out
offsets from a runtime `extern __shared__` base, which puts an address
computation on the critical path of every shared access; a static buffer makes
every offset a compile-time constant.

```python
# before: pool offsets are runtime values, so each access re-derives its address.
pool = T.SMEMPool()
counters = pool.alloc((words,), "uint32")
scan = pool.alloc((scan_words,), "uint32")
pool.commit()

# after: the reference's own shape -- offsets are compile-time constants.
counters = T.alloc_buffer((words,), "uint32", scope="shared")
scan = T.alloc_buffer((scan_words,), "uint32", scope="shared")
```

A union the reference expresses inside one allocation stays a union: alias the
views onto the dominant buffer rather than allocating each separately, so the
byte budget matches.

```python
# the exchange buffers alias the rank grid, as the reference's union does.
counters16 = counters32.view("uint16")
xchg_keys = counters32
xchg_values = counters32
```

## Rationale

The address arithmetic is fixed cost per access, so it is invisible wherever
occupancy hides it and decisive where a kernel is a few microseconds long. On
one block-sort port the switch moved the complete matrix from 39/46 to 43/46
shapes above the gate and the worst shape from 0.977 to 0.984.

## Boundary

The converse case is a hard requirement, not a preference: an arena above the
48 KB static ceiling cannot be a static buffer and must come from the pool. Check
the reference's own declaration before choosing, and verify the realized byte
budget either way -- a `cuobjdump -res-usage` figure includes the driver's
reserved block and will not match the declaration.

Take that declaration from the reference's export, not its source text. A DSL
frontend can allocate from the dynamic pool something whose source reads exactly
like a static array: a fixed-size struct member, no pool or arena in sight,
allocated through the frontend's own allocator and emitted as
`.extern .shared .b8 __dynamic_shmem__[]` with the base broadcast across the
warp. Reading the source there yields the opposite answer and sends a port away
from the reference rather than toward it. Where the reference is already the pool
form, matching it is what this entry asks for, and a static buffer is then a
candidate for beating the reference rather than a fidelity fix.

## Verification

Read the declaration in the generated CUDA, not just the total: the static form
appears as named `__shared__` arrays with constant subscripts, the pool form as
offsets into one extern base.
