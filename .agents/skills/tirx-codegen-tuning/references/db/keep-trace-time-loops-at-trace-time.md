# Keep trace-time loops at trace time

**Symptoms:** `register_spill`, `local_memory`, `runtime_loop`, `massive_slowdown`

## Symptom

A DSL port preserves the source operations but becomes several times slower,
and PTX gains a per-thread local-memory depot with many `ld.local` and
`st.local` instructions. Small fixed register arrays that scalarized in the
reference now remain indexed across a runtime loop.

## What to change

Preserve the staging boundary of fixed Python loops. A plain `range` in a traced
kernel expands the body while tracing; a DSL serial loop emits a device `For`.

```python
# Wrong when STATIC_TRIPS belongs to the traced program's static structure.
with K.serial(STATIC_TRIPS, unroll=False) as i:
    consume(registers[i])

# Keep the index a Python integer and emit each operation while tracing.
for i in range(STATIC_TRIPS):
    consume(registers[i])
```

When porting parser kernels, translate a source `T.serial` to the DSL's runtime
loop construct, but leave a source Python `range` as Python `range`.

## Rationale

Two sparse-attention forward ports changed fixed Python loops into runtime
serial loops. The representative generated PTX acquired a 1280-byte local depot
per thread, 252 local loads, and 304 local stores even though its MMA and
synchronization work had not changed. Restoring trace-time expansion removed
all local loads and stores.

On paired same-GPU measurements, the first representative moved from 11.16 ms
to 1.347 ms, matching the 1.357 ms parser baseline; its sibling moved from
10.83 ms to 1.041 ms, matching the 1.041 ms baseline. Their complete 55-case
correctness matrix passed after the change.

## Boundary

Only apply this to compile-time structural loops. Runtime trip counts and loops
that are deliberately represented by `T.serial` remain device loops; source
`T.unroll` remains an explicit unrolled device loop. Trace expansion can increase
code size, so it is not a general replacement for runtime iteration.

## Verification

Compare PTX for `.local`, `ld.local`, `st.local`, and an unexpected loop
backedge. Then run the affected correctness matrix and a paired same-GPU timing
against the source implementation; instruction-shape similarity alone is not a
performance oracle.
