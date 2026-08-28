# Keep trace-time loops at trace time

**Symptoms:** `register_spill`, `local_memory`, `runtime_loop`, `massive_slowdown`, `long_scoreboard`

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

When the source deliberately spells a small fixed device loop as unrolled, keep
that contract explicit instead of relying on a serial loop with a constant bound.

```python
# before: fixed bound, but the fragment still has a runtime index.
with K.serial(1, GROUPS, unroll=False) as group:
    score = combine(score, registers[group])

# after: preserve the source's explicitly unrolled device loop.
with K.unroll(1, GROUPS) as group:
    score = combine(score, registers[group])
```

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

In a separate fixed-group reduction, changing two constant-bound serial loops
to explicit unroll removed a 512-byte local depot. PTX local
declaration/load/store occurrences fell from 169/8/160 to 0/0/0 while 64
tensor-core issues and 40 tensor-copy instructions stayed unchanged. The
critical short-shape ratio rose from 0.375x to 0.996x, and the final 23-case
correctness matrix passed.

## Boundary

Only apply this to compile-time structural loops. Runtime trip counts and loops
that are deliberately represented by `T.serial` remain device loops; source
`T.unroll` remains an explicit unrolled device loop. Trace expansion can increase
code size, so it is not a general replacement for runtime iteration. Explicit
device unrolling is best reserved for small fixed trip counts where preserving
the source order and scalar indexing is important.

The code-size clause has teeth in the other direction too. A cold path guarded
by a runtime predicate still occupies the loop body that encloses it, so
expanding one at trace time costs the hot iterations even though it never
executes. A tail fixup that a reference writes as a runtime loop over a row, and
that a port expanded into a Python loop, put 256 unreachable global accesses
inside a persistent kernel's per-work-item body; restoring the runtime loop took
the port's excess over the reference from +112 sites per direction to below zero
and moved three near-gate shapes up by 0.004 to 0.016 with correctness
unchanged. Match the reference's staging boundary in both directions: expand
what it expands, and leave as a device loop what it leaves as one.

## Verification

Compare PTX for `.local`, `ld.local`, `st.local`, and an unexpected loop
backedge. Then run the affected correctness matrix and a paired same-GPU timing
against the source implementation; instruction-shape similarity alone is not a
performance oracle.
