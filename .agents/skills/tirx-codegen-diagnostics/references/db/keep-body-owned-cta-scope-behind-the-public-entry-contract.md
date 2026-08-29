# Keep body-owned CTA scope behind the public entry contract

**Symptoms:** `thread_axis_drift`, `sass_schedule_divergence`, `launch_config_drift`, `low_level_ir_contract`

## Symptom

A source kernel declares its CTA coordinates in the body and also names a flat
thread coordinate there. Moving both axes into an entry layout can change the
generated schedule, but exposing a per-kernel thread-layout switch or calling a
raw builder bypasses the Kern entry contract.

## What to change

Leave only the CTA shape body-owned. Select that contract with `grid=False`,
declare the CTA coordinates through `K.cta_id(extents)`, and consume the flat
thread coordinate owned by the entry through `K.thread_id()`.

```python
@K.kernel(warps=THREADS // 32, arch="sm_100a", grid=False)
def kernel(...):
    work, batch = K.cta_id([NUM_WORK, NUM_BATCHES])
    tid = K.thread_id()
```

Do not add a `thread_layout` entry option or import the raw IR builder in a
kernel to recreate the old spelling.

## Rationale

The Kern entry has one canonical flat CTA-local thread axis, while `grid=False`
keeps the source's body-owned CTA shape without creating a second entry
representation. An eight-warp PDL combine migrated from raw body calls for both
axes to this public spelling across three default specializations. The CUDA
text changed, but the final fatbin and complete SASS stayed byte-identical:
1,456 / 1,160 / 1,240 instructions, 87 / 69 / 78 registers, and zero stack,
spill, or local memory.

Earlier entry implementations did produce schedule regressions when they moved
both scopes. The reusable distinction is therefore CTA ownership, not a public
switch for thread ownership.

## Boundary

Binary identity in one kernel does not prove every body-owned thread scope is
irrelevant. If the final SASS changes, treat that as an entry-primitive design
question; do not work around the public contract with a kernel-local raw
builder or a per-entry layout escape hatch.

## Verification

Compile through the real runner path and compare the final fatbin, complete
SASS, and resources. Then measure the real launch scope, especially for a PDL
consumer whose isolated timing can miss producer-consumer effects.
