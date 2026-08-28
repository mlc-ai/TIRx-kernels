# Move whole rows with vector global accesses

**Symptoms:** `fixed_overhead`, `slow_small_shape`, `dispatch_specific_deficit`, `instruction_count_gap`

## Symptom

A persistent kernel seeds or drains a per-thread row of state through global
memory once per work item, and the port issues one scalar access per element
where the reference moves the row 128 bits at a time. The specialization that
carries the row trails the reference, and the deficit shrinks monotonically as
the per-item iteration count grows -- the signature of a fixed per-work-item
cost rather than a steady-state throughput gap.

## What to change

Move the row in 16-byte accesses. A row whose length is a power-of-two multiple
of the element size starts at least that far aligned, so every group is in
bounds and the width is structural rather than something ptxas may or may not
fuse.

```python
# before: one access per element, 128 per row.
for key in range(ROW):
    K.assign(values[key], _load_scalar(src, base + key))

# after: 16 bytes per access.
for group in range(ROW // 4):          # f32 rows
    K.ptx["ld.global.v4.f32"](
        values[group * 4], values[group * 4 + 1],
        values[group * 4 + 2], values[group * 4 + 3],
        src.ptr_to([base + group * 4]),
    )
```

For a narrower element type, load the same 16 bytes as `ld.global.v4.b32` and
unpack the pairs; the access count falls by the packing factor again.

## Rationale

Both directions matter: the seed reads the row and the drain writes it, and each
was one access per element. Widening them took the shortest affected shape from
0.858x to 0.998x while the guard dispatch that compiles the row moves out
measured unchanged (0.0001). The gain tracked the per-item iteration count
inversely, as a fixed per-item cost must: +0.14 where an item covered 32
iterations, +0.004 where it covered 512.

The reason this survives into a mature port is worth naming. A sibling kernel
sharing the same helper had its benchmark dispatch compile the row moves out
entirely, so the scalar form was never on a measured path there and carried over
looking proven. A dispatch that no benchmark covers can hide an arbitrarily
large fixed cost; when a new port puts that dispatch on the gate, check its
access widths against the reference before trusting the inheritance.

## Boundary

Only for contiguous runs whose base alignment follows from the row length, not
from a runtime offset that happens to be aligned. Where a cold path moves the
same row but never executes on the measured shapes, widening it is not the
lever -- keeping it a runtime loop is, because the cost there is the code it
adds to the enclosing loop body rather than the accesses it issues.

## Verification

Compare the state-carrying and state-free builds of the same kernel: take the
difference in `ld.global`/`st.global` sites on each side and require the port's
delta to match the reference's. Absolute counts across two compilers are not
comparable; the delta between two builds of the same program is.
