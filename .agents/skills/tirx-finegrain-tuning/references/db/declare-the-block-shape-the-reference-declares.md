# Declare the block shape the reference declares

**Symptoms:** `special_register_reads`, `instruction_count_bloat`, `slow_latency_bound_shape`

## Symptom

Excess dynamic special-register reads (`S2R`, `S2UR`) and instruction bloat on
latency-bound shapes.

## What to change

Take the block shape from the reference, not from a scaffold's correspondence
table. When the reference launches a flat block and derives warp, lane, and
group indices by division, declare the flat block and derive the indices.

```python
# before: a 2-D declaration costs a special-register read per component,
# at every use, for indices the reference derives by division.
lane, warp = T.thread_id([32, WARPS])

# after: one flat block, indices derived.
tid = T.thread_id([THREADS])
warp: T.int32 = _make_warp_uniform(tid // 32)
lane: T.int32 = tid % 32
lane_quad: T.int32 = lane % 4
```

## Rationale

A multi-dimensional `thread_id` costs a special-register read per component at
every use, so a 2-D declaration adds nothing over the reference's flat launch
and is charged for on every access. One latency-bound kernel declared a
`(32, warps, 1)` block where both the reference and the approved sketch
specified a flat one. It cost 4608 dynamic `S2R` plus 4608 `S2UR`; flattening
moved the worst shape from 0.864x to 1.016x and removed 48 static instructions,
all in the affected phase.

## Verification

Confirm with dynamic special-register counts on the worst shape.
