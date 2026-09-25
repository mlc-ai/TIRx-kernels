# Hoist a read past every loop that does not index it

**Symptoms:** `short_scoreboard`, `mio_throttle`, `bank_conflict_excess`, `instruction_parity_with_deficit`

## Symptom

Shared-load instruction counts match the reference almost exactly, yet bank
conflicts and wavefronts run tens of percent above it, and the kernel spends
more cycles while executing *fewer* instructions overall.

## What to change

Find the shared reads whose index does not depend on the innermost loop, and
lift them past **every** loop that does not index them -- not just the nearest
one. Cache the result in a register array sized by the indices that actually
matter.

```python
# before: read once per innermost iteration. `row` depends on the outer split
# and on nothing else, but the read sits inside the column loop.
for col_pass in range(N_PASS):
    for grp in range(N_GRP):
        row = row_of_lane + lane_base + 8 * parity
        scale = _shared_load_f32(s_scale, slot * STRIDE + row)
        ...

# after: one read per distinct row for the whole region.
scale_cache = T.alloc_local((N_LANE_BASE * N_PARITY,), "float32")
for lb in range(N_LANE_BASE):
    for par in range(N_PARITY):
        row = row_of_lane + lb * LANE_STEP + 8 * par
        scale_cache[lb * N_PARITY + par] = _shared_load_f32(s_scale, slot * STRIDE + row)
for col_pass in range(N_PASS):
    for grp in range(N_GRP):
        scale = scale_cache[(lane_base // LANE_STEP) * N_PARITY + parity]
```

Hoisting to the enclosing loop is not enough and reads as a no-op: the backend
already common-subexpression-eliminates a repeated shared load *within* one
straight-line region, so the instruction count does not move and the change
looks worthless. It does not CSE across the outer loop, which is where the
duplication actually lives.

Values held in one-element local buffers collapse the same way, which is worth
knowing because the storage looks per-thread and real. Binding six such fields,
read 14 and 8 times, to scalars so the reads coalesce left every SASS counter
unmoved -- 3552 static instructions on both sides. Hand-caching within a region
is never the lever; crossing the loop is.

## Rationale

One epilogue read of a per-row reciprocal, issued once per 128-bit store,
accounted for 493,694 shared-load instructions and 1,129,247 bank conflicts --
2.29 conflicts apiece -- against a whole-kernel excess over the reference of
763,220. Cutting it to four reads per thread moved shared-load wavefronts from
+16.1% against the reference to +2.8%, took shared-load instructions to 10.6%
*below* it, and moved the failing shape from 0.9785 to 1.0035. Six of the other
nine shapes improved at the same time.

The hoist-to-the-nearest-loop version of the same change measured a 84K drop in
conflicts out of 3.16M and an unchanged instruction count, and was discarded as
a no-op before the outer hoist was tried.

## Boundary

Only worth it where the read is genuinely loop-invariant in the inner index and
the cache stays small: the register array is live across the whole region, so a
cache sized by an inner index trades the stall for pressure. Reads that a warp
broadcasts (every lane the same address) are already cheap and gain nothing.

Fewer loads is not the same objective as fewer conflicts, and the two can
oppose. Two index arrays read per store were folded into one packed word decoded
by a mask and a shift -- 19% fewer shared-load instructions, correctness intact,
and the shape lost 0.008. Concentrating the accesses into one array cost more in
distribution than the removed loads returned, and the decode lengthened the
dependency chain. Reduce the reads whose index is loop-invariant; do not merge
distinct arrays to reduce the count.

An access pattern that looks conflict-free on paper can still be the worst one
in the kernel. This read distributes as eight distinct rows per warp with
four-way broadcast, which is textbook conflict-free, and it was nonetheless
carrying a third of all conflicts. Do not reason an access out of suspicion --
delete it in a throwaway build and watch the counter.

## Verification

Diff `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`,
`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` and
`smsp__inst_executed_op_shared_ld.sum` against the reference. Equal instruction
counts with unequal conflicts localize the problem to an access pattern rather
than to how much work is issued.
