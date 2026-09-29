# Take shared-exchange widths from the reference

**Symptoms:** `instruction_count_gap`, `sass_divergence`, `slow_small_shape`

## Symptom

A block-exchange or block-scan transcribed with scalar shared accesses where the
reference issues vector ones, leaving several extra `ld.shared.b32` per pass that
no amount of surrounding tuning removes.

## What to change

Two places in a cub-style collective read a contiguous run and are vectorized in
the reference, each under its own condition.

The blocked gather is contiguous exactly when padding is off -- cub's rule is
`ITEMS_PER_THREAD > 4 && is_power_of_two(ITEMS_PER_THREAD)` -- so at four items
per thread there is no padding and the run is one 16-byte load.

```python
if items_per_thread == 4 and not insert_padding(items_per_thread):
    quad = ld_shared_quad_u32(buf, tx * items_per_thread)
    items_reg[0], items_reg[1] = quad[0], quad[1]
    items_reg[2], items_reg[3] = quad[2], quad[3]
else:
    for i in T.unroll(items_per_thread):
        items_reg[i] = ld_shared_u32(buf, padded_offset(...))
```

The warp-aggregate array is read whole, and the width follows the warp count:
`v2` at two warps, `v4` at four, and two `v4` at eight. Cover every warp count
the launcher can reach, not just the ones a favourite shape exercises.

## Rationale

Matching both brought the shared profile to an exact match with the reference --
2 `ld.shared.v4.b32`, 11 scalar `ld.shared.b32`, 28 `st.shared.b32`, 9
`bar.sync` -- and is mostly a parity change: the gather measured neutral to
within 0.001 on shapes that reproduce to that precision. The widest rung, where
eight aggregates become two vector loads instead of eight scalar ones, moved
0.988-0.990 to 0.994-0.998.

## Boundary

Only where the run really is contiguous. With padding on, the gather is strided
and must stay scalar; taking the width from a shape that happens to be unpadded
will read wrong data on the others.

## Verification

Diff the shared-memory opcode histogram against the reference rather than
checking one rung. The eight-warp case here was left scalar through two separate
audits because the four-warp case was present and looked complete.
