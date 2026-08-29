# Pick the TMEM load shape by the stores it implies

**Symptoms:** `uncoalesced_store`, `excessive_sectors`, `slow_epilogue`, `unsaturated_bandwidth`

## Symptom

An epilogue that is bit-exact and writes exactly the reference's bytes, while
issuing twice its sectors. Fewer instructions than the reference and more time,
with the gap in the store path rather than the GEMM. The profiler names it
outright: uncoalesced global accesses, N excessive sectors, roughly half the
total.

## What to change

Choose the `tcgen05.ld` shape by which thread ends up holding which element, not
by which shape is easiest to index. The shape fixes the register-to-element map,
and the map fixes whether a warp's 128-bit stores are contiguous.

`32x32b` makes the thread the M row and the register the column -- an identity
map, and the obvious choice when transcribing. Every lane of one store
instruction then lands on a different row, one row stride apart. `16x256b` puts
four columns of ONE row in lanes 0-3, so a store covers eight rows in contiguous
64-byte runs: 16 sectors for 512 bytes, the minimum.

Measured `16x256b` map, seeded through a `32x32b.st` whose map is the identity:

```text
address = (lane_base << 16) | col_base
row = warp*32 + lane_base + (lane % 32)//4 + 8*((r//2) % 2)
col = col_base + (lane % 4)*2 + (r % 2) + 8*(r//4)
```

One instruction is 32 registers covering 64 rows by 64 columns, so a 128x128
tile takes four, over `(lane_base, col_base)` in `{0,16} x {0,64}`. The second
row half is reached through the **lane field of the address**, not through the
repetition suffix and not through the column field: `.x8` and `.x16` cover the
same rows, and a column offset alone leaves half the tile unread.

Where the reference permutes columns before storing, that permutation belongs to
its fragment layout. Applying the inverse permutation to a different fragment
reproduces the bytes exactly while destroying the contiguity the permutation
existed to create.

## Rationale

The permuted store address is not an ABI curiosity to be undone; it is what
makes the stores whole-sector. One epilogue read with `32x32b` and the inverse
map issued 16,127,606 excessive sectors, 50% of 32,433,928, against 5,750 (0%)
for the reference, whose total sector count was exactly half for identical
bytes written. Switching the read shape and using the reference's own forward
map took two shapes from 0.538 to 0.929 and 0.486 to 0.931.

## Boundary

Only where the fragment feeds global stores. A fragment consumed in registers,
or written to shared memory where a swizzle absorbs the layout, should take
whichever shape indexes most simply. The wide-fragment live-range cost still
applies: reading the whole tile before storing any of it trades sectors for
registers.

## Verification

Compare `l1tex__t_sectors_pipe_lsu_mem_global_op_st` and `dram__bytes_write`
against the reference together. Equal bytes with unequal sectors is the tell,
and it survives every value-level correctness check, so no amount of bitwise
agreement rules it out. Seed a tile with `row * K + col` through the shape whose
map is already known and dump the registers to recover an unfamiliar map rather
than deriving it.
