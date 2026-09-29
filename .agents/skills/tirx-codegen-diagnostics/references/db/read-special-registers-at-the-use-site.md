# Read special registers at the use site

**Symptoms:** `local_memory_traffic`, `register_spill`, `long_scoreboard`, `persistent_grid_regression`

## Symptom

A persistent kernel keeps a special-register value (`gridDim.x` as the
grid stride, a rank, a lane mask) in one kernel-scope local for the whole tile
loop. Dynamic `LDL` rises against the reference, the hottest new stall is the
tile-increment `IMAD.IADD` whose operand comes from a local-memory fill, and the
reference shows thousands of extra `S2R` instead.

## What to change

Emit the special-register read where it is consumed instead of carrying it.

```python
# before: one asm-volatile read lives across every role's persistent loop.
num_bids = txl.local_scalar("uint32", init=txl.cuda.mov_sreg(32, "nctaid.x"))
...
txl.assign(tile_idx, tile_idx + num_bids)

# after: the read is re-issued at each increment; nothing stays live.
def num_bids():
    return txl.cast(txl.cuda.mov_sreg(32, "nctaid.x"), "uint32")
...
txl.assign(tile_idx, tile_idx + num_bids())
```

## Rationale

`txl.cuda.mov_sreg` lowers to `asm volatile("mov.u32 %0, %nctaid.x")`. A volatile
asm result cannot be rematerialized, so under a tight role budget ptxas spills
it to local memory and refills it at every use; a plain PTX special-register
read (what nvcc emits for `gridDim.x`) is rematerialized as `S2R` for free. In a
16-warp persistent attention kernel with 192/80/48 role budgets the carried
value cost 48 `LDL` per tile in the 80-register epilogue role and its fill sat
on the `tile_idx += gridDim.x` critical path. Re-reading it moved the two
streaming benchmark rows from 0.9693x to 1.0074x and from 1.0054x to 1.0457x;
static stack fell from 352 to 336 bytes and `LDL` from 128 to 120.

## Boundary

A once-read local is the right form when it is consumed inside one short region
or when the value must be provably warp-uniform for a later guard. The
re-read costs one `S2R` per use, so keep it out of unrolled inner chains.

Re-reading a special register must also preserve useful uniformity. In a
persistent backward kernel, replacing two compute-role warp-index expressions
with fresh volatile thread-index reads shifted right by five reduced static
local-load sites from 26 to 25, but increased local stores 36 to 42 and stack 16 to 40
bytes. S2R sites rose 29 to 54 and R2UR 296 to 320. Two ordinary complete-operation
profiles slowed 4.17% and 4.96%; two inactive-role controls were flat. Original
assertions and all six output byte comparisons passed. This rewrite was not
adopted. Reading afresh at role entry is not equivalent to rematerializing every
use, and fewer load sites alone do not establish a lifetime improvement.

The analogous plain thread-index expression, without the volatile read, also
regressed the same two ordinary profiles by 3.04% and 3.37%, while fallback
controls were nearly flat. Its stack reached 64 bytes with 42 local-load and 46
local-store sites, despite having the same mathematical warp number. All four
original checks and bytewise comparisons passed. Neither spelling reproduced
the allocation behavior of the original warp-uniform helper.

Adding a warp-uniform broadcast to the fresh volatile value did not recover
the original allocation either. Stack grew to 112 bytes and local-load/store
sites to 70/76; the same two ordinary profiles slowed 12.93% and 13.62%, with
fallback controls nearly flat and all four numerical/bytewise checks passing.
Uniformity and live-range placement must be evaluated together; neither a new
read nor a broadcast guarantees the original compiler allocation.

## Verification

Compare dynamic `LDL`/`STL` and `S2R` counts against the reference and confirm
the increment's operand no longer comes from a local-memory fill in the SASS.
