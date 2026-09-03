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
num_bids = K.local_scalar("uint32", init=K.cuda.mov_sreg(32, "nctaid.x"))
...
K.assign(tile_idx, tile_idx + num_bids)

# after: the read is re-issued at each increment; nothing stays live.
def num_bids():
    return K.cast(K.cuda.mov_sreg(32, "nctaid.x"), "uint32")
...
K.assign(tile_idx, tile_idx + num_bids())
```

## Rationale

`K.cuda.mov_sreg` lowers to `asm volatile("mov.u32 %0, %nctaid.x")`. A volatile
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

## Verification

Compare dynamic `LDL`/`STL` and `S2R` counts against the reference and confirm
the increment's operand no longer comes from a local-memory fill in the SASS.
