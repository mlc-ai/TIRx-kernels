# Stage unrolled loads into distinct registers

**Symptoms:** `long_scoreboard`, `exposed_load_latency`, `insufficient_memory_parallelism`, `unroll_no_effect`

## Symptom

An `unroll=` annotation measurably helps, but the shape is still behind the
reference and the profile still shows exposed load latency. Paired NCU on the
failing shape reports identical global load sectors on both sides and FEWER
executed instructions on ours, with the whole gap in `long_scoreboard`:
22.48 warp-cycles per issued instruction against the reference's 7.02 at
4352 load sectors each, 14336 instructions against 24832.

Reading the generated CUDA shows why the unroll did not buy what it looked like
it bought: every unrolled copy writes the SAME destination local.

```c
alignas(64) uint _ptr_6[1];
tvm_builtin_ptx_ld_global_nc_b32(_ptr_6[0], ...);   // copy 0
tvm_builtin_ptx_st_global_b32(...);                 // consumes it
tvm_builtin_ptx_ld_global_nc_b32(_ptr_6[0], ...);   // copy 1 must wait
```

The copies exist, but each load waits for the previous consumer to free the
register, so no two loads are ever in flight.

## What to change

Stage the whole chunk into distinct registers before consuming any of it: an
explicitly unrolled Python-level loop over a `T.alloc_local((N,), ...)` array,
loads first, uses second.

```python
stage = T.alloc_local((trips,), "int32")
for u in range(trips):                      # python unroll -> one register each
    stage[u] = load(base + u * stride)
for u in range(trips):
    consume(stage[u])
```

This took a page-table transform's trivial path from 0.825x to 0.992x, and the
kernel's complete matrix from five failing shapes to three.

## Rationale

`T.serial(..., unroll=N)` is a hint about loop structure, not about register
allocation. The unrolled body reuses the destination the helper allocated, which
serializes exactly the loads the unroll was meant to overlap. Splitting the
load phase from the use phase is what actually creates the independent
destinations, and `T.alloc_local((N,), ...)` indexed by a Python-level loop
variable is the way to name them.

## Boundary

Applies where the trip count is a compile-time exact multiple of the stride, so
the staged loop needs no per-element bounds check. Where the bound is a runtime
value, the re-check costs more than the parallelism returns: the same transform
on a loop bounded by a runtime row length regressed three shapes
(0.966 -> 0.946, 0.984 -> 0.944, 0.988 -> 0.863). Guard the exact-divisor case
and leave the general path on the ordinary loop.

The staged array is real registers. Keep `trips` small; the win here came at 4.
That ceiling is specific to staging global loads whose destinations were being
serialized. A shared-memory ladder feeding a long arithmetic chain measured the
opposite preference -- 4 regressed every required shape against 8, and the
complete 16-element fragment beat both -- so sweep the width rather than
assuming either direction.

## Verification

Diagnose with paired stall breakdowns, not instruction counts -- fewer
instructions plus more time is the signature. Confirm the fix in the generated
CUDA by checking that the unrolled copies write different destinations, then
accept or reject on bench-suite.
