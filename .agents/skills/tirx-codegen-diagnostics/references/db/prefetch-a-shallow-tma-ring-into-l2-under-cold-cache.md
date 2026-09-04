# Prefetch a shallow TMA ring into L2 under cold cache

**Symptoms:** `cold_cache_regression`, `exposed_load_latency`, `insufficient_memory_parallelism`, `instruction_parity_with_deficit`, `barrier_stall`

## Symptom

A warp-specialized pipeline beats its reference under a warm L2 yet loses to it
in the benchmark suite, whose timer flushes L2 before every launch. The deficit
is per block, independent of wave count, and appears only in the specialization
whose shared-memory budget leaves the fewest ring stages. Barrier attribution
shows the consumer side spinning on the accumulator-ready barrier (the matrix
warp starves on operands) far more than the reference does, while executed
instructions, issue efficiency, and the MMA issue code are equal or better.

## What to change

Have the producer warp issue a TMA L2 prefetch (`cp.async.bulk.prefetch.tensor`,
no shared memory, no barrier) for the block a fixed distance ahead of the one it
loads, and gate it on ring depth: only rings whose in-flight bytes no longer
cover a cold HBM fetch at the port's block time.

```python
# before: the ring is the only prefetch; with 3 slots and 2 loads per block it
# holds 1.5 blocks, so a cold-L2 fetch stalls the matrix warp.
with K.serial(n_blocks - 1, unroll=False) as i:
    n = n_max - 2 - i
    load_k(n)
    load_v(n)

# after: pull block n-2 into L2 while the ring is busy with n and n-1.
PREFETCH = 2 if kv_stage <= 3 else 0
with K.serial(n_blocks - 1, unroll=False) as i:
    n = n_max - 2 - i
    if PREFETCH:
        with K.If(n - PREFETCH >= 0), K.Then():
            with K.If(elected()), K.Then():
                K.ptx["cp.async.bulk.prefetch.tensor.4d.L2.global.tile"](
                    K.address_of(tmap_k), K.int32(0), K.Cast("int32", n_pf * BLK_N), head, batch
                )
                # ... and the V box(es) of the same block
    load_k(n)
    load_v(n)
```

Also prefetch the first two blocks of every tile before the ring's own loads
begin, since the ring is empty at a tile boundary.

## Rationale

The ring depth that hides latency is a function of block time, not a constant of
the algorithm. One attention forward with an e4m3 K plus bf16 V (48 KB per
block) fit only three ring slots in 227 KB; its reference ran 1.2 us per block
and covered the cold fetch, the port ran 1.0 us per block and did not. Measured
with CUPTI, warm versus cold (256 MB flush): port 101.2 -> 125.7 us (+24%),
reference 120.6 -> 124.2 us (+3%); sibling modes with four or more slots moved
+1-4%, and a 32768-key stream 0%. Under the suite's cold protocol the row sat
at 0.984x. With the prefetch, 125.7 -> 111.0 us (1.118x); two sibling rows of
the same mode moved 1.008 -> 1.133x and 1.036 -> 1.096x. Applied to every mode,
the same prefetch cost 0.5-2.3% on rings of 4-13 slots (long bf16 and fp8
streams), so the gate at three slots keeps only the beneficiary.

## Boundary

Only for rings that are provably too shallow for the cold fetch: deeper rings
already overlap the latency and pay the prefetch's issue slots and L2 traffic
for nothing. Prefetch distance must stay within the tile so no out-of-range
block is touched; guard the block index. This does not change bytes moved into
shared memory or any barrier count, so it needs no protocol re-review, but the
touched blocks' L2 footprint grows by the distance.

## Verification

Reproduce the suite's protocol before optimizing: time with a 256 MB L2 flush
between launches and compare against a warm loop on both sides; a large
warm/cold gap on the port only is the signature. Attribute `try_wait` retries
per barrier from the NCU source view to confirm the operand-ready barrier is
the one starving. After the change, re-measure the affected mode and every
deeper-ring sibling at one long and one short shape.
