# Issue loads before conversions

**Symptoms:** `long_scoreboard`, `insufficient_memory_parallelism`, `exposed_load_latency`

## Symptom

Long-scoreboard stalls in a cold-cache, memory-latency-bound kernel; loads
serialized behind dependent conversion chains. Also the residual case where
instruction and sector counts all match and only DRAM throughput differs: one
port issued fewer instructions, requested identical sectors, clustered its loads
better and moved 17% less data, and still reached 81% of the reference's DRAM
throughput.

## What to change

Issue independent global loads before starting dependent widening, unpacking, or
conversion chains: stage the raw bits into a local buffer in one loop, then
widen them in a second loop.

```python
# before: each widening chain sits on the critical path of the next load.
for j in T.unroll(ROWS):
    words = _load_u32x4(state, read_base + T.cast(row_index(j), "int64"))
    for pr in T.unroll(4):
        h_reg[j * 8 + 2 * pr] = _bf16_to_f32(_lo16(words[pr]))
        h_reg[j * 8 + 2 * pr + 1] = _bf16_to_f32(_hi16(words[pr]))

# after: every row's load is issued before any of them is widened.
h_words = T.alloc_local((4 * ROWS,), "uint32")
for j in T.unroll(ROWS):
    words = _load_u32x4(state, read_base + T.cast(row_index(j), "int64"))
    for pr in T.unroll(4):
        h_words[j * 4 + pr] = words[pr]
for j in T.unroll(ROWS):
    for pr in T.unroll(4):
        w: T.uint32 = h_words[j * 4 + pr]
        h_reg[j * 8 + 2 * pr] = _bf16_to_f32(_lo16(w))
        h_reg[j * 8 + 2 * pr + 1] = _bf16_to_f32(_hi16(w))
```

If raw bits can remain live safely, sink the first conversion behind
independent work or a real synchronization boundary. Program order is the
lever; which side of a barrier ptxas finally places the work is not something
the kernel controls.

## Rationale

The benchmark harness zeroes a 256 MB buffer before every timed iteration, so
the measured kernel always starts with an empty L2 and the figure of merit is
how many misses are outstanding. This regime is not optional in the gate.

This has produced 1.7-3.7% gains in short recurrent kernels and much larger
gains when multiple token loads were previously serialized. In the
matching-counters case above, the lever that worked was raising the number of
outstanding misses.

## Boundary

It can regress when the staged raw values spill or the loads usually hit cache.

Design the ablation carefully. Deleting the conversion arithmetic to test
whether it costs anything removed 95 instructions and changed the time by
nothing, because the consumer still depended on the load -- that experiment
answers whether the arithmetic is expensive, not whether the latency is exposed.
The measurement that does answer it is the scheduled-SASS distance from the last
load to its first consumer, which moved from 17 to 215 instructions when the
conversion was written later.

## Verification

Confirm the load issue window and load-to-first-consumer distance in SASS, then
measure long-scoreboard stalls, registers, spills, and the complete shape
matrix.
