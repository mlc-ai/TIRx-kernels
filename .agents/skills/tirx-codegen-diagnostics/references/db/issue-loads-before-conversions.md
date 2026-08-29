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

When an overlap window has an always-valid half and a predicated half, issue
the always-valid group first, then the predicated group, and delay consumers of
both. Otherwise the predicate and the first half's conversion chain can cut the
load issue window in two even though each half is staged internally.

```python
previous_words = K.alloc_local([N], "uint32")
current_words = K.alloc_local([N], "uint32")

# before: the first conversion chain separates the two load groups.
with K.If(has_previous):
    load_group(previous_words, previous_ptr)
    convert_group(previous_values, previous_words)
load_group(current_words, current_ptr)
convert_group(current_values, current_words)

# after: every independent load is issued before either conversion chain.
load_group(current_words, current_ptr)
with K.If(has_previous):
    load_group(previous_words, previous_ptr)
with K.If(has_previous):
    convert_group(previous_values, previous_words)
convert_group(current_values, current_words)
```

If ordinary local-array staging still compiles to the old schedule, a bounded
multi-output device helper can give every load a distinct output register and
make the issue boundary real. Use it only after final SASS proves that the
compiler erased the ordinary rewrite.

## Rationale

The benchmark harness zeroes a 256 MB buffer before every timed iteration, so
the measured kernel always starts with an empty L2 and the figure of merit is
how many misses are outstanding. This regime is not optional in the gate.

This has produced 1.7-3.7% gains in short recurrent kernels and much larger
gains when multiple token loads were previously serialized. In the
matching-counters case above, the lever that worked was raising the number of
outstanding misses.

One eight-token KDA decode showed the mechanism across ten independent BF16
value loads: issuing every raw 16-bit load before conversion and recurrence
removed the failing cold-cache gap. All 28 correctness configurations passed,
and the affected clean 45-round same-GPU row moved from 7.548 to 7.056
microseconds, 0.93482 after/before.

The pattern holds across instruction mixes, not just the one it was found on.
A block-scaled dequantization with a different unpack chain and half the trip
count -- four shared loads per thread rather than eight -- gained on every
specialization that ran it (+0.0015, +0.0056, +0.0022) with an untouched control
dispatch flat, once the batch covered the whole trip count.

An eight-position overlap window supplied the predicated-boundary case. A
plain load-first rewrite and a typed-load helper both collapsed to the same 832
final instructions at 48 registers. Grouping each half separately changed the
binary but still failed both measured shapes. Issuing the eight always-valid
score/value loads before the eight predicated score/value loads, with all 16
ahead of the first BF16 conversion, reduced long-scoreboard samples from 1299
to 552. The final allocation was 53 registers with no spill; all 18 correctness
cases passed, and the complete four-shape benchmark matrix measured
1.0249-1.1635x against the reference.

## Boundary

It can regress when the staged raw values spill or the loads usually hit cache.
The decode result above held with a final build at 56 registers, zero stack, and
zero local memory; it does not extend to a specialization whose raw staging set
spills.

Design the ablation carefully. Deleting the conversion arithmetic to test
whether it costs anything removed 95 instructions and changed the time by
nothing, because the consumer still depended on the load -- that experiment
answers whether the arithmetic is expensive, not whether the latency is exposed.
The measurement that does answer it is the scheduled-SASS distance from the last
load to its first consumer, which moved from 17 to 215 instructions when the
conversion was written later.

Source and PTX order are not enough. Separating a complete shared-word load
phase from its unpack phase made one PTX stream match the reference more closely,
but the final cubin remained identical at 661 instructions, 92 registers, and
the same relevant opcode counts. With no generated-code lever left, the rewrite
was reverted before timing.

Staging width is not monotonic. In the predicated-window experiment, a
half-window boundary paid the code-size and live-range cost without passing the
benchmark, and a smaller group restored the old register count without
recovering performance. Preserve the full issue window only while the extra
outputs remain in registers, and do not use an opaque multi-output helper for
loads whose independence, address validity, or ordering is not statically
established.

## Verification

Confirm the load issue window and load-to-first-consumer distance in SASS, then
measure long-scoreboard stalls, registers, spills, and the complete shape
matrix. For a predicated window, verify that the unconditional and conditional
groups both precede the first consumer rather than checking each group in
isolation.
