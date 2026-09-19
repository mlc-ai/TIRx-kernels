# Keep staged loads when a split buffer widens

**Symptoms:** `performance_regression`, `slow_epilogue`, `excessive_sectors`, `tma_issue_overhead`

## Symptom

Changing a split buffer to a wider element type for numerical accuracy slows
the producer-plus-combine pipeline. The conversion instructions disappear or
shrink, but the combine kernel regresses enough that the end-to-end result is
still slower.

## What to change

Keep a previously validated TMA-to-shared staging path and its row-to-lane
mapping when only the split-buffer type becomes wider. Update the descriptor,
shared allocation, and shared load type together; do not replace the staged
transfer with per-row global loads to avoid a small staging tile.

```python
# before: each row lane fetches its own widened partial directly from global.
for value in txl.unroll(VALUES_PER_LANE):
    partial[value] = _ld_global(split_ptr + row_base + lane_offset + value)

# after: preserve the bulk transfer and consume the widened shared tile.
txl.cuda.tma_load_2d(
    split_smem,
    split_desc,
    split_index,
    barrier=split_ready,
)
txl.cuda.mbarrier_wait_parity(split_ready, phase)
for value in txl.unroll(VALUES_PER_LANE):
    partial[value] = _ld_shared(split_smem + lane_offset + value)
```

Preserve the measured number of lanes assigned to each row as well. More lanes
reduce the bytes handled by one thread but do not reduce the bytes transferred
by the CTA.

## Rationale

In one measured split-and-combine pipeline, widening the partials from one to
two bytes increased end-to-end latency by 21.0% on an eight-way reduction and
13.0% on a four-way reduction. The main producer changed by only about 2% in a
separate profile; the wider producer stores and combine reads accounted for
the remaining cost.

Replacing the widened combine's TMA staging with direct global loads made the
same two shapes slower again, from 145.94 to 164.56 microseconds and from 94.17
to 103.63 microseconds. Doubling the lanes per row also failed to recover the
bandwidth cost, measuring 147.79 and 96.36 microseconds. Retaining the original
TMA staging and row mapping was the fastest exact-format implementation among
the measured candidates.

Doubling each BF16 TMA tile from 32 to 64 KiB did not amortize enough fixed
work to help: the eight-way row was 0.3% slower and the four-way row was 3.3%
slower in paired runs. Publishing row readiness from the persistent producer
and letting the dependent TMA combine run early also regressed by 15.5% and
9.4%, even after reducing the publication atomics to one per token, split, and
slot. Embedding a direct-load combine into the producer was worse still. The
extra producer barrier and global publication protocol outweighed the overlap.

Replacing the contended counters with generation-tagged publication did not
change that conclusion. Packing the row sum and generation into one 64-bit
release store, acquiring it in the dependent combine, and issuing TMA only
after every row in the tile was ready measured 180.83 versus 147.01
microseconds on the eight-way row and 106.67 versus 94.68 microseconds on the
four-way row. A baseline-style alternative with 64 rows per CTA and a
two-stage `cp.async` split pipeline was much closer, but still measured 148.95
versus 146.67 and 95.05 versus 93.82 microseconds. Neither eliminating atomic
RMWs nor reducing CTA count paid for replacing the original TMA tile.

Making the TMA combine persistent and double-buffering consecutive row tiles
also failed. On the eight-way row it used 60 registers with no spills, yet
regressed from 147.48 to 186.64 microseconds. Thousands of short CTAs provided
more useful latency hiding than one long-lived CTA per SM; prefetching the next
32 KiB tile did not compensate for the lost inter-CTA scheduling freedom.

Partitioning a three-sequence workload so each sequence produced and consumed
a smaller BF16 workspace also lost substantially. Sequential per-sequence
pipelines regressed the eight-way row by 45.9% and the four-way row by 62.6%;
launching those pipelines on three CUDA streams reduced the losses only to
34.1% and 48.3%. The smaller working set did not repay repeated persistent-grid
startup and the loss of one globally balanced work queue.

## Boundary

This applies when the algorithm, split count, and reduction are unchanged and
the regression begins with a wider intermediate format. It does not show that
TMA is always preferable to direct global access, and it does not remove the
bandwidth cost of the wider format. Recovering that cost requires reducing
traffic or overlapping it at the pipeline level; changing load syntax or lane
ownership alone is not enough. Do not add a row-readiness protocol solely to
overlap the merge unless the producer already publishes the needed readiness
state or a measured end-to-end run pays back that synchronization.

## Verification

Benchmark the complete producer-plus-combine path, then time the producer and
combine separately. Confirm that the wider path writes and reads the expected
number of bytes and compare global sectors as well as conversion instructions.
Keep the node and workload fixed when evaluating direct-global and lane-layout
alternatives, and retain them only if they improve the end-to-end time while
passing the exact-format correctness matrix.
