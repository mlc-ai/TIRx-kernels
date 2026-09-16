# Join the warp before an elected lane publishes its accesses

**Symptoms:** `stale_shared_data`, `nondeterministic_output`, `read_write_race`

## Symptom

An asynchronous matrix consumer can race with shared-memory writes even when
all producer lanes execute the required proxy fence. Only the elected producer
lane arrives at the publishing barrier, supplying a count of 32.

## What to change

Join all participating lanes before electing the barrier publisher. Preserve
the per-lane proxy fences that order generic stores before async-proxy readers.

```python
# before: election alone does not join preceding accesses from other lanes.
if txl.cuda.elect_sync() != txl.uint32(0):
    ready.arrive(slot, count=32)

# after: publish the joined warp's accesses through the elected lane.
txl.cuda.warp_sync()
if txl.cuda.elect_sync() != txl.uint32(0):
    ready.arrive(slot, count=32)
```

## Rationale

The arrival count supplies barrier accounting; it does not collect another
lane's preceding memory operations. Election selects the issuing lane but does
not replace the memory-ordering warp join. A sparse-attention decode case with
one query, 256 keys and 16 query heads changed from a Racecheck read/write error
to clean after adding the join. NumSim and an actual SM100 GPU launch matched an
independent PyTorch oracle. Three rounds timing graphs of 64 launches measured
6.87 us per launch before and 6.90 us after; direct-launch measurements were noisy.

## Boundary

All lanes named by the warp synchronization mask must participate. Do not add a
full-mask collective inside a branch entered only by the elected lane. This
experiment covers one BF16 decode shape, not a complete performance matrix.

## Verification

Run Racecheck with the full producer and consumer roles, compare the GPU output
with an independent oracle, and measure the affected launch configuration.
