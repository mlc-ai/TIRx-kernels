# Include metadata readers in the slot reuse barrier

**Symptoms:** `read_write_race`, `nondeterministic_output`, `multi_tile_only_failure`

## Symptom

The producer overwrites a shared metadata slot while a matrix-issue warp still
reads its block count or indices. The data consumers have released the slot,
but the metadata reader was omitted from its reuse barrier.

## What to change

Include the metadata-reading warp in the reuse barrier's expected arrivals.
Have it arrive after its final metadata read, including the final nonempty
iteration. Keep arrivals balanced on every producer-visible generation.

## Rationale

Releasing the tensors does not release metadata still needed to issue work.
A sparse-attention prefill launch with 32 queries, 256 keys and 16 query heads
reported a producer store overlapping the matrix warp's metadata load. Adding
that warp's 32 arrivals removed the Racecheck error. NumSim and an actual SM100
GPU launch both matched the independent PyTorch oracle. Graph timings of 64
launches measured 13.97 us per launch before and 11.77 us after, with substantial
run-to-run noise; this is correctness evidence, not a general speedup claim.

## Boundary

Account for empty-work branches separately: a warp that never consumes a slot
must not introduce an unmatched arrival or change another generation's count.
The measured launch is a small BF16 configuration, not the full shape matrix.

## Verification

Check the last nonempty iteration as well as slot wraparound. Run Racecheck and
independent GPU correctness before measuring the affected launch configuration.
