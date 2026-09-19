# Release shared tiles after the last scalar reader

**Symptoms:** `nondeterministic_outputs`, `shared_memory_race`, `barrier_stall`

## Symptom

A scalar fallback reads a shared intermediate after the final tensor-core
consumer. The loader still treats tensor-core completion as permission to reuse
that tile. Outputs are finite but incorrect and change between identical
launches.

## What to change

Extend the tile's reuse protocol to cover its new scalar readers. If delaying an
existing release also delays unrelated transfers, give the new reader lifetime
its own release. Every participant must arrive once per phase, including lanes
that skip the fallback.

```python
# before: the new scalar use outlives the old tile ownership.
wait_tensor_consumers()
start_next_tile_load()
read_scalar_correction()  # races with the next tile

# after: producer reuse follows the actual final reader.
read_scalar_correction_if_needed()
scalar_readers_free.arrive()
# loader role:
wait_tensor_consumers()
scalar_readers_free.wait()
start_next_tile_load()
```

## Rationale

One measured correction extended the lifetime of two derivative tiles into a
scalar epilogue. The previous tensor-completion wait allowed the next gate load
to overwrite them. Query/key normalized RMS errors were approximately 0.65-0.70
and identical launches disagreed. Waiting for the scalar epilogue removed the
large errors and repeatability failures. After compacting the correction into
an earlier pass, a dedicated release preserved correctness while allowing gate
and upstream-gradient prefetch before the remaining epilogue finished. Three
normal-range workloads moved from about 1115/190/323 to 1110/187/318 microseconds.

## Boundary

A shorter lifetime must account for scratch reuse as well as the original
operand. In the measured extension, the scalar pass reused old state tiles for
correction outputs. Their existing later release still had to protect epilogue
reads after the derivative tiles could be released.

Warp ballots or other full-mask collectives used to select a fallback must be
materialized before lane divergence. A barrier arrival by each lane does not
make a warp collective inside a partial-lane branch valid.

## Verification

Map each shared region through producer, tensor consumer, scalar consumer, and
next producer. Test multiple successive work items, mixed fallback predicates,
partial chunks, and repeated poisoned launches. Confirm phase progression for
lanes that skip work, then measure the prefetch overlap that the separate release
is intended to restore.
