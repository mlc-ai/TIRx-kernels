# Merge waits with a protocol proof

**Symptoms:** `redundant_barrier_wait`, `serialized_teardown`, `exposed_store_tail`

## Symptom

Redundant barrier waits, serialized teardown, or an exposed store tail at kernel
exit.

## What to change

Collapse waits only when they guard the same consumer dependency and all
producers can contribute exact byte counts to one completion condition. The byte
count rides the arrive, so a merged wait means one expect-tx carrying the sum:

```python
@T.inline
def _mbarrier_expect_tx(smem_raw, byte_offset, num_bytes):
    T.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
        smem_raw.ptr_to([byte_offset]), T.uint32(num_bytes)
    )


@T.inline
def _mbarrier_arrive(smem_raw, byte_offset):
    T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(smem_raw.ptr_to([byte_offset]))


_mbarrier_wait(smem_raw, full_off + stage * 8, phase)
```

Delay a tail drain when kernel exit or a later protocol edge already supplies
the required ordering; where it is still needed, it is an extra wait past the
last real iteration:

```python
if STAGES == 8:
    dstage: T.uint32 = slot + T.uint32(4) * (dki % T.uint32(2))
    dphase: T.uint32 = dki // T.uint32(2) & T.uint32(1)
    _mbarrier_wait(
        smem_raw, T.uint32(consumed_off) + dstage * T.uint32(8), dphase ^ T.uint32(1)
    )
```

A drain that waits on everything the thread has outstanding merges the same
way. Where an epilogue reads a tile with several `tcgen05.ld` instructions and
waits after each, one wait after the last issue still orders all of them ahead
of the first consuming store, and the reads then overlap instead of draining
one at a time:

```python
# before: each read drains before the next is issued.
for chunk in range(CHUNKS):
    _tmem_load(frag[chunk], addr(chunk))
    _tmem_wait_ld()

# after: issue them all, drain once.
for chunk in range(CHUNKS):
    _tmem_load(frag[chunk], addr(chunk))
_tmem_wait_ld()
```

## Rationale

Before editing, write the producer-consumer happens-before argument; the merge
is sound only when one completion condition covers every producer's
contribution.

## Boundary

Do not merge visibility domains, permit ring overwrite, remove a release
witness, or weaken cross-stream ordering.

A wait that looks like a pure release may still be load-bearing. Where a named
barrier is reused across phases by several warpgroups, replacing its trailing
`bar.sync` with `bar.arrive` -- on the argument that the warpgroup only needs
to release another warp and never needs the return trip -- deadlocked: the
arrival counts fall out of phase with the matching `bar.sync` elsewhere. Match
the reference's instruction only after the phase argument holds for the port's
own barrier assignment, which may differ if the barrier ids were renumbered.
The merged form above is safe by contrast because the drain is per-thread, with
no other participant to fall out of step with.

## Verification

Validate transaction counts, ring wrap, completion visibility, and deadlock
freedom before profiling wait stalls and tail-dominated shapes.
