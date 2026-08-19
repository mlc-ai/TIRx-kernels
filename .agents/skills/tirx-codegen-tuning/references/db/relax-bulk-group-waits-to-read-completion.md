# Relax bulk-group waits to read completion

**Symptoms:** `serialized_tma_pipeline`, `store_load_overlap_blocked`, `instruction_parity_with_deficit`

## Symptom

A TMA pipeline serialized at the slot-reuse wait: the next load cannot issue
until the previous store's write has landed, even though only the SMEM source
needs to be free.

## What to change

Where the SMEM slot's next consumer is the next load's read, wait on read
completion rather than full completion.

```python
# before: waits for the whole previous TMA store -- SMEM read and the HBM or
# NVLink write -- before the next load may issue.
T.ptx.cp.async_.bulk.wait_group(0)

# after: waits only until the TMA engine has finished reading the SMEM source,
# which is exactly the reuse requirement of a single per-warp slot.
T.ptx.cp.async_.bulk.wait_group.read(0)
T.cuda.warp_sync()
```

## Rationale

Relaxing both per-token waits in one dispatch token loop and copy epilogue
overlapped the previous store's write with the next token's load and brought the
kernel to parity. Publication semantics were unchanged because a full commit and
`wait_group(0)` still guard the exit barriers.

A secondary effect is visible in SASS: the full wait lowers to DEPBAR plus a
per-token `CCTL.IVALL`, so the relaxed form also drops a per-token L1 invalidate
from the loop.

## Boundary

The relaxation is valid only where the SMEM slot's next consumer is the next
load's read, not the store's completion. Write that producer-consumer argument
per wait site before editing.

## Verification

Re-verify correctness at scale, and confirm the exit path still carries a full
commit and wait before any cross-rank or cross-CTA barrier.
