# Fold invariant SMEM descriptor fields before long loops

**Symptoms:** `descriptor_spill`, `stack_growth`, `local_memory_traffic`, `short_scoreboard`

## Symptom

Stack bytes and local traffic that trace to the matrix descriptors themselves,
in a kernel whose long-lived operands differ only in their shared-memory start
address.

## What to change

`tcgen05.encode_matrix_descriptor` writes through an addressable `uint64`
temporary, and one temporary per long-lived operand can spill. Where LDO, SDO,
and swizzle are specialization constants, fold them into a compile-time base and
keep only scalar descriptors live.

```python
# Address-free constant: LDO, SDO, and swizzle are specialization constants.
DESC_MN = encode_smem_descriptor_base_uint64(0, 64, K.SW128B.value)

# The runtime start comes from one real K-owned shared view plus compile-time
# view deltas, so each operand is a scalar, not an addressable temporary.
desc = DESC_MN | ((base_addr + VIEW_DELTA) >> 4)
```

Retain a runtime encoder when any layout field is dynamic or the views do not
share a proven allocation anchor. Keep a descriptor issue-local when carrying it
across the loop increases uniform-register pressure.

## Rationale

One measured two-CTA attention-backward core moved from 3304 SASS instructions,
128 registers, and 160 bytes of stack to 3000 instructions, 128 registers, and
8 bytes of stack. All seven correctness configurations passed. A clean 45-round
same-GPU comparison measured 81.557 to 80.716 microseconds, 0.98968
after/before.

## Boundary

The fold is only valid while every folded field is genuinely a specialization
constant and every operand's start address is derived from the same proven
allocation anchor.

## Verification

Compare stack bytes, dynamic local loads and stores, and static SASS on both
sides, and confirm the descriptors no longer occupy addressable temporaries.
