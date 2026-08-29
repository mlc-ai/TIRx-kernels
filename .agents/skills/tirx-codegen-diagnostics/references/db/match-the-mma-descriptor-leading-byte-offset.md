# Match the MMA descriptor's physical leading-byte offset

**Symptoms:** `illegal_shared_access`, `utchmma_fault`, `mma_descriptor_lbo_mismatch`, `second_kphase_fault`

## Symptom

A typed shared-memory view with the right allocation, swizzle, and base address
still faults inside the MMA instruction. The first matrix phase stays in range
and a later phase -- the first one whose address depends on the leading-byte
stride -- faults.

## What to change

Compare the complete descriptor constant and its phase increments, not only the
shared pointer. Where a generic descriptor helper encodes a different physical
leading-byte offset than the reference, use the kernel's canonical raw
descriptor constructor for exactly the operands whose layout differs.

## Rationale

One 128x64 f16/bf16 SW128B view encoded LBO 1024 through the generic MN-major
helper, producing a `0x04000000` low field where the validated kernel required
LBO 512 and `0x02000000`. The second matrix phase was the first to depend on
that stride and faulted at `UTCHMMA`. Overriding only the four MN-major operands
restored `0x4000404002000000`, passed Compute Sanitizer with zero errors, and
made all valid intermediate tiles bit-identical across ten registered
configurations.

## Boundary

Do not replace typed descriptor construction globally. Keep the typed buffer as
the storage owner and override only the descriptor boundary whose physical
layout has been proven against generated CUDA/SASS and correctness data.

## Verification

Print the full 64-bit descriptor constant and the per-phase increments on both
sides, then run Compute Sanitizer and compare intermediate tiles bitwise across
every registered configuration.
