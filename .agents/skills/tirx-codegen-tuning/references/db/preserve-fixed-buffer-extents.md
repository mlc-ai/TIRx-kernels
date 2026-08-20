# Preserve fixed buffer extents when specialization owns them

**Symptoms:** `register_count_gap`, `sass_schedule_divergence`, `instruction_parity_with_deficit`, `slow_small_shape`

## Symptom

A port whose static opcode and address counts match the frozen build still
allocates a different register count and regresses a short shape. The entry
declares fresh symbolic extents for buffers whose exact size the specialization
already fixes.

## What to change

Declare the specialized extents at the entry where the specialization is
already their single source of truth. A fresh symbolic extent is the right
contract for a raw runtime pointer, but it perturbs ptxas when the size is in
fact compile-time.

## Rationale

In a six-token recurrent kernel, restoring the original body and its
1,488-instruction static count while leaving entry buffers symbolic still
allocated 63 instead of 66 registers and measured 1.1792 after/before on the
short row. Static opcode and address counts were therefore not parity.

Declaring the specialized one-dimensional extents at the entry restored the
frozen SASS byte-for-byte for all four dispatched splits, including register
allocations of 128, 132, 132, and 66 with no stack or spill. All 28 correctness
configurations passed.

## Boundary

Keep symbolic extents when sizes are genuinely runtime. Use fixed extents only
when the specialization already owns them, and compare every affected
specialization rather than one shape.

## Verification

Compare registers and final SASS per dispatched split, not only static opcode
and address counts; those can match while the allocation and schedule differ.
