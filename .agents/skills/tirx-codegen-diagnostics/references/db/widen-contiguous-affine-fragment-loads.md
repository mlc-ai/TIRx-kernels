# Widen contiguous affine fragment loads

**Symptoms:** `scalar_global_loads`, `lg_throttle`, `vectorizable_affine_fragment`, `instruction_count_gap`

## Symptom

A normalization epilogue loads a compile-time contiguous affine fragment into
per-thread registers with one scalar global instruction per element.  The
fragment base and extent prove every element is in bounds and aligned, but the
final SASS does not coalesce the scalar source operations.

## What to change

Load each complete aligned fragment with one explicit wide transaction, then
unpack its raw words into the existing scalar registers.  Keep the scalar
predicated path for shapes whose fragment has a tail or whose base alignment is
not structural.

```python
if full_aligned_fragment:
    words = K.alloc_local([4], K.u64)
    K.ptx.ld.global_.v4.b64(
        words[0], words[1], words[2], words[3], source.ptr_to([fragment_base])
    )
    for pair in K.unroll(4):
        K.ptx.mov.b64(values[pair * 2], values[pair * 2 + 1], words[pair])
else:
    for value in K.unroll(FRAGMENT):
        with K.If(fragment_base + value < extent), K.Then():
            K.ptx.ld.global_.b32(values[value], source.ptr_to([fragment_base + value]))
```

## Rationale

For one 16,384-column BF16 normalization, each thread owned two fully valid
eight-FP32 gamma fragments and two beta fragments.  Widening those loads reduced
static global-load sites from 34 to 6 and total static SASS from 240 to 216.
Dynamic warp instructions fell from 929,792 to 827,392, LG-throttle samples
from 819 to 21, and MIO-throttle samples from 119 to 61.  Allocation remained
64 registers per thread with zero local traffic.

All 23 correctness configurations passed.  The four affected row-counts each
beat the external reference in both implementation orders; their combined
ratios ranged from 1.1211x to 1.6153x, while the original failing row measured
1.1325x and 1.1380x in two 15-round directions.

## Boundary

The transaction width must follow from the compile-time fragment layout.  Do
not widen a predicated tail, a runtime-strided fragment, or a base that is only
accidentally aligned in one allocation.  A wide per-thread access can also
reduce coalescing when neighboring threads do not own adjacent fragments, so
verify sectors and timing rather than accepting the lower instruction count
alone.  Scope the path by the same target and shape predicate that proves the
alignment.

## Verification

Inspect final SASS for the intended wide loads, then compare static and dynamic
load instructions, requested sectors, LG throttle, registers, and local
traffic.  Run every affected row count plus the complete correctness matrix;
measure both implementation orders on the formerly failing shape.
