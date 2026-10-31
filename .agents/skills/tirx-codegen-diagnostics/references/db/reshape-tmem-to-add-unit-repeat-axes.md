# Reshape TMEM to add unit repeat axes

**Symptoms:** `tile_lowering_failure`, `tmem_layout_mismatch`

## Symptom

A matrix tile instruction rejects its accumulator with `expected a TMEM image`,
although its non-unit dimensions map rows to `TLane` and columns to `TCol`.

## What to change

Declare the physical two-dimensional TMEM image first, then add the unit
instruction-repeat axes with a view.

```python
image = T.Tensor(
    (128, columns), dtype, scope="tmem",
    layout=TileLayout(S[(128, columns): (1 @ TLane, 1 @ TCol)]),
)
tile = T.bind_(T.cuda.decl_tmem(column, ty_args=[image]))
operand = tile.view(1, 1, 128, columns)
```

Avoid adding these axes with unqualified numeric zero strides in the physical
layout. A numeric stride belongs to the memory axis; even a zero-valued memory
coordinate can violate the TMEM instruction pattern's axis contract.

## Rationale

A four-dimensional declaration with strides `(0, 0, 1@TLane, 1@TCol)` failed
TMEM operand validation. Declaring the two-dimensional image and reshaping it
removed the unwanted memory axis. The resulting attention implementation passed
32 configurations against its instruction-level reference bitwise and against
an independent FP32 reference.

## Boundary

This fixes a layout contract, without establishing performance equivalence.
Use the appropriate datapath layout when the MMA does not use the full 128-row
identity image, and retain intentional replicas.

## Verification

Check the physical axis set and row/column mapping, compile the tile instruction,
and run numerical and paired timing checks separately.
