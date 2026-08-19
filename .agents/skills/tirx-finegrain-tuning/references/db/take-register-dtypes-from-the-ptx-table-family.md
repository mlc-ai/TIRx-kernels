# Take register dtypes from the PTX table family

**Symptoms:** `invalid_ptx_form`, `dtype_mismatch_at_use`, `register_view_friction`

## Symptom

A dtype error at a use site that appears to demand redesigning the data path,
because one register tile is consumed by two instruction families with different
operand types.

## What to change

In the PTX table, operand dtypes are fixed per instruction family, not per use
site, so plan the typed views up front. The casts between them are register
renames and cost nothing in SASS.

```python
# One tile, two typed views across neighbouring instructions: the load
# family's destinations are int32 ...
values_0 = T.alloc_local([16], "int32", align=16)
values_1 = T.alloc_local([16], "int32", align=16)

# ... while add.bf16x2 operands must be uint32. The casts are register
# renames and cost nothing in SASS.
sums = T.alloc_local([4], "uint32")
for w in T.unroll(4):
    T.evaluate(
        T.ptx["add.bf16x2"](
            sums[w],
            T.cast(values_0[j * 4 + w], "uint32"),
            T.cast(values_1[j * 4 + w], "uint32"),
        )
    )
```

Where the reference reads only part of a wide value, match the width rather than
loading wide and converting:

```python
# An i64 index consumed as i32 is a single low-word load in the reference.
idx = T.alloc_local([1], "int32")
T.evaluate(T.ptx.ld.global_.b32(idx[0], topk_i32.ptr_to([index])))
```

## Rationale

Loading i64 and converting is a real instruction-shape divergence, not a
shorthand. Declaring an int32 view of the tensor reproduces the reference's
single low-word `ld.global.b32`.

## Verification

Check the emitted instruction shape, not only that the types compile; a cast
that survives to SASS is a divergence, a rename is not.
