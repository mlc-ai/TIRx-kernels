# Narrow every operand a shortened block reaches

**Symptoms:** `partial_output`, `unported_work_reduction`, `bitwise_mismatch`, `slow_epilogue`

## Symptom

Output that is partly correct, and a specialization the reference treats
specially running measurably slower than the padded sibling it otherwise shares
code with, for no visible reason.

## What to change

When a reference shortens its last tile to the rows that exist, port every
operand that encodes the shortening, together. Three operands carried it in one
grouped layout: the loader's per-CTA coordinate, which shifts by the shortened
height while the transfer box keeps its compile-time size; the matrix
instruction descriptor's row-count field; and the number of epilogue stores.

```python
# 1. The loader coordinate shifts by the shortened height; the box does not.
row_coord = row_base - (TILE_ROWS - ROWS)

# 2. The descriptor's row-count field, either as a literal bit-field ...
instr_desc: T.uint32 = T.uint32(0x08110910 if ROWS == 128 else 0x04110910)

# ... or as the encoder's M field.
INSTR_DESC = encode_instr_descriptor_block_scaled_uint32(
    M=umma_m, N=umma_n, K=UMMA_K, d_dtype="float32", cta_group=cta_group
)

# 3. The epilogue store count.
for i in T.unroll(ROWS // ROWS_PER_STORE):
    ...
```

## Rationale

The work reduction is encoded in several operands at once, and they have to move
together. With only the last two ported, the peer CTA read rows at the
unshortened offset and just the first half of each block came out correct --
which reads like a transfer-shape bug rather than a missing narrowing. With all
three, correctness was restored and the shape moved from 0.964x to 0.996x,
against 0.997x for the zero-padded sibling that shares its code.

## Boundary

Size the expectation before measuring: the saving here is about one partly-empty
block per group.

## Verification

When a specially-treated specialization is slower than its code-sharing sibling,
look for a work reduction transcribed at some of its sites and not the rest.
