# Take a descriptor's rank from the tensor, not the flag that selects it

**Symptoms:** `illegal_instruction`, `config_specific_buffer_load`, `instruction_variant_mismatch`

## Symptom

A launch dies with an illegal instruction on exactly the specializations where
two independent options are both on -- a new tensor layout and a staging or
alternate-descriptor path -- while each option alone runs fine. Nothing in the
generated code looks wrong at the copy site: it names a descriptor and a rank
that are individually correct.

## What to change

Separate the two questions the site is really asking. The *rank* of a bulk
tensor copy is a property of the tensor's layout. *Which* descriptor it reads is
a property of where the data is staged. When a kernel keeps more than one map
over the same tensor -- an MMA-swizzled one and a plain staging one, say -- every
one of them has to be encoded at whatever rank the layout gave the tensor, or
one combination will pair a map of one rank with a copy of another.

```python
# before: rank branches on the layout flag, the map branches on the staging
# flag, and the staging map was only ever encoded at the old rank.
if layout_adds_a_mode:
    T.evaluate(T.ptx[_COPY_4D](dst, T.address_of(stage_map if staged else main_map), ...))
else:
    T.evaluate(T.ptx[_COPY_3D](dst, T.address_of(stage_map if staged else main_map), ...))

# after: each map is encoded at the rank its tensor has, so whichever one the
# staging flag picks already matches the rank the layout flag chose.
for m, elem_bytes in ((main_map, mma_bytes), (stage_map, raw_bytes)):
    if layout_adds_a_mode:
        _encode(m, rank=4, dims=(D, TILE, HEADS, TILES), strides=(...))
    else:
        _encode(m, rank=3, dims=(D, HEADS, TOTAL), strides=(...))
```

## Rationale

A tiled bulk copy takes its coordinate count from the instruction, and the
descriptor carries its own rank from the encode call. The hardware does not
reconcile them: a 4-D copy against a rank-3 map reads a coordinate the map never
described, and the launch traps. Because each option is exercised on its own by
other configurations, the whole matrix passes except the cells where both are
set -- here two of thirty-four configurations, both of them fp8-staged pages,
which is why the failure surfaced only under a bitwise gate that ran every
config rather than a representative few.

## Boundary

Only applies where a kernel holds more than one descriptor over the same tensor.
With a single map the rank cannot disagree with itself. The trap is worth
looking for whenever a port adds a layout axis late -- paging, batching, an
extra head mode -- to a kernel whose descriptors were written when the layout
had one shape.

## Verification

Check every encode's argument count against `4 + rank * 4 + 3`; a short or long
call is the same mistake caught earlier, since the runtime wrapper derives the
shape, stride, box and element-stride counts from the rank it was handed. Then
confirm each copy site's rank matches the rank of the map it names, for every
combination of the flags, not just the ones a default configuration reaches.
