# Issue shared vector loads explicitly

**Symptoms:** `slow_epilogue`, `register_pressure`, `performance_regression`, `instruction_count_gap`

## Symptom

A shared-memory read-back issues scalar `ld.shared.b32` in groups whose
addresses are contiguous and aligned, and ptxas fuses them into `LDS.128` on
some builds of the same kernel and not others.

## What to change

Issue the vector form yourself wherever the run is contiguous and the base is
aligned, instead of leaving the fusion to ptxas.

```python
# before: four scalar loads per chunk, fused only sometimes.
for j in range(4):
    K.ptx.ld.shared.b32(quad[j], smem.ptr_to([base + j * 4]))

# after: the width is structural.
K.ptx["ld.shared.v4.b32"](quad[0], quad[1], quad[2], quad[3], smem.ptr_to([base]))
```

Apply it to the narrow cases too: two adjacent 8-byte-aligned values are one
`ld.shared.v2.b32`, not two loads that may or may not pair.

## Rationale

In one epilogue's transpose read-back this took bare `LDS` from 69 to 9 and
`LDS.128` from 13 to 28, and both rows that ran the reduction improved with
their absolute times falling, while the control row that generates no reduction
did not move at all.

The reason to make it structural is the measured mirror image. An address
rewrite on the *store* side of the same buffer, derived as an exact bit identity,
cut 144 SASS instructions, 45 LOP3 and 12 registers -- and regressed 10-15% on
every row that ran it, with the same control row flat. Its shorter addresses had
stopped ptxas recognising the group, so the reads went back to scalar. Fusion
that a compiler performs under one register budget is not a property you can
rely on after touching register pressure anywhere nearby.

## Boundary

Only where the run really is contiguous and the base alignment is guaranteed by
construction, not by the shape that happens to be under test. A wider load also
ties more results into one register group, so on a path that is already at its
register ceiling the wider form can cost more than the fusion saved.

## Verification

Count bare and vector shared accesses separately in SASS rather than looking at
the instruction total, and keep a shape in the set that does not execute the
region at all -- if it moves, the effect is not the one being attributed.
