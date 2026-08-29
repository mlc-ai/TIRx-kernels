# Contract and reassociate across the inline-asm boundary

**Symptoms:** `inline_asm_boundary`, `instruction_count_bloat`, `slow_epilogue`

## Symptom

An elementwise chain transcribed faithfully from the reference retires more
arithmetic than the reference's own machine code, even though both compute the
same expression with the same operations written down.

## What to change

Where the arithmetic reaches ptxas as inline assembly, write the contracted and
reassociated form by hand: ptxas will not fuse a multiply that feeds an add into
an FMA across an asm boundary, and it will not reassociate around a shared
subexpression.

```python
# before: transcribed as the reference writes it -- a multiply, then an add.
ops["product"](inner, work, inner)
ops["offset"](inner, inner, 1.0)

# after: one instruction.
ops["fused"](inner, work, inner, _spread(1.0))


def _fused(out, left, right, addend):
    """``left * right + addend``; the reference's compiler contracts this."""
    K.ptx["fma.rn.f32x2"](
        packed,
        K.cuda.make_float2(left[0], left[1]),
        K.cuda.make_float2(right[0], right[1]),
        K.cuda.make_float2(addend[0], addend[1]),
    )
    K.ptx["mov.b64"](out[0], out[1], packed)
```

Reassociation is the same job by hand: pull the shared subexpression out, and
fold a terminal multiply into the accumulator it feeds rather than materialising
the product.

```python
# before: two products materialise a value only the accumulator reads.
ops["product"](term, step, gate)
ops["binary"]("add", accum, accum, term)

# after: the accumulation absorbs the multiply.
ops["fused"](accum, step, gate, accum)
```

## Rationale

Read the reference's machine code, not its source: one activation was written as
thirteen packed multiplies and three adds and its compiler retired eleven
multiplies, two adds and one FMA. Writing that form directly removed two packed
multiplies and folded one add into an FMA per element pair, exactly as
predicted, in a sixteen-times-unrolled loop. Extending the same audit to the
other two activations and to every accumulator fold removed 64 and 32
instructions per subtile and took the family that had the most contraction sites
from 0.986 to 1.021 at its smallest shape and 0.988 to 0.999 at its largest. The
gain tracks the number of sites, which is what makes the mechanism credible.

The boundary is asymmetric, and worth knowing before hunting: ptxas *does*
common-subexpression-eliminate two identical asm blocks. Deleting a provably
redundant duplicate of an expression changed nothing at all.

## Boundary

An FMA rounds once where the multiply and the add round twice, so the result
moves within the tolerance the reference comparison already allows -- confirm
the comparison is a tolerance and not an exact match before contracting. Where
a reference pins non-contracted instructions deliberately, this entry does not
apply; check whether its own machine code carries the FMA first.

## Verification

Count packed multiplies, adds and FMAs per unrolled element on both sides. The
saving should be an exact multiple of the unroll factor; if it is not, the
transcription and the reference are not computing the same expression.
