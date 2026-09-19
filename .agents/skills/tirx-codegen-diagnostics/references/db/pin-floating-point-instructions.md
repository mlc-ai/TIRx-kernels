# Pin floating-point instructions

**Symptoms:** `bitwise_mismatch`, `denormal_mismatch`, `unexpected_ftz`, `select_lowered_as_branch`

## Symptom

Bitwise mismatches on denormal inputs; `.ftz` forms appearing where the
reference emits none, or missing where it emits them; a float compare/select
lowered as a branch instead of `selp`, perturbing instruction shape and
scheduling even when normal values agree.

## What to change

When the reference pins an instruction, use the exact PTX operation: non-FTZ
`mul.f32`/`add.f32`, `div.rn.f32`, or explicit `setp` plus `selp`. Retain
`.approx.ftz` only where the reference uses it. Plain TIRx remains appropriate
for integer and index math.

```python
# One helper per pinned instruction; the chain string is the contract.
def _mul(a, b):
    """``mul.f32`` -- non-FTZ, matching the reference."""
    return _ptx_binary("mul.f32", a, b)


def _fma(a, b, c):
    """``fma.rn.f32`` -- non-FTZ."""
    return _ptx_ternary("fma.rn.f32", a, b, c)


def _div_rn(a, b):
    """``div.rn.f32`` -- the reference wants full-precision division here."""
    return _ptx_binary("div.rn.f32", a, b)


def _exp2(a):
    """``.approx.ftz`` retained only where the reference uses it."""
    return _ptx_unary("ex2.approx.ftz.f32", a)
```

A compare/select is pinned as an explicit `setp` plus `selp`, so neither the
codegen nor ptxas chooses the `.ftz` flag or a branch:

```python
def _max(a, b):
    p = T.local_scalar("uint32")
    out = T.alloc_local([1], "float32")
    T.evaluate(T.ptx.setp.gt.f32(p, a, b))
    T.evaluate(T.ptx.selp.f32(out[0], a, b, T.ptx.pred(p)))
    return out[0]
```

Global fast-math off-switches exist for both TVM CUDA compile paths
(`TVM_CUDA_NVCC_NO_FAST_MATH=1` for nvcc, `--ftz=false` via
`TVM_CUDA_NVRTC_EXTRA_OPTS` for NVRTC), but prefer per-op pinning: it holds
regardless of compile defaults and documents intent at the use site.

## Rationale

Two independent mechanisms share this fix. Fast-math defaults (e.g. nvcc
`--use_fast_math`) add `.ftz` to float arithmetic and make division approximate,
causing bitwise mismatches on denormals. Independently, a float compare/select
whose PTX form is unpinned lets the codegen choose whether `setp` carries `.ftz`
and lets ptxas choose between `selp` and a branch.

## Boundary

The direction is a property of the reference, not of the family, and both
families are registered in the PTX table, so the `.ftz` forms are always an
explicit choice. Two siblings ported from a tile-DSL reference emit no `.ftz` at
all and needed non-FTZ helpers to defeat the fast-math build; a third, whose
reference is plain CUDA operators compiled with fast math, emits 108
`fma.rn.ftz.f32`, 69 `mul.ftz.f32`, 53 `add.ftz.f32`, 4 `sub.ftz.f32` and no
plain-`.f32` arithmetic at all. Inheriting a sibling's arithmetic helpers is a
silent divergence in either direction; read the reference's own PTX census
first.

Matching the FTZ flag alone does not justify splitting an FMA into a multiply
and a global FP32 atomic add. An FMA can add a subnormal product to a normal
accumulator without materializing that product; the atomic flushes its
subnormal input. A measured ten-term reduction with every expert value equal
to `2**-124` and weights `[0.5, 1/18, ..., 1/18]` returned approximately
`2**-124` with `fma.rn.ftz.f32`, but only `2**-125` with a separately computed
product and `red.add.f32`. Preserve the fused weighted reduction when this
range is part of the numerical contract; fast-math reference flags do not
authorize the split.

Widening the accumulator is insufficient if its input conversion still uses
fast-math defaults. In the same experiment, an ordinary FP32-to-FP64 cast lost
the small contributions before `red.add.f64`; explicitly emitting
`cvt.f64.f32` preserved them and recovered approximately `2**-124`. Pin the
conversion as well as the arithmetic, and inspect the production compiler
path rather than a separately compiled non-fast-math binary.

## Verification

Denormal inputs plus an instruction-by-instruction PTX comparison.
