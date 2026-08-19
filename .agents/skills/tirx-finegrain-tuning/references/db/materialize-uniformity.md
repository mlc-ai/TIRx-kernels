# Materialize uniformity

**Symptoms:** `warp_retry_region`, `vectorized_uniform_math`, `excess_address_instructions`

## Symptom

WARPSYNC/ENDCOLLECTIVE retry or collective regions, or vector integer/address
work where every participating lane holds the same value.

## What to change

Materialize a warp-uniform proof or broadcast only after proving the value is
identical for every active lane and mask.

```python
def _make_warp_uniform(value):
    """Semantically the identity -- every lane already holds this value --
    but it is the hint that lets ptxas prove the guard it feeds is
    warp-uniform."""
    return T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), value, 0, 32)


# before: ptxas cannot prove the guard below is warp-uniform.
warp: T.int32 = tid // 32

# after:
warp: T.int32 = _make_warp_uniform(tid // 32)
```

The explicit PTX spelling, where the exact operands matter:

```python
out = T.alloc_local((1,), "uint32")
T.evaluate(
    T.ptx.shfl_sync.idx.b32(
        out[0], T.reinterpret("uint32", value), T.uint32(0), T.uint32(31),
        T.uint32(0xFFFFFFFF),
    )
)
```

Do not drop a reference instruction that looks like an identity: a broadcast of
a value every lane in the warp already holds may be the only evidence ptxas has
that the guard it feeds is warp-uniform.

## Rationale

If every participating lane holds the same value but control flow hides that
fact, ptxas may emit vector integer/address work plus retry or collective
regions. Measured cases removed WARPSYNC/ENDCOLLECTIVE regions and tens of
thousands of vector address instructions.

One omitted identity broadcast wrapped a phase's reductions in a
WARPSYNC/ENDCOLLECTIVE retry region whose back-edge spanned most of the kernel,
adding nine shuffles to the reference's count and fourteen warp syncs to its
zero; reinstating the broadcast restored the reference's exact shuffle count and
moved two shapes from 0.997x to 1.003x and 1.000x.

## Boundary

Whole-function complexity can still determine uniform placement, so a local
rewrite is not guaranteed to flip codegen. The cost appears only where the guard
is live, so a specialization launching exactly the guarded warp count shows
nothing and a wider one pays.

## Verification

Confirm vector versus uniform op counts, branch topology, registers, and the
full workload matrix. Sweep the geometry where the guard actually excludes
warps.
