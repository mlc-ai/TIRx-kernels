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

The win is not always the uniform datapath, and reaching for it a second time
can cost. Broadcasting a warp index and a cluster rank dropped 56 instructions
and 5 registers with the uniform opcode count flat at 363 to 362 -- ptxas
simplified the warp predicates once the invariance was stated, and moved nothing
onto the scalar pipe. Extending the same broadcast to a loop counter that ptxas
already knew was warp-invariant added 8 instructions and increased no U-prefixed
opcode at all. Before assuming a reference keeps a value uniform, read its own
machine code at the site: one that holds an index in a uniform register may
still move it into a per-thread register to scale it, in which case its
advantage there is precomputation, not the datapath.

The proof costs a `__shfl_sync`, and it is only repaid when there is
vector-datapath work for it to move. Applying it to a shared-memory base and a
tensor-memory base whose derived addresses were already reloaded into registers
per warp role measured neutral to slightly negative: +0.7%, +1.0%, +0.0%, +0.4%.
Confirm from the generated code that uniform-datapath instructions actually
replace vector ones before keeping the hint.

A CTA reduction result is not necessarily a missing compiler proof. Broadcasting
a `bar.red.popc` result before a whole-CTA early return added one shuffle but
left branch/reconvergence counts, a 16-byte stack and 26/36 local load/store
sites unchanged. Four same-workspace paired profiles retained every output
byte; the two active ordinary paths regressed 0.27–0.67%, while inactive-path
controls were effectively flat. Uniform source semantics alone did not make
this extra broadcast useful.

A producer-aggregated entry word supplied a second negative result: adding a
uniform broadcast to its load, or replacing the ordinary load with `ldu` in a
consumer that never writes the word, produced identical final SASS. Both kept
the same vector load and predicated exit, 168 registers, a 16-byte stack, and
25/36 local load/store sites. The source hint was not a new implementation to
benchmark. Read-only and identical-address semantics permit `ldu`; they do not
require the assembler to use a different hardware datapath.

Uniform fallback selection also cannot repay an expensive detector or duplicated
epilogue by itself. One measured 96-term risk predicate followed by a single
warp ballot passed all targeted numerical cases but regressed the complete long
workload by 27.31%. Replacing it with packed BF16x2 abs/max reductions and one
outer uniform branch still regressed 12.74%. The branch was warp-uniform in both
cases; scanning inputs and retaining two full epilogues remained the dominant
cost. Treat uniformity as a code-generation property, not evidence that a
runtime specialization is cheap.

## Verification

Confirm vector versus uniform op counts, branch topology, registers, and the
full workload matrix. Sweep the geometry where the guard actually excludes
warps.
