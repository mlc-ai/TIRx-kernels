# Take an instruction's modifiers from the reference's PTX

**Symptoms:** `instruction_variant_mismatch`, `unverified_instruction_selection`, `sass_divergence`, `instruction_count_gap`

## Symptom

Same opcode family, same issue count, different instruction: a modifier a
builder derived from kernel-level properties that the reference never selects.
Nothing in a count exposes it.

## What to change

Read each modifier off the reference's own dumped PTX and off the literal
arguments at its call sites, not off the helper's source or off your model of
the algorithm, and check any modifier a builder derives automatically against
that same dump. A derived modifier belongs on the chains the dump shows it on,
and nowhere else:

```python
# The matrix and copy chains carry the cluster modifier ...
mma_chain = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.scale_vec::1X"
utccp_chain = f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"

# ... but the bulk load does not: every reference call site passes a literal 1,
# so each CTA fetches its own slice and the pairing lives in the matrix
# instruction, not in the load.
load_chain = (
    f"cp.async.bulk.tensor.{rank}d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
)
```

Note any modifier containing `::` requires the bracket form `T.ptx[chain](...)`
rather than attribute access.

## Rationale

One port appended a two-CTA modifier to every bulk tensor load because the
cluster held more than one CTA. The reference's copy helper does contain a
two-CTA branch, but every call site passes a literal one and never reaches it --
the pairing lives in the matrix instruction, not in the load, and each CTA
fetches its own slice. The two variants signal completion differently. Dropping
the modifier left correctness unchanged and took the kernel from 55.64M to
54.24M instructions, with the tensor pipe active 74.4% of cycles against the
reference's 73.3%.

## Boundary

A branch present in a reference helper is not evidence that it is taken.

## Verification

Diff the modifier against the reference's dumped PTX at each call site; confirm
correctness and the instruction count after dropping a derived modifier.
