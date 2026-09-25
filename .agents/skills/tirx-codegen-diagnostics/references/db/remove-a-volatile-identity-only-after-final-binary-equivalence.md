# Remove a volatile identity only after final-binary equivalence

**Symptoms:** `volatile_identity`, `inline_asm_boundary`, `sass_equivalence`, `performance_regression`

## Symptom

An inline `asm volatile("mov.u32")` identity in the reference that may or may not
be an intentional optimization barrier. Source form alone does not say which.

## What to change

Replace only the identity with the ordinary typed PTX move, keeping the
surrounding issue order unchanged.

```python
# before: an opaque source helper wrapping asm volatile("mov.u32").
value = _source_opaque_helper(x)

# after: the ordinary typed move, operands kept at the instruction's own width.
out = T.alloc_local([1], "uint32")
T.evaluate(T.ptx.mov.u32(out[0], x))
```

Where the consumer uses a signed view, keep the `.u32` instruction operands as
`uint32` and add a same-width cast for the consumer rather than weakening the
PTX table's dtype contract.

## Rationale

For one paged MQA kernel, replacing two volatile source helpers with
non-volatile typed `mov.u32` produced byte-identical SASS at both affected
bench-suite configs and retained 168 registers with zero stack and local memory.
On one fixed B200 using five rounds and a one-second cooldown, volatile/plain
times were 4.459290/4.459954 us (ratio 0.999851) and 6.350161/6.349622 us (ratio
1.000085); correctness was unchanged.

## Boundary

That result proves those identities were redundant for that lowering, not that
volatile asm is generally redundant. If SASS, resources, correctness, or a
reproducible wall-time gate changes, retain opacity through a general backend
primitive rather than a workload-specific source helper.

## Verification

Compare correctness, final SASS, resources, and fixed-runner wall time before
keeping the removal.
