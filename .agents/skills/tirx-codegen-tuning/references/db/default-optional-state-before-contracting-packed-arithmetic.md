# Default optional state before contracting packed arithmetic

**Symptoms:** `instruction_variant_mismatch`, `branch_in_hot_loop`, `sass_divergence`, `predicated_destination`

## Symptom

A packed recurrence computes `output = left * right - state`, but `state` is
absent on the first iteration. Guarding the subtraction leaves a multiply plus
a conditional subtract in the hot path instead of the reference's packed FMA.
Explicitly negating the optional state before asking for an FMA can be worse:
the negation may become another packed arithmetic instruction.

## What to change

Initialize the packed state carrier to the additive identity, overwrite it only
when state is present, and keep the multiply-subtract chain unconditional. This
keeps only state production under control flow and presents one straight-line,
contractible arithmetic chain to code generation.

```python
# before: optional state also makes the hot arithmetic conditional.
_mul_pair(output, left, right)
with K.If(have_state), K.Then():
    state_pair = _pack_pair(state_lo, state_hi)
    _sub_pair(output, output, state_pair)

# after: the inactive path contributes the exact additive identity.
state_pair = K.local_scalar("uint32", init=K.uint32(0))
with K.If(have_state), K.Then():
    K.assign(state_pair, _pack_pair(state_lo, state_hi))
product = K.local_scalar("uint32")
_mul_pair(product, left, right)
_sub_pair(output, product, state_pair)
```

## Rationale

One attempted direct packed FMA formed a negative addend first. ptxas emitted a
separate `HFMA2` for every explicit negation before the requested `HFMA2`, so
the hot path still contained two arithmetic instructions and the candidate was
discarded before timing.

Defaulting the optional packed state to zero and leaving the multiply-subtract
chain unconditional instead produced eight packed `HFMA2` instructions for
the eight recurrence pairs, without an arithmetic branch around them. The
focused ratio moved from 0.9304x to 0.9341x. All 14 correctness configurations
passed, and the rewrite remained in the later complete performance-matrix
winner.

## Boundary

The inactive value must be the exact additive identity for the packed format,
and contracting the multiply and subtraction must match the reference's
rounding semantics. Keep loads and conversions that are invalid without state
inside the predicate; only their defined packed result is defaulted. Do not
introduce a separate packed negation unless final SASS proves it folds into the
desired instruction.

## Verification

Test both the state-absent first iteration and recurrent state-present
iterations with asymmetric values. In SASS, check for one packed FMA per pair,
no preceding negate instruction, and no new reconvergence around the arithmetic.
Then run the affected correctness and performance matrices.
