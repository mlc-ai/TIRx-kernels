# Do not decompose an index the address recombines

**Symptoms:** `excess_address_math`, `excess_integer_math`, `latency_bound_fixup`, `slow_latency_bound_shape`

## Symptom

A reference decomposes a flat index with a FastDivmod and the port transcribes
it faithfully, yet the address built from the quotient and remainder weights
them by exactly the factors the divmod divided by. The decomposition is
multiplied straight back out, and its seven-instruction magic-multiply chain
sits on the address that gates a load or store.

## What to change

Check the identity before transcribing the divmod. Where a flat index splits as
`idx = outer * N + inner` and the address recombines the parts as
`(base + outer) * N * W + inner * W`, that is `base * N * W + idx * W`: the
address never needs the parts separated.

```python
# before: decompose, then weight the parts by the same factors that split them.
outer, inner = _divmod(idx, N)
addr = (base + outer) * stride + inner * W    # stride == N * W

# after: the flat index carries both terms already.
addr = base * stride + idx * W
```

Keep the divmod only where a consumer genuinely reads the parts apart -- a
per-row count indexed by `outer`, a predicate on `inner`. Drop it at every site
that only re-weights them.

## Rationale

The transcription is faithful and still redundant: the reference computes the
decomposition because other statements in the same region consume it, while the
port's copy at this site feeds nothing but the recombination. Removing it at one
staging address and then at two store addresses took per-thread divmods from 14
to 9 with load and store counts unchanged, and shrank every specialization's
generated code by 3.5-3.8 KB.

The gain is not where instruction count predicts. The chain removed is short,
but it is serial and it gates address formation, so the shapes it frees are the
ones with least else to hide it behind. Nine shapes carried the same change: the
four that had been below the gate crossed together -- 0.9855, 0.9778, 0.9667 and
0.9679 to 0.9992, 1.0004, 1.0142 and 1.0134 -- while the shapes already passing
moved little. Four shapes crossing at once, on a change that touches no memory
access, is the signature of a per-thread scalar cost rather than a shape
effect; a fix that helps only one shape in a family is usually not this.

## Boundary

The identity is arithmetic, not a heuristic: verify it holds for the actual
strides before applying it. It fails as soon as the stride is not the divisor
times the element width -- a padded row, a swizzled layout, or a permuted
component breaks the factorization, and the decomposition is then load-bearing.

## Verification

Count divmod expansions in the generated code and confirm the drop equals the
number of sites removed, with load and store counts unchanged. A store count
that moves means an address changed rather than an equivalent one was formed
more cheaply.
