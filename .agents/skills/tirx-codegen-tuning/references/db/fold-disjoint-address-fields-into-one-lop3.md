# Fold disjoint address fields into one LOP3

**Symptoms:** `excess_address_math`, `excess_integer_math`, `latency_bound_epilogue`

## Symptom

A swizzled shared-memory address is built as an add followed by an XOR, and the
two instructions sit between a barrier wait and the load that depends on it.

## What to change

Where the address terms occupy disjoint bit fields, OR them instead of adding,
and derive the twist from the invariant part alone. `base | (chunk ^ twist)` is
the three-input LOP3 the hardware already has; `(base + chunk) ^ twist` is two
instructions because the compiler cannot prove the carry is impossible.

```python
# before: the trailing byte scale hides the bit structure.
addr = warp_base + (column * ROWS + (chunk * 4 ^ swizzle)) * 4

# after: the scale folded in, so the twist lives in bits 4..6 and the base
# starts at bit 7 -- disjoint, and the chunk ORs in.
column_base = warp_base + column * ROW_BYTES
addr = column_base | (chunk * 16 ^ swizzle)
```

The same folding applies to a lane field: a term masked into high bits ORs into
a low-bit sum instead of adding to it, which turns a mask, an add and a move
into one instruction.

```python
# before: mask, then add.
address = tmem_base + (tid << 16 & 0xE00000) + column

# after: one LOP3.
address = (tid << 16 & 0xE00000) | (tmem_base + column)
```

## Rationale

The address is bit-for-bit identical, so this is free of any numerical
argument. What it buys is out of proportion to its size: sixteen instructions
removed from a kernel executing six million bought 1.6%, because they stand
between a barrier wait and the shared load that depends on it. The lane-field
fold added a second effect -- eleven per-thread operations moved onto the
uniform datapath, which issues on a separate scalar pipe.

## Boundary

The disjointness is arithmetic, not a heuristic. Every caller has to pass a base
aligned to the field width, and the twisted field has to fit inside that
alignment; verify both before rewriting, because an unaligned caller silently
reads the wrong bytes. The payoff also scales with how many terms share one
base: a row wide enough for eight chunks shows it and a two-chunk row does not.

## Verification

Diff the LOP3 and integer-add counts, and confirm the addresses are unchanged by
comparing the emitted offsets, not by re-deriving them.
