# Declare the block shape the reference declares

**Symptoms:** `special_register_reads`, `instruction_count_bloat`, `slow_latency_bound_shape`, `launch_config_drift`, `instruction_variant_mismatch`

## Symptom

Excess dynamic special-register reads (`S2R`, `S2UR`) and instruction bloat on
latency-bound shapes.

## What to change

Take the block shape from the reference, not from a scaffold's correspondence
table. When the reference launches a flat block and derives warp, lane, and
group indices by division, declare the flat block and derive the indices.

```python
# before: a 2-D declaration costs a special-register read per component,
# at every use, for indices the reference derives by division.
lane, warp = T.thread_id([32, WARPS])

# after: one flat block, indices derived.
tid = T.thread_id([THREADS])
warp: T.int32 = _make_warp_uniform(tid // 32)
lane: T.int32 = tid % 32
lane_quad: T.int32 = lane % 4
```

Match the strength of the declaration too. A static thread extent ordinarily
lowers through a launch bound and permits any block no larger than `.maxntid`.
When fresh reference PTX instead requires `.reqntid`, request the exact block
contract explicitly.

```python
# before: the extent can still lower to __launch_bounds__ and .maxntid.
tid = T.thread_id([THREADS])

# after: require the declared thread and cluster extents exactly.
T.attr({"tirx.required_block_size": 1})
tid = T.thread_id([THREADS])
```

## Rationale

A multi-dimensional `thread_id` costs a special-register read per component at
every use, so a 2-D declaration adds nothing over the reference's flat launch
and is charged for on every access. One latency-bound kernel declared a
`(32, warps, 1)` block where both the reference and the approved sketch
specified a flat one. It cost 4608 dynamic `S2R` plus 4608 `S2UR`; flattening
moved the worst shape from 0.864x to 1.016x and removed 48 static instructions,
all in the affected phase.

The exact-block case is a separate contract, not a stronger spelling of the
same launch bound. A CUDA 13 probe showed that `__block_size__` changed
`.maxntid` to `.reqntid`, but also emitted `.reqnctapercluster` and
`.blocksareclusters`. The matching runtime therefore launched with the CUDA
required-block sentinel, omitted a duplicate runtime cluster attribute, and
converted the logical CTA grid to a cluster grid. Cluster-1, cluster-2, and
cluster-16 correctness all passed, as did 1,025 source-domain configurations;
the complete measured matrix retained 1.018-1.046x reference/port ratios.

## Boundary

Use the required-block contract only when fresh reference PTX requires exact
dimensions. It needs CUDA 13 or newer, static thread and cluster extents, and a
runtime that understands CUDA's cluster-grid launch semantics. Do not replace a
register-budget launch bound with it: `.maxntid` and `.reqntid` promise different
things, and the measured result above establishes source-contract matching, not
an isolated speedup from strengthening the directive.

## Verification

Confirm flat-versus-dimensional declarations with dynamic special-register
counts on the worst shape. For an exact declaration, inspect final PTX for
`.reqntid`, inspect the accompanying cluster directives, and launch guards at
cluster 1 and greater than 1 before accepting the performance matrix.
