# Materialize a warp collective before divergence

**Symptoms:** `kernel_deadlock`, `collective_in_divergent_branch`, `repeated_expression`, `partial_active_mask`

## Symptom

A kernel hangs at a use site inside an elected-lane or otherwise divergent
region, where the value being used is an alias for an expression that contains a
warp collective. The collective appears once in the source.

## What to change

Materialize the collective's result once, before divergence, and use the
materialized value everywhere else.

```python
# before: an alias for the shuffle; the tree is re-emitted at every use, so a
# use under an elected-lane guard runs the collective with one lane active.
warp_id = _shuffle_full_mask(raw_warp)

# after: one collective, executed with the full warp, then reused as a value.
warp_id = K.local_scalar("int32")
K.assign(warp_id, _shuffle_full_mask(raw_warp))
```

Reuse the entry's already materialized value, bind the collective before the
guard, or give each site its own local. Never let the subtree be rebuilt under a
partial active mask.

## Rationale

Expressions are trees, and reusing one can emit its complete subtree at every
use. When the subtree contains a warp collective, that is a correctness
boundary, not a size question.

One traced kernel reused a full-mask shuffle that established its warp id.
Inside an elected-lane region the tree was emitted again with only one of 32
lanes active and the kernel deadlocked. Materializing the warp id passed bitwise
stage-4 and stage-8 checks with PDL both disabled and enabled, followed by all
eight correctness workloads.

A recurrent KDA prefill hit the same boundary across 21 triangular pivot
shuffles: plain aliases deadlocked wherever their consumers were divergent,
while one local per pivot, one reused local, and `K.Bind` placed before the
guard all passed the six-case correctness matrix.

## Boundary

The fix is a correctness fix, not a performance lever, and it does not explain a
regression that survives it. On the KDA prefill above, those three
materialization forms measured about 552/484/296 us -- effectively identical --
against same-GPU baselines of 498/431/267 us, so the locals' simultaneous live
ranges were not the remaining 10-12% deficit. Line-mapped SASS instead found
five `STL` plus five `LDL` in a different prep-role address block and 23 versus
8 `S2R` instructions; attribute stack traffic to its generated-source block
before changing how a collective is materialized.

Paired NCU reports sharpened that conclusion. The port kept the baseline's
64-register allocation and added only 1,052,219 dynamic instructions (+0.45%);
its 172,032 local loads and 9,600 local stores were 0.08% of the 234.9
million-instruction stream, while duration rose 9.79%. The material difference
was scheduling: issue activity fell from 0.6681 to 0.6059 warps/cycle, eligible
warps from 1.4052 to 1.0653, and long-scoreboard, short-scoreboard, and wait
ratios rose by 0.6502, 0.5537, and 0.4193, with occupancy unchanged and little
barrier-stall movement. PC sampling placed the dominant long-scoreboard waits
immediately after the same asynchronous barrier-wait pattern on both sides. Do
not treat a small spill count as the cause just because the baseline has zero:
scale it against the full dynamic stream and inspect the load-consumer schedule
after pipeline waits. Re-expressing warp and lane from one thread id yielded no
reproducible timing win, so identifier syntax alone was not the missing schedule
constraint.

## Verification

Read the generated code at every use site of the alias and confirm the
collective appears exactly once, at full-warp scope. Then run the divergent
configurations, which are the ones that hang.
