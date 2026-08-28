# Sweep register budgets by role and shape

**Symptoms:** `register_spill`, `excess_address_math`, `low_occupancy`, `register_budget_mismatch`, `schedule_regression`

## Symptom

Spills, address hoisting, or occupancy loss that shifts across shape regimes or
warp roles.

## What to change

Sweep neighboring register budgets per warp role on representative single-wave
and multi-wave shapes. The budget is the first statement of each role branch,
and the increases must be paid for by matching decreases elsewhere.

```python
if warpgroup_idx == 0:  # compute and epilogue
    T.ptx.setmaxnreg.inc.sync.aligned.u32(144)
    ...
elif warpgroup_idx == 1:  # producer
    T.ptx.setmaxnreg.dec.sync.aligned.u32(96)
    ...
else:  # matrix-issue role
    T.ptx.setmaxnreg.inc.sync.aligned.u32(168)
```

Note `setmaxnreg.sync.aligned` is a four-warp collective: every warp of the
warpgroup must reach it, including otherwise idle ones.

Re-run the sweep after changing descriptor placement, fragment width, or other
live ranges.

## Rationale

Producer, compute, and epilogue warps can need materially different register
budgets. Compiler register level can also trade spills, address hoisting, and
occupancy differently across shape regimes.

One four-warp epilogue sweep was sharply non-monotonic. Requested budgets 48,
56, and 64 passed 1/5, 4/5, and 2/5 targeted rows respectively. Applying 56 to
every output later drove one packed-output specialization to a ptxas allocation
failure at 255 entry registers and regressed several non-singleton paths.
Keeping 56 only on its measured single-cluster BF16 beneficiary and restoring
native allocation elsewhere removed the compile cliff and recovered every FP8
output guard.

A separate FP32 path was limited to one CTA per SM by shared memory and was
long-scoreboard dominated, so raising its epilogue budget from 56 to 64 could
not reduce occupancy. The scoped 64-register form passed all 12 affected FP32
rows on two GPUs, retained zero failures across the correctness matrix, and
helped the final 66-row suite clear its strict gate at a 0.9907x minimum. The
same value had regressed other outputs when applied globally.

Budget redistribution can matter even when total allocation and spill counts
stay fixed. One 16-warp pipeline kept its collective total at 2048 registers
while changing three role budgets from 192/80/48 to 192/88/40. Both forms had
zero local and shared spilling, but the correction-preserving allocation reduced
elapsed cycles from 23,294 to 22,896 and executed instructions from 11,406 to
11,377. It subsequently passed complete targeted and full matrices at 0.995x
and 1.002x minimum ratios.

Instruction count can point at the same role-budget problem from the opposite
direction. One seven-warp persistent pipeline already executed 5,276 fewer
basic-opcode thread instructions than its reference, yet used 72 registers per
thread against 89 and executed 9,269 more synchronization try-wait warp events;
the accumulator-empty wait alone ran 12,926 times against 6,532. Giving only
the four-warp epilogue role an 88-register temporal target moved the directly
affected ratio from 0.9804x to 1.0134x. All 35 correctness configurations and
seven affected or guard configurations passed, and the complete 28-row matrix
cleared at a 0.9908x minimum; an equivalent final build repeated the complete
matrix at a 0.9932x minimum. Once the target already has the shorter instruction
stream, a register deficit plus excess consumer-release polling is a reason to
sweep the consumer role rather than keep deleting arithmetic.

## Boundary

An occupancy proof only shows that a larger budget is affordable; it does not
show that its schedule is better. Scope a budget by the compile-time role,
fragment/output family, cluster regime, and resource limit that produced the
measured response. Let unrelated paths use native allocation when a shared
budget changes their entry allocation or trips an allocator limit.

Two constraints bind before any sweep, and only the first is usually checked.
The budget is warpgroup-uniform, so roles sharing a warpgroup cannot move
independently. The budget must also balance globally: increases are funded only
by what the decreasing roles release. Against a 96-register default, one kernel's
decreasing roles released 14,336 registers for increasing roles that required
12,288; raising one producer role's budget dropped the released pool to 11,264,
and the increasing roles then blocked forever inside `setmaxnreg.inc`. It
compiled cleanly and hung at runtime -- a shape that normally finished in 0.034 s
did not complete in 240 s. Compute the release-versus-require arithmetic for the
whole kernel, not for the role being changed.

Check whether registers bind at all before sweeping. Zero local and zero shared
spilling rules out an obvious capacity failure, but does not prove that role
budgets are schedule-insensitive: redistribution at a fixed total can still
change issue order and the critical path. Increasing the total cap without a
resource argument remains unlikely to help. Where a sweep did pay, it was
narrow: a collective-issuing warpgroup improved by 0.1-1.0% at 56 registers,
and funding a further rise to 72 out of the reduce role regressed the shapes
that role dominates.

Eliminating a spill counter is also not sufficient acceptance evidence. A
shape-scoped rebalance removed six shared-spilling requests, yet its two
critical benchmark ratios were only 0.981x and 0.989x. Continue through the
performance matrix after the generated-code symptom is repaired.

A source register count is only a seed for the sweep. In the seven-warp case,
88 was the nearest legal target below the reference's observed 89, but adjacent
budgets were not measured and the child was not re-profiled. The timings prove
that the scoped register intervention repaired the deficit; they do not prove
that 88 is optimal or that the measured try-wait count itself fell. Do not copy
the value or claim the counter mechanism without the corresponding sweep and
post-change profile.

## Verification

Record realized allocation and dynamic local traffic, not only the requested
cap. Compile every specialization touched by the selector so allocator cliffs
cannot hide outside the timing set, then measure the beneficiary and guard
paths at adjacent budgets. When excess pipeline polling motivated the sweep,
re-profile the retained child and compare the same barrier waits before
attributing the gain to faster stage release.
