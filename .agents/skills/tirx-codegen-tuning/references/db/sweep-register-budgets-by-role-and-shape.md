# Sweep register budgets by role and shape

**Symptoms:** `register_spill`, `excess_address_math`, `low_occupancy`

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

## Boundary

An occupancy proof only shows that a larger budget is affordable; it does not
show that its schedule is better. Scope a budget by the compile-time role,
fragment/output family, cluster regime, and resource limit that produced the
measured response. Let unrelated paths use native allocation when a shared
budget changes their entry allocation or trips an allocator limit.

## Verification

Record realized allocation and dynamic local traffic, not only the requested
cap. Compile every specialization touched by the selector so allocator cliffs
cannot hide outside the timing set, then measure the beneficiary and guard
paths at adjacent budgets.
