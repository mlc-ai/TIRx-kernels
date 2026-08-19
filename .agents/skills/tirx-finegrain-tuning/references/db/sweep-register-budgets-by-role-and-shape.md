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

## Verification

Record realized allocation and dynamic local traffic, not only the requested
cap.
