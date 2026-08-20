# Keep sibling warp roles at sibling syntax depth

**Symptoms:** `kernel_deadlock`, `unreachable_role_branch`, `missing_tcgen05_mma`, `tmem_full_wait`

## Symptom

A protocol role is absent from a deadlock snapshot: producers fill every stage
and wait forever for empty slots while consumers wait on a resource nothing ever
produces. The role's body compiled without error.

## What to change

An elected-lane block and a different warp role are independent protocol
participants. Write the second role as a sibling of the enclosing warp-role
guard, not as an `elif` aligned with the elected-lane guard.

```python
# before: aligned with the elected-lane guard, so tracing nests the MMA role
# under the elected lane's else, and its warp predicate becomes contradictory.
if warp_id == TMA_WARP:
    if lane_id == 0:
        ...          # elected-lane work
    elif warp_id == MMA_WARP:
        ...          # never reached

# after: an independent sibling guard at the warp-role level.
if warp_id == TMA_WARP:
    if lane_id == 0:
        ...
if warp_id == MMA_WARP:
    ...
```

## Rationale

A contradictory warp predicate dead-code-eliminates the complete role without a
compile error. One 12-warp reduce-scatter GEMM lowered its two MMA producer
warps to `if (false && cbx == 0)`: the TMA warp filled four shared-memory stages
and waited forever for empty slots, while every consumer warp waited on TMEM
full. Dedenting the MMA guard to an independent sibling `if` restored the
`tcgen05.mma` and commit path, completed the formerly hanging single launch, and
passed all eight TP1 configurations for 20 reset/relaunch cycles each.

## Boundary

Fix the control-flow ownership first. Changing barrier phases or arrival counts
cannot repair a producer body that was never emitted.

## Verification

Inspect the generated TIR or CUDA whenever a role is absent from a deadlock
snapshot, and confirm the role's instructions are present before adjusting any
barrier accounting.
