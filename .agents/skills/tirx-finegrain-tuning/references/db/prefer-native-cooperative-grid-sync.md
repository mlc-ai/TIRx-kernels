# Prefer native cooperative grid sync over software tickets

**Symptoms:** `rare_data_corruption`, `barrier_releases_early`, `software_grid_barrier`, `cooperative_launch_available`

## Symptom

Rare data corruption from a software grid barrier that releases early. One
barrier built on `atom.add` plus `old == num_sms - 1` was lowered by ptxas into
a warp-aggregated ticket whose compare constant came out wrong (`0x1b` instead
of `0x3f`) inside the full kernel, while the isolated repro compiled correctly;
it released after 27 of 64 CTAs.

## What to change

When the reference already uses a cooperative grid sync and the launch keeps
every CTA resident, use the native form instead of any software barrier.

```python
T.cuda.grid_sync()

# and on the launch params of the kernel that executes the sync:
tags.append("tirx.use_cooperative_launch")
```

Keep a dependent epilogue PDL-only: cooperative launch applies to the main
kernel that executes the grid sync, not automatically to every kernel in the
chain.

## Rationale

Current lowering emits the native cooperative launch attribute, so the software
forms are unnecessary. A monotonic ticketless u64 counter avoids the compiler
hazard but still adds port-only global loads, reductions, and polling.
Replacing two ticketless barriers with the native sync removed the workspace
counters and polling, passed all four correctness configurations, and moved
stable five-round 8-GPU campaigns from a pinned 0.966x ratio to 0.993x and
1.022x.

## Boundary

Use a ticketless monotonic counter only as a fallback when native cooperative
launch is unavailable, and inspect final SASS if a ticket form is ever
unavoidable.

## Verification

Confirm the resolved launch params carry the cooperative attribute, then run the
full correctness matrix and a stable multi-round campaign.
