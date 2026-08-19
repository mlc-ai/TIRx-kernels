# Keep multimem participants alive through kernel exit

**Symptoms:** `illegal_instruction`, `unspecified_launch_failure`, `multimem_early_exit`, `distributed_flake`

## Symptom

Asynchronous CUDA faults at distributed shapes that pass on standalone rerun. A
host barrier after launch cannot keep a rank's device workers alive while peer
ranks are still issuing `multimem.ld_reduce`.

## What to change

Establish rank-local all-worker completion on device, then release-increment a
multicast rank flag and acquire-wait on the local flag until every rank arrives,
before releasing the final device resources or exiting the kernel.

Order the phases: complete the local CTA or cluster drain, perform the
system-scope rank rendezvous, then deallocate TMEM and exit. Keep this barrier
outside the tile protocol.

A bare single flag per rank is insufficient when independently scheduled
persistent workers can finish at different times. Two correct forms are known:
index symmetric flags by physical worker, or first release-aggregate every local
cluster into a rank-local counter and let only the last cluster perform the
cross-rank rendezvous. In the aggregate form, keep both CTAs of that last
cluster behind a cluster barrier until the rank rendezvous completes.

## Rationale

The missing exit rendezvous presented as two different asynchronous CUDA faults
at two different TP4 shapes across two full correctness matrices, while
standalone reruns passed. Adding the per-worker device barrier passed 200
non-blocking relaunches across both failing shapes, the complete 16-shape
TP1/TP4 matrix, and a 1402-configuration full suite.

A later full sweep still observed one standalone-unreproducible TP4 launch
failure with the per-worker form. Replacing 148 multicast arrivals per rank with
a 74-cluster local completion counter plus one multicast arrival per rank passed
the failing shape for 20 relaunches and the complete 16-shape matrix for 320
relaunches, including 160 TP4 launches.

## Boundary

Launch blocking is a localization tool here, not a fix: it passed 200 relaunches
but does not prove the cross-rank kernel-exit invariant.

Do not interpret an isolated `illegal_instruction` from a full-TMEM persistent
kernel until the runner has excluded co-runners. One shared-runner sweep allowed
single-GPU cases to overlap and admitted GPUs already owned by external
processes; its TP1 failure localized to the kernel's intentional `trap` when a
512-column allocation did not start at TMEM column zero. Make full-resource
kernels exclusive and reject externally occupied cards before treating a
remaining failure as evidence about the exit barrier.

## Verification

Use a terminal-state oracle that checks both the local cluster arrivals and all
rank arrivals on every launch, across repeated non-blocking relaunches.
