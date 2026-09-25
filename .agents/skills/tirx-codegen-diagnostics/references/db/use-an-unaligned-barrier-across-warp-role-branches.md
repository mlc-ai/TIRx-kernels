# Use an unaligned barrier across warp role branches

**Symptoms:** `divergent_barrier`, `synccheck_failure`, `warp_specialized_synchronization`

## Symptom

Synccheck reports a divergent named barrier when cooperating warp roles reach
the same barrier ID and participant count at different instruction sites.

## What to change

Use `barrier.cta.sync` without `.aligned` at the cooperating sites, retaining
the same ID, count, and phase protocol.

```python
# before: bar.sync implicitly promises aligned execution.
txl.ptx.bar.sync(txl.uint32(3), txl.uint32(160))

# after: distinct warp-role branches may meet at this named barrier.
txl.ptx.barrier.cta.sync(txl.uint32(3), txl.uint32(160))
```

## Rationale

The [PTX barrier specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier)
defines `bar.sync` as the aligned form. The unaligned form permits different
warps to use distinct instruction sites with the same barrier ID and count.

A measured 160-thread rendezvous between one idle warp and four quantizer
warps failed Synccheck in both the original kernel and its numerical revision.
Changing the two sites passed Synccheck on both affected large shapes and
preserved bitwise output. Same-worker CUDA-event time increased from 57.343 to
57.631 us and from 68.629 to 68.798 us over five rounds. This is a correctness
repair with a measured cost, not a performance shortcut.

## Boundary

This does not repair a wrong participant count, mismatched barrier phases,
partial-warp participation, or missing memory dependencies. Diagnose the
reported sites before changing other named barriers. Direct compilation can
show the same SASS barrier mnemonic for both PTX forms; that does not replace
checking the source-level contract and the actual instrumented execution.

## Verification

Inspect both participating sites and their phase/count agreement. Reproduce
the failure on the frozen baseline, run Synccheck on the repaired kernel,
check repeated outputs, and measure every shape executing the changed sites.
