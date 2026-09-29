# Publish small payloads with an in-word generation

**Symptoms:** `slow_epilogue`, `synchronization_overhead`, `global_counter_contention`

## Symptom

A short cross-CTA reduction spends substantial tail time publishing a small
payload through a separate counter and sharing the acquired counter with its
consumer threads.

## What to change

When the payload and generation fit together in one naturally aligned scalar
word, publish and poll that whole word using GPU-scope relaxed scalar accesses.
The consumer accepts the payload only when the generation in the same loaded
word matches. Keep the payload packed through publication.

```python
# A 32-bit payload and a 32-bit generation share one aligned 64-bit record.
record = txl.local_scalar("uint64")
txl.ptx.mov.b64(record, payload, generation)
txl.ptx.st.relaxed.gpu.global_.u64(records.ptr_to([index]), record)

# The consumer polls with the same scalar width and scope.
observed = txl.local_scalar("uint64")
txl.ptx.ld.relaxed.gpu.global_.u64(observed, records.ptr_to([index]))
```

The loop must inspect the loaded generation before using its payload. Each
record needs a unique producer and a generation lifecycle that cannot accept
an old record. A toggled phase is sufficient only when every producer rewrites
every record on each ordered invocation and invocations do not overlap.

## Rationale

Aligned, morally strong scalar accesses provide
[single-copy atomicity](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#atomicity): the
consumer observes a payload and generation from the same write. No ordering of
separate payload storage is needed. This can remove a publication barrier,
contended completion RMW, and barrier distributing the acquire.

A measured ten-producer BF16-pair reduction retained fixed-order FP32 FMAs but
replaced its counter protocol with tagged scalar words. Same-worker pure GPU
time fell from 7.357 to 6.807 us over five rounds. The final code used aligned
64-bit `STG.E.64.STRONG.GPU` and `LDG.E.64.STRONG.GPU`, 48 registers, and zero
stack. Repeated launches, mutable inputs, hot routing, and exact cancellation
passed. This measurement compares two FP32-reduction implementations; it does
not establish parity with a lower-precision atomic baseline.

## Boundary

This publishes only the bytes in the scalar word. It does not establish an
acquire/release relationship for other memory. A vector of two 32-bit accesses
is not an atomic 64-bit publication, and overlapping mixed-width accesses do
not satisfy the same argument. Initialize tags before the first invocation;
prove phase reuse and wraparound against the complete producer lifecycle.

The extra tag increases traffic. Larger measured reductions regressed when
this doubled their scratch footprint; retain compact payload storage and a
separate publication protocol where that traffic dominates. A polling grid
must also have a progress argument; all producers must be able to run while
consumers wait.

## Verification

Inspect scalar width, alignment, scope, and final SASS. Check repeated launches
with changed inputs, poisoned output, phase reuse, generation wraparound where
applicable, and synchronization diagnostics. Compare the same reduction
arithmetic before and after the publication change, then measure the full
shape matrix because the traffic tradeoff changes with payload volume.
