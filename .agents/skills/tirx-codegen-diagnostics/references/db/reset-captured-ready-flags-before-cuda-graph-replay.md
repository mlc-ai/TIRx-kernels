# Reset captured ready flags before CUDA Graph replay

**Symptoms:** `cuda_graph_replay_nondeterminism`, `stale_epoch_flags`, `cross_cta_dependency_race`, `changed_input_replay`

## Symptom

A persistent kernel is repeatable under ordinary launches but changes output
after a CUDA Graph is replayed with updated inputs. Cross-CTA consumers poll
epoch-stamped ready flags, and the epoch is supplied as a host scalar.

## What to change

When the host epoch cannot advance during replay, capture a clear of the ready
flags before the producer and consumer kernels. Keep ordinary launches on the
existing epoch protocol.

```python
def launch():
    host_epoch["value"] += 1
    if torch.cuda.is_current_stream_capturing():
        ready_flags.zero_()  # becomes the first node on every replay
    producer_consumer_kernel(*args, host_epoch["value"])
```

## Rationale

Graph capture freezes scalar kernel arguments and does not rerun the Python
closure on replay. Flags published by the preceding replay can therefore
satisfy equality or monotonic-progress waits for the captured epoch before the
current producer has written its data. Clearing only the ready flags restored
reference accuracy and bitwise repeatability across ten fused/persistent cases,
including path changes and upstream amplitudes through 2^64. This isolated the
ready flags from a scheduler counter that the kernel already reset at exit.

The capture-only clear left three ordinary complete calls at -0.34%, -0.72%,
and +0.04%. Correct graph replay was 0.93%, 2.73%, and 2.71% slower than the
stale-flag replay on three shapes, below a 5% limit even though the stale replay
could incorrectly skip dependency waits.

A later combined kernel added a compact item retry after the native recurrence.
One captured graph was replayed while its mode changed
`0 -> 4 -> 1 -> 0 -> 4` (ordinary, item retry, launch-wide stable, ordinary,
item retry). Clearing only the recurrence ready flags was still sufficient:
all ten replays were bitwise repeatable, the retry count was zero after every
replay, and the two item-retry results matched the reference elementwise for
all outputs except an existing `1.607e-40` FP32-subnormal `dh0` difference. The
retry kernel already resets its count at completion, while launch-wide mode 1
suppresses item publication. Do not add a second graph memset for such state
without first proving that its owning kernel can leave it live at completion.

The same test without the captured ready-flag clear failed after the first
ordinary-to-retry transition: two nominally identical replays differed in `dv`
by `0.00891113`. This negative control distinguishes the reset from a test that
merely happens to schedule producers before consumers.

Folding the zero stores into an existing range-guard kernel did not reliably
recover graph time. The merged guard passed the same five-mode lifecycle test,
but relative to the separate captured clear its replay times changed by
`+0.24%`, `+0.34%`, and `-0.36%` on one fused and two persistent shapes. Extra
arguments, predicates, and stores can offset removal of a small graph node;
prefer the narrower capture-only clear unless paired measurement shows a
repeatable margin improvement.

## Boundary

Clear the flags before every captured producer/consumer sequence, not between
its producer and consumer. The reset value must fail every wait target used by
the captured epoch. Reset other mutable scheduler state only when the kernel
does not restore it before completion.

This protocol assumes replays that share flags, snapshots, and counters are
serialized. Concurrent graph replays need separate workspace or a device-side
unique epoch; a shared clear does not make them safe.

## Verification

Capture once, then change inputs and replay repeatedly without calling the host
launcher. Alternate between execution paths and amplify upstream values so a
stale snapshot is visible. Require all outputs to match the reference and two
successive replays bitwise. Measure graph replay and ordinary launch separately.

When downstream work owns a queue or counter, read that state after every
replay and require its completion value, including transitions into and out of
the queue-consuming mode. On a combined GB200 candidate, direct production-to-
fixed graph replay costs were 2.05%, 4.33%, and 4.86% on one fused and two mega
shapes; ordinary complete-call costs were 2.45%, 1.67%, and 0.59% on three mega
shapes. Keep the performance claim scoped to the measured device and shapes,
especially when a result is close to its limit.
