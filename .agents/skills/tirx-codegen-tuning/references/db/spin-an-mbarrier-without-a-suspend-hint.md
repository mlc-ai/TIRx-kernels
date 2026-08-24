# Spin an mbarrier without a suspend hint

**Symptoms:** `nanosleep_stall`, `barrier_stall`, `latency_bound_epilogue`, `sass_schedule_divergence`

## Symptom

Warp-state sampling parks most of the kernel's not-issued samples on
`NANOSLEEP.SYNCS` where the reference parks its on `BRA`, and the dependent load
after a barrier issues several instructions late.

## What to change

Drop the suspend-time hint from the `try_wait` the spin loop retries on.

```python
# Four instructions, inline, with the sleep and the re-check standing between
# the barrier and the load that depends on it:
#   TRYWAIT; @!P NANOSLEEP 0x989680; @!P PHASECHK; @!P BRA
K.ptx.mbarrier.try_wait.parity.shared.b64(ready, barrier, phase, K.uint32(10_000_000))

# Two, with the retry out of line and the dependent load directly behind:
#   TRYWAIT; @!P BRA <out-of-line>
K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(ready, barrier, phase)
```

## Rationale

ptxas lowers the two spellings differently. Given a suspend-time hint it expands
the wait inline around a `NANOSLEEP`; without one it emits a two-instruction
check and moves the retry out of line. On a latency-bound kernel with one CTA
per multiprocessor there is nothing else to run while a warp sleeps, so the
sleep buys nothing and the two extra instructions sit on the critical path at
every handshake. On a Blackwell MoE grouped GEMM this was worth 2-3% and took
one required row from 0.976 to 1.004.

## Boundary

Check what the reference actually uses -- CUTLASS mixes both, hintless at its
hot waits and hinted elsewhere, and the hintless form is a busy-wait that
competes for issue slots. It pays where the waiting warps have no useful
neighbours, not where occupancy is high. This is the same trade a hand-written
sleep backoff in a flag spin loses, seen from the other side: here the sleep is
inserted by ptxas from an operand most callers copy without noticing, and it
loses for the same reason -- release-detection latency.

## Verification

Count `NANOSLEEP` sites in the SASS before and after, and confirm the wait
becomes `TRYWAIT; @!P BRA` with the dependent load immediately behind it.
