# Spin backoffs cost their detection latency

**Symptoms:** `backoff_regression`, `barrier_stall`, `cold_start_sensitivity`

## Symptom

A barrier spin that looks wasteful, inviting a sleep-based backoff that then
fails to pay for itself.

## What to change

Keep the reference's spin form. Adding `nanosleep` backoff to a barrier spin
trades poll traffic for release-detection latency, and on this class of kernel
the trade lost.

```python
# The reference busy-polls everywhere; keep that shape.
while not _flag_ready():
    pass

# Treat this as a hypothesis to falsify, not a default.
while not _flag_ready():
    T.cuda.nano_sleep(256)
```

## Rationale

In one dispatch kernel, 256/512 ns backoffs in the grid and NVLink barrier spins
were correct but measured no better than busy-polling across three bench
campaigns, with quiet rounds trending worse (about 0.85-0.93 against 0.95-1.01).
Roughly 0.5-1 us of added detection latency per barrier outweighed the saved L2-
and sys-scope poll traffic, which is already cheap at 64 CTAs on one line plus
one SM's NVLink polls.

## Verification

Compare quiet-round campaigns, not a single run: the loss showed up as a trend
across three campaigns rather than a single decisive measurement.
