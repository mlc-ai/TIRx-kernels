# Adapt row grouping when a fixup grid cannot fill one wave

**Symptoms:** `low_wave_count`, `low_dram_throughput`, `latency_bound_fixup`, `small_grid`

## Symptom

A bandwidth-looking fixup that is actually launch-latency-bound: grouping
several independent rows into each CTA leaves too few blocks to occupy the SMs.
One B200 fixup launched 32 blocks, only 0.11 waves per SM, while using 0.66% of
peak DRAM throughput.

## What to change

Choose the largest supported row group that still launches roughly one wave:
retain the wider group where the state count already provides enough blocks, and
fall back through smaller groups only for underfilled shapes.

```python
# Derived from the state count and the device SM count, not from a named shape.
ROWS_PER_CTA = T.meta_var(
    4 if num_states * 1 >= num_sms * 4 else 2 if num_states * 2 >= num_sms else 1
)
```

## Rationale

Reducing the group from four rows to one raised the grid from 32 to 128 blocks;
the generated row-one kernel used 255 registers, 512 bytes of shared memory, and
no spills. On the same physical GPU over 15 rounds, time improved from 54.641 us
to 53.665 us, a 1.0182 before/current ratio. The wider group remained selected
for shapes with sufficient parallel states.

## Boundary

Keep the grouping decision derived from the state count and device SM count
rather than special-casing a named shape.

## Verification

All ten correctness configurations passed and the complete affected bench matrix
cleared its ratio gate; check registers, shared memory, and spills on the
narrower group as well as wave count.
