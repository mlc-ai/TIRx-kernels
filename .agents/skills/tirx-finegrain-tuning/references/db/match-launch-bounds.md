# Match launch bounds

**Symptoms:** `register_spill`, `register_budget_mismatch`, `local_memory_traffic`, `low_occupancy`

## Symptom

STL/LDL traffic or global rescheduling in a kernel whose reference uses more
registers; realized allocation capped well below the reference's.

## What to change

Set `tirx.launch_bounds_min_blocks_per_sm` from the reference kernel's realized
occupancy target, not from theoretical occupancy. It is a statement attribute
placed right after `T.device_entry()`, not a function attribute.

```python
T.device_entry()
T.attr({"tirx.launch_bounds_min_blocks_per_sm": 8})
```

Do not copy one minimum-block value across block-size families; select it from
the shape when the families differ.

```python
T.device_entry()
if ILP_ROWS == 4 and SEQ_LEN == 8:
    if USE_SMEM_V and NUM_HEADS >= 8:
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 9})
    else:
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 8})
```

## Rationale

The value becomes the second CUDA `__launch_bounds__` argument and imposes a
hard ptxas register budget: roughly 65536 registers divided by (threads per CTA
times the bound), rounded down to the allocation granularity. One measured
512-thread quantization kernel was capped at 32 registers with a bound of 4
while its reference ran at about 50; a bound of 2 restored parity.

In one measured selector, a 160-thread family used nine minimum blocks and a
representative FP16-state specialization moved from 53 to 40 registers, while
its 288-thread family used one minimum block and stayed at 53 registers. Forcing
nine on the larger block cut its allocation to 32 registers before timing. The
shape-aware 9/1 selector cleared its five-workload boundary matrix at
1.003-1.028x.

## Boundary

Treat a large allocation shift as a separate shape A/B even when neither variant
spills: ptxas can trade registers for recomputation and address instructions.

## Verification

Compare resource usage, achieved occupancy, and dynamic local-memory traffic on
both sides; do not infer success from the declared launch bound alone.
