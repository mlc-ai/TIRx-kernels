# Match every kernel's grid, not only its block

**Symptoms:** `instruction_parity_with_deficit`, `unsaturated_bandwidth`, `launch_config_drift`

## Symptom

A port whose loop is instruction-identical to the reference is still slow
because its grid is smaller.

## What to change

Read the host launch call of every kernel in the chain and resolve each grid
from its own source, not from the port's shared SM-count knob. A sibling
kernel's value is not evidence for this one.

When a one-CTA-cluster persistent kernel uses the whole device, resolve the
actual device SM count during CPU preparation instead of freezing an
architecture-wide constant. Include the resolved count in the cached
specialization key so a process cannot reuse a grid compiled for another
device profile.

```python
# Each kernel in the chain carries its own grid, resolved from its own
# launch site: a bandwidth-model count for the main kernel, the full device
# SM count for the bandwidth-bound copy epilogue.
MAIN_SMS = T.meta_var(64)
EPILOGUE_SMS = T.meta_var(num_sms)  # e.g. 148 on this device

# A full-device persistent specialization is keyed by the prepared count.
num_sms = hardware_num_sms(FALLBACK_SMS)
active_clusters = min(cluster_work, num_sms)
```

## Rationale

One dispatch main kernel takes a bandwidth-model count (64 on B200 for e256/k6)
while its copy epilogue is launched with the full device SM count (148), 2.3x
the warps for the same bandwidth-bound copy. A sketch that recorded "same as
kernel 1" -- and passed review with it -- produced a 64-CTA epilogue measuring
112 us against the reference's 97 us; the full-SM launch closed it, and the same
decoupling was then baked into the combine port from the start (main kernel 64,
reduce epilogue 148, 16 warps each).

A six-warp SM107 persistent GEMM supplied the fixed-constant case. Paired
profiles showed a reference grid of 216 and a port grid of 200 while both were
one-CTA-per-SM shared-memory limited, had zero local-memory sectors, near-equal
DRAM work, and identical tensor-memory work. On the benchmark device, CPU
preparation resolved 212 SMs. Replacing only the fixed cap with that prepared
count moved three distinct worst ratios from 0.8878x, 0.9109x, and 0.9345x to
1.0068x, 1.0007x, and 1.0001x. Fifteen other historical worst rows passed at a
1.0027x minimum, and the complete 102-row matrix passed at a 0.9970x minimum;
all 16 correctness specializations remained bitwise exact.

## Boundary

The device SM count is available to CPU-prepare without initializing CUDA, so
resolving a grid does not require a device query in the wrong stage.

The full SM count is not an active-cluster count for multi-CTA clusters. Use an
occupancy- or source-certified concurrent-cluster cap for those shapes. Keep
the resolved hardware value in the specialization cache key whenever one
process can prepare the same logical shape for devices with different SM
counts.

Matching stops where the reference's grid exceeds a persistent kernel's
work-item count: the surplus CTAs run only launch, barrier-init, and drain
code, and that fixed cost lands on the shortest shapes. Launching the
reference's full SM-count grid instead of `min(work_items, sm_count)` measured
0.976x against a 0.984-0.988x band on a one-item-per-CTA shape and trimmed a
second from 1.033x to 1.018x; the clamped grid was retained. Match the grid
when every CTA receives work; clamp it to the work count when it would not.

## Verification

Compare the realized grid of every kernel in the chain against its own reference
launch site, then re-measure the affected dispatch. For device-derived grids,
also verify that the prepared SM count reaches the compiled grid and that
specialization caching distinguishes different device profiles.
