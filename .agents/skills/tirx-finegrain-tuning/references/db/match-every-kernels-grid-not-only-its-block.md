# Match every kernel's grid, not only its block

**Symptoms:** `instruction_parity_with_deficit`, `unsaturated_bandwidth`, `launch_config_drift`

## Symptom

A port whose loop is instruction-identical to the reference is still slow
because its grid is smaller.

## What to change

Read the host launch call of every kernel in the chain and resolve each grid
from its own source, not from the port's shared SM-count knob. A sibling
kernel's value is not evidence for this one.

```python
# Each kernel in the chain carries its own grid, resolved from its own
# launch site: a bandwidth-model count for the main kernel, the full device
# SM count for the bandwidth-bound copy epilogue.
MAIN_SMS = T.meta_var(64)
EPILOGUE_SMS = T.meta_var(num_sms)  # e.g. 148 on this device
```

## Rationale

One dispatch main kernel takes a bandwidth-model count (64 on B200 for e256/k6)
while its copy epilogue is launched with the full device SM count (148), 2.3x
the warps for the same bandwidth-bound copy. A sketch that recorded "same as
kernel 1" -- and passed review with it -- produced a 64-CTA epilogue measuring
112 us against the reference's 97 us; the full-SM launch closed it, and the same
decoupling was then baked into the combine port from the start (main kernel 64,
reduce epilogue 148, 16 warps each).

## Boundary

The device SM count is available to CPU-prepare without initializing CUDA, so
resolving a grid does not require a device query in the wrong stage.

## Verification

Compare the realized grid of every kernel in the chain against its own reference
launch site, then re-measure the affected dispatch.
