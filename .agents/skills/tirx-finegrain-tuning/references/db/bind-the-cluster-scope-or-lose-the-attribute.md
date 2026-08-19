# Bind the cluster scope or lose the attribute

**Symptoms:** `silently_dropped_attribute`, `cluster_dim_mismatch`, `launch_config_drift`

## Symptom

A requested cluster launch that silently falls back to cluster (1,1,1).
Correctness is unaffected, so the drift surfaces only when launch metadata or a
profiled comparison against a cluster-launched reference disagrees.

## What to change

Requesting `clusterCtaIdx.x` in the launch tags does not by itself produce a
cluster launch: the resolved `tirx.kernel_launch_params` carries the cluster
dimension only when the kernel body binds the scope.

```python
# The binding is what materializes the attribute -- keep it even when the
# value is never used.
if CLUSTER_N > 1:
    cbx = T.cta_id_in_cluster([CLUSTER_N])
```

## Rationale

One combine kernel declares cluster (2,1,1) to overlap clustered computation
kernels; adding the dead binding restored
`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = (2,1,1)`, verified through the real
lowering path.

## Boundary

Treat every launch-tag request as unproven until the resolved params are
inspected; the request and the realized attribute are different objects.

## Verification

Inspect the resolved launch params through the real lowering path, not the
requested tag list.
