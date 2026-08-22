# Bind the cluster scope or lose the attribute

**Symptoms:** `silently_dropped_attribute`, `cluster_dim_mismatch`, `launch_config_drift`

## Symptom

A requested cluster launch that silently falls back to an ordinary launch.
Correctness can be unaffected, so the drift surfaces only when launch metadata
or a profiled comparison against a cluster-launched reference disagrees. This
also applies to an explicitly requested cluster `(1,1,1)`: extent one does not
make the launch mechanism equivalent.

## What to change

Requesting `clusterCtaIdx.x` in the launch tags does not by itself produce a
cluster launch: the resolved `tirx.kernel_launch_params` carries the cluster
dimension only when the kernel body binds the scope.

```python
# before: an extent-one specialization silently loses cluster presence.
if CLUSTER_N > 1:
    cbx = T.cta_id_in_cluster([CLUSTER_N])

# after: the binding remains even when every extent is one and the value is
# never used.
if USE_CLUSTER_LAUNCH:
    cbx = T.cta_id_in_cluster([CLUSTER_N])
```

Lowering and the runtime must preserve presence separately from value. Do not
decide whether to emit `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` by testing
whether any resolved dimension differs from one.

## Rationale

One combine kernel declares cluster (2,1,1) to overlap clustered computation
kernels; adding the dead binding restored
`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = (2,1,1)`, verified through the real
lowering path.

An extent-one persistent specialization provided the complementary case. The
target initially profiled as Cluster Size 0 while the reference reported 1.
Preserving its explicit cluster scope through lowering and keying the runtime
attribute on scope presence made both report Cluster Size 1. Five singleton
rows improved by about 0.2-3.0%, all 13 affected and guard rows passed at a
minimum 0.9905x, and all correctness specializations passed. A later complete
matrix had one unrelated high-persistence epilogue failure; the explicit
cluster mechanism remained in the final passing implementation.

## Boundary

Treat every launch-tag request as unproven until the resolved params are
inspected; the request and the realized attribute are different objects. An
ordinary `(1,1,1)` launch and an explicit cluster `(1,1,1)` launch must remain
distinguishable in the launch metadata.

## Verification

Inspect the resolved launch params through the real lowering path, not the
requested tag list. Profile the realized Cluster Size for the extent-one case
as well as a multi-CTA guard, then run every specialization sharing the launch
path.
