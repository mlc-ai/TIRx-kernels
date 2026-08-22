# Cap a persistent cluster grid by concurrent residency

**Symptoms:** `persistent_grid_regression`, `launch_config_drift`, `concurrency_capped`, `low_occupancy`

## Symptom

A persistent clustered kernel is near parity for one- and two-CTA clusters but
falls into a much slower band as the cluster grows. The grid was sized as
`num_sms // cluster_size`, so its scheduler stride assumes every launched
cluster is resident at once even though the hardware admits fewer concurrent
clusters.

## What to change

Resolve the maximum simultaneously active cluster count for the compiled
cluster shape and resource usage, then cap the persistent grid by that count.
Do not derive the cap by dividing the SM count by CTAs per cluster.

```python
# before: geometric capacity is not concurrent residency.
num_clusters = min(cluster_work, NUM_SMS // cluster_size)

# after: use a source- or occupancy-certified active-cluster cap.
active_cap = ACTIVE_CLUSTERS[cluster_size]
num_clusters = min(cluster_work, active_cap)

@K.kernel(grid=[CLUSTER_M, CLUSTER_N, num_clusters])
def kernel(...):
    _, _, cluster_work_id = K.cta_id()
    work = K.local_scalar("int32", init=cluster_work_id)
    with K.While(work < cluster_work):
        ...
        K.assign(work, work + num_clusters)
```

## Rationale

One measured SM100 persistent grid used `floor(148 / cluster_size)`, while the
source's certified active-cluster counts for sizes 1, 2, 4, 8, and 16 were 148,
74, 33, 15, and 7. The geometric rule launched delayed second-wave clusters;
because the grid size was also the persistent scheduler stride, those late
clusters received a different amount of work.

Replacing only the grid cap moved three large-cluster guards from the
0.62-0.67x band to 0.992-0.998x. Across the complete 66-shape matrix, strict
passes rose from 27 to 46 and the worst ratio rose from 0.5546x to 0.9618x,
with all correctness configurations passing.

## Boundary

Active-cluster capacity is specific to the device, cluster shape, dynamic
shared memory, registers, and launch contract. The measured constants above are
evidence for one compiled family, not portable architecture constants. Obtain
the cap from the reference's hardware query or an occupancy result for the
actual specialization.

This changes work distribution, not only occupancy. Re-prove the grid-stride
mapping and tail coverage whenever the cap changes.

## Verification

Compare the realized grid and cluster dimensions with the reference, then
profile the actual active-cluster count. Run correctness across every cluster
shape and benchmark both the large-cluster failures and the small-cluster
guards before keeping the cap.
