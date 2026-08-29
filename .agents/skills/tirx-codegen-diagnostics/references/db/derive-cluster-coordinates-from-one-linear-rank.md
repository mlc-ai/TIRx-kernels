# Derive cluster coordinates from one linear rank

**Symptoms:** `special_register_reads`, `repeated_index_expression`, `excess_address_math`, `persistent_grid_regression`

## Symptom

A persistent clustered kernel reuses the CTA's in-cluster coordinates in
scheduler, multicast-mask, and address expressions. Generated PTX repeatedly
materializes coordinate expressions or diverges from a reference that reads one
linear cluster rank and derives both coordinates from it.

## What to change

Read the linear rank once into a local scalar. For a power-of-two X extent and
the usual X-minor rank, derive X with a mask and Y with a shift, then reuse the
two local scalars everywhere.

```python
rank = K.local_scalar("int32", init=K.cuda.mov_sreg(32, "cluster_ctarank"))
cluster_x = K.local_scalar("int32", init=rank & (CLUSTER_M - 1))
cluster_y = K.local_scalar("int32", init=rank >> (CLUSTER_M.bit_length() - 1))
```

Keep the declared cluster scope that carries launch metadata even when its
returned coordinate expressions are not used directly.

## Rationale

One measured rewrite replaced two materialized scope coordinates and the
reconstructed linear rank with one `cluster_ctarank` read plus mask/shift
derivation. Across 33 directly comparable targeted rows, strict passes moved
from 29 to 33 and the minimum ratio moved from 0.9215x to 0.9909x. Several
stable rows improved by 0.006-0.013x. The selected form passed all 34 rows in
the expanded targeted set and survived the complete correctness and
performance matrices.

The useful mechanism was not merely assigning coordinate expressions to local
variables. Matching the reference's rank source and linearization removed the
remaining generated-code divergence.

## Boundary

Mask/shift derivation requires a power-of-two X extent and rank linearization
`rank = x + extent_x * y`. Use exact division and remainder for any other
mapping. Do not delete the cluster-scope declaration if launch extraction needs
it, and do not substitute a global CTA rank for the in-cluster rank.

## Verification

Check PTX/SASS for one intended in-cluster rank read, the expected mask/shift or
divmod pair, and no repeated coordinate reconstruction. Run every legal cluster
shape and both singleton and multi-CTA guard paths through targeted performance
measurement, then the complete correctness and performance matrices.
