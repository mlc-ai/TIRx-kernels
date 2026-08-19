# Split predicated destination policies around the inactive-path merge

**Symptoms:** `predicated_destination`, `inactive_lane_value`, `sass_divergence`, `performance_regression`

## Symptom

A predicated chain that either carries false input dependencies or loses the
inactive-lane value, depending on which single destination policy was applied to
the whole chain.

## What to change

A predicated instruction with a written destination needs the policy at that
specific program point, not one policy for the whole expression chain. Keep the
default write-only destination before the merge, merge explicitly with `selp`,
then use `preserve_dst=True` on a later predicated transform.

```python
# Write-only (default preserve_dst=False): inactive lanes are not consumed yet.
T.evaluate(T.ptx.ld.shared.f32(s_log[0], s_addr, pred=predicate))
T.evaluate(T.ptx.ld.shared.f32(t_log[0], t_addr, pred=predicate))
T.evaluate(T.ptx.sub.f32(gamma[0], s_log[0], t_log[0]))

# The explicit merge that makes the inactive value defined.
T.evaluate(T.ptx.selp.f32(gamma[0], gamma[0], T.float32(0), T.ptx.pred(predicate)))

# Read-write, because inactive lanes must now retain the merged value.
T.evaluate(T.ptx.ex2.approx.ftz.f32(gamma[0], gamma[0], pred=predicate, preserve_dst=True))
```

Applying read-write binding to the initial loads creates false input
dependencies; applying write-only binding to the final transform loses the
inactive value.

## Rationale

One shared-memory gamma path recovered its original lowering with predicated
undefined shared loads, an unconditional subtract, `selp` to zero inactive
lanes, and a predicated read-write `ex2`. The final SASS was byte-identical to
the source-helper baseline. Across its three bench-suite workloads,
baseline/final times were 54.307/54.317 us, 119.646/119.631 us, and
83.360/82.549 us, with correctness passing.

## Boundary

This sequence is valid only when the undefined values are dominated by the
merge.

## Verification

Verify predicate polarity, inactive-lane consumption, final SASS, and every
control-flow shape.
