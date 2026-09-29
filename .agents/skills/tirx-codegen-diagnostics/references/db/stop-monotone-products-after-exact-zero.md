# Stop monotone products after exact zero

**Symptoms:** `excess_instructions`, `loop_overhead`, `long_dependency_chain`

## Symptom

A bounded product has rounded to zero, but a reduction keeps issuing loads and
FMAs for terms multiplied by that product. Tiling several reductions together
can hide the point at which every remaining contribution is zero.

## What to change

Identify the product that upper-bounds the other live products. End the runtime
loop when that product is exactly zero, provided every subsequent term remains
zero. Keep the reduction order unchanged up to that point.

```python
# before: iterations continue after all products vanish.
for j in txl.serial(count):
    consume_and_advance(products, j)

# after: largest_product bounds every live product in this iteration order.
j = txl.local_scalar("int32", init=0)
with txl.While((j < count) & (largest_product != txl.float32(0.0))):
    consume_and_advance(products, j)
    txl.assign(j, j + 1)
```

## Rationale

In a measured decay correction, adjacent factors lay in [0,1]. In a backward
scan, the earliest output row's product was the last to reach zero; in a forward
scan, the latest row's product was last. Checking those products cut two large
strong-decay cases from 6.33 ms/688 us to 4.80 ms/513 us. Ordinary inputs that
skipped the correction remained around 873/142/240 us. All nine focused
correctness cases passed the original tolerances and repeated poisoned-output
checks. This uses exact zero, not an accuracy-dependent truncation threshold.

## Boundary

Prove the product ordering for both scan directions and partial tiles. A zero
product does not justify skipping a later reset, a nonfinite multiplication,
an independent addend, or an observable side effect. Check signed-zero behavior
if bitwise output identity is part of the contract. A varying exit point also
does not permit full-mask warp collectives inside the divergent loop.

## Verification

Test exact-zero factors, gradual underflow, factors equal to one, mixed lanes,
and partial tiles. Verify every participating lane still reaches the shared
memory release protocol. Measure a case where the loop exits early and a case
where it reaches its original bound.
