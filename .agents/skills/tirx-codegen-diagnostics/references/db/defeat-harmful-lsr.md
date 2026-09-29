# Defeat harmful LSR

**Symptoms:** `imad_prologue_chain`, `excess_address_math`, `register_pressure`, `slow_small_shape`

## Symptom

A long pointer-induction chain in the prologue whose setup cost and live
addresses dominate short workloads, where the reference recomputes compact
addresses inside the loop.

## What to change

Pass only the offending stride or bound through an opaque identity PTX move,
once, outside the loop.

```python
def opaque_i32(x):
    """Identity ``mov.s32``: keeps a loop stride opaque to the host compiler's
    strength reduction, so the loop recomputes addresses per iteration the way
    the reference's own binary does. The value is unchanged."""
    out = T.alloc_local([1], "int32")
    T.evaluate(T.ptx.mov.s32(out[0], x))
    return out[0]


stride = opaque_i32(grid_x * BLOCKS_PER_TB)  # hidden once, before the loop
idx: T.int32 = bx * BLOCKS_PER_TB + tx
while idx < total:
    ...
    idx = idx + stride
```

## Rationale

When nvcc sees a loop stride as an ordinary integer expression, loop-strength
reduction may build a long pointer-induction chain in the prologue. The setup
cost and live addresses can dominate short workloads even if each loop iteration
becomes locally cheaper.

## Boundary

Do not obscure data-path values indiscriminately.

## Verification

Compare the SASS before the first loop branch, plus registers and latency on the
smallest shapes.
