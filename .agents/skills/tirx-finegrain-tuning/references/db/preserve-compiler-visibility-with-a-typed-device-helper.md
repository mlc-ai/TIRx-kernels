# Preserve compiler visibility with a typed device helper

**Symptoms:** `inline_asm_boundary`, `branch_reconvergence`, `excess_control_instructions`, `performance_regression`

## Symptom

Expanding a long register-only sequence into one inline-asm wrapper per PTX
instruction changes whole-function control-flow lowering even though the
data-path instructions are identical. One measured packed reduction moved from
1080 instructions and one reconvergence region to 1088 and two.

## What to change

Where the original compiler-visible function boundary is the only demonstrated
difference, express the sequence as a private scalar-return PrimFunc whose body
contains the ordinary typed PTX ops.

```python
@T.prim_func(private=True)
def wrelu_reduce(
    accum_h: T.handle("float32", "local"), weights_h: T.handle("float32", "local")
) -> T.float32:
    # Pointer parameters carry their real storage scope, or the pipeline
    # invents global loads.
    accum = T.decl_buffer((num_heads,), "float32", data=accum_h, scope="local")
    weights = T.decl_buffer((num_heads,), "float32", data=weights_h, scope="local")
    ...
    return result
```

Bind the helper to the same device target as its caller; otherwise the pipeline
treats the call as a cross-target launch.

## Rationale

The typed private device helper restored 1080 instructions, one reconvergence
region, 168 registers, zero stack, and the same packed arithmetic. On one fixed
GPU the old-helper/current ratios were 38.130/38.087 = 1.0011 and
184.346/182.235 = 1.0116 across the dense and compressed workloads.

NVVM inlined the helper without a force-inline attribute; do not add one unless
a final-binary negative control shows a surviving device call.

## Boundary

Use this only for a reusable typed function boundary, not to hide an arbitrary
source string or to make a workload-specific PTX bundle.

## Verification

Confirm the final binary has no helper call, then compare control topology,
registers, spills, correctness, and fixed-runner wall time across every affected
workload.
