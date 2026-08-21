# Preserve a body-owned flat thread scope when the source owns it

**Symptoms:** `thread_axis_drift`, `short_kernel_regression`, `sass_schedule_divergence`, `long_scoreboard`

## Symptom

A port with the right flat lane mapping, the same registers, the same dynamic
instruction count, and the same global and shared memory work still measures
slower, and the slower stream exposes more long-scoreboard latency. The entry
declares independent CTA, warp, lane, and thread scopes around a source body
that owned one CTA scope and one flat thread scope.

## What to change

Leave both scopes to the body and declare the original axis contract inside it.

```python
@K.kernel(warps=THREADS // 32, arch="sm_100a", grid=False, thread_layout=False)
def kernel(...):
    work, n = K.cta_id([NUM_VALUE_HEADS * VALUE_SPLIT, NUM_SEQS])
    tid = K.thread_id([THREADS])
```

This keeps entry ownership of the ABI and shared-memory structure while removing
the extra axis representation.

## Rationale

A four-warp GDN decode port had the same 64 registers, 3,442,688 dynamic
instructions, and global/shared memory work as the frozen source, but two clean
45-round campaigns measured after/before at 1.01166 and 1.01093. Restoring the
body-owned scopes produced a 1,080-instruction SASS stream that matched the
frozen stream operand-for-operand, targeted correctness passed, and the clean
45-round ratio was 0.99991.

The same boundary applies to a short PDL consumer. An eight-warp sparse decode
combine emitted 576 SASS instructions and 48 registers with entry-owned
three-axis CTA and flat-thread scopes, while the frozen body used 560
instructions and 47 registers. Restoring the body-owned scopes made the final
SASS byte-identical, passed targeted correctness, and moved the clean 45-round
pipeline ratio from 1.01307 to 1.00540.

## Boundary

Keep entry-owned axes as the default; use this opt-out only when the source owns
flat axes and measured codegen shows that reproducing that contract closes the
schedule gap.

Isolated profiles had ranked both candidate kernels faster in the PDL case, so
validate producer-consumer changes in the actual multi-launch timing scope.

## Verification

Diff the final SASS operand-for-operand against the frozen stream, then measure
in the real launch scope rather than in isolation.
