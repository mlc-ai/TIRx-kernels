# Trade cross-warp reduction for aligned thread-local work

**Symptoms:** `reduction_overhead`, `math_pipe_throttle`, `mio_throttle`, `instruction_parity_with_deficit`

## Symptom

A row reduction matches the reference's memory traffic and already executes no
more instructions, yet the wide CTA spends excess instructions and stalls on
cross-warp reduction.  The row width is large enough that halving the CTA can
give each remaining thread another complete aligned vector instead of creating
scalar or misaligned accesses.

## What to change

Halve the CTA width, double the compile-time values handled by each thread, and
retain the original aligned vector transaction as the per-thread chunk.  Apply
the smaller launch only to the measured target and specialization; derive the
number of warp partials and element loops from the selected block size.

```python
threads = T.meta_var(NARROW_THREADS if use_narrow_reduction else SOURCE_THREADS)
values_per_thread = T.meta_var(ROW_WIDTH // threads)
chunks_per_thread = T.meta_var(values_per_thread // ALIGNED_VECTOR_WIDTH)

for chunk in T.unroll(chunks_per_thread):
    values = _load_aligned_vector(base + tid * values_per_thread + chunk * ALIGNED_VECTOR_WIDTH)
    _consume(values)

for partial in T.unroll(threads // WARP_SIZE - 1):
    _merge_warp_partial(partial)
```

## Rationale

On a 20-SM target, changing a 3,072-value fused row normalization from 384
threads with eight values each to 192 threads with sixteen values each reduced
dynamic warp instructions from 4,995,840 to 3,895,680.  Shared loads fell from
57,600 to 28,800 and shared stores from 26,880 to 15,360; math-pipe and MIO
stall samples moved from 86/48 to 31/0.  The realized allocation rose from 72
to 120 registers per thread but remained spill-free and used fewer allocated
registers per CTA because the thread count halved.

The complete six-row affected performance matrix measured 1.0169-1.0793x
reference/port.  Two longer counterbalanced checks measured 1.0264-1.0364x on
the original failing row and 1.0162-1.0328x on the scaled-input guard.  All 43
correctness configurations passed.

## Boundary

This is not a general preference for fewer warps.  The useful narrower block
preserved 16-byte vector chunks and increased thread-local work while reducing
the number of warp partials.  A prior 768-thread alternative moved in the
opposite direction and remained at 0.9762x despite exposing more warps.  Do not
use a block width that forces misaligned vector addresses, scalar tails, local
memory traffic, or a register allocation cliff.  Scope the choice to the
target and compile-time mode/output family that passed the matrix.

## Verification

Check vector alignment, realized registers, local traffic, dynamic
instructions, and shared reduction transactions.  Run the complete affected
correctness and benchmark matrices, including low-row, multi-batch, and any
compile-time scaled-input variants, in both implementation orders.
