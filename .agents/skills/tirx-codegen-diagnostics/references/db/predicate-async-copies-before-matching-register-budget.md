# Predicate asynchronous copies before matching the register budget

**Symptoms:** `reconvergent_branch`, `cp_async`, `register_budget`, `instruction_count_gap`

## Symptom

A normalization kernel wraps an asynchronous row load in a uniform-looking
row-valid branch. The reference predicates the copy instruction directly, while
the generated kernel pays `BSSY`, `BRA`, and `BSYNC` overhead around the copy.
Memory traffic and launch geometry already match, but the generated kernel also
uses a different register allocation.

## What to change

Put the row predicate on `cp.async` while leaving the column-tail source-size
operand intact. Once the control-flow instruction gap closes, sweep the static
register cap and compiler register-usage level together. Scope both changes to
the measured target and specialization.

```python
K.ptx.cp_async(
    shared.ptr_to([shared_offset]),
    source.ptr_to([source_offset]),
    copy_bytes,
    valid_source_bytes,
    pred=row_valid,
)
```

## Rationale

For a Thor H4096 PDL RMSNorm, direct predication reduced static SASS from 720
to 712 instructions, equal to the source. Dynamic warp instructions fell from
45,120 to 44,928 versus 44,800 for the source. It did not improve timing by
itself. Pairing it with a 56-register cap and compiler register-usage level 6
matched the source's 56-register allocation and measured 1.0117x over 30
counterbalanced same-process pairs. Each order subset independently exceeded
parity at 1.0068x and 1.0167x. Global and shared traffic remained equal, local
and spill traffic remained zero, and all 279 correctness configurations passed.

The same instruction-level predicate repaired an H8192 FP8-quantizing
normalization for a different reason. It increased dynamic warp instructions
from 194,048 to 196,352 (the source used 194,816), but changed the allocation
from 79 to 80 registers and reduced sampled long/short-scoreboard stalls from
85/11 to 73/5. Two independent 30-pair runs measured 1.0147x and 1.0065x over
the source, and all 97 affected configurations passed. This is a useful
counterexample to treating instruction count as the objective: direct
predication can improve the dependency schedule even when it adds instructions.

## Boundary

A false instruction predicate suppresses the copy; it does not promise the
same shared-memory contents as an explicit zero-fill. Use this transformation
only when invalid rows never consume or store meaningful results, all threads
still participate in required barriers, and column tails retain their original
source-size zero-fill semantics. Preserve the established path on unmeasured
targets. Closing an instruction-count gap is not sufficient evidence by itself:
the register experiment was required for the measured speedup. A separate
H8192 thread-count change produced a short-run false positive and was rejected
by both subsets of a 30-pair counterbalanced run.

## Verification

Compare final SASS reconvergence instructions, dynamic warp and predicated-on
thread instructions, realized registers, memory transactions, local traffic,
and spills. Benchmark in one process after burn-in, alternate implementation
order, require both order subsets to agree on the key shape, and run the full
affected correctness matrix.
