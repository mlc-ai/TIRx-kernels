# Elide tail predicates only when the specialization proves full tiles

**Symptoms:** `excess_guard_math`, `branch_in_hot_loop`, `dispatch_specific_deficit`, `instruction_count_gap`

## Symptom

A shape-specialized kernel keeps per-element tail compares and selects even
when every sequence admitted by that specialization is an exact multiple of
the processing tile.  The redundant guards sit in the hot loop of one critical
warp role, while ragged specializations still genuinely need them.

## What to change

Derive a compile-time full-tile fact from the exact specialization inputs and
split the emitted code.  Keep the guarded fallback intact for every
specialization that can receive a partial tile.

```python
full_tiles = all(int(length) % TILE == 0 for length in config["seq_lens"])

if full_tiles:
    K.assign(value, transformed)
else:
    K.assign(value, K.if_then_else(index < runtime_extent, transformed, zero))
```

## Rationale

A runtime predicate whose result is statically known still costs compare,
predicate, and select instructions unless the specialization exposes the proof
to tracing.  Removing that work from a drain-bound warp can shorten the whole
persistent pipeline even when the guarded reference kernel remains generic.

In one SM100 persistent backward safe-gate path, the full-tile specialization
removed 16 `FSEL` sites and all 15 `ISETP.LE.AND` sites in the final SASS.  The
safe workload changed from 1642.624 us TIRx / 1616.763 us reference (ratio
0.984256) to 1584.229 us TIRx / 1616.954 us reference (ratio 1.020656), a 3.686%
TIRx latency reduction.  The complete eleven-row matrix had minimum ratio
0.994679.

## Boundary

This is valid only when the compiled artifact is tied to the exact static shape
fact.  Do not infer full tiles from a benchmark sample if the same artifact may
later receive different runtime sequence lengths.  Do not delete the ragged
fallback: the `(17, 33)` safe-gate case was separately checked against both the
standalone source and the mathematical oracle.

Static instruction reduction alone is not the gate.  A related dead-tail
unrolling change elsewhere reduced static code but regressed its complete
matrix; isolate this specialization split and let `bench_suite` decide.

## Verification

Confirm in final SASS that the intended compare/select sites disappear only
from the full-tile specialization.  Run correctness for both an aligned case
and a ragged fallback, then run the complete frozen workload matrix and require
every row, not only the affected one, to pass the project ratio gate.
