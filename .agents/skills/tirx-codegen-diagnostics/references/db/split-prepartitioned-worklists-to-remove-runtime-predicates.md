# Split prepartitioned worklists to remove runtime predicates

**Symptoms:** `runtime_predication`, `branch_in_hot_loop`, `excess_control_instructions`, `mixed_worklist`

## Symptom

A planner already separates exceptional and uniform items into distinct lists,
but the kernel merges them into one traversal and recomputes the list identity
for every item. A large unrolled consumer consequently carries mask selection,
payload addressing, or other exceptional control through the uniform hot path.

## What to change

Traverse the planner's lists separately and pass list identity to the shared
consumer as a trace-time Boolean. Preserve each list's original order,
prefetches, and producer/consumer protocol.

```python
# before: `exceptional` is a runtime value inside every consumer instance.
with K.While(i < exceptional_count + uniform_count):
    consume(item_id(i), exceptional=i < exceptional_count)
    K.assign(i, i + 1)

# after: the helper emits two code shapes from Python Boolean arguments.
with K.While(i < exceptional_count):
    consume(exceptional_id(i), exceptional=True)
    K.assign(i, i + 1)
with K.While(j < uniform_count):
    consume(uniform_id(j), exceptional=False)
    K.assign(j, j + 1)
```

## Rationale

In a masked sparse pipeline, guarding the 128 lane-wise mask selects once per
partial item reduced dynamic `FSEL` executions from 100,864 to 2,816. Three
fixed-layout paths moved from a failing 0.943x worst ratio to 1.028x, 1.031x,
and 1.099x.

The remaining packed path still merged partial and full work. Splitting its
prepartitioned lists and tracing the mask-presence flag reduced TIRx latency
from 26.085 to 25.919 us, a 0.63% improvement, and moved the measured ratio
from 0.983x to 0.990x. Tight source comparison was exact on the target, and a
mixed multi-partial case also passed an independent oracle.

## Boundary

Use this only when list membership is already an input invariant. Do not
repartition records inside the kernel or change an ordering that carries
pipeline phases, prefetch distance, or accumulation semantics. Keep a shared
consumer helper so only the exceptional subregion is specialized; duplicating
an entire large role body for a constant stage measured neutral-to-negative in
the same pipeline.

The split may increase static code size. Retain it only when production-compiler
SASS and the required benchmark paths improve, including shapes with no
exceptional items and shapes with several of them.

## Verification

Check emitted code to confirm payload loads and per-element selects are absent
from the uniform loop. Compare dynamic control/select counts before and after,
then run tight correctness for all-uniform, mixed, empty, and multi-exception
plans. Benchmark the focused deficit first and finish with the frozen full
matrix.
