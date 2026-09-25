# Bypass proven identity indices without rewriting the plan

**Symptoms:** `indirect_global_load`, `long_scoreboard`, `planner_indirection`, `schedule_regression`

## Symptom

A specialized planner materializes an index array whose values are proven equal
to their logical positions. The consumer performs a dependent global load for
every item even though the mapping is identity, but nearby planner metadata and
prefetches also shape the producer/consumer schedule.

## What to change

Bypass only the proven identity index load. Retain counts, offsets, payloads,
prefetches, and traversal structure until each has independent evidence.

```python
def physical_id(logical):
    if IDENTITY_PLAN:  # trace-time proof attached to this specialization
        return logical
    return _load_i32(index_buffer, index_base + logical)
```

Keep the generic indirect fallback in the same kernel family for every plan
that does not carry the proof.

## Rationale

In one sparse persistent pipeline, bypassing only bit-proven identity IDs
reduced TIRx latency from 25.919 to 25.549 us, a 1.43% improvement, and measured
1.006x against the reference. The retained default-NVRTC profile was spill-free,
and exact source plus independent-oracle correctness passed.

A broader rewrite derived counts and offsets, skipped planner prefetches, and
synthesized the tail payload. It was value-correct but regressed to 26.926 us,
3.89% slower than the accepted starting point. Removing memory traffic is not
automatically a win when it changes generated scheduling; isolate the one
dependent load the proof makes unnecessary.

## Boundary

Identity must be established from the exact planner contract for every work
record accepted by the compiled specialization. A semantic label such as
"full" is insufficient: an attempted tail-payload synthesis failed correctness
because the payload also encoded query-row validity. Even a bit-identical
payload synthesis can regress scheduling, so do not bundle it with index
elision.

Tie the proof to compile-time specialization keys. Runtime shapes that can
select a non-identity plan must use the indirect fallback.

## Verification

Compare materialized planner indices with logical positions across every work
record and edge tile before compiling the fast path. Run exact source checks and
an independent oracle for the specialized case plus an irregular fallback.
Finally, use the production compiler and benchmark the isolated index bypass;
also verify in SASS that the intended dependent load disappears without changes
to the remaining planner prefetch and payload schedule.
