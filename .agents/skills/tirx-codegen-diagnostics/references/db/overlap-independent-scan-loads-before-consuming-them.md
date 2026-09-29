# Overlap independent scan loads before consuming them

**Symptoms:** `serial_global_loads`, `long_scoreboard`, `slow_small_shape`

## Symptom

A scan already uses wide, coalesced loads and enough CTAs, but consumes each
load before issuing the next independent iteration. The scan adds a measurable
fixed cost to an otherwise unchanged operation.

## What to change

Unroll a small group of independent iterations, issue their loads into separate
registers, and only then reduce their predicates. Advance by the whole group.
Keep the original partitioning and final reduction.

```python
# before:
with txl.While(position < end):
    load_vector(payload, address(position))
    update_predicate(payload)
    txl.assign(position, position + stride)

# after:
with txl.While(position < end):
    for group in range(4):
        load_vector(payloads[group], bounded_address(position + group * stride))
    for group in range(4):
        update_predicate(payloads[group])
    txl.assign(position, position + 4 * stride)
```

The helper names illustrate the load and predicate sites. `bounded_address`
must preserve the scan's result, not merely make the address legal.

## Rationale

In a measured chunk-end range scan, SASS changed from one `LDG.E.128` followed
by its comparisons to four independent loads before those comparisons. Both
builds used 32 registers, zero stack, and zero local traffic.

With identical inputs, outputs, workspaces and main executable, substituting
only the compiled scan reduced its ordinary latency from 6.24 to 3.75 us. The
active compute kernel remained 143.62 versus 143.64 us. A magnitude-checking
profile reduced the scan from 17.14 to 14.61 us with its compute kernel also
unchanged. Complete ordinary calls improved about 1.6%; a larger family saved
about 1.1 us in the scan but gained little at operation level.

Fifty range-predicate cases agreed with an independent oracle and the original
flags, including short tails, threshold equality and adjacent values, magnitude
rejection, and returning to mild input. Twenty complete-operation profiles kept
all six outputs and decision flags bitwise equal, and the 39-case registered
matrix passed. Four additional short/long layouts reduced complete ordinary
latency by 1.0–1.3% and initialization latency by 0.35–0.60%; their longer
gate scans fell from about 9.43 to 5.41 us.

## Boundary

The measured reduction was existential. Clamping excess positions to the last
real chunk end duplicated an already included predicate and therefore preserved
its result. The same technique would change a sum or a population count. Do not
read undefined padding or assume arbitrary duplicate loads are harmless.

More outstanding loads increase register lifetimes and may reduce occupancy.
Inspect the emitted issue order and resource allocation, and measure the entire
scan, including later branches that share its register budget.

Independent allocation layouts initially produced much larger, mixed changes
in total latency. Those numbers did not isolate the scan intervention. In the
controlled experiment the compute executable and all addresses were identical;
its attributed time remained stable while the scan changed.

Widening a short consumer-entry flag scan was much less useful. Replacing
scalar loads with aligned four-word loads plus a bounded scalar tail emitted
`LDG.E.128`, but retained the 16-byte stack and 26/36 local load/store sites of
the enclosing kernel. Four paired profiles kept all output bytes; complete
ordinary calls improved only 0.05–0.34%, and inactive-path controls were flat.
That result does not justify treating memory width alone as the remaining
latency mechanism. Tail decisions need independent coverage before adopting
such a scan rewrite; those four full-operation profiles did not cover every
possible flag-array remainder.

## Verification

Inspect load issue order, registers and spills. Compare predicates against an
independent oracle on full groups, partial groups and repeated input changes.
Check complete outputs under the original tolerances, and measure the whole
operation as well as the affected scan.
