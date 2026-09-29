# Share sparse channel corrections across warp lanes

**Symptoms:** `branch_divergence`, `underfilled_pipeline`, `register_spill`, `schedule_regression`

## Symptom

A channel-wise scalar fallback executes in only one or two lanes, but keeps the
entire warp busy for a long token scan. A small channel fallback fraction can
therefore affect most warps and produce nearly dense-fallback latency.

## What to change

Ballot the fallback channels before divergence. For sparse masks, enumerate
active channels and assign otherwise idle lanes to their token rows. Retain
the existing channel-per-lane schedule for dense masks. Preserve the bounded
decay products, diagonal treatment, and accumulator order within each row.

Join all helper lanes before the original channel owners release the shared
derivative tiles. An arrival by the owners alone does not prove that helpers
have finished reading or publishing their corrections.

## Rationale

One implementation assigned four channel groups per warp, eight lanes per
group, and four consecutive token rows per lane. It retained the dense loop
above twelve selected channels. Sixteen targeted correctness cases and fourteen
paired performance cases passed with unchanged tolerances. At roughly 5%
selected channels, two shapes improved from 5943 to 3078 us and from 646 to
406 us. At 12.5%, they improved from 5952 to 3712 us and from 647 to 454 us.

## Boundary

The sparse branch can enlarge register lifetimes even when not executed. In
the same experiment, one specialization's local stack grew from 64 to 328
bytes and its ordinary path regressed from 144 to 170 us. The other kept a
16-byte stack and ordinary latency near 875-878 us, but dense-fallback latency
still increased about 6%. Synthetic mixed decay distributions regressed about
7-11%. These results support sparse cooperation as an intervention, not that
particular cutoff, allocation, or unconditional promotion.

Changing only the affected specialization's ptxas register-usage level to 5
reduced its stack to 8 bytes. A new fourteen-case paired matrix passed with
unchanged tolerances. Its ordinary path was 143.63 versus 142.26 us for the
bounded baseline, sparse 5% was 344.71 versus 646.73 us, sparse 12.5% was
390.79 versus 647.50 us, and dense was 639.25 versus 643.71 us. Synthetic
initialization profiles improved about 1-2%. This removed that specialization's
cold-path spill regression, but did not resolve the other specialization's
6-8% dense and initialization regressions. Compiler scheduling and sparse
cooperation need separate acceptance checks.

Sharing only one channel per whole warp helped one- and two-channel masks but
was slower around four channels. The number of cooperating lanes and rows per
lane needs its own crossover measurement. Reducing scalar work does not prove
that the complete kernel improved.

The benefit also depends on the surrounding fast path. Combining four-group
cooperation with guarded exponent balancing passed twelve paired cases under
unchanged tolerances. One synthetic initialization profile improved from
3792 to 2401 us, while a less variable profile increased from 1432 to 1447 us.
The fully dense control increased from 6025 to 6433 us. Its local-load/store
instruction counts grew from 41/23 to 49/31 despite unchanged stack and register
allocations. The other specialization reduced local loads from 201 to 33 and
stack from 40 to 16 bytes; sparse cases improved by 13–41%, but its ordinary
control still increased by about 0.8%. Neither static spill counts nor a single
gate distribution establish a net win. Separate conditioning stress tests also
limited the surrounding balancing policy; this work assignment does not repair
that policy.


Adding compensated tensor operands and predicating their unused residual
arithmetic changed the composition again. Ten paired cases retained bitwise
identical outputs, but the first family's dense control regressed10.4% and
ordinary/init controls1.6-1.9%, while variable initialization and sparse12.5%
improved33.5%/36.6%. The second family's sparse case improved38.5%, but its
local stack grew40->96 bytes and a variable initialization control regressed
2.8%. A register-lifetime win from an earlier fast path is not preserved by
an independently useful predicate change; remeasure the actual composition.

## Verification

Test empty, singleton, partial-group, cutoff, and dense masks, including tails
and repeated poisoned outputs. Check that helpers cannot release or overwrite
shared storage early. Inspect register allocation and local traffic for cold
and hot fallback paths, then measure sparse benefits alongside ordinary and
dense controls. Keep amplitude and gate-range stress tests separate: changing
the work assignment does not make an unsafe exponential factorization safe.
