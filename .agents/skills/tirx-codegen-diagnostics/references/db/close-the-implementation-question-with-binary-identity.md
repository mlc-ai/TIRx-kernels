# Close the implementation question with binary identity

**Symptoms:** `unstable_benchmark`, `gate_flapping`, `sass_equivalence`, `interfered_run`

## Symptom

A required-shape matrix reports a gate crossing on a port whose generated CUDA,
fatbin, and final SASS are byte-identical to the build it is compared against.

## What to change

Nothing in the kernel. Exact executable identity settles the implementation
question: there is no instruction-selection difference left to find, so raise
the round count and retest on a quiet card instead of inventing a change to
explain the ratio.

## Rationale

Byte-identical builds cross the gate under timing pollution alone, and the
crossing does not survive a clean interleaved campaign.

Two byte-identical CUDA/fatbin/SASS pairs crossed at 1.0204 and 1.0231 in a
15-round matrix, then measured 1.0052 and 0.9995 in a clean interleaved
45-round retest. A later 45-round matrix put three more byte-identical pairs at
1.0103, 1.0152, and 1.0395, while a quiet-card interleaved 45-round targeted A/B
measured 1.00114, 0.99978, and 1.00031 with zero retries.

## Boundary

The two claims are distinct: executable identity closes the implementation
question, and the targeted rerun adjudicates only the polluted timing claim. A
targeted rerun cannot excuse a different binary, and it cannot replace
full-roster coverage.

A five-round cross-session overlap check reported after/before regressions of
1.0351, 1.0183, and 1.0130; a quiet-card interleaved 45-round targeted A/B
measured 1.00123, 1.00447, and 1.00189 with zero retries. Two of those pairs
were executable-identical, but the third had a different binary, so its rerun
proved a clean performance pass and not binary parity.

## Verification

Diff generated CUDA, fatbin, and final SASS before accepting a targeted rerun as
adjudication, and state which of the two claims the evidence supports.
