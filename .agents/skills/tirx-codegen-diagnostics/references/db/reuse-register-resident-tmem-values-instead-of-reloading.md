# Reuse register-resident TMEM values instead of reloading

**Symptoms:** `tmem_wait`, `long_scoreboard`, `exposed_load_latency`, `dispatch_specific_deficit`, `data_dependent_correctness_failure`

## Symptom

A per-iteration path issues a second `tcgen05.ld` of accumulator cells whose
values are already live in registers from an earlier read in the same
synchronization window, and the specialization that exercises the path trails
the reference at matched protocol. The source often carries the same redundant
read, so instruction parity hides it.

A numerical variant reloads cells after they were repurposed, although the
consumer needs the earlier snapshot rather than their current value.

The redundant read can also be a global-memory reload of an input already
transposed through TMEM into a live register fragment. Confirm the coordinate
mapping and conversion semantics before forwarding that fragment.

## What to change

When no TMEM store to those cells intervenes between the first read and the
reuse site -- same acquire, same release -- forward the live registers and
delete the second load chain. The gate judges time, not fidelity to a
redundancy the source happens to carry.

If the consumer needs the earlier snapshot across a TMEM overwrite, retain the
existing, unmodified register fragment and delete the reload of the overwritten
cells. This is distinct from substituting stale registers for a current value.

For an input fragment already needed by later output arithmetic, use static
fragment indices to unpack the required elements. If that requires loop
unrolling, measure an unroll-only ablation as well as register forwarding:
unrolling itself changes instruction scheduling and static site counts.

```python
# before: the snapshot path re-reads the cells the repack just read.
state = txl.alloc_local((FRAGMENT,), "float32")
_tmem_load_fragment(state, STATE_COLUMNS)
_publish_packed(state)
with txl.If(do_snapshot), txl.Then():
    snapshot = txl.alloc_local((FRAGMENT,), "float32")
    _tmem_load_fragment(snapshot, STATE_COLUMNS)
    _stage_snapshot(snapshot)

# after: the registers still hold the cells' current value.
state = txl.alloc_local((FRAGMENT,), "float32")
_tmem_load_fragment(state, STATE_COLUMNS)
_publish_packed(state)
with txl.If(do_snapshot), txl.Then():
    _stage_snapshot(state)
```

## Rationale

Inline-PTX TMEM loads are opaque: no backend proves two of them redundant, and
the second one is a long-latency read sitting on the per-iteration critical
path, not shadowed arithmetic. Removing four 32-lane-wide loads per iteration
on a path that fired every iteration moved the focused shape from 0.977x to
0.991x with the guard shapes at 1.001x-1.034x, and the change survived the
complete correctness matrix and the final complete performance-matrix winner.

The win is specific to long-latency loads. Pure instruction-count trims in the
same warpgroup -- folding negations, de-duplicating a replicated register
array, about 620K dynamic operations together -- measured neutral: that work
sat in stall shadow.

Preserving tail-score registers across an overlapping probability store fixed
a sparse-prefill LSE error: maximum absolute error fell from 0.02354 to
4.77e-7 against an independent oracle, also matched by three SM100 launches.
Producer x64 TMEM-load sites fell from four to three, retaining 128 registers
and zero spills on SM100 and SM103. Three paired 15-round SM100 workloads had
after/before time ratios of 0.984-1.002; this is not an SM103 performance result.

In a BF16 gated-backward specialization, forwarding an already-live packed key
fragment removed 32 global-load sites relative to an equally unrolled control.
Launch register metadata stayed at 168, stack at 96 bytes, and static local
loads/stores at 44/26. Against the rolled implementation, two initialized-gate
profiles improved 6.6% and a dense strong-decay profile improved 1.2%. Directly
against unrolling alone, initialization improved 1.4-4.8% but dense decay
regressed 2.1%. Thus forwarding and unrolling gains are not additive across
independent runs. All 39 registered cases and three BF16 boundary profiles
retained all six output tensors byte for byte under the original assertions.

## Boundary

For current-value forwarding, an intervening TMEM store to the same cells, or a
barrier that admits one, makes the registers stale. An earlier-snapshot consumer
may cross that store only if the registers retain the required snapshot.
Neither form establishes asynchronous memory ordering; required lifetime waits
remain a separate obligation.
Forwarding must not stretch the fragment's live range into a later region
either -- keeping a wide fragment alive across an independent load batch pushed
a warpgroup roughly sixty registers past its budget and spilled, regressing the
affected shapes to 0.69x and 0.79x. Reuse pays where the fragment was already
live for another purpose at the reuse site.

An identity-MMA transpose is not automatically equivalent for every input bit
pattern: verify normal values, signed zeros and denormals under the actual
rounding and FTZ behavior. The measured boundary checks establish output
equivalence for those cases, not a universal input-bit preservation proof.
Do not substitute a shared-memory reread merely because it has the same
coordinates: aliased staging storage may already be overwritten by another
warp. Private register forwarding needs no new shared-memory lifetime claim.

## Verification

Confirm the `tcgen05.ld` site count drops in generated code for the affected
specialization and that registers do not spill, then measure the affected and
guard shapes; an instruction drop alone is not evidence, since shadowed-work
removals measure neutral.
For snapshots, also check numerical results with an overwrite between the
original read and the consumer.
For global-to-fragment forwarding, compare against an equally unrolled control
when counting removed loads; expansion can increase total static sites versus
the rolled baseline even while removing every redundant reload.
