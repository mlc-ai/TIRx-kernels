# Reuse register-resident TMEM values instead of reloading

**Symptoms:** `tmem_wait`, `long_scoreboard`, `exposed_load_latency`, `dispatch_specific_deficit`

## Symptom

A per-iteration path issues a second `tcgen05.ld` of accumulator cells whose
values are already live in registers from an earlier read in the same
synchronization window, and the specialization that exercises the path trails
the reference at matched protocol. The source often carries the same redundant
read, so instruction parity hides it.

## What to change

When no TMEM store to those cells intervenes between the first read and the
reuse site -- same acquire, same release -- forward the live registers and
delete the second load chain. The gate judges time, not fidelity to a
redundancy the source happens to carry.

```python
# before: the snapshot path re-reads the cells the repack just read.
state = K.alloc_local((FRAGMENT,), "float32")
_tmem_load_fragment(state, STATE_COLUMNS)
_publish_packed(state)
with K.If(do_snapshot), K.Then():
    snapshot = K.alloc_local((FRAGMENT,), "float32")
    _tmem_load_fragment(snapshot, STATE_COLUMNS)
    _stage_snapshot(snapshot)

# after: the registers still hold the cells' current value.
state = K.alloc_local((FRAGMENT,), "float32")
_tmem_load_fragment(state, STATE_COLUMNS)
_publish_packed(state)
with K.If(do_snapshot), K.Then():
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

## Boundary

Valid only inside one synchronization window: any intervening TMEM store to the
same cells, or a barrier that admits one, makes the registers stale.
Forwarding must not stretch the fragment's live range into a later region
either -- keeping a wide fragment alive across an independent load batch pushed
a warpgroup roughly sixty registers past its budget and spilled, regressing the
affected shapes to 0.69x and 0.79x. Reuse pays where the fragment was already
live for another purpose at the reuse site.

## Verification

Confirm the `tcgen05.ld` site count drops in generated code for the affected
specialization and that registers do not spill, then measure the affected and
guard shapes; an instruction drop alone is not evidence, since shadowed-work
removals measure neutral.
