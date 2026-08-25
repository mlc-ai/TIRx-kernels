# Peel runtime state before hiding TMEM latency

**Symptoms:** `runtime_predication`, `reconvergence`, `tmem_wait`, `long_scoreboard`, `latency_hiding`

## Symptom

A recurrent warp role issues TMEM loads and immediately waits before consuming
them. There is independent shared-memory work that could cover the TMEM latency,
but moving the loads across that work splits a runtime `have_state` predicate
into two regions. Generated code gains a hot reconvergence trampoline, uniform
register transfers, and scheduling NOPs; the intended overlap then performs
worse than the original immediate wait.

## What to change

Peel the first iteration when it is the only iteration with different state
ownership. Generate one first-iteration path with compile-time-known state
presence and one steady-state loop with state presence fixed true. Only then
issue the recurrent TMEM loads before the independent shared-memory loads,
keeping the TMEM wait immediately before the first dependent conversion or
arithmetic instruction.

```python
# before: moving the load would split one runtime predicate into two regions.
with K.serial(num_chunks) as local_chunk:
    have_state = local_chunk > 0
    load_v()
    load_w()
    with K.If(have_state), K.Then():
        wait_state_ready()
        load_state_from_tmem()
        K.ptx.tcgen05.wait__ld.sync.aligned()
    consume_v_w_and_state()

# after: first-iteration semantics and recurrent semantics are specialized.
load_first_v_w_without_state()
with K.serial(num_chunks - 1) as recurrent_chunk:
    load_v()
    wait_state_ready()
    load_state_from_tmem()
    load_w()  # independent work covers part of the TMEM latency
    K.ptx.tcgen05.wait__ld.sync.aligned()
    consume_v_w_and_state()
```

## Rationale

The overlap is useful only when it does not lengthen hot control flow. In one
recurrent prefill kernel, applying the load motion while `have_state` was runtime
added one `BSSY`/`BSYNC` pair, three static `R2UR`, and eight static NOPs. Its
nine-row affected performance matrix fell to a 0.9169 minimum and 0.9506 mean
ratio.

Peeling the first chunk removed the recurrent state predicate. Reapplying the
same load motion then moved both target `LDTM` instructions ahead of the two W
`LDSM` instructions while leaving the parent's counts unchanged: 10 `BSSY`, 10
`BSYNC`, 54 `R2UR`, and 49 NOPs. The complete candidate matrix passed with a
0.9949 minimum and 1.0030 mean ratio; the final complete matrix passed with a
0.9953 minimum and 1.0029 mean ratio.

## Boundary

The peeled iteration must be the only semantic exception, and all recurrent
iterations must have state ready under the same pipeline protocol. The TMEM
wait is not removed: it stays before first use. The work moved between load and
wait must be independent of both the TMEM destination registers and the barrier
that proves the TMEM accumulator ready. Keep the original order for paths where
state presence remains runtime-dependent; the measured kernel does this for its
large-checkpoint specializations.

## Verification

Check generated CUDA or line-info SASS for the intended `LDTM` -> independent
work -> `wait::ld` order, then compare `BSSY`, `BSYNC`, `R2UR`, and NOP counts
against the parent. Reject the change if the overlap creates another runtime
predicate region. Run affected correctness, including the peeled first chunk
and recurrent chunks, then run the complete performance matrix.
