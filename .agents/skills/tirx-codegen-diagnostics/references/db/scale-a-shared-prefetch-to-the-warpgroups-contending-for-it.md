# Scale a shared prefetch to the warpgroups contending for it

**Symptoms:** `short_scoreboard`, `insufficient_memory_parallelism`, `dispatch_specific_deficit`, `schedule_regression`

## Symptom

Batching a shared-memory pass into load-then-convert helps one dispatch and
regresses a sibling. The two differ only in how many warpgroups run the same
pass at the same time.

## What to change

Treat the prefetch depth as a property of the concurrency on that memory, not
of the pass, and take it from the same compile-time predicate that decides how
many warpgroups run it. One warpgroup converting a tile wants its loads batched
so the dependent conversion chain stops sitting on the critical path of the next
load; two warpgroups converting different tiles at once do not, because the
batch only makes twice the threads burst into the same shared memory.

Keep the contended form rolled. A batched rewrite normally replaces a rolled
loop with a counted one over the batch, which unrolls -- so the two forms differ
in prefetch depth *and* in code size unless the rolled path is written back with
a `While`.

```python
# `batch` comes from the dispatch predicate, not from a global constant.
def _convert_tile(src, dst, batch, iters):
    if batch == 1:
        i = T.alloc_local((1,), "int32")
        i[0] = 0
        while i[0] < iters:                 # contended: stays rolled
            words = _load_chunk(src, i[0])
            _convert_and_store(dst, words, i[0])
            i[0] = i[0] + 1
    else:
        staged = T.alloc_local((4 * batch,), "uint32")
        for group in range(iters // batch):  # uncontended: batch, then convert
            for b in range(batch):
                _load_chunk_into(staged, src, group * batch + b, b)
            for b in range(batch):
                _convert_and_store_from(staged, dst, group * batch + b, b)
```

The staging buffer's extent must be a module-level constant. A bare assignment
inside a traced body binds a TIR variable, and the allocation stops being a
constant-size one -- which only shows up on the dispatch that instantiates the
path.

## Rationale

Batching every chunk load before converting any took the single-warpgroup
dispatch from 0.9862 to 1.0410 and the two-warpgroup sibling from 1.0210 to
0.9868. Cutting the depth from the whole tile to four did not recover the
sibling (0.9857), and passing it depth 1 did not either (0.9864) -- so neither
register pressure nor depth was the cause. What remained was the loop form: the
batched rewrite iterated with a counted `range`, which unrolls, where the
original was rolled. Giving each dispatch the form it measured best in recovered
both, at 1.0421 and 1.0208.

## Boundary

The same split governs staging depth outside shared memory. Issuing every
tensor-memory read of an epilogue before its single wait, instead of draining
one column pass at a time, gained on eight shapes of one matrix and lost on two
-- the specializations whose operand dtype left least room for the extra live
fragments, one of them by 13%. Binding the form to the compile-time predicate
that names the operand dtype and the load program kept every gain and undid both
regressions. Raising a load pipeline from two stages to three splits the same
way, helping one dtype and costing another 0.023.

A targeted subset will hide this. The four-shape set used while iterating showed
that epilogue rewrite improving everything it touched; only the complete matrix
exposed the two regressions. Judge a depth or form change on every dispatch that
takes each branch, not on the shapes that motivated it.

The depth is not transferable between kernels or between passes. Concurrency is
one axis among several that set it: the per-thread trip count sets how much
parallelism there is to extract, and the live range of the staged registers sets
a ceiling above which the depth costs more than it returns. All three can bind
at once, and only a sweep separates them.

## Verification

When one rewrite changes two things, vary each alone before believing either
explanation. Two cheap variants -- depth first, then loop form -- separated
them here, and the first two attributions were both wrong. Measure the sibling
dispatches together: a change keyed to a compile-time predicate can only be
judged on the configs that take each branch.
