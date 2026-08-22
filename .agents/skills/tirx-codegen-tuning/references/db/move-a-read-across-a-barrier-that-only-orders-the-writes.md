# Move a read across a barrier that only orders the writes

**Symptoms:** `barrier_stall`, `exposed_load_latency`, `long_scoreboard`

## Symptom

A global load and the shared store that consumes it sit a few instructions
apart immediately after a barrier, and that store is the heaviest-stalling
store site in the kernel -- on the one this was measured on, 517 stall samples
against 2.2% of every sample the kernel took.

## What to change

Ask what the barrier actually orders. A barrier in front of a shared write is
normally there to stop that write from racing the previous iteration's readers.
It says nothing about where the stored *value* comes from. When the value comes
from an input the kernel only reads, issue the read ahead of the barrier into a
local and leave only the store behind it, so the wait absorbs the memory latency
instead of the store paying it afterwards.

```python
# before: the read is issued after the barrier and its consumer is a few
# instructions behind it, so the wait and the latency are paid in series.
bar_sync_named(BAR_ID, PARTICIPANTS)
for i in T.unroll(N):
    if guard(i):
        st_shared_i32(dst, slot + i, ld_global_i32(src, index(i)))
    else:
        st_shared_i32(dst, slot + i, 0)

# after: every read is issued first and the barrier is paid on top of their
# latency. The else-arm folds into the pre-initialized local, which also drops
# one store site.
words = T.alloc_local((N,), "int32")
for i in T.unroll(N):
    words[i] = 0
    if guard(i):
        words[i] = ld_global_i32(src, index(i))
bar_sync_named(BAR_ID, PARTICIPANTS)
for i in T.unroll(N):
    if guard(i):
        st_shared_i32(dst, slot + i, words[i])
```

## Rationale

Inline-PTX loads are opaque to the scheduler, so nothing moves them for you: a
read written after a barrier stays after it, and its consumer waits out the full
memory latency with the whole warpgroup already synchronized.

The trade is instructions for overlap, not instructions saved. Staging the
values costs a local array and a pre-initialization, and measured on a metadata
publish the rewrite raised executed global loads from 173,425 to 266,698 and
shared stores from 611,442 to 704,222 at warp level. It still won: five
alternating paired rounds, 5/5, 0.76% on the target shape, moving its mean ratio
from 0.9875 to 1.0000, and a second shape on the same load program gained 1.6%
without being aimed at. Expect the instruction count to go up and judge the
change on time.

The stall breakdown confirms where the time comes from rather than leaving it
inferred. Samples move out of memory latency and into issue: long_scoreboard
25.3% to 24.7%, no_instructions 10.8% to 10.3%, while selected rises 12.6% to
13.0% and not_selected 4.7% to 5.2%. More warps are ready and fewer are blocked
on a load, which is the whole of the trade.

## Boundary

Only for reads of data the kernel does not write. A value another warp publishes
into shared memory, or a global buffer this kernel also stores to, is precisely
what the barrier exists to order -- hoisting that read is a correctness bug, not
a tuning change. The staged values also occupy registers for the length of the
barrier; on a warp already near its budget the trade can invert, and the same
rewrite that wins on one operand dtype can lose on the one with less headroom.

## Verification

Judge this one on measured time. The instruction count moves the wrong way by
design, so an instruction-count check will read as a regression, and a
per-opcode stall comparison against the reference will not motivate it either --
on the kernel this was measured on the two sides' shared-store stall shares were
2.40% and 2.47%, indistinguishable. What identified the site was its own stall
count, not a gap against anything.

Three ways to misread the machine code here, all of them hit while measuring
this.

A SASS listing keyed on the first token of each line silently drops every
`@P0`-guarded instruction, and guarded loads and stores are exactly what a
publish inside two range checks compiles to. That one filter produced three
different false conclusions in a row: a load-volume gap against the reference
that did not exist, a set of sites the port supposedly had and the reference did
not, and a shared-store stall share off by more than twentyfold. Take the opcode
from the second token when the first begins with `@`.

Shared-memory offsets are assigned per binary, so the same numeric offset can
name different buffers before and after a change. A store that looks like it
moved or vanished at a given offset proves nothing across two builds.

An apparent removal of work deserves a test rather than an explanation:
deliberately corrupt the value being published and confirm the bitwise gate
fails. If it still passes, what was removed was dead work.
