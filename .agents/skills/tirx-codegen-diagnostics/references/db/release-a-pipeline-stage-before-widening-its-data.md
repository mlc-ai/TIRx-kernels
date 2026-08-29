# Release a pipeline stage before widening its data

**Symptoms:** `producer_starvation`, `underfilled_pipeline`, `epilogue_serialization`

## Symptom

A consumer loads a stage into registers, converts it, then releases the stage.
The pipeline has few stages and the producer has no slack.

## What to change

Fence and release as soon as the bytes are in registers; convert afterwards.

```python
for word in range(0, words_per_row, 4):
    K.ptx["ld.shared.v4.b32"](raw[word], raw[word + 1], raw[word + 2], raw[word + 3], ...)
K.ptx["fence.proxy.async.shared::cta"]()
with K.If(K.lane_id() == K.int32(0)), K.Then():
    K.ptx.mbarrier.arrive.shared.b64(pipe.empty.ptr_to([consumer.stage]))
consumer.advance()
_widen(fragment, raw)   # after the release, not before
```

## Rationale

The conversion does not touch shared memory, so holding the stage across it buys
nothing and costs the producer a refill it could already have started. A CUTLASS
reference will usually keep its fragments in the source dtype and convert at the
point of use, which is the same thing by another route. On a four-bit
specialization with two stages against two consumed per subtile -- zero slack --
this was worth about 1%.

## Boundary

Worth the most where stages are few; on a deep pipeline the producer is already
far enough ahead that the hold is invisible.

## Verification

Machine-code counts will not change -- the mbarrier arrive is an ordering point
ptxas cannot move the conversion across, so program order is the whole
mechanism. Measure it.
