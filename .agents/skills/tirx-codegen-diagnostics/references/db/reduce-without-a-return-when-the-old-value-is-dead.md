# Reduce without a return when the old value is dead

**Symptoms:** `long_scoreboard`, `slow_epilogue`, `slow_small_shape`

## Symptom

Long-scoreboard stalls at an epilogue's global accumulate, where the atomic
writes its returned old value into a local nothing reads.

## What to change

Use the non-returning `red.global.*` form for an accumulate whose result is
discarded, and delete the dead destination.

```python
# before: the warp waits on a global round trip for a value it drops.
previous = K.local_scalar("float32")
K.ptx["atom.global.add.f32"](previous, buffer.ptr_to([index]), value)

# after: same accumulate, no destination, no scoreboard entry.
K.ptx["red.global.add.f32"](buffer.ptr_to([index]), value)
```

## Rationale

`atom` names a destination register, so the issuing warp holds a scoreboard
entry until the value comes back from memory even when the value is never used;
`red` performs the same reduction with no destination and no wait. In one
epilogue two such accumulates -- a per-expert maximum and a per-row sum -- moved
the row they dominate from 0.944/0.947 to 0.970, a real 2.5% on what was then
the worst deficit, with the guards unchanged.

## Boundary

Only where the return is genuinely dead. A work-claiming counter in a dynamic
scheduler consumes exactly that value and has to stay an `atom`; auditing by
mnemonic rather than by use will break it.

## Verification

Check that no `red` site has a live destination in the generated code, and
confirm the long-scoreboard stall at the accumulate falls rather than moving
somewhere else.
