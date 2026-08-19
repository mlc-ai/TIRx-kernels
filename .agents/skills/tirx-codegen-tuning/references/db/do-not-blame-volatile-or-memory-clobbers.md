# Do not blame volatile or memory clobbers

**Symptoms:** `scheduling_barrier_suspicion`, `short_scoreboard`, `unexplained_small_shape_deficit`

## Symptom

A scheduling deficit that looks caused by `asm volatile` helpers and memory
clobbers -- roughly 460 such barriers per warp in one kernel.

## What to change

Nothing on this lever. Generated `T.ptx` helpers are `asm volatile`, and global
loads and stores carry a memory clobber; removing either is not a recoverable
win. Look elsewhere for the deficit.

## Rationale

Removing both, one at a time and together, left the kernel at 6.014-6.020 us
across eight independent timings on the quiet shape, with the profiler's
short-scoreboard stall unchanged. The ratio column moved only because the
reference wandered.

An earlier experiment on the same kernel appeared to recover a point, but it
removed barriers and switched to fast-math forms at once; isolating the halves
showed the barriers contributed nothing and the fast-math swap was a divergence
the parity contract forbids.

## Verification

Change one thing per measurement; time each half of a combined edit in isolation
before crediting either.
