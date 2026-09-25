# Do not blame volatile or memory clobbers

**Symptoms:** `scheduling_barrier_suspicion`, `short_scoreboard`, `unexplained_small_shape_deficit`

## Symptom

A scheduling deficit that looks caused by `asm volatile` helpers and memory
clobbers -- roughly 460 such barriers per warp in one kernel.

## What to change

Nothing on this lever. Generated `T.ptx` helpers are `asm volatile`, and global
loads and stores carry a memory clobber; removing either is not a recoverable
win, and on a shared-memory algorithm it is not even available. Look elsewhere
for the deficit.

## Rationale

Removing both, one at a time and together, left the kernel at 6.014-6.020 us
across eight independent timings on the quiet shape, with the profiler's
short-scoreboard stall unchanged. The ratio column moved only because the
reference wandered.

The clobber is also load-bearing rather than conservative, which closes the
lever for good on kernels that stage through shared memory. It is derived, not
chosen -- any instruction carrying an address operand gets it -- and an
address-only memory model leaves ptxas no aliasing information of its own. A
local build rendering plain `ld`/`st` without `volatile` and without `"memory"`
made one kernel compute garbage, every element wrong, because the clobber was
the only thing ordering a counter read-modify-write against the scatter and
gather that follow it. Recovering that scheduling freedom needs an aliasing
model, not a flag.

An earlier experiment on the same kernel appeared to recover a point, but it
removed barriers and switched to fast-math forms at once; isolating the halves
showed the barriers contributed nothing and the fast-math swap was a divergence
the parity contract forbids.

## Verification

Change one thing per measurement; time each half of a combined edit in isolation
before crediting either.
