# Release shared tiles after the last scalar reader

**Symptoms:** `nondeterministic_outputs`, `shared_memory_race`, `barrier_stall`

## Symptom

A scalar fallback reads a shared intermediate after the final tensor-core
consumer. The loader still treats tensor-core completion as permission to reuse
that tile. Outputs are finite but incorrect and change between identical
launches.

## What to change

Extend the tile's reuse protocol to cover its new scalar readers. If delaying an
existing release also delays unrelated transfers, give the new reader lifetime
its own release. Every participant must arrive once per phase, including lanes
that skip the fallback.

```python
# before: the new scalar use outlives the old tile ownership.
wait_tensor_consumers()
start_next_tile_load()
read_scalar_correction()  # races with the next tile

# after: producer reuse follows the actual final reader.
read_scalar_correction_if_needed()
scalar_readers_free.arrive()
# loader role:
wait_tensor_consumers()
scalar_readers_free.wait()
start_next_tile_load()
```

## Rationale

One measured correction extended the lifetime of two derivative tiles into a
scalar epilogue. The previous tensor-completion wait allowed the next gate load
to overwrite them. Query/key normalized RMS errors were approximately 0.65-0.70
and identical launches disagreed. Waiting for the scalar epilogue removed the
large errors and repeatability failures. After compacting the correction into
an earlier pass, a dedicated release preserved correctness while allowing gate
and upstream-gradient prefetch before the remaining epilogue finished. Three
normal-range workloads moved from about 1115/190/323 to 1110/187/318 microseconds.

## Boundary

A shorter lifetime must account for scratch reuse as well as the original
operand. In the measured extension, the scalar pass reused old state tiles for
correction outputs. Their existing later release still had to protect epilogue
reads after the derivative tiles could be released.

Warp ballots or other full-mask collectives used to select a fallback must be
materialized before lane divergence. A barrier arrival by each lane does not
make a warp collective inside a partial-lane branch valid.

A new tensor residual can create the opposite prefetch problem: placing it in
old reduction/input storage forces the next input transfer to wait for a later
matrix completion. A measured alternative moved two residual tiles into old
state storage after its last matrix consumer. The original state-reuse barrier
already outlived the new residual readers, so the next input prefetch could
resume at its original release. A retained compute-group join also protected
prior scalar readers; later scalar scratch writes still waited for tensor
completion. Ten paired cases preserved every output bitwise and the complete
29-case matrix passed. One family improved1-2% on correction-heavy inputs;
the other gained2.1% on one mixed profile but regressed2.9% on its dense control
despite stack reduction48->40 bytes. Prefer an already compatible lifetime
when available, but measure both overlap and compiler-scheduling effects.

A precision-preparation pass can also change an operand before a remaining
state tensor contraction, even without a race. In a measured rewrite, an
intermediate completion marker preceded the final state use of a tile that
was then rebuilt for derivatives. One correlated input changed the state
FP64 normalized RMS error from0.001605 to0.004520. Moving that final state
contraction before the marker preserved its original accumulation order and
restored bitwise equality of both state-dependent outputs against the
baseline across24 inputs. All six outputs repeated exactly, and static
resource and opcode totals were unchanged. The broader matrix retained its
pre-existing numerical failures; this establishes the reader-lifetime fix,
not general accuracy of the new derivative formulation.

Publishing completed scalar writes requires a release from the writers as well
as protecting the eventual reuse. In a measured cross-warp cache, lanes wrote
disjoint token halves, used a warp join, then lane zero arrived with a count
of32. Racecheck still reported26 read/write hazards across10 mixed-path cases;
the older bounded baseline also reported40 hazards on four strong cases.
Replacing that counted arrival with one arrival per writing lane cleared the
same10-case run: zero hazards or warnings and all numerical checks passed.
The arrival count, consumer wait, and reuse release were otherwise unchanged.
This establishes a sanitizer-clean publication protocol; numerical passes alone
did not distinguish a hardware race from incomplete synchronization modeling.
The hot-path cost must still be measured before claiming a performance benefit.

Another measured alias had a cheaper solution than extending the release. A
rounded derivative diagonal remained in `T5` for a scalar epilogue, while the
next gate TMA also targeted `T5`. Delaying the common Q/K/V/gate release made
four ordinary shapes 4.4-7.4% slower. Splitting only the gate write reduced
three shapes below5% but left a long grouped shape at6.0%. The retained scratch
alternative moved the next gate destination to `T6`, whose last tensor reader
was already covered by the loader's existing `chunk_done` wait. Eight targeted
cases then passed repeatedly without a new barrier. The complete operation
still regressed about5.3-6.0% because of the correction arithmetic, so it was
not accepted. When an alias extends a scalar lifetime, first look for a region
whose existing completion wait already proves it dead; then qualify the whole
operation independently of the synchronization win.

The physical replacement tile can itself dominate the cost even when an
existing completion wait proves the reuse safe. In an ablation of that same
epilogue, moving the gate from `T5` to `T6` without any correction arithmetic
regressed the complete long workload by 6.66%. A tile-base sweep found bases 0
and 2 unsafe, producing nonfinite query gradients, while bases 4 and 6 passed
the default correctness case but regressed 6.77% and 6.86%. Existing lifetime
proof is necessary for correctness; it does not imply a neutral physical stage
or schedule. Measure the relocation alone before attributing the combined cost
to the new arithmetic.

Aliasing an output as transient global scratch was substantially worse in the
same epilogue. Packing 64 BF16 diagonal values into otherwise unwritten `db`
slots, synchronizing the four consumer warps, reading each pair in the q and k
passes, and finally overwriting every slot with its normal result passed eight
directed cancellation cases; both kernel families returned exact zero for the
affected gradient. Five-round complete-operation timings nevertheless regressed
13.56% on the packed fused workload and 18.24% on the long grouped workload.
The buffer required no new allocation and every address was work-item-private,
but repeated global reloads plus another workgroup join dominated. An output's
unused lifetime proves alias legality, not that it is a cheap substitute for a
shared tile or short live range.


A helper warp that already waited for the rounded tile provided a cheaper
on-chip publication point than a compute-group join, but the benefit depended
strongly on kernel duration. It copied 64 raw BF16 diagonal values into an
existing shared region, and the scalar q/k epilogues loaded that copy. Eight
directed cases passed with exact cancellation. Complete-operation timing
regressed only 0.99% on the long fused workload, but 16.17% on the shorter
grouped megakernel. The same shared-memory protocol can fit under an exposed
latency budget in one family and dominate another; qualify each dispatch family
separately, and avoid per-work-item publication on short persistent schedules.

An ablation later separated publication from consumption in that grouped
megakernel. Moving the same 64-value helper copy after the final matrix reader
and retaining its fence and release, but deleting every compute-warp scalar
load and correction, changed the long complete operation from 230.76 to
230.68 microseconds (-0.04%). The combined correction remained 14.19% slower.
Thus a helper publication that appears expensive only in the combined patch
can be fully hidden; measure a copy-only child before moving storage or adding
another lifetime protocol. In this case the consumer arithmetic and control
flow, rather than the shared producer, owned essentially all exposed cost.

## Verification

Map each shared region through producer, tensor consumer, scalar consumer, and
next producer. Test multiple successive work items, mixed fallback predicates,
partial chunks, and repeated poisoned launches. Confirm phase progression for
lanes that skip work, then measure the prefetch overlap that the separate release
is intended to restore.
