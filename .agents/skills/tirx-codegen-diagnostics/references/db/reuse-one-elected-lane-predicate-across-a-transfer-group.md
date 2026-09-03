# Reuse one elected-lane predicate across a transfer group

**Symptoms:** `repeated_expression`, `instruction_count_bloat`, `branch_reconvergence`, `slow_latency_bound_shape`

## Symptom

A warp-specialized loop elects a leader separately around several adjacent
single-lane transfers.  The transfers belong to one logical issue group and no
operation between them changes warp membership or the active mask, but generated
SASS contains multiple `ELECT` instructions per loop iteration.

## What to change

Elect the leader once before the transfer group and reuse the resulting predicate
for every guarded issue in that group.

```python
# before: each helper call emits another warp election.
with K.If(_elect_one()):
    issue_first_transfer(...)
with K.If(_elect_one()):
    issue_second_transfer(...)

# after: one election owns the complete transfer group.
leader = _elect_one()
with K.If(leader):
    issue_first_transfer(...)
with K.If(leader):
    issue_second_transfer(...)
```

## Rationale

An elected-lane helper is an effectful warp operation, not a common scalar
expression that later compiler passes can reliably merge.  Repeating it in a
deep transfer loop duplicated both election and control-flow work.

In one measured rewrite, dynamic warp instructions fell from 6.61 million to
5.33 million, static SASS fell from 928 to 824 instructions, and static `ELECT`
sites matched the reference at three.  Register allocation and spill behavior
were unchanged.  The affected 4096-square row moved from 0.962x to 1.010x, all
14 affected and guard workloads passed, the complete 88-workload matrix had a
minimum ratio of 0.9959x, and all 59 correctness configurations passed.

## Boundary

Reuse is valid only while every lane that participates in the election remains
in the same active warp and no intervening divergence can change the active
mask.  Re-elect after reconvergence boundaries, changes in participating lanes,
or control flow whose inactive lanes do not execute the original election.  Do
not extend one predicate across unrelated pipeline phases merely to reduce the
static instruction count.

Keep the election inside the control region it governs. In one counterexample,
moving an election above a CTA-uniform leader branch and reusing it for that
branch's barrier issue plus two following transfer regions was numerically
correct but changed the generated schedule catastrophically: only 6 of 34
targeted rows passed and the minimum ratio was 0.6662x. The apparent active-mask
equivalence did not make the three regions one profitable issue group. Hoist
only after the source PTX/SASS shows one election governing the same contiguous
region.

A second counterexample combined this hoist with plain-scalar descriptor
halves in a 16-warp block-scaled attention matrix-issue role (four
`tcgen05.cp` scale copies plus two block-scaled MMAs per group, thirteen-slot
FP4 operand ring). The two target rows did not move (18.54 and 126.07 us) and
the long 4096-key stream regressed from 131.6 to 167.2 us (1.012x -> 0.797x).
The two changes were applied together, so the regression is not attributed to
either alone; the reviewed per-instruction form was restored and the sentinel
recovered to 131.6 us. Measure the long-stream sentinel of every ring depth
before keeping a hoist that only the latency-bound shape motivated.

## Verification

Confirm in SASS that the hot loop contains one election for the logical transfer
group, compare reconvergence instructions, registers, stack, and spills, then run
both the affected and guard performance workloads plus the complete correctness
matrix.
