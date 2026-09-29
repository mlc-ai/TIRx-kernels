# Sweep register budgets by role and shape

**Symptoms:** `register_spill`, `excess_address_math`, `low_occupancy`, `register_budget_mismatch`, `schedule_regression`

## Symptom

Spills, address hoisting, or occupancy loss that shifts across shape regimes or
warp roles.

## What to change

Sweep neighboring register budgets per warp role on representative single-wave
and multi-wave shapes. The budget is the first statement of each role branch,
and the increases must be paid for by matching decreases elsewhere.

```python
if warpgroup_idx == 0:  # compute and epilogue
    T.ptx.setmaxnreg.inc.sync.aligned.u32(144)
    ...
elif warpgroup_idx == 1:  # producer
    T.ptx.setmaxnreg.dec.sync.aligned.u32(96)
    ...
else:  # matrix-issue role
    T.ptx.setmaxnreg.inc.sync.aligned.u32(168)
```

Note `setmaxnreg.sync.aligned` is a four-warp collective: every warp of the
warpgroup must reach it, including otherwise idle ones.

Re-run the sweep after changing descriptor placement, fragment width, or other
live ranges.

## Rationale

Producer, compute, and epilogue warps can need materially different register
budgets. Compiler register level can also trade spills, address hoisting, and
occupancy differently across shape regimes.

One four-warp epilogue sweep was sharply non-monotonic. Requested budgets 48,
56, and 64 passed 1/5, 4/5, and 2/5 targeted rows respectively. Applying 56 to
every output later drove one packed-output specialization to a ptxas allocation
failure at 255 entry registers and regressed several non-singleton paths.
Keeping 56 only on its measured single-cluster BF16 beneficiary and restoring
native allocation elsewhere removed the compile cliff and recovered every FP8
output guard.

A separate FP32 path was limited to one CTA per SM by shared memory and was
long-scoreboard dominated, so raising its epilogue budget from 56 to 64 could
not reduce occupancy. The scoped 64-register form passed all 12 affected FP32
rows on two GPUs, retained zero failures across the correctness matrix, and
helped the final 66-row suite clear its strict gate at a 0.9907x minimum. The
same value had regressed other outputs when applied globally.

Budget redistribution can matter even when total allocation and spill counts
stay fixed. One 16-warp pipeline kept its collective total at 2048 registers
while changing three role budgets from 192/80/48 to 192/88/40. Both forms had
zero local and shared spilling, but the correction-preserving allocation reduced
elapsed cycles from 23,294 to 22,896 and executed instructions from 11,406 to
11,377. It subsequently passed complete targeted and full matrices at 0.995x
and 1.002x minimum ratios.

Instruction count can point at the same role-budget problem from the opposite
direction. One seven-warp persistent pipeline already executed 5,276 fewer
basic-opcode thread instructions than its reference, yet used 72 registers per
thread against 89 and executed 9,269 more synchronization try-wait warp events;
the accumulator-empty wait alone ran 12,926 times against 6,532. Giving only
the four-warp epilogue role an 88-register temporal target moved the directly
affected ratio from 0.9804x to 1.0134x. All 35 correctness configurations and
seven affected or guard configurations passed, and the complete 28-row matrix
cleared at a 0.9908x minimum; an equivalent final build repeated the complete
matrix at a 0.9932x minimum. Once the target already has the shorter instruction
stream, a register deficit plus excess consumer-release polling is a reason to
sweep the consumer role rather than keep deleting arithmetic.

The ptxas `--register-usage-level` knob has the same regime dependence and can
flip as a step, not a slope. One 16-warp block-scaled attention forward with a
quantized P was 18.46 us at levels 4-10 and 15.47 us at levels 1-3 on a
single-wave grid (64 tiles on 152 SMs, eight blocks per tile; reference 16.23
us, ratio 0.877x -> 1.049x), with identical executed-instruction counts and an
identical softmax-role binary: only the matrix-issue and correction roles were
rescheduled (R2UR 46 -> 25, IMAD 71 -> 48 in the issue role). The same level 3
ran 168.8 us against 130.9 us (+28%) on the 4096-key, 24-head stream of the
same kernel, and every other mode of the kernel measured flat across levels
3-10. The level was therefore keyed on the grid regime (`num_tiles <= num_sms`)
for that one specialization; a per-mode key alone would have traded the small
shape against the long streams.

Adding a cold numerical fallback to a 12-warp persistent backward kernel grew
the generated local stack from 16 to 80 bytes even on inputs that skipped the
fallback. After shortening that fallback's live ranges, a fresh role-budget
sweep helped: eight compute warps and four auxiliary warps at 208/88 took
155/261 us on two ordinary grouped-head shapes; 224/56 took 149/243 us,
232/40 took 143/238 us, and 240/24 regressed to 146/245 us. All keep the same
64,512-register CTA allocation. Mixed fallback predicates passed at every
retained point. The fused family responded differently, so its budget was
measured separately. Earlier results from a larger fallback had not predicted
this sweep; repeat it after changing the fragment and branch structure.

Adding exponent balancing and explicit diagonal cancellation changed that
conclusion again. On a fused long-chain workload, 224/56, 232/40, and 240/24
measured approximately 1070, 972, and 898 us, respectively, against the original
kernel's 844 us in alternating same-device rounds. The grouped-head family did
not benefit: 232/40 and 240/24 were about 161 us. The previously rejected
240/24 budget therefore became worth retaining for the fused experiment only;
the earlier sweep was evidence for the old live ranges, not a permanent cap.

A phase-specific split was not better in that experiment. Restoring 208/88
for the recurrence streams, then redistributing to 232/40 at the token phase
boundary, left ordinary timings near 144/238/155/226 us and moved the large
strong-decay case from roughly 513-518 to 523 us. Correctness passed, but the
extra collective transitions did not justify retaining another allocation
policy. A phase boundary makes redistribution legal, not automatically useful.

An unshifted derivative rewrite changed the tradeoff again. For the same fused
shape,240/24 used104 stack bytes with57/36 static local loads/stores;
232/40 used80 bytes and42/22, while224/56 used112 bytes and48/29. Four paired
profiles retained every output bitwise at each point. Relative to240/24,
232/40 improved ordinary latency3.05% but regressed initialization, sparse,
and dense corrections3.07%,5.45%,and3.47%. The224/56 split similarly improved
ordinary2.50% but regressed those corrections3.89%,8.79%,and6.05%. Neither
reduced-cap split was retained: improved ordinary spilling did not justify
slower workloads for which the numerical fallback was added.

The separately compiled ordinary body had another allocation cliff at the
same total register budget. Moving compute/auxiliary roles from 208/88 to
216/72 increased the stack from 16 to 216 bytes and local load/store sites
from 26/36 to 98/104. Two ordinary profiles regressed 6.5–6.7%; inactive-path
controls remained flat. Moving to 200/104 instead used a 24-byte stack with
29/39 sites and improved ordinary timing only 0.57–0.63%. All eight paired
profile checks preserved output bytes. That small opposite-direction gain
shows why spill counts alone cannot rank neighboring allocations, but it did
not resolve the remaining complete-operation latency target.

Combining that 200/104 allocation with a separately measured reciprocal-lifetime
reduction restored a 16-byte stack and 27/36 local load/store sites. Four
additional paired checks remained bytewise equal, but ordinary timing was flat
on the shorter workload and regressed 1.0% on the longer one. The individual
sub-percent gains did not compose; fewer spills than the reduced-budget
variant alone did not make the combined schedule faster than the original
allocation.

Warp count is likewise not a useful register knob without inspecting the
resulting allocation. In an independent D128/V128 gradient prototype, reducing
eight warps to four lowered the reported register count from 255 to 168 but
increased spills from roughly 396/366 to 1108/1076 across two shapes. The
gradient phase grew to 3134/5524 us and the complete operation to 32.8x/36.1x
the original. Ten FP64-oracle cases passed, so the result isolates a performance
failure rather than an arithmetic change.

Fragment width had its own non-monotonic point. Sweeping the value tile from
16 through 128 made 64 the best complete-call width at 19.73x/22.11x, while
128 was the fastest gradient-only width but slowed the recurrent state phases.
Using width 64 for recurrence and 128 for gradients improved the complete call
to 15.72x/17.56x and passed twenty FP64-oracle cases. Select widths by phase
and complete-operation timing; a phase-local optimum can enlarge another
phase's live fragment enough to lose overall.

## Boundary

An occupancy proof only shows that a larger budget is affordable; it does not
show that its schedule is better. Scope a budget by the compile-time role,
fragment/output family, cluster regime, and resource limit that produced the
measured response. Let unrelated paths use native allocation when a shared
budget changes their entry allocation or trips an allocator limit.

Two constraints bind before any sweep, and only the first is usually checked.
The budget is warpgroup-uniform, so roles sharing a warpgroup cannot move
independently. The budget must also balance globally: increases are funded only
by what the decreasing roles release. Against a 96-register default, one kernel's
decreasing roles released 14,336 registers for increasing roles that required
12,288; raising one producer role's budget dropped the released pool to 11,264,
and the increasing roles then blocked forever inside `setmaxnreg.inc`. It
compiled cleanly and hung at runtime -- a shape that normally finished in 0.034 s
did not complete in 240 s. Compute the release-versus-require arithmetic for the
whole kernel, not for the role being changed.

Check whether registers bind at all before sweeping. Zero local and zero shared
spilling rules out an obvious capacity failure, but does not prove that role
budgets are schedule-insensitive: redistribution at a fixed total can still
change issue order and the critical path. Increasing the total cap without a
resource argument remains unlikely to help. Where a sweep did pay, it was
narrow: a collective-issuing warpgroup improved by 0.1-1.0% at 56 registers,
and funding a further rise to 72 out of the reduce role regressed the shapes
that role dominates.

Eliminating a spill counter is also not sufficient acceptance evidence. A
shape-scoped rebalance removed six shared-spilling requests, yet its two
critical benchmark ratios were only 0.981x and 0.989x. Continue through the
performance matrix after the generated-code symptom is repaired.

Changing the warp layout to distribute live fragments can change matrix
instruction selection before it changes the register budget. In an independent
SM100 recurrence component, increasing eight warps to sixteen replaced120
static `tcgen05.mma` sites with480 `mma.sync` sites and removed TMEM use. The
reported per-thread allocation fell from255 to32, but the stack grew from136
to7664 bytes and static local load/store sites from28/31 to5587/4386.
Twenty-five state-oracle checks still passed. Two complete-component timings
grew from937/1638 us to21593/38465 us in alternating same-device rounds.
This was a different tensor lowering, not evidence of successful register
relief or absence of tensor-core instructions. Inspect the selected matrix
instruction family and accumulator placement whenever a warp-count change
appears to produce an unexpectedly low register count; do not carry this
particular compiler response over to another lowering without checking it.

A source register count is only a seed for the sweep. In the seven-warp case,
88 was the nearest legal target below the reference's observed 89, but adjacent
budgets were not measured and the child was not re-profiled. The timings prove
that the scoped register intervention repaired the deficit; they do not prove
that 88 is optimal or that the measured try-wait count itself fell. Do not copy
the value or claim the counter mechanism without the corresponding sweep and
post-change profile.

In a no-new-buffer native epilogue, the default208/88 compute/auxiliary split
was also the best fixed-total allocation. Moving to200/104 was effectively
flat, while216/72,224/56,232/40, and240/24 made the long grouped workload
roughly10-13% slower versus an original kernel. Sweeping ptxas register levels
2-8 found level3 best, but its complete-operation ratio was still1.056 versus
the original and therefore missed a5% acceptance limit. A role or compiler
budget sweep can bound the remaining cost without producing an acceptable
point; retain the arithmetic/synchronization diagnosis and reject the patch
rather than treating the best sampled configuration as a win.

After that epilogue was narrowed to a launch-uniform cold correction, the
compiler-level result changed again: default level10 cost 5.31% and 4.44% on
short and long grouped workloads, while forcing level3 cost 9.42% and 7.72%.
The earlier level3 minimum did not transfer across the changed branch and live
ranges. Re-run the compiler-level sweep after structurally narrowing a fallback;
do not retain a level selected for its larger predecessor.

Neighboring role budgets did not repair the narrowed form either. Its default
208/88 compute/auxiliary split assembled with a 56-byte stack and 35/46 static
local load/store sites. Moving to 200/104 produced 80 bytes and 40/51 sites;
moving to 216/72 produced 240 bytes and 108/119 sites. The latter gave compute
warps more registers but starved the auxiliary group that now owned the risk
reduction. Inspect the whole specialized function: a budget intended to relieve
one role can move the stack cliff into the producer of a new cold-path flag.

## Verification

Record realized allocation and dynamic local traffic, not only the requested
cap. Compile every specialization touched by the selector so allocator cliffs
cannot hide outside the timing set, then measure the beneficiary and guard
paths at adjacent budgets. When excess pipeline polling motivated the sweep,
re-profile the retained child and compare the same barrier waits before
attributing the gain to faster stage release.
