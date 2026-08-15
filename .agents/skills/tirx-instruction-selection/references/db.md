# TIRx instruction-selection DB

## E1: Materialize and forward reused values

**Symptoms:** `repeated_expression`, `excess_address_math`, `excess_unpack_math`, `instruction_count_bloat`, `register_count_gap`

TIRx expressions are trees. Reusing a `PrimExpr` can emit its complete subtree
at every use; ptxas does not reliably recover the intended common subexpression.

Bind swizzle offsets, unpacked lanes, scale products, and other reused values to
local scalars or buffers exactly once. Helpers used multiple times should return
materialized values rather than rebuild an expression.

Confirm by counting instructions in the corresponding PTX/SASS basic block. A
1.5-2x count increase with the same control flow is a typical signal.

The reverse direction has a boundary. Hoisting address invariants the backend
already merges changes almost nothing: one such hoist moved static SASS by five
instructions and left the shift count untouched. But no backend merges across
opaque inline asm, so where the reference's own compiler CSEd two accesses to
the same element, the port must reuse the earlier value explicitly; two excess
loads and a redundant integer max disappeared that way.

An in/out inline-PTX destination written directly into a dynamically indexed
local buffer is another opaque boundary. If the next instruction consumes the
just-written value, bind the PTX result to a scalar, write that scalar back, and
forward it to the next instruction instead of rereading the buffer expression.
In one recurrent state update this changed no PTX line count, but reduced the
realized register allocation from 60 to 48 and static SASS from 1080 to 1000
instructions; `IMAD.MOV.U32` fell from 84 to 31 and `MOV` from 45 to 20, with no
spill. The resulting build then cleared all 43 gate workloads. Inspect ptxas
resources and SASS for this pattern; source or PTX size can remain unchanged.

## E2: Pin floating-point instructions

**Symptoms:** `bitwise_mismatch`, `denormal_mismatch`, `unexpected_ftz`, `select_lowered_as_branch`

Fast-math defaults can add `.ftz`, approximate division, or change a compare and
select into different control flow. This causes bitwise mismatches on denormals
and may perturb scheduling even when normal values agree.

When the reference pins an instruction, use the exact PTX operation: non-FTZ
`mul.f32`/`add.f32`, `div.rn.f32`, or explicit `setp` plus `selp`. Retain
`.approx.ftz` only where the reference uses it. Plain TIRx remains appropriate
for integer and index math.

Confirm with denormal inputs and an instruction-by-instruction PTX comparison.

## E3: Match launch bounds

**Symptoms:** `register_spill`, `register_budget_mismatch`, `local_memory_traffic`, `low_occupancy`

`tirx.launch_bounds_min_blocks_per_sm` becomes the second CUDA
`__launch_bounds__` argument and imposes a hard ptxas register budget. A value
chosen from theoretical occupancy can starve a kernel whose reference uses more
registers, producing STL/LDL or global rescheduling.

Set the bound from the reference kernel's realized occupancy target. Compare
resource usage, achieved occupancy, and dynamic local-memory traffic on both
sides; do not infer success from the declared launch bound alone.

Do not copy one minimum-block value across block-size families. In one measured
selector, a 160-thread family used nine minimum blocks and a representative
FP16-state specialization moved from 53 to 40 registers, while its 288-thread
family used one minimum block and stayed at 53 registers. Forcing nine on the
larger block cut its allocation to 32 registers before timing. The shape-aware
9/1 selector cleared its five-workload boundary matrix at 1.003-1.028x. Treat
such a large allocation shift as a separate shape A/B even when neither variant
spills: ptxas can trade registers for recomputation and address instructions.

## E4: Defeat harmful LSR

**Symptoms:** `imad_prologue_chain`, `excess_address_math`, `register_pressure`, `slow_small_shape`

When nvcc sees a loop stride as an ordinary integer expression, loop-strength
reduction may build a long pointer-induction chain in the prologue. The setup
cost and live addresses can dominate short workloads even if each loop
iteration becomes locally cheaper.

If the reference recomputes compact addresses inside the loop, pass only the
offending stride or bound through an opaque identity PTX move. Do not obscure
data-path values indiscriminately.

Confirm by comparing the SASS before the first loop branch, plus registers and
latency on the smallest shapes.

## E5: Match conversion and packing idioms

**Symptoms:** `bitwise_mismatch`, `packing_mismatch`, `excess_shift_or`, `store_width_mismatch`

- Check operand order for packed conversions; paired FP formats commonly take
  the high element first. Use asymmetric correctness inputs.
- Pack two 16-bit halves with the native two-input 32-bit move and two 32-bit
  halves with the corresponding 64-bit move.
- For an unavailable packed form, reuse a native equivalent already validated
  in another kernel instead of inventing a new lowering.
- Match shuffle masks and saturation/rounding modifiers exactly.

Confirm packed words bitwise before judging instruction count.

Keep conversion width, packing width, and transaction width independent. A
validated 128-bit state store used eight scalar FP32-to-16-bit conversions, four
two-input `mov.b32` packs, and one `st.global.v4.b32`; ptxas then selected paired
`F2FP.*.F32.PACK_AB` instructions feeding the 128-bit SASS store. A packed PTX
conversion is therefore not required to obtain a packed store. Trace the
conversion-to-store def-use chain in SASS instead of comparing conversion
mnemonics in isolation.

## E6: Prefer binary parity

**Symptoms:** `local_rewrite_regression`, `sass_divergence`, `register_allocation_change`

In a parity port, a semantically cheaper rewrite can change global ptxas
scheduling, register allocation, and numerical behavior. Local instruction
count is not the target.

First match PTX shape, register count, and SASS structure. Deviate only after a
measured full-matrix result. A generated-code improvement that does not improve
the production benchmark is not a retained optimization.

## E7: Map SASS in both directions

**Symptoms:** `unattributed_sass_hotspot`, `instruction_count_gap`, `register_count_gap`

Use line information to walk both sides of a gap:

```text
slow shape -> dynamic opcode difference -> SASS PCs -> generated source
           -> TIRx operation / reference instruction block
```

Preserve generated CUDA, PTX, cubin, and line information. Compare SASS and
registers, not only PTX: ptxas performs the final selection and scheduling.

Counting is where this walk goes wrong silently. A `--source-in-ptx` dump echoes
the reference as `//` comments, so a plain opcode grep also matches echoed
inline-asm text; inline-asm bodies are brace-wrapped and predicated statements
carry an `@%pN` prefix, so a line-anchored regex undercounts both; and statement
lines are not instructions -- one body measured 5112 against 3553, turning a
phase's true 55% share into a reported 39%. Line information names several
files, and inlined code is attributable only through its `inlined_at` field, so
keying on the leading line number files every shuffle and packed conversion
under an unrelated region.

Never claim two instruction streams are identical from opcode tokens alone. One
comparison that matched form-by-form on tokens showed 15 of ~25 shapes differing
once register, uniform-register, and immediate operands were normalised: half
the shared loads took a uniform-register offset where the reference took a pure
immediate, and the constant-bank traffic split between the uniform datapath and
per-thread loads. That identity claim had been load-bearing for a "no lever
remains" conclusion.

A difference that does not survive to SASS is not an alignment target. One port
carried an extra shuffle in every PTX dump, synthesized by the toolchain for any
kernel that takes a thread index as a launch parameter; dynamic SASS shuffles
matched the reference exactly, so there was nothing there to fix.

## E8: Keep benchmark evidence clean

**Symptoms:** `unstable_benchmark`, `interfered_run`, `gate_flapping`

Shared GPUs move ratios. Iterate with targeted workloads, but accept or reject
only from the latest complete required-shape matrix. Let the suite retry
interfered measurements.

Promote a baseline only from a complete clean run, never by hand-editing ratios
or merging partial evidence. Keep timer scope explicit: kernel-only timings are
not end-to-end latency.

Four failure modes, each observed:

- Sample gating removes contaminated samples, not session-to-session drift. The
  same build measured 3-4% apart across sessions with zero samples discarded, so
  an A/B is valid only inside one interleaved campaign.
- The suite's intruder check does not catch everything. Two builds emitting
  identical code for a shape measured 1.077 and 0.995 while both were reported
  ok; one row silently absorbed about 8%. Re-measure any single-row claim inside
  about 5% on a quiet card, and verify clocks before and after.
- Choosing which rows to measure by what you expect to find hides regressions.
  Removing a shape-conditional knob after A/B-ing only the rows that wanted it
  gone moved an unmeasured row from 1.077 to 0.855. Run the whole matrix
  whenever a knob is touched -- the knob exists because some row disagreed.
- Editing the kernel while a matrix runs corrupts that run: harnesses re-import
  per config and later subprocesses measure a body that no longer exists. Check
  the module digest across the run, and pin the device on every side command so
  it does not land on the benchmark GPU.

Raising round count converts drift into margin: dips that appear at five rounds
can vanish at fifteen, which means the margin was inside the noise, not above
it.

Read absolute times, not only ratios. The contaminated run that motivated two
code changes reported plausible ratios while its absolute times ran about 50%
high, and that signal was available at the time and went unchecked. A single
crossing is not a pass either: one row at 0.9977 was called a first pass on a
host that drifts 1-2% between runs on identical builds. And when the failing row
*moves* between runs while within-run variance stays at a few hundredths, the
deficit is drift rather than a per-shape effect.

Gate samples on the device the measurement actually uses. A gate requiring that
no foreign benchmark process exists never fires on a busy machine, and one
requiring every card idle blocks on cards the measurement never touches. Check
that the device is quiet immediately before and after each sample, discard the
sample if it moved, interleave the variants, and keep a comparison inside one
campaign.

## E9: Issue loads before conversions

**Symptoms:** `long_scoreboard`, `insufficient_memory_parallelism`, `exposed_load_latency`

For cold-cache, memory-latency-bound kernels, issue independent global loads
before starting dependent widening, unpacking, or conversion chains. If raw bits
can remain live safely, sink the first conversion behind independent work or a
real synchronization boundary.

This has produced 1.7-3.7% gains in short recurrent kernels and much larger
gains when multiple token loads were previously serialized. It can regress when
the staged raw values spill or the loads usually hit cache.

This regime is not optional in the gate: the benchmark harness zeroes a 256 MB
buffer before every timed iteration, so the measured kernel always starts with
an empty L2 and the figure of merit is how many misses are outstanding.

Design the ablation carefully. Deleting the conversion arithmetic to test
whether it costs anything removed 95 instructions and changed the time by
nothing, because the consumer still depended on the load -- that experiment
answers whether the arithmetic is expensive, not whether the latency is exposed.
The measurement that does answer it is the scheduled-SASS distance from the last
load to its first consumer, which moved from 17 to 215 instructions when the
conversion was written later. Program order is the lever; which side of a
barrier ptxas finally places the work is not something the kernel controls.

Confirm the load issue window and load-to-first-consumer distance in SASS, then
measure long-scoreboard stalls, registers, spills, and the complete shape matrix.

## E10: Bound hoisting by live-range cost

**Symptoms:** `register_pressure`, `local_memory_traffic`, `low_occupancy`

Hoist work only when hidden latency outweighs the added lifetime. A small
operation can be expensive when its results remain live across a recurrent loop,
large fragment, or synchronization chain.

One measured FP32 bias hoist regressed 6.6%; by contrast, staging a larger load
set won when it created enough outstanding DRAM misses. Tile wide epilogue
fragments so only the next consumed tile remains live.

Compare registers and dynamic LDL/STL before instruction count, and sweep the
tightest specialization where one spill can reverse the result.

## E11: Expose predication and uniform control

**Symptoms:** `branch_reconvergence`, `warp_divergence`, `excess_control_instructions`, `branch_in_hot_loop`, `serialized_stores`

For one isolated load or store, express the predicate on the PTX instruction
when an outer branch blocks if-conversion. For a loop-invariant uniform
condition, hoist it and duplicate the hot loop only when that exposes a dense
path without changing recurrence state.

When the condition is runtime at kernel entry but constant throughout the CTA,
an inline TIRx helper with a `T.constexpr` mode can force the two branch bodies
to specialize independently: dispatch once on the runtime condition, then call
the helper with `True` or `False`. Replacing repeated pad checks with this shape
removed 262,144 dynamic `CS2R` instructions in the profiled specialization by
letting each copy dead-code-eliminate the opposite path. This is a control-flow
lowering tool, not permission to change dispatch semantics; test both copies and
compare code size, registers, and every affected workload.

Do not predicate substantial computation or duplicate a body that causes
instruction-cache pressure, spills, or lower occupancy.

The reference's source text does not reveal whether it wants this. nvcc
routinely duplicates a loop around a store predicate the reference wrote per
iteration, so transcribing the text faithfully keeps a per-iteration branch the
reference never compiles to. Detect it by counting static arithmetic against the
reference: a recurrence block appearing an odd multiple of its logical count is
duplicated, and matching that multiple is the target.

One store-heavy recurrence carried the predicate per row and issued stores too
sparsely to saturate DRAM -- 61.1% against the reference's 62.5%, at identical
occupancy with no spill. Hoisting the predicate and duplicating the loop matched
the reference's static counts exactly and moved the largest shapes from 0.988x,
1.000x and 1.003x to 1.013x, 1.023x and 1.026x.

Predicating a store is an operand-level change, and the API distinction matters:
marking an operand as a predicate register is not the same as `@p` predication,
which is a separate keyword on the instruction. One output store written as a
guarded branch cost a real branch plus a reconvergence per CTA; reissuing it as
a predicated store matched the reference exactly at BSYNC 9216 to 6144 and BRA
7680 to 6144, and moved two shapes from 0.982x and 0.988x to 1.007-1.019x and
1.012x.

Not every predicate is worth rewriting. Rebuilding an integer-materialized
condition as a boolean conjunction, aimed at an excess of logic ops and
reconvergence, changed nothing: both forms lowered identically, down to equal
totals. Check the lowering before assuming the written form survived.

Confirm predicate polarity and inactive-lane memory behavior, then compare BRA,
BSYNC, reconvergence, code size, registers, and both control-flow outcomes.

## E12: Materialize uniformity

**Symptoms:** `warp_retry_region`, `vectorized_uniform_math`, `excess_address_instructions`

If every participating lane holds the same value but control flow hides that
fact, ptxas may emit vector integer/address work plus retry or collective
regions. Materialize a warp-uniform proof or broadcast only after proving the
value is identical for every active lane and mask.

Measured cases removed WARPSYNC/ENDCOLLECTIVE regions and tens of thousands of
vector address instructions. Whole-function complexity can still determine
uniform placement, so a local rewrite is not guaranteed to flip codegen.

A reference may already carry the proof as an instruction that looks like an
identity: a broadcast of a value every lane in the warp already holds. Dropping
it as redundant removes the only evidence ptxas has that the guard it feeds is
warp-uniform. One such omission wrapped a phase's reductions in a
WARPSYNC/ENDCOLLECTIVE retry region whose back-edge spanned most of the kernel,
adding nine shuffles to the reference's count and fourteen warp syncs to its
zero; reinstating the broadcast restored the reference's exact shuffle count and
moved two shapes from 0.997x to 1.003x and 1.000x.

The cost appears only where the guard is live, so a specialization launching
exactly the guarded warp count shows nothing and a wider one pays. Sweep the
geometry where the guard actually excludes warps.

Confirm vector versus uniform op counts, branch topology, registers, and the
full workload matrix.

## E13: Select address lowering by shape

**Symptoms:** `excess_address_math`, `register_pressure`, `schedule_regression`

Native `[base+imm]` addressing removes explicit pointer arithmetic only when
the byte offset is a compile-time immediate. It can still alter scheduling,
allocation, dependency chains, or pointer lifetimes.

Keep native offsets and explicit pointer arithmetic as alternatives until a
full shape picker is measured. One FP32 MTP matrix selected native offsets for
56 of 97 configurations and explicit arithmetic for the other 41; neither form
was globally best.

Likewise, replacing `step * stride` with a moving cursor is not intrinsically
cheaper. Five live cursors in one recurrent specialization increased registers
from 58 to 60 and static SASS from 1072 to 1080 instructions, with no spill and
no consistent full-matrix gain. ptxas had already strength-reduced much of the
original indexing, while the explicit cursors extended five live ranges and
added loop-back updates. Retain cursor induction only when the emitted address
chain and the affected shapes demonstrate a gain.

Confirm normalized PTX/SASS addresses, register counts, integer address ops,
spills, and latency per specialization.

## E14: Tune pipeline shape and transfer granularity

**Symptoms:** `barrier_stall`, `exposed_epilogue_tail`, `underfilled_pipeline`, `late_stage_completion`, `tma_issue_overhead`

Choose pipeline depth from wave count and exposed latency, not from the largest
possible ring. Single-wave work may benefit from deeper accumulator buffering;
multi-wave work may need more producer stages. Never reduce a protocol ring
below its proven safe depth.

Split a large TMA box only when earlier sub-box completion benefits a real
steady state. Preserve alignment, swizzle atoms, coverage, and exact barrier
byte accounting. Extra TMA issue instructions often regress short shapes.

Benchmark both sides of every dispatch boundary and validate deadlock freedom,
footprints, registers, stage completion timing, and issue counts.

## E15: Merge waits with a protocol proof

**Symptoms:** `redundant_barrier_wait`, `serialized_teardown`, `exposed_store_tail`

Collapse waits only when they guard the same consumer dependency and all
producers can contribute exact byte counts to one completion condition. Delay a
tail drain when kernel exit or a later protocol edge already supplies the
required ordering.

Before editing, write the producer-consumer happens-before argument. Do not
merge visibility domains, permit ring overwrite, remove a release witness, or
weaken cross-stream ordering.

Validate transaction counts, ring wrap, completion visibility, and deadlock
freedom before profiling wait stalls and tail-dominated shapes.

## E16: Size swizzles and fragments physically

**Symptoms:** `smem_bank_conflict`, `register_spill`, `slow_epilogue`

Derive shared-memory swizzles from the lane-to-bank map. Keep a live register
fragment no wider than the next consumed tile, especially across barriers and
epilogue casts.

A measured swizzle removed a 96x store-conflict gap; tiling a 128-register
epilogue fragment down to at most 16 live registers removed dynamic local
traffic. Smaller is not automatically better when it adds synchronization or
breaks vector alignment.

Shared-pipe conflicts appear even where a kernel allocates no shared memory,
because shuffles use that pipe and divergence serializes them. One such kernel
ran at nearly double the reference's conflict rate with zero shared allocation;
removing the branch around a guarded store took its conflicts to zero against
the reference's 612.

Measure bank conflicts, static registers, dynamic LDL/STL, and writeback depth
together across all affected shapes.

## E17: Sweep register budgets by role and shape

**Symptoms:** `register_spill`, `excess_address_math`, `low_occupancy`

Producer, compute, and epilogue warps can need materially different register
budgets. Compiler register level can also trade spills, address hoisting, and
occupancy differently across shape regimes.

Sweep neighboring budgets on representative single-wave and multi-wave shapes.
Re-run the sweep after changing descriptor placement, fragment width, or other
live ranges. Record realized allocation and dynamic local traffic, not only the
requested cap.

## E18: Declare the block shape the reference declares

**Symptoms:** `special_register_reads`, `instruction_count_bloat`, `slow_latency_bound_shape`

A multi-dimensional `thread_id` costs a special-register read per component at
every use. When the reference launches a flat block and derives warp, lane, and
group indices by division, a 2-D declaration adds nothing and is charged for on
every access.

One latency-bound kernel declared a `(32, warps, 1)` block where both the
reference and the approved sketch specified a flat one. It cost 4608 dynamic
`S2R` plus 4608 `S2UR`; flattening moved the worst shape from 0.864x to 1.016x
and removed 48 static instructions, all in the affected phase.

Take the block shape from the reference, not from a scaffold's correspondence
table, and confirm with dynamic special-register counts on the worst shape.

## E19: Treat a dead tail's unrolling as a change, not a cleanup

**Symptoms:** `instruction_count_bloat`, `unreachable_code_expansion`, `dead_tail_loop`

A loop with a runtime bound that is empty in every dispatched regime is still
unrolled. One orphan tail loop expanded roughly 33x, emitting 70 narrow global
stores where the reference emitted 18, and disabling unrolling removed 112
static instructions from the phase.

Removing that expansion is not automatically free. The same change measured a
small gain on the contaminated run that motivated it, then measured 0.924-0.979
on one shape across five later campaigns, reproducibly the worst variant there;
that shape cleared the gate only after the unroll change was reverted. Code that
never executes costs issue bandwidth and instruction cache, not cycles, so the
static win is large while the runtime effect is small and can be negative.

Identify the expansion, then put the change through the complete matrix like any
other. Do not land it alongside a second change: these two were introduced
together, and attributing their combined delta to the other one produced a
specialization axis that had to be withdrawn.

## E20: Do not blame volatile or memory clobbers

**Symptoms:** `scheduling_barrier_suspicion`, `short_scoreboard`, `unexplained_small_shape_deficit`

Generated `T.ptx` helpers are `asm volatile`, and global loads and stores carry
a memory clobber. Roughly 460 such barriers per warp look like an obvious cause
of a scheduling deficit. Measurement says otherwise: removing both, one at a
time and together, left the kernel at 6.014-6.020 us across eight independent
timings on the quiet shape, with the profiler's short-scoreboard stall
unchanged. The ratio column moved only because the reference wandered.

An earlier experiment on the same kernel appeared to recover a point, but it
removed barriers and switched to fast-math forms at once; isolating the halves
showed the barriers contributed nothing and the fast-math swap was an
instruction-selection divergence that the parity contract forbids. Change one
thing per measurement.

## E21: Predict the cross-shape ordering before measuring

**Symptoms:** `hypothesis_not_falsifiable`, `placement_search`, `gate_flapping`

State what ordering across shapes your mechanism implies, then check that
ordering rather than the headline number. A memory-level-parallelism fix
predicted the largest gain where the grid least fills the machine and where the
overlapped dimension is longest; it delivered 1.07x, 1.28x and 1.65x in exactly
that order while leaving instruction counts unchanged, which no
instruction-count explanation predicts.

The same discipline ends searches. After two placements of one operation had
been compared, a third was predicted to fix two shapes and keep a third; it
matched the incumbent within run-to-run drift, so the hypothesis was falsified
and position enumeration stopped. Enumerating implementation forms without a
mechanism is the same error as sweeping parameter values without one.

Holding per-CTA work fixed and sweeping the grid is how such an ordering gets
generated. The same code at 64, 256, 1024 and 8192 CTAs over 148 SMs separates a
per-CTA critical-path deficit from a throughput one, because only the grids that
leave an SM holding a single CTA expose the per-CTA chain. Read the sweep as a
family rather than as a boundary: one clean-looking separation, drawn from a
single matrix, dissolved on the next run.

Record rejected hypotheses with their measurements. Several of them are the
obvious explanations for a latency-bound gap, and each is easy to assert without
measuring: one rewrite that removed 31 instructions from a pre-barrier critical
path made a shape 6.3% slower with non-overlapping ranges.

## E22: Allocate shared memory beyond the static ceiling explicitly

**Symptoms:** `illegal_memory_access`, `zero_dynamic_smem`, `smem_capacity_limit`

A per-CTA arena above the 48 KB static limit cannot be a static buffer. Allocate
it from the shared-memory pool at the reference's alignment, keep every region
byte offset unchanged, and declare the dynamic-shared launch parameter on the
kernel.

Omitting that declaration is not a compile error. The launch reserves zero
dynamic bytes and every arena access faults at runtime, which reads like a
transcription bug anywhere in the kernel. Prove the path before writing the
body: compile and run each specialization's arena size through the real swizzled
store and matrix-load shapes, and include an oversized arena as a negative
control so a passing result means something.

Static versus dynamic is a declaration detail, not a configuration one. Measured
on both sides of one port, the L1/shared split was the same 65536 bytes either
way and the reference's carveout hint changed nothing. Scaffold-stage notes that
such a hint is "not needed" are assumptions until a counter is read; this one
sat unexamined while the port chased a latency deficit a wrong split could
plausibly have caused.

## E23: Prove a PTX form by compiling it

**Symptoms:** `invalid_ptx_form`, `unverified_instruction_selection`, `attribute_chain_resolves`

Attribute-chain access to the PTX table validates only the leading family, so a
chain that resolves may name an instruction that does not exist -- a
double-precision flush-to-zero form resolves cleanly and is not a real
instruction. Resolution is not evidence.

Prove each form the port needs by compiling a kernel that uses it, with a
known-good positive control and a known-bad negative control in the same run. A
negative control that fails with a precise diagnostic is what makes the positive
rows mean anything. Where the emitted operand form differs from the reference's
-- registers where the reference wrote immediates -- check the SASS before
treating it as a divergence; ptxas commonly folds it back.

## E24: Give a specialization axis a mechanism

**Symptoms:** `shape_conditional_constexpr`, `confounded_ab`, `gate_flapping`

A conditional constexpr is an assertion made in code rather than in prose and
deserves the same evidentiary bar. Two builds differing by two changes cannot
attribute their delta to either one, and a comparison run that way produced a
token-count deferral boundary that encoded a confounded measurement; separating
the changes showed the effect belonged to the other one.

Before writing the axis, measure the metric that justifies the mechanism at
every value of the axis. That falsified this boundary directly: the
load-to-consumer distance the deferral was supposed to buy was reproduced at
every token count, so the claim that the larger counts already covered the load
was false. The register rationale offered alongside it failed the same check.

Prefer recording a boundary in the log over encoding it in the kernel. Two
candidate axes in one port, one on token count and one on occupancy, were both
constructed from single runs, and both dissolved once the real mechanism was
fixed and the matrix was re-run unconditionally.

## E25: Match the profiler's regime to the gate's

**Symptoms:** `profiler_disagrees_with_gate`, `unstable_benchmark`, `unexplained_small_shape_deficit`

A shape can be faster under every profiler and still fail the gate. On one row a
clock-locked profile, an unlocked profile and a CUPTI loop all put the port
ahead by 0.3-1.2% while the gate's own timer put it 2.3% behind. The gate
flushes a large buffer before every timed iteration and the others do not, so
they were comparing a warm cache against a cold one; adding the same flush to
the CUPTI loop reproduced the gate's verdict, and the whole gap was the
cold-start delta, 2.07 us against the reference's 1.80 us.

So a profiler ranks two builds back to back inside one campaign; it does not
adjudicate whether a shape clears the threshold, and on these kernels it read
2-3% high. It can also fail to reproduce the phenomenon at all: one change
measured uniformly 3-4% harmful under the profiler and shape-dependently harmful
under the gate, leaving no mechanism to act on and no axis to justify. Confirm
any single-shape claim with the gate's timer before acting on it.

When instruction and sector counts all match and only DRAM throughput differs,
the remaining variable is how the address stream lands in DRAM over time. One
port issued fewer instructions, requested identical sectors, clustered its loads
better and moved 17% less data, and still reached 81% of the reference's DRAM
throughput; neither the opcode nor the memory table exposes that, and the lever
that worked was raising the number of outstanding misses.
