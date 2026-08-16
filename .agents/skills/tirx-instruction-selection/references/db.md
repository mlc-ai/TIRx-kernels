# TIRx instruction-selection DB

## Materialize and forward reused values

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

## Pin floating-point instructions

**Symptoms:** `bitwise_mismatch`, `denormal_mismatch`, `unexpected_ftz`, `select_lowered_as_branch`

Two independent mechanisms share one fix. Fast-math defaults (e.g. nvcc
`--use_fast_math`) add `.ftz` to float arithmetic and make division
approximate, causing bitwise mismatches on denormals. Independently, a float
compare/select whose PTX form is unpinned lets the codegen choose whether
`setp` carries `.ftz` and lets ptxas choose between `selp` and a branch; that
perturbs instruction shape and scheduling even when normal values agree.

When the reference pins an instruction, use the exact PTX operation: non-FTZ
`mul.f32`/`add.f32`, `div.rn.f32`, or explicit `setp` plus `selp`. Retain
`.approx.ftz` only where the reference uses it. Plain TIRx remains appropriate
for integer and index math.

Global fast-math off-switches exist for both TVM CUDA compile paths
(`TVM_CUDA_NVCC_NO_FAST_MATH=1` for nvcc, `--ftz=false` via
`TVM_CUDA_NVRTC_EXTRA_OPTS` for NVRTC), but prefer per-op pinning: it holds
regardless of compile defaults and documents intent at the use site.

The direction is a property of the reference, not of the family, and both
families are registered in the PTX table, so the `.ftz` forms are always an
explicit choice. Two siblings ported from a tile-DSL reference emit no `.ftz` at
all and needed non-FTZ helpers to defeat the fast-math build; a third, whose
reference is plain CUDA operators compiled with fast math, emits 108
`fma.rn.ftz.f32`, 69 `mul.ftz.f32`, 53 `add.ftz.f32`, 4 `sub.ftz.f32` and no
plain-`.f32` arithmetic at all. Inheriting a sibling's arithmetic helpers is a
silent divergence in either direction; read the reference's own PTX census
first.

Confirm with denormal inputs and an instruction-by-instruction PTX comparison.

## Match launch bounds

**Symptoms:** `register_spill`, `register_budget_mismatch`, `local_memory_traffic`, `low_occupancy`

`tirx.launch_bounds_min_blocks_per_sm` becomes the second CUDA
`__launch_bounds__` argument and imposes a hard ptxas register budget: roughly
65536 registers divided by (threads per CTA times the bound), rounded down to
the allocation granularity. A value chosen from theoretical occupancy can
starve a kernel whose reference uses more registers, producing STL/LDL or
global rescheduling. One measured 512-thread quantization kernel was capped at
32 registers with a bound of 4 while its reference ran at about 50; a bound of
2 restored parity.

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

## Defeat harmful LSR

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

## Match conversion and packing idioms

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

## Prefer binary parity

**Symptoms:** `local_rewrite_regression`, `sass_divergence`, `register_allocation_change`

In a parity port, a semantically cheaper rewrite can change global ptxas
scheduling, register allocation, and numerical behavior. Local instruction
count is not the target.

First match PTX shape, register count, and SASS structure. Deviate only after a
measured full-matrix result. A generated-code improvement that does not improve
the production benchmark is not a retained optimization.

## Map SASS in both directions

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

## Keep benchmark evidence clean

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

Give both sides the same memory state, not only the same arithmetic. An
accumulating GEMM whose reference wrapper copies C into D inside the call under
measurement does work the port does not: per-kernel attribution never charges
that copy to the kernel, but it leaves the output resident in L2, and the
reference's own kernel then measured 81.50 us against 83.48 us for the identical
call made in place. Seeding the accumulator once outside the timed closure moved
three variants from 0.963x, 0.989x and 1.003x to 0.990x, 1.004x and 1.006x. A
single output buffer shared by two implementations biases the same way, in
favour of whichever runs second. Read what the reference wrapper does inside the
timed region, and give each implementation its own output.

## Issue loads before conversions

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

## Bound hoisting by live-range cost

**Symptoms:** `register_pressure`, `local_memory_traffic`, `low_occupancy`

Hoist work only when hidden latency outweighs the added lifetime. A small
operation can be expensive when its results remain live across a recurrent loop,
large fragment, or synchronization chain.

One measured FP32 bias hoist regressed 6.6%; by contrast, staging a larger load
set won when it created enough outstanding DRAM misses. Tile wide epilogue
fragments so only the next consumed tile remains live.

Compare registers and dynamic LDL/STL before instruction count, and sweep the
tightest specialization where one spill can reverse the result.

## Expose predication and uniform control

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

A whole elected-lane region in a warp-specialized mainloop is the same case at
larger scale. Guarding a block of single-issue matrix and copy instructions on
the elected lane costs a reconvergence pair per iteration, measured at 4.53
instructions per K block against the reference's roughly zero, because the
reference's compiler predicates those instructions individually instead.
Flattening the region and predicating each issue -- in the mainloop and in both
epilogue store paths -- took reconvergence-marked BSSY and BSYNC from 494,814
and 989,628 to 222 and 444, the kernel from 58.81M to 55.64M instructions, and
the tensor pipe from 67.3% to 74.2% of cycles against the reference's 73.5%. On
the gate the mainloop half is what moved the family, from 0.939-0.983x to
0.995-1.035x; extending the rewrite to the epilogue produced no separation from
run-to-run drift, so that is where to stop.

Confirm predicate polarity and inactive-lane memory behavior, then compare BRA,
BSYNC, reconvergence, code size, registers, and both control-flow outcomes.

## Preserve compiler visibility with a typed device helper

**Symptoms:** `inline_asm_boundary`, `branch_reconvergence`, `excess_control_instructions`, `performance_regression`

Expanding a long register-only sequence into one inline-asm wrapper per PTX
instruction can change whole-function control-flow lowering even when the data
path instructions are identical.  If the original compiler-visible function
boundary is the only demonstrated difference, express the sequence as a
private scalar-return PrimFunc whose body contains the ordinary typed PTX ops.
Give pointer parameters their real storage scope and bind the helper to the
same device target as its caller; otherwise the pipeline either invents global
loads or treats the call as a cross-target launch.

One measured packed reduction moved from 1080 instructions and one
reconvergence region to 1088 and two when expanded directly.  A typed private
device helper restored 1080 instructions, one reconvergence region, 168
registers, zero stack, and the same packed arithmetic.  On one fixed GPU, the
old-helper/current ratios were 38.130/38.087 = 1.0011 and
184.346/182.235 = 1.0116 across the dense and compressed workloads.  NVVM
inlined the helper without a force-inline attribute; do not add one unless a
final-binary negative control shows a surviving device call.

Use this only for a reusable typed function boundary, not to hide an arbitrary
source string or make a workload-specific PTX bundle.  Confirm the final
binary has no helper call, then compare control topology, registers, spills,
correctness, and fixed-runner wall time across every affected workload.

## Remove a volatile identity only after final-binary equivalence

**Symptoms:** `volatile_identity`, `inline_asm_boundary`, `sass_equivalence`, `performance_regression`

An inline `asm volatile("mov.u32")` identity may look like an intentional
optimization barrier, but do not preserve or remove it based on source form
alone.  Replace only the identity with the ordinary typed PTX move, keep the
surrounding issue order unchanged, and compare correctness, final SASS,
resources, and fixed-runner wall time.  Where the consumer uses a signed view,
keep the `.u32` instruction operands as `uint32` and use a same-width cast for
the consumer instead of weakening the PTX table's dtype contract.

For one paged MQA kernel, replacing two volatile source helpers with
non-volatile typed `mov.u32` produced byte-identical SASS at both affected
bench-suite configs and retained 168 registers with zero stack and local
memory.  On one fixed B200 using five Proton rounds and one-second cooldown,
volatile/plain times were 4.459290/4.459954 us (ratio 0.999851) and
6.350161/6.349622 us (ratio 1.000085); correctness was unchanged.  This proves
those identities were redundant for that lowering, not that volatile asm is
generally redundant.  If SASS, resources, correctness, or a reproducible
wall-time gate changes, retain opacity through a general backend primitive
rather than a workload-specific source helper.

## Split predicated destination policies around the inactive-path merge

**Symptoms:** `predicated_destination`, `inactive_lane_value`, `sass_divergence`, `performance_regression`

A predicated instruction with a written destination needs the policy at that
specific program point, not one policy for the whole expression chain.  Keep
the default `preserve_dst=False` write-only destination when inactive lanes
cannot be consumed before an explicit merge, perform that merge with `selp`,
then use `preserve_dst=True` on a later predicated transform when inactive lanes
must retain the merged value.  Applying read-write binding to the initial loads
creates false input dependencies; applying write-only binding to the final
transform loses the inactive value.

One shared-memory gamma path recovered its original lowering with predicated
undefined shared loads, an unconditional subtract, `selp` to zero inactive
lanes, and a predicated read-write `ex2`.  The final SASS was byte-identical to
the source-helper baseline.  Across its three bench-suite workloads,
baseline/final times were 54.307/54.317 us, 119.646/119.631 us, and
83.360/82.549 us, while correctness passed.  Verify predicate polarity,
inactive-lane consumption, final SASS, and every control-flow shape; this
sequence is valid only when the undefined values are dominated by the merge.

## Keep acquire polling in a typed private function boundary

**Symptoms:** `polling_loop_hang`, `acquire_load`, `device_function_boundary`, `unstable_benchmark_ratio`

When a source helper contains an acquire load followed by a sleep-and-retry
loop, replacing the arbitrary source call does not require flattening the loop
into its caller.  Express it as a private scalar-return PrimFunc containing the
typed acquire load, retry condition, sleep, and reload, and call that function
through the IR module.  Bind the helper and entrypoint to the same CUDA target;
the function boundary preserves compiler-visible control flow without keeping
an arbitrary CUDA source string.

For one distributed reduce-scatter polling loop, the typed private helper and
the source helper produced byte-identical loadable SASS and kernel metadata.
TP1 source/private timing was approximately 590.430/590.049 us, with a later
final run at 590.412 us; TP4 source/private timing was 277.541/278.789 us
(ratio 0.9955).  TP1 and TP4 correctness both completed without permanent
spin.  Multi-reference TP4 campaigns were order-sensitive and unstable, so do
not replace paired fixed-topology A/B or final-binary comparison with an
unpaired aggregate.  Stop immediately on a hang, acquire-scope change,
different final control flow, or a reproducible ratio at or below the gate.

## Materialize uniformity

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

## Select address lowering by shape

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

## Tune pipeline shape and transfer granularity

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

## Merge waits with a protocol proof

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

## Keep multimem participants alive through kernel exit

**Symptoms:** `illegal_instruction`, `unspecified_launch_failure`, `multimem_early_exit`, `distributed_flake`

A host barrier after launch cannot keep a rank's device workers alive while
peer ranks are still issuing `multimem.ld_reduce`.  Give every persistent
worker its own symmetric exit flag, release-increment the multicast flag after
all local multimem users finish, and acquire-wait on the local flag until every
rank arrives before releasing device resources or exiting the kernel.

Keep this barrier outside the tile protocol: first complete the local CTA or
cluster drain, then perform the system-scope rank rendezvous, then deallocate
TMEM and exit.  A single flag per rank is insufficient when independently
scheduled persistent workers can finish at different times; index the flags by
the physical worker or SM identity used by the grid.

The missing exit rendezvous presented as two different asynchronous CUDA
faults at two different TP4 shapes across two full correctness matrices, while
standalone reruns passed.  Adding the per-worker device barrier passed 200
non-blocking relaunches across both failing shapes, the complete 16-shape
TP1/TP4 matrix, and a 1402-configuration full suite.  Launch blocking is a
localization tool here, not a fix: it passed 200 relaunches but does not prove
the cross-rank kernel-exit invariant.

## Size swizzles and fragments physically

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

## Sweep register budgets by role and shape

**Symptoms:** `register_spill`, `excess_address_math`, `low_occupancy`

Producer, compute, and epilogue warps can need materially different register
budgets. Compiler register level can also trade spills, address hoisting, and
occupancy differently across shape regimes.

Sweep neighboring budgets on representative single-wave and multi-wave shapes.
Re-run the sweep after changing descriptor placement, fragment width, or other
live ranges. Record realized allocation and dynamic local traffic, not only the
requested cap.

## Adapt row grouping when a fixup grid cannot fill one wave

**Symptoms:** `low_wave_count`, `low_dram_throughput`, `latency_bound_fixup`, `small_grid`

A bandwidth-looking fixup can instead be launch-latency-bound when grouping
several independent rows into each CTA leaves too few blocks to occupy the SMs.
Choose the largest supported row group that still launches roughly one wave:
retain the wider group when the state count already provides enough blocks,
then fall back through smaller groups only for underfilled shapes.  Keep the
grouping decision derived from the state count and device SM count rather than
special-casing a named shape.

One B200 fixup launched 32 blocks, only 0.11 waves per SM, while using 0.66% of
peak DRAM throughput.  Reducing its group from four rows to one raised the grid
to 128 blocks; the generated row-one kernel used 255 registers, 512 bytes of
shared memory, and no spills.  On the same physical GPU over 15 rounds, time
improved from 54.641 us to 53.665 us, a 1.0182 before/current ratio.  The wider
group remained selected for shapes with sufficient parallel states, all ten
correctness configurations passed, and the complete affected bench matrix
cleared its 0.99 ratio gate.

## Declare the block shape the reference declares

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

## Treat a dead tail's unrolling as a change, not a cleanup

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

## Do not blame volatile or memory clobbers

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

## Measure elided warp synchronization as a scheduling constraint

**Symptoms:** `warp_sync_schedule_shift`, `small_shape_regression`, `bar_warp_elided`

A converged full-mask `bar.warp.sync` may remain in PTX while ptxas emits no
`BAR.WARP` instruction. That does not make the source change performance
neutral: the ordering constraint can still reschedule surrounding instructions
differently for each specialization.

One B200 `sm_100a` one-warp decode sweep added exactly one PTX
`bar.warp.sync`, emitted no SASS barrier, and kept registers and spills
unchanged. Cold-L2 time nevertheless moved by +2.07%, -0.76%, and -1.17%
across its small, medium, and large default shapes; both benchmark orderings
agreed. Treat the movement as specialization-specific scheduling, not as a
fixed barrier latency. Inspect PTX, SASS, and resources for every affected
shape, then A/B the complete dispatch matrix even when the machine barrier is
elided.

## Predict the cross-shape ordering before measuring

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

## Allocate shared memory beyond the static ceiling explicitly

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

## Prove a PTX form by compiling it

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

The compiler itself is the same kind of cheap elimination. One epilogue built
through the TIRx NVRTC path and through nvcc -O3 produced byte-identical SASS,
so when a gap persists, diffing the two build paths first costs minutes and
rules the toolchain out before any deeper analysis.

## Give a specialization axis a mechanism

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

## Match the profiler's regime to the gate's

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

## Fold the loop index into the operand it indexes

**Symptoms:** `unroll_no_effect`, `instruction_count_bloat`, `excess_guard_math`, `branch_in_hot_loop`

Unrolling a mainloop by replicating its body is not by itself an
instruction-selection change. Where a reference unrolls, the payoff is usually
that the unroll factor turns a value the body reads at runtime into a per-copy
compile-time constant; a transcription that keeps reading the runtime value
takes the code growth and none of the folding.

One mainloop unrolled four ways with a per-copy predicate, the shape nvcc emits
for a trip count that is not a multiple of four, changed nothing: 813 us against
810 us, dynamic instruction count unmoved. The body was still computing its
scale-factor index as the loop counter modulo four. Re-indexing the loop as
`counter = 4 * outer + inner` makes `inner` that index, and the per-copy
constants then fold: the bit-field inserts that place the index into the
instruction descriptor, and the guard around the scale-factor copy, both
disappear in three of every four copies. The mainloop warp fell from 58.1 to
42.0 instructions per K block against the reference's 37.4, and the kernel from
61.68M to 58.81M instructions.

Round the trip count up and cover the tail with one predicate rather than
duplicating the body, and advance the counter outside that predicate -- inside
it, the rounded loop never reaches its bound and the kernel hangs.

The boundary is the fold, not the unroll. On the specializations where the
folded quantity was already constant, one scale-factor stage per load instead of
four, the same code measured unchanged, as predicted. A runtime trip count is
not itself the cost either: the same kernel compiled with a dynamic K bound
measured 0.9908x of the static-bound build.

## Divide unsigned

**Symptoms:** `excess_integer_math`, `excess_address_math`, `instruction_count_gap`, `slow_small_shape`

A signed `//` or `%` lowers to the full floordiv/floormod fixup sequence
whenever the dividend's sign cannot be proven. A scheduler counter read from a
signed integer global carries no such proof even when every value it can hold is
a count, so each division emits an absolute value, a sign compare and a chain of
moves that a reference written in unsigned arithmetic throughout does not have.

One masked-layout shape carried 49,920 absolute-value and 38,460 signed-compare
instructions against the reference's none. Casting each counter to unsigned
before dividing, and routing the swizzled-block arithmetic through the same
helpers, took that shape from 3.868M to 3.537M instructions against the
reference's 3.388M, and from 0.976x to above the gate.

What the cast removes is the correction, not the division: these divisors are
runtime values whenever the group sizes are, so both sides issue a real integer
divide. Count the fixup opcodes rather than the divide, and apply the cast
across the whole family -- these counters feed every role, so the sites that
matter are usually more numerous than the one the profile attributed.

## Hoist warp-converging instructions out of the loop

**Symptoms:** `excess_control_instructions`, `branch_in_hot_loop`, `vectorized_uniform_math`, `instruction_count_gap`

A lane election, vote or shuffle is a warp-converging instruction with real
issue cost, and a reference that calls one at each use site does not necessarily
execute it there: nvcc hoists a loop-invariant election into a single uniform
predicate carried across the loop. Transcribing the call at every use is
faithful to the text and diverges from the compiled form.

One mainloop re-elected twice per K block, executing 495,036 elections against
the reference's 36,284, while the reference executed 1.1M more uniform-register
moves -- the predicate it was holding instead. Electing once outside the loop
and consuming the resulting predicate on each instruction took the kernel from
809.5 to 806.1 us and from 62.27M to 61.68M instructions.

Hoisting is not deleting. Where a reference's election or broadcast is the only
evidence a later guard has that its condition is warp-uniform, removing it costs
more than the instruction saves; lift it to the enclosing region and keep the
predicate live instead of dropping the proof.

## Take an instruction's modifiers from the reference's PTX

**Symptoms:** `instruction_variant_mismatch`, `unverified_instruction_selection`, `sass_divergence`, `instruction_count_gap`

A helper that assembles an instruction chain from kernel-level properties can
select a modifier the reference never selects, and nothing in a count exposes
it: same opcode family, same issue count, different instruction.

One port appended a two-CTA modifier to every bulk tensor load because the
cluster held more than one CTA. The reference's copy helper does contain a
two-CTA branch, but every call site passes a literal one and never reaches it --
the pairing lives in the matrix instruction, not in the load, and each CTA
fetches its own slice. The two variants signal completion differently. Dropping
the modifier left correctness unchanged and took the kernel from 55.64M to
54.24M instructions, with the tensor pipe active 74.4% of cycles against the
reference's 73.3%.

A branch present in a reference helper is not evidence that it is taken. Read
each modifier off the reference's own dumped PTX and off the literal arguments
at its call sites, not off the helper's source or off your model of the
algorithm, and check any modifier a builder derives automatically against that
same dump.

## Narrow every operand a shortened block reaches

**Symptoms:** `partial_output`, `unported_work_reduction`, `bitwise_mismatch`, `slow_epilogue`

When a reference shortens its last tile to the rows that exist, that saving is
encoded in several operands at once, and they have to move together. Porting a
subset yields output that is partly correct and a specialization that is slower
than its padded sibling for no visible reason.

Three operands carried it in one grouped layout: the loader's per-CTA
coordinate, which shifts by the shortened height while the transfer box keeps
its compile-time size; the matrix instruction descriptor's row-count field; and
the number of epilogue stores. With only the last two ported, the peer CTA read
rows at the unshortened offset and just the first half of each block came out
correct -- which reads like a transfer-shape bug rather than a missing
narrowing. With all three, correctness was restored and the shape moved from
0.964x to 0.996x, against 0.997x for the zero-padded sibling that shares its
code.

Size the expectation before measuring: the saving here is about one partly-empty
block per group. When a specialization the reference treats specially is
measurably slower than the one it otherwise shares code with, look for a work
reduction transcribed at some of its sites and not the rest.

## Preserve cache policy across launch-local handoffs

**Symptoms:** `cold_cache_regression`, `inter_kernel_handoff`, `global_store_policy`, `dispatch_specific_deficit`

A global store into a workspace that the next launch consumes is not a terminal
streaming store. Its cache operator is part of the producer-consumer schedule:
adding `L1::no_allocate` can change the next launch's cache behavior even when
the address stream and vector width match. When the source SASS uses the default
`STG.E.128` form, preserve that default policy instead of adding a cache hint on
general streaming-store intuition.

In one four-launch recurrent chain, changing the 128-bit FP32 fixed-state stores
from `st.global.L1::no_allocate.v4.b32` to the default `st.global.v4.b32` reduced
the weakest 128-row shape from 86.56 to 82.93 us and moved source/port from 0.962
to 1.005. A 64-row guard stayed effectively flat at 119.65 versus 119.85 us,
which localized the gain to the affected dispatch. A later clean seven-shape
matrix retained the result at 82.68 us for the port versus 82.95 us for the
source, or 1.003 source/port.

This does not justify default caching for final outputs or write-only
workspaces; the boundary is an immediate cross-launch consumer. Verify the
source and target SASS cache operators and vector width, then time the producer
and consumer in the same cold-cache benchmark scope. Re-run the affected
dispatch, a different-dispatch guard, and the full matrix before keeping the
change.

## Match every kernel's grid, not only its block

**Symptoms:** `instruction_parity_with_deficit`, `unsaturated_bandwidth`, `launch_config_drift`

A port whose loop is instruction-identical to the reference can still be slow
when its grid is smaller. The launch SMs of a kernel pair are set per kernel
at the host call site, and a sibling's value is not evidence: DeepEP's
dispatch main kernel takes a bandwidth-model count (64 on B200 for e256/k6)
while its copy epilogue is launched with the full device SM count
(`device_runtime->get_num_sms()` = 148), 2.3x the warps for the same
bandwidth-bound copy. A sketch that recorded "same as kernel 1" -- and passed
review with it -- produced a 64-CTA epilogue measuring 112 us against the
reference's 97 us; the full-SM launch closed it, and the same decoupling was
then baked into the combine port from the start (main kernel 64, reduce
epilogue 148, 16 warps each).

Read the host launch call of every kernel in the chain and resolve each grid
from its own source, not from the port's shared num_sms knob. On the bench
suite, the device SM count is available to CPU-prepare as
`TIRX_PREPARE_NUM_SMS` without initializing CUDA.

## Relax bulk-group waits to read completion

**Symptoms:** `serialized_tma_pipeline`, `store_load_overlap_blocked`, `instruction_parity_with_deficit`

The full `cp.async.bulk.wait_group 0` waits for the whole previous TMA store
-- SMEM read and the HBM or NVLink write -- before the next TMA load may
issue. The `.read` form waits only until the TMA engine has finished reading
the SMEM source, which is exactly the reuse requirement of a single per-warp
SMEM slot. Relaxing both per-token waits in DeepEP's dispatch token loop and
copy epilogue overlapped the previous store's write with the next token's
load and brought the kernel to parity; publication semantics were unchanged
because a full commit and wait still guard the exit barriers. A secondary
effect visible in SASS: the full wait lowers to DEPBAR plus a per-token
`CCTL.IVALL` (lineinfo attributes the invalidate to the wait helper), so the
relaxed form also drops a per-token L1 invalidate from the loop.

The relaxation is valid only where the SMEM slot's next consumer is the next
load's read, not the store's completion; write that producer-consumer
argument per wait site before editing, and re-verify correctness at scale.

## Spend warps before pipeline slots

**Symptoms:** `deeper_pipeline_slower`, `concurrency_capped`, `underfilled_pipeline`

For a single-buffered per-warp TMA copy loop, concurrency is warps times one
outstanding bulk copy. A two-slot per-warp pipeline at constant SMEM halves
the warp count, and the lost concurrency can cost more than the deeper
pipeline recovers: an A/B-paired two-slot variant of DeepEP's dispatch
epilogue (8 warps x 2 slots, token B's load overlapping token A's wait and
store) was correct but measured 126.5 us against the single-slot form's
112.4 us p25, reference 99.2 us. The source's 16-warps x 1-slot shape was
already near the bandwidth ceiling, so the extra slots bought latency hiding
that the warp count was already providing.

Count outstanding bytes (warps x slot bytes) before adding per-warp depth,
and scale depth only after the SM's warp slots are full.

## Prefer native cooperative grid sync over software tickets

**Symptoms:** `rare_data_corruption`, `barrier_releases_early`, `software_grid_barrier`, `cooperative_launch_available`

A software grid barrier built on `atom.add` plus `old == num_sms - 1` was
lowered by ptxas into a warp-aggregated ticket whose compare constant came
out wrong (`0x1b` instead of `0x3f`) inside the full kernel, while the
isolated repro compiled correctly. The barrier released after 27 of 64 CTAs
and caused rare token corruption. A monotonic ticketless u64 counter avoids
that compiler hazard, but it still adds port-only global loads, reductions,
and polling.

When the source already uses `cooperative_groups::this_grid().sync()` and
the launch keeps every CTA resident, use `T.cuda.grid_sync()` together with
`tirx.use_cooperative_launch`. Current lowering emits the native cooperative
launch attribute, so DeepEP dispatch no longer needs either software form.
Replacing its two ticketless barriers removed the workspace counters and
polling, passed all four correctness configurations, and moved stable
five-round 8-GPU campaigns from the pinned 0.966x ratio to 0.993x and 1.022x.
Keep the dependent epilogue PDL-only: cooperative launch applies to the main
kernel that executes the grid sync, not automatically to every kernel in the
chain. Use a ticketless monotonic counter only as a fallback when native
cooperative launch is unavailable, and inspect final SASS if a ticket form is
ever unavoidable.

## Express low-level memory access through raw PTX

**Symptoms:** `low_level_ir_contract`, `buffer_load_violation`, `buffer_store_violation`, `precompile_test_failure`

The public low-level IR contract rejects `BufferLoad` and `BufferStore` in
global or shared scope even when their eventual CUDA happens to match the
source. Keep buffers for shape and pointer ownership, but perform memory
access with `T.ptx.ld.*` and `T.ptx.st.*` through `buffer.ptr_to([index])`.
The pointer operand is recorded as an address-only load and is contract-safe.

Do not reinterpret an arbitrary rvalue solely to feed a bit-typed store:
reinterpreting a literal such as zero can lower to an invalid address-of-rvalue
expression in CUDA. Cast integer values to the store's bit type, or reinterpret
a real register-backed lvalue when exact floating-point bits are required.
Migrating DeepEP dispatch this way reduced 29 violations across its two public
functions to zero (50 address-only pointer operands remained), passed all four
correctness configurations, and retained 0.993x and 1.022x in stable five-round
8-GPU campaigns.

## Bind the cluster scope or lose the attribute

**Symptoms:** `silently_dropped_attribute`, `cluster_dim_mismatch`, `launch_config_drift`

Requesting `clusterCtaIdx.x` in the TIRx launch tags does not by itself
produce a cluster launch: the resolved `tirx.kernel_launch_params` carries
the cluster dimension only when the kernel body binds the scope
(`T.cta_id_in_cluster([cluster])`, even if the value is never used). Without
the binding the launch silently falls back to cluster (1,1,1). Correctness
is unaffected, so the drift surfaces only when launch metadata or a
profiled comparison against a cluster-launched reference disagrees. DeepEP's
combine kernel declares cluster (2,1,1) to overlap clustered computation
kernels; adding the dead binding restored
`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = (2,1,1)`, verified through the real
lowering path.

Treat every launch-tag request as unproven until the resolved params are
inspected; the request and the realized attribute are different objects.

## Take register dtypes from the PTX table family

**Symptoms:** `invalid_ptx_form`, `dtype_mismatch_at_use`, `register_view_friction`

In the TIRx PTX table, operand dtypes are fixed per instruction family, not
per use site: `ld...v4.s32` destinations must be int32 while `add.bf16x2`
operands must be uint32, so one 16-byte register tile can need two typed
views across neighboring instructions. The casts between them are register
renames and cost nothing in SASS; plan the views up front instead of letting
a type error redesign the data path.

Where the source reads only part of a wide value, match the width: an i64
top-k index consumed as i32 is a single low-word `ld.global.b32` in the
reference, so the port declares an int32 view of the tensor rather than
loading i64 and converting -- the latter is a real instruction-shape
divergence, not a shorthand.

## A PDL wait without a trigger releases at completion

**Symptoms:** `pdl_chain_suspicion`, `missing_trigger`, `launch_config_drift`

`griddepcontrol.wait` in the dependent kernel releases when the primary
kernel completes; an explicit `griddepcontrol.launch_dependents` is required
only where the source has one. DeepEP's dispatch triggers right after its
data-arrival barrier, but its combine main kernel contains no
`cudaTriggerProgrammaticLaunchCompletion` -- the reduce epilogue's wait
releases at kernel-1 completion, overlapping only its prologue, and adding a
trigger would start the epilogue early on unguarded data. Check the source
for the trigger's existence and exact position before wiring the chain: a
missing trigger is not a bug to fix, and an added one is a semantic
reordering.

## Spin backoffs cost their detection latency

**Symptoms:** `backoff_regression`, `barrier_stall`, `cold_start_sensitivity`

Adding `nanosleep` backoff to a barrier spin trades poll traffic for
release-detection latency, and on this class of kernel the trade lost. In
DeepEP's dispatch, 256/512 ns backoffs in the grid and NVLink barrier spins
were correct but measured no better than busy-polling across three bench
campaigns, with quiet rounds trending worse (~0.85-0.93 vs ~0.95-1.01):
about 0.5-1 us of added detection latency per barrier outweighed the saved
L2- and sys-scope poll traffic, which is already cheap at 64 CTAs on one
line plus one SM's NVLink polls. The reference busy-polls everywhere; keep
the reference's spin form and treat sleep-based throttling as a hypothesis
to falsify, not a default.

## Keep CPU prepare free of CUDA and compilation

**Symptoms:** `prepare_stage_failure`, `cuda_init_in_prepare`, `compile_in_gpu_stage`

The bench suite's two-stage contract is enforced, not advisory: CPU
`prepare_bench` must return a process-local prepared object without
initializing CUDA, and the `run_gpu` stage rejects `tvm.compile` outright.
Three sequential workload failures map to the three rules -- a missing
`prepare_bench` (the one-stage fallback is gone), a device query or a
reference-model call that initializes CUDA (DeepEP's theoretical-SM model
ends in `torch.cuda.get_device_properties`; mirror the arithmetic or take
the suite's `TIRX_PREPARE_NUM_SMS`), and a compile left in `run_gpu`.
Multi-GPU workloads additionally sit outside the default sweep
(`default: false`, single-GPU-only default) and run explicitly through
`--workloads` with `load_kernel_configs('<kernel>')`; `--filter` matches
only the single-GPU sweep.

Compile and export in prepare, spawn ranks in `run_gpu`, and make any
shared launcher accept prebuilt libraries so the GPU stage never compiles.

## Guard CUDA enum constants by toolkit version

**Symptoms:** `unsupported_swizzle_enum`, `valid_driver_enum_rejected`, `descriptor_creation_failure`

CUDA driver enum constants are C/C++ identifiers, not preprocessor macros, so
`#ifdef CU_TENSOR_MAP_SWIZZLE_*` is always false even when `cuda.h` declares the
enumerator. Guard toolkit-specific cases with `CUDA_VERSION` instead. On the
SM100 target stack, the 128B atom swizzles are present in every inspected CUDA
12.8 through 13.2 header, so `CUDA_VERSION >= 12080` admits values 4-6 while
preserving builds against older headers.

Do not change a kernel's shared-memory layout to work around this runtime
validation failure. With the enum checks compiled under the version guard, a
GDN prefill case that previously rejected swizzle value 4 passed unchanged;
an NVSHMEM all-gather case passed against the same rebuilt runtime, confirming
that the fix belongs to descriptor validation rather than kernel lowering.

## Lower specialized global accesses explicitly

**Symptoms:** `low_level_ir_contract_failure`, `config_specific_buffer_load`, `config_specific_buffer_store`

TIRx configuration specialization can hide raw global `BufferLoad` and
`BufferStore` nodes from the commonly exercised shape while leaving them in a
less frequent branch. The low-level contract intentionally rejects those nodes:
express the access with the kernel's typed `ld.global` / `st.global` helper and
preserve signed values through bit reinterpretation rather than conversion.

Configuration-wide inspection exposed this in 82 radix-top-k configurations
and all four DeepEP combine configurations. Replacing only the raw accesses
with existing typed PTX helpers left zero violations across the complete
234-config radix and four-config DeepEP matrices; all 86 previously rejected
configurations then passed their upstream oracles under a 16-worker run.
Inspect every public configuration after specialization; validating only one
default `get_kernel()` result is insufficient.
