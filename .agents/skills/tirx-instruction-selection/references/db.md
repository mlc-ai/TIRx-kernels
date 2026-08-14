# TIRx instruction-selection DB

## E1: Materialize reused expressions

**Symptoms:** `repeated_expression`, `excess_address_math`, `excess_unpack_math`, `instruction_count_bloat`

TIRx expressions are trees. Reusing a `PrimExpr` can emit its complete subtree
at every use; ptxas does not reliably recover the intended common subexpression.

Bind swizzle offsets, unpacked lanes, scale products, and other reused values to
local scalars or buffers exactly once. Helpers used multiple times should return
materialized values rather than rebuild an expression.

Confirm by counting instructions in the corresponding PTX/SASS basic block. A
1.5-2x count increase with the same control flow is a typical signal.

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

**Symptoms:** `bitwise_mismatch`, `packing_mismatch`, `excess_shift_or`

- Check operand order for packed conversions; paired FP formats commonly take
  the high element first. Use asymmetric correctness inputs.
- Pack two 16-bit halves with the native two-input 32-bit move and two 32-bit
  halves with the corresponding 64-bit move.
- For an unavailable packed form, reuse a native equivalent already validated
  in another kernel instead of inventing a new lowering.
- Match shuffle masks and saturation/rounding modifiers exactly.

Confirm packed words bitwise before judging instruction count.

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

## E8: Keep benchmark evidence clean

**Symptoms:** `unstable_benchmark`, `interfered_run`, `gate_flapping`

Shared GPUs move ratios. Iterate with targeted workloads, but accept or reject
only from the latest complete required-shape matrix. Let the suite retry
interfered measurements.

Promote a baseline only from a complete clean run, never by hand-editing ratios
or merging partial evidence. Keep timer scope explicit: kernel-only timings are
not end-to-end latency.

## E9: Issue loads before conversions

**Symptoms:** `long_scoreboard`, `insufficient_memory_parallelism`, `exposed_load_latency`

For cold-cache, memory-latency-bound kernels, issue independent global loads
before starting dependent widening, unpacking, or conversion chains. If raw bits
can remain live safely, sink the first conversion behind independent work or a
real synchronization boundary.

This has produced 1.7-3.7% gains in short recurrent kernels and much larger
gains when multiple token loads were previously serialized. It can regress when
the staged raw values spill or the loads usually hit cache.

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

Do not predicate substantial computation or duplicate a body that causes
instruction-cache pressure, spills, or lower occupancy.

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
