# Writer Phase: Performance Gate

## Goal

Improve the TIRx implementation until it matches the source implementation on
every required shape. Bench-suite timing is the performance acceptance metric.
NCU reports, generated PTX/SASS, source comparisons, and codegen-database entries
are diagnostic evidence used to understand and improve performance; they cannot
accept a performance change by themselves.

The final performance gate is:

```text
source_time / tirx_time > 0.99 for every required shape
```

Maintain one append-only ledger throughout this stage. It must record the
optimization process, including unsuccessful hypotheses and regressions, so that
later work can build on earlier evidence instead of repeating it.

The writer may create a goal for this stage. Do not complete the goal or claim
PASS while any required shape is at or below `0.99`.

## Repo-Local Guidance

For a target under `tirx-kernels`, follow `tirx-kernel-integration` for benchmark
scope, references, rounds, artifacts, GPU locking, and invalid-run handling.

Also obey the target checkout's repo-local instructions. Before changing emitted
instructions, issue order, predication, uniformity, register lifetimes, address
lowering, memory width or cache hints, pipeline depth, or synchronization for
performance, read and apply the checkout's
`.agents/skills/tirx-codegen-diagnostics/SKILL.md`. Use the target checkout's copy;
do not substitute a copied or cached version.

## Hard Acceptance Gate

Every final threshold decision must use the `tirx-kernels` bench-suite tool:

```bash
python -m tirx_kernels.bench_suite
```

Ensure the selected workload set contains every required target shape. Final
evidence must come from the latest complete bench-suite run. Do not use
`python -m tirx_kernels.bench`, an ad hoc timer, a partial shape set, selected
profiler counters, or an average ratio as final acceptance evidence.

Targeted bench-suite workloads may be used during an investigation, but final
PASS requires the latest complete required-shape matrix.

## 1. Mandatory Preparation Report

Preparation is mandatory. Do not begin performance-optimization iterations until
the preparation report passes this entry gate.

### Establish the initial performance matrix

Run the complete required-shape bench-suite matrix for the correctness-gate
implementation. Use:

```bash
python -m tirx_kernels.bench_suite
```

Record every required shape, the paired source and TIRx times, and
`source_time / tirx_time`. Select several of the worst-performing required
configs, primarily by the lowest ratios. The selected set should cover distinct
important failing regimes rather than repeatedly profiling equivalent configs.
Record why each config was selected.

### Collect paired NCU reports

For every selected config, collect paired reports for the source baseline and the
current TIRx implementation. Both runs must use the same:

- config and input regime;
- timing and profiling scope;
- launch boundaries and kernel instance;
- profiler sections and collection method.

Collect at least:

- `InstructionStats`;
- `MemoryWorkloadAnalysis_Tables`;
- `SourceCounters`;
- dynamic SASS opcode counts;
- predicated-on thread instruction counts;
- source/PTX/SASS line information.

A suitable command shape is:

```bash
ncu \
  --section InstructionStats \
  --section MemoryWorkloadAnalysis_Tables \
  --section SourceCounters \
  --import-sass yes \
  --import-source yes \
  --source-folders <source-and-generated-code-roots> \
  --export <report-path> \
  <equivalent-single-config-command>
```

Build with line information and retain the raw NCU reports plus the generated
source, PTX, SASS, and lineinfo needed to trace observations back to source and
TIRx operations. `TVM_KERNEL_DUMP=<absolute-directory>` may be used for generated
code. Serialize NCU and benchmark commands through the repository's shared GPU
lock, and reject contaminated measurements according to repository conventions.

### Preparation artifact and hard gate

Write:

```text
${PORT_DIR}/perf_gate/preparation_report.md
```

The report must contain:

- the initial complete bench-suite matrix;
- the selected worst-performing configs and selection rationale;
- the exact source and TIRx identities being compared;
- the exact profiling commands and measurement scope;
- paths to every paired source and TIRx NCU report;
- the paired times and ratio for every selected config;
- an initial comparison of memory behavior and dynamic SASS opcodes;
- immediately visible source, PTX, or SASS differences.

This report is a hard gate. Missing, unmatched, contaminated, or
non-reproducible paired reports do not pass preparation. Do not enter the
optimization loop or claim performance PASS until the report is complete.

## 2. Performance Investigation Loop

After preparation passes, run an evidence-driven optimization loop:

1. Review the current benchmark results, preparation report, ledger, source,
   implementation, generated code, and other available evidence.
2. Choose one promising performance direction and the configs that expose it.
3. Investigate that direction deeply enough to form or reject a concrete
   hypothesis. An investigation does not have to produce a code change.
4. When the evidence supports a change, make one focused experiment whose result
   can be interpreted independently of unrelated optimizations.
5. Check correctness and measure the affected configs with the bench-suite.
6. Append the evidence, change, measurements, and conclusion to the ledger.
7. Continue the same direction while it remains productive. Otherwise choose a
   new direction from the accumulated evidence. Periodically rerun the complete
   required-shape matrix to detect cross-shape regressions.

Use engineering judgment to choose the next investigation. Disproved hypotheses,
no-change conclusions, compile failures, correctness failures, and performance
regressions are useful results and must be recorded.

## Strongly Recommended Investigation Directions

The following directions often expose performance gaps between a source kernel
and its TIRx port. They are strongly recommended, but they are not an exhaustive
checklist and need not be investigated in a fixed order.

### 1. Memory behavior

Use paired `MemoryWorkloadAnalysis_Tables`, SASS, and lineinfo to compare the
complete memory behavior of the source and TIRx kernels.

#### Local memory and register spills

Investigate:

- local-memory load and store traffic;
- dynamic `LDL` and `STL` instructions;
- spill load/store bytes and transactions;
- ptxas spill warnings, register count, and live ranges;
- whether wide or long-lived values caused spilling;
- the SASS PCs and originating TIRx operations responsible for the traffic.

Do not infer spills from register count alone. Confirm them using local-memory
traffic, compiler evidence, or generated instructions.

#### Global memory and L2

Investigate:

- total global/device-memory read and write instructions, bytes, and transactions;
- L2 read/write sectors and meaningful hit or miss behavior;
- unexpected repeated loads or stores;
- access width, vectorization, coalescing, and transaction amplification;
- cache policies and cache-hint differences;
- where the source retains or reuses data that TIRx reloads or rewrites.

Compare whole-kernel traffic, not only one selected instruction. Trace important
differences to contributing SASS PCs and source/TIRx operations.

#### Shared-memory traffic

Investigate:

- total shared-memory load and store operations;
- requests, bytes, transactions, sectors, or wavefronts;
- repeated or redundant shared-memory accesses;
- access-width differences;
- differences in staging, fragment movement, and reuse.

A larger shared-memory allocation is not by itself evidence of a problem. Focus
on executed traffic and its role in the kernel.

#### Shared-memory bank conflicts

Investigate:

- bank conflicts reported for shared loads and stores;
- ideal versus actual shared-memory wavefronts or transactions;
- instruction PCs that produce excessive wavefronts;
- the explicit address arithmetic and source swizzle responsible for the access;
- whether the TIRx mapping reproduces the source kernel's bank behavior.

Trace a bank-conflict observation through SASS and lineinfo before changing the
address mapping.

### 2. Dynamic SASS opcode statistics

Compare the complete union of dynamic opcode counts from the paired source and
TIRx reports, including:

- warp-level `sass__inst_executed_per_opcode`;
- predicated-on thread-level
  `sass__thread_inst_executed_true_per_opcode`.

Rank the opcodes for which TIRx executes more instructions than the source, then
investigate the largest actionable differences. For each selected opcode, trace:

```text
dynamic opcode difference
  -> contributing SASS PCs
  -> generated source/PTX/SASS line
  -> originating TIRx operation
  -> corresponding source-kernel operation
  -> concrete explanation or focused candidate change
```

Exclude `SYNC` and `NANOSLEEP` (including `NANOSLEEP.SYNCS` forms) from the
initial excess-opcode ranking. Their dynamic counts commonly differ because
`mbarrier.wait` is implemented as a runtime polling loop whose iteration count
varies between executions.

Do not permanently ignore them. Investigate `SYNC` or `NANOSLEEP` when evidence
shows that the synchronization protocol, wait construction, pipeline timing, or
backoff behavior itself differs from the source. Do not assume every extra opcode
is harmful; establish why the direction and magnitude matter to measured runtime.

### 3. Launch, resource-constraint, and ptxas configuration sweep

Systematically sweep the kernel's legal launch constraints, register controls,
and ptxas configuration. These settings can change register allocation, spilling,
occupancy, instruction scheduling, and generated SASS even when the kernel body
is unchanged. This sweep is strongly recommended early in the investigation
loop, especially when source and TIRx have similar structure but different
register counts, local-memory traffic, occupancy, or instruction sequences.

#### Launch and kernel-entry controls

Sweep the applicable presence and values of:

- cluster launch configuration, including the target interface's no-cluster mode
  versus a one-CTA cluster (`cluster_size=0` versus `cluster_size=1` where those
  are the exposed settings), other source-compatible cluster shapes, and
  preferred cluster dimensions;
- thread-block dimensions and warp count when configurable without changing the
  source implementation's thread-role decomposition;
- `__launch_bounds__` constraints: maximum threads per block, minimum blocks per
  SM, and maximum blocks per cluster;
- the CUDA 13 `__maxnreg__(N)` qualifier, represented by
  `tirx.max_registers`;
- the CUDA 13 `__block_size__` exact block/cluster contract, represented by
  `tirx.required_block_size=1`;
- dynamic shared-memory size and applicable launch-time shared-memory settings;
- other source-relevant launch attributes, such as cooperative or
  programmatic-dependent launch.

Distinguish the entry-level `__maxnreg__` constraint from PTX `setmaxnreg`.
`setmaxnreg` changes register budgets dynamically for warpgroup roles and must
preserve the source kernel's role structure and collective requirements.

Do not combine incompatible controls. In this checkout,
`tirx.max_registers` is mutually exclusive with the launch-bounds attributes and
with `tirx.required_block_size`. Changing the actual CTA or cluster decomposition
is allowed only when it remains compatible with the source algorithm,
synchronization protocol, and thread roles.

#### ptxas register optimization and related controls

Sweep:

```text
--register-usage-level=0..10
```

In this checkout it is forwarded through:

```bash
TVM_CUDA_PTXAS_REG_LEVEL=<0..10>
```

The checkout currently passes `10` when no override is provided, while native
ptxas defaults to `5`. Higher values permit optimizations that may consume more
registers; lower values inhibit register-increasing optimizations. Neither
direction is universally faster. This is a beta ptxas option whose behavior may
change between toolkit versions, so record the CUDA toolkit and ptxas version.

Also consider source-relevant controls such as:

- `--maxrregcount`;
- ptxas optimization level;
- `--allow-expensive-optimizations=true|false`;
- load/store cache-policy options;
- other ptxas options used by the source build.

Use `TVM_CUDA_PTXAS_EXTRA_OPTS` when that is the checkout's supported forwarding
mechanism.

Do not require a full Cartesian product of every knob. Start with
one-dimensional or small structured sweeps, identify sensitive regions, and
refine promising combinations. For every tested configuration, record:

- the complete launch and compiler configuration;
- environment variables, exact compiler flags, toolkit version, and ptxas version;
- compile status and compiler warnings;
- registers per thread, spill traffic, and local-memory traffic;
- shared-memory use and resulting occupancy;
- relevant generated PTX/SASS differences;
- correctness results and per-config bench-suite timings.

Reduced register count or higher theoretical occupancy alone is not a performance
improvement; bench-suite timing must confirm it.

### 4. Other evidence-driven directions

Other useful directions may include:

- comparing the source code and TIRx implementation structure;
- comparing generated source, PTX, and SASS;
- matching an observed symptom against the repo-local codegen database;
- instruction selection, instruction width, address calculation, or integer work;
- predication, uniformity, dependency chains, or instruction scheduling;
- register lifetime, occupancy, pipeline depth, or producer/consumer overlap;
- synchronization, barrier protocols, launch specialization, or shape-dependent
  behavior.

These are examples, not a fixed strategy list. Follow whichever evidence most
plausibly explains the current performance gap. When using the codegen database,
search its `**Symptoms:**` rows, read matching entries in full, and apply an entry
only when its preconditions match the observed symptom.

## Investigation Ledger

Maintain:

```text
${PORT_DIR}/perf_gate/ledger.jsonl
```

Initialize it with the preparation work, then append an entry for every meaningful
investigation or experiment. Each entry must contain enough information to
understand and reproduce the work, including:

- a stable iteration or investigation ID and implementation identity;
- configs under investigation;
- selected direction and reason for selecting it;
- concrete hypothesis;
- NCU, benchmark, source, PTX, SASS, or database evidence;
- relevant artifact paths;
- focused code change or reason no change was made;
- compile and correctness commands and results;
- bench-suite commands, per-config times, and ratios;
- conclusion: improved, regressed, neutral, disproved, or inconclusive;
- recommended next step.

Record failed compilation, failed correctness, performance regressions, and
disproved hypotheses. Do not keep only successful optimizations. The ledger
documents the optimization history; it does not rank candidates or prescribe the
next investigation.

Only the main writer performs the optimization loop, changes the target
implementation, and appends the canonical ledger entries. Do not start another
reviewer during this stage.

## Validation and Measurement

Use the repository's normal correctness commands after implementation changes.
Targeted correctness and targeted bench-suite workloads may be used while
investigating a direction. Run broader validation in proportion to the scope and
risk of the change, and periodically run the complete required-shape matrix to
catch cross-shape regressions.

Diagnostic evidence explains a change; bench-suite timing judges it. A generated-
code, profiler, register-count, or occupancy improvement without a bench-suite
improvement is not a performance win.

Before claiming PASS:

1. run the complete required correctness set;
2. run the complete required-shape bench-suite matrix;
3. confirm that the measured implementation is the implementation in the main
   working tree;
4. append the final commands, results, implementation identity, and artifact
   paths to the ledger.

## Stage Boundary

The approved sketch and both reviewer gates are closed before this stage. Do not
edit the sketch, return to an earlier stage, or restart either reviewer.

Every performance change must preserve the porting contract, including the
source implementation structure, one-dimensional linear shared-memory storage,
explicit scalar address arithmetic, and the prohibition on first-class layouts
and tile primitives.

## PASS Checklist

The performance gate is PASS only when all are true:

- the mandatory preparation report contains valid paired source/TIRx NCU reports
  for the selected worst-performing configs;
- the ledger records the preparation and performance-investigation history;
- the latest complete `tirx_kernels.bench_suite` matrix contains every required
  shape;
- every required shape has `source_time / tirx_time > 0.99`;
- the final implementation passes the required correctness checks;
- the measured final implementation is the implementation in the main working
  tree.
