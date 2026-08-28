# Writer Phase: Performance Gate

## Goal

Improve TIRx performance as measured by the `tirx-kernels` bench-suite on the
required shapes. Bench-suite timing is the only performance metric in this gate.
Generated code, NCU counters, opcode tables, memory tables, lineinfo, and static
instruction counts are diagnostic evidence only; none of them can accept or
reject a performance change.

The gate is:

```text
source_time / tirx_time > 0.99 for every required shape
```

The writer may create a goal for this stage. Do not complete the goal or claim
PASS while any required shape is at or below `0.99`.

## Required Repo-Local Skill

For a target under `tirx-kernels`, read and follow
`<TARGET_REPO_ROOT>/.agents/skills/tirx-codegen-tuning/SKILL.md` at the start of
this gate. Use the skill from the target checkout so its guidance stays aligned
with that checkout. Before changing emitted instructions, issue order,
predication, uniformity, register lifetimes, address lowering, memory width or
cache hints, pipeline depth, or synchronization for performance, apply that
skill's symptom-indexed workflow.

If the target is not a `tirx-kernels` checkout or does not contain that skill,
continue with the diagnostics below; do not silently substitute an unrelated
cached copy.

## Hard Acceptance Gate

Every final threshold decision MUST use the `tirx-kernels` bench-suite tool:

```bash
python -m tirx_kernels.bench_suite
```

Ensure the selected workload set contains every required target shape. Final
evidence must come from the latest complete bench-suite run. Do not use
`python -m tirx_kernels.bench`, an ad hoc timer, a partial run, selected shapes,
or an average ratio as final acceptance evidence.

Follow `tirx-kernel-integration` for benchmark scope, references,
rounds, artifacts, and invalid-run handling.

Use bench-suite measurements to decide whether a candidate improved performance
and should be retained. Targeted bench-suite workloads may be used during an
iteration, but final PASS requires the latest complete required-shape matrix.

## Read Once and Freeze Validation Commands

Read this file and the repo-local codegen-tuning skill once when entering the
performance stage. Do not reread this file, `correctness_gate.md`, or the root
porting skill during each search expansion. Do not restart either reviewer.

At stage entry, write one validation manifest under:

```text
${PORT_DIR}/perf_gate/validation_manifest.json
```

Record the exact source/reference identity, required and guard shapes,
correctness commands, tolerances, targeted bench-suite commands, complete-matrix
command, and measurement protocol. Search iterations execute this frozen manifest;
they do not rediscover or reinterpret the gates. Update the manifest only when the
user changes the task or an actual code-interface change makes a recorded command
invalid. A timing value is not frozen: paired source/TIRx timings may still be
remeasured under the fixed protocol.

## Global Variant Ledger

Maintain one append-only global ledger for the complete performance search:

```text
${PORT_DIR}/perf_gate/variant_ledger.jsonl
```

Initialize it with the correctness-gate implementation as the root variant.
Every explored result must receive a stable `variant_id` and a ledger row,
including variants that fail to compile, fail correctness, regress performance,
duplicate an existing program, or produce no useful change. Preserve each unique
program under `${PORT_DIR}/perf_gate/variants/<variant_id>/` as an immutable commit,
tree hash, or complete reproducible patch plus its parent identity. The ledger is
the search history, not a list containing only winners.

Each row must record at least:

- `variant_id`, `parent_id`, lineage depth, selected strategy, and code/tree hash;
- strategy-selection policy, selection reason or probability, and random seed
  when selection is stochastic;
- expansion ID and the parent's expansion count at selection time;
- exact hypothesis and focused code change;
- compile and correctness status plus validation level: `provisional` or
  `fully_validated`;
- targeted and full bench-suite commands, artifacts, per-shape times and ratios;
- diagnostic artifact paths and a short evidence summary;
- energy value and the exact energy-function version/parameters;
- eligibility state: `eligible`, `ineligible`, or `duplicate`;
- failure, rejection, or duplicate reason when applicable.

Only the main writer performs the search, writes strategy-local artifacts,
produces candidate patches/commits, appends canonical ledger rows, and changes
eligibility state.

## Global Energy-Guided Search

The candidate pool contains every quick-validation-passing, reproducible,
nonduplicate variant, regardless of lineage depth, validation level, or how many
times it has already been expanded. Select the parent from this complete eligible
pool. Expansion does not consume or retire a parent: the same parent may be
selected and expanded again any number of times, including repeatedly with the
same strategy, as long as each expansion records a concrete new hypothesis or
attempt.

Before the first selection, run the complete required-shape bench-suite matrix
for the root variant and store it in the root ledger row. If the root already
passes the hard gate, no expansion is needed. Otherwise use its failing shapes
and measurements to initialize the candidate-pool energy values.

Selection may be greedy or stochastic. Define and record an energy function from
measured bench-suite results and optional novelty/complexity terms. A suitable
policy is:

```text
with probability p_greedy:
    choose argmax energy(eligible_pool)
otherwise:
    sample eligible_pool with probability softmax(energy / temperature)
```

Use nonzero exploration probability so a performance-average variant can still
be selected. The energy function ranks search candidates only; it must not replace
the per-shape hard acceptance gate. Do not use profiler counters as if they were
performance scores. Record the random seed whenever stochastic selection is used.
The energy function may include an expansion-count or novelty term to avoid
starvation, but it must not make a previously expanded valid parent ineligible.

After selecting the parent, select exactly one of the three strategies below for
this expansion. The strategy choice may be deterministic from current evidence or
sampled from an adaptive probability distribution. Record the chosen strategy,
the reason or probability assigned to it, and the random seed when sampled. Do
not run the other two strategies during the same expansion and do not silently
switch strategies after work on the selected strategy starts.

For each selected parent:

1. Select one strategy and materialize one isolated working copy at exactly the
   selected parent program.
2. In the main writer session, execute that strategy using the parent ID, exact
   target/guard shapes, source baseline, approved sketch, and relevant
   generated-code paths.
3. Produce one focused candidate. A blocked or no-change result is still recorded;
   do not silently substitute work from another strategy.
4. Do not run GPU measurements that overlap contaminating work. Serialize
   NCU and bench-suite commands through the environment's shared GPU lock and
   reject measurements with intruding work according to repository conventions.
5. Materialize the child and run only the quick validation recorded in the
   manifest: compile, affected/target correctness, guard correctness, and targeted
   target/guard bench-suite measurements.
6. Register the outcome in the global ledger. Add a unique, quick-validation-
   passing, reproducible child to the eligible pool as `provisional` even when it
   is not faster than the parent; mark an invalid or duplicate child ineligible
   while retaining its ledger row. Keep the parent eligible.
7. Increment the parent's expansion count, then perform the next global selection.
   Periodically run the complete required-shape matrix for the best current variant.

Continue expanding the candidate pool until one variant passes the latest complete
required-shape matrix or the search is genuinely blocked. Do not overwrite the
main target implementation on every experiment. Promote a selected candidate to
the main working tree only after its immutable variant and measurements are in
the ledger.

## Performance Strategies

### Strategy 1: Paired NCU and lineinfo

Profile the source and selected parent on the same representative failing shape,
input regime, timing scope, launch boundaries, and kernel instance. Export paired
reports with at least:

```bash
ncu \
  --section InstructionStats \
  --section MemoryWorkloadAnalysis_Tables \
  --section SourceCounters \
  --import-sass yes \
  --import-source yes \
  --source-folders <source-roots-and-generated-code-dump> \
  --export <source-or-tirx-report> \
  <equivalent-single-shape-command>
```

Compare the complete union of:

- dynamic warp-level `sass__inst_executed_per_opcode`;
- predicated-on thread-level
  `sass__thread_inst_executed_true_per_opcode`;
- shared-memory, L1/TEX, L2, and device-memory rows from
  `MemoryWorkloadAnalysis_Tables`.

Choose one actionable metric where TIRx is demonstrably behind the source. Do not
equate a large difference with a regression without establishing the harmful
direction. Trace that metric through its contributing SASS PCs and lineinfo:

```text
dynamic NCU or memory difference
  -> contributing SASS PC(s) and instruction(s)
  -> generated/source file:line
  -> originating TIRx operation and source-kernel operation
  -> one focused candidate change
```

Build with line information and retain generated TIRx, CUDA, PTX, and SASS needed
for the trace. `TVM_KERNEL_DUMP=<absolute-directory>` may be used for this. Make
one change intended to close the selected metric, then return the candidate and
the paired reports, exported tables, source map, and hypothesis. `SYNC` and
`NANOSLEEP` commonly originate from `mbarrier.wait` busy-wait loops and are not
standalone targets unless evidence shows a controllable protocol difference.

### Strategy 2: Codegen database

Follow the target checkout's repo-local `tirx-codegen-tuning` skill. Search only
the `**Symptoms:**` rows in its `references/db/`, match the selected parent's
observed symptom, and read every matching entry in full. Select one applicable
codegen-database intervention, state why its preconditions match, and make one
focused candidate change. Return the candidate, selected database entry, observed
symptom, expected generated-code effect, and resulting generated-code evidence.
If no entry applies, return a recorded no-change result instead of inventing a
database rule or silently switching strategies.

### Strategy 3: Free exploration

Independently inspect the parent implementation, source, generated code, benchmark
shape behavior, and existing evidence. Form one concrete performance hypothesis
and make one focused change. This strategy is not required to use NCU or the codegen
database, but it must preserve correctness and return a reproducible candidate,
the hypothesis, and the expected mechanism. Do not combine several unrelated
optimizations into one child.

## Candidate Evaluation

Diagnostic evidence explains a child; bench-suite timing judges it. Validation
has exactly two levels.

### Quick validation for every child

Run only the commands frozen in the manifest for:

1. compilation;
2. correctness on the affected or target config plus predefined guard configs;
3. targeted bench-suite workloads for the target and guard shapes;
4. contamination and timing-scope checks.

A unique child that passes this level is `provisional` and may enter the eligible
pool or become a parent. Retain slower provisional children so the energy policy
can explore through a temporary regression. Do not run all correctness configs or
the complete performance matrix merely to register an ordinary child.

### Full validation for selected candidates

Run the manifest's complete correctness set and complete required-shape
bench-suite matrix only when:

- a provisional child is being promoted to replace the current champion;
- the recorded periodic checkpoint is due; or
- the writer is preparing to claim final PASS.

A candidate that passes both becomes `fully_validated`. If it fails, record the
failure and mark that variant ineligible; continue the search. Full validation
means executing the frozen commands, not rereading gate documents, rescanning the
repository for validation scope, or starting a reviewer.

A diagnostic improvement without a bench-suite improvement is not a performance
win, but the child remains part of the ledger. Final PASS still requires the
latest complete required-shape matrix, not a strategy-local or targeted
measurement.

## Stage Boundary

The initial sketch review is complete before this stage. Treat that sketch as a
completed initial design artifact. Do not edit it, return to the sketch stage,
or start another reviewer. Correctness was established by the preceding gate
and remains a prerequisite for retaining implementation changes, but it is not
an additional performance metric.

## Must Not

- Do not use `tirx_kernels.bench` for final performance acceptance.
- Do not stop on a partial matrix or an average above `0.99`.
- Do not pass when even one required shape is `<= 0.99x` source.
- Do not compare timings from different scopes or setup boundaries.
- Do not edit the approved sketch or start another sketch reviewer.
- Do not introduce any first-class layout or any non-linear/multidimensional SMEM
  tensor in a performance variant. Every variant must retain one-dimensional
  linear SMEM and explicit scalar offset/index arithmetic.
- Do not reread correctness/performance gate documents or rescan validation scope
  during each expansion; execute the frozen validation manifest.
- Do not run full correctness and the complete performance matrix for every
  ordinary child.
- Do not call a generated-code or profiler difference a performance improvement
  unless bench-suite measurements improve.

## PASS Checklist

The performance gate is PASS only when all are true:

- the latest complete `tirx_kernels.bench_suite` matrix contains every required
  shape;
- every required shape has `source_time / tirx_time > 0.99`;
- the winning variant and its complete matrix are present in the global ledger;
- the winning variant is marked `fully_validated`;
- the target implementation in the main working tree is byte-identical to the
  recorded winning variant and still passes required correctness checks.
