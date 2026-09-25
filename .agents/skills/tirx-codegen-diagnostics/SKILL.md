---
name: tirx-codegen-diagnostics
description: Diagnose TIRx CUDA kernel problems once evidence localizes them in the generated code rather than in the algorithm, parallel decomposition, or high-level pipeline strategy. Use for performance gaps seen in PTX/SASS (instruction selection or order, register pressure, scoreboard stalls, address lowering, predication, uniformity, memory access, synchronization, packing) and for the failures low-level codegen choices cause (bitwise or denormal mismatches, launch failures, illegal instructions or accesses, hangs, ptxas rejections). Provides symptom-indexed, measured generated-code interventions.
---

# TIRx codegen diagnostics

Use this workflow for low-level problems along TIRx -> TVM CUDA codegen ->
nvcc/ptxas -> SASS: performance gaps, and the failures that low-level codegen
choices cause on the way (bitwise mismatches, launch failures, illegal
instructions, hangs). It applies both when matching a hand-written
CUDA, CuTe-DSL, or inline-assembly reference and when optimizing a kernel
against its own measured baseline. The entry condition is evidence, not a
stage of the port: use this skill once generated code, ptxas, profiling, or
resource data localizes the problem below the algorithm, the parallel
decomposition, and the WASP or high-level pipeline strategy. For a
performance gap that means first ruling those three out — do not invoke it
merely because a kernel is slow. For a compile or correctness failure the
localizing evidence is usually the failure itself (a ptxas message, a faulting
instruction, a mismatching lane), and the port need not be finished. A
database intervention may change a kernel idiom or pipeline parameter, but
its justification must be that measured mechanism.

## Workflow

1. Name the observed symptom from correctness, PTX/SASS, profiling, resource
   usage, or benchmark behavior. Correctness failures are indexed like
   performance ones (`bitwise_mismatch`, `kernel_deadlock`,
   `unspecified_launch_failure`, ...); when one appears, search it directly.
   When the task is a performance change, any correctness or bitwise
   difference it introduces is a constraint on that change, not a reason to
   change the algorithm.
2. Each entry is one file under [references/db/](references/db/). Search only
   the `Symptoms:` rows:

   ```bash
   rg -l -i '^\*\*Symptoms:\*\*.*long_scoreboard' \
     .agents/skills/tirx-codegen-diagnostics/references/db/
   ```

3. Read each matching file in full. If no exact tag matches, search the
   `Symptoms:` rows for a close observable term.
4. Change one lever, then verify at the level the symptom lives:
   - performance gap: compare generated PTX and SASS against the reference or
     the previous measured candidate, then run the affected correctness and
     performance matrices;
   - compile or correctness failure: inspect the lowest artifact that exists
     (the ptxas message, the PTX, the SASS if it assembled, the faulting
     instruction), then rebuild and run the correctness validation. Re-run
     the performance matrix only if the fix touched a hot path.

For reference-matching perf gates, treat the bench-suite ratio as reference
time divided by TIRx time; the gate requires `> 0.99x`. For standalone
optimization, freeze the measured baseline and acceptance metric before the
first change.

## Maintain the database

One entry per file in `references/db/`, filename the kebab-case of the entry
title. Add or update an entry only after correctness passes and a measured
experiment shows a reusable generated-code change. Structure each
file as:

```markdown
# <Title>

**Symptoms:** `two_to_five`, `observable_snake_case_terms`

## Symptom
## What to change
## Rationale
## Boundary
## Verification
```

`What to change` names the concrete kernel-code action, with code where a
specific idiom exists. `Rationale` carries the mechanism and the essential
measured numbers. Omit `Boundary` only when none is known.

Code blocks are genericized TIRx idioms distilled from kernels where the change
was measured, short enough to read at a glance, and marked `# before:` /
`# after:` when the entry is a rewrite. Never cite a file path, line number, or
kernel name in one; an entry that prescribes no edit carries no code.

Keep `Symptoms:` as the only index. Reuse existing terms when they fit
(`rg -No '^\*\*Symptoms:\*\*.*' references/db/` lists them), two to five per
entry. Entries record kernel-code changes, not benchmark or measurement
methodology. Do not add a separate table, taxonomy, commit hash, run ID, file
path, line number, raw artifact, baseline-only promotion, algorithm
substitution, or unmeasured CUDA folklore. Delete a stale entry's file instead
of preserving compatibility metadata. Use unique, descriptive, unnumbered
titles and filenames; do not assign sequence IDs.
