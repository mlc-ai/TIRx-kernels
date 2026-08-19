---
name: tirx-codegen-tuning
description: Use when porting, tuning, or diagnosing TIRx CUDA kernels, especially for perf-gate failures, PTX/SASS divergence, bitwise mismatches, register spills, scoreboard stalls, address arithmetic, predication, uniformity, pipeline depth, TMA, shared-memory conflicts, exact floating-point semantics, or packing. Provides a symptom-indexed database of generated-code changes with measured mechanisms.
---

# TIRx codegen tuning

Use this workflow for TIRx -> TVM CUDA codegen -> nvcc/ptxas -> SASS work when
matching a hand-written CUDA, CuTe-DSL, or inline-assembly reference.

## Workflow

1. Name the observed symptom from correctness, PTX/SASS, profiling, resource
   usage, or benchmark behavior.
2. Each entry is one file under [references/db/](references/db/). Search only
   the `Symptoms:` rows:

   ```bash
   rg -l -i '^\*\*Symptoms:\*\*.*long_scoreboard' \
     .agents/skills/tirx-codegen-tuning/references/db/
   ```

3. Read each matching file in full. If no exact tag matches, search the
   `Symptoms:` rows for a close observable term.
4. Compare generated PTX and SASS on both sides, change one lever, then run the
   affected correctness and performance matrices.

Treat the bench-suite ratio as reference time divided by TIRx time; the gate
requires `> 0.99x`.

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
