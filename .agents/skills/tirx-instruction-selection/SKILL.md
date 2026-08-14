---
name: tirx-instruction-selection
description: Use when porting, tuning, or diagnosing TIRx CUDA kernels, especially for perf-gate failures, PTX/SASS divergence, bitwise mismatches, register spills, scoreboard stalls, address arithmetic, predication, uniformity, pipeline depth, TMA, shared-memory conflicts, exact floating-point semantics, packing, or unstable benchmark ratios. Provides a symptom-indexed instruction-selection database and measured verification guidance.
---

# TIRx instruction selection

Use this workflow for TIRx -> TVM CUDA codegen -> nvcc/ptxas -> SASS work when
matching a hand-written CUDA, CuTe-DSL, or inline-assembly reference.

## Workflow

1. Name the observed symptom from correctness, PTX/SASS, profiling, resource
   usage, or benchmark behavior.
2. Search only the `Symptoms:` rows in [references/db.md](references/db.md):

   ```bash
   rg -n -i '^\*\*Symptoms:\*\*.*long_scoreboard' \
     .agents/skills/tirx-instruction-selection/references/db.md
   ```

3. Read each matching entry from its heading through the next heading. If no
   exact tag matches, search the `Symptoms:` rows for a close observable term.
4. Compare generated PTX and SASS on both sides, change one lever, then run the
   affected correctness and performance matrices.

Treat the bench-suite ratio as reference time divided by TIRx time; the gate
requires `> 0.99x`.

## Maintain the database

Add or update an entry only after correctness passes and a measured experiment
shows a reusable mechanism. Include the action, boundary, verification, and
essential numbers in the entry itself.

Keep `Symptoms:` as the only index. Add two to five observable snake-case terms
per entry and reuse existing terms when they fit. Do not add a separate table,
taxonomy, commit hash, run ID, file path, line number, raw artifact,
baseline-only promotion, algorithm substitution, or unmeasured CUDA folklore.
Delete stale advice instead of preserving compatibility metadata.
