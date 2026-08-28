# Writer Phase: Correctness Gate

## Goal And Two Required Gates

Implement the reviewer-approved sketch as a concrete TIRx kernel, verify that
implementation against the sketch, then make it numerically correct.

The correctness stage has exactly two required gates:

1. `implementation review`: the concrete kernel faithfully realizes the approved
   sketch's resources, warp roles, pipeline, control flow, tile data flow, and
   key-operation instruction selections in plain TIRx or local PTX;
2. `numerical correctness`: the target compiles, launches safely, and produces
   correct outputs for every required case.

Both must pass before entering the performance gate. Performance, profiler
artifacts, and auxiliary documents are not criteria for this stage.

The initial sketch review is already complete. Use the approved sketch as a
binding implementation plan and the source implementation as the final authority.
Do not edit the sketch or return to the sketch-reviewer stage.

## Procedure

1. Realize the approved sketch as a concrete TIRx kernel while retaining its
   resource allocation, warp-role split, pipeline, control flow, and tile-based
   data-flow structure. For every key copy, compute, and sync/async operation,
   implement the sketch's `instruction_selection` with plain TIRx or exact local
   PTX as needed. Use no first-class layouts. Allocate every SMEM tensor as
   one-dimensional linear storage and implement all logical coordinates,
   swizzles, transposes, stages, and aliases through explicit scalar offsets.
2. Start an actual `correctness_reviewer` subagent and require it to follow the
   reviewer instructions below. If it returns FAIL, fix all scoped findings and
   rerun that reviewer until it passes. The writer must not self-review.
3. Start with the smallest deterministic target config.
4. Compile and run against the trusted source implementation or reference.
5. Fix compile, launch, synchronization, memory-safety, and numerical failures
   directly in the TIRx implementation.
6. Rerun the failing case after each fix.
7. Run every required `CONFIGS` case, specialization branch, boundary case,
   tail case, and relevant correctness test.
8. Remove temporary correctness instrumentation.
9. If correctness work changed any reviewer-scoped resource, role, pipeline,
   control-flow, tile-data-flow, or key instruction-selection decision, the old
   reviewer PASS is invalid. Rerun the same correctness reviewer until both gates
   pass on the same implementation. Incidental address arithmetic and local
   scalar plumbing do not invalidate the review.

Complete the target module according to `tirx-kernel-integration`,
including the kernel, local helpers, launch metadata, `prepare_data`, `run_test`,
`run_bench`, `CONFIGS`, and optional `BENCH_CONFIGS` required by the target.

## Correctness Reviewer Subagent

The reviewer reads the approved sketch and the concrete TIRx kernel. It may inspect
the source and generated line-info PTX only as needed to resolve ambiguity or
verify a key instruction selection. It must not edit any file or debug numerical
correctness.

Review only these points:

1. resources: smem, registers, barriers, and other explicitly sketched storage
   are represented consistently;
2. task split: warp/lane/producer/consumer roles match the sketch;
3. pipeline and synchronization structure match the sketch;
4. control flow and per-task tile data flow broadly match the sketch, with no
   missing, invented, reordered, or collapsed key copy/compute/sync operation;
5. every key operation's sketched `instruction_selection` is concretely realized
   by plain TIRx or local PTX, and generated PTX confirms the selection where
   compilation is available;
6. the implementation contains no `TilePrimitiveCall`, `tirx.tile.*`, or
   operation categorized as `tile_primitive`;
7. neither the sketch nor implementation uses a first-class layout object,
   value, annotation, or layout-bearing helper, and every implementation SMEM
   allocation is one-dimensional and linear with explicit offset arithmetic.

Ignore incidental address calculations, pointer arithmetic, descriptor assembly,
temporary scalar plumbing, routine casts, and other details below the sketch's
abstraction level unless they change one of the seven points above.

Return exactly `PASS`, `FAIL`, or `BLOCKED`, followed by concise scoped findings.
PASS requires all seven points to pass. BLOCKED is not PASS.

## PASS

This gate passes only when both conditions hold on the same implementation:

- the actual correctness reviewer has returned PASS;
- every required case compiles and launches successfully;
- synchronization and memory behavior are valid;
- every checked output matches the trusted source/reference under the target
  tolerances;
- no known correctness failure remains.

After this PASS, enter the performance gate and permanently stop using the
correctness reviewer. Performance changes do not require reviewer revalidation.

## Must Not

- Do not edit or re-review the approved sketch.
- Do not simulate the correctness reviewer in the writer session.
- Do not use tile primitives; use plain TIRx and exact local PTX when needed.
- Do not introduce a first-class layout or a non-linear/multidimensional SMEM
  tensor. Use one-dimensional linear SMEM and explicit scalar offset arithmetic.
- Workflow-stage terminology belongs only in workflow artifacts. Do not copy
  stage labels or artifact status into target code. Kernel comments and docstrings
  must describe source provenance, implementation structure, and runtime semantics
  directly.
- Do not perform performance alignment in this stage.
- Do not claim PASS without both a current reviewer PASS and complete numerical
  correctness. Do not pass while any required case is failing, skipped without
  justification, or untested.
