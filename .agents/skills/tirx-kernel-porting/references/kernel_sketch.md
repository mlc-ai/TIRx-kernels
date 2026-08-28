# Writer Phase: Kernel Sketch

## Goal

Write a non-executable kernel sketch in the style of the current examples under
[tirx-kernels `.agents/sketch/`](https://github.com/mlc-ai/tirx-kernels/tree/main/.agents/sketch).

Current canonical upstream sketch/kernel pairs:

| sketch | corresponding TIRx kernel source |
| --- | --- |
| `.agents/sketch/flashkda_bf16_m128.md` | `tirx_kernels/flashkda/bf16_fused_m128.py` |
| `.agents/sketch/gdn_prefill_sm100.md` | `tirx_kernels/attention/gdn_prefill_sm100.py` |

Only files under the canonical upstream `.agents/sketch/` directory belong in
this list. Do not treat kernel-local design notes or files named `*_sketch.md`
elsewhere in the repository as sketch examples for this workflow.

This list will grow. Before every use, enumerate the current upstream sketch
directory, identify any newly added sketch/kernel pairs, and study them too. Do
not treat this table as a closed or exhaustive list.

Before writing, read every sketch in that directory and the corresponding TIRx
kernel linked by each sketch. Learn the style from the sketch/kernel pairs, not
from one example or from this file alone.

Treat the FlashKDA and GDN prefill sketches as the target abstraction level and
detail density. A sketch is the kernel's semantic execution skeleton, not a
source-level transcription or an expanded derivation of every implementation
expression.

**Kernel sketch = resource allocation (SMEM/registers/mbarriers) + task split
(warps/roles) + per-task tile-based dataflow.**

## Linear Storage Constraint

The sketch must not use any first-class layout. Do not attach a layout object,
layout value, layout annotation, or `layout=` argument to a tensor, tile, buffer,
view, alias, register fragment, TMEM object, or operation. This prohibition applies
even when the source framework represents its mapping with CuTe, CuTeDSL, Gluon,
Triton, or another layout abstraction.

Every SMEM declaration must be a one-dimensional linear allocation. Represent
logical dimensions, pipeline stages, aliases, swizzles, transposes, and operand
mappings with named scalar offset/index functions over that linear allocation.
Keep routine address arithmetic summarized, but show every mapping that changes
the selected bytes, instruction descriptor, role ownership, or dataflow. Hardware
descriptor fields, strides, swizzle immediates, and PTX operands may be written
explicitly because they select instructions; they are not first-class layouts.

Examples of forbidden sketch forms include `tile(..., layout=...)`,
`view(..., layout=...)`, `alias(..., layout=...)`, layout-bearing register/TMEM
fragments, and declarations that model SMEM as a multidimensional layout object.
Use a linear SMEM base plus explicit offsets instead.

Write the result to:

```text
${TARGET_REPO_ROOT}/.agents/sketch/<target-kernel-name>.md
```

## Sketch File Contents

The two current sketches share this overall structure:

1. **Title and scope**
   - `<kernel>: coarse WASP pipeline sketch`
   - state that the sketch is non-executable
   - link the corresponding TIRx kernel module as the source of truth
   - state the fixed specialization and out-of-scope variants
2. **Pipeline at a glance**
   - a table of warp/warpgroup/CTA roles
   - each role's tile program and publication/reuse edges
3. **Primitive vocabulary**
   - structural ops: linear storage, explicit offset/index functions, slices,
     and register storage without first-class layouts
   - directional copy ops between GMEM, SMEM, TMEM, and registers
   - basic compute ops such as `gemm`, `fill`, `cast`, `exp`, `add`, and `mul`
   - explicit schedule ops for pipes, barriers, waits, commits, releases, fences,
     stages, and phases
4. **Complete sketch**
   - static specialization, runtime ABI, and launch
   - necessary SMEM, TMEM, register, and mbarrier declarations, explicit physical
     mappings, aliases, and lifetimes
   - pipeline and synchronization construction
   - source-order role selection and complete role bodies
   - necessary scheduler/phase scalar state, loops, branches, tails, and output
     stores
   - instruction selection for key copy, compute, and sync/async operations
5. **Kernel-specific tables when needed**
   - logical GEMM shapes and owners
   - descriptor or TensorMap fields
   - storage-alias lifetimes
   - TIRx module and benchmark contract
   - static specialization boundary
6. **Instruction-selection summary**
   - summarize how placement, explicit physical mapping, shape, and schedule
     select PTX/SASS
   - keep opcode names and counts as evidence, not as hidden computation

Use this as the file outline, but include only kernel-specific sections supported
by the target implementation.

## Required Abstraction

The sketch body must be dominated by the two operation classes that carry the
kernel's semantics:

1. **Data movement**: explicit directional copies between GMEM, SMEM, TMEM, and
   registers.
2. **Data computation**: explicit basic compute ops such as `gemm`, `fill`,
   `cast`, `exp`, `add`, and `mul`. Do not replace these with a generic
   `compute` op.

Include only the supporting structure needed to explain how those operations
execute:

- warp/lane/warpgroup/CTA roles and their control-flow branches;
- loops, `while` loops, tails, and predicates that change executed work;
- pipe, mbarrier, fence, wait, commit, release, and other sync/async operations;
- necessary SMEM, TMEM, register, and mbarrier declarations;
- necessary scalar control state such as phase, stage, tile scheduler, loop
  counters, and bounds.

Do not expand incidental address arithmetic, pointer arithmetic, descriptor bit
assembly, constant folding, temporary scalar plumbing, or equivalent mechanical
setup. Retain such a detail only when it changes the selected tile or memory
region, mask/bounds behavior, role ownership, storage alias/lifetime,
synchronization protocol, control flow, or instruction selection. Express the
resulting logical tile/region directly at the copy or compute operation.

The sketch must explicitly show:

- warp, warpgroup, CTA, producer, and consumer roles used by the kernel;
- pipes, stages, phases, publication/reuse edges, and asynchronous synchronization;
- storage placement, linear buffers, explicit physical mappings, aliases, and
  lifetimes;
- source-order control flow, loops, branches, tails, and output paths.

Express data movement and computation with basic ops such as `copy`, `gemm`,
`fill`, `cast`, `exp`, `add`, and `mul`.

Each key copy, compute, or sync/async op may represent only:

- one PTX instruction;
- one tile of the same PTX instruction family; or
- one explicit loop of the same PTX instruction family.

An op must not contain work more complex than that. It must not hide another
algorithmic phase, role, branch, pipe transition, synchronization, epilogue, or
output path.

Every key copy, compute, and sync/async op occurrence must be followed by its
instruction selection:

```python
op(...)
# instruction_selection: <PTX instruction or family>; extent: <scalar/vector/tile/loop>
```

Storage declarations, structural views, control-flow syntax, and scalar
bookkeeping for phase/stage/scheduling do not need instruction-selection
annotations unless they themselves emit a key operation. Do not annotate omitted
address derivations merely to make the sketch look line-by-line complete.

## Steps

1. Read the scaffold artifacts and target source implementation.
2. **Export the source specialization's line-info PTX before writing anything.**
   Build it through its normal path with line information enabled, preserve the
   export under `${PORT_DIR}`, and preserve the generated intermediate source
   when the input is CuTeDSL, Gluon, Triton, or another code generator. See
   [source_export.md](source_export.md) for per-source-kind recipes.
3. Read all current upstream `.agents/sketch/*.md` files and each linked TIRx
   kernel.
4. Identify the target kernel's roles, storage, pipes, synchronization, control
   flow, and primitive PTX-level dataflow. Read the export for every fact the
   source text cannot settle, and reconcile the scaffold's provisional launch and
   tensor claims against it; record any correction, because a wrong claim there
   propagates into the sketch and then into the implementation.
5. Write a same-style sketch under `${TARGET_REPO_ROOT}/.agents/sketch/`.
6. Check that every key copy, compute, and sync/async op stays within the allowed
   abstraction bound and has an instruction-selection annotation **derived from
   the export**.
7. Continue to the sketch-reviewer gate. The sketch reviewer must be a subagent.
   It produces its own independent export; yours does not replace it, and having
   yours is what changes the review from "did the writer guess the lowering" to
   "did the writer read it correctly".

## Must Not

- Do not write a mathematical reference or high-level algorithm summary.
- Do not define compound ops such as `attention`, `softmax`, `normalize`,
  `inverse`, `update_state`, `run_pipeline`, or `load_and_gemm`.
- Do not hide warp roles, pipes, async synchronization, or control flow inside an
  op.
- Do not omit or defer instruction selection for key copy, compute, or
  sync/async operations.
- Do not annotate instruction selection from the source text. The source
  systematically misrepresents at least these, and each one changes what the
  port must transcribe, so look each up in the export rather than inferring it:
  - **storage class** -- a declaration that reads like a static shared struct may
    be allocated from the dynamic pool, which selects a different TIRx form;
  - **which threads execute an operation** -- the compiler sinks a load into a
    predicated block, or hoists one out, so a value the source reads on every
    thread may in fact be read by one, which decides whether a broadcast is
    load-bearing or redundant;
  - **whether a loop-invariant load is hoisted** -- a memory clobber, such as an
    atomic in the loop body, can keep it re-issued every iteration;
  - **operand form and width** -- immediate versus register, and sign-extending
    versus plain loads, follow from how the value is consumed downstream;
  - **whether a loop is unrolled** -- an explicit `unroll=1` in the source, or its
    absence, is not the same as what the backend emits.
- Do not bury the semantic execution skeleton under address calculations or
  other mechanical source details.
- Do not implement or debug the target TIRx kernel in this stage.
