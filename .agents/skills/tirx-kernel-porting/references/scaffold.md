# Writer Phase: Scaffold

## 1. Goal

This is the scaffold stage.

Create the integration scaffold once. The expected on-disk outputs are:

- target TIRx module
- `${PORT_DIR}/source_overview.md`
- `${PORT_DIR}/launch_config.md`
- `${PORT_DIR}/tensor_overview.md`
- `${PORT_DIR}/scaffold_manifest.yaml`

The target TIRx module must include:

- `KERNEL_META`
- `CONFIGS`
- `BENCH_CONFIGS` when benchmark-only cases are useful
- `prepare_data`
- `run_test`
- `run_bench`
- an `@T.primfunc` target entry with an empty body and a visible start marker for
  the kernel-sketch stage

### File Format

`${PORT_DIR}/source_overview.md` must use this Markdown structure:

```markdown
# Source Overview

## Primary Source
- source_kind: <cuda|cutedsl|gluon|triton|other>
- source_path: <path>
- expected_source_entry: <symbol>

## Files Inspected
| path | relevant symbols | reason |
| --- | --- | --- |

## Call Graph
- <public/host entry> -> <dispatched kernel or program> -> <helpers>

## Specialization And Launch Facts
- <template/JIT/runtime facts needed later>

## Supported Modes And Branches
- <mode/dtype/layout/compression branch and condition>

## Source-Order Algorithm Phases
- <ordered list of major source regions, for orientation only>

## Unresolved Items
- <missing file/helper or none>

## Risks
- <DSL/codegen/runtime risks or none>
```

`${PORT_DIR}/launch_config.md` must use this Markdown structure:

```markdown
# Launch Config

## Logical Inputs And Outputs
- <name>: <shape/dtype/role>

## Runtime Parameters
- <parameter>: <meaning and source>

## Launch Topology
- grid_or_program_shape: <expression>
- block_or_program_shape: <expression or none>
- threads_or_warps: <expression or none>
- cta_group: <mode>
- cluster: <shape or none>

## Memory Resources And Launch Attributes
- shared_memory_bytes: <expression or none>
- pipeline_stages: <expression or none>
- attributes: <list>

## Compile-Time Or JIT Parameters
- <parameter>: <value/source>

## Source To TIRx Correspondence
| source launch value | TIRx value | notes |
| --- | --- | --- |
```

`${PORT_DIR}/tensor_overview.md` must use this Markdown structure:

```markdown
# Tensor Overview

| tensor | storage | dtype | logical shape | size | alignment | offset/start | source physical mapping/swizzle (documentation only) | lifetime | source location |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

`${PORT_DIR}/scaffold_manifest.yaml` is an entry index, not an edit contract. It must include only these fields:

```yaml
port_dir: "<PORT_DIR>"
target_repo_root: "<TARGET_REPO_ROOT>"
source_entry:
  kind: "<cuda|cutedsl|gluon|triton|other>"
  source: "<absolute source path>"
  symbol: "<source kernel, program, or host entry symbol where tracing starts>"
target_entry:
  module: "tirx_kernels/.../<kernel>.py"
  symbol: "<TIRx primfunc symbol where implementation starts>"
  start_marker: "<optional exact marker string or null>"
```

The start marker is only a visual anchor for where the kernel-sketch stage should
begin writing. It is not an edit boundary.

The overview documents are mandatory orientation notes for later phases. They are
not edit permissions, not an implementation spec, and not a substitute for reading
the source implementation and TIRx source.

## 2. Steps

Step 1: Read the user task, repo-local instructions, and nearby TIRx kernel modules needed to match local style.

Step 2: Read only enough source dispatch/API/test/benchmark code to identify the
source entry point, the target TIRx entry point, and the shape/config coverage
needed for `prepare_data`, `run_test`, and `run_bench`.

Step 3: Create the target TIRx module scaffold.

Step 4: Write `${PORT_DIR}/source_overview.md`, `${PORT_DIR}/launch_config.md`, `${PORT_DIR}/tensor_overview.md`, and `${PORT_DIR}/scaffold_manifest.yaml` using the File Format above.

Step 5: Continue to the kernel-sketch stage in the main session.

## 3. What You Must Follow

- Do not implement source-kernel behavior inside the `@T.primfunc`.
- Do not create or edit `line_by_line_mapping.yaml` or `partial_line_by_line_mapping.yaml`.
- Do not claim correctness passes.
- Do not run expensive benchmarks.
- Keep the scaffold consistent with `tirx-kernel-integration`.
- Include existing source test or benchmark shape/config coverage when it is discoverable.
- Keep `source_overview.md`, `launch_config.md`, and `tensor_overview.md` useful enough to orient later phases.
- Treat their storage-class, launch, and per-thread claims as **provisional**.
  They are written from the source text, which misrepresents exactly those: a
  declaration that reads static may be allocated dynamically, and a value the
  source reads on every thread may be read by one after the compiler sinks it.
  The kernel-sketch stage exports line-info PTX and reconciles these documents
  against it. Do not let a claim that the export later overturns stay in them --
  a stale overview propagates into the sketch and then into the implementation.
- Do not turn scaffold facts into edit permissions. The manifest tells the
  kernel-sketch stage where source tracing starts and where to start in TIRx; it
  does not restrict required target-module edits.
- Do not stop in scaffold because entry or coverage information is incomplete. Resolve it by reading more local source, using the user-provided entry, or recording an explicit assumption in the overview documents.
