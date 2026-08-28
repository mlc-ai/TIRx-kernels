---
name: tirx-kernel-integration
description: Integrate kernels into tirx-kernels using its current module, registry, licensing, correctness, benchmark, and bench-suite conventions. Use when adding, porting, testing, benchmarking, or reviewing a kernel in the repository.
---

# TIRx Kernel Integration

Use this skill for repository integration around a TIRx kernel. Use
`tirx-kernel-porting` separately when the task requires implementation-trace-
preserving transcription from CUDA, CuTeDSL, Gluon, Triton, or another source.

## 1. Confirm the Live Contract

Before editing a kernel, inspect the current versions of:

- `tirx_kernels/runner.py`
- `tirx_kernels/registry.py`
- `tirx_kernels/bench/__main__.py`
- `tirx_kernels/test/__main__.py`
- `tirx_kernels/bench_suite/README.md` and the relevant file under
  `tirx_kernels/bench_suite/config/`
- `LICENSE`, `NOTICE`, `licenses/`, and the License section of `README.md`
- `tests/lint/check_license_headers.py` and the license hook in
  `.pre-commit-config.yaml`
- the `license` and `license-files` fields in `pyproject.toml`
- `tvm.tirx.bench.bench` in the paired TVM worktree
- tests covering the affected CLI or protocol
- one recently maintained module with similar runtime and reference behavior

Also inspect `tirx_kernels/_protocol.py`, but do not treat it as authoritative when
it lags the executable code. If sources conflict, follow this order:

1. runner, CLI, and tests
2. recently maintained kernel modules
3. `_protocol.py`
4. this skill

The paired worktrees matter: the host `~/tvm` or `~/tirx-kernels` checkout may not
match the branch under development. If live code differs from this skill, follow
the live code and update the skill.

## 2. Standard Module Shape

A discoverable module under `tirx_kernels/<category>/` should normally expose:

```python
KERNEL_META = {
    "name": "kernel_name",
    "category": "gemm",
    "compute_capability": 10,
}

CONFIGS = [
    {"label": "m1024_n1024_k1024", "M": 1024, "N": 1024, "K": 1024},
]

# Optional when the performance sweep intentionally differs from CONFIGS.
BENCH_CONFIGS = [
    {"label": "m4096_n4096_k4096", "M": 4096, "N": 4096, "K": 4096},
]


def get_kernel(...):
    ...


def prepare_data(...):
    ...


def run_test(...):
    ...


def run_bench(
    ..., *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0
):
    ...
```

Important:

- `compute_capability=10` means SM100. Do not write `100`.
- `KERNEL_META["name"]` must be globally unique and is the CLI kernel name.
- `category` is discovered from package directories. A new category needs an
  `__init__.py`; registry infrastructure directories are intentionally skipped.
- `get_kernel` returns the TIRx `PrimFunc`, or a list for a multi-kernel workload.
- `run_test` is the correctness entry point used by the runner.
- `run_bench` is optional and is the benchmark entry point used by the runner.
- `check_correctness` may be a useful internal helper, but current execution uses
  `run_test`; do not add a second abstraction only to satisfy a stale protocol.
- Keep `__all__` synchronized when the surrounding package uses it.

## 3. License and Provenance

Treat a mechanically transcribed or substantially copied kernel as a port, not
as native TIRx code. Determine provenance and applicable terms before writing the
implementation. Inspect the upstream repository license, file-local header,
`NOTICE` or authors files, and the exact source commit. Do not infer a license
from the project name or from another checkout revision. If the source has no
clear redistributable license, has conflicting terms, or requires a compatibility
decision that is not already represented by repository policy, stop and report
the issue instead of copying the implementation.

Current repository organization is:

- `LICENSE` contains Apache-2.0 plus a third-party component-to-license map.
- `NOTICE` carries project notices and upstream notices that must propagate.
- `licenses/` contains verbatim upstream license and required authors texts.
- `pyproject.toml` currently declares
  `Apache-2.0 AND BSD-3-Clause AND MIT` and includes `LICENSE`, `NOTICE`, and
  `licenses/*.txt` in distributions.
- `tests/lint/check_license_headers.py` binds source-project buckets and exceptional
  files to their required citations and SPDX expressions.

The current path-to-license map is:

- native TIRx code, including `tirx_kernels/basic/`: `Apache-2.0`
- `tirx_kernels/deepgemm/`: `Apache-2.0 AND MIT`
- `tirx_kernels/flashmla/`: `Apache-2.0 AND MIT`
- `tirx_kernels/flashattention/`: `Apache-2.0 AND BSD-3-Clause`
- `tirx_kernels/flashinfer/`: normally `Apache-2.0`
- `tirx_kernels/flashinfer/gdn_prefill/gdn_prefill_sm100.py`:
  `Apache-2.0 AND BSD-3-Clause`, with the upstream BSD conditions and disclaimer
  retained verbatim in the file header

Every tracked Python source file must have exactly one SPDX license expression and
the TIRx copyright line. Native files use exactly:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
```

A ported file must instead name the upstream project, cite the canonical URL and
exact source commit, retain the relevant upstream copyright notice, and declare
both the TIRx and upstream terms. Follow the live checker for exact spelling:

```python
# This file is a TIRx port of code from <Project>
# (<canonical URL> @ <commit>), <upstream copyright notice>
# SPDX-License-Identifier: Apache-2.0 AND <upstream SPDX license>
# SPDX-FileCopyrightText: Copyright TIRx authors
```

Also put the exact upstream source path or paths in the module docstring. Derived
helpers carry the port header too; package markers and genuinely native harnesses
use the native header only when the checker classifies them as exceptions. Never
run the checker's `--fix` mode on a port: it intentionally synthesizes only the
native Apache header and cannot determine upstream terms.

For a source project already represented in the repository, use its existing
bucket, license text, citation shape, and checker rule. A file with terms different
from its bucket needs a narrow file override. Preserve any file-local conditions
or disclaimer that the upstream license requires to travel with the source; SPDX
tags do not replace required text.

For a new source project or license family, update all applicable surfaces in the
same change:

1. Add the verbatim upstream license under `licenses/`, plus required `AUTHORS` or
   notice text.
2. Add the component or precise file exception to the third-party map in `LICENSE`.
3. Propagate relevant upstream `NOTICE` attribution when required.
4. Extend the `pyproject.toml` license expression if it introduces a new license;
   keep every license/notice file included by `license-files`.
5. Add the path-bound project, URL, and SPDX expectation to
   `tests/lint/check_license_headers.py`, using a file override for exceptional
   terms and an exception only for genuinely native files.
6. Extend the checker's self-tests so a wrong license, URL, missing copyright, or
   missing required text cannot pass.
7. Update the README License section when the documented source-project set or
   packaging description changes.

Do not remove or rewrite verbatim files under `licenses/`, strip an upstream
header, label a port as plain Apache-2.0, or choose `AND`/`OR` expressions from
memory. Match the inspected source terms and the repository's live policy.

Validate with:

```bash
python tests/lint/check_license_headers.py
python tests/lint/check_license_headers.py --self-test
pre-commit run --all-files
```

## 4. Configuration Layers

There are three distinct configuration layers. Do not collapse their roles:

1. `CONFIGS` is the labeled correctness/default module matrix.
2. Optional `BENCH_CONFIGS` is the module benchmark matrix. The benchmark CLI
   prefers it and falls back to `CONFIGS` when it is absent.
3. `tirx_kernels/bench_suite/config/<kernel>.yaml` selects the curated regression
   sweep and marks each benchmark config `default: true|false`.

Every module config must have a stable, meaningful `label`; the runner removes
`label` and passes the remaining keys as keyword arguments. A bench-suite `config`
value must match a label accepted by the module benchmark matrix.

Keep every config in the implementation's real dispatch domain. For a port of a
specific source specialization, record and enforce the source dispatch predicates;
do not add nearby shapes that dispatch to a different implementation.

Use labels that expose meaningful dimensions or modes. Avoid labels such as
`small` or `fast` when a parameter-encoded label is practical.

## 5. Imports and Data Preparation

Kernel discovery must work without optional reference packages. Import FlashInfer,
FlashMLA, FlashAttention, SGLang, DeepGEMM, FlashKDA, and similar dependencies only
inside the preparation, execution, or lazy reference-builder path that needs them.
Do not import optional packages at module import time.

`prepare_data` should make logical inputs and implementation-specific state explicit.

- Use deterministic seeds for correctness data.
- Allocate inputs before timed closures are created.
- Give TIRx and references equivalent logical values.
- Give implementations independent mutable outputs and workspaces unless sharing
  is required by the actual API.
- Prefer torch tensors when the compiled module accepts them directly.
- Do not add torch -> NumPy -> TVM copies without a runtime requirement.
- Keep quantization, packing, scaling, layout conversion, metadata generation,
  JIT compilation, autotuning, workspace setup, and validation outside the timed
  launch unless the production operation intentionally includes them.
- For debugging, record shape, dtype, seed, device, and derived launch values.

For distributed kernels, make rank-local tensors, process-group assumptions,
required GPU count, communication state, and timing stream explicit. Do not reuse
a stale free-GPU decision.

## 6. Correctness

`run_test(**config)` must compile, launch, and validate one config internally.

- Raise `AssertionError` on mismatch.
- Raise `unittest.SkipTest` when hardware, GPU count, a reference package, or a
  runtime dependency is unavailable.
- Start with a debugger-friendly case, but also cover predicates and boundary
  conditions that distinguish the target implementation.
- Compare against the source implementation or another trusted reference when
  doing a port. A scalar mathematical implementation is useful only as an
  additional oracle; it does not establish implementation alignment.
- Fold stable regression coverage into `run_test`, `CONFIGS`, or repository tests
  rather than leaving it only in scratch scripts.

Run one config with:

```bash
python -m tirx_kernels.test --kernel <name> --config <label>
```

Run the default-sweep import gate with:

```bash
python -m tirx_kernels.bench_suite --check-imports
```

## 7. Benchmark Entry Point

Use the paired TVM worktree's `tvm.tirx.bench.bench` as the standard timing API.
The local benchmark API takes our no-argument launch closures and optional lazy
reference builders:

```python
from tvm.tirx.bench import bench


def run_bench(
    ..., *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0
):
    data = prepare_data(...)
    tirx_launch = build_tirx_launch(data)

    def build_reference():
        # Heavy import, JIT, tuning, setup, and validation happen here.
        return build_reference_launch(data)

    return bench(
        {"tirx": tirx_launch},
        references={"reference": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )
```

Rules:

- `funcs` contains only our implementations. External implementations belong in
  `references`.
- Launch closures are no-argument callables. Inputs, compilation, and setup are
  outside the timed closure unless intentionally part of the measured operation.
- `references` maps a name to a no-argument builder; the builder returns the
  no-argument launch closure.
- A reference-builder failure is returned by `bench` as a `BASELINE_ERROR` while
  preserving our result. The bench suite treats that error as a failed workload;
  it is not valid regression evidence.
- Pass `timer=None`, `warmup=None`, and `repeat=None` to inherit central defaults
  unless the module has a justified override. `warmup` and `repeat` are millisecond
  budgets, not iteration counts.
- `rounds` is the number of independent measurement rounds.
- `cooldown_s` is applied before each implementation in each round.
- Return the complete `bench` result instead of legacy `*_ms` fields.

Use `tirx` for a new single implementation and `tirx_<variant>` for multiple TIRx
variants. The bench suite also recognizes legacy `tir` and `tir_<variant>` names.
All other implementation names are treated as references.

Result values under `impls` and `round_samples` are microseconds:

```python
{
    "impls": {"tirx": tirx_us, "reference": reference_us},
    "round_samples": {"tirx": [tirx_us, ...]},
    "errors": {},
    "timer": "proton",
    "benchmark_protocol": {...},
}
```

## 8. Timer Semantics

Local timers:

- `timer=None` resolves to `proton`.
- `proton` attributes per-kernel GPU time and excludes host dispatch overhead.
- `event` measures CUDA-event wall time for the launch closure.
- `cudagraph_proton` uses CUDA Graph replay plus Proton attribution. It uses
  `cudagraph_rep`, not `warmup` or `repeat`, and should be used only when every
  implementation captures and attributes correctly.

Distributed timing:

- Supplying a `DistributedBenchContext` makes `timer=None` resolve to `kineto`.
- Distributed benchmarking supports only `kineto`.
- Kineto uses fixed iteration counts and rejects `warmup`, `repeat`, and
  `cudagraph_rep` overrides.
- A distributed `prepare={name: callback}` resets implementation-specific state
  outside the measured span. Local benchmarks reject `prepare`.
- Auxiliary streams must synchronize correctly with the context's timing stream.

`megamoe` is a specialized CLI/module protocol for the DeepGEMM MegaMoE path, not
a general `tvm.tirx.bench.bench` timer. It fixes its own schedule; do not pass
`warmup` or `repeat` overrides to it.

Do not mix values from different timers in one unlabeled comparison. Do not claim
drop-in latency from a kernel-only Proton result.

Run a local benchmark with:

```bash
python -m tirx_kernels.bench \
  --kernel <name> --config <label> \
  --timer proton --rounds 5 --cooldown 1
```

The runner forwards `warmup`, `repeat`, and `timer` only when explicitly supplied.
The CLI and bench suite normally forward their defaults of `rounds=5` and
`cooldown_s=1.0`; a direct module call normally defaults to one round.

## 9. Kernel-Only and End-to-End

Keep these as separate measurements:

- Kernel-only times only the target launch or intended one-to-one kernel sequence.
- Drop-in/end-to-end times the real repeated operation, including required setup
  kernels, metadata work, communication, and wrapper overhead.

For an implementation-trace-preserving port, kernel-only is the direct generated-
code comparison. End-to-end answers whether replacing the production call path
helps. State exactly which one each result represents.

## 10. Bench Suite

The curated sweep is defined by one file per kernel:

```yaml
kernel: kernel_name
defaults:
  num_gpus: 1
configs:
  - {config: m1024_n1024_k1024, default: true}
  - {config: m4096_n4096_k4096, default: false}
```

Each entry requires `config` and `default`. Optional file-level defaults and
per-entry overrides include `timer`, `warmup`, `repeat`, and `num_gpus`. A custom
file passed through `--workloads` instead uses a flat `workloads:` list where every
entry includes its own `kernel`.

Run the default sweep with:

```bash
python -m tirx_kernels.bench_suite
```

Suite rules:

- Do not set `CUDA_VISIBLE_DEVICES`; the suite acquires and monitors GPUs atomically.
- The default is five independent rounds with arithmetic-mean aggregation.
- A workload benchmarks our implementation and every declared reference.
- The first real `FAIL`, including a missing/failing reference, fail-fasts the sweep.
- `INTERFERED` is requeued. `SKIP` is accepted without retry.
- Run JSON, logs, and reports live under `.bench-suite/` and are not committed.
- Reference adapters may use the absolute `TIRX_BENCH_CACHE_DIR` supplied by the
  suite for version- and GPU-qualified caches.
- A report ratio is `fastest non-ours reference / ours`; values greater than one
  mean our implementation is faster.

Promote a complete default sweep by replacing the baseline:

```bash
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<id>.json
```

Promote a targeted run without dropping unrelated rows:

```bash
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<id>.json --merge
```

Both forms regenerate `baseline.md`. Never copy a run JSON over `baseline.json`.

## 11. Review Checklist

- Does registry discovery find a unique `KERNEL_META["name"]`?
- Does every Python file have the correct native or port SPDX header?
- Does each port cite the canonical upstream URL, exact commit, copyright, and
  source path without dropping required file-local license text?
- For a new source family, are `LICENSE`, `NOTICE` when applicable, `licenses/`,
  `pyproject.toml`, README, and the path-bound license checker synchronized?
- Do the license checker, its self-test, and the pre-commit license hook pass?
- Is `compute_capability` encoded as `10` for SM100 rather than `100`?
- Do optional dependencies remain lazy so import discovery succeeds?
- Are `CONFIGS` and optional `BENCH_CONFIGS` serving their distinct purposes?
- Does the bench-suite config reference valid labels and set `default` deliberately?
- Does every test and benchmark config dispatch to the intended implementation?
- Does `run_test` validate useful boundary cases and raise on mismatch?
- Are logical inputs equivalent and mutable workspaces independent?
- Does `run_bench` forward timer, rounds, and cooldown semantics correctly?
- Are setup, JIT, tuning, validation, and allocations outside timed closures?
- Are external implementations supplied as lazy reference builders?
- Are implementation names classified correctly as ours versus references?
- Are benchmark values treated as microseconds and timer semantics labeled?
- Are distributed context, prepare callbacks, streams, and GPU count explicit?
- Are kernel-only and end-to-end claims kept separate?
- Does the bench suite pass its import gate, and is baseline promotion mode correct?
