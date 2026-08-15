# bench-suite

Pre-commit regression benchmark for TIRx kernels. Runs a curated workload
sweep against the **working tree**, assigns GPUs automatically, and writes run
JSON + reports under `.bench-suite/`.

`config/` holds one file per kernel listing every config that kernel can bench,
each flagged `default: true|false`. The files are bucketed to mirror the kernel
tree, so a kernel's configs sit at `config/<bucket>/<kernel>.yaml`. With no
`--workloads`, the flagged configs across all files are assembled into
`.bench-suite/workloads.generated.yaml` and that is what runs. The generated
file is the inspectable source of truth for the current representative sweep.
Every kernel has at most three default small/medium/large representatives; the current tree assembles 133
workloads. Widening or narrowing the sweep is a YAML `default`
flag flip, not a scheduler rule or a second selection file. Multi-GPU configs
are deliberately absent from the default measured sweep but remain available
to explicit workload files.
For each kernel retaining three curated defaults, the same YAML also owns a
`selection_rationale` and the config loader requires exactly one
small/medium/large selection.

```bash
cd /path/to/tirx-kernels
pip install -e .

export TVM_PATH=/path/to/tvm
export PYTHONPATH="${TVM_PATH}/python"
export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"
# Do NOT set CUDA_VISIBLE_DEVICES — GPU selection is automatic.
```

Entry point: `python -m tirx_kernels.bench_suite` (same flags as `run.py`).

### Default-sweep prerequisites

Every row benches our kernel **and all of its reference impls**; a reference
that fails to build is recorded as a baseline error and **fails the workload**
(which fail-fasts the whole sweep). The host requires:

- a CUDA 13.2-aligned Python stack, including PyTorch, `cuda-toolkit`, NVRTC,
  and extensions rebuilt against that PyTorch ABI;
- **SGLang** on `PYTHONPATH` (plus its CUTLASS DSL) for the fp8 paged MQA
  reference rows.

Explicit GemmComm workloads additionally require NVSHMEM and absolute
runtime-library locks:
`TIRX_NCCL_LIBRARY`, `TIRX_CUBLAS_LIBRARY`, `TIRX_CUBLASMP_LIBRARY`, and
`TIRX_NVSHMEM_LIBRARY`. `NVSHMEM_HOME` points to the development installation
used while compiling the TIRx kernels. These locks affect only the spawned
GemmComm rank workers.

Import-check the kernels selected by the current workload file without
preparing, compiling, or using a GPU:

```bash
python -m tirx_kernels.bench_suite --check-imports
```

### SGLang FP8 paged MQA exploration

The `sglang_cutedsl` reference is required wherever it appears — the fp8 paged
MQA rows of the default sweep as well as this exploration sweep (see
"Default-sweep prerequisites" above). To run the 80-shape SM100 comparison
against SGLang's current production picker, expose a matching SGLang checkout
and install the CUTLASS DSL version required by that checkout:

```bash
export SGLANG_PATH=/path/to/sglang
export PYTHONPATH="${SGLANG_PATH}/python:${PYTHONPATH}"

python -m tirx_kernels.bench_suite --filter deepgemm_sm100_fp8_paged_mqa_logits \
  --workloads <(python -c "import yaml,sys; from tirx_kernels.bench_suite import run; \
yaml.safe_dump({'workloads': run.load_kernel_configs('deepgemm_sm100_fp8_paged_mqa_logits')}, sys.stdout)")
```

This is a kernel-only Proton comparison: Q/context reshaping, schedule metadata,
and CuTe JIT compilation happen outside the timed region. Runs are written under
`.bench-suite/`; inspect that run's `errors` and require every row to contain
`tirx`, `deepgemm`, and `sglang_cutedsl`. Do not promote this exploratory sweep to
the pinned baseline until its shape set and winning regions have been reviewed.

### Sparse FlashMLA decode matrices

The compact matrix contains the 14 public-h_q=64 upstream `gen_testcase()`
performance cases, plus the h_q=64 DeepSeek-V4 primary:

```bash
export FLASH_MLA_PATH=/path/to/FlashMLA
python -m tirx_kernels.bench_suite \
  --workloads <(python -c "import yaml,sys; from tirx_kernels.bench_suite import run; \
yaml.safe_dump({'workloads': run.load_kernel_configs('sparse_flashmla_decode_head64')}, sys.stdout)")
```

The workload has exactly 15 rows: all 14 upstream public-h_q=64 cases whose
`num_runs` is nonzero, followed by the task-required coverage of the h_q=64
DeepSeek-V4 shape (listed first for discoverability). The 2,358 upstream
h_q=64 correctness/corner cases explicitly set `num_runs=0` and are not
benchmark shapes. All public h_q=128 cases are outside this benchmark scope.

This compares the complete TIRx main-plus-combine launch against FlashMLA's
public SM100 sparse-FP8 decode dispatch. Scheduler construction and allocation
are outside both timed closures. For this port, only the results emitted by
`tirx_kernels.bench_suite` are accepted as performance evidence.

## Directory layout

| Kind | Files |
|------|--------|
| **Run** | `__main__.py` (the `python -m tirx_kernels.bench_suite` entry point), `run.py`, `config/<bucket>/<kernel>.yaml` (one per kernel, per-config `default:` flag) |
| **Pinned baseline (git)** | `baseline.json`, `baseline.md` |
| **Promote / report** | `promote_baseline.py`, `ratio_diff.py`, `baseline_view.py`, `impls.py` (impl-name classification shared by the reports) |
| **Package** | `__init__.py` |

Run artifacts (logs, `runs/*.json`, `reports/*`) live under `.bench-suite/` and are not committed.

## Execution strategy (TL;DR)

1. **Pinned baseline lives in git** (`baseline.json`, `baseline.md`).
2. **One fresh process per workload.** Each child performs CPU prepare exactly
   once, then one or more GPU attempts before `RESULT_READY → accepted RESULT → exit`. Import/parsing,
   config resolution, specialization, IR generation, and compilation happen in
   CPU prepare without initializing CUDA or owning a GPU. The compiled executable
   remains in that process; children and runtime objects are never recycled across
   workloads or serialized across processes.
3. **Bounded CPU/GPU pipeline.** The orchestrator starts multiple one-shot prepare
   children up to `--max-prepare-processes`, bounds PREPARING+READY children with
   `--ready-backlog`, and late-binds an atomic GPU claim only after READY. Prepare
   children retain all physical GPUs visible; after ASSIGN the child selects the
   physical index with `torch.cuda.set_device()` and verifies it through the UUID
   handshake. GPU stages run concurrently across currently eligible cards and
   serially per card. Live-process `CUDA_VISIBLE_DEVICES` mutation is not a valid
   late-binding mechanism.
   An accepted `RESULT_READY` releases dispatch immediately; an interfered
   `RESULT_READY` returns the same child to READY instead. Polling is only for
   foreign GPU occupancy changes. The initial occupancy snapshot starts alongside
   the first CPU prepares; assignment remains blocked until that snapshot is complete.
4. **Measurement semantics stay in the GPU stage.** Each child benches our kernel
   and every reference implementation, retaining the original implementation
   order, correctness/reference setup, timer, warmup/repeat, five rounds, 1.0s
   cooldown before every implementation/round, raw samples, and arithmetic mean.
5. **Fail fast**: the first workload/subprocess `FAIL` stops new scheduling,
   terminates the suite's in-flight subprocesses, writes the partial run for
   diagnosis, and exits with code 1. `INTERFERED` is not a workload failure and
   is retried in the same child without repeating CPU prepare; `SKIP` is accepted
   without retry. The old claim is released before reassignment, and the child
   rebuilds all GPU tensors, references, workspaces, and timer state on the newly
   assigned card. Every interference retry records the workload, attempt,
   intruder PIDs, and `retry_in_place: true`; retried results otherwise follow
   the ordinary measurement path.
6. **Ratio regression report** compares current ref/ours ratio vs the pinned
   `baseline.json` ratio (computed from its ours + ref impls). Promote a run over
   the baseline with `promote_baseline.py`.

## Baseline files (git-tracked)

| File | Contents | Refresh when |
|------|----------|--------------|
| `baseline.json` | Our kernel times + reference impl times per workload | Kernel changes, env / library upgrades |
| `baseline.md` | Human view: ours + ref + ratio | Auto on promote |

Promote through `promote_baseline.py` only (never bare `cp`).

## Workflows

### Daily: kernel iteration

```bash
python -m tirx_kernels.bench_suite
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<id>.json --merge
```

The default is already five independent rounds. Use `--rounds 1` only for a quick
diagnostic run that will not be promoted. The requested protocol remains recorded
in the run JSON.

### Refresh the pinned baseline (rare)

```bash
python -m tirx_kernels.bench_suite
python tirx_kernels/bench_suite/promote_baseline.py .bench-suite/runs/<id>.json
```

The replacement form above is used for a complete default sweep so removed
workloads do not remain in `baseline.json`. For a targeted update,
`promote_baseline.py <run>.json --merge` patches only the ok rows by
`(kernel, config)`. Both forms regenerate `baseline.md`. Promoted runs use the
arithmetic mean of all five samples; no samples are trimmed or silently dropped.

Spot-check one workload: `python -m tirx_kernels.bench --kernel ... --config ... --rounds 5`

## Workload fields

Each `config/<bucket>/<kernel>.yaml` entry requires `config` and `default`; the file
supplies `kernel` and an optional file-level `defaults:` mapping merged into
every entry. Optional per-entry fields are `timer`, `warmup`, `repeat`, and
`num_gpus` (default `1`). A file passed via `--workloads` uses the flat
`workloads:` list form instead, where each entry carries its own `kernel`.
Single-GPU jobs receive a physical index and UUID and select that index after
assignment. Multi-GPU jobs receive the complete ordered physical-index/UUID claim;
rank workers synchronize their runtime device selection only after that atomic
claim, and all assigned cards are monitored for interference.

Multi-GPU, distributed, Kineto, and MegaMoE adapters use the same pipeline-only
lifecycle. The scheduler rejects assignment-count mismatches and only launches
rank/CUDA runtimes after a complete atomic claim. Their barriers, sample-wise max
aggregation, Kineto spans, and process-group cleanup are preserved. On the shared
benchmark host, multi-GPU runtime validation is intentionally recorded as
`exempted_by_human_unmeasured`: this is neither `passed` nor `missing`, and must
never be represented by zero, null, or an empty cell. Explicit multi-GPU workload
files remain supported, but the default sweep and routine acceptance do not run them.

Implementation note (2026-08-14): the set-device/in-place-retry implementation
supersedes the former live-mask/fresh-prepare retry behavior. External-device
audits use runtime reachability, not grep presence:
FlashInfer MegaMoE's device-0 CLI/benchmark/debug functions are not imported by
this project and are non-blocking. This project's MegaMoE worker does call the
out-of-tree pinned DeepGEMM `utils.dist.init_dist`, which executes
`torch.cuda.set_device(local_rank)` and can override a nonzero physical assignment
once all devices remain visible. Under the former mask path that override was
latent because logical device 0 was the assigned card. The installed package
fails still earlier because it lacks `fp8_fp4_mega_moe`.

The approved implementation preserves `init_dist()` and its process group, then
restores the assigned physical device and revalidates its UUID before case
construction and timing. This is one instance of the general position invariant:
after any reachable external call that may change current device, restore and
prove the assigned device before allocation or launch. The default single-GPU
MegaMoE configs are therefore covered by the ordinary default sweep. External
source edits, monkey-patches, and fallback to masking remain prohibited. The
one-rank MegaMoE path also retains its TCP rendezvous/process-group setup and
32-attempt EADDRINUSE handling as a known deferred overhead.

MegaMoE entries use `timer: megamoe`, which invokes the dedicated DeepGEMM
`bench_kineto` protocol. Do not set `warmup` or `repeat` for this timer because
the protocol fixes its own 30-test schedule. Both compared MegaMoE launches have
one target CUDA kernel, so its named-kernel measurement is the same target GPU
span used by a full-span measurement. GemmComm entries use `timer: kineto`, which
measures the complete correlated GPU activity span across all streams after
preparation and applies the same cold-cache setup before every sample.

The suite exports an absolute, report-directory-independent `TIRX_BENCH_CACHE_DIR`
under `${XDG_CACHE_HOME:-~/.cache}/tirx-kernels/bench-suite/`.
Reference adapters may use it for version/GPU-qualified autotune caches, but must
finish cache loading, tuning, workspace setup, and validation before returning their
timed launch closure. The NVFP4 FlashInfer adapter uses one cache file per shape and
records its requested backend and selected runner/tactic in the result metadata. It
defaults to FlashInfer's `auto` backend; set
`TIRX_NVFP4_FLASHINFER_BACKEND=cutlass` to force an independently cached CUTLASS run.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--rounds N` | `5` | Complete standard-timer calls per implementation/workload |
| `--cooldown` | `0.0` | Seconds before every implementation in every round |
| `--util-threshold` | `0` | Skip GPUs above this utilization; requeue if a foreign process exceeds it during a run |
| `--mem-threshold` | `0` | Skip GPUs with compute-app memory-used percent above this percent |
| `--max-prepare-processes N` | host/GPU-derived | Maximum concurrent one-shot CPU prepare children |
| `--ready-backlog N` | at least prepare bound and 2× visible GPUs | Maximum PREPARING+READY children awaiting assignment |
| `--check-imports` | off | Import every kernel selected by the workload file and exit |

Round aggregation is always the arithmetic mean. The raw five-element sample arrays
remain in the run JSON for variance and outlier inspection.

## Ratio rules

- **ref impl** = fastest non-ours impl in baseline, fixed across runs.
- **ratio** = ref/ours (>1 means ours is faster).
- **ratio Δ** in `bench.md` = current ratio vs the baseline ratio (computed from
  `baseline.json`'s ours + ref impls).

## Outputs

| Path | Description |
|------|-------------|
| `.bench-suite/runs/<id>.json` | Aggregated results, raw samples, GPU assignment, and retry metadata |
| `.bench-suite/reports/<id>/summary.md` | Provenance and per-row times |
| `.bench-suite/reports/<id>/bench.md` | Main diff report (ratio Δ vs pinned baseline) |
| `.bench-suite/logs/*__a<N>.log` | Benchmark subprocess stdout for each attempt |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions over threshold (or no baseline yet) |
| `1` | A workload failed; the suite stopped immediately |
| `2` | Config error |
| `3` | One or more regressions exceeded `--threshold` |
