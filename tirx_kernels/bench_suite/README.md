# bench-suite

Pre-commit regression benchmark for TIRx kernels. Runs a curated workload
sweep against the **working tree**, assigns GPUs automatically, and writes run
JSON + reports under `.bench-suite/`.

`config/` holds one file per kernel listing every config that kernel can bench,
each flagged `default: true|false`. With no `--workloads`, the flagged configs
across all files are assembled into `.bench-suite/workloads.generated.yaml` and
that is what runs -- 128 representative single-GPU workloads, including the TP1
AllGather+GEMM and GEMM+ReduceScatter profiles. Widening or narrowing the sweep
is a flag flip, not a new file.

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
(which fail-fasts the whole sweep). The default `workloads.yaml` therefore has
two hard host requirements beyond torch/DeepGEMM:

- **SGLang** checkout on `PYTHONPATH` (plus its CUTLASS DSL): the fp8 paged MQA
  rows bench the `sglang_cutedsl` reference unconditionally.
- **NVSHMEM**: required by the `allgather_gemm` / `gemm_reduce_scatter`
  (GemmComm) rows.

GemmComm benchmark rows additionally require absolute runtime-library locks:
`TIRX_NCCL_LIBRARY`, `TIRX_CUBLAS_LIBRARY`, `TIRX_CUBLASMP_LIBRARY`, and
`TIRX_NVSHMEM_LIBRARY`. `NVSHMEM_HOME` points to the development installation
used while compiling the TIRx kernels. These locks affect only the spawned
GemmComm rank workers.

`--filter` is include-only and cannot exclude rows. On a host without NVSHMEM,
run a trimmed workload list:

```bash
grep -vE "allgather_gemm|gemm_reduce_scatter" \
  .bench-suite/workloads.generated.yaml > /tmp/workloads_no_comm.yaml
python -m tirx_kernels.bench_suite --workloads /tmp/workloads_no_comm.yaml
```

Import gate (kernels in the assembled default sweep only):

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
| **Run** | `run.py`, `config/<kernel>.yaml` (one per kernel, per-config `default:` flag) |
| **Pinned baseline (git)** | `baseline.json`, `baseline.md` |
| **Promote / report** | `promote_baseline.py`, `ratio_diff.py`, `baseline_view.py` |

Run artifacts (logs, `runs/*.json`, `reports/*`) live under `.bench-suite/` and are not committed.

## Strategy (TL;DR)

1. **Pinned baseline lives in git** (`baseline.json`, `baseline.md`).
2. **One job = one workload** (kernel + config). A worker atomically acquires the
   workload's requested GPU count, then runs **one**
   bench subprocess that always benches our kernel **and** every reference impl:
   compile/prepare once, then **`--rounds N` in-bench**. Each round is one complete
   call to the selected Triton-standard timer (including its own estimate, warmup,
   and measurement), and `--cooldown` is applied before every implementation in
   every round. The suite retains all round samples and reports their arithmetic mean.
3. **Fail fast**: the first workload/subprocess `FAIL` stops new scheduling,
   terminates the suite's in-flight subprocesses, writes the partial run for
   diagnosis, and exits with code 1. `INTERFERED` is not a workload failure and
   is requeued; `SKIP` workloads are accepted without retry.
4. **Dynamic free GPU queue** (one worker per probe-OK GPU):
   workers pull jobs from a shared queue; each job atomically claims all required
   free cards, runs one subprocess, then releases the full set. Whoever finishes
   first grabs the next satisfiable job — no static workload→GPU binding and no
   partial multi-GPU claims. Larger waiting claims take priority so single-GPU
   traffic cannot starve 2/4/6-GPU workloads.
5. **Ratio regression report** compares current ref/ours ratio vs the pinned
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
diagnostic run that will not be promoted.

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

Each `config/<kernel>.yaml` entry requires `config` and `default`; the file
supplies `kernel` and an optional file-level `defaults:` mapping merged into
every entry. Optional per-entry fields are `timer`, `warmup`, `repeat`, and
`num_gpus` (default `1`). A file passed via `--workloads` uses the flat
`workloads:` list form instead, where each entry carries its own `kernel`. Multi-GPU jobs receive
the acquired physical indices as an ordered, comma-separated
`CUDA_VISIBLE_DEVICES` value and all assigned cards are monitored for interference.

MegaMoE entries use `timer: megamoe`, which invokes the dedicated DeepGEMM
`bench_kineto` protocol. Do not set `warmup` or `repeat` for this timer because
the protocol fixes its own 30-test schedule. Both compared MegaMoE launches have
one target CUDA kernel, so its named-kernel measurement is the same target GPU
span used by a full-span measurement. GemmComm entries use `timer: kineto`, which
measures the complete correlated GPU activity span across all streams after
preparation and applies the same cold-cache setup before every sample.

The suite exports an absolute `TIRX_BENCH_CACHE_DIR` under `.bench-suite/cache/`.
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
| `--cooldown` | `1.0` | Seconds before every implementation in every round |
| `--util-threshold` | `0` | Skip GPUs with SM utilization above this percent |
| `--mem-threshold` | `0` | Skip GPUs with compute-app memory-used percent above this percent |

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
| `.bench-suite/runs/<id>.json` | Aggregated run results (times in microseconds) |
| `.bench-suite/reports/<id>/summary.md` | Run overview: label, git provenance, ok/fail counts, per-row table |
| `.bench-suite/reports/<id>/bench.md` | Main diff report (ratio Δ vs pinned baseline) |
| `.bench-suite/logs/*__a<N>.log` | Benchmark subprocess stdout for each attempt |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions over threshold (or no baseline yet) |
| `1` | A workload failed; the suite stopped immediately |
| `2` | Config error |
| `3` | One or more regressions exceeded `--threshold` |
