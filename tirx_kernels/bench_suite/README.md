# bench-suite

The bench suite measures registered kernels against one pinned before-run
baseline. Its acceptance verdict is direct before/after wall time from
`ratio_diff.py`. By default the suite runs only the TIRx implementation;
passing `--with-references` also runs every declared external reference
(baseline) implementation, which enables the reference-ratio regression
report against `baseline.json`. References remain diagnostics — they never
replace the direct verdict. The same flag exists on
`python -m tirx_kernels.bench` for single-workload diagnostics.

## Thor performance

The following measurements cover 16 representative workloads from 15 kernels
still declared for Jetson AGX Thor (`sm_110a`, 20 SM). They were collected on
2026-09-04 before the rebase; full required-shape validation on the current
revision is pending. `Source/TIRx` is the external baseline latency divided by
TIRx latency, so values above 1 mean TIRx is faster. This comparison is separate
from the suite's direct before/after gate.

Each row reports arithmetic means from 15 Proton rounds on the same GPU, with
1000 ms warmup, 100 ms repeat and 1 s cooldown. CV is the population standard
deviation divided by the mean, shown as TIRx/source percentages.

| Kernel | Config | External baseline | TIRx (µs) | Source (µs) | Source/TIRx | CV (%) | Run |
|---|---|---|---:|---:|---:|---:|:---:|
| `fast_topk_clusters` | `f32_plain_b64_l16384_k256` | FlashInfer | 79.021 | 198.150 | 2.5076 | 5.2/1.1 | A |
| `filtered_topk` | `f32_plain_r64_l8192_k256` | FlashInfer | 46.489 | 52.644 | 1.1324 | 6.5/6.6 | A |
| `flash_attention4` | `s4096_h32kv4_causal` | Upstream FA4 CuTeDSL | 899.927 | 948.046 | 1.0535 | 3.3/3.2 | A |
| `flashinfer_fused_add_rmsnorm` | `fused_bf16_m32_h4096_xc_rc_pdl1` | FlashInfer CuTeDSL | 16.229 | 16.491 | 1.0162 | 5.5/3.5 | A |
| `flashinfer_fused_dit_layernorm` | `grgb_bf16_b1_r1920` | FlashInfer CUDA | 395.515 | 400.998 | 1.0139 | 3.3/4.0 | A |
| `flashinfer_layernorm` | `bf16_m128_h16384_xc_yc_pdl0_eps1e6` | FlashInfer CuTeDSL | 88.357 | 90.838 | 1.0281 | 5.0/2.9 | A |
| `flashinfer_qk_rmsnorm` | `rms_bf16_b32_n32_h128_xc_yc_pdl0` | FlashInfer CuTeDSL | 7.671 | 7.631 | 0.9948 | 4.7/2.4 | A |
| `flashinfer_rmsnorm` | `rms_bf16_m32_h4096_xc_yc_pdl1` | FlashInfer CuTeDSL | 12.014 | 12.018 | 1.0003 | 3.6/3.9 | B |
| `fp16_bf16_gemm` | `bf16_4096x4096x4096` | cuBLAS | 1225.117 | 1535.717 | 1.2535 | 19.1/7.9 | A |
| `fp16_bf16_gemm` | `fp16_4096x4096x4096` | cuBLAS | 1173.072 | 1473.222 | 1.2559 | 1.8/2.5 | A |
| `gdn_decode_bf16_ilp4` | `t4_b4_h8_hv16_tv16` | FlashInfer CuTeDSL | 94.771 | 97.945 | 1.0335 | 4.0/2.2 | A |
| `mxfp4_quantize` | `fp16_linear_m4096_k4096` | FlashInfer | 376.032 | 393.498 | 1.0464 | 9.0/7.0 | A |
| `nvfp4_gemm` | `4096x4096x4096` | FlashInfer CUTLASS FP4 | 417.271 | 423.874 | 1.0158 | 2.0/3.5 | A |
| `radix_topk_multi_cta` | `f32_basic_r4_l115188_k256_ctas3` | FlashInfer | 69.935 | 71.246 | 1.0187 | 4.3/4.2 | A |
| `radix_topk_single_cta` | `f32_basic_r64_l32768_k512` | FlashInfer | 186.331 | 194.371 | 1.0432 | 0.5/0.6 | A |
| `selective_state_update_mtp_horizontal` | `b512_h64_d64_s128_t6_r8_statebf16_official` | FlashInfer CUDA | 3528.172 | 3830.267 | 1.0856 | 0.0/0.0 | A |

Run A is `final-classic-4d26851-official-15r/runs/1.json`, measured at
`4d26851f`. Run B is `rms-static-cap56-level6-15r/runs/1.json`, a later targeted
RMSNorm rerun at `4d26851f-dirty` with the restored level-6/56-register schedule;
it replaces that row's earlier 0.9705 ratio. It is a separate run, not a sample
replacement within Run A. Both used TVM `15b607d6`, FlashInfer `f2e04400`,
upstream FA4 `0251105a`, and PyTorch `2.9.1+cu130` where applicable. Raw runs
remain local; these measurements do not replace `baseline.json` or its generated
`baseline.md`.

Run A allowed up to four preparing children; Run B prepared one child. Both
recorded zero GPU interference retries. CPU preparation overlap was not ruled
out in Run A; use `--serial-prepare` for subsequent Thor measurements. The BF16
GEMM row also has 19.1% TIRx CV, as shown above.

Act-and-Mul, RMSNorm Quant, Recurrent KDA Grouped and NVFP4 Quantize have since
been withdrawn from Thor declarations because of below-gate measurements;
their four classic rows are omitted. The table describes the listed shapes and
measured revisions, and does not certify every configuration of these kernels.

## Workloads

`config/<category>/<kernel>.yaml` is the workload source of truth. Each
registered kernel has one file. Files with `default_suite: true` select one to
three representative single-GPU rows with `default: true`; curated three-row
files label them `small`, `medium`, and `large`. The current default roster is
284 rows across 97 device kernels: 260 rows from 89 kernels validated on
`sm_100a`, `sm_103a`, and `sm_107a`, plus three rows each from eight
single-architecture kernels (the `sm_107a`-only Rubin BMM and SM107 block-scaled
GEMM, the `sm_100a`-only Cake VSA blk128/longseq/ultrasparse ports, and the
`sm_103a`-only Cake VSA longseq, fast.cu NVFP4 GEMM, and FP4 FA4 forward). Thus
an SM107 run selects 266 rows while SM100 and SM103 runs select 269.
GPU runs retain only rows registered for the pool's exact architecture.

With no `--workloads`, the suite writes the selected rows to
`.bench-suite/workloads.generated.yaml`. Inspect that file before freezing a
baseline.

```bash
python -m tirx_kernels.bench_suite --check-imports
```

The import check resolves every selected kernel across architectures without
compiling or using a GPU; execution filters the default roster to the exact GPU
architecture.

## Environment

```bash
pip install -e .
python scripts/install_reference_dependencies.py

export TVM_PATH=/path/to/tvm
export PYTHONPATH="${TVM_PATH}/python"
export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"
```

Do not set `CUDA_VISIBLE_DEVICES`; the runner assigns physical GPUs. Reference
revisions used by explicit diagnostic runs are pinned in
`reference-dependencies.json`.

Explicit distributed workloads additionally require their NCCL, cuBLAS,
cuBLASMp, and NVSHMEM runtime dependencies.

For Thor DeepGEMM and FlashMLA baselines, build separate variants using the
worker's Python, PyTorch and CUDA environment after installing the pinned
references above:

```bash
python scripts/setup_thor_source_references.py --name deep-gemm \
  --variant-root /path/to/new/deep-gemm-thor --build
python scripts/setup_thor_source_references.py --name flash-mla \
  --variant-root /path/to/new/flash-mla-thor --build
```

Apply each variant's generated `tirx-thor-environment.json` environment settings
before starting its worker, preserving other required `PYTHONPATH` entries.
Use the setup script's `--help` for checking or registering existing builds.

## Freeze the before baseline

Freeze one complete clean run before changing kernels:

```bash
python -m tirx_kernels.bench_suite --label before-rewrite
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<before-run>.json
```

Promotion replaces `baseline.json`, preserves the complete raw evidence, and
regenerates `baseline.md`. Incremental `--merge` promotion is disabled: mixing
rows from different revisions would create competing baseline identities.

## Run the after gate

Use the same workload set, TVM, dependencies, runner, timer, round count, and
implementation order:

```bash
python -m tirx_kernels.bench_suite \
  --label after-rewrite \
  --baseline tirx_kernels/bench_suite/baseline.json \
  --threshold 1.0
```

Regenerate the report directly with:

```bash
python tirx_kernels/bench_suite/ratio_diff.py \
  .bench-suite/runs/<after-run>.json \
  --baseline tirx_kernels/bench_suite/baseline.json \
  --threshold 1.0
```

For each expected `(kernel, config, ours implementation)`:

```text
speedup = (before_us - after_us) / before_us
pass iff speedup > -1%
```

Equality at `-1%` fails. Missing, duplicate, failed, interfered, dirty, or
otherwise incomparable rows fail. A complete matrix discovers crossings;
after that, rerun only configs that are missing, changed, failed, or polluted.
An explicit workload file or filter records a targeted selection, so the gate
requires exactly those after rows while still requiring the immutable before
baseline to contain the complete roster for the run's exact CUDA architecture.
Do not rerun clean passing rows or splice selected samples into the baseline.
Byte-identical CUDA, fatbin, and final SASS already establish implementation
alignment.

## Run a same-GPU paired A/B

Use `--ab-before` when the before and after samples must be collected in one
campaign instead of comparing against the pinned run:

```bash
python -m tirx_kernels.bench_suite \
  --ab-before <before-revision> \
  --workloads workloads.yaml \
  --rounds 15
```

The current checkout is the after side and must be clean and committed. The
suite runs the current benchmark harness against both kernel revisions. Each
workload is assigned to one available physical GPU; its before and after sides
run on that same GPU UUID, while other workloads may run concurrently on other
GPUs. If either side observes interference, both samples are discarded and the
pair is retried. Artifacts are written under `.bench-suite/ab/` and the direct
gate remains strict `after/before < 1.01`.

## Execution model

- One fresh child process prepares each workload before GPU assignment.
- Compilation and host preparation remain outside the timed region.
- GPU assignment is automatic and atomic for multi-GPU workloads.
- Foreign activity discards and retries the affected sample in the same child.
- Standard workloads aggregate independent timer rounds with the arithmetic
  mean.
- Terminal workload failures are collected without cancelling unrelated work;
  the complete sweep exits nonzero after reporting every failure.
- External references are excluded from the suite's direct before/after
  acceptance denominator. They run only under `--with-references`; a
  missing or failing enabled reference (`BASELINE_ERROR`) fails its
  workload.

Useful options:

| Option | Meaning |
|---|---|
| `--workloads PATH` | Run an explicit workload list |
| `--ab-before REV` | Run REV/current as a same-GPU paired campaign |
| `--filter TEXT` | Keep selected kernel names containing `TEXT` |
| `--rounds N` | Independent standard-timer samples |
| `--serial-prepare` | Run one child through preparation, GPU work, and exit before starting another |
| `--cooldown S` | Delay before each implementation |
| `--timer NAME` | Override every selected workload timer (for example, `event` when Proton/CUPTI is unavailable) |
| `--threshold PCT` | Report threshold; the direct gate remains fixed at 1% |
| `--out-dir PATH` | Artifact root |
| `--check-imports` | Resolve selected imports and exit |
| `--no-report` | Skip report generation |
| `--with-references` | Also run external references and the ratio report (not with `--ab-before`) |

On systems with shared CPU/GPU memory, such as Thor, use `--serial-prepare`
to keep one workload's host compilation from overlapping another workload's
GPU measurements. This option serializes children across the whole GPU pool,
including retries and process cleanup. A multi-GPU workload still receives its
required GPU group. The default remains concurrent preparation and GPU work.
Setting `--max-prepare-processes 1 --ready-backlog 1` only limits buffering and
does not provide this isolation. The run's `pipeline.serial_prepare` and
`pipeline.max_active_children` fields record the scheduling choice. Timer,
rounds, cooldown, implementation order, and cache behavior are unchanged.
`--serial-prepare` does not apply to the separate `--ab-before` runner.

## Artifacts

| Kind | Location |
|---|---|
| Runner | `run.py` |
| Workload definitions | `config/<category>/<kernel>.yaml` |
| Canonical before baseline | `baseline.json`, `baseline.md` |
| Baseline replacement | `promote_baseline.py` |
| Direct gate | `ratio_diff.py` |
| Raw runs | `.bench-suite/runs/<id>.json` |
| Per-workload logs | `.bench-suite/logs/<id>/` |
| Reports | `.bench-suite/reports/<id>/bench.md` |

## Reading a ratio you do not trust

Small, short kernels expose measurement effects that larger ones average away.
Three are worth knowing before attributing a ratio to the kernel.

**A per-implementation buffer allocation is a placement draw.** An in-place
kernel needs its own working set per implementation, and cloning once per side
puts the copies in different allocations, hence different L2 partitions. On a
grid too small to spread across the device one side is local and the other
remote, and remote-partition latency -- fixed in nanoseconds -- lands entirely on
whichever side drew it. One shape measured 0.957, 0.974, 1.058 and 1.099 across
four allocation arrangements with **no code change**, and two byte-identical
kernels read 0.937 and 1.082. In the low mode both sides are faster in absolute
time and only the ratio moves, so absolute times look innocent.

It bites when all three hold: the kernel is in place, so the per-side allocation
is on the read path rather than a shared input or write-only output; the working
set fits in one partition (a few KB); and the kernel is short enough that a
fixed latency term is a visible fraction. Sibling kernels failing any of these
showed at most 0.005. To cancel it, have both implementations alternate over the
same two working sets in opposite phase, so each spends half its calls on each
buffer, then **run the swap test**: exchange which side gets which buffer and
confirm the ratio does not move. A contiguous split of one buffer looks
symmetric and fails that test.

**Absolute microseconds do not survive a session boundary.** They drift between
sessions while ratios hold: one kernel read 7.746-7.773 us in one session and
8.238 us in another, both tight, at a ratio of 1.160-1.164 throughout. Compare
ratios, or measure both sides in one process.

**A complete matrix is the acceptance instrument, not the iteration
instrument.** Isolated single-config repeats reproduced to about +/-0.001 where
the matrix carried about +/-0.05 on the same shapes, and the matrix inflated
individual rows. Decide changes on isolated repeats; accept on the matrix.

Two controls make all three visible: keep a pair of configs that compile to
byte-identical code, since any spread between them is measurement rather than
kernel; and sweep the grid, since a deficit that disappears once the SM array
fills is occupancy, not code.

When a profiler and the suite disagree about the same two kernels, check the
instrument settings before either result. NCU locks clocks to base and flushes
caches by default; the suite runs at boost, warm. Memory latency costs more SM
cycles the faster the clock runs, so a kernel with exposed latency can lead at
base and trail at boost. Reproduce the suite's regime with
`ncu --cache-control none --clock-control none`.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions over threshold (or no baseline yet) |
| `1` | One or more workloads failed; all workloads were allowed to finish |
| `2` | Config error |
| `3` | One or more regressions exceeded `--threshold` |

Run artifacts are not committed. Promote a baseline only after reviewing its
complete roster, samples, retries, and provenance.
