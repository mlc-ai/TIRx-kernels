# bench-suite

The bench suite measures registered kernels against one pinned before-run
baseline. Its acceptance verdict is direct before/after wall time from
`ratio_diff.py`. By default the suite runs only the TIRx implementation;
passing `--with-references` also runs every declared external reference
(baseline) implementation, which enables the reference-ratio regression
report against `baseline.json`. References remain diagnostics — they never
replace the direct verdict. The same flag exists on
`python -m tirx_kernels.bench` for single-workload diagnostics.

## Workloads

`config/<category>/<kernel>.yaml` is the workload source of truth. Each
registered kernel has one file. Files with `default_suite: true` select one to
three representative single-GPU rows with `default: true`; curated three-row
files label them `small`, `medium`, and `large`. The current default roster is
263 rows across 90 device kernels: 260 rows from 89 kernels validated on
`sm_100a`, `sm_103a`, and `sm_107a`, plus 3 rows from the `sm_107a`-only Rubin
BMM. Thus an SM107 run selects all 263 rows, while SM100 and SM103 runs retain
the 260-row subset.
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
| `--cooldown S` | Delay before each implementation |
| `--threshold PCT` | Report threshold; the direct gate remains fixed at 1% |
| `--out-dir PATH` | Artifact root |
| `--check-imports` | Resolve selected imports and exit |
| `--no-report` | Skip report generation |
| `--with-references` | Also run external references and the ratio report (not with `--ab-before`) |

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
