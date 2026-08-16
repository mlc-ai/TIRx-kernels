# bench-suite

`bench-suite` is the fixed-runner absolute-performance gate for TIRx kernels.
By default it runs only the TIRx candidate, records its GPU time in
microseconds, and compares that time with the checked-in timing for the same
kernel/config on the same runner. External optimized implementations remain
available only through the explicit diagnostic `--with-references` mode.

## Contracts

- Kernel modules expose small correctness cases as `CONFIGS` and production
  benchmark cases as a separate `BENCH_CONFIGS` list.
- Every benchmark calls `bench({"tirx": launch}, references={...}, ...)` when a
  peer exists. Reference entries are lazy builders: the default mode discards
  them without importing or constructing their external packages. Input
  construction, compilation, workspace setup, and correctness math stay
  outside the timed closure.
- `config/**/*.yaml` is the benchmark workload source of truth. Its
  `default: true` rows form the default sweep; explicit workload files can
  select any production row.
- `baseline.json` contains only pinned TIRx timings and their timing methods. A
  run is comparable only when both its timing method and exact `runner`
  identity match the pin. The identity
  includes hostname, GPU model/SM profile, immutable GPU UUIDs and PCI bus IDs,
  max clock and power policy, NVIDIA driver, and the complete NVLink/PCIe
  topology matrix.
- `current / pinned > 1` is slower. A regression must exceed both the measured
  default noise limits: 4% and 5µs.

## Run

```bash
cd /path/to/tirx-kernels
pip install -e .

export TVM_PATH=/path/to/tvm
export PYTHONPATH="${TVM_PATH}/python"
export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"

python -m tirx_kernels.bench_suite
```

GPU selection is automatic. The suite excludes cards occupied at startup,
probes the remaining idle cards, admits only one homogeneous SM100 compile
profile, prepares each workload once in a fresh child, then assigns an idle GPU
for the candidate launch. Multi-GPU workloads receive an atomic ordered card
claim and use the same candidate-only timing contract.

Useful commands:

```bash
# Validate imports for the selected workloads without compiling or using CUDA.
python -m tirx_kernels.bench_suite --check-imports

# Run one workload family.
python -m tirx_kernels.bench_suite --filter mxfp8_quantize

# Explicit diagnostic comparison with that family's installed reference.
python -m tirx_kernels.bench_suite --filter mxfp8_quantize --with-references

# Inspect one kernel/config outside the suite.
python -m tirx_kernels.bench --kernel <kernel> --config <label> --rounds 5
```

Run artifacts are written below `.bench-suite/`; they are not committed.

Reference mode records every implementation in the run JSON and summary, but
does not compare against or update the pinned TIRx baseline. Missing or
incompatible reference packages fail the selected diagnostic workload. When
two references require incompatible CuTeDSL versions, run each kernel filter
in its matching environment; the default sweep needs neither environment.

## Pinned timing workflow

The checked-in files are:

| File | Meaning |
|---|---|
| `baseline.json` | Fixed-runner TIRx timings and their runner identity |
| `baseline.md` | Generated human-readable view of `baseline.json` |
| `timing_diff.py` | Absolute current-versus-pinned timing report |
| `promote_baseline.py` | The only supported baseline update path |

Replace the full pinned sweep:

```bash
python -m tirx_kernels.bench_suite
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<id>.json
```

Patch selected successful rows only when the run and baseline have identical
runner identities:

```bash
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<id>.json --merge
```

Promotion keeps only successful `tirx` timings, their timing methods, and the
aggregation contract, then regenerates `baseline.md`.
A legacy baseline without a runner identity cannot be merged or used for a
regression verdict; replace it with a complete run from the intended fixed
runner.

## Measurement lifecycle

Each workload uses one fresh child process. CPU preparation completes before GPU
assignment. The GPU stage constructs operands once and, by default, times only
the TIRx launch. `--with-references` constructs and times the selected lazy
peers in that same child. Local workloads use CUDA-event wall timing;
distributed workloads opt into Kineto explicitly. Timers record five rounds by
default and aggregate their arithmetic mean. The scheduler rejects foreign GPU
activity and retries the same prepared child on an idle eligible card; failed
workloads stop the sweep.

The run JSON retains diagnostic probe/pipeline/sample metadata. Promotion strips
that ephemeral state and stores only the pinned runner, source revisions,
aggregation contract, and absolute TIRx time.

## Main flags

| Flag | Default | Meaning |
|---|---:|---|
| `--rounds N` | 5 | Complete timing rounds per workload |
| `--cooldown S` | 0 | Cooldown before each round |
| `--threshold PCT` | 4 | Relative slowdown floor versus the pinned TIRx time |
| `--absolute-threshold-us US` | 5 | Absolute slowdown floor versus the pin |
| `--filter TEXT` | none | Select kernel names containing `TEXT` |
| `--workloads PATH` | generated defaults | Explicit flat workload YAML |
| `--with-references` | false | Import and time lazy external peers; diagnostic only |
| `--no-report` | false | Run measurements without comparing to the pin |
| `--check-imports` | false | Import-only validation |
