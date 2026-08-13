# Bench Suite CPU/GPU Pipeline Draft

## Problem

The benchmark suite currently acquires GPU resources before starting each benchmark subprocess. A workload therefore holds an idle GPU while Python starts, the target module is parsed, the TIRx program is specialized and generated, and TVM/NVRTC compilation runs. These CPU-heavy phases repeat across workloads and serialize behind the GPU allocation boundary.

The benchmark measurement protocol itself must remain unchanged: the same implementations, implementation order, reference setup, timer, warmup/repeat budgets, round count, cooldown, raw round samples, aggregation, fail-fast behavior, and interference retry semantics must be preserved.

## Confirmed Target

Split benchmark execution into an explicit two-stage contract:

1. CPU prepare stage: import/parse the target benchmark module, specialize the workload, generate IR, and compile the executable. This stage must not initialize CUDA, allocate GPU memory, run setup kernels, or acquire a GPU token.
2. GPU stage: after preparation reports READY, dynamically acquire the workload's required number of currently eligible GPUs, bind them to the already-prepared child process, then perform CUDA initialization, allocation, reference setup/autotune, correctness warmup, and the unchanged benchmark timing protocol.

Run many CPU prepare stages concurrently. Run up to N GPU stages concurrently when N eligible idle GPUs exist, while retaining atomic multi-GPU claims. Adapt to external GPU load using the existing utilization/memory gates and interference monitoring.

Reduce the default measured sweep at its configuration source: for every kernel currently carrying more than three `default: true` configs, retain exactly three production-relevant small, medium, and large representatives. Keep every other config in the same kernel YAML with `default: false`, so it remains benchable through an explicit workload file. Do not truncate configs dynamically in the scheduler. With the current repository inventory this changes the generated default sweep from 234 to 112 workloads, a reduction of 122 workloads (52.1%).

The desired steady-state critical path is:

```text
time to first prepared workload
+ GPU-stage list-scheduling makespan over the GPUs that are actually eligible over time
+ negligible scheduler-controlled dispatch gaps
```

After the first workload becomes READY, an internally free eligible GPU must not sit idle when the ready queue contains a satisfiable job. Waiting caused by foreign GPU load must be measured separately from scheduler overhead.

## Prototype Evidence

An `fp16_bf16_gemm` prototype separated `prepare_bench()` from `PreparedBench.run_gpu()` without changing `run_bench()` behavior.

For fp16 square GEMMs at 1024, 4096, and 8192, using one B200, five rounds, and a diagnostic 0.1 second cooldown:

- Sequential end-to-end wall time: 29.70 seconds.
- Concurrent CPU prepare plus one serialized GPU stage: 19.17 seconds.
- Observed speedup: 1.55x; wall-time reduction: 35.4%.
- Per-workload CPU preparation after process startup: approximately 3.9 to 4.2 seconds.
- Serialized GPU stages: approximately 3.3, 4.0, and 4.5 seconds.
- The process-start/import cost overlapped with CPU preparation and belongs in the first-READY latency, rather than recurring on the steady-state GPU critical path.
- Direct target-module import plus specialization/TVM/NVRTC compilation did not initialize CUDA.

These measurements establish direction and a cost model; they are not fixed performance thresholds for unrelated workloads or machines.

## Required Design Properties

- Keep the prepared executable in the same child process from CPU preparation through GPU execution; do not serialize opaque runtime modules between processes.
- Use exactly one fresh child process per workload attempt. The child performs one CPU prepare, waits for one late GPU assignment, runs one GPU stage, reports one result, and exits. Never recycle a post-GPU child for another prepare; do not add resident workers, reusable worker pools, fork templates, or same-interpreter compile thread pools.
- Use an explicit parent/child READY and GPU-assignment protocol.
- Preserve dynamic `GpuPool` selection, atomic `num_gpus` acquisition, cancellation, fail-fast, and interference retry behavior.
- Use event-driven dispatch for internally released GPUs. Polling is permitted only for observing external GPU state changes.
- Bound concurrent prepare processes and ready-queue depth to avoid host-memory exhaustion, while exposing enough parallelism to keep eligible GPUs fed.
- Record phase timestamps and causes for GPU idle time: initial preparation, ready-queue starvation, external occupancy, unsatisfied multi-GPU topology, interference retry, GPU execution, and result/finalization.
- Migrate every benchable workload to the two-stage contract. Kernel aliases, custom reference builders, alternate timers, multi-GPU jobs, and distributed/rank-based workloads may use specialized adapters behind the same contract, but none may retain or silently select the old one-stage execution path.
- Resolve every public kernel name to exactly one Python module without importing the whole kernel tree. Derive direct mappings where possible and handle alias-named kernels from the same canonical source metadata; validate the resolved module against runtime `KERNEL_META`. Do not add a competing hand-maintained manifest.
- Treat each kernel YAML's `default` flags as the sole authority for default measured coverage. Kernels with more than three currently measured configs must be curated to exactly three small/medium/large representatives; kernels already at three or fewer remain unchanged. Selection must span increasing semantic work and, where possible, distinct production dispatch regimes rather than using lexicographic labels or runtime truncation.
- Keep all non-selected configs present and pipeline-capable. The smaller default sweep changes routine coverage, not the complete set of supported benchmark workloads.
- Do not reduce rounds, timer budgets, cooldown, reference coverage, or correctness semantics as a performance shortcut.
- Do not run the full suite while another session is benchmarking. Use a small representative workload matrix for implementation evidence.
- Migrate multi-GPU, distributed, Kineto, and MegaMoE code to the same pipeline-only lifecycle, but do not occupy multiple GPUs for acceptance testing on the shared host. Keep multi-GPU configs out of the default measured sweep while preserving explicit execution.
- Report multi-GPU runtime validation as `exempted_by_human_unmeasured`. This status is distinct from `passed` and `missing`: it must include the migrated lifecycle semantics, the explicit human exemption, and the fact that no multi-GPU runtime evidence was collected. Never encode the exemption as pass, zero, null, or an empty cell.
- Retain non-multi-GPU structural evidence for multi-GPU semantics: reject assignment-count mismatches, never degrade an unsatisfied atomic claim into a partial allocation, and validate rank lifecycle ordering with a single-GPU or mocked runtime.

## Quantitative Interpretation

The critical-path model is a hard structural requirement. Exact absolute seconds are workload- and machine-dependent and are trend targets, not universal hard requirements.

Scheduler-controlled overhead is independently measurable and should be negligible relative to GPU-stage work. When an eligible internally free GPU and a satisfiable READY workload coexist, dispatch should be event-driven with sub-100-millisecond p95 latency on the target host. Representative targeted runs should show no recurring per-workload CPU preparation on the GPU critical path after the first READY event.

## Scope Notes

Start by generalizing the proven GEMM split into reusable runner and scheduler primitives, then use representative kernels from different families to harden the contract. Complete the same code migration for every benchable kernel and workload, including alias-named modules and multi-GPU/distributed rank lifecycles, before declaring implementation coverage complete. Atomic GPU claims, barriers, rank aggregation, Kineto/MegaMoE timing, correctness, and interference isolation remain mandatory. Multi-GPU code completion is not runtime validation: runtime evidence remains explicitly exempted and unmeasured until a human authorizes a later multi-GPU run.

Timer/cooldown changes, CUDA kernel performance tuning, and full-suite baseline promotion are separate concerns. Full migration does not authorize running the complete generated timed sweep while another session is benchmarking; development and acceptance use static all-workload gates, no-card structural protocol tests, and a small single-GPU timed matrix until the machine is available for a later complete 112-workload sweep. Multi-GPU runtime testing is outside this acceptance run even if multiple cards appear idle.
