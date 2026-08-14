# bench-suite run 1

- label: `cudagraph-after-6416bb6-gpu2-schema3`
- git: tir=`ea0950ab`  tirx-kernels=`6416bb61-dirty`  tirx-bench-ci=`-`
- status: ok=1 (over 1 workloads)

## Pipeline critical path

Parallel CPU preparation overlaps across one-shot children; only the first READY latency is charged before the GPU schedule.

- process model: `one_shot_child_per_workload`
- cost-model schema: `3`
- cost-model evidence: `measured`
- observed critical wall: `28.963s`
- first child spawn / spawn span: `0.132s` / `0.000s`
- first READY latency: `3.621s`
- ideal GPU list schedule (all workloads READY at time zero): `23.978s`
- CPU-READY-constrained GPU list schedule: `23.978s`
- attempt-READY-constrained GPU list schedule: `24.819s`
- eligibility-constrained GPU list schedule: `25.343s`
- GPU claim card-time total / by index: `23.978s` / `{'2': 23.977768182754517}`
- post-GPU_START execution time total / by index: `23.077s` / `{'2': 23.076583862304688}`
- CPU READY starvation: `0.000s`
- interference retry READY delay / interrupted GPU ownership: `0.841s` / `4.160s`
- foreign GPU wait: `0.524s`
- expected critical wall: `28.439s`
- expected critical wall including foreign occupancy: `28.963s`
- unexplained residual: `0.000s`
- internal dispatch latency p95/max: `0.040s` / `0.040s`
- raw / foreign-overlap dispatch wait p95: `0.801s` / `0.801s`
- ASSIGN-to-GPU-start p95/max: `0.841s` / `0.841s`
- initial-attempt dispatch p95/max: `0.040s` / `0.040s`
- retry-attempt dispatch p95/max: `0.000s` / `0.000s`
- final process-reap tail: `1.018s`
- observed bounds: preparing=1, READY=1, buffered=1, active children=1
- host resource peaks: process tree=5, RSS=1.994 GiB, FDs=191

### Workload phase breakdown

Times are wall-clock seconds. `specialize/compile` includes workload specialization, IR generation, and compilation after config resolution.

| workload | GPU | startup | CLI bootstrap | framework import | exact import | config resolve | specialize/compile | CPU prepare | READY wait | ASSIGN handoff | GPU stage | result handoff | reap tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `act_and_mul/silu_fp16_d4096_t1` | `2` | 0.035 | 0.003 | 2.407 | 0.421 | 0.000 | 0.623 | 3.453 | 5.525 | 0.052 | 19.766 | 0.001 | 1.018 |

## Multi-GPU runtime validation exemption

- validation status: `exempted_by_human_unmeasured`
- runtime evidence: not collected, by explicit human direction; this is neither a pass nor a missing-result placeholder
- migrated semantics: pipeline-only late assignment, atomic full-GPU claim before rank/CUDA startup, barriers, sample-wise max aggregation, Kineto spans, and process-group cleanup
- non-multi-GPU evidence: protocol assignment-count rejection, atomic claim failure, and rank lifecycle ordering

## Baseline impl provenance

- `cutlass`: v4.7.0
- `deep_gemm`: v2.3.0
- `flash_attn`: v2.8.4
- `flash_kda`: v0.0.1+1ce47ea @`1ce47ea3` (/home/hongyij/workspace/kda-backends/FlashKDA)
- `flashinfer`: v0.6.17
- `sglang`: not installed
- `torch`: v2.11.0+cu130 cuda=13.0 torch_git=70d99e998b49

## `act_and_mul`

_baseline impl_: `flashinfer` · _ours_: `tirx` · _ratio_ = baseline/ours · `>1` means ours is faster

| config | flashinfer | tirx | flashinfer/tirx | attempt | gpus |
|---|---:|---:|---:|---:|---:|
| silu_fp16_d4096_t1 | 2.21 | 2.00 | 1.105 | 3 | 2 |
