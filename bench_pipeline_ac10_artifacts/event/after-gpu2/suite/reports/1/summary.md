# bench-suite run 1

- label: `event-after-a5abeca-gpu2-schema3`
- git: tir=`ea0950ab`  tirx-kernels=`a5abecab-dirty`  tirx-bench-ci=`-`
- status: ok=1 (over 1 workloads)

## Pipeline critical path

Parallel CPU preparation overlaps across one-shot children; only the first READY latency is charged before the GPU schedule.

- process model: `one_shot_child_per_workload`
- cost-model schema: `3`
- cost-model evidence: `measured`
- observed critical wall: `30.651s`
- first child spawn / spawn span: `0.188s` / `0.000s`
- first READY latency: `3.296s`
- ideal GPU list schedule (all workloads READY at time zero): `24.822s`
- CPU-READY-constrained GPU list schedule: `24.822s`
- attempt-READY-constrained GPU list schedule: `26.689s`
- eligibility-constrained GPU list schedule: `27.355s`
- GPU claim card-time total / by index: `24.822s` / `{'2': 24.8216814994812}`
- post-GPU_START execution time total / by index: `23.629s` / `{'2': 23.628719091415405}`
- CPU READY starvation: `0.000s`
- interference retry READY delay / interrupted GPU ownership: `1.868s` / `13.560s`
- foreign GPU wait: `0.665s`
- expected critical wall: `29.986s`
- expected critical wall including foreign occupancy: `30.651s`
- unexplained residual: `0.000s`
- internal dispatch latency p95/max: `0.059s` / `0.059s`
- raw / foreign-overlap dispatch wait p95: `0.875s` / `0.875s`
- ASSIGN-to-GPU-start p95/max: `1.099s` / `1.099s`
- initial-attempt dispatch p95/max: `0.059s` / `0.059s`
- retry-attempt dispatch p95/max: `0.000s` / `0.000s`
- final process-reap tail: `0.666s`
- observed bounds: preparing=1, READY=1, buffered=1, active children=1
- host resource peaks: process tree=5, RSS=1.867 GiB, FDs=177

### Workload phase breakdown

Times are wall-clock seconds. `specialize/compile` includes workload specialization, IR generation, and compilation after config resolution.

| workload | GPU | startup | CLI bootstrap | framework import | exact import | config resolve | specialize/compile | CPU prepare | READY wait | ASSIGN handoff | GPU stage | result handoff | reap tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `act_and_mul/silu_fp16_d4096_t1` | `2` | 0.033 | 0.003 | 2.252 | 0.292 | 0.000 | 0.528 | 3.075 | 16.093 | 0.028 | 11.234 | 0.000 | 0.666 |

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
| silu_fp16_d4096_t1 | 6.21 | 6.17 | 1.006 | 5 | 2 |
