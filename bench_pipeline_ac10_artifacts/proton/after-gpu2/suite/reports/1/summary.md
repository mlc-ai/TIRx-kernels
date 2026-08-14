# bench-suite run 1

- label: `proton-after-0400e58-gpu2-schema3`
- git: tir=`ea0950ab`  tirx-kernels=`0400e585-dirty`  tirx-bench-ci=`-`
- status: ok=3 (over 3 workloads)

## Pipeline critical path

Parallel CPU preparation overlaps across one-shot children; only the first READY latency is charged before the GPU schedule.

- process model: `one_shot_child_per_workload`
- cost-model schema: `3`
- cost-model evidence: `measured`
- observed critical wall: `46.575s`
- first child spawn / spawn span: `0.116s` / `0.002s`
- first READY latency: `5.313s`
- ideal GPU list schedule (all workloads READY at time zero): `41.215s`
- CPU-READY-constrained GPU list schedule: `41.215s`
- attempt-READY-constrained GPU list schedule: `41.215s`
- eligibility-constrained GPU list schedule: `41.215s`
- GPU claim card-time total / by index: `41.215s` / `{'2': 41.21505117416382}`
- post-GPU_START execution time total / by index: `38.511s` / `{'2': 38.51134753227234}`
- CPU READY starvation: `0.000s`
- interference retry READY delay / interrupted GPU ownership: `0.000s` / `0.000s`
- foreign GPU wait: `0.000s`
- expected critical wall: `46.528s`
- expected critical wall including foreign occupancy: `46.528s`
- unexplained residual: `0.047s`
- internal dispatch latency p95/max: `0.043s` / `0.043s`
- raw / foreign-overlap dispatch wait p95: `0.043s` / `0.000s`
- ASSIGN-to-GPU-start p95/max: `1.008s` / `1.008s`
- initial-attempt dispatch p95/max: `0.043s` / `0.043s`
- final process-reap tail: `0.616s`
- observed bounds: preparing=3, READY=2, buffered=3, active children=3
- host resource peaks: process tree=16, RSS=3.617 GiB, FDs=254

### Workload phase breakdown

Times are wall-clock seconds. `specialize/compile` includes workload specialization, IR generation, and compilation after config resolution.

| workload | GPU | startup | CLI bootstrap | framework import | exact import | config resolve | specialize/compile | CPU prepare | READY wait | ASSIGN handoff | GPU stage | result handoff | reap tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp16_bf16_gemm/fp16_1024x1024x1024` | `2` | 0.033 | 0.003 | 2.255 | 0.416 | 0.000 | 2.490 | 5.164 | 0.043 | 0.758 | 12.418 | 0.000 | 0.649 |
| `fp16_bf16_gemm/fp16_4096x4096x4096` | `2` | 0.032 | 0.003 | 2.253 | 0.417 | 0.000 | 2.601 | 5.274 | 13.111 | 1.008 | 13.023 | 0.000 | 0.729 |
| `fp16_bf16_gemm/fp16_2048x2048x2048` | `2` | 0.032 | 0.003 | 2.254 | 0.426 | 0.000 | 2.801 | 5.483 | 26.936 | 0.937 | 13.069 | 0.000 | 0.616 |

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

## `fp16_bf16_gemm`

_baseline impl_: `torch-cublas` · _ours_: `tir` · _ratio_ = baseline/ours · `>1` means ours is faster

| config | tir | torch-cublas | torch-cublas/tir | attempt | gpus |
|---|---:|---:|---:|---:|---:|
| fp16_1024x1024x1024 | 6.90 | 6.01 | **0.871** | 1 | 2 |
| fp16_2048x2048x2048 | 16.45 | 15.85 | **0.964** | 1 | 2 |
| fp16_4096x4096x4096 | 95.75 | 91.96 | **0.960** | 1 | 2 |
