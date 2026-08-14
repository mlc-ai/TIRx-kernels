# bench-suite run 1

- label: `kineto-after-cbd6b01-gpu2-schema3`
- git: tir=`ea0950ab`  tirx-kernels=`cbd6b016-dirty`  tirx-bench-ci=`-`
- status: FAIL=1 (over 1 workloads)

## Pipeline critical path

Parallel CPU preparation overlaps across one-shot children; only the first READY latency is charged before the GPU schedule.

- process model: `one_shot_child_per_workload`
- cost-model schema: `3`
- cost-model evidence: `missing`
- missing reason: no workload produced an ok result with a complete timeline and valid GPU assignment
- complete GPU timelines: `0` / `1`
- complete cost-model measurements: `0` / `1`
- expected wall, residual, starvation, foreign wait, and latency percentiles are intentionally unpublished because the run lacks complete measurement evidence
- final process-reap tail: `0.000s`
- observed bounds: preparing=1, READY=0, buffered=1, active children=1
- host resource peaks: process tree=6, RSS=3.640 GiB, FDs=348

### Workload phase breakdown

Times are wall-clock seconds. `specialize/compile` includes workload specialization, IR generation, and compilation after config resolution.

| workload | GPU | startup | CLI bootstrap | framework import | exact import | config resolve | specialize/compile | CPU prepare | READY wait | ASSIGN handoff | GPU stage | result handoff | reap tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `allgather_gemm/tp1_m8192_n24576_k4096_fp16_dynamic` | `2` | 0.032 | 0.003 | 2.356 | 0.412 | 0.000 | 4.791 | 7.562 | 27.374 | 0.002 | 6.963 | 0.002 | 0.265 |

## Multi-GPU runtime validation exemption

- validation status: `exempted_by_human_unmeasured`
- runtime evidence: not collected, by explicit human direction; this is neither a pass nor a missing-result placeholder
- migrated semantics: pipeline-only late assignment, atomic full-GPU claim before rank/CUDA startup, barriers, sample-wise max aggregation, Kineto spans, and process-group cleanup
- non-multi-GPU evidence: protocol assignment-count rejection, atomic claim failure, and rank lifecycle ordering

## Baseline impl provenance

- `cutlass`: not installed
- `deep_gemm`: v2.6.1 @`1097cce8-dirty` (/home/hongyij/workspace/tirx-kernels)
- `flash_attn`: v2.8.4
- `flash_kda`: v0.0.1+1ce47ea @`1ce47ea3` (/home/hongyij/workspace/kda-backends/FlashKDA)
- `flashinfer`: v0.6.17
- `sglang`: v0.0.0.dev0 @`96a04cb1` (/home/hongyij/workspace/tirx-kernels/.porting/deps/sglang-96a04cb1)
- `torch`: v2.11.0+cu130 cuda=13.0 torch_git=70d99e998b49

## `allgather_gemm`

| config | ratio | attempt | gpus |
|---|---:|---:|---:|
| tp1_m8192_n24576_k4096_fp16_dynamic **[FAIL]** | — | 3 | 2 |
