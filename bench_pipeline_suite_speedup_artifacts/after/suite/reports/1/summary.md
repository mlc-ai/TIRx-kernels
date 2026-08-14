# bench-suite run 1

- label: `suite-speedup-after-0d28f50-cold`
- git: tir=`ea0950ab`  tirx-kernels=`0d28f507-dirty`  tirx-bench-ci=`-`
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
- observed bounds: preparing=16, READY=1, buffered=16, active children=22
- host resource peaks: process tree=59, RSS=17.559 GiB, FDs=1001

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

## `deepgemm_fp8_fp4_mega_moe`

| config | ratio | attempt | gpus |
|---|---:|---:|---:|
| t64_m64_h7168_i3072_e384_k6_g1 **[FAIL]** | — | 1 |  |
