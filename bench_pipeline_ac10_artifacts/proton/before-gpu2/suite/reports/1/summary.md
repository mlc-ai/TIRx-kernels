# bench-suite run 1

- label: `proton-before-a91a1b7-gpu2-uuid`
- git: tir=`ea0950ab`  tirx-kernels=`a91a1b76`  tirx-bench-ci=`-`
- status: ok=3 (over 3 workloads)

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
| fp16_1024x1024x1024 | 6.92 | 6.02 | **0.870** | 1 | 2 |
| fp16_2048x2048x2048 | 16.54 | 15.85 | **0.958** | 1 | 2 |
| fp16_4096x4096x4096 | 95.70 | 92.71 | **0.969** | 1 | 2 |
