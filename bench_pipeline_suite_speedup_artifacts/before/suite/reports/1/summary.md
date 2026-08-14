# bench-suite run 1

- label: `suite-speedup-before-a91a1b7-cold`
- git: tir=`ea0950ab`  tirx-kernels=`a91a1b76`  tirx-bench-ci=`-`
- status: FAIL=1, ok=7 (over 8 workloads)

## Baseline impl provenance

- `cutlass`: not installed
- `deep_gemm`: v2.6.1 @`1097cce8-dirty` (/home/hongyij/workspace/tirx-kernels)
- `flash_attn`: v2.8.4
- `flash_kda`: v0.0.1+1ce47ea @`1ce47ea3` (/home/hongyij/workspace/kda-backends/FlashKDA)
- `flashinfer`: v0.6.17
- `sglang`: v0.0.0.dev0 @`96a04cb1` (/home/hongyij/workspace/tirx-kernels/.porting/deps/sglang-96a04cb1)
- `torch`: v2.11.0+cu130 cuda=13.0 torch_git=70d99e998b49

## `allgather_gemm`

| config | cublas_nccl_cudagraph | cublasmp_split_p2p | tirx | ratio | attempt | gpus |
|---|---:|---:|---:|---:|---:|---:|
| tp1_m8192_n106496_k16384_fp16_dynamic | 18008.76 | 18238.14 | 19705.71 | — | 1 | 2 |
| tp1_m8192_n24576_k4096_fp16_dynamic | 1009.10 | 1058.15 | 1051.56 | — | 1 | 7 |
| tp1_m8192_n57344_k8192_fp16_dynamic | 4899.78 | 5225.68 | 6668.66 | — | 1 | 6 |

## `fp16_bf16_gemm`

_baseline impl_: `torch-cublas` · _ours_: `tir` · _ratio_ = baseline/ours · `>1` means ours is faster

| config | deepgemm-bf16 | deepgemm-cublaslt | tir | torch-cublas | torch-cublas/tir | attempt | gpus |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16_4096x4096x4096 | 88.95 | 89.89 | 92.72 | 89.91 | **0.970** | 1 | 1 |
| fp16_1024x1024x1024 | — | — | 6.82 | 6.06 | **0.889** | 1 | 3 |
| fp16_16384x16384x16384 | — | — | 5904.79 | 6906.39 | 1.170 | 1 | 4 |

## `gemm_reduce_scatter`

| config | cublas_nccl_cudagraph | cublasmp_split_p2p | tirx | ratio | attempt | gpus |
|---|---:|---:|---:|---:|---:|---:|
| tp1_m8192_n16384_k53248_fp16_dynamic **[FAIL]** | — | — | — | — | 1 | 4 |
| tp1_m8192_n4096_k12288_fp16_dynamic | 565.89 | 752.28 | 600.00 | — | 1 | 5 |
