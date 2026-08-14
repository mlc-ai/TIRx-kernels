# bench-suite run 1

- label: `kineto-before-a91a1b7-gpu2-uuid`
- git: tir=`ea0950ab`  tirx-kernels=`a91a1b76`  tirx-bench-ci=`-`
- status: ok=1 (over 1 workloads)

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
| tp1_m8192_n24576_k4096_fp16_dynamic | 1009.16 | 1056.45 | 1051.64 | — | 1 | 2 |
