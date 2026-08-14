# bench-suite run 1

- label: `event-before-a91a1b7-gpu2-uuid`
- git: tir=`ea0950ab`  tirx-kernels=`a91a1b76`  tirx-bench-ci=`-`
- status: ok=1 (over 1 workloads)

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
| silu_fp16_d4096_t1 | 6.94 | 6.19 | 1.121 | 1 | 2 |
