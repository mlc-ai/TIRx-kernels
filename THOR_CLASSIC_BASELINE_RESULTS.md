# NVIDIA Thor classic-kernel same-device baselines

Measured on 2026-09-03 on one NVIDIA Jetson AGX Thor Developer Kit. The table combines the original representative campaign with the classic-family additions. Every speedup is `reference latency / TIRx latency`; values above 1.0 favor TIRx.

## Summary

| Metric | Result |
|---|---:|
| Measured exact-shape rows | **20/20 passed** |
| Main classic families with at least one numeric row | **12/17** |
| TIRx faster by more than 5% | **5/20** |
| Within 5% | **10/20** |
| Reference faster by more than 5% | **5/20** |
| Geometric-mean TIRx speedup | **0.999x** |
| Rows with either CV above 10% | **9/20** |

The mixed geomean is descriptive only: it gives one vote to each selected workload, not to each model invocation. The per-row numbers are the result to use.
The source-to-benchmark mapping for every row is audited in [THOR_SOURCE_BENCHMARK_AUDIT.md](THOR_SOURCE_BENCHMARK_AUDIT.md).

## Complete numeric table

| Family | Kernel / config | TIRx µs | CV | Reference | Reference µs | CV | Speedup |
|---|---|---:|---:|---|---:|---:|---:|
| Dense GEMM | `fp16_bf16_gemm/bf16_4096x4096x4096` | 1346.893 | 28.8% | `torch-cublas` | 1358.055 | 16.7% | **1.008x** |
| Dense GEMM | `fp16_bf16_gemm/fp16_4096x4096x4096` | 998.801 | 0.7% | `torch-cublas` | 1282.266 | 0.9% | **1.284x** |
| Quantized GEMM | `nvfp4_gemm/4096x4096x4096` | 374.545 | 8.9% | `flashinfer` | 347.978 | 8.2% | **0.929x** |
| Attention | `flash_attention4/s4096_h32kv4_causal` | 985.535 | 3.5% | `flashattn_fa4_cutedsl` | 964.548 | 1.5% | **0.979x** |
| Normalization | `flashinfer_fused_add_rmsnorm/fused_bf16_m32_h4096_xc_rc_pdl1` | 16.030 | 19.2% | `flashinfer_cutedsl` | 13.827 | 9.1% | **0.863x** |
| Normalization | `flashinfer_fused_dit_layernorm/grgb_bf16_b1_r1920` | 334.494 | 11.2% | `flashinfer_cuda` | 325.428 | 10.3% | **0.973x** |
| Normalization | `flashinfer_layernorm/bf16_m128_h16384_xc_yc_pdl0_eps1e6` | 79.794 | 7.2% | `flashinfer_cutedsl` | 78.551 | 7.0% | **0.984x** |
| Normalization | `flashinfer_qk_rmsnorm/rms_bf16_b32_n32_h128_xc_yc_pdl0` | 7.001 | 12.9% | `flashinfer_cutedsl` | 6.856 | 7.9% | **0.979x** |
| Normalization | `flashinfer_rmsnorm/rms_bf16_m32_h4096_xc_yc_pdl1` | 11.330 | 21.1% | `flashinfer_cutedsl` | 9.885 | 9.6% | **0.872x** |
| Normalization | `flashinfer_rmsnorm_quant/bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | 22.736 | 18.0% | `flashinfer_cutedsl` | 20.016 | 8.3% | **0.880x** |
| Activation / quantization | `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 2389.711 | 11.0% | `flashinfer` | 2386.804 | 12.6% | **0.999x** |
| Activation / quantization | `mxfp4_quantize/fp16_linear_m4096_k4096` | 280.820 | 11.1% | `flashinfer` | 323.658 | 14.2% | **1.153x** |
| Activation / quantization | `nvfp4_quantize/fp16_linear_m4096_k4096` | 254.796 | 11.1% | `flashinfer` | 254.155 | 9.4% | **0.997x** |
| TopK | `fast_topk_clusters/f32_plain_b64_l16384_k256` | 198.190 | 0.9% | `flashinfer` | 197.375 | 0.3% | **0.996x** |
| TopK | `filtered_topk/f32_plain_r64_l8192_k256` | 41.798 | 4.5% | `flashinfer` | 47.615 | 3.3% | **1.139x** |
| TopK | `radix_topk_multi_cta/f32_basic_r4_l115188_k256_ctas3` | 66.400 | 2.2% | `flashinfer` | 67.430 | 2.0% | **1.016x** |
| TopK | `radix_topk_single_cta/f32_basic_r64_l32768_k512` | 174.729 | 3.6% | `flashinfer` | 187.086 | 1.4% | **1.071x** |
| Recurrent / SSM | `gdn_decode_bf16_ilp4/t4_b4_h8_hv16_tv16` | 83.903 | 5.4% | `flashinfer_cutedsl` | 82.843 | 7.0% | **0.987x** |
| Recurrent / SSM | `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 431.529 | 8.2% | `flashinfer_cutedsl` | 383.748 | 9.8% | **0.889x** |
| Recurrent / SSM | `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 3526.523 | 0.0% | `flashinfer_cuda` | 3827.694 | 0.0% | **1.085x** |

The NVFP4 row also measured cuBLASLt at **360.902 µs** (CV 7.1%), or **0.964x** relative to TIRx. FlashInfer is retained as that row's primary baseline to follow the requested priority.

The attention row uses upstream FA4 CuTeDSL as its primary baseline. On the same row, FlashInfer CuTeDSL measured **971.336 µs** and the legacy FlashInfer FA2 control measured **2944.449 µs**; neither secondary control enters the geomean.

## Main-family coverage without a publishable Thor number

| Family | Status | Reason |
|---|---|---|
| Dense/batched FP8 GEMM | N/A | The exact pinned DeepGEMM entry rejects compute capability 11; FlashInfer `mm_fp8` has a different low-latency/scale contract. |
| Grouped GEMM | N/A | The exact pinned DeepGEMM host dispatch rejects Thor; no contract-matched independent launch has passed yet. |
| Fused MoE | N/A | `sm100_fp8_fp4_mega_moe` remains blocked by the compute-10-only DeepGEMM scale-layout host path; some cases also require multiple GPUs. |
| Block-sparse / sparse-MLA attention | N/A | TIRx is numerically validated, but the inspected exact external source dispatchers have no compute-11 Thor path. |
| MQA logits / indexer | N/A | TIRx is numerically validated; DeepGEMM and SGLang CuTeDSL timing peers are disabled on Thor by their compute-10 host dispatch. |

These are unavailable comparisons, not zero performance and not failed TIRx correctness. They must not be included in the geomean until an exact independent Thor baseline launches.

## Measurement provenance

| Field | Value |
|---|---|
| GPU | NVIDIA Jetson AGX Thor Developer Kit, 20 SMs |
| CUDA architecture | `sm_110a` |
| Power mode | `MAXN`; dynamic clocks, because `jetson_clocks` requires root |
| Timer | Proton, cold-L2 per timed iteration |
| Rounds / aggregation | 5 / arithmetic mean |
| Warmup / repeat | 1000 ms / 100 ms per implementation per round |
| TVM/TIR revision | `15b607d6` |
| TIRx-kernels revision | `d3e698c7` |
| FlashInfer version / revision | `0.6.18` / `f2e04400` |
| FlashAttention-4 revision | `0251105a` |

Rows above 10% CV are retained and visibly flagged by their CV columns. In particular, the BF16 4096-cube GEMM switched between fast and slow clock regimes within its five rounds. Its absolute mean and near-threshold ratio should be rerun with `sudo jetson_clocks` before publication.

## Raw evidence

- Original representative run: `/home/tlopexh/thor-validation/source-bench-final/runs/1.json`
- Classic additions run: `/home/tlopexh/thor-validation/classic-baseline-additions/runs/3.json`
- Both runs used five round samples per implementation and had no interference retries in the selected final artifacts.
