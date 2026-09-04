# NVIDIA Thor classic-kernel same-device baselines

Measured on 2026-09-04 on one NVIDIA Jetson AGX Thor Developer Kit. The table is the post-tuning classic-family campaign plus two longer confirmation rows. Every speedup is `reference latency / TIRx latency`; values above 1.0 favor TIRx.

## Summary

| Metric | Result |
|---|---:|
| Measured exact-shape rows | **20/20 passed** |
| Main classic families with at least one numeric row | **12/17** |
| TIRx faster by more than 5% | **3/20** |
| Within 5% | **17/20** |
| Reference faster by more than 5% | **0/20** |
| Geometric-mean TIRx speedup | **1.017x** |
| Rows with either CV above 10% | **8/20** |

The mixed geomean is descriptive only: it gives one vote to each selected workload, not to each model invocation. The per-row numbers are the result to use.
The source-to-benchmark mapping for every row is audited in [THOR_SOURCE_BENCHMARK_AUDIT.md](THOR_SOURCE_BENCHMARK_AUDIT.md).

## Complete numeric table

| Family | Kernel / config | TIRx µs | CV | Reference | Reference µs | CV | Speedup |
|---|---|---:|---:|---|---:|---:|---:|
| Dense GEMM | `fp16_bf16_gemm/bf16_4096x4096x4096` | 855.206 | 14.3% | `torch-cublas` | 956.604 | 9.6% | **1.119x** |
| Dense GEMM | `fp16_bf16_gemm/fp16_4096x4096x4096` | 1125.359 | 35.5% | `torch-cublas` | 1179.606 | 22.8% | **1.048x** |
| Quantized GEMM | `nvfp4_gemm/4096x4096x4096` | 355.540 | 6.9% | `flashinfer` | 366.789 | 8.7% | **1.032x** |
| Attention | `flash_attention4/s4096_h32kv4_causal` | 957.935 | 2.4% | `flashattn_fa4_cutedsl` | 937.332 | 2.1% | **0.978x** |
| Normalization | `flashinfer_fused_add_rmsnorm/fused_bf16_m32_h4096_xc_rc_pdl1` | 14.249 | 7.0% | `flashinfer_cutedsl` | 13.873 | 5.4% | **0.974x** |
| Normalization | `flashinfer_fused_dit_layernorm/grgb_bf16_b1_r1920` | 267.990 | 0.5% | `flashinfer_cuda` | 266.755 | 0.6% | **0.995x** |
| Normalization | `flashinfer_layernorm/bf16_m128_h16384_xc_yc_pdl0_eps1e6` | 73.997 | 3.6% | `flashinfer_cutedsl` | 73.668 | 2.8% | **0.996x** |
| Normalization | `flashinfer_qk_rmsnorm/rms_bf16_b32_n32_h128_xc_yc_pdl0` | 5.505 | 3.4% | `flashinfer_cutedsl` | 5.270 | 2.1% | **0.957x** |
| Normalization | `flashinfer_rmsnorm/rms_bf16_m32_h4096_xc_yc_pdl1` | 9.450 | 10.2% | `flashinfer_cutedsl` | 9.366 | 10.4% | **0.991x** |
| Normalization | `flashinfer_rmsnorm_quant/bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | 18.403 | 10.3% | `flashinfer_cutedsl` | 18.138 | 8.5% | **0.986x** |
| Activation / quantization | `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 2233.054 | 3.6% | `flashinfer` | 2233.962 | 3.4% | **1.000x** |
| Activation / quantization | `mxfp4_quantize/fp16_linear_m4096_k4096` | 232.590 | 12.7% | `flashinfer` | 242.089 | 6.0% | **1.041x** |
| Activation / quantization | `nvfp4_quantize/fp16_linear_m4096_k4096` | 210.232 | 7.1% | `flashinfer` | 212.055 | 13.3% | **1.009x** |
| TopK | `fast_topk_clusters/f32_plain_b64_l16384_k256` | 193.904 | 1.0% | `flashinfer` | 193.215 | 0.7% | **0.996x** |
| TopK | `filtered_topk/f32_plain_r64_l8192_k256` | 43.536 | 5.9% | `flashinfer` | 48.248 | 2.1% | **1.108x** |
| TopK | `radix_topk_multi_cta/f32_basic_r4_l115188_k256_ctas3` | 65.835 | 5.4% | `flashinfer` | 64.276 | 3.2% | **0.976x** |
| TopK | `radix_topk_single_cta/f32_basic_r64_l32768_k512` | 177.806 | 2.4% | `flashinfer` | 184.081 | 2.4% | **1.035x** |
| Recurrent / SSM | `gdn_decode_bf16_ilp4/t4_b4_h8_hv16_tv16` | 76.791 | 9.8% | `flashinfer_cutedsl` | 77.799 | 10.1% | **1.013x** |
| Recurrent / SSM | `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 404.786 | 12.2% | `flashinfer_cutedsl` | 414.203 | 9.6% | **1.023x** |
| Recurrent / SSM | `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 3529.556 | 0.0% | `flashinfer_cuda` | 3830.212 | 0.0% | **1.085x** |

The NVFP4 row also measured cuBLASLt at **382.956 µs** (CV 3.8%), or **1.077x** relative to TIRx. FlashInfer is retained as that row's primary baseline to follow the requested priority.

The attention row uses upstream FA4 CuTeDSL as its primary baseline. On the same row, FlashInfer CuTeDSL measured **980.931 µs** and the legacy FlashInfer FA2 control measured **2945.198 µs**; neither secondary control enters the geomean.

The complete campaign contained all 20 rows. NVFP4 GEMM and Fused Add RMSNorm then received isolated 30-round confirmations because their full-campaign ratios contradicted prior paired runs; the table uses those longer confirmations. This substitution is reflected in the summary counts and geometric mean.

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
| Rounds / aggregation | 15; two confirmation rows use 30 / arithmetic mean |
| Warmup / repeat | 1000 ms / 100 ms per implementation per round |
| TVM/TIR revision | `15b607d6` |
| TIRx-kernels revision | `86fe2638-dirty` |
| FlashInfer version / revision | `0.6.18` / `f2e04400` |
| FlashAttention-4 revision | `0251105a` |

Rows above 10% CV are retained and visibly flagged by their CV columns. In particular, the 4096-cube GEMMs switched between fast and slow clock regimes within their rounds. Their absolute means and near-threshold ratios should not be treated as locked-clock absolute latency. The paired same-process ratios remain the comparison used here.

## Raw evidence

- Complete 20-row run: `/home/tlopexh/thor-validation/final-classic-after-tuning-15r/runs/1.json`
- NVFP4 30-round confirmation: `/home/tlopexh/TIRx-kernels/.porting/nvfp4_gemm/perf_gate/recheck-after-full-30r/runs/1.json`
- Fused Add RMSNorm 30-round confirmation: `/home/tlopexh/TIRx-kernels/.porting/flashinfer_fused_add_rmsnorm/perf_gate/recheck-current-after-full-30r/runs/1.json`
- All selected final artifacts had no failures or interference retries.
