# NVIDIA Thor classic-kernel same-device baselines

Measured on 2026-09-04 on one NVIDIA Jetson AGX Thor Developer Kit. The table combines the original representative campaign with the classic-family additions. Every speedup is `reference latency / TIRx latency`; values above 1.0 favor TIRx.

## Summary

| Metric | Result |
|---|---:|
| Measured exact-shape rows | **20/20 passed** |
| Main classic families with at least one numeric row | **12/17** |
| TIRx faster by more than 5% | **2/20** |
| Within 5% | **15/20** |
| Reference faster by more than 5% | **3/20** |
| Geometric-mean TIRx speedup | **1.001x** |
| Rows with either CV above 10% | **5/20** |

The mixed geomean is descriptive only: it gives one vote to each selected workload, not to each model invocation. The per-row numbers are the result to use.
The source-to-benchmark mapping for every row is audited in [THOR_SOURCE_BENCHMARK_AUDIT.md](THOR_SOURCE_BENCHMARK_AUDIT.md).

## Complete numeric table

| Family | Kernel / config | TIRx µs | CV | Reference | Reference µs | CV | Speedup |
|---|---|---:|---:|---|---:|---:|---:|
| Dense GEMM | `fp16_bf16_gemm/bf16_4096x4096x4096` | 1008.219 | 33.5% | `torch-cublas` | 1015.492 | 14.3% | **1.007x** |
| Dense GEMM | `fp16_bf16_gemm/fp16_4096x4096x4096` | 1480.844 | 29.1% | `torch-cublas` | 1490.247 | 15.6% | **1.006x** |
| Quantized GEMM | `nvfp4_gemm/4096x4096x4096` | 287.421 | 6.2% | `flashinfer` | 290.775 | 0.8% | **1.012x** |
| Attention | `flash_attention4/s4096_h32kv4_causal` | 985.467 | 3.9% | `flashattn_fa4_cutedsl` | 934.073 | 2.7% | **0.948x** |
| Normalization | `flashinfer_fused_add_rmsnorm/fused_bf16_m32_h4096_xc_rc_pdl1` | 11.358 | 5.5% | `flashinfer_cutedsl` | 10.671 | 1.9% | **0.940x** |
| Normalization | `flashinfer_fused_dit_layernorm/grgb_bf16_b1_r1920` | 265.420 | 8.9% | `flashinfer_cuda` | 265.092 | 8.9% | **0.999x** |
| Normalization | `flashinfer_layernorm/bf16_m128_h16384_xc_yc_pdl0_eps1e6` | 85.094 | 8.6% | `flashinfer_cutedsl` | 83.778 | 8.5% | **0.985x** |
| Normalization | `flashinfer_qk_rmsnorm/rms_bf16_b32_n32_h128_xc_yc_pdl0` | 5.496 | 2.6% | `flashinfer_cutedsl` | 5.417 | 2.9% | **0.986x** |
| Normalization | `flashinfer_rmsnorm/rms_bf16_m32_h4096_xc_yc_pdl1` | 9.743 | 3.5% | `flashinfer_cutedsl` | 9.662 | 3.2% | **0.992x** |
| Normalization | `flashinfer_rmsnorm_quant/bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | 18.370 | 9.7% | `flashinfer_cutedsl` | 18.014 | 9.0% | **0.981x** |
| Activation / quantization | `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 2346.760 | 7.6% | `flashinfer` | 2289.409 | 1.0% | **0.976x** |
| Activation / quantization | `mxfp4_quantize/fp16_linear_m4096_k4096` | 259.440 | 22.0% | `flashinfer` | 271.856 | 18.8% | **1.048x** |
| Activation / quantization | `nvfp4_quantize/fp16_linear_m4096_k4096` | 261.147 | 12.2% | `flashinfer` | 262.187 | 12.6% | **1.004x** |
| TopK | `fast_topk_clusters/f32_plain_b64_l16384_k256` | 194.392 | 0.5% | `flashinfer` | 194.424 | 0.7% | **1.000x** |
| TopK | `filtered_topk/f32_plain_r64_l8192_k256` | 43.005 | 3.5% | `flashinfer` | 48.077 | 1.8% | **1.118x** |
| TopK | `radix_topk_multi_cta/f32_basic_r4_l115188_k256_ctas3` | 59.658 | 6.1% | `flashinfer` | 59.924 | 3.0% | **1.004x** |
| TopK | `radix_topk_single_cta/f32_basic_r64_l32768_k512` | 165.448 | 4.7% | `flashinfer` | 173.161 | 3.9% | **1.047x** |
| Recurrent / SSM | `gdn_decode_bf16_ilp4/t4_b4_h8_hv16_tv16` | 76.087 | 6.9% | `flashinfer_cutedsl` | 76.039 | 8.2% | **0.999x** |
| Recurrent / SSM | `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 479.960 | 10.7% | `flashinfer_cutedsl` | 437.735 | 10.7% | **0.912x** |
| Recurrent / SSM | `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 3528.442 | 0.0% | `flashinfer_cuda` | 3830.180 | 0.0% | **1.086x** |

The NVFP4 row also measured cuBLASLt at **276.512 µs** (CV 3.4%), or **0.962x** relative to TIRx. FlashInfer is retained as that row's primary baseline to follow the requested priority.

The attention row uses upstream FA4 CuTeDSL as its primary baseline. On the same row, FlashInfer CuTeDSL measured **977.107 µs** and the legacy FlashInfer FA2 control measured **2940.869 µs**; neither secondary control enters the geomean.

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
| Rounds / aggregation | 15 / arithmetic mean |
| Warmup / repeat | 1000 ms / 100 ms per implementation per round |
| TVM/TIR revision | `15b607d6` |
| TIRx-kernels revision | `65228c58-dirty` |
| FlashInfer version / revision | `0.6.18` / `f2e04400` |
| FlashAttention-4 revision | `0251105a` |

Rows above 10% CV are retained and visibly flagged by their CV columns. In particular, the 4096-cube GEMMs switched between fast and slow clock regimes within their 15 rounds. Their absolute means and near-threshold ratios should not be treated as locked-clock absolute latency. The paired same-process ratios remain the comparison used here.

## Raw evidence

- Original representative run: `/home/tlopexh/thor-validation/final-tuned-representative-15r/runs/1.json`
- Classic additions run: `/home/tlopexh/thor-validation/final-tuned-additions-15r/runs/1.json`
- Both runs used 15 round samples per implementation and had no interference retries in the selected final artifacts.
