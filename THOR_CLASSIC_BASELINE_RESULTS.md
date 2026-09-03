# NVIDIA Thor classic-kernel same-device baselines

Measured on 2026-09-03 on one NVIDIA Jetson AGX Thor Developer Kit. The table combines the original representative campaign with the classic-family additions. Every speedup is `reference latency / TIRx latency`; values above 1.0 favor TIRx.

## Summary

| Metric | Result |
|---|---:|
| Measured exact-shape rows | **20/20 passed** |
| Main classic families with at least one numeric row | **12/17** |
| TIRx faster by more than 5% | **4/20** |
| Within 5% | **12/20** |
| Reference faster by more than 5% | **4/20** |
| Geometric-mean TIRx speedup | **1.050x** |
| Rows with either CV above 10% | **8/20** |

The mixed geomean is descriptive only: it gives one vote to each selected workload, not to each model invocation. The per-row numbers are the result to use.

## Complete numeric table

| Family | Kernel / config | TIRx µs | CV | Reference | Reference µs | CV | Speedup |
|---|---|---:|---:|---|---:|---:|---:|
| Dense GEMM | `fp16_bf16_gemm/bf16_4096x4096x4096` | 1176.022 | 27.3% | `torch-cublas` | 1231.584 | 0.8% | **1.047x** |
| Dense GEMM | `fp16_bf16_gemm/fp16_4096x4096x4096` | 1205.268 | 27.5% | `torch-cublas` | 1219.785 | 0.4% | **1.012x** |
| Quantized GEMM | `nvfp4_gemm/4096x4096x4096` | 384.989 | 7.8% | `flashinfer` | 353.804 | 7.0% | **0.919x** |
| Attention | `flash_attention4/s4096_h32kv4_causal` | 971.528 | 2.9% | `flashinfer_fa2` | 2943.756 | 0.1% | **3.030x** |
| Normalization | `flashinfer_fused_add_rmsnorm/fused_bf16_m32_h4096_xc_rc_pdl1` | 17.154 | 16.1% | `flashinfer_cutedsl` | 15.028 | 4.2% | **0.876x** |
| Normalization | `flashinfer_fused_dit_layernorm/grgb_bf16_b1_r1920` | 325.743 | 13.2% | `flashinfer_cuda` | 311.435 | 13.0% | **0.956x** |
| Normalization | `flashinfer_layernorm/bf16_m128_h16384_xc_yc_pdl0_eps1e6` | 76.161 | 4.6% | `flashinfer_cutedsl` | 76.394 | 6.0% | **1.003x** |
| Normalization | `flashinfer_qk_rmsnorm/rms_bf16_b32_n32_h128_xc_yc_pdl0` | 6.136 | 9.2% | `flashinfer_cutedsl` | 6.142 | 7.4% | **1.001x** |
| Normalization | `flashinfer_rmsnorm/rms_bf16_m32_h4096_xc_yc_pdl1` | 11.129 | 9.0% | `flashinfer_cutedsl` | 10.228 | 11.3% | **0.919x** |
| Normalization | `flashinfer_rmsnorm_quant/bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | 18.377 | 4.3% | `flashinfer_cutedsl` | 17.728 | 8.2% | **0.965x** |
| Activation / quantization | `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 2434.706 | 11.1% | `flashinfer` | 2262.402 | 0.6% | **0.929x** |
| Activation / quantization | `mxfp4_quantize/fp16_linear_m4096_k4096` | 308.695 | 8.8% | `flashinfer` | 328.513 | 8.5% | **1.064x** |
| Activation / quantization | `nvfp4_quantize/fp16_linear_m4096_k4096` | 235.974 | 9.3% | `flashinfer` | 227.148 | 8.8% | **0.963x** |
| TopK | `fast_topk_clusters/f32_plain_b64_l16384_k256` | 193.887 | 1.3% | `flashinfer` | 193.280 | 0.8% | **0.997x** |
| TopK | `filtered_topk/f32_plain_r64_l8192_k256` | 44.757 | 3.6% | `flashinfer` | 49.180 | 1.6% | **1.099x** |
| TopK | `radix_topk_multi_cta/f32_basic_r4_l115188_k256_ctas3` | 66.482 | 2.2% | `flashinfer` | 65.924 | 2.4% | **0.992x** |
| TopK | `radix_topk_single_cta/f32_basic_r64_l32768_k512` | 174.994 | 3.6% | `flashinfer` | 180.461 | 5.5% | **1.031x** |
| Recurrent / SSM | `gdn_decode_bf16_ilp4/t4_b4_h8_hv16_tv16` | 81.317 | 7.1% | `flashinfer_cutedsl` | 85.258 | 10.2% | **1.048x** |
| Recurrent / SSM | `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 398.642 | 9.9% | `flashinfer_cutedsl` | 393.236 | 10.9% | **0.986x** |
| Recurrent / SSM | `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 3526.438 | 0.0% | `flashinfer_cuda` | 3829.149 | 0.0% | **1.086x** |

The NVFP4 row also measured cuBLASLt at **349.897 µs** (CV 5.6%), or **0.909x** relative to TIRx. FlashInfer is retained as that row's primary baseline to follow the requested priority.

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
| TIRx-kernels revision | `bb69d704-dirty` |
| FlashInfer version / revision | `0.6.18` / `f2e04400` |

Rows above 10% CV are retained and visibly flagged by their CV columns. In particular, the two 4096-cube GEMMs saw a slow first TIRx round under dynamic Thor clocks. Their absolute means and near-threshold ratios should be rerun with `sudo jetson_clocks` before publication.

## Raw evidence

- Original representative run: `/home/tlopexh/thor-validation/flashinfer-native-final/runs/2.json`
- Classic additions run: `/home/tlopexh/thor-validation/classic-baseline-additions/runs/2.json`
- Both runs used five round samples per implementation and had no interference retries in the selected final artifacts.
