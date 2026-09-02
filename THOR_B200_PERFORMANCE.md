# NVIDIA Thor versus B200 TIRx performance

Measured on 2026-09-02 using the default representative workload roster.

## Result

- Thor completed **69/69** workloads with **0 interference retries**.
- **56** rows have a usable TIR/TIRx timing in the repository's historical SM100/B200 baseline; **1** exact baseline row failed, and **12** new BSA rows are absent there.
- Across the 56 matched rows, geometric-mean Thor/B200 latency is **13.250x**; equivalently, Thor delivers **7.5%** of B200's throughput on this workload mix.
- The median latency ratio is **17.546x** and the observed range is **1.377x--46.682x**.
- Thor has 20 SMs versus 148 on B200 (7.4x as many). The aggregate per-SM-normalized throughput is about **55.8%** of the B200 baseline, but this is only a rough diagnostic because the table mixes compute-, bandwidth-, and latency-bound kernels.

The geometric mean is a descriptive summary, not a model-level score. The detailed rows below are the authoritative data.

## Measurement provenance

| Field | Thor | Repository SM100/B200 baseline |
|---|---|---|
| GPU | NVIDIA Jetson AGX Thor Developer Kit | B200 attribution inferred from `sm_100a` and the suite's 148-SM B200 annotations; the JSON does not store a product name |
| SM count | 20 | 148 |
| CUDA architecture | `sm_110a` | `sm_100a` |
| Power/clock state | `MAXN`; dynamic clocks (`jetson_clocks` was not locked) | Not recorded |
| Timer | `proton` | `proton` |
| Rounds | 5, arithmetic mean | 5, arithmetic mean |
| Warmup / repeat budget | 25 ms / 100 ms | 25 ms / 100 ms |
| TVM/TIR revision | `15b607d6` | `73e38d3f` |
| TIRx-kernels revision | `6c25cb07` | `f727de3d-dirty` |
| CUDA/PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 | CUDA 13.2 / PyTorch 2.13.0+cu132 |

Thor's Triton 3.5.1 package bundles CUDA 12.8 CUPTI, which cannot initialize against this CUDA 13.1 Thor stack. The run therefore set `TRITON_CUPTI_LIB_PATH=/usr/local/cuda-13.1/extras/CUPTI/lib64`; both final columns still use the same Proton timer protocol.

This is a historical cross-machine comparison, not a controlled hardware-only A/B: the TVM, TIRx, CUDA, PyTorch, and CUPTI revisions differ, and the repository baseline labels its TIRx checkout as dirty. A publication-grade comparison requires rerunning this exact TIRx/TVM revision on a B200.

## Cross-session stability

Against the preceding same-protocol Thor run, 57 common rows have geometric-mean current/prior latency **1.009x** and median **0.994x**. The largest individual shift is `nvfp4_gemm/16384x16384x16384`: 27373.257 µs to 71375.402 µs (2.607x).

The aggregate is repeatable, but individual absolute times can move substantially between sessions under dynamic clocks. The complete table uses only the final, single-piece 69-row run; no samples were spliced from the earlier run.

## Kernel summary

`Thor/B200 latency > 1` means Thor is slower. Relative throughput is its reciprocal.

| Kernel | Thor rows | Matched B200 rows | Geomean Thor/B200 latency | Thor relative throughput |
|---|---:|---:|---:|---:|
| `act_and_mul` | 3 | 3 | 9.070x | 11.0% |
| `cudnn_sm100_bsa_backward_blk128` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_backward_blk64` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_forward_blk128` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_forward_blk64` | 3 | 0 | — | — |
| `flash_attention_backward_sm100` | 3 | 3 | 12.006x | 8.3% |
| `fp16_bf16_gemm` | 3 | 3 | 21.991x | 4.5% |
| `gdn_decode_bf16_ilp4` | 3 | 3 | 6.052x | 16.5% |
| `gdn_decode_bf16_wide_vec_mtp` | 3 | 3 | 21.375x | 4.7% |
| `gdn_decode_bf16_wide_vec_t1` | 3 | 3 | 23.432x | 4.3% |
| `gdn_decode_fp32_mtp_warp` | 3 | 2 | 31.857x | 3.1% |
| `mxfp4_quantize` | 3 | 3 | 15.768x | 6.3% |
| `mxfp8_quantize` | 3 | 3 | 12.824x | 7.8% |
| `nvfp4_gemm` | 3 | 3 | 18.982x | 5.3% |
| `nvfp4_quantize` | 3 | 3 | 11.722x | 8.5% |
| `nvfp4_quantize_per_token` | 3 | 3 | 10.453x | 9.6% |
| `recurrent_kda_decode_grouped` | 3 | 3 | 11.686x | 8.6% |
| `recurrent_kda_decode_one_warp` | 3 | 3 | 18.887x | 5.3% |
| `rmsnorm` | 3 | 3 | 4.651x | 21.5% |
| `silu_and_mul_nvfp4_experts_quantize` | 3 | 3 | 10.985x | 9.1% |
| `sparse_flashmla_prefill_head128_phase1` | 3 | 3 | 14.051x | 7.1% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | 3 | 3 | 12.739x | 7.8% |
| `sparse_flashmla_prefill_head64_phase1` | 3 | 3 | 12.818x | 7.8% |

## GEMM effective throughput

Throughput uses the conventional `2*M*N*K` operation count. NVFP4 values are effective throughput and include neither scale-processing operations nor any sparsity multiplier.

| Kernel | Config | Thor µs | B200 µs | Thor effective TFLOP/s | B200 effective TFLOP/s | Thor/B200 throughput |
|---|---|---:|---:|---:|---:|---:|
| `fp16_bf16_gemm` | `bf16_4096x4096x4096` | 1779.692 | 92.495 | 77.226 | 1485.903 | 5.2% |
| `fp16_bf16_gemm` | `fp16_1024x1024x1024` | 81.309 | 6.626 | 26.411 | 324.114 | 8.1% |
| `fp16_bf16_gemm` | `fp16_16384x16384x16384` | 256838.878 | 5702.039 | 34.248 | 1542.622 | 2.2% |
| `nvfp4_gemm` | `1024x1024x1024` | 54.912 | 5.211 | 39.108 | 412.138 | 9.5% |
| `nvfp4_gemm` | `16384x16384x16384` | 71375.402 | 1528.963 | 123.237 | 5752.981 | 2.1% |
| `nvfp4_gemm` | `4096x4096x4096` | 407.381 | 29.305 | 337.372 | 4690.013 | 7.2% |

The FP16 16384³ row reaches only 34.25 effective TFLOP/s on Thor, below the 4096³ BF16 row's 77.23 TFLOP/s. That inversion is a concrete tuning target: the current B200-oriented schedule does not scale well to Thor's 20-SM device at that shape.

## Complete workload table

Thor CV is the population coefficient of variation across five round means. **8** rows exceed 10% and are marked `†`; repeat those rows under locked clocks before using small differences for tuning decisions.

| Kernel | Config | Thor Proton µs | Thor CV | B200 Proton µs | Thor/B200 latency | Thor relative throughput |
|---|---|---:|---:|---:|---:|---:|
| `act_and_mul` | `gelu_tanh_fp16_d11008_t8192` | 2372.865 | 1.7% | 97.637 | 24.303x | 4.1% |
| `act_and_mul` | `silu_bf16_d16384_t32768` | 13764.394 | 0.3% | 617.343 | 22.296x | 4.5% |
| `act_and_mul` | `silu_fp16_d4096_t1` | 2.860 | 4.7% | 2.077 | 1.377x | 72.6% |
| `cudnn_sm100_bsa_backward_blk128` | `p00_b1_h1_d64_sq128_skv4096_kv16` | 40.924 | 21.7% † | — | — | — |
| `cudnn_sm100_bsa_backward_blk128` | `p02_b2_h8_d128_sq4096_skv8191_kv32` | 3779.466 | 0.4% | — | — | — |
| `cudnn_sm100_bsa_backward_blk128` | `p11_b1_h2_d128_sq524288_skv8192_kv32_qb4096_g8` | 60119.944 | 3.0% | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p00_b1_h1_sq64_skv4096_kv16_nomask` | 106.846 | 10.5% † | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask` | 2241.032 | 1.8% | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p11_b1_h1_sq192000_skv8192_maxkv16_var_mask_auto1024_i64kv` | 4923.554 | 1.2% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p00_bf16_d64_mha` | 496.406 | 0.9% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p08_bf16_d128_mqa` | 537.508 | 2.2% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p13_fp16_d96_gqa` | 485.938 | 3.9% | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p00_b1_h1_sq64_skv4096_kv16_nomask_s1_static` | 14.010 | 4.7% | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask_s1_clc` | 431.573 | 10.9% † | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p09_b1_h8_sq2048_skv65536_kv512_nomask_s8_static` | 4320.318 | 3.1% | — | — | — |
| `flash_attention_backward_sm100` | `b1_s2048_h16_causal` | 926.246 | 0.7% | 82.141 | 11.276x | 8.9% |
| `flash_attention_backward_sm100` | `b1_s8192_h16_noncausal` | 13511.986 | 0.2% | 1081.811 | 12.490x | 8.0% |
| `flash_attention_backward_sm100` | `b4_s8192_h16_noncausal` | 53697.286 | 0.6% | 4370.638 | 12.286x | 8.1% |
| `fp16_bf16_gemm` | `bf16_4096x4096x4096` | 1779.692 | 2.1% | 92.495 | 19.241x | 5.2% |
| `fp16_bf16_gemm` | `fp16_1024x1024x1024` | 81.309 | 1.8% | 6.626 | 12.272x | 8.1% |
| `fp16_bf16_gemm` | `fp16_16384x16384x16384` | 256838.878 | 1.7% | 5702.039 | 45.043x | 2.2% |
| `gdn_decode_bf16_ilp4` | `t1_b1_h2_hv4_tv16` | 5.282 | 9.9% | 3.070 | 1.720x | 58.1% |
| `gdn_decode_bf16_ilp4` | `t4_b8_h4_hv8_tv16` | 77.721 | 4.9% | 6.663 | 11.664x | 8.6% |
| `gdn_decode_bf16_ilp4` | `t8_b4_h8_hv16_tv16` | 114.750 | 1.7% | 10.386 | 11.048x | 9.1% |
| `gdn_decode_bf16_wide_vec_mtp` | `t2_b4_h16_hv32_tv32` | 82.007 | 3.6% | 6.331 | 12.953x | 7.7% |
| `gdn_decode_bf16_wide_vec_mtp` | `t4_b64_h8_hv16_tv128` | 961.926 | 3.4% | 33.810 | 28.451x | 3.5% |
| `gdn_decode_bf16_wide_vec_mtp` | `t8_b512_h16_hv32_tv128` | 30840.244 | 1.6% | 1163.720 | 26.501x | 3.8% |
| `gdn_decode_bf16_wide_vec_t1` | `b128_h8_hv16_tv128` | 667.287 | 1.2% | 24.894 | 26.806x | 3.7% |
| `gdn_decode_bf16_wide_vec_t1` | `b16_h16_hv32_tv64` | 192.498 | 5.1% | 11.252 | 17.107x | 5.8% |
| `gdn_decode_bf16_wide_vec_t1` | `b512_h4_hv8_tv128` | 1294.730 | 0.9% | 46.151 | 28.054x | 3.6% |
| `gdn_decode_fp32_mtp_warp` | `t2_b4_h16_hv64_tv16_ilp2_sv0` | 378.820 | 1.0% | 14.341 | 26.415x | 3.8% |
| `gdn_decode_fp32_mtp_warp` | `t4_b64_h8_hv32_tv64_ilp4_sv1` | 4915.043 | 1.6% | `FAIL` | — | — |
| `gdn_decode_fp32_mtp_warp` | `t8_b256_h16_hv64_tv64_ilp4_sv1` | 68136.165 | 2.2% | 1773.392 | 38.421x | 2.6% |
| `mxfp4_quantize` | `fp16_128x4_m128_k1024` | 8.631 | 3.4% | 2.491 | 3.465x | 28.9% |
| `mxfp4_quantize` | `fp16_128x4_m16384_k7168` | 2028.738 | 0.5% | 52.500 | 38.643x | 2.6% |
| `mxfp4_quantize` | `fp16_linear_m4096_k4096` | 301.022 | 12.5% † | 10.282 | 29.276x | 3.4% |
| `mxfp8_quantize` | `fp16_128x4_m128_k1024` | 5.937 | 3.7% | 2.679 | 2.216x | 45.1% |
| `mxfp8_quantize` | `fp16_128x4_m16384_k7168` | 1996.871 | 1.8% | 60.634 | 32.933x | 3.0% |
| `mxfp8_quantize` | `fp16_linear_m4096_k4096` | 327.001 | 1.6% | 11.316 | 28.897x | 3.5% |
| `nvfp4_gemm` | `1024x1024x1024` | 54.912 | 36.5% † | 5.211 | 10.539x | 9.5% |
| `nvfp4_gemm` | `16384x16384x16384` | 71375.402 | 6.3% | 1528.963 | 46.682x | 2.1% |
| `nvfp4_gemm` | `4096x4096x4096` | 407.381 | 11.4% † | 29.305 | 13.902x | 7.2% |
| `nvfp4_quantize` | `fp16_128x4_m128_k1024` | 5.788 | 6.9% | 2.497 | 2.318x | 43.1% |
| `nvfp4_quantize` | `fp16_128x4_m16384_k7168` | 1601.960 | 1.3% | 55.273 | 28.983x | 3.5% |
| `nvfp4_quantize` | `fp16_linear_m4096_k4096` | 245.831 | 7.2% | 10.253 | 23.977x | 4.2% |
| `nvfp4_quantize_per_token` | `fp16_128x4_m128_k1024` | 5.825 | 2.9% | 2.689 | 2.166x | 46.2% |
| `nvfp4_quantize_per_token` | `fp16_128x4_m16384_k7168` | 1443.940 | 1.2% | 57.766 | 24.996x | 4.0% |
| `nvfp4_quantize_per_token` | `fp16_linear_m4096_k4096` | 255.476 | 5.0% | 12.110 | 21.096x | 4.7% |
| `recurrent_kda_decode_grouped` | `dec_hv16_b1` | 11.426 | 6.2% | 3.309 | 3.453x | 29.0% |
| `recurrent_kda_decode_grouped` | `ver_t8_hv12_b16` | 411.806 | 0.6% | 22.114 | 18.622x | 5.4% |
| `recurrent_kda_decode_grouped` | `ver_t8_hv16_b128` | 4001.059 | 1.5% | 161.221 | 24.817x | 4.0% |
| `recurrent_kda_decode_one_warp` | `hv12_b64_tr16_lb` | 275.969 | 4.1% | 12.643 | 21.828x | 4.6% |
| `recurrent_kda_decode_one_warp` | `hv16_b128_tr16_lb` | 632.899 | 2.1% | 26.661 | 23.739x | 4.2% |
| `recurrent_kda_decode_one_warp` | `hv16_b8_tr8_lb` | 68.230 | 7.0% | 5.247 | 13.003x | 7.7% |
| `rmsnorm` | `hs128_bs32` | 3.635 | 8.6% | 2.313 | 1.571x | 63.6% |
| `rmsnorm` | `hs4096_bs128` | 22.312 | 2.3% | 3.520 | 6.339x | 15.8% |
| `rmsnorm` | `hs8192_bs4113` | 725.002 | 2.3% | 71.758 | 10.103x | 9.9% |
| `silu_and_mul_nvfp4_experts_quantize` | `bf16_b8_m512_k2048` | 173.626 | 16.2% † | 9.048 | 19.189x | 5.2% |
| `silu_and_mul_nvfp4_experts_quantize` | `fp16_b128_m2048_k2048` | 5987.055 | 1.4% | 273.842 | 21.863x | 4.6% |
| `silu_and_mul_nvfp4_experts_quantize` | `fp16_b8_m16_k2048` | 10.555 | 4.2% | 3.341 | 3.159x | 31.7% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | 32607.851 | 8.5% | 1813.111 | 17.984x | 5.6% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | 18002.580 | 0.6% | 1676.518 | 10.738x | 9.3% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | 25497.001 | 9.6% | 1775.025 | 14.364x | 7.0% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | 13079.694 | 11.5% † | 1126.813 | 11.608x | 8.6% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | 21663.051 | 8.3% | 1196.729 | 18.102x | 5.5% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | 11725.676 | 0.7% | 1191.661 | 9.840x | 10.2% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk512_hq64_s4096_kv65536_topk512` | 8691.969 | 0.4% | 381.845 | 22.763x | 4.4% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk512_hq64_s4096_kv8192_topk512` | 4174.274 | 0.3% | 366.517 | 11.389x | 8.8% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk576_hq64_s4096_kv32768_topk512` | 7001.477 | 0.5% | 861.845 | 8.124x | 12.3% |

## Correctness scope

The performance roster contains only the 23 kernels already admitted for exact `sm_110a` runtime support after their complete correctness matrices passed: 485/485 configurations. The 69 timed rows are three representative performance shapes per kernel; they do not replace the complete numerical validation matrices.

## Raw evidence

- Thor run: `/home/tlopexh/thor-validation/bench-proton-23/runs/1.json`
- B200 baseline: `/home/tlopexh/TIRx-kernels/tirx_kernels/bench_suite/baseline.json`
- Prior Thor stability run: `/home/tlopexh/thor-validation/bench-proton/runs/1.json`
- Thor run status: 69 `ok`, 0 failures, 0 interference retries
- Usable B200 matches: 56 rows; failed B200 baseline rows: 1; absent new BSA rows: 12
