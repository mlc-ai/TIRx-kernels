# NVIDIA Thor versus B200 TIRx performance

Measured on 2026-09-02 using the default representative workload roster.

## Result

- Thor completed **57/57** workloads with **0 interference retries**.
- **45** rows have an exact `(kernel, config)` match in the repository's historical SM100/B200 baseline; **12** new BSA rows have no B200 value there.
- Across the 45 matched rows, geometric-mean Thor/B200 latency is **12.347x**; equivalently, Thor delivers **8.1%** of B200's throughput on this workload mix.
- The median latency ratio is **14.879x** and the observed range is **1.195x--50.644x**.
- Thor has 20 SMs versus 148 on B200 (7.4x as many). The aggregate per-SM-normalized throughput is about **59.9%** of the B200 baseline, but this is only a rough diagnostic because the table mixes compute-, bandwidth-, and latency-bound kernels.

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
| TIRx-kernels revision | `a8988b6e` | `f727de3d-dirty` |
| CUDA/PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 | CUDA 13.2 / PyTorch 2.13.0+cu132 |

Thor's Triton 3.5.1 package bundles CUDA 12.8 CUPTI, which cannot initialize against this CUDA 13.1 Thor stack. The run therefore set `TRITON_CUPTI_LIB_PATH=/usr/local/cuda-13.1/extras/CUPTI/lib64`; both final columns still use the same Proton timer protocol.

This is a historical cross-machine comparison, not a controlled hardware-only A/B: the TVM, TIRx, CUDA, PyTorch, and CUPTI revisions differ, and the repository baseline labels its TIRx checkout as dirty. A publication-grade comparison requires rerunning this exact TIRx/TVM revision on a B200.

## Kernel summary

`Thor/B200 latency > 1` means Thor is slower. Relative throughput is its reciprocal.

| Kernel | Thor rows | Matched B200 rows | Geomean Thor/B200 latency | Thor relative throughput |
|---|---:|---:|---:|---:|
| `act_and_mul` | 3 | 3 | 9.110x | 11.0% |
| `cudnn_sm100_bsa_backward_blk128` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_backward_blk64` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_forward_blk128` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_forward_blk64` | 3 | 0 | — | — |
| `flash_attention_backward_sm100` | 3 | 3 | 12.108x | 8.3% |
| `fp16_bf16_gemm` | 3 | 3 | 21.971x | 4.6% |
| `mxfp4_quantize` | 3 | 3 | 15.240x | 6.6% |
| `mxfp8_quantize` | 3 | 3 | 12.696x | 7.9% |
| `nvfp4_gemm` | 3 | 3 | 13.964x | 7.2% |
| `nvfp4_quantize` | 3 | 3 | 11.694x | 8.6% |
| `nvfp4_quantize_per_token` | 3 | 3 | 10.430x | 9.6% |
| `recurrent_kda_decode_grouped` | 3 | 3 | 13.549x | 7.4% |
| `recurrent_kda_decode_one_warp` | 3 | 3 | 19.415x | 5.2% |
| `rmsnorm` | 3 | 3 | 4.148x | 24.1% |
| `silu_and_mul_nvfp4_experts_quantize` | 3 | 3 | 11.188x | 8.9% |
| `sparse_flashmla_prefill_head128_phase1` | 3 | 3 | 14.368x | 7.0% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | 3 | 3 | 12.825x | 7.8% |
| `sparse_flashmla_prefill_head64_phase1` | 3 | 3 | 13.155x | 7.6% |

## GEMM effective throughput

Throughput uses the conventional `2*M*N*K` operation count. NVFP4 values are effective throughput and include neither scale-processing operations nor any sparsity multiplier.

| Kernel | Config | Thor µs | B200 µs | Thor effective TFLOP/s | B200 effective TFLOP/s | Thor/B200 throughput |
|---|---|---:|---:|---:|---:|---:|
| `fp16_bf16_gemm` | `bf16_4096x4096x4096` | 1279.050 | 92.495 | 107.454 | 1485.903 | 7.2% |
| `fp16_bf16_gemm` | `fp16_1024x1024x1024` | 100.348 | 6.626 | 21.400 | 324.114 | 6.6% |
| `fp16_bf16_gemm` | `fp16_16384x16384x16384` | 288775.659 | 5702.039 | 30.460 | 1542.622 | 2.0% |
| `nvfp4_gemm` | `1024x1024x1024` | 53.262 | 5.211 | 40.319 | 412.138 | 9.8% |
| `nvfp4_gemm` | `16384x16384x16384` | 27373.257 | 1528.963 | 321.339 | 5752.981 | 5.6% |
| `nvfp4_gemm` | `4096x4096x4096` | 436.011 | 29.305 | 315.219 | 4690.013 | 6.7% |

The FP16 16384³ row reaches only 30.46 effective TFLOP/s on Thor, below the 4096³ BF16 row's 107.45 TFLOP/s. That inversion is a concrete tuning target: the current B200-oriented schedule does not scale well to Thor's 20-SM device at that shape.

## Complete workload table

Thor CV is the population coefficient of variation across five round means. **8** rows exceed 10% and are marked `†`; repeat those rows under locked clocks before using small differences for tuning decisions.

| Kernel | Config | Thor Proton µs | Thor CV | B200 Proton µs | Thor/B200 latency | Thor relative throughput |
|---|---|---:|---:|---:|---:|---:|
| `act_and_mul` | `gelu_tanh_fp16_d11008_t8192` | 2352.477 | 1.0% | 97.637 | 24.094x | 4.2% |
| `act_and_mul` | `silu_bf16_d16384_t32768` | 13828.223 | 0.4% | 617.343 | 22.400x | 4.5% |
| `act_and_mul` | `silu_fp16_d4096_t1` | 2.910 | 4.8% | 2.077 | 1.401x | 71.4% |
| `cudnn_sm100_bsa_backward_blk128` | `p00_b1_h1_d64_sq128_skv4096_kv16` | 44.956 | 24.3% † | — | — | — |
| `cudnn_sm100_bsa_backward_blk128` | `p02_b2_h8_d128_sq4096_skv8191_kv32` | 3793.977 | 1.4% | — | — | — |
| `cudnn_sm100_bsa_backward_blk128` | `p11_b1_h2_d128_sq524288_skv8192_kv32_qb4096_g8` | 59959.875 | 2.9% | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p00_b1_h1_sq64_skv4096_kv16_nomask` | 106.033 | 18.6% † | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask` | 2304.815 | 2.8% | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p11_b1_h1_sq192000_skv8192_maxkv16_var_mask_auto1024_i64kv` | 4967.877 | 1.3% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p00_bf16_d64_mha` | 519.477 | 1.3% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p08_bf16_d128_mqa` | 530.871 | 2.8% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p13_fp16_d96_gqa` | 483.637 | 5.2% | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p00_b1_h1_sq64_skv4096_kv16_nomask_s1_static` | 13.240 | 3.2% | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask_s1_clc` | 441.000 | 15.9% † | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p09_b1_h8_sq2048_skv65536_kv512_nomask_s8_static` | 4345.198 | 1.5% | — | — | — |
| `flash_attention_backward_sm100` | `b1_s2048_h16_causal` | 929.403 | 1.1% | 82.141 | 11.315x | 8.8% |
| `flash_attention_backward_sm100` | `b1_s8192_h16_noncausal` | 13556.478 | 0.3% | 1081.811 | 12.531x | 8.0% |
| `flash_attention_backward_sm100` | `b4_s8192_h16_noncausal` | 54710.451 | 0.2% | 4370.638 | 12.518x | 8.0% |
| `fp16_bf16_gemm` | `bf16_4096x4096x4096` | 1279.050 | 28.3% † | 92.495 | 13.828x | 7.2% |
| `fp16_bf16_gemm` | `fp16_1024x1024x1024` | 100.348 | 6.4% | 6.626 | 15.145x | 6.6% |
| `fp16_bf16_gemm` | `fp16_16384x16384x16384` | 288775.659 | 1.5% | 5702.039 | 50.644x | 2.0% |
| `mxfp4_quantize` | `fp16_128x4_m128_k1024` | 8.609 | 8.2% | 2.491 | 3.456x | 28.9% |
| `mxfp4_quantize` | `fp16_128x4_m16384_k7168` | 2033.929 | 3.2% | 52.500 | 38.741x | 2.6% |
| `mxfp4_quantize` | `fp16_linear_m4096_k4096` | 271.769 | 2.5% | 10.282 | 26.431x | 3.8% |
| `mxfp8_quantize` | `fp16_128x4_m128_k1024` | 6.133 | 8.6% | 2.679 | 2.289x | 43.7% |
| `mxfp8_quantize` | `fp16_128x4_m16384_k7168` | 2045.225 | 2.0% | 60.634 | 33.731x | 3.0% |
| `mxfp8_quantize` | `fp16_linear_m4096_k4096` | 299.903 | 3.6% | 11.316 | 26.503x | 3.8% |
| `nvfp4_gemm` | `1024x1024x1024` | 53.262 | 20.6% † | 5.211 | 10.222x | 9.8% |
| `nvfp4_gemm` | `16384x16384x16384` | 27373.257 | 1.2% | 1528.963 | 17.903x | 5.6% |
| `nvfp4_gemm` | `4096x4096x4096` | 436.011 | 8.2% | 29.305 | 14.879x | 6.7% |
| `nvfp4_quantize` | `fp16_128x4_m128_k1024` | 5.577 | 1.5% | 2.497 | 2.233x | 44.8% |
| `nvfp4_quantize` | `fp16_128x4_m16384_k7168` | 1643.709 | 2.3% | 55.273 | 29.738x | 3.4% |
| `nvfp4_quantize` | `fp16_linear_m4096_k4096` | 246.852 | 8.6% | 10.253 | 24.077x | 4.2% |
| `nvfp4_quantize_per_token` | `fp16_128x4_m128_k1024` | 6.110 | 8.2% | 2.689 | 2.272x | 44.0% |
| `nvfp4_quantize_per_token` | `fp16_128x4_m16384_k7168` | 1484.013 | 3.5% | 57.766 | 25.690x | 3.9% |
| `nvfp4_quantize_per_token` | `fp16_linear_m4096_k4096` | 235.436 | 7.1% | 12.110 | 19.441x | 5.1% |
| `recurrent_kda_decode_grouped` | `dec_hv16_b1` | 14.446 | 13.8% † | 3.309 | 4.366x | 22.9% |
| `recurrent_kda_decode_grouped` | `ver_t8_hv12_b16` | 465.209 | 6.7% | 22.114 | 21.036x | 4.8% |
| `recurrent_kda_decode_grouped` | `ver_t8_hv16_b128` | 4366.166 | 1.5% | 161.221 | 27.082x | 3.7% |
| `recurrent_kda_decode_one_warp` | `hv12_b64_tr16_lb` | 273.899 | 4.1% | 12.643 | 21.664x | 4.6% |
| `recurrent_kda_decode_one_warp` | `hv16_b128_tr16_lb` | 652.198 | 0.7% | 26.661 | 24.463x | 4.1% |
| `recurrent_kda_decode_one_warp` | `hv16_b8_tr8_lb` | 72.463 | 5.1% | 5.247 | 13.809x | 7.2% |
| `rmsnorm` | `hs128_bs32` | 2.764 | 3.5% | 2.313 | 1.195x | 83.7% |
| `rmsnorm` | `hs4096_bs128` | 21.150 | 4.7% | 3.520 | 6.009x | 16.6% |
| `rmsnorm` | `hs8192_bs4113` | 713.113 | 1.9% | 71.758 | 9.938x | 10.1% |
| `silu_and_mul_nvfp4_experts_quantize` | `bf16_b8_m512_k2048` | 185.063 | 18.8% † | 9.048 | 20.453x | 4.9% |
| `silu_and_mul_nvfp4_experts_quantize` | `fp16_b128_m2048_k2048` | 6073.372 | 2.8% | 273.842 | 22.178x | 4.5% |
| `silu_and_mul_nvfp4_experts_quantize` | `fp16_b8_m16_k2048` | 10.313 | 2.8% | 3.341 | 3.087x | 32.4% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | 34438.195 | 2.4% | 1813.111 | 18.994x | 5.3% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | 18105.164 | 1.1% | 1676.518 | 10.799x | 9.3% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | 25668.868 | 10.9% † | 1775.025 | 14.461x | 6.9% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | 12389.150 | 1.8% | 1126.813 | 10.995x | 9.1% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | 23127.459 | 7.5% | 1196.729 | 19.326x | 5.2% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | 11831.779 | 0.8% | 1191.661 | 9.929x | 10.1% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk512_hq64_s4096_kv65536_topk512` | 9216.621 | 6.9% | 381.845 | 24.137x | 4.1% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk512_hq64_s4096_kv8192_topk512` | 4086.647 | 0.3% | 366.517 | 11.150x | 9.0% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk576_hq64_s4096_kv32768_topk512` | 7290.432 | 0.4% | 861.845 | 8.459x | 11.8% |

## Correctness scope

The performance roster contains only the 19 kernels already admitted for exact `sm_110a` runtime support after their complete correctness matrices passed: 316/316 configurations. The 57 timed rows are three representative performance shapes per kernel; they do not replace the complete numerical validation matrices.

## Raw evidence

- Thor run: `/home/tlopexh/thor-validation/bench-proton/runs/1.json`
- B200 baseline: `/home/tlopexh/TIRx-kernels/tirx_kernels/bench_suite/baseline.json`
- Thor run status: 57 `ok`, 0 failures, 0 interference retries
- B200 matches: 45 rows; new BSA without historical B200 rows: 12
