# bench-suite baseline view: `baseline.json`

- Timestamp: `14`
- Label:     `post-refactor`
- Git:       `{'tir': 'ea0950ab', 'tirx-kernels': 'b3be0123', 'tirx-bench-ci': None}`
- Workloads: 258 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## act_and_mul

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gelu_fp16_d11008_t8192` | tirx | 154.2156 | flashinfer | 253.8506 | 1.646 | — |
| `gelu_tanh_fp16_d11008_t8192` | tirx | 97.4212 | flashinfer | 109.3614 | 1.123 | — |
| `silu_bf16_d16384_t32768` | tirx | 615.7810 | flashinfer | 686.5466 | 1.115 | — |
| `silu_bf16_d4096_t8192` | tirx | 33.2499 | flashinfer | 36.5619 | 1.100 | — |
| `silu_fp16_d11008_t8192` | tirx | 97.6846 | flashinfer | 107.9017 | 1.105 | — |
| `silu_fp16_d16384_t32768` | tirx | 453.6279 | flashinfer | 470.2372 | 1.037 | — |
| `silu_fp16_d4096_t1` | tirx | 2.5687 | flashinfer | 2.7795 | 1.082 | — |
| `silu_fp16_d4096_t8192` | tirx | 41.3473 | flashinfer | 46.5259 | 1.125 | — |

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 21469.3738 | cublas_nccl_cudagraph | 21463.2041 | 1.000 | cublasmp_split_p2p=21790.5044 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1066.3879 | cublas_nccl_cudagraph | 1034.9734 | 0.971 | cublasmp_split_p2p=1079.9175 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2876.2938 | cublas_nccl_cudagraph | 2717.1455 | 0.945 | cublasmp_split_p2p=2772.5312 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 6103.6119 | cublas_nccl_cudagraph | 5984.3328 | 0.980 | cublasmp_split_p2p=6148.8383 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1298.6000 | deepgemm | 1288.0000 | 0.992 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3377.4000 | deepgemm | 3377.6000 | 1.000 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 61.4546 | deepgemm | 63.9326 | 1.040 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 184.3678 | deepgemm | 182.3390 | 0.989 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.2063 | deepgemm | 6.5250 | 1.051 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 6.2833 | deepgemm | 7.2835 | 1.159 | — |

## deepgemm_sm100_fp8_bmm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bhd_bhr_hdr_b4096_h8_r4096_d1024` | tirx | 176.1994 | deepgemm | 188.5498 | 1.070 | — |
| `bhd_hdr_bhr_b8192_h8_r4096_d1024` | tirx | 227.1852 | deepgemm | 253.6922 | 1.117 | — |
| `bhr_hdr_bhd_b4096_h8_r4096_d1024` | tirx | 138.1472 | deepgemm | 139.6637 | 1.011 | — |
| `bhr_hdr_bhd_b8192_h8_r4096_d1024` | tirx | 197.2879 | deepgemm | 196.8100 | 0.998 | — |

## deepgemm_sm100_fp8_gemm_1d1d

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m4096_n2112_k7168` | tirx | 49.7282 | deepgemm | 49.7913 | 1.001 | — |
| `m4096_n24576_k1536` | tirx | 148.6603 | deepgemm | 148.3340 | 0.998 | — |
| `m4096_n24576_k1536_bfp4` | tirx | 150.7022 | deepgemm | 149.7895 | 0.994 | — |
| `m4096_n32768_k512` | tirx | 70.6411 | deepgemm | 72.2801 | 1.023 | — |
| `m4096_n4096_k7168` | tirx | 123.2633 | deepgemm | 124.5363 | 1.010 | — |
| `m4096_n4096_k7168_bfp4` | tirx | 81.0394 | deepgemm | 77.0748 | 0.951 | — |
| `m4096_n576_k7168` | tirx | 18.6026 | deepgemm | 18.9120 | 1.017 | — |
| `m4096_n7168_k16384` | tirx | 325.3169 | deepgemm | 323.9807 | 0.996 | — |
| `m4096_n7168_k16384_bfp4` | tirx | 300.5845 | deepgemm | 297.4391 | 0.990 | — |
| `m4096_n7168_k2048` | tirx | 42.2952 | deepgemm | 42.3046 | 1.000 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 66.4714 | deepgemm | 68.1229 | 1.025 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 181.8032 | deepgemm | 191.9038 | 1.056 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.7193 | deepgemm | 6.9450 | 1.034 | sglang_cutedsl=7.0011 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5840 | sglang_cutedsl | 4.7747 | 1.042 | deepgemm=5.3156 |

## deepgemm_sm100_k_grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g16_m7168_n2048_k2048_gran128_al128` | tirx | 579.1164 | deepgemm | 581.8843 | 1.005 | — |
| `g4_m4096_n7168_k8192_gran128_al128` | tirx | 1013.3178 | deepgemm | 1009.7449 | 0.996 | — |
| `g4_m4096_n7168_k8192_gran128_al128_psum` | tirx | 859.9106 | deepgemm | 876.6517 | 1.019 | — |
| `g4_m4096_n7168_k8192_gran32_al32` | tirx | 834.9966 | deepgemm | 868.6466 | 1.040 | — |
| `g8_m4096_n7168_k4096_gran128_al128` | tirx | 876.7165 | deepgemm | 872.7994 | 0.996 | — |
| `g8_m4096_n7168_k4096_gran32_al160` | tirx | 1126.3837 | deepgemm | 1181.2470 | 1.049 | — |

## deepgemm_sm100_m_grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g4_m8192_n4096_k2048` | tirx | 205.5935 | deepgemm | 207.5108 | 1.009 | — |
| `g4_m8192_n4096_k4096` | tirx | 363.9075 | deepgemm | 367.8614 | 1.011 | — |
| `g4_m8192_n4096_k4096_psum` | tirx | 452.7249 | deepgemm | 452.4756 | 0.999 | — |
| `g4_m8192_n6144_k7168` | tirx | 1001.1147 | deepgemm | 983.4413 | 0.982 | — |
| `g4_m8192_n6144_k7168_bfp4` | tirx | 977.0830 | deepgemm | 963.9919 | 0.987 | — |
| `g4_m8192_n7168_k3072` | tirx | 580.0537 | deepgemm | 582.0827 | 1.003 | — |
| `g8_m4096_n4096_k2048` | tirx | 194.3954 | deepgemm | 197.0576 | 1.014 | — |
| `g8_m4096_n4096_k2048_bfp4` | tirx | 232.1981 | deepgemm | 234.1759 | 1.009 | — |
| `g8_m4096_n4096_k4096` | tirx | 361.9380 | deepgemm | 369.1489 | 1.020 | — |
| `g8_m4096_n6144_k7168` | tirx | 1109.6160 | deepgemm | 1105.3470 | 0.996 | — |
| `g8_m4096_n7168_k3072` | tirx | 594.7954 | deepgemm | 594.0340 | 0.999 | — |
| `g8_m4096_n7168_k3072_psum_zp` | tirx | 601.4369 | deepgemm | 603.6703 | 1.004 | — |

## deepgemm_sm100_m_grouped_fp8_gemm_masked

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g32_m192_n4096_k2048` | tirx | 72.7861 | deepgemm | 72.9036 | 1.002 | — |
| `g32_m192_n4096_k4096_bfp4` | tirx | 114.2318 | deepgemm | 113.9582 | 0.998 | — |
| `g32_m192_n6144_k7168` | tirx | 336.2360 | deepgemm | 329.7036 | 0.981 | — |
| `g6_m1024_n4096_k2048` | tirx | 60.6740 | deepgemm | 60.5707 | 0.998 | — |
| `g6_m1024_n6144_k7168` | tirx | 200.7467 | deepgemm | 200.6686 | 1.000 | — |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.3073 | deepgemm | 5.2196 | 0.983 | — |
| `m137_n24_k7680_s16` | tirx | 5.3061 | deepgemm | 5.3535 | 1.009 | — |
| `m13_n24_k7168_s1` | tirx | 33.8115 | deepgemm | 33.4763 | 0.990 | — |
| `m4096_n24_k28672_s16` | tirx | 57.8921 | deepgemm | 62.8364 | 1.085 | — |
| `m4096_n24_k7168_s1` | tirx | 36.0125 | deepgemm | 35.7021 | 0.991 | — |
| `m64_n24_k28672_s112` | tirx | 7.2039 | deepgemm | 7.2959 | 1.013 | — |
| `m8192_n24_k16384_s1` | tirx | 50.5342 | deepgemm | 60.5773 | 1.199 | — |
| `m8192_n24_k28672_s1` | tirx | 83.9610 | deepgemm | 91.6330 | 1.091 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.6227 | flashattn_sm100 | 20.1902 | 1.029 | — |
| `s1024_h32kv16_causal` | tir | 20.1335 | flashattn_sm100 | 20.3510 | 1.011 | — |
| `s1024_h32kv32` | tir | 32.1967 | flashattn_sm100 | 33.0879 | 1.028 | — |
| `s1024_h32kv32_causal` | tir | 21.1454 | flashattn_sm100 | 21.5064 | 1.017 | — |
| `s1024_h32kv4` | tir | 19.1192 | flashattn_sm100 | 19.9600 | 1.044 | — |
| `s1024_h32kv4_causal` | tir | 18.7431 | flashattn_sm100 | 19.7290 | 1.053 | — |
| `s1024_h32kv8` | tir | 19.5072 | flashattn_sm100 | 20.0500 | 1.028 | — |
| `s1024_h32kv8_causal` | tir | 19.1711 | flashattn_sm100 | 19.7744 | 1.031 | — |
| `s2048_h32kv16` | tir | 57.6003 | flashattn_sm100 | 57.7701 | 1.003 | — |
| `s2048_h32kv16_causal` | tir | 36.3420 | flashattn_sm100 | 38.5848 | 1.062 | — |
| `s2048_h32kv32` | tir | 59.8262 | flashattn_sm100 | 60.0877 | 1.004 | — |
| `s2048_h32kv32_causal` | tir | 64.4690 | flashattn_sm100 | 64.6407 | 1.003 | — |
| `s2048_h32kv4` | tir | 94.3612 | flashattn_sm100 | 94.9303 | 1.006 | — |
| `s2048_h32kv4_causal` | tir | 35.2785 | flashattn_sm100 | 37.7566 | 1.070 | — |
| `s2048_h32kv8` | tir | 56.1241 | flashattn_sm100 | 56.6827 | 1.010 | — |
| `s2048_h32kv8_causal` | tir | 35.4594 | flashattn_sm100 | 37.8575 | 1.068 | — |
| `s4096_h32kv16` | tir | 211.9596 | flashattn_sm100 | 213.8201 | 1.009 | — |
| `s4096_h32kv16_causal` | tir | 113.2938 | flashattn_sm100 | 118.4429 | 1.045 | — |
| `s4096_h32kv32` | tir | 218.1091 | flashattn_sm100 | 220.0999 | 1.009 | — |
| `s4096_h32kv32_causal` | tir | 121.4913 | flashattn_sm100 | 120.1097 | 0.989 | — |
| `s4096_h32kv4` | tir | 208.2616 | flashattn_sm100 | 210.2081 | 1.009 | — |
| `s4096_h32kv4_causal` | tir | 174.5619 | flashattn_sm100 | 182.6275 | 1.046 | — |
| `s4096_h32kv8` | tir | 207.4134 | flashattn_sm100 | 209.5260 | 1.010 | — |
| `s4096_h32kv8_causal` | tir | 112.0452 | flashattn_sm100 | 117.5551 | 1.049 | — |
| `s8192_h32kv16` | tir | 780.7842 | flashattn_sm100 | 779.7321 | 0.999 | — |
| `s8192_h32kv16_causal` | tir | 422.0594 | flashattn_sm100 | 428.6703 | 1.016 | — |
| `s8192_h32kv32` | tir | 796.6754 | flashattn_sm100 | 800.0535 | 1.004 | — |
| `s8192_h32kv32_causal` | tir | 440.1180 | flashattn_sm100 | 436.9789 | 0.993 | — |
| `s8192_h32kv4` | tir | 984.6224 | flashattn_sm100 | 989.6464 | 1.005 | — |
| `s8192_h32kv4_causal` | tir | 564.6177 | flashattn_sm100 | 582.7476 | 1.032 | — |
| `s8192_h32kv8` | tir | 766.1723 | flashattn_sm100 | 773.2631 | 1.009 | — |
| `s8192_h32kv8_causal` | tir | 415.7060 | flashattn_sm100 | 425.8563 | 1.024 | — |

## flash_attention_backward_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_s2048_h16_causal` | tir | 81.8951 | flashattn_sm100 | 81.9815 | 1.001 | — |
| `b1_s4096_h16_causal` | tir | 323.1004 | flashattn_sm100 | 328.2527 | 1.016 | — |
| `b1_s8192_h16_causal` | tir | 931.6856 | flashattn_sm100 | 935.9016 | 1.005 | — |
| `b1_s8192_h16_noncausal` | tir | 1100.8997 | flashattn_sm100 | 1117.1258 | 1.015 | — |
| `b2_s4096_h16_causal` | tir | 388.0339 | flashattn_sm100 | 390.3185 | 1.006 | — |
| `b4_s8192_h16_causal` | tir | 2473.1056 | flashattn_sm100 | 2496.3857 | 1.009 | — |
| `b4_s8192_h16_noncausal` | tir | 5549.1120 | flashattn_sm100 | 5662.1554 | 1.020 | — |

## flashkda_bf16_fused_m128

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `h64_mixed` | tirx | 268.2399 | flashinfer_m128 | 271.9395 | 1.014 | flashkda_raw=667.1473 |
| `h64_uniform` | tirx | 465.6187 | flashinfer_m128 | 473.5961 | 1.017 | flashkda_raw=764.6676 |
| `h96_fixed8192` | tirx | 500.8359 | flashinfer_m128 | 508.1553 | 1.015 | flashkda_raw=1075.1109 |
| `h96_mixed` | tirx | 609.2198 | flashinfer_m128 | 622.6901 | 1.022 | flashkda_raw=1403.8307 |
| `h96_uniform` | tirx | 432.0196 | flashinfer_m128 | 436.5378 | 1.010 | flashkda_raw=708.6740 |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.8719 | torch-cublas | 5.9959 | 0.873 | deepgemm-bf16=6.8735, deepgemm-cublaslt=6.0038 |
| `bf16_16384x16384x16384` | tir | 5423.8178 | torch-cublas | 5486.0778 | 1.011 | deepgemm-bf16=5934.2415, deepgemm-cublaslt=5488.1450 |
| `bf16_2048x2048x2048` | tir | 27.4191 | torch-cublas | 26.4829 | 0.966 | deepgemm-bf16=29.1607, deepgemm-cublaslt=26.4901 |
| `bf16_4096x4096x4096` | tir | 90.7458 | deepgemm-bf16 | 87.5186 | 0.964 | deepgemm-cublaslt=88.1432, torch-cublas=88.7432 |
| `bf16_8192x8192x8192` | tir | 694.8820 | deepgemm-cublaslt | 708.0456 | 1.019 | deepgemm-bf16=732.9954, torch-cublas=712.9988 |
| `fp16_1024x1024x1024` | tir | 10.3797 | torch-cublas | 8.7730 | 0.845 | — |
| `fp16_16384x16384x16384` | tir | 5957.4736 | torch-cublas | 5888.4751 | 0.988 | — |
| `fp16_2048x2048x2048` | tir | 16.4784 | torch-cublas | 15.8591 | 0.962 | — |
| `fp16_4096x4096x4096` | tir | 148.5279 | torch-cublas | 140.2774 | 0.944 | — |
| `fp16_8192x8192x8192` | tir | 738.8704 | torch-cublas | 762.3584 | 1.032 | — |

## gdn_prefill_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hq16_hv16_s4096+4096` | tirx | 127.3157 | flashinfer_cutedsl | 133.4357 | 1.048 | — |
| `hq16_hv64_s1x8192` | tirx | 385.7228 | flashinfer_cutedsl | 402.0258 | 1.042 | — |
| `hq2_hv8_s1x65536` | tirx | 1758.8466 | flashinfer_cutedsl | 1781.1827 | 1.013 | — |
| `hq32_hv32_s8192x16` | tirx | 1558.5000 | flashinfer_cutedsl | 1622.4510 | 1.041 | — |
| `hq8_hv32_s1024x8` | tirx | 92.2448 | flashinfer_cutedsl | 95.8803 | 1.039 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 11267.0222 | cublas_nccl_cudagraph | 10951.6801 | 0.972 | cublasmp_split_p2p=12201.0029 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 1000.4833 | cublas_nccl_cudagraph | 943.5559 | 0.943 | cublasmp_split_p2p=1243.5105 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1493.2130 | cublas_nccl_cudagraph | 1436.5057 | 0.962 | cublasmp_split_p2p=1840.6237 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2402.0207 | cublas_nccl_cudagraph | 2383.3390 | 0.992 | cublasmp_split_p2p=2819.7509 |

## mxfp4_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_128x4_m4096_k4096` | tirx | 10.7420 | flashinfer | 10.7219 | 0.998 | — |
| `bf16_linear_m4096_k4096` | tirx | 10.3308 | flashinfer | 10.6924 | 1.035 | — |
| `fp16_128x4_m1024_k2048` | tirx | 3.7288 | flashinfer | 3.7532 | 1.007 | — |
| `fp16_128x4_m128_k1024` | tirx | 2.8414 | flashinfer | 2.8668 | 1.009 | — |
| `fp16_128x4_m16384_k7168` | tirx | 66.8505 | flashinfer | 68.2974 | 1.022 | — |
| `fp16_128x4_m4096_k4096` | tirx | 10.7370 | flashinfer | 10.6849 | 0.995 | — |
| `fp16_linear_m1024_k2048` | tirx | 3.5864 | flashinfer | 3.6517 | 1.018 | — |
| `fp16_linear_m128_k1024` | tirx | 2.3680 | flashinfer | 2.4079 | 1.017 | — |
| `fp16_linear_m16384_k7168` | tirx | 50.5177 | flashinfer | 50.6019 | 1.002 | — |
| `fp16_linear_m4096_k4096` | tirx | 13.9417 | flashinfer | 14.0012 | 1.004 | — |

## mxfp8_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_128x4_m4096_k4096` | tirx | 11.7643 | flashinfer | 11.8278 | 1.005 | — |
| `bf16_linear_m4096_k4096` | tirx | 11.2872 | flashinfer | 11.1933 | 0.992 | — |
| `fp16_128x4_m1024_k2048` | tirx | 3.8630 | flashinfer | 3.9447 | 1.021 | — |
| `fp16_128x4_m128_k1024` | tirx | 3.6176 | flashinfer | 3.7711 | 1.042 | — |
| `fp16_128x4_m16384_k7168` | tirx | 71.5604 | flashinfer | 76.6297 | 1.071 | — |
| `fp16_128x4_m4096_k4096` | tirx | 11.7617 | flashinfer | 11.8319 | 1.006 | — |
| `fp16_linear_m1024_k2048` | tirx | 3.7724 | flashinfer | 3.8231 | 1.013 | — |
| `fp16_linear_m128_k1024` | tirx | 2.6200 | flashinfer | 2.6025 | 0.993 | — |
| `fp16_linear_m16384_k7168` | tirx | 62.3071 | flashinfer | 63.2526 | 1.015 | — |
| `fp16_linear_m4096_k4096` | tirx | 13.8346 | flashinfer | 13.5341 | 0.978 | — |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.4514 | cublaslt_nvfp4 | 4.5272 | 0.830 | flashinfer=4.5366 |
| `16384x16384x16384` | tir | 1516.7496 | cublaslt_nvfp4 | 1431.0615 | 0.944 | flashinfer=1445.1355 |
| `2048x2048x2048` | tir | 8.8312 | flashinfer | 7.7123 | 0.873 | cublaslt_nvfp4=8.5846 |
| `4096x4096x4096` | tir | 29.8666 | cublaslt_nvfp4 | 27.7979 | 0.931 | flashinfer=28.9586 |
| `8192x8192x8192` | tir | 253.0963 | cublaslt_nvfp4 | 238.7013 | 0.943 | flashinfer=240.3198 |

## nvfp4_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_128x4_m4096_k3584_silu` | tirx | 20.6098 | flashinfer | 22.3594 | 1.085 | — |
| `bf16_128x4_m4096_k4096` | tirx | 10.7824 | flashinfer | 11.1453 | 1.034 | — |
| `bf16_linear_m4096_k4096` | tirx | 10.0129 | flashinfer | 10.2862 | 1.027 | — |
| `fp16_128x4_m1024_k2048` | tirx | 3.6911 | flashinfer | 3.6802 | 0.997 | — |
| `fp16_128x4_m128_k1024` | tirx | 3.4540 | flashinfer | 3.4875 | 1.010 | — |
| `fp16_128x4_m16384_k7168` | tirx | 54.6462 | flashinfer | 55.1040 | 1.008 | — |
| `fp16_128x4_m4096_k3584_silu` | tirx | 20.3774 | flashinfer | 21.8474 | 1.072 | — |
| `fp16_128x4_m4096_k4096` | tirx | 11.0861 | flashinfer | 10.8557 | 0.979 | — |
| `fp16_linear_m1024_k2048` | tirx | 3.6253 | flashinfer | 3.5903 | 0.990 | — |
| `fp16_linear_m128_k1024` | tirx | 2.4949 | flashinfer | 2.4904 | 0.998 | — |
| `fp16_linear_m16384_k7168` | tirx | 50.5884 | flashinfer | 51.1227 | 1.011 | — |
| `fp16_linear_m4096_k4096` | tirx | 10.3428 | flashinfer | 10.4800 | 1.013 | — |

## nvfp4_quantize_per_token

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_128x4_m4096_k4096` | tirx | 12.7112 | flashinfer | 13.7694 | 1.083 | — |
| `bf16_linear_m4096_k4096` | tirx | 12.1999 | flashinfer | 13.2294 | 1.084 | — |
| `fp16_128x4_m1024_k2048` | tirx | 3.9813 | flashinfer | 4.1212 | 1.035 | — |
| `fp16_128x4_m128_k1024` | tirx | 2.8930 | flashinfer | 3.0564 | 1.056 | — |
| `fp16_128x4_m16384_k7168` | tirx | 57.8227 | flashinfer | 59.3197 | 1.026 | — |
| `fp16_128x4_m4096_k4096` | tirx | 12.6939 | flashinfer | 13.5944 | 1.071 | — |
| `fp16_linear_m1024_k2048` | tirx | 3.8549 | flashinfer | 3.9801 | 1.032 | — |
| `fp16_linear_m128_k1024` | tirx | 2.9066 | flashinfer | 3.0670 | 1.055 | — |
| `fp16_linear_m16384_k7168` | tirx | 55.8258 | flashinfer | 57.4695 | 1.029 | — |
| `fp16_linear_m4096_k4096` | tirx | 17.3447 | flashinfer | 19.5163 | 1.125 | — |

## recurrent_kda_decode_grouped

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `dec_hv12_b4` | tirx | 3.6504 | flashinfer_cutedsl | 3.9219 | 1.074 | — |
| `dec_hv12_b8` | tirx | 4.2584 | flashinfer_cutedsl | 4.6137 | 1.083 | — |
| `dec_hv16_b1` | tirx | 3.2388 | flashinfer_cutedsl | 3.5628 | 1.100 | — |
| `dec_hv16_b4` | tirx | 3.7759 | flashinfer_cutedsl | 4.0781 | 1.080 | — |
| `ver_t2_hv16_b8` | tirx | 6.8376 | flashinfer_cutedsl | 7.3421 | 1.074 | — |
| `ver_t4_hv16_b32` | tirx | 24.7130 | flashinfer_cutedsl | 33.8884 | 1.371 | — |
| `ver_t8_hv12_b16` | tirx | 22.0309 | flashinfer_cutedsl | 31.3530 | 1.423 | — |
| `ver_t8_hv12_b64` | tirx | 65.9184 | flashinfer_cutedsl | 92.4159 | 1.402 | — |
| `ver_t8_hv16_b1` | tirx | 9.0308 | flashinfer_cutedsl | 14.8158 | 1.641 | — |
| `ver_t8_hv16_b128` | tirx | 161.9329 | flashinfer_cutedsl | 219.3316 | 1.354 | — |
| `ver_t8_hv16_b16` | tirx | 26.0173 | flashinfer_cutedsl | 34.9513 | 1.343 | — |
| `ver_t8_hv16_b32` | tirx | 44.5975 | flashinfer_cutedsl | 63.6727 | 1.428 | — |
| `ver_t8_hv16_b4` | tirx | 10.2479 | flashinfer_cutedsl | 16.9288 | 1.652 | — |
| `ver_t8_hv16_b64` | tirx | 83.1184 | flashinfer_cutedsl | 112.7169 | 1.356 | — |

## recurrent_kda_decode_one_warp

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv12_b16_tr16_lb` | tirx | 5.6951 | flashinfer_cutedsl | 5.9730 | 1.049 | — |
| `hv12_b64_tr16_lb` | tirx | 12.8366 | flashinfer_cutedsl | 13.8690 | 1.080 | — |
| `hv16_b128_tr16_lb` | tirx | 26.6851 | flashinfer_cutedsl | 29.1858 | 1.094 | — |
| `hv16_b16_tr16_lb` | tirx | 6.3473 | flashinfer_cutedsl | 6.4838 | 1.022 | — |
| `hv16_b32_tr16_lb` | tirx | 9.6974 | flashinfer_cutedsl | 10.2393 | 1.056 | — |
| `hv16_b64_tr16_lb` | tirx | 15.2689 | flashinfer_cutedsl | 16.9040 | 1.107 | — |
| `hv16_b8_tr8_lb` | tirx | 5.2472 | flashinfer_cutedsl | 5.3377 | 1.017 | — |

## selective_state_update_mtp_horizontal

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 7.0351 | flashinfer_cuda | 7.5239 | 1.069 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 1388.3803 | flashinfer_cuda | 1509.4559 | 1.087 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 352.2852 | flashinfer_cuda | 382.5711 | 1.086 | — |
| `b64_h64_d64_s128_t6_r8_statebf16_official` | tirx | 82.9846 | flashinfer_cuda | 89.3754 | 1.077 | — |

## selective_state_update_mtp_simple

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 6.5280 | flashinfer_cuda | 7.8631 | 1.205 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 1520.4572 | flashinfer_cuda | 1584.4758 | 1.042 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 384.8073 | flashinfer_cuda | 400.9095 | 1.042 | — |
| `b64_h64_d64_s128_t6_r8_statebf16_official` | tirx | 89.6541 | flashinfer_cuda | 93.7083 | 1.045 | — |

## selective_state_update_mtp_vertical

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 28.4328 | flashinfer_cuda | 29.8231 | 1.049 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 2826.8122 | flashinfer_cuda | 2905.6069 | 1.028 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 1208.4806 | flashinfer_cuda | 1242.4536 | 1.028 | — |
| `b64_h64_d64_s128_t6_r8_statebf16_official` | tirx | 98.6174 | flashinfer_cuda | 101.2316 | 1.027 | — |

## selective_state_update_stp_horizontal

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 48.8682 | flashinfer_cuda | 49.7236 | 1.018 | — |
| `b64_h64_d64_s128_r64` | tirx | 40.5516 | flashinfer_cuda | 42.5661 | 1.050 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 41.8943 | flashinfer_cuda | 43.2139 | 1.031 | — |
| `b64_h64_d64_s256_r8` | tirx | 67.7627 | flashinfer_cuda | 68.3421 | 1.009 | — |

## selective_state_update_stp_simple

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 65.4906 | flashinfer_cuda | 67.4403 | 1.030 | — |
| `b64_h64_d64_s128_r64` | tirx | 54.2322 | flashinfer_cuda | 58.5379 | 1.079 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 37.4572 | flashinfer_cuda | 38.1160 | 1.018 | — |
| `b64_h64_d64_s256_r8` | tirx | 74.9952 | flashinfer_cuda | 83.5618 | 1.114 | — |

## selective_state_update_stp_vertical

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 47.5355 | flashinfer_cuda | 54.3434 | 1.143 | — |
| `b64_h64_d64_s128_r64` | tirx | 27.5996 | flashinfer_cuda | 31.3183 | 1.135 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 27.6318 | flashinfer_cuda | 31.3378 | 1.134 | — |
| `b64_h64_d64_s256_r8` | tirx | 67.9484 | flashinfer_cuda | 79.1539 | 1.165 | — |

## silu_and_mul_nvfp4_experts_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_b128_m2048_k2048` | tirx | 285.1640 | flashinfer | 319.4270 | 1.120 | — |
| `bf16_b8_m512_k2048` | tirx | 13.5666 | flashinfer | 16.0305 | 1.182 | — |
| `fp16_b128_m2048_k2048` | tirx | 421.6170 | flashinfer | 503.8167 | 1.195 | — |
| `fp16_b4_m128_k4096` | tirx | 5.5833 | flashinfer | 5.7942 | 1.038 | — |
| `fp16_b8_m16_k2048` | tirx | 4.8038 | flashinfer | 6.0456 | 1.259 | — |
| `fp16_b8_m512_k2048` | tirx | 8.8492 | flashinfer | 10.2845 | 1.162 | — |

## sparse_flashmla_decode_head64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | tirx | 136.8763 | flashmla | 144.2464 | 1.054 | — |
| `model1_b148_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 74.9601 | flashmla | 78.1596 | 1.043 | — |
| `model1_b256_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 153.6155 | flashmla | 163.3648 | 1.063 | — |
| `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 28.5783 | flashmla | 34.1026 | 1.193 | — |
| `v32_b148_sq2_sk32768_topk16384_p64` | tirx | 910.6713 | flashmla | 953.2395 | 1.047 | — |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1728.4820 | flashmla | 1761.2495 | 1.019 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 2489.6685 | flashmla | 2492.2068 | 1.001 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1729.9883 | flashmla | 1768.0810 | 1.022 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1859.8779 | flashmla | 1867.4771 | 1.004 | trtllm_gen=2090.3868 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 1990.8174 | flashmla | 1991.3282 | 1.000 | trtllm_gen=2192.8898 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 2421.3491 | flashmla | 2408.3092 | 0.995 | trtllm_gen=2839.9839 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1152.0658 | flashmla | 1171.3121 | 1.017 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1668.5951 | flashmla | 1677.7430 | 1.005 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1149.5010 | flashmla | 1161.5816 | 1.011 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 370.0398 | flashmla | 378.4910 | 1.023 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 377.4105 | flashmla | 383.6426 | 1.017 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 582.4460 | flashmla | 595.1229 | 1.022 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 572.5333 | flashmla | 585.4490 | 1.023 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 388.0184 | flashmla | 401.0920 | 1.034 | trtllm_gen=464.0309 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 584.6850 | flashmla | 605.9564 | 1.036 | trtllm_gen=719.6944 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 405.1609 | flashmla | 418.7721 | 1.034 | trtllm_gen=488.0644 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 372.2002 | flashmla | 381.5993 | 1.025 | trtllm_gen=447.2225 |

## tinygemm2_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b13_o1024_k2048` | tirx | 4.2666 | flashinfer_sm100 | 4.2510 | 0.996 | — |
| `b16_o2880_k2880` | tirx | 7.9593 | flashinfer_sm100 | 8.0375 | 1.010 | — |
| `b1_o128_k720` | tirx | 2.9497 | flashinfer_sm100 | 2.9316 | 0.994 | — |
| `b2_o16_k256` | tirx | 2.7903 | flashinfer_sm100 | 2.8247 | 1.012 | — |
| `b4_o2880_k2880` | tirx | 9.6865 | flashinfer_sm100 | 9.7997 | 1.012 | — |
| `b64_o4096_k3072` | tirx | 35.2649 | flashinfer_sm100 | 35.4422 | 1.005 | — |
| `b7_o128_k4096` | tirx | 4.9619 | flashinfer_sm100 | 4.9197 | 0.991 | — |
| `b8_o1024_k1024` | tirx | 4.8111 | flashinfer_sm100 | 4.8135 | 1.001 | — |
