# bench-suite baseline view: `baseline.json`

- Timestamp: `14`
- Label:     `post-refactor`
- Git:       `{'tir': 'f5d998f1', 'tirx-kernels': 'a27292c8-dirty', 'tirx-bench-ci': None}`
- Workloads: 150 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## act_and_mul

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gelu_fp16_d11008_t8192` | tirx | 153.7020 | flashinfer | 253.0105 | 1.646 | — |
| `gelu_tanh_fp16_d11008_t8192` | tirx | 97.3507 | flashinfer | 109.2135 | 1.122 | — |
| `silu_bf16_d16384_t32768` | tirx | 454.3596 | flashinfer | 474.5232 | 1.044 | — |
| `silu_bf16_d4096_t8192` | tirx | 33.3754 | flashinfer | 36.2414 | 1.086 | — |
| `silu_fp16_d11008_t8192` | tirx | 97.6393 | flashinfer | 108.0944 | 1.107 | — |
| `silu_fp16_d16384_t32768` | tirx | 453.1996 | flashinfer | 468.4406 | 1.034 | — |
| `silu_fp16_d4096_t1` | tirx | 2.5989 | flashinfer | 2.8029 | 1.078 | — |
| `silu_fp16_d4096_t8192` | tirx | 33.0855 | flashinfer | 35.9270 | 1.086 | — |

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 20628.0216 | cublas_nccl_cudagraph | 17975.1971 | 0.871 | cublasmp_split_p2p=18194.6035 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1047.3683 | cublas_nccl_cudagraph | 1029.8326 | 0.983 | cublasmp_split_p2p=1075.0385 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2840.5361 | cublas_nccl_cudagraph | 2669.6732 | 0.940 | cublasmp_split_p2p=2723.3628 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 5271.5981 | cublas_nccl_cudagraph | 4994.9124 | 0.948 | cublasmp_split_p2p=5084.1313 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1300.0000 | deepgemm | 1290.2000 | 0.992 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3417.2000 | deepgemm | 3418.0000 | 1.000 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 37.7683 | deepgemm | 39.6569 | 1.050 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 182.1997 | deepgemm | 183.4966 | 1.007 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.3675 | deepgemm | 6.5311 | 1.026 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.4052 | deepgemm | 5.1028 | 1.158 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 39.7206 | deepgemm | 40.9378 | 1.031 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 176.5759 | deepgemm | 188.0171 | 1.065 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.5965 | deepgemm | 6.8616 | 1.040 | sglang_cutedsl=6.9308 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5151 | deepgemm | 4.9688 | 1.100 | sglang_cutedsl=4.9862 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.0570 | deepgemm | 5.0612 | 1.001 | — |
| `m137_n24_k7680_s16` | tirx | 5.2931 | deepgemm | 5.3676 | 1.014 | — |
| `m13_n24_k7168_s1` | tirx | 21.0857 | deepgemm | 20.8208 | 0.987 | — |
| `m4096_n24_k28672_s16` | tirx | 57.5820 | deepgemm | 62.5876 | 1.087 | — |
| `m4096_n24_k7168_s1` | tirx | 23.1313 | deepgemm | 23.6549 | 1.023 | — |
| `m64_n24_k28672_s112` | tirx | 5.1753 | deepgemm | 5.1880 | 1.002 | — |
| `m8192_n24_k16384_s1` | tirx | 51.1783 | deepgemm | 60.5043 | 1.182 | — |
| `m8192_n24_k28672_s1` | tirx | 84.0412 | deepgemm | 92.6510 | 1.102 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.6227 | flashattn_sm100 | 20.1902 | 1.029 | — |
| `s1024_h32kv16_causal` | tir | 20.1335 | flashattn_sm100 | 20.3510 | 1.011 | — |
| `s1024_h32kv32` | tir | 20.1687 | flashattn_sm100 | 20.6800 | 1.025 | — |
| `s1024_h32kv32_causal` | tir | 21.2254 | flashattn_sm100 | 21.5249 | 1.014 | — |
| `s1024_h32kv4` | tir | 19.1319 | flashattn_sm100 | 20.0153 | 1.046 | — |
| `s1024_h32kv4_causal` | tir | 18.8343 | flashattn_sm100 | 19.6094 | 1.041 | — |
| `s1024_h32kv8` | tir | 19.5072 | flashattn_sm100 | 20.0500 | 1.028 | — |
| `s1024_h32kv8_causal` | tir | 19.1711 | flashattn_sm100 | 19.7744 | 1.031 | — |
| `s2048_h32kv16` | tir | 57.6003 | flashattn_sm100 | 57.7701 | 1.003 | — |
| `s2048_h32kv16_causal` | tir | 36.3420 | flashattn_sm100 | 38.5848 | 1.062 | — |
| `s2048_h32kv32` | tir | 59.3939 | flashattn_sm100 | 59.8279 | 1.007 | — |
| `s2048_h32kv32_causal` | tir | 39.9021 | flashattn_sm100 | 40.1212 | 1.005 | — |
| `s2048_h32kv4` | tir | 54.8049 | flashattn_sm100 | 55.8229 | 1.019 | — |
| `s2048_h32kv4_causal` | tir | 35.0373 | flashattn_sm100 | 37.7036 | 1.076 | — |
| `s2048_h32kv8` | tir | 56.1241 | flashattn_sm100 | 56.6827 | 1.010 | — |
| `s2048_h32kv8_causal` | tir | 35.4594 | flashattn_sm100 | 37.8575 | 1.068 | — |
| `s4096_h32kv16` | tir | 211.9596 | flashattn_sm100 | 213.8201 | 1.009 | — |
| `s4096_h32kv16_causal` | tir | 113.2938 | flashattn_sm100 | 118.4429 | 1.045 | — |
| `s4096_h32kv32` | tir | 215.5509 | flashattn_sm100 | 217.4530 | 1.009 | — |
| `s4096_h32kv32_causal` | tir | 121.7270 | flashattn_sm100 | 121.7182 | 1.000 | — |
| `s4096_h32kv4` | tir | 208.7661 | flashattn_sm100 | 209.4963 | 1.003 | — |
| `s4096_h32kv4_causal` | tir | 110.9106 | flashattn_sm100 | 115.9296 | 1.045 | — |
| `s4096_h32kv8` | tir | 207.4134 | flashattn_sm100 | 209.5260 | 1.010 | — |
| `s4096_h32kv8_causal` | tir | 112.0452 | flashattn_sm100 | 117.5551 | 1.049 | — |
| `s8192_h32kv16` | tir | 780.7842 | flashattn_sm100 | 779.7321 | 0.999 | — |
| `s8192_h32kv16_causal` | tir | 422.0594 | flashattn_sm100 | 428.6703 | 1.016 | — |
| `s8192_h32kv32` | tir | 776.4990 | flashattn_sm100 | 784.8277 | 1.011 | — |
| `s8192_h32kv32_causal` | tir | 434.3788 | flashattn_sm100 | 433.8064 | 0.999 | — |
| `s8192_h32kv4` | tir | 758.4225 | flashattn_sm100 | 761.1707 | 1.004 | — |
| `s8192_h32kv4_causal` | tir | 414.1863 | flashattn_sm100 | 424.4719 | 1.025 | — |
| `s8192_h32kv8` | tir | 766.1723 | flashattn_sm100 | 773.2631 | 1.009 | — |
| `s8192_h32kv8_causal` | tir | 415.7060 | flashattn_sm100 | 425.8563 | 1.024 | — |

## flashkda_bf16_fused_m128

| config | tirx (µs) | tirx_tx_tile (µs) | flashinfer_m128 (µs) | flashkda_raw (µs) |
|---|---:|---:|---:|---:|
| `h64_mixed` | 271.4726 | 270.5494 | 271.1238 | 665.1911 |
| `h64_uniform` | 295.8269 | 295.6376 | 294.2116 | 478.2878 |
| `h96_fixed8192` | 505.9878 | 507.6734 | 507.5801 | 1076.2557 |
| `h96_mixed` | 389.6571 | 390.3374 | 394.1816 | 883.8601 |
| `h96_uniform` | 438.7699 | 438.9795 | 436.1381 | 708.7317 |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.7218 | torch-cublas | 5.7967 | 0.862 | deepgemm-bf16=6.7270, deepgemm-cublaslt=5.8036 |
| `bf16_16384x16384x16384` | tir | 5755.9696 | deepgemm-cublaslt | 5536.5311 | 0.962 | deepgemm-bf16=6769.4673, torch-cublas=5547.0202 |
| `bf16_2048x2048x2048` | tir | 16.2937 | torch-cublas | 15.6522 | 0.961 | deepgemm-bf16=17.2255, deepgemm-cublaslt=15.6594 |
| `bf16_4096x4096x4096` | tir | 93.3913 | deepgemm-bf16 | 89.0707 | 0.954 | deepgemm-cublaslt=90.4138, torch-cublas=90.0437 |
| `bf16_8192x8192x8192` | tir | 689.1778 | deepgemm-cublaslt | 718.6884 | 1.043 | deepgemm-bf16=743.0745, torch-cublas=733.8203 |
| `fp16_1024x1024x1024` | tir | 6.7709 | torch-cublas | 5.8691 | 0.867 | — |
| `fp16_16384x16384x16384` | tir | 5959.6308 | torch-cublas | 5856.6750 | 0.983 | — |
| `fp16_2048x2048x2048` | tir | 16.5415 | torch-cublas | 15.8525 | 0.958 | — |
| `fp16_4096x4096x4096` | tir | 97.4104 | torch-cublas | 93.6657 | 0.962 | — |
| `fp16_8192x8192x8192` | tir | 737.0429 | torch-cublas | 744.8045 | 1.011 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.7796 | deepgemm | 49.4835 | 0.974 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 113.9491 | deepgemm | 113.6677 | 0.998 | — |
| `deepgemm_m4096_n32768_k512` | tir | 67.8366 | deepgemm | 71.9740 | 1.061 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 80.7791 | deepgemm | 80.7767 | 1.000 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.0995 | deepgemm | 19.0024 | 0.945 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 340.2924 | deepgemm | 332.4038 | 0.977 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 43.3182 | deepgemm | 42.1357 | 0.973 | — |

## gdn_prefill_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hq16_hv16_s4096+4096` | tirx | 126.3829 | flashinfer_cutedsl | 133.2174 | 1.054 | — |
| `hq16_hv64_s1x8192` | tirx | 238.9151 | flashinfer_cutedsl | 251.4505 | 1.052 | — |
| `hq2_hv8_s1x65536` | tirx | 1735.8840 | flashinfer_cutedsl | 1785.6235 | 1.029 | — |
| `hq32_hv32_s8192x16` | tirx | 1073.0538 | flashinfer_cutedsl | 1120.0307 | 1.044 | — |
| `hq8_hv32_s1024x8` | tirx | 91.7895 | flashinfer_cutedsl | 95.5388 | 1.041 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 9517.6854 | cublas_nccl_cudagraph | 9146.2667 | 0.961 | cublasmp_split_p2p=9947.0179 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 602.2152 | cublas_nccl_cudagraph | 564.3116 | 0.937 | cublasmp_split_p2p=754.4997 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1501.6823 | cublas_nccl_cudagraph | 1411.5482 | 0.940 | cublasmp_split_p2p=1811.8186 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2499.8367 | cublas_nccl_cudagraph | 2450.8611 | 0.980 | cublasmp_split_p2p=2886.1382 |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 162.4744 | deepgemm | 165.6884 | 1.020 | — |
| `large_g4_m8192_n4096_k4096` | tir | 349.7351 | deepgemm | 367.1022 | 1.050 | — |
| `large_g4_m8192_n6144_k7168` | tir | 979.6092 | deepgemm | 989.7458 | 1.010 | — |
| `large_g4_m8192_n7168_k3072` | tir | 496.1302 | deepgemm | 529.0817 | 1.066 | — |
| `large_g8_m4096_n4096_k2048` | tir | 192.7758 | deepgemm | 199.0057 | 1.032 | — |
| `large_g8_m4096_n4096_k4096` | tir | 349.4112 | deepgemm | 347.7435 | 0.995 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1094.3548 | deepgemm | 1126.7154 | 1.030 | — |
| `large_g8_m4096_n7168_k3072` | tir | 501.1544 | deepgemm | 530.3688 | 1.058 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 33.8177 | 37.1881 | 33.2740 | 53.2202 | 60.1598 |
| `moe_a3b_bs8_all` | 96.4288 | 100.0675 | 103.5619 | 126.5397 | 131.4599 |
| `moe_a3b_bs32_all` | 192.3405 | 192.4070 | 198.5807 | 224.5717 | 218.3641 |
| `moe_a3b_bs128_all` | 222.1318 | 220.2841 | 228.6052 | 245.3700 | 247.8075 |
| `moe_a3b_bs512_all` | 239.2628 | 231.7671 | 244.3510 | 287.6833 | 274.4938 |
| `moe_a3b_bs1024_all` | 259.9133 | 257.9804 | 275.5271 | 338.3083 | 312.4617 |
| `moe_a3b_bs2048_all` | 339.3372 | 343.8899 | 355.3639 | 430.7005 | 392.3249 |
| `moe_a3b_bs4096_all` | 562.7819 | 559.7292 | 574.0259 | 667.4069 | 604.9371 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.4250 | cublaslt_nvfp4 | 4.4581 | 0.822 | flashinfer=4.5164 |
| `16384x16384x16384` | tir | 1527.0347 | flashinfer | 1430.8557 | 0.937 | cublaslt_nvfp4=1456.2751 |
| `2048x2048x2048` | tir | 8.5260 | flashinfer | 7.5789 | 0.889 | cublaslt_nvfp4=7.6341 |
| `4096x4096x4096` | tir | 29.5830 | cublaslt_nvfp4 | 27.6822 | 0.936 | flashinfer=28.8911 |
| `8192x8192x8192` | tir | 186.1670 | cublaslt_nvfp4 | 177.5714 | 0.954 | flashinfer=177.6061 |

## silu_and_mul_nvfp4_experts_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_b128_m2048_k2048` | tirx | 285.8931 | flashinfer | 321.6687 | 1.125 | — |
| `bf16_b8_m512_k2048` | tirx | 8.8665 | flashinfer | 10.7135 | 1.208 | — |
| `fp16_b128_m2048_k2048` | tirx | 276.2779 | flashinfer | 304.7120 | 1.103 | — |
| `fp16_b4_m128_k4096` | tirx | 5.5820 | flashinfer | 5.8108 | 1.041 | — |
| `fp16_b8_m16_k2048` | tirx | 3.4218 | flashinfer | 4.2766 | 1.250 | — |
| `fp16_b8_m512_k2048` | tirx | 8.7735 | flashinfer | 10.5524 | 1.203 | — |

## sparse_flashmla_decode_head64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | tirx | 136.4290 | flashmla | 143.7901 | 1.054 | — |
| `model1_b148_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 75.0438 | flashmla | 78.8342 | 1.051 | — |
| `model1_b256_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 104.3486 | flashmla | 108.5835 | 1.041 | — |
| `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 16.9367 | flashmla | 21.4151 | 1.264 | — |
| `v32_b148_sq2_sk32768_topk16384_p64` | tirx | 921.0947 | flashmla | 959.5411 | 1.042 | — |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1737.3124 | flashmla | 1739.4067 | 1.001 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1879.5763 | flashmla | 1887.1304 | 1.004 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1725.9254 | flashmla | 1748.1179 | 1.013 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1850.5348 | flashmla | 1825.1426 | 0.986 | trtllm_gen=2072.0612 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 1993.4430 | flashmla | 1976.7194 | 0.992 | trtllm_gen=2182.0294 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1833.5276 | flashmla | 1830.6983 | 0.998 | trtllm_gen=2064.3804 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1152.1517 | flashmla | 1170.8088 | 1.016 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1184.8009 | flashmla | 1201.8090 | 1.014 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1145.0310 | flashmla | 1159.9359 | 1.013 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 368.8593 | flashmla | 378.9462 | 1.027 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 371.4774 | flashmla | 381.1018 | 1.026 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 376.9243 | flashmla | 388.6337 | 1.031 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 362.6400 | flashmla | 375.0309 | 1.034 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 388.1928 | flashmla | 402.6150 | 1.037 | trtllm_gen=463.0734 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 390.2824 | flashmla | 404.9671 | 1.038 | trtllm_gen=474.2063 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 402.6227 | flashmla | 418.9807 | 1.041 | trtllm_gen=488.9936 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 371.5140 | flashmla | 383.3754 | 1.032 | trtllm_gen=449.4779 |

## tinygemm2_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b13_o1024_k2048` | tirx | 4.3372 | flashinfer_sm100 | 4.3214 | 0.996 | — |
| `b16_o2880_k2880` | tirx | 7.9576 | flashinfer_sm100 | 8.0430 | 1.011 | — |
| `b1_o128_k720` | tirx | 3.3080 | flashinfer_sm100 | 3.3084 | 1.000 | — |
| `b2_o16_k256` | tirx | 3.4074 | flashinfer_sm100 | 3.4165 | 1.003 | — |
| `b4_o2880_k2880` | tirx | 7.0696 | flashinfer_sm100 | 7.0592 | 0.999 | — |
| `b64_o4096_k3072` | tirx | 22.1238 | flashinfer_sm100 | 22.2666 | 1.006 | — |
| `b7_o128_k4096` | tirx | 4.9490 | flashinfer_sm100 | 4.9171 | 0.994 | — |
| `b8_o1024_k1024` | tirx | 3.5398 | flashinfer_sm100 | 3.5290 | 0.997 | — |
