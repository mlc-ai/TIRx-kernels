# bench-suite baseline view: `baseline.json`

- Timestamp: `14`
- Label:     `post-refactor`
- Git:       `{'tir': 'fabb698a', 'tirx-kernels': '6c2af771', 'tirx-bench-ci': None}`
- Workloads: 113 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 20324.1247 | cublas_nccl_cudagraph | 18118.8980 | 0.891 | cublasmp_split_p2p=18339.1959 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1053.2129 | cublas_nccl_cudagraph | 1032.3811 | 0.980 | cublasmp_split_p2p=1078.3585 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2806.9690 | cublas_nccl_cudagraph | 2699.1326 | 0.962 | cublasmp_split_p2p=2756.7483 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 5361.5593 | cublas_nccl_cudagraph | 4892.5433 | 0.913 | cublasmp_split_p2p=4985.0454 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1297.0000 | deepgemm | 1289.0000 | 0.994 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3439.2000 | deepgemm | 3440.0000 | 1.000 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 37.5969 | deepgemm | 39.6371 | 1.054 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 181.1046 | deepgemm | 181.3527 | 1.001 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.4656 | deepgemm | 6.4663 | 1.000 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.0549 | deepgemm | 4.9944 | 1.232 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 40.0279 | deepgemm | 41.2420 | 1.030 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 180.1963 | deepgemm | 189.3540 | 1.051 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.6756 | deepgemm | 6.8586 | 1.027 | sglang_cutedsl=6.9025 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6250 | sglang_cutedsl | 4.8811 | 1.055 | deepgemm=5.4470 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.2311 | deepgemm | 5.1865 | 0.991 | — |
| `m137_n24_k7680_s16` | tirx | 5.3731 | deepgemm | 5.3226 | 0.991 | — |
| `m13_n24_k7168_s1` | tirx | 21.4164 | deepgemm | 20.8460 | 0.973 | — |
| `m4096_n24_k28672_s16` | tirx | 58.1708 | deepgemm | 62.9098 | 1.081 | — |
| `m4096_n24_k7168_s1` | tirx | 23.2634 | deepgemm | 23.6653 | 1.017 | — |
| `m64_n24_k28672_s112` | tirx | 5.1863 | deepgemm | 5.1776 | 0.998 | — |
| `m8192_n24_k16384_s1` | tirx | 51.6240 | deepgemm | 61.6711 | 1.195 | — |
| `m8192_n24_k28672_s1` | tirx | 83.7983 | deepgemm | 91.7821 | 1.095 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.7438 | flashattn_sm100 | 20.1868 | 1.022 | — |
| `s1024_h32kv16_causal` | tir | 20.1373 | flashattn_sm100 | 20.2434 | 1.005 | — |
| `s1024_h32kv32` | tir | 20.1770 | flashattn_sm100 | 20.5041 | 1.016 | — |
| `s1024_h32kv32_causal` | tir | 20.6680 | flashattn_sm100 | 21.4776 | 1.039 | — |
| `s1024_h32kv4` | tir | 19.2911 | flashattn_sm100 | 19.9360 | 1.033 | — |
| `s1024_h32kv4_causal` | tir | 18.9146 | flashattn_sm100 | 19.6163 | 1.037 | — |
| `s1024_h32kv8` | tir | 19.5000 | flashattn_sm100 | 19.9824 | 1.025 | — |
| `s1024_h32kv8_causal` | tir | 19.0961 | flashattn_sm100 | 19.7242 | 1.033 | — |
| `s2048_h32kv16` | tir | 57.8216 | flashattn_sm100 | 58.2600 | 1.008 | — |
| `s2048_h32kv16_causal` | tir | 36.2784 | flashattn_sm100 | 38.6564 | 1.066 | — |
| `s2048_h32kv32` | tir | 59.3314 | flashattn_sm100 | 59.6296 | 1.005 | — |
| `s2048_h32kv32_causal` | tir | 39.8301 | flashattn_sm100 | 40.4032 | 1.014 | — |
| `s2048_h32kv4` | tir | 56.3555 | flashattn_sm100 | 56.6166 | 1.005 | — |
| `s2048_h32kv4_causal` | tir | 35.5954 | flashattn_sm100 | 37.9587 | 1.066 | — |
| `s2048_h32kv8` | tir | 56.2651 | flashattn_sm100 | 56.5705 | 1.005 | — |
| `s2048_h32kv8_causal` | tir | 36.1257 | flashattn_sm100 | 37.8445 | 1.048 | — |
| `s4096_h32kv16` | tir | 213.4173 | flashattn_sm100 | 214.3931 | 1.005 | — |
| `s4096_h32kv16_causal` | tir | 112.9986 | flashattn_sm100 | 118.0209 | 1.044 | — |
| `s4096_h32kv32` | tir | 216.7032 | flashattn_sm100 | 220.5039 | 1.018 | — |
| `s4096_h32kv32_causal` | tir | 119.3166 | flashattn_sm100 | 120.5545 | 1.010 | — |
| `s4096_h32kv4` | tir | 209.4276 | flashattn_sm100 | 209.6753 | 1.001 | — |
| `s4096_h32kv4_causal` | tir | 109.7150 | flashattn_sm100 | 114.5406 | 1.044 | — |
| `s4096_h32kv8` | tir | 209.7788 | flashattn_sm100 | 211.4472 | 1.008 | — |
| `s4096_h32kv8_causal` | tir | 111.8660 | flashattn_sm100 | 117.7443 | 1.053 | — |
| `s8192_h32kv16` | tir | 772.2544 | flashattn_sm100 | 773.7900 | 1.002 | — |
| `s8192_h32kv16_causal` | tir | 414.9246 | flashattn_sm100 | 424.7156 | 1.024 | — |
| `s8192_h32kv32` | tir | 795.9493 | flashattn_sm100 | 798.4008 | 1.003 | — |
| `s8192_h32kv32_causal` | tir | 433.7231 | flashattn_sm100 | 433.6129 | 1.000 | — |
| `s8192_h32kv4` | tir | 780.6390 | flashattn_sm100 | 770.4900 | 0.987 | — |
| `s8192_h32kv4_causal` | tir | 412.2221 | flashattn_sm100 | 421.1178 | 1.022 | — |
| `s8192_h32kv8` | tir | 771.2404 | flashattn_sm100 | 775.7926 | 1.006 | — |
| `s8192_h32kv8_causal` | tir | 415.8717 | flashattn_sm100 | 428.5254 | 1.030 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.9013 | torch-cublas | 6.0055 | 0.870 | deepgemm-bf16=6.8659, deepgemm-cublaslt=6.0092 |
| `bf16_16384x16384x16384` | tir | 5628.1333 | torch-cublas | 5610.1962 | 0.997 | deepgemm-bf16=5976.2569, deepgemm-cublaslt=5613.8253 |
| `bf16_2048x2048x2048` | tir | 16.4056 | torch-cublas | 15.5389 | 0.947 | deepgemm-bf16=17.2813, deepgemm-cublaslt=15.5431 |
| `bf16_4096x4096x4096` | tir | 93.7913 | deepgemm-bf16 | 88.7105 | 0.946 | deepgemm-cublaslt=89.6318, torch-cublas=89.8241 |
| `bf16_8192x8192x8192` | tir | 695.5359 | deepgemm-cublaslt | 708.2811 | 1.018 | deepgemm-bf16=733.1626, torch-cublas=713.1888 |
| `fp16_1024x1024x1024` | tir | 6.8571 | torch-cublas | 5.9095 | 0.862 | — |
| `fp16_16384x16384x16384` | tir | 5753.1710 | torch-cublas | 5759.5303 | 1.001 | — |
| `fp16_2048x2048x2048` | tir | 16.4258 | torch-cublas | 15.8087 | 0.962 | — |
| `fp16_4096x4096x4096` | tir | 96.1417 | torch-cublas | 93.1778 | 0.969 | — |
| `fp16_8192x8192x8192` | tir | 727.8093 | torch-cublas | 759.2175 | 1.043 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.7469 | deepgemm | 49.6232 | 0.978 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 113.8359 | deepgemm | 113.0782 | 0.993 | — |
| `deepgemm_m4096_n32768_k512` | tir | 67.9943 | deepgemm | 71.2934 | 1.049 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 82.0894 | deepgemm | 82.2106 | 1.001 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.1172 | deepgemm | 19.0448 | 0.947 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 333.0388 | deepgemm | 330.8140 | 0.993 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 42.9358 | deepgemm | 42.3531 | 0.986 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 9622.9860 | cublas_nccl_cudagraph | 9198.5443 | 0.956 | cublasmp_split_p2p=10003.0208 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 599.1303 | cublas_nccl_cudagraph | 561.3896 | 0.937 | cublasmp_split_p2p=748.3171 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1500.3251 | cublas_nccl_cudagraph | 1431.1354 | 0.954 | cublasmp_split_p2p=1833.8248 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2434.1435 | cublas_nccl_cudagraph | 2403.8429 | 0.988 | cublasmp_split_p2p=2844.8883 |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 161.1956 | deepgemm | 168.0708 | 1.043 | — |
| `large_g4_m8192_n4096_k4096` | tir | 351.4085 | deepgemm | 361.4099 | 1.028 | — |
| `large_g4_m8192_n6144_k7168` | tir | 997.3748 | deepgemm | 1043.1590 | 1.046 | — |
| `large_g4_m8192_n7168_k3072` | tir | 507.5144 | deepgemm | 526.4080 | 1.037 | — |
| `large_g8_m4096_n4096_k2048` | tir | 189.1060 | deepgemm | 195.4129 | 1.033 | — |
| `large_g8_m4096_n4096_k4096` | tir | 354.3510 | deepgemm | 359.8891 | 1.016 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1071.8308 | deepgemm | 1095.4849 | 1.022 | — |
| `large_g8_m4096_n7168_k3072` | tir | 506.7036 | deepgemm | 535.0379 | 1.056 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 33.6535 | 36.9689 | 34.7909 | 56.0104 | 65.0726 |
| `moe_a3b_bs8_all` | 95.6029 | 99.9491 | 104.5142 | 129.9420 | 134.8303 |
| `moe_a3b_bs32_all` | 190.3210 | 192.1916 | 199.6875 | 223.8428 | 222.4272 |
| `moe_a3b_bs128_all` | 223.4190 | 219.2493 | 227.1432 | 245.0598 | 248.4439 |
| `moe_a3b_bs512_all` | 237.7760 | 230.4128 | 243.2862 | 288.8539 | 276.8774 |
| `moe_a3b_bs1024_all` | 260.3320 | 256.4390 | 274.0187 | 335.9667 | 311.6647 |
| `moe_a3b_bs2048_all` | 338.4892 | 339.4944 | 357.9046 | 431.9871 | 391.4657 |
| `moe_a3b_bs4096_all` | 563.0525 | 558.4133 | 573.4422 | 665.7490 | 606.7889 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.3781 | cublaslt_nvfp4 | 4.4346 | 0.825 | flashinfer=4.5210 |
| `16384x16384x16384` | tir | 1489.9813 | flashinfer | 1444.8755 | 0.970 | cublaslt_nvfp4=1466.0304 |
| `2048x2048x2048` | tir | 8.7511 | flashinfer | 7.4529 | 0.852 | cublaslt_nvfp4=7.5843 |
| `4096x4096x4096` | tir | 29.4367 | cublaslt_nvfp4 | 27.6113 | 0.938 | flashinfer=28.7620 |
| `8192x8192x8192` | tir | 185.4257 | flashinfer | 176.1684 | 0.950 | cublaslt_nvfp4=179.2211 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1725.6573 | flashmla | 1757.7074 | 1.019 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1888.3586 | flashmla | 1895.1349 | 1.004 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1698.9118 | flashmla | 1712.2236 | 1.008 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1794.0651 | flashmla | 1817.7636 | 1.013 | trtllm_gen=2054.2305 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2021.1135 | flashmla | 2009.2543 | 0.994 | trtllm_gen=2203.1653 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1793.0457 | flashmla | 1816.3394 | 1.013 | trtllm_gen=2050.9292 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1148.6985 | flashmla | 1157.5796 | 1.008 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1197.4755 | flashmla | 1218.7120 | 1.018 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1158.1683 | flashmla | 1159.3792 | 1.001 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 373.9933 | flashmla | 379.0438 | 1.014 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 373.3247 | flashmla | 383.1354 | 1.026 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 382.1048 | flashmla | 389.4611 | 1.019 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 363.6449 | flashmla | 373.5008 | 1.027 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 383.8314 | flashmla | 395.1274 | 1.029 | trtllm_gen=461.2470 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 392.5180 | flashmla | 403.6945 | 1.028 | trtllm_gen=474.9567 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 402.3719 | flashmla | 416.8263 | 1.036 | trtllm_gen=483.9187 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 375.3681 | flashmla | 383.5837 | 1.022 | trtllm_gen=452.4132 |
