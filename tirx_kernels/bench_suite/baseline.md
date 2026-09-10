# bench-suite baseline view: `baseline.json`

- Timestamp: `4`
- Label:     `f727de3d-dirty`
- Git:       `{'tir': '73e38d3f', 'tirx-kernels': 'f727de3d-dirty', 'tirx-bench-ci': None}`
- Workloads: 182 ok, 9 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## act_and_mul

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gelu_tanh_fp16_d11008_t8192` | tirx | 97.6374 | flashinfer | 109.4522 | 1.121 | — |
| `silu_bf16_d16384_t32768` | tirx | 617.3432 | flashinfer | 1296.4386 | 2.100 | — |
| `silu_fp16_d4096_t1` | tirx | 2.0772 | flashinfer | 2.2879 | 1.101 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 37.5576 | deepgemm | 39.7304 | 1.058 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 183.7198 | deepgemm | 187.3286 | 1.020 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.3495 | deepgemm | 6.5476 | 1.031 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.0313 | deepgemm | 4.7520 | 1.179 | — |

## deepgemm_sm100_fp8_bmm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bhd_bhr_hdr_b4096_h8_r4096_d1024` | tirx | 136.5538 | deepgemm | 139.1567 | 1.019 | — |
| `bhd_hdr_bhr_b8192_h8_r4096_d1024` | tirx | 222.8038 | deepgemm | 244.7085 | 1.098 | — |
| `bhr_hdr_bhd_b4096_h8_r4096_d1024` | tirx | 99.4255 | deepgemm | 99.3295 | 0.999 | — |

## deepgemm_sm100_fp8_gemm_1d1d

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m4096_n4096_k7168_bfp4` | tirx | 130.6360 | deepgemm | 91.0264 | 0.697 | — |
| `m4096_n576_k7168` | tirx | 18.8025 | deepgemm | 19.0987 | 1.016 | — |
| `m4096_n7168_k16384` | tirx | 336.4438 | deepgemm | 325.5337 | 0.968 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 47.2485 | deepgemm | 54.2831 | 1.149 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 176.4521 | deepgemm | 186.8705 | 1.059 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.7530 | deepgemm | 6.8781 | 1.019 | sglang_cutedsl=6.9006 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2384 | sglang_cutedsl | 4.7356 | 1.117 | deepgemm=5.0161 |

## deepgemm_sm100_k_grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g16_m7168_n2048_k2048_gran128_al128` | tirx | 912.2861 | deepgemm | 918.9562 | 1.007 | — |
| `g4_m4096_n7168_k8192_gran128_al128_psum` | tirx | 857.4816 | deepgemm | 860.4166 | 1.003 | — |
| `g8_m4096_n7168_k4096_gran32_al160` | tirx | 907.9104 | deepgemm | 930.4703 | 1.025 | — |

## deepgemm_sm100_m_grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g4_m8192_n6144_k7168` | tirx | 1000.2049 | deepgemm | 1029.7921 | 1.030 | — |
| `g8_m4096_n4096_k2048_bfp4` | tirx | 186.0463 | deepgemm | 187.4825 | 1.008 | — |
| `g8_m4096_n7168_k3072_psum_zp` | tirx | 511.4467 | deepgemm | 507.7964 | 0.993 | — |

## deepgemm_sm100_m_grouped_fp8_gemm_masked

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g32_m192_n4096_k4096_bfp4` | tirx | 115.2601 | deepgemm | 114.5373 | 0.994 | — |
| `g32_m192_n6144_k7168` | tirx | 338.7765 | deepgemm | 326.7344 | 0.964 | — |
| `g6_m1024_n4096_k2048` | tirx | 39.1687 | deepgemm | 39.2882 | 1.003 | — |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.2401 | deepgemm | 5.2343 | 0.999 | — |
| `m4096_n24_k7168_s1` | tirx | 23.1650 | deepgemm | 23.8279 | 1.029 | — |
| `m8192_n24_k28672_s1` | tirx | 84.4301 | deepgemm | 92.6280 | 1.097 | — |

## fast_topk_clusters

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_plain_b16_l4096_k256` | tirx | 6.6451 | flashinfer | 7.2568 | 1.092 | — |
| `f32_plain_b64_l16384_k256` | tirx | 13.7256 | flashinfer | 14.2128 | 1.035 | — |
| `f32_plain_b64_l65536_k1024` | tirx | 20.8370 | flashinfer | 21.3537 | 1.025 | — |

## filtered_topk

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_plain_det_r2_l524288_k256_endbit` | tirx | 114.9716 | flashinfer | 125.9324 | 1.095 | — |
| `f32_plain_r4_l8192_k256` | tirx | 7.7729 | flashinfer | 9.0389 | 1.163 | — |
| `f32_plain_r64_l8192_k256` | tirx | 8.4706 | flashinfer | 9.7603 | 1.152 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv4` | tir | 19.3743 | flashattn_sm100 | 20.0717 | 1.036 | — |
| `s4096_h32kv4_causal` | tir | 112.4341 | flashattn_sm100 | 115.8063 | 1.030 | — |
| `s8192_h32kv32` | tir | 778.7393 | flashattn_sm100 | 785.1523 | 1.008 | — |

## flash_attention_backward_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_s2048_h16_causal` | tir | 82.1407 | flashattn_sm100 | 82.1089 | 1.000 | — |
| `b1_s8192_h16_noncausal` | tir | 1081.8105 | flashattn_sm100 | 1092.8711 | 1.010 | — |
| `b4_s8192_h16_noncausal` | tir | 4370.6380 | flashattn_sm100 | 4471.5627 | 1.023 | — |

## flashinfer_add_rmsnorm_fp4quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_nv_3d_bf16_b32_s32_h128_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 2.9600 | flashinfer_cutedsl | 2.8960 | 0.978 | — |
| `bench_nv_bf16_m32_h4096_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 5.8104 | flashinfer_cutedsl | 5.8124 | 1.000 | — |
| `bench_nv_large_bf16_m64_h8192_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 6.0163 | flashinfer_cutedsl | 6.0226 | 1.001 | — |

## flashinfer_fused_add_rmsnorm_quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_e4m3_m32_h4096_xc_rc_yc_pdl0_s1` | tirx | 3.5368 | flashinfer_cutedsl | 3.5666 | 1.008 | — |
| `bf16_e4m3_m32_h4096_xc_rc_yc_pdl1_s1` | tirx | 3.5717 | flashinfer_cutedsl | 3.6109 | 1.011 | — |
| `bf16_e4m3_m64_h8192_xc_rc_yc_pdl0_s1` | tirx | 3.7283 | flashinfer_cutedsl | 3.7940 | 1.018 | — |

## flashinfer_fused_dit_layernorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `grgb_bf16_b1_r1920` | tirx | 16.6008 | flashinfer_cuda | 16.8442 | 1.015 | — |
| `grss_bf16_b4_r1920` | tirx | 72.9835 | flashinfer_cuda | 73.4669 | 1.007 | — |
| `rss_bf16_b1_r768` | tirx | 8.7467 | flashinfer_cuda | 8.6655 | 0.991 | — |

## flashinfer_layernorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_m128_h1024_xc_yc_pdl0_eps1e6` | tirx | 3.1305 | flashinfer_cutedsl | 3.1160 | 0.995 | — |
| `bf16_m128_h16384_xc_yc_pdl0_eps1e6` | tirx | 8.9714 | flashinfer_cutedsl | 8.9680 | 1.000 | — |
| `bf16_m1_h128_xc_yc_pdl0_eps1e6` | tirx | 2.2494 | flashinfer_cutedsl | 2.2566 | 1.003 | — |

## flashinfer_qk_rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gemma_bf16_b32_n32_h128_xc_yc_pdl0` | tirx | 2.7697 | flashinfer_cutedsl | 2.8056 | 1.013 | — |
| `rms_bf16_b32_n32_h128_xc_yc_pdl0` | tirx | 2.6324 | flashinfer_cutedsl | 2.6516 | 1.007 | — |
| `rms_f16_b16_n64_h128_xc_yc_pdl0` | tirx | 2.5852 | flashinfer_cutedsl | 2.5963 | 1.004 | — |

## flashinfer_rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gemma_bf16_m64_h8192_xc_yc_pdl0` | tirx | 3.4718 | flashinfer_cutedsl | 3.5704 | 1.028 | — |
| `rms_bf16_m32_h4096_xc_yc_pdl0` | tirx | 3.2981 | flashinfer_cutedsl | 3.4361 | 1.042 | — |
| `rms_bf16_m32_h4096_xc_yc_pdl1` | tirx | 3.3285 | flashinfer_cutedsl | 3.3229 | 0.998 | — |

## flashinfer_rmsnorm_fp4quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_nv_3d_bf16_b32_s32_h128_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 2.7762 | flashinfer_cutedsl | 2.7551 | 0.992 | — |
| `bench_nv_bf16_m32_h4096_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 4.8426 | flashinfer_cutedsl | 4.8704 | 1.006 | — |
| `bench_nv_large_bf16_m64_h8192_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 4.9428 | flashinfer_cutedsl | 4.9394 | 0.999 | — |

## flashinfer_rmsnorm_quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_e4m3_m32_h4096_xc_yc_pdl0_s1` | tirx | 3.2766 | flashinfer_cutedsl | 3.2204 | 0.983 | — |
| `bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | tirx | 3.4049 | flashinfer_cutedsl | 3.3867 | 0.995 | — |
| `bf16_e5m2_m3_h1048576_xc_yc_pdl1_s1_cluster16_sync` | tirx | 16.6427 | flashinfer_cutedsl | 16.5748 | 0.996 | — |

## flashkda_bf16_fused_m128

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `h64_mixed` | tirx | 267.8367 | flashinfer_m128 | 272.7128 | 1.018 | flashkda_raw=665.3760 |
| `h96_fixed8192` | tirx | 500.9243 | flashinfer_m128 | 506.0955 | 1.010 | flashkda_raw=1073.3854 |
| `h96_uniform` | tirx | 434.0923 | flashinfer_m128 | 436.1191 | 1.005 | flashkda_raw=705.6305 |

## flashkda_decode_t1_precomputed

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv16h16_b128_s8` | tirx | 26.6199 | flashinfer_cake | 31.5408 | 1.185 | — |
| `hv16h16_b1_s16` | tirx | 4.1159 | flashinfer_cake | 5.8536 | 1.422 | — |
| `hv32h16_b32_s8` | tirx | 15.2390 | flashinfer_cake | 19.4038 | 1.273 | — |

## flashkda_decode_t2_precomputed

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv12h12_b8_t2` | tirx | 6.3762 | flashinfer_cake | 8.7845 | 1.378 | — |
| `hv16h16_b64_t2` | tirx | 25.1064 | flashinfer_cake | 30.2963 | 1.207 | — |
| `hv32h16_b128_t2` | tirx | 81.6001 | flashinfer_cake | 95.2481 | 1.167 | — |

## flashkda_decode_t3_lower_bound

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv16h16_b16_t3` | tirx | 13.4194 | flashinfer_cake | 14.0665 | 1.048 | — |
| `hv16h16_b1_t3` | tirx | 5.5565 | flashinfer_cake | 6.0026 | 1.080 | — |
| `hv16h16_b4_t3` | tirx | 6.6822 | flashinfer_cake | 6.9711 | 1.043 | — |

## flashkda_decode_t4_precomputed

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv12h12_b8_t4` | tirx | 8.7613 | flashinfer_cake | 11.0596 | 1.262 | — |
| `hv16h16_b64_t4` | tirx | 39.3332 | flashinfer_cake | 42.0130 | 1.068 | — |
| `hv32h16_b128_t4` | tirx | 132.0186 | flashinfer_cake | 137.9693 | 1.045 | — |

## flashkda_decode_t5_gram

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv32h16_b128_s1` | tirx | 158.9776 | flashinfer_cake | 160.0385 | 1.007 | — |
| `hv32h16_b1_s8` | tirx | 7.0295 | flashinfer_cake | 9.3121 | 1.325 | — |
| `hv32h16_b3_s4` | tirx | 8.5994 | flashinfer_cake | 10.2351 | 1.190 | — |

## flashkda_decode_t6_gram

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv32h16_b128_s1` | tirx | 181.6806 | flashinfer_cake | 195.8678 | 1.078 | — |
| `hv32h16_b1_s8` | tirx | 7.2978 | flashinfer_cake | 8.7352 | 1.197 | — |
| `hv32h16_b3_s4` | tirx | 14.0607 | flashinfer_cake | 16.2512 | 1.156 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_4096x4096x4096` | tir | 92.4952 | deepgemm-bf16 | 89.9498 | 0.972 | deepgemm-cublaslt=90.7190, torch-cublas=90.3239 |
| `fp16_1024x1024x1024` | tir | 6.6257 | torch-cublas | 5.8395 | 0.881 | — |
| `fp16_16384x16384x16384` | tir | 5702.0394 | torch-cublas | 5612.5350 | 0.984 | — |

## gdn_cp_prefill_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_q16_k16_v16_s4096+4096_init_f16_i64` | tirx | 117.7848 | flashinfer_cutedsl | 128.4645 | 1.091 | — |
| `fp16_q16_k16_v64_s192+64_initfinal_f16_i64` | tirx | 82.5471 | flashinfer_cutedsl | 83.7373 | 1.014 | — |
| `fp16_q1_k1_v1_s2048_none_i32` | tirx | 53.7819 | flashinfer_cutedsl | 67.5602 | 1.256 | — |

## gdn_decode_bf16_ilp4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t1_b1_h2_hv4_tv16` | tirx | 3.0704 | flashinfer_cutedsl | 3.2716 | 1.066 | — |
| `t4_b8_h4_hv8_tv16` | tirx | 6.6630 | flashinfer_cutedsl | 7.1216 | 1.069 | — |
| `t8_b4_h8_hv16_tv16` | tirx | 10.3861 | flashinfer_cutedsl | 11.7190 | 1.128 | — |

## gdn_decode_bf16_wide_vec_mtp

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t2_b4_h16_hv32_tv32` | tirx | 6.3312 | flashinfer_cutedsl | 6.8442 | 1.081 | — |
| `t4_b64_h8_hv16_tv128` | tirx | 33.8103 | flashinfer_cutedsl | 46.7374 | 1.382 | — |
| `t8_b512_h16_hv32_tv128` | tirx | 1163.7200 | flashinfer_cutedsl | 1330.0720 | 1.143 | — |

## gdn_decode_bf16_wide_vec_t1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b128_h8_hv16_tv128` | tirx | 24.8936 | flashinfer_cutedsl | 25.6358 | 1.030 | — |
| `b16_h16_hv32_tv64` | tirx | 11.2524 | flashinfer_cutedsl | 20.2887 | 1.803 | — |
| `b512_h4_hv8_tv128` | tirx | 46.1512 | flashinfer_cutedsl | 47.0845 | 1.020 | — |

## gdn_decode_fp32_mtp_warp

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t2_b4_h16_hv64_tv16_ilp2_sv0` | tirx | 14.3412 | flashinfer_cutedsl | 15.8320 | 1.104 | — |
| `t8_b256_h16_hv64_tv64_ilp4_sv1` | tirx | 1773.3917 | flashinfer_cutedsl | 1778.8057 | 1.003 | — |

## gdn_prefill_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hq16_hv64_s1x8192` | tirx | 240.6253 | flashinfer_cutedsl | 250.7749 | 1.042 | — |
| `hq32_hv32_s8192x16` | tirx | 1084.1739 | flashinfer_cutedsl | 1109.6317 | 1.023 | — |
| `hq8_hv32_s1024x8` | tirx | 159.0958 | flashinfer_cutedsl | 187.6231 | 1.179 | — |

## msa_sparse_atten_fwd_nvfp4_kv_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `ring48k_bf16q_qh16_t16` | tirx | 1029.3983 | msa | 1387.7469 | 1.348 | — |

## msa_sparse_atten_fwd_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `ring48k_bf16_qh16_t16` | tirx | 1361.0732 | msa | 1832.0996 | 1.346 | — |
| `ring48k_fp8_qh16_t16` | tirx | 1334.7881 | msa | 1526.8879 | 1.144 | — |

## msa_sparse_prepare_flat_schedule_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `decode_b128_k65536_h4` | tirx | 8694.0625 | msa | 12213.3396 | 1.405 | — |
| `prefill_b1_k8192_h2` | tirx | 3.8856 | msa | 4.2417 | 1.092 | — |

## msa_sparse_prepare_fwd_split_atomic_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `decode_b128_k65536_h4` | tirx | 216.8929 | msa | 219.2295 | 1.011 | — |
| `prefill_b1_k131072_h1` | tirx | 33.1634 | msa | 33.3473 | 1.006 | — |
| `prefill_b1_k8192_h2` | tirx | 7.1920 | msa | 7.3364 | 1.020 | — |

## mxfp4_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.4907 | flashinfer | 2.5397 | 1.020 | — |
| `fp16_128x4_m16384_k7168` | tirx | 52.5001 | flashinfer | 52.8837 | 1.007 | — |
| `fp16_linear_m4096_k4096` | tirx | 10.2822 | flashinfer | 10.6286 | 1.034 | — |

## mxfp8_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.6790 | flashinfer | 2.7309 | 1.019 | — |
| `fp16_128x4_m16384_k7168` | tirx | 60.6338 | flashinfer | 60.5192 | 0.998 | — |
| `fp16_linear_m4096_k4096` | tirx | 11.3160 | flashinfer | 11.2650 | 0.995 | — |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.2106 | cublaslt_nvfp4 | 4.3806 | 0.841 | flashinfer=4.4708 |
| `16384x16384x16384` | tir | 1528.9627 | flashinfer | 1443.7832 | 0.944 | cublaslt_nvfp4=1445.3852 |
| `4096x4096x4096` | tir | 29.3046 | cublaslt_nvfp4 | 27.3885 | 0.935 | flashinfer=28.5923 |

## nvfp4_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.4972 | flashinfer | 2.5140 | 1.007 | — |
| `fp16_128x4_m16384_k7168` | tirx | 55.2725 | flashinfer | 54.8886 | 0.993 | — |
| `fp16_linear_m4096_k4096` | tirx | 10.2527 | flashinfer | 10.3761 | 1.012 | — |

## nvfp4_quantize_per_token

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.6892 | flashinfer | 2.8470 | 1.059 | — |
| `fp16_128x4_m16384_k7168` | tirx | 57.7658 | flashinfer | 59.1917 | 1.025 | — |
| `fp16_linear_m4096_k4096` | tirx | 12.1103 | flashinfer | 12.9942 | 1.073 | — |

## radix_topk_multi_cta

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_basic_r2_l524288_k256_large` | tirx | 40.0140 | flashinfer | 42.4567 | 1.061 | — |
| `f32_basic_r4_l115188_k256_ctas3` | tirx | 35.3704 | flashinfer | 35.9263 | 1.016 | — |
| `f32_basic_r4_l57596_k256_vec4` | tirx | 30.1728 | flashinfer | 31.8387 | 1.055 | — |

## radix_topk_single_cta

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_basic_r256_l57592_k1024_maxchunk` | tirx | 80.2137 | flashinfer | 80.4478 | 1.003 | — |
| `f32_basic_r64_l32768_k512` | tirx | 23.1822 | flashinfer | 25.7455 | 1.111 | — |
| `f32_basic_r8_l8192_k256` | tirx | 11.1426 | flashinfer | 12.5080 | 1.123 | — |

## recurrent_kda_decode_grouped

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `dec_hv16_b1` | tirx | 3.3088 | flashinfer_cutedsl | 3.5166 | 1.063 | — |
| `ver_t8_hv12_b16` | tirx | 22.1145 | flashinfer_cutedsl | 31.1857 | 1.410 | — |
| `ver_t8_hv16_b128` | tirx | 161.2211 | flashinfer_cutedsl | 222.8499 | 1.382 | — |

## recurrent_kda_decode_one_warp

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv12_b64_tr16_lb` | tirx | 12.6430 | flashinfer_cutedsl | 13.8379 | 1.095 | — |
| `hv16_b128_tr16_lb` | tirx | 26.6607 | flashinfer_cutedsl | 29.8146 | 1.118 | — |
| `hv16_b8_tr8_lb` | tirx | 5.2474 | flashinfer_cutedsl | 5.4464 | 1.038 | — |

## rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hs128_bs32` | tir | 2.3129 | flashinfer | 2.0438 | 0.884 | — |
| `hs4096_bs128` | tir | 3.5200 | flashinfer | 3.4994 | 0.994 | — |
| `hs8192_bs4113` | tir | 71.7585 | flashinfer | 23.9442 | 0.334 | — |

## selective_state_update_mtp_horizontal

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 6.9503 | flashinfer_cuda | 7.4360 | 1.070 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 1658.4259 | flashinfer_cuda | 1814.7010 | 1.094 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 347.0377 | flashinfer_cuda | 382.4105 | 1.102 | — |

## selective_state_update_mtp_simple

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 4.3164 | flashinfer_cuda | 5.1695 | 1.198 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 1494.9751 | flashinfer_cuda | 1584.5818 | 1.060 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 378.0140 | flashinfer_cuda | 400.7631 | 1.060 | — |

## selective_state_update_mtp_vertical

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 17.2645 | flashinfer_cuda | 18.1947 | 1.054 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 2689.2282 | flashinfer_cuda | 2903.4872 | 1.080 | — |

## selective_state_update_stp_horizontal

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 47.8065 | flashinfer_cuda | 49.7971 | 1.042 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 26.5000 | flashinfer_cuda | 30.0204 | 1.133 | — |
| `b64_h64_d64_s256_r8` | tirx | 47.4491 | flashinfer_cuda | 47.3090 | 0.997 | — |

## selective_state_update_stp_simple

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 65.2790 | flashinfer_cuda | 67.5019 | 1.034 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 37.3374 | flashinfer_cuda | 38.1243 | 1.021 | — |
| `b64_h64_d64_s256_r8` | tirx | 54.0660 | flashinfer_cuda | 61.2385 | 1.133 | — |

## selective_state_update_stp_vertical

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 47.3534 | flashinfer_cuda | 54.2497 | 1.146 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 27.0785 | flashinfer_cuda | 31.3608 | 1.158 | — |
| `b64_h64_d64_s256_r8` | tirx | 46.2417 | flashinfer_cuda | 48.7500 | 1.054 | — |

## silu_and_mul_nvfp4_experts_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_b8_m512_k2048` | tirx | 9.0482 | flashinfer | 10.8425 | 1.198 | — |
| `fp16_b128_m2048_k2048` | tirx | 273.8417 | flashinfer | 301.9139 | 1.103 | — |
| `fp16_b8_m16_k2048` | tirx | 3.3409 | flashinfer | 4.1893 | 1.254 | — |

## sm100_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1326.6000 | deepgemm | 1308.0000 | 0.986 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3359.2000 | deepgemm | 3363.4000 | 1.001 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1_s1` | tirx | 3742.4000 | deepgemm | 3746.0000 | 1.001 | — |

## sparse_flashmla_decode_head64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | tirx | 136.7393 | flashmla | 144.7348 | 1.058 | — |
| `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 16.7063 | flashmla | 21.1248 | 1.264 | — |
| `v32_b148_sq2_sk32768_topk16384_p64` | tirx | 1526.2413 | flashmla | 1817.5149 | 1.191 | — |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1813.1113 | flashmla | 1871.2985 | 1.032 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1676.5177 | flashmla | 1732.7519 | 1.034 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1775.0254 | flashmla | 1816.8810 | 1.024 | trtllm_gen=2033.6984 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1126.8128 | flashmla | 1152.5680 | 1.023 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1196.7294 | flashmla | 1216.7862 | 1.017 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1191.6612 | flashmla | 1205.0408 | 1.011 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 381.8447 | flashmla | 390.1592 | 1.022 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 366.5173 | flashmla | 372.7173 | 1.017 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 861.8450 | flashmla | 763.2616 | 0.886 | trtllm_gen=775.0955 |

## tinygemm2_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_o2880_k2880` | tirx | 7.9359 | flashinfer_sm100 | 8.0420 | 1.013 | — |
| `b1_o128_k720` | tirx | 2.9564 | flashinfer_sm100 | 2.9851 | 1.010 | — |
| `b64_o4096_k3072` | tirx | 21.9170 | flashinfer_sm100 | 22.0918 | 1.008 | — |

## Failed (9)

- `gdn_decode_fp32_mtp_warp/t4_b64_h8_hv32_tv64_ilp4_sv1`: prepared child exited None without a terminal message
- `msa_sparse_atten_fwd_nvfp4_kv_sm100/ring48k_fp8q_qh16_t16`: baseline error(s): msa: DSLRuntimeError: 🧊🧊🧊 ICE 🧊🧊🧊
- `msa_sparse_atten_fwd_nvfp4_kv_sm100/varlen_b3_s8192_bf16q_qh4_t16`: baseline error(s): msa: DSLRuntimeError: 🧊🧊🧊 ICE 🧊🧊🧊
- `msa_sparse_atten_fwd_sm100/varlen_b3_s8192_qh4_t16`: baseline error(s): msa: DSLRuntimeError: 🧊🧊🧊 ICE 🧊🧊🧊
- `msa_sparse_prepare_flat_schedule_sm100/decode_b64_k16384_h4_varlen`: baseline error(s): msa: partially initialized module 'torch._dynamo' has no attribute 'decorators' (most likely due to a circular import)
- `selective_state_update_mtp_vertical/b512_h64_d64_s128_t6_r8_statebf16_official`: prepared child exited None without a terminal message
- `stable_sort_topk_by_value/f32_r4_k128`: prepare: RuntimeError: CPU prepare changed CUDA initialization state from False to True
- `stable_sort_topk_by_value/f32_r64_k2048`: prepare: RuntimeError: CPU prepare changed CUDA initialization state from False to True
- `stable_sort_topk_by_value/f32_r64_k256`: prepare: RuntimeError: CPU prepare changed CUDA initialization state from False to True

## Jetson AGX Thor (`sm_110a`)

### agent_evolved_kda_forward_b1_t8192

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `h96_fixed` | tirx | 6531.8080 | flash_kda | 16013.1360 | 2.452 | — |
| `h96_uniform` | tirx | 6719.4230 | flash_kda | 17103.4660 | 2.545 | — |
| `h64_uniform` | tirx | 4590.0820 | flash_kda | 11915.7180 | 2.596 | — |

### cudnn_sm100_gdn_prefill_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b16_s8192_h64_nostate` | tirx | 48456.4055 | cudnn_frontend | 52156.1827 | 1.076 | — |
| `perf_b16_s8192_h64_state` | tirx | 71834.9177 | cudnn_frontend | 75027.4035 | 1.044 | — |
| `perf_b1_s8192_h64_nostate` | tirx | 3373.5589 | cudnn_frontend | 3566.0714 | 1.057 | — |
| `perf_b1_s8192_h64_state` | tirx | 5417.8802 | cudnn_frontend | 5562.0063 | 1.027 | — |
| `perf_b2_s8192_h64_nostate` | tirx | 6463.8619 | cudnn_frontend | 6917.2765 | 1.070 | — |
| `perf_b2_s8192_h64_state` | tirx | 10431.7077 | cudnn_frontend | 10610.2611 | 1.017 | — |
| `perf_b4_s16384_h64_nostate` | tirx | 25375.6592 | cudnn_frontend | 27125.7366 | 1.069 | — |
| `perf_b4_s16384_h64_state` | tirx | 41456.0619 | cudnn_frontend | 43139.1787 | 1.041 | — |
| `perf_b4_s2048_h64_nostate` | tirx | 3392.6928 | cudnn_frontend | 3608.9931 | 1.064 | — |
| `perf_b4_s2048_h64_state` | tirx | 5434.0191 | cudnn_frontend | 5449.2560 | 1.003 | — |
| `perf_b4_s32768_h64_nostate` | tirx | 47628.0536 | cudnn_frontend | 48091.0625 | 1.010 | — |
| `perf_b4_s32768_h64_state` | tirx | 73405.5111 | cudnn_frontend | 76815.0665 | 1.046 | — |
| `perf_b4_s4096_h64_nostate` | tirx | 6533.7592 | cudnn_frontend | 6961.6331 | 1.065 | — |
| `perf_b4_s4096_h64_state` | tirx | 10595.4457 | cudnn_frontend | 10616.4183 | 1.002 | — |
| `perf_b4_s8192_h64_nostate` | tirx | 12767.4998 | cudnn_frontend | 13657.9609 | 1.070 | — |
| `perf_b4_s8192_h64_state` | tirx | 20791.4739 | cudnn_frontend | 21065.2219 | 1.013 | — |
| `perf_b8_s8192_h64_nostate` | tirx | 25599.3496 | cudnn_frontend | 27247.8958 | 1.064 | — |
| `perf_b8_s8192_h64_state` | tirx | 41343.4019 | cudnn_frontend | 42291.6150 | 1.023 | — |

### cudnn_sm100_gdn2_prefill_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b16_s8192_h64_nostate` | tirx | 78938.2119 | cudnn_frontend | 78879.5303 | 0.999 | — |
| `perf_b16_s8192_h64_state` | tirx | 180406.9697 | cudnn_frontend | 181084.9209 | 1.004 | — |
| `perf_b1_s8192_h64_nostate` | tirx | 6255.8655 | cudnn_frontend | 6251.1093 | 0.999 | — |
| `perf_b1_s8192_h64_state` | tirx | 13170.8713 | cudnn_frontend | 13347.0925 | 1.013 | — |
| `perf_b2_s8192_h64_nostate` | tirx | 12370.3989 | cudnn_frontend | 12396.8920 | 1.002 | — |
| `perf_b2_s8192_h64_state` | tirx | 26657.2028 | cudnn_frontend | 26803.9108 | 1.006 | — |
| `perf_b4_s16384_h64_nostate` | tirx | 48496.3097 | cudnn_frontend | 48350.7485 | 0.997 | — |
| `perf_b4_s16384_h64_state` | tirx | 90927.5278 | cudnn_frontend | 91365.3754 | 1.005 | — |
| `perf_b4_s2048_h64_nostate` | tirx | 6435.7558 | cudnn_frontend | 6380.4742 | 0.991 | — |
| `perf_b4_s2048_h64_state` | tirx | 13426.7858 | cudnn_frontend | 13415.9042 | 0.999 | — |
| `perf_b4_s32768_h64_nostate` | tirx | 79049.2524 | cudnn_frontend | 79937.7245 | 1.011 | — |
| `perf_b4_s32768_h64_state` | tirx | 181559.6007 | cudnn_frontend | 183771.2308 | 1.012 | — |
| `perf_b4_s4096_h64_nostate` | tirx | 12451.8207 | cudnn_frontend | 12536.4522 | 1.007 | — |
| `perf_b4_s4096_h64_state` | tirx | 26711.5806 | cudnn_frontend | 26665.7060 | 0.998 | — |
| `perf_b4_s8192_h64_nostate` | tirx | 24713.4659 | cudnn_frontend | 24845.8069 | 1.005 | — |
| `perf_b4_s8192_h64_state` | tirx | 51844.8926 | cudnn_frontend | 51587.0895 | 0.995 | — |
| `perf_b8_s8192_h64_nostate` | tirx | 48076.4737 | cudnn_frontend | 49628.3408 | 1.032 | — |
| `perf_b8_s8192_h64_state` | tirx | 90596.7627 | cudnn_frontend | 91286.9752 | 1.008 | — |

### cudnn_sm100_gdn2_recompute_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b16_s8192_h64_nostate` | tirx | 155070.8620 | cudnn_frontend | 159703.9119 | 1.030 | — |
| `perf_b16_s8192_h64_state` | tirx | 156298.5425 | cudnn_frontend | 161099.5008 | 1.031 | — |
| `perf_b1_s8192_h64_nostate` | tirx | 11509.1100 | cudnn_frontend | 11647.3288 | 1.012 | — |
| `perf_b1_s8192_h64_state` | tirx | 11614.8180 | cudnn_frontend | 11779.7155 | 1.014 | — |
| `perf_b2_s8192_h64_nostate` | tirx | 23095.8568 | cudnn_frontend | 23228.4264 | 1.006 | — |
| `perf_b2_s8192_h64_state` | tirx | 23240.6585 | cudnn_frontend | 23429.0805 | 1.008 | — |
| `perf_b4_s16384_h64_nostate` | tirx | 77818.0166 | cudnn_frontend | 79805.2748 | 1.026 | — |
| `perf_b4_s16384_h64_state` | tirx | 77934.0186 | cudnn_frontend | 79998.0010 | 1.026 | — |
| `perf_b4_s2048_h64_nostate` | tirx | 11353.1798 | cudnn_frontend | 11619.6198 | 1.023 | — |
| `perf_b4_s2048_h64_state` | tirx | 11764.0243 | cudnn_frontend | 11979.3038 | 1.018 | — |
| `perf_b4_s32768_h64_nostate` | tirx | 155591.2415 | cudnn_frontend | 159374.3670 | 1.024 | — |
| `perf_b4_s32768_h64_state` | tirx | 156032.5627 | cudnn_frontend | 159735.5274 | 1.024 | — |
| `perf_b4_s4096_h64_nostate` | tirx | 22992.7032 | cudnn_frontend | 23278.9249 | 1.012 | — |
| `perf_b4_s4096_h64_state` | tirx | 23258.3996 | cudnn_frontend | 23580.7693 | 1.014 | — |
| `perf_b4_s8192_h64_nostate` | tirx | 45392.1619 | cudnn_frontend | 46261.4040 | 1.019 | — |
| `perf_b4_s8192_h64_state` | tirx | 45835.6203 | cudnn_frontend | 46391.9832 | 1.012 | — |
| `perf_b8_s8192_h64_nostate` | tirx | 77944.0310 | cudnn_frontend | 79614.7813 | 1.021 | — |
| `perf_b8_s8192_h64_state` | tirx | 77929.7797 | cudnn_frontend | 80522.5522 | 1.033 | — |

### fast_topk_clusters

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_plain_b16_l4096_k256` | tirx | 10.9391 | flashinfer | 11.1809 | 1.022 | — |
| `f32_plain_b64_l16384_k256` | tirx | 67.7273 | flashinfer | 192.1225 | 2.837 | — |
| `f32_plain_b64_l65536_k1024` | tirx | 226.0682 | flashinfer | 304.5217 | 1.347 | — |

### filtered_topk

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_plain_det_r2_l524288_k256_endbit` | tirx | 143.5166 | flashinfer | 162.3314 | 1.131 | — |
| `f32_plain_r4_l8192_k256` | tirx | 10.4883 | flashinfer | 12.1341 | 1.157 | — |
| `f32_plain_r64_l8192_k256` | tirx | 39.3581 | flashinfer | 45.6034 | 1.159 | — |

### flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv4` | tir | 155.3829 | flashattn_sm100 | 165.1326 | 1.063 | — |
| `s4096_h32kv4_causal` | tir | 890.7954 | flashattn_sm100 | 929.1618 | 1.043 | — |
| `s8192_h32kv32` | tir | 8086.6973 | flashattn_sm100 | 8267.8324 | 1.022 | — |

### flash_attention_backward_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_s2048_h16_causal` | tir | 936.8841 | flashattn_sm100 | 937.3432 | 1.000 | — |
| `b1_s4096_h16_causal` | tir | 2302.5478 | flashattn_sm100 | 2339.1055 | 1.016 | — |
| `b1_s8192_h16_causal` | tir | 7479.3079 | flashattn_sm100 | 7585.5064 | 1.014 | — |
| `b1_s8192_h16_noncausal` | tir | 12660.2311 | flashattn_sm100 | 13064.2535 | 1.032 | — |
| `b2_s4096_h16_causal` | tir | 4714.6003 | flashattn_sm100 | 4738.3381 | 1.005 | — |
| `b4_s8192_h16_causal` | tir | 30542.4081 | flashattn_sm100 | 30776.4327 | 1.008 | — |
| `b4_s8192_h16_noncausal` | tir | 50738.2705 | flashattn_sm100 | 52405.0081 | 1.033 | — |

### flashinfer_layernorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_m1_h128_xc_yc_pdl0_eps1e6` | tirx | 2.9773 | flashinfer_cutedsl | 3.1874 | 1.071 | — |
| `bf16_m128_h1024_xc_yc_pdl0_eps1e6` | tirx | 6.2452 | flashinfer_cutedsl | 7.2593 | 1.162 | — |
| `bf16_m128_h16384_xc_yc_pdl0_eps1e6` | tirx | 64.8166 | flashinfer_cutedsl | 75.6959 | 1.168 | — |

### flashinfer_qk_rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gemma_bf16_b32_n32_h128_xc_yc_pdl0` | tirx | 6.0794 | flashinfer_cutedsl | 6.1496 | 1.012 | — |
| `rms_bf16_b32_n32_h128_xc_yc_pdl0` | tirx | 6.2123 | flashinfer_cutedsl | 6.3458 | 1.021 | — |
| `rms_f16_b16_n64_h128_xc_yc_pdl0` | tirx | 6.1372 | flashinfer_cutedsl | 6.2640 | 1.021 | — |

### flashinfer_rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gemma_bf16_m64_h8192_xc_yc_pdl0` | tirx | 18.3253 | flashinfer_cutedsl | 19.4243 | 1.060 | — |
| `rms_bf16_m32_h4096_xc_yc_pdl0` | tirx | 6.5895 | flashinfer_cutedsl | 6.8275 | 1.036 | — |
| `rms_bf16_m32_h4096_xc_yc_pdl1` | tirx | 8.7481 | flashinfer_cutedsl | 8.8077 | 1.007 | — |

### fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_4096x4096x4096` | tir | 1172.8616 | torch-cublas | 1337.8177 | 1.141 | — |
| `fp16_1024x1024x1024` | tir | 96.8199 | torch-cublas | 98.3945 | 1.016 | — |
| `fp16_16384x16384x16384` | tir | 193286.4660 | torch-cublas | 211708.9267 | 1.095 | — |

### msa_sparse_prepare_flat_schedule_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `decode_b128_k65536_h4` | tirx | 93666.9573 | msa | 119334.1068 | 1.274 | — |
| `decode_b64_k16384_h4_varlen` | tirx | 3136.1864 | msa | 3995.1724 | 1.274 | — |
| `prefill_b1_k8192_h2` | tirx | 5.7109 | msa | 6.2449 | 1.094 | — |

### mxfp4_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 6.7874 | flashinfer | 6.9652 | 1.026 | — |
| `fp16_128x4_m16384_k7168` | tirx | 1936.2057 | flashinfer | 1948.9107 | 1.007 | — |
| `fp16_linear_m4096_k4096` | tirx | 344.2575 | flashinfer | 353.9275 | 1.028 | — |

### selective_state_update_mtp_simple

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 24.1904 | flashinfer_cuda | 25.3989 | 1.050 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 15063.7808 | flashinfer_cuda | 15868.6225 | 1.053 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 3788.1454 | flashinfer_cuda | 3988.6067 | 1.053 | — |

### selective_state_update_mtp_vertical

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 47.7000 | flashinfer_cuda | 50.0233 | 1.049 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 25012.4818 | flashinfer_cuda | 26609.4659 | 1.064 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 6220.0814 | flashinfer_cuda | 6629.1134 | 1.066 | — |

### selective_state_update_mtp_horizontal

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 26.0536 | flashinfer_cuda | 29.3040 | 1.125 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 14069.8720 | flashinfer_cuda | 15273.7740 | 1.086 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 3537.5044 | flashinfer_cuda | 3875.5075 | 1.096 | — |
