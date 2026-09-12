# bench-suite baseline view: `baseline.json`

- Timestamp: `4`
- Label:     `remote-baseline`
- Git:       `{'tir': '7a8c0703', 'tirx-kernels': '52d04aed', 'tirx-bench-ci': None}`
- Workloads: 271 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## act_and_mul

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gelu_tanh_fp16_d11008_t8192` | tirx | 97.5791 | flashinfer | 85.4028 | 0.875 | — |
| `silu_bf16_d16384_t32768` | tirx | 454.3756 | flashinfer | 469.3271 | 1.033 | — |
| `silu_fp16_d4096_t1` | tirx | 2.1158 | flashinfer | 2.9639 | 1.401 | — |

## agent_evolved_kda_backward_packed

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `p04_hq2_hv4_t18432` | tirx | 150.5622 | fla_chunk_kda_bwd | 916.3090 | 6.086 | — |
| `p05_hq4_hv8_t18432` | tirx | 244.7955 | fla_chunk_kda_bwd | 1723.5605 | 7.041 | — |
| `packed_1024x8_h96` | tirx | 922.1302 | fla_chunk_kda_bwd | 8176.3468 | 8.867 | — |

## agent_evolved_kda_forward_b1_t8192

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `h64_uniform` | tirx | 191.4043 | flash_kda | 494.0019 | 2.581 | — |
| `h96_fixed` | tirx | 311.5623 | flash_kda | 1066.3652 | 3.423 | — |
| `h96_uniform` | tirx | 275.3779 | flash_kda | 729.4685 | 2.649 | — |

## agent_evolved_moe_fp8_blockscale_dsv3

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t1` | tirx | 49.3324 | flashinfer_trtllm_fp8_block_scale_moe | 64.8516 | 1.315 | — |
| `t14107` | tirx | 675.0185 | flashinfer_trtllm_fp8_block_scale_moe | 2318.1395 | 3.434 | — |
| `t901` | tirx | 271.0422 | flashinfer_trtllm_fp8_block_scale_moe | 341.5536 | 1.260 | — |

## cake_vsa_blk128_compact_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m16384_n16384_h16_sel16_lse` | tirx | 427.5383 | flashinfer | 432.0306 | 1.011 | — |
| `m32768_n32768_h24_sel32` | tirx | 2377.7613 | flashinfer | 2405.0680 | 1.011 | — |
| `m4096_n4096_h8_sel8` | tirx | 36.9984 | flashinfer | 37.2053 | 1.006 | — |

## cake_vsa_longseq_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m16384_n32768_h8_sel32` | tirx | 339.5920 | flashinfer | 343.1110 | 1.010 | — |
| `m4096_n16384_h8_sel1` | tirx | 16.8586 | flashinfer | 16.6366 | 0.987 | — |
| `m8192_n32768_h8_sel192` | tirx | 996.3080 | flashinfer | 1003.1753 | 1.007 | — |

## cake_vsa_ultrasparse_bsr_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m131072_n32768_h8_sel6` | tirx | 500.4929 | flashinfer | 524.3966 | 1.048 | — |
| `m262144_n131072_h8_sel6` | tirx | 1039.9807 | flashinfer | 1075.2621 | 1.034 | — |
| `m80000_n8192_h8_sel6` | tirx | 307.1484 | flashinfer | 321.9290 | 1.048 | — |

## cudnn_sm100_bsa_backward_blk128

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `p00_b1_h1_d64_sq128_skv4096_kv16` | tirx | 14.6209 | cudnn_frontend | 22.6397 | 1.548 | — |
| `p02_b2_h8_d128_sq4096_skv8191_kv32` | tirx | 363.5484 | cudnn_frontend | 379.2763 | 1.043 | — |
| `p11_b1_h2_d128_sq524288_skv8192_kv32_qb4096_g8` | tirx | 5559.2697 | cudnn_frontend | 5568.8322 | 1.002 | — |

## cudnn_sm100_bsa_backward_blk64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `p00_b1_h1_sq64_skv4096_kv16_nomask` | tirx | 13.4447 | cudnn_frontend | 14.8741 | 1.106 | — |
| `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask` | tirx | 152.9698 | cudnn_frontend | 158.5197 | 1.036 | — |
| `p11_b1_h1_sq192000_skv8192_maxkv16_var_mask_auto1024_i64kv` | tirx | 542.5334 | cudnn_frontend | 545.2254 | 1.005 | — |

## cudnn_sm100_bsa_forward_blk128

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `p00_bf16_d64_mha` | tirx | 57.7046 | cudnn_frontend | 58.1627 | 1.008 | — |
| `p08_bf16_d128_mqa` | tirx | 71.3320 | cudnn_frontend | 114.5157 | 1.605 | — |
| `p13_fp16_d96_gqa` | tirx | 59.7031 | cudnn_frontend | 69.6275 | 1.166 | — |

## cudnn_sm100_bsa_forward_blk64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `p00_b1_h1_sq64_skv4096_kv16_nomask_s1_static` | tirx | 9.4589 | cudnn_frontend | 9.5237 | 1.007 | — |
| `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask_s1_clc` | tirx | 42.3952 | cudnn_frontend | 52.0923 | 1.229 | — |
| `p09_b1_h8_sq2048_skv65536_kv512_nomask_s8_static` | tirx | 344.3742 | cudnn_frontend | 373.5086 | 1.085 | — |

## cudnn_sm100_bsa_forward_combine_blk64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h4_sq1024_s2` | tirx | 3.6072 | cudnn_frontend | 3.8366 | 1.064 | — |
| `b1_h4_sq2048_s4` | tirx | 6.4799 | cudnn_frontend | 6.6503 | 1.026 | — |
| `b1_h8_sq2048_s8` | tirx | 17.2641 | cudnn_frontend | 17.1135 | 0.991 | — |

## cudnn_sm100_csa_compressor_fwd

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b1_s8192_d128_c2` | tirx | 6.4495 | cudnn_frontend | 6.6218 | 1.027 | — |
| `perf_b3_s8192_d128_c2` | tirx | 11.8304 | cudnn_frontend | 12.3396 | 1.043 | — |
| `perf_b3_s8192_d512_c2` | tirx | 35.4869 | cudnn_frontend | 36.7574 | 1.036 | — |

## cudnn_sm100_dense_blockscaled_gemm_persistent_amax

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf00_m1024_n1024_k1024_e4_e8v32_bf16_kk_m_t128x128_c1x1_l1` | tirx | 5.3836 | cudnn_frontend | 5.5442 | 1.030 | — |
| `perf03_m4096_n4096_k4096_f4_e8v16_f16_kk_m_t128x128_c2x1_l1` | tirx | 33.4749 | cudnn_frontend | 33.7878 | 1.009 | — |
| `perf07_m8192_n8192_k8192_f4_e4v16_f4_kk_n_t128x128_c4x2_l2` | tirx | 439.9182 | cudnn_frontend | 440.8189 | 1.002 | — |

## cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `anchor_m1024_n1024_k1024_l1` | tirx | 9.0323 | cudnn_frontend | 9.3835 | 1.039 | — |
| `anchor_m4096_n4096_k4096_l1` | tirx | 45.8773 | cudnn_frontend | 46.7592 | 1.019 | — |
| `anchor_m8192_n8192_k8192_l2` | tirx | 402.5961 | cudnn_frontend | 407.9243 | 1.013 | — |

## cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf00_m4096_n4096_k4096_l1_e4_e8v32_bf16_bf16_mn_n_t256x256_c2x1_vf32_l1` | tirx | 53.3804 | cudnn_frontend | 53.9648 | 1.011 | — |
| `perf02_m1024_n1024_k1024_l1_f4_e8v16_bf16_bf16_kk_n_t128x128_c1x1_vf32_l1` | tirx | 5.5948 | cudnn_frontend | 5.7717 | 1.032 | — |
| `perf22_m8192_n8192_k8192_l2_f4_e8v16_bf16_bf16_kk_n_t256x64_c2x1_vf32_l2` | tirx | 602.6963 | cudnn_frontend | 608.9405 | 1.010 | — |

## cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf00_m1024_n1024_k1024_e4_e8v32_bf16_bf16_kk_m_t128x128_c1x1_sf32_l1` | tirx | 6.6133 | cudnn_frontend | 6.9837 | 1.056 | — |
| `perf15_m4096_n4096_k4096_e4_e8v32_e4_f32_kk_n_t128x128_c1x1_sf32_l1` | tirx | 68.8639 | cudnn_frontend | 69.5048 | 1.009 | — |
| `perf30_m8192_n8192_k8192_e4_e8v32_f32_f32_kk_n_t128x128_c1x1_vf32_l2` | tirx | 3719.8476 | cudnn_frontend | 3810.0987 | 1.024 | — |

## cudnn_sm100_dense_gemm_persistent_swiglu

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf00_m1024_n1024_k1024_bf16_f32_bf16_bf16_kk_m_t128x128_c1x1_l1` | tirx | 8.2674 | cudnn_frontend | 8.3091 | 1.005 | — |
| `perf03_m4096_n4096_k4096_f16_f16_bf16_bf16_kk_n_t256x64_c2x2_l1` | tirx | 137.8674 | cudnn_frontend | 139.8134 | 1.014 | — |
| `perf09_m8192_n8192_k8192_e5_f32_f32_bf16_mn_n_t256x128_c4x4_l2` | tirx | 1038.5087 | cudnn_frontend | 1044.4848 | 1.006 | — |

## cudnn_sm100_flex_attention_backward

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `p00_d128_dv128_causal_128k` | tirx | 30696.0496 | cudnn_frontend | 29705.9146 | 0.968 | — |
| `p16_d8_dv8_float16_fixed_causal` | tirx | 56.6381 | cudnn_frontend | 57.1289 | 1.009 | — |
| `p32_d64_dv64_coverage0` | tirx | 89.5611 | cudnn_frontend | 94.4810 | 1.055 | — |

## cudnn_sm100_flex_attention_forward_hd256

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `p_causal_bf16_cta1_fixed_mha_lse` | tirx | 23108.8898 | cudnn_frontend | 22208.0511 | 0.961 | — |
| `p_hstu_bf16_cta2_fixed_mha_lse` | tirx | 1802.4006 | cudnn_frontend | 1784.6763 | 0.990 | — |
| `p_local_fp16_cta2_varlen_gqa4_lse` | tirx | 535.0359 | cudnn_frontend | 1312.0992 | 2.452 | — |

## cudnn_sm100_gdn2_bprop_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_basic_b1_s2048_h16` | tirx | 505.2923 | cudnn_frontend | 407.1309 | 0.806 | — |
| `perf_l2_b1_s8192_h16` | tirx | 2418.9015 | cudnn_frontend | 2159.3301 | 0.893 | — |
| `perf_l2_b4_s8192_h64` | tirx | 6862.5210 | cudnn_frontend | 5633.9480 | 0.821 | — |

## cudnn_sm100_gdn2_prefill_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b1_s8192_h64_nostate` | tirx | 455.2358 | cudnn_frontend | 499.8491 | 1.098 | — |
| `perf_b4_s32768_h64_state` | tirx | 5962.6556 | cudnn_frontend | 6067.3652 | 1.018 | — |
| `perf_b4_s8192_h64_nostate` | tirx | 930.6561 | cudnn_frontend | 1008.7050 | 1.084 | — |

## cudnn_sm100_gdn2_recompute_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b1_s8192_h64_nostate` | tirx | 514.9804 | cudnn_frontend | 547.5272 | 1.063 | — |
| `perf_b4_s32768_h64_state` | tirx | 5228.7495 | cudnn_frontend | 5257.2478 | 1.005 | — |
| `perf_b4_s8192_h64_nostate` | tirx | 1294.0518 | cudnn_frontend | 1305.9527 | 1.009 | — |

## cudnn_sm100_gdn_bprop_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b4_s32768_h64` | tirx | 6444.4880 | cudnn_frontend | 7018.0083 | 1.089 | — |
| `perf_basic_b1_s2048_h16` | tirx | 175.8941 | cudnn_frontend | 180.9795 | 1.029 | — |
| `perf_basic_b1_s8192_h16` | tirx | 685.6477 | cudnn_frontend | 676.7349 | 0.987 | — |

## cudnn_sm100_gdn_prefill_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b1_s8192_h64_nostate` | tirx | 248.4162 | cudnn_frontend | 232.5437 | 0.936 | — |
| `perf_b4_s32768_h64_state` | tirx | 2892.6430 | cudnn_frontend | 2835.5281 | 0.980 | — |
| `perf_b4_s8192_h64_nostate` | tirx | 526.5453 | cudnn_frontend | 501.7368 | 0.953 | — |

## cudnn_sm100_gdn_recompute_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_b1_s8192_h64_nostate` | tirx | 262.3492 | cudnn_frontend | 261.9147 | 0.998 | — |
| `perf_b4_s32768_h64_state` | tirx | 2170.9101 | cudnn_frontend | 2157.0647 | 0.994 | — |
| `perf_b4_s8192_h64_nostate` | tirx | 555.8501 | cudnn_frontend | 553.7154 | 0.996 | — |

## cudnn_sm100_gemm_proj_rope_mxfp8_bf16in

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t2048_k1536_h128_w_out_in_false` | tirx | 124.1561 | cudnn_frontend | 131.6592 | 1.060 | — |
| `t4096_k1536_h128_w_out_in_false` | tirx | 249.4522 | cudnn_frontend | 261.7480 | 1.049 | — |
| `t4096_k1536_h128_w_out_in_true` | tirx | 251.0521 | cudnn_frontend | 256.3644 | 1.021 | — |

## cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t2048_k1536_h128` | tirx | 95.6796 | cudnn_frontend | 116.5403 | 1.218 | — |
| `t4096_k1536_h128` | tirx | 189.4431 | cudnn_frontend | 225.5675 | 1.191 | — |

## cudnn_sm100_kda_bprop_f16

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf_basic_b1_s2048_h16` | tirx | 445.6378 | cudnn_frontend | 409.7328 | 0.919 | — |
| `perf_l2_b1_s8192_h16` | tirx | 2277.4873 | cudnn_frontend | 2279.4413 | 1.001 | — |
| `perf_l2_b4_s8192_h64` | tirx | 6442.3399 | cudnn_frontend | 5128.1410 | 0.796 | — |

## cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf00_e4_t4096_n2048_k2048_dnst_sw_e4_e8v32_cbf16_de4_bk_t256x256_c2x1_bpv` | tirx | 44.8532 | cudnn_frontend | 44.8657 | 1.000 | — |
| `perf00_e8_t16384_n4096_k8192_dnst_sw_e4_e8v32_cbf16_de4_bk_t256x256_c2x1_bpv` | tirx | 450.2465 | cudnn_frontend | 458.1922 | 1.018 | — |
| `perf00_e8_t32768_n4096_k7168_dnst_sw_e4_e8v32_cbf16_de4_bk_t256x256_c2x1_bpv` | tirx | 812.2998 | cudnn_frontend | 811.2848 | 0.999 | — |

## cudnn_sm100_moe_grouped_gemm_dglu_dbias

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `perf00_e4_t4096_n2048_k2048_dnst_sw_cbf16_dbf16_bk_t256x256_c2x1_vb` | tirx | 41.4363 | cudnn_frontend | 41.8645 | 1.010 | — |
| `perf00_e8_t16384_n4096_k8192_dnst_sw_cbf16_dbf16_bk_t256x256_c2x1_vb` | tirx | 744.9379 | cudnn_frontend | 743.8721 | 0.999 | — |
| `perf00_e8_t32768_n4096_k7168_dnst_sw_cbf16_dbf16_bk_t256x256_c2x1_vb` | tirx | 1344.0361 | cudnn_frontend | 1347.4321 | 1.003 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 37.3890 | deepgemm | 39.6435 | 1.060 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 172.5455 | deepgemm | 182.1195 | 1.055 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.3262 | deepgemm | 6.5873 | 1.041 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 3.9429 | deepgemm | 4.7384 | 1.202 | — |

## deepgemm_sm100_fp8_bmm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bhd_bhr_hdr_b4096_h8_r4096_d1024` | tirx | 133.5706 | deepgemm | 137.0898 | 1.026 | — |
| `bhd_hdr_bhr_b8192_h8_r4096_d1024` | tirx | 226.9867 | deepgemm | 248.2419 | 1.094 | — |
| `bhr_hdr_bhd_b4096_h8_r4096_d1024` | tirx | 98.0146 | deepgemm | 98.6466 | 1.006 | — |

## deepgemm_sm100_fp8_gemm_1d1d

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m4096_n4096_k7168_bfp4` | tirx | 77.7860 | deepgemm | 78.4380 | 1.008 | — |
| `m4096_n576_k7168` | tirx | 18.7219 | deepgemm | 19.0530 | 1.018 | — |
| `m4096_n7168_k16384` | tirx | 328.2075 | deepgemm | 323.8813 | 0.987 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 39.8320 | deepgemm | 41.0239 | 1.030 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 179.6966 | deepgemm | 193.3338 | 1.076 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.7409 | sglang_cutedsl | 6.8303 | 1.013 | deepgemm=6.9287 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.1029 | sglang_cutedsl | 4.6704 | 1.138 | deepgemm=4.9748 |

## deepgemm_sm100_k_grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g16_m7168_n2048_k2048_gran128_al128` | tirx | 576.8235 | deepgemm | 573.6039 | 0.994 | — |
| `g4_m4096_n7168_k8192_gran128_al128_psum` | tirx | 837.0762 | deepgemm | 845.2882 | 1.010 | — |
| `g8_m4096_n7168_k4096_gran32_al160` | tirx | 901.4059 | deepgemm | 932.3985 | 1.034 | — |

## deepgemm_sm100_m_grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g4_m8192_n6144_k7168` | tirx | 986.4308 | deepgemm | 1001.6994 | 1.015 | — |
| `g8_m4096_n4096_k2048_bfp4` | tirx | 183.2535 | deepgemm | 184.2996 | 1.006 | — |
| `g8_m4096_n7168_k3072_psum_zp` | tirx | 507.1197 | deepgemm | 517.7624 | 1.021 | — |

## deepgemm_sm100_m_grouped_fp8_gemm_masked

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `g32_m192_n4096_k4096_bfp4` | tirx | 114.4696 | deepgemm | 113.8030 | 0.994 | — |
| `g32_m192_n6144_k7168` | tirx | 323.1794 | deepgemm | 322.0381 | 0.996 | — |
| `g6_m1024_n4096_k2048` | tirx | 39.2476 | deepgemm | 39.2947 | 1.001 | — |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.2588 | deepgemm | 5.2914 | 1.006 | — |
| `m4096_n24_k7168_s1` | tirx | 23.0464 | deepgemm | 23.6728 | 1.027 | — |
| `m8192_n24_k28672_s1` | tirx | 84.4612 | deepgemm | 92.4647 | 1.095 | — |

## fast_topk_clusters

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_plain_b16_l4096_k256` | tirx | 6.4824 | flashinfer | 7.1067 | 1.096 | — |
| `f32_plain_b64_l16384_k256` | tirx | 13.6070 | flashinfer | 14.0971 | 1.036 | — |
| `f32_plain_b64_l65536_k1024` | tirx | 20.8469 | flashinfer | 21.3315 | 1.023 | — |

## filtered_topk

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_plain_det_r2_l524288_k256_endbit` | tirx | 114.3760 | flashinfer | 129.8991 | 1.136 | — |
| `f32_plain_r4_l8192_k256` | tirx | 7.7872 | flashinfer | 9.0702 | 1.165 | — |
| `f32_plain_r64_l8192_k256` | tirx | 8.4275 | flashinfer | 9.7389 | 1.156 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv4` | tir | 19.3260 | flashattn_sm100 | 23.7789 | 1.230 | — |
| `s4096_h32kv4_causal` | tir | 112.3773 | flashattn_sm100 | 110.0823 | 0.980 | — |
| `s8192_h32kv32` | tir | 770.2086 | flashattn_sm100 | 954.9975 | 1.240 | — |

## flash_attention_backward_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_s2048_h16_causal` | tir | 80.7875 | flashattn_sm100 | 82.5767 | 1.022 | — |
| `b1_s8192_h16_noncausal` | tir | 1078.0363 | flashattn_sm100 | 1089.1990 | 1.010 | — |
| `b4_s8192_h16_noncausal` | tir | 4329.7294 | flashattn_sm100 | 4394.7338 | 1.015 | — |

## flashinfer_add_rmsnorm_fp4quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_nv_3d_bf16_b32_s32_h128_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 2.8270 | flashinfer_cutedsl | 2.7334 | 0.967 | — |
| `bench_nv_bf16_m32_h4096_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 5.4974 | flashinfer_cutedsl | 3.6816 | 0.670 | — |
| `bench_nv_large_bf16_m64_h8192_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 5.7809 | flashinfer_cutedsl | 3.8447 | 0.665 | — |

## flashinfer_fused_add_rmsnorm_quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_e4m3_m32_h4096_xc_rc_yc_pdl0_s1` | tirx | 3.3654 | flashinfer_cutedsl | 3.4347 | 1.021 | — |
| `bf16_e4m3_m32_h4096_xc_rc_yc_pdl1_s1` | tirx | 3.5447 | flashinfer_cutedsl | 3.6256 | 1.023 | — |
| `bf16_e4m3_m64_h8192_xc_rc_yc_pdl0_s1` | tirx | 3.7400 | flashinfer_cutedsl | 3.7264 | 0.996 | — |

## flashinfer_fused_dit_layernorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `grgb_bf16_b1_r1920` | tirx | 16.6263 | flashinfer_cuda | 16.7781 | 1.009 | — |
| `grss_bf16_b4_r1920` | tirx | 72.9458 | flashinfer_cuda | 73.7001 | 1.010 | — |
| `rss_bf16_b1_r768` | tirx | 8.6388 | flashinfer_cuda | 8.6127 | 0.997 | — |

## flashinfer_layernorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_m128_h1024_xc_yc_pdl0_eps1e6` | tirx | 3.0274 | flashinfer_cutedsl | 3.0205 | 0.998 | — |
| `bf16_m128_h16384_xc_yc_pdl0_eps1e6` | tirx | 8.9432 | flashinfer_cutedsl | 8.9038 | 0.996 | — |
| `bf16_m1_h128_xc_yc_pdl0_eps1e6` | tirx | 2.2431 | flashinfer_cutedsl | 2.2454 | 1.001 | — |

## flashinfer_qk_rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gemma_bf16_b32_n32_h128_xc_yc_pdl0` | tirx | 2.6520 | flashinfer_cutedsl | 2.6660 | 1.005 | — |
| `rms_bf16_b32_n32_h128_xc_yc_pdl0` | tirx | 2.5786 | flashinfer_cutedsl | 2.5751 | 0.999 | — |
| `rms_f16_b16_n64_h128_xc_yc_pdl0` | tirx | 2.5874 | flashinfer_cutedsl | 2.5930 | 1.002 | — |

## flashinfer_rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `gemma_bf16_m64_h8192_xc_yc_pdl0` | tirx | 3.3734 | flashinfer_cutedsl | 3.4629 | 1.027 | — |
| `rms_bf16_m32_h4096_xc_yc_pdl0` | tirx | 3.1684 | flashinfer_cutedsl | 3.3103 | 1.045 | — |
| `rms_bf16_m32_h4096_xc_yc_pdl1` | tirx | 3.3150 | flashinfer_cutedsl | 3.3022 | 0.996 | — |

## flashinfer_rmsnorm_fp4quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_nv_3d_bf16_b32_s32_h128_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 2.6520 | flashinfer_cutedsl | 2.7612 | 1.041 | — |
| `bench_nv_bf16_m32_h4096_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 4.6501 | flashinfer_cutedsl | 4.9169 | 1.057 | — |
| `bench_nv_large_bf16_m64_h8192_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | tirx | 4.7706 | flashinfer_cutedsl | 5.0411 | 1.057 | — |

## flashinfer_rmsnorm_quant

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_e4m3_m32_h4096_xc_yc_pdl0_s1` | tirx | 3.1304 | flashinfer_cutedsl | 3.1470 | 1.005 | — |
| `bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | tirx | 3.3240 | flashinfer_cutedsl | 3.2919 | 0.990 | — |
| `bf16_e5m2_m3_h1048576_xc_yc_pdl1_s1_cluster16_sync` | tirx | 16.6090 | flashinfer_cutedsl | 16.6953 | 1.005 | — |

## flashkda_bf16_fused_m128

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `h64_mixed` | tirx | 267.2602 | flashinfer_m128 | 246.6408 | 0.923 | flashkda_raw=674.6022 |
| `h96_fixed8192` | tirx | 501.0936 | flashinfer_m128 | 474.7650 | 0.947 | flashkda_raw=1088.5055 |
| `h96_uniform` | tirx | 435.1291 | flashinfer_m128 | 393.4816 | 0.904 | flashkda_raw=720.7827 |

## flashkda_decode_t1_precomputed

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv16h16_b128_s8` | tirx | 26.5421 | flashinfer_cake | 31.3697 | 1.182 | — |
| `hv16h16_b1_s16` | tirx | 3.9732 | flashinfer_cake | 4.8230 | 1.214 | — |
| `hv32h16_b32_s8` | tirx | 15.3485 | flashinfer_cake | 18.9134 | 1.232 | — |

## flashkda_decode_t2_precomputed

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv12h12_b8_t2` | tirx | 6.3397 | flashinfer_cake | 8.2341 | 1.299 | — |
| `hv16h16_b64_t2` | tirx | 25.0377 | flashinfer_cake | 29.5018 | 1.178 | — |
| `hv32h16_b128_t2` | tirx | 81.4915 | flashinfer_cake | 94.9095 | 1.165 | — |

## flashkda_decode_t3_lower_bound

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv16h16_b16_t3` | tirx | 13.3315 | flashinfer_cake | 13.9673 | 1.048 | — |
| `hv16h16_b1_t3` | tirx | 5.5619 | flashinfer_cake | 5.8770 | 1.057 | — |
| `hv16h16_b4_t3` | tirx | 6.6880 | flashinfer_cake | 6.8354 | 1.022 | — |

## flashkda_decode_t4_precomputed

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv12h12_b8_t4` | tirx | 8.5967 | flashinfer_cake | 10.2990 | 1.198 | — |
| `hv16h16_b64_t4` | tirx | 39.2970 | flashinfer_cake | 41.8347 | 1.065 | — |
| `hv32h16_b128_t4` | tirx | 131.5728 | flashinfer_cake | 137.3647 | 1.044 | — |

## flashkda_decode_t5_gram

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv32h16_b128_s1` | tirx | 159.1978 | flashinfer_cake | 159.7295 | 1.003 | — |
| `hv32h16_b1_s8` | tirx | 7.0499 | flashinfer_cake | 9.0548 | 1.284 | — |
| `hv32h16_b3_s4` | tirx | 8.6346 | flashinfer_cake | 9.8383 | 1.139 | — |

## flashkda_decode_t6_gram

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv32h16_b128_s1` | tirx | 177.4392 | flashinfer_cake | 179.3353 | 1.011 | — |
| `hv32h16_b1_s8` | tirx | 7.1819 | flashinfer_cake | 8.3343 | 1.160 | — |
| `hv32h16_b3_s4` | tirx | 14.0978 | flashinfer_cake | 15.6588 | 1.111 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_4096x4096x4096` | tir | 90.2884 | deepgemm-bf16 | 88.3829 | 0.979 | deepgemm-cublaslt=89.2035, torch-cublas=89.0057 |
| `fp16_1024x1024x1024` | tir | 6.6206 | torch-cublas | 5.9120 | 0.893 | — |
| `fp16_16384x16384x16384` | tir | 5712.7390 | torch-cublas | 5698.9725 | 0.998 | — |

## gdn_cp_prefill_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_q16_k16_v16_s4096+4096_init_f16_i64` | tirx | 117.2520 | flashinfer_cutedsl | 128.1965 | 1.093 | — |
| `fp16_q16_k16_v64_s192+64_initfinal_f16_i64` | tirx | 82.4385 | flashinfer_cutedsl | 82.6132 | 1.002 | — |
| `fp16_q1_k1_v1_s2048_none_i32` | tirx | 51.9827 | flashinfer_cutedsl | 67.1099 | 1.291 | — |

## gdn_decode_bf16_ilp4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t1_b1_h2_hv4_tv16` | tirx | 3.0016 | flashinfer_cutedsl | 3.2423 | 1.080 | — |
| `t4_b8_h4_hv8_tv16` | tirx | 6.5213 | flashinfer_cutedsl | 7.0249 | 1.077 | — |
| `t8_b4_h8_hv16_tv16` | tirx | 10.3580 | flashinfer_cutedsl | 11.6218 | 1.122 | — |

## gdn_decode_bf16_wide_vec_mtp

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t2_b4_h16_hv32_tv32` | tirx | 6.2602 | flashinfer_cutedsl | 6.8052 | 1.087 | — |
| `t4_b64_h8_hv16_tv128` | tirx | 33.6230 | flashinfer_cutedsl | 41.0639 | 1.221 | — |
| `t8_b512_h16_hv32_tv128` | tirx | 858.3107 | flashinfer_cutedsl | 859.2888 | 1.001 | — |

## gdn_decode_bf16_wide_vec_t1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b128_h8_hv16_tv128` | tirx | 24.7195 | flashinfer_cutedsl | 25.5661 | 1.034 | — |
| `b16_h16_hv32_tv64` | tirx | 8.8470 | flashinfer_cutedsl | 8.9838 | 1.015 | — |
| `b512_h4_hv8_tv128` | tirx | 46.3505 | flashinfer_cutedsl | 47.0252 | 1.015 | — |

## gdn_decode_fp32_mtp_warp

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t2_b4_h16_hv64_tv16_ilp2_sv0` | tirx | 14.4962 | flashinfer_cutedsl | 16.4998 | 1.138 | — |
| `t4_b64_h8_hv32_tv64_ilp4_sv1` | tirx | 138.8128 | flashinfer_cutedsl | 137.9819 | 0.994 | — |
| `t8_b256_h16_hv64_tv64_ilp4_sv1` | tirx | 1775.0352 | flashinfer_cutedsl | 1766.3158 | 0.995 | — |

## gdn_prefill_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hq16_hv64_s1x8192` | tirx | 239.6501 | flashinfer_cutedsl | 250.6522 | 1.046 | — |
| `hq32_hv32_s8192x16` | tirx | 1086.0588 | flashinfer_cutedsl | 1116.3724 | 1.028 | — |
| `hq8_hv32_s1024x8` | tirx | 90.3195 | flashinfer_cutedsl | 96.2150 | 1.065 | — |

## merge_state

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_s16384_h32_d128` | tirx | 64.4122 | flashinfer | 64.7978 | 1.006 | — |
| `fp16_s128_h32_d128` | tirx | 2.9027 | flashinfer | 2.9429 | 1.014 | — |
| `fp16_s2048_h32_d128` | tirx | 10.8166 | flashinfer | 11.4167 | 1.055 | — |

## mxfp4_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.4399 | flashinfer | 2.5485 | 1.045 | — |
| `fp16_128x4_m16384_k7168` | tirx | 57.3076 | flashinfer | 52.9975 | 0.925 | — |
| `fp16_linear_m4096_k4096` | tirx | 10.1725 | flashinfer | 10.3875 | 1.021 | — |

## mxfp8_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.5626 | flashinfer | 2.7126 | 1.059 | — |
| `fp16_128x4_m16384_k7168` | tirx | 60.5406 | flashinfer | 61.3838 | 1.014 | — |
| `fp16_linear_m4096_k4096` | tirx | 11.3288 | flashinfer | 11.2904 | 0.997 | — |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.3076 | cublaslt_nvfp4 | 4.4365 | 0.836 | flashinfer=4.4796 |
| `16384x16384x16384` | tir | 1478.9889 | flashinfer | 1407.5581 | 0.952 | cublaslt_nvfp4=1416.3645 |
| `4096x4096x4096` | tir | 29.5340 | cublaslt_nvfp4 | 27.5528 | 0.933 | flashinfer=30.5954 |

## nvfp4_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.5167 | flashinfer | 2.5309 | 1.006 | — |
| `fp16_128x4_m16384_k7168` | tirx | 55.4643 | flashinfer | 55.0608 | 0.993 | — |
| `fp16_linear_m4096_k4096` | tirx | 9.9870 | flashinfer | 10.1721 | 1.019 | — |

## nvfp4_quantize_per_token

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `fp16_128x4_m128_k1024` | tirx | 2.6851 | flashinfer | 2.8048 | 1.045 | — |
| `fp16_128x4_m16384_k7168` | tirx | 57.6700 | flashinfer | 59.2157 | 1.027 | — |
| `fp16_linear_m4096_k4096` | tirx | 11.9714 | flashinfer | 13.0295 | 1.088 | — |

## radix_topk_multi_cta

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_basic_r2_l524288_k256_large` | tirx | 40.4073 | flashinfer | 42.9759 | 1.064 | — |
| `f32_basic_r4_l115188_k256_ctas3` | tirx | 34.8782 | flashinfer | 36.1100 | 1.035 | — |
| `f32_basic_r4_l57596_k256_vec4` | tirx | 29.9921 | flashinfer | 31.9152 | 1.064 | — |

## radix_topk_single_cta

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_basic_r256_l57592_k1024_maxchunk` | tirx | 80.2695 | flashinfer | 80.8206 | 1.007 | — |
| `f32_basic_r64_l32768_k512` | tirx | 22.9188 | flashinfer | 25.6207 | 1.118 | — |
| `f32_basic_r8_l8192_k256` | tirx | 10.6622 | flashinfer | 12.1313 | 1.138 | — |

## recurrent_kda_decode_grouped

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `dec_hv16_b1` | tirx | 3.2321 | flashinfer_cutedsl | 3.5129 | 1.087 | — |
| `ver_t8_hv12_b16` | tirx | 22.0557 | flashinfer_cutedsl | 31.4705 | 1.427 | — |
| `ver_t8_hv16_b128` | tirx | 161.1780 | flashinfer_cutedsl | 220.3767 | 1.367 | — |

## recurrent_kda_decode_one_warp

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hv12_b64_tr16_lb` | tirx | 12.6241 | flashinfer_cutedsl | 13.7859 | 1.092 | — |
| `hv16_b128_tr16_lb` | tirx | 26.6699 | flashinfer_cutedsl | 29.5450 | 1.108 | — |
| `hv16_b8_tr8_lb` | tirx | 5.2476 | flashinfer_cutedsl | 5.3684 | 1.023 | — |

## rmsnorm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `hs128_bs32` | tir | 2.3163 | flashinfer | 2.0116 | 0.868 | — |
| `hs4096_bs128` | tir | 3.4469 | flashinfer | 3.3860 | 0.982 | — |
| `hs8192_bs4113` | tir | 70.2542 | flashinfer | 23.9197 | 0.340 | — |

## selective_state_update_mtp_horizontal

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 6.6558 | flashinfer_cuda | 7.4063 | 1.113 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 1340.8386 | flashinfer_cuda | 1509.0593 | 1.125 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 339.5400 | flashinfer_cuda | 381.9644 | 1.125 | — |

## selective_state_update_mtp_simple

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 4.3017 | flashinfer_cuda | 5.1615 | 1.200 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 1461.1855 | flashinfer_cuda | 1583.2450 | 1.084 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 369.8327 | flashinfer_cuda | 400.5019 | 1.083 | — |

## selective_state_update_mtp_vertical

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | tirx | 17.3331 | flashinfer_cuda | 18.2566 | 1.053 | — |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx | 2689.0516 | flashinfer_cuda | 2903.3396 | 1.080 | — |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | tirx | 678.8074 | flashinfer_cuda | 732.9689 | 1.080 | — |

## selective_state_update_stp_horizontal

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 47.6435 | flashinfer_cuda | 49.7336 | 1.044 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 26.6081 | flashinfer_cuda | 28.7173 | 1.079 | — |
| `b64_h64_d64_s256_r8` | tirx | 46.9917 | flashinfer_cuda | 47.1111 | 1.003 | — |

## selective_state_update_stp_simple

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 65.2544 | flashinfer_cuda | 67.5100 | 1.035 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 37.0292 | flashinfer_cuda | 38.0836 | 1.028 | — |
| `b64_h64_d64_s256_r8` | tirx | 54.2165 | flashinfer_cuda | 61.0571 | 1.126 | — |

## selective_state_update_stp_vertical

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b64_h64_d128_s128_r8` | tirx | 47.2782 | flashinfer_cuda | 54.3635 | 1.150 | — |
| `b64_h64_d64_s128_r8_base` | tirx | 26.9979 | flashinfer_cuda | 31.3632 | 1.162 | — |
| `b64_h64_d64_s256_r8` | tirx | 45.9849 | flashinfer_cuda | 48.9395 | 1.064 | — |

## silu_and_mul_nvfp4_experts_quantize

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_b8_m512_k2048` | tirx | 8.7921 | flashinfer | 10.6553 | 1.212 | — |
| `fp16_b128_m2048_k2048` | tirx | 274.1134 | flashinfer | 303.1528 | 1.106 | — |
| `fp16_b8_m16_k2048` | tirx | 3.3388 | flashinfer | 4.2194 | 1.264 | — |

## sm100_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1297.8000 | deepgemm | 1287.8000 | 0.992 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3413.0000 | deepgemm | 3411.0000 | 0.999 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1_s1` | tirx | 3803.8000 | deepgemm | 3805.4000 | 1.000 | — |

## sparse_flashmla_decode_head64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | tirx | 135.9081 | flashmla | 143.9894 | 1.059 | — |
| `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 16.6388 | flashmla | 20.9862 | 1.261 | — |
| `v32_b148_sq2_sk32768_topk16384_p64` | tirx | 910.4335 | flashmla | 949.6846 | 1.043 | — |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1847.4694 | flashmla | 1881.8889 | 1.019 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1680.5710 | flashmla | 1717.1620 | 1.022 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1792.1016 | flashmla | 1828.3010 | 1.020 | trtllm_gen=2071.9938 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1143.0908 | flashmla | 1164.9478 | 1.019 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1176.6594 | flashmla | 1200.4979 | 1.020 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1135.7600 | flashmla | 1154.5922 | 1.017 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 383.4403 | flashmla | 387.6547 | 1.011 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 366.7225 | flashmla | 374.8375 | 1.022 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 384.1339 | flashmla | 396.5597 | 1.032 | trtllm_gen=468.1361 |

## stable_sort_topk_by_value

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `f32_r4_k128` | tirx | 4.8004 | flashinfer | 4.8560 | 1.012 | — |
| `f32_r64_k2048` | tirx | 11.4025 | flashinfer | 11.5270 | 1.011 | — |
| `f32_r64_k256` | tirx | 6.2369 | flashinfer | 6.7179 | 1.077 | — |

## tinygemm2_sm100

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_o2880_k2880` | tirx | 7.8855 | flashinfer_sm100 | 7.9961 | 1.014 | — |
| `b1_o128_k720` | tirx | 2.9323 | flashinfer_sm100 | 2.9278 | 0.998 | — |
| `b64_o4096_k3072` | tirx | 21.8806 | flashinfer_sm100 | 22.0385 | 1.007 | — |
