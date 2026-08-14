# 106-workload supplemental sweep breakdown

This is supplemental evidence outside every acceptance criterion.

## End-to-end result

- migration-before: `476.645473386s` (`7.9441 min`)
- pipeline: `363.171956028s` (`6.0529 min`)
- before / after: `1.312451x`
- wall reduction: `113.473517358s` (`23.8067%`)

Both sides used isolated cold caches. This increases the prepare share and makes the measured speedup an upper bound for ordinary warm-cache use.

The pipeline command measured commit 5429283 before the later NVFP4 cuBLASLt build-only move; no adjusted suite wall is inferred from the targeted follow-up.

The migration-before runner has no equivalent phase timeline or card-time cost model; those fields are unavailable, not zero.

## Pipeline workload phase breakdown

All durations are wall-clock seconds.

| workload | startup | CLI | framework import | exact import | config | specialize/compile | CPU prepare | READY wait | ASSIGN | GPU stage | result | reap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp16_bf16_gemm/fp16_1024x1024x1024` | 0.043 | 0.027 | 2.421 | 1.048 | 0.000 | 2.909 | 6.405 | 18.036 | 3.889 | 12.389 | 0.000 | 0.736 |
| `fp16_bf16_gemm/fp16_16384x16384x16384` | 0.043 | 0.026 | 2.376 | 1.095 | 0.000 | 2.928 | 6.427 | 18.863 | 3.079 | 15.724 | 0.000 | 0.700 |
| `fp16_bf16_gemm/bf16_4096x4096x4096` | 0.043 | 0.027 | 2.388 | 1.064 | 0.000 | 2.921 | 6.399 | 18.013 | 3.877 | 25.536 | 0.000 | 0.736 |
| `nvfp4_gemm/1024x1024x1024` | 0.043 | 0.026 | 2.393 | 1.054 | 0.000 | 1.832 | 5.305 | 0.148 | 5.522 | 156.112 | 0.001 | 0.930 |
| `nvfp4_gemm/4096x4096x4096` | 0.041 | 0.026 | 2.394 | 1.080 | 0.000 | 1.978 | 5.479 | 0.142 | 5.162 | 157.011 | 0.000 | 0.828 |
| `nvfp4_gemm/16384x16384x16384` | 0.042 | 0.025 | 2.337 | 1.076 | 0.000 | 2.045 | 5.483 | 18.063 | 1.440 | 143.488 | 0.000 | 1.563 |
| `deepgemm_fp8_fp4_mega_moe/t64_m64_h7168_i3072_e384_k6_g1` | 0.042 | 0.026 | 2.351 | 0.993 | 0.000 | 6.878 | 10.248 | 33.841 | 0.954 | 19.235 | 0.001 | 1.008 |
| `deepgemm_fp8_fp4_mega_moe/t8192_m8192_h7168_i3072_e384_k6_g1` | 0.043 | 0.026 | 2.362 | 1.092 | 0.000 | 7.420 | 10.900 | 49.046 | 1.240 | 34.697 | 0.001 | 0.920 |
| `deepgemm_sm100_fp4_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | 0.043 | 0.027 | 2.389 | 1.066 | 0.000 | 2.135 | 5.617 | 18.471 | 4.215 | 13.276 | 0.000 | 0.709 |
| `deepgemm_sm100_fp4_mqa_logits/s4096_skv8192_h64_d128_bf16_compressed_nocp` | 0.042 | 0.026 | 2.318 | 1.063 | 0.000 | 2.067 | 5.474 | 10.253 | 0.799 | 14.398 | 0.000 | 0.746 |
| `deepgemm_sm100_fp4_paged_mqa_logits/b1_n1_mp1_ps32_h64_d128_f32_fixed` | 0.044 | 0.027 | 2.364 | 1.071 | 0.000 | 1.691 | 5.153 | 0.035 | 5.585 | 13.649 | 0.000 | 0.960 |
| `deepgemm_sm100_fp4_paged_mqa_logits/b16_n1_mp128_ps64_h64_d128_bf16_fixed` | 0.043 | 0.026 | 2.374 | 1.075 | 0.000 | 1.938 | 5.413 | 90.467 | 0.023 | 12.599 | 0.000 | 0.716 |
| `deepgemm_sm100_fp8_bmm/bhr_hdr_bhd_b4096_h8_r4096_d1024` | 0.043 | 0.026 | 2.465 | 0.978 | 0.000 | 1.273 | 4.741 | 0.083 | 5.883 | 13.687 | 0.000 | 0.856 |
| `deepgemm_sm100_fp8_bmm/bhd_hdr_bhr_b8192_h8_r4096_d1024` | 0.043 | 0.026 | 2.340 | 1.112 | 0.000 | 1.223 | 4.702 | 0.016 | 5.968 | 14.575 | 0.000 | 2.248 |
| `deepgemm_sm100_fp8_bmm/bhd_bhr_hdr_b4096_h8_r4096_d1024` | 0.043 | 0.029 | 2.360 | 1.075 | 0.000 | 1.090 | 4.553 | 0.049 | 5.728 | 13.195 | 0.001 | 0.740 |
| `deepgemm_sm100_fp8_gemm_1d1d/m4096_n576_k7168` | 0.043 | 0.026 | 2.478 | 0.977 | 0.000 | 1.158 | 4.640 | 0.019 | 5.759 | 13.656 | 0.000 | 0.776 |
| `deepgemm_sm100_fp8_gemm_1d1d/m4096_n7168_k16384` | 0.042 | 0.027 | 2.454 | 1.437 | 0.000 | 1.241 | 5.159 | 98.656 | 0.069 | 12.565 | 0.000 | 0.688 |
| `deepgemm_sm100_fp8_gemm_1d1d/m4096_n4096_k7168_bfp4` | 0.043 | 0.027 | 2.440 | 1.399 | 0.000 | 1.253 | 5.119 | 21.054 | 0.946 | 13.350 | 0.000 | 0.685 |
| `deepgemm_sm100_fp8_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | 0.043 | 0.027 | 2.442 | 1.377 | 0.000 | 1.961 | 5.807 | 34.627 | 1.017 | 13.698 | 0.000 | 0.719 |
| `deepgemm_sm100_fp8_mqa_logits/s4096_skv8192_h64_d128_bf16_compressed_nocp` | 0.045 | 0.028 | 2.495 | 1.127 | 0.000 | 2.013 | 5.664 | 44.532 | 0.992 | 14.681 | 0.000 | 0.697 |
| `deepgemm_sm100_fp8_paged_mqa_logits/b1_n1_mp1_ps64_h64_d128_f32_fixed` | 0.044 | 0.026 | 2.388 | 0.929 | 0.000 | 1.928 | 5.271 | 85.462 | 0.012 | 19.058 | 0.001 | 1.023 |
| `deepgemm_sm100_fp8_paged_mqa_logits/b16_n1_mp128_ps64_h64_d128_bf16_fixed` | 0.060 | 0.036 | 2.378 | 0.893 | 0.000 | 1.880 | 5.188 | 45.487 | 0.870 | 23.607 | 0.000 | 1.012 |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g8_m4096_n7168_k4096_gran32_al160` | 0.048 | 0.026 | 2.538 | 0.718 | 0.000 | 1.257 | 4.538 | 31.468 | 1.035 | 13.564 | 0.001 | 0.678 |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g16_m7168_n2048_k2048_gran128_al128` | 0.043 | 0.028 | 2.432 | 0.679 | 0.000 | 1.218 | 4.357 | 30.691 | 1.004 | 13.459 | 0.001 | 0.676 |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g4_m4096_n7168_k8192_gran128_al128_psum` | 0.045 | 0.027 | 2.386 | 0.792 | 0.000 | 1.256 | 4.460 | 35.493 | 1.356 | 13.907 | 0.000 | 0.654 |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g4_m8192_n6144_k7168` | 0.043 | 0.027 | 2.549 | 1.014 | 0.000 | 1.230 | 4.819 | 98.455 | 0.006 | 13.207 | 0.001 | 0.702 |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g8_m4096_n4096_k2048_bfp4` | 0.069 | 0.041 | 2.407 | 0.770 | 0.000 | 1.232 | 4.449 | 41.077 | 0.995 | 13.471 | 0.000 | 0.786 |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g8_m4096_n7168_k3072_psum_zp` | 0.045 | 0.027 | 2.399 | 0.702 | 0.000 | 1.257 | 4.385 | 49.727 | 0.915 | 13.477 | 0.000 | 0.728 |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked/g32_m192_n6144_k7168` | 0.045 | 0.027 | 2.343 | 0.292 | 0.000 | 1.257 | 3.918 | 83.660 | 0.008 | 13.028 | 0.000 | 0.582 |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked/g6_m1024_n4096_k2048` | 0.044 | 0.027 | 2.274 | 0.282 | 0.000 | 1.222 | 3.805 | 45.145 | 1.041 | 13.266 | 0.000 | 0.663 |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked/g32_m192_n4096_k4096_bfp4` | 0.045 | 0.027 | 2.296 | 0.317 | 0.000 | 1.350 | 3.991 | 22.244 | 0.884 | 13.258 | 0.000 | 0.687 |
| `deepgemm_sm100_tf32_hc_prenorm_gemm/m128_n24_k16384_s64` | 0.045 | 0.027 | 2.280 | 0.597 | 0.000 | 1.711 | 4.616 | 34.920 | 1.054 | 13.143 | 0.000 | 0.604 |
| `deepgemm_sm100_tf32_hc_prenorm_gemm/m8192_n24_k28672_s1` | 0.043 | 0.027 | 2.327 | 0.295 | 0.000 | 1.742 | 4.391 | 36.107 | 0.939 | 12.492 | 0.000 | 0.692 |
| `deepgemm_sm100_tf32_hc_prenorm_gemm/m4096_n24_k7168_s1` | 0.043 | 0.027 | 2.298 | 0.335 | 0.000 | 1.730 | 4.391 | 43.596 | 1.021 | 13.218 | 0.000 | 0.713 |
| `flash_attention4/s1024_h32kv4` | 0.046 | 0.027 | 2.322 | 0.447 | 0.000 | 4.412 | 7.208 | 33.167 | 0.937 | 16.009 | 0.001 | 1.030 |
| `flash_attention4/s4096_h32kv4_causal` | 0.046 | 0.027 | 2.305 | 0.384 | 0.000 | 4.511 | 7.227 | 35.637 | 0.989 | 17.685 | 0.000 | 0.868 |
| `flash_attention4/s8192_h32kv32` | 0.046 | 0.026 | 2.327 | 0.429 | 0.000 | 4.604 | 7.386 | 43.887 | 1.333 | 20.930 | 0.000 | 0.973 |
| `flash_attention_backward_sm100/b1_s2048_h16_causal` | 0.043 | 0.026 | 2.310 | 0.831 | 0.000 | 6.758 | 9.926 | 41.295 | 0.757 | 20.079 | 0.000 | 0.864 |
| `flash_attention_backward_sm100/b1_s8192_h16_noncausal` | 0.044 | 0.027 | 2.341 | 0.452 | 0.000 | 6.588 | 9.408 | 42.700 | 0.968 | 19.260 | 0.001 | 0.784 |
| `flash_attention_backward_sm100/b4_s8192_h16_noncausal` | 0.045 | 0.027 | 2.408 | 0.285 | 0.000 | 6.467 | 9.188 | 100.979 | 0.008 | 16.880 | 0.000 | 1.740 |
| `act_and_mul/silu_fp16_d4096_t1` | 0.044 | 0.026 | 2.356 | 0.506 | 0.000 | 0.617 | 3.506 | 46.179 | 1.112 | 16.163 | 0.000 | 0.778 |
| `act_and_mul/silu_bf16_d16384_t32768` | 0.045 | 0.027 | 2.349 | 0.283 | 0.000 | 0.682 | 3.341 | 45.389 | 1.353 | 16.203 | 0.000 | 1.261 |
| `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 0.046 | 0.027 | 2.373 | 0.347 | 0.000 | 0.768 | 3.516 | 46.647 | 0.902 | 17.190 | 0.000 | 0.806 |
| `silu_and_mul_nvfp4_experts_quantize/fp16_b128_m2048_k2048` | 0.046 | 0.026 | 2.308 | 0.289 | 0.000 | 0.739 | 3.362 | 48.152 | 0.835 | 13.596 | 0.000 | 0.811 |
| `silu_and_mul_nvfp4_experts_quantize/bf16_b8_m512_k2048` | 0.045 | 0.027 | 2.405 | 0.304 | 0.000 | 0.822 | 3.557 | 43.128 | 0.901 | 13.728 | 0.000 | 0.710 |
| `silu_and_mul_nvfp4_experts_quantize/fp16_b8_m16_k2048` | 0.046 | 0.026 | 2.474 | 0.284 | 0.000 | 0.787 | 3.571 | 47.948 | 1.712 | 13.837 | 0.000 | 0.771 |
| `gdn_decode_bf16_wide_vec_t1/b16_h16_hv32_tv64` | 0.046 | 0.028 | 2.419 | 0.723 | 0.000 | 1.092 | 4.262 | 47.032 | 1.767 | 15.204 | 0.001 | 0.821 |
| `gdn_decode_bf16_wide_vec_t1/b128_h8_hv16_tv128` | 0.045 | 0.027 | 2.406 | 0.601 | 0.000 | 1.300 | 4.334 | 97.479 | 0.114 | 15.879 | 0.001 | 0.775 |
| `gdn_decode_bf16_wide_vec_t1/b512_h4_hv8_tv128` | 0.045 | 0.027 | 2.342 | 0.299 | 0.000 | 1.272 | 3.939 | 47.428 | 0.902 | 16.313 | 0.000 | 1.834 |
| `gdn_prefill_sm100/hq8_hv32_s1024x8` | 0.047 | 0.028 | 2.371 | 0.309 | 0.000 | 4.020 | 6.728 | 36.435 | 0.913 | 18.870 | 0.000 | 0.833 |
| `gdn_prefill_sm100/hq16_hv64_s1x8192` | 0.045 | 0.030 | 2.372 | 0.307 | 0.000 | 4.034 | 6.742 | 40.016 | 0.914 | 18.812 | 0.001 | 0.805 |
| `gdn_prefill_sm100/hq32_hv32_s8192x16` | 0.045 | 0.027 | 2.344 | 0.316 | 0.000 | 3.963 | 6.651 | 43.104 | 0.852 | 18.154 | 0.000 | 0.734 |
| `tinygemm2_sm100/b1_o128_k720` | 0.045 | 0.027 | 2.349 | 0.314 | 0.000 | 0.929 | 3.619 | 45.158 | 1.226 | 19.155 | 0.000 | 1.555 |
| `tinygemm2_sm100/b16_o2880_k2880` | 0.044 | 0.027 | 2.344 | 0.324 | 0.000 | 0.917 | 3.611 | 45.123 | 1.174 | 17.115 | 0.000 | 1.791 |
| `tinygemm2_sm100/b64_o4096_k3072` | 0.044 | 0.026 | 2.333 | 0.324 | 0.000 | 0.935 | 3.618 | 42.293 | 2.919 | 14.664 | 0.007 | 1.544 |
| `flashkda_bf16_fused_m128/h96_fixed8192` | 0.045 | 0.027 | 2.320 | 0.607 | 0.000 | 2.550 | 5.505 | 33.770 | 2.608 | 27.559 | 0.000 | 1.963 |
| `flashkda_bf16_fused_m128/h96_uniform` | 0.045 | 0.027 | 2.349 | 0.378 | 0.000 | 2.607 | 5.361 | 31.645 | 2.263 | 27.339 | 0.000 | 0.952 |
| `flashkda_bf16_fused_m128/h64_mixed` | 0.044 | 0.026 | 2.320 | 0.316 | 0.000 | 2.600 | 5.263 | 32.955 | 1.063 | 24.472 | 0.000 | 0.817 |
| `recurrent_kda_decode_grouped/dec_hv16_b1` | 0.042 | 0.027 | 2.329 | 0.277 | 0.000 | 1.004 | 3.637 | 33.629 | 0.891 | 14.565 | 0.000 | 0.747 |
| `recurrent_kda_decode_grouped/ver_t8_hv16_b128` | 0.042 | 0.026 | 2.284 | 0.279 | 0.000 | 1.291 | 3.880 | 39.348 | 2.483 | 17.493 | 0.000 | 1.169 |
| `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 0.046 | 0.029 | 2.273 | 0.337 | 0.000 | 1.326 | 3.965 | 33.389 | 2.506 | 17.452 | 0.000 | 1.056 |
| `recurrent_kda_decode_one_warp/hv16_b8_tr8_lb` | 0.045 | 0.027 | 2.285 | 0.352 | 0.000 | 0.972 | 3.636 | 32.431 | 2.488 | 15.101 | 0.000 | 0.774 |
| `recurrent_kda_decode_one_warp/hv16_b128_tr16_lb` | 0.043 | 0.026 | 2.275 | 0.274 | 0.000 | 1.002 | 3.577 | 35.931 | 1.009 | 16.105 | 0.000 | 0.836 |
| `recurrent_kda_decode_one_warp/hv12_b64_tr16_lb` | 0.043 | 0.026 | 2.327 | 0.301 | 0.000 | 1.205 | 3.859 | 40.613 | 2.463 | 15.774 | 0.000 | 0.728 |
| `selective_state_update_mtp_horizontal/b1_h64_d64_s128_t6_r8_statebf16_official` | 0.043 | 0.026 | 2.297 | 0.286 | 0.000 | 1.279 | 3.889 | 37.405 | 2.948 | 29.647 | 0.000 | 0.736 |
| `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 0.045 | 0.026 | 2.284 | 0.584 | 0.000 | 1.248 | 4.142 | 30.541 | 3.026 | 31.449 | 0.000 | 1.143 |
| `selective_state_update_mtp_horizontal/b2048_h64_d64_s128_t6_r8_statebf16_official` | 0.046 | 0.027 | 2.276 | 1.054 | 0.000 | 1.343 | 4.700 | 32.424 | 0.890 | 30.221 | 0.009 | 1.172 |
| `selective_state_update_mtp_simple/b1_h64_d64_s128_t6_r8_statebf16_official` | 0.044 | 0.026 | 2.323 | 0.533 | 0.000 | 1.233 | 4.115 | 33.169 | 1.866 | 24.774 | 0.000 | 1.097 |
| `selective_state_update_mtp_simple/b512_h64_d64_s128_t6_r8_statebf16_official` | 0.045 | 0.027 | 2.290 | 0.275 | 0.000 | 1.167 | 3.759 | 32.930 | 1.870 | 24.355 | 0.000 | 1.298 |
| `selective_state_update_mtp_simple/b2048_h64_d64_s128_t6_r8_statebf16_official` | 0.042 | 0.026 | 2.308 | 0.697 | 0.000 | 1.198 | 4.229 | 34.152 | 1.030 | 22.324 | 0.000 | 1.594 |
| `selective_state_update_mtp_vertical/b1_h64_d64_s128_t6_r8_statebf16_official` | 0.043 | 0.026 | 2.319 | 0.699 | 0.000 | 1.466 | 4.511 | 101.038 | 0.025 | 13.567 | 0.000 | 1.029 |
| `selective_state_update_mtp_vertical/b512_h64_d64_s128_t6_r8_statebf16_official` | 0.043 | 0.026 | 2.322 | 0.580 | 0.000 | 1.535 | 4.463 | 38.439 | 1.052 | 16.990 | 0.000 | 0.997 |
| `selective_state_update_mtp_vertical/b2048_h64_d64_s128_t6_r8_statebf16_official` | 0.044 | 0.026 | 2.287 | 0.280 | 0.000 | 1.430 | 4.023 | 50.045 | 1.166 | 14.344 | 0.000 | 0.762 |
| `selective_state_update_stp_horizontal/b64_h64_d64_s128_r8_base` | 0.045 | 0.026 | 2.284 | 0.336 | 0.000 | 1.742 | 4.388 | 44.650 | 4.306 | 31.326 | 0.000 | 1.303 |
| `selective_state_update_stp_horizontal/b64_h64_d128_s128_r8` | 0.042 | 0.026 | 2.290 | 0.321 | 0.000 | 1.472 | 4.110 | 40.908 | 3.893 | 20.052 | 0.000 | 1.358 |
| `selective_state_update_stp_horizontal/b64_h64_d64_s256_r8` | 0.042 | 0.026 | 2.286 | 0.308 | 0.000 | 1.553 | 4.173 | 41.588 | 3.828 | 34.177 | 0.000 | 0.834 |
| `selective_state_update_stp_simple/b64_h64_d64_s128_r8_base` | 0.043 | 0.026 | 2.312 | 0.314 | 0.000 | 0.821 | 3.473 | 41.010 | 4.202 | 30.194 | 0.000 | 0.869 |
| `selective_state_update_stp_simple/b64_h64_d128_s128_r8` | 0.044 | 0.026 | 2.303 | 0.290 | 0.000 | 0.836 | 3.455 | 29.866 | 0.932 | 29.682 | 0.000 | 1.216 |
| `selective_state_update_stp_simple/b64_h64_d64_s256_r8` | 0.045 | 0.026 | 2.409 | 0.718 | 0.000 | 0.839 | 3.992 | 31.430 | 3.172 | 32.029 | 0.000 | 1.123 |
| `selective_state_update_stp_vertical/b64_h64_d64_s128_r8_base` | 0.055 | 0.029 | 2.397 | 0.291 | 0.000 | 1.260 | 3.977 | 31.040 | 2.770 | 29.943 | 0.000 | 0.843 |
| `selective_state_update_stp_vertical/b64_h64_d128_s128_r8` | 0.045 | 0.027 | 2.297 | 0.318 | 0.000 | 1.362 | 4.004 | 84.954 | 0.016 | 14.588 | 0.000 | 1.216 |
| `selective_state_update_stp_vertical/b64_h64_d64_s256_r8` | 0.045 | 0.027 | 2.406 | 0.319 | 0.000 | 1.506 | 4.259 | 41.494 | 0.787 | 23.171 | 0.001 | 0.925 |
| `mxfp4_quantize/fp16_linear_m4096_k4096` | 0.042 | 0.026 | 2.301 | 0.964 | 0.000 | 0.655 | 3.946 | 48.196 | 1.854 | 16.839 | 0.001 | 0.834 |
| `mxfp4_quantize/fp16_128x4_m16384_k7168` | 0.046 | 0.027 | 2.324 | 0.738 | 0.000 | 0.603 | 3.692 | 55.959 | 1.149 | 17.060 | 0.000 | 0.740 |
| `mxfp4_quantize/fp16_128x4_m128_k1024` | 0.045 | 0.027 | 2.321 | 0.630 | 0.000 | 0.585 | 3.563 | 45.819 | 1.866 | 17.811 | 0.000 | 0.718 |
| `mxfp8_quantize/fp16_linear_m4096_k4096` | 0.045 | 0.027 | 2.309 | 0.397 | 0.000 | 0.554 | 3.287 | 54.313 | 2.129 | 16.813 | 0.000 | 1.670 |
| `mxfp8_quantize/fp16_128x4_m16384_k7168` | 0.044 | 0.027 | 2.289 | 0.297 | 0.000 | 0.570 | 3.182 | 48.555 | 2.012 | 16.565 | 0.000 | 1.773 |
| `mxfp8_quantize/fp16_128x4_m128_k1024` | 0.045 | 0.026 | 2.339 | 0.299 | 0.000 | 0.556 | 3.219 | 49.257 | 2.187 | 16.864 | 0.000 | 0.839 |
| `nvfp4_quantize/fp16_linear_m4096_k4096` | 0.043 | 0.026 | 2.471 | 1.904 | 0.000 | 0.574 | 4.974 | 33.728 | 1.564 | 17.344 | 0.001 | 0.921 |
| `nvfp4_quantize/fp16_128x4_m16384_k7168` | 0.045 | 0.026 | 2.362 | 0.937 | 0.000 | 0.598 | 3.924 | 33.903 | 1.620 | 17.438 | 0.000 | 0.718 |
| `nvfp4_quantize/fp16_128x4_m128_k1024` | 0.046 | 0.029 | 2.301 | 0.922 | 0.000 | 0.593 | 3.845 | 37.658 | 1.333 | 17.613 | 0.000 | 0.871 |
| `nvfp4_quantize_per_token/fp16_linear_m4096_k4096` | 0.056 | 0.033 | 2.378 | 0.748 | 0.000 | 0.747 | 3.906 | 47.058 | 1.112 | 17.332 | 0.000 | 0.830 |
| `nvfp4_quantize_per_token/fp16_128x4_m16384_k7168` | 0.045 | 0.026 | 2.280 | 0.771 | 0.000 | 0.683 | 3.760 | 48.083 | 2.365 | 18.473 | 0.000 | 2.274 |
| `nvfp4_quantize_per_token/fp16_128x4_m128_k1024` | 0.045 | 0.026 | 2.279 | 0.294 | 0.000 | 0.704 | 3.303 | 47.737 | 2.331 | 17.515 | 0.000 | 0.724 |
| `sparse_flashmla_decode_head64/deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | 0.043 | 0.027 | 2.294 | 0.292 | 0.000 | 5.651 | 8.264 | 44.392 | 1.160 | 14.856 | 0.000 | 1.224 |
| `sparse_flashmla_decode_head64/model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | 0.043 | 0.027 | 2.267 | 0.276 | 0.000 | 5.737 | 8.306 | 33.766 | 2.460 | 14.474 | 0.000 | 0.938 |
| `sparse_flashmla_decode_head64/v32_b148_sq2_sk32768_topk16384_p64` | 0.045 | 0.027 | 2.291 | 0.298 | 0.000 | 5.890 | 8.506 | 27.871 | 1.203 | 16.451 | 0.000 | 0.749 |
| `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | 0.043 | 0.026 | 2.312 | 0.376 | 0.000 | 3.252 | 5.966 | 26.507 | 2.303 | 13.927 | 0.000 | 0.718 |
| `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | 0.045 | 0.027 | 2.300 | 0.744 | 0.000 | 3.696 | 6.767 | 28.981 | 1.296 | 15.415 | 0.000 | 0.568 |
| `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | 0.045 | 0.027 | 2.292 | 0.580 | 0.000 | 3.553 | 6.452 | 29.834 | 1.954 | 47.494 | 0.000 | 0.866 |
| `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | 0.042 | 0.026 | 2.292 | 0.939 | 0.000 | 2.124 | 5.381 | 43.790 | 0.021 | 12.532 | 0.000 | 0.602 |
| `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | 0.043 | 0.026 | 2.309 | 0.323 | 0.000 | 2.055 | 4.712 | 30.857 | 3.932 | 13.478 | 0.000 | 0.655 |
| `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | 0.046 | 0.026 | 2.307 | 0.388 | 0.000 | 1.967 | 4.688 | 30.192 | 3.619 | 13.506 | 0.000 | 0.851 |
| `sparse_flashmla_prefill_head64_phase1/bench_dqk512_hq64_s4096_kv8192_topk512` | 0.050 | 0.026 | 2.280 | 0.281 | 0.000 | 2.825 | 5.412 | 29.004 | 3.589 | 13.163 | 0.000 | 0.591 |
| `sparse_flashmla_prefill_head64_phase1/bench_dqk512_hq64_s4096_kv65536_topk512` | 0.045 | 0.026 | 2.430 | 0.288 | 0.000 | 2.717 | 5.461 | 25.635 | 3.049 | 13.406 | 0.000 | 0.776 |
| `sparse_flashmla_prefill_head64_phase1/bench_dqk576_hq64_s4096_kv32768_topk512` | 0.045 | 0.026 | 2.311 | 0.289 | 0.000 | 2.837 | 5.464 | 29.389 | 1.253 | 40.940 | 0.000 | 0.871 |

## GPU-stage residual, descending

The floor includes only mandatory cooldown: implementations x 5 rounds x 1.0s. Timer setup, correctness, allocation, loading, warmup/repeat and real GPU execution remain in the residual, so it is a triage signal rather than movable-work accounting.

| rank | workload | GPU stage | cooldown floor | residual |
|---:|---|---:|---:|---:|
| 1 | `nvfp4_gemm/4096x4096x4096` | 157.011 | 15.000 | 142.011 |
| 2 | `nvfp4_gemm/1024x1024x1024` | 156.112 | 15.000 | 141.112 |
| 3 | `nvfp4_gemm/16384x16384x16384` | 143.488 | 15.000 | 128.488 |
| 4 | `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | 47.494 | 15.000 | 32.494 |
| 5 | `sparse_flashmla_prefill_head64_phase1/bench_dqk576_hq64_s4096_kv32768_topk512` | 40.940 | 15.000 | 25.940 |
| 6 | `deepgemm_fp8_fp4_mega_moe/t8192_m8192_h7168_i3072_e384_k6_g1` | 34.697 | 10.000 | 24.697 |
| 7 | `selective_state_update_stp_horizontal/b64_h64_d64_s256_r8` | 34.177 | 10.000 | 24.177 |
| 8 | `selective_state_update_stp_simple/b64_h64_d64_s256_r8` | 32.029 | 10.000 | 22.029 |
| 9 | `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 31.449 | 10.000 | 21.449 |
| 10 | `selective_state_update_stp_horizontal/b64_h64_d64_s128_r8_base` | 31.326 | 10.000 | 21.326 |
| 11 | `selective_state_update_mtp_horizontal/b2048_h64_d64_s128_t6_r8_statebf16_official` | 30.221 | 10.000 | 20.221 |
| 12 | `selective_state_update_stp_simple/b64_h64_d64_s128_r8_base` | 30.194 | 10.000 | 20.194 |
| 13 | `selective_state_update_stp_vertical/b64_h64_d64_s128_r8_base` | 29.943 | 10.000 | 19.943 |
| 14 | `selective_state_update_stp_simple/b64_h64_d128_s128_r8` | 29.682 | 10.000 | 19.682 |
| 15 | `selective_state_update_mtp_horizontal/b1_h64_d64_s128_t6_r8_statebf16_official` | 29.647 | 10.000 | 19.647 |
| 16 | `selective_state_update_mtp_simple/b1_h64_d64_s128_t6_r8_statebf16_official` | 24.774 | 10.000 | 14.774 |
| 17 | `selective_state_update_mtp_simple/b512_h64_d64_s128_t6_r8_statebf16_official` | 24.355 | 10.000 | 14.355 |
| 18 | `selective_state_update_stp_vertical/b64_h64_d64_s256_r8` | 23.171 | 10.000 | 13.171 |
| 19 | `flashkda_bf16_fused_m128/h96_fixed8192` | 27.559 | 15.000 | 12.559 |
| 20 | `flashkda_bf16_fused_m128/h96_uniform` | 27.339 | 15.000 | 12.339 |
| 21 | `selective_state_update_mtp_simple/b2048_h64_d64_s128_t6_r8_statebf16_official` | 22.324 | 10.000 | 12.324 |
| 22 | `flash_attention4/s8192_h32kv32` | 20.930 | 10.000 | 10.930 |
| 23 | `flash_attention_backward_sm100/b1_s2048_h16_causal` | 20.079 | 10.000 | 10.079 |
| 24 | `selective_state_update_stp_horizontal/b64_h64_d128_s128_r8` | 20.052 | 10.000 | 10.052 |
| 25 | `flashkda_bf16_fused_m128/h64_mixed` | 24.472 | 15.000 | 9.472 |
| 26 | `flash_attention_backward_sm100/b1_s8192_h16_noncausal` | 19.260 | 10.000 | 9.260 |
| 27 | `deepgemm_fp8_fp4_mega_moe/t64_m64_h7168_i3072_e384_k6_g1` | 19.235 | 10.000 | 9.235 |
| 28 | `tinygemm2_sm100/b1_o128_k720` | 19.155 | 10.000 | 9.155 |
| 29 | `gdn_prefill_sm100/hq8_hv32_s1024x8` | 18.870 | 10.000 | 8.870 |
| 30 | `gdn_prefill_sm100/hq16_hv64_s1x8192` | 18.812 | 10.000 | 8.812 |
| 31 | `deepgemm_sm100_fp8_paged_mqa_logits/b16_n1_mp128_ps64_h64_d128_bf16_fixed` | 23.607 | 15.000 | 8.607 |
| 32 | `nvfp4_quantize_per_token/fp16_128x4_m16384_k7168` | 18.473 | 10.000 | 8.473 |
| 33 | `gdn_prefill_sm100/hq32_hv32_s8192x16` | 18.154 | 10.000 | 8.154 |
| 34 | `mxfp4_quantize/fp16_128x4_m128_k1024` | 17.811 | 10.000 | 7.811 |
| 35 | `flash_attention4/s4096_h32kv4_causal` | 17.685 | 10.000 | 7.685 |
| 36 | `nvfp4_quantize/fp16_128x4_m128_k1024` | 17.613 | 10.000 | 7.613 |
| 37 | `nvfp4_quantize_per_token/fp16_128x4_m128_k1024` | 17.515 | 10.000 | 7.515 |
| 38 | `recurrent_kda_decode_grouped/ver_t8_hv16_b128` | 17.493 | 10.000 | 7.493 |
| 39 | `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 17.452 | 10.000 | 7.452 |
| 40 | `nvfp4_quantize/fp16_128x4_m16384_k7168` | 17.438 | 10.000 | 7.438 |
| 41 | `nvfp4_quantize/fp16_linear_m4096_k4096` | 17.344 | 10.000 | 7.344 |
| 42 | `nvfp4_quantize_per_token/fp16_linear_m4096_k4096` | 17.332 | 10.000 | 7.332 |
| 43 | `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 17.190 | 10.000 | 7.190 |
| 44 | `tinygemm2_sm100/b16_o2880_k2880` | 17.115 | 10.000 | 7.115 |
| 45 | `mxfp4_quantize/fp16_128x4_m16384_k7168` | 17.060 | 10.000 | 7.060 |
| 46 | `selective_state_update_mtp_vertical/b512_h64_d64_s128_t6_r8_statebf16_official` | 16.990 | 10.000 | 6.990 |
| 47 | `flash_attention_backward_sm100/b4_s8192_h16_noncausal` | 16.880 | 10.000 | 6.880 |
| 48 | `mxfp8_quantize/fp16_128x4_m128_k1024` | 16.864 | 10.000 | 6.864 |
| 49 | `mxfp4_quantize/fp16_linear_m4096_k4096` | 16.839 | 10.000 | 6.839 |
| 50 | `mxfp8_quantize/fp16_linear_m4096_k4096` | 16.813 | 10.000 | 6.813 |
| 51 | `mxfp8_quantize/fp16_128x4_m16384_k7168` | 16.565 | 10.000 | 6.565 |
| 52 | `sparse_flashmla_decode_head64/v32_b148_sq2_sk32768_topk16384_p64` | 16.451 | 10.000 | 6.451 |
| 53 | `gdn_decode_bf16_wide_vec_t1/b512_h4_hv8_tv128` | 16.313 | 10.000 | 6.313 |
| 54 | `act_and_mul/silu_bf16_d16384_t32768` | 16.203 | 10.000 | 6.203 |
| 55 | `act_and_mul/silu_fp16_d4096_t1` | 16.163 | 10.000 | 6.163 |
| 56 | `recurrent_kda_decode_one_warp/hv16_b128_tr16_lb` | 16.105 | 10.000 | 6.105 |
| 57 | `flash_attention4/s1024_h32kv4` | 16.009 | 10.000 | 6.009 |
| 58 | `gdn_decode_bf16_wide_vec_t1/b128_h8_hv16_tv128` | 15.879 | 10.000 | 5.879 |
| 59 | `recurrent_kda_decode_one_warp/hv12_b64_tr16_lb` | 15.774 | 10.000 | 5.774 |
| 60 | `fp16_bf16_gemm/fp16_16384x16384x16384` | 15.724 | 10.000 | 5.724 |
| 61 | `fp16_bf16_gemm/bf16_4096x4096x4096` | 25.536 | 20.000 | 5.536 |
| 62 | `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | 15.415 | 10.000 | 5.415 |
| 63 | `gdn_decode_bf16_wide_vec_t1/b16_h16_hv32_tv64` | 15.204 | 10.000 | 5.204 |
| 64 | `recurrent_kda_decode_one_warp/hv16_b8_tr8_lb` | 15.101 | 10.000 | 5.101 |
| 65 | `sparse_flashmla_decode_head64/deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | 14.856 | 10.000 | 4.856 |
| 66 | `deepgemm_sm100_fp8_mqa_logits/s4096_skv8192_h64_d128_bf16_compressed_nocp` | 14.681 | 10.000 | 4.681 |
| 67 | `tinygemm2_sm100/b64_o4096_k3072` | 14.664 | 10.000 | 4.664 |
| 68 | `selective_state_update_stp_vertical/b64_h64_d128_s128_r8` | 14.588 | 10.000 | 4.588 |
| 69 | `deepgemm_sm100_fp8_bmm/bhd_hdr_bhr_b8192_h8_r4096_d1024` | 14.575 | 10.000 | 4.575 |
| 70 | `recurrent_kda_decode_grouped/dec_hv16_b1` | 14.565 | 10.000 | 4.565 |
| 71 | `sparse_flashmla_decode_head64/model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | 14.474 | 10.000 | 4.474 |
| 72 | `deepgemm_sm100_fp4_mqa_logits/s4096_skv8192_h64_d128_bf16_compressed_nocp` | 14.398 | 10.000 | 4.398 |
| 73 | `selective_state_update_mtp_vertical/b2048_h64_d64_s128_t6_r8_statebf16_official` | 14.344 | 10.000 | 4.344 |
| 74 | `deepgemm_sm100_fp8_paged_mqa_logits/b1_n1_mp1_ps64_h64_d128_f32_fixed` | 19.058 | 15.000 | 4.058 |
| 75 | `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | 13.927 | 10.000 | 3.927 |
| 76 | `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g4_m4096_n7168_k8192_gran128_al128_psum` | 13.907 | 10.000 | 3.907 |
| 77 | `silu_and_mul_nvfp4_experts_quantize/fp16_b8_m16_k2048` | 13.837 | 10.000 | 3.837 |
| 78 | `silu_and_mul_nvfp4_experts_quantize/bf16_b8_m512_k2048` | 13.728 | 10.000 | 3.728 |
| 79 | `deepgemm_sm100_fp8_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | 13.698 | 10.000 | 3.698 |
| 80 | `deepgemm_sm100_fp8_bmm/bhr_hdr_bhd_b4096_h8_r4096_d1024` | 13.687 | 10.000 | 3.687 |
| 81 | `deepgemm_sm100_fp8_gemm_1d1d/m4096_n576_k7168` | 13.656 | 10.000 | 3.656 |
| 82 | `deepgemm_sm100_fp4_paged_mqa_logits/b1_n1_mp1_ps32_h64_d128_f32_fixed` | 13.649 | 10.000 | 3.649 |
| 83 | `silu_and_mul_nvfp4_experts_quantize/fp16_b128_m2048_k2048` | 13.596 | 10.000 | 3.596 |
| 84 | `selective_state_update_mtp_vertical/b1_h64_d64_s128_t6_r8_statebf16_official` | 13.567 | 10.000 | 3.567 |
| 85 | `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g8_m4096_n7168_k4096_gran32_al160` | 13.564 | 10.000 | 3.564 |
| 86 | `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | 13.506 | 10.000 | 3.506 |
| 87 | `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | 13.478 | 10.000 | 3.478 |
| 88 | `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g8_m4096_n7168_k3072_psum_zp` | 13.477 | 10.000 | 3.477 |
| 89 | `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g8_m4096_n4096_k2048_bfp4` | 13.471 | 10.000 | 3.471 |
| 90 | `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g16_m7168_n2048_k2048_gran128_al128` | 13.459 | 10.000 | 3.459 |
| 91 | `sparse_flashmla_prefill_head64_phase1/bench_dqk512_hq64_s4096_kv65536_topk512` | 13.406 | 10.000 | 3.406 |
| 92 | `deepgemm_sm100_fp8_gemm_1d1d/m4096_n4096_k7168_bfp4` | 13.350 | 10.000 | 3.350 |
| 93 | `deepgemm_sm100_fp4_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | 13.276 | 10.000 | 3.276 |
| 94 | `deepgemm_sm100_m_grouped_fp8_gemm_masked/g6_m1024_n4096_k2048` | 13.266 | 10.000 | 3.266 |
| 95 | `deepgemm_sm100_m_grouped_fp8_gemm_masked/g32_m192_n4096_k4096_bfp4` | 13.258 | 10.000 | 3.258 |
| 96 | `deepgemm_sm100_tf32_hc_prenorm_gemm/m4096_n24_k7168_s1` | 13.218 | 10.000 | 3.218 |
| 97 | `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g4_m8192_n6144_k7168` | 13.207 | 10.000 | 3.207 |
| 98 | `deepgemm_sm100_fp8_bmm/bhd_bhr_hdr_b4096_h8_r4096_d1024` | 13.195 | 10.000 | 3.195 |
| 99 | `sparse_flashmla_prefill_head64_phase1/bench_dqk512_hq64_s4096_kv8192_topk512` | 13.163 | 10.000 | 3.163 |
| 100 | `deepgemm_sm100_tf32_hc_prenorm_gemm/m128_n24_k16384_s64` | 13.143 | 10.000 | 3.143 |
| 101 | `deepgemm_sm100_m_grouped_fp8_gemm_masked/g32_m192_n6144_k7168` | 13.028 | 10.000 | 3.028 |
| 102 | `deepgemm_sm100_fp4_paged_mqa_logits/b16_n1_mp128_ps64_h64_d128_bf16_fixed` | 12.599 | 10.000 | 2.599 |
| 103 | `deepgemm_sm100_fp8_gemm_1d1d/m4096_n7168_k16384` | 12.565 | 10.000 | 2.565 |
| 104 | `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | 12.532 | 10.000 | 2.532 |
| 105 | `deepgemm_sm100_tf32_hc_prenorm_gemm/m8192_n24_k28672_s1` | 12.492 | 10.000 | 2.492 |
| 106 | `fp16_bf16_gemm/fp16_1024x1024x1024` | 12.389 | 10.000 | 2.389 |

## Per-implementation sweep means

Values are microseconds and are independently recomputable from the five round samples embedded in the JSON evidence and its hashed raw run artifacts.

| workload | migration-before impls (us) | pipeline impls (us) |
|---|---|---|
| `fp16_bf16_gemm/fp16_1024x1024x1024` | tir=6.898133; torch-cublas=6.104186 | tir=6.908679; torch-cublas=6.056907 |
| `fp16_bf16_gemm/fp16_16384x16384x16384` | tir=5965.665677; torch-cublas=5875.866178 | tir=5992.390388; torch-cublas=5910.402856 |
| `fp16_bf16_gemm/bf16_4096x4096x4096` | tir=92.474375; torch-cublas=89.603452; deepgemm-cublaslt=89.592488; deepgemm-bf16=89.875600 | tir=91.814553; torch-cublas=89.322559; deepgemm-cublaslt=89.344196; deepgemm-bf16=89.464480 |
| `nvfp4_gemm/1024x1024x1024` | tir=5.346785; flashinfer=4.567333; cublaslt_nvfp4=4.552319 | tir=5.226142; flashinfer=4.458617; cublaslt_nvfp4=4.526540 |
| `nvfp4_gemm/4096x4096x4096` | tir=29.622323; flashinfer=30.909655; cublaslt_nvfp4=28.676675 | tir=29.645417; flashinfer=31.043186; cublaslt_nvfp4=28.869528 |
| `nvfp4_gemm/16384x16384x16384` | tir=1498.113512; flashinfer=1438.492350; cublaslt_nvfp4=1468.875286 | tir=1516.869516; flashinfer=1441.617783; cublaslt_nvfp4=1433.946040 |
| `deepgemm_fp8_fp4_mega_moe/t64_m64_h7168_i3072_e384_k6_g1` | tirx=1298.800000; deepgemm=1287.600000 | tirx=1300.000000; deepgemm=1287.000000 |
| `deepgemm_fp8_fp4_mega_moe/t8192_m8192_h7168_i3072_e384_k6_g1` | tirx=3460.600000; deepgemm=3461.200000 | tirx=3408.000000; deepgemm=3407.600000 |
| `deepgemm_sm100_fp4_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | tirx=38.173507; deepgemm=39.799527 | tirx=37.903728; deepgemm=39.316323 |
| `deepgemm_sm100_fp4_mqa_logits/s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx=183.699809; deepgemm=181.520306 | tirx=185.430262; deepgemm=182.754323 |
| `deepgemm_sm100_fp4_paged_mqa_logits/b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx=4.068858; deepgemm=4.946223 | tirx=4.360358; deepgemm=5.132609 |
| `deepgemm_sm100_fp4_paged_mqa_logits/b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx=6.286884; deepgemm=6.530648 | tirx=6.357254; deepgemm=6.531888 |
| `deepgemm_sm100_fp8_bmm/bhr_hdr_bhd_b4096_h8_r4096_d1024` | tirx=98.914063; deepgemm=99.272053 | tirx=98.715197; deepgemm=99.235837 |
| `deepgemm_sm100_fp8_bmm/bhd_hdr_bhr_b8192_h8_r4096_d1024` | tirx=231.572833; deepgemm=251.218421 | tirx=228.153917; deepgemm=246.285250 |
| `deepgemm_sm100_fp8_bmm/bhd_bhr_hdr_b4096_h8_r4096_d1024` | tirx=133.145240; deepgemm=136.289274 | tirx=132.856168; deepgemm=136.752348 |
| `deepgemm_sm100_fp8_gemm_1d1d/m4096_n576_k7168` | tirx=18.738403; deepgemm=19.036224 | tirx=18.800738; deepgemm=19.088620 |
| `deepgemm_sm100_fp8_gemm_1d1d/m4096_n7168_k16384` | tirx=332.287677; deepgemm=338.087204 | tirx=322.753503; deepgemm=323.322007 |
| `deepgemm_sm100_fp8_gemm_1d1d/m4096_n4096_k7168_bfp4` | tirx=78.666903; deepgemm=79.141876 | tirx=78.203308; deepgemm=78.519600 |
| `deepgemm_sm100_fp8_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | tirx=39.509401; deepgemm=40.522276 | tirx=40.073678; deepgemm=41.087783 |
| `deepgemm_sm100_fp8_mqa_logits/s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx=180.406142; deepgemm=190.396401 | tirx=181.402290; deepgemm=192.065115 |
| `deepgemm_sm100_fp8_paged_mqa_logits/b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx=4.386609; deepgemm=4.799550; sglang_cutedsl=4.841366 | tirx=4.578870; deepgemm=5.061698; sglang_cutedsl=4.989692 |
| `deepgemm_sm100_fp8_paged_mqa_logits/b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx=6.682834; deepgemm=6.917722; sglang_cutedsl=6.932099 | tirx=6.661127; deepgemm=6.922735; sglang_cutedsl=6.901245 |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g8_m4096_n7168_k4096_gran32_al160` | tirx=908.243662; deepgemm=941.844105 | tirx=914.057067; deepgemm=942.945320 |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g16_m7168_n2048_k2048_gran128_al128` | tirx=584.083536; deepgemm=581.147481 | tirx=582.348883; deepgemm=583.675095 |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous/g4_m4096_n7168_k8192_gran128_al128_psum` | tirx=844.794867; deepgemm=856.848120 | tirx=872.001593; deepgemm=867.541099 |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g4_m8192_n6144_k7168` | tirx=999.715424; deepgemm=1005.461647 | tirx=1006.264731; deepgemm=1021.890973 |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g8_m4096_n4096_k2048_bfp4` | tirx=183.715934; deepgemm=183.732646 | tirx=187.306609; deepgemm=188.747146 |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous/g8_m4096_n7168_k3072_psum_zp` | tirx=533.912933; deepgemm=534.195381 | tirx=560.741171; deepgemm=532.147814 |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked/g32_m192_n6144_k7168` | tirx=327.944223; deepgemm=334.033802 | tirx=330.222273; deepgemm=325.653440 |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked/g6_m1024_n4096_k2048` | tirx=39.468498; deepgemm=39.527588 | tirx=39.413859; deepgemm=39.371697 |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked/g32_m192_n4096_k4096_bfp4` | tirx=115.071050; deepgemm=114.424131 | tirx=113.162021; deepgemm=113.574970 |
| `deepgemm_sm100_tf32_hc_prenorm_gemm/m128_n24_k16384_s64` | tirx=5.263110; deepgemm=5.204370 | tirx=5.256475; deepgemm=5.236095 |
| `deepgemm_sm100_tf32_hc_prenorm_gemm/m8192_n24_k28672_s1` | tirx=83.819590; deepgemm=91.920960 | tirx=83.807994; deepgemm=91.909255 |
| `deepgemm_sm100_tf32_hc_prenorm_gemm/m4096_n24_k7168_s1` | tirx=23.178778; deepgemm=23.811789 | tirx=23.062398; deepgemm=23.682490 |
| `flash_attention4/s1024_h32kv4` | tir=19.420471; flashattn_sm100=20.022291 | tir=19.515974; flashattn_sm100=20.127386 |
| `flash_attention4/s4096_h32kv4_causal` | tir=111.779515; flashattn_sm100=115.700627 | tir=112.036653; flashattn_sm100=115.918669 |
| `flash_attention4/s8192_h32kv32` | tir=772.275308; flashattn_sm100=772.919778 | tir=777.363072; flashattn_sm100=792.673064 |
| `flash_attention_backward_sm100/b1_s2048_h16_causal` | tir=82.619035; flashattn_sm100=81.850796 | tir=82.596874; flashattn_sm100=82.255012 |
| `flash_attention_backward_sm100/b1_s8192_h16_noncausal` | tir=1126.035022; flashattn_sm100=1122.301408 | tir=1107.690467; flashattn_sm100=1105.810184 |
| `flash_attention_backward_sm100/b4_s8192_h16_noncausal` | tir=4417.096018; flashattn_sm100=4451.041369 | tir=4353.467658; flashattn_sm100=4392.820720 |
| `act_and_mul/silu_fp16_d4096_t1` | tirx=2.682575; flashinfer=2.818537 | tirx=2.122868; flashinfer=2.298195 |
| `act_and_mul/silu_bf16_d16384_t32768` | tirx=454.466604; flashinfer=474.854253 | tirx=454.444624; flashinfer=474.881283 |
| `act_and_mul/gelu_tanh_fp16_d11008_t8192` | tirx=97.266651; flashinfer=109.254863 | tirx=97.324502; flashinfer=109.253487 |
| `silu_and_mul_nvfp4_experts_quantize/fp16_b128_m2048_k2048` | tirx=276.450433; flashinfer=301.837906 | tirx=276.790465; flashinfer=297.809545 |
| `silu_and_mul_nvfp4_experts_quantize/bf16_b8_m512_k2048` | tirx=8.916110; flashinfer=10.695360 | tirx=8.891477; flashinfer=10.524191 |
| `silu_and_mul_nvfp4_experts_quantize/fp16_b8_m16_k2048` | tirx=3.310810; flashinfer=4.106696 | tirx=3.423162; flashinfer=4.220119 |
| `gdn_decode_bf16_wide_vec_t1/b16_h16_hv32_tv64` | tirx=8.804985; flashinfer_cutedsl=9.049522 | tirx=8.850474; flashinfer_cutedsl=9.102667 |
| `gdn_decode_bf16_wide_vec_t1/b128_h8_hv16_tv128` | tirx=24.981902; flashinfer_cutedsl=25.811885 | tirx=24.935856; flashinfer_cutedsl=25.481746 |
| `gdn_decode_bf16_wide_vec_t1/b512_h4_hv8_tv128` | tirx=46.244634; flashinfer_cutedsl=46.994240 | tirx=46.364855; flashinfer_cutedsl=47.165866 |
| `gdn_prefill_sm100/hq8_hv32_s1024x8` | tirx=92.815583; flashinfer_cutedsl=95.867342 | tirx=92.236881; flashinfer_cutedsl=95.630230 |
| `gdn_prefill_sm100/hq16_hv64_s1x8192` | tirx=238.880781; flashinfer_cutedsl=249.518142 | tirx=240.395420; flashinfer_cutedsl=250.073768 |
| `gdn_prefill_sm100/hq32_hv32_s8192x16` | tirx=1088.275460; flashinfer_cutedsl=1115.174245 | tirx=1078.554438; flashinfer_cutedsl=1118.187087 |
| `tinygemm2_sm100/b1_o128_k720` | tirx=2.958236; flashinfer_sm100=2.956175 | tirx=2.958391; flashinfer_sm100=2.953334 |
| `tinygemm2_sm100/b16_o2880_k2880` | tirx=7.996638; flashinfer_sm100=8.080661 | tirx=8.047295; flashinfer_sm100=8.145350 |
| `tinygemm2_sm100/b64_o4096_k3072` | tirx=22.066281; flashinfer_sm100=22.250988 | tirx=21.985783; flashinfer_sm100=22.146687 |
| `flashkda_bf16_fused_m128/h96_fixed8192` | tirx=499.125823; flashinfer_m128=506.474148; flashkda_raw=1101.831257 | tirx=498.945085; flashinfer_m128=506.003648; flashkda_raw=1101.770761 |
| `flashkda_bf16_fused_m128/h96_uniform` | tirx=434.088763; flashinfer_m128=438.108010; flashkda_raw=725.677098 | tirx=433.994948; flashinfer_m128=436.565682; flashkda_raw=723.953752 |
| `flashkda_bf16_fused_m128/h64_mixed` | tirx=265.064451; flashinfer_m128=270.347391; flashkda_raw=687.277730 | tirx=268.312593; flashinfer_m128=272.570408; flashkda_raw=688.709372 |
| `recurrent_kda_decode_grouped/dec_hv16_b1` | tirx=3.262802; flashinfer_cutedsl=3.705359 | tirx=3.362606; flashinfer_cutedsl=3.543588 |
| `recurrent_kda_decode_grouped/ver_t8_hv16_b128` | tirx=161.874361; flashinfer_cutedsl=219.105364 | tirx=161.990418; flashinfer_cutedsl=217.457176 |
| `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | tirx=22.103771; flashinfer_cutedsl=31.204223 | tirx=22.293624; flashinfer_cutedsl=32.640861 |
| `recurrent_kda_decode_one_warp/hv16_b8_tr8_lb` | tirx=5.250936; flashinfer_cutedsl=5.428764 | tirx=5.189009; flashinfer_cutedsl=5.424336 |
| `recurrent_kda_decode_one_warp/hv16_b128_tr16_lb` | tirx=26.643509; flashinfer_cutedsl=29.357490 | tirx=26.559925; flashinfer_cutedsl=29.426928 |
| `recurrent_kda_decode_one_warp/hv12_b64_tr16_lb` | tirx=12.675478; flashinfer_cutedsl=13.799970 | tirx=12.841775; flashinfer_cutedsl=13.868557 |
| `selective_state_update_mtp_horizontal/b1_h64_d64_s128_t6_r8_statebf16_official` | tirx=6.980793; flashinfer_cuda=7.455799 | tirx=7.071193; flashinfer_cuda=7.533226 |
| `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | tirx=351.678343; flashinfer_cuda=382.052219 | tirx=351.707970; flashinfer_cuda=381.949872 |
| `selective_state_update_mtp_horizontal/b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx=1388.982272; flashinfer_cuda=1509.772084 | tirx=1389.485693; flashinfer_cuda=1510.499625 |
| `selective_state_update_mtp_simple/b1_h64_d64_s128_t6_r8_statebf16_official` | tirx=4.418844; flashinfer_cuda=5.216036 | tirx=4.429795; flashinfer_cuda=5.228621 |
| `selective_state_update_mtp_simple/b512_h64_d64_s128_t6_r8_statebf16_official` | tirx=384.387913; flashinfer_cuda=400.474250 | tirx=384.443253; flashinfer_cuda=400.494444 |
| `selective_state_update_mtp_simple/b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx=1519.802228; flashinfer_cuda=1583.588430 | tirx=1519.495788; flashinfer_cuda=1583.293754 |
| `selective_state_update_mtp_vertical/b1_h64_d64_s128_t6_r8_statebf16_official` | tirx=17.467338; flashinfer_cuda=18.229035 | tirx=17.413828; flashinfer_cuda=18.226044 |
| `selective_state_update_mtp_vertical/b512_h64_d64_s128_t6_r8_statebf16_official` | tirx=713.255883; flashinfer_cuda=733.343069 | tirx=713.748836; flashinfer_cuda=733.829412 |
| `selective_state_update_mtp_vertical/b2048_h64_d64_s128_t6_r8_statebf16_official` | tirx=2826.134306; flashinfer_cuda=2905.006135 | tirx=2825.755755; flashinfer_cuda=2904.520900 |
| `selective_state_update_stp_horizontal/b64_h64_d64_s128_r8_base` | tirx=27.729307; flashinfer_cuda=28.693216 | tirx=27.962399; flashinfer_cuda=28.876863 |
| `selective_state_update_stp_horizontal/b64_h64_d128_s128_r8` | tirx=48.842483; flashinfer_cuda=49.705608 | tirx=48.704971; flashinfer_cuda=50.138437 |
| `selective_state_update_stp_horizontal/b64_h64_d64_s256_r8` | tirx=47.003245; flashinfer_cuda=47.360600 | tirx=46.904918; flashinfer_cuda=47.096347 |
| `selective_state_update_stp_simple/b64_h64_d64_s128_r8_base` | tirx=37.374017; flashinfer_cuda=38.054288 | tirx=37.332965; flashinfer_cuda=38.104842 |
| `selective_state_update_stp_simple/b64_h64_d128_s128_r8` | tirx=65.192967; flashinfer_cuda=67.340051 | tirx=65.237490; flashinfer_cuda=67.751001 |
| `selective_state_update_stp_simple/b64_h64_d64_s256_r8` | tirx=53.929718; flashinfer_cuda=61.249216 | tirx=54.196454; flashinfer_cuda=61.134536 |
| `selective_state_update_stp_vertical/b64_h64_d64_s128_r8_base` | tirx=27.677038; flashinfer_cuda=31.319779 | tirx=27.622558; flashinfer_cuda=31.458312 |
| `selective_state_update_stp_vertical/b64_h64_d128_s128_r8` | tirx=47.378879; flashinfer_cuda=54.389882 | tirx=47.114604; flashinfer_cuda=54.240921 |
| `selective_state_update_stp_vertical/b64_h64_d64_s256_r8` | tirx=46.160755; flashinfer_cuda=48.865675 | tirx=46.183453; flashinfer_cuda=48.991419 |
| `mxfp4_quantize/fp16_linear_m4096_k4096` | tirx=10.186385; flashinfer=10.338594 | tirx=10.269894; flashinfer=10.444208 |
| `mxfp4_quantize/fp16_128x4_m16384_k7168` | tirx=52.566248; flashinfer=52.973546 | tirx=52.500389; flashinfer=53.121518 |
| `mxfp4_quantize/fp16_128x4_m128_k1024` | tirx=2.872089; flashinfer=2.897688 | tirx=2.500328; flashinfer=2.530755 |
| `mxfp8_quantize/fp16_linear_m4096_k4096` | tirx=11.277346; flashinfer=11.286787 | tirx=11.312196; flashinfer=11.232849 |
| `mxfp8_quantize/fp16_128x4_m16384_k7168` | tirx=60.063539; flashinfer=60.471364 | tirx=60.953667; flashinfer=60.406779 |
| `mxfp8_quantize/fp16_128x4_m128_k1024` | tirx=2.542393; flashinfer=2.600443 | tirx=2.608777; flashinfer=2.674433 |
| `nvfp4_quantize/fp16_linear_m4096_k4096` | tirx=10.409722; flashinfer=10.492753 | tirx=10.504392; flashinfer=10.444213 |
| `nvfp4_quantize/fp16_128x4_m16384_k7168` | tirx=55.974532; flashinfer=55.820552 | tirx=54.997001; flashinfer=55.364002 |
| `nvfp4_quantize/fp16_128x4_m128_k1024` | tirx=2.528560; flashinfer=2.547946 | tirx=2.447097; flashinfer=2.479128 |
| `nvfp4_quantize_per_token/fp16_linear_m4096_k4096` | tirx=12.114072; flashinfer=13.146416 | tirx=12.007335; flashinfer=13.190352 |
| `nvfp4_quantize_per_token/fp16_128x4_m16384_k7168` | tirx=57.657632; flashinfer=59.177569 | tirx=57.623681; flashinfer=59.192759 |
| `nvfp4_quantize_per_token/fp16_128x4_m128_k1024` | tirx=2.716754; flashinfer=2.847174 | tirx=2.696318; flashinfer=2.870678 |
| `sparse_flashmla_decode_head64/deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | tirx=136.921414; flashmla=144.831201 | tirx=137.647645; flashmla=145.409300 |
| `sparse_flashmla_decode_head64/model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx=16.686527; flashmla=21.368004 | tirx=16.835748; flashmla=21.378831 |
| `sparse_flashmla_decode_head64/v32_b148_sq2_sk32768_topk16384_p64` | tirx=913.107551; flashmla=975.140784 | tirx=911.876367; flashmla=975.045608 |
| `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx=1708.361759; flashmla=1723.298454 | tirx=1705.811238; flashmla=1742.349197 |
| `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx=1859.063621; flashmla=1879.723558 | tirx=1854.835107; flashmla=1899.239663 |
| `sparse_flashmla_prefill_head128_phase1/bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx=1856.457190; flashmla=1852.304343; trtllm_gen=2075.737100 | tirx=1850.019805; flashmla=1862.199221; trtllm_gen=2070.276479 |
| `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx=1149.234290; flashmla=1167.094273 | tirx=1154.760153; flashmla=1156.304111 |
| `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx=1160.496776; flashmla=1181.984296 | tirx=1158.299271; flashmla=1177.398345 |
| `sparse_flashmla_prefill_head128_small_topk_phase1/bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx=1191.151501; flashmla=1211.628474 | tirx=1181.296290; flashmla=1206.219014 |
| `sparse_flashmla_prefill_head64_phase1/bench_dqk512_hq64_s4096_kv8192_topk512` | tirx=364.688573; flashmla=369.111778 | tirx=364.338972; flashmla=371.115704 |
| `sparse_flashmla_prefill_head64_phase1/bench_dqk512_hq64_s4096_kv65536_topk512` | tirx=380.382487; flashmla=385.602793 | tirx=380.165862; flashmla=385.864564 |
| `sparse_flashmla_prefill_head64_phase1/bench_dqk576_hq64_s4096_kv32768_topk512` | tirx=384.218218; flashmla=393.249710; trtllm_gen=452.617355 | tirx=385.904261; flashmla=394.314121; trtllm_gen=460.361347 |
