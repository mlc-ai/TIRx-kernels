# NVIDIA Thor versus B200 TIRx performance

> This historical cross-machine comparison is retained as broad porting
> evidence, not as the primary performance baseline. For controlled same-Thor
> comparisons against FlashInfer, see
> [THOR_NATIVE_BASELINE.md](THOR_NATIVE_BASELINE.md).

Measured on 2026-09-03 using the default representative workload roster.

## At a glance

| Question | Result | Interpretation |
|---|---:|---|
| Does the admitted Thor set run correctly? | **90 kernels / 9,204 configs passed** | Every admitted config compiled, launched, and passed its numerical oracle |
| Did the performance suite finish? | **254/254 workloads** across 87 kernels | 0 failures; 0 interference retries |
| How much has an exact B200 comparison? | **183/254 rows** | 5 matching baseline rows failed and 66 rows are absent |
| What is the aggregate absolute performance? | **9.6% of B200 throughput** | Geometric-mean Thor/B200 latency is 10.433x; median is 12.205x |
| What is the rough per-SM efficiency? | **70.9% of B200** | Normalized for 20 versus 148 SMs; not a hardware peak metric |

## Bottom line

The admitted single-GPU Thor port is functionally complete: 9,204 numerical configurations and all 254 representative performance workloads pass. Absolute throughput averages 9.6% of the historical B200 baseline. Pure SM-count scaling would suggest 13.5%; the measured result therefore reaches roughly 70.9% of that per-SM-normalized level on this mixed suite.

This says the port works, not that it is fully tuned. Large GEMMs and several attention/SSM schedules remain the clearest 20-SM tuning targets. The geometric mean mixes compute-, bandwidth-, and latency-bound kernels and is not a model-level or theoretical-peak score.

The unsupported boundary, including the single-GPU MegaMoE host-layout blocker, is documented in [THOR_VALIDATION.md](THOR_VALIDATION.md).

## Performance by kernel family

Relative throughput is the reciprocal of the Thor/B200 latency ratio. The per-SM column multiplies it by 148/20 and should be read only as a coarse scheduling diagnostic. Values above 100% are expected for latency-bound small operations and do not imply a higher Thor SM hardware peak.

| Family | Thor kernels | Thor rows | B200 matches | Thor/B200 latency | Relative throughput | Per-SM normalized |
|---|---:|---:|---:|---:|---:|---:|
| GEMM / MoE | 22 | 61 | 35 | 14.495x | 6.9% | 51.1% |
| Attention | 16 | 48 | 27 | 11.831x | 8.5% | 62.5% |
| Normalization | 9 | 27 | 27 | 4.493x | 22.3% | 164.7% |
| Recurrent / SSM | 28 | 82 | 61 | 13.999x | 7.1% | 52.9% |
| TopK / sorting | 6 | 18 | 15 | 4.737x | 21.1% | 156.2% |
| Activation / quantization | 6 | 18 | 18 | 11.493x | 8.7% | 64.4% |

## Representative GEMM throughput

Throughput uses the conventional `2*M*N*K` operation count. NVFP4 values are effective throughput and include neither scale-processing operations nor any sparsity multiplier.

| Kernel | Config | Thor µs | B200 µs | Thor effective TFLOP/s | B200 effective TFLOP/s | Thor/B200 throughput |
|---|---|---:|---:|---:|---:|---:|
| `fp16_bf16_gemm` | `bf16_4096x4096x4096` | 1251.646 | 92.495 | 109.807 | 1485.903 | 7.4% |
| `fp16_bf16_gemm` | `fp16_1024x1024x1024` | 98.186 | 6.626 | 21.872 | 324.114 | 6.7% |
| `fp16_bf16_gemm` | `fp16_16384x16384x16384` | 312464.042 | 5702.039 | 28.151 | 1542.622 | 1.8% |
| `nvfp4_gemm` | `1024x1024x1024` | 51.733 | 5.211 | 41.511 | 412.138 | 10.1% |
| `nvfp4_gemm` | `16384x16384x16384` | 36218.299 | 1528.963 | 242.863 | 5752.981 | 4.2% |
| `nvfp4_gemm` | `4096x4096x4096` | 534.702 | 29.305 | 257.038 | 4690.013 | 5.5% |

The FP16 16384³ row reaches only 28.15 effective TFLOP/s on Thor, below the 4096³ BF16 row's 109.81 TFLOP/s. That inversion is a concrete tuning target: the current B200-oriented schedule does not scale well to Thor's 20-SM device at that shape.

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
| TIRx-kernels revision | `3c25997b` | `f727de3d-dirty` |
| CUDA/PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 | CUDA 13.2 / PyTorch 2.13.0+cu132 |

Thor's Triton 3.5.1 package bundles CUDA 12.8 CUPTI, which cannot initialize against this CUDA 13.1 Thor stack. The run therefore set `TRITON_CUPTI_LIB_PATH=/usr/local/cuda-13.1/extras/CUPTI/lib64`; both final columns still use the same Proton timer protocol.

This is a historical cross-machine comparison, not a controlled hardware-only A/B: the TVM, TIRx, CUDA, PyTorch, and CUPTI revisions differ, and the repository baseline labels its TIRx checkout as dirty. A publication-grade comparison requires rerunning this exact TIRx/TVM revision on a B200.

## Cross-session stability

Against the preceding same-protocol Thor run, 69 common rows have geometric-mean current/prior latency **0.993x** and median **0.995x**. The largest individual shift is `nvfp4_gemm/16384x16384x16384`: 71375.402 µs to 36218.299 µs (0.507x).

The aggregate is repeatable, but individual absolute times can move substantially between sessions under dynamic clocks. The complete table uses only the final, single-piece 254-row run; no samples were spliced from the earlier run.

## Appendix A: per-kernel summary

`Thor/B200 latency > 1` means Thor is slower. Relative throughput is its reciprocal.

| Kernel | Thor rows | Matched B200 rows | Geomean Thor/B200 latency | Thor relative throughput |
|---|---:|---:|---:|---:|
| `act_and_mul` | 3 | 3 | 8.843x | 11.3% |
| `agent_evolved_kda_forward_b1_t8192_h96` | 1 | 0 | — | — |
| `cudnn_sm100_bsa_backward_blk128` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_backward_blk64` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_forward_blk128` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_forward_blk64` | 3 | 0 | — | — |
| `cudnn_sm100_bsa_forward_combine_blk64` | 3 | 0 | — | — |
| `cudnn_sm100_csa_compressor_fwd` | 3 | 0 | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` | 3 | 0 | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` | 3 | 0 | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | 3 | 0 | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | 3 | 0 | — | — |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | 3 | 0 | — | — |
| `cudnn_sm100_gdn2_bprop_f16` | 3 | 0 | — | — |
| `cudnn_sm100_gdn2_prefill_f16` | 3 | 0 | — | — |
| `cudnn_sm100_gdn_bprop_f16` | 3 | 0 | — | — |
| `cudnn_sm100_gdn_prefill_f16` | 3 | 0 | — | — |
| `cudnn_sm100_gdn_recompute_f16` | 3 | 0 | — | — |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | 3 | 0 | — | — |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | 2 | 0 | — | — |
| `cudnn_sm100_kda_bprop_f16` | 3 | 0 | — | — |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | 3 | 0 | — | — |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | 3 | 0 | — | — |
| `deepgemm_sm100_fp4_mqa_logits` | 2 | 2 | 10.909x | 9.2% |
| `deepgemm_sm100_fp4_paged_mqa_logits` | 2 | 2 | 4.759x | 21.0% |
| `deepgemm_sm100_fp8_bmm` | 3 | 3 | 19.277x | 5.2% |
| `deepgemm_sm100_fp8_gemm_1d1d` | 3 | 3 | 14.749x | 6.8% |
| `deepgemm_sm100_fp8_mqa_logits` | 2 | 2 | 10.313x | 9.7% |
| `deepgemm_sm100_fp8_paged_mqa_logits` | 2 | 2 | 5.614x | 17.8% |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | 3 | 3 | 19.174x | 5.2% |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | 3 | 3 | 14.873x | 6.7% |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | 3 | 3 | 25.551x | 3.9% |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | 3 | 3 | 19.320x | 5.2% |
| `fast_topk_clusters` | 3 | 3 | 8.245x | 12.1% |
| `filtered_topk` | 3 | 3 | 2.334x | 42.8% |
| `flash_attention4` | 3 | 3 | 10.478x | 9.5% |
| `flash_attention_backward_sm100` | 3 | 3 | 11.923x | 8.4% |
| `flashinfer_add_rmsnorm_fp4quant` | 3 | 3 | 3.626x | 27.6% |
| `flashinfer_fused_add_rmsnorm_quant` | 3 | 3 | 4.505x | 22.2% |
| `flashinfer_fused_dit_layernorm` | 3 | 3 | 25.439x | 3.9% |
| `flashinfer_layernorm` | 3 | 3 | 3.312x | 30.2% |
| `flashinfer_qk_rmsnorm` | 3 | 3 | 2.385x | 41.9% |
| `flashinfer_rmsnorm` | 3 | 3 | 3.404x | 29.4% |
| `flashinfer_rmsnorm_fp4quant` | 3 | 3 | 2.506x | 39.9% |
| `flashinfer_rmsnorm_quant` | 3 | 3 | 5.851x | 17.1% |
| `flashkda_bf16_fused_m128` | 3 | 3 | 12.126x | 8.2% |
| `flashkda_decode_t1_precomputed` | 3 | 3 | 11.692x | 8.6% |
| `flashkda_decode_t2_precomputed` | 3 | 3 | 19.388x | 5.2% |
| `flashkda_decode_t3_lower_bound` | 3 | 3 | 8.587x | 11.6% |
| `flashkda_decode_t4_precomputed` | 3 | 3 | 20.942x | 4.8% |
| `flashkda_decode_t5_gram` | 3 | 3 | 15.134x | 6.6% |
| `flashkda_decode_t6_gram` | 3 | 3 | 12.486x | 8.0% |
| `fp16_bf16_gemm` | 3 | 3 | 22.232x | 4.5% |
| `gdn_cp_prefill_sm100` | 3 | 3 | 5.767x | 17.3% |
| `gdn_decode_bf16_ilp4` | 3 | 3 | 6.083x | 16.4% |
| `gdn_decode_bf16_wide_vec_mtp` | 3 | 3 | 21.460x | 4.7% |
| `gdn_decode_bf16_wide_vec_t1` | 3 | 3 | 24.059x | 4.2% |
| `gdn_decode_fp32_mtp_warp` | 3 | 2 | 31.811x | 3.1% |
| `gdn_prefill_sm100` | 3 | 3 | 10.846x | 9.2% |
| `msa_sparse_atten_fwd_combine_sm100` | 3 | 0 | — | — |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100` | 3 | 3 | 25.337x | 3.9% |
| `msa_sparse_atten_fwd_sm100` | 3 | 3 | 27.718x | 3.6% |
| `msa_sparse_prepare_flat_schedule_sm100` | 3 | 3 | 5.191x | 19.3% |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | 3 | 3 | 7.242x | 13.8% |
| `mxfp4_quantize` | 3 | 3 | 15.295x | 6.5% |
| `mxfp8_quantize` | 3 | 3 | 12.425x | 8.0% |
| `nvfp4_gemm` | 3 | 3 | 16.250x | 6.2% |
| `nvfp4_quantize` | 3 | 3 | 12.045x | 8.3% |
| `nvfp4_quantize_per_token` | 3 | 3 | 10.180x | 9.8% |
| `radix_topk_multi_cta` | 3 | 3 | 1.938x | 51.6% |
| `radix_topk_single_cta` | 3 | 3 | 5.064x | 19.7% |
| `recurrent_kda_decode_grouped` | 3 | 3 | 12.478x | 8.0% |
| `recurrent_kda_decode_one_warp` | 3 | 3 | 18.103x | 5.5% |
| `rmsnorm` | 3 | 3 | 4.553x | 22.0% |
| `selective_state_update_mtp_horizontal` | 3 | 3 | 6.686x | 15.0% |
| `selective_state_update_mtp_simple` | 3 | 3 | 7.817x | 12.8% |
| `selective_state_update_mtp_vertical` | 3 | 2 | 4.874x | 20.5% |
| `selective_state_update_stp_horizontal` | 3 | 3 | 47.835x | 2.1% |
| `selective_state_update_stp_simple` | 3 | 3 | 22.745x | 4.4% |
| `selective_state_update_stp_vertical` | 3 | 3 | 29.626x | 3.4% |
| `silu_and_mul_nvfp4_experts_quantize` | 3 | 3 | 11.184x | 8.9% |
| `sparse_flashmla_decode_head64` | 3 | 3 | 8.267x | 12.1% |
| `sparse_flashmla_prefill_head128_phase1` | 3 | 3 | 12.780x | 7.8% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | 3 | 3 | 12.633x | 7.9% |
| `sparse_flashmla_prefill_head64_phase1` | 3 | 3 | 13.032x | 7.7% |
| `stable_sort_topk_by_value` | 3 | 0 | — | — |
| `tinygemm2_sm100` | 3 | 3 | 11.719x | 8.5% |

## Appendix B: complete workload results

Thor CV is the population coefficient of variation across five round means. **60** rows exceed 10% and are marked `†`; repeat those rows under locked clocks before using small differences for tuning decisions.

| Kernel | Config | Thor Proton µs | Thor CV | B200 Proton µs | Thor/B200 latency | Thor relative throughput |
|---|---|---:|---:|---:|---:|---:|
| `act_and_mul` | `gelu_tanh_fp16_d11008_t8192` | 2294.413 | 1.0% | 97.637 | 23.499x | 4.3% |
| `act_and_mul` | `silu_bf16_d16384_t32768` | 13709.707 | 1.0% | 617.343 | 22.208x | 4.5% |
| `act_and_mul` | `silu_fp16_d4096_t1` | 2.753 | 6.7% | 2.077 | 1.325x | 75.5% |
| `agent_evolved_kda_forward_b1_t8192_h96` | `b1_t8192_h96_k128_v128_bf16` | 6300.395 | 2.9% | — | — | — |
| `cudnn_sm100_bsa_backward_blk128` | `p00_b1_h1_d64_sq128_skv4096_kv16` | 45.402 | 16.7% † | — | — | — |
| `cudnn_sm100_bsa_backward_blk128` | `p02_b2_h8_d128_sq4096_skv8191_kv32` | 3753.887 | 0.6% | — | — | — |
| `cudnn_sm100_bsa_backward_blk128` | `p11_b1_h2_d128_sq524288_skv8192_kv32_qb4096_g8` | 59745.606 | 2.3% | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p00_b1_h1_sq64_skv4096_kv16_nomask` | 104.379 | 17.6% † | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask` | 2249.968 | 1.0% | — | — | — |
| `cudnn_sm100_bsa_backward_blk64` | `p11_b1_h1_sq192000_skv8192_maxkv16_var_mask_auto1024_i64kv` | 4798.705 | 0.5% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p00_bf16_d64_mha` | 512.478 | 0.3% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p08_bf16_d128_mqa` | 537.603 | 0.5% | — | — | — |
| `cudnn_sm100_bsa_forward_blk128` | `p13_fp16_d96_gqa` | 509.720 | 0.2% | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p00_b1_h1_sq64_skv4096_kv16_nomask_s1_static` | 15.479 | 22.5% † | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p04_b1_h8_sq4096_skv8192_maxkv32_var_mask_s1_clc` | 441.089 | 13.0% † | — | — | — |
| `cudnn_sm100_bsa_forward_blk64` | `p09_b1_h8_sq2048_skv65536_kv512_nomask_s8_static` | 4455.363 | 5.9% | — | — | — |
| `cudnn_sm100_bsa_forward_combine_blk64` | `b1_h4_sq1024_s2` | 52.417 | 6.0% | — | — | — |
| `cudnn_sm100_bsa_forward_combine_blk64` | `b1_h4_sq2048_s4` | 175.493 | 1.0% | — | — | — |
| `cudnn_sm100_bsa_forward_combine_blk64` | `b1_h8_sq2048_s8` | 387.768 | 10.1% † | — | — | — |
| `cudnn_sm100_csa_compressor_fwd` | `perf_b1_s8192_d128_c2` | 82.357 | 10.1% † | — | — | — |
| `cudnn_sm100_csa_compressor_fwd` | `perf_b3_s8192_d128_c2` | 219.362 | 4.9% | — | — | — |
| `cudnn_sm100_csa_compressor_fwd` | `perf_b3_s8192_d512_c2` | 675.013 | 6.5% | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` | `perf00_m1024_n1024_k1024_e4_e8v32_bf16_kk_m_t128x128_c1x1_l1` | 75.152 | 6.3% | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` | `perf03_m4096_n4096_k4096_f4_e8v16_f16_kk_m_t128x128_c2x1_l1` | 485.707 | 14.8% † | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_amax` | `perf07_m8192_n8192_k8192_f4_e4v16_f4_kk_n_t128x128_c4x2_l2` | 15241.653 | 0.5% | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` | `anchor_m1024_n1024_k1024_l1` | 96.676 | 7.4% | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` | `anchor_m4096_n4096_k4096_l1` | 760.458 | 10.8% † | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant` | `anchor_m8192_n8192_k8192_l2` | 19286.529 | 3.6% | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | `perf00_m4096_n4096_k4096_l1_e4_e8v32_bf16_bf16_mn_n_t256x256_c2x1_vf32_l1` | 1079.552 | 20.0% † | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | `perf02_m1024_n1024_k1024_l1_f4_e8v16_bf16_bf16_kk_n_t128x128_c1x1_vf32_l1` | 60.717 | 8.2% | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant` | `perf22_m8192_n8192_k8192_l2_f4_e8v16_bf16_bf16_kk_n_t256x64_c2x1_vf32_l2` | 44359.160 | 15.6% † | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | `perf00_m1024_n1024_k1024_e4_e8v32_bf16_bf16_kk_m_t128x128_c1x1_sf32_l1` | 58.029 | 13.8% † | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | `perf15_m4096_n4096_k4096_e4_e8v32_e4_f32_kk_n_t128x128_c1x1_sf32_l1` | 968.327 | 0.8% | — | — | — |
| `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant` | `perf30_m8192_n8192_k8192_e4_e8v32_f32_f32_kk_n_t128x128_c1x1_vf32_l2` | 56673.186 | 2.2% | — | — | — |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | `perf00_m1024_n1024_k1024_bf16_f32_bf16_bf16_kk_m_t128x128_c1x1_l1` | 116.451 | 4.2% | — | — | — |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | `perf03_m4096_n4096_k4096_f16_f16_bf16_bf16_kk_n_t256x64_c2x2_l1` | 2766.611 | 0.9% | — | — | — |
| `cudnn_sm100_dense_gemm_persistent_swiglu` | `perf09_m8192_n8192_k8192_e5_f32_f32_bf16_mn_n_t256x128_c4x4_l2` | 30779.489 | 0.5% | — | — | — |
| `cudnn_sm100_gdn2_bprop_f16` | `perf_basic_b1_s2048_h16` | 1134.353 | 8.3% | — | — | — |
| `cudnn_sm100_gdn2_bprop_f16` | `perf_l2_b1_s8192_h16` | 4090.847 | 8.8% | — | — | — |
| `cudnn_sm100_gdn2_bprop_f16` | `perf_l2_b4_s8192_h64` | 60734.482 | 2.8% | — | — | — |
| `cudnn_sm100_gdn2_prefill_f16` | `perf_b1_s8192_h64_nostate` | 5394.394 | 7.5% | — | — | — |
| `cudnn_sm100_gdn2_prefill_f16` | `perf_b4_s32768_h64_state` | 188344.224 | 1.0% | — | — | — |
| `cudnn_sm100_gdn2_prefill_f16` | `perf_b4_s8192_h64_nostate` | 20125.443 | 0.5% | — | — | — |
| `cudnn_sm100_gdn_bprop_f16` | `perf_b4_s32768_h64` | 103169.995 | 1.6% | — | — | — |
| `cudnn_sm100_gdn_bprop_f16` | `perf_basic_b1_s2048_h16` | 507.548 | 0.5% | — | — | — |
| `cudnn_sm100_gdn_bprop_f16` | `perf_basic_b1_s8192_h16` | 1825.979 | 0.2% | — | — | — |
| `cudnn_sm100_gdn_prefill_f16` | `perf_b1_s8192_h64_nostate` | 2931.226 | 9.3% | — | — | — |
| `cudnn_sm100_gdn_prefill_f16` | `perf_b4_s32768_h64_state` | 71593.266 | 2.1% | — | — | — |
| `cudnn_sm100_gdn_prefill_f16` | `perf_b4_s8192_h64_nostate` | 10583.083 | 1.3% | — | — | — |
| `cudnn_sm100_gdn_recompute_f16` | `perf_b1_s8192_h64_nostate` | 3332.707 | 6.3% | — | — | — |
| `cudnn_sm100_gdn_recompute_f16` | `perf_b4_s32768_h64_state` | 46115.295 | 1.3% | — | — | — |
| `cudnn_sm100_gdn_recompute_f16` | `perf_b4_s8192_h64_nostate` | 12744.545 | 5.8% | — | — | — |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | `t2048_k1536_h128_w_out_in_false` | 4082.140 | 0.5% | — | — | — |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | `t4096_k1536_h128_w_out_in_false` | 5755.312 | 0.8% | — | — | — |
| `cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` | `t4096_k1536_h128_w_out_in_true` | 6961.600 | 1.1% | — | — | — |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | `t2048_k1536_h128` | 3054.062 | 2.2% | — | — | — |
| `cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in` | `t4096_k1536_h128` | 4109.924 | 4.9% | — | — | — |
| `cudnn_sm100_kda_bprop_f16` | `perf_basic_b1_s2048_h16` | 906.781 | 9.2% | — | — | — |
| `cudnn_sm100_kda_bprop_f16` | `perf_l2_b1_s8192_h16` | 3055.183 | 9.7% | — | — | — |
| `cudnn_sm100_kda_bprop_f16` | `perf_l2_b4_s8192_h64` | 46220.720 | 1.3% | — | — | — |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | `perf00_e4_t4096_n2048_k2048_dnst_sw_e4_e8v32_cbf16_de4_bk_t256x256_c2x1_bpv` | 807.248 | 3.2% | — | — | — |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | `perf00_e8_t16384_n4096_k8192_dnst_sw_e4_e8v32_cbf16_de4_bk_t256x256_c2x1_bpv` | 14900.818 | 3.7% | — | — | — |
| `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias` | `perf00_e8_t32768_n4096_k7168_dnst_sw_e4_e8v32_cbf16_de4_bk_t256x256_c2x1_bpv` | 36736.292 | 6.0% | — | — | — |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | `perf00_e4_t4096_n2048_k2048_dnst_sw_cbf16_dbf16_bk_t256x256_c2x1_vb` | 953.843 | 6.7% | — | — | — |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | `perf00_e8_t16384_n4096_k8192_dnst_sw_cbf16_dbf16_bk_t256x256_c2x1_vb` | 38267.536 | 2.1% | — | — | — |
| `cudnn_sm100_moe_grouped_gemm_dglu_dbias` | `perf00_e8_t32768_n4096_k7168_dnst_sw_cbf16_dbf16_bk_t256x256_c2x1_vb` | 72759.509 | 1.6% | — | — | — |
| `deepgemm_sm100_fp4_mqa_logits` | `s2048_skv4096_h64_d128_f32_dense_cp` | 395.260 | 0.9% | 37.558 | 10.524x | 9.5% |
| `deepgemm_sm100_fp4_mqa_logits` | `s4096_skv8192_h64_d128_bf16_compressed_nocp` | 2077.529 | 0.1% | 183.720 | 11.308x | 8.8% |
| `deepgemm_sm100_fp4_paged_mqa_logits` | `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | 78.333 | 7.5% | 6.349 | 12.337x | 8.1% |
| `deepgemm_sm100_fp4_paged_mqa_logits` | `b1_n1_mp1_ps32_h64_d128_f32_fixed` | 7.402 | 5.3% | 4.031 | 1.836x | 54.5% |
| `deepgemm_sm100_fp8_bmm` | `bhd_bhr_hdr_b4096_h8_r4096_d1024` | 3145.221 | 4.3% | 136.554 | 23.033x | 4.3% |
| `deepgemm_sm100_fp8_bmm` | `bhd_hdr_bhr_b8192_h8_r4096_d1024` | 4034.757 | 8.9% | 222.804 | 18.109x | 5.5% |
| `deepgemm_sm100_fp8_bmm` | `bhr_hdr_bhd_b4096_h8_r4096_d1024` | 1707.425 | 4.2% | 99.425 | 17.173x | 5.8% |
| `deepgemm_sm100_fp8_gemm_1d1d` | `m4096_n4096_k7168_bfp4` | 1121.287 | 5.6% | 130.636 | 8.583x | 11.7% |
| `deepgemm_sm100_fp8_gemm_1d1d` | `m4096_n576_k7168` | 342.577 | 3.2% | 18.803 | 18.220x | 5.5% |
| `deepgemm_sm100_fp8_gemm_1d1d` | `m4096_n7168_k16384` | 6903.150 | 26.2% † | 336.444 | 20.518x | 4.9% |
| `deepgemm_sm100_fp8_mqa_logits` | `s2048_skv4096_h64_d128_f32_dense_cp` | 394.976 | 3.3% | 47.249 | 8.360x | 12.0% |
| `deepgemm_sm100_fp8_mqa_logits` | `s4096_skv8192_h64_d128_bf16_compressed_nocp` | 2244.913 | 0.6% | 176.452 | 12.723x | 7.9% |
| `deepgemm_sm100_fp8_paged_mqa_logits` | `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | 115.228 | 5.6% | 6.753 | 17.063x | 5.9% |
| `deepgemm_sm100_fp8_paged_mqa_logits` | `b1_n1_mp1_ps64_h64_d128_f32_fixed` | 7.828 | 3.0% | 4.238 | 1.847x | 54.1% |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | `g16_m7168_n2048_k2048_gran128_al128` | 16204.831 | 3.6% | 912.286 | 17.763x | 5.6% |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | `g4_m4096_n7168_k8192_gran128_al128_psum` | 15082.651 | 4.7% | 857.482 | 17.589x | 5.7% |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | `g8_m4096_n7168_k4096_gran32_al160` | 20484.889 | 4.8% | 907.910 | 22.563x | 4.4% |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | `g4_m8192_n6144_k7168` | 12982.313 | 1.7% | 1000.205 | 12.980x | 7.7% |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | `g8_m4096_n4096_k2048_bfp4` | 3384.927 | 6.1% | 186.046 | 18.194x | 5.5% |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | `g8_m4096_n7168_k3072_psum_zp` | 7125.828 | 3.7% | 511.447 | 13.933x | 7.2% |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | `g32_m192_n4096_k4096_bfp4` | 3028.458 | 0.4% | 115.260 | 26.275x | 3.8% |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | `g32_m192_n6144_k7168` | 12530.496 | 4.4% | 338.776 | 36.988x | 2.7% |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | `g6_m1024_n4096_k2048` | 672.276 | 6.0% | 39.169 | 17.164x | 5.8% |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | `m128_n24_k16384_s64` | 63.685 | 10.7% † | 5.240 | 12.153x | 8.2% |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | `m4096_n24_k7168_s1` | 433.191 | 6.6% | 23.165 | 18.700x | 5.3% |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | `m8192_n24_k28672_s1` | 2678.967 | 4.5% | 84.430 | 31.730x | 3.2% |
| `fast_topk_clusters` | `f32_plain_b16_l4096_k256` | 12.773 | 31.9% † | 6.645 | 1.922x | 52.0% |
| `fast_topk_clusters` | `f32_plain_b64_l16384_k256` | 239.070 | 39.8% † | 13.726 | 17.418x | 5.7% |
| `fast_topk_clusters` | `f32_plain_b64_l65536_k1024` | 348.857 | 37.9% † | 20.837 | 16.742x | 6.0% |
| `filtered_topk` | `f32_plain_det_r2_l524288_k256_endbit` | 173.290 | 35.3% † | 114.972 | 1.507x | 66.3% |
| `filtered_topk` | `f32_plain_r4_l8192_k256` | 11.637 | 25.5% † | 7.773 | 1.497x | 66.8% |
| `filtered_topk` | `f32_plain_r64_l8192_k256` | 47.733 | 33.0% † | 8.471 | 5.635x | 17.7% |
| `flash_attention4` | `s1024_h32kv4` | 235.342 | 23.2% † | 19.374 | 12.147x | 8.2% |
| `flash_attention4` | `s4096_h32kv4_causal` | 950.625 | 4.3% | 112.434 | 8.455x | 11.8% |
| `flash_attention4` | `s8192_h32kv32` | 8722.082 | 0.5% | 778.739 | 11.200x | 8.9% |
| `flash_attention_backward_sm100` | `b1_s2048_h16_causal` | 911.697 | 0.7% | 82.141 | 11.099x | 9.0% |
| `flash_attention_backward_sm100` | `b1_s8192_h16_noncausal` | 13535.919 | 0.2% | 1081.811 | 12.512x | 8.0% |
| `flash_attention_backward_sm100` | `b4_s8192_h16_noncausal` | 53345.010 | 0.4% | 4370.638 | 12.205x | 8.2% |
| `flashinfer_add_rmsnorm_fp4quant` | `bench_nv_3d_bf16_b32_s32_h128_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | 10.338 | 19.7% † | 2.960 | 3.493x | 28.6% |
| `flashinfer_add_rmsnorm_fp4quant` | `bench_nv_bf16_m32_h4096_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | 12.379 | 13.7% † | 5.810 | 2.130x | 46.9% |
| `flashinfer_add_rmsnorm_fp4quant` | `bench_nv_large_bf16_m64_h8192_b16_e4m3_sw0_both0_yn0_pdl0_eps1e6_gsnone_preallocated_random` | 38.561 | 14.1% † | 6.016 | 6.409x | 15.6% |
| `flashinfer_fused_add_rmsnorm_quant` | `bf16_e4m3_m32_h4096_xc_rc_yc_pdl0_s1` | 10.123 | 5.1% | 3.537 | 2.862x | 34.9% |
| `flashinfer_fused_add_rmsnorm_quant` | `bf16_e4m3_m32_h4096_xc_rc_yc_pdl1_s1` | 12.828 | 4.1% | 3.572 | 3.591x | 27.8% |
| `flashinfer_fused_add_rmsnorm_quant` | `bf16_e4m3_m64_h8192_xc_rc_yc_pdl0_s1` | 33.155 | 6.9% | 3.728 | 8.893x | 11.2% |
| `flashinfer_fused_dit_layernorm` | `grgb_bf16_b1_r1920` | 397.982 | 6.9% | 16.601 | 23.974x | 4.2% |
| `flashinfer_fused_dit_layernorm` | `grss_bf16_b4_r1920` | 1880.636 | 4.7% | 72.984 | 25.768x | 3.9% |
| `flashinfer_fused_dit_layernorm` | `rss_bf16_b1_r768` | 233.106 | 6.4% | 8.747 | 26.651x | 3.8% |
| `flashinfer_layernorm` | `bf16_m128_h1024_xc_yc_pdl0_eps1e6` | 8.500 | 25.5% † | 3.130 | 2.715x | 36.8% |
| `flashinfer_layernorm` | `bf16_m128_h16384_xc_yc_pdl0_eps1e6` | 86.180 | 23.2% † | 8.971 | 9.606x | 10.4% |
| `flashinfer_layernorm` | `bf16_m1_h128_xc_yc_pdl0_eps1e6` | 3.132 | 9.7% | 2.249 | 1.392x | 71.8% |
| `flashinfer_qk_rmsnorm` | `gemma_bf16_b32_n32_h128_xc_yc_pdl0` | 6.133 | 5.4% | 2.770 | 2.214x | 45.2% |
| `flashinfer_qk_rmsnorm` | `rms_bf16_b32_n32_h128_xc_yc_pdl0` | 6.340 | 4.6% | 2.632 | 2.408x | 41.5% |
| `flashinfer_qk_rmsnorm` | `rms_f16_b16_n64_h128_xc_yc_pdl0` | 6.578 | 9.7% | 2.585 | 2.545x | 39.3% |
| `flashinfer_rmsnorm` | `gemma_bf16_m64_h8192_xc_yc_pdl0` | 20.623 | 5.3% | 3.472 | 5.940x | 16.8% |
| `flashinfer_rmsnorm` | `rms_bf16_m32_h4096_xc_yc_pdl0` | 7.228 | 10.3% † | 3.298 | 2.191x | 45.6% |
| `flashinfer_rmsnorm` | `rms_bf16_m32_h4096_xc_yc_pdl1` | 10.083 | 3.8% | 3.329 | 3.029x | 33.0% |
| `flashinfer_rmsnorm_fp4quant` | `bench_nv_3d_bf16_b32_s32_h128_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | 5.728 | 12.0% † | 2.776 | 2.063x | 48.5% |
| `flashinfer_rmsnorm_fp4quant` | `bench_nv_bf16_m32_h4096_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | 8.753 | 8.0% | 4.843 | 1.808x | 55.3% |
| `flashinfer_rmsnorm_fp4quant` | `bench_nv_large_bf16_m64_h8192_b16_e4m3_sw0_pdl0_eps1e6_gsnone_preallocated_random` | 20.847 | 6.0% | 4.943 | 4.218x | 23.7% |
| `flashinfer_rmsnorm_quant` | `bf16_e4m3_m32_h4096_xc_yc_pdl0_s1` | 6.883 | 11.9% † | 3.277 | 2.101x | 47.6% |
| `flashinfer_rmsnorm_quant` | `bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | 17.509 | 8.7% | 3.405 | 5.142x | 19.4% |
| `flashinfer_rmsnorm_quant` | `bf16_e5m2_m3_h1048576_xc_yc_pdl1_s1_cluster16_sync` | 308.610 | 23.3% † | 16.643 | 18.543x | 5.4% |
| `flashkda_bf16_fused_m128` | `h64_mixed` | 3564.335 | 1.7% | 267.837 | 13.308x | 7.5% |
| `flashkda_bf16_fused_m128` | `h96_fixed8192` | 5297.110 | 1.4% | 500.924 | 10.575x | 9.5% |
| `flashkda_bf16_fused_m128` | `h96_uniform` | 5500.077 | 1.0% | 434.092 | 12.670x | 7.9% |
| `flashkda_decode_t1_precomputed` | `hv16h16_b128_s8` | 686.724 | 8.1% | 26.620 | 25.797x | 3.9% |
| `flashkda_decode_t1_precomputed` | `hv16h16_b1_s16` | 10.403 | 9.7% | 4.116 | 2.528x | 39.6% |
| `flashkda_decode_t1_precomputed` | `hv32h16_b32_s8` | 373.556 | 6.0% | 15.239 | 24.513x | 4.1% |
| `flashkda_decode_t2_precomputed` | `hv12h12_b8_t2` | 75.565 | 6.5% | 6.376 | 11.851x | 8.4% |
| `flashkda_decode_t2_precomputed` | `hv16h16_b64_t2` | 570.319 | 10.4% † | 25.106 | 22.716x | 4.4% |
| `flashkda_decode_t2_precomputed` | `hv32h16_b128_t2` | 2208.976 | 9.1% | 81.600 | 27.071x | 3.7% |
| `flashkda_decode_t3_lower_bound` | `hv16h16_b16_t3` | 216.632 | 7.8% | 13.419 | 16.143x | 6.2% |
| `flashkda_decode_t3_lower_bound` | `hv16h16_b1_t3` | 20.300 | 9.5% | 5.556 | 3.653x | 27.4% |
| `flashkda_decode_t3_lower_bound` | `hv16h16_b4_t3` | 71.728 | 5.0% | 6.682 | 10.734x | 9.3% |
| `flashkda_decode_t4_precomputed` | `hv12h12_b8_t4` | 101.178 | 6.6% | 8.761 | 11.548x | 8.7% |
| `flashkda_decode_t4_precomputed` | `hv16h16_b64_t4` | 1066.704 | 6.2% | 39.333 | 27.120x | 3.7% |
| `flashkda_decode_t4_precomputed` | `hv32h16_b128_t4` | 3871.597 | 6.7% | 132.019 | 29.326x | 3.4% |
| `flashkda_decode_t5_gram` | `hv32h16_b128_s1` | 4934.124 | 7.5% | 158.978 | 31.037x | 3.2% |
| `flashkda_decode_t5_gram` | `hv32h16_b1_s8` | 49.261 | 4.5% | 7.030 | 7.008x | 14.3% |
| `flashkda_decode_t5_gram` | `hv32h16_b3_s4` | 137.059 | 2.6% | 8.599 | 15.938x | 6.3% |
| `flashkda_decode_t6_gram` | `hv32h16_b128_s1` | 5655.059 | 7.0% | 181.681 | 31.126x | 3.2% |
| `flashkda_decode_t6_gram` | `hv32h16_b1_s8` | 48.390 | 7.4% | 7.298 | 6.631x | 15.1% |
| `flashkda_decode_t6_gram` | `hv32h16_b3_s4` | 132.600 | 2.9% | 14.061 | 9.431x | 10.6% |
| `fp16_bf16_gemm` | `bf16_4096x4096x4096` | 1251.646 | 25.3% † | 92.495 | 13.532x | 7.4% |
| `fp16_bf16_gemm` | `fp16_1024x1024x1024` | 98.186 | 6.2% | 6.626 | 14.819x | 6.7% |
| `fp16_bf16_gemm` | `fp16_16384x16384x16384` | 312464.042 | 2.0% | 5702.039 | 54.799x | 1.8% |
| `gdn_cp_prefill_sm100` | `fp16_q16_k16_v16_s4096+4096_init_f16_i64` | 1583.193 | 9.1% | 117.785 | 13.441x | 7.4% |
| `gdn_cp_prefill_sm100` | `fp16_q16_k16_v64_s192+64_initfinal_f16_i64` | 674.512 | 21.0% † | 82.547 | 8.171x | 12.2% |
| `gdn_cp_prefill_sm100` | `fp16_q1_k1_v1_s2048_none_i32` | 93.903 | 22.4% † | 53.782 | 1.746x | 57.3% |
| `gdn_decode_bf16_ilp4` | `t1_b1_h2_hv4_tv16` | 5.380 | 13.3% † | 3.070 | 1.752x | 57.1% |
| `gdn_decode_bf16_ilp4` | `t4_b8_h4_hv8_tv16` | 70.328 | 9.4% | 6.663 | 10.555x | 9.5% |
| `gdn_decode_bf16_ilp4` | `t8_b4_h8_hv16_tv16` | 126.394 | 8.8% | 10.386 | 12.170x | 8.2% |
| `gdn_decode_bf16_wide_vec_mtp` | `t2_b4_h16_hv32_tv32` | 84.577 | 3.1% | 6.331 | 13.359x | 7.5% |
| `gdn_decode_bf16_wide_vec_mtp` | `t4_b64_h8_hv16_tv128` | 932.007 | 1.0% | 33.810 | 27.566x | 3.6% |
| `gdn_decode_bf16_wide_vec_mtp` | `t8_b512_h16_hv32_tv128` | 31230.094 | 3.2% | 1163.720 | 26.836x | 3.7% |
| `gdn_decode_bf16_wide_vec_t1` | `b128_h8_hv16_tv128` | 670.560 | 0.9% | 24.894 | 26.937x | 3.7% |
| `gdn_decode_bf16_wide_vec_t1` | `b16_h16_hv32_tv64` | 203.596 | 7.3% | 11.252 | 18.094x | 5.5% |
| `gdn_decode_bf16_wide_vec_t1` | `b512_h4_hv8_tv128` | 1318.688 | 0.6% | 46.151 | 28.573x | 3.5% |
| `gdn_decode_fp32_mtp_warp` | `t2_b4_h16_hv64_tv16_ilp2_sv0` | 386.637 | 1.2% | 14.341 | 26.960x | 3.7% |
| `gdn_decode_fp32_mtp_warp` | `t4_b64_h8_hv32_tv64_ilp4_sv1` | 5209.246 | 2.7% | `FAIL` | — | — |
| `gdn_decode_fp32_mtp_warp` | `t8_b256_h16_hv64_tv64_ilp4_sv1` | 66563.923 | 1.6% | 1773.392 | 37.535x | 2.7% |
| `gdn_prefill_sm100` | `hq16_hv64_s1x8192` | 2078.314 | 3.2% | 240.625 | 8.637x | 11.6% |
| `gdn_prefill_sm100` | `hq32_hv32_s8192x16` | 20421.442 | 0.8% | 1084.174 | 18.836x | 5.3% |
| `gdn_prefill_sm100` | `hq8_hv32_s1024x8` | 1247.581 | 2.4% | 159.096 | 7.842x | 12.8% |
| `msa_sparse_atten_fwd_combine_sm100` | `fp8p_s16384_qh8_t8` | 1553.803 | 3.4% | — | — | — |
| `msa_sparse_atten_fwd_combine_sm100` | `ring48k_fp32p_qh16_t16` | 29815.575 | 9.4% | — | — | — |
| `msa_sparse_atten_fwd_combine_sm100` | `scale_bf16p_temp_s16384_qh4_t16` | 2720.310 | 11.0% † | — | — | — |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100` | `ring48k_bf16q_qh16_t16` | 30137.148 | 7.5% | 1029.398 | 29.276x | 3.4% |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100` | `ring48k_fp8q_qh16_t16` | 24945.276 | 6.4% | 1384.614 | 18.016x | 5.6% |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100` | `varlen_b3_s8192_bf16q_qh4_t16` | 7868.873 | 7.6% | 255.158 | 30.839x | 3.2% |
| `msa_sparse_atten_fwd_sm100` | `ring48k_bf16_qh16_t16` | 52842.633 | 6.6% | 1361.073 | 38.824x | 2.6% |
| `msa_sparse_atten_fwd_sm100` | `ring48k_fp8_qh16_t16` | 24864.729 | 7.6% | 1334.788 | 18.628x | 5.4% |
| `msa_sparse_atten_fwd_sm100` | `varlen_b3_s8192_qh4_t16` | 7843.580 | 8.2% | 266.394 | 29.444x | 3.4% |
| `msa_sparse_prepare_flat_schedule_sm100` | `decode_b128_k65536_h4` | 93665.642 | 0.0% | 8694.062 | 10.774x | 9.3% |
| `msa_sparse_prepare_flat_schedule_sm100` | `decode_b64_k16384_h4_varlen` | 3127.682 | 0.1% | 318.735 | 9.813x | 10.2% |
| `msa_sparse_prepare_flat_schedule_sm100` | `prefill_b1_k8192_h2` | 5.141 | 31.3% † | 3.886 | 1.323x | 75.6% |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | `decode_b128_k65536_h4` | 2095.600 | 1.1% | 216.893 | 9.662x | 10.3% |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | `prefill_b1_k131072_h1` | 327.069 | 2.4% | 33.163 | 9.862x | 10.1% |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | `prefill_b1_k8192_h2` | 28.670 | 2.1% | 7.192 | 3.986x | 25.1% |
| `mxfp4_quantize` | `fp16_128x4_m128_k1024` | 7.829 | 12.5% † | 2.491 | 3.143x | 31.8% |
| `mxfp4_quantize` | `fp16_128x4_m16384_k7168` | 1996.881 | 1.5% | 52.500 | 38.036x | 2.6% |
| `mxfp4_quantize` | `fp16_linear_m4096_k4096` | 307.695 | 8.2% | 10.282 | 29.925x | 3.3% |
| `mxfp8_quantize` | `fp16_128x4_m128_k1024` | 6.233 | 14.2% † | 2.679 | 2.327x | 43.0% |
| `mxfp8_quantize` | `fp16_128x4_m16384_k7168` | 1855.464 | 0.5% | 60.634 | 30.601x | 3.3% |
| `mxfp8_quantize` | `fp16_linear_m4096_k4096` | 304.881 | 1.7% | 11.316 | 26.943x | 3.7% |
| `nvfp4_gemm` | `1024x1024x1024` | 51.733 | 22.9% † | 5.211 | 9.928x | 10.1% |
| `nvfp4_gemm` | `16384x16384x16384` | 36218.299 | 38.3% † | 1528.963 | 23.688x | 4.2% |
| `nvfp4_gemm` | `4096x4096x4096` | 534.702 | 26.1% † | 29.305 | 18.246x | 5.5% |
| `nvfp4_quantize` | `fp16_128x4_m128_k1024` | 5.619 | 9.9% | 2.497 | 2.250x | 44.4% |
| `nvfp4_quantize` | `fp16_128x4_m16384_k7168` | 1570.230 | 1.0% | 55.273 | 28.409x | 3.5% |
| `nvfp4_quantize` | `fp16_linear_m4096_k4096` | 280.266 | 1.1% | 10.253 | 27.336x | 3.7% |
| `nvfp4_quantize_per_token` | `fp16_128x4_m128_k1024` | 6.034 | 10.8% † | 2.689 | 2.244x | 44.6% |
| `nvfp4_quantize_per_token` | `fp16_128x4_m16384_k7168` | 1393.113 | 5.1% | 57.766 | 24.117x | 4.1% |
| `nvfp4_quantize_per_token` | `fp16_linear_m4096_k4096` | 236.095 | 8.9% | 12.110 | 19.495x | 5.1% |
| `radix_topk_multi_cta` | `f32_basic_r2_l524288_k256_large` | 100.682 | 11.0% † | 40.014 | 2.516x | 39.7% |
| `radix_topk_multi_cta` | `f32_basic_r4_l115188_k256_ctas3` | 67.377 | 21.5% † | 35.370 | 1.905x | 52.5% |
| `radix_topk_multi_cta` | `f32_basic_r4_l57596_k256_vec4` | 45.795 | 26.1% † | 30.173 | 1.518x | 65.9% |
| `radix_topk_single_cta` | `f32_basic_r256_l57592_k1024_maxchunk` | 781.768 | 0.5% | 80.214 | 9.746x | 10.3% |
| `radix_topk_single_cta` | `f32_basic_r64_l32768_k512` | 180.771 | 22.8% † | 23.182 | 7.798x | 12.8% |
| `radix_topk_single_cta` | `f32_basic_r8_l8192_k256` | 19.041 | 25.5% † | 11.143 | 1.709x | 58.5% |
| `recurrent_kda_decode_grouped` | `dec_hv16_b1` | 14.148 | 6.1% | 3.309 | 4.276x | 23.4% |
| `recurrent_kda_decode_grouped` | `ver_t8_hv12_b16` | 418.157 | 8.5% | 22.114 | 18.909x | 5.3% |
| `recurrent_kda_decode_grouped` | `ver_t8_hv16_b128` | 3873.613 | 4.3% | 161.221 | 24.027x | 4.2% |
| `recurrent_kda_decode_one_warp` | `hv12_b64_tr16_lb` | 274.451 | 2.6% | 12.643 | 21.708x | 4.6% |
| `recurrent_kda_decode_one_warp` | `hv16_b128_tr16_lb` | 643.138 | 3.3% | 26.661 | 24.123x | 4.1% |
| `recurrent_kda_decode_one_warp` | `hv16_b8_tr8_lb` | 59.452 | 0.8% | 5.247 | 11.330x | 8.8% |
| `rmsnorm` | `hs128_bs32` | 3.367 | 9.2% | 2.313 | 1.456x | 68.7% |
| `rmsnorm` | `hs4096_bs128` | 21.575 | 5.8% | 3.520 | 6.129x | 16.3% |
| `rmsnorm` | `hs8192_bs4113` | 758.798 | 12.3% † | 71.758 | 10.574x | 9.5% |
| `selective_state_update_mtp_horizontal` | `b1_h64_d64_s128_t6_r8_statebf16_official` | 24.095 | 25.1% † | 6.950 | 3.467x | 28.8% |
| `selective_state_update_mtp_horizontal` | `b2048_h64_d64_s128_t6_r8_statebf16_official` | 14066.984 | 0.0% | 1658.426 | 8.482x | 11.8% |
| `selective_state_update_mtp_horizontal` | `b512_h64_d64_s128_t6_r8_statebf16_official` | 3526.607 | 0.0% | 347.038 | 10.162x | 9.8% |
| `selective_state_update_mtp_simple` | `b1_h64_d64_s128_t6_r8_statebf16_official` | 20.476 | 21.1% † | 4.316 | 4.744x | 21.1% |
| `selective_state_update_mtp_simple` | `b2048_h64_d64_s128_t6_r8_statebf16_official` | 15068.646 | 0.1% | 1494.975 | 10.080x | 9.9% |
| `selective_state_update_mtp_simple` | `b512_h64_d64_s128_t6_r8_statebf16_official` | 3775.694 | 0.0% | 378.014 | 9.988x | 10.0% |
| `selective_state_update_mtp_vertical` | `b1_h64_d64_s128_t6_r8_statebf16_official` | 43.536 | 23.4% † | 17.265 | 2.522x | 39.7% |
| `selective_state_update_mtp_vertical` | `b2048_h64_d64_s128_t6_r8_statebf16_official` | 25332.209 | 0.3% | 2689.228 | 9.420x | 10.6% |
| `selective_state_update_mtp_vertical` | `b512_h64_d64_s128_t6_r8_statebf16_official` | 6207.857 | 0.1% | `FAIL` | — | — |
| `selective_state_update_stp_horizontal` | `b64_h64_d128_s128_r8` | 2219.514 | 0.7% | 47.807 | 46.427x | 2.2% |
| `selective_state_update_stp_horizontal` | `b64_h64_d64_s128_r8_base` | 1062.000 | 1.6% | 26.500 | 40.075x | 2.5% |
| `selective_state_update_stp_horizontal` | `b64_h64_d64_s256_r8` | 2791.364 | 0.3% | 47.449 | 58.829x | 1.7% |
| `selective_state_update_stp_simple` | `b64_h64_d128_s128_r8` | 1445.675 | 4.2% | 65.279 | 22.146x | 4.5% |
| `selective_state_update_stp_simple` | `b64_h64_d64_s128_r8_base` | 739.812 | 8.0% | 37.337 | 19.814x | 5.0% |
| `selective_state_update_stp_simple` | `b64_h64_d64_s256_r8` | 1449.751 | 7.2% | 54.066 | 26.814x | 3.7% |
| `selective_state_update_stp_vertical` | `b64_h64_d128_s128_r8` | 1434.668 | 7.5% | 47.353 | 30.297x | 3.3% |
| `selective_state_update_stp_vertical` | `b64_h64_d64_s128_r8_base` | 767.195 | 6.3% | 27.079 | 28.332x | 3.5% |
| `selective_state_update_stp_vertical` | `b64_h64_d64_s256_r8` | 1400.766 | 4.1% | 46.242 | 30.292x | 3.3% |
| `silu_and_mul_nvfp4_experts_quantize` | `bf16_b8_m512_k2048` | 171.847 | 10.5% † | 9.048 | 18.992x | 5.3% |
| `silu_and_mul_nvfp4_experts_quantize` | `fp16_b128_m2048_k2048` | 5650.664 | 0.8% | 273.842 | 20.635x | 4.8% |
| `silu_and_mul_nvfp4_experts_quantize` | `fp16_b8_m16_k2048` | 11.926 | 14.3% † | 3.341 | 3.570x | 28.0% |
| `sparse_flashmla_decode_head64` | `deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | 2152.344 | 0.9% | 136.739 | 15.740x | 6.4% |
| `sparse_flashmla_decode_head64` | `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | 56.582 | 14.5% † | 16.706 | 3.387x | 29.5% |
| `sparse_flashmla_decode_head64` | `v32_b148_sq2_sk32768_topk16384_p64` | 16176.357 | 0.9% | 1526.241 | 10.599x | 9.4% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | 26742.656 | 0.8% | 1813.111 | 14.750x | 6.8% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | 17897.781 | 1.0% | 1676.518 | 10.676x | 9.4% |
| `sparse_flashmla_prefill_head128_phase1` | `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | 23531.298 | 19.4% † | 1775.025 | 13.257x | 7.5% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | 13078.023 | 13.8% † | 1126.813 | 11.606x | 8.6% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | 21543.110 | 5.7% | 1196.729 | 18.002x | 5.6% |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | 11500.324 | 2.7% | 1191.661 | 9.651x | 10.4% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk512_hq64_s4096_kv65536_topk512` | 8726.118 | 0.4% | 381.845 | 22.853x | 4.4% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk512_hq64_s4096_kv8192_topk512` | 4103.796 | 6.3% | 366.517 | 11.197x | 8.9% |
| `sparse_flashmla_prefill_head64_phase1` | `bench_dqk576_hq64_s4096_kv32768_topk512` | 7454.335 | 1.6% | 861.845 | 8.649x | 11.6% |
| `stable_sort_topk_by_value` | `f32_r4_k128` | 7.588 | 38.8% † | `FAIL` | — | — |
| `stable_sort_topk_by_value` | `f32_r64_k2048` | 61.134 | 45.8% † | `FAIL` | — | — |
| `stable_sort_topk_by_value` | `f32_r64_k256` | 12.423 | 42.5% † | `FAIL` | — | — |
| `tinygemm2_sm100` | `b16_o2880_k2880` | 262.040 | 0.4% | 7.936 | 33.019x | 3.0% |
| `tinygemm2_sm100` | `b1_o128_k720` | 6.095 | 12.0% † | 2.956 | 2.062x | 48.5% |
| `tinygemm2_sm100` | `b64_o4096_k3072` | 518.157 | 6.6% | 21.917 | 23.642x | 4.2% |

## Raw evidence

- Thor run: `/home/tlopexh/thor-validation/bench-proton-final-clean/runs/1.json`
- B200 baseline: `/home/tlopexh/TIRx-kernels/tirx_kernels/bench_suite/baseline.json`
- Prior Thor stability run: `/home/tlopexh/thor-validation/bench-proton-23/runs/1.json`
- Thor run status: 254 `ok`, 0 failures, 0 interference retries
- Usable B200 matches: 183 rows; failed B200 baseline rows: 5; workload rows absent from the B200 baseline: 66
