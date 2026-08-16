# bench-suite baseline view: `baseline.json`

- Timestamp: `12`
- Label:     `restore-proton-timer-3`
- Runner:    `{'hostname': 'catalyst-fleet1.cs.cmu.edu', 'gpu': {'name': 'NVIDIA B200', 'compute_capability': [10, 0], 'cuda_arch': 'sm_100a', 'num_sms': 148}, 'gpu_inventory': [{'index': '0', 'uuid': 'GPU-ef5a8300-fa8a-64f7-b6a5-ca2d0dbb6128', 'pci.bus_id': '00000000:1B:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}, {'index': '1', 'uuid': 'GPU-e8754e6d-624e-e1d0-595a-f9444588960a', 'pci.bus_id': '00000000:43:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}, {'index': '2', 'uuid': 'GPU-f8a4f1df-8b46-4cbf-3244-a33b90e06aa9', 'pci.bus_id': '00000000:52:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}, {'index': '3', 'uuid': 'GPU-51c31609-c1ae-1fa6-14b9-da2a172ffd67', 'pci.bus_id': '00000000:61:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}, {'index': '4', 'uuid': 'GPU-feda1f0f-e1ab-30f7-dd6d-65c8ebe11acc', 'pci.bus_id': '00000000:9D:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}, {'index': '5', 'uuid': 'GPU-087b0f09-0d22-7908-e4cb-bb49fd81a455', 'pci.bus_id': '00000000:C3:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}, {'index': '6', 'uuid': 'GPU-e56ad157-72b3-2e86-4cd9-5769dc1f229c', 'pci.bus_id': '00000000:D1:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}, {'index': '7', 'uuid': 'GPU-aff76307-71d5-37df-fd90-83fe611b5e73', 'pci.bus_id': '00000000:DF:00.0', 'name': 'NVIDIA B200', 'clocks.max.sm': '1965', 'clocks.max.memory': '3996', 'power.limit': '1000.00', 'persistence_mode': 'Enabled'}], 'gpu_topology': ['\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tNIC0\tNIC1\tNIC2\tNIC3\tNIC4\tNIC5\tNIC6\tNIC7\tNIC8\tNIC9\tNIC10\tNIC11\tNIC12\tNIC13\tNIC14\tNIC15\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m', 'GPU0\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNODE\tNODE\tNODE\tNODE\tPXB\tNODE\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t0-55,112-167\t0\t\tN/A', 'GPU1\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tPXB\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t0-55,112-167\t0\t\tN/A', 'GPU2\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tPXB\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t0-55,112-167\t0\t\tN/A', 'GPU3\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tPXB\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t0-55,112-167\t0\t\tN/A', 'GPU4\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tPXB\tNODE\tNODE\tNODE\tNODE\tNODE\t56-111,168-223\t1\t\tN/A', 'GPU5\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tPXB\tNODE\tNODE\t56-111,168-223\t1\t\tN/A', 'GPU6\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tPXB\tNODE\t56-111,168-223\t1\t\tN/A', 'GPU7\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tNODE\tPXB\t56-111,168-223\t1\t\tN/A', 'NIC0\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\t X \tPIX\tPIX\tPIX\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC1\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tPIX\t X \tPIX\tPIX\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC2\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tPIX\tPIX\t X \tPIX\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC3\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tPIX\tPIX\tPIX\t X \tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC4\tPXB\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\t X \tNODE\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC5\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tPIX\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC6\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tNODE\tPIX\t X \tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC7\tNODE\tPXB\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC8\tNODE\tNODE\tPXB\tNODE\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC9\tNODE\tNODE\tNODE\tPXB\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t\t\t\t', 'NIC10\tSYS\tSYS\tSYS\tSYS\tPXB\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t X \tNODE\tNODE\tNODE\tNODE\tNODE\t\t\t\t', 'NIC11\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\t X \tPIX\tNODE\tNODE\tNODE\t\t\t\t', 'NIC12\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\tPIX\t X \tNODE\tNODE\tNODE\t\t\t\t', 'NIC13\tSYS\tSYS\tSYS\tSYS\tNODE\tPXB\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\t X \tNODE\tNODE\t\t\t\t', 'NIC14\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tPXB\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\t X \tNODE\t\t\t\t', 'NIC15\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tPXB\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tNODE\tNODE\tNODE\t X \t\t\t\t', '', 'Legend:', '', '  X    = Self', '  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)', '  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node', '  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)', '  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)', '  PIX  = Connection traversing at most a single PCIe bridge', '  NV#  = Connection traversing a bonded set of # NVLinks', '', 'NIC Legend:', '', '  NIC0: mlx5_0', '  NIC1: mlx5_1', '  NIC2: mlx5_2', '  NIC3: mlx5_3', '  NIC4: mlx5_4', '  NIC5: mlx5_5', '  NIC6: mlx5_6', '  NIC7: mlx5_7', '  NIC8: mlx5_8', '  NIC9: mlx5_9', '  NIC10: mlx5_10', '  NIC11: mlx5_11', '  NIC12: mlx5_12', '  NIC13: mlx5_13', '  NIC14: mlx5_14', '  NIC15: mlx5_15'], 'driver_version': '595.58.03'}`
- Git:       `{'tir': '27c2e019-dirty', 'tirx-kernels': 'bcd09d07-dirty'}`
- Workloads: 137 ok, 0 failed

One row per config with the pinned TIRx absolute GPU time.

## act_and_mul

| config | timer | tirx (µs) |
|---|---|---:|
| `gelu_tanh_fp16_d11008_t8192` | `proton` | 97.6652 |
| `silu_bf16_d16384_t32768` | `proton` | 454.4366 |
| `silu_fp16_d4096_t1` | `proton` | 2.5013 |

## deepgemm_sm100_fp4_mqa_logits

| config | timer | tirx (µs) |
|---|---|---:|
| `s2048_skv4096_h64_d128_f32_dense_cp` | `proton` | 37.8445 |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | `proton` | 186.6705 |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | timer | tirx (µs) |
|---|---|---:|
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | `proton` | 4.0888 |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | `proton` | 6.1129 |

## deepgemm_sm100_fp8_bmm

| config | timer | tirx (µs) |
|---|---|---:|
| `bhd_bhr_hdr_b4096_h8_r4096_d1024` | `proton` | 133.7840 |
| `bhd_hdr_bhr_b8192_h8_r4096_d1024` | `proton` | 224.3210 |
| `bhr_hdr_bhd_b4096_h8_r4096_d1024` | `proton` | 97.7707 |

## deepgemm_sm100_fp8_gemm_1d1d

| config | timer | tirx (µs) |
|---|---|---:|
| `m4096_n576_k7168` | `proton` | 18.8133 |
| `m4096_n4096_k7168_bfp4` | `proton` | 77.4168 |
| `m4096_n7168_k16384` | `proton` | 332.6277 |

## deepgemm_sm100_fp8_mqa_logits

| config | timer | tirx (µs) |
|---|---|---:|
| `s2048_skv4096_h64_d128_f32_dense_cp` | `proton` | 39.9464 |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | `proton` | 193.0693 |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | timer | tirx (µs) |
|---|---|---:|
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | `proton` | 4.5831 |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | `proton` | 6.7690 |

## deepgemm_sm100_k_grouped_fp8_gemm_contiguous

| config | timer | tirx (µs) |
|---|---|---:|
| `g4_m4096_n7168_k8192_gran128_al128_psum` | `proton` | 852.6776 |
| `g8_m4096_n7168_k4096_gran32_al160` | `proton` | 927.9266 |
| `g16_m7168_n2048_k2048_gran128_al128` | `proton` | 578.0204 |

## deepgemm_sm100_m_grouped_fp8_gemm_contiguous

| config | timer | tirx (µs) |
|---|---|---:|
| `g4_m8192_n6144_k7168` | `proton` | 1034.9156 |
| `g8_m4096_n4096_k2048_bfp4` | `proton` | 187.4108 |
| `g8_m4096_n7168_k3072_psum_zp` | `proton` | 518.9476 |

## deepgemm_sm100_m_grouped_fp8_gemm_masked

| config | timer | tirx (µs) |
|---|---|---:|
| `g6_m1024_n4096_k2048` | `proton` | 39.6386 |
| `g32_m192_n4096_k4096_bfp4` | `proton` | 113.5337 |
| `g32_m192_n6144_k7168` | `proton` | 342.1666 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | timer | tirx (µs) |
|---|---|---:|
| `m128_n24_k16384_s64` | `proton` | 5.3209 |
| `m4096_n24_k16384_s2` | `proton` | 28.7617 |
| `m8192_n24_k28672_s1` | `proton` | 83.2368 |

## flash_attention4

| config | timer | tirx (µs) |
|---|---|---:|
| `s1024_h32kv4` | `proton` | 19.5685 |
| `s4096_h32kv4_causal` | `proton` | 110.3583 |
| `s8192_h32kv32` | `proton` | 770.3913 |

## flash_attention_backward_sm100

| config | timer | tirx (µs) |
|---|---|---:|
| `b1_s2048_h16_causal` | `proton` | 81.5927 |
| `b1_s8192_h16_noncausal` | `proton` | 1095.4509 |
| `b4_s8192_h16_noncausal` | `proton` | 4406.9603 |

## flashkda_bf16_fused_m128

| config | timer | tirx (µs) |
|---|---|---:|
| `h64_mixed` | `proton` | 265.4007 |
| `h96_fixed8192` | `proton` | 500.4741 |
| `h96_uniform` | `proton` | 433.3607 |

## flashkda_decode_t1_precomputed

| config | timer | tirx (µs) |
|---|---|---:|
| `hv16h16_b1_s16` | `proton` | 4.0068 |
| `hv16h16_b128_s8` | `proton` | 28.7018 |
| `hv32h16_b32_s8` | `proton` | 16.6525 |

## flashkda_decode_t2_precomputed

| config | timer | tirx (µs) |
|---|---|---:|
| `hv12h12_b8_t2` | `proton` | 6.3919 |
| `hv16h16_b64_t2` | `proton` | 25.3409 |
| `hv32h16_b128_t2` | `proton` | 83.2439 |

## flashkda_decode_t3_lower_bound

| config | timer | tirx (µs) |
|---|---|---:|
| `hv16h16_b1_t3` | `proton` | 5.8324 |
| `hv16h16_b4_t3` | `proton` | 6.6956 |
| `hv16h16_b16_t3` | `proton` | 13.4767 |

## flashkda_decode_t4_precomputed

| config | timer | tirx (µs) |
|---|---|---:|
| `hv12h12_b8_t4` | `proton` | 8.7843 |
| `hv16h16_b64_t4` | `proton` | 41.1599 |
| `hv32h16_b128_t4` | `proton` | 135.6346 |

## flashkda_decode_t5_gram

| config | timer | tirx (µs) |
|---|---|---:|
| `hv32h16_b1_s8` | `proton` | 7.5336 |
| `hv32h16_b3_s4` | `proton` | 8.8665 |
| `hv32h16_b128_s1` | `proton` | 158.0077 |

## flashkda_decode_t6_gram

| config | timer | tirx (µs) |
|---|---|---:|
| `hv32h16_b1_s8` | `proton` | 7.3664 |
| `hv32h16_b3_s4` | `proton` | 14.5141 |
| `hv32h16_b128_s1` | `proton` | 177.0170 |

## fp16_bf16_gemm

| config | timer | tirx (µs) |
|---|---|---:|
| `bf16_4096x4096x4096` | `proton` | 92.3262 |
| `fp16_1024x1024x1024` | `proton` | 6.7428 |
| `fp16_16384x16384x16384` | `proton` | 5811.4929 |

## gdn_cp_prefill_sm100

| config | timer | tirx (µs) |
|---|---|---:|
| `fp16_q1_k1_v1_s2048_none_i32` | `proton` | 55.1199 |
| `fp16_q16_k16_v16_s4096+4096_init_f16_i64` | `proton` | 119.3954 |
| `fp16_q16_k16_v64_s192+64_initfinal_f16_i64` | `proton` | 82.8498 |

## gdn_decode_bf16_ilp4

| config | timer | tirx (µs) |
|---|---|---:|
| `t1_b1_h2_hv4_tv16` | `proton` | 2.9442 |
| `t4_b8_h4_hv8_tv16` | `proton` | 6.5627 |
| `t8_b4_h8_hv16_tv16` | `proton` | 10.4689 |

## gdn_decode_bf16_wide_vec_mtp

| config | timer | tirx (µs) |
|---|---|---:|
| `t2_b4_h16_hv32_tv32` | `proton` | 6.3269 |
| `t4_b64_h8_hv16_tv128` | `proton` | 34.0307 |
| `t8_b512_h16_hv32_tv128` | `proton` | 858.6385 |

## gdn_decode_bf16_wide_vec_t1

| config | timer | tirx (µs) |
|---|---|---:|
| `b16_h16_hv32_tv64` | `proton` | 8.8088 |
| `b128_h8_hv16_tv128` | `proton` | 24.8050 |
| `b512_h4_hv8_tv128` | `proton` | 46.1153 |

## gdn_decode_fp32_mtp_warp

| config | timer | tirx (µs) |
|---|---|---:|
| `t2_b4_h16_hv64_tv16_ilp2_sv0` | `proton` | 14.6836 |
| `t4_b64_h8_hv32_tv64_ilp4_sv1` | `proton` | 142.5750 |
| `t8_b256_h16_hv64_tv64_ilp4_sv1` | `proton` | 1775.8250 |

## gdn_prefill_sm100

| config | timer | tirx (µs) |
|---|---|---:|
| `hq8_hv32_s1024x8` | `proton` | 90.8626 |
| `hq16_hv64_s1x8192` | `proton` | 240.2070 |
| `hq32_hv32_s8192x16` | `proton` | 1087.3939 |

## mxfp4_quantize

| config | timer | tirx (µs) |
|---|---|---:|
| `fp16_128x4_m128_k1024` | `proton` | 2.9521 |
| `fp16_128x4_m16384_k7168` | `proton` | 52.4637 |
| `fp16_linear_m4096_k4096` | `proton` | 10.3753 |

## mxfp8_quantize

| config | timer | tirx (µs) |
|---|---|---:|
| `fp16_128x4_m128_k1024` | `proton` | 2.5602 |
| `fp16_128x4_m16384_k7168` | `proton` | 60.7462 |
| `fp16_linear_m4096_k4096` | `proton` | 11.3961 |

## nvfp4_gemm

| config | timer | tirx (µs) |
|---|---|---:|
| `1024x1024x1024` | `proton` | 5.4477 |
| `4096x4096x4096` | `proton` | 29.4994 |
| `16384x16384x16384` | `proton` | 1465.8600 |

## nvfp4_quantize

| config | timer | tirx (µs) |
|---|---|---:|
| `fp16_128x4_m128_k1024` | `proton` | 2.3898 |
| `fp16_128x4_m16384_k7168` | `proton` | 54.9947 |
| `fp16_linear_m4096_k4096` | `proton` | 10.2663 |

## nvfp4_quantize_per_token

| config | timer | tirx (µs) |
|---|---|---:|
| `fp16_128x4_m128_k1024` | `proton` | 2.7746 |
| `fp16_128x4_m16384_k7168` | `proton` | 57.6668 |
| `fp16_linear_m4096_k4096` | `proton` | 12.0264 |

## recurrent_kda_decode_grouped

| config | timer | tirx (µs) |
|---|---|---:|
| `dec_hv16_b1` | `proton` | 3.3447 |
| `ver_t8_hv12_b16` | `proton` | 22.0083 |
| `ver_t8_hv16_b128` | `proton` | 161.7264 |

## recurrent_kda_decode_one_warp

| config | timer | tirx (µs) |
|---|---|---:|
| `hv12_b64_tr16_lb` | `proton` | 12.8025 |
| `hv16_b8_tr8_lb` | `proton` | 5.2857 |
| `hv16_b128_tr16_lb` | `proton` | 26.6729 |

## selective_state_update_mtp_horizontal

| config | timer | tirx (µs) |
|---|---|---:|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 6.9388 |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 352.3265 |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 1388.8939 |

## selective_state_update_mtp_simple

| config | timer | tirx (µs) |
|---|---|---:|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 4.3882 |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 385.2828 |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 1519.0504 |

## selective_state_update_mtp_vertical

| config | timer | tirx (µs) |
|---|---|---:|
| `b1_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 17.3860 |
| `b512_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 713.3219 |
| `b2048_h64_d64_s128_t6_r8_statebf16_official` | `proton` | 2824.6168 |

## selective_state_update_stp_horizontal

| config | timer | tirx (µs) |
|---|---|---:|
| `b64_h64_d64_s128_r8_base` | `proton` | 28.1392 |
| `b64_h64_d64_s256_r8` | `proton` | 46.8917 |
| `b64_h64_d128_s128_r8` | `proton` | 48.6830 |

## selective_state_update_stp_simple

| config | timer | tirx (µs) |
|---|---|---:|
| `b64_h64_d64_s128_r8_base` | `proton` | 37.3738 |
| `b64_h64_d64_s256_r8` | `proton` | 54.1183 |
| `b64_h64_d128_s128_r8` | `proton` | 65.8215 |

## selective_state_update_stp_vertical

| config | timer | tirx (µs) |
|---|---|---:|
| `b64_h64_d64_s128_r8_base` | `proton` | 27.6699 |
| `b64_h64_d64_s256_r8` | `proton` | 46.1006 |
| `b64_h64_d128_s128_r8` | `proton` | 47.3161 |

## silu_and_mul_nvfp4_experts_quantize

| config | timer | tirx (µs) |
|---|---|---:|
| `bf16_b8_m512_k2048` | `proton` | 8.8716 |
| `fp16_b8_m16_k2048` | `proton` | 3.3804 |
| `fp16_b128_m2048_k2048` | `proton` | 275.2879 |

## sm100_fp8_fp4_mega_moe

| config | timer | tirx (µs) |
|---|---|---:|
| `t64_m64_h7168_i3072_e384_k6_g1` | `kineto` | 1298.9994 |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | `kineto` | 3375.8997 |
| `t8192_m8192_h7168_i3072_e384_k6_g1_s1` | `kineto` | 3765.0909 |

## sparse_flashmla_decode_head64

| config | timer | tirx (µs) |
|---|---|---:|
| `deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | `proton` | 137.9485 |
| `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | `proton` | 16.7443 |
| `v32_b148_sq2_sk32768_topk16384_p64` | `proton` | 910.8162 |

## sparse_flashmla_prefill_head128_phase1

| config | timer | tirx (µs) |
|---|---|---:|
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | `proton` | 1704.5195 |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | `proton` | 1878.0577 |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | `proton` | 1828.7784 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | timer | tirx (µs) |
|---|---|---:|
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | `proton` | 1143.6162 |
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | `proton` | 1145.9278 |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | `proton` | 1186.1721 |

## sparse_flashmla_prefill_head64_phase1

| config | timer | tirx (µs) |
|---|---|---:|
| `bench_dqk512_hq64_s4096_kv8192_topk512` | `proton` | 367.1772 |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | `proton` | 378.8519 |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | `proton` | 388.9157 |

## tinygemm2_sm100

| config | timer | tirx (µs) |
|---|---|---:|
| `b1_o128_k720` | `proton` | 2.9702 |
| `b16_o2880_k2880` | `proton` | 7.9564 |
| `b64_o4096_k3072` | `proton` | 22.2163 |
