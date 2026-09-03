# NVIDIA Thor kernel comparison summary

## Summary

| Metric | Result |
|---|---:|
| Comparisons completed | **20/20** |
| TIRx faster by more than 5% | **5/20** |
| Difference within 5% | **10/20** |
| Baseline faster by more than 5% | **5/20** |
| Geometric-mean TIRx speedup | **0.999x (approximately equal overall)** |

Speedup is `baseline latency / TIRx latency`; values above 1.0 favor TIRx.

## Kernel comparison results

| Kernel | Contract-matched baseline | TIRx (us) | Baseline (us) | TIRx speedup | Result |
|---|---|---:|---:|---:|---|
| BF16 GEMM | cuBLAS | 1346.893 | 1358.055 | 1.008x | Within 5% |
| FP16 GEMM | cuBLAS | 998.801 | 1282.266 | **1.284x** | TIRx faster by 28.4% |
| NVFP4 GEMM | FlashInfer CUTLASS FP4 | 374.545 | 347.978 | 0.929x | Baseline faster by 7.1% |
| FlashAttention-4 | Upstream FA4 CuTeDSL | 985.535 | 964.548 | 0.979x | Within 5% |
| Fused Add RMSNorm | FlashInfer CuTeDSL | 16.030 | 13.827 | 0.863x | Baseline faster by 13.7% |
| Fused DiT LayerNorm | FlashInfer CUDA | 334.494 | 325.428 | 0.973x | Within 5% |
| LayerNorm | FlashInfer CuTeDSL | 79.794 | 78.551 | 0.984x | Within 5% |
| QK RMSNorm | FlashInfer CuTeDSL | 7.001 | 6.856 | 0.979x | Within 5% |
| RMSNorm | FlashInfer CuTeDSL | 11.330 | 9.885 | 0.872x | Baseline faster by 12.8% |
| RMSNorm Quant | FlashInfer CuTeDSL | 22.736 | 20.016 | 0.880x | Baseline faster by 12.0% |
| GELU-and-Mul | FlashInfer CUDA | 2389.711 | 2386.804 | 0.999x | Within 5% |
| MXFP4 Quantize | FlashInfer CuTeDSL | 280.820 | 323.658 | **1.153x** | TIRx faster by 15.3% |
| NVFP4 Quantize | FlashInfer CuTeDSL | 254.796 | 254.155 | 0.997x | Within 5% |
| Fast TopK Clusters | FlashInfer | 198.190 | 197.375 | 0.996x | Within 5% |
| Filtered TopK | FlashInfer | 41.798 | 47.615 | **1.139x** | TIRx faster by 13.9% |
| Radix TopK Multi-CTA | FlashInfer | 66.400 | 67.430 | 1.016x | Within 5% |
| Radix TopK Single-CTA | FlashInfer | 174.729 | 187.086 | **1.071x** | TIRx faster by 7.1% |
| GDN Decode BF16 ILP4 | FlashInfer CuTeDSL | 83.903 | 82.843 | 0.987x | Within 5% |
| Recurrent KDA Grouped | FlashInfer CuTeDSL | 431.529 | 383.748 | 0.889x | Baseline faster by 11.1% |
| Mamba SSU Horizontal | FlashInfer CUDA | 3526.523 | 3827.694 | **1.085x** | TIRx faster by 8.5% |
