# NVIDIA Thor native-baseline performance

Measured on 2026-09-03 on one NVIDIA Jetson AGX Thor Developer Kit. Every row compares TIRx with a FlashInfer implementation on the same GPU and exact input shape.

## At a glance

| Question | Result |
|---|---|
| Correctness and execution | **13/13 passed**; 0 failures and 0 interference retries |
| TIRx faster by more than 5% | **3/13** |
| Within 5% | **3/13** |
| FlashInfer faster by more than 5% | **7/13** |
| Geometric-mean TIRx speedup | **1.011x** |

The aggregate is nearly even, but it hides a wide spread: TIRx FlashAttention4 is 3.005x the throughput of FlashInfer FA2 at the selected GQA-prefill shape, while FlashInfer leads most selected normalization and TopK paths. Use the per-family and per-kernel rows for tuning decisions rather than the single mixed-workload mean.

## Results by family

Speedup is `FlashInfer latency / TIRx latency`; values above 1.0 favor TIRx.

| Family | Rows | Geomean TIRx speedup |
|---|---:|---:|
| Attention | 1 | 3.005x |
| Normalization | 2 | 0.862x |
| Activation / quantization | 3 | 0.958x |
| TopK | 4 | 0.886x |
| Recurrent / SSM | 3 | 0.983x |

## Complete representative table

The 5% band is descriptive, not a statistical significance test.

| Family | Kernel / config | TIRx µs | TIRx CV | FlashInfer implementation | FlashInfer µs | FI CV | TIRx speedup | Result |
|---|---|---:|---:|---|---:|---:|---:|---|
| Attention | `flash_attention4/s4096_h32kv4_causal` | 979.241 | 2.7% | `flashinfer_fa2` | 2942.277 | 0.1% | **3.005x** | TIRx faster |
| Normalization | `flashinfer_rmsnorm/rms_bf16_m32_h4096_xc_yc_pdl1` | 12.064 | 25.1% | `flashinfer_cutedsl` | 10.297 | 9.1% | **0.854x** | FlashInfer faster |
| Normalization | `flashinfer_rmsnorm_quant/bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | 20.767 | 24.3% | `flashinfer_cutedsl` | 18.089 | 8.9% | **0.871x** | FlashInfer faster |
| Activation / quantization | `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 2627.917 | 11.9% | `flashinfer` | 2581.898 | 13.5% | **0.982x** | within 5% |
| Activation / quantization | `mxfp4_quantize/fp16_linear_m4096_k4096` | 341.252 | 29.9% | `flashinfer` | 334.277 | 14.5% | **0.980x** | within 5% |
| Activation / quantization | `nvfp4_quantize/fp16_linear_m4096_k4096` | 318.155 | 15.7% | `flashinfer` | 290.755 | 9.0% | **0.914x** | FlashInfer faster |
| TopK | `fast_topk_clusters/f32_plain_b64_l16384_k256` | 243.972 | 41.9% | `flashinfer` | 194.933 | 1.4% | **0.799x** | FlashInfer faster |
| TopK | `filtered_topk/f32_plain_r64_l8192_k256` | 42.098 | 5.4% | `flashinfer` | 47.456 | 2.8% | **1.127x** | TIRx faster |
| TopK | `radix_topk_multi_cta/f32_basic_r4_l115188_k256_ctas3` | 74.850 | 38.4% | `flashinfer` | 61.841 | 4.2% | **0.826x** | FlashInfer faster |
| TopK | `radix_topk_single_cta/f32_basic_r64_l32768_k512` | 213.299 | 41.2% | `flashinfer` | 176.958 | 4.6% | **0.830x** | FlashInfer faster |
| Recurrent / SSM | `gdn_decode_bf16_ilp4/t4_b4_h8_hv16_tv16` | 89.817 | 24.2% | `flashinfer_cutedsl` | 80.138 | 10.0% | **0.892x** | FlashInfer faster |
| Recurrent / SSM | `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 440.968 | 9.9% | `flashinfer_cutedsl` | 432.091 | 10.2% | **0.980x** | within 5% |
| Recurrent / SSM | `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 3529.687 | 0.1% | `flashinfer_cuda` | 3832.441 | 0.1% | **1.086x** | TIRx faster |

## Selection and interpretation

The roster deliberately samples serving-relevant operator families rather than every shape: attention, RMSNorm, fused activation, FP4 quantization, four TopK variants, and three recurrent/SSM paths. Each reference adapter lives beside its kernel and executes the same fused operation on the same generated inputs. The workload roster is [`scripts/thor_flashinfer_representative.yaml`](scripts/thor_flashinfer_representative.yaml).

The GDN and grouped-KDA choices follow production dispatch shapes present in SGLang's kernel configuration manifests. FlashInfer remains the timed implementation baseline; SGLang supplies shape provenance rather than a second timing column. FlashInfer has no FA3 binary for `sm_110a` in this environment, so the attention baseline is its supported FA2 backend.

This is a kernel-launch microbenchmark, not end-to-end request throughput. It does not measure scheduler, KV-cache management, batching policy, CPU work, or network overhead. The 13 rows demonstrate representative performance, while the exhaustive numerical coverage remains documented in [THOR_VALIDATION.md](THOR_VALIDATION.md).

## Measurement provenance

| Field | Value |
|---|---|
| GPU | NVIDIA Jetson AGX Thor Developer Kit, 20 SMs |
| CUDA architecture | `sm_110a` |
| Power/clock state | `MAXN`; dynamic clocks (`jetson_clocks` was not locked) |
| Timer | `proton` |
| Rounds / aggregation | 5 / arithmetic mean |
| Warmup / repeat budget | 25 ms / 100 ms per implementation per round |
| TVM/TIR revision | `15b607d6` |
| TIRx-kernels revision | `d64108df` |
| FlashInfer version / revision | `0.6.18` / `f2e04400` |
| CUDA / PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 |

The population coefficient of variation exceeded 10% for either implementation in **10/13** rows. Dynamic clocks can move absolute latency between rounds; lock the production power mode and clocks before treating small differences as tuning wins. Both sides in every row nevertheless share the same process, GPU, timer, and five-round protocol.

## Reproduce

```bash
python -m tirx_kernels.bench_suite \
  --workloads scripts/thor_flashinfer_representative.yaml \
  --out-dir /home/tlopexh/thor-validation/flashinfer-native-final \
  --with-references --timer proton --rounds 5 --cooldown 0 \
  --max-prepare-processes 1 --ready-backlog 1 --no-probe --no-report
python scripts/report_thor_native.py \
  --run /home/tlopexh/thor-validation/flashinfer-native-final/runs/1.json \
  --output THOR_NATIVE_BASELINE.md
```

The Thor CUDA, TVM, CUPTI, and `sm_110a` environment variables described in [THOR_VALIDATION.md](THOR_VALIDATION.md) must be set first.

## Raw evidence

- Run JSON: `/home/tlopexh/thor-validation/flashinfer-native-final/runs/1.json`
- Run status: 13 `ok`; every row contains five TIRx and five FlashInfer round samples
