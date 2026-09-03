# NVIDIA Thor native-baseline performance

Measured on 2026-09-03 on one NVIDIA Jetson AGX Thor Developer Kit. Every row compares TIRx with a contract-matched implementation on the same GPU and exact input shape. FA4 uses upstream FlashAttention-4 CuTeDSL as its primary baseline.

## At a glance

| Question | Result |
|---|---|
| Correctness and execution | **13/13 passed**; 0 failures and 0 interference retries |
| TIRx faster by more than 5% | **4/13** |
| Within 5% | **6/13** |
| Reference faster by more than 5% | **3/13** |
| Geometric-mean TIRx speedup | **1.001x** |

The mixed aggregate hides a wide spread: at the selected GQA-prefill shape, the upstream FA4 CuTeDSL latency divided by TIRx latency is 0.979x. The plain RMSNorm and GELU activation paths favor FlashInfer, and most other rows are close. Use the per-family and per-kernel rows for tuning decisions rather than the single mixed-workload mean.

## Results by family

Speedup is `primary-reference latency / TIRx latency`; values above 1.0 favor TIRx.

| Family | Rows | Geomean TIRx speedup |
|---|---:|---:|
| Attention | 1 | 0.979x |
| Normalization | 2 | 0.876x |
| Activation / quantization | 3 | 1.047x |
| TopK | 4 | 1.054x |
| Recurrent / SSM | 3 | 0.984x |

## Complete representative table

The 5% band is descriptive, not a statistical significance test.

| Family | Kernel / config | TIRx µs | TIRx CV | Primary reference | Reference µs | Ref CV | TIRx speedup | Result |
|---|---|---:|---:|---|---:|---:|---:|---|
| Attention | `flash_attention4/s4096_h32kv4_causal` | 985.535 | 3.5% | `flashattn_fa4_cutedsl` | 964.548 | 1.5% | **0.979x** | within 5% |
| Normalization | `flashinfer_rmsnorm/rms_bf16_m32_h4096_xc_yc_pdl1` | 11.330 | 21.1% | `flashinfer_cutedsl` | 9.885 | 9.6% | **0.872x** | Reference faster |
| Normalization | `flashinfer_rmsnorm_quant/bf16_e4m3_m64_h8192_xc_yc_pdl0_s1` | 22.736 | 18.0% | `flashinfer_cutedsl` | 20.016 | 8.3% | **0.880x** | Reference faster |
| Activation / quantization | `act_and_mul/gelu_tanh_fp16_d11008_t8192` | 2389.711 | 11.0% | `flashinfer` | 2386.804 | 12.6% | **0.999x** | within 5% |
| Activation / quantization | `mxfp4_quantize/fp16_linear_m4096_k4096` | 280.820 | 11.1% | `flashinfer` | 323.658 | 14.2% | **1.153x** | TIRx faster |
| Activation / quantization | `nvfp4_quantize/fp16_linear_m4096_k4096` | 254.796 | 11.1% | `flashinfer` | 254.155 | 9.4% | **0.997x** | within 5% |
| TopK | `fast_topk_clusters/f32_plain_b64_l16384_k256` | 198.190 | 0.9% | `flashinfer` | 197.375 | 0.3% | **0.996x** | within 5% |
| TopK | `filtered_topk/f32_plain_r64_l8192_k256` | 41.798 | 4.5% | `flashinfer` | 47.615 | 3.3% | **1.139x** | TIRx faster |
| TopK | `radix_topk_multi_cta/f32_basic_r4_l115188_k256_ctas3` | 66.400 | 2.2% | `flashinfer` | 67.430 | 2.0% | **1.016x** | within 5% |
| TopK | `radix_topk_single_cta/f32_basic_r64_l32768_k512` | 174.729 | 3.6% | `flashinfer` | 187.086 | 1.4% | **1.071x** | TIRx faster |
| Recurrent / SSM | `gdn_decode_bf16_ilp4/t4_b4_h8_hv16_tv16` | 83.903 | 5.4% | `flashinfer_cutedsl` | 82.843 | 7.0% | **0.987x** | within 5% |
| Recurrent / SSM | `recurrent_kda_decode_grouped/ver_t8_hv12_b16` | 431.529 | 8.2% | `flashinfer_cutedsl` | 383.748 | 9.8% | **0.889x** | Reference faster |
| Recurrent / SSM | `selective_state_update_mtp_horizontal/b512_h64_d64_s128_t6_r8_statebf16_official` | 3526.523 | 0.0% | `flashinfer_cuda` | 3827.694 | 0.0% | **1.085x** | TIRx faster |

Attention secondary controls (same tensors and shape; not used in the aggregate):

| Implementation | Latency µs | CV | latency / TIRx |
|---|---:|---:|---:|
| `flashinfer_cutedsl` | 971.336 | 1.7% | 0.986x |
| `flashinfer_fa2` | 2944.449 | 0.0% | 2.988x |

## Selection and interpretation

The roster deliberately samples serving-relevant operator families rather than every shape: attention, RMSNorm, fused activation, FP4 quantization, four TopK variants, and three recurrent/SSM paths. Each reference adapter lives beside its kernel and executes the same fused operation on the same generated inputs. The workload roster is [`scripts/thor_flashinfer_representative.yaml`](scripts/thor_flashinfer_representative.yaml).
The per-kernel source and upstream-benchmark decisions are documented in [THOR_SOURCE_BENCHMARK_AUDIT.md](THOR_SOURCE_BENCHMARK_AUDIT.md).

Because attention is the most prominent result, a separate complete 32-config sequence-length, GQA-ratio, and causal-mask sweep is reported in [THOR_FLASHINFER_ATTENTION.md](THOR_FLASHINFER_ATTENTION.md). That older sweep is explicitly a secondary FA2 generation comparison, not the FA4 headline baseline.

The GDN and grouped-KDA choices follow production dispatch shapes present in SGLang's kernel configuration manifests. FlashInfer remains the timed implementation baseline for those ported kernels; SGLang supplies shape provenance rather than a second timing column. Attention is different: upstream FA4 CuTeDSL is primary, FlashInfer's own CuTeDSL path is a serving-library peer, and FA2 is retained only as a legacy control.

This is a kernel-launch microbenchmark, not end-to-end request throughput. It does not measure scheduler, KV-cache management, batching policy, CPU work, or network overhead. The 13 rows demonstrate representative performance, while the exhaustive numerical coverage remains documented in [THOR_VALIDATION.md](THOR_VALIDATION.md).

## Measurement provenance

| Field | Value |
|---|---|
| GPU | NVIDIA Jetson AGX Thor Developer Kit, 20 SMs |
| CUDA architecture | `sm_110a` |
| Power/clock state | `MAXN`; dynamic clocks (`jetson_clocks` was not locked) |
| Timer | `proton` |
| Rounds / aggregation | 5 / arithmetic mean |
| Warmup / repeat budget | 1000 ms / 100 ms per implementation per round |
| TVM/TIR revision | `15b607d6` |
| TIRx-kernels revision | `cfce35dd-dirty` |
| FlashInfer version / revision | `0.6.18` / `f2e04400` |
| FlashAttention-4 revision | `0251105a` |
| CUDA / PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 |

The population coefficient of variation exceeded 10% for either implementation in **5/13** rows. Dynamic clocks can move absolute latency between rounds; lock the production power mode and clocks before treating small differences as tuning wins. Both sides in every row nevertheless share the same process, GPU, timer, and five-round protocol.

## Reproduce

```bash
python -m tirx_kernels.bench_suite \
  --workloads scripts/thor_flashinfer_representative.yaml \
  --out-dir /home/tlopexh/thor-validation/source-bench-final \
  --with-references --timer proton --rounds 5 --cooldown 0 \
  --max-prepare-processes 1 --ready-backlog 1 --no-probe --no-report
python scripts/report_thor_native.py \
  --run /home/tlopexh/thor-validation/source-bench-final/runs/1.json \
  --output THOR_NATIVE_BASELINE.md
```

The Thor CUDA, TVM, CUPTI, and `sm_110a` environment variables described in [THOR_VALIDATION.md](THOR_VALIDATION.md) must be set first.

## Raw evidence

- Run JSON: `/home/tlopexh/thor-validation/source-bench-final/runs/1.json`
- Run status: 13 `ok`; every reported implementation has five round samples
