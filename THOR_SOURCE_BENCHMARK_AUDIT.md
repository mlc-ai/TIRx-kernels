# Thor source-benchmark baseline audit

This audit starts from each selected TIRx kernel's source implementation and
its corresponding upstream benchmark.  It does not choose a framework first
and then look for a vaguely similar operator.

## Selection rule

1. A source-port kernel is compared first with the exact source launch, using
   the same dtype, layout, shape, fusion boundary, flags, and output contract.
2. A generic kernel with no imported source is compared with the conventional
   vendor implementation of the same operation.
3. Providers from an upstream benchmark that answer a different question are
   retained only as secondary controls.  Examples are eager PyTorch for fusion
   benefit, or `torch.topk` for library-level rather than algorithm-level TopK.
4. An unavailable source implementation is reported as unavailable; it is not
   replaced silently with an older algorithm generation.

## FlashInfer attention is not only FA2

The pinned FlashInfer `prefill.py` accepts `fa2`, `fa3`, `cudnn`,
`trtllm-gen`, `cutlass`, and `cute-dsl` backends.  Its benchmark scripts use
different ones for different questions:

| FlashInfer benchmark | Actual attention path |
|---|---|
| `bench_batch_attention.py` | Explicit `backend="fa2"` for the old paged-prefill/decode comparison |
| `bench_blackwell_attention.py` | Explicit `backend="cutlass"` |
| `bench_blackwell_attention_cutedsl.py` | FlashInfer CUTLASS versus `BatchPrefillCuteDSLWrapper` |

The pinned FA3 package image is not loadable for `sm_110a` in this environment,
but both FA2 and FlashInfer CuTeDSL run on Thor.  Neither should be mislabeled
as Dao-AILab FlashAttention-4: FA4 has its own CuTeDSL source and benchmark.

## Audit of the 20 measured rows

| TIRx kernel(s) | Source and corresponding upstream bench | What that bench actually compares | Correct primary for this TIRx comparison | Audit result |
|---|---|---|---|---|
| `fp16_bf16_gemm` (FP16 and BF16 rows) | TIRx-native kernel; no imported upstream kernel or source benchmark | N/A | `torch.matmul(..., out=...)`, which dispatches the exact dense NT GEMM to cuBLAS | **Keep.** cuBLAS is the vendor control; this is not a FlashInfer-port row. |
| `nvfp4_gemm` | FlashInfer [`routines/gemm.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/routines/gemm.py), routine `mm_fp4` | FlashInfer enumerates and validates supported `mm_fp4` backends | FlashInfer `mm_fp4`; on this Thor run `auto` resolved to `CutlassFp4GemmRunner`, tactic 28. Direct cuBLASLt NVFP4 remains an independent secondary control | **Keep, but name the resolved backend.** Both use the same packed E2M1 data, UE4M3 scale layout, block size, alpha, and BF16 output. |
| `flash_attention4` | Dao-AILab FA4 [`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/cute/flash_fwd_sm100.py) and [`benchmark_flash_attention_fp8.py`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/cute/benchmark_flash_attention_fp8.py) | The upstream bench times `_flash_attn_fwd` as `FA4-CuTe-BF16` / `FA4-CuTe-FP8`; PyTorch is the numerical reference and cuDNN is optional for FP8 | Upstream `_flash_attn_fwd` with FP16 inputs, reported as `flashattn_fa4_cutedsl` | **Corrected.** FlashInfer CuTeDSL is a useful serving-library peer; FlashInfer FA2 is only a legacy-generation control. |
| `flashinfer_rmsnorm`, `flashinfer_fused_add_rmsnorm`, `flashinfer_rmsnorm_quant` | FlashInfer CuTeDSL norm kernels and [`routines/norm.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/routines/norm.py) | The source bench dispatches the CuTeDSL backend and uses PyTorch equations for reference checking | The exact pinned FlashInfer CuTeDSL source launch | **Keep.** These are source-to-port comparisons with the same fusion and quantization boundary. |
| `flashinfer_layernorm` | FlashInfer `LayerNormKernel` in `flashinfer/norm/kernels/layernorm.py`; no standalone plain-LayerNorm row in the inspected routine driver | The public FlashInfer entry dispatches the same CuTeDSL source | Exact pinned FlashInfer CuTeDSL `layernorm` launch | **Keep.** Direct source A/B is stronger than substituting a generic framework LayerNorm. |
| `flashinfer_qk_rmsnorm` | FlashInfer `QKRMSNormKernel` / `qk_rmsnorm_cute`; the public combined-QK benchmark is [`bench_fused_qk_rmsnorm_rope.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_fused_qk_rmsnorm_rope.py) | The public benchmark has an extra RoPE fusion and therefore is not the same standalone contract | Exact pinned standalone `qk_rmsnorm_cute` source launch | **Keep.** Do not use the fused-QK-RMSNorm+RoPE timing as the primary number. |
| `flashinfer_fused_dit_layernorm` | FlashInfer [`bench_fused_dit_layernorm.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_fused_dit_layernorm.py) | Fused FlashInfer CUDA kernel versus an eager sequence of PyTorch operations | Exact fused FlashInfer CUDA source launch | **Keep.** Eager PyTorch is useful for fusion benefit, but is not the closest port-quality baseline. |
| `act_and_mul` | FlashInfer CUDA `act_and_mul_kernel` behind `silu_and_mul`, `gelu_and_mul`, and `gelu_tanh_and_mul`; no exact standalone comparison script was found | Related benches use the op as part of larger fused chains | Exact pinned FlashInfer CUDA source launch for the selected activation | **Keep.** A fused-chain number would change the operation boundary. |
| `mxfp4_quantize`, `nvfp4_quantize` | FlashInfer [`bench_mxfp4_quantize_backend_comparison.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_mxfp4_quantize_backend_comparison.py) and [`bench_nvfp4_quantize_backend_comparison.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_nvfp4_quantize_backend_comparison.py) | FlashInfer CUDA versus FlashInfer CuTeDSL, with backend-agreement checks | Exact pinned CuTeDSL source launch because these TIRx kernels port the CuTeDSL implementation | **Keep with caveat.** MXFP4 uses the same quantization contract but FlashInfer selects its 4-threads-per-scale-factor schedule on low-SM Thor while TIRx retains the ported 1-thread schedule. |
| `fast_topk_clusters`, `filtered_topk`, `radix_topk_single_cta`, `radix_topk_multi_cta` | FlashInfer [`bench_topk.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_topk.py) | FlashInfer selected algorithms versus `torch.topk`, with optional SGLang and deterministic/tie-break variants | The pinned corresponding FlashInfer internal source launch, with the algorithm explicitly forced when dispatch could choose another sibling | **Keep.** `torch.topk` is a useful secondary library control, but it is not the correct primary for judging individual FlashInfer algorithm ports. |
| `gdn_decode_bf16_ilp4` | FlashInfer [`bench_gdn_decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_gdn_decode.py) | Exact FlashInfer GDN kernels and optional Triton references, including MTP cases | Exact pinned FlashInfer CuTeDSL BF16-state ILP4 launch | **Keep.** Triton can be an additional implementation column after its state/layout contract is matched. |
| `recurrent_kda_decode_grouped` | FlashInfer [`bench_recurrent_kda.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_recurrent_kda.py) | The production `recurrent_kda` dispatch | Exact pinned grouped-CTA CuTeDSL source launch, not a one-warp sibling | **Keep.** The selected verify shape exercises the same grouped source specialization. |
| `selective_state_update_mtp_horizontal` | FlashInfer [`bench_ssu_sweep_mtp.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_ssu_sweep_mtp.py) | Triton reference plus FlashInfer `simple`, `vertical`, `horizontal`, and `auto` algorithms | FlashInfer CUDA `horizontal` source launch | **Keep.** `auto` or Triton would answer a best-library comparison, not whether the horizontal source port is competitive. |

## Corrected FA4 result on Thor

All four implementations below use the same batch 1, sequence 4096, 32 query
heads, 4 KV heads, head dimension 128, causal mask, FP16 Q/K/V storage, NHD
layout, and `1/sqrt(128)` softmax scale.  Every reference output is checked
against TIRx before timing.

| Implementation | Mean latency | CV | Relative to TIRx |
|---|---:|---:|---:|
| TIRx FA4 | 985.535 µs | 3.5% | 1.000x |
| Upstream FA4 CuTeDSL | 964.548 µs | 1.5% | 0.979x |
| FlashInfer CuTeDSL | 971.336 µs | 1.7% | 0.986x |
| FlashInfer FA2 | 2944.449 µs | 0.0% | 2.988x |

The defensible conclusion is that TIRx FA4 is in the same performance band as
the two CuTeDSL peers on this shape, with upstream FA4 about 2.2% faster in this
run.  The roughly 3x result against FA2 is real for that older implementation,
but it is not evidence that TIRx beats FA4.

Raw run: `/home/tlopexh/thor-validation/source-bench-final/runs/1.json`.
