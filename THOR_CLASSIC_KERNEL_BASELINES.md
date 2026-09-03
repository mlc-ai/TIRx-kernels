# Thor classic-kernel baseline inventory

This document inventories benchmark sources before any new Thor performance
sweep is run.  It groups kernels by operator semantics, not by Python module or
registry entry, so multiple implementations of the same operation do not
inflate the coverage count.

## Scope and evidence

"Classic" means an operator family that is either already represented in
TIRx-kernels and relevant to an LLM runtime, or is one of the serving primitives
whose absence would be important when describing coverage.  The inventory was
checked against:

- FlashInfer commit
  [`f2e04400`](https://github.com/flashinfer-ai/flashinfer/tree/f2e04400e330fb2debe0bf8730d9424a1d37927f),
  which is pinned by `reference-dependencies.json`.
- SGLang commit
  [`c7c03ec5`](https://github.com/sgl-project/sglang/tree/c7c03ec53b1e664c2d415db4f02e43f86661f31d),
  which is pinned by `reference-dependencies.json`.
- vLLM commit
  [`cee0f92c`](https://github.com/vllm-project/vllm/tree/cee0f92c02112ace7120da45896d27e0fe95ea1e),
  inspected from current `main` on 2026-09-03.  vLLM is not currently pinned or
  installed by this repository.

The status codes deliberately distinguish a benchmark from an implementation:

- **E**: a standalone benchmark with matching core operator semantics exists.
  Tensor layout, dtype, shape, outputs, fusion boundary, and timing method must
  still be aligned before publishing a ratio.
- **A**: adaptable only.  A related benchmark or implementation exists, but its
  fusion, dtype, layout, algorithm, or provider differs, or only a correctness
  test was found.
- **--**: no reusable standalone benchmark was found in the inspected tree.
- **Gap**: TIRx-kernels does not currently expose this operator as a standalone
  kernel.

An **E** is therefore an availability result, not a claim that the script will
run unmodified on Thor (`sm_110a`).  Architecture support must be probed next.

## A. Main inference-kernel families already in TIRx-kernels

| # | Operator family | TIRx-kernels coverage | FlashInfer | SGLang | vLLM | Baseline to try first |
|---:|---|---|---|---|---|---|
| 1 | Dense FP16/BF16 GEMM | `fp16_bf16_gemm`, `tinygemm2_sm100` | **E** persistent GEMM | **A** only specialized/router paths found | **A** tests/specialized paths, no plain standalone bench found | FlashInfer persistent GEMM; retain cuBLAS as the vendor control |
| 2 | Dense/batched FP8 GEMM | `deepgemm_sm100_fp8_gemm_1d1d`, two FP8 BMMs | **E** | **E** | **E** | FlashInfer `mm_fp8`; use matching per-tensor/per-block scale contract |
| 3 | FP4/MXFP4/NVFP4 block-scaled GEMM | `nvfp4_gemm` and block-scaled cuDNN/DeepGEMM paths | **E** | **E** | **E** | FlashInfer block-scaled GEMM |
| 4 | Grouped GEMM | cuDNN and DeepGEMM M/K-grouped variants | **E** | **E** | **E** | FlashInfer grouped GEMM, split into contiguous and masked contracts |
| 5 | Fused MoE | `sm100_fp8_fp4_mega_moe` plus grouped-MoE building blocks | **E** | **E** | **E** | FlashInfer fused MoE; MegaMoE needs its exact FP8/FP4 contract |
| 6 | Dense attention forward/prefill | `flash_attention4` | **E** | **A** inspected FA4 bench is FP8-specific | **A** inspected prefill bench delegates to FlashInfer/TRT-LLM | FlashInfer FA2 on Thor; FA3 has no loadable `sm_110a` image in the installed build |
| 7 | Block-sparse / sparse-MLA attention forward | BSA, FlashMLA, and MSA forward paths | **E** | **E** | **A** implementations/tests found, no standalone sparse-attention bench found | FlashInfer, but benchmark BSA, sparse MLA, and MSA separately |
| 8 | RMSNorm and fused-add/quantized RMSNorm | native RMSNorm plus FlashInfer norm ports | **E** | **E** | **E** | FlashInfer exact entry point for each fusion boundary |
| 9 | LayerNorm / fused DiT LayerNorm | FlashInfer LayerNorm and fused DiT ports | **E** | **E** generic norm | **E** generic LayerNorm | FlashInfer for fused DiT; SGLang or vLLM for plain LayerNorm |
| 10 | Activation-and-multiply / fused activation+quant | `act_and_mul`, SiLU-and-mul NVFP4 expert quantization | **A** exact op is present inside a fused-chain bench | **E** | **E** | FlashInfer for the fused chain; SGLang/vLLM for standalone activation-and-mul |
| 11 | FP8/MXFP8/FP4/NVFP4 quantization | four standalone FlashInfer ports plus norm/activation fusions | **E** | **E** | **E** | FlashInfer backend-comparison benches |
| 12 | TopK and MoE routing | five TopK variants, including filtered/radix/stable-sort forms | **E** | **E** | **E** | FlashInfer; keep plain TopK, page-table TopK, and MoE routing as separate contracts |
| 13 | RoPE, QK norm, and fused norm+RoPE | QK RMSNorm and projection-GEMM+YARN-RoPE fusions | **E** RoPE and fused QK-RMSNorm+RoPE | **E** | **E** standalone RoPE; fused forms have tests | FlashInfer where the fusion matches; SGLang for standalone QKNorm |
| 14 | GDN linear attention | decode, prefill, recompute, and backward-family ports | **E** decode/prefill | **E** decode/prefill | **A** implementations/tests found, no standalone GDN bench found | FlashInfer for decode/prefill |
| 15 | KDA linear attention | agent-evolved, FlashKDA/recurrent, and cuDNN backward ports | **E** | **E** | **A** Kimi-K3-specific benchmark is ROCm-only | FlashInfer for recurrent KDA; match state update and MTP semantics |
| 16 | Mamba selective-state update | six STP/MTP and scheduling variants | **E** | **A** runtime implementation, no standalone Mamba/SSU bench found | **E** | FlashInfer SSU/Mamba benchmark |
| 17 | MQA logits / sparse-attention indexer | FP4/FP8 dense and paged MQA logits | **A** related MSA proxy/TopK routines, not the same contract | **E** paged MQA logits vs DeepGEMM | **A** implementations/tests found, no standalone MQA-logits bench found | SGLang paged-MQA benchmark |

## B. Specialized, training, or multi-GPU families in TIRx-kernels

These are real repository coverage, but they should not be mixed into the main
single-GPU inference geomean.

| # | Operator family | TIRx-kernels coverage | FlashInfer | SGLang | vLLM | Correct comparison |
|---:|---|---|---|---|---|---|
| 18 | Dense attention backward | `flash_attention_backward_sm100` | **--** inference library | **A** generic backward timing helper, no ready matching framework baseline | **--** inference runtime | Upstream FlashAttention backward at the pinned commit |
| 19 | Sparse attention backward | BSA backward and DSA sparse-attention backward | **--** | **--** | **--** | Original cuDNN Frontend kernels; do not substitute a forward kernel |
| 20 | Fused GEMM epilogues | amax, SwiGLU, SReLU/DSReLU+quant, projection+RoPE | **A** some component/fused-chain benches | **A** related fusions | **A** related fusions | Original cuDNN Frontend exact fusion; component timings are diagnostic only |
| 21 | TF32 HC pre-norm/router GEMM | `deepgemm_sm100_tf32_hc_prenorm_gemm` | **A** router GEMM differs | **A** router GEMM differs | **A** router GEMM differs | Pinned DeepGEMM implementation |
| 22 | CSA compression / sparse preparation | cuDNN CSA compressor and MSA preparation kernels | **A** MSA proxy/indexer differs | **A** CSA runtime implementation found, no standalone matching bench | **--** | Original cuDNN/MSA sources; build an exact harness if this becomes a headline kernel |
| 23 | Fused collectives / expert communication | allgather-GEMM, GEMM-reduce-scatter, DeepEP dispatch/combine | **A** all-reduce/all-to-all benches | **A** all-reduce/fused-collective benches | **A** fused-collective benches | Exact NVSHMEM/DeepEP operation on a multi-GPU system; out of scope for single-GPU Thor |

## C. Classic serving primitives not currently exposed by TIRx-kernels

These should be reported as coverage gaps, not silently counted as passed.

| # | Operator family | TIRx-kernels | FlashInfer | SGLang | vLLM | Consequence |
|---:|---|---|---|---|---|---|
| 24 | Paged/decode attention | **Gap** (sparse decode is present, generic paged decode is not) | **E** | **E** | **E** | No TIRx-vs-framework number until a TIRx kernel is added |
| 25 | KV-cache append/reshape/store | **Gap** | **E** | **E** | **E** | No TIRx-vs-framework number until a TIRx kernel is added |
| 26 | Sampling (top-k/top-p/from-probs) | **Gap** | **E** | **A** inspected SGLang script calls FlashInfer sampling | **--** no standalone kernel benchmark found | FlashInfer is available, but there is no TIRx candidate to compare |

## Concrete benchmark evidence

### FlashInfer (preferred when exact)

- GEMM: [`bench_persistent_gemm.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_persistent_gemm.py),
  [`bench_mm_fp8.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_mm_fp8.py), and
  [`bench_cute_dsl_blockscaled_gemm.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_cute_dsl_blockscaled_gemm.py).
- Grouped GEMM/MoE: [`bench_deepgemm_blackwell.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_deepgemm_blackwell.py) and
  [`bench_cutlass_fused_moe.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_cutlass_fused_moe.py).
- Attention: [`bench_batch_attention.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_batch_attention.py),
  [`bench_batch_decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_batch_decode.py),
  [`bench_block_sparse_attention.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_block_sparse_attention.py), and
  [`bench_deepseek_mla.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_deepseek_mla.py).
- Norm/activation/quantization: [`bench_fused_add_rmsnorm.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_fused_add_rmsnorm.py),
  [`bench_fused_dit_layernorm.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_fused_dit_layernorm.py),
  [`bench_silu_and_mul_nvfp4_quantize.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_silu_and_mul_nvfp4_quantize.py), and
  [`bench_nvfp4_quantize_backend_comparison.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_nvfp4_quantize_backend_comparison.py).
- RoPE/TopK: [`bench_rope.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_rope.py),
  [`bench_fused_qk_rmsnorm_rope.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_fused_qk_rmsnorm_rope.py), and
  [`bench_topk.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_topk.py).
- Linear/state-space: [`bench_gdn_decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_gdn_decode.py),
  [`bench_gdn_prefill.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_gdn_prefill.py),
  [`bench_recurrent_kda.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_recurrent_kda.py), and
  [`bench_mamba_ssd_combined.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_mamba_ssd_combined.py).
- Serving/communication: [`bench_append_paged_kv_cache.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_append_paged_kv_cache.py),
  [`bench_sampling.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/bench_sampling.py), and
  [`bench_quantized_allreduce.py`](https://github.com/flashinfer-ai/flashinfer/blob/f2e04400e330fb2debe0bf8730d9424a1d37927f/benchmarks/comm/bench_quantized_allreduce.py).

### SGLang (fallback when FlashInfer is absent or not exact)

- Linear attention: [`bench_gdn_decode.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/benchmark/bench_linear_attention/bench_gdn_decode.py),
  [`bench_gdn_prefill.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/benchmark/bench_linear_attention/bench_gdn_prefill.py), and
  [`bench_kda_decode.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/benchmark/bench_linear_attention/bench_kda_decode.py).
- GEMM/MoE: [`benchmark_deepgemm_fp8_gemm.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py),
  [`bench_fp4_gemm.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/python/sglang/kernels/aot/benchmark/bench_fp4_gemm.py), and
  [`benchmark_vllm_vs_sglang_fused_moe_triton.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py).
- Norm/activation/quantization/RoPE: [`bench_rmsnorm.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/python/sglang/kernels/aot/benchmark/bench_rmsnorm.py),
  [`bench_activation.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/python/sglang/kernels/aot/benchmark/bench_activation.py),
  [`bench_per_tensor_quant_fp8.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/python/sglang/kernels/aot/benchmark/bench_per_tensor_quant_fp8.py), and
  [`bench_fused_qknorm_rope.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/test/registered/kernels/benchmark/attention/bench_fused_qknorm_rope.py).
- Sparse/indexer: [`bench_sparse_mla_q8kv8_prefill_sm90.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/test/registered/kernels/benchmark/attention/bench_sparse_mla_q8kv8_prefill_sm90.py) and
  [`benchmark_cute_dsl_fp8_paged_mqa_logits.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py).
- TopK/sampling/KV cache: [`bench_topk.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/test/registered/kernels/benchmark/attention/bench_topk.py),
  [`bench_top_k_top_p_sampling.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/python/sglang/kernels/aot/benchmark/bench_top_k_top_p_sampling.py), and
  [`bench_store_cache.py`](https://github.com/sgl-project/sglang/blob/c7c03ec53b1e664c2d415db4f02e43f86661f31d/test/registered/kernels/benchmark/kvcache/bench_store_cache.py).

### vLLM (secondary fallback; not yet pinned locally)

- GEMM/MoE: [`benchmark_fp8_gemm.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_fp8_gemm.py),
  [`benchmark_nvfp4_gemm.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_nvfp4_gemm.py),
  [`benchmark_grouped_gemm_cutlass.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_grouped_gemm_cutlass.py), and
  [`benchmark_moe.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_moe.py).
- Core elementwise: [`benchmark_rmsnorm.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_rmsnorm.py),
  [`benchmark_activation.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_activation.py),
  [`benchmark_nvfp4_quant.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_nvfp4_quant.py),
  [`benchmark_rope.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_rope.py), and
  [`benchmark_fused_topk.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_fused_topk.py).
- Attention/state/serving: [`benchmark_paged_attention.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_paged_attention.py),
  [`benchmark_trtllm_prefill_attention.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_trtllm_prefill_attention.py),
  [`benchmark_selective_state_update.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_selective_state_update.py),
  [`benchmark_kimi_k3_kda_decode.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_kimi_k3_kda_decode.py), and
  [`benchmark_reshape_and_cache.py`](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/benchmarks/kernels/benchmark_reshape_and_cache.py).

## Proposed benchmark order

Do not run one aggregate sweep across mismatched semantics.  The defensible
sequence is:

1. Probe whether each selected reference imports and launches on `sm_110a`.
2. Start with the exact FlashInfer rows: dense attention forward, RMSNorm,
   quantization, TopK, GDN, KDA, Mamba SSU, and matching GEMM variants.
3. Use SGLang only for exact gaps, initially paged MQA logits and standalone
   activation/QKNorm where needed.
4. Use vLLM only when it provides an independent exact kernel not already
   covered, rather than timing a vLLM script that calls FlashInfer.
5. Record backward, special fused epilogues, and communication in separate
   tables with their native upstream baselines.
6. Report serving gaps (paged decode, KV cache, sampling) as gaps until TIRx
   candidates exist.

Every final performance row must record the exact semantic contract and the
reference provider actually invoked, not merely the repository containing the
benchmark script.
