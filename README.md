# TIRx kernels

High-performance GPU kernels authored in
[Kern](tirx_kernels/kern/README.md) and compiled through
[TIRx](https://github.com/apache/tvm).

## Kernels

Registered kernels declare their CUDA target architecture in `KERNEL_META`.
Each linked name below is the public registry name accepted by `--kernel`; the
link opens its implementation. Run `python -m tirx_kernels.registry --format json`
for the authoritative config list.

### Native TIRx

- **GEMM:**
  [`fp16_bf16_gemm`](tirx_kernels/basic/fp16_bf16_gemm.py),
  [`nvfp4_gemm`](tirx_kernels/basic/nvfp4_gemm.py)
- **Normalization:**
  [`rmsnorm`](tirx_kernels/basic/rmsnorm.py)
- **Distributed:**
  [`allgather_gemm`](tirx_kernels/basic/allgather_gemm.py),
  [`gemm_reduce_scatter`](tirx_kernels/basic/gemm_reduce_scatter.py)

### Agent-evolved TIRx

Curated kernels selected from measured agent-evolution runs. Their canonical
modules retain the supported workload and stable provenance; run logs and
intermediate candidates remain outside this package. See the
[measured speedups](tirx_kernels/agent_evolved/README.md) for the benchmark
contract and results.

- **KDA forward:**
  [`agent_evolved_kda_forward_b1_t8192`](tirx_kernels/agent_evolved/kda_forward_b1_t8192.py)

### cuDNN Frontend ports

- **Persistent GEMM:**
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_amax`](tirx_kernels/cudnn/amax/dense_blockscaled_gemm_persistent_amax.py),
  [`cudnn_sm100_dense_gemm_persistent_swiglu`](tirx_kernels/cudnn/swiglu/dense_gemm_persistent_swiglu.py),
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant`](tirx_kernels/cudnn/swiglu/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py),
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant`](tirx_kernels/cudnn/srelu/dense_blockscaled_gemm_persistent_srelu_quant.py),
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant`](tirx_kernels/cudnn/dsrelu/dense_blockscaled_gemm_persistent_dsrelu_quant.py),
  [`cudnn_sm100_gemm_proj_rope_mxfp8_bf16in`](tirx_kernels/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_bf16in.py),
  [`cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in`](tirx_kernels/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_mxfp8in.py)
- **Grouped GEMM:**
  [`cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias`](tirx_kernels/cudnn/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py),
  [`cudnn_sm100_moe_grouped_gemm_dglu_dbias`](tirx_kernels/cudnn/dglu/moe_grouped_gemm_dglu_dbias.py)
- **Linear attention:**
  [`cudnn_sm100_kda_bprop_f16`](tirx_kernels/cudnn/linear_attention/kda_bprop_f16.py),
  [`cudnn_sm100_gdn_prefill_f16`](tirx_kernels/cudnn/linear_attention/gdn_prefill_f16.py),
  [`cudnn_sm100_gdn_recompute_f16`](tirx_kernels/cudnn/linear_attention/gdn_recompute_f16.py),
  [`cudnn_sm100_gdn2_prefill_f16`](tirx_kernels/cudnn/linear_attention/gdn2_prefill_f16.py),
  [`cudnn_sm100_gdn2_recompute_f16`](tirx_kernels/cudnn/linear_attention/gdn2_recompute_f16.py),
  [`cudnn_sm100_gdn2_bprop_f16`](tirx_kernels/cudnn/linear_attention/gdn2_bprop_f16.py),
  [`cudnn_sm100_gdn_bprop_f16`](tirx_kernels/cudnn/linear_attention/gdn_bprop_f16.py)
- **CSA compression:**
  [`cudnn_sm100_csa_compressor_fwd`](tirx_kernels/cudnn/csa/compressor_fwd_sm100.py)
- **Sparse attention:**
  [`cudnn_sm100_dsa_sparse_attention_backward`](tirx_kernels/cudnn/dsa/sparse_attention_backward.py),
  [`cudnn_sm100_bsa_forward_blk128`](tirx_kernels/cudnn/bsa/block_sparse_attention_forward_sm100_blk128.py),
  [`cudnn_sm100_bsa_forward_blk64`](tirx_kernels/cudnn/bsa/block_sparse_attention_forward_sm100_blk64.py),
  [`cudnn_sm100_bsa_forward_combine_blk64`](tirx_kernels/cudnn/bsa/block_sparse_attention_forward_combine_sm100_blk64.py),
  [`cudnn_sm100_bsa_backward_blk128`](tirx_kernels/cudnn/bsa/block_sparse_attention_backward_sm100_blk128.py),
  [`cudnn_sm100_bsa_backward_blk64`](tirx_kernels/cudnn/bsa/block_sparse_attention_backward_sm100_blk64.py)

### FlashAttention ports

- **Forward:**
  [`flash_attention4`](tirx_kernels/flashattention/flash_attention4.py)
- **Backward:**
  [`flash_attention_backward_sm100`](tirx_kernels/flashattention/flash_attention_backward.py)

### FlashInfer ports

Grouped by the FlashInfer Python entry point each port backs.

- **`flashinfer.activation`:**
  [`act_and_mul`](tirx_kernels/flashinfer/activation/act_and_mul.py),
  [`silu_and_mul_nvfp4_experts_quantize`](tirx_kernels/flashinfer/activation/silu_and_mul_nvfp4_experts_quantize.py)
- **`flashinfer.quantization`:**
  [`nvfp4_quantize`](tirx_kernels/flashinfer/quantization/nvfp4_quantize.py),
  [`nvfp4_quantize_per_token`](tirx_kernels/flashinfer/quantization/nvfp4_quantize_per_token.py),
  [`mxfp4_quantize`](tirx_kernels/flashinfer/quantization/mxfp4_quantize.py),
  [`mxfp8_quantize`](tirx_kernels/flashinfer/quantization/mxfp8_quantize.py)
- **`flashinfer.norm`:**
  [`flashinfer_rmsnorm`](tirx_kernels/flashinfer/norm/rmsnorm.py),
  [`flashinfer_rmsnorm_quant`](tirx_kernels/flashinfer/norm/rmsnorm_quant.py),
  [`flashinfer_rmsnorm_fp4quant`](tirx_kernels/flashinfer/norm/rmsnorm_fp4quant.py),
  [`flashinfer_add_rmsnorm_fp4quant`](tirx_kernels/flashinfer/norm/add_rmsnorm_fp4quant.py),
  [`flashinfer_layernorm`](tirx_kernels/flashinfer/norm/layernorm.py),
  [`flashinfer_fused_add_rmsnorm`](tirx_kernels/flashinfer/norm/fused_add_rmsnorm.py),
  [`flashinfer_fused_add_rmsnorm_quant`](tirx_kernels/flashinfer/norm/fused_add_rmsnorm_quant.py),
  [`flashinfer_fused_dit_layernorm`](tirx_kernels/flashinfer/norm/fused_dit_layernorm.py),
  [`flashinfer_qk_rmsnorm`](tirx_kernels/flashinfer/norm/qk_rmsnorm.py)
- **`flashinfer.mamba`:**
  [`selective_state_update_stp_simple`](tirx_kernels/flashinfer/mamba/selective_state_update_stp_simple.py),
  [`selective_state_update_stp_vertical`](tirx_kernels/flashinfer/mamba/selective_state_update_stp_vertical.py),
  [`selective_state_update_stp_horizontal`](tirx_kernels/flashinfer/mamba/selective_state_update_stp_horizontal.py),
  [`selective_state_update_mtp_simple`](tirx_kernels/flashinfer/mamba/selective_state_update_mtp_simple.py),
  [`selective_state_update_mtp_vertical`](tirx_kernels/flashinfer/mamba/selective_state_update_mtp_vertical.py),
  [`selective_state_update_mtp_horizontal`](tirx_kernels/flashinfer/mamba/selective_state_update_mtp_horizontal.py)
- **`flashinfer.kda`:**
  [`flashkda_bf16_fused_m128`](tirx_kernels/flashinfer/kda/bf16_fused_m128.py),
  [`recurrent_kda_decode_one_warp`](tirx_kernels/flashinfer/kda/recurrent_kda_decode_one_warp.py),
  [`recurrent_kda_decode_grouped`](tirx_kernels/flashinfer/kda/recurrent_kda_decode_grouped.py),
  [`flashkda_decode_t1_precomputed`](tirx_kernels/flashinfer/kda/flashkda_decode_t1_precomputed.py),
  [`flashkda_decode_t2_precomputed`](tirx_kernels/flashinfer/kda/flashkda_decode_t2_precomputed.py),
  [`flashkda_decode_t3_lower_bound`](tirx_kernels/flashinfer/kda/flashkda_decode_t3_lower_bound.py),
  [`flashkda_decode_t4_precomputed`](tirx_kernels/flashinfer/kda/flashkda_decode_t4_precomputed.py),
  [`flashkda_decode_t5_gram`](tirx_kernels/flashinfer/kda/flashkda_decode_t5_gram.py),
  [`flashkda_decode_t6_gram`](tirx_kernels/flashinfer/kda/flashkda_decode_t6_gram.py)
- **`flashinfer.gdn_decode`:**
  [`gdn_decode_bf16_ilp4`](tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_ilp4.py),
  [`gdn_decode_bf16_wide_vec_t1`](tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_t1.py),
  [`gdn_decode_bf16_wide_vec_mtp`](tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_mtp.py),
  [`gdn_decode_fp32_mtp_warp`](tirx_kernels/flashinfer/gdn_decode/gdn_decode_fp32_mtp_warp.py)
- **`flashinfer.gdn_prefill`:**
  [`gdn_prefill_sm100`](tirx_kernels/flashinfer/gdn_prefill/gdn_prefill_sm100.py),
  [`gdn_cp_prefill_sm100`](tirx_kernels/flashinfer/gdn_prefill/gdn_cp_prefill_sm100.py)
- **`flashinfer.cake_vsa`:**
  [`cake_vsa_blk128_compact_sm100`](tirx_kernels/flashinfer/cake_vsa/cake_vsa_blk128_compact_sm100.py),
  [`cake_vsa_ultrasparse_bsr_sm100`](tirx_kernels/flashinfer/cake_vsa/cake_vsa_ultrasparse_bsr_sm100.py),
  [`cake_vsa_longseq_sm100`](tirx_kernels/flashinfer/cake_vsa/cake_vsa_longseq_sm100.py),
  [`cake_vsa_longseq_sm103`](tirx_kernels/flashinfer/cake_vsa/cake_vsa_longseq_sm103.py)
- **`flashinfer.msa_ops`:**
  [`blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103`](tirx_kernels/flashinfer/msa_ops/blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103.py),
  [`blackwell_msa_decode_uniform_fp8_qkv_paged_sm100`](tirx_kernels/flashinfer/msa_ops/blackwell_msa_decode_uniform_fp8_qkv_paged_sm100.py),
  [`blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100`](tirx_kernels/flashinfer/msa_ops/blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100.py),
  [`blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103`](tirx_kernels/flashinfer/msa_ops/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103.py)
- **`flashinfer.gemm`:**
  [`tinygemm2_sm100`](tirx_kernels/flashinfer/gemm/tinygemm2_sm100.py),
  [`bmm_fp8_rubin`](tirx_kernels/flashinfer/gemm/bmm_fp8_rubin.py),
  [`grouped_gemm_masked_rubin`](tirx_kernels/flashinfer/gemm/grouped_gemm_masked_rubin.py)
- **`flashinfer.fused_moe`:**
  [`blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin`](tirx_kernels/flashinfer/fused_moe/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin.py)
- **`flashinfer.topk`:**
  [`fast_topk_clusters`](tirx_kernels/flashinfer/topk/fast_topk_clusters.py),
  [`filtered_topk`](tirx_kernels/flashinfer/topk/filtered_topk.py),
  [`radix_topk_multi_cta`](tirx_kernels/flashinfer/topk/radix_topk_multi_cta.py),
  [`radix_topk_single_cta`](tirx_kernels/flashinfer/topk/radix_topk_single_cta.py),
  [`stable_sort_topk_by_value`](tirx_kernels/flashinfer/topk/stable_sort_topk_by_value.py)

### FlashMLA ports

- **Sparse prefill:**
  [`sparse_flashmla_prefill_head64_phase1`](tirx_kernels/flashmla/sparse_prefill_head64_phase1.py),
  [`sparse_flashmla_prefill_head128_phase1`](tirx_kernels/flashmla/sparse_prefill_head128_phase1.py),
  [`sparse_flashmla_prefill_head128_small_topk_phase1`](tirx_kernels/flashmla/sparse_prefill_head128_small_topk_phase1.py)
- **Sparse decode:**
  [`sparse_flashmla_decode_head64`](tirx_kernels/flashmla/sparse_decode_head64.py)
- **Sparse forward:**
  [`flash_mla_sparse_fwd`](tirx_kernels/flashmla/flash_mla_sparse_fwd.py)

### DeepGEMM ports

- **Dense and grouped GEMM:**
  [`deepgemm_sm100_fp8_gemm_1d1d`](tirx_kernels/deepgemm/fp8_gemm_1d1d.py),
  [`deepgemm_sm100_m_grouped_fp8_gemm_contiguous`](tirx_kernels/deepgemm/m_grouped_fp8_gemm_contiguous.py),
  [`deepgemm_sm100_m_grouped_fp8_gemm_masked`](tirx_kernels/deepgemm/m_grouped_fp8_gemm_masked.py),
  [`deepgemm_sm100_k_grouped_fp8_gemm_contiguous`](tirx_kernels/deepgemm/k_grouped_fp8_gemm_contiguous.py),
  [`deepgemm_sm100_fp8_bmm`](tirx_kernels/deepgemm/fp8_bmm.py),
  [`deepgemm_sm100_tf32_hc_prenorm_gemm`](tirx_kernels/deepgemm/tf32_hc_prenorm_gemm.py)
- **MQA logits:**
  [`deepgemm_sm100_fp4_mqa_logits`](tirx_kernels/deepgemm/mqa_logits_fp4.py),
  [`deepgemm_sm100_fp8_mqa_logits`](tirx_kernels/deepgemm/mqa_logits_fp8.py),
  [`deepgemm_sm100_fp4_paged_mqa_logits`](tirx_kernels/deepgemm/paged_mqa_logits_fp4.py),
  [`deepgemm_sm100_fp8_paged_mqa_logits`](tirx_kernels/deepgemm/paged_mqa_logits_fp8.py)
- **MoE:**
  [`sm100_fp8_fp4_mega_moe`](tirx_kernels/deepgemm/sm100_fp8_fp4_mega_moe.py)

### DeepEP ports

- **Elastic communication:**
  [`deepep_dispatch`](tirx_kernels/deepep/dispatch.py),
  [`deepep_combine`](tirx_kernels/deepep/combine.py)

### MSA ports

- **Sparse-attention preparation:**
  [`msa_sparse_prepare_flat_schedule_sm100`](tirx_kernels/msa/sparse_prepare_flat_schedule.py),
  [`msa_sparse_prepare_fwd_split_atomic_sm100`](tirx_kernels/msa/sparse_prepare_fwd_split_atomic.py)
- **Sparse-attention forward:**
  [`msa_sparse_atten_fwd_sm100`](tirx_kernels/msa/sparse_atten_fwd.py),
  [`msa_sparse_atten_fwd_nvfp4_kv_sm100`](tirx_kernels/msa/sparse_atten_fwd_nvfp4_kv.py),
  [`msa_sparse_atten_fwd_combine_sm100`](tirx_kernels/msa/sparse_atten_fwd_combine.py)

## Performance

Per-workload numbers — our kernel time, every reference impl, and the
ref/ours ratio (>1 means ours is faster) — are pinned in
[`tirx_kernels/bench_suite/baseline.md`](tirx_kernels/bench_suite/baseline.md),
regenerated on every baseline promotion. See the
[bench-suite README](tirx_kernels/bench_suite/README.md) for how the sweep runs
and how to refresh the baseline.

## Installation

```bash
pip install tirx-kernels          # from a release
# or, from a checkout:
pip install -e .
```

### External dependencies

Correctness uses the original upstream implementations. Install the exact,
mutually compatible revisions from the repository lock:

```bash
python scripts/install_reference_dependencies.py
```

[`reference-dependencies.json`](reference-dependencies.json) is the single
source of truth for reference revisions and the shared CUTLASS DSL version.
The same command installs the pinned pytest/xdist runner. `torch` and `tvm.tirx`
remain externally managed runtime/compiler dependencies.

| Dependency       | Needed by                          | Notes                                                  |
| ---------------- | ---------------------------------- | ------------------------------------------------------ |
| `tvm.tirx`       | all kernels (compile + run)        | The TIRx compiler. Put it on `PYTHONPATH`, e.g. `/path/to/tir/python`. |
| `torch`          | all kernels                        | CUDA build matching your GPU.                          |
| `deep_gemm`      | FP8 GEMM and `deepgemm_*` baselines | Used for optimized reference kernels and the MegaMoE timer. |
| cuDNN Frontend (`cudnn`) | `cudnn_*` correctness and baselines | Source install pinned in the lock (v1.28.0); replaces any released `nvidia-cudnn-frontend` wheel, which lacks the CuTeDSL kernel sources. |
| `flashinfer`     | `nvfp4_gemm` baseline | Used for reference implementations. |
| `flash-attn` + CUTLASS DSL | `flash_attention_backward_sm100` baseline | Current SM100 forward/backward reference. |
| `sglang` (+ CUTLASS DSL) | `deepgemm_sm100_fp8_paged_mqa_logits` reference | Optional `sglang_cutedsl` benchmark reference. |
| `flash_mla`      | `sparse_flashmla_*` / `flash_mla_sparse_fwd` baselines | Reference impls. |
| `deep_ep`        | `deepep_*` correctness and baselines | Reference implementation. |
| `flash-linear-attention` | `agent_evolved_kda_forward_b1_t8192` correctness | Independent FLA BF16/Triton chunk reference. |
| `flash_kda`      | `flashkda_*` optional baselines | Raw FlashKDA benchmark peer. |
| `fmha_sm100` (MSA) | `msa_*` correctness and baselines | Reference implementation; set `MSA_PATH` to use a checkout elsewhere. |
| NVSHMEM          | `allgather_gemm`, `gemm_reduce_scatter` | Required to compile/run the GemmComm kernels. |

Correctness tests import and run these upstream implementations. The bench suite
does not launch or time benchmark reference implementations by default (kernel
data-preparation helpers may still import their upstream package). Pass
`--with-references` to enable reference launches; a missing enabled reference
fails its workload. See
[`tirx_kernels/bench_suite/README.md`](tirx_kernels/bench_suite/README.md)
for the prerequisites and workarounds.

## Usage

### Command line

```bash
# List discovered kernels (with their config labels)
python -m tirx_kernels.registry --format json

# Run correctness tests (optionally filter by kernel / config label)
pytest -n 16 tests/test_correctness.py

# Benchmark
python -m tirx_kernels.bench --kernel nvfp4_gemm
python -m tirx_kernels.bench --kernel nvfp4_gemm --with-references

# Pre-commit regression benchmark sweep (see tirx_kernels/bench_suite/README.md)
python -m tirx_kernels.bench_suite
```

### Programmatic API

Every kernel module exposes a small, uniform interface (see
`tirx_kernels/_protocol.py`):

```python
from tirx_kernels.registry import discover_kernels

kernels = discover_kernels()          # {name: module}
mod = kernels["fp16_bf16_gemm"]

mod.run_test(M=1024, N=1024, K=1024)  # compile + run + correctness check
mod.run_bench(M=1024, N=1024, K=1024) # profile (needs a GPU)

func = mod.get_kernel(M=1024, N=1024, K=1024)  # the TIRx PrimFunc
```

Each module also provides `KERNEL_META`: its name, category, exact
`runtime_cuda_archs`, and optional correctness-only `reference_requirements`.
The registry and test harness reject unsupported architectures before compile,
and skip correctness before GPU work when a declared reference package, version,
or Git source identity is unavailable. `CONFIGS` contains the test parameter sweep.

## License

Except where otherwise noted, this project is licensed under the Apache
License 2.0; see [LICENSE](LICENSE). Required Apache attribution notices are
collected in [NOTICE](NOTICE).

Every Python source file carries SPDX tags. Kernel ports derived from third-party projects
(cuDNN Frontend, DeepGEMM, fast.cu, FlashMLA, flash-attention, flash-attention-fp4, FlashInfer, MSA) additionally cite the upstream
project and the exact commit ported, retain the upstream copyright notice, and
declare the combined terms — for example `Apache-2.0 AND MIT`. Where an upstream
license requires its conditions text to travel with the source, that text is kept
in the file verbatim. The third-party section at the end of [LICENSE](LICENSE)
lists which components fall under which license, and [`licenses/`](licenses)
holds the corresponding license texts.
