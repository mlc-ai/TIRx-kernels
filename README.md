# TIRx kernels

High-performance GPU kernels authored in
[tirx-lite](tirx_kernels/tirx_lite/README.md) and compiled through
[TIRx](https://github.com/apache/tvm).

## Kernels

Task sets live under the root-level `tirx_kernels/` package as
`tirx_kernels/<task_set>/<task>/<device>/`.
Each task has one home; distinct implementations remain side by side, with no
required dispatcher. `definition.json`, `workloads.json`, and task-local
`tirx_kernels.bench.py` are optional. Existing Python configuration lists and suite YAML
retain their formats. See [the layout guide](docs/layout.md).

The hardware directory identifies the implementation's primary target. The exact
execution allowlist remains `KERNEL_META["runtime_cuda_archs"]`. Unless annotated,
implementations support `sm_100a`, `sm_103a`, and `sm_107a`; annotations list a
restricted set or additional architectures. Existing registry names, including
`curated_*`, remain stable for workload and result compatibility. Curated kernel
provenance and measurements are retained in [these notes](docs/curated-results.md).

### basic

- **allgather_gemm:**
  [`allgather_gemm`](tirx_kernels/basic/allgather_gemm/b200/allgather_gemm.py) ⟨sm_100a⟩
- **fp16_bf16_gemm:**
  [`fp16_bf16_gemm`](tirx_kernels/basic/fp16_bf16_gemm/b200/fp16_bf16_gemm.py) ⟨+sm_110a⟩
- **gemm_reduce_scatter:**
  [`gemm_reduce_scatter`](tirx_kernels/basic/gemm_reduce_scatter/b200/gemm_reduce_scatter.py) ⟨sm_100a⟩
- **kda_backward_packed:**
  [`curated_kda_backward_packed`](tirx_kernels/basic/kda_backward_packed/b200/kda_backward_packed.py) ⟨sm_100a⟩
- **kda_forward:**
  [`curated_kda_forward_portfolio_multishape`](tirx_kernels/basic/kda_forward/b200/kda_forward_portfolio_multishape.py) ⟨sm_100a⟩
- **nvfp4_gemm:**
  [`fastcu_nvfp4_gemm_gb300`](tirx_kernels/basic/nvfp4_gemm/gb300/nvfp4_gemm_gb300.py) ⟨sm_103a⟩,
  [`nvfp4_gemm`](tirx_kernels/basic/nvfp4_gemm/b200/nvfp4_gemm.py)
- **rmsnorm:**
  [`flashinfer_rmsnorm`](tirx_kernels/basic/rmsnorm/b200/flashinfer_rmsnorm.py) ⟨+sm_110a⟩,
  [`rmsnorm`](tirx_kernels/basic/rmsnorm/b200/rmsnorm.py)

### cudnn

- **block_sparse_attention_backward:**
  [`cudnn_sm100_bsa_backward_blk128`](tirx_kernels/cudnn/block_sparse_attention_backward/b200/block_sparse_attention_backward_sm100_blk128.py),
  [`cudnn_sm100_bsa_backward_blk64`](tirx_kernels/cudnn/block_sparse_attention_backward/b200/block_sparse_attention_backward_sm100_blk64.py)
- **block_sparse_attention_forward:**
  [`cudnn_sm100_bsa_forward_blk128`](tirx_kernels/cudnn/block_sparse_attention_forward/b200/block_sparse_attention_forward_sm100_blk128.py),
  [`cudnn_sm100_bsa_forward_blk64`](tirx_kernels/cudnn/block_sparse_attention_forward/b200/block_sparse_attention_forward_sm100_blk64.py)
- **block_sparse_attention_forward_combine:**
  [`cudnn_sm100_bsa_forward_combine_blk64`](tirx_kernels/cudnn/block_sparse_attention_forward_combine/b200/block_sparse_attention_forward_combine_sm100_blk64.py)
- **compressor_fwd:**
  [`cudnn_sm100_csa_compressor_fwd`](tirx_kernels/cudnn/compressor_fwd/b200/compressor_fwd_sm100.py)
- **dense_blockscaled_gemm_persistent_amax:**
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_amax`](tirx_kernels/cudnn/dense_blockscaled_gemm_persistent_amax/b200/dense_blockscaled_gemm_persistent_amax.py)
- **dense_blockscaled_gemm_persistent_dsrelu_quant:**
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant`](tirx_kernels/cudnn/dense_blockscaled_gemm_persistent_dsrelu_quant/b200/dense_blockscaled_gemm_persistent_dsrelu_quant.py)
- **dense_blockscaled_gemm_persistent_srelu_quant:**
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant`](tirx_kernels/cudnn/dense_blockscaled_gemm_persistent_srelu_quant/b200/dense_blockscaled_gemm_persistent_srelu_quant.py)
- **dense_blockscaled_gemm_persistent_swiglu_interleaved_quant:**
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant`](tirx_kernels/cudnn/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant/b200/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py)
- **dense_gemm_persistent_swiglu:**
  [`cudnn_sm100_dense_gemm_persistent_swiglu`](tirx_kernels/cudnn/dense_gemm_persistent_swiglu/b200/dense_gemm_persistent_swiglu.py)
- **flex_attention_backward:**
  [`cudnn_sm100_flex_attention_backward`](tirx_kernels/cudnn/flex_attention_backward/b200/flex_attention_backward_sm100.py) ⟨sm_100a⟩
- **flex_attention_forward:**
  [`cudnn_sm100_flex_attention_forward_hd256`](tirx_kernels/cudnn/flex_attention_forward/b200/forward_hd256_sm100.py) ⟨sm_100a sm_103a⟩,
  [`cudnn_sm103_flex_attention_forward`](tirx_kernels/cudnn/flex_attention_forward/gb300/forward_sm103.py) ⟨sm_103a⟩
- **gdn2_bprop_f16:**
  [`cudnn_sm100_gdn2_bprop_f16`](tirx_kernels/cudnn/gdn2_bprop_f16/b200/gdn2_bprop_f16.py)
- **gdn2_prefill_f16:**
  [`cudnn_sm100_gdn2_prefill_f16`](tirx_kernels/cudnn/gdn2_prefill_f16/b200/gdn2_prefill_f16.py) ⟨+sm_110a⟩
- **gdn2_recompute_f16:**
  [`cudnn_sm100_gdn2_recompute_f16`](tirx_kernels/cudnn/gdn2_recompute_f16/b200/gdn2_recompute_f16.py) ⟨+sm_110a⟩
- **gdn_bprop_f16:**
  [`cudnn_sm100_gdn_bprop_f16`](tirx_kernels/cudnn/gdn_bprop_f16/b200/gdn_bprop_f16.py)
- **gdn_prefill_f16:**
  [`cudnn_sm100_gdn_prefill_f16`](tirx_kernels/cudnn/gdn_prefill_f16/b200/gdn_prefill_f16.py) ⟨+sm_110a⟩
- **gdn_recompute_f16:**
  [`cudnn_sm100_gdn_recompute_f16`](tirx_kernels/cudnn/gdn_recompute_f16/b200/gdn_recompute_f16.py) ⟨+sm_110a⟩
- **gemm_proj_rope_mxfp8:**
  [`cudnn_sm100_gemm_proj_rope_mxfp8_bf16in`](tirx_kernels/cudnn/gemm_proj_rope_mxfp8/b200/gemm_proj_rope_mxfp8_bf16in.py),
  [`cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in`](tirx_kernels/cudnn/gemm_proj_rope_mxfp8/b200/gemm_proj_rope_mxfp8_mxfp8in.py)
- **kda_bprop_f16:**
  [`cudnn_sm100_kda_bprop_f16`](tirx_kernels/cudnn/kda_bprop_f16/b200/kda_bprop_f16.py) ⟨+sm_110a⟩
- **moe_blockscaled_grouped_gemm_dglu_dbias:**
  [`cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias`](tirx_kernels/cudnn/moe_blockscaled_grouped_gemm_dglu_dbias/b200/moe_blockscaled_grouped_gemm_dglu_dbias.py)
- **moe_grouped_gemm_dglu_dbias:**
  [`cudnn_sm100_moe_grouped_gemm_dglu_dbias`](tirx_kernels/cudnn/moe_grouped_gemm_dglu_dbias/b200/moe_grouped_gemm_dglu_dbias.py)
- **sparse_attention_backward:**
  [`cudnn_sm100_dsa_sparse_attention_backward`](tirx_kernels/cudnn/sparse_attention_backward/b200/sparse_attention_backward.py)

### deepep

- **combine:**
  [`deepep_combine`](tirx_kernels/deepep/combine/b200/combine.py) ⟨sm_100a⟩
- **dispatch:**
  [`deepep_dispatch`](tirx_kernels/deepep/dispatch/b200/dispatch.py) ⟨sm_100a⟩

### deepgemm

- **fp8_bmm:**
  [`deepgemm_sm100_fp8_bmm`](tirx_kernels/deepgemm/fp8_bmm/b200/fp8_bmm.py)
- **fp8_fp4_mega_moe:**
  [`sm100_fp8_fp4_mega_moe`](tirx_kernels/deepgemm/fp8_fp4_mega_moe/b200/sm100_fp8_fp4_mega_moe.py)
- **fp8_gemm_1d1d:**
  [`deepgemm_sm100_fp8_gemm_1d1d`](tirx_kernels/deepgemm/fp8_gemm_1d1d/b200/fp8_gemm_1d1d.py)
- **k_grouped_fp8_gemm_contiguous:**
  [`deepgemm_sm100_k_grouped_fp8_gemm_contiguous`](tirx_kernels/deepgemm/k_grouped_fp8_gemm_contiguous/b200/k_grouped_fp8_gemm_contiguous.py)
- **m_grouped_fp8_gemm_contiguous:**
  [`deepgemm_sm100_m_grouped_fp8_gemm_contiguous`](tirx_kernels/deepgemm/m_grouped_fp8_gemm_contiguous/b200/m_grouped_fp8_gemm_contiguous.py)
- **m_grouped_fp8_gemm_masked:**
  [`deepgemm_sm100_m_grouped_fp8_gemm_masked`](tirx_kernels/deepgemm/m_grouped_fp8_gemm_masked/b200/m_grouped_fp8_gemm_masked.py)
- **mqa_logits:**
  [`deepgemm_sm100_fp4_mqa_logits`](tirx_kernels/deepgemm/mqa_logits/b200/mqa_logits_fp4.py),
  [`deepgemm_sm100_fp8_mqa_logits`](tirx_kernels/deepgemm/mqa_logits/b200/mqa_logits_fp8.py)
- **paged_mqa_logits:**
  [`deepgemm_sm100_fp4_paged_mqa_logits`](tirx_kernels/deepgemm/paged_mqa_logits/b200/paged_mqa_logits_fp4.py),
  [`deepgemm_sm100_fp8_paged_mqa_logits`](tirx_kernels/deepgemm/paged_mqa_logits/b200/paged_mqa_logits_fp8.py)
- **tf32_hc_prenorm_gemm:**
  [`deepgemm_sm100_tf32_hc_prenorm_gemm`](tirx_kernels/deepgemm/tf32_hc_prenorm_gemm/b200/tf32_hc_prenorm_gemm.py)

### flashattention

- **attention_backward:**
  [`flash_attention_backward_sm100`](tirx_kernels/flashattention/attention_backward/b200/flash_attention_backward.py) ⟨+sm_110a⟩
- **attention_forward:**
  [`flash_attention4`](tirx_kernels/flashattention/attention_forward/b200/flash_attention4.py) ⟨+sm_110a⟩,
  [`flash_attention4_fp4`](tirx_kernels/flashattention/attention_forward/gb300/flash_attention4_fp4.py) ⟨sm_103a⟩

### flashinfer

- **act_and_mul:**
  [`act_and_mul`](tirx_kernels/flashinfer/act_and_mul/b200/act_and_mul.py) ⟨+sm_110a⟩
- **alphamoe_fp8_blockscale_qwen3next:**
  [`curated_alphamoe_fp8_blockscale_qwen3next`](tirx_kernels/flashinfer/alphamoe_fp8_blockscale_qwen3next/b200/alphamoe_fp8_blockscale_qwen3next.py) ⟨sm_100a⟩
- **blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion:**
  [`blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin`](tirx_kernels/flashinfer/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion/rubin/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin.py) ⟨sm_107a⟩
- **bmm_fp8:**
  [`bmm_fp8_rubin`](tirx_kernels/flashinfer/bmm_fp8/rubin/bmm_fp8_rubin.py) ⟨sm_107a⟩
- **dense_blockscaled_gemm:**
  [`dense_blockscaled_gemm_sm107`](tirx_kernels/flashinfer/dense_blockscaled_gemm/rubin/dense_blockscaled_gemm_sm107.py) ⟨sm_107a⟩
- **fused_add_rmsnorm:**
  [`flashinfer_fused_add_rmsnorm`](tirx_kernels/flashinfer/fused_add_rmsnorm/b200/fused_add_rmsnorm.py) ⟨sm_100a⟩
- **fused_add_rmsnorm_quant:**
  [`flashinfer_add_rmsnorm_fp4quant`](tirx_kernels/flashinfer/fused_add_rmsnorm_quant/b200/add_rmsnorm_fp4quant.py),
  [`flashinfer_fused_add_rmsnorm_quant`](tirx_kernels/flashinfer/fused_add_rmsnorm_quant/b200/fused_add_rmsnorm_quant.py)
- **fused_dit_layernorm:**
  [`flashinfer_fused_dit_layernorm`](tirx_kernels/flashinfer/fused_dit_layernorm/b200/fused_dit_layernorm.py)
- **gdn_cp_prefill:**
  [`gdn_cp_prefill_sm100`](tirx_kernels/flashinfer/gdn_cp_prefill/b200/gdn_cp_prefill_sm100.py)
- **gdn_decode:**
  [`gdn_decode_bf16_ilp4`](tirx_kernels/flashinfer/gdn_decode/b200/gdn_decode_bf16_ilp4.py),
  [`gdn_decode_bf16_wide_vec_mtp`](tirx_kernels/flashinfer/gdn_decode/b200/gdn_decode_bf16_wide_vec_mtp.py),
  [`gdn_decode_bf16_wide_vec_t1`](tirx_kernels/flashinfer/gdn_decode/b200/gdn_decode_bf16_wide_vec_t1.py),
  [`gdn_decode_fp32_mtp_warp`](tirx_kernels/flashinfer/gdn_decode/b200/gdn_decode_fp32_mtp_warp.py)
- **gdn_prefill:**
  [`gdn_prefill_sm100`](tirx_kernels/flashinfer/gdn_prefill/b200/gdn_prefill_sm100.py)
- **grouped_gemm_masked:**
  [`grouped_gemm_masked_rubin`](tirx_kernels/flashinfer/grouped_gemm_masked/rubin/grouped_gemm_masked_rubin.py) ⟨sm_107a⟩
- **kda_decode:**
  [`curated_kda_decode_multishape`](tirx_kernels/flashinfer/kda_decode/b200/kda_decode_multishape.py) ⟨sm_100a⟩,
  [`recurrent_kda_decode_grouped`](tirx_kernels/flashinfer/kda_decode/b200/recurrent_kda_decode_grouped.py),
  [`recurrent_kda_decode_one_warp`](tirx_kernels/flashinfer/kda_decode/b200/recurrent_kda_decode_one_warp.py) ⟨+sm_110a⟩
- **layernorm:**
  [`flashinfer_layernorm`](tirx_kernels/flashinfer/layernorm/b200/layernorm.py) ⟨+sm_110a⟩
- **merge_state:**
  [`merge_state`](tirx_kernels/flashinfer/merge_state/b200/merge_state.py) ⟨+sm_110a⟩
- **mla_dsv4:**
  [`curated_mla_dsv4_multishape`](tirx_kernels/flashinfer/mla_dsv4/b200/mla_dsv4_multishape.py) ⟨sm_100a⟩
- **moe_fp8_blockscale_dsv3:**
  [`curated_moe_fp8_blockscale_dsv3`](tirx_kernels/flashinfer/moe_fp8_blockscale_dsv3/b200/moe_fp8_blockscale_dsv3.py) ⟨sm_100a⟩
- **mxfp4_quantize:**
  [`mxfp4_quantize`](tirx_kernels/flashinfer/mxfp4_quantize/b200/mxfp4_quantize.py) ⟨+sm_110a⟩
- **mxfp8_quantize:**
  [`mxfp8_quantize`](tirx_kernels/flashinfer/mxfp8_quantize/b200/mxfp8_quantize.py)
- **nvfp4_quantize:**
  [`nvfp4_quantize`](tirx_kernels/flashinfer/nvfp4_quantize/b200/nvfp4_quantize.py)
- **nvfp4_quantize_per_token:**
  [`nvfp4_quantize_per_token`](tirx_kernels/flashinfer/nvfp4_quantize_per_token/b200/nvfp4_quantize_per_token.py)
- **qk_rmsnorm:**
  [`flashinfer_qk_rmsnorm`](tirx_kernels/flashinfer/qk_rmsnorm/b200/qk_rmsnorm.py) ⟨+sm_110a⟩
- **rmsnorm_quant:**
  [`flashinfer_rmsnorm_fp4quant`](tirx_kernels/flashinfer/rmsnorm_quant/b200/rmsnorm_fp4quant.py),
  [`flashinfer_rmsnorm_quant`](tirx_kernels/flashinfer/rmsnorm_quant/b200/rmsnorm_quant.py)
- **selective_state_update:**
  [`selective_state_update_mtp_horizontal`](tirx_kernels/flashinfer/selective_state_update/b200/selective_state_update_mtp_horizontal.py) ⟨+sm_110a⟩,
  [`selective_state_update_mtp_simple`](tirx_kernels/flashinfer/selective_state_update/b200/selective_state_update_mtp_simple.py) ⟨+sm_110a⟩,
  [`selective_state_update_mtp_vertical`](tirx_kernels/flashinfer/selective_state_update/b200/selective_state_update_mtp_vertical.py) ⟨+sm_110a⟩,
  [`selective_state_update_stp_horizontal`](tirx_kernels/flashinfer/selective_state_update/b200/selective_state_update_stp_horizontal.py),
  [`selective_state_update_stp_simple`](tirx_kernels/flashinfer/selective_state_update/b200/selective_state_update_stp_simple.py),
  [`selective_state_update_stp_vertical`](tirx_kernels/flashinfer/selective_state_update/b200/selective_state_update_stp_vertical.py)
- **silu_and_mul_nvfp4_experts_quantize:**
  [`silu_and_mul_nvfp4_experts_quantize`](tirx_kernels/flashinfer/silu_and_mul_nvfp4_experts_quantize/b200/silu_and_mul_nvfp4_experts_quantize.py)
- **stable_sort_topk_by_value:**
  [`stable_sort_topk_by_value`](tirx_kernels/flashinfer/stable_sort_topk_by_value/b200/stable_sort_topk_by_value.py)
- **topk:**
  [`fast_topk_clusters`](tirx_kernels/flashinfer/topk/b200/fast_topk_clusters.py) ⟨+sm_110a⟩,
  [`filtered_topk`](tirx_kernels/flashinfer/topk/b200/filtered_topk.py) ⟨+sm_110a⟩,
  [`radix_topk_multi_cta`](tirx_kernels/flashinfer/topk/b200/radix_topk_multi_cta.py),
  [`radix_topk_single_cta`](tirx_kernels/flashinfer/topk/b200/radix_topk_single_cta.py)
- **vsa:**
  [`curated_vsa_multishape`](tirx_kernels/flashinfer/vsa/b200/vsa_multishape.py) ⟨sm_100a⟩

### flashmla

- **sparse_decode:**
  [`sparse_flashmla_decode_head64`](tirx_kernels/flashmla/sparse_decode/b200/sparse_decode_head64.py)
- **sparse_prefill:**
  [`flash_mla_sparse_fwd`](tirx_kernels/flashmla/sparse_prefill/b200/flash_mla_sparse_fwd.py) ⟨sm_100a⟩,
  [`sparse_flashmla_prefill_head128_phase1`](tirx_kernels/flashmla/sparse_prefill/b200/sparse_prefill_head128_phase1.py),
  [`sparse_flashmla_prefill_head128_small_topk_phase1`](tirx_kernels/flashmla/sparse_prefill/b200/sparse_prefill_head128_small_topk_phase1.py),
  [`sparse_flashmla_prefill_head64_phase1`](tirx_kernels/flashmla/sparse_prefill/b200/sparse_prefill_head64_phase1.py)

### msa

- **decode:**
  [`curated_msa_decode_multishape`](tirx_kernels/msa/decode/b200/msa_decode_multishape.py) ⟨sm_100a⟩
- **prefill:**
  [`curated_msa_prefill_multishape`](tirx_kernels/msa/prefill/b200/msa_prefill_multishape.py) ⟨sm_100a⟩
- **sparse_atten_fwd:**
  [`msa_sparse_atten_fwd_nvfp4_kv_sm100`](tirx_kernels/msa/sparse_atten_fwd/b200/sparse_atten_fwd_nvfp4_kv.py),
  [`msa_sparse_atten_fwd_sm100`](tirx_kernels/msa/sparse_atten_fwd/b200/sparse_atten_fwd.py)
- **sparse_atten_fwd_combine:**
  [`msa_sparse_atten_fwd_combine_sm100`](tirx_kernels/msa/sparse_atten_fwd_combine/b200/sparse_atten_fwd_combine.py)
- **sparse_prepare_flat_schedule:**
  [`msa_sparse_prepare_flat_schedule_sm100`](tirx_kernels/msa/sparse_prepare_flat_schedule/b200/sparse_prepare_flat_schedule.py) ⟨+sm_110a⟩
- **sparse_prepare_fwd_split_atomic:**
  [`msa_sparse_prepare_fwd_split_atomic_sm100`](tirx_kernels/msa/sparse_prepare_fwd_split_atomic/b200/sparse_prepare_fwd_split_atomic.py)

## Performance

Per-workload numbers — our kernel time, every reference impl, and the
ref/ours ratio (>1 means ours is faster) — are pinned in
[`tirx_kernels/bench/baseline.md`](tirx_kernels/bench/baseline.md),
regenerated on every baseline promotion. See the
[bench-suite README](tirx_kernels/bench/README.md) for how the sweep runs
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
| `flashinfer`     | all `flashinfer.*` ports, `nvfp4_gemm` and `rmsnorm` baselines | Correctness reference and optimized baseline. |
| `flash-attn` + CUTLASS DSL | `flash_attention_backward_sm100` baseline | Current SM100 forward/backward reference. |
| SGLang CuTeDSL kernels (vendored, + CUTLASS DSL) | `deepgemm_sm100_fp8_paged_mqa_logits` reference | `sglang_cutedsl` benchmark reference; copied into `tirx_kernels/deepgemm/_shared/_sglang_cutedsl/`, no SGLang install needed. |
| `flash_mla`      | `sparse_flashmla_*` / `flash_mla_sparse_fwd` baselines | Reference impls. |
| `deep_ep`        | `deepep_*` correctness and baselines | Reference implementation. |
| `flash-linear-attention` | the `curated_kda_forward_*` kernels and `curated_kda_backward_packed` correctness | Independent FLA BF16/Triton chunk reference. |
| `flash_kda`      | `flashkda_*` and `curated_kda_forward_*` optional baselines | Raw FlashKDA benchmark peer. |
| `fmha_sm100` (MSA) | `msa_*` correctness and baselines | Reference implementation; set `MSA_PATH` to use a checkout elsewhere. |
| NVSHMEM          | `allgather_gemm`, `gemm_reduce_scatter` | Required to compile/run the GemmComm kernels. |

Correctness tests import and run these upstream implementations. The bench suite
does not launch or time benchmark reference implementations by default (kernel
data-preparation helpers may still import their upstream package). Pass
`--with-references` to enable reference launches; a missing enabled reference
fails its workload. See
[`tirx_kernels/bench/README.md`](tirx_kernels/bench/README.md)
for the prerequisites and workarounds.

## Usage

### Command line

```bash
# List discovered kernels (with their config labels)
python -m tirx_kernels.bench.registry --format json

# Run correctness tests (optionally filter by kernel / config label)
pytest -n 16 tests/test_correctness.py

# Benchmark
python -m tirx_kernels.bench run --kernel nvfp4_gemm
python -m tirx_kernels.bench run --kernel nvfp4_gemm --with-references

# Pre-commit regression benchmark sweep on the kcoral benchmark server
# (see tirx_kernels/bench/README.md; needs `pip install -e '.[remote]'`)
python -m tirx_kernels.bench suite --server http://127.0.0.1:8901
```

### Programmatic API

Every kernel module exposes a small, uniform interface (see
`tirx_kernels/bench/_protocol.py`):

```python
from tirx_kernels.bench.registry import discover_kernels

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
(cuDNN Frontend, DeepGEMM, DeepEP, fast.cu, FlashMLA, flash-attention, flash-attention-fp4, FlashInfer, MSA) additionally cite the upstream
project and the exact commit ported, retain the upstream copyright notice, and
declare the combined terms — for example `Apache-2.0 AND MIT`. Where an upstream
license requires its conditions text to travel with the source, that text is kept
in the file verbatim. The third-party section at the end of [LICENSE](LICENSE)
lists which components fall under which license, and [`licenses/`](licenses)
holds the corresponding license texts.
