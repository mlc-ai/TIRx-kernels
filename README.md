# TIRx kernels

High-performance GPU kernels authored in
[tirx-lite](tirx_kernels/tirx_lite/README.md) and compiled through
[TIRx](https://github.com/apache/tvm).

## Kernels

Native kernels are grouped by workload under `tirx_kernels/<workload>/`.
External ports live under `tirx_kernels/ported/<upstream>/`, with subpackages
for upstream entry points. Unless annotated, a kernel runs on `sm_100a`,
`sm_103a` and `sm_107a`, and its performance was measured and tuned on
`sm_100a`; every listed architecture is gated by the registry correctness tests.
Annotations mark the exceptions:
⟨sm_103a⟩ runs only on that architecture and was tuned there, and ⟨+sm_110a⟩
runs on that architecture in addition to the default three. Annotations follow
`KERNEL_META["runtime_cuda_archs"]` and `tests/lint/check_readme_kernels.py`
keeps them in sync.

### Native TIRx

- **GEMM:**
  [`fp16_bf16_gemm`](tirx_kernels/gemm/fp16_bf16_gemm.py) ⟨+sm_110a⟩,
  [`nvfp4_gemm`](tirx_kernels/gemm/nvfp4_gemm.py)
- **Normalization:**
  [`rmsnorm`](tirx_kernels/norm/rmsnorm.py)
- **Distributed:**
  [`allgather_gemm`](tirx_kernels/gemm/allgather_gemm.py) ⟨sm_100a⟩,
  [`gemm_reduce_scatter`](tirx_kernels/gemm/gemm_reduce_scatter.py) ⟨sm_100a⟩
- **KDA forward:**
  [`kda_forward_portfolio_multishape`](tirx_kernels/kda/kda_forward_portfolio_multishape.py) ⟨sm_100a⟩
- **Qwen3-Next TP4 Alpha-MoE FP8:**
  [`alphamoe_fp8_blockscale_qwen3next`](tirx_kernels/moe/alphamoe_fp8_blockscale_qwen3next.py) ⟨sm_100a⟩
- **KDA backward:**
  [`kda_backward_packed`](tirx_kernels/kda/kda_backward_packed.py) ⟨sm_100a⟩
- **KDA decode:**
  [`kda_decode_multishape`](tirx_kernels/kda/kda_decode_multishape.py) ⟨sm_100a⟩
- **MSA prefill:**
  [`msa_prefill_multishape`](tirx_kernels/msa/msa_prefill_multishape.py) ⟨sm_100a⟩
- **MSA decode:**
  [`msa_decode_multishape`](tirx_kernels/msa/msa_decode_multishape.py) ⟨sm_100a⟩
- **VSA forward:**
  [`vsa_multishape`](tirx_kernels/vsa/vsa_multishape.py) ⟨sm_100a⟩
- **DeepSeek-V4 sparse MLA:**
  [`mla_dsv4_multishape`](tirx_kernels/mla/mla_dsv4_multishape.py) ⟨sm_100a⟩

### cuDNN Frontend ports

- **Persistent GEMM:**
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_amax`](tirx_kernels/ported/cudnn/amax/dense_blockscaled_gemm_persistent_amax.py),
  [`cudnn_sm100_dense_gemm_persistent_swiglu`](tirx_kernels/ported/cudnn/swiglu/dense_gemm_persistent_swiglu.py),
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant`](tirx_kernels/ported/cudnn/swiglu/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py),
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant`](tirx_kernels/ported/cudnn/srelu/dense_blockscaled_gemm_persistent_srelu_quant.py),
  [`cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant`](tirx_kernels/ported/cudnn/dsrelu/dense_blockscaled_gemm_persistent_dsrelu_quant.py),
  [`cudnn_sm100_gemm_proj_rope_mxfp8_bf16in`](tirx_kernels/ported/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_bf16in.py),
  [`cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in`](tirx_kernels/ported/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_mxfp8in.py)
- **Grouped GEMM:**
  [`cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias`](tirx_kernels/ported/cudnn/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py),
  [`cudnn_sm100_moe_grouped_gemm_dglu_dbias`](tirx_kernels/ported/cudnn/dglu/moe_grouped_gemm_dglu_dbias.py)
- **Linear attention:**
  [`cudnn_sm100_kda_bprop_f16`](tirx_kernels/ported/cudnn/linear_attention/kda_bprop_f16.py) ⟨+sm_110a⟩,
  [`cudnn_sm100_gdn_prefill_f16`](tirx_kernels/ported/cudnn/linear_attention/gdn_prefill_f16.py) ⟨+sm_110a⟩,
  [`cudnn_sm100_gdn_recompute_f16`](tirx_kernels/ported/cudnn/linear_attention/gdn_recompute_f16.py) ⟨+sm_110a⟩,
  [`cudnn_sm100_gdn2_prefill_f16`](tirx_kernels/ported/cudnn/linear_attention/gdn2_prefill_f16.py) ⟨+sm_110a⟩,
  [`cudnn_sm100_gdn2_recompute_f16`](tirx_kernels/ported/cudnn/linear_attention/gdn2_recompute_f16.py) ⟨+sm_110a⟩,
  [`cudnn_sm100_gdn2_bprop_f16`](tirx_kernels/ported/cudnn/linear_attention/gdn2_bprop_f16.py),
  [`cudnn_sm100_gdn_bprop_f16`](tirx_kernels/ported/cudnn/linear_attention/gdn_bprop_f16.py)
- **CSA compression:**
  [`cudnn_sm100_csa_compressor_fwd`](tirx_kernels/ported/cudnn/csa/compressor_fwd_sm100.py)
- **Flex attention:**
  [`cudnn_sm100_flex_attention_forward_hd256`](tirx_kernels/ported/cudnn/flex_attention/forward_hd256_sm100.py) ⟨sm_100a sm_103a⟩,
  [`cudnn_sm103_flex_attention_forward`](tirx_kernels/ported/cudnn/flex_attention/forward_sm103.py) ⟨sm_103a⟩,
  [`cudnn_sm100_flex_attention_backward`](tirx_kernels/ported/cudnn/flex_attention/flex_attention_backward_sm100.py) ⟨sm_100a⟩
- **Sparse attention:**
  [`cudnn_sm100_dsa_sparse_attention_backward`](tirx_kernels/ported/cudnn/dsa/sparse_attention_backward.py),
  [`cudnn_sm100_bsa_forward_blk128`](tirx_kernels/ported/cudnn/bsa/block_sparse_attention_forward_sm100_blk128.py),
  [`cudnn_sm100_bsa_forward_blk64`](tirx_kernels/ported/cudnn/bsa/block_sparse_attention_forward_sm100_blk64.py),
  [`cudnn_sm100_bsa_forward_combine_blk64`](tirx_kernels/ported/cudnn/bsa/block_sparse_attention_forward_combine_sm100_blk64.py),
  [`cudnn_sm100_bsa_backward_blk128`](tirx_kernels/ported/cudnn/bsa/block_sparse_attention_backward_sm100_blk128.py),
  [`cudnn_sm100_bsa_backward_blk64`](tirx_kernels/ported/cudnn/bsa/block_sparse_attention_backward_sm100_blk64.py)

### FlashAttention ports

- **Forward:**
  [`flash_attention4`](tirx_kernels/ported/flashattention/flash_attention4.py) ⟨+sm_110a⟩,
  [`flash_attention4_fp4`](tirx_kernels/ported/flashattention/flash_attention4_fp4.py) ⟨sm_103a⟩
- **Backward:**
  [`flash_attention_backward_sm100`](tirx_kernels/ported/flashattention/flash_attention_backward.py) ⟨+sm_110a⟩

### FlashInfer ports

Grouped by the FlashInfer Python entry point each port backs.

- **`flashinfer.activation`:**
  [`act_and_mul`](tirx_kernels/ported/flashinfer/activation/act_and_mul.py) ⟨+sm_110a⟩,
  [`silu_and_mul_nvfp4_experts_quantize`](tirx_kernels/ported/flashinfer/activation/silu_and_mul_nvfp4_experts_quantize.py)
- **`flashinfer.cascade`:**
  [`merge_state`](tirx_kernels/ported/flashinfer/cascade/merge_state.py) ⟨+sm_110a⟩
- **`flashinfer.quantization`:**
  [`nvfp4_quantize`](tirx_kernels/ported/flashinfer/quantization/nvfp4_quantize.py),
  [`nvfp4_quantize_per_token`](tirx_kernels/ported/flashinfer/quantization/nvfp4_quantize_per_token.py),
  [`mxfp4_quantize`](tirx_kernels/ported/flashinfer/quantization/mxfp4_quantize.py) ⟨+sm_110a⟩,
  [`mxfp8_quantize`](tirx_kernels/ported/flashinfer/quantization/mxfp8_quantize.py)
- **`flashinfer.norm`:**
  [`flashinfer_rmsnorm`](tirx_kernels/ported/flashinfer/norm/rmsnorm.py) ⟨+sm_110a⟩,
  [`flashinfer_rmsnorm_quant`](tirx_kernels/ported/flashinfer/norm/rmsnorm_quant.py),
  [`flashinfer_rmsnorm_fp4quant`](tirx_kernels/ported/flashinfer/norm/rmsnorm_fp4quant.py),
  [`flashinfer_add_rmsnorm_fp4quant`](tirx_kernels/ported/flashinfer/norm/add_rmsnorm_fp4quant.py),
  [`flashinfer_layernorm`](tirx_kernels/ported/flashinfer/norm/layernorm.py) ⟨+sm_110a⟩,
  [`flashinfer_fused_add_rmsnorm`](tirx_kernels/ported/flashinfer/norm/fused_add_rmsnorm.py) ⟨sm_100a⟩,
  [`flashinfer_fused_add_rmsnorm_quant`](tirx_kernels/ported/flashinfer/norm/fused_add_rmsnorm_quant.py),
  [`flashinfer_fused_dit_layernorm`](tirx_kernels/ported/flashinfer/norm/fused_dit_layernorm.py),
  [`flashinfer_qk_rmsnorm`](tirx_kernels/ported/flashinfer/norm/qk_rmsnorm.py) ⟨+sm_110a⟩
- **`flashinfer.mamba`:**
  [`selective_state_update_stp_simple`](tirx_kernels/ported/flashinfer/mamba/selective_state_update_stp_simple.py),
  [`selective_state_update_stp_vertical`](tirx_kernels/ported/flashinfer/mamba/selective_state_update_stp_vertical.py),
  [`selective_state_update_stp_horizontal`](tirx_kernels/ported/flashinfer/mamba/selective_state_update_stp_horizontal.py),
  [`selective_state_update_mtp_simple`](tirx_kernels/ported/flashinfer/mamba/selective_state_update_mtp_simple.py) ⟨+sm_110a⟩,
  [`selective_state_update_mtp_vertical`](tirx_kernels/ported/flashinfer/mamba/selective_state_update_mtp_vertical.py) ⟨+sm_110a⟩,
  [`selective_state_update_mtp_horizontal`](tirx_kernels/ported/flashinfer/mamba/selective_state_update_mtp_horizontal.py) ⟨+sm_110a⟩
- **`flashinfer.kda`:**
  [`recurrent_kda_decode_one_warp`](tirx_kernels/ported/flashinfer/kda/recurrent_kda_decode_one_warp.py) ⟨+sm_110a⟩,
  [`recurrent_kda_decode_grouped`](tirx_kernels/ported/flashinfer/kda/recurrent_kda_decode_grouped.py)
- **`flashinfer.gdn_decode`:**
  [`gdn_decode_bf16_ilp4`](tirx_kernels/ported/flashinfer/gdn_decode/gdn_decode_bf16_ilp4.py),
  [`gdn_decode_bf16_wide_vec_t1`](tirx_kernels/ported/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_t1.py),
  [`gdn_decode_bf16_wide_vec_mtp`](tirx_kernels/ported/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_mtp.py),
  [`gdn_decode_fp32_mtp_warp`](tirx_kernels/ported/flashinfer/gdn_decode/gdn_decode_fp32_mtp_warp.py)
- **`flashinfer.gdn_prefill`:**
  [`gdn_prefill_sm100`](tirx_kernels/ported/flashinfer/gdn_prefill/gdn_prefill_sm100.py),
  [`gdn_cp_prefill_sm100`](tirx_kernels/ported/flashinfer/gdn_prefill/gdn_cp_prefill_sm100.py)
- **`flashinfer.gemm`:**
  [`bmm_fp8_rubin`](tirx_kernels/ported/flashinfer/gemm/bmm_fp8_rubin.py) ⟨sm_107a⟩,
  [`dense_blockscaled_gemm_sm107`](tirx_kernels/ported/flashinfer/gemm/dense_blockscaled_gemm_sm107.py) ⟨sm_107a⟩,
  [`grouped_gemm_masked_rubin`](tirx_kernels/ported/flashinfer/gemm/grouped_gemm_masked_rubin.py) ⟨sm_107a⟩
- **`flashinfer.fused_moe`:**
  [`blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin`](tirx_kernels/ported/flashinfer/fused_moe/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin.py) ⟨sm_107a⟩
- **`flashinfer.topk`:**
  [`fast_topk_clusters`](tirx_kernels/ported/flashinfer/topk/fast_topk_clusters.py) ⟨+sm_110a⟩,
  [`filtered_topk`](tirx_kernels/ported/flashinfer/topk/filtered_topk.py) ⟨+sm_110a⟩,
  [`radix_topk_multi_cta`](tirx_kernels/ported/flashinfer/topk/radix_topk_multi_cta.py),
  [`radix_topk_single_cta`](tirx_kernels/ported/flashinfer/topk/radix_topk_single_cta.py),
  [`stable_sort_topk_by_value`](tirx_kernels/ported/flashinfer/topk/stable_sort_topk_by_value.py)

### FlashMLA ports

- **Sparse prefill:**
  [`sparse_flashmla_prefill_head64_phase1`](tirx_kernels/ported/flashmla/sparse_prefill_head64_phase1.py),
  [`sparse_flashmla_prefill_head128_phase1`](tirx_kernels/ported/flashmla/sparse_prefill_head128_phase1.py),
  [`sparse_flashmla_prefill_head128_small_topk_phase1`](tirx_kernels/ported/flashmla/sparse_prefill_head128_small_topk_phase1.py)
- **Sparse decode:**
  [`sparse_flashmla_decode_head64`](tirx_kernels/ported/flashmla/sparse_decode_head64.py)
- **Sparse forward:**
  [`flash_mla_sparse_fwd`](tirx_kernels/ported/flashmla/flash_mla_sparse_fwd.py) ⟨sm_100a⟩

### DeepGEMM ports

- **Dense and grouped GEMM:**
  [`deepgemm_sm100_fp8_gemm_1d1d`](tirx_kernels/ported/deepgemm/fp8_gemm_1d1d.py),
  [`deepgemm_sm100_m_grouped_fp8_gemm_contiguous`](tirx_kernels/ported/deepgemm/m_grouped_fp8_gemm_contiguous.py),
  [`deepgemm_sm100_m_grouped_fp8_gemm_masked`](tirx_kernels/ported/deepgemm/m_grouped_fp8_gemm_masked.py),
  [`deepgemm_sm100_k_grouped_fp8_gemm_contiguous`](tirx_kernels/ported/deepgemm/k_grouped_fp8_gemm_contiguous.py),
  [`deepgemm_sm100_fp8_bmm`](tirx_kernels/ported/deepgemm/fp8_bmm.py),
  [`deepgemm_sm100_tf32_hc_prenorm_gemm`](tirx_kernels/ported/deepgemm/tf32_hc_prenorm_gemm.py)
- **MQA logits:**
  [`deepgemm_sm100_fp4_mqa_logits`](tirx_kernels/ported/deepgemm/mqa_logits_fp4.py),
  [`deepgemm_sm100_fp8_mqa_logits`](tirx_kernels/ported/deepgemm/mqa_logits_fp8.py),
  [`deepgemm_sm100_fp4_paged_mqa_logits`](tirx_kernels/ported/deepgemm/paged_mqa_logits_fp4.py),
  [`deepgemm_sm100_fp8_paged_mqa_logits`](tirx_kernels/ported/deepgemm/paged_mqa_logits_fp8.py)
- **MoE:**
  [`sm100_fp8_fp4_mega_moe`](tirx_kernels/ported/deepgemm/sm100_fp8_fp4_mega_moe.py)

### DeepEP ports

- **Elastic communication:**
  [`deepep_dispatch`](tirx_kernels/ported/deepep/dispatch.py) ⟨sm_100a⟩,
  [`deepep_combine`](tirx_kernels/ported/deepep/combine.py) ⟨sm_100a⟩

### fast.cu ports

- **NVFP4 GEMM:**
  [`fastcu_nvfp4_gemm_gb300`](tirx_kernels/ported/fastcu/nvfp4_gemm_gb300.py) ⟨sm_103a⟩

### MSA ports

- **Sparse-attention preparation:**
  [`msa_sparse_prepare_flat_schedule_sm100`](tirx_kernels/ported/msa/sparse_prepare_flat_schedule.py) ⟨+sm_110a⟩,
  [`msa_sparse_prepare_fwd_split_atomic_sm100`](tirx_kernels/ported/msa/sparse_prepare_fwd_split_atomic.py)
- **Sparse-attention forward:**
  [`msa_sparse_atten_fwd_sm100`](tirx_kernels/ported/msa/sparse_atten_fwd.py),
  [`msa_sparse_atten_fwd_nvfp4_kv_sm100`](tirx_kernels/ported/msa/sparse_atten_fwd_nvfp4_kv.py),
  [`msa_sparse_atten_fwd_combine_sm100`](tirx_kernels/ported/msa/sparse_atten_fwd_combine.py)

## Performance

Per-workload numbers — our kernel time, every reference impl, and the
ref/ours ratio (>1 means ours is faster) — are pinned in
[`tirx_kernels/bench_suite/baseline.md`](tirx_kernels/bench_suite/baseline.md),
regenerated on every baseline promotion. See the
[bench-suite README](tirx_kernels/bench_suite/README.md) for how the sweep runs
and how to refresh the baseline.

## Installation

```bash
pip install tirx-kernels
# or, from a checkout:
pip install -e .
```

The kcoral client for remote benchmarking is installed automatically.

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
| SGLang CuTeDSL kernels (vendored, + CUTLASS DSL) | `deepgemm_sm100_fp8_paged_mqa_logits` reference | `sglang_cutedsl` benchmark reference; copied into `tirx_kernels/ported/deepgemm/_sglang_cutedsl/`, no SGLang install needed. |
| `flash_mla`      | `sparse_flashmla_*` / `flash_mla_sparse_fwd` baselines | Reference impls. |
| `deep_ep`        | `deepep_*` correctness and baselines | Reference implementation. |
| `flash-linear-attention` | `kda_forward_portfolio_multishape` and `kda_backward_packed` correctness | Independent FLA BF16/Triton chunk reference. |
| `flash_kda`      | `flashkda_*` and `kda_forward_portfolio_multishape` optional baselines | Raw FlashKDA benchmark peer. |
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

# Pre-commit regression benchmark sweep on the kcoral benchmark server
# (see tirx_kernels/bench_suite/README.md)
python -m tirx_kernels.bench_suite --server http://127.0.0.1:8901
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

A native module uses its workload as `KERNEL_META["category"]` (e.g. `kda`);
a port uses `ported.<upstream>` (e.g. `ported.flashinfer`). Registry category
filters use those same names. CLI kernel names are independent of module paths.

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
