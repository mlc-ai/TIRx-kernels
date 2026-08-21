# TIRx kernels

High-performance GPU kernels written in [TIRx](https://github.com/apache/tvm).

## Kernels

All kernels target `sm_100a`. **Kernel** is the registry name accepted by the
`--kernel` CLI filters; **Module** is its source file, relative to the bucket
directory under `tirx_kernels/`. Each bucket holds the kernels ported from one
upstream project.

`basic/` — native TIRx kernels, with no single upstream project:

| Kernel                | Module                   | What it is |
| --------------------- | ------------------------ | ---------- |
| `fp16_bf16_gemm`      | `fp16_bf16_gemm.py`      | Dense GEMM |
| `nvfp4_gemm`          | `nvfp4_gemm.py`          | Dense GEMM |
| `rmsnorm`             | `rmsnorm.py`             | RMSNorm |
| `allgather_gemm`      | `allgather_gemm.py`      | AllGather + GEMM (multi-GPU, NVSHMEM) |
| `gemm_reduce_scatter` | `gemm_reduce_scatter.py` | GEMM + ReduceScatter (multi-GPU, NVSHMEM) |

`flashattention/` — Dao-AILab flash-attention ports:

| Kernel                           | Module                        | What it is |
| -------------------------------- | ----------------------------- | ---------- |
| `flash_attention4`               | `flash_attention4.py`         | FlashAttention-4 |
| `flash_attention_backward_sm100` | `flash_attention_backward.py` | Two-CTA FlashAttention backward (D=128); [schedule sketch](tirx_kernels/flashattention/flash_attention_backward_sm100_sketch.md) |

`flashinfer/` — FlashInfer ports, sub-bucketed by the FlashInfer Python entry
point each port backs (`activation`, `quantization`, `norm` — CuTe-DSL backend —,
`mamba`, `kda`, `gdn_decode`, `gdn_prefill`, `gemm`):

| Kernel                                  | Module                                              | What it is |
| --------------------------------------- | --------------------------------------------------- | ---------- |
| `act_and_mul`                           | `activation/act_and_mul.py`                         | `silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul` (one templated kernel) |
| `silu_and_mul_nvfp4_experts_quantize`   | `activation/silu_and_mul_nvfp4_experts_quantize.py` | SiLU*mul fused with per-expert NVFP4 quantization |
| `nvfp4_quantize`                        | `quantization/nvfp4_quantize.py`                    | Block quantization, linear and swizzled SF layouts; also the `silu_and_mul` variant |
| `nvfp4_quantize_per_token`              | `quantization/nvfp4_quantize_per_token.py`          | Per-token-activation quantization |
| `mxfp4_quantize`                        | `quantization/mxfp4_quantize.py`                    | Block quantization with UE8M0 scales |
| `mxfp8_quantize`                        | `quantization/mxfp8_quantize.py`                    | Block quantization with UE8M0 scales |
| `flashinfer_rmsnorm`                    | `norm/rmsnorm.py`                                   | Shared 2-D RMSNorm / Gemma RMSNorm family with compact, int64-strided, PDL, async-copy, and cluster-reduction paths |
| `flashinfer_rmsnorm_fp4quant`           | `norm/rmsnorm_fp4quant.py`                          | RMSNorm fused with packed E2M1 FP4 quantization, E4M3 or UE8M0 block scales, optional scale swizzling, PDL, and cluster reduction |
| `flashinfer_add_rmsnorm_fp4quant`       | `norm/add_rmsnorm_fp4quant.py`                      | Residual add and RMSNorm fused with packed E2M1 FP4 quantization, optional dual scale layouts and normalized output, PDL, and cluster reduction |
| `flashinfer_layernorm`                  | `norm/layernorm.py`                                 | BF16 LayerNorm with FP32 affine parameters, independent int64 row strides, and optional PDL |
| `flashinfer_fused_add_rmsnorm`          | `norm/fused_add_rmsnorm.py`                         | Shared fused residual-add RMSNorm / Gemma family with compact, int64-strided, PDL, async-copy, and cluster-reduction paths |
| `flashinfer_fused_add_rmsnorm_quant`    | `norm/fused_add_rmsnorm_quant.py`                   | Fused residual add, RMSNorm, and FP8 quantization with compact, int64-strided, PDL, async-copy, and cluster-reduction paths |
| `flashinfer_fused_dit_layernorm`        | `norm/fused_dit_layernorm.py`                       | WAN DIT fused gate/residual LayerNorm with gamma/beta or scale/shift epilogues and BF16, NVFP4, or MXFP8 output |
| `flashinfer_qk_rmsnorm`                 | `norm/qk_rmsnorm.py`                                | Shared 3-D QK RMSNorm / Gemma RMSNorm family with arbitrary int64 batch/head strides, PDL, and sync/async copy paths |
| `selective_state_update_stp_simple`     | `mamba/selective_state_update_stp_simple.py`        | Single-token, `algorithm="simple"` |
| `selective_state_update_stp_vertical`   | `mamba/selective_state_update_stp_vertical.py`      | Single-token, `algorithm="vertical"` |
| `selective_state_update_stp_horizontal` | `mamba/selective_state_update_stp_horizontal.py`    | Single-token, `algorithm="horizontal"` |
| `selective_state_update_mtp_simple`     | `mamba/selective_state_update_mtp_simple.py`        | Multi-token, `algorithm="simple"` |
| `selective_state_update_mtp_vertical`   | `mamba/selective_state_update_mtp_vertical.py`      | Multi-token, `algorithm="vertical"` |
| `selective_state_update_mtp_horizontal` | `mamba/selective_state_update_mtp_horizontal.py`    | Multi-token, `algorithm="horizontal"` |
| `flashkda_bf16_fused_m128`              | `kda/bf16_fused_m128.py`                            | Recurrent KDA prefill, M128 schedule |
| `recurrent_kda_decode_one_warp`         | `kda/recurrent_kda_decode_one_warp.py`              | Recurrent KDA decode, one warp per CTA (T=1, `sequence_heads >= 128`) |
| `recurrent_kda_decode_grouped`          | `kda/recurrent_kda_decode_grouped.py`               | Recurrent KDA decode, grouped CTA (small-batch T=1 and all speculative T>1) |
| `flashkda_decode_t1_precomputed`        | `kda/flashkda_decode_t1_precomputed.py`             | FlashKDA "cake" decode, T=1 precomputed gate |
| `flashkda_decode_t2_precomputed`        | `kda/flashkda_decode_t2_precomputed.py`             | FlashKDA "cake" decode, T=2 precomputed gate (WY, tensor cores) |
| `flashkda_decode_t3_lower_bound`        | `kda/flashkda_decode_t3_lower_bound.py`             | FlashKDA "cake" decode, T=3 lower-bound gate computed in-kernel |
| `flashkda_decode_t4_precomputed`        | `kda/flashkda_decode_t4_precomputed.py`             | FlashKDA "cake" decode, T=4 precomputed gate (WY, tensor cores) |
| `flashkda_decode_t5_gram`               | `kda/flashkda_decode_t5_gram.py`                    | FlashKDA "cake" decode, T=5 coefficient-Gram gate (tensor-core WY coefficients, all four value splits) |
| `flashkda_decode_t6_gram`               | `kda/flashkda_decode_t6_gram.py`                    | FlashKDA "cake" decode, T=6 coefficient-Gram gate (dynamic shared memory, all four value splits) |
| `gdn_decode_bf16_ilp4`                  | `gdn_decode/gdn_decode_bf16_ilp4.py`                | Gated Delta Net decode, bf16 state, ILP4 fallback schedule (T=1/2/4/8) |
| `gdn_decode_bf16_wide_vec_t1`           | `gdn_decode/gdn_decode_bf16_wide_vec_t1.py`         | Gated Delta Net decode, bf16 state, wide-vector single-token (T=1) |
| `gdn_decode_bf16_wide_vec_mtp`          | `gdn_decode/gdn_decode_bf16_wide_vec_mtp.py`        | Gated Delta Net decode, bf16 state, wide-vector multi-token (T=2/4/8) |
| `gdn_decode_fp32_mtp_warp`              | `gdn_decode/gdn_decode_fp32_mtp_warp.py`            | Gated Delta Net decode, fp32 state, multi-token warp schedule |
| `gdn_prefill_sm100`                     | `gdn_prefill/gdn_prefill_sm100.py`                  | Gated Delta Net prefill |
| `gdn_cp_prefill_sm100`                  | `gdn_prefill/gdn_cp_prefill_sm100.py`               | Gated Delta Net chunk-parallel prefill (T precompute, M/N precompute, chunk-state fixup, CP prefill) |
| `tinygemm2_sm100`                       | `gemm/tinygemm2_sm100.py`                           | TinyGEMM2 |

`flashmla/` — FlashMLA sparse attention ports:

| Kernel                                              | Module                                        | What it is |
| --------------------------------------------------- | --------------------------------------------- | ---------- |
| `sparse_flashmla_prefill_head64_phase1`             | `sparse_prefill_head64_phase1.py`             | Sparse prefill, 64 q-heads (phase 1) |
| `sparse_flashmla_prefill_head128_phase1`            | `sparse_prefill_head128_phase1.py`            | Sparse prefill, 128 q-heads (phase 1) |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | `sparse_prefill_head128_small_topk_phase1.py` | Sparse prefill, 128 q-heads, small top-k (phase 1) |
| `sparse_flashmla_decode_head64`                     | `sparse_decode_head64.py`                     | Sparse decode, 64 q-heads (main + combine launch) |
| `flash_mla_sparse_fwd`                              | `flash_mla_sparse_fwd.py`                     | Sparse forward |

`deepgemm/` — DeepGEMM ports:

| Kernel                                         | Module                             | What it is |
| ---------------------------------------------- | ---------------------------------- | ---------- |
| `deepgemm_sm100_fp8_gemm_1d1d`                 | `fp8_gemm_1d1d.py`                 | Dense blockwise-scaled GEMM |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | `m_grouped_fp8_gemm_contiguous.py` | M-grouped contiguous GEMM (incl. psum layout) |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked`     | `m_grouped_fp8_gemm_masked.py`     | M-grouped masked GEMM |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | `k_grouped_fp8_gemm_contiguous.py` | K-grouped contiguous GEMM (wgrad, incl. psum layout) |
| `deepgemm_sm100_fp8_bmm`                       | `fp8_bmm.py`                       | Batched GEMM behind `fp8_einsum` |
| `deepgemm_sm100_fp4_mqa_logits`                | `mqa_logits_fp4.py`                | MQA attention logits |
| `deepgemm_sm100_fp8_mqa_logits`                | `mqa_logits_fp8.py`                | MQA attention logits |
| `deepgemm_sm100_fp4_paged_mqa_logits`          | `paged_mqa_logits_fp4.py`          | Paged-KV MQA attention logits |
| `deepgemm_sm100_fp8_paged_mqa_logits`          | `paged_mqa_logits_fp8.py`          | Paged-KV MQA attention logits |
| `deepgemm_sm100_tf32_hc_prenorm_gemm`          | `tf32_hc_prenorm_gemm.py`          | Prenorm GEMM |
| `sm100_fp8_fp4_mega_moe`                       | `sm100_fp8_fp4_mega_moe.py`        | Fused MoE megakernel (MegaMoE) |

`deepep/` — DeepEP ports:

| Kernel            | Module        | What it is |
| ----------------- | ------------- | ---------- |
| `deepep_dispatch` | `dispatch.py` | V2 elastic dispatch, single-domain NVLink path (multi-GPU, 8 ranks) |
| `deepep_combine` | `combine.py` | V2 elastic combine, single-domain NVLink path (multi-GPU, 8 ranks) |

`msa/` — MSA sparse attention ports:

| Kernel                                     | Module                              | What it is |
| ------------------------------------------ | ----------------------------------- | ---------- |
| `msa_sparse_prepare_flat_schedule_sm100`   | `sparse_prepare_flat_schedule.py`   | Flat work-list preparation for sparse attention |
| `msa_sparse_prepare_fwd_split_atomic_sm100` | `sparse_prepare_fwd_split_atomic.py` | Forward split-slot preparation (packed `q_idx`/slot metadata) |
| `msa_sparse_atten_fwd_sm100`               | `sparse_atten_fwd.py`               | CSR block-sparse attention forward, split-partial output |
| `msa_sparse_atten_fwd_nvfp4_kv_sm100`      | `sparse_atten_fwd_nvfp4_kv.py`      | CSR block-sparse attention forward over NVFP4 K/V |

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
| `flashinfer`     | `nvfp4_gemm` baseline | Used for reference implementations. |
| `flash-attn` + CUTLASS DSL | `flash_attention_backward_sm100` baseline | Current SM100 forward/backward reference. |
| `sglang` (+ CUTLASS DSL) | `deepgemm_sm100_fp8_paged_mqa_logits` reference | Optional `sglang_cutedsl` benchmark reference. |
| `flash_mla`      | `sparse_flashmla_*` / `flash_mla_sparse_fwd` baselines | Reference impls. |
| `deep_ep`        | `deepep_*` correctness and baselines | Reference implementation. |
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

Each module also provides `KERNEL_META` (name / category / `compute_capability`)
and `CONFIGS` (the test/bench parameter sweeps) that the registry and CLI use.

## License

Except where otherwise noted, this project is licensed under the Apache
License 2.0; see [LICENSE](LICENSE). Required Apache attribution notices are
collected in [NOTICE](NOTICE).

Every Python source file carries SPDX tags. Kernel ports derived from third-party projects
(DeepGEMM, FlashMLA, flash-attention, FlashInfer, MSA) additionally cite the upstream
project and the exact commit ported, retain the upstream copyright notice, and
declare the combined terms — for example `Apache-2.0 AND MIT`. Where an upstream
license requires its conditions text to travel with the source, that text is kept
in the file verbatim. The third-party section at the end of [LICENSE](LICENSE)
lists which components fall under which license, and [`licenses/`](licenses)
holds the corresponding license texts.
