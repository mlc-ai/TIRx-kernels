# tirx-kernels

High-performance GPU kernels written in [TIRx](https://github.com/apache/tvm).

## Kernels

All kernels target `sm_100a`. Names are the registry names accepted by the
`--kernel` CLI filters. Each category directory holds the kernels ported from
one upstream project; `basic/` holds the native TIRx kernels with no single
upstream project.

`basic/` — native TIRx kernels:

| Kernel | dtype | What it is |
| ------ | ----- | ---------- |
| `fp16_bf16_gemm` | fp16 / bf16 | Dense GEMM |
| `nvfp4_gemm` | nvfp4 | Dense GEMM |
| `rmsnorm` | fp16 / bf16 | RMSNorm |
| `allgather_gemm` | fp16 | AllGather + GEMM (multi-GPU, NVSHMEM) |
| `gemm_reduce_scatter` | fp16 | GEMM + ReduceScatter (multi-GPU, NVSHMEM) |

`flashattention/` — Dao-AILab flash-attention ports:

| Kernel | dtype | What it is |
| ------ | ----- | ---------- |
| `flash_attention4` | bf16 | FlashAttention-4 |
| `flash_attention_backward_sm100` | fp16 | Two-CTA FlashAttention backward (D=128); [schedule sketch](tirx_kernels/flashattention/flash_attention_backward_sm100_sketch.md) |

`flashinfer/` — FlashInfer ports:

| Kernel | dtype | What it is |
| ------ | ----- | ---------- |
| `tinygemm2_sm100` | bf16 | TinyGEMM2 |
| `gdn_prefill_sm100` | fp16 | Gated Delta Net prefill |
| `flashkda_bf16_fused_m128` | bf16 | Recurrent KDA prefill, M128 schedule |

Testing/benching against the `flashinfer_m128` reference requires
`FLASHKDA_PR_WORKTREE` pointing at the flashinfer-ai/flashinfer#4262 head
worktree containing the m128 kernel.

`flashmla/` — FlashMLA sparse attention ports:

| Kernel | dtype | What it is |
| ------ | ----- | ---------- |
| `sparse_flashmla_prefill_head64_phase1` | bf16 | Sparse prefill, 64 q-heads (phase 1) |
| `sparse_flashmla_prefill_head128_phase1` | bf16 | Sparse prefill, 128 q-heads (phase 1) |
| `sparse_flashmla_prefill_head128_small_topk_phase1` | bf16 | Sparse prefill, 128 q-heads, small top-k (phase 1) |
| `flash_mla_sparse_fwd` | bf16 | Sparse forward |

`deepgemm/` — DeepGEMM ports:

| Kernel | dtype | What it is |
| ------ | ----- | ---------- |
| `deepgemm_sm100_fp8_gemm_1d1d` | fp8 + fp4 / bf16 | Dense blockwise-scaled GEMM |
| `deepgemm_sm100_m_grouped_fp8_gemm_contiguous` | fp8 + fp4 / bf16 | M-grouped contiguous GEMM (incl. psum layout) |
| `deepgemm_sm100_m_grouped_fp8_gemm_masked` | fp8 + fp4 / bf16 | M-grouped masked GEMM |
| `deepgemm_sm100_k_grouped_fp8_gemm_contiguous` | fp8 / fp32 | K-grouped contiguous GEMM (wgrad, incl. psum layout) |
| `deepgemm_sm100_fp8_bmm` | fp8 / bf16 + fp32 | Batched GEMM behind `fp8_einsum` |
| `deepgemm_sm100_fp4_mqa_logits` | fp4 / bf16 | MQA attention logits |
| `deepgemm_sm100_fp8_mqa_logits` | fp8 / bf16 | MQA attention logits |
| `deepgemm_sm100_fp4_paged_mqa_logits` | fp4 / bf16 | Paged-KV MQA attention logits |
| `deepgemm_sm100_fp8_paged_mqa_logits` | fp8 / bf16 | Paged-KV MQA attention logits |
| `deepgemm_sm100_tf32_hc_prenorm_gemm` | tf32 / bf16 | Prenorm GEMM |
| `deepgemm_fp8_fp4_mega_moe` | fp8 + fp4 | Fused MoE megakernel (MegaMoE) |

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

These are **not on PyPI** and must be installed/available separately. They are
imported lazily, so `import tirx_kernels` and kernel discovery work without
them — they are only needed to actually compile/run a kernel:

| Dependency       | Needed by                          | Notes                                                  |
| ---------------- | ---------------------------------- | ------------------------------------------------------ |
| `tvm.tirx`       | all kernels (compile + run)        | The TIRx compiler. Put it on `PYTHONPATH`, e.g. `/path/to/tir/python`. |
| `torch`          | all kernels                        | CUDA build matching your GPU.                          |
| `deep_gemm`      | FP8 GEMM and `deepgemm_*` baselines | Used for optimized reference kernels. |
| `flashinfer`     | `nvfp4_gemm` data/baseline | Used for nvfp4 quantization and reference impls. |
| `flash-attn` + CUTLASS DSL | `flash_attention_backward_sm100` data/baseline | Current SM100 forward/backward reference. |
| `flash_kda`      | `flashkda_bf16_fused_m128` baseline | Uses the installed package; results record its package, source, extension, and CUTLASS provenance when available. |
| `sglang` (+ CUTLASS DSL) | `deepgemm_sm100_fp8_paged_mqa_logits` reference | `sglang_cutedsl` reference; checkout on `PYTHONPATH`. |
| `flash_mla`      | `sparse_flashmla_*` / `flash_mla_sparse_fwd` baselines | Reference impls. |
| NVSHMEM          | `allgather_gemm`, `gemm_reduce_scatter` | Required to compile/run the GemmComm kernels. |

The bench_suite **default sweep** hard-requires several of these (a missing
reference fails the whole sweep) — see
[`tirx_kernels/bench_suite/README.md`](tirx_kernels/bench_suite/README.md)
for the prerequisites and workarounds.

## Usage

### Command line

```bash
# List discovered kernels (with their config labels)
python -m tirx_kernels.registry --format json

# Run correctness tests (optionally filter by kernel / config label)
python -m tirx_kernels.test
python -m tirx_kernels.test --kernel fp16_bf16_gemm
python -m tirx_kernels.test --kernel fp16_bf16_gemm --config bf16_1024x1024x1024

# Benchmark
python -m tirx_kernels.bench --kernel nvfp4_gemm

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

Kernel ports derived from third-party projects (DeepGEMM, FlashMLA,
flash-attention, FlashInfer) retain their upstream terms and per-file
attribution headers; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
for the corresponding copyright notices and license texts.
