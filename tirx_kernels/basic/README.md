<!--
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Basic kernels

Native TIRx kernels with no single upstream project: dense GEMMs
(`fp16_bf16_gemm`, `nvfp4_gemm`), RMSNorm (`rmsnorm`), and the distributed
GEMM+collective kernels below.

## Distributed GEMM kernels

The distributed SM100, tensor-parallel kernels were recovered from the
megakernel performance branch. Both workloads register TP1 and TP4
specializations for eight model shapes:

| Registry name | Global operation | Rank-local result |
| --- | --- | --- |
| `allgather_gemm` | `A[M, K] @ W[N, K].T` after gathering row shards of `A` | `[M, N / TP]` |
| `gemm_reduce_scatter` | sum of rank-local `A[M, K / TP] @ W[N, K / TP].T`, scattered over `M` | `[M / TP, N]` |

The second source was historically called GEMM+AllReduce, but its actual
protocol and output shape are ReduceScatter. The public name reflects the
implemented operation. The registered matrix contains the Qwen3-8B,
LLaMA-3.1-8B, Gemma-2-9B, Gemma-2-27B, Qwen3-32B, LLaMA-3.1-70B,
GPT-3-175B, and LLaMA-3.1-405B shapes. All use `M=8192` and FP16. TP values
other than 1 or 4 and the legacy static scheduler are rejected.

The parent compiles and exports one module, then `torch.multiprocessing.spawn`
starts one rank-local worker per GPU. NCCL bootstraps the process group and
broadcasts the NVSHMEM UID; every worker explicitly initializes NVSHMEM, loads
the local module, and owns its Device API streams. No Disco session or remote
runtime object is created. Mutable queues, semaphores, workspaces, and outputs
are reset independently on every rank before each measured launch.

Both registry entries are direct kernel ports. This PR contains no megakernel
DSL, scheduling policy, execution region, or MoE change. AllGather+GEMM keeps
the already validated dynamic queue implementation. GEMM+ReduceScatter uses
the hand-transcribed fused persistent kernel in `gemm_reduce_scatter.py`: a
two-CTA GEMM queue feeds an initially
empty ReduceScatter queue, and the final contributor publishes each tile with
system-scope release/acquire ordering. TP4 ReduceScatter uses NVLS
`multimem.ld_reduce` directly from the multicast partial output; TP1 uses the
same fused queue and a local vectorized writeback. There is no host peer
transfer, staging buffer, or separate reduction kernel.

## Validation and benchmarking

GEMM+ReduceScatter correctness checks every rank's full partial GEMM and local
ReduceScatter output for 20 consecutive reset/relaunch cycles, including queue
tails, task consumption, semaphore counts, and NaN tile coverage.

The headline benchmark uses cold-cache Kineto full spans: 5 warmups and 30
measured launches per implementation and round, with DeepGEMM's 8 GB L2 flush,
GPU sleep, and rank barrier before every launch. Mutable state is reset before
the profiler scope. Each sample spans the earliest through latest correlated
CUDA activity across all streams, is reduced by the slowest rank, and contributes
to the round median; multiple rounds use their arithmetic mean. The cuBLAS+NCCL
reference initializes both libraries and captures the complete GEMM-plus-
collective sequence before timing; its measured closure is one CUDA Graph replay.
TIRx and cuBLASMp retain their direct launch closures. All headline values use
the same Kineto full-span protocol, so ratios never mix timers.

cuBLASMp 0.10 requires nvmath-python, NCCL4Py, and a compatible recent NCCL.
Every benchmark requires absolute paths for all four runtime dependencies so a
loader-path change cannot silently alter the comparison:

```bash
export TIRX_NCCL_LIBRARY=/path/to/libnccl.so.2
export TIRX_CUBLAS_LIBRARY=/path/to/libcublas.so
export TIRX_CUBLASMP_LIBRARY=/path/to/libcublasmp.so.0
export TIRX_NVSHMEM_LIBRARY=/path/to/libnvshmem_host.so
export PYTHONPATH=/path/to/nvmath-python:/path/to/cublasmp-package:$PYTHONPATH
```

The selected files are preloaded only in newly spawned rank workers. Each
result records the actual shared object resolving the NCCL, cuBLAS, cuBLASMp,
and NVSHMEM API symbol together with its runtime version, and fails if any
loaded file differs from its configured lock. cuBLASMp builder failures remain
visible in `errors`, which bench-suite treats as a failed workload.

The result's `ratios` mapping is always `baseline_us / tirx_us`; values greater
than one mean TIRx is faster.

For a B200 host, select any registered config explicitly when needed:

```bash
python -m tirx_kernels.test --kernel allgather_gemm \
  --config tp4_m8192_n51200_k5120_fp16_dynamic
python -m tirx_kernels.test --kernel gemm_reduce_scatter \
  --config tp4_m8192_n5120_k25600_fp16_dynamic

python -m tirx_kernels.bench --kernel allgather_gemm \
  --config tp4_m8192_n51200_k5120_fp16_dynamic --timer kineto --rounds 5 --json
python -m tirx_kernels.bench --kernel gemm_reduce_scatter \
  --config tp4_m8192_n5120_k25600_fp16_dynamic --timer kineto --rounds 5 --json
```

The distributed Kineto protocol uses fixed iteration counts, so `--warmup` and
`--repeat` overrides are rejected. The reported value is the arithmetic mean of
the five round results.
