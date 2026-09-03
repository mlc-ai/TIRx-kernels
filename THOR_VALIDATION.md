# NVIDIA Thor (`sm_110a`) validation

This document records the validation state for retargeting the repository's
SM100 kernels to NVIDIA Jetson AGX Thor.  A kernel is added to its exact
`KERNEL_META["runtime_cuda_archs"]` allowlist only after every entry in that
module's `CONFIGS` has compiled, launched on Thor, and passed its numerical
correctness check.

## Baseline

- Hardware: NVIDIA Thor, compute capability 11.0, 20 SMs
- CUDA toolkit: 13.1
- TVM: Apache TVM `main` at `15b607d6bf`, including Thor target tag PR #20259
- TIRx-kernels base: `0512291`
- Candidate scope: 95 SM100 kernels and 9,282 correctness configurations
- Fully validated: 90 kernels and 9,204 configurations

The remaining five modules contain 78 configurations. Two collectives require
NVSHMEM (and half of their cases require four GPUs), both DeepEP modules require
eight GPUs, and MegaMoE combines a compute-10-only DeepGEMM host path with 15
multi-GPU cases. They remain outside the exact Thor allowlist; the boundary is
recorded below rather than treating missing hardware or an unavailable
reference as a correctness pass.

## Performance baseline

The final performance campaign uses the same Proton timer, 25 ms warmup budget,
100 ms repeat budget, five independent rounds, and arithmetic-mean aggregation
as the repository's historical SM100/B200 baseline. It includes every
admitted Thor kernel that has a default benchmark workload.

The final single-piece run completed 254/254 representative workloads across
87 kernels with no failures or interference retries. Of those rows, 183 have
usable exact-shape timings in the historical B200 baseline. Their geometric-
mean Thor/B200 latency ratio is 10.433x (9.6% relative throughput), and the
median is 12.205x. This is a cross-machine comparison with different software
revisions and dynamic Thor clocks, not a controlled hardware-only A/B.

See [THOR_B200_PERFORMANCE.md](THOR_B200_PERFORMANCE.md) for the complete table,
per-kernel summaries, GEMM effective throughput, round variability, provenance,
and comparison limitations.  Generate it from the raw JSON with
`scripts/report_thor_b200.py`.

On this Thor environment, set
`TRITON_CUPTI_LIB_PATH=/usr/local/cuda-13.1/extras/CUPTI/lib64` before running
Proton.  Triton 3.5.1 otherwise selects its bundled CUDA 12.8 CUPTI, whose
`cuptiSubscribe` cannot initialize against the CUDA 13.1 driver stack.
Set `TRITON_PTXAS_PATH=/usr/local/cuda-13.1/bin/ptxas` as well when a reference
uses Triton; its bundled CUDA 12.8 assembler does not recognize `sm_110a`.

## Mamba stochastic-rounding compatibility

PTX exposes `cvt.rs.f16x2.f32` only on `sm_100a` and `sm_103a`; CUDA 13.1
`ptxas` rejects that instruction for `sm_110a`.  The shared
`K.idioms.cvt_rs_f16x2_f32` helper therefore keeps the native instruction on
the two supported architectures and mirrors the pinned FlashInfer software
path on Thor: add each supplied 13-bit random field to the discarded FP32
mantissa bits, truncate, and rebias to FP16.

All 12 Philox configurations across the six selective-state-update kernels
initially reproduced the unsupported-instruction error.  After the fallback,
the complete family passes 236/236 configurations, including STP/MTP,
horizontal/vertical/simple, and intermediate-state variants.

## Persistent-cluster and FP8 reference compatibility

Thor rejects the non-portable 16-block cluster shapes accepted by B200 with
`CUDA_ERROR_INVALID_CLUSTER_SIZE`.  The target helper preserves every B200
shape at eight blocks or below and reduces only 16-block shapes to eight on an
explicit `sm_110a` prepare.  Both the TIRx and pinned CUTLASS DSL paths receive
the same adjusted schedule; the operation and tensor shapes do not change.

PyTorch cannot export its FP8 tensor dtypes through DLPack on this environment.
The dense SwiGLU reference adapter therefore exports their byte views and sets
the corresponding CUTLASS element type explicitly, preserving the exact input
bytes.  All 75 dense SwiGLU and 42 grouped dGLU configurations then compile and
launch.  Two batched N-major, two-CTA dense SwiGLU specializations expose a
pinned-source C-store bug: TIRx agrees with the direct FP32 equation on every
element while the source can store the other batch's value.  Those exact two
cases use the mathematical oracle; the other 73 retain the source comparison.

Together with FastTopK, fused DiT LayerNorm, and stable-sort TopK, this batch
passes 242/242 configurations on Thor.

## Block-scaled GEMM and DSA compatibility

The five block-scaled cuDNN Frontend ports now pass 245/245 configurations on
Thor: persistent Amax, DSReLU quantization, SReLU quantization, interleaved
SwiGLU quantization, and grouped MoE dGLU/dbias.  Their TIRx schedules use the
same explicit cluster reduction described above.  FP8 and packed-FP4 tensors
are passed to the pinned CuTe source through byte-preserving typed DLPack
views.

The pinned source's NVVM path cannot lower four of these kernels for
`sm_110a`, so their existing structured FP32 equations are the numerical
oracle on Thor.  Interleaved SwiGLU retains the pinned-source comparison on 56
configurations.  Two 256x64-tile schedules make the source cross AB12 epilogue
tiles after retargeting; for those exact schedules, TIRx agrees exactly with
the independently evaluated structured GEMM while the source stores values
from another tile.

The four-stage DSA sparse-attention backward chain passes 12/12 configurations
against its full FP32 attention/gradient oracle.  Coverage includes BF16 and
FP16, ragged and empty rows, negative indices, sink folding, and no-sink
execution.  The pinned DSA source has no compute-11 host dispatch, so it is not
used as a Thor timing peer.

## Retargeted source references and TinyGEMM2

The FlashInfer GDN context-parallel prefill source has a single host-side
compute-10.x guard; its pinned CuTe DSL bodies accept an explicit architecture.
Compiling those unchanged bodies as `sm_110a` preserves the independent source
comparison, and all 10/10 TIRx configurations pass it.

Both cuDNN projection-plus-RoPE MXFP8 kernels pass 12/12 configurations after
their pinned source adapters use the same byte-preserving FP8 DLPack bridge as
the other cuDNN kernels.  This covers BF16 and MXFP8 inputs, both weight
orientations, and token counts from 128 through 4096.

TinyGEMM2 passes 8/8 problem shapes, with both normal and programmatic dependent
launch paths checked per shape.  Because its frozen CUDA binary has no Thor
target, correctness is measured against an independent FP32
`input @ weight.T + bias` equation rounded to BF16.

## TopK, RMSNorm, and FlashKDA

All three FlashInfer TopK implementations pass their complete Thor matrices:
FilteredTopK 107/107, single-CTA radix TopK 234/234, and multi-CTA radix TopK
354/354. Coverage includes FP32, FP16, and BF16; plain, page-table, and ragged
index transforms; deterministic and atomic collection; tie handling; and the
workspace reset/reuse paths.

The base RMSNorm family passes 960/960 configurations: RMSNorm 279/279, fused
add RMSNorm 281/281, and QK RMSNorm 400/400. Three extreme hidden dimensions
select a 16-CTA cluster in the pinned B200 schedule. On Thor, both the TIRx
kernel and the unchanged pinned CuTe body are recomputed as complete 8-CTA
schedules, preserving the source comparison while staying within Thor's
cluster limit.

The four quantized RMSNorm modules pass 5,378/5,378 configurations: FP4
RMSNorm 1,025/1,025, fused-add FP4 RMSNorm 1,245/1,245, RMSNorm quantization
1,552/1,552, and fused-add RMSNorm quantization 1,556/1,556. Coverage includes
BF16/FP16 inputs, E4M3/E5M2/int8 outputs, PDL, scale modes, swizzled layouts,
residual paths, automatic allocation, fragment boundaries, and hidden sizes
through 1,048,576.

The six FlashKDA decode stages pass 126/126 configurations and the BF16 M128
fused prefill kernel passes 6/6. Their Thor checks use independent FP32
recurrent gated-delta-rule equations, including speculative checkpoint and
accepted-token semantics. The BF16 comparison floor is one representable step;
this covers the handful of hundred-million-element state outputs that differ
from a differently ordered FP32 recurrence by exactly one BF16 ULP.

The agent-evolved KDA forward kernel passes its sole production shape
(`B=1,T=8192,H=96,D=128`) against the pinned FLA numerical reference. B200's
148 SMs launch its 96 work items as 96 CTAs, but the generic 20-SM Thor launch
collapsed them to 20 CTAs and reused barrier/TMEM state for multiple heads,
which deadlocked. Thor now preserves one work item per CTA and lets CUDA issue
the 96 CTAs in waves; other architectures keep their original launch rule.

## BSA forward-combine oracle

`cudnn_sm100_bsa_forward_combine_blk64` retains a bitwise-exact comparison
between TIRx and the pinned CUDA source for both O and LSE.  Its auxiliary BF16
mathematical-oracle check now combines the existing `0.0078125` absolute floor
with one representable BF16 step at the expected value's scale.  This accepts
the single one-ULP reduction difference previously observed at larger
magnitudes without weakening the primary source gate.  The complete matrix now
passes 9/9 configurations.

## DeepGEMM kernels

Ten DeepGEMM modules pass 382/382 configurations on Thor.  The five shared
FP8 GEMM/BMM families account for 60 cases; paged FP4/FP8 MQA logits account
for 209; and non-paged FP4/FP8 MQA logits plus TF32 HC pre-norm GEMM account
for 113.  Their Thor paths preserve the generated kernel bodies and replace
only B200-specific host preparation or unavailable DeepGEMM reference
dispatch with independent structured equations.

The paged MQA schedule derives the persistent launch shape from Thor's 20 SMs
instead of retaining a 148-SM B200 assumption. The prepared benchmark path
also compiles with that hardware-resolved SM count; otherwise its fixed
148-CTA launch indexes schedule metadata allocated for only 20 SMs. B200 keeps
the original 148-SM default. Correctness includes both logical page layouts,
invalid entries, variable lengths, FP4/FP8 scale layouts, and the full
published configuration matrices.

## MSA sparse attention

All five MSA modules pass 91/91 configurations: flat-schedule preparation
28/28, split-atomic preparation 27/27, sparse attention forward 10/10,
NVFP4-KV forward 12/12, and forward combine 14/14.  Thor uses independent
PyTorch scheduling and attention equations where the pinned MSA package has
no compute-11 host dispatch.  The matrices cover masking, ragged sequences,
page tables, FP4 dequantization, split scheduling, empty work, and combine
reductions.

## FlashMLA sparse decode

Sparse FlashMLA head-64 decode passes 15/15 configurations for both V3.2 and
Model1 cache layouts, including the 148/256-batch, 16K-top-k pressure cases.
The Thor oracle decodes only indexed FP8 cache rows and evaluates attention in
chunks, following the pinned FlashMLA mathematical reference without relying
on its compute-10-only host dispatcher.  Correctness fixtures may alias
physical pages to keep their cache below Thor's unified-memory budget while
preserving logical sequence, top-k, scheduler, page-table, and masking
semantics.  Performance preparation retains the original full physical cache.

## Explicitly blocked modules

| Module | Configurations | Thor evidence / blocker |
|---|---:|---|
| `allgather_gemm` | 16 | Eight one-GPU cases fail before compile because NVSHMEM is not installed; eight require four GPUs. |
| `gemm_reduce_scatter` | 16 | Eight one-GPU cases fail before compile because NVSHMEM is not installed; eight require four GPUs. |
| `deepep_dispatch` | 4 | Every configuration requires eight GPUs. |
| `deepep_combine` | 4 | Every configuration requires eight GPUs. |
| `sm100_fp8_fp4_mega_moe` | 38 | The smallest one-GPU case reaches DeepGEMM's compute-10-only scale-layout host guard; 15 cases additionally require 2/4/6 GPUs. |

This is 5/95 modules and 78/9,282 configurations. The validated fraction is
90/95 modules (94.7%) and 9,204/9,282 configurations (99.2%).

## Validation artifacts

Resumable evidence from the final phase is outside the repository under
`/home/tlopexh/thor-validation/correctness-final`:

- `rmsnorm/*.jsonl`: 5,378 unique PASS results
- `flashmla-compact/all.jsonl`: 15 unique PASS results
- `kda-system-ptxas.jsonl`: one PASS result
- `communication-blockers.jsonl`: the 40 exact NVSHMEM/topology blocker results

Earlier full matrices were written to `/tmp` during the staged investigation
and did not survive the machine restart; their aggregate results and the fixes
they gated are recorded by the sections above and their focused commits.

These files are evidence from this machine, not portable repository inputs.
Repeat validation through `scripts/validate_thor.py`; do not infer support only
from the saved summary.
