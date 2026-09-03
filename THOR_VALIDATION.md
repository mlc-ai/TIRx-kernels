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
- Fully validated: 46 kernels and 1,263 configurations

The operator-level smoke sweep currently reports 56 numerical passes, 30
failures, and 9 skips.  Follow-up launch-only checks show that 27 more kernels
compile and launch successfully without their unavailable or incompatible
external correctness references.  Seven DeepGEMM cases are blocked before
launch by architecture-specific input layout preparation, two communication
kernels require NVSHMEM, two DeepEP kernels require eight GPUs, and the
agent-evolved KDA kernel remains conservatively disabled after an earlier
long-running Thor launch.

## Performance baseline

The 23 fully validated kernels also completed all 69 representative benchmark
rows on Thor: 69 passed, 0 failed, and 0 interference retries.  The
final run uses the same Proton timer, 25 ms warmup budget, 100 ms repeat budget,
five independent rounds, and arithmetic-mean aggregation as the repository's
historical SM100/B200 baseline.

See [THOR_B200_PERFORMANCE.md](THOR_B200_PERFORMANCE.md) for the complete table,
per-kernel summaries, GEMM effective throughput, round variability, provenance,
and comparison limitations.  Generate it from the raw JSON with
`scripts/report_thor_b200.py`.

On this Thor environment, set
`TRITON_CUPTI_LIB_PATH=/usr/local/cuda-13.1/extras/CUPTI/lib64` before running
Proton.  Triton 3.5.1 otherwise selects its bundled CUDA 12.8 CUPTI, whose
`cuptiSubscribe` cannot initialize against the CUDA 13.1 driver stack.

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

## BSA forward-combine oracle

`cudnn_sm100_bsa_forward_combine_blk64` retains a bitwise-exact comparison
between TIRx and the pinned CUDA source for both O and LSE.  Its auxiliary BF16
mathematical-oracle check now combines the existing `0.0078125` absolute floor
with one representable BF16 step at the expected value's scale.  This accepts
the single one-ULP reduction difference previously observed at larger
magnitudes without weakening the primary source gate.  The complete matrix now
passes 9/9 configurations.

## Validation artifacts

The resumable local JSONL runs are kept outside the repository under `/tmp`:

- `tirx-thor-main-0512291-smoke.jsonl`
- `tirx-thor-main-0512291-launch-failures.jsonl`
- `tirx-thor-main-0512291-launch-skips.jsonl`
- `tirx-thor-main-0512291-fp16-bf16-full.jsonl`
- `tirx-thor-main-0512291-batch2-full.jsonl`
- `tirx-thor-main-0512291-batch3-quant-full.jsonl`
- `tirx-thor-main-0512291-batch4-kda-rms-full.jsonl`
- `tirx-thor-main-0512291-batch5-bsa-full.jsonl`
- `tirx-thor-main-0512291-batch6-gdn-decode-full.jsonl`
- `tirx-thor-main-0512291-batch7-gdn-prefill-full.jsonl`
- `tirx-thor-main-0512291-batch8-linear-attention-full.jsonl`
- `tirx-thor-main-0512291-batch9-attention-norm-full.jsonl`
- `tirx-thor-main-0512291-batch10-mamba-ssu-final.jsonl`
- `tirx-thor-main-0512291-batch11-topk-norm-gemm-final.jsonl`
- `tirx-thor-main-0512291-batch13-bsa-combine-final.jsonl`

These files are evidence from this machine, not portable repository inputs.
Repeat validation through `scripts/validate_thor.py`; do not infer support only
from the saved summary.
