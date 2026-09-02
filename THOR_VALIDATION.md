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
- Fully validated: 34 kernels and 776 configurations

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

## BSA forward-combine observation

`cudnn_sm100_bsa_forward_combine_blk64` passes 8 of its 9 correctness
configurations.  The remaining configuration is
`b1_h4_sq1024_s2` (seed 8606).

For that configuration, the TIRx output and the pinned CUDA source output are
bitwise identical for both O and LSE.  The subsequent, auxiliary comparison
between the pinned source and the mathematical oracle rejects one BF16 element
out of 524,288:

- index: `(0, 232, 2, 5)`
- absolute difference: `0.015625`
- current absolute tolerance: `0.0078125`

At the value's magnitude this is one BF16 representable step.  This is not
evidence of a TVM or TIRx/source mismatch.  The follow-up is to make the
secondary oracle comparison BF16-ULP-aware while preserving the existing
bitwise TIRx-versus-pinned-source gate.  The kernel must not be added to the
`sm_110a` allowlist until all nine `CONFIGS` pass the final check.

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

These files are evidence from this machine, not portable repository inputs.
Repeat validation through `scripts/validate_thor.py`; do not infer support only
from the saved summary.
