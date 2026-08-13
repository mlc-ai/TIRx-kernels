# Bench Suite Pipeline Validation

Validation date: 2026-08-13.

This report records implementation and acceptance evidence for the one-shot
benchmark pipeline. It deliberately distinguishes static migration, structural
tests, measured single-GPU evidence, and the human-directed multi-GPU runtime
exemption.

## Implementation status

- Lifecycle: every workload attempt uses one fresh process and follows
  `PREPARING → READY → ASSIGNED → RUNNING_GPU → RESULT → exit`.
- CPU prepare owns exact module import/parsing, config resolution,
  specialization, IR generation, compilation, and any compile-cache population.
- GPU assignment is late-bound after READY. The prepared executable remains in
  its creating process and cannot be serialized or executed from another PID.
- Generic adapters compile canonical `get_kernel()` output during prepare and
  consume the executable through `compile_kernel_lazy()` after assignment. The
  capability audit rejects eager GPU-stage generation/compilation.
- The orchestrator uses a bounded number of one-shot children, a bounded READY
  backlog, condition-driven internal GPU release/dispatch, and polling only for
  external occupancy changes.
- Distributed, Kineto, and MegaMoE adapters compile/export before assignment and
  start rank CUDA/runtime state only from `run_gpu()` after the complete claim.
- Default measurement semantics remain 5 rounds, 1.0 second cooldown, the same
  timer budgets, reference construction, correctness work, raw samples, and
  arithmetic-mean aggregation.

## Static and no-card evidence

The capability audit reports:

- 41 registered kernels.
- 992 module-owned benchmark configs.
- 992 YAML inventory entries, with no module-only or YAML-only config.
- 21 generic lazy-replay adapters checked at AST level.
- 112 single-GPU default workloads, with at most three defaults per kernel.
- 27 multi-GPU configs, none selected by the default measured sweep.

`tests/test_bench_pipeline_protocol.py` has 36 passing behavior tests covering:

- one-shot process/IPC behavior, log isolation, cancellation, fail-fast, dynamic
  external eligibility, fresh-process interference retry, and resource bounds;
- same-GPU serialization, concurrent logical-GPU execution, complete atomic
  claims, assignment-count/duplicate/index rejection, and no partial ownership;
- process-local prepared objects, serialization rejection, and CUDA prepare
  guards including driver-level contexts;
- strict result schemas for event, Proton, CUDA-graph Proton, Kineto, and
  MegaMoE timers without changing the requested rounds or cooldown;
- Kineto complete correlated activity spans, rank barriers, sample-wise max,
  and protocol metadata;
- MegaMoE alternating implementation order, sample-wise rank max aggregation,
  mismatched rank-round rejection, and distributed cleanup on failure;
- lazy replay skipping its builder exactly once, with under/over-consumption
  rejected, plus a static gate against moving builder work back to GPU stage.

## CPU prepare evidence without GPU assignment

Three fresh prepared children were run only through READY and then cancelled.
`TIRX_PREPARE_NUM_SMS=148` supplied the compile profile; no GPU was assigned.

| workload | wall to READY | framework import | exact import | specialize/generate/compile | result |
|---|---:|---:|---:|---:|---|
| `fp16_bf16_gemm/fp16_1024x1024x1024` | 6.341s | 2.920s | 0.470s | 2.908s | READY, clean CANCEL |
| `act_and_mul/silu_fp16_d4096_t1` | 4.975s | 3.485s | 0.480s | 0.970s | READY, clean CANCEL |
| `flash_attention4/s1024_h32kv4` | 9.424s | 3.768s | 0.932s | 4.687s | READY, clean CANCEL |

These measurements demonstrate that parsing/generation/compilation—not only
imports—are on the parallel CPU side of the boundary.

## Targeted single-GPU measured evidence

No full sweep was run. All targeted measurements retained 5 rounds and a 1.0
second cooldown.

The three-small-GEMM pipeline run produced:

- observed critical wall: 46.787s;
- first READY: 7.087s;
- one-GPU list schedule: 39.635s;
- READY starvation and foreign wait: 0s;
- dispatch p95: 0.056s;
- unexplained residual: 0.065s (0.14% of observed wall).

This run has no matching same-protocol migration-before artifact. The older
prototype evidence—29.70s sequential versus 19.17s pipelined, 1.55×—is retained
only as directional cost-model evidence, not claimed as a final A/B.

After the generic lazy-replay correction, one small `act_and_mul` workload was
run on one automatically selected idle B200:

- CPU prepare: 4.082s;
- GPU stage: 14.918s;
- observed/expected critical wall: 19.085s / 19.037s;
- unexplained residual: 0.047s;
- dispatch p95: 0.040s;
- TIRx/FlashInfer: 2.606µs / 2.789µs;
- 5 raw samples per implementation, Proton timer, 1.0s cooldown;
- no interference retry;
- host peaks: 3 processes, 1.999 GiB RSS, 218 file descriptors.

## Multi-GPU runtime validation exemption

Validation status: `exempted_by_human_unmeasured`.

The exempt workloads are all configs with `num_gpus > 1` in:

- `allgather_gemm` (8 configs);
- `gemm_reduce_scatter` (8 configs);
- `deepgemm_fp8_fp4_mega_moe` (11 configs).

Their code migration includes the same pipeline-only lifecycle, assignment-count
validation, complete atomic claims before rank/CUDA startup, compile/export before
assignment, barriers, Kineto spans, sample-wise max aggregation, MegaMoE round
ordering, and process-group/runtime cleanup.

No multi-GPU runtime measurement was performed, by explicit human direction.
Therefore these workloads are not marked passed and are not represented as
`MISSING`, zero, null, or empty values. The complete per-config exemption inventory
is generated in `.bench-suite/reports/pipeline-capability.md`.

## Remaining evidence boundary

- The implementation, static all-config migration gate, no-card structural tests,
  CPU-only prepare evidence, and targeted single-GPU execution are complete.
- Multi-GPU runtime behavior remains intentionally unmeasured.
- A final same-protocol migration-before versus migration-after single-GPU A/B is
  not available. It must not be inferred from the prototype or unrelated runs.
- The full 112-workload measured sweep and baseline promotion remain deferred
  until the shared machine is available; they were not required or run here.
