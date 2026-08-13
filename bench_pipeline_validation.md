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
- 5 DeepGEMM direct-compile adapters with exact prepared-cache key and
  consumption validation.
- 15 explicit/custom adapters covering process-local executables, dispatcher
  delegation, hardware-profile compile caches, and distributed export/load
  lifecycles; together the three adapter classes account for 41/41 kernels.
- 112 single-GPU default workloads, with at most three defaults per kernel.
- 33 historically over-broad default selections, each now exactly three points
  with its rationale stored beside the canonical `default` flags in the same
  kernel YAML and emitted by the capability report.
- 27 multi-GPU configs, none selected by the default measured sweep.

`tests/test_bench_pipeline_protocol.py` has 56 passing behavior tests covering:

- one-shot process/IPC behavior, log isolation, cancellation, fail-fast, dynamic
  external eligibility, fresh-process interference retry, bounded-backlog
  drain/refill without dropped or reused attempts, and resource bounds;
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
  rejected, plus a static gate against moving builder work back to GPU stage;
- strict prepared-cache replay for the five DeepGEMM `compile_spec` users:
  mismatched keys, missing consumption, and extra consumption all fail before a
  GPU-stage compilation can silently enter the critical path;
- missing/incomplete cost-model evidence never publishes expected wall,
  residual, starvation, foreign-wait, or latency values as zero;
- summaries watermark non-default rounds/cooldown as diagnostic, including old
  run JSON whose protocol must be derived from its result rows;
- standalone `run_kernel_bench()` composes the same prepare/run-GPU contract;
- timeline validation rejects missing transitions, reversed timestamps, and
  overlapping ownership intervals on the same GPU;
- capability accounting proves 41/41 adapters and 33/33 reviewed three-point
  selections from canonical sources.

## CPU prepare evidence without GPU assignment

Three representative fresh prepared children were run only through READY and
then cancelled.
`TIRX_PREPARE_NUM_SMS=148` supplied the compile profile; no GPU was assigned.

| workload | wall to READY | framework import | exact import | specialize/generate/compile | result |
|---|---:|---:|---:|---:|---|
| `fp16_bf16_gemm/fp16_1024x1024x1024` | 6.341s | 2.920s | 0.470s | 2.908s | READY, clean CANCEL |
| `act_and_mul/silu_fp16_d4096_t1` | 4.975s | 3.485s | 0.480s | 0.970s | READY, clean CANCEL |
| `flash_attention4/s1024_h32kv4` | 9.424s | 3.768s | 0.932s | 4.687s | READY, clean CANCEL |

These measurements demonstrate that parsing/generation/compilation—not only
imports—are on the parallel CPU side of the boundary.

After strict prepared-cache replay was added, each of the five DeepGEMM
`compile_spec` adapters was also run in a fresh child through READY and then
cancelled, again without a GPU assignment:

| workload family | specialize/generate/compile | result |
|---|---:|---|
| FP8 BMM | 1.284s | READY, clean CANCEL |
| FP8 GEMM 1D1D | 1.217s | READY, clean CANCEL |
| K-grouped contiguous | 1.224s | READY, clean CANCEL |
| M-grouped contiguous | 1.275s | READY, clean CANCEL |
| M-grouped masked | 1.300s | READY, clean CANCEL |

This proves that each family can create its specialization in the CPU-only
phase. Exact replay consumption is independently enforced by the process-local
cache-key tests and the all-config AST capability gate described above.

### Concurrent large-shape resource envelope

One CPU-only run prepared three deliberately large representatives concurrently:

- `fp16_bf16_gemm/fp16_16384x16384x16384`;
- `flash_attention4/s8192_h32kv32`;
- `gdn_prefill_sm100/hq32_hv32_s8192x16`.

All three fresh children reached READY within 8.780 seconds. The owned process
tree peaked at 4 processes, 3,828,740,096 RSS bytes (3.566 GiB), and 287 open
file descriptors. The orchestrator then sent CANCEL without ever assigning a
GPU; the PID registry was empty afterward, all temporary directories were
removed, and CUDA remained uninitialized. The tracked artifact
`bench_pipeline_cpu_prepare_evidence.json` contains the individual phase times,
resource peaks, protocol, and cleanup state, and its invariants are checked by
the protocol test suite.

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

The older prototype evidence—29.70s sequential versus 19.17s pipelined, 1.55×—
is retained only as directional cost-model evidence. A same-protocol A/B is
reported separately below.

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

### Default-protocol migration A/B

The migration-before side was checked out in a detached worktree at `a91a1b7`,
which still contains the one-stage scheduler. Both sides used exactly these
three explicit workloads:

- `fp16_bf16_gemm/fp16_1024x1024x1024`;
- `fp16_bf16_gemm/fp16_2048x2048x2048`;
- `fp16_bf16_gemm/fp16_4096x4096x4096`.

Both sides were restricted to physical GPU 6, ran serial GPU stages, skipped the
probe, and retained 5 rounds, 1.0 second cooldown, Proton warmup/repeat defaults,
both implementations, correctness/reference setup, raw samples, and arithmetic
mean. An outer monotonic timer measured the complete command:

| path | wall | result |
|---|---:|---|
| one-stage `a91a1b7` | 95.712s | 3/3 ok, no interference |
| pipeline | 50.792s | 3/3 ok, no interference |

For this fixed three-workload matrix, the pipeline is 1.884× faster and reduces
wall time by 46.9%. The pipeline's internal model measured 44.237s from scheduler
start through final RESULT: first READY 6.208s, GPU list schedule 37.980s, no
READY starvation or foreign wait, dispatch p95 0.036s, and unexplained residual
0.049s. The additional outer-command time is startup/provenance/report work that
both complete commands include.

The tracked artifact `bench_pipeline_ab_evidence.json` retains both sides' five
raw samples, protocol, exact source-artifact SHA-256 values, after-side timeline
offsets, and cost-model fields. Its derived values are independently
recomputable without relying on gitignored `.bench-suite/` state. The after-side
provenance intentionally remains `b427de3b-dirty`, the working tree that was
actually measured; it is not relabeled as a later cleanup commit.

A prior back-to-back pair on the same matrix also showed the pipeline win. Its
old/new kernel ratios differed by -0.75%, -0.55%, and -0.13% for the 1024, 2048,
and 4096 shapes. In the precisely outer-timed pair, the 1024 and 4096 ratio deltas
were +0.04% and +0.11%; the migration-before 2048 reference had one 34.329µs raw
sample while its other four samples were 15.977–16.011µs, so that row's aggregate
ratio is explicitly treated as an outlier-contaminated measurement rather than
evidence of a pipeline semantic shift. No samples were trimmed or replaced.

This A/B isolates orchestration by using the same three workload identities on
both sides. The independent default-coverage change from 234 to 112 workloads is
not included in this speedup. It reduces routine sweep work by 52.1% at the YAML
configuration source and must be reported as a separate coverage/runtime effect,
never multiplied into or merged with the 1.884× pipeline result.

### Single-GPU timer-family evidence ledger

The plan requires a migration-before versus pipeline A/B for every timer family
that can run on one GPU. Only Proton currently has complete runtime evidence;
the other rows remain explicitly unmeasured rather than being inferred from
their structural tests.

| timer family | runtime evidence | status |
|---|---|---|
| Proton | Fixed three-GEMM default-protocol A/B above, with tracked raw samples and timelines | measured, targeted A/B passed |
| Event | The one-stage side completed at the default protocol; three fresh pipeline attempts were invalidated by foreign PIDs before a valid result, and a fourth was cancelled | pipeline side missing due shared-machine interference; no A/B claim |
| CUDA-graph Proton | Not launched after repeated Event-timer interference established that the shared machine was unsuitable for another targeted run | unmeasured due shared-machine interference |
| Kineto | Correlated-span, barrier, sample-wise-max, schema, and cleanup behavior pass structurally; the runtime path also requires the locked NCCL/cuBLAS/cuBLASMp/NVSHMEM environment | structural only; runtime A/B unmeasured |
| MegaMoE | Alternating order, per-rank samples, sample-wise max, mismatch rejection, and cleanup pass structurally | structural only; runtime A/B unmeasured due shared-machine interference |

The completed one-stage Event run and the interfered pipeline attempts exist
only in gitignored local run logs, so they are not presented as persistent
review evidence. After three independently spawned attempts encountered foreign
processes, further GPU measurement stopped in accordance with the shared-machine
discipline. No reduced rounds, cooldown, timer budget, reference coverage, or
correctness work was used to manufacture a result.

### DeepGEMM strict-cache runtime evidence boundary

A default-protocol single-GPU attempt was made for
`deepgemm_sm100_m_grouped_fp8_gemm_masked/g32_m192_n6144_k7168`. It completed
CPU prepare, reported READY, received a one-GPU assignment, and then failed
before launch construction because the installed `deep_gemm 2.3.0+35c4bc8`
package does not export `cast_back_from_fp4` or `cast_back_from_fp8` from
`deep_gemm.utils.math`, while the existing reference-data builder imports those
helpers. This is an external reference-package API mismatch, not evidence that
the strict replay path passed or failed at launch.

The local partial-run artifact's cost model was correctly published as
`measurement_status: missing`, with 0 complete GPU timelines out of 1 record and
no `expected_s`, `unexplained_s`, starvation, foreign-wait, or latency fields.
This diagnostic artifact is gitignored and is not claimed as persistent review
evidence. No replacement dequantization algorithm was introduced solely to
manufacture runtime evidence.

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

## Acceptance ledger

| criterion | status | evidence boundary |
|---|---|---|
| AC-1 | satisfied | Unified process-local prepare/run-GPU contract, CUDA prepare guards, serialization rejection, and standalone composition tests |
| AC-2 | satisfied | Dedicated protocol, late assignment, pre-run CUDA recheck, assignment cardinality/index/duplicate rejection, and no-assignment no-GPU tests |
| AC-3 | satisfied | Bounded one-shot concurrency/backlog, condition-driven dispatch, same-GPU serialization, logical multi-GPU concurrency, and dynamic eligibility tests |
| AC-4 | satisfied | Default 5 rounds/1.0s unchanged, baseline-equivalent finalization, raw-sample/schema checks for every timer family, and tracked Proton A/B |
| AC-5 | satisfied | State-aware fail-fast, Ctrl-C cleanup, fresh-process interference retry, process-group/PID/temp-dir/GPU-claim cleanup tests |
| AC-6 | satisfied | Bounded process/RSS/FD evidence, cancellation cleanup, immediate internal release, and resource accounting tests |
| AC-7 | satisfied | Complete timeline validation, no-data cost-model gating, diagnostic-protocol watermarking, and recomputable tracked cost model |
| AC-8 | satisfied | Canonical `KERNEL_META` exact-load index, runtime metadata validation, duplicate rejection, cache invalidation, and all-config resolution gate |
| AC-9 | satisfied for migration and structural coverage | 41/41 adapters and 992/992 configs pass the pipeline-only gate; one-stage execution is removed; multi-GPU runtime remains separately exempted |
| AC-10 | incomplete | Proton targeted A/B passes the wall-time/residual/ratio requirements; Event, CUDA-graph Proton, Kineto, and MegaMoE runtime A/B evidence remains unmeasured as itemized above |
| AC-11 | satisfied | 112 defaults, all 992 configs retained, and 33/33 reviewed three-point selections with YAML-owned small/medium/large roles and rationale |

The plan as a whole is therefore not marked complete: AC-10 still requires the
missing single-GPU timer-family runtime evidence on a suitable machine. The
multi-GPU runtime rows are outside that remaining requirement because their
status is the explicit human-directed exemption, not missing evidence.

## Engineering-principles audit

- **Occam's razor:** the implementation uses one one-shot child lifecycle and
  two explicit replay mechanisms (generic lazy replay and strict keyed replay);
  it introduces no resident workers, reusable pools, fork templates, or compile
  thread pools.
- **Single source of truth:** kernel identity comes from `KERNEL_META`, complete
  config/default/selection metadata comes from the kernel YAML, and measurement
  defaults remain owned by the runner. Reports and gates derive from those
  authorities rather than maintaining competing manifests.
- **No slop:** the one-stage path is deleted, every registered adapter is
  accounted for, selection audit metadata is stripped before execution, and
  missing, exempted, diagnostic, and measured evidence are distinct states.
- **Broad understanding and independent evidence:** the retained tests protect
  lifecycle, scheduling, measurement, cleanup, and inventory contracts; tracked
  artifacts preserve the reproducible wall-time A/B and CPU resource evidence.
- **Optimize the real objective:** the retained performance result is complete
  command wall time on a fixed workload/GPU/protocol matrix, not a proxy such as
  process count or compiler concurrency.
- **Cost model and falsifiability:** expected critical time is reconstructed
  from first READY plus GPU scheduling, with foreign wait and residual separate;
  incomplete timelines publish no numeric performance fields. Only the
  reproducible Proton A/B win is retained as a performance conclusion.
- **Stop low-quality experiments:** Event-timer validation stopped after three
  fresh attempts were independently interfered with; the remaining rows are
  reported as unmeasured instead of weakening the protocol or claiming success.

## Remaining evidence boundary

- The implementation, static all-config migration gate, no-card structural
  tests, CPU-only prepare evidence, tracked default-protocol orchestration A/B,
  and generic lazy-replay single-GPU execution are complete.
- Proton has complete targeted runtime A/B evidence. Event has only a completed
  before-side run, while CUDA-graph Proton, Kineto, and MegaMoE lack valid
  runtime A/B evidence; AC-10 and the overall plan remain incomplete.
- The five DeepGEMM adapters have strict key/consumption behavior tests and
  CPU-only READY evidence; their real GPU replay attempt is blocked before
  launch by the installed DeepGEMM reference API mismatch and is not marked
  passed.
- Multi-GPU runtime behavior remains intentionally unmeasured.
- The full 112-workload measured sweep and baseline promotion remain deferred
  until the shared machine is available; they were not required or run here.
