# Bench Suite Pipeline Validation

Validation date: 2026-08-14.

This report records implementation and acceptance evidence for the one-shot
benchmark pipeline. It deliberately distinguishes static migration, structural
tests, invalidated or missing single-GPU evidence, and the human-directed
multi-GPU runtime exemption.

## Current implementation status

- Every workload owns one fresh process. It performs CPU prepare exactly once,
  then one or more GPU attempts in that same process before terminal result and
  exit. Processes that have run a GPU stage are never reused for another workload.
- The lifecycle follows
  `PREPARING → READY → ASSIGNED → RUNNING_GPU → RESULT_READY → RESULT → exit`;
  interference returns the same child from GPU cleanup to READY.
- The first verified assignment establishes process-level GPU affinity. An
  in-place retry releases ownership but can only reacquire that exact atomic GPU
  set; other READY workloads remain free to use the other eligible cards.
- CPU prepare owns exact module import/parsing, config resolution,
  specialization, IR generation, compilation, and any compile-cache population.
- GPU assignment is late-bound after READY. The prepared executable remains in
  its creating process and cannot be serialized or executed from another PID.
- Prepare children keep all physical GPUs visible. ASSIGN selects the physical
  ordinal with `torch.cuda.set_device()`, verifies exact UUID identity, and
  rejects CUDA contexts on cards never assigned to the process. Live-process
  `CUDA_VISIBLE_DEVICES` mutation is not used for assignment.
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

## Set-device decision and reachability audit

The implemented design keeps all physical GPUs visible in the prepare child,
so logical and physical ordinals are identical. CPU prepare must still leave CUDA
uninitialized. After each ASSIGN, the child must call
`torch.cuda.set_device(physical_index)` and prove the current physical device with
the existing UUID handshake before allocation or launch. Mutating
`CUDA_VISIBLE_DEVICES` in a live prepared process is no longer permitted.

For interference handling, a workload owns one one-shot process and performs CPU
prepare exactly once. After an interfered GPU attempt, the orchestrator releases
the old atomic claim but retains its exact GPU-set identity as process affinity.
The child waits until that same complete set becomes eligible, selects it again,
and rebuilds all GPU-side tensors, references, workspaces, and timer state without
repeating import, specialization, generation, or compilation. The run artifact
must mark `retry_in_place: true` and preserve per-attempt assignment, UUID,
ownership, and phase records.

This affinity is required by reachable external-runtime behavior, not by one
workload special case. The inspected DeepGEMM runtime has process-global caches:
`csrc/jit/cache.hpp` keys `KernelRuntime` only by JIT directory, while the cached
runtime owns a CUDA module/function loaded in the first device context; its
process-global `DeviceRuntime` also owns a cuBLASLt handle, CUDA workspace, and
cached device properties. No public reset is available. Moving the same prepared
process between physical devices can therefore reuse invalid context-bound
objects. Exact-claim retry turns that external assumption into one scheduler
invariant while preserving dynamic first assignment.

A retry occurs in a process that has already executed a GPU attempt. The human
ruled that this does not create a separate evidence class without measured proof
of a distinguishable hot-process effect. In-place retry results remain ordinary
measurement evidence and may enter clean AC-10 A/B. Each record must still mark
`retry_in_place: true` so accumulated same-config data can later compare retried
and first-attempt measurements without running a dedicated experiment.

The first external-library audit found these device-0 source matches in the
installed FlashInfer package:

- `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/comm.py:84` calls
  `torch.cuda.set_device(0)` in the real single-rank `bootstrap_dist()` path, and
  line 89 constructs `Device(0)`;
- `shim/nvfp4.py` and `shim/mxfp8.py` call that helper;
- `src/moe_mxfp8_glu/mega_runner.py:891`,
  `src/moe_nvfp4_swapab/mega_runner.py:2736`, and
  `src/moe_nvfp4_swapab/benchmark_p2p.py:512` also call
  `torch.cuda.set_device(0)` on executable runner/benchmark paths, followed in at
  least one path by bare `device="cuda"` allocations.

The human review corrected the audit criterion from source presence to runtime
reachability. AST inspection confirms every binding above is inside a function;
the runner matches are CLI/benchmark/debug entry points, `flashinfer.__init__`
does not import `moe_ep`, and this repository has no `moe_ep` or
`cutedsl_megamoe` import or call. They are therefore not blockers and must not be
presented as such. The durable rule is: a blocker requires evidence of who
imports and calls the binding. Docstrings/examples, standalone CLI or benchmark
scripts, and unimported subpackages do not qualify. A module-level binding or a
function on a bench workload's real call graph does qualify; once confirmed, it
must be reported without modifying the external library, adding a workaround,
or reverting to masking.

The continued audit found a reachable prerequisite for the intended DeepGEMM
MegaMoE dependency:

```text
PreparedMegaMoeBench.run_gpu
  -> tirx_kernels.deepgemm.mega_moe._run_distributed
  -> tirx_kernels.deepgemm.mega_moe._run_worker
  -> deep_gemm.utils.dist.init_dist(local_rank, num_processes)
  -> torch.cuda.set_device(local_rank)
```

The repository identifies this port against DeepGEMM `559d79fb`; the evidence is
from the out-of-tree pinned dependency copy at
`/home/hongyij/workspace/tirx-kernels/.porting/deps/deep_gemm-559d79fb/deep_gemm/utils/dist.py:33`,
which calls `torch.cuda.set_device(local_rank)`.
For a one-rank workload assigned physical GPU 6, `local_rank` remains 0 and the
dependency switches execution to GPU 0. This is a real call edge from the bench
workload, not a grep-only finding. Under the superseded mask implementation this
was latent rather than an existing correctness bug: device 0 was the masked
assigned card. It became material when the all-visible
`set_device(physical_index)` design was adopted.

The base interpreter's `deep_gemm` package lacks `fp8_fp4_mega_moe`, so an
unpinned local run raises `SkipTest` before the code can reach `utils.dist`.
This is an earlier API mismatch, not evidence that the intended dependency is
safe. The locked benchmark environment instead resolves `deep_gemm` from
`venv-bench-sglang96a04cb-nccl4py031-cublasmp010`, where
`fp8_fp4_mega_moe` and `utils.dist` are both present; that is the environment
required for runtime evidence. The MegaMoE YAML has two default single-GPU entries,
`t64_m64_h7168_i3072_e384_k6_g1` and
`t8192_m8192_h7168_i3072_e384_k6_g1`, so the path participates in the 109-item
default sweep.

The human approved a repository-owned call-site fix. The target keeps
`init_dist()` and its returned group, separates logical distributed rank from the
assigned physical device, restores the assigned device immediately after the
external call, and validates its UUID before case construction and timing. More
generally, every external call that can change current device is followed by the
same position invariant. A reachable override is blocking only if correction
requires modifying external source, monkey-patching, or reverting to masking.
None of those prohibited mechanisms is used.

Separately, `_run_distributed()` still acquires a TCP rendezvous port and creates
a one-rank process group when `num_processes == 1`, with the existing 32-attempt
EADDRINUSE retry. This is a known deferred overhead and is not changed here.

The remaining external audit found no runtime AST hard-coded device-0 calls in
the imported `deep_gemm`, `flash_attn`, CUTLASS DSL, or `flash_kda` Python trees.
That statement is scoped to the inspected Python call sites and is not a GPU
runtime validation claim.

Three repository no-op calls were also inspected and left unchanged:

- `deepgemm/tf32_hc_prenorm_gemm.py:285`;
- `deepgemm/mqa_logits_fp4.py:217`;
- `deepgemm/mqa_logits_fp8.py:229`.

Each calls `torch.cuda.set_device(torch.cuda.current_device())` at the start of
GPU `prepare_data()`, before capability checks and bare CUDA allocations. They
date to the original import commit, but no explanatory commit context was found.
Their intent is therefore recorded as unresolved rather than deleted or used as
evidence that the new design works.

## Static and no-card evidence

The capability audit reports:

- 49 registered kernels.
- 1180 module-owned benchmark configs.
- 1180 YAML inventory entries, with no module-only or YAML-only config.
- 27 generic lazy-replay adapters checked at AST level.
- 11 strict-cache adapters with exact prepared-cache key and consumption
  validation.
- 11 explicit/custom adapters covering process-local executables, dispatcher
  delegation, hardware-profile compile caches, and distributed export/load
  lifecycles; together the three adapter classes account for 49/49 kernels.
- 133 single-GPU default workloads, with at most three defaults per kernel.
- 40 current curated default selections, each exactly three points
  with its rationale stored beside the canonical `default` flags in the same
  kernel YAML and emitted by the capability report.
- 27 multi-GPU configs, none selected by the default measured sweep.

Final post-rebase non-GPU verification on 2026-08-14 passed Ruff, Python bytecode
compilation, all 106 protocol/evidence tests, the full capability audit above,
and `git diff --check origin/main...HEAD` over source/document paths. The diff
check excludes byte-preserved raw artifact trees: several captured failure logs
contain upstream/runtime trailing whitespace that must remain unchanged as
evidence. No benchmark workload or multi-GPU runtime was launched for this
verification pass.

The protocol suite covers:

- one-shot process/IPC behavior, log isolation, cancellation, fail-fast, dynamic
  external eligibility, same-child GPU retry without repeated prepare, bounded-backlog
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
- strict prepared-cache replay for both DeepGEMM forms: the five
  `compile_spec`/`build_launch` users and the five direct custom-compiler users.
  Mismatched keys, missing consumption, extra consumption, and specialization
  drift all fail before a GPU-stage compilation can enter the critical path;
- a runner-level GPU-stage guard rejects any `tvm.compile` after assignment,
  closing cache-miss fallback for every in-process adapter rather than relying
  only on per-adapter discipline;
- missing/incomplete cost-model evidence never publishes expected wall,
  residual, starvation, foreign-wait, or latency values as zero;
- summaries watermark non-default rounds/cooldown as diagnostic, including old
  run JSON whose protocol must be derived from its result rows;
- standalone `run_kernel_bench()` composes the same prepare/run-GPU contract;
- timeline validation rejects missing transitions, reversed timestamps, and
  overlapping ownership intervals on the same GPU;
- capability accounting proves 49/49 adapters and 40/40 reviewed three-point
  selections from canonical sources.

The current additions also exercise physical-UUID lookup without context creation,
UUID mismatch rejection, restoration after an external device override, rejection
of contexts on never-assigned cards, exact per-attempt ownership ordering,
RESULT_READY acceptance/retry, and per-record `retry_in_place` provenance. The
targeted runtime evidence below validates one real single-GPU set-device path and
one controlled same-child card-switch retry. It does not validate every kernel or
replace the still-missing migration-before AC-10 A/B.

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

The five newly migrated direct custom-compiler adapters were separately run in
fresh children through READY and then cancelled without assignment:

| workload | wall to READY | framework import | module load | specialize/generate/compile | result |
|---|---:|---:|---:|---:|---|
| `deepgemm_sm100_tf32_hc_prenorm_gemm/m64_n24_k28672_s112` | 4.702s | 2.245s | 0.716s | 1.705s | READY, clean CANCEL |
| `deepgemm_sm100_fp4_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | 4.338s | 2.216s | 0.287s | 1.801s | READY, clean CANCEL |
| `deepgemm_sm100_fp8_mqa_logits/s2048_skv4096_h64_d128_f32_dense_cp` | 4.388s | 2.226s | 0.299s | 1.828s | READY, clean CANCEL |
| `deepgemm_sm100_fp4_paged_mqa_logits/b1_n1_mp1_ps32_h64_d128_f32_fixed` | 4.591s | 2.339s | 0.324s | 1.893s | READY, clean CANCEL |
| `deepgemm_sm100_fp8_paged_mqa_logits/b1_n1_mp4_ps64_h64_d128_f32_fixed` | 4.227s | 2.215s | 0.340s | 1.636s | READY, clean CANCEL |

`bench_pipeline_custom_cache_prepare_evidence.json` records the exact values and
their boundary. The timings were transcribed from command output and have no
persisted raw run JSON or log, so this is a repository CPU-prepare ledger rather
than independently rehashable source evidence. It proves neither prepared replay
after assignment, GPU correctness/runtime, nor AC-10 performance.

The TF32 HC prepare initially exposed an offline compiler defect: passing only
`nvcc -arch sm_100a` selected a non-family intermediate target and `ptxas`
rejected `tcgen05` instructions. Offline materialization now emits
`-gencode arch=compute_100a,code=sm_100a` (and the analogous `sm_100f` form),
preserving the Blackwell family architecture without initializing the driver.

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

### Current set-device and in-place-retry evidence

`bench_pipeline_set_device_evidence.json` tracks the acceptance-relevant raw
fields and exact SHA-256 hashes for two local, gitignored run JSONs. These are
implementation/runtime checks for the new path, not a migration-before A/B.

The fresh run used
`fp16_bf16_gemm/fp16_1024x1024x1024` on physical GPU 2, UUID
`GPU-f8a4f1df-8b46-4cbf-3244-a33b90e06aa9`:

- `retry_in_place: false`, one GPU attempt, status `ok`;
- TIRx 6.899µs and torch-cuBLAS 6.010µs, with all five raw samples retained;
- first READY 5.203s, dispatch latency 0.040s, ASSIGN-to-GPU-start 0.925s;
- GPU ownership/list-schedule interval 12.645s;
- critical wall 18.814s, unexplained 0.965s, and final reap tail 0.616s;
- complete outer scheduler/process wall 19.430s.

This proves real all-visible `set_device` binding, UUID verification, canonical
measurement, and the RESULT_READY/ACCEPT_RESULT handoff. The unexplained value is
slightly above the plan's `max(0.5s, 5%)` target for this one-workload run, and
there is no paired migration-before artifact, so it is not promoted to AC-10
performance evidence.

The historical controlled retry run used the same workload. A synthetic monitored intruder
forced attempt 1 to stop on physical GPU 2 and attempt 2 to run on physical GPU 1,
UUID `GPU-e8754e6d-624e-e1d0-595a-f9444588960a`:

- both attempts used child PID 54244, with only one prepare timeline and one log;
- the old claim was released before the second assignment, and the process never
  held both cards simultaneously;
- the record has `attempt: 2` and `retry_in_place: true`; retried results remain
  ordinary measurement evidence under the human ruling;
- final TIRx was 6.877µs and torch-cuBLAS 6.016µs, with five raw samples each;
- critical wall was 18.762s, unexplained 0.311s, and outer wall 19.529s;
- descendant and forced-kill lists were empty for this local GEMM.

This run predates the exact-claim retry invariant. It remains evidence for
same-child prepare reuse, provenance, cleanup, and the cost of abandoning a
context, but it is not evidence for the current retry assignment policy.

After attempt-1 cleanup and again after reassignment, the abandoned GPU 2 kept
645,922,816 bytes (exactly 616 MiB) resident in the still-live workload process.
This is the requested primary-context cost of an in-process card switch. Its
acceptability is intentionally left to the human. After the child exited, a
read-only query at 2026-08-14T00:34:37Z showed GPU 2 at 124 MiB, 0% utilization,
and no listed compute PID, so the measured 616 MiB context was no longer resident.
GPU 1 showed unattributed host memory with no compute PID; it is recorded only as
a host observation and is not attributed to this run.

The fresh and retry measurements finish on different GPUs, so their kernel times
must not be used as a controlled hot-versus-fresh comparison. The provenance flag
exists so future same-config accumulated data can make that comparison without a
dedicated experiment.

The historical three-small-GEMM pipeline run produced the following internal
timeline. It predates physical-UUID verification, so its requested GPU index is
not proof of the physical device used and the numbers are retained only as
directional cost-model data:

- observed critical wall: 46.787s;
- first READY: 7.087s;
- one-GPU list schedule: 39.635s;
- READY starvation and foreign wait: 0s;
- dispatch p95: 0.056s;
- unexplained residual: 0.065s (0.14% of observed wall).

The older prototype evidence—29.70s sequential versus 19.17s pipelined, 1.55×—
is likewise retained only as directional cost-model evidence. The historical
same-protocol pair below is also invalidated for performance acceptance.

After the generic lazy-replay correction, one small `act_and_mul` workload was
recorded after requesting an automatically selected idle B200. This also
predates the UUID handshake; the scheduler's idle-card selection cannot prove
which physical GPU the prepared child ultimately used, so this is not accepted
single-GPU runtime evidence:

- CPU prepare: 4.082s;
- GPU stage: 14.918s;
- observed/expected critical wall: 19.085s / 19.037s;
- unexplained residual: 0.047s;
- dispatch p95: 0.040s;
- TIRx/FlashInfer: 2.606µs / 2.789µs;
- 5 raw samples per implementation, Proton timer, 1.0s cooldown;
- no interference retry;
- host peaks: 3 processes, 1.999 GiB RSS, 218 file descriptors.

### Invalidated historical default-protocol A/B

The migration-before side was checked out in a detached worktree at `a91a1b7`,
which still contains the one-stage scheduler. Both sides used exactly these
three explicit workloads:

- `fp16_bf16_gemm/fp16_1024x1024x1024`;
- `fp16_bf16_gemm/fp16_2048x2048x2048`;
- `fp16_bf16_gemm/fp16_4096x4096x4096`.

Both commands requested GPU index 6, ran serial GPU stages, skipped the probe,
and retained 5 rounds, 1.0 second cooldown, Proton warmup/repeat defaults, both
implementations, correctness/reference setup, raw samples, and arithmetic mean.
The old pipeline path did not persist or verify the physical UUID. It has since
been reproduced mapping `assigned_env=6` to logical device 0 UUID
`GPU-ef5a8300-...` (NVML index 0), while NVML index 6 is
`GPU-e56ad157-...`. The pipeline side's actual physical identity is therefore
unknown, and the premise that both sides used the same physical GPU is false as
an evidence claim. The recorded outer monotonic values were:

| path | wall | result |
|---|---:|---|
| one-stage `a91a1b7` | 95.712s | 3/3 ok, no interference |
| pipeline | 50.792s | 3/3 ok, no interference |

The quotient is 1.884× and the arithmetic reduction is 46.9%, but this is now
classified as an internally reproducible historical calculation, not a valid
migration speedup. It must not be used for AC-10 or any default-protocol
performance claim until both sides are remeasured on the same UUID-verified idle
GPU. The pipeline's internal model remains internally checkable: 44.237s from
scheduler start through final RESULT, first READY 6.208s, GPU list schedule
37.980s, no READY starvation or foreign wait, dispatch p95 0.036s, and
unexplained residual 0.049s.

The tracked artifact `bench_pipeline_ab_evidence.json` retains both sides' five
raw samples, protocol, recorded source-artifact SHA-256 values, after-side
timeline offsets, and cost-model fields. Its derived arithmetic is internally
recomputable from fields in that same JSON; that is not independent provenance
for those inputs. Both hashes name gitignored run JSONs. They match the copies
currently present in this workspace, but the before worktree/artifact was absent
in the third-party audit environment and neither source artifact is tracked.
Artifact-backed verification must therefore explicitly skip when a local file is
unavailable, never silently pass. The two outer-wall inputs exist only in the
tracked evidence JSON: the after run JSON supports 44.237s critical wall plus
0.616s final reap tail, not the additional 5.939s needed to reach 50.792s. Thus
the headline quotient has no persisted raw outer-timer artifact on either side.
The after-side provenance intentionally remains `b427de3b-dirty`, the working
tree that was actually measured; it is not relabeled as a later cleanup commit.

A prior back-to-back pair on the same matrix also showed the pipeline win. Its
old/new kernel ratios differed by -0.75%, -0.55%, and -0.13% for the 1024, 2048,
and 4096 shapes. In the precisely outer-timed pair, the 1024 and 4096 ratio deltas
were +0.04% and +0.11%; the migration-before 2048 reference had one 34.329µs raw
sample while its other four samples were 15.977–16.011µs, so that row's aggregate
ratio is explicitly treated as an outlier-contaminated measurement rather than
evidence of a pipeline semantic shift. No samples were trimmed or replaced.

The historical pair used the same three workload identities, but it does not
isolate orchestration because same-device identity is unverified. The independent
default-coverage change from 234 to 112 workloads is not included in the recorded
quotient. It reduces routine sweep work by 52.1% at the YAML configuration source
and must be reported as a separate coverage/runtime effect, never multiplied into
or merged with any future UUID-verified pipeline result.

### Single-GPU timer-family evidence ledger

Proton has a persisted same-physical-GPU schema-3 pair satisfying every named
structural AC-10 check. Its earlier retry-heavy attempt is retained as diagnostic
evidence, not substituted for the passing pair. Event and CUDA-graph Proton have
source-reproducible UUID-verified pairs but no speedup claim. Kineto and MegaMoE
remain missing/unmeasured through the new UUID-verified path. Kineto is more
specifically classified as missing because the pinned external TVM NVSHMEM
runtime violates the assigned-device invariant before timing. By the human scope
decision of 2026-08-14, no additional timer-family GPU collection is required;
the ledger states remain unchanged rather than being promoted to pass.

The reproducible inputs are tracked as `bench_pipeline_ac10_workloads.yaml`
(Proton), `bench_pipeline_ac10_event_workload.yaml`,
`bench_pipeline_ac10_cudagraph_workload.yaml`,
`bench_pipeline_ac10_kineto_workload.yaml`, and
`bench_pipeline_ac10_megamoe_workload.yaml`. After collection,
`scripts/build_bench_pipeline_ac10_evidence.py` must open and hash both raw run
JSONs, both independent outer timers, and their stdout/stderr logs. It rejects
missing sources, cross-side UUID differences, non-default protocols, incomplete
samples/timelines, non-reproducible means or cost fields, and does not emit an
evidence file on rejection. A hand-filled but arithmetically consistent summary
is therefore insufficient for the replacement AC-10 evidence.

| timer family | runtime evidence | status |
|---|---|---|
| Proton | On physical GPU 2, UUID `GPU-f8a4f1df-8b46-4cbf-3244-a33b90e06aa9`, migration-before completed in 71.602s and schema-3 pipeline in 56.415s: 1.2692× / 21.21% measured wall improvement. Both sides retained 3/3 default-protocol records and all raw samples; implementation means changed by -0.81% to +0.06%. Pipeline critical wall was 46.575s versus 46.528s expected, leaving 0.047s unexplained; dispatch p95 was 42.8ms, CPU READY starvation 0, and retries 0 | measured, reviewable, and passes the Proton AC-10 checks |
| Event | On physical GPU 2 with the same UUID as Proton, migration-before completed in 24.642s and pipeline in 37.451s (0.6580×). The pipeline result had four explicitly recorded in-place retries caused by foreign PIDs, so no speedup is claimed. Its critical wall was 30.651103s versus 30.651093s expected with foreign wait, leaving 10µs unexplained; dispatch p95 was 59.2ms and CPU READY starvation 0. TIRx changed by -0.34%; FlashInfer's -10.61% mean shift is explained by one 9.800µs first-round baseline outlier while its other four baseline samples were 6.214–6.239µs | measured and source-reviewable; structural cost-model checks pass, but not a reproducible wall-time win |
| CUDA-graph Proton | On physical GPU 2 with the same UUID, migration-before completed in 33.356s and pipeline in 37.677s (0.8853×). The pipeline result had two explicitly recorded in-place retries, so no speedup is claimed. Its critical wall was 28.963493s versus 28.963464s expected with foreign wait, leaving 28.8µs unexplained; dispatch p95 was 40.2ms and CPU READY starvation 0. TIRx changed by +1.77% and FlashInfer by +0.08%, within the retained raw-sample spread | measured and source-reviewable; structural and implementation-ratio checks pass, but not a reproducible wall-time win |
| Kineto | Migration-before completed on physical GPU 2 in 56.722s with zero retries and five raw Kineto samples for each implementation. Pipeline could not enter timing: pinned TIR `ea0950ab` calls `cudaSetDevice(worker_id)` and `cudaSetDevice(mype_node)` in `src/runtime/extra/contrib/nvshmem/init.cc:64,68`; for one rank both are 0, creating a never-assigned physical GPU-0 context while the scheduler assigned GPU 2. The UUID/context guard correctly failed. An earlier attempt reached `nvshmem_finalize` and aborted with an invalid context, which led to the same root cause. | `missing`, with tracked before and failure artifacts; completing it requires an external TVM change or a prohibited mask/monkey-patch, so no A/B or numeric cost model is published |
| MegaMoE | Alternating order, per-rank samples, sample-wise max, mismatch rejection, and cleanup pass structurally. The representative default config now also completes both CPU-only stats/no-stats compilations for the architecture-specific `sm_100a` target without initializing CUDA | structural and CPU-prepare evidence only; runtime A/B unmeasured |

The older Event and CUDA-graph runs existed only in gitignored local logs and did
not prove physical identity. The new tracked pairs supersede both old results.
No reduced rounds, cooldown, timer budget, reference coverage, or correctness
work was used to manufacture a result.

The tracked passing Proton pair lives under
`bench_pipeline_ac10_artifacts/proton/{before-gpu2,after-gpu2}/`, with its derived
record in `evidence-gpu2-schema3.json`. The builder hashes and reopens both raw run
JSONs, both outer timers, and all four outer logs. Each outer snapshot records
110 MiB, zero utilization, no compute process, and the same UUID before and after
the command. The after run's source-tree fingerprint exactly matches
`0400e58:tirx_kernels`; its `-dirty` label comes from untracked run artifacts, not
source drift.

The corresponding Event sources live under
`bench_pipeline_ac10_artifacts/event/{before-gpu2,after-gpu2}/`, with the same
derived evidence filename and independently hashed raw sources. Its after
source-tree fingerprint exactly matches `a5abeca:tirx_kernels`; its `-dirty`
label likewise reflects untracked run artifacts rather than source drift.
CUDA-graph follows the same layout under `bench_pipeline_ac10_artifacts/cudagraph/`;
its after source-tree fingerprint exactly matches `6416bb6:tirx_kernels`.
Kineto's explicit missing record is
`bench_pipeline_ac10_artifacts/kineto/evidence-missing.json`. It hashes the
successful migration-before sources, both pipeline failure attempts, and the
host-local pinned TVM source that couples rank 0 to CUDA device 0. It contains no
wall-speedup, residual, latency percentile, or expected-wall value.
The same record preserves the second API audit: no registered TVM full-init
entry point separates PE rank from device ordinal; `worker_id_start` is also
`myrank`; `mype_node` restores device 0; and NVSHMEM's public full-init wrapper is
header-inline over a non-exported implementation symbol. The exported hostlib
initializer is not the complete device initialization required by these kernels.
A third audit inspected every local TVM ref and worktree plus alternative
workspace/TIR checkouts. The only TVM worktree is detached at `ea0950ab`, and no
available revision or second implementation separates PE rank from physical
device ordinal. There is therefore no existing dependency version to select as
a repository-local resolution.

The earlier GPU-1 pair remains in `evidence-attempt-1.json` because it exposed two
cost-model defects rather than passing them silently. It is not the Proton
acceptance result.

This pair is deliberately retained even though it is not an acceptance pass.
All three workloads initially reached READY within 33ms, yet the persisted
schema-1 cost model reports 8.329s of `ready_starvation_s`. Inspection shows that
delay comes from retry readiness and ASSIGN-to-GPU_START ownership after seven
interrupted GPU attempts, not repeated CPU prepare. Recomputing the same raw
attempts with schema 3 reports `ready_starvation_s = 0`,
`interference_retry_ready_delay_s = 4.889s`, 53.208s of claim card-time versus
49.764s after `GPU_START`, and a 0.660s unexplained residual (4ms below the
schema-1 result because the claim-based list schedule preserves the exact
ASSIGN ordering). Schema 3
also records transient foreign-PID
intervals and subtracts their overlap from internal dispatch latency; attempt 1
predates that telemetry, so its 1.873s dispatch p95 cannot be retroactively
reclassified. The implementation and behavioral tests are corrected; the later
passing Proton schema-3 artifact supplies the required measured evidence.

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

That failed attempt covers one of the original five
`compile_spec`/`build_launch` adapters. The five direct custom-compiler adapters
have the CPU-only READY→CANCEL evidence above, but none has UUID-verified GPU
runtime evidence. Neither subgroup is therefore marked as runtime-passed.

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
| AC-2 | satisfied for implementation and single-GPU targeted evidence | ASSIGN uses physical indices with `set_device`, exact per-attempt UUID proof, position validation after reachable external calls, and rejection of never-assigned-card contexts. Multi-GPU runtime remains explicitly exempted rather than passed |
| AC-3 | satisfied | Bounded one-shot concurrency/backlog, condition-driven dispatch, same-GPU serialization, logical multi-GPU concurrency, and dynamic eligibility tests |
| AC-4 | satisfied | Default 5 rounds/1.0s, finalization, raw samples, timer schemas, correctness/reference work, and evidence eligibility are unchanged. Every terminal record explicitly marks `retry_in_place` |
| AC-5 | implementation and in-scope structural evidence satisfied; Kineto runtime remains terminal missing | Same-child retry preserves prepared CPU state, rebuilds GPU state, and records exact attempt ownership. The pinned TVM NVSHMEM rank/device coupling creates an unassigned GPU-0 context and prevents the one-rank Kineto path from completing NVSHMEM/process-group cleanup; the 2026-08-14 scope decision removed that runtime collection from pending work without converting it to pass, and multi-rank runtime remains separately exempted |
| AC-6 | satisfied | Bounded process/RSS/FD evidence, cancellation cleanup, immediate internal release, and resource accounting tests |
| AC-7 | satisfied | Cost-model schema 3 separates initial CPU READY constraints, retry READY delay, ASSIGN-held card time, post-GPU_START execution, transient foreign-PID wait, and internal dispatch latency; complete-timeline/no-data gating remains intact. The clean Proton run measured 0.047s unexplained residual and 42.8ms internal dispatch p95 |
| AC-8 | satisfied | Canonical `KERNEL_META` exact-load index, runtime metadata validation, duplicate rejection, cache invalidation, and all-config resolution gate |
| AC-9 | satisfied for migration and structural coverage | 49/49 adapters and 1180/1180 configs pass the pipeline-only gate; one-stage execution is removed; multi-GPU runtime remains separately exempted |
| AC-10 | terminal evidence ledger closed by human scope decision | Proton has a passing persisted same-UUID schema-3 A/B. Event and CUDA-graph retain their honest non-winning measurements. Kineto remains explicitly missing because of the pinned external TVM rank/device coupling and MegaMoE remains unmeasured; the human decision of 2026-08-14 removed all remaining timer-family GPU runs from the required work queue without converting missing evidence into pass |
| AC-11 | satisfied | 133 defaults, all 1180 configs retained, and 40/40 reviewed three-point selections with YAML-owned small/medium/large roles and rationale; all 16 `gemm_reduce_scatter` configs remain explicitly runnable |

The set-device and same-child retry implementation is complete at its structural
anchors and has targeted single-GPU runtime evidence. Proton, Event, and
CUDA-graph retain the statuses stated above; no remaining timer family is a
required completion item. Kineto and MegaMoE stay missing/unmeasured rather than
being rewritten as pass, and multi-GPU runtime rows remain the explicit
human-directed exemption.

### Completion audit

The final in-scope implementation is complete. This conclusion is based on the
current tracked sources and independently executable gates, not on the absence
of open notes:

- `audit_pipeline_capabilities()` reports `static_pass`, `execution_mode:
  pipeline`, 49 kernels, 1180 module configs, 1180 YAML configs, 133 defaults, 40
  curated three-point selections, and 27 multi-GPU rows explicitly classified
  `exempted_by_human_unmeasured`;
- all 106 collected protocol/evidence tests pass, including lifecycle,
  assignment/UUID, no-CUDA prepare, strict cache consumption, all-config
  migration, measurement schema, cost-model no-data gating, raw artifact hash
  verification, and suite A/B recomputation;
- the license-header gate and its self-test pass, and the GDN no-tile structural
  lints pass for all 42 wide-vector T1 and all 67 ILP4 pre-dispatch
  specializations;
- repository searches find no legacy/fallback execution mode, reusable prepare
  pool, process pool, or compilation thread pool. The only
  `ThreadPoolExecutor` probes independent candidate GPUs in short-lived
  subprocesses before workload preparation; it does not specialize or compile
  workloads in the orchestrator interpreter;
- default rounds/cooldown remain owned once by `runner.py` as 5 and 1.0s, and
  every numeric evidence publisher reopens and hashes its raw sources.

The pinned-TVM Kineto failure, unmeasured MegaMoE/multi-GPU runtime, and old
invalidated measurements remain explicit terminal evidence states. They are not
passes, but the 2026-08-14 human scope decision removed them from pending work.

## AC-external full-suite speedup supplement

This supplement is outside AC-10 and every other acceptance criterion. The
historical 112-workload attempt remains in
`bench_pipeline_suite_speedup_evidence.json`: both sides fail-fast stopped, so it
correctly publishes no speedup. Its recorded 234-to-112 coverage context remains
historical evidence and is not rewritten.

The completed supplemental matrix is
`bench_pipeline_suite_workloads_106.yaml` (SHA-256
`4c3ad6acf86a89ef38aa6f10acee98452089fec5f38f5fffb89b397c2081177d`).
Its metadata is the single authority for all six exclusions:

- the three `gemm_reduce_scatter` TP1 configs
  (`tp1_m8192_n4096_k12288_fp16_dynamic`,
  `tp1_m8192_n8192_k28672_fp16_dynamic`, and
  `tp1_m8192_n16384_k53248_fp16_dynamic`) left the canonical default sweep after
  the migration-before runner hit the known NCCL illegal-instruction failure;
  this was not introduced by the pipeline migration;
- the three `allgather_gemm` TP1 configs
  (`tp1_m8192_n24576_k4096_fp16_dynamic`,
  `tp1_m8192_n57344_k8192_fp16_dynamic`, and
  `tp1_m8192_n106496_k16384_fp16_dynamic`) are excluded only from this
  supplement because pinned TVM couples NVSHMEM PE rank to physical CUDA device.
  The default YAML remains unchanged and the missing Kineto status is not
  promoted.

Both sides completed all 106 workloads with status `ok`, default 5 rounds and
1.0s cooldown, isolated cold caches, persisted outer timers, and one-second
all-GPU monitoring:

| side | outer wall | minutes | participating GPUs | retries |
|---|---:|---:|---:|---:|
| migration-before `a91a1b7` | 476.645473386s | 7.9441 | 8 / 8 | unavailable in the old artifact |
| pipeline | 363.171956028s | 6.0529 | 8 / 8 | 11 in-place retries |

The measured quotient is **1.312451x**, saving 113.473517358s or 23.8067% wall
time. `bench_pipeline_suite_speedup_106_evidence.json` hashes and reopens the
matrix, both raw run JSONs, both outer timers, and both stdout/stderr logs; it
requires all 106 rows to be `ok`, validates the default protocol, and recomputes
every implementation mean from five raw round samples. The raw before/after run
hashes are respectively
`0e60d794907a13649340d4a811e103007916fe9a298e179cc442b31f30d0de06`
and `089874f16e5e611ea46d8a2090ac5c103d95b386bd8e00f0ae5de66bd5cad14a`;
the outer-timer hashes are
`069fe3a680740edda8353ef3244b2eff9abdca57e7cb4113327369f3198d52c4`
and `ef44658404e1fc3ee8f1d98ab23f465170ba848d74c80c740a47a48d599016ea`.
The pipeline run reports `5429283d-dirty` because its source probe also sees
untracked measurement directories; tracked source had no diff when the command
started. This full-suite number therefore measures commit `5429283` before the
later NVFP4 cuBLASLt build-only move described below. No synthetic adjustment is
added to the 363.171956028s wall from the targeted follow-up.

The number requires four explicit caveats:

1. Both sides are cold-cache runs. Cold compilation increases the prepare share,
   which is exactly what the pipeline overlaps, so 1.312451x is an upper bound;
   normal warm-cache speedup should be lower.
2. It includes CPU/GPU pipeline overlap, but not multi-GPU worker parallelism.
   The migration-before scheduler already ran one worker per available card.
3. It is not combined, multiplied, or summarized with the independent
   234-to-112/109/106 coverage changes.
4. GPU eligibility differs even though both commands eventually used all eight
   physical UUIDs. The pipeline rejects unattributed resident VRAM above a
   512 MiB allowance; `a91a1b7` only considered utilization/compute-process
   evidence and could accept a card with resident memory but no listed compute
   process. Therefore the raw wall quotient is approximate real-machine evidence,
   not a fixed-runner microbenchmark.

The migration-before artifact has no equivalent phase timeline or card-time
cost model, so no card-time ratio is published. That evidence is unavailable,
not zero. This result remains supplemental and does not enter the AC ledger.

## Per-config phase and GPU-stage residual evidence

`bench_pipeline_suite_breakdown_106.md` and the schema-2 JSON evidence publish
all 106 pipeline phase rows: startup, CLI bootstrap, framework import, exact
import, config resolution, specialize/generate/compile, total CPU prepare,
READY wait, ASSIGN handoff, GPU stage, result handoff, and reap tail. They also
publish migration-before and pipeline per-implementation microseconds plus all
five raw round samples. The old runner cannot supply an equivalent phase split,
which is stated explicitly.

For triage, each row computes
`GPU stage - implementation_count * 5 * 1.0s`. This is only a cooldown lower-bound
residual: it still includes timer setup, correctness, allocation, loading,
warmup/repeat, and real GPU execution. Across the 106 rows, p50 is 6.258s, p90 is
20.208s, and the maximum is 142.011s. The eleven rows at or above p90 fall into
four source-identical groups:

- all three `nvfp4_gemm` configs: device tensors, quantization, FlashInfer
  backend loading/autotune, and a cuBLASLt PyTorch extension build were inside
  the claimed GPU window;
- the two `d_qk=576` sparse FlashMLA configs: the third TRT-LLM reference creates
  device KV storage, a 128 MiB workspace, block tables, and a tactic probe;
- the large single-rank MegaMoE config: a TCP rendezvous and one-rank process
  group are established after claim, followed by two device cases, barriers,
  correctness, and the fixed MegaMoE timer protocol;
- five selective-state configs: device cases plus first-use FlashInfer reference
  import/JIT, correctness execution, and reference warmup dominate the residual.

Only the NVFP4 cuBLASLt extension build is both device-independent and material.
It now builds before READY under `cuda_initialization_guard()` with explicit
`sm_100a` code generation; ASSIGN-time code consumes the exact keyed artifact and
only loads the shared library. A cold default-protocol targeted comparison added
12.582-12.584s to CPU prepare and saved 1.394s and 16.488s of GPU-stage wall on
the two clean configs. The 1024 config showed an apparent 82.753s reduction but
had 18 in-place retries, so that number is explicitly non-attributable. All
three configs retained five samples per implementation; clean per-implementation
mean changes stayed within 1.54%.

The no-GPU structural probe in
`bench_pipeline_prepare_cuda_import_evidence.json` explains why the remaining
large setup was not moved. Even with `FLASHINFER_CUDA_ARCH_LIST=10.0a`, importing
the reachable FlashInfer FP4 JIT, selective-state reference, or TRT-LLM decode
path changed CUDA initialization from false to true and was rejected by the
guard. FlashKDA peer import/provenance passed the guard but took only 24ms;
DeepGEMM MegaMoE import also passed at about 0.2s but is not portable to spawned
rank workers. Both are below the stopping threshold and the material remainder
is device/rank work. Full derivation and source hashes are in
`bench_pipeline_gpu_stage_cpu_work_evidence.json`.

## Engineering-principles audit

- **Occam's razor:** the implementation uses one one-shot child lifecycle and
  two explicit replay mechanisms (generic lazy replay and strict keyed replay);
  it introduces no resident workers, reusable pools, fork templates, or compile
  thread pools. The current design preserves one process per workload and
  removes repeated CPU prepare rather than adding a reusable worker system.
- **Single source of truth:** kernel identity comes from `KERNEL_META`, complete
  config/default/selection metadata comes from the kernel YAML, and measurement
  defaults remain owned by the runner. Reports and gates derive from those
  authorities rather than maintaining competing manifests.
- **No slop:** the one-stage path is deleted, every registered adapter is
  accounted for, selection audit metadata is stripped before execution, and
  missing, exempted, diagnostic, and measured evidence are distinct states.
- **Broad understanding and independent evidence:** the retained tests protect
  lifecycle, scheduling, measurement, cleanup, and inventory contracts; tracked
  artifacts preserve CPU resource evidence and explicitly delimit the historical
  A/B's provenance gaps instead of presenting it as independently verified.
- **Optimize the real objective:** complete-command wall time on a fixed
  workload/protocol matrix remains the oracle. The passing Proton pair measures
  1.2692× on one UUID, and the completed 106-workload cold-cache supplement
  measures 1.312451× across the machine while stating its eligibility-policy and
  cache-state caveats.
- **Cost model and falsifiability:** incomplete timelines still publish no numeric
  performance fields. The first persisted retry-heavy pair falsified schema 1's
  `ready_starvation_s` interpretation. Schema 3 now gives CPU READY constraints,
  retry READY delay, ASSIGN-held card time, post-GPU_START execution, transient
  foreign-PID wait, and internal dispatch latency separate canonical fields.
- **Stop low-quality experiments:** invalid runs remain unmeasured instead of
  weakening the protocol or claiming success. In-place retries retain the UUID
  handshake and explicit per-record provenance, while remaining ordinary
  measurement evidence under the human ruling.

## Terminal evidence boundary

- The implementation, static all-config migration gate, no-card structural
  tests, CPU-only prepare evidence, set-device binding, and same-child retry
  lifecycle are complete. One fresh and one controlled-retry single-GPU run are
  tracked as targeted implementation evidence. Historical generic
  lazy-replay GPU runs predate the UUID handshake and are not runtime acceptance
  evidence; the tracked A/B is explicitly invalidated.
- Proton has a passing, source-reproducible same-UUID schema-3 targeted A/B. Its
  earlier attempt 1 remains explicitly non-passing diagnostic evidence. Event
  has source-reproducible same-UUID evidence with four in-place retries and no
  speedup claim. CUDA-graph has the same evidence class with two in-place retries
  and no speedup claim. Kineto has tracked `missing` evidence showing the pinned
  TVM NVSHMEM helper creates a never-assigned GPU-0 context; MegaMoE still lacks
  valid runtime A/B evidence. These are terminal missing/unmeasured ledger states
  under the 2026-08-14 human scope decision, not pending GPU work.
- The original five DeepGEMM `compile_spec`/`build_launch` adapters have strict
  key/consumption tests and CPU-only READY evidence; their attempted real GPU
  replay is blocked before launch by the installed DeepGEMM reference API
  mismatch and is not marked passed.
- The five newly migrated DeepGEMM direct custom-compiler adapters have strict
  key/consumption tests and a repository CPU-only READY→CANCEL ledger, but no
  UUID-verified GPU runtime evidence and no runtime pass claim.
- Multi-GPU runtime behavior remains intentionally unmeasured.
- The former FlashInfer device-0 finding is explicitly reclassified as
  unreachable and non-blocking. The DeepGEMM MegaMoE call chain above is a
  reachable prerequisite for the new binding design, with an approved
  repository-owned restore-and-validate fix; it is neither a runtime pass/fail
  placeholder nor an implicit exemption. No external workaround is permitted.
- The historical 112-workload pair and 109-workload pipeline attempt both failed
  before completion and remain missing evidence. The authorized 106-workload
  supplemental pair completed and publishes the separately scoped 1.312451×
  cold-cache result. The former pinned-TVM authorization request was withdrawn;
  Kineto/allgather status remains missing rather than pass.
