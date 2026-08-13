# Bench Suite Pipeline Validation

Validation date: 2026-08-13.

This report records implementation and acceptance evidence for the one-shot
benchmark pipeline. It deliberately distinguishes static migration, structural
tests, invalidated or missing single-GPU evidence, and the human-directed
multi-GPU runtime exemption.

## Current implementation status before the superseding binding decision

- The current implementation snapshot uses the earlier design: every workload
  attempt uses one fresh process, late binding mutates `CUDA_VISIBLE_DEVICES`,
  and interference starts a fresh child that repeats CPU prepare. Those details
  are now obsolete and are not accepted as the target implementation.
- Its lifecycle follows
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

## Superseding set-device decision and reachability audit

The human-directed target keeps all physical GPUs visible in the prepare child,
so logical and physical ordinals are identical. CPU prepare must still leave CUDA
uninitialized. After each ASSIGN, the child must call
`torch.cuda.set_device(physical_index)` and prove the current physical device with
the existing UUID handshake before allocation or launch. Mutating
`CUDA_VISIBLE_DEVICES` in a live prepared process is no longer permitted.

Interference handling is also superseded. A workload owns one one-shot process
and performs CPU prepare exactly once. After an interfered GPU attempt, the
orchestrator must release the old atomic claim and may assign the same or another
card to that same child. The child must select the new card and rebuild all
GPU-side tensors, references, workspaces, and timer state without repeating
import, specialization, generation, or compilation. The run artifact must mark
`retry_in_place: true` and preserve per-attempt assignment, UUID, ownership, and
phase records.

This changes the measurement condition: a retry occurs in a process that has
already executed a GPU attempt. It is therefore a distinct hot-process evidence
class, not a clean fresh-process/first-attempt sample. Unless the human explicitly
approves that semantic change, in-place retries cannot be mixed into clean AC-10
A/B evidence. No such approval has been recorded.

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

The continued audit found such a reachable path in the intended DeepGEMM
MegaMoE dependency:

```text
PreparedMegaMoeBench.run_gpu
  -> tirx_kernels.deepgemm.mega_moe._run_distributed
  -> tirx_kernels.deepgemm.mega_moe._run_worker
  -> deep_gemm.utils.dist.init_dist(local_rank, num_processes)
  -> torch.cuda.set_device(local_rank)
```

The repository identifies this port against DeepGEMM `559d79fb`; that pinned
source calls `torch.cuda.set_device(local_rank)` in `deep_gemm/utils/dist.py:33`.
For a one-rank workload assigned physical GPU 6, `local_rank` remains 0 and the
dependency switches execution to GPU 0. This is a real call edge from the bench
workload, not a grep-only finding. The currently installed `deep_gemm` package
lacks `utils.dist`, so this machine fails the path earlier with an API mismatch;
that absence is not positive evidence for the intended dependency. Work on the
binding/retry migration is paused pending a human decision for this reachable
dependency. No external-library modification, monkey-patch, replacement
initializer, or mask fallback has been added.

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

- 41 registered kernels.
- 992 module-owned benchmark configs.
- 992 YAML inventory entries, with no module-only or YAML-only config.
- 21 generic lazy-replay adapters checked at AST level.
- 10 strict-cache adapters with exact prepared-cache key and consumption
  validation: five DeepGEMM `compile_spec`/`build_launch` adapters and five
  direct custom-compiler adapters.
- 10 explicit/custom adapters covering process-local executables, dispatcher
  delegation, hardware-profile compile caches, and distributed export/load
  lifecycles; together the three adapter classes account for 41/41 kernels.
- 112 single-GPU default workloads, with at most three defaults per kernel.
- 33 historically over-broad default selections, each now exactly three points
  with its rationale stored beside the canonical `default` flags in the same
  kernel YAML and emitted by the capability report.
- 27 multi-GPU configs, none selected by the default measured sweep.

Before the superseding set-device decision,
`tests/test_bench_pipeline_protocol.py` had 73 passing behavior tests covering:

- one-shot process/IPC behavior, log isolation, cancellation, fail-fast, dynamic
  external eligibility, the now-obsolete fresh-process interference retry,
  bounded-backlog
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
- capability accounting proves 41/41 adapters and 33/33 reviewed three-point
  selections from canonical sources.

Those historical tests remain evidence for the contracts they actually exercise,
but they do not prove the new `set_device` binding, non-assigned-card
allocation/launch rejection, one-prepare in-place retry, hot-process evidence
classification, or abandoned-card primary-context VRAM accounting.

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

The plan requires a migration-before versus pipeline A/B for every timer family
that can run on one GPU. None currently has admissible same-physical-GPU A/B
evidence. Completed runs are recorded below but remain unmeasured for AC-10 until
repeated through the UUID-verified path.

| timer family | runtime evidence | status |
|---|---|---|
| Proton | The fixed three-GEMM pair retained default 5 rounds/1.0s data, but the old pipeline side did not verify its physical UUID; its 1.884× arithmetic also relies on unpersisted outer timers | invalidated; AC-10 unmeasured pending UUID-verified same-card rerun |
| Event | Clean zero-retry default-protocol runs exist locally on both sides; the pipeline result is TIRx 6.180µs and requested index 6, but the old pipeline path did not verify physical UUID | completed local runs, but physical identity invalidates the A/B; no claim |
| CUDA-graph Proton | Clean default-protocol runs exist locally on both sides; before TIRx is 1.675µs and pipeline TIRx is 1.931µs. This discrepancy helped expose the binding defect; the old pipeline path did not verify physical UUID | completed local runs, but physical identity invalidates the A/B; no claim |
| Kineto | Correlated-span, barrier, sample-wise-max, schema, and cleanup behavior pass structurally; the runtime path also requires the locked NCCL/cuBLAS/cuBLASMp/NVSHMEM environment | structural only; runtime A/B unmeasured |
| MegaMoE | Alternating order, per-rank samples, sample-wise max, mismatch rejection, and cleanup pass structurally | structural only; runtime A/B unmeasured due shared-machine interference |

The Event and CUDA-graph runs exist only in gitignored local run logs, so they are
not persistent review evidence. Their clean completion corrects the earlier
ledger rationale, but does not cure the unverified physical identity. No reduced
rounds, cooldown, timer budget, reference coverage, or correctness work was used
to manufacture a result.

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
| AC-2 | incomplete under superseding design | The earlier mask-based protocol has assignment validation and CUDA guards, but live-process masking is obsolete. `set_device(physical_index)`, per-attempt UUID proof, and non-assigned-card allocation/launch rejection are not complete; the reachable DeepGEMM MegaMoE `init_dist -> set_device(local_rank)` path requires a human decision |
| AC-3 | satisfied | Bounded one-shot concurrency/backlog, condition-driven dispatch, same-GPU serialization, logical multi-GPU concurrency, and dynamic eligibility tests |
| AC-4 | satisfied for first attempts; unresolved for in-place retry | Default 5 rounds/1.0s, finalization, raw samples, and timer schemas are unchanged. A retry in an already-used GPU process is a distinct hot-process condition and cannot enter clean AC-10 evidence without human approval |
| AC-5 | incomplete under superseding design | Existing fail-fast and cleanup evidence remains valid, but the fresh-child retry is obsolete. One-prepare in-place GPU retry, exact ownership transfer, GPU-state rebuild, and abandoned-card primary-context VRAM reporting are not implemented |
| AC-6 | satisfied | Bounded process/RSS/FD evidence, cancellation cleanup, immediate internal release, and resource accounting tests |
| AC-7 | satisfied | Complete timeline validation, no-data cost-model gating, diagnostic-protocol watermarking, and tracked internal cost-model arithmetic; source-artifact verification is conditional on gitignored artifacts being present |
| AC-8 | satisfied | Canonical `KERNEL_META` exact-load index, runtime metadata validation, duplicate rejection, cache invalidation, and all-config resolution gate |
| AC-9 | satisfied for migration and structural coverage | 41/41 adapters and 992/992 configs pass the pipeline-only gate; one-stage execution is removed; multi-GPU runtime remains separately exempted |
| AC-10 | incomplete | The former Proton claim is invalidated by unverified physical identity and unpersisted outer timers; Event and CUDA-graph runs have the same identity defect, while Kineto and MegaMoE runtime A/B evidence remains unmeasured |
| AC-11 | satisfied | 112 defaults, all 992 configs retained, and 33/33 reviewed three-point selections with YAML-owned small/medium/large roles and rationale |

The plan as a whole is therefore not marked complete. The immediate blocker is
the human decision required by the reachable DeepGEMM MegaMoE device override; after that is
resolved, AC-2 and AC-5 require the superseding implementation and structural
evidence. AC-10 still requires admissible single-GPU timer-family runtime
evidence on a suitable machine. Multi-GPU runtime rows remain the explicit
human-directed exemption, not missing evidence.

## Engineering-principles audit

- **Occam's razor:** the implementation uses one one-shot child lifecycle and
  two explicit replay mechanisms (generic lazy replay and strict keyed replay);
  it introduces no resident workers, reusable pools, fork templates, or compile
  thread pools. The superseding target preserves one process per workload and
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
  UUID-verified workload/GPU/protocol matrix remains the oracle; no current
  performance result meets that evidence boundary.
- **Cost model and falsifiability:** expected critical time is reconstructed
  from first READY plus GPU scheduling, with foreign wait and residual separate;
  incomplete timelines publish no numeric performance fields. The historical
  Proton quotient is retained only as invalidated ledger arithmetic.
- **Stop low-quality experiments:** invalid runs remain unmeasured instead of
  weakening the protocol or claiming success. Future in-place retries must retain
  the UUID handshake and be labeled as hot-process evidence, not silently folded
  into a clean first-attempt A/B.

## Remaining evidence boundary

- The implementation, static all-config migration gate, no-card structural
  tests, and CPU-only prepare evidence for the earlier pipeline are complete.
  The superseding device-binding and retry lifecycle are not implemented.
  Historical generic
  lazy-replay GPU runs predate the UUID handshake and are not runtime acceptance
  evidence; the tracked A/B is explicitly invalidated.
- Proton, Event, and CUDA-graph Proton have no admissible UUID-verified targeted
  A/B. Kineto and MegaMoE also lack valid runtime A/B evidence; AC-10 and the
  overall plan remain incomplete.
- The original five DeepGEMM `compile_spec`/`build_launch` adapters have strict
  key/consumption tests and CPU-only READY evidence; their attempted real GPU
  replay is blocked before launch by the installed DeepGEMM reference API
  mismatch and is not marked passed.
- The five newly migrated DeepGEMM direct custom-compiler adapters have strict
  key/consumption tests and a repository CPU-only READY→CANCEL ledger, but no
  UUID-verified GPU runtime evidence and no runtime pass claim.
- Multi-GPU runtime behavior remains intentionally unmeasured.
- The former FlashInfer device-0 finding is explicitly reclassified as
  unreachable and non-blocking. The DeepGEMM MegaMoE call chain above is the
  current feasibility blocker; it is neither a runtime pass/fail placeholder nor
  an implicit exemption. No external workaround or GPU validation was attempted.
- The full 112-workload measured sweep and baseline promotion remain deferred
  until the shared machine is available; they were not required or run here.
