# Grouped GEMM Masked Rubin Kernel Sketch

Status: writer draft; immutable after the first reviewer PASS.

This is an operation-level sketch for the TIRx port of FlashInfer
`grouped_gemm_masked_rubin.py` at commit
`012cfdb97f217e0d48bc9352c17a74068c9e495b`. It describes source semantics,
not executable TIRx. It deliberately uses only linear storage and scalar byte/
element mappings. No first-class layout object is part of this sketch.

## Frozen source and production specialization

- Entry: `grouped_gemm_nt_masked` -> `_grouped_gemm_nt_masked_sm107` ->
  `MaskedBatchedMatmulCuteDSLRubin` ->
  `Sm107BlockScaledPersistentDenseGemmKernel.kernel`.
- Source:
  `/root-vol/aarch64-ws/kernel-libs/vr200/flashinfer/flashinfer/gemm/kernels/grouped_gemm_masked_rubin.py`
  (`sha256=729ead8b8e3cfc66b0ec57e4b452f571c95f185758444a9ed697b11e60005639`).
- Scheduler source:
  `grouped_gemm_masked_blackwell.py:103-418`.
- Production specialization: A/B `float4_e2m1fn`, SFA/SFB
  `float8_e4m3fn`, vector length 16, C `bfloat16`, tile
  `(128,128,256)`, instruction `(128,128,128)`, cluster `(1,1)`, no alpha,
  no destination signals, K-major A/B, N-major C.
- Writer line-info PTX:
  `.porting/grouped_gemm_masked_rubin/kernel_sketch/source_export_production/`
  `cutlass___call___flashinfergemmkernelsgrouped_gemm_masked_rubinMaskedBatchedMatmulCuteDSLRubin_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_None_None_CUstream0x0.sm_107a.ptx`
  (`sha256=5092bf530b543b24442a30486bc4f3807ed6d09c77b531ca01e13b33b7b609c6`,
  `.version 9.4`, `.target sm_107a`, `.reqntid 192,1,1`).

The production benchmark consists of the 102 unique combinations of 51
`(num_groups, expected_m_per_group)` profiles and `(N,K)` in
`{(4096,7168),(7168,2048)}`, with `max_m=4096`.

## Parameterization and derived constants

The public grouped wrapper has one A/B dtype. The reachable dtype classes are:

- FP4: A/B E2M1 with scale E4M3, E8M0, or NV-E5M3, vector 16 or 32 where
  accepted by the source validation; tile K 256 and instruction K 128.
- FP8: A/B E4M3 or E5M2 with E8M0 scale and vector 32; tile K 128 and
  instruction K 64.
- C: FP32, FP16, or BF16. Optional per-expert alpha has the declared alpha
  dtype. Optional destination signals are uint32 release counters.
- instruction M is 128 (CTA group 1) or 256 (CTA group 2); instruction N and
  tile N are 64, 128, 192, or 256; tile M is instruction M or twice it
  (B-reuse); cluster M/N are powers of two in `{1,2,4}`, total <=16, with
  cluster M divisible by the CTA group.

For one specialization derive:

- `cta_group = instruction_m / 128`, `cta_m = tile_m / cta_group`,
  `b_rows = tile_n`, `k_tiles = ceil_div(K,tile_k)`.
- `acc_stages = 1` only for B-reuse with N 192 or 256, otherwise 2.
- AB stages and C stages are the exact source `SharedStorage.get_smem_capacity`
  solution within the SM107 dynamic-SMEM limit; production is AB=8, ACC=2,
  C=4, dynamic SMEM=328704 bytes, and TMEM=576 columns.
- `m_tiles(group)=ceil_div(masked_m[group],cta_m)` and
  `n_cluster_tiles=ceil_div(ceil_div(N,tile_n),cluster_n)`.
- Grid is `(cluster_m,cluster_n,min(max_active_clusters,total_tile_clusters))`.
  In production, the persistent z extent is bounded by 200 singleton clusters.

Specialization predicates are compile-time. Runtime shape facts are limited to
the masked row counts and persistent scheduler state.

## Linear storage and scalar mappings

All global arguments are one-dimensional pointer views over the wrapper's
physical allocations. All shared allocations below are byte-linear.

| object | storage and lifetime | linear mapping / placement |
| --- | --- | --- |
| A | GMEM packed FP4 or FP8, whole launch | production byte `((g*M+m)*K+k)//2`; low/high nibble selects even/odd logical `k`. FP8 replaces `/2` by one byte per logical element |
| B | GMEM packed FP4 or FP8, whole launch | production byte `((g*N+n)*K+k)//2`; low/high nibble selects even/odd logical `k`. FP8 replaces `/2` by one byte per logical element |
| SFA/SFB | GMEM 8-bit scale tensors, whole launch | for outer index `o`, scale index `j=k//sf_vec`, `sf_inner=ceil(K/(4*sf_vec))*4`, `padded_outer=ceil(outer/128)*128`, byte `g*padded_outer*sf_inner + ((j//4)*4+(o//128)*sf_inner)*128 + (o%32)*16 + ((o%128)//32)*4 + j%4`; outer is M for SFA and N for SFB |
| C | GMEM FP32/FP16/BF16 | production byte `((g*M+m)*N+n)*2`. A scheduled final M tile writes every row through `ceil_div(masked_m[g],128)*128-1` (bounded by max M), including rows at or above `masked_m[g]`; masking is tile-granular, not element-granular |
| masked_m | GMEM int32[num_groups] | byte `4*g`, read only after `g < num_groups` has been established by nested control flow |
| alpha | optional GMEM | one element per expert, read for the current valid expert only |
| dst_signals | optional GMEM uint32 | source scheduler-defined expert completion counters, release atomic updates |
| protocol | dynamic SMEM byte interval from 0 | production AB-full `[0,64)`, AB-empty `[64,128)`, ACC-full `[128,144)`, ACC-empty `[144,160)`, dealloc barrier `[160,168)`, TMEM holding slot `[168,172)` |
| sC | dynamic SMEM byte-linear `[1024,33792)` | production stage size 8192. For stage `s`, vector `v=0..3`, warp `w=0..3`, lane `l=0..31`, let `u=smem_base+1024+s*8192+w*2048+l*64+v*16`; byte offset is `(u xor ((u>>3)&48))-smem_base` and the 16 bytes hold four consecutive packed BF16x2 words |
| sA | dynamic SMEM byte-linear `[33792,164864)` | production stage base `33792+s*16384`; MMA descriptor is `0x4000404000010000 OR (((smem_base+33792)>>4)&0x7fff)`, then adds `s*1024+kblock*4` in 16-byte descriptor units |
| sB | dynamic SMEM byte-linear `[164864,295936)` | production stage base `164864+s*16384`; same descriptor base/immediate as A, adding `s*1024+kblock*4` |
| sSFA | dynamic SMEM byte-linear `[295936,312320)` | production stage base `295936+s*2048`; scale-copy descriptor `0x0000400800010000 OR (((smem_base+295936)>>4)&0x7fff)`, then `s*128+chunk*32`, `chunk=0..3` |
| sSFB | dynamic SMEM byte-linear `[312320,328704)` | production stage base `312320+s*2048`; same scale descriptor base, then `s*128+chunk*32`, `chunk=0..3`; other N specializations apply the source N=64 slice and N=192 overlap equation before this scalar offset |
| tAcc | TMEM columns `[0,256)` in production | stage `s` starts at column `s*128`; epilogue warp `w` contributes address bit field `w<<21`; subtile `q=0..3` adds `q*32` |
| tSFA | TMEM columns `[256,272)` in production | each `chunk=0..3` copies 512 shared bytes to column `256+4*chunk` |
| tSFB | TMEM columns `[272,288)` in production | each `chunk=0..3` copies 512 shared bytes to column `272+4*chunk`; columns `[288,576)` remain part of the exact source allocation and provide alternate-specialization capacity |

Intervals are computed at specialization time and asserted non-overlapping. The
dynamic SMEM base is aligned to 1024 bytes. TMEM is exactly the source column
allocation and is allocated/deallocated by epilogue warp 0.

## Barrier and pipeline contract

1. AB-full has `ab_stages`, initial phase 0 for consumers, one producer arrival,
   and expected bytes equal to all A+B+SFA+SFB TMA pieces for a stage and CTA
   group (production exactly 36864 bytes). AB-empty has `ab_stages`, initial producer phase 1, and arrival count
   `cluster_n + cluster_m/cta_group - 1` with source cluster multicast masks.
2. ACC-full has `acc_stages`, initial consumer phase 0, one MMA commit arrival.
   ACC-empty has `acc_stages`, initial producer phase 1 and `128*cta_group`
   per-thread epilogue arrivals (production exactly 128).
3. Warp 0 elects one lane to initialize AB/ACC barriers. TMA warp elects one
   lane to initialize the 2-CTA TMEM deallocation barrier. Each pipeline
   construction issues its own `fence.mbarrier_init.release.cluster` followed
   by its CTA synchronization; the explicit source fence follows, for three
   ordered init fences and two built-in CTA barriers in production. Cluster
   size >1 then uses relaxed cluster arrive and later cluster wait. Production
   cluster `(1,1)` emits neither cluster operation and instead executes the
   later singleton fallback `bar.sync 0,192` before TMEM use (three CTA barrier
   points total including the two pipeline-construction synchronizations).
4. Each ring cursor advances stage and flips phase only at wrap. Speculative
   `try_wait.parity.acquire.cta.shared::cta` can suppress the later plain wait,
   but never changes publication/reuse order.
5. TMA C store uses proxy-async CTA fence, named barrier 1 across 128 epilogue
   threads, bulk-group commit/acquire, and final tail. ACC-empty is released only
   after every subtile has left the accumulator. Optional signal publication
   waits for the required store group before a release atomic.

## Persistent masked scheduler

Each physical cluster starts at linear work index `blockIdx.z`; advance adds
`num_persistent_clusters`. Scheduler state is `(work,g,accum_m_tiles,executed)`.
For every query:

1. Set `keep_running = g < num_groups`.
2. While `keep_running`, and only inside that guarded region, load
   `rows=masked_m[g]`, compute its clustered M-tile span, and skip empty or
   exhausted experts whose cumulative span times N-cluster tiles does not cover
   `work`. Record any pending completion bookkeeping before incrementing g.
3. Recheck `g < num_groups`; only then may `masked_m[g]` be loaded again.
4. Map `work` to `(cluster_tile_m,cluster_tile_n,g)`, then add the physical CTA
   coordinate times cluster shape. Validity is false for exhausted experts and
   trailing CTA coordinates outside `ceil_div(masked_m[g],cta_m)`.
5. TMA, MMA, and epilogue roles instantiate identical independent scheduler
   state and advance in the same source order. The epilogue additionally carries
   packed pending signal counts across work advances.

This nested guard is a safety invariant: no combined expression may speculatively
read `masked_m[num_groups]`.

## Execution skeleton

### Prologue and roles

1. Launch six warps. Warps 0-3 are epilogue, warp 4 is MMA, warp 5 is TMA.
   The source chain-dispatch ordering is preserved.
2. TMA warp prefetches A, B, SFA, SFB, and C tensor-map descriptors.
   `instruction_selection: five prefetch.tensormap instructions; source PTX .loc 1026-1030.`
3. Initialize the barrier arrays and conditional deallocation barrier, issue the
   three ordered init fences/two pipeline CTA barriers, then take the cluster or
   singleton-CTA branch specified above.
   `instruction_selection: production has 20 mbarrier.init.shared.b64, three fence.mbarrier_init.release.cluster, two built-in bar.sync 0 plus later bar.sync 0,192, and no barrier.cluster instruction; source PTX .loc 1068-1106 and 1273.`

### TMA producer warp 5

For each valid scheduled output tile and every K tile (production exactly 28 K tiles):

1. Speculatively acquire then, if needed, wait for AB-empty at the current ring
   stage. The leader CTA arrives-with-expected-bytes on AB-full.
   `instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 plus mbarrier.try_wait.parity.shared.b64 and mbarrier.arrive.expect_tx.shared.b64; source PTX .loc 1331-1343.`
2. One elected lane issues the A tile copy at `(k,m,g)` into the stage SMEM sA
   interval. Production emits exactly
   `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`
   with destination, A tensor map, three coordinates, AB-full barrier, and cache
   hint; PTX `.loc 1345`, line 292. Cluster-N alternatives replace shared CTA by
   shared cluster and add multicast, and CTA-group 2 adds its modifier.
3. Issue the B tile copy at `(n,k,g)` into sB. Production emits the identical
   tensor.3d shared-CTA opcode family with B tensor map and no multicast/CTA-group
   modifier; PTX `.loc 1351`, line 300. Cluster-M alternatives are separate
   shared-cluster forms.
4. Issue SFA and SFB packed-scale tile copies into their stage intervals, with
   the exact N=64 SFB source slice and cluster multicast policy.
   `instruction_selection: production issues one SFA and one SFB cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint, each with four coordinates, AB-full barrier, cache hint, and no multicast/CTA-group modifier; source PTX .loc 1358-1365, lines 314 and 323. Cluster/CTA-group alternatives use their distinct source forms.`
5. Advance the AB producer cursor, perform the next speculative wait, and after
   the last work tile drain every AB-empty stage in ring order.

### MMA warp 4

After named barrier 2 across warps 0-4, load the allocated TMEM base. For each
valid scheduled output tile:

1. Speculatively acquire AB-full; leader CTA acquires the current ACC-empty
   stage. Apply the source odd-tile TMEM base correction for N=64/192.
2. For each K tile, wait AB-full if the speculative wait missed. Copy all scale
   chunks from the current sSFA/sSFB stage to tSFA/tSFB before MMA, except for the
   source FP4 B-reuse interleaved case where copy and MMA issue order is
   interleaved exactly by K block.
   `instruction_selection: one tcgen05.cp.cta_group::{1|2}.32x128b.warpx4 per 512-byte scale chunk, exact TMEM/SMEM descriptor offsets; production has exactly four SFA plus four SFB copies (8 total) per K tile, PTX .loc 591-598 and 1511-1547, lines 794-857.`
3. For each instruction K block, issue block-scaled MMA. B-reuse alternates the
   source B-keep/B-reuse descriptors and accumulator halves; non-reuse uses the
   base descriptor. The first issue clears/initializes, later issues accumulate.
   `instruction_selection: tcgen05.mma.cta_group::{1|2}.kind::{mxf4nvf4|mxf8f6f4}.block_scale.block{16|32}.collector::a::discard, exact instruction descriptor and scale TMEM operands, one issue per source K block/reuse half; production is exactly two block16 CTA-group-1 MMA issues per K tile (56 per output tile), PTX .loc 1591-1599, lines 884 and 903.`
4. Commit AB-empty only after the stage's last MMA use, then advance AB consumer
   state and speculatively wait for the next stage.
   `instruction_selection: tcgen05.commit.cta_group::{1|2}.mbarrier::arrive::one.shared::cluster with multicast only for CTA-group 2/source mask; production PTX .loc 1603-1604, PTX line 910.`
5. Commit ACC-full after the final K tile, advance ACC producer state and the
   scheduler. At tail, advance once more and wait for final ACC-empty reuse.
   `instruction_selection: tcgen05.commit...mbarrier arrival to ACC-full; production PTX .loc 1617-1619, PTX line 937.`

### Epilogue warps 0-3

1. All 32 lanes of warp 0 execute the warp-synchronous allocation of the specialization's TMEM columns; named
   barrier 2 publishes the pointer to warps 0-4.
   `instruction_selection: tcgen05.alloc{.exclusive for CTA-group 1}.cta_group::{1|2}.sync.aligned.shared::cta.b32; production PTX .loc 1639-1646, PTX line 1051.`
2. For each valid scheduled output tile, wait ACC-full. Iterate source epilogue
   subtiles (production exactly four). Each warp loads one 32x32-bit TMEM
   fragment into 32 registers.
   `instruction_selection: exactly one tcgen05.ld.sync.aligned.32x32b.x32.b32 per warp/subtile; it is synchronous and production emits no separate tcgen05.wait::ld; PTX .loc 1762-1768, line 1196.`
3. If alpha is present, load `alpha[g]` and multiply every FP32 accumulator
   lane. Convert with source rounding to FP32, packed FP16, or packed BF16.
   `instruction_selection: ld.global alpha per valid expert when enabled; mul.f32x2 for paired values; cvt.rn.f16x2.f32 or cvt.rn.bf16x2.f32, or bit-preserving FP32 words; source .loc 1771-1778.`
4. Store register vectors into the current byte-linear sC stage using the scalar
   XOR-swizzled byte mapping. Fence async shared and named-barrier-sync 128
   epilogue threads.
   `instruction_selection: production has exactly four st.shared.v4.b32 per thread/subtile, then fence.proxy.async.shared::cta and bar.sync 1,128; source .loc 1778-1794, PTX lines 1251-1261.`
5. All 32 lanes of epilogue warp 0 execute the warp-synchronous issue of the C subtile TMA store at `(n,m,g)`,
   commits the bulk group, and acquires/waits according to the C-stage ring.
   `instruction_selection: cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint, cp.async.bulk.commit_group, and production cp.async.bulk.wait_group.read 3; final tail is wait_group.read 0; PTX .loc 1797-1825 and 1902-1903, lines 1270-1275 and 1395.`
6. Barrier-sync before reusing sC. When signals are enabled, lane 0 publishes
   every newly completed expert only after its final relevant C store group,
   using the source release atomic and packed pending counters.
   `instruction_selection: source signal atomic atom.add.release.gpu.global.s32 with signed 32-bit add semantics; source .loc 1832-1847 and 1886-1901; absent from production PTX by specialization.`
7. After all four production subtiles, every one of the 128 participating
   epilogue threads arrives on ACC-empty. CTA-group 2 uses the corresponding
   remote-peer target, but this is never an elected-lane release. Advance ACC
   state and scheduler, then drain C stores at tail.
8. Warp 0 relinquishes the TMEM allocation permit. CTA-group 2 performs the
   remote deallocation-barrier handshake; then warp 0 deallocates TMEM.
   `instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::{1|2}.sync.aligned; optional mapa + mbarrier arrive/wait; tcgen05.dealloc{.exclusive for group 1}.cta_group::{1|2}.sync.aligned.b32; production PTX .loc 1865-1877, PTX line 1393.`

## Source coverage map

| source lines | sketch section | semantic coverage |
| --- | --- | --- |
| benchmark `bench_cute_dsl_blockscaled_gemm.py:16-19,27-47,92-202` | Frozen source / parameterization / linear storage | exact production dtype/config selection, 102-workload family, physical tensor creation, mask generation, public call |
| public `grouped_gemm_masked_wrapper.py:44-257` | Frozen source / parameterization | SM107 dispatch, public tile translation, rejected fusion/multirank/swap, alpha forwarding |
| Rubin 120-205, 1905-2216 | Parameterization | dtype, instruction/tile, B-reuse, cluster, output, and alignment validation |
| Rubin 207-319 | Derived constants / linear storage | exact ACC/AB/C stage computation from SMEM capacity |
| Rubin 321-653 | Linear storage / MMA warp | MMA/cluster shapes, all staged SMEM and TMEM scalar mappings, scale-copy ordering |
| Rubin 655-983 | Frozen source / parameterization | host tensors, TMA descriptors, scheduler params, grid, shared structure and launch |
| Rubin 986-1107 | Prologue and barrier contract | roles, five descriptor prefetches, pipelines, three init fences, CTA/cluster branch |
| Rubin 1110-1275 | Linear storage / prologue | all global/shared/TMEM partitions, TMA partitions, singleton pre-use sync |
| Rubin 1278-1391 | TMA producer | persistent scheduler and A/B/SFA/SFB TMA pipeline and tail |
| Rubin 1395-1631 | MMA warp | TMEM scale copies, blockscaled MMA, B reuse, commits and tails |
| Rubin 1635-1903 | Epilogue | TMEM allocation/load, alpha/cast, sC, C TMA, signals, per-thread ACC release, deallocation |
| Rubin 2219-2594 | Frozen source / linear storage | internal physical pointer/tensor mapping, packed scale transform, compiled callable, optional argument specialization |
| Rubin 2597-2688 | Frozen source / parameterization | SM107 adapter, C-layout check, FP4 logical-K recovery, defaults/kwargs and invocation |
| inherited Blackwell 103-184,185-421 | Persistent scheduler | scheduler parameters, persistent grid/initial state, strict guarded expert scan, tile mapping, advance, signal bookkeeping |
| inherited Blackwell 588-699 | Parameterization / roles | base constructor, warp and named-barrier IDs, epilogue tile and common attributes |
| inherited Blackwell 2232-2379 | Linear storage / epilogue | TMEM-to-register and C-store partition semantics used by Rubin |
| inherited Blackwell 2516-2549 | Persistent scheduler | exact cta-tiler scheduler parameters and grid helper |

Every sketch operation above is justified by one of these source regions; the
descriptor-bit arithmetic and incidental temporary scalars are intentionally
contained within the corresponding semantic copy/compute row.

## Non-negotiable implementation constraints

- Implement only with `import tirx_kernels.kern as K`; no handwritten PrimFunc
  body, first-class layout API, `K.cuda.func_call`, inline CUDA function call,
  or change under `tirx_kernels/kern/`.
- Preserve the strict masked-row guard, persistent advance, exact pipeline
  stages/phases, warp roles, TMA/TMEM operations, optional alpha and signal
  branches, and source output bytes.
- Production performance is judged only by bench_suite with the pinned
  FlashInfer reference under PTX 9.4; broad non-production branches are
  correctness coverage, not extra benchmark rows.
