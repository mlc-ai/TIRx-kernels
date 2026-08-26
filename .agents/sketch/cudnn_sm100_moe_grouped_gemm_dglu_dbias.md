<!--
This file describes a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cudnn_sm100_moe_grouped_gemm_dglu_dbias: coarse WASP pipeline sketch

This document is **not executable**. It fixes the resource allocation, the task
split across warps, and the per-task tile dataflow for the TIRx port at
`tirx_kernels/cudnn/dglu/_moe_grouped_gemm_dglu_dbias/kernel.py`, which is the
executable source of truth. The sketch is frozen once the sketch reviewer
passes it, and neither the correctness gate nor the performance gate may edit
it.

## Source identity

- source: `/home/bohanhou/kernel-libs/cudnn-frontend/python/cudnn/gemm/cutedsl/grouped/dglu/moe_grouped_gemm_dglu_dbias.py`
- commit: `aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5` (`1.25.0.dev-250-gaded9909`)
- sha256: `d448b5c9ddd4514f96340aa1a620894a48e884f4216c56715119d56c37bd9e38`, **2284 lines**
- entry: `MoEGroupedGemmDgluDbiasBf16Kernel.__call__` -> optional `helper_kernel`, then `kernel`

Citation shorthands: a bare `N` is a line in the source above; `sched N` is
`grouped/moe_persistent_scheduler.py`, `utils N` is `grouped/moe_utils.py`,
`ext N` is `grouped/moe_sched_extension.py`, `helpers N` is
`grouped/moe_kernel_helpers.py`. `PTX n` is a **line number in the anchor
export** named in the evidence table, never a source line.

## PTX `.file` numbering

| `.file` | path |
| --- | --- |
| 1 | `grouped/dglu/moe_grouped_gemm_dglu_dbias.py` |
| 2 | `grouped/moe_persistent_scheduler.py` |
| 3 | `grouped/moe_utils.py` |
| 4 | `grouped/moe_sched_extension.py` |

## Evidence

Writer exports under `.porting/moe_grouped_gemm_dglu_dbias/writer_source_export/`,
from `export_ptx.py` with `CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=ptx
CUTE_DSL_NO_CACHE=1`. Each run asserts its outputs are finite and non-zero, so a
degenerate export cannot be annotated.

| branch | axis it turns on | ptx lines | `.loc` | sha256 | MMA idesc | AB stages |
| --- | --- | --- | --- | --- | --- | --- |
| `anchor` | the annotated specialization | 3793 | 1159 | `ab1d446e` | `0x10400490` | 5 |
| `discrete` | discrete-B weights + helper descriptors | 3868 | 1173 | `3162ca17` | `0x10400490` | 5 |
| `dynamic` | atomic-ticket scheduler | 4004 | 1221 | `53e3cb42` | `0x10400490` | 5 |
| `dgeglu` | the second activation | 4840 | 1481 | `14b95ffa` | `0x10400490` | 5 |
| `cf32_nodbias` | FP32 C, dBias off | 3063 | 925 | `1d38f93a` | `0x10400490` | 5 |
| `bmajor_n` | n-major B | 3796 | 1153 | `6b9cb225` | `0x10410490` | 5 |
| `tile128_c1x1` | one-CTA tile, singleton cluster | 3604 | 1115 | `f425a929` | `0x08400490` | 3 |
| `tile_n64` | narrow tile_n | 3683 | 1127 | `098f01b9` | `0x08100490` | 6 |
| `scalar_f32` | scalar (non-packed) epilogue math | 3501 | 849 | `201ea257` | `0x10400490` | 5 |
| `nodbias` | bf16 C, dBias off -- the depth-6 point | 3197 | 952 | `354e50c1` | `0x10400490` | 6 |

Instruction counts follow the corpus convention, instruction lines minus
predicated lines: `anchor` = 2263 - 120 = **2143**.

## Anchor values

| quantity | value | source |
| --- | --- | --- |
| mode | dense, static, dSwiGLU, C/D bf16, k-major B, packed-f32, dBias on | |
| `mma_tiler_mn` | `(256, 256)`, `use_2cta_instrs`, `atom_thr = 2` | `:123-133` |
| `cta_tile_shape_mnk` | `(128, 256, 64)` | `:212-230` |
| `cluster_shape_mn` | `(2, 1)` | |
| shape | 4 experts x 256 tokens, `N = 512`, `K = 512` | |
| `epi_tile` | `(128, 32)`, 8 subtile pairs per tile | `:261` |
| `k_tile` | 64 = instruction K 16 x `mma_inst_tile_k` 4 | `:224-230` |
| stages | acc 2, **AB 5**, C 2, D 2, tile-info 2 | `:2239-2282` |
| AB `expect_tx` | **65536** = `(a_stage 16384 + b_stage 16384) * atom_thr 2` | `PTX 903-904` |
| C `expect_tx` | **8192** = `128 * 32 * 2 B` | `PTX 3658` |
| TMEM columns | **512** = `clamp(next_pow2(num_acc_stage * cta_tile_n), 32, 512)` | `:289-294`, `PTX 1427-1428` |
| threads | 256 = 8 warps, `.maxntid 256, 1, 1`, `.minnctapersm 1` | `PTX 38-39` |
| grid | `(cluster_m, cluster_n, max_active_clusters)`, persistent | `:551`, helpers:988-992 |
| shared memory | one dynamic `.extern .shared .align 1024 .b8` arena | `PTX 14` |

**AB depth is 5, and dBias does not change it at this tile.** The solve is
`(232448 - reserved) // (a_stage + b_stage)` over `a_stage = b_stage = 16 KiB`,
which lands on 5 both with dBias (`anchor`) and with FP32 C and no dBias
(`cf32_nodbias`, whose larger sC cancels the sDbias saving). Dropping dBias at a
bf16 C does raise it, to 6 (`nodbias`: 12 AB mbarriers, sA-to-sB span
132096 - 33792 = 98304 = 6 x 16384). The depth also moves with the tile:
`tile128_c1x1` gets 3 (one CTA, so a full 32 KiB sB stage) and `tile_n64` gets 6.
Every depth in the evidence table is read from an export, none inferred.

**TMEM columns are derived, not pinned.** `anchor` and `tile128_c1x1` allocate
512; `tile_n64` allocates **128** (`tcgen05.alloc` immediate). This is the
largest structural departure from the block-scaled sibling, which always
reserves 512 and hand-partitions scale-factor regions inside it. The literal 2
in the closed form is `num_acc_stage`.

## The MMA instruction descriptor

Lifted from the exports, never adapted from the block-scaled encoder. Built as
`selp`-selected constants immediately before the MMA (`PTX 1250-1252`), issued
as `tcgen05.mma.cta_group::2.kind::f16` (`PTX 1264`, `.loc 1 1568`).

```text
idesc = base | (p1 ? 1<<13 : 0) | (p2 ? 1<<14 : 0)
base(anchor) = 0x10400490
```

| bits | field | anchor | evidence |
| --- | --- | --- | --- |
| 4-6 | D format | 1 (f32) | constant across branches |
| 7-9 | A format | 1 (bf16) | constant across branches |
| 10-12 | B format | 1 (bf16) | constant across branches |
| 13 | A negate | runtime `%p1` | `TiledMMA` kernel param, `PTX 46-52` |
| 14 | B negate | runtime `%p2` | `TiledMMA` kernel param, `PTX 46-52` |
| 16 | transpose B | 0 here, **1** in `bmajor_n` | `0x10400490` vs `0x10410490` |
| 17-22 | `N >> 3` | 32 -> N 256 | `tile_n64` gives 8 -> N 64 |
| 23 | SF format | **0** | the block-scaled kinds set this; `kind::f16` never does |
| 24-28 | `M >> 4` | 16 -> M 256 | `tile128_c1x1` gives 8 -> M 128 |

Bits 13/14 arrive as `TiledMMA` kernel parameters this kernel never sets (it
only ever sets `tcgen05.Field.ACCUMULATE`), so both predicates are false at run
time and the port emits a compile-time constant descriptor:

```text
idesc = (1 << 4) | (1 << 7) | (1 << 10)
      | (transpose_b << 16) | ((tile_n >> 3) << 17) | ((tile_m >> 4) << 24)
```

`# instruction_selection: tcgen05.mma.cta_group::{1,2}.kind::f16; extent: one 16-deep k-block, four issues per k tile`

## Waits come in two forms

Of 33 `mbarrier.try_wait` in the anchor, **29** are blocking acquires
`mbarrier.try_wait.parity.shared.b64` carrying the suspend hint **10000000**,
and **4** are non-blocking peeks
`mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` with **no** hint
(`PTX 863, 919, 1170, 1241` at `.loc 1 1441 / 1459 / 1534 / 1555`). The peeks
look one stage ahead and their status predicates the following acquire.

The hint value is a per-kernel fact read off the export: this family spins with
10000000, where the linear-attention family spins with 1.

`# instruction_selection: mbarrier.try_wait.parity.shared.b64 with suspend hint 10000000; extent: one blocking acquire per handshake`
`# instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64, no hint; extent: one look-ahead peek per producer loop`

## No first-class layouts

Neither this sketch nor the device kernel introduces a first-class layout, a
layout algebra object, or a multidimensional shared-memory tensor. Every shared
region is a one-dimensional byte range inside a single flat `u8` arena, indexed
by explicit scalar offset arithmetic; matrix descriptors are assembled from
hardware immediates. The upstream `SharedStorage` struct and its `cute` layouts
document the byte map only.

## Pipeline at a glance

Eight warps, 256 threads, one CTA per multiprocessor, persistent over work
tiles handed out by the scheduler warp. Roles and ids are unchanged from the
block-scaled sibling (`:158-162`). Sections below appear in **source order**.

| warp | role | tile program | publishes / consumes |
| --- | --- | --- | --- |
| 7 | scheduler | walk work tiles, flatten each into `sInfo`, terminate with `expert_idx = -1` | publishes `tile_info_full`; consumes `tile_info_empty` |
| 5 | A/B TMA | per k tile: peek, acquire, `expect_tx` 65536, two cluster-multicast `cp.async.bulk.tensor` | consumes `ab_empty`; publishes `ab_full` |
| 4 | MMA | per k tile: four `tcgen05.mma.kind::f16`; leader CTA only | consumes `ab_full`; releases `ab_empty` **via `tcgen05.commit`**; publishes `acc_full` via `tcgen05.commit` |
| 0-3 | epilogue | per subtile pair: TMEM read, `alpha^2` scale, C read, two GLU gradients, dprob and dBias accumulation, D store | consumes `acc_full`, `c_full`; releases `acc_empty`, `c_empty`; warp 0 owns TMEM alloc and the D stores |
| 6 | C load | per subtile pair: two `shared::cta` `cp.async.bulk.tensor`, gate then up, into two separate C stages | consumes `c_empty`; publishes `c_full` |

Named barriers (`:183-198`): id 0 the CTA-wide init barrier (`bar.sync 0`,
`PTX 233`), id 2 epilogue (128 threads), id 3 TMEM lifecycle (160 = MMA warp
plus four epilogue warps), id 4 scheduler (32). Barrier **id 1 (256 threads)
does not occur in the anchor** — it appears only in the singleton-cluster
branches, where it replaces the cluster wait. Anchor totals: 12 `bar.sync`
(1x id 0, 2x id 4, 2x id 3, 7x id 2), 23 `mbarrier.init`, 33
`mbarrier.try_wait`, 25 `elect.sync`.

## Primitive vocabulary

Each is one PTX instruction, one tile, or one loop of one family:

- `tma_load_multicast_3d(dst_byte, desc, coords, barrier, mask)` — one
  `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::N`
  (A and B only)
- `tma_load_3d(dst_byte, desc, coords, barrier)` — one
  `cp.async.bulk.tensor.3d.shared::cta.global.tile...` (C only)
- `tma_store_3d(desc, coords, src_byte)` — one S2G bulk tensor copy
- `bulk_commit()` / `bulk_wait(0)` — one `cp.async.bulk.commit_group` /
  `cp.async.bulk.wait_group.read`
- `expect_tx(barrier, bytes)` / `arrive(barrier)` / `acquire(barrier, phase)` /
  `peek(barrier, phase)` — `mbarrier.arrive.expect_tx` / `mbarrier.arrive` /
  the hinted blocking `try_wait` / the unhinted `try_wait.acquire.cta`
- `umma_arrive(barrier)` — one
  `tcgen05.commit...mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`
  (the MMA warp's AB release and its accumulator publish; **not** an
  `mbarrier.arrive`)
- `mma_f16(acc_col, a_desc, b_desc, idesc, accumulate)` — one
  `tcgen05.mma.cta_group::N.kind::f16`
- `tmem_alloc(cols)` / `tmem_relinquish()` / `tmem_dealloc(cols)`
- `tmem_load_32x32b_x32(regs, col)` — one `tcgen05.ld.sync.aligned.32x32b.x32.b32`
- `smem_ld_v4(byte)` / `smem_st_v4(byte, regs)` — `ld.shared.v4.b32` /
  `st.shared.v4.b32`
- `smem_ld_b32(byte)` / `smem_st_b32(byte, v)` — scalar `ld.shared.b32` /
  `st.shared.b32`
- `cvt_f32_bf16(x)` / `pack_bf16x2(a, b)` — `cvt.f32.bf16` / `cvt.rn.bf16x2.f32`
- `fmul2(a, b)` / `fadd2(a, b)` — `mul.rn.f32x2` / `add.rn.f32x2`
- `exp2(x)` / `rcp(x)` — `ex2.approx.ftz.f32` / `rcp.approx.ftz.f32`
- `atomic_add_f32(addr, v)` — one `atom.global.add.f32`
- `atomic_add_bf16x2(addr, lo, hi)` — one `cvt.rn.bf16x2.f32` plus one
  `red.global.add.noftz.bf16x2`
- `named_barrier(id, threads)` — one `bar.sync`
- `cluster_arrive_relaxed()` / `cluster_wait()` — `barrier.cluster.arrive.relaxed` /
  `barrier.cluster.wait`
- `fence_async_shared()` / `fence_mbarrier_init()` — `fence.proxy.async.shared::cta` /
  `fence.mbarrier_init.release.cluster`
- `elect_one()` — one `elect.sync`
- `mapa(addr, cta)` — one `mapa.shared::cluster.u32` (peer-CTA barrier address)

## Complete sketch

```python
# ==========================================================================
# Static specialization, runtime ABI, and launch
# source 123-345, 416-566; PTX 14-39
# ==========================================================================
# Compile-time: weight_mode, sched, act, c_dtype, d_dtype, b_major, tile_m,
# tile_n, cluster_shape, vectorized_f32, with_dbias, expert_cnt, N, K,
# group_m_list, linear_offset.
#
# Runtime operands, in ABI order:
#   a, b (dense tensor or discrete int64 pointer array), c, d,
#   padded_offsets, alpha, beta, prob, dprob, dbias?, workspace?
#
# grid   = (cluster_m, cluster_n, MAX_ACTIVE_CLUSTERS[cluster_m * cluster_n])
# block  = 256 threads = 8 warps
# cluster= (cluster_m, cluster_n), bound unconditionally including (1, 1)
#          # instruction_selection: cta_id_in_cluster with preferred=; extent: whole kernel

# ==========================================================================
# Storage and synchronization objects
# source 568-611, 1189-1250; PTX 14, 118-243
# ==========================================================================
# One flat `u8` arena, `K.alloc_buffer((SMEM_BYTES,), K.u8, scope="shared.dyn",
# align=1024)`, carved by explicit scalar byte offsets in upstream declaration
# order. Anchor byte map, read back from the export's TMA destinations:
#
#   ab_mbar        0     .. 79      5 full + 5 empty, 8 B each
#   acc_mbar       80    .. 111     2 full + 2 empty
#   sched block    112   .. 175     tile_info_mbar 4x8 B, sInfo 4x2x4 B align 16
#                                   (+ cluster_mbar 2x8 B, cluster_bcast 4x4 B
#                                    when the scheduler is dynamic)
#   c_full         176   .. 191
#   c_empty        192   .. 207
#   tmem_dealloc   208   .. 215
#   tmem_holding   216   .. 219
#   sC             1024  .. 17407   128*32*2 stages * 2 B
#   sD             17408 .. 33791   128*32*2 stages * 2 B
#   sA             33792 .. 115711  5 stages * 16384 B
#   sB             115712.. 197631  5 stages * 16384 B   (tile_n/atom_thr = 128 wide)
#   sDbias         197632.. 230399  128 * 64 * 4 B
#                                   total <= 232448
#
# A/B multicast masks, built once from the cluster layout and this CTA's
# in-cluster coordinate (source 1303-1307): the A mask spans the cluster's n
# extent, the B mask its m extent, so each tile is fetched once per cluster and
# broadcast.
#
# TMEM: one allocation of clamp(next_pow2(num_acc_stage * cta_tile_n), 32, 512)
# columns, split into num_acc_stage accumulator regions of cta_tile_n columns.
#   # instruction_selection: tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32; extent: once per kernel
#
# Init order (source 1189-1250, 1272-1282, 1325-1328):
#   mbarrier_init(...)                       # 22 objects, PTX 118-225
#     # ab_full/empty 5+5 at 0..72, acc_full/empty 2+2 at 80..104,
#     # tile_info_full/empty 2+2 at 112..136, c_full/empty 2+2 at 176..200.
#   fence_mbarrier_init()                    # PTX 232
#   named_barrier(0, 256)                    # PTX 233  -- CTA-wide init barrier
#   if elect_one():                          # PTX 238
#       mbarrier_init(tmem_dealloc, 32)      # source 1272, PTX 243 -- the 23rd
#   fence_mbarrier_init()                    # PTX 246, publishes that one barrier
#   cluster_arrive_relaxed()                 # PTX 248
#   ... smem tensor setup, the two multicast masks, MMA fragment construction ...
#   cluster_wait()                           # PTX 262
#     # The source comment at 1322-1323 is explicit: cluster wait BEFORE the
#     # tensor-memory allocation. The allocation itself is issued much later, in
#     # the epilogue role at source 1608 / PTX 1427-1428.
#     # When cluster_size == 1 the cluster wait degenerates and the source uses
#     # named_barrier(1, 256) instead; that is the only place barrier 1 appears.
#
# total_tokens = padded_offsets[expert_cnt - 1]     # source 1161
# if total_tokens <= 0: exit()                      # source 1330-1331
#
# k_tile_cnt = ceil_div(K, k_tile)                  # source 1332 -- KERNEL-WIDE
#   constant, identical for every work tile. sInfo[3] carries the same value and
#   is published for the record, but the loop bound is this constant.

# ==========================================================================
# Optional pre-kernel: per-expert B descriptors and the scheduler counter
# source 349-414, 617-637
# ==========================================================================
# Emitted only when weight_mode is discrete or the scheduler is dynamic.
# grid = (L, 1, 1) discrete else (1, 1, 1); one thread.
#
# for expert in this_block:                      # discrete only
#     read b_ptrs[expert]                        # reads precede writes
#     build one 128-byte TMA descriptor image for (N, K) at that base
#     store it into workspace[expert * 128 : expert * 128 + 128]
#       # instruction_selection: st.global.v4.b32 x8; extent: one 128 B image per expert
# if sched is dynamic:
#     zero the 4-byte ticket counter at workspace[L * 128]
#
# Single "b" slot at a 128-byte stride; the block-scaled sibling carries two
# slots at 256 B because it also images SFB.

# ==========================================================================
# Warp 7: persistent tile scheduler
# source 1337-1372; sched 420-449; PTX 497-729
# ==========================================================================
# with role(sched_warp):
#     state = scheduler_create(padded_offsets, block_idx, grid_dim, counter?)
#     tile  = initial_work_tile(state)
#     while True:
#         acquire(tile_info_empty[stage], phase)
#         if elect_one():                       # source 1345
#             sInfo[0, stage] = expert_idx if valid else -1
#             sInfo[1, stage] = tile_m_idx
#             sInfo[2, stage] = tile_n_idx
#             sInfo[3, stage] = k_tile_cnt
#               # instruction_selection: st.shared.v4.b32; extent: one 16 B work-tile record
#               # The four words are contiguous i32 and vectorize; the
#               # termination record is the same single store of {-1, 0, 0, 0}.
#         fence_async_shared()
#         named_barrier(4, 32)                  # PTX 520, 726
#         arrive(tile_info_full[stage])
#         if not valid: break
#         tile = advance_to_next_work(state)
#           # static : linear_idx = bidz + i * stride
#           # dynamic: one atom.global.add.u32 ticket, broadcast across the cluster
#         stage, phase = advance(stage, phase, num_tile_stage)
#     producer_tail(tile_info_empty, num_tile_stage)   # source 1369
#       # Consumers release stage S right after reading its four words, so the
#       # producer is free to refill S while the consumers still work that tile.
#       # instruction_selection: 2x blocking try_wait; extent: drain both stages

# ==========================================================================
# Warp 5: A/B TMA loads
# source 1374-1495; PTX 84-93, 863-978
# ==========================================================================
# with role(tma_warp):
#     prefetch the A, B, C and D descriptors             # source 1166-1172
#       # instruction_selection: prefetch.tensormap x4; extent: once per kernel, PTX 84-93
#     # Prologue record, read and released before the loop. The wait, the read,
#     # the fence and the release are adjacent everywhere, so a tile's work runs
#     # from registers with its tile-info stage already free -- that is what keeps
#     # the two-stage pipeline a full tile ahead.
#     acquire(tile_info_full[stage], phase)
#     expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#     fence_async_shared(); arrive(tile_info_empty[stage])
#     stage, phase = advance(stage, phase, num_tile_stage)
#     for each work tile:
#         if expert_idx < 0: break
#         update_expert_info(padded_offsets, expert_idx)     # ext, token range
#         status = True                                      # source 1439
#         if k_tile_cnt > 0:                                 # source 1440
#             status = peek(ab_empty[ab_stage], phase)       # source 1441
#           # No is_leader_cta here, unlike the MMA warp at 1533: warp 5 runs on
#           # both CTAs of the cluster and both must acquire their AB stage. A
#           # true status means SKIP the blocking acquire, so over-guarding this
#           # peek would let a non-leader CTA write into an unreleased stage.
#         for k_tile in range(k_tile_cnt):
#             acquire(ab_empty[ab_stage], phase, unless=status)
#             leader = elect_one()
#             expect_tx(ab_full[ab_stage], 65536, pred=leader)     # PTX 903-904
#               # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: (a+b) x atom_thr bytes
#             tma_load_multicast_3d(sA + ab_stage * 16384, desc_a,
#                                   (k_tile * 64, row_base, 0),
#                                   ab_full[ab_stage], mask_a, pred=leader)
#               # instruction_selection: cp.async.bulk.tensor.3d.shared::cluster...cta_group::2; extent: one 128x64 bf16 tile
#             tma_load_multicast_3d(sB + ab_stage * 16384, desc_b_or_image,
#                                   (k_tile * 64, col_base, expert_idx),
#                                   ab_full[ab_stage], mask_b, pred=leader)
#               # instruction_selection: cp.async.bulk.tensor.3d.shared::cluster...cta_group::2; extent: one 128x64 bf16 half-tile
#             ab_stage, phase = advance(ab_stage, phase, num_ab_stage)
#             if k_tile < k_tile_cnt - 1:                     # source 1458
#                 status = peek(ab_empty[ab_stage], phase)    # source 1459
#         acquire(tile_info_full[stage], phase)   # next tile's record
#         expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#         fence_async_shared(); arrive(tile_info_empty[stage])
#         stage, phase = advance(stage, phase, num_tile_stage)
#     producer_tail(ab_empty, num_ab_stage)                    # source 1495
#       # instruction_selection: 5x blocking try_wait; extent: drain all AB stages
#
# The waits sit outside the elected region; only the expect_tx and the two
# issues carry the lane guard as an instruction predicate, so the warp never
# diverges. In discrete mode `desc_b_or_image` is the workspace slot the helper
# built, bound through a descriptor pointer.

# ==========================================================================
# Warp 4: MMA
# source 1500-1599; PTX 1107-1408
# ==========================================================================
# with role(mma_warp):
#     named_barrier(3, 160)                     # PTX 1107, TMEM base published
#     # Prologue record, read and released before the loop. The wait, the read,
#     # the fence and the release are adjacent everywhere, so a tile's work runs
#     # from registers with its tile-info stage already free -- that is what keeps
#     # the two-stage pipeline a full tile ahead.
#     acquire(tile_info_full[stage], phase)
#     expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#     fence_async_shared(); arrive(tile_info_empty[stage])
#     stage, phase = advance(stage, phase, num_tile_stage)
#     for each work tile:
#         if expert_idx < 0: break
#         status = True                                      # source 1532
#         if k_tile_cnt > 0 and is_leader_cta:               # source 1533
#             status = peek(ab_full[ab_stage], phase)        # source 1534
#         if is_leader_cta:                                  # source 1546
#             acquire(acc_empty[acc_stage], phase)           # source 1547
#             acc_col = acc_stage * cta_tile_n
#             for k_tile in range(k_tile_cnt):
#                 acquire(ab_full[ab_stage], phase, unless=status)
#                 a_desc = smem_descriptor(sA + ab_stage * 16384)
#                 b_desc = smem_descriptor(sB + ab_stage * 16384)
#                 for kblock in range(4):        # k_tile 64 / instruction K 16
#                     if elect_one():
#                         mma_f16(acc_col, a_desc + kblock * a_step,
#                                 b_desc + kblock * b_step, IDESC,
#                                 accumulate=(k_tile != 0 or kblock != 0))
#                           # instruction_selection: tcgen05.mma.cta_group::2.kind::f16; extent: one 256x256x16 issue
#                 umma_arrive(ab_empty[ab_stage])            # source 1579
#                   # instruction_selection: tcgen05.commit...multicast::cluster.b64; extent: AB release, PTX 1325
#                 ab_stage, phase = advance(ab_stage, phase, num_ab_stage)
#                 if k_tile < k_tile_cnt - 1:                # source 1554
#                     status = peek(ab_full[ab_stage], phase)  # source 1555
#             umma_arrive(acc_full[acc_stage])               # source 1581
#               # instruction_selection: tcgen05.commit...multicast::cluster.b64; extent: accumulator publish, PTX 1340
#         acc_stage, phase = advance(acc_stage, phase, num_acc_stage)
#         acquire(tile_info_full[stage], phase)   # next tile's record
#         expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#         fence_async_shared(); arrive(tile_info_empty[stage])
#         stage, phase = advance(stage, phase, num_tile_stage)
#     producer_tail(acc_empty, 1)                              # source 1599
#
# `accumulate` is false only on the very first issue of a tile, which clears the
# accumulator without a separate zeroing pass.
# On a NON-leader CTA the entire guarded block is skipped: no AB acquire, no MMA,
# no AB release, no accumulator acquire or commit. The peer CTA only advances
# `acc_stage` and runs the tile-info handshake; its operand halves reach the
# leader through the cluster multicast, not through this warp.

# ==========================================================================
# Warps 0-3: epilogue
# source 1604-1967; PTX 1427-3556
# ==========================================================================
# with role(epilogue_warps):
#     warp 0 only: tmem_alloc(tmem_cols)        # PTX 1427-1428
#     named_barrier(3, 160)                     # PTX 1431
#     # Prologue record, read and released before the loop. The wait, the read,
#     # the fence and the release are adjacent everywhere, so a tile's work runs
#     # from registers with its tile-info stage already free -- that is what keeps
#     # the two-stage pipeline a full tile ahead.
#     acquire(tile_info_full[stage], phase)
#     expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#     fence_async_shared(); arrive(tile_info_empty[stage])
#     stage, phase = advance(stage, phase, num_tile_stage)
#     for each work tile:
#         if expert_idx < 0: break
#         square_alpha = alpha[expert] * alpha[expert]
#         beta_e       = beta[expert]
#         p            = prob[row_of_this_thread]
#         dProbVal     = 0.0                                   # source 1727
#         acquire(acc_full[acc_stage], phase)                  # PTX 1538
#         for subtile in range(epi_tile_cnt):                  # 8 pairs
#             acc = tmem_load_32x32b_x32(acc_col + subtile * 32)
#               # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: 32 accumulator columns
#               # No tcgen05.wait::ld follows: the reference emits none.
#             g   = fmul2(acc, square_alpha)
#
#             acquire(c_full[gate_stage], phase)
#             gate = smem_ld_v4(sC + gate_stage * 8192)   # 4 issues
#               # instruction_selection: ld.shared.v4.b32 x4; extent: 32 C values per lane
#             fence_async_shared(); arrive(c_empty[gate_stage])   # source 1776-1778
#             acquire(c_full[up_stage], phase)
#             up   = smem_ld_v4(sC + up_stage * 8192)      # 4 issues
#             fence_async_shared(); arrive(c_empty[up_stage])     # source 1785-1787
#             # Both C stages are released BEFORE any activation work, which is
#             # what lets the C-load warp run a full subtile ahead.
#
#             x1 = fmul2(cvt_f32_bf16(gate), beta_e)
#             x2 = fmul2(cvt_f32_bf16(up),   beta_e)
#
#             # ---- dSwiGLU (source 766-942) ----------------------------------
#             #   s      = 1 / (1 + exp2(-LOG2_E * x1))
#             #     # instruction_selection: ex2.approx.ftz.f32 then rcp.approx.ftz.f32; extent: 32 values
#             #   swish  = x1 * s
#             #   dprob_sub  = g * x2 * swish          # ASSIGNED, fresh per subtile
#             #   d1 = g * p * x2 * s * (1 + x1 * (1 - s))
#             #   d2 = g * p * swish
#             #
#             # ---- dGeGLU (source 944-1083) ----------------------------------
#             #   y1 = min(x1, 7.0);  y2 = clamp(x2, -7.0, 7.0)
#             #   s  = 1 / (1 + exp2(-LOG2_E * 1.702 * y1))
#             #   dprob_sub  = g * s * (y2 + linear_offset) * y1   # ASSIGNED per subtile
#             #   d1 = g * s * (1 + 1.702 * y1 * (1 - s)) * (y2 + linear_offset) * p
#             #   d2 = g * y1 * s * p
#             #   d1 *= (x1 <= 7.0 ? y1 : 0.0)
#             #   d2 *= (x2 >= -7.0 ? y2 : 0.0)
#             #     The upper bound plays no part in the d2 filter: for x2 > 7 the
#             #     vectorized path multiplies by y2 (= 7.0), not by zero.
#             #     The SCALAR path differs here -- it yields 0 for x2 > 7 -- so the
#             #     two compiled programs genuinely disagree on that input, and the
#             #     oracle must follow whichever one the specialization selects.
#             #
#             # Packed form when vectorized_f32:
#             #   # instruction_selection: mul.rn.f32x2 / add.rn.f32x2, rnd=rn ftz=false; extent: 16 issues over 32 values
#
#             if with_dbias:
#                 for i in range(32):                        # source 703-705
#                     smem_st_b32(sDbias + warp_base + i * 32 + m_slot, d1[i])
#                     smem_st_b32(sDbias + warp_base + (epi_n + i) * 32 + m_slot, d2[i])
#                   # instruction_selection: st.shared.b32 x64; extent: 64 f32 per lane per subtile
#                   # Scalar, not vectorized: the (32, 1, epi_n*2*32) stride puts
#                   # consecutive n 128 B apart.
#
#             if with_dprob:                                   # source 1808-1829, indent 20
#                 dprob_acc = packed_pairwise_sum(dprob_sub)   # 32 -> 2
#                   # instruction_selection: add.rn.f32x2 x16; extent: 32-element reduction
#                 dProbVal += dprob_acc[0] + dprob_acc[1]      # source 1823, scalar fold
#                 # Reduce-then-sum, once per subtile -- NOT accumulate-then-reduce.
#                 # The summation tree is load-bearing: the oracle reproduces this
#                 # same 32-column subtile grouping rather than summing the row.
#
#             if with_dbias:                                   # source 1834-1848, indent 20
#                 n_base_d1 = tile_n_idx * (tile_n * 2) + (2 * subtile + 0) * 32
#                 n_base_d2 = tile_n_idx * (tile_n * 2) + (2 * subtile + 1) * 32
#                 # dbias_reduction, source 686-765: SMEM transpose, no shuffles.
#                 # Runs once PER SUBTILE, so all 8 column pairs of the work tile
#                 # are reduced and each carries its own n_base.
#                 named_barrier(2, 128)                           # source 707, PTX 2843
#                 col_a = 2 * lane if lane < 16 else epi_n + 2 * (lane - 16)
#                 col_b = col_a + 1
#                 sum_a = sum_b = 0.0
#                 for g in range(8):                              # source 721-733
#                     m_base = g * 4
#                     off_a  = warp_base + col_a * 32 + (m_base ^ (((col_a >> 1) & 0x7) << 2))
#                     off_b  = warp_base + col_b * 32 + (m_base ^ (((col_b >> 1) & 0x7) << 2))
#                     sum_a += sum(smem_ld_b32(off_a + j) for j in range(4))
#                     sum_b += sum(smem_ld_b32(off_b + j) for j in range(4))
#                   # instruction_selection: ld.shared.b32 x64; extent: 32 rows per column pair
#                   # The XOR swizzle applies to the ROW-GROUP BASE, not the column, and
#                   # the 128-bit copy atom decays to scalars because the swizzled offset
#                   # is not provably 16-byte aligned.
#                 named_barrier(2, 128)                           # source 741, PTX 3242
#                 smem_st_b32 x2 -> this warp's 64-bit partial slot   # source 746
#                 named_barrier(2, 128)                           # source 747, PTX 3255
#                 warp 0: total = sum of the four warps' partials      # source 755, ld.shared.b32 x8
#                         if n_offset < dbias_n_total:
#                             atomic_add_bf16x2(&dbias[expert, n_offset], total)
#                               # instruction_selection: cvt.rn.bf16x2.f32 + red.global.add.noftz.bf16x2; extent: one column pair, PTX 3316
#
#             d1_bits = pack_bf16x2(d1 pairs)   # source 1853, 16 issues
#             d2_bits = pack_bf16x2(d2 pairs)   # source 1854, 16 issues
#               # instruction_selection: cvt.rn.bf16x2.f32 x16 each; extent: 32 values per half
#             warp 0: bulk_wait(0)                            # source 1900, PTX 3325
#             named_barrier(2, 128)                           # source 1901, PTX 3383
#             smem_st_v4(sD + d1_stage * 8192, d1_bits)       # source 1904, 4 issues
#             smem_st_v4(sD + d2_stage * 8192, d2_bits)       # source 1911, 4 issues
#               # instruction_selection: st.shared.v4.b32 x4 each; extent: one 128x32 D block per half
#             fence_async_shared()
#             named_barrier(2, 128)                           # source 1917, PTX 3425
#             warp 0: tma_store_3d(desc_d, (col_d1, row, 0), sD + d1_stage * 8192)
#             warp 0: tma_store_3d(desc_d, (col_d2, row, 0), sD + d2_stage * 8192)
#               # instruction_selection: cp.async.bulk.tensor.3d S2G x2; extent: two 128x32 D blocks
#             warp 0: bulk_commit()                           # source 1929, PTX 3449
#             named_barrier(2, 128)                           # source 1930, PTX 3452
#             d1_stage, d2_stage = next two values of (subtile_counter % num_d_stage)
#
#         if elect_one(): arrive(acc_empty[acc_stage])        # source 1935-1936, PTX 3465
#         acc_stage, phase = advance(acc_stage, phase, num_acc_stage)
#         acquire(tile_info_full[stage], phase)          # next tile's record
#         expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#         fence_async_shared(); arrive(tile_info_empty[stage])   # source 1942-1948
#         stage, phase = advance(stage, phase, num_tile_stage)
#         atomic_add_f32(&dprob[row_of_this_thread], dProbVal)   # source 1954
#           # instruction_selection: atom.global.add.f32; extent: one per epilogue thread per work tile
#           # The record is fully released before the flush, as at source 1942-1955.
#
#     tmem_relinquish()                        # source 1960, PTX 3524
#     named_barrier(2, 128)                    # source 1961, PTX 3529
#     arrive(tmem_dealloc_mbar); acquire(tmem_dealloc_mbar)   # source 1962-1965, PTX 3536, 3543
#       # The two-CTA TMEM deallocation handshake, addressed with mapa when the
#       # peer barrier lives in the other CTA.
#     warp 0: tmem_dealloc(tmem_cols)          # PTX 3553
#     bulk_wait(0)                             # PTX 3556 -- after the dealloc
#
# dprob is accumulated per thread across every subtile and flushed once, so its
# summation order is the kernel's 32-column subtile order. dBias and dprob are
# both atomic accumulations into caller-zeroed buffers, so neither is
# bit-reproducible between runs; both are compared with reduction-aware
# tolerances.

# ==========================================================================
# Warp 6: C loads
# source 1971-2041; PTX 3643-3776
# ==========================================================================
# with role(c_load_warp):
#     # Prologue record, read and released before the loop. The wait, the read,
#     # the fence and the release are adjacent everywhere, so a tile's work runs
#     # from registers with its tile-info stage already free -- that is what keeps
#     # the two-stage pipeline a full tile ahead.
#     acquire(tile_info_full[stage], phase)
#     expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#     fence_async_shared(); arrive(tile_info_empty[stage])
#     stage, phase = advance(stage, phase, num_tile_stage)
#     for each work tile:
#         if expert_idx < 0: break
#         for subtile in range(epi_tile_cnt):           # 8 pairs
#             for half in (0, 1):                       # gate block, then up block
#                 acquire(c_empty[c_stage], phase)
#                 leader = elect_one()
#                 expect_tx(c_full[c_stage], 8192, pred=leader)     # PTX 3658, 3691
#                 tma_load_3d(sC + c_stage * 8192, desc_c,
#                             (col_base + (2 * subtile + half) * 32, row_base, 0),
#                             c_full[c_stage], pred=leader)
#                   # instruction_selection: cp.async.bulk.tensor.3d.shared::cta...; extent: one 128x32 C block, PTX 3667/3703
#                 c_stage, phase = advance(c_stage, phase, num_c_stage)
#         acquire(tile_info_full[stage], phase)   # next tile's record
#         expert_idx, tile_m_idx, tile_n_idx, _ = sInfo[:, stage]
#         fence_async_shared(); arrive(tile_info_empty[stage])
#         stage, phase = advance(stage, phase, num_tile_stage)
#     producer_tail(c_empty, num_c_stage)                       # source 2041
#
# The gate and up halves land in SEPARATE C stages, which is why num_c_stage is
# 2 and why the epilogue consumes them as a pair. C is the only `shared::cta`
# bulk load; A and B are cluster-multicast.
```

## Source / sketch / PTX correspondence

The PTX column carries **anchor-export line numbers**.

| region | source | sketch section | PTX |
| --- | --- | --- | --- |
| specialization and launch | 123-345, 416-566 | Static specialization | 14, 38-39 |
| shared storage, mbarrier init | 568-611, 1189-1250 | Storage and synchronization | 118-243 |
| init fence, CTA barrier, cluster sync | 1245, 1272-1282, 1325-1328 | Storage and synchronization | 232-233, 246-262 |
| descriptor prefetch | 1166-1172 | Warp 5 | 84-93 |
| descriptor pre-kernel | 349-414, 617-637 | Optional pre-kernel | separate entry |
| scheduler warp | 1337-1372 | Warp 7 | 497-729 |
| A/B TMA warp | 1374-1495 | Warp 5 | 863-978 |
| MMA warp | 1500-1599 | Warp 4 | 1107-1408 |
| TMEM allocation | 1604-1613 | Warps 0-3 | 1427-1431 |
| C s2r and release | 1770-1787 | Warps 0-3 | 1651-1729 |
| activations | 766-942, 944-1083 | Warps 0-3 | 1978 onward |
| dprob packed reduction | 1808-1829 | Warps 0-3 | `.loc 1 1817` |
| D convert and store | 1853-1930 | Warps 0-3 | 3325-3452 |
| dBias reduction | 686-765 | Warps 0-3 | 2843-3316 |
| accumulator release | 1935-1937 | Warps 0-3 | 3465 |
| dprob flush | 1950-1955 | Warps 0-3 | 3513 |
| teardown | 1960-1967 | Warps 0-3 | 3524-3556 |
| C-load warp | 1971-2041 | Warp 6 | 3643-3776 |
| stage solve | 2239-2282 | Anchor values | n/a |

## Out of scope

Unreachable in this port's domain, with the predicate that excludes each:
- every scale-factor, quantization, amax and `dsituglu` path — absent from the
  bf16 source entirely (`sfa|sfb|SFD|amax` occurs 353 times in the block-scaled
  sibling and **0** times here);
- `store_d_directly` and its `stg_256` path — hard-coded false (`:203`), so the
  256-bit STG epilogue is dead code;
- `epilogue_prefetch_more` — hard-coded false (`:324`);
- a `prob`-less call — `generate_dprob` is unconditionally true (`:325`) and the
  host API rejects a missing `prob`/`dprob` pair.
