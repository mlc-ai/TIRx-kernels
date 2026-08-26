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
- sha256: `d448b5c9ddd4514f96340aa1a620894a48e884f4216c56715119d56c37bd9e38`, 2284 lines
- entry: `MoEGroupedGemmDgluDbiasBf16Kernel.__call__` -> optional `helper_kernel`, then `kernel`

Citation shorthands used below: a bare `N` is a line in the source above;
`sched N` is `grouped/moe_persistent_scheduler.py`, `utils N` is
`grouped/moe_utils.py`, `ext N` is `grouped/moe_sched_extension.py`, and
`helpers N` is `grouped/moe_kernel_helpers.py`. `PTX n` is a line in the anchor
export named in the evidence table.

## PTX `.file` numbering

The exports number the same four files consistently across every branch:

| `.file` | path |
| --- | --- |
| 1 | `grouped/dglu/moe_grouped_gemm_dglu_dbias.py` |
| 2 | `grouped/moe_persistent_scheduler.py` |
| 3 | `grouped/moe_utils.py` |
| 4 | `grouped/moe_sched_extension.py` |

## Evidence

Writer exports under `.porting/moe_grouped_gemm_dglu_dbias/writer_source_export/`,
produced by `export_ptx.py` with `CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=ptx
CUTE_DSL_NO_CACHE=1`. Every run asserts its outputs are finite and non-zero, so
a degenerate export cannot be annotated. The reviewer's export is independent
and does not reuse these.

| branch | axis it turns on | ptx lines | `.loc` | sha256 | MMA idesc |
| --- | --- | --- | --- | --- | --- |
| `anchor` | the annotated specialization | 3793 | 1159 | `ab1d446e` | `0x10400490` |
| `discrete` | discrete-B weights + helper descriptors | 3868 | 1173 | `3162ca17` | `0x10400490` |
| `dynamic` | atomic-ticket scheduler | 4004 | 1221 | `53e3cb42` | `0x10400490` |
| `dgeglu` | the second activation | 4840 | 1481 | `14b95ffa` | `0x10400490` |
| `cf32_nodbias` | FP32 C, dBias off | 3063 | 925 | `1d38f93a` | `0x10400490` |
| `bmajor_n` | n-major B | 3796 | 1153 | `6b9cb225` | `0x10410490` |
| `tile128_c1x1` | one-CTA tile, singleton cluster | 3604 | 1115 | `f425a929` | `0x08400490` |
| `tile_n64` | narrow tile_n | 3683 | 1127 | `098f01b9` | `0x08100490` |
| `scalar_f32` | scalar (non-packed) epilogue math | 3501 | 849 | `201ea257` | `0x10400490` |

Instruction counts below follow the corpus convention: instruction lines minus
predicated lines. For `anchor` that is 2263 - 120 = **2143**.

## Anchor values

The annotated specialization, and the values every `extent:` annotation refers
to unless it says otherwise:

| quantity | value | source |
| --- | --- | --- |
| mode | dense weights, static scheduler, dSwiGLU, C and D bf16, k-major B, packed-f32 epilogue, dBias on | |
| `mma_tiler_mn` | `(256, 256)`, so `use_2cta_instrs`, `atom_thr = 2` | `:123-133` |
| `cta_tile_shape_mnk` | `(128, 256, 64)` | `:212-230` |
| `cluster_shape_mn` | `(2, 1)` | |
| shape | 4 experts x 256 tokens, `N = 512`, `K = 512` | |
| `epi_tile` | `(128, 32)`, 8 subtile pairs per tile | `:328` |
| `k_tile` | 64 = instruction K 16 x `mma_inst_tile_k` 4 | `:224-230` |
| stages | acc 2, AB 3 (4 without dBias), C 2, D 2, tile-info 2 | `:2239-2282` |
| TMEM columns | **512** (`PTX 1427-1428`), `= clamp(next_pow2(2 * cta_tile_n), 32, 512)` | `:284-289` |
| threads | 256 = 8 warps, `.maxntid 256, 1, 1`, `.minnctapersm 1` | `PTX 38-39` |
| grid | `(cluster_m, cluster_n, max_active_clusters)` persistent | `:551`, helpers:988-992 |
| shared memory | one dynamic `.extern .shared .align 1024 .b8` arena | `PTX 14` |

**The TMEM column count is derived, not pinned.** The closed form above was
checked against three exports: `anchor` and `tile128_c1x1` allocate 512 columns,
`tile_n64` allocates **128** (`tcgen05.alloc` immediate). This is the single
largest structural departure from the block-scaled sibling, which always
reserves 512 and hand-partitions scale-factor regions inside it.

## The MMA instruction descriptor

Lifted from the exports, never adapted from the block-scaled encoder. The
reference builds it as `selp`-selected constants immediately before the MMA
(`PTX 1250-1252`), then issues
`tcgen05.mma.cta_group::2.kind::f16` (`PTX 1264`, `.loc 1 1568`):

```text
idesc = base | (p1 ? 1<<13 : 0) | (p2 ? 1<<14 : 0)
base(anchor) = 0x10400490
```

Field decode, confirmed by differencing the branch exports rather than by
reading a header:

| bits | field | anchor | evidence |
| --- | --- | --- | --- |
| 4-6 | D format | 1 (f32) | constant across branches |
| 7-9 | A format | 1 (bf16) | constant across branches |
| 10-12 | B format | 1 (bf16) | constant across branches |
| 13 | A negate | runtime `%p1` | `TiledMMA` kernel param byte, `PTX 48-52` |
| 14 | B negate | runtime `%p2` | `TiledMMA` kernel param byte, `PTX 48-52` |
| 16 | transpose B | 0 here, **1** in `bmajor_n` | `0x10400490` vs `0x10410490` |
| 17-22 | `N >> 3` | 32 -> N 256 | `tile_n64` gives 8 -> N 64 |
| 23 | SF format | **0** | the block-scaled kinds set this; `kind::f16` never does |
| 24-28 | `M >> 4` | 16 -> M 256 | `tile128_c1x1` gives 8 -> M 128 |

Bits 13 and 14 arrive as `TiledMMA` **kernel parameters** that this kernel never
sets, so both predicates are false at run time. The port therefore emits a
compile-time constant descriptor:

```text
idesc = (1 << 4) | (1 << 7) | (1 << 10)
      | (transpose_b << 16) | ((tile_n >> 3) << 17) | ((tile_m >> 4) << 24)
```

`# instruction_selection: tcgen05.mma.cta_group::{1,2}.kind::f16; extent: one k-block of the 64-wide k tile, four issues per tile`

## No first-class layouts

Neither this sketch nor the device kernel introduces a first-class layout, a
layout algebra object, or a multidimensional shared-memory tensor. Every shared
region is a one-dimensional byte range inside a single flat `u8` arena, indexed
by explicit scalar offset arithmetic; matrix descriptors are assembled from
hardware immediates. The upstream `SharedStorage` struct and its `cute` layouts
are documentation of the byte map only.

## Pipeline at a glance

Eight warps, 256 threads, one CTA per multiprocessor, persistent over work
tiles handed out by the scheduler warp. Roles and their ids are unchanged from
the block-scaled sibling (`:157-181`).

| warp | role | tile program | publishes / consumes |
| --- | --- | --- | --- |
| 0-3 | epilogue | per subtile pair: read accumulator from TMEM, scale by `alpha^2`, read C, form the two GLU gradients, accumulate dprob and dBias, write D through SMEM | consumes `acc_full`, `c_full`; releases `acc_empty`, `c_empty`; warp 0 additionally owns the TMEM allocation and the D TMA store |
| 4 | MMA | per k tile: issue four `tcgen05.mma.kind::f16`, accumulate into the current TMEM stage | consumes `ab_full`; releases `ab_empty`; publishes `acc_full` via `tcgen05.commit` |
| 5 | A/B TMA | per k tile: `expect_tx` then two `cp.async.bulk.tensor` into the AB ring | consumes `ab_empty`; publishes `ab_full` |
| 6 | C load | per subtile pair: two `cp.async.bulk.tensor` (gate block, then up block) into the two C stages | consumes `c_empty`; publishes `c_full` |
| 7 | scheduler | walk work tiles, flatten each into `sInfo`, terminate with `expert_idx = -1` | publishes `tile_info`; consumes `tile_info` release |

Named barriers (`:186-206`), unchanged: id 1 CTA-wide (256 threads), id 2
epilogue (128), id 3 TMEM allocation (160 = MMA warp + four epilogue warps),
id 4 scheduler (32). `PTX` shows 12 `bar.sync`, 25 `mbarrier.init`, 33
`mbarrier.try_wait`, 25 `elect.sync`.

**Suspend hint.** Every reference `try_wait` carries the immediate `10000000`
(`PTX 1498`, `.loc 1 1498`). The port matches it. This is the opposite choice
from the linear-attention family, where the reference spins with a hint of 1;
the hint is a per-kernel fact to be read off the export, not a global default.

`# instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 with suspend hint 10000000; extent: one spin per pipeline handshake`

## Primitive vocabulary

The sketch uses only these, each of which is one PTX instruction, one tile, or
one loop of one family:

- `tma_load_3d(dst_byte, desc, coords, barrier)` — one
  `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`
- `tma_store_3d(desc, coords, src_byte)` — one S2G bulk tensor copy
- `expect_tx(barrier, bytes)` / `arrive(barrier)` / `wait(barrier, phase)` —
  one `mbarrier.arrive.expect_tx` / `mbarrier.arrive` / `mbarrier.try_wait.parity`
- `mma_f16(acc_tmem, a_desc, b_desc, idesc, accumulate)` — one
  `tcgen05.mma.cta_group::N.kind::f16`
- `mma_commit(barrier)` — one `tcgen05.commit...multicast::cluster.b64`
- `tmem_alloc(cols)` / `tmem_dealloc` / `tmem_relinquish` — one tcgen05 alloc,
  dealloc, or relinquish
- `tmem_load_32x32b_x32(regs, col)` — one `tcgen05.ld.sync.aligned.32x32b.x32.b32`
- `smem_ld_v4(regs, byte)` / `smem_st_v4(byte, regs)` — one `ld.shared.v4.b32` /
  `st.shared.v4.b32`
- `cvt_f32_bf16(x)` / `pack_bf16x2(lo, hi)` — one `cvt.f32.bf16` /
  `cvt.rn.bf16x2.f32`
- `fmul2(a, b)` / `fadd2(a, b)` — one `mul.rn.f32x2` / `add.rn.f32x2`
- `exp2(x)` / `rcp(x)` — one `ex2.approx.ftz.f32` / `rcp.approx.ftz.f32`
- `atomic_add_f32(addr, v)` — one `atom.global.add.f32`
- `atomic_add_bf16x2(addr, lo, hi)` — one `cvt.rn.bf16x2.f32` plus one
  `red.global.add.noftz.bf16x2`
- `named_barrier(id, threads)` — one `bar.sync`
- `elect_one()` — one `elect.sync`

## Complete sketch

```python
# ==========================================================================
# Static specialization, runtime ABI, and launch
# source 123-345, 416-566; PTX 14-39
# ==========================================================================
# Compile-time: weight_mode, sched, act, c_dtype, d_dtype, b_major, tile_m,
# tile_n, cluster_shape, vectorized_f32, with_dbias, expert_cnt, N, K,
# group_m_list, linear_offset. Everything the reference takes at run time and
# the port can know statically is baked, which is what makes the byte map exact.
#
# Runtime operands, in ABI order:
#   a, b (dense tensor or discrete int64 pointer array), c, d,
#   padded_offsets, alpha, beta, prob, dprob, dbias?, workspace?
#
# grid   = (cluster_m, cluster_n, MAX_ACTIVE_CLUSTERS[cluster_m * cluster_n])
# block  = 256 threads = 8 warps
# cluster= (cluster_m, cluster_n) bound unconditionally, including (1, 1)
#          # instruction_selection: cta_id_in_cluster with preferred=; extent: whole kernel
#          The block-scaled sibling binds it the same way; `preferred=` is what
#          keeps the extent-1 case from folding away and losing the cluster
#          launch attribute.

# ==========================================================================
# Storage and synchronization objects
# source 568-676; PTX 14
# ==========================================================================
# One flat `u8` arena, `K.alloc_buffer((SMEM_BYTES,), K.u8, scope="shared.dyn",
# align=1024)`, carved by explicit scalar byte offsets in the upstream
# declaration order. Sizes for the anchor:
#
#   ab_mbar        num_ab_stage * 2 * 8 B      full/empty per AB stage
#   acc_mbar       num_acc_stage * 2 * 8 B     full/empty per accumulator stage
#   sched block    tile_info_mbar 4 * 8 B
#                  sInfo          4 * num_tile_stage * 4 B, align 16
#                  cluster_mbar   2 * 8 B      dynamic scheduler only
#                  cluster_bcast  4 * 4 B      dynamic scheduler only
#   c_full/c_empty num_c_stage * 8 B each
#   tmem_dealloc   8 B
#   tmem_holding   4 B
#   sC             128 * 32 * num_c_stage * sizeof(c_dtype)   = 16 KiB
#   sD             128 * 32 * num_d_stage * sizeof(d_dtype)   = 16 KiB
#   sA             128 * 64 * 2 B per stage                   = 16 KiB/stage
#   sB             256 *  64 * 2 B per stage                  = 32 KiB/stage
#   sDbias         128 * 64 * 4 B (f32) when with_dbias        = 32 KiB
#
# num_ab_stage is whatever the remaining budget allows against the 232448 B
# capacity: 3 with dBias, 4 without.
#
# TMEM: one allocation of `clamp(next_pow2(2 * cta_tile_n), 32, 512)` columns,
# split into num_acc_stage accumulator regions of cta_tile_n columns each.
#   # instruction_selection: tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32; extent: once per kernel
# Warp 0 allocates, publishes the base through `tmem_holding`, and the CTA
# reads it after named barrier 3.

# ==========================================================================
# Optional pre-kernel: per-expert B descriptors and the scheduler counter
# source 350-414, 617-637
# ==========================================================================
# Emitted only when weight_mode is discrete or the scheduler is dynamic.
# grid = (L, 1, 1) discrete else (1, 1, 1); one thread.
#
# for expert in this_block:                      # discrete only
#     read b_ptrs[expert]
#     build one 128-byte TMA descriptor image for (N, K) at that base
#     store it into workspace[expert * 128 : expert * 128 + 128]
#       # instruction_selection: st.global.v4.b32 x8; extent: one 128 B image per expert
# if sched is dynamic:
#     zero the 4-byte ticket counter at workspace[L * 128]
#
# Single "b" slot at a 128-byte stride -- the block-scaled sibling carries two
# slots at 256 B because it also images SFB. Reads precede writes so an expert
# never reads a slot another block has already overwritten.

# ==========================================================================
# Warp 7: persistent tile scheduler
# source 1252-1361; sched 420-449; PTX .file 2
# ==========================================================================
# with role(sched_warp):
#     state = scheduler_create(padded_offsets, block_idx, grid_dim, counter?)
#     tile  = initial_work_tile(state)
#     while True:
#         wait(tile_info_empty[stage], phase)
#         if tile is valid:
#             sInfo[0, stage] = expert_idx
#             sInfo[1, stage] = tile_m_idx
#             sInfo[2, stage] = tile_n_idx
#             sInfo[3, stage] = k_tile_cnt
#         else:
#             sInfo[0, stage] = -1            # the termination token
#         fence_proxy_async_shared()
#         named_barrier(4, 32)
#         arrive(tile_info_full[stage])
#         if not valid: break
#         tile = advance_to_next_work(state)
#           # static : linear_idx = bidz + i * stride
#           # dynamic: one atom.global.add.u32 ticket, broadcast across the cluster
#         stage, phase = advance(stage, phase, num_tile_stage)
#
# Every consumer warp reads the same sInfo words, so the tile identity is
# published once rather than recomputed per warp.

# ==========================================================================
# Warp 5: A/B TMA loads
# source 1165-1172, 1321-1416
# ==========================================================================
# with role(tma_warp):
#     prefetch the A, B, C and D descriptors                 # source 1165-1172
#       # instruction_selection: prefetch.tensormap; extent: once per kernel
#     for each work tile:
#         wait(tile_info_full[stage], phase); read sInfo
#         if expert_idx < 0: break
#         update_expert_info(padded_offsets, expert_idx)     # ext, token range
#         for k_tile in range(k_tile_cnt):
#             wait(ab_empty[ab_stage], phase)
#             leader = elect_one()
#               # instruction_selection: elect.sync; extent: one per transfer group
#             expect_tx(ab_full[ab_stage], a_bytes + b_bytes, pred=leader)
#             tma_load_3d(sA + ab_stage * a_stage_bytes, desc_a,
#                         (k_tile * 64, token_offset + tile_m_idx * tile_m, 0),
#                         ab_full[ab_stage], pred=leader)
#               # instruction_selection: cp.async.bulk.tensor.3d...L2::cache_hint; extent: one 128x64 bf16 tile
#             tma_load_3d(sB + ab_stage * b_stage_bytes, desc_b_or_expert_image,
#                         (k_tile * 64, tile_n_idx * tile_n, expert_idx),
#                         ab_full[ab_stage], pred=leader)
#               # instruction_selection: cp.async.bulk.tensor.3d...L2::cache_hint; extent: one 256x64 bf16 tile
#             ab_stage, phase = advance(ab_stage, phase, num_ab_stage)
#
# The waits sit outside the elected region and the two issues carry the lane
# guard as an instruction predicate, so the warp never diverges.
# In discrete mode `desc_b_or_expert_image` is the workspace slot the helper
# built, bound through the descriptor pointer rather than a static descriptor.

# ==========================================================================
# Warp 4: MMA
# source 1471-1660; PTX 1250-1318
# ==========================================================================
# with role(mma_warp):
#     named_barrier(3, 160)                     # TMEM base is published
#     for each work tile:
#         wait(tile_info_full[stage], phase); read sInfo
#         if expert_idx < 0: break
#         wait(acc_empty[acc_stage], phase)
#         acc_col = acc_stage * cta_tile_n
#         for k_tile in range(k_tile_cnt):
#             wait(ab_full[ab_stage], phase)
#             a_desc = smem_descriptor(sA + ab_stage * a_stage_bytes)
#             b_desc = smem_descriptor(sB + ab_stage * b_stage_bytes)
#             for kblock in range(4):           # k_tile 64 / instruction K 16
#                 mma_f16(acc_col, a_desc + kblock * a_step,
#                         b_desc + kblock * b_step, IDESC,
#                         accumulate=(k_tile != 0 or kblock != 0))
#                   # instruction_selection: tcgen05.mma.cta_group::2.kind::f16; extent: one 256x256x16 issue
#             arrive(ab_empty[ab_stage])
#             ab_stage, phase = advance(ab_stage, phase, num_ab_stage)
#         mma_commit(acc_full[acc_stage])
#           # instruction_selection: tcgen05.commit...multicast::cluster.b64; extent: one per accumulator stage
#         acc_stage, phase = advance(acc_stage, phase, num_acc_stage)
#
# `accumulate` is false only on the very first issue of a tile, which is what
# clears the accumulator without a separate zeroing pass.
# Under a two-CTA atom only the leader CTA issues; the peer contributes its
# operand halves and waits on the same barriers.

# ==========================================================================
# Warp 6: C loads
# source 2005-2027
# ==========================================================================
# with role(c_load_warp):
#     for each work tile:
#         wait(tile_info_full[stage], phase); read sInfo
#         if expert_idx < 0: break
#         for subtile in range(epi_tile_cnt):           # 8 pairs for tile_n 256
#             for half in (0, 1):                       # gate block, then up block
#                 wait(c_empty[c_stage], phase)
#                 leader = elect_one()
#                 expect_tx(c_full[c_stage], c_tile_bytes, pred=leader)
#                 tma_load_3d(sC + c_stage * c_stage_bytes, desc_c,
#                             (col_base + (2 * subtile + half) * 32,
#                              row_base, 0),
#                             c_full[c_stage], pred=leader)
#                   # instruction_selection: cp.async.bulk.tensor.3d G2S; extent: one 128x32 C block
#                 c_stage, phase = advance(c_stage, phase, num_c_stage)
#
# The gate and up halves land in *separate* C stages, which is why num_c_stage
# is 2 and why the epilogue consumes them as a pair.

# ==========================================================================
# Warps 0-3: epilogue
# source 1700-2172, 2344-2384; PTX 1766-1911, 3316, 3513
# ==========================================================================
# with role(epilogue_warps):
#     warp 0 only: tmem_alloc(tmem_cols); publish base; tmem_relinquish
#     named_barrier(3, 160)
#     for each work tile:
#         wait(tile_info_full[stage], phase); read sInfo
#         if expert_idx < 0: break
#         square_alpha = alpha[expert] * alpha[expert]
#         beta_e       = beta[expert]
#         p            = prob[row_of_this_thread]
#         dprob_acc    = 0.0
#         wait(acc_full[acc_stage], phase)
#         for subtile in range(epi_tile_cnt):
#             acc = tmem_load_32x32b_x32(acc_col + subtile * 32)
#               # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: 32 accumulator columns
#             g   = fmul2(acc, square_alpha)
#             wait(c_full[gate_stage], phase);  gate = smem_ld_v4(sC + gate_stage * ...)
#             wait(c_full[up_stage],   phase);  up   = smem_ld_v4(sC + up_stage   * ...)
#               # instruction_selection: ld.shared.v4.b32; extent: 4 C values per lane per issue
#             x1 = fmul2(cvt_f32_bf16(gate), beta_e)
#             x2 = fmul2(cvt_f32_bf16(up),   beta_e)
#
#             # ---- dSwiGLU (source 767-943) ----------------------------------
#             #   s      = 1 / (1 + exp2(-LOG2_E * x1))
#             #     # instruction_selection: ex2.approx.ftz.f32 then rcp.approx.ftz.f32; extent: one per value
#             #   swish  = x1 * s
#             #   dprob_acc += g * x2 * swish
#             #   d1 = g * p * x2 * s * (1 + x1 * (1 - s))
#             #   d2 = g * p * swish
#             #
#             # ---- dGeGLU (source 945-1086) ----------------------------------
#             #   y1 = min(x1, 7.0);  y2 = clamp(x2, -7.0, 7.0)
#             #   s  = 1 / (1 + exp2(-LOG2_E * 1.702 * y1))
#             #   dprob_acc += g * s * (y2 + linear_offset) * y1
#             #   d1 = g * s * (1 + 1.702 * y1 * (1 - s)) * (y2 + linear_offset) * p
#             #   d2 = g * y1 * s * p
#             #   d1 *= (x1 <= 7.0 ? y1 : 0.0)          # the mask carries y1, not 1
#             #   d2 *= (|x2| <= 7.0 ? y2 : 0.0)
#             #
#             # Both chains run on packed pairs when vectorized_f32:
#             #   # instruction_selection: mul.rn.f32x2 / add.rn.f32x2, rnd=rn ftz=false; extent: two values per issue
#             # The scalar branch is the same algebra on single values and is a
#             # separate compiled program, not a runtime choice.
#
#             if with_dbias:
#                 sDbias[column-major slot for this warp] = d1, d2
#                   # instruction_selection: st.shared.v4.b32; extent: 4 f32 per lane
#             smem_st_v4(sD + d_stage * ..., pack_bf16x2(d1, d2))
#               # instruction_selection: st.shared.v4.b32; extent: one 128x32 D block per half
#             arrive(c_empty[gate_stage]); arrive(c_empty[up_stage])
#             warp 0: tma_store_3d(desc_d, (col, row, 0), sD + d_stage * ...)
#               # instruction_selection: cp.async.bulk.tensor.3d S2G; extent: one 128x32 D block
#
#         arrive(acc_empty[acc_stage])
#         atomic_add_f32(&dprob[row_of_this_thread], dprob_acc)
#           # instruction_selection: atom.global.add.f32; extent: one per epilogue thread per work tile
#
#         if with_dbias:
#             # dbias_reduction, source 686-765: SMEM transpose, no shuffles
#             named_barrier(2, 128)
#             col_a = 2 * lane if lane < 16 else epi_n + 2 * (lane - 16)
#             col_b = col_a + 1
#             swizzled = col ^ (((col >> 1) & 0x7) << 2)      # bank-conflict free
#             sum_a, sum_b = 0.0, 0.0
#             for i in range(8):
#                 sum_a += smem_ld_v4(sDbias + swizzled_a + i * 16)   # 4 rows each
#                 sum_b += smem_ld_v4(sDbias + swizzled_b + i * 16)
#               # instruction_selection: ld.shared.v4.b32; extent: 32 rows per column
#             store (sum_a, sum_b) as one 64-bit slot for this warp
#             named_barrier(2, 128)
#             warp 0: total = sum of the four warps' slots
#                     if n_offset < dbias_n_total:
#                         atomic_add_bf16x2(&dbias[expert, n_offset], total)
#                           # instruction_selection: cvt.rn.bf16x2.f32 + red.global.add.noftz.bf16x2; extent: one column pair
#
#     cp_async_bulk_wait_group(0)                 # PTX 1967
#     named_barrier(1, 256)
#     warp 0: tmem_dealloc(tmem_cols)             # PTX 1968
#
# dprob is accumulated per thread across every subtile of the tile and flushed
# once, so its summation order is the kernel's 32-column subtile order -- the
# oracle reproduces that order deliberately rather than summing the row.
# dBias and dprob are both atomic accumulations into caller-zeroed buffers, so
# neither is bit-reproducible between runs and both are compared with
# reduction-aware tolerances.
```

## Source / sketch / PTX correspondence

| region | source | sketch section | PTX |
| --- | --- | --- | --- |
| specialization and launch | 123-345, 416-566 | Static specialization | 14-39 |
| shared storage and mbarriers | 568-676, 1195-1249 | Storage and synchronization | 1197, 1211, 1227, 1245 |
| descriptor pre-kernel | 350-414, 617-637 | Optional pre-kernel | separate entry |
| scheduler warp | 1252-1361 | Warp 7 | `.file` 2 |
| A/B TMA warp | 1165-1172, 1321-1416 | Warp 5 | 2010-2019 |
| MMA warp | 1471-1660 | Warp 4 | 1250-1318 |
| C-load warp | 2005-2027 | Warp 6 | 2011, 2013, 2018, 2021 |
| epilogue and activations | 767-1086, 1700-2172 | Warps 0-3 | 1766-1911 |
| dprob flush | 2344-2384 | Warps 0-3 | 3513 |
| dBias reduction | 686-765 | Warps 0-3 | 3316 |
| teardown | 1960-1985 | Warps 0-3 | 1967-1981 |

## Out of scope

Unreachable in this port's domain, with the predicate that excludes each:
- every scale-factor, quantization, amax and `dsituglu` path — absent from the
  bf16 source entirely (`sfa|sfb|SFD|amax` occurs 353 times in the block-scaled
  sibling and **0** times here);
- `store_d_directly` and its `stg_256` path — the source hard-codes the flag
  false (`:202`), so the 256-bit STG epilogue is dead code;
- `epilogue_prefetch_more` — hard-coded false (`:328`);
- a `prob`-less call — `generate_dprob` is unconditionally true (`:329`) and the
  host API rejects a missing `prob`/`dprob` pair.
