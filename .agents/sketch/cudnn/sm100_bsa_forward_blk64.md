<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 blk64 block-sparse attention forward: coarse WASP pipeline sketch

This is a non-executable execution sketch. It freezes the launch contract,
linear storage, sixteen-warp role split, sparse gather order, asynchronous
protocols, online row-statistics dataflow, split-P publication, correction
exchange, output path, singleton-cluster CLC branch, and split-KV combine for
[`tirx_kernels/cudnn/bsa/block_sparse_attention_forward_sm100_blk64.py`](../../../tirx_kernels/cudnn/bsa/block_sparse_attention_forward_sm100_blk64.py).
That module and its private implementation package are the executable source of
truth after the reviewer gate; this sketch remains frozen.

The source is the production `bsa_attn_fwd_blk64_cutedsl` path in cuDNN
Frontend. The producer comes from
`python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_sm100.py`;
the split reduction comes from adjacent `bsa_fwd_combine.py`; the wrapper and
compile policy are in `python/cudnn/block_sparse_attention/_interface.py`.
Source-line citations below use those files at commit
`7b5327b32907b9dd21d85a393d62f9573d7f0116`.

First-class layouts are forbidden throughout the sketch and implementation.
All device SMEM is one linear byte allocation. The source's layout objects are
resolved here into scalar extents, byte offsets, index functions, swizzle bits,
TensorMap fields, and raw MMA descriptors. TMEM regions are integer columns;
register fragments are ordinary scalar/vector arrays. “Tile” in prose means a
logical rectangular region, never a TIRx tile primitive or layout-bearing value.

## Scope and specializations

The fixed producer geometry is BF16 with a constexpr non-packed Q-to-KV head map
(`H_q = r * H_kv`), `D=DV=128`, Q rows 64, one grouped KV iteration of 256 tokens
assembled from four indexed sparse blocks of 64, one-CTA UMMA, and FP32
score/output accumulation. The producer always uses 512 threads, one Q stage,
three aliased K/V stages, two alternating score/P stages, two O accumulator
stages, and all 512 TMEM columns. Source `bsa_fwd_sm100.py:53-188`.

In scope:

| axis | accepted values | device-code consequence |
| --- | --- | --- |
| head map | MHA `r=1` / GQA / MQA, any `r` with `H_q % H_kv == 0` | only the K/V TMA head coordinate becomes `h//r`; Q, sparse metadata, O, LSE and the grid all stay `H_q`-indexed, and batch remains a separate coordinate contribution folded into the K/V block axis, never into the head. `r=1` folds at trace time and emits nothing, matching the source. For `r>1` the emitted form depends on the work coordinate's provenance — see the head-map instruction selection at the warp-14 body |
| input presentation | BHSD / BSHD | BSHD is made contiguous as BHSD by the wrapper before either timed launch; producer addressing is the same |
| sparse count | fixed / per-Q-block variable | fixed reads one scalar; variable loads `block_nums[b,h,qb]`; the latter may specialize an empty-tile branch |
| block tail mask | present / absent | present loads physical `block_sizes[block_id]`; absent substitutes 64 without a GMEM access |
| scheduling | static / CLC | static uses ordinary `K.cta_id()`; CLC explicitly calls `K.cta_id_in_cluster([1,1,1])` and dedicates warp 15 to cluster launch control |
| split-KV | 1..256, including wrapper `auto` | values greater than one produce FP32 partial O/LSE, disable CLC, and launch the independent 128-thread combine |
| K/V stride width | normal i32 / active i64 | only the K/V global address and TensorMap stride path widens; normal metadata and loop counters stay i32 |
| output dtype | BF16 final / FP32 producer partial | FP32 only when split-KV is active; final combine converts to BF16 |
| scale | default `1/sqrt(128)` / runtime f32 | producer carries both the natural and log2-domain scale |

Out of scope: FP16, PackGQA (the packed Q-head layout), causal attention, `D` or
`DV` other than 128, sparse block sizes other than 64, Q tiles other than 64,
grouped KV tiles other than 256, multi-CTA MMA, cluster dimensions greater than
one, and every SM90, SM110-specific, or SM120 sibling. Tile primitives are out of
scope.

The two head-adjacent exclusions are properties of the source, not choices. FP16
is excluded because the source's own conversion helpers emit BF16 unconditionally
(`bsa_fwd_helpers.py:136-148` for the softmax P operand and `:322-379` for the
non-FP32 output store), so an FP16 launch would disagree with the instruction
descriptor rather than fail. PackGQA is excluded because the source's packed LSE
row bound (`bsa_fwd_sm100.py:2211`) uses the unpacked `seqlen_q` where the packed
layout needs `seqlen_q * r`, silently dropping rows; the blk128 sibling carries
the corrected form. Both are refused by the source's own host surface.

Static and split producer launches are ordinary launches. CLC is an explicit
singleton-cluster launch even though every dimension has extent one; presence
of `clusterCtaIdx.*`, not the numeric extent, is the contract. Split-KV and CLC
are mutually exclusive (`_interface.py:1167-1179`).

## Line-info export used for instruction selection

The writer exported and ran three structurally distinct production paths and a
generated intermediate. The manifest is preserved at
`.porting/block_sparse_attention_forward_sm100_blk64/writer_source_export/manifest.md`.

| export | result | artifact SHA256 | relevant launch |
| --- | --- | --- | --- |
| static | source O/LSE and FP32 gathered oracle PASS | `1b43896ab9956289433237434a80d47c26bd3f7a3626cbb32acf918f5d4e9ee9` | `.reqntid 512,1,1`, ordinary CTA |
| CLC | source O/LSE and oracle PASS | `0c88704ea28668e20e0bf46a96621d8c771ae8f3f16eb41988a1246b1d484779` | `.reqntid 512,1,1`, CLC instructions and singleton cluster request |
| split producer | source partial/final result and oracle PASS | `83cd2a7372f594088dc12a21e05120020107a7bfe51830cbd12177893f8247d3` | `.reqntid 512,1,1`, ordinary CTA |
| split combine | same split PASS | `ce81938a9ed60c37576f4384a5ec28e86c93772d3b8987f52ac7a4a636763a70` | `.reqntid 128,1,1`, ordinary CTA |
| static clean MLIR | same static PASS | `0ddeda5829230f9b85cc82e2e0f794a844c2ebe2a22b350a0d57c8c2083fe1de` | resolves dynamic storage fields and source mappings |

The head map was exported separately, as three static/fixed producers differing
only in `qhead_per_kvhead`, so the map's instruction selection is read from a
diff rather than inferred. Same manifest.

| export | `H_q`/`H_kv` | artifact SHA256 | emitted at `.loc 1 1186 26` |
| --- | --- | --- | --- |
| grouped `r=1` | 8 / 8 | `f93b3903d6f4f938cfdf0003b4d5f822b083f4c7eabed496f616b6d3db9585b3` | nothing; the expression folds away |
| grouped `r=3` | 6 / 2 | `8f346ea25b5f3e0e71b36312af88ccf809b64bc1f78a6ff111d0d88f873431d9` | `cvt.u16.u32` / `mul.hi.u16 %rs, %rs, -21845` / `shr.u16 %rs, %rs, 1` / `cvt.u32.u16` |
| grouped `r=4` | 8 / 2 | `fb46dfb64585e7240c6857d0a59d156a9f5c170f70dc93eeb9a82078fdc72b1c` | `shr.u32 %r, %r, 2` |

Those three share the static, fixed-count specialization, where `head_idx` is the
raw `%ctaid.y`. Two further exports cover the scheduling modes that obtain the
head index differently, and they select a different sequence for the same ratio:

| export | `H_q`/`H_kv` | scheduling | artifact SHA256 | emitted at `.loc 1 1186 26` |
| --- | --- | --- | --- | --- |
| grouped `r=4` CLC, variable counts | 8 / 2 | CLC persistent | `973f7d3653034137…` | ten-instruction signed floor division: `shr.s32 ,31` / `shr.u32 ,30` / `add.s32` / `shr.s32 ,2` / `and.b32 ,-4` / `setp.ne.b32` / `setp.lt.s32` / `and.pred` / `selp.b32 ,-1,0` / `add.s32` |
| grouped `r=8` MQA, split-KV 2 | 8 / 1 | static, split | `93f3642c14259182…` | the same ten-instruction sequence with `,29` / `,3` / `,-8` |

The difference is the operand's provable sign, not the ratio: `%ctaid.y` is
non-negative, while the split scheduler's `divmod` result and the CLC response
field are `Int32` of unproven sign, which forces the negative-operand
correction. Ratio 1 folds away in every mode, static, CLC and split alike.

With register names normalized, that `.loc 1 1186` hunk is the *only* semantic
difference between the three **static** exports; everything else is register
renumbering from the one extra live value. Each grouped export was also checked against a
head-replication oracle — the stock wrapper on the same Q with K/V
`repeat_interleave`d to `H_q` heads — matching at `atol=0, rtol=0` on O, on LSE,
and on the LSE finiteness mask.

The static producer contains 64 `tcgen05.mma.ws.cta_group::1.kind::f16`,
32 `tcgen05.ld.sync.aligned.32x32b.x32.b32`, 16
`tcgen05.st.sync.aligned.32x32b.x16.b32`, eight x32 TMEM stores, 15
`tcgen05.commit...mbarrier::arrive`, 40 tensor bulk copies, and 518
`ex2.approx.ftz.f32` sites in its statically replicated bodies. Counts describe
the exported anchor, not dynamic work. The CLC export adds 26
`clusterlaunchcontrol` sites while retaining the same MMA/TMEM families. The
combine export confirms scalar LSE `cp.async.ca`, 16-byte O
`cp.async.cg`, `shfl.sync.bfly`, packed f32x2 multiply/add, BF16x2 conversion,
and vector global stores.

## Pipeline at a glance

Producer: 16 warps. Static and split specializations use `SingleTileScheduler`:
each CTA owns exactly its launch coordinate `(q_block, q_head, batch, split)`,
executes it once, and then becomes invalid. Only the CLC specialization is
persistent; every role consumes the same CLC response sequence and communicates
only through the named pipelines.

| warps | role-local program | publication / reuse edges |
| --- | --- | --- |
| 0..3 | stage-0 score consumer: load 128 FP32 S values per row from TMEM, apply two 64-token masks, update online max/sum, convert P to BF16 and store it back to the aliased TMEM stage | consumes `SPO.full[0]`; early first 32 P columns release `SPO.empty[0]`; last P columns commit `P_LAST.full[0]`; named barriers publish stats readiness while `SMSTAT.empty[0]` alone protects stats-storage reuse |
| 4..7 | identical stage-1 score consumer for the alternating S/P region | same edges for stage 1 |
| 8..11 | correction: consume softmax rescale factors, rescale previous TMEM O stages, wait for final O, combine the two stage statistics, exchange/reduce warp pairs in aliased K/V SMEM, form BF16 O (or FP32 split partial) and LSE | consumes named softmax-stat arrivals and `O_ACC.full`; releases `SMSTAT.empty`/`SPO.empty`; produces `O_EPI.full` |
| 12 | sole MMA issuer and TMEM allocator: QK into alternating S stages, then split-arrival PV into alternating O stages | consumes Q/K/V TMA-full, `SPO.empty`, `P_LAST.full`; releases Q/K/V stages; produces `SPO.full` and `O_ACC.full` |
| 13 | epilogue: waits for corrected shared O and TMA-stores one 64x128 output region | consumes `O_EPI.full`, bulk-group wait releases shared output |
| 14 | sole TMA producer: load Q once, then reverse sparse K/V stream into the three-stage union | consumes Q/KV empty barriers; TMA transaction completion produces their full barriers |
| 15 | static/split: register-decreased idle warp; CLC: acquire the empty CLC slot and issue `try_cancel` into its full mbarrier | all 512 threads consume each CLC response and release the empty slot |

The load order for `N = padded_sparse_blocks/4` grouped iterations is exactly
`Q, K[N-1], K[N-2], {V[N-1-i], K[N-3-i]} for i=0..N-3, V[1], V[0]`.
The MMA order is `S0, S1`, then alternating `PV(stage), QK(stage)` pairs, then
the two final `PV` operations. Sparse block count is rounded to a multiple of
eight before division by four, so `N` is even and both stage epilogues exist.

The correction warps initially release both O stages because no old O exists.
For later score groups they use the online-softmax rescale supplied by the
matching score warpgroup before letting MMA overwrite/accumulate that stage.
The first 32 columns of each P stage are deliberately published separately
from the last 96; this is load-bearing overlap, not an optional optimization.

## Primitive vocabulary

Structural declarations carry no computation and no layout object:

```python
specialize(...)                    # static Python/JIT branch values
launch(...)                        # grid, block, cluster-presence and SMEM bytes
gmem_region(name, dtype, extents, strides)
smem_bytes(name, base, bytes, alignment)
tmem_columns(name, first, count, dtype)
rmem_words(name, count, dtype)
tensormap(name, rank, dtype, box, strides, swizzle, oob_fill)
byte_ptr(linear_base, scalar_byte_offset)
pipeline_state(stages, index, phase, count)
work_cursor(q_blocks, heads, batch, splits, persistent_limit)  # heads is H_q
```

Every physical SMEM mapping is a scalar function. `swizzle343(x)` means the
source/export's `S<3,4,3>` bit permutation,
`x ^ (((x >> 7) & 7) << 4)`, applied to the source element address before
scaling by two BF16 bytes. The logical tuple decomposition below is explicit so
the implementation never reconstructs a layout object.

Directional movement:

```python
copy_g2s_tma(tmap, coords, dst_byte, mbarrier)
copy_s2g_tma(tmap, coords, src_byte)
copy_g2s_async(src_ptr, dst_byte, bytes, valid_bytes)
copy_t2r(src_tmem_col, dst_regs, words)
copy_r2t(src_regs, dst_tmem_col, words)
copy_t2s(src_tmem_col, dst_byte, words)
copy_r2s(src_regs, dst_byte, words)
copy_s2r(src_byte, dst_regs, words)
load_global(src_ptr, dtype, lanes)
store_global(dst_ptr, values, dtype, lanes)
load_shared(src_byte, dtype, lanes)
store_shared(dst_byte, values, dtype, lanes)
```

Basic computation remains decomposed:

```python
fill(dst, value)
maximum(dst, lhs, rhs)
add(dst, lhs, rhs, lanes=1)
mul(dst, lhs, rhs, lanes=1)
fma(dst, lhs, rhs, acc, lanes=1)
exp2(dst, src)
log2(dst, src)
rcp(dst, src)
cast(dst, src, rounding="rn")
gemm(dst_tmem, a_smem_or_tmem, b_smem, accumulate, descriptor)
reduce_max_warp(dst, src, width)
reduce_sum_warp(dst, src, width)
```

Synchronization remains visible:

```python
elect_one(); arrive(barrier); arrive_expect_tx(barrier, bytes)
wait(barrier, phase); release(barrier); commit_tmem(barrier)
fence_mbarrier_init_cluster(); fence_async_shared_cta()
fence_after_thread_sync(); fence_tmem_store(); sync_warp(); sync_cta()
cp_async_commit_group(); cp_async_wait_group(pending)
named_barrier_arrive(id, threads); named_barrier_arrive_wait(id, threads)
alloc_tmem(columns); free_tmem(columns)
cluster_try_cancel(full_mbarrier, response); cluster_query_wait()
```

## Complete producer sketch

```python
# ---------------------------------------------------------------------------
# Fixed specialization, ABI, and launch
# source main:53-188,226-677; export PTX:14-43
# ---------------------------------------------------------------------------
P = specialize(
    B, H, HKV, SQ, SKV,
    has_block_sizes, has_variable_counts, allow_empty,
    kv_splits, use_clc, use_i64_kv_strides, output_dtype,
)
# `H` is always the Q head count. `HKV` divides it; `R = H // HKV` is the
# non-packed head map's ratio and is constexpr, so `h // R` resolves at trace
# time to nothing (R=1), a shift (R power of two), or a reciprocal multiply.
R = H // HKV
M = 64
N = 256
D = DV = 128
SPARSE = 64
WARPS = 16
THREADS = 512
Q_STAGES = 1
KV_STAGES = 3
SP_STAGES = 2
SPLIT_P_COLS = 32

ABI = (
    Q_bf16, K_bf16, V_bf16, O_bf16_or_f32, LSE_f32,
    map_Q, map_K, map_V, map_O,
    softmax_scale_log2, softmax_scale,
    scheduler_params,
    block_index_i32, optional_block_sizes_i32,
    uniform_block_count_i32, optional_block_nums_i32,
    optional_split_offsets_i32,
)

if P.use_clc:
    cta = cta_id_in_cluster([1, 1, 1])
    # instruction_selection: runtime cluster-dimension launch attribute plus
    #   `clusterlaunchcontrol.*`; extent: explicit singleton cluster launch
else:
    cta = cta_id()
    # instruction_selection: ordinary `%ctaid.*` launch metadata with no
    #   cluster launch attribute; extent: one ordinary producer launch

launch(
    grid=((ceil_div(SQ,64), H, B) if P.use_clc
          else (ceil_div(SQ,64), H*kv_splits, B)),   # H is H_q for every ratio
    block=(512, 1, 1),
    arch="sm_100a",
    min_blocks_per_sm=1,
    dynamic_smem_bytes=217088,
)
# instruction_selection: `.reqntid 512,1,1`, `.minnctapersm 1`,
#   `.extern .shared .align 1024`; extent: one producer specialization
# Register reallocation is source-exact: softmax warps 0..7 request 184,
# correction warps 8..11 request 88, and MMA/load/epilogue/idle or CLC warps
# 12..15 decrease to 48 via `setmaxnreg`.

# ---------------------------------------------------------------------------
# One-dimensional SMEM and explicit physical mappings
# generated MLIR fields at writer export lines 188-215
# ---------------------------------------------------------------------------
SMEM = smem_bytes("producer", base=0, bytes=217088, alignment=1024)

Q_PIPE       = smem_bytes("q_mbars",       0,   16, 8)  # full@0, empty@8
KV_PIPE      = smem_bytes("kv_mbars",      16,  48, 8)  # full@16..32, empty@40..56
SPO_PIPE     = smem_bytes("spo_mbars",     64,  32, 8)  # full@64,72; empty@80,88
P_LAST_PIPE  = smem_bytes("plast_mbars",   96,  32, 8)  # full@96,104; empty@112,120
O_ACC_PIPE   = smem_bytes("oacc_mbars",    128, 32, 8)  # full@128,136; empty@144,152
SMSTAT_PIPE  = smem_bytes("smstat_mbars",  160, 32, 8)  # full@160,168; empty@176,184
O_EPI_PIPE   = smem_bytes("oepi_mbars",    192, 32, 8)  # full@192,200; empty@208,216
TMEM_HOLD    = smem_bytes("tmem_pointer",  224, 4,  4)
REDUCE_MBAR  = smem_bytes("reduce_mbars",  232, 16, 8)
SCALE_STATS  = smem_bytes("row_stats",     248, 2048, 8)
PAIR_STATS   = smem_bytes("pair_stats",    2296,1024, 8)

# CLC only: 16 bytes of mbarrier storage at 3320 and a 16-byte-aligned,
# 16-byte response at 3344. Static/split reserve zero bytes. Alignment leaves
# the following large regions unchanged in every mode.
CLC_MBAR     = smem_bytes("clc_mbars",      3320, 16 if P.use_clc else 0, 8)
CLC_RESPONSE = smem_bytes("clc_response",  3344, 16 if P.use_clc else 0, 16)
Q_SMEM       = smem_bytes("Q",              4096, 16384, 1024)
KV_UNION     = smem_bytes("K_V_exchange_O", 20480, 196608, 1024)

# Exact source/export address functions, returning BF16 element offsets
# relative to the named linear region.
def q_elem(m0, m1, d0, d1):
    return swizzle343(64*m0 + m1 + 16*d0 + 4096*d1)

def k_elem(stage, n0, n1, d0, d1):
    return swizzle343(64*n0 + n1 + 16*d0 + 16384*d1 + 32768*stage)

def v_elem(stage, n0, sparse4, d16, d4, dhalf):
    return swizzle343(n0 + 4096*sparse4 + 64*d16
                      + 1024*d4 + 16384*dhalf + 32768*stage)

# Per-sparse-block TMA destinations inside one K/V stage.
K_SLOT = (0, 2, 1, 3)
def k_tma_elem(stage, sub, token64, dim64, half):
    return 32768*stage + 4096*K_SLOT[sub] + swizzle343(64*token64 + dim64 + 16384*half)

def v_tma_elem(stage, sub, dim64, token64, half):
    return 32768*stage + 8192*sub + swizzle343(dim64 + 64*token64 + 4096*half)

# The correction exchange aliases the beginning of KV_UNION as FP32.
def exchange_f32(corr_warp, lane, chunk, word):
    return corr_warp*(4*32*32) + chunk*(32*32) + lane*4 + word

# Corrected O aliases KV_UNION after 32768 BF16 elements = 65536 bytes, so its
# absolute SMEM base is 20480+65536=86016. Both output dtypes use that byte base
# but have distinct source-IR layouts and TensorMap boxes.
# BF16: S<3,4,3> o ((8,8),(64,2),(1,2)):((64,512),(1,4096),(0,8192)).
def o_bf16_elem(r8, r_outer8, c64, c_half, stage):
    return swizzle343(64*r8 + 512*r_outer8 + c64 + 4096*c_half + 8192*stage)

# FP32 split partial: S<3,4,3> o
# ((8,8),(32,4),(1,2)):((32,256),(1,2048),(0,8192)).
def o_f32_elem(r8, r_outer8, c32, c_quarter, stage):
    return swizzle343(32*r8 + 256*r_outer8 + c32
                      + 2048*c_quarter + 8192*stage)

O_SMEM_BASE = KV_UNION.base + 65536

S_TMEM = (0, 128)       # FP32 score columns
P_TMEM = (0, 128)       # BF16 P aliases S, source address uses 2x column scale
O_TMEM = (256, 384)     # FP32 output accumulators
TMEM_COLUMNS = 512

# SCALE_STATS planes are [stage][128 correction lanes]: sum at plane 0,
# row-max at plane 1. PAIR_STATS stores (sum,max) for four correction warps.
def scale_stat(stage, lane128, plane):
    return 248 + 4*(lane128 + stage*128 + plane*2*128)

def pair_stat(warp, lane, field):
    return 2296 + 4*(warp*64 + lane*2 + field)

# ---------------------------------------------------------------------------
# TensorMaps and raw descriptors
# source main:388-548; export clean MLIR types 17-43,112
# ---------------------------------------------------------------------------
map_Q = tensormap("Q", rank=4, dtype=bf16, box=(64,64,1,1),
                  swizzle="128B/S<3,4,3>", oob_fill="zero")
map_K = tensormap("K sparse sub-block", rank=5, dtype=bf16,
                  box=(64,64,1,1,(1,1)), swizzle="128B/S<3,4,3>",
                  stride_width=(i64 if P.use_i64_kv_strides else i32))
map_V = tensormap("V sparse sub-block", rank=5, dtype=bf16,
                  box=(64,64,2,1,(1,1)), swizzle="128B/S<3,4,3>",
                  stride_width=(i64 if P.use_i64_kv_strides else i32))
if P.output_dtype == bf16:
    map_O = tensormap("O bf16", rank=4, dtype=bf16, box=(64,64,1,1),
                      swizzle="128B/S<3,4,3>")
else:
    map_O = tensormap("O partial f32", rank=4, dtype=f32, box=(32,64,1,1),
                      swizzle="128B/S<3,4,3>")

# ---------------------------------------------------------------------------
# Pipeline construction and prologue
# source main:733-922; export PTX source locations 725-819
# ---------------------------------------------------------------------------
Q_PROD   = pipeline_state(1, index=0, phase=1, count=0)
Q_CONS   = pipeline_state(1, index=0, phase=0, count=0)
KV_PROD  = pipeline_state(3, index=0, phase=1, count=0)
KV_CONS  = pipeline_state(3, index=0, phase=0, count=0)
SPO_CONS_PHASE = [0, 0]          # MMA waits on P/old-O availability
SPO_FULL_PHASE = [0, 0]          # softmax waits on score publication
P_LAST_CONS_PHASE = [0, 0]       # MMA; toggles with the matching SPO stage
SMSTAT_PROD_PHASE = [1, 1]       # softmax acquires empty stats slots
SMSTAT_CONS_PHASE = [0, 0]       # correction releases them
OACC_CONS_PHASE = 0              # correction; toggles only for live tiles
OEPI_PROD_PHASE = 1              # correction; toggles for live and empty tiles
OEPI_CONS_PHASE = 0              # epilogue; toggles once for every tile
CLC_PROD = pipeline_state(1, index=0, phase=1, count=0)
CLC_CONS = pipeline_state(1, index=0, phase=0, count=0)

if elected_lane_of_warp(0):
    prefetch_descriptor(map_Q)
    # instruction_selection: `prefetch.tensormap`; extent: one descriptor
    prefetch_descriptor(map_K)
    # instruction_selection: `prefetch.tensormap`; extent: one descriptor
    prefetch_descriptor(map_V)
    # instruction_selection: `prefetch.tensormap`; extent: one descriptor
    prefetch_descriptor(map_O)
    # instruction_selection: `prefetch.tensormap`; extent: one descriptor

if elected_lane_of_warp(0):
    init_barrier(Q_PIPE.full[0], arrivals=1)
    init_barrier(Q_PIPE.empty[0], arrivals=1)
    # instruction_selection: `mbarrier.init.shared.b64`; extent: 1 full + 1 empty
    for stage in 0..2:
        init_barrier(KV_PIPE.full[stage], arrivals=1)
        init_barrier(KV_PIPE.empty[stage], arrivals=1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: 3 full + 3 empty
    for stage in 0..1:
        init_barrier(SPO_PIPE.full[stage], arrivals=1)
        init_barrier(SPO_PIPE.empty[stage], arrivals=256)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: 2 full + 2 empty
        init_barrier(P_LAST_PIPE.full[stage], arrivals=4)
        init_barrier(P_LAST_PIPE.empty[stage], arrivals=1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: 2 full + 2 empty
        init_barrier(O_ACC_PIPE.full[stage], arrivals=1)
        init_barrier(O_ACC_PIPE.empty[stage], arrivals=128)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: 2 full + 2 empty
        init_barrier(SMSTAT_PIPE.full[stage], arrivals=128)
        init_barrier(SMSTAT_PIPE.empty[stage], arrivals=128)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: 2 full + 2 empty
        init_barrier(O_EPI_PIPE.full[stage], arrivals=128)
        init_barrier(O_EPI_PIPE.empty[stage], arrivals=32)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: 2 full + 2 empty
if elected_lane_of_warp(15):
    init_barrier(REDUCE_MBAR + 0, arrivals=64)
    # instruction_selection: `mbarrier.init.shared.b64`; extent: scalar
    init_barrier(REDUCE_MBAR + 8, arrivals=64)
    # instruction_selection: `mbarrier.init.shared.b64`; extent: scalar
fence_mbarrier_init_cluster()
# instruction_selection: first `fence.mbarrier_init.release.cluster` after all
#   ordinary pipeline and reduction mbarriers; extent: CTA, with no sync yet

if P.use_clc:
    if elected_lane_of_warp(0):
        init_barrier(CLC_MBAR.full[0], arrivals=1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: one CLC full slot
        init_barrier(CLC_MBAR.empty[0], arrivals=512)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: one CLC empty slot
    fence_mbarrier_init_cluster()
    # instruction_selection: second `fence.mbarrier_init.release.cluster`;
    #   extent: CLC full/empty pair only
    sync_cta()
    # instruction_selection: first `bar.sync 0`; extent: CLC-create CTA sync
sync_cta()
# instruction_selection: static/split's only `bar.sync 0`, or CLC's second
#   `bar.sync 0`; extent: `pipeline_init_wait` across all 512 threads

if warp == 12:
    alloc_tmem(512)
    # instruction_selection: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`;
    #   extent: one elected MMA-warp lane
sync_named_tmem_pointer(warps=0..12)
# instruction_selection: `bar.sync`; extent: named 13-warp rendezvous

# The actual source dispatch order is CLC/empty -> load -> MMA -> epilogue ->
# softmax -> correction. The role bodies below are factored by role for review,
# but implementation branch order remains identical to the source.
#
# Static/split scheduler contract used by every role:
#   work = launch coordinate; scheduler.advance() marks it invalid. There is no
#   grid-stride or persistent cursor in these modes.
# Every role begins with `work = scheduler.initial_work_tile_info()`.
#
# CLC scheduler contract. Warp 15 overlaps production of the *next* work record
# with execution of the current one. For every current valid record it performs:
if P.use_clc and warp == 15:
    work = scheduler.initial_work_tile_info()
    while work.valid:
        wait(CLC_MBAR.empty[CLC_PROD.index], CLC_PROD.phase)
        # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
        #   extent: one producer acquire of the 512-arrival empty slot
        cluster_try_cancel(CLC_MBAR.full[CLC_PROD.index], CLC_RESPONSE)
        # instruction_selection: `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes`;
        #   extent: one 16-byte next-work query into the full barrier
        advance(CLC_PROD)

        # Warp 15 is also one of the 512 consumers, so it executes the same
        # converged response protocol as every computational role.
        sync_cta()
        # instruction_selection: `bar.sync`; extent: all 512 threads before response
        wait(CLC_MBAR.full[CLC_CONS.index], CLC_CONS.phase)
        # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
        #   extent: all consumer threads observe the completed response
        work = decode_clc_response(CLC_RESPONSE)
        # instruction_selection: `clusterlaunchcontrol.query_cancel` and scalar
        #   response decode; extent: one next `(q_block,head,batch,0)` record
        release(CLC_MBAR.empty[CLC_CONS.index])
        # instruction_selection: `mbarrier.arrive.shared.b64`; extent: all 512
        advance(CLC_CONS)

    # Producer tail is separate from response consumption.
    wait(CLC_MBAR.empty[CLC_PROD.index], CLC_PROD.phase)
    # instruction_selection: CLC producer-tail mbarrier wait; extent: the final
    #   outstanding query slot, separate from the per-record consumer step

# In every load/MMA/epilogue/softmax/correction CLC role below, each written
# `work = scheduler.advance()` expands to: `sync_cta`; wait CLC full at
# `CLC_CONS.phase`; decode the 16-byte response; every thread releases CLC empty;
# advance `CLC_CONS`. Static/split expansion only invalidates the one work item.

# ---------------------------------------------------------------------------
# Warp 14: Q and reverse sparse K/V TMA producer
# source main:1149-1300,2322-2418; exported .loc 1227-1301,2341-2409
# ---------------------------------------------------------------------------
if warp == 14:
    while work.valid:
        qb, h, b, split = work.coordinates
        # The non-packed head map. `h` stays the Q head everywhere else --
        # sparse metadata, Q, O and LSE are all Q-head-indexed -- and only the
        # K/V tiles come from the grouped KV head. Recomputed per work item, so
        # a persistent CLC producer re-derives it after every advance.
        hkv = u32(h) // R
        # instruction_selection: nothing when R==1, in every scheduling mode.
        #   For R>1 the source's emitted form depends on the provenance of `h`,
        #   not on R alone. Static non-split takes `h` straight from `%ctaid.y`,
        #   which is provably non-negative, so `.loc 1 1186` is one unsigned op:
        #   `shr.u32` for a power-of-two R, or `cvt.u16.u32` /
        #   `mul.hi.u16 ,-21845` / `shr.u16 ,1` / `cvt.u32.u16` for R==3. Under
        #   split-KV `h` comes from the scheduler's `divmod` and under CLC from
        #   the decoded response; both are Int32 of unproven sign, so the source
        #   emits full floor-division with the negative-operand correction --
        #   ten instructions for a power-of-two R, nine including
        #   `mul.hi.s32 ,1431655766` for R==3.
        #   The port narrows `h` to unsigned first, which is a deliberate and
        #   benign divergence: `h` is a grid/scheduler head index and is always
        #   non-negative, so the two forms are equal, and the port keeps the
        #   one-instruction form in all three scheduling modes rather than
        #   reproducing the source's ten-instruction signed sequence.
        #   extent: one scalar integer expression per work item in the load
        #   role, on no inner loop, in every case
        raw_count, split_start = sparse_count_and_offset(work)
        process = (raw_count > 0) if P.allow_empty else True
        NITER = round_up(raw_count, 8) // 4

        # Each requested index is clamped to the last live entry so padded
        # groups are safe; the block-size mask later zeros phantom columns.
        def sparse_id(i):
            return block_index[b,h,qb,split_start + min(i, raw_count-1)]

        if process:
            wait(Q_PIPE.empty[0], Q_PROD.phase)
            # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
            #   extent: one stage wait
            arrive_expect_tx(Q_PIPE.full[0], 64*128*2)
            # instruction_selection: `mbarrier.arrive.expect_tx.shared.b64`;
            #   extent: elected TMA lane
            for d_half in 0..1:
                copy_g2s_tma(map_Q, (d_half*64, qb*64, h, b),
                             Q_SMEM + 2*q_elem(..., d_half), Q_PIPE.full[0])
                # instruction_selection: `cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
                #   extent: exactly two 64x64 BF16 issues, one per D half
            advance(Q_PROD)

            for kind, reverse_group in load_sequence(NITER):
                wait(KV_PIPE.empty[KV_PROD.index], KV_PROD.phase)
                # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
                #   extent: one three-stage acquire
                arrive_expect_tx(KV_PIPE.full[KV_PROD.index], 4*64*128*2)
                # instruction_selection: `mbarrier.arrive.expect_tx.shared.b64`;
                #   extent: elected TMA lane
                for sub in 0..3:
                    sid = sparse_id(reverse_group*4 + sub)
                    if kind == K:
                        for d_half in 0..1:
                            copy_g2s_tma(map_K, (d_half, sid, hkv, b),
                                         KV_UNION + 2*k_tma_elem(
                                             KV_PROD.index, sub, ..., d_half),
                                         KV_PIPE.full[KV_PROD.index])
                            # instruction_selection: `cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
                            #   extent: exactly two 64x64 BF16 issues per K
                            #   sparse sub-block, eight issues for the K group
                    else:
                        copy_g2s_tma(map_V, (sid, hkv, b),
                                     KV_UNION + 2*v_tma_elem(KV_PROD.index, sub, ...),
                                     KV_PIPE.full[KV_PROD.index])
                        # instruction_selection: same 5-D tensor-bulk family;
                        #   extent: exactly one 64x128 BF16 issue per V sparse
                        #   sub-block, four issues for the V group
                advance(KV_PROD)

        scheduler.prefetch_next()
        work = scheduler.advance()

    wait(KV_PIPE.empty[KV_PROD.index], KV_PROD.phase)
    # instruction_selection: `mbarrier.try_wait.parity.shared.b64`; extent: producer tail
    wait(Q_PIPE.empty[0], Q_PROD.phase)
    # instruction_selection: same mbarrier wait family; extent: Q producer tail

# ---------------------------------------------------------------------------
# Warp 12: sole WS QK/PV issuer
# source main:1301-1580; exported .loc 1385-1519 and helper-attributed sites
# ---------------------------------------------------------------------------
if warp == 12:
    while work.valid:
        raw_count = sparse_count(work)
        process = (raw_count > 0) if P.allow_empty else True
        NITER = round_up(raw_count, 8) // 4

        if process:
            wait(Q_PIPE.full[0], Q_CONS.phase)
            # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
            #   extent: one Q wait
            fence_after_thread_sync()
            # instruction_selection: `tcgen05.fence::after_thread_sync`;
            #   extent: elected MMA warp after the Q full wait
            advance(Q_CONS)

            for stage in (0, 1):
                wait(KV_PIPE.full[KV_CONS.index], KV_CONS.phase)
                # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
                #   extent: one K stage
                fence_after_thread_sync()
                # instruction_selection: `tcgen05.fence::after_thread_sync`;
                #   extent: elected MMA warp after each K full wait
                for k_phase in 0..7:
                    gemm(S_TMEM[stage], q_desc_at(Q_SMEM,k_phase),
                         k_desc_at(KV_UNION,KV_CONS.index,k_phase),
                         accumulate=(k_phase != 0),
                         descriptor=QK_ISSUE_64x256x16_WS_SS)
                    # instruction_selection: `tcgen05.mma.ws.cta_group::1.kind::f16`;
                    #   extent: exactly eight issues; k_phase 0 carries
                    #   accumulate=0 and k_phase 1..7 carry accumulate=1
                commit_tmem(SPO_PIPE.full[stage])
                # instruction_selection: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
                #   extent: one score stage
                release(KV_PIPE.empty[KV_CONS.index])
                # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one stage
                advance(KV_CONS)

            initialized = [False, False]
            for pair in 0..((NITER-2)//2 - 1):
                for stage in (0, 1):
                    wait(SPO_PIPE.empty[stage], SPO_CONS_PHASE[stage])
                    # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
                    #   extent: one P/old-O reuse wait
                    wait(KV_PIPE.full[KV_CONS.index], KV_CONS.phase)
                    # instruction_selection: same mbarrier wait family; extent: one V stage
                    fence_after_thread_sync()
                    # instruction_selection: `tcgen05.fence::after_thread_sync`;
                    #   extent: elected MMA warp after each V full wait
                    for issue in 0..1:
                        gemm(O_TMEM[stage], P_TMEM[stage] + issue*PV_A_STEP,
                             KV_UNION + 2*v_elem(KV_CONS.index, issue,...),
                             accumulate=(initialized[stage] or issue > 0),
                             descriptor=PV_ISSUE_64x32x128_WS_TS)
                        # instruction_selection: `tcgen05.mma.ws.cta_group::1.kind::f16`;
                        #   extent: exactly two issues using the early 32 P columns
                    wait(P_LAST_PIPE.full[stage], P_LAST_CONS_PHASE[stage])
                    # instruction_selection: `mbarrier.try_wait.parity.shared::cta.b64`;
                    #   extent: the split arrival, after issue 2 and before issue 3
                    for issue in 2..7:
                        gemm(O_TMEM[stage], P_TMEM[stage] + issue*PV_A_STEP,
                             KV_UNION + 2*v_elem(KV_CONS.index, issue,...),
                             accumulate=True, descriptor=PV_ISSUE_64x32x128_WS_TS)
                        # instruction_selection: same WS f16 MMA family;
                        #   extent: exactly six remaining issues
                    release(KV_PIPE.empty[KV_CONS.index])
                    # instruction_selection: `mbarrier.arrive.shared.b64`; extent: V stage
                    advance(KV_CONS)

                    wait(KV_PIPE.full[KV_CONS.index], KV_CONS.phase)
                    # instruction_selection: mbarrier wait family; extent: next K stage
                    fence_after_thread_sync()
                    # instruction_selection: `tcgen05.fence::after_thread_sync`;
                    #   extent: elected MMA warp after the next K full wait
                    for k_phase in 0..7:
                        gemm(S_TMEM[stage], q_desc_at(Q_SMEM,k_phase),
                             k_desc_at(KV_UNION,KV_CONS.index,k_phase),
                             accumulate=(k_phase != 0),
                             descriptor=QK_ISSUE_64x256x16_WS_SS)
                        # instruction_selection: WS f16 MMA family; extent:
                        #   exactly eight issues, first zero-init and seven accumulate
                    commit_tmem(SPO_PIPE.full[stage])
                    # instruction_selection: tcgen05 commit-to-mbarrier; extent: score stage
                    release(KV_PIPE.empty[KV_CONS.index])
                    # instruction_selection: mbarrier arrive; extent: K stage
                    advance(KV_CONS)
                    SPO_CONS_PHASE[stage] ^= 1
                    P_LAST_CONS_PHASE[stage] ^= 1
                    initialized[stage] = True

            release(Q_PIPE.empty[0])
            # instruction_selection: `mbarrier.arrive.shared.b64`; extent: Q stage

            for stage in (0, 1):
                wait(SPO_PIPE.empty[stage], SPO_CONS_PHASE[stage])
                # instruction_selection: mbarrier wait family; extent: final P stage
                wait(KV_PIPE.full[KV_CONS.index], KV_CONS.phase)
                # instruction_selection: mbarrier wait family; extent: final V stage
                fence_after_thread_sync()
                # instruction_selection: `tcgen05.fence::after_thread_sync`;
                #   extent: elected MMA warp after the final V full wait
                for issue in 0..1:
                    gemm(O_TMEM[stage], P_TMEM[stage] + issue*PV_A_STEP,
                         KV_UNION + 2*v_elem(KV_CONS.index, issue,...),
                         accumulate=(initialized[stage] or issue > 0),
                         descriptor=PV_ISSUE_64x32x128_WS_TS)
                    # instruction_selection: WS f16 MMA family; extent: two early issues
                wait(P_LAST_PIPE.full[stage], P_LAST_CONS_PHASE[stage])
                # instruction_selection: CTA-qualified mbarrier parity wait;
                #   extent: after issue 2 and before issue 3
                for issue in 2..7:
                    gemm(O_TMEM[stage], P_TMEM[stage] + issue*PV_A_STEP,
                         KV_UNION + 2*v_elem(KV_CONS.index, issue,...),
                         accumulate=True, descriptor=PV_ISSUE_64x32x128_WS_TS)
                    # instruction_selection: WS f16 MMA family; extent: six late issues
                commit_tmem(O_ACC_PIPE.full[stage])
                # instruction_selection: tcgen05 commit-to-mbarrier; extent: O stage
                release(KV_PIPE.empty[KV_CONS.index])
                # instruction_selection: mbarrier arrive; extent: final V stage
                advance(KV_CONS)
                SPO_CONS_PHASE[stage] ^= 1
                P_LAST_CONS_PHASE[stage] ^= 1

        work = scheduler.advance()

# ---------------------------------------------------------------------------
# Warp 13: corrected O epilogue (source dispatches this before softmax/correction)
# source main:2218-2311; exported .loc 2291-2296
# ---------------------------------------------------------------------------
if warp == 13:
    epi_consumer_phase = OEPI_CONS_PHASE
    while work.valid:
        wait(O_EPI_PIPE.full[0], epi_consumer_phase)
        # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
        #   extent: one corrected shared O region
        issue_count = 2 if P.output_dtype == bf16 else 4
        for issue in 0..issue_count-1:
            output_row = issue*(64 if P.output_dtype == bf16 else 32)
            copy_s2g_tma(map_O, output_coordinates(work, output_row),
                         O_SMEM_BASE + issue*8192)
            # instruction_selection: `cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint`;
            #   extent: exactly two issues for BF16 final O or four issues for
            #   FP32 split partial O, all sourced from absolute SMEM base 86016
        cp_async_commit_group()
        # instruction_selection: `cp.async.bulk.commit_group`; extent: one O group
        cp_async_wait_group(0)
        # instruction_selection: `cp.async.bulk.wait_group.read 0`; extent: O group
        release(O_EPI_PIPE.empty[0])
        # instruction_selection: `mbarrier.arrive.shared.b64`; extent: 32 epilogue lanes
        epi_consumer_phase ^= 1
        work = scheduler.advance()
    OEPI_CONS_PHASE = epi_consumer_phase

# ---------------------------------------------------------------------------
# Warps 0..7: two four-warp online row-statistics groups
# source main:1581-1867; exported .loc 1665-1854
# ---------------------------------------------------------------------------
if warp in 0..7:
    stage = 0 if warp < 4 else 1
    local_warp = warp & 3
    lane128 = local_warp*32 + lane
    s_phase = SPO_FULL_PHASE[stage]
    stat_empty_phase = SMSTAT_PROD_PHASE[stage]

    while work.valid:
        raw_count = sparse_count(work)
        has_work = (raw_count > 0) if P.allow_empty else True
        NITER = round_up(raw_count, 8) // 4
        WG_ITERS = NITER // 2
        row_max = -inf
        row_sum = 0.0

        wait(SMSTAT_PIPE.empty[stage], stat_empty_phase)
        # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
        #   extent: one stats slot acquire
        stat_empty_phase ^= 1

        if has_work:
            for it in 0..WG_ITERS-1:
                wait(SPO_PIPE.full[stage], s_phase)
                # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
                #   extent: one score-stage wait
                copy_t2r(S_TMEM[stage] + 0, score[0:32], 32)
                # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x32.b32`;
                #   extent: one 32-column score fragment
                copy_t2r(S_TMEM[stage] + 32, score[32:64], 32)
                # instruction_selection: same TMEM-load family; extent: one fragment
                copy_t2r(S_TMEM[stage] + 64, score[64:96], 32)
                # instruction_selection: same TMEM-load family; extent: one fragment
                copy_t2r(S_TMEM[stage] + 96, score[96:128], 32)
                # instruction_selection: same TMEM-load family; extent: one fragment

                bs_lo, bs_hi = block_sizes_for(it, stage, local_warp//2)
                for j in 0..63:
                    score[j] = score[j] if j < bs_lo else -inf
                    # instruction_selection: `shr.u32`, full/zero `setp` plus
                    #   `bra.uni`, then per-bit `and.b32`, `setp.eq.u32`, and
                    #   predicated `mov.f32`; extent: two 32-value branches
                    score[64+j] = score[64+j] if j < bs_hi else -inf
                    # instruction_selection: same branch/and/setp/predicated-move
                    #   family; extent: the second 64-token sparse block

                tile_max = reduce_max_register_local(score[0:128])
                # instruction_selection: unrolled register-local `max.ftz.f32`;
                #   extent: 128 values owned by this thread, no warp shuffle
                if it == 0:
                    new_max = tile_max
                    row_max_safe = new_max if new_max != -inf else 0.0
                    old_scale = 0.0
                    # instruction_selection: `setp`/branch/moves; extent: first group
                else:
                    new_max = maximum(row_max, tile_max)
                    row_max_safe = new_max if new_max != -inf else 0.0
                    delta = row_max-row_max_safe
                    # instruction_selection: scalar `sub.f32`; extent: one row
                    delta_scaled = delta*softmax_scale_log2
                    # instruction_selection: scalar `mul.f32`; extent: one row
                    old_scale = exp2(delta_scaled)
                    # instruction_selection: `ex2.approx.ftz.f32`; extent:
                    #   scalar row rescale after the separate subtract/multiply
                    if delta_scaled >= -8.0:
                        new_max = row_max
                        row_max_safe = row_max
                        old_scale = 1.0
                        # instruction_selection: `setp.ge.f32` plus branch/moves;
                        #   extent: exact BF16 `rescale_threshold=8` path
                if it > 0:
                    store_shared(scale_stat(stage,lane128,0), old_scale)
                    # instruction_selection: `st.shared.b32`; extent: scalar
                named_barrier_arrive(stage*4 + local_warp, threads=64)
                # instruction_selection: `bar.arrive`; extent: softmax/correction warp pair

                negative_scale_log2 = -softmax_scale_log2
                # instruction_selection: scalar `neg.f32`; extent: one row
                negative_rowmax_scaled = row_max_safe*negative_scale_log2
                # instruction_selection: scalar `mul.f32`; extent: one row,
                #   prepared separately before the packed score FMAs
                for pair in 0..63:
                    score_scaled[2*pair:2*pair+2] = fma(
                        score[2*pair:2*pair+2],
                        (softmax_scale_log2, softmax_scale_log2),
                        (negative_rowmax_scaled, negative_rowmax_scaled), lanes=2)
                    # instruction_selection: `fma.rn.f32x2`; extent: 64 packed
                    #   scale-and-subtract pairs before any exponentiation
                    prob_f32[2*pair] = exp2(score_scaled[2*pair])
                    prob_f32[2*pair+1] = exp2(score_scaled[2*pair+1])
                    # instruction_selection: `ex2.approx.ftz.f32`;
                    #   extent: two values per explicit pair
                row_sum = row_sum*old_scale + reduce_sum(prob_f32)
                # instruction_selection: one scalar `mul.f32` for old
                #   `row_sum*old_scale`, an explicit packed `add.rn.f32x2`
                #   probability-reduction tree, then one scalar add;
                #   extent: one complete 128-value row

                for chunk in 0..3:
                    cast(p_bf16x2[chunk], prob_f32[chunk*32:(chunk+1)*32], rounding="rn")
                    # instruction_selection: `cvt.rn.satfinite.bf16x2.f32`;
                    #   extent: explicit 16-pair loop
                    copy_r2t(p_bf16x2[chunk], P_TMEM[stage] + chunk*16, 16)
                    # instruction_selection: `tcgen05.st.sync.aligned.32x32b.x16.b32`;
                    #   extent: one 32-column BF16 P fragment
                    if chunk == 0:
                        fence_tmem_store()
                        # instruction_selection: `tcgen05.wait::st.sync.aligned`;
                        #   extent: first 32 columns
                        release(SPO_PIPE.empty[stage])
                        # instruction_selection: `mbarrier.arrive.shared.b64`;
                        #   extent: early P publication

                fence_tmem_store()
                # instruction_selection: `tcgen05.wait::st.sync.aligned`; extent: P stage
                sync_warp()
                # instruction_selection: `bar.warp.sync`; extent: one softmax warp
                if elected_lane:
                    arrive(P_LAST_PIPE.full[stage])
                    # instruction_selection: `mbarrier.arrive.shared.b64`;
                    #   extent: last-96-column publication
                wait(SMSTAT_PIPE.empty[stage], stat_empty_phase)
                # instruction_selection: mbarrier parity wait; extent: correction reuse
                stat_empty_phase ^= 1
                row_max = new_max
                s_phase ^= 1

            store_shared(scale_stat(stage,lane128,0), row_sum)
            # instruction_selection: `st.shared.b32`; extent: final row sum
            store_shared(scale_stat(stage,lane128,1), row_max)
            # instruction_selection: `st.shared.b32`; extent: final row max
            named_barrier_arrive(stage*4 + local_warp, threads=64)
            # instruction_selection: `bar.arrive`; extent: final stats publication
        else:
            named_barrier_arrive(stage*4 + local_warp, threads=64)
            # instruction_selection: `bar.arrive`; extent: synthetic empty publication

        work = scheduler.advance()

    SPO_FULL_PHASE[stage] = s_phase
    SMSTAT_PROD_PHASE[stage] = stat_empty_phase

    wait(SMSTAT_PIPE.empty[stage], stat_empty_phase)
    # instruction_selection: mbarrier parity wait; extent: stats producer tail
    arrive(TMEM_ALLOC_NAMED)
    # instruction_selection: `bar.arrive`; extent: TMEM lifetime handoff

# ---------------------------------------------------------------------------
# Warps 8..11: correction, warp-pair exchange, O/LSE production
# source main:1868-2217; exported .loc 1903-2215
# ---------------------------------------------------------------------------
if warp in 8..11:
    corr_warp = warp - 8
    lane128 = corr_warp*32 + lane
    sm_stats_consumer_phase = SMSTAT_CONS_PHASE[0]
    o_corr_consumer_phase = OACC_CONS_PHASE
    corr_epi_producer_phase = OEPI_PROD_PHASE
    for stage in (0, 1):
        release(SPO_PIPE.empty[stage])
        # instruction_selection: `mbarrier.arrive.shared.b64`; extent: initial O-empty token

    while work.valid:
        raw_count = sparse_count(work)
        has_work = (raw_count > 0) if P.allow_empty else True
        if has_work:
            # The first score group establishes stats but has no old O to rescale.
            named_barrier_arrive_wait(0*4+corr_warp, threads=64)
            # instruction_selection: `bar.sync`; extent: paired softmax/correction warps
            release(SMSTAT_PIPE.empty[0])
            # instruction_selection: `mbarrier.arrive.shared.b64`; extent: stage 0 stats
            named_barrier_arrive_wait(1*4+corr_warp, threads=64)
            # instruction_selection: `bar.sync`; extent: paired warps
            sm_stats_consumer_phase ^= 1

            for pair in 0..((round_up(raw_count,8)//4 - 2)//2 - 1):
                for stage in (0, 1):
                    named_barrier_arrive_wait(stage*4+corr_warp, threads=64)
                    # instruction_selection: `bar.sync`; extent: paired warps
                    scale = load_shared(scale_stat(stage,lane128,0), f32, 1)
                    # instruction_selection: `ld.shared.b32`; extent: scalar
                    if warp_vote_any(scale < 1.0):
                        copy_t2r(O_TMEM[stage], o_regs, 32)
                        # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x32.b32`;
                        #   extent: explicit four-fragment 4x32dp32 load
                        mul(o_regs, o_regs, scale, lanes=2)
                        # instruction_selection: `mul.rn.f32x2`; extent: explicit packed loop
                        copy_r2t(o_regs, O_TMEM[stage], 32)
                        # instruction_selection: `tcgen05.st.sync.aligned.32x32b.x32.b32`;
                        #   extent: explicit four-fragment store
                        fence_tmem_store()
                        # instruction_selection: `tcgen05.wait::st.sync.aligned`; extent: O stage
                    release(SPO_PIPE.empty[stage])
                    # instruction_selection: mbarrier arrive; extent: rescaled O reuse
                    release(SMSTAT_PIPE.empty[1-stage])
                    # instruction_selection: mbarrier arrive; extent: stats reuse
                sm_stats_consumer_phase ^= 1

            release(SMSTAT_PIPE.empty[1])
            # instruction_selection: mbarrier arrive; extent: final loop balance
            for stage in (0, 1):
                named_barrier_arrive_wait(stage*4+corr_warp, threads=64)
                # instruction_selection: `bar.sync`; extent: final stats
                row_sum[stage] = load_shared(scale_stat(stage,lane128,0), f32, 1)
                # instruction_selection: `ld.shared.b32`; extent: scalar
                row_max[stage] = load_shared(scale_stat(stage,lane128,1), f32, 1)
                # instruction_selection: `ld.shared.b32`; extent: scalar
                release(SMSTAT_PIPE.empty[stage])
                # instruction_selection: mbarrier arrive; extent: final stats reuse

            max_local = maximum(valid_max(row_max[0]), valid_max(row_max[1]))
            # instruction_selection: `max.ftz.f32`; extent: scalar row
            delta0 = row_max[0] - max_local
            delta1 = row_max[1] - max_local
            # instruction_selection: `sub.f32`; extent: one per live stage
            scaled_delta0 = delta0*softmax_scale_log2
            scaled_delta1 = delta1*softmax_scale_log2
            # instruction_selection: `mul.f32`; extent: one per live stage
            scale0 = exp2(scaled_delta0) if row_sum[0] > 0 else 0
            scale1 = exp2(scaled_delta1) if row_sum[1] > 0 else 0
            # instruction_selection: `ex2.approx.ftz.f32`; extent: one per stage
            sum_stage1 = row_sum[1]*scale1
            # instruction_selection: `mul.f32`; extent: scalar stage-1 term
            sum_local = fma(row_sum[0], scale0, sum_stage1)
            # instruction_selection: `fma.rn.f32`; extent: scalar stage-0 term

            for stage in (0, 1):
                wait(O_ACC_PIPE.full[stage], o_corr_consumer_phase)
                # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
                #   extent: final O stage
                fence_after_thread_sync()
                # instruction_selection: `tcgen05.fence::after_thread_sync`;
                #   extent: each correction warp after each O-accumulator wait
            wait(O_EPI_PIPE.empty[0], corr_epi_producer_phase)
            # instruction_selection: mbarrier parity wait; extent: output region acquire
        else:
            for stage in (0, 1):
                named_barrier_arrive_wait(stage*4+corr_warp, threads=64)
                # instruction_selection: `bar.sync`; extent: synthetic empty stats
                release(SMSTAT_PIPE.empty[stage])
                # instruction_selection: mbarrier arrive; extent: stats reuse
            scale0 = scale1 = sum_local = max_local = 0.0
            wait(O_EPI_PIPE.empty[0], corr_epi_producer_phase)
            # instruction_selection: mbarrier parity wait; extent: output acquire

        partner = corr_warp ^ 2
        store_shared(pair_stat(partner,lane,0), sum_local)
        # instruction_selection: `st.shared.b32`; extent: scalar
        store_shared(pair_stat(partner,lane,1), max_local)
        # instruction_selection: `st.shared.b32`; extent: scalar
        mbarrier_arrive_and_wait(REDUCE_MBAR + 8*(corr_warp&1), phase=0)
        # instruction_selection: `mbarrier.arrive.shared.b64` plus parity wait;
        #   extent: first rendezvous of one 64-thread correction warp pair
        partner_sum = load_shared(pair_stat(corr_warp,lane,0), f32, 1)
        # instruction_selection: `ld.shared.b32`; extent: scalar
        partner_max = load_shared(pair_stat(corr_warp,lane,1), f32, 1)
        # instruction_selection: `ld.shared.b32`; extent: scalar
        max_total = maximum(max_local, partner_max)
        # instruction_selection: `max.ftz.f32`; extent: scalar
        own_delta = max_local-max_total
        peer_delta = partner_max-max_total
        # instruction_selection: `sub.f32`; extent: one own + one peer scalar
        own_delta_scaled = own_delta*softmax_scale_log2
        peer_delta_scaled = peer_delta*softmax_scale_log2
        # instruction_selection: `mul.f32`; extent: one own + one peer scalar
        own_rescale = exp2(own_delta_scaled) if sum_local > 0 else 0
        peer_rescale = exp2(peer_delta_scaled) if partner_sum > 0 else 0
        # instruction_selection: `ex2.approx.ftz.f32`; extent: own + peer
        peer_sum_scaled = partner_sum*peer_rescale
        # instruction_selection: `mul.f32`; extent: scalar peer contribution
        total_sum = fma(sum_local, own_rescale, peer_sum_scaled)
        # instruction_selection: `fma.rn.f32`; extent: scalar own contribution
        inv_total = rcp(total_sum) if total_sum > 0 else 0
        # instruction_selection: `rcp.approx.ftz.f32`; extent: scalar

        if has_work and not (P.allow_empty and scale0 == 0 and scale1 == 0):
            copy_t2r(O_TMEM[0], o0_regs, 32)
            # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x32.b32`;
            #   extent: explicit 4x32dp32 fragments
            copy_t2r(O_TMEM[1], o1_regs, 32)
            # instruction_selection: same TMEM-load family; extent: explicit fragments
            own_weight = own_rescale*inv_total
            own_scale0 = scale0*own_weight
            own_scale1 = scale1*own_weight
            # instruction_selection: `mul.f32`; extent: exactly three scalar
            #   weight products (`own_weight`, `own_scale0`, `own_scale1`)
            mul(o1_scaled, o1_regs, own_scale1, lanes=2)
            fma(exchange_regs, o0_regs, own_scale0, o1_scaled, lanes=2)
            # instruction_selection: packed `mul.rn.f32x2` for O1 followed by
            #   packed `fma.rn.f32x2` for O0; extent: explicit fragment loop
            copy_r2s(exchange_regs, KV_UNION + 4*exchange_f32(corr_warp,lane,...), 32)
            # instruction_selection: `st.shared.v4.b32`; extent: explicit exchange loop
        else:
            fill(exchange_regs, 0.0)
            # instruction_selection: `mov.f32`; extent: explicit fragment loop
            copy_r2s(exchange_regs, KV_UNION + 4*exchange_f32(corr_warp,lane,...), 32)
            # instruction_selection: `st.shared.v4.b32`; extent: explicit exchange loop

        mbarrier_arrive_and_wait(REDUCE_MBAR + 8*(corr_warp&1), phase=1)
        # instruction_selection: mbarrier arrive and parity wait; extent: second
        #   rendezvous of the same warp pair; the next tile starts again at phase 0
        if corr_warp < 2:
            copy_s2r(own_exchange, own_regs, 32)
            # instruction_selection: `ld.shared.v4.b32`; extent: explicit vector loop
            copy_s2r(peer_exchange, peer_regs, 32)
            # instruction_selection: `ld.shared.v4.b32`; extent: explicit vector loop
            add(final_regs, own_regs, peer_regs, lanes=2)
            # instruction_selection: `add.rn.f32x2`; extent: explicit packed loop
            if P.output_dtype == bf16:
                cast(final_bf16x2, final_regs, rounding="rn")
                # instruction_selection: `cvt.rn.satfinite.bf16x2.f32`;
                #   extent: explicit packed loop
                copy_r2s(final_bf16x2, O_SMEM_BASE + 2*o_bf16_elem(...), 16)
                # instruction_selection: `st.shared.v4.b32`; extent: BF16 O vectors
            else:
                copy_r2s(final_regs, O_SMEM_BASE + 4*o_f32_elem(...), 32)
                # instruction_selection: `st.shared.v4.b32`; extent: FP32 split
                #   partial vectors in the same corrected shared-O region

        fence_async_shared_cta()
        # instruction_selection: `fence.proxy.async.shared::cta`; extent: correction warps
        if corr_warp < 2 and row_is_in_SQ:
            log_sum = log2(total_sum) if total_sum > 0 else -inf
            # instruction_selection: `lg2.approx.ftz.f32`; extent: scalar row
            lse_log2 = fma(max_total, softmax_scale_log2, log_sum)
            # instruction_selection: `fma.rn.f32`; extent: scalar row
            lse = lse_log2*ln2 if total_sum > 0 else -inf
            # instruction_selection: `mul.f32`; extent: final scalar ln2 conversion
            store_global(LSE_ptr(work,row), lse, f32, 1)
            # instruction_selection: `st.global.b32`; extent: scalar row

        if has_work:
            for stage in (0, 1):
                release(SPO_PIPE.empty[stage])
                # instruction_selection: mbarrier arrive; extent: O accumulator reuse
        arrive(O_EPI_PIPE.full[0])
        # instruction_selection: `mbarrier.arrive.shared.b64`; extent: all 128
        #   correction threads publish the corrected shared O
        if has_work:
            o_corr_consumer_phase ^= 1
            # extent: only live tiles consumed O_ACC full generations
        sm_stats_consumer_phase ^= 1
        corr_epi_producer_phase ^= 1
        work = scheduler.advance()

    OACC_CONS_PHASE = o_corr_consumer_phase
    SMSTAT_CONS_PHASE = [sm_stats_consumer_phase, sm_stats_consumer_phase]
    OEPI_PROD_PHASE = corr_epi_producer_phase
    wait(O_EPI_PIPE.empty[0], corr_epi_producer_phase)
    # instruction_selection: mbarrier parity wait; extent: correction producer tail
    arrive(TMEM_ALLOC_NAMED)
    # instruction_selection: `bar.arrive`; extent: TMEM lifetime handoff

# MMA warp waits for score/correction arrivals, then owns deallocation.
if warp == 12:
    relinquish_tmem_alloc_permit()
    # instruction_selection: `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`;
    #   extent: elected MMA warp before the lifetime barrier
    sync_named_tmem_pointer(warps=0..12)
    # instruction_selection: `bar.sync 2,416`; extent: 13-warp TMEM lifetime rendezvous
    free_tmem(512)
    # instruction_selection: `tcgen05.dealloc.cta_group::1.sync.aligned.b32`;
    #   extent: one elected MMA-warp lane
```

The query-tail predicate is applied only at final O/LSE stores; TMA O uses
TensorMap bounds. Block-tail masking happens before maximum/exp. An empty sparse
tile does not issue Q/K/V or MMA, does not release nonexistent `SPO` generations,
still balances the stats and output pipelines, writes O zero, and writes LSE
`-inf`. Split offsets select disjoint sparse-index ranges. With
`aligned_base=floor(floor(valid/kv_splits)/8)*8`, the source uses aligned
offsets `min(aligned_base*s + min(valid-aligned_base*kv_splits, 8*s), valid)`
when `aligned_base>0`; otherwise—including the 256-split path—it uses even
ceil-distributed offsets `ceil(valid*s/kv_splits)`. Each split producer follows
the same program and folds `split*H` into its output-head axis (`H` is H_q, so the combine is unaffected by the head map).

## Complete split-KV combine sketch

The combine exists only for `kv_splits > 1`. Its production specialization is
`tile_m=16`, `k_block_size=64`, `stages=4`, `threads=128`,
`max_splits=2**ceil(log2(kv_splits))`. The physical split extent used by the
source LSE layout is `max(8,max_splits)`, even when `max_splits==2`. It is an ordinary launch and has no TMEM,
TMA, cluster, or warp specialization; all four warps execute the same row/column
program. Source combine `22-147,250-594`; wrapper `_interface.py:885-1005`.

```python
launch(
    grid=(ceil_div(SQ*H,16), ceil_div(128,64), B),
    block=(128,1,1), arch="sm_100a",
    dynamic_smem_bytes=combine_storage_bytes(max(8,max_splits)),
)
# instruction_selection: `.reqntid 128,1,1`, ordinary CTA metadata, and
#   `.extern .shared .align 1024`; extent: one combine specialization
cta = cta_id()
# instruction_selection: ordinary `%ctaid.*`; extent: one combine CTA

LSE_SPLIT_STORAGE = max(8, max_splits)
COMB = smem_bytes("combine", 0, combine_storage_bytes(LSE_SPLIT_STORAGE), 128)
LSE_S = smem_bytes("split_lse", 0, LSE_SPLIT_STORAGE*16*4, 128)
MAX_SPLIT_S = smem_bytes("max_valid_split", align_up(end(LSE_S),128), 16*4, 128)
O_RING = smem_bytes("partial_o_ring", align_up(end(MAX_SPLIT_S),128),
                    4*16*64*4, 128)
# For max_splits=2 this is exactly LSE=[0,512), MAX_SPLIT=[512,576),
# padding=[576,640), and O_RING=[640,17024), for 17024 dynamic SMEM bytes.

# Source atom is S<4,0,4> composed with ordered (8,16), order=(1,0),
# tiled along split. Resolving it gives the scalar element XOR below. O is a
# plain stage-major 16x64 FP32 ring.
def lse_smem(split, row):
    linear = 16*split + row
    return LSE_S.base + 4*(linear ^ ((linear >> 4) & 0xF))

def o_ring(stage, row, col):
    return O_RING.base + 4*(col + 64*row + 16*64*stage)

if dynamic_num_splits_is_one:
    return

row0 = tidx // 16
row1 = row0 + 8
col0 = (tidx % 16) * 4
# extent: each of 128 threads owns exactly two rows separated by 8 and four
# adjacent FP32/BF16 output columns in each row.

for assigned_row in lse_copy_rows_for_thread(tidx, tile_m=16):
    for split in per_thread_splits(max_splits):
        if assigned_row_is_valid and split < num_splits:
            copy_g2s_async(LSE_partial[split,b,assigned_row,h],
                           lse_smem(split,assigned_row), 4, 4)
            # instruction_selection: `cp.async.ca.shared.global ...,4,4`;
            #   extent: scalar LSE
        else:
            store_shared(lse_smem(split,assigned_row), -inf, f32, 1)
            # instruction_selection: `st.shared.b32`; extent: scalar fill
cp_async_commit_group()
# instruction_selection: `cp.async.commit_group`; extent: LSE generation

for stage in 0..2:
    if stage < num_splits:
        if row0_is_valid:
            copy_g2s_async(O_partial[stage,b,row0,h,col0:col0+4],
                           o_ring(stage,row0,col0), 16, 16)
        if row1_is_valid:
            copy_g2s_async(O_partial[stage,b,row1,h,col0:col0+4],
                           o_ring(stage,row1,col0), 16, 16)
        # instruction_selection: `cp.async.cg.shared.global ...,16,16`;
        #   extent: exactly two FP32x4 issues, one per owned row
    cp_async_commit_group()
    # instruction_selection: `cp.async.commit_group`; extent: one O-ring stage

cp_async_wait_group(3)
# instruction_selection: `cp.async.wait_group 3`; extent: LSE/initial O
sync_cta()
# instruction_selection: `bar.sync 0`; extent: CTA

copy_s2r(LSE_S, lse_regs, assigned_split_row_values)
# instruction_selection: `ld.shared.b32`; extent: explicit split values
lse_max = reduce_max_warp(max(lse_regs), width=8)
# instruction_selection: `max.f32` plus `shfl.sync.bfly.b32` with 4,2,1;
#   extent: one row across split lanes
max_valid_split = reduce_max_warp(last_non_inf_split(lse_regs), width=8)
# instruction_selection: `max.s32` plus `shfl.sync.bfly.b32`;
#   extent: one row
sum_exp = 0.0
for split in assigned_splits:
    lse_log2 = lse_regs[split]*log2e
    max_log2 = safe(lse_max)*log2e
    # instruction_selection: `mul.f32`; extent: exactly two scalar products
    scale_delta = lse_log2-max_log2
    # instruction_selection: `sub.f32`; extent: scalar split
    scale[split] = exp2(scale_delta)
    # instruction_selection: `ex2.approx.ftz.f32`; extent: scalar split
    sum_exp = add(sum_exp, scale[split])
    # instruction_selection: `add.f32`; extent: scalar split
sum_exp = reduce_sum_warp(sum_exp, width=8)
# instruction_selection: `shfl.sync.bfly.b32` plus `add.f32`; extent: one row
final_lse = log2(sum_exp)*ln2 + lse_max
# instruction_selection: `lg2.approx.ftz.f32` plus `fma.rn.f32`; extent: row
inv_sum = rcp_rn(sum_exp) if finite_positive(sum_exp) else 0.0
# instruction_selection: `rcp.rn.f32`; extent: scalar, not approximate reciprocal
for split in assigned_splits:
    scale[split] = mul(scale[split], inv_sum)
    # instruction_selection: `mul.f32`; extent: scalar split
    store_shared(lse_smem(split,row), scale[split], f32, 1)
    # instruction_selection: `st.shared.b32`; extent: scalar split weight
store_shared(MAX_SPLIT_S + 4*row, max_valid_split, i32, 1)
# instruction_selection: `st.shared.b32`; extent: scalar row bound
if k_block == 0 and row_is_valid:
    store_global(final_LSE[b,row,h], final_lse, f32, 1)
    # instruction_selection: `st.global.b32`; extent: scalar row

sync_cta()
# instruction_selection: `bar.sync 0`; extent: CTA
fill(o_acc_f32[row0], 0.0)
fill(o_acc_f32[row1], 0.0)
# instruction_selection: `mov.f32`; extent: two independent FP32x4 row accumulators
load_stage = 3
compute_stage = 0
thread_max_valid_split = max(max_valid_split[row0], max_valid_split[row1])
for split in 0..thread_max_valid_split:
    weight0 = load_shared(lse_smem(split,row0), f32, 1)
    weight1 = load_shared(lse_smem(split,row1), f32, 1)
    # instruction_selection: `ld.shared.b32`; extent: one scalar per owned row
    next_split = split + 3
    if next_split <= thread_max_valid_split:
        if row0_is_valid:
            copy_g2s_async(O_partial[next_split,b,row0,h,col0:col0+4],
                           o_ring(load_stage,row0,col0), 16, 16)
        if row1_is_valid:
            copy_g2s_async(O_partial[next_split,b,row1,h,col0:col0+4],
                           o_ring(load_stage,row1,col0), 16, 16)
        # instruction_selection: `cp.async.cg.shared.global ...,16,16`;
        #   extent: exactly two FP32x4 issues, one per owned row
    cp_async_commit_group()
    # instruction_selection: `cp.async.commit_group`; extent: O-ring generation
    load_stage = (load_stage + 1) % 4
    cp_async_wait_group(3)
    # instruction_selection: `cp.async.wait_group 3`; extent: current O stage
    copy_s2r(o_ring(compute_stage,row0,col0), partial0, 4)
    copy_s2r(o_ring(compute_stage,row1,col0), partial1, 4)
    # instruction_selection: `ld.shared.v2.b64`; extent: two independent FP32x4 rows
    compute_stage = (compute_stage + 1) % 4
    if row0_is_valid and weight0 > 0:
        for pair in 0..1:
            mul(weighted0[pair], partial0[pair], weight0, lanes=2)
            # instruction_selection: `mul.f32x2`; extent: two pairs for row0
        for pair in 0..1:
            add(o_acc_f32[row0,pair], o_acc_f32[row0,pair],
                weighted0[pair], lanes=2)
            # instruction_selection: `add.f32x2`; extent: two pairs for row0,
            #   kept separate from multiply to preserve source rounding
    if row1_is_valid and weight1 > 0:
        for pair in 0..1:
            mul(weighted1[pair], partial1[pair], weight1, lanes=2)
            # instruction_selection: `mul.f32x2`; extent: two pairs for row1
        for pair in 0..1:
            add(o_acc_f32[row1,pair], o_acc_f32[row1,pair],
                weighted1[pair], lanes=2)
            # instruction_selection: `add.f32x2`; extent: two pairs for row1

cast(o0_bf16x2, o_acc_f32[row0], rounding="rn")
cast(o1_bf16x2, o_acc_f32[row1], rounding="rn")
# instruction_selection: `cvt.rn.bf16x2.f32`; extent: two pairs per owned row
if row0_is_valid:
    store_global(final_O[b,row0,h,col0:col0+4], o0_bf16x2, bf16, 4)
    # instruction_selection: `st.global.v2.b32`; extent: one BF16x4 row store
if row1_is_valid:
    store_global(final_O[b,row1,h,col0:col0+4], o1_bf16x2, bf16, 4)
    # instruction_selection: `st.global.v2.b32`; extent: one BF16x4 row store
```

Rows outside `SQ*H` are filled with `-inf` before the max-valid-split reduction
so they cannot read a stale O stage. A split with LSE `-inf` has zero weight.
When every split is empty, final LSE remains `-inf` and O remains zero. A
dynamic one-split combine request returns early, although the wrapper normally
does not launch this kernel for one split.

## Static specialization boundary

The implementation may construct separate TIRx PrimFuncs for the following
compile-time branches, but must not change the execution skeleton:

| branch | specialized differences |
| --- | --- |
| `qhead_per_kvhead` | the K/V TMA head coordinate only. `R=1` folds to the plain Q head and must stay byte-identical to the MHA program; other ratios add one scalar integer op per work item, the port narrowing the head index to unsigned so the one-instruction form is kept under CLC and split-KV too |
| `has_block_sizes` | physical-block size GMEM loads and two 64-column masks versus constant 64 |
| fixed / variable count | scalar fixed count versus `block_nums[b,h,qb]`; variable count changes scheduler arguments |
| `allow_empty` | empty-tile control path and zero/-inf output; split buckets can also be empty when the requested split count exceeds a row's live sparse blocks |
| static / CLC | ordinary versus explicit singleton-cluster launch tags, CLC storage and warp-15 body |
| `kv_splits` | scheduler split axis, split-offset metadata, output dtype/head folding, and independent combine specialization |
| i32 / i64 K/V strides | K/V TensorMap and address stride width only |
| final BF16 / partial FP32 O | correction writes the specialized dtype into the same shared-O region; only O descriptor/output ABI and issue count differ |

In CLC, the persistent work order and all pipeline phase variables survive
across response records; a role must not restart a phase because sparse counts
can change between adjacent records. Static/split executes one launch coordinate
per CTA, so its phase state has only that one record. Both acquisition modes use
the same role body after the work coordinate has been obtained.

## TIRx module, correctness, and benchmark contract

The public module exports `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`,
`get_kernel`, `prepare_data`, `prepare_bench`, `run_test`, `run_gpu`, and
`run_bench` following repository conventions. `get_kernel` returns one PrimFunc
for `kv_splits=1` and producer plus combine for `kv_splits>1`. Device code uses
`import tirx_kernels.kern as K` exclusively.

Correctness is the module's fixed 35-row matrix: BHSD/BSHD, Q tails 1/63/65,
masked/unmasked, custom scale, fixed/variable/empty counts, static/CLC,
split 1/2/3/8/256 and auto-to-2, the active i64 K/V-stride case, and the
non-packed head map at ratios 2, 3, 4 and 8 (MQA included) crossed with those
axes. Every row must run with no skip. O and LSE are compared both with the production source
and a gathered FP32 oracle; empty rows require exact O zero and LSE `-inf`.
Split rows additionally validate producer FP32 partial O/LSE before combine.

Performance truth is only `python -m tirx_kernels.bench_suite`. The frozen
23-row matrix covers short/long and batch/head scaling, static/CLC, fixed,
variable and empty counts, split 2/4/8, i64 K/V addressing, and a grouped-head
mirror of those axes at ratios 2, 3, 4 and 8. Every row must
contain five finite positive Proton samples for source and TIRx, `status == ok`,
no error, and strict `mean(source)/mean(tirx) > 0.99`.

## Instruction-selection summary

- Producer launch: 512 threads, one CTA UMMA, 217088 bytes dynamic SMEM aligned
  to 1024, minimum one CTA/SM. CLC alone carries explicit singleton cluster
  metadata and cluster-launch-control instructions.
- The head map contributes at most one scalar integer instruction per work
  item, in the load role only: nothing at ratio 1, `shr.u32` at a power-of-two
  ratio, a 16-bit reciprocal multiply otherwise. It is on no inner loop and
  changes no issue count. The source itself keeps that one-instruction form only
  where the head index is the raw `%ctaid.y`; under split-KV and CLC its head
  index is a signed `Int32` and it emits a ten-instruction floor division. The
  port narrows to unsigned and keeps one instruction in every mode -- a
  deliberate divergence, sound because a head index is never negative.
- Q uses exactly two 4-D tensor-bulk G2S issues. Each K group uses eight 5-D
  issues (two per sparse sub-block), each V group uses four (one per sparse
  sub-block), and O uses exactly two BF16 or four FP32 4-D tensor-bulk S2G
  issues from absolute shared address 86016.
- QK is SMEM/SMEM WS f16 MMA into FP32 TMEM. PV is TMEM/SMEM WS f16 MMA into
  FP32 TMEM. `tcgen05.commit` publishes score and output generations.
- Score/P uses x32 FP32 TMEM loads, BF16x2 RN conversion, and x16 BF16 TMEM
  stores. Correction uses x32 TMEM loads/stores and packed f32x2 arithmetic.
- All stage reuse uses mbarrier parity state. Named barriers pair each softmax
  warp with one correction warp. Two 64-arrival mbarriers protect correction
  warp-pair exchange.
- Combine uses scalar `cp.async.ca` for LSE, 16-byte `cp.async.cg` for partial O,
  four committed stages with wait-group 3, width-8 butterfly reductions,
  packed f32x2 weighted accumulation, BF16x2 conversion, and vector stores.

These choices are selected by explicit placement, scalar physical mappings,
descriptor fields, role ownership, and schedule. Opcode counts are evidence
from the preserved exports; no hidden computation is encoded in them.
