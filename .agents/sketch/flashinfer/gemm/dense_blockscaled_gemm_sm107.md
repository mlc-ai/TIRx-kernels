<!--
This file is a design sketch for a TIRx port of FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ 012cfdb97f217e0d48bc9352c17a74068c9e495b),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer SM107 dense block-scaled FP4 GEMM kernel sketch

This non-executable, operation-level sketch freezes the intended transcription
of `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm107.py`, the alpha-scaled
TMA epilogue in `epilogue_utils.py`, the production `mm_fp4` runner in
`gemm_base.py`, and its SM107 tactic selector in `kernels/utils.py`. The pinned
FlashInfer commit is `012cfdb97f217e0d48bc9352c17a74068c9e495b`; the primary
source SHA-256 is
`c9def937d2bf76b363bb321aa4053ffd39055febdcef3c890572bd2c4f2946ad`.
The inherited CUTLASS example is pinned to commit
`cdcf8d86daa9b417840fd99875a1b1af685d389d`, file SHA-256
`1517d4cde6b7988d5f44eca5fc2de4516b6f582ae60c37a5d1455a09c647453b`.

The writer exported and executed three independent line-info PTX anchors under
`.porting/dense_blockscaled_gemm_sm107/writer_export/` with both source caches
disabled. They cover B keep/reuse, CTA-group 2, NVFP4/MXFP4, FP16/BF16,
row/column output, N=192, explicit-zero and automatic prefetch:

| anchor | specialization | PTX SHA-256 |
| --- | --- | --- |
| `breuse_nvfp4_lineinfo` | tile 256x128, instruction 128x128, cluster 2x1, NVFP4, FP16, no swap, prefetch 0 | `88aa75ff4db0908589bd1fd54cfc3fad63cf17cdacb33e6f3ac3c803c3ab00f7` |
| `twocta_mxfp4_swap_lineinfo` | tile/instruction 256x128, cluster 2x1, MXFP4, BF16, swap, prefetch 2 | `78e563e3d4fc381c0f19d32a2178bca959b9e9756bcf9dfd0324c5aa825d61cc` |
| `n192_auto_prefetch_lineinfo` | tile/instruction 128x192, cluster 1x2, NVFP4, FP16, no swap, automatic prefetch | `d3d40f43a93970309c9d18c7d6c752d66a437a3fcf5f30db2cb94fb47cc2776e` |

Every artifact declares PTX 9.4, `.target sm_107a`, `.reqntid 192,1,1`,
1024-byte-aligned dynamic shared memory, and maps its line records directly to
the pinned source, CUTLASS parent, and `epilogue_utils.py`.

After the independent sketch reviewer returns PASS, this file is immutable.

## Transcription boundary and invariants

The executable module is one parameterized K-language port and imports the
device language only as `import tirx_kernels.kern as K`. It may use raw
`K.ptx[...]`, scalar K control flow, opaque TensorMaps, rank-one shared
allocations, local register arrays, and ordinary integer helpers. It must not
use a tile primitive, a first-class mapping/layout object, `K.cuda.func_call`,
an inline-CUDA function-call exemption, or any change under
`tirx_kernels/kern/`.

The production boundary is the SM107 branch reachable from `mm_fp4`: both
operands are packed E2M1 FP4; scales are E4M3 with vector size 16 (NVFP4) or
E8M0 with vector size 32 (MXFP4); output is FP16 or BF16; batch is one. It
includes swap, CTA-group 1/2, B keep/reuse, N tiles 64/128/192/256, legal
1/2/4-by-1/2/4 clusters, and prefetch distances 0, 2, and automatic. The
separate mixed-cluster class and direct-only FP8/mixed-input modes are out of
scope.

For public matrices A `(M,K/2)` and B `(K/2,N)`, packed nibbles decode to
E2M1 values. With source-layout scale storage `SFA` and `SFB`, the operation is

```text
C[m,n] = cast_output(
    float32(cast_output(alpha_f32)) * sum_k(
        decode_fp4(A[m,k]) * decode_scale(SFA[m,k//V]) *
        decode_fp4(B[k,n]) * decode_scale(SFB[n,k//V])
    )
)
```

where `V` is 16 or 32. Accumulation and alpha multiplication are FP32, but the
source first rounds the loaded FP32 alpha to the selected FP16/BF16 output
dtype and widens that scalar back to FP32. The widened rounded alpha is applied
before the final output conversion. A/B and their scales exchange roles under
`swap_ab`; the output descriptor becomes column-major so public storage remains
row-major. TensorMap OOB fill supplies zero for partial M/N tiles. K remains
16-byte aligned as required by the public runner and every live production
shape has K divisible by 32.

All source mappings lower before tracing to scalar extents, strides, byte
offsets, XOR swizzle arithmetic, TensorMap fields, and integer UMMA descriptor
bits. Every shared object is a byte interval in one rank-one arena. Every TMEM
reference is a scalar row/column address.

## Static specialization

```python
P = specialize_static(M, N, K, sf_mode, output_dtype, tactic)
TILE_M, TILE_N = P.mma_tiler_mn
INST_M, INST_N = P.mma_instruction_mn
assert TILE_M in (128, 256) and TILE_N in (64, 128, 192, 256)
assert INST_M in (128, 256) and INST_N == TILE_N
assert TILE_M == INST_M or TILE_M == 2*INST_M

CTA_GROUP = INST_M // 128                 # one or two
B_REUSE = TILE_M == 2*INST_M             # group-one only in production
CTA_M = TILE_M // CTA_GROUP               # 128 normally; 256 for B-reuse
CTA_N = TILE_N
K_TILE = 256
INSTRUCTION_K = 128
K_BLOCKS = 2
SF_VECTOR = 16 if sf_mode == "nvfp4" else 32
SF_DTYPE = E4M3 if sf_mode == "nvfp4" else E8M0
ACC_STAGES = 1 if B_REUSE and TILE_N in (192, 256) else 2
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
WARPS = 6
EPI_N = source_epilogue_n(CTA_GROUP, CTA_N, output_dtype, swap_ab)

A_STAGE_BYTES = source_a_stage_bytes(P)
B_STAGE_BYTES = source_b_stage_bytes(P)
SFA_STAGE_BYTES = source_sfa_stage_bytes(P)
SFB_STAGE_BYTES = source_sfb_stage_bytes(P)
C_STAGE_BYTES = source_c_stage_bytes(P)
AB_STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES + SFA_STAGE_BYTES + SFB_STAGE_BYTES
SM107_SMEM_CAPACITY = 334848             # pinned get_smem_capacity_in_bytes("sm_107")
AB_STAGES = (SM107_SMEM_CAPACITY - (1024 + 2*C_STAGE_BYTES)) // AB_STAGE_BYTES
C_STAGES = 2 + (
    SM107_SMEM_CAPACITY - AB_STAGES*AB_STAGE_BYTES - (1024 + 2*C_STAGE_BYTES)
) // C_STAGE_BYTES
PREFETCH_DISTANCE = AB_STAGES if tactic.prefetch_dist is None else tactic.prefetch_dist

NUM_ACC_TMEM_COLS = CTA_N * ACC_STAGES * (2 if B_REUSE else 1)
NUM_SFA_TMEM_COLS = SFA_STAGE_BYTES // 128
NUM_SFB_TMEM_COLS = SFB_STAGE_BYTES // 128
TMEM_ALLOC_COLS = 576                     # SM107 max allocation, always requested
assert NUM_ACC_TMEM_COLS + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS <= 576
```

The specialization rejects source-illegal alignment, cluster, and tile
combinations. The validation matrix is generated from all legal production
modes and reduced only by structural coverage tokens: scale format/vector,
output conversion, swap/output major, CTA group, B-reuse collector sequence,
N=64/128/192/256, cluster multicast directions, stage tuple, and prefetch
0/2/automatic. Generated-PTX hashing may collapse cases only when the entire
PTX is byte-identical.

## Primitive vocabulary

```python
tensor_map(base, rank, extents, byte_strides, box_extents,
           element_strides, dtype, swizzle_code, oob_zero)
smem_interval(base, byte_offset, byte_count)
tmem_address(base_column, row, column)
static_scheduler_cursor(problem_ctas, cluster_shape, grid_clusters, raster="m")
prefetch_tensormap(map)
prefetch_tma_tile(map, coordinates, rank, cache_hint)
tma_load(map, coordinates, shared_address, full_barrier,
         cta_group, optional_multicast_mask, cache_hint)
tma_store(map, coordinates, shared_address, cache_hint)
umma_scale_copy(cta_group, tmem_address, shared_descriptor)
umma_blockscaled(cta_group, collector, accumulator, a_descriptor, b_descriptor,
                 instruction_descriptor, sfa_address, sfb_address, accumulate)
try_wait_mbarrier(address, phase, acquire)
wait_mbarrier(address, phase)
arrive_mbarrier(address)
umma_commit(address, multicast_mask)
alloc_tmem(cta_group, columns); relinquish_tmem(cta_group); dealloc_tmem(cta_group)
```

## Complete operation sketch

```python
# ========================================================================
# 1. Host descriptors and persistent launch
# Source primary 864-1207, 1913-1980; runner 6361-6795.
# ========================================================================
if tactic.swap_ab:
    kernel_M, kernel_N = N, M
    kernel_A, kernel_B = transpose_storage_view(B), A
    kernel_SFA, kernel_SFB = transpose_storage_view(SFB), SFA
    C_MAJOR = "m"
else:
    kernel_M, kernel_N = M, N
    kernel_A, kernel_B = A, transpose_storage_view(B)
    kernel_SFA, kernel_SFB = SFA, transpose_storage_view(SFB)
    C_MAJOR = "n"

map_A = tensor_map(kernel_A, rank=2, extents=(K//2, kernel_M),
                   byte_strides=(1, K//2), box_extents=source_a_box(P),
                   dtype=U8, swizzle_code=source_a_swizzle(P), oob_zero=True)
map_B = tensor_map(kernel_B, rank=2, extents=(K//2, kernel_N),
                   byte_strides=(1, K//2), box_extents=source_b_box(P),
                   dtype=U8, swizzle_code=source_b_swizzle(P), oob_zero=True)
map_SFA = source_scale_tensormap(kernel_SFA, kernel_M, K, SF_VECTOR, operand="A")
map_SFB = source_scale_tensormap(kernel_SFB, kernel_N, K, SF_VECTOR, operand="B")
map_C = source_output_tensormap(C, M, N, output_dtype, C_MAJOR)

problem_ctas = (ceil_div(kernel_M, CTA_M), ceil_div(kernel_N, CTA_N), 1)
problem_clusters = source_cluster_count(problem_ctas, tactic.cluster)
grid_clusters = min(problem_clusters, source_max_active_clusters(tactic.cluster))
launch(grid=(cluster_m, cluster_n, grid_clusters), block_threads=192,
       cluster=(cluster_m, cluster_n, 1), arch="sm_107a",
       dynamic_smem_bytes=P.dynamic_smem_bytes, min_blocks_per_sm=1)
# instruction_selection: PTX 9.4 `.target sm_107a`, `.reqntid 192,1,1`,
# `.minnctapersm 1`, and `.extern .shared .align 1024`; one persistent launch.

# ========================================================================
# 2. Exact rank-one shared arena and scalar descriptors
# Source 414-719, 1264-1474; anchor prologues.
# ========================================================================
AB_FULL = smem_interval(smem, 0, 8*AB_STAGES)
AB_EMPTY = smem_interval(smem, 8*AB_STAGES, 8*AB_STAGES)
ACC_FULL = smem_interval(smem, 16*AB_STAGES, 8*ACC_STAGES)
ACC_EMPTY = smem_interval(smem, 16*AB_STAGES + 8*ACC_STAGES, 8*ACC_STAGES)
TMEM_DEALLOC = smem_interval(smem, source_tmem_dealloc_offset(P), 8)
TMEM_POINTER = smem_interval(smem, source_tmem_pointer_offset(P), 4)
C_OFFSET = 1024
A_OFFSET = C_OFFSET + C_STAGES*C_STAGE_BYTES
B_OFFSET = A_OFFSET + AB_STAGES*A_STAGE_BYTES
SFA_OFFSET = B_OFFSET + AB_STAGES*B_STAGE_BYTES
SFB_OFFSET = source_align_1024(SFA_OFFSET + AB_STAGES*SFA_STAGE_BYTES)

sC = smem_interval(smem, C_OFFSET, C_STAGES*C_STAGE_BYTES)
sA = smem_interval(smem, A_OFFSET, AB_STAGES*A_STAGE_BYTES)
sB = smem_interval(smem, B_OFFSET, AB_STAGES*B_STAGE_BYTES)
sSFA = smem_interval(smem, SFA_OFFSET, AB_STAGES*SFA_STAGE_BYTES)
sSFB = smem_interval(smem, SFB_OFFSET, AB_STAGES*SFB_STAGE_BYTES)

# One exclusive 576-column allocation. ACC, SFA, and the SFB copy region do not
# alias; the separate SFB MMA read view intentionally overlaps its copy region.
# ACC is live from ACC_EMPTY acquire through the four-warp ACC_EMPTY release;
# SFA/SFB are overwritten only after AB_FULL acquire and are consumed before
# the matching AB_EMPTY commit. The tail after the shifted-view reserve is
# untouched.
TMEM_ACC = tmem_column_interval(tmem_base, 0, NUM_ACC_TMEM_COLS)
TMEM_SFA = tmem_column_interval(tmem_base, NUM_ACC_TMEM_COLS,
                                NUM_SFA_TMEM_COLS)
SFB_COL = NUM_ACC_TMEM_COLS + NUM_SFA_TMEM_COLS
SFB_SHIFT_RESERVE = 2 if CTA_N in (64, 192) else 0
TMEM_SFB_COPY = tmem_column_interval(tmem_base, SFB_COL, NUM_SFB_TMEM_COLS)
# This is an overlapping read alias, never a scale-copy destination. On an odd
# N tile it begins two columns later; the instruction's N=64/192 logical extent
# omits the corresponding padded 64-column portion at the other end.
TMEM_SFB_MMA = lambda tile_n: tmem_column_interval(
    tmem_base, SFB_COL + (2 if CTA_N in (64,192) and tile_n % 2 else 0),
    NUM_SFB_TMEM_COLS)
TMEM_UNUSED = tmem_column_interval(
    tmem_base, SFB_COL + NUM_SFB_TMEM_COLS + SFB_SHIFT_RESERVE,
    576 - SFB_COL - NUM_SFB_TMEM_COLS - SFB_SHIFT_RESERVE)

def acc_stage_column(stage, reuse_slice=0):
    # reuse_slice is 0=B-keep and 1=B-reuse; it is statically zero otherwise.
    return tmem_base + CTA_N * ((2 if B_REUSE else 1)*stage + reuse_slice)
def sfa_base_column():
    return tmem_base + NUM_ACC_TMEM_COLS
def sfb_copy_base_column():
    return tmem_base + SFB_COL
def sfb_mma_base_column(tile_n):
    # The source applies this only to MMA reads; copies remain at the fixed base.
    return tmem_base + SFB_COL + (2 if CTA_N in (64,192) and tile_n % 2 else 0)

def a_stage_address(stage, row, kk):
    return A_OFFSET + stage*A_STAGE_BYTES + source_a_smem_byte_offset(P, row, kk)
def b_stage_address(stage, col, kk):
    return B_OFFSET + stage*B_STAGE_BYTES + source_b_smem_byte_offset(P, col, kk)
def sfa_stage_address(stage, logical_chunk):
    return SFA_OFFSET + stage*SFA_STAGE_BYTES + source_sfa_byte_offset(P, logical_chunk)
def sfb_stage_address(stage, logical_chunk):
    return SFB_OFFSET + stage*SFB_STAGE_BYTES + source_sfb_byte_offset(P, logical_chunk)
def c_stage_address(stage, row, col):
    return C_OFFSET + stage*C_STAGE_BYTES + source_c_smem_byte_offset(P, row, col)

A_DESC_BASE = source_smem_descriptor_base(P, operand="A")
B_DESC_BASE = source_smem_descriptor_base(P, operand="B")
SF_DESC_BASE = source_smem_descriptor_base(P, operand="SF")
# instruction_selection: scalar shifts, masks, XORs and ORs only. Shared
# descriptor address fields are `(shared_address >> 4) & 0x3fff`; no runtime
# layout/encoder object and no inline function call.

# ========================================================================
# 3. Coordinates, barriers, and cluster publication
# Source 1248-1479; all anchor prologues.
# ========================================================================
alpha_f32 = load_global_f32(alpha_ptr)
alpha_output = cvt_rn_scalar_f32(alpha_f32, output_dtype)
alpha_acc_f32 = widen_scalar_to_f32(alpha_output)
tid = thread_id_x(); warp = warp_uniform(tid >> 5); lane = tid & 31
cluster_rank = block_idx_in_cluster()
cta_v = block_idx_x() % CTA_GROUP
leader_cta = cta_v == 0
cluster_m_coord, cluster_n_coord = source_cluster_coord(cluster_rank, P)

if warp == TMA_WARP:
    for map in (map_A, map_B, map_SFA, map_SFB, map_C):
        prefetch_tensormap(map)
# instruction_selection: five `prefetch.tensormap`; one per opaque descriptor.

if warp == 0 and elect_one():
    for stage in static_range(AB_STAGES): init_mbarrier(AB_FULL+8*stage, 1)
    for stage in static_range(AB_STAGES):
        init_mbarrier(AB_EMPTY+8*stage, source_ab_empty_arrivals(P))
    for stage in static_range(ACC_STAGES): init_mbarrier(ACC_FULL+8*stage, 1)
    for stage in static_range(ACC_STAGES):
        init_mbarrier(ACC_EMPTY+8*stage, 4*(2 if CTA_GROUP == 2 else 1))
if CTA_GROUP == 2 and warp == 0 and elect_one():
    init_mbarrier(TMEM_DEALLOC, 32)
# instruction_selection: `mbarrier.init.shared.b64`; exactly
# `2*AB_STAGES + 2*ACC_STAGES` elected initializations for CTA-group one and
# one additional warp-0 initialization with arrival count 32 for CTA-group two.

fence_mbarrier_init_cluster()
if CTA_GROUP == 2:
    fence_mbarrier_init_cluster()          # publishes TMEM_DEALLOC initialization
if tactic.cluster_m * tactic.cluster_n > 1:
    cluster_arrive_relaxed()
prepare_scalar_multicast_masks_and_descriptors()
if tactic.cluster_m * tactic.cluster_n > 1:
    cluster_wait()
else:
    cta_sync()
common_scheduler = static_scheduler(problem_ctas, tactic.cluster, grid_clusters, "m")
initial_work = common_scheduler.initial_work()
# instruction_selection: one common `fence.mbarrier_init.release.cluster` and
# one additional fence only for CTA-group two. Non-singleton clusters issue a
# relaxed cluster arrival and delayed `barrier.cluster.wait`; singleton clusters
# issue neither cluster operation and use `bar.sync 0` after the one common fence.

# ========================================================================
# 4. Persistent TMA producer, warp 5
# Source 1481-1622; anchor producer regions.
# ========================================================================
with role(warp == TMA_WARP):
    work = common_scheduler.cursor(initial_work)
    ab_producer = pipeline_state(AB_STAGES, initial_phase=1)
    while work.valid:
        tile_m, tile_n = work.coordinate
        ab_producer.reset()
        if PREFETCH_DISTANCE > 0:
            for pf in runtime_range(min(PREFETCH_DISTANCE, ceil_div(K, K_TILE))):
                prefetch_tma_tile(map_A, source_a_coord(tile_m, pf), 2, source_l2_hint)
                prefetch_tma_tile(map_B, source_b_coord(tile_n, pf), 2, source_l2_hint)
                prefetch_tma_tile(map_SFA, source_sfa_coord(tile_m, pf), 3, source_l2_hint)
                prefetch_tma_tile(map_SFB, source_sfb_coord(tile_n, pf), 3, source_l2_hint)
        speculative_empty = try_wait_mbarrier(
            AB_EMPTY+8*ab_producer.stage, ab_producer.phase, acquire=True)
        for k_tile in runtime_range(ceil_div(K, K_TILE)):
            wait_if_needed(AB_EMPTY+8*ab_producer.stage,
                           ab_producer.phase, speculative_empty)
            if leader_cta and elect_one():
                arrive_expect_tx(AB_FULL+8*ab_producer.stage,
                                 source_tma_transaction_bytes(P))
            if elect_one():
                current_full = AB_FULL + 8*ab_producer.stage
                tma_load(map_A, source_a_coord(tile_m,k_tile),
                         a_stage_address(ab_producer.stage,0,0), current_full,
                         CTA_GROUP, source_a_multicast_mask(P), source_l2_hint)
                tma_load(map_B, source_b_coord(tile_n,k_tile),
                         b_stage_address(ab_producer.stage,0,0), current_full,
                         CTA_GROUP, source_b_multicast_mask(P), source_l2_hint)
                tma_load(map_SFA, source_sfa_coord(tile_m,k_tile),
                         sfa_stage_address(ab_producer.stage,0), current_full,
                         CTA_GROUP, source_sfa_multicast_mask(P), source_l2_hint)
                tma_load(map_SFB, source_sfb_coord(tile_n,k_tile),
                         sfb_stage_address(ab_producer.stage,0), current_full,
                         CTA_GROUP, source_sfb_multicast_mask(P), source_l2_hint)
            if PREFETCH_DISTANCE > 0 and k_tile < ceil_div(K,K_TILE)-PREFETCH_DISTANCE:
                rolling_k = k_tile + PREFETCH_DISTANCE
                prefetch_tma_tile(map_A, source_a_coord(tile_m,rolling_k), 2, source_l2_hint)
                prefetch_tma_tile(map_B, source_b_coord(tile_n,rolling_k), 2, source_l2_hint)
                prefetch_tma_tile(map_SFA, source_sfa_coord(tile_m,rolling_k), 3, source_l2_hint)
                prefetch_tma_tile(map_SFB, source_sfb_coord(tile_n,rolling_k), 3, source_l2_hint)
            advance(ab_producer); speculative_empty = next_try_wait_if_live()
        work.advance()
    producer_tail_wait_every_stage(ab_producer)
# instruction_selection: elected `mbarrier.arrive.expect_tx.shared.b64`; four
# TMA loads per K tile. A/B select
# `cp.async.bulk.tensor.2d.shared::{cta|cluster}.global.tile.mbarrier::complete_tx::bytes`
# and SFA/SFB select the otherwise identical `tensor.3d` family. A/SFA use
# shared::cta when cluster-N is one and otherwise shared::cluster with
# `.multicast::cluster`; B/SFB use shared::cta when the number of cluster-M
# groups is one and otherwise shared::cluster with `.multicast::cluster`.
# Every form ends in `.L2::cache_hint`; CTA-group two uses shared::cluster,
# its leader completion barrier, and the final `.cta_group::2` modifier, with
# multicast retained when the operand spans multiple same-direction peers.
# Each enabled initial iteration and guarded rolling
# iteration emits A/B `cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint`
# and SFA/SFB `cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint`.
# The initial extent is `min(PREFETCH_DISTANCE, ceil_div(K,256))`; rolling has
# exactly `ceil_div(K,256)-PREFETCH_DISTANCE` iterations when positive.
# Zero-prefetch emits no data-prefetch instructions. Empty-buffer speculative
# acquires lower to `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
# unresolved waits and the all-`AB_STAGES` producer-tail drain lower to parity
# `mbarrier.try_wait.parity.shared.b64` loops at the current stage address.

# ========================================================================
# 5. Persistent MMA, warp 4
# Source 1624-1779; anchor MMA regions.
# ========================================================================
with role(warp == MMA_WARP):
    named_barrier(2, 160)
    tmem_base = load_shared_u32(TMEM_POINTER)
    work = common_scheduler.cursor(initial_work)
    ab_consumer = pipeline_state(AB_STAGES, initial_phase=0)
    acc_producer = pipeline_state(ACC_STAGES, initial_phase=1)
    while work.valid:
        tile_m, tile_n = work.coordinate
        ab_consumer.reset()
        speculative_full = True
        if leader_cta and ceil_div(K,K_TILE) > 0:
            speculative_full = try_wait_mbarrier(
                AB_FULL+8*ab_consumer.stage, ab_consumer.phase, acquire=True)
        if leader_cta:
            wait_mbarrier(ACC_EMPTY+8*acc_producer.stage, acc_producer.phase,
                          acquire=False)
        accumulate = False
        for k_tile in runtime_range(ceil_div(K, K_TILE)):
            if leader_cta:
                wait_if_needed(AB_FULL+8*ab_consumer.stage,
                               ab_consumer.phase, speculative_full)
            if leader_cta:
                if B_REUSE:
                    for k_block in static_range(K_BLOCKS):
                        # Exact interleave: SFA-keep, SFB, SFA-reuse, then two MMAs.
                        for issue in static_range(2):
                            umma_scale_copy(1, source_sfa_keep_tmem(P,k_block)+4*issue,
                                source_sfa_keep_desc(P,ab_consumer.stage,k_block,issue))
                        for issue in static_range(2):
                            umma_scale_copy(
                                1,
                                sfb_copy_base_column()
                                + source_sfb_kblock_offset(P,k_block)+4*issue,
                                source_sfb_desc(P,ab_consumer.stage,k_block,issue))
                        for issue in static_range(2):
                            umma_scale_copy(1, source_sfa_reuse_tmem(P,k_block)+4*issue,
                                source_sfa_reuse_desc(P,ab_consumer.stage,k_block,issue))
                        umma_blockscaled(1, "a::discard,b::fill",
                            source_acc_keep(P,acc_producer.stage),
                            source_a_keep_desc(P,ab_consumer.stage,k_block),
                            source_b_desc(P,ab_consumer.stage,k_block),
                            source_instruction_desc(P,"keep"),
                            source_sfa_keep_tmem(P,k_block),
                            sfb_mma_base_column(tile_n)+source_sfb_kblock_offset(P,k_block),
                            accumulate)
                        umma_blockscaled(1, "a::discard,b::lastuse",
                            source_acc_reuse(P,acc_producer.stage),
                            source_a_reuse_desc(P,ab_consumer.stage,k_block),
                            source_b_desc(P,ab_consumer.stage,k_block),
                            source_instruction_desc(P,"reuse"),
                            source_sfa_reuse_tmem(P,k_block),
                            sfb_mma_base_column(tile_n)+source_sfb_kblock_offset(P,k_block),
                            accumulate)
                        accumulate = True
                else:
                    for sfa_issue in static_range(SFA_STAGE_BYTES // 512):
                        umma_scale_copy(CTA_GROUP,
                            sfa_base_column()+4*sfa_issue,
                            source_sfa_issue_desc(P,ab_consumer.stage,sfa_issue))
                    for sfb_issue in static_range(SFB_STAGE_BYTES // 512):
                        umma_scale_copy(CTA_GROUP,
                            sfb_copy_base_column()+4*sfb_issue,
                            source_sfb_issue_desc(P,ab_consumer.stage,sfb_issue))
                    for k_block in static_range(K_BLOCKS):
                        umma_blockscaled(CTA_GROUP, "a::discard",
                            source_acc(P,acc_producer.stage),
                            source_a_desc(P,ab_consumer.stage,k_block),
                            source_b_desc(P,ab_consumer.stage,k_block),
                            source_instruction_desc(P),
                            source_sfa_tmem(P,k_block),
                            sfb_mma_base_column(tile_n)+source_sfb_kblock_offset(P,k_block),
                            accumulate)
                        accumulate = True
                umma_commit(AB_EMPTY+8*ab_consumer.stage,
                            source_ab_empty_multicast_mask(P,cluster_rank))
            advance(ab_consumer)
            if leader_cta and k_tile+1 < ceil_div(K,K_TILE):
                speculative_full = try_wait_mbarrier(
                    AB_FULL+8*ab_consumer.stage, ab_consumer.phase, acquire=True)
        if leader_cta:
            umma_commit(ACC_FULL+8*acc_producer.stage,
                        source_acc_full_multicast_mask(P,cluster_rank))
        advance(acc_producer); work.advance()
    if leader_cta:
        for unused in static_range(ACC_STAGES-1):
            advance(acc_producer)
        wait_mbarrier(ACC_EMPTY+8*acc_producer.stage, acc_producer.phase,
                      acquire=False)
# instruction_selection: `tcgen05.cp.cta_group::{1|2}.32x128b.warpx4`;
# `tcgen05.mma.cta_group::{1|2}.kind::mxf4nvf4.block_scale.block{16|32}`.
# Ordinary cases use `.collector::a::discard`. B-reuse cases use paired
# `.collector::a::discard.collector::b::fill` and `...b::lastuse` in the exact
# interleaved order. The elected leader CTA issues one AB_EMPTY commit per K
# tile and one ACC_FULL commit per output tile. CTA-group two and clustered
# group one use
# `tcgen05.commit.cta_group::{2|1}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`
# with the exact source AB-consumer or ACC-producer mask. Singleton group one
# uses
# `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`
# with no multicast operand.
# Ordinary SFA and SFB loops have `SFA_STAGE_BYTES/512` then
# `SFB_STAGE_BYTES/512` static issues, respectively (for example 4 then 8 for
# N=192 NVFP4). In B-reuse NVFP4, each SFA-keep, SFB, and SFA-reuse source-copy
# call expands to two static issues per K block: six per block, twelve total.
# AB_FULL speculative probes use current parity and
# `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`; unresolved AB_FULL
# waits and the blocking ACC_EMPTY producer acquire use non-acquire
# `mbarrier.try_wait.parity.shared.b64` loops. At the tail, only the leader CTA
# advances accumulator state `ACC_STAGES-1` times and performs one final
# non-acquire ACC_EMPTY wait. AB release commits target the current AB_EMPTY
# stage and ACC publication targets the current ACC_FULL stage.

# ========================================================================
# 6. Four persistent alpha-scaled TMA epilogue warps
# Source 1781-1911; epilogue_utils 486-574; anchor epilogues.
# ========================================================================
with role(warp in EPI_WARPS):
    if warp == 0:
        alloc_tmem(CTA_GROUP, TMEM_ALLOC_COLS, destination=TMEM_POINTER,
                   exclusive=True)
        # tcgen05.alloc.exclusive.cta_group::{1|2}.sync.aligned.shared::cta.b32
    named_barrier(2, 160)
    # allocation retrieval: bar.sync 2,160
    tmem_base = load_shared_u32(TMEM_POINTER)
    work = common_scheduler.cursor(initial_work)  # reset to source initial work
    acc_consumer = pipeline_state(ACC_STAGES, initial_phase=0)
    c_sequence = 0
    while work.valid:
        tile_m, tile_n = work.coordinate
        wait_mbarrier(ACC_FULL+8*acc_consumer.stage, acc_consumer.phase,
                      acquire=False)
        for subtile in static_range(source_epilogue_subtiles(P)):
            values = tmem_load_source_fragment(P, tmem_base,
                                               acc_consumer.stage, warp, subtile)
            for pair in static_range(16):
                values[2*pair:2*pair+2] = mul_f32x2(
                    values[2*pair:2*pair+2], alpha_acc_f32)
            packed = cvt_rn_x2_f32(values, output_dtype)
            c_stage = c_sequence % C_STAGES
            source_register_store_shared(P, packed,
                                         c_stage_address(c_stage,0,0), warp, lane)
            fence_async_shared_cta()       # fence.proxy.async.shared::cta
            named_barrier(1,128)           # bar.sync 1,128
            if warp == 0:
                for source_store_piece in source_output_store_pieces(P):
                    tma_store(map_C, source_c_coord(tile_m,tile_n,subtile,source_store_piece),
                              source_c_piece_address(P,c_stage,source_store_piece),
                              source_l2_hint)
                tma_commit()                # cp.async.bulk.commit_group
                tma_wait(pending=C_STAGES-1) # cp.async.bulk.wait_group.read C_STAGES-1
            named_barrier(1,128)           # bar.sync 1,128
            c_sequence += 1
        named_barrier(1,128)               # bar.sync 1,128 before ACC release
        if elect_one():
            if CTA_GROUP == 1:
                mbarrier_arrive_shared(
                    ACC_EMPTY+8*acc_consumer.stage)  # mbarrier.arrive.shared.b64
            else:
                acc_owner = source_acc_empty_peer_rank(cluster_rank, warp)
                remote_acc_empty = mapa_shared_cluster_u32(
                    ACC_EMPTY+8*acc_consumer.stage, acc_owner)
                mbarrier_arrive_shared_cluster(
                    remote_acc_empty)       # mbarrier.arrive.shared::cluster.b64
        advance(acc_consumer); work.advance()
    tma_wait(pending=0)                    # cp.async.bulk.wait_group.read 0
    if warp == 0:
        relinquish_tmem(CTA_GROUP)
        # tcgen05.relinquish_alloc_permit.cta_group::{1|2}.sync.aligned
        if CTA_GROUP == 2:
            peer_rank = cluster_rank ^ 1
            peer_dealloc = mapa_shared_cluster_u32(TMEM_DEALLOC, peer_rank)
            # mapa.shared::cluster.u32
            mbarrier_arrive_shared_cluster(peer_dealloc)
            # mbarrier.arrive.shared::cluster.b64
            wait_mbarrier(TMEM_DEALLOC, phase=0, acquire=False)
            # mbarrier.try_wait.parity.shared.b64, phase 0
        dealloc_tmem(CTA_GROUP, tmem_base, TMEM_ALLOC_COLS, exclusive=True)
# instruction_selection: output-major N, for either CTA group, uses one
# `tcgen05.ld.sync.aligned.32x32b.x32.b32` and exactly four
# `st.shared.v4.b32` per lane per subtile.
# Output-major M, for either CTA group, uses two
# `tcgen05.ld.sync.aligned.16x256b.x4.b32` and four
# `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16` per subtile. Each family is
# repeated exactly `source_epilogue_subtiles(P)` times. Output-major N issues
# one and output-major M issues two
# `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint`
# instructions per subtile before one TMA-store commit/wait sequence.
# Alpha first uses scalar `cvt.rn.f16.f32; cvt.f32.f16` or
# `cvt.rn.bf16.f32; cvt.f32.bf16`, then sixteen FP32x2 multiplies precede
# sixteen `cvt.rn.{f16x2|bf16x2}.f32`. Stores are
# rank-2 `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group`, followed
# by `cp.async.bulk.commit_group`,
# `cp.async.bulk.wait_group.read C_STAGES-1`, and `bar.sync 1,128`;
# the tail is `cp.async.bulk.wait_group.read 0`. ACC_FULL waits use non-acquire
# `mbarrier.try_wait.parity.shared.b64` with the current parity;
# after all subtiles, each epilogue warp elects one lane to issue one ACC_EMPTY
# arrival: group one targets its local shared::cta barrier and group two uses
# the source remote cluster destination. Store tail drains to pending zero.
# Both groups use `tcgen05.relinquish_alloc_permit.cta_group::{1|2}.sync.aligned`
# then `tcgen05.dealloc.exclusive.cta_group::{1|2}.sync.aligned.b32` for 576
# columns. Only group two executes, in between, XOR-peer
# `mapa.shared::cluster.u32`, remote `mbarrier.arrive.shared::cluster.b64`,
# and a local phase-zero `mbarrier.try_wait.parity.shared.b64` loop.
```

## Required verification boundary

Correctness must exercise a structural cover of every production-reachable
SM107 mode, with explicit rows for B-reuse, CTA-group 2, both scale modes, both
output dtypes, swap, every N tile, every cluster multicast direction, all three
prefetch settings, non-tile-multiple M/N boundaries, and multiple K tiles.
Inputs use deterministic nibble patterns and exactly representable scale/alpha
values, plus randomized finite payloads. Output storage has front/back canaries
and every case is rerun for determinism.

The first comparison is byte-for-byte against the pinned source on the exact
same tensors and forced tactic. An independent FP64 decode-and-matmul oracle
checks both outputs with dtype-tight error bounds, using
`float32(cast_output(alpha_f32))` exactly: zero tolerance whenever the
reference result is exactly representable; otherwise at most one output ULP
plus the explicitly computed FP32 accumulation bound. NaNs, infinities,
unwritten values, canary damage, or non-determinism are hard failures. Low-level
IR must reject `K.cuda.func_call` and inline-CUDA function-call exemptions.

Performance is a later, permanently reviewer-free gate. Only `bench_suite`
paired source/reference rows count, both sides must compile PTX 9.4 for
`sm_107a`, and every frozen row must satisfy the strict ratio
`source_time / tirx_time > 0.99`.
