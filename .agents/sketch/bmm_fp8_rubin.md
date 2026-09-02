<!--
This file is a design sketch for a TIRx port of FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ 012cfdb97f217e0d48bc9352c17a74068c9e495b),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer SM107 FP8 batched GEMM kernel sketch

This non-executable, operation-level sketch freezes the intended transcription
of `flashinfer/gemm/kernels/bmm_fp8_rubin.py` together with the inherited
scheduler in `bmm_fp8_blackwell.py` and the scaled TMA epilogue in
`epilogue_utils.py`. The pinned source commit is
`012cfdb97f217e0d48bc9352c17a74068c9e495b`; the primary file SHA-256 is
`f10b5ee03096af8394b57cfbe7abb6ee3103baf87c6a58a60f576ead3f4386f3`.

The writer independently exported all 48 tactic/dtype specializations with
CuTeDSL line information under
`.porting/bmm_fp8_rubin/source_export/writer/`. Every artifact targets
`sm_107a`, declares 192 threads, contains `.loc` records, and directly maps
files 1/2/3 to `bmm_fp8_rubin.py`, `bmm_fp8_blackwell.py`, and
`epilogue_utils.py`. The primary tactic-0 E4M3-to-BF16 artifact has SHA-256
`3f90d9c1260b601e6de70e655963a435c256b06a79e59b5eabc6068c79712072`
and 295 `.loc` records. “Primary PTX” below means that artifact.

After the independent sketch reviewer returns PASS, this file is immutable.

## Transcription boundary and invariants

The executable module is one parameterized K-language port and must import the
device language only as `import tirx_kernels.kern as K`. It may use raw
`K.ptx[...]` instructions, scalar K control flow, opaque TensorMaps, rank-one
shared allocations, local register arrays, and ordinary integer helpers. It
must not use a tile primitive, a first-class mapping object, a direct TVM script
namespace, `K.cuda.func_call`, or any inline-CUDA function-call exemption. It
must not modify anything under `tirx_kernels/kern/`.

All source mappings are lowered before tracing into scalar extents, strides,
byte offsets, XOR swizzle arithmetic, TensorMap fields, and integer UMMA
descriptor bits. Every shared object is a byte interval in one rank-one arena;
every TMEM reference is a scalar row/column address.

The supported semantic operation is

```text
C[b,m,n] = cast_c_dtype(
    output_scale_f32 * sum_k(float32(A[b,m,k]) * float32(B[b,k,n]))
)
```

where A and B have the same FP8 type, either E4M3FN or E5M2; accumulation and
scaling are FP32; C is BF16, FP16, or FP32. A is contiguous in K, B and C are
contiguous in N. M/N/K are positive multiples of 16. TensorMap out-of-bounds
fill supplies zeros for the final M/N/K tiles, and the output TensorMap drops
out-of-bounds stores. The source loads `output_scale[0]` inside the kernel and
never performs an unscaled branch.

The source's B-keep/B-reuse branch is out of scope because its executable
predicate is `mma_tiler_m // mma_instruction_m == 2`, which is false for all
eight live tactics (`256 // 256 == 1`). Every writer PTX therefore uses only
`collector::a::discard`; no `fill`, `use`, or `lastuse` collector occurs.

## Static specialization table

Every row uses CTA-group 2, four epilogue warps (0..3), MMA warp 4, TMA warp 5,
one source swizzle unit, TMA output stores, two FP32 accumulator stages, a
128x32 epilogue subtile, and a per-CTA M extent of 128.

| tactic | MMA tile | instruction | cluster | raster | AB stages BF16/FP16/FP32 | C stages BF16/FP16/FP32 | TMEM columns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 256x256x128 | 256x256x64 | 2x1 | M | 9/9/9 | 4/4/2 | 512 |
| 1 | 256x128x128 | 256x128x64 | 2x1 | M | 12/12/12 | 4/4/2 | 256 |
| 2 | 256x256x64 | 256x256x32 | 2x1 | M | 19/19/18 | 2/2/2 | 512 |
| 3 | 256x256x64 | 256x256x64 | 2x1 | M | 19/19/18 | 2/2/2 | 512 |
| 4 | 256x256x128 | 256x256x64 | 2x2 | M | 9/9/9 | 4/4/2 | 512 |
| 5 | 256x128x128 | 256x128x64 | 4x1 | M | 12/12/12 | 4/4/2 | 256 |
| 6 | 256x256x128 | 256x256x64 | 2x1 | N | 9/9/9 | 4/4/2 | 512 |
| 7 | 256x128x128 | 256x128x64 | 2x1 | N | 12/12/12 | 4/4/2 | 256 |

The stage counts are evidenced by emitted mbarrier groups:
`AB_STAGES, AB_STAGES, 2, 2, 1`. The output depth is evidenced by
`cp.async.bulk.wait_group.read 3` for four stages and `...read 1` for two.
Input dtype changes only descriptor format bits. Output dtype changes the
conversion/store sequence and, for tactics 2/3 FP32, reduces AB depth by one.

## Primitive vocabulary

The pseudocode uses structural names that carry only scalars or opaque hardware
descriptors:

```python
specialize_static(...)
launch(grid, block_threads, cluster, dynamic_smem_bytes)
tensor_map(base, rank, extents, byte_strides, box_extents,
           element_strides, dtype, swizzle_code, oob_zero)
smem_interval(base, byte_offset, byte_count)
tmem_address(base_column, row, column)
encode_smem_descriptor(base_bits, shared_address)
static_scheduler_cursor(problem_ctas, cluster_shape, grid_clusters, raster)
```

Movement, compute, and synchronization are explicit:

```python
load_global_f32(pointer)
prefetch_tensormap(map)
tma_load_group2(map, coordinates, shared_address, full_barrier,
                optional_multicast_mask, cache_hint)
tma_store(map, coordinates, shared_address, cache_hint)
tmem_load_32x32_f32(tmem_address, 32_register_words)
register_store_shared(words, shared_address)
umma_f8f6f4_group2(accumulator, a_descriptor, b_descriptor,
                   instruction_descriptor, accumulate)
init_mbarrier(address, arrivals)
arrive_expect_tx(address, bytes)
try_wait_mbarrier(address, phase, acquire)
wait_mbarrier(address, phase)
arrive_mbarrier(address)
umma_commit(address, multicast_mask)
fence_mbarrier_init_cluster()
cluster_arrive_relaxed(); cluster_wait(); named_barrier(id, threads)
fence_async_shared_cta(); tma_commit(); tma_wait(pending)
alloc_tmem_group2(columns); relinquish_tmem_group2(); dealloc_tmem_group2(columns)
```

## Complete operation sketch

```python
# ==========================================================================
# 1. Specialization, opaque TensorMaps, and persistent launch
# Source Rubin 182-257 and 721-817; Blackwell 762-784; primary PTX 8-42.
# ==========================================================================
P = specialize_static(B, M, N, K, ab_dtype, c_dtype, tactic, target="sm_107a")
assert ab_dtype in (E4M3FN, E5M2)
assert c_dtype in (BF16, FP16, FP32)
assert B > 0 and M % 16 == 0 and N % 16 == 0 and K % 16 == 0

CTA_GROUP = 2
CTA_M = 128
CTA_N = P.mma_tile_n
K_TILE = P.mma_tile_k
INSTRUCTION_K = P.instruction_k
K_PHASES = K_TILE // INSTRUCTION_K
WARPS = 6
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
ACC_STAGES = 2
AB_STAGES = P.ab_stages
C_STAGES = P.c_stages
EPI_M = 128
EPI_N = 32
TMEM_COLUMNS = P.tmem_columns

# TensorMap logical dimensions are ordered by physical contiguity. All fields
# below are scalar integers fixed by the specialization; maps themselves are
# opaque 128-byte launch arguments.
map_A = tensor_map(
    A_ptr, rank=3, extents=(K, M, B), byte_strides=(1, K, M*K),
    box_extents=(K_TILE, 256, 1), element_strides=(1, 1, 1),
    dtype=ab_dtype, swizzle_code=SW128B, oob_zero=True,
)
map_B = tensor_map(
    B_ptr, rank=3, extents=(N, K, B), byte_strides=(1, N, K*N),
    box_extents=(CTA_N, K_TILE, 1), element_strides=(1, 1, 1),
    dtype=ab_dtype, swizzle_code=SW128B, oob_zero=True,
)
C_BYTES = 2 if c_dtype in (BF16, FP16) else 4
map_C = tensor_map(
    C_ptr, rank=3, extents=(N, M, B),
    byte_strides=(C_BYTES, N*C_BYTES, M*N*C_BYTES),
    box_extents=(EPI_N, EPI_M, 1), element_strides=(1, 1, 1),
    dtype=c_dtype, swizzle_code=source_output_swizzle(c_dtype), oob_zero=False,
)

problem_ctas = (ceil_div(M, CTA_M), ceil_div(N, CTA_N), B)
problem_clusters = cluster_count(problem_ctas, P.cluster_m, P.cluster_n)
grid_clusters = min(problem_clusters, source_max_active_clusters(P.cluster_size))
launch(
    grid=(P.cluster_m, P.cluster_n, grid_clusters), block_threads=192,
    cluster=(P.cluster_m, P.cluster_n, 1), arch="sm_107a",
    dynamic_smem_bytes=P.dynamic_smem_bytes, min_blocks_per_sm=1,
)
# instruction_selection: `.target sm_107a`, `.reqntid 192,1,1`, dynamic
# shared declaration aligned to 1024, and cluster launch metadata. Extent: one
# specialized launch. `ctaid.x/y` are the CTA coordinates inside one cluster;
# `ctaid.z` is the initial persistent-cluster work index. The three role
# cursors below use exactly the same scalar StaticPersistentTileScheduler
# decode and advance by `grid_clusters`.

# ==========================================================================
# 2. Exact rank-one shared storage
# Source Rubin 308-388, 609-616; primary PTX 110-229, 376-434, 1260-1280.
# ==========================================================================
AB_FULL = smem_interval(smem, 0, 8 * AB_STAGES)
AB_EMPTY = smem_interval(smem, 8 * AB_STAGES, 8 * AB_STAGES)
ACC_FULL = smem_interval(smem, 16 * AB_STAGES, 16)
ACC_EMPTY = smem_interval(smem, 16 * AB_STAGES + 16, 16)
TMEM_DEALLOC = smem_interval(smem, 16 * AB_STAGES + 32, 8)
TMEM_POINTER = smem_interval(smem, 16 * AB_STAGES + 40, 4)

if AB_STAGES <= 12:
    A_OFFSET = 256
else:
    A_OFFSET = 384
A_STAGE_BYTES = 16384 if K_TILE == 128 else 8192
B_STAGE_BYTES = 16384 if CTA_N == 256 and K_TILE == 128 else 8192
B_OFFSET = A_OFFSET + AB_STAGES * A_STAGE_BYTES
C_OFFSET = B_OFFSET + AB_STAGES * B_STAGE_BYTES
C_STAGE_BYTES = 8192 if c_dtype in (BF16, FP16) else 16384

sA = smem_interval(smem, A_OFFSET, AB_STAGES * A_STAGE_BYTES)
sB = smem_interval(smem, B_OFFSET, AB_STAGES * B_STAGE_BYTES)
sC = smem_interval(smem, C_OFFSET, C_STAGES * C_STAGE_BYTES)

# Concrete writer offsets:
#   tactics 0/4/6: A=256, B=147712, C=295168, end=327936
#   tactics 1/5/7: A=256, B=196864, C=295168, end=327936
#   tactics 2/3 BF16/FP16: A=384, B=156032, C=311680, end=328064
#   tactics 2/3 FP32: A=384, B=147840, C=295296, end=328064

def a_stage_address(stage, row, kk):
    return sA + stage*A_STAGE_BYTES + source_sw128_a_byte_offset(row, kk, K_TILE)

def b_stage_address(stage, col, kk):
    return sB + stage*B_STAGE_BYTES + source_sw128_b_byte_offset(col, kk, CTA_N, K_TILE)

def c_stage_address(stage, row, col):
    return sC + stage*C_STAGE_BYTES + source_output_byte_offset(row, col, c_dtype)

# Each source_* offset function is scalar integer/XOR arithmetic matching the
# corresponding TensorMap swizzle. It returns a byte count and constructs no
# mapping value.

# ==========================================================================
# 3. Raw descriptor constants and scalar address insertion
# Source Rubin 133-180, 441-447, 583-592; primary PTX 756-805.
# ==========================================================================
# Dense SM107 instruction descriptor bits are folded on the host. E4M3 uses A/B
# format 0; E5M2 uses A/B format 1. D format is FP32, trans_A=0, trans_B=1,
# negate/saturate/sparse are zero. SM107 K=64 sets bit 29 in addition to the
# common SM100 fields. Writer PTX fixes these exact images:
if CTA_N == 256 and INSTRUCTION_K == 64:
    INSTR_DESC = 0x30410010 if ab_dtype == E4M3FN else 0x30410490
elif CTA_N == 128 and INSTRUCTION_K == 64:
    INSTR_DESC = 0x30210010 if ab_dtype == E4M3FN else 0x30210490
else:  # CTA_N=256, INSTRUCTION_K=32
    INSTR_DESC = 0x10410010 if ab_dtype == E4M3FN else 0x10410490

# Matrix descriptor constant fields are likewise compile-time integers. Only
# the 14-bit shared address is inserted per stage/K phase at run time.
A_DESC_BASE = source_smem_descriptor_base(
    leading_units=source_a_leading_16B_units(K_TILE),
    stride_units=source_a_stride_16B_units(K_TILE), swizzle_code=SW128B,
)
B_DESC_BASE = source_smem_descriptor_base(
    leading_units=source_b_leading_16B_units(CTA_N, K_TILE),
    stride_units=source_b_stride_16B_units(CTA_N, K_TILE), swizzle_code=SW128B,
)

def descriptor_with_address(base_bits, shared_address):
    return base_bits | ((shared_address >> 4) & 0x3FFF)

def a_descriptor(stage, kphase):
    return descriptor_with_address(
        A_DESC_BASE,
        a_stage_address(stage, source_a_descriptor_row(kphase),
                        source_a_descriptor_k(kphase, INSTRUCTION_K)),
    )

def b_descriptor(stage, kphase):
    return descriptor_with_address(
        B_DESC_BASE,
        b_stage_address(stage, source_b_descriptor_col(kphase),
                        source_b_descriptor_k(kphase, INSTRUCTION_K)),
    )

def accumulator_address(tmem_base, acc_stage):
    return tmem_address(tmem_base + acc_stage*CTA_N, row=0, column=0)

# instruction_selection: ordinary integer shifts/ORs plus raw 64-bit UMMA
# descriptors. No runtime encoder helper and no inline CUDA function call.

# ==========================================================================
# 4. Coordinates, barrier initialization, and cluster publication
# Source Rubin 282-450; primary PTX 78-322.
# ==========================================================================
output_scale_f32 = load_global_f32(output_scale_ptr)
# instruction_selection: one `ld.global.b32`; extent: one FP32 scalar load per
# thread before role dispatch, exactly source Rubin 282 and primary PTX 83-84.
tid = thread_id_x()
warp = warp_uniform(tid >> 5)
lane = tid & 31
cluster_rank = block_idx_in_cluster()
mma_tile_coord_v = block_idx_x() % CTA_GROUP
leader_cta = mma_tile_coord_v == 0
cluster_m_coord = cluster_rank % P.cluster_m
cluster_n_coord = (cluster_rank // P.cluster_m) % P.cluster_n

if warp == TMA_WARP:
    prefetch_tensormap(map_A)
    prefetch_tensormap(map_B)
    prefetch_tensormap(map_C)
# instruction_selection: three `prefetch.tensormap`; extent: one per opaque map.

AB_EMPTY_ARRIVALS = (
    source_num_mcast_ctas_a(P.cluster_m, P.cluster_n)
    + source_num_mcast_ctas_b(P.cluster_m, P.cluster_n) - 1
)
# Concrete values are 1 for tactics 0/1/2/3/6/7 and 2 for tactics 4/5.

if warp == 0 and elect_one():
    for stage in static_range(AB_STAGES):
        init_mbarrier(AB_FULL + 8*stage, arrivals=1)
    for stage in static_range(AB_STAGES):
        init_mbarrier(AB_EMPTY + 8*stage, arrivals=AB_EMPTY_ARRIVALS)
    for stage in static_range(ACC_STAGES):
        init_mbarrier(ACC_FULL + 8*stage, arrivals=1)
    for stage in static_range(ACC_STAGES):
        init_mbarrier(ACC_EMPTY + 8*stage, arrivals=8)

# instruction_selection: `mbarrier.init.shared.b64`; extent: exactly
# `2*AB_STAGES + 4` elected-lane initializations, with arrivals
# `1, AB_EMPTY_ARRIVALS, 1, 8` for the four consecutive groups. Writer PTX
# groups are 9/9/2/2, 12/12/2/2, 19/19/2/2, or 18/18/2/2.

if warp == TMA_WARP and elect_one():
    init_mbarrier(TMEM_DEALLOC, arrivals=32)
# instruction_selection: one `mbarrier.init.shared.b64` with arrival count 32;
# extent: the group-2 teardown barrier.

# CuTeDSL's two deferred pipeline constructors emit two publication fences,
# then pipeline_init_arrive performs the cluster arrive. The source waits only
# after shared/TMEM fragments and scalar descriptor coordinates are prepared.
fence_mbarrier_init_cluster()
fence_mbarrier_init_cluster()
if P.cluster_size > 1:
    cluster_arrive_relaxed()

A_MULTICAST_MASK = source_a_multicast_mask(P.cluster_m, P.cluster_n,
                                           cluster_m_coord, cluster_n_coord)
B_MULTICAST_MASK = source_b_multicast_mask(P.cluster_m, P.cluster_n,
                                           cluster_m_coord, cluster_n_coord)
prepare_scalar_scheduler_and_descriptor_terms()

if P.cluster_size > 1:
    cluster_wait()
else:
    cta_sync()
# instruction_selection: `fence.mbarrier_init.release.cluster` twice,
# `barrier.cluster.arrive.relaxed`, then delayed `barrier.cluster.wait`; extent:
# one prologue. Tactic 4's A load uses explicit multicast across cluster N;
# tactic 5's B load uses explicit multicast across the two CTA-group pairs in
# cluster M. CTA-group 2 itself does not require the explicit multicast suffix.

# ==========================================================================
# 5. Role: persistent TMA producer warp 5
# Source Rubin 452-502; primary PTX 323-704.
# ==========================================================================
with role(warp == TMA_WARP):
    work = static_scheduler_cursor(problem_ctas, P.cluster, grid_clusters, P.raster)
    tma_state = pipeline_state(AB_STAGES, initial_phase=1)
    while work.valid:
        tile_m_cta, tile_n_cta, batch = work.cta_coordinate
        mma_m = tile_m_cta // CTA_GROUP
        k_tiles = ceil_div(K, K_TILE)
        tma_state.reset()
        speculative_empty = try_wait_mbarrier(
            AB_EMPTY + 8*tma_state.stage, tma_state.phase, acquire=True,
        )
        for k_tile in runtime_range(k_tiles, unroll_hint=1):
            wait_if_needed(AB_EMPTY + 8*tma_state.stage,
                           tma_state.phase, speculative_empty)
            if elect_one():
                arrive_expect_tx(AB_FULL + 8*tma_state.stage, P.tma_transaction_bytes)
                # P.tma_transaction_bytes is 65536 for N256/K128,
                # 49152 for N128/K128, and 32768 for N256/K64.
                tma_load_group2(
                    map_A,
                    coordinates=(k_tile*K_TILE, mma_m*256, batch),
                    shared_address=a_stage_address(tma_state.stage, 0, 0),
                    full_barrier=AB_FULL + 8*tma_state.stage,
                    optional_multicast_mask=(A_MULTICAST_MASK if tactic == 4 else None),
                    cache_hint=source_l2_cache_hint,
                )
                tma_load_group2(
                    map_B,
                    coordinates=(tile_n_cta*CTA_N, k_tile*K_TILE, batch),
                    shared_address=b_stage_address(tma_state.stage, 0, 0),
                    full_barrier=AB_FULL + 8*tma_state.stage,
                    optional_multicast_mask=(B_MULTICAST_MASK if tactic == 5 else None),
                    cache_hint=source_l2_cache_hint,
                )
            advance(tma_state)
            speculative_empty = True
            if k_tile + 1 < k_tiles:
                speculative_empty = try_wait_mbarrier(
                    AB_EMPTY + 8*tma_state.stage, tma_state.phase, acquire=True,
                )
        work.advance()
    for unused_stage in static_range(AB_STAGES):
        wait_mbarrier(AB_EMPTY + 8*tma_state.stage, tma_state.phase)
        advance(tma_state)
# instruction_selection: elected `mbarrier.arrive.expect_tx.shared.b64`;
# two rank-3 `cp.async.bulk.tensor...cta_group::2` loads per K tile; optional
# `.multicast::cluster`; parity waits and a full producer tail.

# ==========================================================================
# 6. Role: persistent leader-CTA MMA warp 4
# Source Rubin 504-607; primary PTX 705-1068.
# ==========================================================================
with role(warp == MMA_WARP):
    named_barrier(2, 160)
    # instruction_selection: `bar.sync 2,160`; extent: all 32 MMA-warp lanes.
    tmem_base = load_shared_u32(TMEM_POINTER)
    work = static_scheduler_cursor(problem_ctas, P.cluster, grid_clusters, P.raster)
    ab_state = pipeline_state(AB_STAGES, initial_phase=0)
    acc_state = pipeline_state(ACC_STAGES, initial_phase=1)
    while work.valid:
        k_tiles = ceil_div(K, K_TILE)
        ab_state.reset()
        speculative_full = True
        if leader_cta:
            speculative_full = try_wait_mbarrier(
                AB_FULL + 8*ab_state.stage, ab_state.phase, acquire=True,
            )
            wait_mbarrier(ACC_EMPTY + 8*acc_state.stage, acc_state.phase)
        accumulate = False
        for k_tile in runtime_range(k_tiles):
            if leader_cta:
                wait_if_needed(AB_FULL + 8*ab_state.stage,
                               ab_state.phase, speculative_full)
                for kphase in static_range(K_PHASES):
                    if elect_one():
                        umma_f8f6f4_group2(
                            accumulator_address(tmem_base, acc_state.stage),
                            a_descriptor(ab_state.stage, kphase),
                            b_descriptor(ab_state.stage, kphase),
                            INSTR_DESC,
                            accumulate=accumulate,
                            collector="a::discard",
                            disabled_scale_registers=(0,0,0,0,0,0,0,0),
                        )
                    accumulate = True
                if elect_one():
                    umma_commit(AB_EMPTY + 8*ab_state.stage,
                                source_ab_empty_multicast_mask(P, cluster_rank))
                advance(ab_state)
                speculative_full = True
                if k_tile + 1 < k_tiles:
                    speculative_full = try_wait_mbarrier(
                        AB_FULL + 8*ab_state.stage, ab_state.phase, acquire=True,
                    )
        if leader_cta and elect_one():
            umma_commit(ACC_FULL + 8*acc_state.stage,
                        source_acc_full_multicast_mask(P, cluster_rank))
        advance(acc_state)
        work.advance()
    for unused_stage in static_range(ACC_STAGES):
        wait_mbarrier(ACC_EMPTY + 8*acc_state.stage, acc_state.phase)
        advance(acc_state)
# instruction_selection: raw
# `tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard`; two issues per
# source K tile for tactics 0/1/2/4/5/6/7 and one for tactic 3; raw
# `tcgen05.commit.cta_group::2.mbarrier::arrive::one...`; extent: every K tile
# and accumulator publication. Only the leader CTA issues UMMA.

# ==========================================================================
# 7. Roles: four persistent scaled TMA epilogue warps
# Source Rubin 609-666; epilogue_utils 100-255; primary PTX 1069-1412.
# ==========================================================================
with role(warp in EPI_WARPS):
    if warp == 0:
        alloc_tmem_group2(TMEM_COLUMNS, destination=TMEM_POINTER)
        # instruction_selection:
        # `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32`; extent:
        # the full warp 0 in both paired CTAs, 512 columns for N256 or 256 for
        # N128. There is deliberately no elected-lane predicate.
    named_barrier(2, 160)
    # instruction_selection: `bar.sync 2,160`; extent: all 160 lanes in the
    # MMA plus four epilogue warps after warp 0 publishes the TMEM pointer.
    tmem_base = load_shared_u32(TMEM_POINTER)
    work = static_scheduler_cursor(problem_ctas, P.cluster, grid_clusters, P.raster)
    acc_state = pipeline_state(ACC_STAGES, initial_phase=0)
    c_sequence = 0
    while work.valid:
        tile_m_cta, tile_n_cta, batch = work.cta_coordinate
        wait_mbarrier(ACC_FULL + 8*acc_state.stage, acc_state.phase)
        for subtile in static_range(CTA_N // EPI_N):
            # Each epilogue warp owns 32 rows. Each lane receives the 32 FP32
            # columns of its row as 32 b32 registers.
            acc_words = register_words(32)
            tmem_load_32x32_f32(
                tmem_address(
                    tmem_base + acc_state.stage*CTA_N + subtile*EPI_N,
                    row=warp*32, column=0,
                ),
                acc_words,
            )
            # Source vectorizes two FP32 values in each 64-bit register pair.
            for pair in static_range(16):
                acc_words[2*pair:2*pair+2] = mul_f32x2(
                    acc_words[2*pair:2*pair+2], output_scale_f32,
                )

            c_stage = c_sequence % C_STAGES
            if c_dtype == BF16:
                out_words = register_words(16)
                for pair in static_range(16):
                    out_words[pair] = cvt_rn_bf16x2_f32(
                        acc_words[2*pair+1], acc_words[2*pair],
                    )
                register_store_shared(
                    out_words,
                    c_stage_address(c_stage, warp*32, lane),
                    vector_bytes=16, vectors_per_lane_group=4,
                )
                # instruction_selection: four `st.shared.v4.b32` per lane;
                # extent: one 128x32 BF16 subtile across the four warps.
            elif c_dtype == FP16:
                out_words = register_words(16)
                for pair in static_range(16):
                    out_words[pair] = cvt_rn_f16x2_f32(
                        acc_words[2*pair+1], acc_words[2*pair],
                    )
                register_store_shared(
                    out_words,
                    c_stage_address(c_stage, warp*32, lane),
                    vector_bytes=16, vectors_per_lane_group=4,
                )
                # instruction_selection: four `st.shared.v4.b32` per lane;
                # extent: one 128x32 FP16 subtile across the four warps.
            else:
                register_store_shared(
                    acc_words,
                    c_stage_address(c_stage, warp*32, lane),
                    vector_bytes=16, vectors_per_lane_group=8,
                )
                # instruction_selection: eight `st.shared.v4.b32` per lane;
                # extent: one 128x32 FP32 subtile across the four warps.

            fence_async_shared_cta()
            named_barrier(1, 128)
            if warp == 0:
                tma_store(
                    map_C,
                    coordinates=(tile_n_cta*CTA_N + subtile*EPI_N,
                                 tile_m_cta*CTA_M, batch),
                    shared_address=c_stage_address(c_stage, 0, 0),
                    cache_hint=source_l2_cache_hint,
                )
                tma_commit()
                tma_wait(pending=C_STAGES - 1)
            named_barrier(1, 128)
            c_sequence += 1
        named_barrier(1, 128)
        if elect_one():
            # One elected lane in each of four warps and each paired CTA gives
            # the accumulator-empty barrier's eight arrivals.
            remote_empty = map_shared_to_accumulator_owner(
                ACC_EMPTY + 8*acc_state.stage, cluster_rank,
            )
            arrive_mbarrier(remote_empty)
        advance(acc_state)
        work.advance()
    tma_wait(pending=0)

    if warp == 0:
        relinquish_tmem_group2()
        # Both paired CTAs' warp 0 map to the XOR partner, signal that partner,
        # wait on their own local barrier, and execute group-2 deallocation.
        partner_rank = cluster_rank ^ 1
        remote_dealloc = map_shared_to_cluster_rank(TMEM_DEALLOC, partner_rank)
        arrive_mbarrier(remote_dealloc)
        wait_mbarrier(TMEM_DEALLOC, phase=0)
        dealloc_tmem_group2(tmem_base, TMEM_COLUMNS)
# instruction_selection per subtile:
#   one `tcgen05.ld.sync.aligned.32x32b.x32.b32` per epilogue warp;
#   sixteen `mul.f32x2`; sixteen BF16x2/FP16x2 conversions or none for FP32;
#   64 or 128 shared bytes per lane-group store sequence;
#   `fence.proxy.async.shared::cta`, `bar.sync 1,128`;
#   full warp-0 rank-3 TMA store, `cp.async.bulk.commit_group`, and
#   wait-group read 3/1 for the complete warp-collective store sequence.
# All four epilogue warps first execute `cp.async.bulk.wait_group.read 0`.
# Teardown then uses only full warp 0 in both paired CTAs:
# `tcgen05.relinquish_alloc_permit.cta_group::2`, XOR-partner
# `mapa.shared::cluster` arrival, local parity wait, and
# `tcgen05.dealloc.cta_group::2.sync.aligned.b32`. No elected-lane or
# single-owner-CTA predicate appears.
```

## Required verification boundary

Correctness must exercise every one of the 48 tactic/input/output
specializations on guarded, non-tile-multiple M/N/K shapes, plus the source
anchor `(B,M,N,K)=(1,256,10304,2688)`. Inputs use deterministic small exactly
representable FP8 integers and an exactly representable non-unit combined
scale. The comparison is against the pinned source kernel on the same tensors,
with output canaries and low-level IR rejection of any inline CUDA function
call. Bitwise equality is the first criterion; any dtype-specific fallback
tolerance must be no looser than one output ULP and must be justified from an
observed source/TIRx conversion-order difference.

Performance is a later gate and is not evidence for this sketch. At that gate,
only `bench_suite` reference rows count, and every frozen row must satisfy the
strict ratio `source_time / tirx_time > 0.99`.
