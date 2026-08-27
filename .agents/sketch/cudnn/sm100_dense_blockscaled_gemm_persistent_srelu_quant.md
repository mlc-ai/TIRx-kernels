<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 dense block-scaled persistent GEMM + sReLU quant: coarse WASP pipeline sketch

This is the non-executable execution sketch for
[`tirx_kernels/cudnn/srelu/dense_blockscaled_gemm_persistent_srelu_quant.py`](../../tirx_kernels/cudnn/srelu/dense_blockscaled_gemm_persistent_srelu_quant.py).
It freezes the six-warp persistent program, the A/B/SFA/SFB block-scaled
mainloop, the independent C and D store pipelines, `relu(C)^2 * prob`, mandatory
FP32 amax, and one-/two-CTA TMEM lifetime. After the first reviewer PASS this
file is immutable; the module becomes the executable source of truth.

The source is
`python/cudnn/gemm/cutedsl/dense/srelu/dense_blockscaled_gemm_persistent_srelu_quant.py`
at commit `7b5327b32907b9dd21d85a393d62f9573d7f0116`, SHA256
`63aaee0a2acdd94c16c261157c9befb1a4265ca514b9f83dec1e5f5679508b4c`.
Source citations below refer to that file. Writer exports are under
`.porting/dense_blockscaled_gemm_persistent_srelu_quant/writer_export/`; the
primary PTX is `ptx/anchor.ptx`, 1,643 lines, SHA256
`4ab8daa4e85649c89e14f7ad76505bab356615d100d35c628f1e184c3aac6bc6`.
Every export was compiled through the public source class, launched three
consecutive times, and checked against a dequantized FP32 oracle before its PTX
was admitted as evidence.

The instruction-annotated body uses this primary specialization:

| axis | anchor value |
| --- | --- |
| shape | `M=N=256, K=512, L=2` |
| A/B | FP4 E2M1, K-major/K-major |
| SFA/SFB | E8M0, vector size 16 |
| C/D | BF16/BF16, N-major |
| MMA tile / cluster | `(256,256)` / `(2,1)` |
| fixed epilogue | `alpha=2/3`, sReLU active, amax active, SFD absent |
| stage tuple | accumulator/AB/C/D = `1/5/2/2` |
| storage | 512 TMEM columns, 229,376 bytes dynamic SMEM |

The mathematical contract is exact at the FP32 operation boundary:

```text
C_fp32 = alpha * (dequant(A,SFA) @ dequant(B,SFB)^T)
C       = cast(C_fp32, c_dtype)
D_fp32 = prob * max(C_fp32, 0)^2
D       = cast(D_fp32, d_dtype)
amax    = max_over_all_elements(abs(D_fp32))       # FP32 global result
```

The source only applies sReLU and `prob` inside `generate_amax`; therefore this
port fixes `generate_amax=True` and `generate_sfd=False`. The SFD branch only
prints `SFD not implemented` and cannot produce valid D. `vector_f32` remains a
public specialization key for compatibility, but source lines 139-140 only
store it and never read it. Independent `true`/`false` anchor exports are
byte-identical.

No first-class layout, tile, or fragment object may appear in device code.
The implementation imports only `tirx_kernels.kern as K`. Shared memory is one
rank-1 `u8` arena. Stages, swizzles, TensorMap coordinates, UMMA descriptors,
register fragments, and TMEM rows/columns are scalar integer expressions.

## Pipeline at a glance

| warp / role | persistent program | ownership edge |
| --- | --- | --- |
| warp 5, TMA | prefetch six TensorMaps; for every output work item and K tile, wait for AB-empty and issue A/B/SFA/SFB TMA loads | AB-empty -> four TMA loads sharing one transaction barrier -> AB-full; producer tail drains every AB stage |
| warp 4, MMA | wait for TMEM publication; leader CTA waits for AB-full and accumulator-empty, copies SFA/SFB SMEM to TMEM, performs block-scaled tcgen05 MMA, releases AB, publishes accumulator | AB-full -> `tcgen05.cp`/MMA -> AB-empty; ACC-empty -> MMA -> ACC-full |
| warps 0-3, epilogue | warp 0 allocates 512 TMEM columns; all four warps load 128x32 subtiles, apply alpha, store C, compute sReLU/prob and amax, store D, then release accumulator | ACC-full -> TMEM load -> independent C/D TMA rings -> ACC-empty; named barrier 1 brackets every SMEM/TMA reuse and the CTA amax reduction |
| CTA/cluster prologue/tail | initialize and publish barriers, derive cluster coordinates and raw descriptors, then synchronize; epilogue relinquishes allocation and frees after all work | cluster init fence/arrive/wait; named barrier 2 publishes TMEM pointer; CTA2 uses peer deallocation arrival/wait before `tcgen05.dealloc` |

All three role-local schedulers start from the same cluster work id and advance
by the persistent grid's number of resident clusters. There is no scheduler
handoff: only the AB and accumulator barriers transfer ownership. Source
`873-1171`, `1173-1459`; anchor PTX `311-1020`, `1023-1636`.

## Primitive vocabulary

Every structural helper below returns an integer, scalar state, raw pointer, or
raw descriptor; none returns a first-class mapping value.

```python
specialize(...); launch(...)
ceil_div(x, y); align_up(x, y); integer_offset(...)
byte_ptr(rank1_u8_arena, byte_offset)
pipeline_state(stages, phase, stage, count)
encode_tensormap(raw_descriptor, dtype, rank, pointer, integer_fields...)
encode_umma_descriptor(smem_byte_address, ldo, sdo, swizzle, transpose)
tmem_address(base_column, warp_row, stage_column, subtile_column)

copy_g2s(tensormap, integer_coords, smem_byte_ptr, mbarrier, multicast_mask)
copy_s2t(raw_smem_descriptor, tmem_u32_address)
copy_t2r(tmem_u32_address, scalar_registers)
copy_r2s(scalar_registers, smem_byte_ptr)
copy_s2g(smem_byte_ptr, tensormap, integer_coords)

gemm_blockscaled(acc_tmem, a_desc, b_desc, sfa_tmem, sfb_tmem, accumulate)
cast(value, dtype); mul(a, b); max(a, b); abs(value)
warp_reduce_max_nan(value); atomic_max_nonnegative_f32_bits(pointer, value)

elect_one(); try_wait(barrier, phase); wait(barrier, phase)
arrive_expect_tx(barrier, bytes); release(barrier, mask)
advance(stage, phase, count); fence_mbarrier_init_cluster()
cluster_arrive_relaxed(); cluster_wait(); fence_async_shared_cta()
named_barrier(id, thread_count); tma_commit_group(); tma_wait_group(pending)
alloc_tmem(columns); relinquish_tmem(); dealloc_tmem(columns)
```

## Complete sketch

```python
# ===========================================================================
# 1. Static specialization, raw ABI, host descriptors, and launch
# source 37-140, 145-299, 301-600; anchor PTX 12-119
# ===========================================================================
P = specialize(
    M, N, K, L,
    ab_dtype, sf_dtype, sf_vec_size, c_dtype, d_dtype,
    a_major, b_major, c_major,
    mma_tile_m, mma_tile_n, cluster_m, cluster_n,
    vector_f32,
)
assert P in source_validated_capability_manifest
assert c_major == "n"                 # M-major produced incorrect D in source
assert generate_amax is True
assert generate_sfd is False

CTA_GROUP = 2 if mma_tile_m == 256 else 1
CTA_M = 128
N_TILE = mma_tile_n                    # 64, 128, 192, or 256
K_TILE = 256 if ab_dtype is FP4 else 128
WARPS = 6
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
ACC_STAGES = 1 if N_TILE == 256 else 2
C_STAGES = 2
D_STAGES = 2
AB_STAGES = source_stage_formula(P)
TMEM_COLUMNS = 512

# Fixed pointer ABI. The host prelude creates raw TensorMaps for A/B/SFA/SFB
# loads and C/D stores, preserving dtype, major, strides, tail fill, tile box,
# and multicast pieces. prob and amax remain ordinary global pointers.
ABI = (A, B, SFA, SFB, C, D, prob, amax, alpha)
HOST_DESCRIPTORS = (
    encode_tensormap(A_MAP, A_DTYPE, 3, A, A_INTEGER_FIELDS),
    encode_tensormap(B_MAP, B_DTYPE, 3, B, B_INTEGER_FIELDS),
    encode_tensormap(SFA_MAP, SF_DTYPE, 4, SFA, SFA_INTEGER_FIELDS),
    encode_tensormap(SFB_MAP, SF_DTYPE, 4, SFB, SFB_INTEGER_FIELDS),
    encode_tensormap(C_MAP, C_DTYPE, 3, C, C_INTEGER_FIELDS),
    encode_tensormap(D_MAP, D_DTYPE, 3, D, D_INTEGER_FIELDS),
)

m_tiles = ceil_div(M, CTA_M)
n_tiles = ceil_div(N, N_TILE)
cluster_m_tiles = ceil_div(m_tiles, cluster_m)
cluster_n_tiles = ceil_div(n_tiles, cluster_n)
cluster_work = cluster_m_tiles * cluster_n_tiles * L
resident_clusters = min(cluster_work, source_max_active_clusters(cluster_m * cluster_n))
launch(
    grid=(cluster_m, cluster_n, resident_clusters),
    block=(192, 1, 1),
    cluster=(cluster_m, cluster_n, 1),
    arch="sm_100a",
    dynamic_smem=SHARED_BYTES,
)
# instruction_selection: source `595-600`; anchor `.reqntid 192,1,1` at PTX
#   `40`, `.extern .shared .align 1024` at `12`. Extent: all specializations.

# ===========================================================================
# 2. One rank-1 SMEM arena and explicit byte/descriptor/TMEM arithmetic
# source 170-299, 471-559, 670-743, 1636-1750; anchor PTX 138-210
# ===========================================================================
ab_bits = dtype_bits(ab_dtype)
c_bits = dtype_bits(c_dtype)
d_bits = dtype_bits(d_dtype)
B_ROWS = N_TILE // CTA_GROUP
A_STAGE_BYTES = CTA_M * K_TILE * ab_bits // 8
B_STAGE_BYTES = B_ROWS * K_TILE * ab_bits // 8
SFA_STAGE_BYTES = CTA_M * K_TILE // sf_vec_size
SFB_STAGE_BYTES = align_up(N_TILE, 128) * K_TILE // sf_vec_size
AB_STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES + SFA_STAGE_BYTES + SFB_STAGE_BYTES
C_STAGE_BYTES = CTA_M * 32 * c_bits // 8
D_STAGE_BYTES = CTA_M * 32 * d_bits // 8
AB_STAGES = (232448 - (1024 + 2*C_STAGE_BYTES + 2*D_STAGE_BYTES + 16)) // AB_STAGE_BYTES

AB_FULL_OFFSET = 0
AB_EMPTY_OFFSET = AB_FULL_OFFSET + AB_STAGES * 8
ACC_FULL_OFFSET = AB_EMPTY_OFFSET + AB_STAGES * 8
ACC_EMPTY_OFFSET = ACC_FULL_OFFSET + ACC_STAGES * 8
TMEM_DEALLOC_OFFSET = ACC_EMPTY_OFFSET + ACC_STAGES * 8
TMEM_PTR_OFFSET = TMEM_DEALLOC_OFFSET + 8
C_OFFSET = 1024
D_OFFSET = C_OFFSET + C_STAGES * C_STAGE_BYTES
A_OFFSET = D_OFFSET + D_STAGES * D_STAGE_BYTES
B_OFFSET = A_OFFSET + AB_STAGES * A_STAGE_BYTES
SFA_OFFSET = B_OFFSET + AB_STAGES * B_STAGE_BYTES
SFB_OFFSET = align_up(SFA_OFFSET + AB_STAGES * SFA_STAGE_BYTES, 1024)
AMAX_OFFSET = align_up(SFB_OFFSET + AB_STAGES * SFB_STAGE_BYTES, 1024)
SHARED_BYTES = align_up(AMAX_OFFSET + 16, 1024)
assert SHARED_BYTES <= 232448
smem = K.alloc_buffer((SHARED_BYTES,), K.u8, scope="shared.dyn")

# Anchor byte intervals, directly observed in anchor TMA addresses:
#   AB_FULL 0..39; AB_EMPTY 40..79; ACC_FULL 80..87; ACC_EMPTY 88..95
#   TMEM_DEALLOC 96..103; TMEM_PTR 104..107; protocol padding to 1024
#   C    1024..17407      (2 * 8192)
#   D   17408..33791      (2 * 8192)
#   A   33792..115711     (5 * 16384)
#   B  115712..197631     (5 * 16384)
#   SFA 197632..207871    (5 * 2048)
#   SFB 207872..228351    (5 * 4096)
#   amax 228352..228367; whole arena rounds to 229376 bytes.
# instruction_selection: anchor A/B/SFA/SFB TMA destinations are PTX
#   `383`, `392`, `408`, `423`; C/D store sources are `1300`, `1497`; amax
#   shared writes/reads are `1525`, `1535-1547`.

# All accessors below expand to scalar byte arithmetic, including stage and
# swizzle XORs. They are functions of integer row/column/major/dtype fields,
# never stored or passed as first-class layouts.
a_byte = A_OFFSET + stage*A_STAGE_BYTES + explicit_a_swizzle(row, k, P)
b_byte = B_OFFSET + stage*B_STAGE_BYTES + explicit_b_swizzle(n, k, P)
sfa_byte = SFA_OFFSET + stage*SFA_STAGE_BYTES + explicit_sf_offset(m, k, P)
sfb_byte = SFB_OFFSET + stage*SFB_STAGE_BYTES + explicit_sf_offset(n, k, P)
c_byte = C_OFFSET + c_stage*C_STAGE_BYTES + explicit_epi_swizzle(row, col, c_bits)
d_byte = D_OFFSET + d_stage*D_STAGE_BYTES + explicit_epi_swizzle(row, col, d_bits)

# Raw 64-bit UMMA descriptors encode the shared byte base/16, leading offset,
# stride offset, transpose bit, and swizzle enum. K-major FP4 anchor uses
# `ldo=1, sdo=64, swizzle=3`; MN-major variants select the source's 32/64/128B
# swizzle integers and split copies without constructing a mapping value.
a_desc = encode_umma_descriptor(a_byte, A_LDO, A_SDO, A_SWIZZLE, A_TRANSPOSE)
b_desc = encode_umma_descriptor(b_byte, B_LDO, B_SDO, B_SWIZZLE, B_TRANSPOSE)
sf_desc = encode_umma_descriptor(sfa_byte, 1, 8, 0, 0)

ACC_COLUMNS = ACC_STAGES * N_TILE
SFA_TMEM_COLUMN = ACC_COLUMNS
SFA_CHUNKS = SFA_STAGE_BYTES // 512
SFB_TMEM_COLUMN = SFA_TMEM_COLUMN + SFA_CHUNKS * 4
SFB_CHUNKS = SFB_STAGE_BYTES // 512
assert SFB_TMEM_COLUMN + SFB_CHUNKS * 4 <= 512

def sfb_tmem_shift(tile_n_idx):
    # Source `1061-1077`: N192 odd tiles and N64 alternating half-tiles skip
    # the first 64 scale columns by shifting two TMEM words.
    if N_TILE == 192:
        return 2 if tile_n_idx % 2 else 0
    if N_TILE == 64:
        return (tile_n_idx % 2) * 2
    return 0

# ===========================================================================
# 3. Cluster coordinates, barriers, descriptor prefetch, and publication
# source 641-870; anchor PTX 119-274
# ===========================================================================
tid = thread_idx_x
warp = uniform(tid >> 5)
lane = tid & 31
cluster_rank = block_rank_in_cluster
cluster_x = cluster_rank % cluster_m
cluster_y = cluster_rank // cluster_m
cta_v = cluster_x % CTA_GROUP
cluster_m_group = cluster_x // CTA_GROUP
leader_cta = (cta_v == 0)

with warp == TMA_WARP:
    prefetch(A_MAP); prefetch(B_MAP); prefetch(SFA_MAP); prefetch(SFB_MAP)
    prefetch(C_MAP); prefetch(D_MAP)
# instruction_selection: source `647-653`; six `prefetch.tensormap` operations
#   begin at anchor PTX `104`. Extent: all 32 lanes of warp 5; there is no
#   `elect.sync` predicate around descriptor prefetch.

AB_EMPTY_ARRIVALS = cluster_n + (cluster_m // CTA_GROUP) - 1
ACC_EMPTY_ARRIVALS = 4 * CTA_GROUP
with warp == 0:
    # Each constructor performs its own election. Source construction order is
    # also the physical barrier/election order.
    with elect_one():
        for s in range(AB_STAGES):
            mbarrier_init(AB_FULL[s], 1)
    with elect_one():
        for s in range(AB_STAGES):
            mbarrier_init(AB_EMPTY[s], AB_EMPTY_ARRIVALS)
    with elect_one():
        for s in range(ACC_STAGES):
            mbarrier_init(ACC_FULL[s], 1)
    with elect_one():
        for s in range(ACC_STAGES):
            mbarrier_init(ACC_EMPTY[s], ACC_EMPTY_ARRIVALS)
    if CTA_GROUP == 2:
        with elect_one():
            mbarrier_init(TMEM_DEALLOC, 32)
if CTA_GROUP == 2:
    # TmemAllocator publishes its two-CTA deallocation barrier independently.
    fence_mbarrier_init_cluster()
# `pipeline_init_arrive` always supplies the second publication fence; the
# AB/ACC constructors use `defer_sync=True` and emit no construction fence.
fence_mbarrier_init_cluster()
if cluster_m * cluster_n > 1:
    cluster_arrive_relaxed()
    cluster_wait()
else:
    cta_sync()
# instruction_selection: anchor PTX `145-163`, `177`, `190`, `203` contains
#   the 5/5/1/1/dealloc `mbarrier.init`; publication and cluster arrival are
#   `208-211`. CTA1 evidence is in `cta1_n64.ptx`.

def scheduler_coords(work):
    # Default source scheduler is M-raster: cluster M changes fastest.
    cluster_m_idx = work % cluster_m_tiles
    quotient = work // cluster_m_tiles
    cluster_n_idx = quotient % cluster_n_tiles
    batch_idx = quotient // cluster_n_tiles
    tile_m_idx = cluster_m_idx * cluster_m + cluster_x
    tile_n_idx = cluster_n_idx * cluster_n + cluster_y
    return tile_m_idx, tile_n_idx, batch_idx

work = block_idx_z
WORK_STRIDE = grid_dim_z

# ===========================================================================
# 4. Warp 5: persistent A/B/SFA/SFB TMA producer
# source 873-967; anchor PTX 311-594
# ===========================================================================
with warp == TMA_WARP:
    ab_prod = pipeline_state(AB_STAGES, phase=1, stage=0, count=0)
    while work < cluster_work:
        tile_m, tile_n, batch = scheduler_coords(work)
        k_count = 0
        speculative = try_wait(AB_EMPTY[ab_prod.stage], ab_prod.phase)
        while k_count < ceil_div(K, K_TILE):
            wait_if_needed(AB_EMPTY[ab_prod.stage], ab_prod.phase, speculative)
            # The pipeline arrival and four copy atoms each perform their own
            # source election. They are five distinct synchronization points.
            if leader_cta:
                with elect_one():
                    arrive_expect_tx(AB_FULL[ab_prod.stage], AB_STAGE_BYTES * CTA_GROUP)

            # Integer coordinates include cluster pieces and zero-fill tails.
            # A/SFA multicast across cluster N; B/SFB multicast across the
            # cluster-M group. CTA2 addresses the leader's cluster SMEM.
            with elect_one():
                copy_g2s(A_MAP, a_coords(tile_m,k_count,batch,cluster_y),
                         byte_ptr(smem, a_stage_piece(ab_prod.stage,cluster_y)),
                         AB_FULL[ab_prod.stage], a_multicast_mask)
            with elect_one():
                copy_g2s(B_MAP, b_coords(tile_n,k_count,batch,cluster_m_group,cta_v),
                         byte_ptr(smem, b_stage_piece(ab_prod.stage,cluster_m_group)),
                         AB_FULL[ab_prod.stage], b_multicast_mask)
            with elect_one():
                copy_g2s(SFA_MAP, sfa_coords(tile_m,k_count,batch,cluster_y),
                         byte_ptr(smem, sfa_stage_piece(ab_prod.stage,cluster_y)),
                         AB_FULL[ab_prod.stage], sfa_multicast_mask)
            with elect_one():
                copy_g2s(SFB_MAP, sfb_coords(tile_n,k_count,batch,cluster_m_group),
                         byte_ptr(smem, sfb_stage_piece(ab_prod.stage,cluster_m_group)),
                         AB_FULL[ab_prod.stage], sfb_multicast_mask)
            advance(ab_prod)
            k_count += 1
            speculative = try_wait(AB_EMPTY[ab_prod.stage], ab_prod.phase) if k_count < k_tiles else 1
        work += WORK_STRIDE

    # Producer tail: revisit each AB stage and wait until MMA has released it.
    for _ in range(AB_STAGES):
        wait(AB_EMPTY[ab_prod.stage], ab_prod.phase)
        advance(ab_prod)
# instruction_selection: source `910-967`; anchor PTX `321`, `349`, `368`
#   implement try/wait/expect-tx. Anchor TMA instructions are 3D CTA-group-2
#   A/B at `383`, `392` and 4D SFA/SFB at `408`, `423`. CTA1, multicast,
#   FP8, and alternate-major opcodes are covered by the writer variants.

# ===========================================================================
# 5. Warp 4: persistent SMEM->TMEM scale copies and block-scaled tcgen05 MMA
# source 970-1171; anchor PTX 603-1020
# ===========================================================================
with warp == MMA_WARP:
    named_barrier(2, 160)               # wait for TMEM pointer publication
    tmem_base = load_smem_u32(TMEM_PTR)
    ab_cons = pipeline_state(AB_STAGES, phase=0, stage=0, count=0)
    acc_prod = pipeline_state(ACC_STAGES, phase=1, stage=0, count=0)
    work = block_idx_z
    while work < cluster_work:
        tile_m, tile_n, batch = scheduler_coords(work)
        if leader_cta:
            speculative = try_wait(AB_FULL[ab_cons.stage], ab_cons.phase)
            wait(ACC_EMPTY[acc_prod.stage], acc_prod.phase)
        accumulate = False
        for k_count in range(ceil_div(K, K_TILE)):
            if leader_cta:
                wait_if_needed(AB_FULL[ab_cons.stage], ab_cons.phase, speculative)

                # Copy compact scale blocks. Each tcgen05.cp consumes one raw
                # SMEM descriptor and one scalar TMEM address.
                for chunk in range(SFA_CHUNKS):
                    with elect_one():
                        copy_s2t(sf_smem_desc(SFA_OFFSET, ab_cons.stage, chunk),
                                 tmem_base + SFA_TMEM_COLUMN + 4*chunk)
                for chunk in range(SFB_CHUNKS):
                    with elect_one():
                        copy_s2t(sf_smem_desc(SFB_OFFSET, ab_cons.stage, chunk),
                                 tmem_base + SFB_TMEM_COLUMN + 4*chunk)

                for kblock in range(4):
                    sfa_addr = integer_sfa_tmem_address(tmem_base, kblock)
                    sfb_addr = integer_sfb_tmem_address(
                        tmem_base, kblock, sfb_tmem_shift(tile_n)
                    )
                    with elect_one():
                        gemm_blockscaled(
                            acc=tmem_base + acc_prod.stage*N_TILE,
                            a=a_desc_for_stage_kblock(ab_cons.stage,kblock),
                            b=b_desc_for_stage_kblock(ab_cons.stage,kblock),
                            sfa=sfa_addr,
                            sfb=sfb_addr,
                            accumulate=accumulate,
                        )
                    accumulate = True
                with elect_one():
                    release(AB_EMPTY[ab_cons.stage], ab_empty_multicast_mask)
            advance(ab_cons)
            if k_count + 1 < k_tiles and leader_cta:
                speculative = try_wait(AB_FULL[ab_cons.stage], ab_cons.phase)

        if leader_cta:
            with elect_one():
                tcgen05_commit(ACC_FULL[acc_prod.stage], acc_full_multicast_mask)
        advance(acc_prod)
        work += WORK_STRIDE

    if leader_cta:
        for _ in range(ACC_STAGES):
            wait(ACC_EMPTY[acc_prod.stage], acc_prod.phase)
            advance(acc_prod)
# instruction_selection: source `1092-1146`; anchor PTX has twelve CTA2
#   `tcgen05.cp.cta_group::2.32x128b.warpx4` operations at `768-861`, four
#   `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16` at
#   `884-938`, AB release commit at `949`, and accumulator commit at `986`.
#   CTA1 and MXF8 instruction forms are in `cta1_n64.ptx` and
#   `cta1_n128_f8_f32_f16_c4x4.ptx`.

# ===========================================================================
# 6. Warps 0-3: C store, FP32 sReLU/prob/amax, and independent D store
# source 1173-1447; anchor PTX 1023-1595
# ===========================================================================
with warp in EPI_WARPS:
    if warp == 0:
        alloc_tmem(TMEM_COLUMNS)
    named_barrier(2, 160)
    tmem_base = load_smem_u32(TMEM_PTR)
    acc_cons = pipeline_state(ACC_STAGES, phase=0, stage=0, count=0)
    c_store_count = 0
    d_store_count = 0
    work = block_idx_z
    while work < cluster_work:
        tile_m, tile_n, batch = scheduler_coords(work)
        wait(ACC_FULL[acc_cons.stage], acc_cons.phase)
        thread_tile_amax = f32(0.0)

        # Each epilogue warp owns one 32-row slice of every 128x32 subtile.
        # `prob` is one FP32 per output row and batch, reused across N subtiles.
        global_row = tile_m * CTA_M + tid
        row_prob = load_global_f32(prob + global_row + batch*M)
        for subtile in range(N_TILE // 32):
            regs = copy_t2r(
                tmem_base + (warp << 21) + acc_cons.stage*N_TILE + subtile*32,
                scalar_f32_registers_for_32x32(),
            )
            # Source order and issue widths are mandatory. The x32 TMEM load
            # yields 32 scalar FP32 registers; alpha is applied as 16 packed
            # FP32-pair multiplies before C conversion.
            for pair in packed_f32_pairs(regs, count=16):
                pair = mul_f32x2(pair, splat_f32x2(alpha))

            c_stage = c_store_count % C_STAGES
            c_regs = convert_32_f32(regs, c_dtype)       # distinct conversion
            copy_r2s(c_regs,                             # four vector SMEM stores
                     byte_ptr(smem, c_byte_offset(c_stage,warp,lane,subtile,P)))
            fence_async_shared_cta()
            named_barrier(1, 128)
            if warp == 0:
                copy_s2g(byte_ptr(smem, C_OFFSET + c_stage*C_STAGE_BYTES),
                         C_MAP, c_coords(tile_m,tile_n,subtile,batch))
                tma_commit_group()
                tma_wait_group(C_STAGES - 1)
            named_barrier(1, 128)
            c_store_count += 1

            # Active source branch, entirely FP32 and independent of rounded C:
            # 32 scalar ReLU instructions, 16 packed squares, then 16 packed
            # probability multiplies. Do not fold these instruction families.
            relu_regs = scalar_f32_registers_for_32x32()
            for i in range(32):
                relu_regs[i] = max_f32(regs[i], f32(0.0))
            d_regs = scalar_f32_registers_for_32x32()
            for pair in range(16):
                d_regs.pair[pair] = mul_f32x2(relu_regs.pair[pair], relu_regs.pair[pair])
            for pair in range(16):
                d_regs.pair[pair] = mul_f32x2(d_regs.pair[pair], splat_f32x2(row_prob))

            abs_d_regs = scalar_f32_registers_for_32x32()
            for i in range(32):
                abs_d_regs[i] = abs_f32(d_regs[i])
            # The source vector reduction lowers to a fixed 17-instruction
            # max.NaN tree over these 32 values, then one ordinary running fmax.
            subtile_amax = reduce_max_nan_32(abs_d_regs, instruction_count=17)
            thread_tile_amax = max_f32(thread_tile_amax, subtile_amax)

            d_stage = d_store_count % D_STAGES
            d_output_regs = convert_32_f32(d_regs, d_dtype)  # distinct conversion
            copy_r2s(d_output_regs,                          # four vector SMEM stores
                     byte_ptr(smem, d_byte_offset(d_stage,warp,lane,subtile,P)))
            fence_async_shared_cta()
            named_barrier(1, 128)
            if warp == 0:
                copy_s2g(byte_ptr(smem, D_OFFSET + d_stage*D_STAGE_BYTES),
                         D_MAP, d_coords(tile_m,tile_n,subtile,batch))
                tma_commit_group()
                tma_wait_group(D_STAGES - 1)
            named_barrier(1, 128)
            d_store_count += 1

        warp_amax = warp_reduce_max_nan(thread_tile_amax)
        if lane == 0:
            store_smem_f32(AMAX_OFFSET + warp*4, warp_amax)
        named_barrier(1, 128)
        if warp == 0 and lane == 0:
            block_amax = f32(0.0)
            for slot in range(4):
                block_amax = max(block_amax, load_smem_f32(AMAX_OFFSET + slot*4))
            atomic_max_nonnegative_f32_bits(amax, block_amax)

        with elect_one():
            if CTA_GROUP == 2:
                # AsyncThread arrival targets the even leader CTA's physical
                # ACC-empty barrier; this is a rank mapping, not multicast.
                acc_empty_destination_rank = (cluster_rank // 2) * 2
                arrive(remote(ACC_EMPTY[acc_cons.stage], acc_empty_destination_rank), 1)
            else:
                arrive(ACC_EMPTY[acc_cons.stage], 1)
        advance(acc_cons)
        work += WORK_STRIDE

# instruction_selection: source `1298-1305`; anchor PTX loads prob at `1191-1197`
#   and accumulator with `tcgen05.ld.sync.aligned.32x32b.x32.b32` at `1205`.
# instruction_selection: source `1315-1349`; anchor PTX applies alpha with
#   packed FP32 multiplies `1223-1238`, converts C `1245-1276`, uses
#   `st.shared.v4.b32` `1282-1288`, `fence.proxy.async.shared::cta`/barrier
#   `1290-1292`, and C TMA commit/wait-one `1300-1304`.
# instruction_selection: source `1354-1368`; anchor PTX uses `max.f32` ReLU
#   `1309-1356`, `mul.f32x2` square `1357-1372`, prob multiply `1376-1391`,
#   `abs.f32` `1394-1440`, and `max.NaN.f32` `1442-1458`.
# instruction_selection: source `1379-1406`; anchor PTX converts D at
#   `1462-1477`, stores it at `1481-1487`, and issues the distinct D TMA
#   commit/wait-one at `1497-1501` after its own fence/barriers.
# instruction_selection: source `1408-1434`; anchor PTX emits
#   `redux.sync.max.NaN.f32` at `1517`, four shared values `1525-1547`, and
#   `atom.global.max.s32` at `1551`. Because values are nonnegative FP32,
#   signed integer bit ordering implements atomic fmax without a CAS loop.

# ===========================================================================
# 7. TMEM teardown and store tails
# source 1449-1459; anchor PTX 1596-1636
# ===========================================================================
with warp in EPI_WARPS:
    if warp == 0:
        relinquish_tmem()
    named_barrier(1, 128)
    if warp == 0:
        if CTA_GROUP == 2:
            peer_rank = cluster_rank ^ 1
            # SPMD extent: each of warp 0's 32 lanes sends exactly one arrival.
            arrive(remote(TMEM_DEALLOC, peer_rank), 1)
            wait(TMEM_DEALLOC, phase=0)
        dealloc_tmem(tmem_base, TMEM_COLUMNS)
    # After the warp-0 free branch rejoins, all 128 epilogue threads execute
    # both independent store tails in source order.
    tma_wait_group(0)                    # C pipeline tail
    tma_wait_group(0)                    # D pipeline tail
# instruction_selection: source `1452-1459`; anchor PTX emits CTA2 relinquish
#   `1602`, 128-thread barrier `1607`, peer `mapa`/arrival `1611-1614`, wait
#   `1621`, 512-column deallocation `1631`, and two independent wait-zero tails
#   `1634`, `1636`. CTA1 omits only the peer protocol.
```

## Pipeline inventory

| pipeline | stages | producer -> consumer | full event | empty/reuse event |
| --- | ---: | --- | --- | --- |
| AB/SF | source formula; anchor 5 | warp 5 -> warp 4 leader CTA | one transaction mbarrier accounts for A+B+SFA+SFB and CTA group bytes | tcgen05 multicast commit after all scale copies and four K-block MMAs |
| accumulator | 1 for N256, otherwise 2 | warp 4 leader -> warps 0-3 in one or two CTAs | tcgen05 commit to ACC_FULL | one elected arrival per epilogue warp; CTA1 targets local ACC-empty, CTA2 maps all eight arrivals to the even leader CTA's count-8 barrier |
| C store | fixed 2 | warps 0-3 -> warp 0 TMA issue | async shared fence + named barrier 1 + TMA commit | `wait_group.read 1` before ring reuse, final `read 0` |
| D store | fixed 2 | warps 0-3 -> warp 0 TMA issue | independent async shared fence + named barrier 1 + TMA commit | independent `wait_group.read 1`, final `read 0` |
| amax | tile-local scalar + 4 shared FP32 slots | thread -> warp -> CTA -> global | scalar max, `redux.sync.max.NaN.f32`, named barrier 1 | one nonnegative FP32 bitwise atomic max per output tile |

The source stage formula is part of the specialization and SMEM ABI. The
validated capability matrix observed twelve stage/storage tuples, including
anchor `1/5/2/2`, N64 `2/6/2/2`, N192 `2/5/2/2`, FP8 N128
`2/5/2/2`, FP8 N256 `1/4/2/2`, and FP4 V32 N128 `2/7/2/2`.

## TensorMap, tail, and persistent-scheduler contract

- A is logically `(M,K,L)`, B `(N,K,L)`, and the MMA computes A times B
  transposed over K. SFA/SFB use the source block-scale atom packing.
- Host TensorMaps carry the exact runtime extents and zero-fill out-of-bounds
  A/B/SF loads. C/D stores are 128x32 subtiles and keep predicate/tail
  semantics in the raw descriptor coordinates.
- The N64 SFB TensorMap divides the tile coordinate by two; N64 and N192 then
  apply the explicit two-word TMEM shift shown above. N192's padded 256-column
  SFB storage participates in AB transaction bytes.
- CTA2 uses a 256-row MMA instruction over two 128-row CTAs. Only CTA-v 0 owns
  tcgen05 scale-copy/MMA/commit instructions; output coordinates remain per
  128-row CTA. Cluster M must therefore be divisible by two.
- The role-local work cursor is persistent and advances by resident cluster
  count. Tail shapes and `L=2` were admitted only after three consecutive
  source launches matched C, D, and amax.

## Static specialization boundary

Only capability-manifest PASS modes may enter `CONFIGS` or `BENCH_CONFIGS`:

- A/B: FP4 E2M1 or same-type FP8 E4M3/E5M2.
- SF: FP4 accepts E8M0/V16, E8M0/V32, or E4M3/V16; FP8 accepts E8M0/V32.
- C: FP16, BF16, FP32, FP8 E4M3, or FP8 E5M2. D: FP16, BF16, or FP32.
- A/B major: FP4 K/K; FP8 A K/M and B K/N. C and D are N-major only.
- MMA M: 128 (CTA1) or 256 (CTA2); N: 64/128/192/256.
- CTA1 clusters use axes in `{1,2,4}`. CTA2 cluster M is `{2,4}` and cluster N
  is `{1,2,4}`. Cluster axes and product still obey the source public checks.
- L is 1 or 2 in the port matrix. `vector_f32` accepts both values but does not
  change generated code; benchmark coverage keeps one representative.
- amax is always active and FP32; SFD, D=FP8, D=FP4, the `uint8` FP4 alias, and
  source-rejected or three-run-oracle-failing combinations are excluded.

The single parameterized kernel must preserve the source stage formula,
descriptor form, instruction family, store conversion path, and CTA topology
for every admitted mode. `CONFIGS` will be the deterministic pairwise-token
cover plus CTA1/CTA2, N64/N192, L2, M/N/K tails, stage, vector flag, and cluster
boundary guards. Performance representatives are a separate deterministic
minimal cover.

## Writer PTX evidence

`analyze_writer_exports.py` tokenized each `.loc file:line | PTX opcode` pair.
Greedy selection maximized newly covered branch tokens and broke ties by label;
reverse redundancy pruning retained seven exports covering all 581 tokens.

| label | tile / cluster | dtype and output path | stages | PTX SHA256 |
| --- | --- | --- | --- | --- |
| `anchor` | 256x256 / 2x1 | FP4 E8V16, BF16/BF16, K/K/N | 1/5/2/2 | `4ab8daa4e85649c89e14f7ad76505bab356615d100d35c628f1e184c3aac6bc6` |
| `cta1_n64` | 128x64 / 1x1 | FP4 E8V16, BF16/BF16, tail | 2/6/2/2 | `353615e53ade976ad302cf75696a5634d857e135a22407c6ef4a791440fad5bb` |
| `cta2_n128_f4v32_e4_f32_c4x2` | 256x128 / 4x2 | FP4 E8V32, FP8E4/FP32 | 2/7/2/2 | `249bde96666be193f61083cb933f4e6c5083d607c49609fb83877bc00d392d56` |
| `cta2_n192` | 256x192 / 2x1 | FP4 E8V16, BF16/BF16, tail | 2/5/2/2 | `97275a36a62d03cc5580c2a4824cee237fa95fc9fcd76835f5dfebdffef13429` |
| `cta1_n128_f8_f32_f16_c4x4` | 128x128 / 4x4 | FP8E5 E8V32, FP32/FP16, M/N/N | 2/5/2/2 | `4d3e40a0ac7bd491af7723c6b01bad906283bbd2e0b08ecadbd3b917ab853648` |
| `cta1_n256_f8_e5_bf16_c1x4` | 128x256 / 1x4 | FP8E4 E8V32, FP8E5/BF16 | 1/4/2/2 | `fbe0c7e62667bf6b1264f5fa1c67890d0721d99a7e87c9fccb2dcaf43f3613eb` |
| `cta2_n256_f4e4_f16_bf16_c2x4` | 256x256 / 2x4 | FP4 E4V16, FP16/BF16 | 1/5/2/2 | `0ff652711857f2be85cb3cc1bc9460a297a3ac1d52313d6f563950d2185f9fc6` |

`anchor_scalar` is a separately compiled and validated `vector_f32=False`
specialization. Its SHA256 equals `anchor` exactly, so it adds no PTX branch
token. Raw results, generated intermediates, and the complete token ledger are
preserved beneath `writer_export/`.

## Executable module and validation contract

The executable module exposes `KERNEL_META`, deterministic `CONFIGS`,
deterministic `BENCH_CONFIGS`, `get_kernel`, `prepare_data`, `run_test`,
`prepare_bench`, `run_gpu`, and `run_bench`. It must use a lazy importlib source
reference, not `cudnn._compiled_module`.

`prepare_data` generates deterministic signed FP4/FP8 patterns, nontrivial
per-row `prob`, `alpha=2/3`, and representable K-dependent scales. TIRx and
source get identical logical inputs but separate C/D/amax outputs. `run_test`
checks TIRx against the independently compiled source and both against the
dequantized FP32 oracle for C, D, and mandatory FP32 amax. FP32 uses
`atol=rtol=1e-4`; FP16/BF16 and amax use `atol=0.12, rtol=0.02`; FP8 C compares
against an oracle converted to the same FP8 dtype.

`prepare_bench` compiles TIRx. The GPU child lazily builds the source reference,
keeps source JIT, allocation, validation, reset, and synchronization outside
the timed closure, and registers it as `cudnn_frontend`. Every timed closure
contains exactly one launch and forwards timer/warmup/repeat/rounds/cooldown to
the central benchmark API unchanged.

Structural gates reject `TilePrimitiveCall`, `tirx.tile.*`, `tile_primitive`,
any first-class layout, non-rank-1 shared buffers, or imports other than the
normal Python/runtime dependencies plus `tirx_kernels.kern as K` for device
construction. Generated PTX must retain the instruction-selection and protocol
edges cited above.
