<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 dense block-scaled persistent GEMM + dSReLU quant: coarse WASP pipeline sketch

This is the non-executable execution contract for
[`tirx_kernels/cudnn/dsrelu/dense_blockscaled_gemm_persistent_dsrelu_quant.py`](../../tirx_kernels/cudnn/dsrelu/dense_blockscaled_gemm_persistent_dsrelu_quant.py).
It freezes the seven-warp persistent program, block-scaled A/B mainloop,
read-only C load pipeline, D store pipeline, row-wise `dprob` reduction,
optional FP32 amax, and CTA1/CTA2 TMEM lifetime. After the first reviewer PASS
this file is immutable; the executable module becomes the implementation source
of truth.

The authority is
`python/cudnn/gemm/cutedsl/dense/dsrelu/dense_blockscaled_gemm_persistent_dsrelu_quant.py`
at commit `aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5`, SHA256
`eb2b22e69e74f3637eb8f6e86222c83e9349a6e730b64a0fc1d58ba37a57003f`.
Source citations below refer to that file. The writer export is
`.porting/dense_blockscaled_gemm_persistent_dsrelu_quant/writer_export/anchor_lineinfo.ptx`,
SHA256 `9eb4daafa4699d9642cbfd1570cfc41df75ae68919ae152be00c73d1d135edfd`.
It contains 460 `.loc` records mapping 159 source lines. It was produced by the
normal cache-disabled CuTeDSL compile path, launched three times with nonempty
signed production data, and checked for D, dprob, and amax before admission.

The instruction-annotated anchor specialization is:

| axis | anchor value |
| --- | --- |
| shape | `M=N=256, K=512, L=2` |
| A/B | FP4 E2M1, K-major/K-major |
| SFA/SFB | E8M0, vector size 16 |
| C/D | BF16/BF16, N-major |
| MMA tile / cluster | `(256,256)` / `(2,1)` |
| epilogue | `alpha=2/3`, amax active, SFD absent, `vector_f32=False` |
| stages | accumulator/AB/C/D = `1/5/2/2` |
| launch storage | 224 threads, 512 TMEM columns, 229,376 dynamic-SMEM bytes |

The FP32 operation-boundary contract is:

```text
X[m,n,l] = alpha * blockscaled_gemm(A, B, SFA, SFB)[m,n,l]
R[m,n,l] = max(X[m,n,l], 0)
D_fp32[m,n,l] = 2 * R[m,n,l] * (C[m,n,l] * prob[m,l])
D[m,n,l] = cast(D_fp32[m,n,l], d_dtype)
dprob[m,l] += sum_n(R[m,n,l] * R[m,n,l] * C[m,n,l])
amax = max(amax, max_mnl(abs(D_fp32[m,n,l])))       # only when enabled
```

Operation order is part of the contract: alpha precedes ReLU; C is loaded and
converted independently; the dSReLU helper performs ReLU, doubling, and
`C*prob` multiplication; dprob squares the same FP32 ReLU registers, multiplies
by C, then uses the source scalar ADD reduction order. The global dprob array is
zeroed by the host before launch because different N tiles atomically accumulate
into the same `(m,l)` element.

SFD is excluded: the source branch prints `SFD not implemented` instead of
producing an output. FP8/FP4 D and unproven public-predicate modes are also
excluded. The target imports only `tirx_kernels.kern as K`; device code uses no
tile primitive, first-class layout or fragment, multidimensional shared buffer,
CUDA source/function call, or low-level-IR exemption. Low-level operations are
spelled only through established `K.ptx[...]` forms in the target module.

## Pipeline at a glance

| warp / role | persistent program | ownership edge |
| --- | --- | --- |
| warp 5, AB TMA | prefetch six TensorMaps; issue A/B/SFA/SFB G2S loads for every work item and K tile | AB-empty -> four TMA loads sharing one transaction barrier -> AB-full |
| warp 4, MMA | retrieve TMEM, copy SFA/SFB S2T, issue block-scaled tcgen05 MMA, release AB and publish accumulator | AB-full -> scale copy/MMA -> AB-empty; ACC-empty -> MMA -> ACC-full |
| warp 6, C TMA | follow the same persistent work cursor; issue one C G2S load for each 128x32 subtile | C-empty -> one TMA load/expect-tx -> C-full; producer tail drains all C stages |
| warps 0-3, epilogue | load TMEM, wait/load/release C, compute dSReLU and dprob, optionally reduce amax, stage and TMA-store D | ACC-full + C-full -> FP32 math -> D S2G; C-empty and ACC-empty returned independently |
| cluster prologue/tail | initialize/publish barriers and TMEM pointer; CTA2 peer release precedes deallocation | cluster init fence/arrive/wait; named barrier 2 publishes TMEM; named barrier 1 synchronizes 128 epilogue threads |

Each role constructs the same static persistent scheduler. There is no scheduler
warp and no cross-role scheduler handoff: role-local cursors start at
`block_idx_z`, advance by `grid_dim_z`, and are coupled only by AB, accumulator,
and C pipeline ownership. Source `906-1202`, `1206-1546`; writer `.loc` evidence
at source lines 945-1004, 1084-1207, 1299-1475, and 1522-1545.

## Primitive vocabulary

All helpers return scalar integers, scalar state, raw pointers, or raw encoded
descriptors. They never return a layout, tile, tensor, or fragment object.

```python
specialize(...); launch(...)
ceil_div(x, y); align_up(x, y); integer_offset(...)
byte_ptr(rank1_u8_arena, byte_offset)
pipeline_state(stages, phase, stage, count); advance(state)
encode_tensormap(raw_descriptor, dtype, rank, pointer, integer_fields...)
encode_umma_descriptor(byte_address, ldo, sdo, swizzle, transpose)
tmem_address(base_column, warp_row, stage_column, subtile_column)

copy_g2s(tensormap, integer_coords, smem_byte_ptr, mbarrier, multicast_mask)
copy_s2t(raw_smem_descriptor, tmem_u32_address)
copy_t2r(tmem_u32_address, scalar_registers)
copy_r2s(scalar_registers, smem_byte_ptr)
copy_s2g(smem_byte_ptr, tensormap, integer_coords)
gemm_blockscaled(acc_tmem, a_desc, b_desc, sfa_tmem, sfb_tmem, accumulate)

elect_one(); try_wait(barrier, phase); wait(barrier, phase)
arrive_expect_tx(barrier, bytes); release(barrier, destination)
fence_mbarrier_init_cluster(); cluster_arrive_relaxed(); cluster_wait()
fence_async_shared_cta(); named_barrier(id, thread_count)
tma_commit_group(); tma_wait_group(pending)
alloc_tmem(columns); relinquish_tmem(); dealloc_tmem(columns)
```

## Complete sketch

```python
# ===========================================================================
# 1. Static specialization, raw ABI, TensorMaps, and launch
# source 37-309, 311-600; writer export prologue and source loc 667-672
# ===========================================================================
P = specialize(
    M, N, K, L,
    ab_dtype, sf_dtype, sf_vec_size, c_dtype, d_dtype,
    a_major, b_major, c_major,
    mma_tile_m, mma_tile_n, cluster_m, cluster_n,
    vector_f32, with_amax,
)
assert P in source_capability_manifest.PASS
assert c_major == "n"
assert with_sfd is False

CTA_GROUP = 2 if mma_tile_m == 256 else 1
CTA_M = 128
N_TILE = mma_tile_n                   # 64, 128, 192, 256
K_TILE = 256 if ab_dtype is FP4 else 128
WARPS = 7
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
AB_TMA_WARP = 5
C_TMA_WARP = 6
ACC_STAGES = 1 if N_TILE == 256 else 2
C_STAGES = 2
D_STAGES = 2
AB_STAGES = source_stage_formula(P)
TMEM_COLUMNS = 512

# Fixed raw pointer ABI. amax remains present even in no-amax specializations;
# compile-time with_amax guards every access. C is read-only and D is write-only.
# prob has physical stride (1,M) over (m,l), whereas contiguous source dprob
# has shape (M,1,L), stride (L,L,1), hence address m*L+l.
ABI = (A, B, SFA, SFB, C, D, prob, dprob, amax, alpha)
HOST_DESCRIPTORS = (
    encode_tensormap(A_MAP, A_DTYPE, 3, A, A_FIELDS),
    encode_tensormap(B_MAP, B_DTYPE, 3, B, B_FIELDS),
    encode_tensormap(SFA_MAP, SF_DTYPE, 4, SFA, SFA_FIELDS),
    encode_tensormap(SFB_MAP, SF_DTYPE, 4, SFB, SFB_FIELDS),
    encode_tensormap(C_MAP, C_DTYPE, 3, C, C_FIELDS),
    encode_tensormap(D_MAP, D_DTYPE, 3, D, D_FIELDS),
)

m_tiles = ceil_div(M, CTA_M)
n_tiles = ceil_div(N, N_TILE)
cluster_m_tiles = ceil_div(m_tiles, cluster_m)
cluster_n_tiles = ceil_div(n_tiles, cluster_n)
cluster_work = cluster_m_tiles * cluster_n_tiles * L
resident_clusters = min(cluster_work, source_max_active_clusters(cluster_m * cluster_n))
launch(
    grid=(cluster_m, cluster_n, resident_clusters),
    block=(224, 1, 1),
    cluster=(cluster_m, cluster_n, 1),
    arch="sm_100a",
    dynamic_smem=SHARED_BYTES,
)
# instruction_selection: source 121-130 fixes warps 0-3/4/5/6 and 224
# threads. Anchor export reports dynamic SMEM 229376 and 74 resident clusters.
# Extent: every admitted specialization.

# ===========================================================================
# 2. One rank-1 u8 SMEM arena and explicit address arithmetic
# source 145-309, 471-578, 693-743; writer loc 699/713/726/735
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
AB_STAGES = source_capacity_formula(P, ACC_STAGES, C_STAGES, D_STAGES)

AB_FULL_OFFSET = 0
AB_EMPTY_OFFSET = AB_FULL_OFFSET + AB_STAGES * 8
ACC_FULL_OFFSET = AB_EMPTY_OFFSET + AB_STAGES * 8
ACC_EMPTY_OFFSET = ACC_FULL_OFFSET + ACC_STAGES * 8
C_FULL_OFFSET = ACC_EMPTY_OFFSET + ACC_STAGES * 8
C_EMPTY_OFFSET = C_FULL_OFFSET + C_STAGES * 8
TMEM_DEALLOC_OFFSET = C_EMPTY_OFFSET + C_STAGES * 8
TMEM_PTR_OFFSET = TMEM_DEALLOC_OFFSET + 8

C_OFFSET = 1024
D_OFFSET = C_OFFSET + C_STAGES * C_STAGE_BYTES
A_OFFSET = D_OFFSET + D_STAGES * D_STAGE_BYTES
B_OFFSET = A_OFFSET + AB_STAGES * A_STAGE_BYTES
SFA_OFFSET = B_OFFSET + AB_STAGES * B_STAGE_BYTES
SFB_OFFSET = align_up(SFA_OFFSET + AB_STAGES * SFA_STAGE_BYTES, 1024)
AMAX_OFFSET = align_up(SFB_OFFSET + AB_STAGES * SFB_STAGE_BYTES, 1024)
SHARED_BYTES = align_up(AMAX_OFFSET + 4 * 4, 1024)
assert SHARED_BYTES <= 232448
smem = K.alloc_buffer((SHARED_BYTES,), K.u8, scope="shared.dyn")

# Anchor byte intervals, recovered from source storage order and PTX addresses:
#   AB_FULL 0..39; AB_EMPTY 40..79
#   ACC_FULL 80..87; ACC_EMPTY 88..95
#   C_FULL 96..111; C_EMPTY 112..127
#   TMEM_DEALLOC 128..135; TMEM_PTR 136..139; padding through 1023
#   C 1024..17407; D 17408..33791
#   A 33792..115711; B 115712..197631
#   SFA 197632..207871; SFB 207872..228351
#   amax 228352..228367; arena 229376 bytes.
# instruction_selection: loc 699 emits ten AB mbarrier.init, loc 713 two ACC,
# loc 726 four C plus a CTA sync, and loc 735 one TMEM-dealloc barrier.

# Every access expands to explicit scalar byte arithmetic. Swizzle XORs,
# stage strides, descriptor bits, and TMEM columns never escape as objects.
a_byte = A_OFFSET + stage*A_STAGE_BYTES + explicit_a_swizzle(row, k, P)
b_byte = B_OFFSET + stage*B_STAGE_BYTES + explicit_b_swizzle(n, k, P)
sfa_byte = SFA_OFFSET + stage*SFA_STAGE_BYTES + explicit_sf_offset(m, k, P)
sfb_byte = SFB_OFFSET + stage*SFB_STAGE_BYTES + explicit_sf_offset(n, k, P)
c_byte = C_OFFSET + c_stage*C_STAGE_BYTES + explicit_epi_swizzle(row, col, c_bits)
d_byte = D_OFFSET + d_stage*D_STAGE_BYTES + explicit_epi_swizzle(row, col, d_bits)

a_desc = encode_umma_descriptor(a_byte, A_LDO, A_SDO, A_SWIZZLE, A_TRANSPOSE)
b_desc = encode_umma_descriptor(b_byte, B_LDO, B_SDO, B_SWIZZLE, B_TRANSPOSE)
ACC_COLUMNS = ACC_STAGES * N_TILE
SFA_TMEM_COLUMN = ACC_COLUMNS
SFA_CHUNKS = SFA_STAGE_BYTES // 512
SFB_TMEM_COLUMN = SFA_TMEM_COLUMN + 4*SFA_CHUNKS
SFB_CHUNKS = SFB_STAGE_BYTES // 512
assert SFB_TMEM_COLUMN + 4*SFB_CHUNKS <= TMEM_COLUMNS

def sfb_tmem_shift(tile_n_idx):
    if N_TILE == 192:
        return 2 if tile_n_idx % 2 else 0
    if N_TILE == 64:
        return (tile_n_idx % 2) * 2
    return 0

# ===========================================================================
# 3. Cluster coordinates, barrier construction, and publication
# source 641-903; writer loc 667-735
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

with warp == AB_TMA_WARP:
    prefetch(A_MAP); prefetch(B_MAP); prefetch(SFA_MAP); prefetch(SFB_MAP)
    prefetch(C_MAP); prefetch(D_MAP)
# instruction_selection: source 667-672 maps one-to-one to six static
# `prefetch.tensormap` instructions. Extent: warp 5, all its lanes as source.

AB_EMPTY_ARRIVALS = cluster_n + (cluster_m // CTA_GROUP) - 1
ACC_EMPTY_ARRIVALS = 4 * CTA_GROUP
C_EMPTY_ARRIVALS = 4
with warp == 0:
    with elect_one():
        for s in range(AB_STAGES): mbarrier_init(AB_FULL[s], 1)
    with elect_one():
        for s in range(AB_STAGES): mbarrier_init(AB_EMPTY[s], AB_EMPTY_ARRIVALS)
    with elect_one():
        for s in range(ACC_STAGES): mbarrier_init(ACC_FULL[s], 1)
    with elect_one():
        for s in range(ACC_STAGES): mbarrier_init(ACC_EMPTY[s], ACC_EMPTY_ARRIVALS)
    with elect_one():
        for s in range(C_STAGES): mbarrier_init(C_FULL[s], 1)
    with elect_one():
        for s in range(C_STAGES): mbarrier_init(C_EMPTY[s], C_EMPTY_ARRIVALS)

# PipelineTmaAsync publishes the C barriers to every CTA thread. Both operations
# execute after reconvergence, not inside warp 0.
fence_mbarrier_init_cluster()
cta_sync()
# instruction_selection: source/PTX loc 726 emits
# `fence.mbarrier_init.release.cluster` followed by `bar.sync 0` once each;
# execution extent is all 224 CTA threads after four C-ring initializations.

with warp == 0:
    if CTA_GROUP == 2:
        with elect_one(): mbarrier_init(TMEM_DEALLOC, 32)
if CTA_GROUP == 2:
    fence_mbarrier_init_cluster()
# instruction_selection: source/PTX loc 735 emits the second
# `fence.mbarrier_init.release.cluster` after the CTA2 deallocation barrier;
# the fence executes in all 224 threads and CTA1 omits this allocator edge.

fence_mbarrier_init_cluster()
if cluster_m * cluster_n > 1:
    cluster_arrive_relaxed()
# instruction_selection: source/PTX loc 744 emits the third
# `fence.mbarrier_init.release.cluster` and one
# `barrier.cluster.arrive.relaxed` per CTA, in all-thread SPMD form.

# Source performs the scalar shared/TMEM partition and descriptor setup between
# pipeline_init_arrive and pipeline_init_wait. Those integer-only mappings were
# frozen in section 2; no memory or async instruction is hidden here.
if cluster_m * cluster_n > 1:
    cluster_wait()
else:
    cta_sync()
# instruction_selection: source/PTX loc 901 emits
# `barrier.cluster.wait` for the anchor. The independent CTA1 guard emits one
# `bar.sync 0` at the same source loc 901, executed by all 224 CTA threads.

def scheduler_coords(work):
    cluster_m_idx = work % cluster_m_tiles
    quotient = work // cluster_m_tiles
    cluster_n_idx = quotient % cluster_n_tiles
    batch_idx = quotient // cluster_n_tiles
    tile_m_idx = cluster_m_idx * cluster_m + cluster_x
    tile_n_idx = cluster_n_idx * cluster_n + cluster_y
    return tile_m_idx, tile_n_idx, batch_idx

WORK_STRIDE = grid_dim_z

# ===========================================================================
# 4. Warp 5: persistent A/B/SFA/SFB TMA producer
# source 906-1004; writer loc 945-1004
# ===========================================================================
with warp == AB_TMA_WARP:
    work = block_idx_z
    ab_prod = pipeline_state(AB_STAGES, phase=1, stage=0, count=0)
    while work < cluster_work:
        tile_m, tile_n, batch = scheduler_coords(work)
        k_tiles = ceil_div(K, K_TILE)
        speculative = try_wait(AB_EMPTY[ab_prod.stage], ab_prod.phase)
        for k_count in range(k_tiles):
            wait_if_needed(AB_EMPTY[ab_prod.stage], ab_prod.phase, speculative)
            if leader_cta:
                with elect_one():
                    arrive_expect_tx(AB_FULL[ab_prod.stage], AB_STAGE_BYTES*CTA_GROUP)
            with elect_one():
                copy_g2s(A_MAP, a_coords(tile_m,k_count,batch,cluster_y),
                         byte_ptr(smem,a_stage_piece(ab_prod.stage,cluster_y)),
                         AB_FULL[ab_prod.stage], a_multicast_mask)
            with elect_one():
                copy_g2s(B_MAP, b_coords(tile_n,k_count,batch,cluster_m_group,cta_v),
                         byte_ptr(smem,b_stage_piece(ab_prod.stage,cluster_m_group)),
                         AB_FULL[ab_prod.stage], b_multicast_mask)
            with elect_one():
                copy_g2s(SFA_MAP, sfa_coords(tile_m,k_count,batch,cluster_y),
                         byte_ptr(smem,sfa_stage_piece(ab_prod.stage,cluster_y)),
                         AB_FULL[ab_prod.stage], sfa_multicast_mask)
            with elect_one():
                copy_g2s(SFB_MAP, sfb_coords(tile_n,k_count,batch,cluster_m_group),
                         byte_ptr(smem,sfb_stage_piece(ab_prod.stage,cluster_m_group)),
                         AB_FULL[ab_prod.stage], sfb_multicast_mask)
            advance(ab_prod)
            speculative = try_wait(AB_EMPTY[ab_prod.stage], ab_prod.phase) if k_count+1 < k_tiles else 1
        work += WORK_STRIDE
    for _ in range(AB_STAGES):
        wait(AB_EMPTY[ab_prod.stage], ab_prod.phase); advance(ab_prod)
# instruction_selection: loc 945 is
# `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`; loc 954 contains the
# blocking `mbarrier.try_wait.parity.shared.b64` spin before the copy. Loc 951
# emits one `mbarrier.arrive.expect_tx.shared.b64` with 77,824 anchor bytes per
# K iteration. Loc 954/961 each emit one
# `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2`
# A/B load; loc 968 emits the corresponding 4D SFA form; loc 975 emits the 4D
# SFB form with `.multicast::cluster`. The anchor runtime loop repeats each
# family twice for K=512/K_TILE=256. Loc 987 is the next-stage acquire, and loc
# 1001 emits five ordinary parity waits for the AB producer tail.

# ===========================================================================
# 5. Warp 4: scale S2T copies and persistent block-scaled tcgen05 MMA
# source 1007-1207; writer loc 1007-1207
# ===========================================================================
with warp == MMA_WARP:
    named_barrier(2, 160)
    tmem_base = load_smem_u32(TMEM_PTR)
    work = block_idx_z
    ab_cons = pipeline_state(AB_STAGES, phase=0, stage=0, count=0)
    acc_prod = pipeline_state(ACC_STAGES, phase=1, stage=0, count=0)
    while work < cluster_work:
        tile_m, tile_n, batch = scheduler_coords(work)
        if leader_cta:
            speculative = try_wait(AB_FULL[ab_cons.stage], ab_cons.phase)
            wait(ACC_EMPTY[acc_prod.stage], acc_prod.phase)
        accumulate = False
        for k_count in range(ceil_div(K, K_TILE)):
            if leader_cta:
                wait_if_needed(AB_FULL[ab_cons.stage], ab_cons.phase, speculative)
                for chunk in range(SFA_CHUNKS):
                    with elect_one():
                        copy_s2t(sfa_smem_desc(ab_cons.stage,chunk),
                                 tmem_base + SFA_TMEM_COLUMN + 4*chunk)
                for chunk in range(SFB_CHUNKS):
                    with elect_one():
                        copy_s2t(sfb_smem_desc(ab_cons.stage,chunk),
                                 tmem_base + SFB_TMEM_COLUMN + 4*chunk)
                for kblock in range(4):
                    with elect_one():
                        gemm_blockscaled(
                            tmem_base + acc_prod.stage*N_TILE,
                            a_desc_for(ab_cons.stage,kblock),
                            b_desc_for(ab_cons.stage,kblock),
                            sfa_tmem_addr(tmem_base,kblock),
                            sfb_tmem_addr(tmem_base,kblock,sfb_tmem_shift(tile_n)),
                            accumulate,
                        )
                    accumulate = True
                with elect_one(): release(AB_EMPTY[ab_cons.stage], ab_empty_destination)
            advance(ab_cons)
            if leader_cta and k_count+1 < ceil_div(K,K_TILE):
                speculative = try_wait(AB_FULL[ab_cons.stage], ab_cons.phase)
        if leader_cta:
            with elect_one(): tcgen05_commit(ACC_FULL[acc_prod.stage], acc_full_multicast_mask)
        advance(acc_prod)
        work += WORK_STRIDE
    if leader_cta:
        for _ in range(ACC_STAGES):
            wait(ACC_EMPTY[acc_prod.stage], acc_prod.phase); advance(acc_prod)
# instruction_selection: loc 1007 is named barrier 2; loc 1012 loads the TMEM
# pointer. Loc 1084 is the speculative AB-full
# `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`; loc 1093 is the
# blocking ACC-empty `mbarrier.try_wait.parity.shared.b64`; loc 1124 is the
# blocking AB-full parity wait, and loc 1184 is the next-stage acquire. Loc
# 1133 emits four and loc 1138 eight
# `tcgen05.cp.cta_group::2.32x128b.warpx4`; loc 1165 emits four
# `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16`; loc 1177
# releases AB and loc 1190 commits the accumulator. Loc 1205 emits the anchor's
# one ordinary parity wait for the ACC producer tail (two for non-N256 modes).

# ===========================================================================
# 6. Warp 6: persistent read-only C G2S pipeline
# source 1478-1546; writer loc 1522-1545
# ===========================================================================
with warp == C_TMA_WARP:
    work = block_idx_z
    c_prod = pipeline_state(C_STAGES, phase=1, stage=0, count=0)
    while work < cluster_work:
        tile_m, tile_n, batch = scheduler_coords(work)
        subtile_count = N_TILE // 32
        for subtile in range(subtile_count):
            # overlapping_accum is hard-disabled at source line 307, so the
            # real subtile index is always the forward loop index.
            wait(C_EMPTY[c_prod.stage], c_prod.phase)
            with elect_one():
                arrive_expect_tx(C_FULL[c_prod.stage], C_STAGE_BYTES)
            with elect_one():
                copy_g2s(C_MAP, c_coords(tile_m,tile_n,subtile,batch),
                         byte_ptr(smem,C_OFFSET+c_prod.stage*C_STAGE_BYTES),
                         C_FULL[c_prod.stage], 0)
            advance(c_prod)
        work += WORK_STRIDE
    for _ in range(C_STAGES):
        wait(C_EMPTY[c_prod.stage], c_prod.phase); advance(c_prod)
# instruction_selection: anchor unrolls eight N256 subtiles. Loc 1525 emits
# eight `mbarrier.try_wait.parity.shared.b64` C-empty waits. Loc 1522 emits
# eight `mbarrier.arrive.expect_tx.shared.b64` operations with 8,192 bytes
# each. Loc 1523 emits eight
# `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`
# C loads, with no multicast or CTA-group modifier. Loc 1543 emits the two
# ordinary parity waits in the producer tail. Extent: warp 6, one elected lane
# issues each G2S operation.

# ===========================================================================
# 7. Warps 0-3: C consumption, dSReLU, dprob, optional amax, and D S2G
# source 1206-1456; writer loc 1210-1456 plus helper loc 309 / driver loc 269
# ===========================================================================
with warp in EPI_WARPS:
    if warp == 0:
        alloc_tmem(TMEM_COLUMNS)
    named_barrier(2, 160)
    tmem_base = load_smem_u32(TMEM_PTR)
    work = block_idx_z
    acc_cons = pipeline_state(ACC_STAGES, phase=0, stage=0, count=0)
    c_cons = pipeline_state(C_STAGES, phase=0, stage=0, count=0)
    d_store_count = 0
    while work < cluster_work:
        tile_m, tile_n, batch = scheduler_coords(work)
        wait(ACC_FULL[acc_cons.stage], acc_cons.phase)
        global_row = tile_m*CTA_M + tid
        row_prob = load_global_f32(prob + global_row + batch*M)
        row_dprob = f32(0.0)
        thread_tile_amax = f32(0.0)

        for subtile in range(N_TILE // 32):
            x = copy_t2r(
                tmem_base + (warp << 21) + acc_cons.stage*N_TILE + subtile*32,
                32_scalar_f32_registers,
            )
            for pair in 16_packed_pairs(x):
                pair = mul_f32x2(pair, splat_f32x2(alpha))

            wait(C_FULL[c_cons.stage], c_cons.phase)
            c_raw = copy_s2r_four_v4(
                byte_ptr(smem,c_byte_offset(c_cons.stage,warp,lane,P)))
            c = convert_32_values_to_f32(c_raw, c_dtype)
            fence_async_shared_cta()
            with elect_one(): release(C_EMPTY[c_cons.stage], local_barrier)
            advance(c_cons)

            # Preserve source grouping and packed widths.
            relu_x = 32_scalar_f32_registers
            for i in range(32):
                relu_x[i] = max_f32(x[i], f32(0.0))
            c_prob = 16_packed_pairs
            for pair in range(16):
                c_prob[pair] = mul_f32x2(c.pair[pair], splat_f32x2(row_prob))
            d_fp32 = 16_packed_pairs
            for pair in range(16):
                doubled_relu = add_f32x2(relu_x.pair[pair], relu_x.pair[pair])
                d_fp32[pair] = mul_f32x2(doubled_relu, c_prob[pair])

            # Source reduction is not reassociated: square and C multiply stay
            # packed, then their 32 lanes feed the same scalar ADD chain.
            dprob_terms = 16_packed_pairs
            for pair in range(16):
                squared = mul_f32x2(relu_x.pair[pair], relu_x.pair[pair])
                dprob_terms[pair] = mul_f32x2(squared, c.pair[pair])
            row_dprob += reduce_add_32_source_order(dprob_terms)

            if with_amax:
                abs_d = [abs_f32(d_fp32[i]) for i in range(32)]
                subtile_amax = reduce_max_nan_32(abs_d, instruction_count=17)
                thread_tile_amax = max_f32(thread_tile_amax, subtile_amax)

            d_stage = d_store_count % D_STAGES
            d_out = convert_32_f32(d_fp32, d_dtype)
            copy_r2s_four_v4(d_out,
                byte_ptr(smem,d_byte_offset(d_stage,warp,lane,P)))
            fence_async_shared_cta()
            named_barrier(1, 128)
            if warp == 0:
                copy_s2g(byte_ptr(smem,D_OFFSET+d_stage*D_STAGE_BYTES),
                         D_MAP, d_coords(tile_m,tile_n,subtile,batch))
                tma_commit_group()
                tma_wait_group(D_STAGES - 1)
            named_barrier(1, 128)
            d_store_count += 1

        if with_amax:
            warp_amax = warp_reduce_max_nan(thread_tile_amax)
            if lane == 0: store_smem_f32(AMAX_OFFSET + warp*4, warp_amax)
            named_barrier(1, 128)
            if warp == 0 and lane == 0:
                block_amax = f32(0.0)
                for slot in range(4):
                    block_amax = max_f32(block_amax, load_smem_f32(AMAX_OFFSET+slot*4))
                atomic_max_nonnegative_f32_bits(amax, block_amax)

        # One atomic contribution per output tile and row. Multiple N tiles
        # complete the global sum without a separate reduction kernel.
        atomic_add_f32(dprob + global_row*L + batch, row_dprob)

        with elect_one():
            if CTA_GROUP == 2:
                release(remote(ACC_EMPTY[acc_cons.stage], even_peer_rank), 1)
            else:
                release(ACC_EMPTY[acc_cons.stage], 1)
        advance(acc_cons)
        work += WORK_STRIDE

# instruction_selection: loc 1210 is CTA2 TMEM allocation, 1215 named barrier
# 2, and 1220 pointer load. Loc 1299 waits ACC-full; 1311 loads prob;
# 1318 emits `tcgen05.ld.sync.aligned.32x32b.x32.b32`; 1329 emits 16 packed
# alpha multiplies. Loc 1335 emits four `ld.shared.v4.b32`; 1337 waits C-full;
# 1340 emits `fence.proxy.async.shared::cta`; 1341 releases C-empty with
# `mbarrier.arrive.shared.b64`.
# instruction_selection: driver loc 269 emits 32 scalar ReLU max, 16 packed
# doubling ADDs, and 16 packed D multiplies. Loc 1350 converts 32 BF16 C values
# and emits 16 packed C*prob multiplies. Loc 1356 emits 32 packed square/C
# multiplies and the 33-scalar-add dprob chain. Helper loc 309 emits 32
# `abs.f32` and one `redux.sync.max.NaN.f32`; loc 1372 emits 17
# `max.NaN.f32`, and loc 1377 one running `max.f32`.
# instruction_selection: loc 1389 emits 16 BF16x2 D conversions; 1395 four
# `st.shared.v4.b32`; loc 1401 emits `fence.proxy.async.shared::cta`;
# 1402/1415 emit `bar.sync 1,128`; loc 1407 emits one
# `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint`
# D S2G per static loop body; loc 1413 emits `cp.async.bulk.commit_group` and
# loc 1414 `cp.async.bulk.wait_group.read 1`. Loc 1428/1431/1437/1438/1443 is
# shared store, barrier, one v4 load, four scalar maxima, and global atomic max;
# the warp redux itself is the helper-loc-309 instruction above.
# Source helper loc 309 contains `atom.global.add.f32` for dprob. Loc 1455
# returns accumulator ownership through a cluster barrier arrival.

# ===========================================================================
# 8. TMEM teardown and D store tail
# source 1458-1475; writer loc 1468-1475
# ===========================================================================
with warp in EPI_WARPS:
    if warp == 0:
        relinquish_tmem()
    named_barrier(1, 128)
    if warp == 0:
        if CTA_GROUP == 2:
            peer_rank = cluster_rank ^ 1
            arrive(remote(TMEM_DEALLOC, peer_rank), 1)  # all 32 warp lanes
            wait(TMEM_DEALLOC, phase=0)
        dealloc_tmem(tmem_base, TMEM_COLUMNS)
    tma_wait_group(0)
# instruction_selection: loc 1467 emits
# `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned`; loc 1468 is
# named barrier 1; loc 1469 maps the peer and arrives; loc 1472 waits the
# deallocation barrier; loc 1473 emits `cp.async.bulk.wait_group.read 0` for
# the D tail; loc 1475 emits
# `tcgen05.dealloc.cta_group::2.sync.aligned.b32` for 512 columns.
```

## Pipeline inventory

| pipeline | stages | producer -> consumer | full event | empty/reuse event |
| --- | ---: | --- | --- | --- |
| AB/SF | source formula; anchor 5 | warp 5 -> warp 4 leader | one transaction barrier accounts for A+B+SFA+SFB and CTA-group bytes | tcgen05 multicast commit after scale copies and all K-block MMAs |
| accumulator | N256: 1; otherwise 2 | warp 4 leader -> warps 0-3, one/two CTAs | tcgen05 commit to ACC-full | one elected arrival per epilogue warp; CTA2 maps eight total arrivals to leader storage |
| C load | fixed 2 | warp 6 -> warps 0-3 | expect-tx plus one 128x32 G2S TensorMap load | four elected epilogue-warp arrivals return C-empty; producer tail drains both stages |
| D store | fixed 2 | warps 0-3 -> warp 0 TMA issue | async-shared fence, named barrier 1, TMA commit | wait-group-one before reuse, named barrier 1, final wait-zero tail |
| dprob | per-thread FP32 scalar | each row thread -> global `(m,l)` | exact ordered reduction across each tile's N values | one `atom.global.add.f32` per tile/row; host initializes zero |
| amax | optional scalar + four shared FP32 slots | thread -> warp -> CTA -> global | abs/max tree, warp redux, named barrier 1 | one nonnegative-FP32 bitwise atomic max per output tile |

## TensorMap, tails, and persistent scheduling

- A is logically `(M,K,L)`, B `(N,K,L)`, and the MMA computes A times B
  transposed over K. C and D are `(M,N,L)`. Prob's physical address is
  `m + l*M`; contiguous source dprob has shape/stride `(M,1,L)/(L,L,1)` and
  physical address `m*L+l`.
- TensorMaps use runtime extents and zero-fill A/B/SF/C tail loads. D stores
  retain source-aligned 128x32 TensorMap boxes; admitted tails respect the
  source output-alignment predicate.
- A/SFA multicast across cluster N. B/SFB multicast across the cluster-M group.
  CTA2 uses a 256-row MMA over two 128-row CTAs; only CTA-v 0 issues tcgen05,
  while each CTA consumes/stores its own 128 rows of C/D and dprob.
- N64 divides the SFB TensorMap tile coordinate by two. N64 and odd N192 tiles
  apply the explicit two-word TMEM shift. N192 retains padded SFB storage in AB
  transaction bytes.
- Warp 6 and the epilogue use identical forward subtile order because source
  `overlapping_accum=False`. Changing that order is a pipeline/control-flow
  change requiring correctness re-review.
- M/N/K tails, L=1/2, and long-running CTA1/CTA2 persistent cases were admitted
  only after three consecutive source launches passed D, dprob, and amax.

## Static specialization boundary

Only rows marked PASS in
`.porting/dense_blockscaled_gemm_persistent_dsrelu_quant/source_capability_manifest.json`
may enter `CONFIGS` or `BENCH_CONFIGS`:

- A/B: FP4 E2M1 or same-type FP8 E4M3/E5M2.
- SF: FP4 admits E8M0/V16, E8M0/V32, and E4M3/V16. FP8 admits E8M0/V32.
- C/D: FP16, BF16, or FP32 combinations that the manifest compiled and checked.
- A/B major: FP4 K/K. FP8 admits K/K, K/N, M/K, and M/N. C is N-major.
- MMA M is 128 (CTA1) or 256 (CTA2); N is 64/128/192/256.
- CTA1 cluster axes are in `{1,2,4}` with product at most 16. CTA2 cluster M
  is `{2,4}`, cluster N is `{1,2,4}`, and the same product limit applies.
- L is 1 or 2 in the admitted matrix. Both amax-on and amax-off are source
  validated. The current manifest admits `vector_f32=False`; it does not infer
  the unused source flag's other value.
- Excluded modes include SFD, FP8/FP4 D, the uint8 FP4 alias, FP4 E4/V32,
  FP8 with V16/E4 scales, FP4 non-K major, C-major M, invalid cluster shapes,
  and source-rejected or three-run-oracle-failing alignments.

`CONFIGS` is a deterministic pairwise-token cover of these observed PASS rows,
with explicit anchor, CTA1/CTA2, N64/N128/N192/N256, L1/L2, cluster, major,
dtype, amax, persistent, and M/N/K-tail guards. `BENCH_CONFIGS` is a compact
structure-token cover of validated modes plus square 1024/2048/4096/8192 and
L1/L2 boundaries. The required performance roster is every `BENCH_CONFIGS`
entry, not only YAML defaults.

## Writer PTX evidence and five-way instruction check

The preserved analyzer result is
`.porting/dense_blockscaled_gemm_persistent_dsrelu_quant/writer_export/ptx_evidence.json`.
The anchor's static opcode inventory includes:

| source operation | anchor instruction evidence |
| --- | --- |
| descriptor prefetch | six `prefetch.tensormap` at source 667-672 |
| barriers | 17 `mbarrier.init`, 7 `bar.sync`, 4 acquire try-waits, 22 ordinary waits |
| AB/SF G2S | two 3D and two 4D CTA2 TensorMap loads per static K-loop body |
| scale S2T | twelve `tcgen05.cp.cta_group::2.32x128b.warpx4` |
| MMA | four `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16` |
| accumulator | one `tcgen05.ld.sync.aligned.32x32b.x32.b32` per static subtile body |
| C G2S | eight 3D shared-CTA TensorMap loads and eight expect-tx arrivals |
| dSReLU | 32 ReLU max, 16 packed doubling ADD, packed C/prob and D multiplies |
| dprob | packed square/C multiplies, 33 scalar ADDs, one `atom.global.add.f32` |
| D S2G | four shared v4 stores, one static S2G TensorMap store, wait-group-one |
| amax | 32 abs, 17 max.NaN, warp redux, shared v4 read, `atom.global.max.s32` |
| TMEM lifetime | one CTA2 alloc, two commits, peer arrival/wait, one dealloc |

The reviewer must independently export the same specialization and prove both
directions `source lines <-> sketch operations <-> PTX opcodes`. A structural
statement without a source and PTX edge is insufficient for instruction
selection, and an opcode not assigned to a sketch operation is unexplained.

## Executable module and validation contract

The module exposes `KERNEL_META`, deterministic `CONFIGS`, deterministic
`BENCH_CONFIGS`, `get_kernel`, `prepare_data`, `run_test`, `prepare_bench`,
`run_gpu`, and `run_bench`. `KERNEL_META["name"]` is
`cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant`, category is
`cudnn`, and compute capability is 10.

`prepare_data` emits deterministic positive and negative A/B/C values,
nonconstant positive prob, nontrivial scales, runtime alpha, and separately
allocated D/dprob/amax for TIRx and the pinned source. Source import, compilation,
reset, synchronization, and validation stay outside timing. Each timed closure
issues exactly one target or reference kernel launch.

`run_test` compares D, dprob, and enabled amax to the pinned source and a host
oracle. The source test fixes low-precision output checks at `atol=0.12,
rtol=0.02`; FP32 paths use their separately recorded FP32 tolerance. Every
declared config must pass without unexplained skips.

Structural gates reject `TilePrimitiveCall`, `tirx.tile.*`, tile helpers,
first-class layouts/fragments, non-rank-1 shared buffers, `K.cuda.func_call`,
embedded CUDA, and low-level-IR exemptions. `inspect_low_level_ir(...).ok` must
hold with `func_calls == ()`, while `tirx_kernels/kern/` and the exception table
remain unchanged.
