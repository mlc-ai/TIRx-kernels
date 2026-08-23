<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 dense block-scaled persistent GEMM + amax: coarse WASP pipeline sketch

This is a non-executable execution sketch. It freezes the storage, six-warp
role split, persistent scheduling, asynchronous protocols, tile dataflow, C
epilogue, and FP32 amax path for the single parameterized TIRx module
[`tirx_kernels/cudnn/amax/dense_blockscaled_gemm_persistent_amax.py`](../../tirx_kernels/cudnn/amax/dense_blockscaled_gemm_persistent_amax.py).
After the reviewer gate that module is the executable source of truth; this
sketch remains frozen.

The source is
`python/cudnn/gemm/cutedsl/dense/amax/dense_blockscaled_gemm_persistent_amax.py`
at commit `7b5327b32907b9dd21d85a393d62f9573d7f0116`, SHA256
`637088ed4fb7db391d7e9325b3e9f952265ff206ebfc96aed0cdfd1d3365d7f1`.
Source-line citations below are to that file. The primary evidence is the
writer's exact line-info export under
`.porting/dense_blockscaled_gemm_persistent_amax/writer_source_export/`, PTX
SHA256 `c3bf02ae4c6affaff3f9aef132eff81815a4b6fbb58bec3b2f3fff40eb87d6bb`.
`PTX n` means the numbered line in that 1,348-line artifact.

The instruction-annotated body is the fixed primary specialization:

| axis | anchor value |
| --- | --- |
| problem | `M=N=K=1024`, `L=1` |
| A/B | FP4 E2M1, K-major/K-major |
| scale | E8M0, vector size 16 |
| C | FP16, N-major |
| MMA tile / cluster | `(128,128)` / `(2,1)` |
| fixed | FP32 accumulator, M tile 128, overlap margin 0 |

The same sketch owns every accepted compile-time branch listed in **Static
specialization boundary**. Those branches alter shapes, descriptor constants,
conversion families, pipeline depths, and predicates; they do not add a role,
kernel, or workflow. M tile 256/two-CTA MMA and the non-working public `uint8`
FP4 alias are out of scope.

First-class layouts are forbidden throughout both this sketch and the device
kernel. No layout object is constructed, passed, returned, stored on a buffer,
or manipulated by layout algebra. Source layouts are resolved before device
tracing into ordinary integer extents, byte strides, byte offsets, swizzle
enum fields, TensorMap bytes, and raw MMA descriptor bits. Every device-side
SMEM buffer is linear, every TMEM region is named by integer row/column
addresses, and every address helper below returns scalar integers or raw
pointers only. Abstract `tile` wording means a logical rectangular region of
the algorithm; it never denotes a TIRx tile primitive or a layout-bearing
value.

## Pipeline at a glance

| warp / role | tile program | publication / reuse edges |
| --- | --- | --- |
| warps 0-3, epilogue | warp 0 allocates all 512 TMEM columns; all four wait on named barrier 3, persistently consume one accumulator tile, load four anchor subtiles TMEM -> registers, accumulate absolute FP32 maxima, convert C, stage C in a five-stage SMEM ring, and TMA-store it; then reduce amax warp -> CTA -> global atomic | accumulator-full mbarrier -> TMEM load; named barrier 2 brackets each SMEM C stage and the four-warp amax handoff; elected consumer release publishes accumulator-empty; TMA bulk-group wait protects C-stage reuse |
| warp 4, MMA | waits for the TMEM pointer on named barrier 3; for each persistent output tile, waits for an accumulator stage and each AB stage, copies SFA/SFB SMEM -> TMEM, issues four block-scaled MMA instructions per anchor K tile, releases AB, and commits the accumulator | AB-full -> scale copy/MMA; `tcgen05.commit` publishes AB-empty after each K tile and accumulator-full after each output tile; accumulator-empty gates reuse |
| warp 5, TMA | prefetches all five TensorMaps, then the warp walks the same persistent tiles and K tiles; its waits/control are warp-uniform and each arrive/TMA issue selects one lane | AB-empty -> four TMA loads -> AB-full transaction completion; producer tail waits until all five stages are reusable |
| whole CTA/cluster prologue | initialize and publish AB mbarriers, independently initialize and publish accumulator mbarriers, then perform the source's third publication arrive before descriptor/address setup and its matching wait afterward | three `fence.mbarrier_init.release.cluster` epochs; the first two each perform cluster arrive/wait, while the third separates arrive from the later wait; named barrier 3 publishes TMEM pointer to MMA+epilogue |

The three persistent role-local scheduler cursors traverse the identical static
work list. They communicate tiles only through the two pipelines; there is no
extra scheduler handoff. Source `833-917`, `984-1107`, `1160-1320`; anchor PTX
`221-559`, `587-956`, `974-1334`.

## Primitive vocabulary

Structural forms describe placement and views without moving data:

```python
specialize(...)                 # compile-time dtype/major/tile/cluster branch
launch(...)                     # grid, six warps, cluster, dynamic SMEM, min occupancy
gmem_region(name, shape, dtype, major)
smem_bytes(name, offset, byte_count, alignment)
tmem_region(name, start_column, row_count, column_count, dtype)
rmem_words(name, count, dtype)
byte_ptr(base, scalar_byte_offset)
raw_mma_descriptor(base_ptr, ldo, sdo, swizzle_enum)
mbarrier_array(name, stages, arrivals)
pipeline_state(name, stages, phase, index, count)
persistent_scheduler(problem_tiles, cluster_shape, grid)
logical_region(tensor, scalar_coordinates, scalar_extents)
```

Directional movement is explicit:

```python
copy_g2s(src_gmem, dst_smem, tensor_map, mbarrier, multicast_mask)
copy_s2t(raw_smem_descriptor_u64, dst_tmem)
copy_t2r(src_tmem, dst_rmem)
copy_r2s(src_rmem, dst_smem)
copy_s2g(src_smem, dst_gmem, tensor_map)
load_smem(src, dst)
store_smem(src, dst)
```

Basic computation remains decomposed:

```python
gemm(acc_tmem, a_smem, sfa_tmem, b_smem, sfb_tmem, accumulate)
abs(src_f32, dst_f32)
reduce_max(src, dst, nan_semantics)
cast(src_f32, dst_c)
atomic_max_bits(src_nonnegative_f32, dst_f32)
```

Schedule operations stay visible:

```python
elect_one(); wait(barrier, phase); arrive_expect_tx(barrier, bytes)
try_wait_acquire(barrier, phase); wait_plain_if_not_ready(barrier, phase, token)
reset_count(state); advance(state.index, state.phase, state.count)
commit(barrier, cta_mask); release(barrier, cta_mask)
fence_mbarrier_init_cluster(); cluster_arrive(); cluster_wait(); cta_sync()
fence_async_shared_cta(); named_barrier(id, threads)
tma_commit_group(); tma_wait_group(pending)
alloc_tmem(columns); dealloc_tmem(columns)
```

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# source 134-200, 201-338, 340-580; PTX 14-42
# ===========================================================================
P = specialize(
    M, N, K, L,
    ab_dtype, sf_dtype, sf_vec_size, c_dtype,
    a_major, b_major, c_major,
    n_tile, cluster_m, cluster_n,
)
M_TILE = 128
K_TILE = 4 * mma_instruction_k(P)       # anchor: 4 * 64 == 256
N_TILE = P.n_tile                       # 128 or 256
ACC_DTYPE = f32
CTA_GROUP = 1                           # M tile 256 is excluded
WARPS = 6
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
AB_STAGES = compute_source_ab_stages(P) # anchor: 5
ACC_STAGES = 2 if N_TILE == 128 else 1
C_STAGES = compute_source_c_stages(P)   # anchor: 5

# Raw tensor pointers remain in the launch ABI because amax is not a TensorMap.
# Five opaque TensorMaps describe A/B/SFA/SFB/C, including runtime shapes,
# strides, OOB fill, boxes, swizzle enum fields, and multicast-capable load
# forms. They are already-encoded descriptors, not first-class layouts.
ABI = (
    A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, amax_ptr,
    map_A, map_B, map_SFA, map_SFB, map_C,
    scheduler_shape_and_fast_divmod,
)

launch(
    grid=persistent_grid(ceil_div(M, 128), ceil_div(N, N_TILE), L,
                         (cluster_m, cluster_n)),
    block=(192, 1, 1),
    cluster=(cluster_m, cluster_n, 1),
    arch="sm_100a",
    min_blocks_per_sm=1,
    dynamic_smem=source_shared_storage_size(P),
)
# instruction_selection: `.reqntid 192,1,1`, `.minnctapersm 1`, and
#   `.extern .shared .align 1024`; extent: one launch specialization
#   (anchor PTX 14, 36-37; source 574-580)

# ===========================================================================
# Storage and synchronization objects
# source 504-548, 643-706, 809-816; PTX 122-218, 563-586
# ===========================================================================
AB_FULL  = mbarrier_array(stages=AB_STAGES, arrivals=1)
AB_EMPTY = mbarrier_array(stages=AB_STAGES,
                          arrivals=cluster_n + cluster_m - 1)
ACC_FULL = mbarrier_array(stages=ACC_STAGES, arrivals=1)
ACC_EMPTY = mbarrier_array(stages=ACC_STAGES, arrivals=4)
TMEM_DEALLOC = smem_scalar(i64)          # reserved; inactive for CTA_GROUP=1
TMEM_PTR = smem_scalar(u32)

# Anchor dynamic-SMEM byte map. Non-anchor offsets are recomputed from the same
# declaration order and alignments; no two live objects alias.
#   AB_FULL 0..39; AB_EMPTY 40..79; ACC_FULL 80..95; ACC_EMPTY 96..111
#   TMEM_DEALLOC 112..119; TMEM_PTR 120..123; padding to 1024
#   sC 1024..41983; sA 41984..123903; sB 123904..205823
#   sSFA 205824..216063; sSFB 216064..226303; sAmax 226304..226319
#   shared struct rounded to 227328 bytes.
sC = smem_bytes("sC", source_sC_offset(P), source_sC_bytes(P), alignment=1024)
sA = smem_bytes("sA", source_sA_offset(P), source_sA_bytes(P), alignment=1024)
sB = smem_bytes("sB", source_sB_offset(P), source_sB_bytes(P), alignment=1024)
sSFA = smem_bytes("sSFA", source_sSFA_offset(P), source_sSFA_bytes(P), alignment=1024)
sSFB = smem_bytes("sSFB", source_sSFB_offset(P), source_sSFB_bytes(P), alignment=1024)
sAmax = smem_bytes("sAmax", source_sAmax_offset(P), 4 * sizeof(f32), alignment=16)

# These are pure compile-time/scalar-integer address formulas. They neither
# create nor consume a layout value. Each returns a raw address in its linear
# byte buffer; swizzle XOR and stage strides are folded into integer arithmetic.
sC_addr = scalar_address_formula(P, region="C")
sA_addr = scalar_address_formula(P, region="A")
sB_addr = scalar_address_formula(P, region="B")
sSFA_addr = scalar_address_formula(P, region="SFA")
sSFB_addr = scalar_address_formula(P, region="SFB")
sAmax_addr = lambda warp_or_slot: 4 * warp_or_slot

# The MMA shared descriptors are raw u64 values. Their base address, LDO, SDO,
# transposition bit, and swizzle enum are ordinary integers fixed by P.
a_desc = raw_mma_descriptor(sA, A_LDO(P), A_SDO(P), A_SWIZZLE_ENUM(P))
b_desc = raw_mma_descriptor(sB, B_LDO(P), B_SDO(P), B_SWIZZLE_ENUM(P))
sfa_desc = raw_mma_descriptor(sSFA, SF_LDO(P), SF_SDO(P), 0)
sfb_desc = raw_mma_descriptor(sSFB, SF_LDO(P), SF_SDO(P), 0)
def sfa_desc_for(stage, sf_chunk):
    return raw_descriptor_address_formula(sfa_desc, stage, sf_chunk)

def sfb_desc_for(stage, sf_chunk):
    return raw_descriptor_address_formula(sfb_desc, stage, sf_chunk)

# All 512 columns are allocated. ACC_STAGES selects disjoint accumulator
# stages; SFA then SFB occupy the following source-computed column regions.
tAcc = tmem_region("acc", start_column=0, row_count=128,
                   column_count=accumulator_columns(P, ACC_STAGES), dtype=f32)
tSFA = tmem_region("SFA", start_column=after_accumulator_columns(P),
                   row_count=128, column_count=SFA_COLUMNS(P), dtype=sf_dtype)
tSFB = tmem_region("SFB", start_column=after_accumulator_and_SFA_columns(P),
                   row_count=128, column_count=SFB_COLUMNS(P), dtype=sf_dtype)

rAcc = rmem_words("acc", source_thread_acc_word_count(P), f32)
rC = rmem_words("C", source_thread_c_word_count(P), c_dtype)

# ===========================================================================
# Coordinates, descriptor prefetch, and barrier initialization
# source 614-719, 819-824; PTX 82-218
# ===========================================================================
warp = warp_uniform(warp_id())
lane = lane_id()
cluster_rank = block_idx_in_cluster()
cluster_m_coord = cluster_rank % cluster_m
cluster_n_coord = (cluster_rank // cluster_m) % cluster_n

# Integer CTA masks replace every source-side inferred cluster mapping. The AB
# mask is the union of the same-M row and same-N column, with rank encoded as
# `m + cluster_m * n`; duplicate self bits collapse under integer OR. Its
# anchor value is 0b11. The accumulator mask is the one-hot local CTA rank.
ab_consumer_mask = 0
for peer_n in range(cluster_n):
    ab_consumer_mask |= 1 << (cluster_m_coord + cluster_m * peer_n)
for peer_m in range(cluster_m):
    ab_consumer_mask |= 1 << (peer_m + cluster_m * cluster_n_coord)
acc_producer_mask = 1 << cluster_rank

if warp == TMA_WARP:
    for descriptor in (map_A, map_B, map_SFA, map_SFB, map_C):
        prefetch(descriptor)
        # instruction_selection: `prefetch.tensormap`; extent: five scalar
        #   descriptor prefetches (source 620-625; PTX 93-111)

if warp == 0 and elect_one():
    for stage in range(AB_STAGES):
        init(AB_FULL[stage], arrivals=1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: five
        #   anchor barriers at offsets 0..32 (source 648-659; PTX 126-135)
if warp == 0 and elect_one():
    for stage in range(AB_STAGES):
        init(AB_EMPTY[stage], arrivals=cluster_n + cluster_m - 1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: five
        #   anchor barriers, arrival count 2, offsets 40..72 (PTX 140-149)

# Publication epoch 1 belongs to PipelineTmaUmma construction. AB full and
# empty initialization have independent elected issuers, and the complete AB
# group is published before any accumulator barrier is initialized.
fence_mbarrier_init_cluster()
# instruction_selection: `fence.mbarrier_init.release.cluster`; extent:
#   AB-pipeline publication (source 648-659 plus PipelineTmaUmma.create;
#   PTX 157)
if cluster_m * cluster_n > 1:
    cluster_arrive()
    cluster_wait()
    # instruction_selection: `barrier.cluster.arrive.relaxed` then
    #   `barrier.cluster.wait`; extent: first cluster synchronization
    #   (PTX 158-159)
else:
    cta_sync()
    # instruction_selection: `bar.sync 0`; extent: default CTA-wide singleton
    #   fallback from PipelineTmaUmma.create

if warp == 0 and elect_one():
    for stage in range(ACC_STAGES):
        init(ACC_FULL[stage], arrivals=1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: two
        #   anchor barriers at offsets 80,88 (source 661-671; PTX 163-170)
if warp == 0 and elect_one():
    for stage in range(ACC_STAGES):
        init(ACC_EMPTY[stage], arrivals=4)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: two
        #   anchor barriers at offsets 96,104 (PTX 176-182)

# Publication epoch 2 belongs to PipelineUmmaAsync construction and is
# separate from both AB publication and the explicit source publication.
fence_mbarrier_init_cluster()
# instruction_selection: `fence.mbarrier_init.release.cluster`; extent:
#   accumulator-pipeline publication (source 661-671 plus
#   PipelineUmmaAsync.create; PTX 187)
if cluster_m * cluster_n > 1:
    cluster_arrive()
    cluster_wait()
    # instruction_selection: `barrier.cluster.arrive.relaxed` then
    #   `barrier.cluster.wait`; extent: second cluster synchronization
    #   (PTX 188-189)
else:
    cta_sync()
    # instruction_selection: `bar.sync 0`; extent: default CTA-wide singleton
    #   fallback from PipelineUmmaAsync.create

# Publication epoch 3 is the explicit source fence/arrive. Its matching wait
# intentionally occurs only after the scalar address, descriptor, partition,
# and mask setup represented above; it is not adjacent to this arrive.
fence_mbarrier_init_cluster()
# instruction_selection: `fence.mbarrier_init.release.cluster`; extent:
#   explicit publication (source 673-687; PTX 191)
if cluster_m * cluster_n > 1:
    cluster_arrive()
    # instruction_selection: `barrier.cluster.arrive.relaxed`; extent: third
    #   cluster arrival (source 685-687; PTX 193)

perform_scalar_address_descriptor_partition_and_mask_setup()

if cluster_m * cluster_n > 1:
    cluster_wait()
    # instruction_selection: `barrier.cluster.wait`; extent: delayed matching
    #   wait before role-local staged-tensor use (source 819-824; PTX 218)
else:
    named_barrier(id=1, threads=192)
    # instruction_selection: `bar.sync 1, 192`; extent: the third epoch's CTA
    #   fallback at the delayed wait site

# Multicast ownership is structural and independent of CTA_GROUP:
#   A and SFA multicast across equal M coordinates (cluster N dimension);
#   B and SFB multicast across equal N coordinates (cluster M dimension).
# A one-element multicast dimension selects the non-multicast CTA destination.

# ===========================================================================
# Role 1: warp 5, persistent TMA producer with elected issue sites
# source 829-917; PTX 221-559
# ===========================================================================
if warp == TMA_WARP:
    sched = persistent_scheduler(problem_tiles=(ceil_div(M, 128),
                                                ceil_div(N, N_TILE), L),
                                 cluster_shape=(cluster_m, cluster_n),
                                 grid=grid_dim())
    ab_prod = pipeline_state(stages=AB_STAGES, phase=1, index=0, count=0)

    while sched.valid:
        tile_m, tile_n, tile_l = sched.current_tile()
        # cluster-local CTA coordinates choose this CTA's A/B/SF partitions;
        # TensorMap OOB fill supplies aligned M/N/K tails.
        reset_count(ab_prod)
        ab_empty_ready = True
        if ab_prod.count < ceil_div(K, K_TILE):
            ab_empty_ready = try_wait_acquire(AB_EMPTY[ab_prod.index],
                                              ab_prod.phase)
            # instruction_selection:
            #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            #   extent: initial speculative AB-empty probe (source 860-864;
            #   PTX 290-297)

        while ab_prod.count < ceil_div(K, K_TILE): # anchor body is nounroll
            k_tile = ab_prod.count
            wait_plain_if_not_ready(AB_EMPTY[ab_prod.index], ab_prod.phase,
                                    ab_empty_ready)
            # instruction_selection: only when the speculative token is false,
            #   retry loop containing `mbarrier.try_wait.parity.shared.b64
            #   ...,10000000`; extent: conditional blocking retry per K tile
            #   (source 868-870; PTX 305-327)

            if elect_one():
                arrive_expect_tx(AB_FULL[ab_prod.index], source_stage_bytes(P))
                # instruction_selection: `mbarrier.arrive.expect_tx.shared.b64`;
                #   extent: one elected-lane arrival, anchor 36,864 bytes
                #   (source 870; PTX 330-336)

            if elect_one():
                copy_g2s(logical_region(A, A_coords(tile_m, k_tile, tile_l), A_box(P)),
                         byte_ptr(sA, sA_addr(ab_prod.index, 0, 0)),
                         map_A, AB_FULL[ab_prod.index], multicast_A_mask)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
            #   extent: one FP4 A tile (source 873-879; PTX 342-351)
            if elect_one():
                copy_g2s(logical_region(B, B_coords(tile_n, k_tile, tile_l), B_box(P)),
                         byte_ptr(sB, sB_addr(ab_prod.index, 0, 0)),
                         map_B, AB_FULL[ab_prod.index], multicast_B_mask)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint`;
            #   extent: one FP4 B tile, mask 0b11 (source 880-886; PTX 353-364)
            if elect_one():
                copy_g2s(logical_region(SFA, SFA_coords(tile_m, k_tile, tile_l), SFA_box(P)),
                         byte_ptr(sSFA, sSFA_addr(ab_prod.index, 0, 0)),
                         map_SFA, AB_FULL[ab_prod.index], multicast_SFA_mask)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
            #   extent: one E8M0 SFA tile (source 887-893; PTX 370-380)
            if elect_one():
                copy_g2s(logical_region(SFB, SFB_coords(tile_n, k_tile, tile_l), SFB_box(P)),
                         byte_ptr(sSFB, sSFB_addr(ab_prod.index, 0, 0)),
                         map_SFB, AB_FULL[ab_prod.index], multicast_SFB_mask)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint`;
            #   extent: one E8M0 SFB tile, mask 0b11 (source 894-900; PTX 382-397)

            advance(ab_prod.index, ab_prod.phase, ab_prod.count, AB_STAGES)
            ab_empty_ready = True
            if ab_prod.count < ceil_div(K, K_TILE):
                ab_empty_ready = try_wait_acquire(AB_EMPTY[ab_prod.index],
                                                  ab_prod.phase)
                # instruction_selection:
                #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                #   extent: next-stage speculative probe after the full cursor
                #   advance (source 902-906; PTX 398-413)
        sched.advance()

    for live_stage in producer_tail(ab_prod, AB_STAGES):
        wait(AB_EMPTY[live_stage.index], live_stage.phase)
        # instruction_selection: five retry loops containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: producer
        #   tail over the five anchor stages (source 914-917; PTX 451-559)

# ===========================================================================
# Role 2: warp 4, persistent block-scaled MMA consumer/producer
# source 922-1107; PTX 563-956
# ===========================================================================
if warp == MMA_WARP:
    named_barrier(id=3, threads=160)
    # instruction_selection: `bar.sync 3,160`; extent: MMA plus four epilogue
    #   warps (source 922-926; PTX 566-572)
    load_smem(TMEM_PTR, acc_tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one pointer word
    #   (source 929-966; PTX 571-586)

    sched = persistent_scheduler(...same work list...)
    ab_cons = pipeline_state(stages=AB_STAGES, phase=0, index=0, count=0)
    acc_prod = pipeline_state(stages=ACC_STAGES, phase=1, index=0, count=0)

    while sched.valid:
        reset_count(ab_cons)
        ab_full_ready = True
        if ab_cons.count < ceil_div(K, K_TILE):
            ab_full_ready = try_wait_acquire(AB_FULL[ab_cons.index],
                                             ab_cons.phase)
            # instruction_selection:
            #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            #   extent: initial speculative AB-full probe before the ACC wait
            #   (source 1003-1007; PTX 664-675)

        wait(ACC_EMPTY[acc_prod.index], acc_prod.phase)
        # instruction_selection: blocking retry loop containing plain
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
        #   accumulator-stage acquire per output tile (source 1010-1013;
        #   PTX 679-697)
        accumulate = False
        while ab_cons.count < ceil_div(K, K_TILE):
            k_tile = ab_cons.count
            wait_plain_if_not_ready(AB_FULL[ab_cons.index], ab_cons.phase,
                                    ab_full_ready)
            # instruction_selection: only when the speculative token is false,
            #   retry loop containing `mbarrier.try_wait.parity.shared.b64
            #   ...,10000000`; extent: conditional AB-full retry per K tile
            #   (source 1023-1027; PTX 705-723)

            for sf_chunk in range(4):
                copy_s2t(sfa_desc_for(ab_cons.index, sf_chunk),
                         tmem_addr(tSFA, sf_chunk))
                # instruction_selection:
                #   `tcgen05.cp.cta_group::1.32x128b.warpx4`; extent: four
                #   elected-lane SFA issues consuming stage/chunk-specific raw
                #   u64 SMEM descriptors (source 1028-1042; PTX 728-760)
            for sf_chunk in range(4):
                copy_s2t(sfb_desc_for(ab_cons.index, sf_chunk),
                         tmem_addr(tSFB, sf_chunk))
                # instruction_selection:
                #   `tcgen05.cp.cta_group::1.32x128b.warpx4`; extent: four
                #   elected-lane SFB issues consuming stage/chunk-specific raw
                #   u64 SMEM descriptors (source 1043-1047; PTX 762-792)

            for kblock in range(4):
                gemm(tmem_addr(tAcc, acc_prod.index),
                     a_desc_for(ab_cons.index, kblock), tmem_addr(tSFA, kblock),
                     b_desc_for(ab_cons.index, kblock), tmem_addr(tSFB, kblock),
                     accumulate=accumulate)
                # instruction_selection: anchor
                #   `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16`;
                #   extent: four elected-lane MMA issues per 256-wide K tile,
                #   first issue uses the clear predicate and later issues
                #   accumulate (source 1049-1079; PTX 800-870)
                accumulate = True

            release(AB_EMPTY[ab_cons.index], cta_mask=ab_consumer_mask)
            # instruction_selection:
            #   `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`;
            #   extent: one elected-lane AB-stage release with explicit anchor
            #   CTA mask 0b11; no intervening warp barrier exists (source
            #   1081-1082; PTX 869-877)
            advance(ab_cons.index, ab_cons.phase, ab_cons.count, AB_STAGES)
            ab_full_ready = True
            if ab_cons.count < ceil_div(K, K_TILE):
                ab_full_ready = try_wait_acquire(AB_FULL[ab_cons.index],
                                                 ab_cons.phase)
                # instruction_selection:
                #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                #   extent: next-stage speculative probe after the full cursor
                #   advance (source 1084-1089; PTX 878-893)

        commit(ACC_FULL[acc_prod.index], cta_mask=acc_producer_mask)
        # instruction_selection:
        #   `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`;
        #   extent: one elected-lane accumulator publication per output tile,
        #   with explicit one-hot local CTA-rank mask (source 1091-1096;
        #   PTX 903-908)
        advance(acc_prod.index, acc_prod.phase, acc_prod.count, ACC_STAGES)
        sched.advance()

    if cluster_rank % 2 == 0:
        for unused in range(ACC_STAGES - 1):
            advance(acc_prod.index, acc_prod.phase, acc_prod.count, ACC_STAGES)
        wait(ACC_EMPTY[acc_prod.index], acc_prod.phase)
        # instruction_selection: rank-even predicate, exactly ACC_STAGES-1
        #   state advances, then one retry loop containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
        #   terminal accumulator wait (source 1104-1107 plus
        #   PipelineUmmaAsync.producer_tail; PTX 927-956)

# ===========================================================================
# Role 3: warps 0-3, persistent C + amax epilogue
# source 1111-1342; PTX 958-1342
# ===========================================================================
if warp in EPI_WARPS:
    if warp == 0:
        alloc_tmem(columns=512, dst=TMEM_PTR)
        # instruction_selection:
        #   `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`;
        #   extent: one 512-column allocation (source 1113-1120; PTX 962-969)
    named_barrier(id=3, threads=160)
    # instruction_selection: `bar.sync 3,160`; extent: pointer publication
    #   (source 1122-1126; PTX 970-973)
    load_smem(TMEM_PTR, acc_tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one pointer word
    #   (source 1129-1137; PTX 972-973)

    sched = persistent_scheduler(...same work list...)
    acc_cons = pipeline_state(stages=ACC_STAGES, phase=0, index=0, count=0)
    c_stage = 0

    while sched.valid:
        wait(ACC_FULL[acc_cons.index], acc_cons.phase)
        # instruction_selection: retry loop containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
        #   accumulator wait per tile (source 1201-1205; PTX 1075-1089)
        thread_tile_amax = 0.0f

        for subtile in range(source_epilogue_subtiles(P)): # anchor: four
            copy_t2r(tmem_addr(tAcc, acc_cons.index, subtile), rAcc)
            # instruction_selection:
            #   `tcgen05.ld.sync.aligned.32x32b.x32.b32`; extent: one 32-register
            #   FP32 tile per epilogue thread and subtile (source 1219-1224;
            #   PTX 1103-1107)

            for value in rAcc: # anchor: 32 FP32 values
                abs(value, abs_value)
                # instruction_selection: `abs.f32`; extent: 32 scalar values
                #   per thread/subtile (source 1226-1231; PTX 1125-1172)
            reduce_max(abs_rAcc, subtile_amax, nan_semantics="source")
            # instruction_selection: `max.NaN.f32` reduction tree followed by
            #   `max.NaN.f32` against zero; only the subsequent running update
            #   is plain `max.f32`; extent: one 32-value register tile (source
            #   1232-1237; PTX 1173-1192)
            thread_tile_amax = max(thread_tile_amax, subtile_amax)
            # instruction_selection: `max.f32`; extent: one scalar running max
            #   (source 1237; PTX 1191-1192)

            cast(rAcc, rC, dtype=c_dtype)
            # instruction_selection: anchor `cvt.rn.f16x2.f32`; extent: 16
            #   packed pair conversions per thread/subtile (source 1239-1244;
            #   PTX 1193-1209)
            for vector in range(4):
                copy_r2s(rC[vector], byte_ptr(sC, sC_addr(c_stage, warp, lane, vector)))
                # instruction_selection: `st.shared.v4.b32`; extent: four
                #   16-byte vector stores per thread/subtile (source 1247-1254;
                #   PTX 1210-1221)

            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta`; extent:
            #   one visibility fence (source 1255-1259; PTX 1222-1223)
            named_barrier(id=2, threads=128)
            # instruction_selection: `bar.sync 2,128`; extent: all epilogue
            #   threads before TMA reads sC (source 1260; PTX 1224-1225)

            if warp == 0:
                copy_s2g(byte_ptr(sC, sC_addr(c_stage, 0, 0, 0)),
                         C_coords(sched.tile, subtile), map_C)
                # instruction_selection: anchor
                #   `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint`;
                #   extent: one FP16 C subtile (source 1263-1270; PTX 1228-1234)
                tma_commit_group()
                # instruction_selection: `cp.async.bulk.commit_group`; extent:
                #   one store group (source 1272; PTX 1235-1236)
                tma_wait_group(C_STAGES - 1)
                # instruction_selection: `cp.async.bulk.wait_group.read 4`;
                #   extent: anchor keeps four older groups pending
                #   (source 1273; PTX 1237-1238)
            named_barrier(id=2, threads=128)
            # instruction_selection: `bar.sync 2,128`; extent: sC-stage reuse
            #   after issue (source 1274; PTX 1240-1241)
            c_stage = (c_stage + 1) % C_STAGES

        warp_amax = reduce_max(thread_tile_amax, lanes=32,
                               nan_semantics="source")
        # instruction_selection: `redux.sync.max.NaN.f32`; extent: one
        #   warp-wide reduction per output tile (source 1276-1282; PTX 1252-1259)
        if lane == 0:
            store_smem(warp_amax, byte_ptr(sAmax, sAmax_addr(warp)), dtype=f32)
            # instruction_selection: `st.shared.b32`; extent: one scalar per
            #   epilogue warp (source 1284-1286; PTX 1260-1263)
        named_barrier(id=2, threads=128)
        # instruction_selection: `bar.sync 2,128`; extent: four-warp amax
        #   publication (source 1288-1289; PTX 1264-1267)

        if warp == 0 and lane == 0:
            for i in range(4):
                load_smem(byte_ptr(sAmax, sAmax_addr(i)), warp_max[i], dtype=f32)
                # instruction_selection: `ld.shared.b32`; extent: four scalar
                #   loads (source 1291-1296; PTX 1270-1284)
            reduce_max(warp_max, block_amax, nan_semantics="source")
            # instruction_selection: `max.f32`; extent: four scalar inputs
            #   with zero identity (source 1293-1296; PTX 1272-1285)
            atomic_max_bits(block_amax, amax_ptr)
            # instruction_selection: `atom.global.max.s32`; extent: one scalar
            #   signed integer-bit atomic on non-negative FP32 (source
            #   1298-1308; PTX 1286-1287)

        if elect_one():
            release(ACC_EMPTY[acc_cons.index])
            # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one
            #   elected arrival per epilogue warp, four arrivals complete the
            #   stage (source 1310-1314; PTX 1289-1296)
        advance(acc_cons.index, acc_cons.phase, acc_cons.count, ACC_STAGES)
        sched.advance()

    named_barrier(id=2, threads=128)
    # instruction_selection: `bar.sync 2,128`; extent: final epilogue rendezvous
    #   before TMEM teardown (source 1322-1326; PTX 1335-1337)
    dealloc_tmem(columns=512)
    # instruction_selection:
    #   `tcgen05.dealloc.cta_group::1.sync.aligned.b32`; extent: one 512-column
    #   deallocation reached by the source's four-warp epilogue domain
    #   (source 1333-1338; PTX 1338-1340)
    tma_wait_group(0)
    # instruction_selection: `cp.async.bulk.wait_group.read 0`; extent: drain
    #   every C store before return (source 1340-1342; PTX 1341-1342)
```

## Pipeline inventory

| pipeline | anchor stages | producer | consumer | full publication | reuse publication |
| --- | ---: | --- | --- | --- | --- |
| AB/SF | 5 | elected lane, warp 5 | elected lane, warp 4 | TMA transaction completion on `AB_FULL`, 36,864 destination bytes | `tcgen05.commit` to `AB_EMPTY` after four MMAs |
| accumulator | 2 | elected lane, warp 4 | four epilogue warps | `tcgen05.commit` to `ACC_FULL` after all K tiles | one elected `mbarrier.arrive` per epilogue warp completes `ACC_EMPTY` |
| C store ring | 5 | all four epilogue warps stage; every lane of warp 0 collectively issues the TMA store/commit/wait path | TMA engine | SMEM fence + named barrier 2 + bulk-group commit | `wait_group.read 4` before the ring advances; second named barrier prevents early overwrite |

Every AB and accumulator cursor has `(index, phase, count)`. Index wraps at its
stage count and toggles phase; the producer and consumer use complementary
initial phases exactly as the source pipeline constructors do. The C cursor has
no mbarrier phase: it is `(tiles_executed * subtile_count + subtile) % C_STAGES`.

## TensorMap fields and tail behavior

| map | logical GMEM tensor | anchor rank | box / SMEM destination | multicast axis |
| --- | --- | ---: | --- | --- |
| A | `(M,K,L)` in source major mode | 3 | CTA A tile in the selected AB stage | cluster N; anchor extent one, so CTA destination |
| B | `(N,K,L)` in source major mode | 3 | CTA B tile in the selected AB stage | cluster M; anchor mask `0b11` |
| SFA | physical block-scaled atom byte order for `(M,ceil_div(K,V),L)` | 4 | CTA SFA tile in the selected AB stage | cluster N; anchor extent one |
| SFB | physical block-scaled atom byte order for `(N,ceil_div(K,V),L)` | 4 | CTA SFB tile in the selected AB stage | cluster M under the SFB-specific integer coordinate formula; anchor mask `0b11` |
| C | `(M,N,L)` in source major mode | 3 | one C epilogue subtile | none; shared CTA -> global |

M/N/K tails remain valid only when the source's 16-byte contiguous-dimension
alignment predicate holds. TensorMap bounds/OOB fill and the scheduler's
ceil-div tile counts carry the tail; the device sketch does not invent an
independent scalar cleanup role. L is the third logical scheduler coordinate.

## Static specialization boundary

There is one builder and one device-kernel definition. The following are one
compile key, not separate kernels:

| axis | accepted values | emitted-code consequence |
| --- | --- | --- |
| A/B dtype | FP4 E2M1; FP8 E4M3; FP8 E5M2, matching | TMA element/packing, SMEM descriptor, block-scaled MMA kind |
| SF | FP4 input: E8M0/V16, E8M0/V32, E4M3/V16; FP8 input: E8M0/V32 | physical SF descriptor, SF TMEM column addresses, block16/block32 MMA descriptor |
| C dtype | FP32, FP16, BF16, FP8 E4M3/E5M2, FP4 E2M1 | epilogue register conversion/packing, C SMEM size/swizzle/TensorMap; FP4 output only with FP4 input and N-major C |
| A/B major | FP4 K/K; FP8 A K/M and B K/N | TensorMap strides, shared byte-address formulas, MMA operand descriptor/transposition |
| C major | M or N, except FP4 output N only | epilogue tile, TMEM load partition, shared swizzle, C TensorMap |
| N tile | 128 or 256 | accumulator columns/stages, epilogue partition; N256 has one accumulator stage |
| cluster | `{1,2,4} x {1,2,4}` | launch cluster, cluster coordinate, A/SFA and B/SFB multicast masks and AB-empty arrival count |
| shape | aligned positive M/N/K and positive L | TensorMap extents, persistent tile count, K loop, tail predicates |

The source's known failing combinations stay outside the compile key: FP8
input with FP8 output; N tile 256 + SF vector 16 + FP32/FP16/BF16 output; M tile
256; and the `uint8` packed-FP4 public alias. The working `int8` scale alias is
normalized to the identical E8M0 bits before specialization, so it creates no
device branch.

The anchor proves the exact instructions annotated above. Each non-anchor
branch must be source-exported during implementation/correctness validation to
verify its conversion, MMA descriptor, TensorMap rank/major mode, stage counts,
and predicates; no branch may change this role/pipeline skeleton.

## TIRx module and benchmark contract

- Registry name: `cudnn_sm100_dense_blockscaled_gemm_persistent_amax`, category
  `cudnn`, compute capability 10.
- The device definition imports only `tirx_kernels.kern as K` and uses
  `K.kernel`, `K.specialize`, `K.Pipeline`/barrier state, `K.TensorMap`, and
  `K.ptx`. Direct `T`, `Tx`, `I`, tile primitives, and `tirx.tile.*` are absent.
- The module exports `KERNEL_META`, deterministic `CONFIGS`/`BENCH_CONFIGS`,
  `get_kernel`, `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, and
  `run_bench`. The 1,350 structural modes are a coverage universe, never 1,350
  kernel definitions.
- Correctness compares C against both the pinned standalone cuDNN source kernel
  and a dequantized FP32 mathematical reference. Tolerances are 0.01 for
  FP32/FP16/BF16 C, 0.1 for FP8/FP4 C, and 0.1 for FP32 amax. The E8M0 `int8`
  alias is also checked bitwise after normalization.
- `prepare_bench` compiles only TIRx and performs no Torch/cuDNN/CuTeDSL import
  or CUDA initialization. `run_gpu` lazily verifies and compiles the source
  after a CUDA context exists. Both timed closures use identical inputs,
  independent C/amax outputs, and the canonical Proton timer.
- Final performance is one complete `bench_suite` run with references, five
  rounds, zero cooldown. A pure JSON validator reads only those raw samples and
  requires `mean(cudnn_frontend_us) / mean(tirx_us) > 0.99` independently on
  every workload row.

## Instruction selection is a lowering consequence

The anchor export, rather than source API names, fixes the instruction families:

| placement / schedule | anchor PTX consequence |
| --- | --- |
| dynamic staged storage | `.extern .shared .align 1024`; concrete regions listed above |
| TensorMap producer | A/SFA use CTA destinations; B/SFB use cluster multicast; one expected-byte mbarrier completes all four |
| scale SMEM -> TMEM placement | eight `tcgen05.cp.cta_group::1.32x128b.warpx4` per K tile |
| FP4 + E8M0/V16 + 128x128 | four `tcgen05.mma...kind::mxf4nvf4.block_scale.block16` per K tile |
| FP32 TMEM fragment shape | one `tcgen05.ld.sync.aligned.32x32b.x32.b32` per thread/subtile |
| FP16 N-major epilogue | 16 `cvt.rn.f16x2.f32`, four `st.shared.v4.b32`, one 3-D TMA store per thread/subtile/tile ownership described above |
| absolute maximum semantics | 32 `abs.f32`, `max.NaN.f32` register tree, `redux.sync.max.NaN.f32`, four scalar SMEM loads/maxima, `atom.global.max.s32` |
| stage reuse | mbarrier parity waits/commits for AB and accumulator; `wait_group.read 4` for the five-stage C ring; `wait_group.read 0` at tail |

All counts are per static occurrence or per explicit loop iteration as stated;
they are evidence for the semantic operations, not hidden computation.
