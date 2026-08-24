<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 block-scaled persistent GEMM + interleaved SwiGLU quant: coarse WASP pipeline sketch

This is a non-executable execution sketch for the single parameterized module
[`tirx_kernels/cudnn/swiglu/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py`](../../tirx_kernels/cudnn/swiglu/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py).
It freezes the six-warp split, persistent scheduler, A/B/SFA/SFB block-scaled
mainloop, paired AB12 stores, interleaved SwiGLU, optional amax, optional SFC
quantization, C stores, and two-CTA teardown. After its first reviewer PASS this
file is immutable; the module becomes the executable source of truth.

The source is
`python/cudnn/gemm/cutedsl/dense/swiglu/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py`
at commit `7b5327b32907b9dd21d85a393d62f9573d7f0116`, SHA256
`f7ec00ae82e79266fcc8b9cf9fdd1b7521bf846ee16ff227781ac204f3787fc1`.
Source citations below refer to that file. Writer evidence is under
`.porting/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant/writer_export/`.
The primary 1,968-line PTX is `anchor/anchor_lineinfo.ptx`, SHA256
`0140539cfe79c4af52558077cffd8acbcee7837bae97b0b7dfd87320345da02e`.
The independently useful variant paths and hashes are recorded in the final
evidence table; `ptx_evidence.json` contains every `.loc` -> opcode occurrence.

The instruction-annotated body uses this primary specialization:

| axis | anchor value |
| --- | --- |
| shape | `M=N=K=1024`, `L=1` |
| A/B | FP4 E2M1, K-major/K-major |
| SFA/SFB | E8M0, vector size 16 |
| AB12/C | BF16/BF16, N-major |
| MMA tile / cluster | `(256,256)` / `(2,2)` |
| epilogue | packed `vector_f32=True`, amax active, SFC inactive |
| fixed | FP32 accumulator, AB12 stages 4, overlap margin 0 |

The same skeleton covers every branch in **Static specialization boundary**.
FP8 C activates SFC; FP4 input plus BF16 C activates amax. Inactive paths still
receive separate dummy buffers through the fixed ABI and disappear at compile
time. The source's `uint8` packed-FP4 alias is not executable and is rejected.

No first-class mapping object is used here or in device code. All shared memory
is one rank-1 `u8` arena. Logical dimensions, stages, transposes, and swizzles
are scalar integer offset functions. TMEM is addressed by integer columns and
rows. UMMA shared descriptors and TensorMaps are raw encoded operands. The word
“region” below means scalar extents plus a base offset, never a tile primitive.

## Pipeline at a glance

| warp / role | persistent program | publication and reuse edges |
| --- | --- | --- |
| warp 5, TMA | prefetch six TensorMaps; for each output tile and K tile, wait on an AB-empty stage and issue A/B/SFA/SFB TMA loads | one transaction-completion AB-full mbarrier covers all four loads; A/SFA multicast in cluster N, B/SFB in cluster M; producer tail waits for every stage |
| warp 4, MMA | wait for the TMEM pointer; leader CTA waits for AB and accumulator stages, copies SF SMEM -> TMEM, performs four block-scaled MMAs per anchor K tile, releases AB, publishes accumulator | AB-full -> `tcgen05.cp`/MMA -> AB-empty; accumulator-empty -> MMA -> accumulator-full; 2CTA leader owns instructions |
| warps 0-3, epilogue | warp 0 allocates 512 TMEM columns; pairs accumulator subtiles as up/gate, stores AB12, computes SwiGLU, optionally computes amax and/or SFC, stores C, releases accumulator | named barrier 1 fences AB12/C stage visibility and reuse; TMA bulk-group waits protect four AB12 and `C_STAGES` C rings; four elected releases complete accumulator-empty |
| CTA/cluster prologue and tail | initialize AB/accumulator/deallocation barriers, publish them once, compute integer descriptors and masks, then synchronize; warp 0 relinquishes/frees TMEM while all epilogue warps drain both store pipes | AB and accumulator construction defer synchronization; CTA2 allocator construction fences its deallocation barrier, then source line 773 performs the sole pipeline publication; named barrier 2 publishes TMEM pointer; two-CTA teardown uses 32 lane-wise peer arrivals |

The three role-local persistent cursors enumerate the same static work list.
They exchange ownership only through the AB and accumulator pipelines; no
separate scheduler handoff exists. Source `935-1028`, `1033-1233`, `1237-1720`;
anchor PTX `277-594`, `603-1023`, `1025-1962`.

## Primitive vocabulary

Structural forms return scalar integers, raw pointers, or encoded descriptors:

```python
specialize(...)
launch(...)
smem_bytes(base, offset, byte_count, alignment)
tmem_region(start_column, row_count, column_count, dtype)
rmem_words(count, dtype)
byte_ptr(base, scalar_byte_offset)
raw_mma_descriptor(base_ptr, ldo, sdo, transpose_bit, swizzle_enum)
pipeline_state(stages, phase, index, count)
persistent_scheduler(problem_tiles, cluster_shape, grid)
logical_region(pointer, scalar_coordinates, scalar_extents)
```

Data movement is directional and explicit:

```python
copy_g2s(gmem_region, smem_ptr, tensormap, mbarrier, multicast_mask)
copy_s2t(smem_descriptor_u64, tmem_address_u32)
copy_t2r(tmem_address_u32, registers)
copy_r2s(registers, smem_ptr)
copy_s2g(smem_ptr, gmem_region, tensormap)
load_global(pointer, register)
store_global(register, pointer)
load_smem(pointer, register)
store_smem(register, pointer)
```

Computation remains decomposed:

```python
gemm(acc_tmem, a_smem, sfa_tmem, b_smem, sfb_tmem, accumulate)
cast(source, destination_dtype)
mul(a, b); add(a, b); exp2(a); reciprocal(a)
abs(source); reduce_max(source, nan_semantics)
atomic_max_bits(nonnegative_f32, global_f32)
```

Scheduling and synchronization stay visible:

```python
elect_one(); try_wait(barrier, phase); wait(barrier, phase)
arrive_expect_tx(barrier, byte_count); commit(barrier, cta_mask)
release(barrier, cta_mask); advance(index, phase, count)
fence_mbarrier_init_cluster(); cluster_arrive(); cluster_wait(); cta_sync()
fence_async_shared_cta(); named_barrier(id, threads)
tma_commit_group(); tma_wait_group(pending)
alloc_tmem(columns); relinquish_tmem(); dealloc_tmem(columns)
```

## Complete sketch

```python
# ===========================================================================
# 1. Static specialization, fixed pointer ABI, and launch
# source 134-200, 201-654; anchor PTX 14-44
# ===========================================================================
P = specialize(
    M, N, K, L,
    ab_dtype, sf_dtype, sf_vec_size, ab12_dtype, c_dtype,
    a_major, b_major, c_major,
    mma_tile_m, mma_tile_n, cluster_m, cluster_n,
    vector_f32,
)
CTA_GROUP = 2 if P.mma_tile_m == 256 else 1
CTA_M = P.mma_tile_m // CTA_GROUP       # always 128
N_TILE = P.mma_tile_n                   # 64, 128, 192, or 256
K_TILE = 256 if P.ab_dtype is FP4 else 128
ACC_DTYPE = f32
AB12_STAGES = 4
WARPS = 6
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
ACC_STAGES = source_acc_stages(P)       # anchor 1; N64/128/192 export 2
AB_STAGES = source_ab_stages(P)         # anchor 3
C_STAGES = source_c_stages(P)           # anchor 4
GENERATE_AMAX = P.ab_dtype is FP4 and P.c_dtype is BF16
GENERATE_SFC = P.c_dtype is FP8

# Exact caller-visible device ABI. TensorMaps are encoded in the host prelude
# from these pointers and captured as raw descriptor arguments; inactive output
# pointers name distinct one-element dummy allocations.
ABI = (A, B, SFA, SFB, C, AB12, amax, SFC, norm_const, alpha)

launch(
    grid=persistent_grid(ceil_div(M, CTA_M), ceil_div(N, N_TILE), L,
                         (cluster_m, cluster_n), CTA_GROUP),
    block=(192, 1, 1),
    cluster=(cluster_m, cluster_n, 1),
    arch="sm_100a",
    min_blocks_per_sm=1,                 # source stage-capacity heuristic only
    dynamic_smem=source_shared_bytes(P),
)
# instruction_selection: `.reqntid 192,1,1` at anchor `39` and
#   `.extern .shared .align 1024` at anchor `14`; extent: one specialization.
#   Source `655-660`. Normal-path PTX has no `.minnctapersm`; occupancy one is
#   used only by the source's host-side stage-capacity calculation.

# ===========================================================================
# 2. One rank-1 SMEM arena, protocol header, integer mappings, and TMEM
# source 563-617, 731-803, 918-925; anchor PTX 145-244, 770-940
# ===========================================================================
arena = smem_bytes(base=dynamic_smem, offset=0,
                   byte_count=source_shared_bytes(P), alignment=1024)
pool = protocol_pool(base=arena)        # the only pool
AB_FULL = pool.pipeline_full(AB_STAGES, producer="tma")
AB_EMPTY = pool.pipeline_empty(AB_STAGES, consumer="tcgen05")
ACC_FULL = pool.pipeline_full(ACC_STAGES, producer="tcgen05")
ACC_EMPTY = pool.pipeline_empty(ACC_STAGES, consumer="mbarrier")
TMEM_DEALLOC = pool.alloc_u64()
TMEM_PTR = pool.alloc_u32()

# Anchor byte intervals. Other specializations recompute the same declaration
# order with scalar sizes and 1024-byte alignments.
#   protocol header: 0..75, padded to 1024
#   sC:     1024..33791       (4 * 8192)
#   sAB12: 33792..99327       (4 * 16384)
#   sA:    99328..148479      (3 * 16384)
#   sB:    148480..197631     (3 * 16384)
#   sSFA:  197632..203775     (3 * 2048)
#   sSFB:  203776..216063     (3 * 4096)
#   sAmax: 216064..216079; whole struct rounds to 217088 bytes.
sC = byte_ptr(arena, source_c_offset(P))
sAB12 = byte_ptr(arena, source_ab12_offset(P))
sA = byte_ptr(arena, source_a_offset(P))
sB = byte_ptr(arena, source_b_offset(P))
sSFA = byte_ptr(arena, source_sfa_offset(P))
sSFB = byte_ptr(arena, source_sfb_offset(P))
sAmax = byte_ptr(arena, source_amax_offset(P))

# Each helper below is integer arithmetic over stage, row, column, dtype bits,
# major, and swizzle XOR bits. No helper creates a mapping value.
sC_addr = integer_smem_address(region="C", specialization=P)
sAB12_addr = integer_smem_address(region="AB12", specialization=P)
sA_addr = integer_smem_address(region="A", specialization=P)
sB_addr = integer_smem_address(region="B", specialization=P)
sSFA_addr = integer_smem_address(region="SFA", specialization=P)
sSFB_addr = integer_smem_address(region="SFB", specialization=P)

a_desc = raw_mma_descriptor(sA, A_LDO(P), A_SDO(P), A_TRANS(P), A_SWIZZLE(P))
b_desc = raw_mma_descriptor(sB, B_LDO(P), B_SDO(P), B_TRANS(P), B_SWIZZLE(P))
sfa_desc = raw_mma_descriptor(sSFA, SF_LDO(P), SF_SDO(P), 0, 0)
sfb_desc = raw_mma_descriptor(sSFB, SF_LDO(P), SF_SDO(P), 0, 0)

# Exactly 512 columns are allocated. Disjoint integer column intervals hold
# accumulator stages, SFA, then SFB; N64/N192 apply the source's alternating
# two-column SFB address shift. CTA_GROUP selects paired-CTA addressing.
tAcc = tmem_region(start_column=0, row_count=128,
                   column_count=accumulator_columns(P), dtype=f32)
tSFA = tmem_region(start_column=after_accumulator(P), row_count=128,
                   column_count=sfa_columns(P), dtype=sf_dtype)
tSFB = tmem_region(start_column=after_accumulator_and_sfa(P), row_count=128,
                   column_count=sfb_columns(P), dtype=sf_dtype)
rUp = rmem_words(source_thread_acc_words(P), f32)
rGate = rmem_words(source_thread_acc_words(P), f32)
rAB12 = rmem_words(source_thread_output_words(P, ab12_dtype), ab12_dtype)
rCompute = rmem_words(source_thread_acc_words(P), f32)
rC = rmem_words(source_thread_output_words(P, c_dtype), c_dtype)
rSFC = rmem_words(source_thread_sfc_words(P), sf_dtype) if GENERATE_SFC else none

# ===========================================================================
# 3. Coordinates, descriptor prefetch, barrier initialization/publication
# source 701-930; anchor PTX 72-274
# ===========================================================================
warp = warp_uniform(warp_id())
lane = lane_id()
cluster_rank = cluster_rank_uniform()
cluster_x = cluster_rank % cluster_m
cluster_y = (cluster_rank // cluster_m) % cluster_n
cta_v = cluster_x % CTA_GROUP
leader_cta = cta_v == 0
leader_rank = cluster_x - cta_v + cluster_m * cluster_y

if warp == TMA_WARP:
    for descriptor in (map_A, map_B, map_SFA, map_SFB, map_C, map_AB12):
        prefetch(descriptor)
        # instruction_selection: `prefetch.tensormap`; extent: six descriptor
        #   prefetches. Source `707-713`; anchor PTX `94-138`.

# AB-empty arrivals equal the union of A/SFA and B/SFB multicast consumers;
# accumulator-full goes from the leader CTA to both CTA partners in 2CTA mode.
a_mask = integer_same_m_mask(cluster_x, cluster_y, cluster_m, cluster_n)
b_mask = integer_same_n_mask(cluster_x, cluster_y, cluster_m, cluster_n)
sfa_mask = a_mask
sfb_mask = integer_sfb_same_n_mask(P, cluster_x, cluster_y)
ab_empty_arrivals = source_ab_consumer_count(P)
acc_full_mask = ((1 << CTA_GROUP) - 1) << leader_rank
acc_empty_arrivals = 4 * CTA_GROUP

if warp == 0 and elect_one():
    for stage in range(AB_STAGES):
        init(AB_FULL[stage], arrivals=1)
        init(AB_EMPTY[stage], arrivals=ab_empty_arrivals)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: anchor
        #   three full and three empty barriers. Source `740`; anchor PTX
        #   `155-169`. Construction has `defer_sync=True`: there is no fence,
        #   cluster arrival, wait, or CTA sync here.

if warp == 0 and elect_one():
    for stage in range(ACC_STAGES):
        init(ACC_FULL[stage], arrivals=1)
        init(ACC_EMPTY[stage], arrivals=acc_empty_arrivals)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: one full
        #   and one empty anchor barrier. Source `754`; anchor PTX `199,212`.
        #   This construction also has `defer_sync=True` and adds no epoch.

if CTA_GROUP == 2 and warp == 0 and elect_one():
    init(TMEM_DEALLOC, arrivals=32)
    # instruction_selection: `mbarrier.init.shared.b64`; extent: one 2CTA
    #   deallocation barrier with immediate arrival count 32. Source
    #   `764-770`; anchor PTX `233-237`.
if CTA_GROUP == 2:
    fence_mbarrier_init_cluster()
    # instruction_selection: allocator-owned
    #   `fence.mbarrier_init.release.cluster`; extent: CTA2 only. Source
    #   `764-770`; anchor PTX `242`. CTA1 has no deallocation-barrier fence.

fence_mbarrier_init_cluster()
# instruction_selection: `fence.mbarrier_init.release.cluster`; extent: source
#   line-773 publication after all deferred construction. Source `773`; anchor
#   PTX `244`; one-CTA PTX `158`.
if CTA_GROUP == 2:
    cluster_arrive_relaxed()
    # instruction_selection: `barrier.cluster.arrive.relaxed`; extent: exactly
    #   one CTA2 publication arrival. Source `773`; anchor PTX `245`. CTA1 has
    #   no arrive-side `bar.sync`.

prepare_integer_addresses_descriptors_partitions_and_masks()
if CTA_GROUP == 2:
    cluster_wait()
    # instruction_selection: `barrier.cluster.wait`; extent: one delayed wait.
    #   Source `930`; anchor PTX `273-274`.
else:
    cta_sync()
    # instruction_selection: `bar.sync 0`; extent: one delayed singleton wait.
    #   Source `930`; one-CTA PTX `182-183`.

# ===========================================================================
# 4. Warp 5: persistent A/B/SFA/SFB TMA producer
# source 935-1028; anchor PTX 277-594
# ===========================================================================
if warp == TMA_WARP:
    sched = persistent_scheduler((ceil_div(M, CTA_M), ceil_div(N, N_TILE), L),
                                 (cluster_m, cluster_n), grid_dim())
    ab_prod = pipeline_state(AB_STAGES, phase=1, index=0, count=0)
    while sched.valid:
        tile_m, tile_n, tile_l = sched.current()
        reset_count(ab_prod)
        ready = try_wait(AB_EMPTY[ab_prod.index], ab_prod.phase)
        # instruction_selection: `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        #   extent: one speculative initial probe. Source `971-975`; anchor PTX
        #   `366-378`.
        for k_tile in range(ceil_div(K, K_TILE)):  # nounroll body
            wait_if_not_ready(AB_EMPTY[ab_prod.index], ab_prod.phase, ready)
            # instruction_selection: retry loop with
            #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
            #   conditional wait per K tile. Source `979-981`; anchor PTX
            #   `383-399`.
            if elect_one():
                arrive_expect_tx(AB_FULL[ab_prod.index], source_tma_bytes(P))
                # instruction_selection: `mbarrier.arrive.expect_tx.shared.b64`;
                #   extent: one elected transaction count. Source `981-1009`;
                #   anchor PTX `413`.
            if elect_one():
                copy_g2s(A_region(tile_m, k_tile, tile_l),
                         byte_ptr(sA, sA_addr(ab_prod.index)), map_A,
                         AB_FULL[ab_prod.index], a_mask)
            # instruction_selection: `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes[.multicast]...cta_group::2`;
            #   extent: one A region. Source `984-990`; anchor PTX `430`.
            if elect_one():
                copy_g2s(B_region(tile_n, k_tile, tile_l),
                         byte_ptr(sB, sB_addr(ab_prod.index)), map_B,
                         AB_FULL[ab_prod.index], b_mask)
            # instruction_selection: same 3-D TMA family; extent: one B region.
            #   Source `991-997`; anchor PTX `440`.
            if elect_one():
                copy_g2s(SFA_region(tile_m, k_tile, tile_l),
                         byte_ptr(sSFA, sSFA_addr(ab_prod.index)), map_SFA,
                         AB_FULL[ab_prod.index], sfa_mask)
            # instruction_selection: `cp.async.bulk.tensor.4d...multicast...cta_group::2`;
            #   extent: one SFA region. Source `998-1004`; anchor PTX `459`.
            if elect_one():
                copy_g2s(SFB_region(tile_n, k_tile, tile_l),
                         byte_ptr(sSFB, sSFB_addr(ab_prod.index)), map_SFB,
                         AB_FULL[ab_prod.index], sfb_mask)
            # instruction_selection: same 4-D TMA family; extent: one SFB
            #   region. Source `1005-1011`; anchor PTX `473`.
            advance(ab_prod.index, ab_prod.phase, ab_prod.count, AB_STAGES)
            ready = try_wait_next_if_live(AB_EMPTY, ab_prod)
            # instruction_selection: acquire-form parity probe; extent: next
            #   live stage. Source `1013-1017`; anchor PTX `489-504`.
        sched.advance()
    for live_stage in producer_tail(ab_prod):
        wait(AB_EMPTY[live_stage.index], live_stage.phase)
        # instruction_selection: parity retry loop; extent: every live AB
        #   stage. Source `1026-1028`; anchor PTX `514-589`.

# ===========================================================================
# 5. Warp 4: persistent block-scaled tcgen05 MMA
# source 1033-1233; anchor PTX 603-1023
# ===========================================================================
if warp == MMA_WARP:
    named_barrier(id=2, threads=160)
    # instruction_selection: `bar.sync 2,160`; extent: MMA plus four epilogue
    #   warps. Source `1037`; anchor PTX `603`.
    load_smem(TMEM_PTR, tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one pointer word. Source
    #   `1042`; anchor PTX `605`.
    sched = persistent_scheduler(...same work list...)
    ab_cons = pipeline_state(AB_STAGES, phase=0, index=0, count=0)
    acc_prod = pipeline_state(ACC_STAGES, phase=1, index=0, count=0)
    while sched.valid:
        reset_count(ab_cons)
        ready = leader_cta and try_wait(AB_FULL[ab_cons.index], ab_cons.phase)
        # instruction_selection: acquire-form parity probe; extent: leader CTA
        #   initial AB-full stage. Source `1110-1114`; anchor PTX `697-706`.
        if leader_cta:
            wait(ACC_EMPTY[acc_prod.index], acc_prod.phase)
            # instruction_selection: plain parity retry loop; extent: one
            #   accumulator stage. Source `1117-1120`; anchor PTX `712-725`,
            #   wait at `719`.

        sfb_column_shift = 2 * (sched.tile_n & 1) if N_TILE in (64, 192) else 0
        # This scalar shift is present for the N64/N192 source branches. The
        # source's N192 multi-tile AB12 defect is frozen as an excluded mode.
        accumulate = false
        for k_tile in range(ceil_div(K, K_TILE)):
            if leader_cta:
                wait_if_not_ready(AB_FULL[ab_cons.index], ab_cons.phase, ready)
                # instruction_selection: parity retry loop; extent: one AB-full
                #   stage. Source `1149-1152`; anchor PTX `741-759`, wait at
                #   `751`.
                for sf_chunk in source_sfa_chunks(P):
                    copy_s2t(sfa_desc_for(ab_cons.index, sf_chunk),
                             tmem_sfa_addr(sf_chunk))
                    # instruction_selection: `tcgen05.cp.cta_group::2.32x128b.warpx4`;
                    #   extent: anchor four SFA issues. Source `1164-1168`;
                    #   anchor PTX `770-794`.
                for sf_chunk in source_sfb_chunks(P):
                    copy_s2t(sfb_desc_for(ab_cons.index, sf_chunk),
                             tmem_sfb_addr(sf_chunk, sfb_column_shift))
                    # instruction_selection: same family; extent: anchor eight
                    #   SFB issues. Source `1169-1173`; anchor PTX `805-863`.
                for kblock in range(4):
                    gemm(tmem_acc_addr(acc_prod.index),
                         a_desc_for(ab_cons.index, kblock), tmem_sfa_addr(kblock),
                         b_desc_for(ab_cons.index, kblock), tmem_sfb_addr(kblock),
                         accumulate)
                    # instruction_selection:
                    #   `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16`;
                    #   extent: four elected MMAs per anchor K tile, with clear
                    #   predicate only on the first. Source `1175-1205`; anchor
                    #   PTX `886,905,921,940`.
                    accumulate = true
                release(AB_EMPTY[ab_cons.index], cta_mask=ab_consumer_mask(P))
                # instruction_selection:
                #   `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`;
                #   extent: one AB-stage release. Source `1207-1208`; anchor PTX
                #   `951`.
            advance(ab_cons.index, ab_cons.phase, ab_cons.count, AB_STAGES)
            ready = leader_cta and try_wait_next_if_live(AB_FULL, ab_cons)
            # instruction_selection: acquire parity probe; extent: next live
            #   stage. Source `1210-1215`; anchor PTX `958-969`.
        if leader_cta:
            commit(ACC_FULL[acc_prod.index], cta_mask=acc_full_mask)
            # instruction_selection:
            #   `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`;
            #   extent: one accumulator publication. Source `1218-1222`; anchor
            #   PTX `987`.
        advance(acc_prod.index, acc_prod.phase, acc_prod.count, ACC_STAGES)
        sched.advance()
    producer_tail_wait_accumulator_empty(acc_prod)
    # instruction_selection: parity retry loop on leader CTA; extent: source
    #   producer tail, including the wait instruction. Source `1230-1233`;
    #   anchor PTX `1009-1023` (`1015` is the parity wait).

# ===========================================================================
# 6. Warps 0-3: paired AB12 store, SwiGLU, optional SFC/amax, C store
# source 1237-1708; anchor PTX 1025-1921; SFC PTX 1035-2071
# ===========================================================================
if warp in EPI_WARPS:
    if warp == 0:
        alloc_tmem(columns=512, dst=TMEM_PTR)
        # instruction_selection: `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32`;
        #   extent: one 512-column allocation. Source `1241`; anchor PTX `1035`.
    named_barrier(id=2, threads=160)
    # instruction_selection: `bar.sync 2,160`; extent: pointer publication.
    #   Source `1246`; anchor PTX `1038`.
    load_smem(TMEM_PTR, tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one pointer word. Source
    #   `1251`; anchor PTX `1040`.

    sched = persistent_scheduler(...same work list...)
    acc_cons = pipeline_state(ACC_STAGES, phase=0, index=0, count=0)
    if GENERATE_SFC:
        norm = load_global(norm_const)
        # instruction_selection: `ld.global.b32`; extent: one loop-invariant
        #   normalization scalar. Source `1305-1307`; SFC PTX `1114`.
    while sched.valid:
        wait(ACC_FULL[acc_cons.index], acc_cons.phase)
        # instruction_selection: parity retry loop; extent: one accumulator
        #   tile. Source `1364`; anchor PTX `1124-1142`, wait at `1133`.
        if GENERATE_AMAX:
            thread_amax = 0.0f
        for pair in range(source_accumulator_subtiles(P) // 2):
            up_subtile = 2 * pair
            gate_subtile = up_subtile + 1
            subtile_cnt = source_accumulator_subtiles(P)
            num_prev_subtiles = sched.tiles_executed * subtile_cnt
            # Preserve the source's two different AB12 stage expressions.  Do
            # not repair them by feeding the R2S stage to the TMA read.
            ab12_write_stage = (num_prev_subtiles + pair) % AB12_STAGES
            ab12_tma_stage = pair % AB12_STAGES
            c_stage = (num_prev_subtiles + pair) % C_STAGES
            copy_t2r(tmem_acc_subtile(acc_cons.index, up_subtile), rUp)
            # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x32.b32`;
            #   extent: one up fragment. Source `1378-1385`; anchor PTX `1177`.
            copy_t2r(tmem_acc_subtile(acc_cons.index, gate_subtile), rGate)
            # instruction_selection: same TMEM-load family; extent: one gate
            #   fragment. Source `1378-1385`; anchor PTX `1195`.
            for i in range(0, len(rUp), 2):
                rUp[i:i+2] = mul_packed_f32x2(rUp[i:i+2], (alpha, alpha))
                rGate[i:i+2] = mul_packed_f32x2(rGate[i:i+2], (alpha, alpha))
                # instruction_selection: `mul.f32x2`; extent: 16 packed
                #   two-lane issues for each 32-value up/gate fragment,
                #   independent of vector_f32. Source `1392-1399`; anchor PTX
                #   `1213-1245`; scalar-F32 PTX `1213-1262`.

            cast(rUp, rAB12, ab12_dtype)
            copy_r2s(
                rAB12,
                byte_ptr(sAB12, sAB12_addr(ab12_write_stage, pair, 0)),
            )
            # instruction_selection: dtype-specific `cvt` plus shared vector
            #   stores; extent: one 128x32 up half. Source `1409-1414`; anchor
            #   PTX `1252-1294`.
            cast(rGate, rAB12, ab12_dtype)
            copy_r2s(
                rAB12,
                byte_ptr(sAB12, sAB12_addr(ab12_write_stage, pair, 1)),
            )
            # instruction_selection: dtype-specific `cvt` plus shared vector
            #   stores; extent: one 128x32 gate half. Source `1415-1420`;
            #   anchor PTX `1295-1350`.
            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta`; extent:
            #   AB12 visibility. Source `1423`; anchor PTX `1352`.
            named_barrier(id=1, threads=128)
            # instruction_selection: `bar.sync 1,128`; extent: four epilogue
            #   warps before AB12 TMA. Source `1427`; anchor PTX `1354`.
            if warp == 0:
                copy_s2g(byte_ptr(sAB12, sAB12_addr(ab12_tma_stage, pair, 0)),
                         AB12_region(sched.current(), pair), map_AB12)
                # instruction_selection:
                #   `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group...`;
                #   extent: one 128x64 AB12 region. Source `1431-1436`; anchor
                #   PTX `1360`.
                tma_commit_group()
                # instruction_selection: `cp.async.bulk.commit_group`; extent:
                #   one AB12 store group. Source `1438`; anchor PTX `1362`.
                tma_wait_group(AB12_STAGES - 1)
                # instruction_selection: `cp.async.bulk.wait_group.read 3`;
                #   extent: protect the four-stage AB12 ring. Source `1439`;
                #   anchor PTX `1364`.
            named_barrier(id=1, threads=128)
            # instruction_selection: `bar.sync 1,128`; extent: AB12 stage
            #   reuse. Source `1440`; anchor PTX `1391`.

            if vector_f32:
                for i in range(0, len(rGate), 2):
                    neg_log2e_gate = mul(rGate[i:i+2], -1.4426950408889634)
                    denominator = add(exp2(neg_log2e_gate), 1.0f)
                    sigmoid = reciprocal(denominator)
                    rCompute[i:i+2] = mul(mul(sigmoid, rGate[i:i+2]), rUp[i:i+2])
                    # instruction_selection: packed `mul.rn.f32x2` at all
                    #   three multiply sites, two scalar
                    #   `ex2.approx.ftz.f32`, two `add.f32`, two
                    #   `rcp.approx.ftz.f32`; extent:
                    #   two FP32 lanes per unrolled step. Source `1444-1470`;
                    #   anchor PTX `1395-1697`.
            else:
                for i in range(len(rGate)):
                    denominator = add(exp2(mul(rGate[i], -1.4426950408889634)), 1.0f)
                    rCompute[i] = mul(rUp[i], mul(rGate[i], reciprocal(denominator)))
                    # instruction_selection: scalar `mul.f32`,
                    #   `ex2.approx.ftz.f32`, `add.f32`, `rcp.approx.ftz.f32`;
                    #   extent: one unrolled FP32 value. Source `1471-1474`;
                    #   scalar PTX source locs 1474/24, PTX `1229-1744`.

            if GENERATE_AMAX:
                subtile_amax = reduce_max(abs(rCompute), nan_semantics="source")
                thread_amax = max(thread_amax, subtile_amax)
                # instruction_selection: `abs.f32`, `max.NaN.f32` register
                #   reduction, then running `max.f32`; extent: one compute
                #   fragment. Source `1477-1489`; anchor PTX `1699-1750`.

            if GENERATE_SFC:
                for group in groups_of(rCompute, sf_vec_size):
                    raw_scale[group] = mul(
                        mul(reduce_max(abs(group), nan_semantics="source"),
                            reciprocal_dtype_limit(c_dtype)),
                        norm)
                    # instruction_selection: `abs.f32`, `max.NaN.f32`, and
                    #   scalar/paired `mul.f32`; extent: one SF vector. Source
                    #   `1508-1560`; SFC PTX `1702-1785`.
                if pair == 3:
                    rSFC_store_full = cast(raw_scale.full_register, sf_dtype)
                    store_global_u32(
                        pack_sfc_word(rSFC_store_full, 0),
                        SFC_region_word(sched.current(), 0),
                    )
                    store_global_u32(
                        pack_sfc_word(rSFC_store_full, 1),
                        SFC_region_word(sched.current(), 1),
                    )
                    # instruction_selection: full-register FP32 -> SF
                    #   conversion followed by exactly two scalar
                    #   `st.global.b32` packed-scale stores. Source
                    #   `1562-1574`; SFC PTX `1794-1808`.

                # This second full-register conversion is unconditional on
                # every pair and is distinct from the pair-3 store conversion.
                rSFC_quant_full = cast(raw_scale.full_register, sf_dtype)
                rSFC_quant_f32_full = cast(rSFC_quant_full, f32)
                current_scale = select_quarter(rSFC_quant_f32_full, pair)
                # instruction_selection: complete SF conversion and complete
                #   SF -> FP32 upcast on every pair. Source `1577-1584`; SFC
                #   PTX `1811-1838`.
                quant_scale = reciprocal(current_scale)
                quant_scale = mul(norm, quant_scale)
                quant_scale = min_f32(quant_scale, max_f32)
                rCompute = mul(rCompute, quant_scale)
                # instruction_selection: `rcp.approx.ftz.f32`, then
                #   `mul.rn.f32x2` by norm, exactly two inline `min.f32`, then
                #   scalar/paired quantization multiplies. Source `1586-1613`;
                #   SFC PTX `1843-1944`.

            cast(rCompute, rC, c_dtype)
            # instruction_selection: selected output conversion family;
            #   extent: one C fragment. Ordinary source `1616-1622`; anchor PTX
            #   `1751-1767`; FP8-C source `1615`, SFC PTX `1946-1991`.
            copy_r2s(rC, byte_ptr(sC, sC_addr(c_stage, pair)))
            # instruction_selection: shared vector stores; extent: one 128x32
            #   C region. Source `1624-1632`; anchor PTX `1773-1779`.
            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta`; extent:
            #   C visibility. Source `1634`; anchor PTX `1781`.
            named_barrier(id=1, threads=128)
            # instruction_selection: `bar.sync 1,128`; extent: before C TMA.
            #   Source `1638`; anchor PTX `1783`.
            if warp == 0:
                copy_s2g(byte_ptr(sC, sC_addr(c_stage, pair)),
                         C_region(sched.current(), pair), map_C)
                # instruction_selection:
                #   `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group...`;
                #   extent: one C region. Source `1642-1647`; anchor PTX `1792`.
                tma_commit_group()
                # instruction_selection: `cp.async.bulk.commit_group`; extent:
                #   one C store group. Source `1649`; anchor PTX `1794`.
                tma_wait_group(C_STAGES - 1)
                # instruction_selection: source-selected `wait_group.read`;
                #   extent: protect the C ring. Source `1650`; anchor PTX `1796`.
            named_barrier(id=1, threads=128)
            # instruction_selection: `bar.sync 1,128`; extent: C-stage reuse.
            #   Source `1651`; anchor PTX `1799`.

        if GENERATE_AMAX:
            warp_amax = reduce_max(thread_amax, lanes=32, nan_semantics="source")
            # instruction_selection: shuffle/max reduction family; extent: one
            #   warp scalar. Source `1653-1658`; anchor PTX `1813-1822`.
            if lane == 0:
                store_smem(warp_amax, byte_ptr(sAmax, 4 * warp))
                # instruction_selection: `st.shared.b32`; extent: one scalar
                #   per epilogue warp. Source `1659-1661`; anchor PTX `1829`.
            named_barrier(id=1, threads=128)
            # instruction_selection: `bar.sync 1,128`; extent: four-warp amax
            #   publication. Source `1663-1664`; anchor PTX `1835`.
            if warp == 0 and lane == 0:
                block_amax = reduce_max(
                    [load_smem(byte_ptr(sAmax, 4 * i)) for i in range(4)],
                    nan_semantics="source")
                # instruction_selection: four `ld.shared.b32` plus `max.f32`;
                #   extent: one CTA scalar. Source `1666-1671`; anchor PTX
                #   `1839-1853`.
                atomic_max_bits(block_amax, amax)
                # instruction_selection: `atom.global.max.s32`; extent: one
                #   non-negative FP32 bit-pattern atomic. Source `1673-1695`;
                #   anchor PTX `1855`.

        if elect_one():
            if CTA_GROUP == 2:
                leader_empty_address = cluster_mapped_address(
                    ACC_EMPTY[acc_cons.index], leader_rank
                )
                release_cluster(leader_empty_address)
                # instruction_selection:
                #   `mbarrier.arrive.shared::cluster.b64`; extent: one elected
                #   arrival per epilogue warp in each CTA, eight arrivals at
                #   the leader-owned empty stage. Source `1697-1702`; anchor
                #   PTX `1863` (address mapping `1086-1089`).
            else:
                release_local(ACC_EMPTY[acc_cons.index])
                # instruction_selection: `mbarrier.arrive.shared.b64`;
                #   extent: one elected arrival per epilogue warp, four local
                #   arrivals. Source `1697-1702`; one-CTA PTX `1717`.
        advance(acc_cons.index, acc_cons.phase, acc_cons.count, ACC_STAGES)
        sched.advance()

# ===========================================================================
# 7. TMEM teardown and store tails
# source 1710-1720; anchor PTX 1928-1962
# ===========================================================================
if warp in EPI_WARPS:
    if warp == 0:
        relinquish_tmem()
        # instruction_selection:
        #   `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned`;
        #   extent: allocator warp 0 only. Source `1713`; anchor PTX `1928`.
    named_barrier(id=1, threads=128)
    # instruction_selection: `bar.sync 1,128`; extent: epilogue teardown
    #   rendezvous. Source `1714`; anchor PTX `1933`.
    if warp == 0:
        if CTA_GROUP == 2:
            peer_dealloc_address = cluster_mapped_peer_address(TMEM_DEALLOC)
            # Every lane in allocator warp 0 arrives once at the peer CTA's
            # barrier.  Its immediate count is 32, not the CTA-group size.
            arrive_cluster_lane_wise(peer_dealloc_address, lanes=32)
            wait(TMEM_DEALLOC, phase=0)
            dealloc_tmem_cta_group_2(columns=512)
            # instruction_selection: one static
            #   `mbarrier.arrive.shared::cluster.b64` executed by all 32 lanes,
            #   local parity wait, then
            #   `tcgen05.dealloc.cta_group::2.sync.aligned.b32`; extent:
            #   allocator warp 0 only. Source `1715`; anchor PTX `1936-1957`.
        else:
            dealloc_tmem_cta_group_1(columns=512)
            # instruction_selection:
            #   `tcgen05.dealloc.cta_group::1.sync.aligned.b32`; extent:
            #   allocator warp 0 only, without a deallocation mbarrier. Source
            #   `1715`; one-CTA PTX `1763-1779`.
    tma_wait_group(0, pipe="C")
    # instruction_selection: `cp.async.bulk.wait_group.read 0`; extent: drain C
    #   stores. Source `1719`; anchor PTX `1960`.
    tma_wait_group(0, pipe="AB12")
    # instruction_selection: `cp.async.bulk.wait_group.read 0`; extent: drain
    #   AB12 stores. Source `1720`; anchor PTX `1962`.
```

## Pipeline inventory

| pipeline | anchor stages | producer | consumer | full publication | reuse publication |
| --- | ---: | --- | --- | --- | --- |
| AB/SF | 3 | elected lane, warp 5 | leader CTA, warp 4 | one expected-byte mbarrier completed by A/B/SFA/SFB TMA transactions | `tcgen05.commit` to AB-empty after four MMAs |
| accumulator | 1 | leader CTA, warp 4 | four epilogue warps in each CTA | `tcgen05.commit` multicast to the CTA pair | CTA2 maps four elected arrivals per CTA to the leader barrier; CTA1 uses four local arrivals |
| AB12 store | fixed 4 | four epilogue warps write `ab12_write_stage`, warp 0 reads `ab12_tma_stage` | TMA engine | shared fence + named barrier 1 + bulk commit | `wait_group.read 3`, then named barrier 1; retained shapes guarantee the source's distinct stage expressions agree |
| C store | 4 | four epilogue warps stage, warp 0 issues | TMA engine | shared fence + named barrier 1 + bulk commit | `wait_group.read C_STAGES-1`, then named barrier 1 |

AB and accumulator cursors each carry `(index, phase, count)`, with complementary
producer/consumer initial phases. C uses the persistent tile count. Source AB12
R2S uses `(tiles_executed * subtile_cnt + pair) % 4`, while its TMA read uses
`pair % 4`; these remain distinct in the implementation and capability guard.

## TensorMap and tail contract

| descriptor | logical tensor | source PTX rank | region / special behavior |
| --- | --- | ---: | --- |
| A | `(M,K,L)` | 3 | CTA-local part of 128-row A region; FP4 K-major or FP8 selected major; cluster-N multicast |
| B | `(N,K,L)` | 3 | N region; cluster-M multicast; N-major FP8 may require multiple static TMA issues |
| SFA | physical block-scale bytes | 4 | scalar atom coordinates derived from M/K/V; cluster-N multicast |
| SFB | physical block-scale bytes | 4 | N64 slices every two scheduler tiles; N192 has special physical order; cluster-M multicast |
| AB12 | `(M,N,L)` | 3 | 128x64 paired up/gate region; output dtype and major select swizzle and conversion |
| C | `(M,N/2,L)` | 3 | 128x32 output region; output dtype and major select swizzle and conversion |

Ordinary AB12/C paths use TensorMap bounds and OOB fill for aligned M/N/K tails.
SFC is a direct register-to-global path with no source predicate. Retained SFC
specializations require N-major, tile-N 256, and either `N % 256 == 0` or
`N % 256 == 192`; the latter leaves enough rounded physical SF storage for the
two packed scale words and passed N=192/448/704 mathematical probes. Remainders
64/128 are unsafe and produced cross-run PASS/FAIL behavior. M-tail is valid
because physical SF storage rounds M to 128. L is the third scheduler
coordinate. Every contiguous dimension must remain 16-byte aligned.

## Static specialization boundary

There is one builder and one device definition. The accepted compile key is the
frozen `source_capability_manifest.json`, summarized here:

| axis | accepted values | emitted consequence |
| --- | --- | --- |
| A/B | FP4 E2M1; FP8 E4M3/E5M2, matching | element width, TensorMaps, shared descriptors, block-scaled MMA kind |
| SF | FP4: E8M0/V16, E8M0/V32, E4M3/V16; FP8: E8M0/V32 | SF bytes, TMEM columns, block16/block32 instruction descriptor |
| AB12 | FP32, FP16, BF16, FP8 E4M3/E5M2 | register conversion, 128x64 SMEM bytes/swizzle, TensorMap |
| C | FP32, FP16, BF16, FP8 E4M3/E5M2 | ordinary conversion or SFC quant path, 128x32 SMEM bytes/swizzle, TensorMap |
| majors | FP4 A/B K/K; FP8 A M/K and B N/K; AB12/C same M or N | TensorMap strides and raw MMA/shared descriptor fields; SFC only N-major |
| MMA tile | M 128/256; N 64/128/192/256 | CTA_GROUP, stage counts, SF copy count, TMEM SFB shift, epilogue pair count |
| cluster | positive power-of-two axes in `{1,2,4}`, product <=16; M256 needs even cluster-M | launch shape, masks, arrival counts, 1CTA/2CTA protocols |
| vector FP32 | true or false; FP8 input + FP32 AB12 is retained only when true | packed or scalar SwiGLU/SFC arithmetic; frozen serializer cross-guard |
| shape | aligned positive M/N/K, `L` 1/2; N64/N192 require at most one work tile per launched persistent CTA, and N192 additionally requires `N=192` | persistent work and K-loop bounds; ordinary tails; alpha is runtime FP32; preserves source AB12 write/TMA stage equivalence |

The actual source runs retained by this port include outer-API-forbidden FP8
AB12, FP8 C/SFC, and FP4-input plus FP8-output branches. `uint8` packed-FP4 is
explicitly rejected. N64 and N192 have `subtile_cnt % 4 == 2`; patterned-input
probes show their AB12 write/TMA stages diverge once a persistent CTA receives a
second work tile. N192 additionally remains source-correct only for `N=192`:
both N=384 and N=3072 leave AB12 columns unwritten. SFC with tile-N 64/128/192,
M-major output, or N remainder 64/128 is excluded by direct mathematical
validation. FP8 input + FP32 AB12 is retained only for `vector_f32=True`.

For the N64/N192 persistence guard, host specialization computes
`problem_clusters = ceil(ceil(M/128)/cluster_m) *
ceil(ceil(N/N_TILE)/cluster_n) * L` and requires it to equal the source launch's
`min(problem_clusters, max_active_clusters)`. The patterned 91-mode matrix has
75 PASS, 10 expected FAIL, and six explicitly unsafe SFC modes (three trials
each, every mode reproducing at least one failure); its SHA256 is
`5a72803105f9a1a559924a9543fa24b578e9e74a362467f22be8753befcb48ce`.

## Writer PTX evidence

| specialization | line-info PTX | SHA256 | decisive evidence |
| --- | --- | --- | --- |
| anchor 2CTA/amax | `writer_export/anchor/anchor_lineinfo.ptx` | `0140539cfe79c4af52558077cffd8acbcee7837bae97b0b7dfd87320345da02e` | 12 SF copies, four 2CTA MXFP4 MMAs, paired AB12/C stores, amax atomic, 2CTA teardown |
| SFC/FP8 C | `writer_export/sfc_fp8_c/sfc_fp8_c_lineinfo.ptx` | `88a00fa0b142c38ae091c02650710f8488cd954285a7d88cf89e58be1f18ca37` | full-register scale conversions, two scalar packed SFC stores, reciprocal/norm/`min.f32` ordering, FP8 C casts |
| FP8 input | `writer_export/fp8_input/fp8_input_lineinfo.ptx` | `780df4af67f3280510324a1c1f138ce98fa8897705493dac233a44a89e91ec67` | FP8 TensorMaps, block32 MMA descriptor, three static SF copies |
| 1CTA | `writer_export/one_cta/one_cta_lineinfo.ptx` | `964e208b24e68b0e6d5696206f6404d58c2784e0adf6c89451780ff45a04f541` | CTA-group-1 alloc/copy/MMA/commit/dealloc and singleton barriers |
| scalar FP32 | `writer_export/scalar_f32/scalar_f32_lineinfo.ptx` | `45d13b7622397631d0b66e6fb71cd75b72b5b97e2d44667a99ce332f9af5da6b` | scalar helper `.loc 24` exp2/reciprocal path |
| N64 | `writer_export/n64/n64_lineinfo.ptx` | `e9ab3934a907675d7f4c059432af067be7f30355981c11cf5bc1a8f422ad6e64` | four SF copies, alternating SFB shift, two accumulator stages, distinct dynamic AB12 write and fixed TMA stage expressions |
| N192 single tile | `writer_export/n192/n192_lineinfo.ptx` | `c3c2b3a16f489c3e6b3c6e8f64648bc91a6bad329dc09f08284b46a0a7db6218` | twelve SF copies, N192 SFB rule, three C stages; only N=192 without persistent CTA reuse is retained |
| FP8 AB12 / FP32 C casts | `writer_export/output_casts/output_casts_lineinfo.ptx` | `fb3b08f2199a32b7aa0b9834200f6c9ee8ba29ec80073c1f0b2bc265afb31835` | FP8 AB12 conversion and FP32 C store families |

Static counts use instruction lines minus predicated lines. They are evidence for
the explicit loops above, not hidden computation. Every key operation in the
sketch maps in both directions through `writer_export/ptx_evidence.json`:
sketch section -> cited source line -> `.loc` PTX line, and every recorded key
PTX opcode belongs to one cited section.

## Executable module and validation contract

- Registry name is
  `cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant`,
  category `cudnn`, compute capability 10.
- Device code imports only `tirx_kernels.kern as K`, declares one rank-1 dynamic
  `u8` arena, and uses only integer offsets/strides/swizzle bits/raw descriptors.
- The fixed pointer ABI is `A/B/SFA/SFB/C/AB12/amax/SFC/norm_const/alpha`.
- The module exports `KERNEL_META`, deterministic pairwise `CONFIGS`,
  `BENCH_CONFIGS`, `get_kernel`, `prepare_data`, `run_test`, `prepare_bench`,
  `run_gpu`, and `run_bench`.
- Correctness compares independent TIRx/source outputs and a mathematical
  reference: 0.01 for ordinary AB12/C/SwiGLU, 0.1 for FP8/SFC/dequantized C and
  amax. `prepare_bench` remains CPU-only; reference compilation is lazy.
- Final performance must use one fresh, complete, unspliced five-round
  bench-suite result and require `mean(cudnn_frontend) / mean(tirx) > 0.99` on
  every retained row. The frozen capability exclusions cannot be weakened in
  performance tuning.
