<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 persistent dense GEMM + interleaved SwiGLU sketch

This is a non-executable execution sketch for the single parameterized TIRx
module
[`tirx_kernels/cudnn/swiglu/dense_gemm_persistent_swiglu.py`](../../tirx_kernels/cudnn/swiglu/dense_gemm_persistent_swiglu.py).
It freezes the source kernel's storage, six-warp role split, persistent
scheduler, asynchronous protocols, dense MMA mainloop, paired accumulator
epilogue, three-output TMA store sequence, and teardown. After the independent
reviewer returns PASS this file is immutable.

The source is
`python/cudnn/gemm/cutedsl/dense/swiglu/dense_gemm_persistent_swiglu.py` at
commit `7b5327b32907b9dd21d85a393d62f9573d7f0116`. Source citations below refer to
that file. The writer's primary line-info PTX is under
`.porting/dense_gemm_persistent_swiglu/writer_ptx/anchor/`, SHA256
`cf83547eb988fbbce454d08f13daaedf67addf2f175861d17f65d5e699a15587`.
`PTX n` refers to a numbered line in that artifact.

The instruction-annotated body fixes this primary specialization:

| axis | anchor value |
| --- | --- |
| problem | `M=N=K=1024`, `L=1`, `alpha=1.0` |
| A/B | BF16, K-major/K-major |
| accumulator | FP32 |
| AB12 / C | BF16 / BF16, N-major |
| MMA tile / cluster | `(256,256)` / `(2,2)` |
| derived stages | A/B 5, accumulator 2, AB12 4, C 2 |

The same sketch covers every compile-time branch in **Static specialization
boundary**. Those branches change integer constants, raw descriptor bits,
instruction kinds, register packing, stage counts, and predicates; they never
create another kernel or role.

## No first-class layout invariant

Neither this sketch nor the executable kernel may construct, pass, return, or
store a first-class layout value. Source mappings are resolved before device
tracing into integer extents, byte strides, byte offsets, XOR swizzles,
TensorMap fields, and raw UMMA descriptor bits. Every SMEM object is a slice of
one rank-1 `u8` arena. Every TMEM object is addressed with scalar row/column
integers. Every helper below accepts and returns only scalars, raw pointers, or
opaque hardware descriptors. “Tile” below means an algorithmic rectangle, not
a tile primitive or a layout-bearing object.

The executable device body must use only `import tirx_kernels.kern as K` and
the `K` language surface. Direct TVM script namespaces, index-map helpers,
tile primitives, and the tile namespace are forbidden.

## Pipeline at a glance

| role | persistent work | publication and reuse edges |
| --- | --- | --- |
| warp 5, TMA | prefetch four TensorMaps; walk output tiles and K blocks; acquire an empty A/B stage and issue one A plus one B TMA load | A/B empty -> `arrive.expect_tx` -> two loads -> A/B full completion; producer tail waits for all stages |
| warp 4, MMA | retrieve TMEM pointer; walk the identical work list; acquire an accumulator stage; consume every A/B stage; issue dense UMMA; publish accumulator | A/B full -> MMA -> UMMA commit releases A/B empty; accumulator empty -> all K blocks -> UMMA commit publishes accumulator full |
| warps 0-3, epilogue | warp 0 allocates 512 TMEM columns; all four retrieve the pointer; consume paired accumulator subtiles; compute scaled AB12 and interleaved SwiGLU C; stage two AB12 blocks plus one C block; warp 0 launches three TMA stores | accumulator full -> two TMEM loads; named barrier 1 brackets each SMEM/store group; `wait_group.read 3` protects four-stage AB12 reuse; elected release publishes accumulator empty |
| whole CTA/cluster | initialize A/B and accumulator barriers in separate publication epochs; initialize CTA-group-2 teardown barrier; perform a third publication arrive and delayed wait | two pipeline publication fence/arrive/wait sequences plus the source's explicit third fence/arrive and later wait; named barrier 2 publishes TMEM pointer to five warps |

The three role-local scheduler cursors traverse exactly the same deterministic
static work list. They communicate only through the two mbarrier pipelines.
Source `690-759`, `764-878`, `970-1137`; anchor PTX `319-585`, `630-875`,
`1022-1570`.

## Primitive vocabulary

Structural pseudo-operations do not move data and never carry a first-class
layout:

```python
specialize_static(...)
launch(grid, block_threads, cluster, dynamic_smem_bytes)
tensor_map(base_ptr, rank, extents, byte_strides, box_extents, element_strides,
           element_dtype, interleave, swizzle, oob_fill)
smem_bytes(base, scalar_offset, scalar_count)
tmem_address(base_column, scalar_row, scalar_column)
raw_umma_descriptor(smem_ptr, leading_byte_delta, stride_byte_delta,
                    base_offset, swizzle_mode)
persistent_cursor(problem_tiles, cluster_shape, grid_shape)
```

Directional movement and synchronization remain explicit:

```python
tma_load(map, coordinates, smem_ptr, full_mbarrier, multicast_mask)
tma_store(map, coordinates, smem_ptr)
tmem_load(tmem_addr, register_words, packed_16b)
register_store_shared(register_words, smem_ptr)
umma_dense(acc_tmem, a_desc, b_desc, accumulate, instruction_kind, cta_group)
init_mbarrier(ptr, arrivals)
try_wait_mbarrier(ptr, phase)
wait_mbarrier(ptr, phase, token)
arrive_expect_tx(ptr, byte_count)
umma_commit(ptr, multicast_mask)
fence_mbarrier_init_cluster(); cluster_arrive(); cluster_wait(); cta_sync()
fence_async_shared_cta(); named_barrier(id, threads)
tma_store_commit(); tma_store_wait(pending)
alloc_tmem(columns, cta_group); dealloc_tmem(columns, cta_group)
```

## Complete sketch

```python
# ==========================================================================
# 1. Static specialization, runtime ABI, and launch
# source 174-292, 295-484; PTX 14-42
# ==========================================================================
P = specialize_static(
    M, N, K, L,
    ab_dtype, acc_dtype, ab12_dtype, c_dtype,
    a_major, b_major, output_major,
    tile_m, tile_n, cluster_m, cluster_n,
)
assert source_legality_predicates(P)

CTA_GROUP = 1 if P.tile_m == 128 else 2
CTA_M = 128
CTA_N = P.tile_n
K_TILE = source_instruction_k(P.ab_dtype)
WARPS = 6
EPI_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
ACC_STAGES = 2
AB12_STAGES = 4
C_STAGES = 2
AB_STAGES = source_capacity_stage_count(P, smem_capacity=232448)

# Runtime ABI. Maps are opaque encoded TensorMap objects. Tensor pointers are
# also retained for correctness/reference wrappers; the device consumes maps.
ABI = (
    A_ptr, B_ptr, AB12_ptr, C_ptr,
    map_A, map_B, map_AB12, map_C,
    alpha_f32,
)

problem_tiles = (ceil_div(M, CTA_M), ceil_div(N, CTA_N), L)
grid = source_persistent_cluster_grid(problem_tiles, P.cluster, max_active_clusters)
launch(
    grid=grid,
    block_threads=192,
    cluster=(P.cluster_m, P.cluster_n, 1),
    arch="sm_100a",
    dynamic_smem_bytes=source_shared_bytes(P),
)
# instruction_selection: `.reqntid 192,1,1` and `.extern .shared .align
#   1024`; extent: the entire launch specialization. Source `295-484`; PTX
#   `14,33`. There is deliberately no minimum-block launch bound: source
#   `occupancy=1` only selects AB_STAGES and writer PTX has no `.minnctapersm`.

# ==========================================================================
# 2. Rank-1 storage, scalar address formulas, and raw descriptors
# source 174-292, 541-676, 1156-1502; PTX 116-249
# ==========================================================================
AB_FULL = smem_bytes(smem, 0, 8 * AB_STAGES)
AB_EMPTY = smem_bytes(smem, 8 * AB_STAGES, 8 * AB_STAGES)
ACC_FULL = smem_bytes(smem, 16 * AB_STAGES, 8 * ACC_STAGES)
ACC_EMPTY = smem_bytes(smem, 16 * AB_STAGES + 8 * ACC_STAGES,
                       8 * ACC_STAGES)
TMEM_DEALLOC = smem_bytes(smem, source_tmem_dealloc_offset(P), 8)
TMEM_PTR = smem_bytes(smem, source_tmem_ptr_offset(P), 4)

# Anchor byte map after the 1024-byte protocol header. These are four linear
# byte intervals, even though source syntax presents multidimensional tensors:
#   sAB12: offset 1024,   4 * 8192 bytes, ends 33792
#   sC:    offset 33792,  2 * 8192 bytes, ends 50176
#   sA:    offset 50176,  5 * 16384 bytes, ends 132096
#   sB:    offset 132096, 5 * 16384 bytes, ends 214016
sAB12 = smem_bytes(smem, source_ab12_offset(P), source_ab12_bytes(P))
sC = smem_bytes(smem, source_c_offset(P), source_c_bytes(P))
sA = smem_bytes(smem, source_a_offset(P), source_a_bytes(P))
sB = smem_bytes(smem, source_b_offset(P), source_b_bytes(P))

# Each function below is scalar integer arithmetic over lane, stage, logical
# row/column, dtype size, and compile-time major. XOR terms reproduce the
# source swizzle. No mapping object is produced.
def sA_byte_offset(stage, row, kk):
    return source_a_stage_stride(P) * stage + source_a_swizzled_offset(P, row, kk)

def sB_byte_offset(stage, col, kk):
    return source_b_stage_stride(P) * stage + source_b_swizzled_offset(P, col, kk)

def sAB12_byte_offset(stage, row, col):
    return source_ab12_stage_stride(P) * stage + source_output_swizzled_offset(P, row, col)

def sC_byte_offset(stage, row, col):
    return source_c_stage_stride(P) * stage + source_output_swizzled_offset(P, row, col)

def a_desc(stage, kphase):
    return raw_umma_descriptor(
        sA + sA_byte_offset(stage, source_a_row(P), source_a_k(P, kphase)),
        source_a_ld_bytes(P), source_a_sd_bytes(P),
        source_a_desc_base(P), source_a_swizzle_mode(P),
    )

def b_desc(stage, kphase):
    return raw_umma_descriptor(
        sB + sB_byte_offset(stage, source_b_col(P), source_b_k(P, kphase)),
        source_b_ld_bytes(P), source_b_sd_bytes(P),
        source_b_desc_base(P), source_b_swizzle_mode(P),
    )

# Every mode allocates exactly the source-computed accumulator columns. The
# anchor folds this specialization function to 512 columns; other tile/dtype
# modes may differ. Scalar column arithmetic selects stage and epilogue
# subtile; CTA_GROUP selects local versus paired-CTA addressing.
TMEM_COLUMNS = source_num_tmem_alloc_cols(P)
def acc_tmem(tmem_base, stage, row, col):
    return tmem_address(tmem_base + source_acc_stage_column(P, stage) + col,
                        row, source_acc_column(P, row, col))

# instruction_selection: raw UMMA shared descriptors are 64-bit integer
#   operands and TMEM addresses are 32-bit integers; extent: all A/B K phases
#   and accumulator stages. Source `587-676,1156-1502`; PTX `700-791,1052-1077`.

# ==========================================================================
# 3. Coordinates, descriptor prefetch, and barrier publication
# source 514-605, 678-684; PTX 82-249
# ==========================================================================
warp = warp_uniform(warp_id())
lane = lane_id()
cluster_rank = block_idx_in_cluster()
mma_tile_coord_v = block_idx_x() % CTA_GROUP
leader_cta = mma_tile_coord_v == 0
cluster_m_coord = cluster_rank % P.cluster_m
cluster_n_coord = (cluster_rank // P.cluster_m) % P.cluster_n

if warp == TMA_WARP:
    prefetch(map_A)
    prefetch(map_B)
    prefetch(map_AB12)
    prefetch(map_C)
    # instruction_selection: four `prefetch.tensormap`; extent: one per map.
    #   Source `517-524`; reviewer PTX `98,101,104,107`, with `.loc 1
    #   521-524`.

# The anchor has full offsets 0..32 with arrival 1, empty offsets 40..72 with
# arrival 2, accumulator-full offsets 80/88 with arrival 1, accumulator-empty
# offsets 96/104 with arrival 8, teardown offset 112, pointer offset 120.
if warp == 0 and elect_one():
    for stage in range(AB_STAGES):
        init_mbarrier(AB_FULL + 8 * stage, arrivals=1)
if warp == 0 and elect_one():
    for stage in range(AB_STAGES):
        init_mbarrier(AB_EMPTY + 8 * stage,
                      arrivals=source_ab_empty_arrivals(P))
# instruction_selection: ten anchor `mbarrier.init.shared.b64`; extent: five
#   full plus five empty stages. Source `549-560`; PTX `138-142,152-156`.

fence_mbarrier_init_cluster()
if P.cluster_m * P.cluster_n > 1:
    cluster_arrive(); cluster_wait()
else:
    cta_sync()
# instruction_selection: exact sequence
#   `fence.mbarrier_init.release.cluster`,
#   `barrier.cluster.arrive.relaxed`, `barrier.cluster.wait` (or singleton
#   `bar.sync`); extent: AB pipeline publication. Source `549-560`; PTX
#   `162-164`.

if warp == 0 and elect_one():
    for stage in range(ACC_STAGES):
        init_mbarrier(ACC_FULL + 8 * stage, arrivals=1)
if warp == 0 and elect_one():
    for stage in range(ACC_STAGES):
        init_mbarrier(ACC_EMPTY + 8 * stage,
                      arrivals=8 if CTA_GROUP == 2 else 4)
# instruction_selection: four `mbarrier.init.shared.b64`; extent: two full and
#   two empty stages. Source `562-572`; PTX `173-174,186-187`.

fence_mbarrier_init_cluster()
if P.cluster_m * P.cluster_n > 1:
    cluster_arrive(); cluster_wait()
else:
    cta_sync()
# instruction_selection: exact sequence
#   `fence.mbarrier_init.release.cluster`,
#   `barrier.cluster.arrive.relaxed`, `barrier.cluster.wait` (or singleton
#   `bar.sync`); extent: accumulator pipeline publication. Source `562-572`;
#   PTX `202-204`.

if CTA_GROUP == 2 and warp == TMA_WARP and elect_one():
    init_mbarrier(TMEM_DEALLOC, arrivals=32)
# instruction_selection: one `mbarrier.init.shared.b64` at anchor offset 112;
#   extent: only two-CTA modes. Source `574-580`; PTX `213`.

fence_mbarrier_init_cluster()
if P.cluster_m * P.cluster_n > 1:
    cluster_arrive()
# instruction_selection: exact
#   `fence.mbarrier_init.release.cluster` then
#   `barrier.cluster.arrive.relaxed`; extent: explicit source publication.
#   Source `580-585`; PTX `218,220`.

# Scalar tensor coordinates, TensorMap coordinates, multicast masks, scheduler
# divisions, and raw descriptor constants are prepared while that third
# cluster synchronization is in flight.
prepare_scalar_coordinates_masks_and_descriptors()

if P.cluster_m * P.cluster_n > 1:
    cluster_wait()
else:
    cta_sync()
# instruction_selection: delayed exact `barrier.cluster.wait` or CTA
#   `bar.sync`;
#   extent: one prologue synchronization before all roles. Source `678-684`;
#   PTX `249`.

# A multicast follows equal M coordinates. B multicast follows equal N
# coordinates. CTA_GROUP=2 participates in the same source masks. Each mask is
# an ordinary 16-bit integer made by setting the corresponding cluster ranks.
a_multicast_mask = source_a_multicast_mask(P, cluster_m_coord, cluster_n_coord)
b_multicast_mask = source_b_multicast_mask(P, cluster_m_coord, cluster_n_coord)
acc_multicast_mask = source_accumulator_multicast_mask(P, cluster_rank)

# ==========================================================================
# 4. Warp 5: persistent A/B TMA producer
# source 687-759; PTX 319-585
# ==========================================================================
if warp == TMA_WARP:
    sched = persistent_cursor(problem_tiles, P.cluster, grid)
    ab_prod = producer_state(stages=AB_STAGES, index=0, phase=1, count=0)
    while sched.valid:
        tile_m, tile_n, batch = source_tile_coordinates(sched, CTA_GROUP)
        ab_prod.count = 0
        token = True
        if ab_prod.count < k_block_count:
            token = try_wait_mbarrier(AB_EMPTY + 8 * ab_prod.index,
                                      ab_prod.phase)
            # instruction_selection: fast probe
            #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            #   extent: initial empty-stage probe per output tile. Source
            #   `716-720`; PTX `345`.
        for k_block in range(k_block_count):
            wait_mbarrier(AB_EMPTY + 8 * ab_prod.index, ab_prod.phase, token)
            # instruction_selection: a ready token skips the fallback;
            #   otherwise `mbarrier.try_wait.parity.shared.b64` loops with the
            #   source timeout. Extent: one empty-stage acquire per K block.
            #   Source `724-726`; PTX `364`.
            if leader_cta and elect_one():
                arrive_expect_tx(AB_FULL + 8 * ab_prod.index,
                                 source_a_tile_bytes(P) + source_b_tile_bytes(P))
                # instruction_selection:
                #   `mbarrier.arrive.expect_tx.shared.b64`; extent: one
                #   elected leader-CTA transaction per K block; nonleader CTAs
                #   branch around only this operation. Source `724-742`; PTX
                #   predicate/election/arrival `374-383`.
            if elect_one():
                tma_load(map_A, (source_a_coord(P, tile_m, k_block, batch)),
                         sA + source_a_stage_stride(P) * ab_prod.index,
                         AB_FULL + 8 * ab_prod.index, a_multicast_mask)
            if elect_one():
                tma_load(map_B, (source_b_coord(P, tile_n, k_block, batch)),
                         sB + source_b_stage_stride(P) * ab_prod.index,
                         AB_FULL + 8 * ab_prod.index, b_multicast_mask)
                # instruction_selection: A is
                #   `cp.async.bulk.tensor.3d.shared::cluster.global.tile.
                #   mbarrier::complete_tx::bytes.multicast::cluster.
                #   L2::cache_hint.cta_group::2`; B is the same exact family
                #   without `multicast::cluster`. Extent: one A and one B load
                #   at two independent election sites per K block. Source
                #   `728-742`; PTX election/load pairs `391/398` and
                #   `401/409`; expect-tx has its own election at `378`.
            advance(ab_prod)
            token = True
            if ab_prod.count < k_block_count:
                token = try_wait_mbarrier(AB_EMPTY + 8 * ab_prod.index,
                                          ab_prod.phase)
                # instruction_selection: next-stage fast probe uses
                #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                #   extent: one conditional probe per non-final K block.
                #   Source `744-748`; PTX `426`.
        sched.advance()
    producer_tail_wait_all_empty(AB_EMPTY, ab_prod, AB_STAGES)
    # instruction_selection: parity try-wait loops; extent: all five anchor
    #   empty stages before warp exit. Source `750-759`; PTX
    #   `476,499,522,545,568` only; `426` is the preceding steady-state probe.

# ==========================================================================
# 5. Warp 4: persistent dense UMMA consumer/accumulator producer
# source 761-878; PTX 630-875
# ==========================================================================
if warp == MMA_WARP:
    named_barrier(id=2, threads=160)
    # instruction_selection: `bar.sync 2,160`; extent: pointer publication to
    #   MMA plus four epilogue warps. Source `766-781`; PTX `585`.
    tmem_base = load_u32(TMEM_PTR)
    # instruction_selection: `ld.shared.b32`; extent: one TMEM-pointer word
    #   loaded by every participating MMA-warp thread after the named barrier.
    #   Source `766-781`; PTX `585-587`, load at `587`.
    sched = persistent_cursor(problem_tiles, P.cluster, grid)
    ab_cons = consumer_state(stages=AB_STAGES, index=0, phase=0, count=0)
    acc_prod = producer_state(stages=ACC_STAGES, index=0, phase=1, count=0)
    while sched.valid:
        ab_cons.count = 0
        ab_token = True
        if ab_cons.count < k_block_count and leader_cta:
            ab_token = try_wait_mbarrier(AB_FULL + 8 * ab_cons.index,
                                         ab_cons.phase)
            # instruction_selection: fast probe
            #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            #   extent: leader's initial A/B-full probe per output tile.
            #   Source `807-812`; PTX `656`.
        if leader_cta:
            wait_empty_accumulator(ACC_EMPTY + 8 * acc_prod.index,
                                   acc_prod.phase)
            # instruction_selection: accumulator-empty fallback is
            #   `mbarrier.try_wait.parity.shared.b64` with timeout loop;
            #   extent: leader CTA once per output tile. Source `814-817`;
            #   PTX `673`.
        accumulate = False
        for k_block in range(k_block_count):
            if leader_cta:
                wait_mbarrier(AB_FULL + 8 * ab_cons.index,
                              ab_cons.phase, ab_token)
                # instruction_selection: A/B-full fallback is
                #   `mbarrier.try_wait.parity.shared.b64` with timeout loop;
                #   extent: leader once per K block. Source `827-830`; PTX
                #   `706`.
                for kphase in range(source_kphases(P)):
                    if elect_one():
                        umma_dense(
                            acc_tmem(tmem_base, acc_prod.index, 0, 0),
                            a_desc(ab_cons.index, kphase),
                            b_desc(ab_cons.index, kphase),
                            accumulate=accumulate,
                            instruction_kind=source_mma_kind(P.ab_dtype),
                            cta_group=CTA_GROUP,
                        )
                    accumulate = True
                    # instruction_selection: anchor emits four
                    #   `tcgen05.mma.cta_group::2.kind::f16`; extent: four K
                    #   phases at four independent election sites per BF16 K
                    #   block. Source `827-850`; PTX elections
                    #   `726,744,758,779`, instructions `737,756,772,791`.
                    #   Other instruction kinds are frozen below.
                if elect_one():
                    umma_commit(AB_EMPTY + 8 * ab_cons.index,
                                source_ab_release_mask(P, cluster_rank))
                # instruction_selection: `tcgen05.commit.cta_group::2.
                #   mbarrier::arrive::one.shared::cluster.multicast::cluster.
                #   b64`; extent: one AB release at its independent election
                #   site per K block. Source `852-853`; PTX election `796`,
                #   commit `800`.
            advance(ab_cons)
            ab_token = True
            if ab_cons.count < k_block_count and leader_cta:
                ab_token = try_wait_mbarrier(AB_FULL + 8 * ab_cons.index,
                                             ab_cons.phase)
                # instruction_selection: next-stage fast probe uses
                #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                #   extent: one leader probe per non-final K block. Source
                #   `855-860`; PTX `820`.
        if leader_cta:
            if elect_one():
                umma_commit(ACC_FULL + 8 * acc_prod.index,
                            acc_multicast_mask)
            # instruction_selection: `tcgen05.commit.cta_group::2.
            #   mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`;
            #   extent: one accumulator publication at an independent election
            #   site per output tile. Source `862-867`; PTX election `830`,
            #   commit `834`.
        advance(acc_prod)
        sched.advance()
    if leader_cta:
        for _ in range(ACC_STAGES - 1):
            advance(acc_prod)
        wait_empty_accumulator(ACC_EMPTY + 8 * acc_prod.index,
                               acc_prod.phase)
        # instruction_selection: `mbarrier.try_wait.parity.shared.b64`
        #   timeout loop; extent: only the leader CTA advances once for the
        #   anchor two-stage pipe and waits once for the last-used accumulator
        #   stage. Source `875-878`; PTX state advance `854-857` and the sole
        #   parity-wait loop `875`.

# ==========================================================================
# 6. Warps 0-3: paired accumulator epilogue and three TMA stores
# source 880-1137; PTX 884-1570
# ==========================================================================
if warp < MMA_WARP:
    if warp == 0:
        alloc_tmem(columns=TMEM_COLUMNS, cta_group=CTA_GROUP,
                   pointer_output=TMEM_PTR)
        # instruction_selection: `tcgen05.alloc.cta_group::2.sync.aligned.
        #   shared::cta.b32`;
        #   extent: warp-collective allocation of source_num_tmem_alloc_cols,
        #   anchor 512. Source `884-891,1480-1502`; PTX `895`.
    named_barrier(id=2, threads=160)
    # instruction_selection: `bar.sync 2,160`; extent: one allocation
    #   publication. Source `893-909`; PTX `898`.
    tmem_base = load_u32(TMEM_PTR)
    # instruction_selection: `ld.shared.b32`; extent: one TMEM-pointer word
    #   loaded by every participating epilogue thread after the named barrier.
    #   Source `893-909`; PTX `898-900`, load at `900`.

    sched = persistent_cursor(problem_tiles, P.cluster, grid)
    acc_cons = consumer_state(stages=ACC_STAGES, index=0, phase=0, count=0)
    while sched.valid:
        tile_m, tile_n, batch = source_tile_coordinates(sched, CTA_GROUP)
        wait_full_accumulator(ACC_FULL + 8 * acc_cons.index, acc_cons.phase)
        # instruction_selection: accumulator-full wait is
        #   `mbarrier.try_wait.parity.shared.b64` with timeout loop; extent:
        #   every epilogue warp once per output tile. Source `1017-1026`; PTX
        #   `1040`.

        subtile_count = source_epilogue_subtile_count(P)
        previous_subtiles = sched.executed_tiles * subtile_count
        for subtile in range(0, subtile_count, 2):
            # Paired blocks are `[X_i | G_i]` in the source's N direction.
            tmem_load(source_epilogue_tmem_addr(P, tmem_base, acc_cons.index,
                                                subtile + 1, warp, lane),
                      r_gate, packed_16b=(P.acc_dtype == f16))
            tmem_load(source_epilogue_tmem_addr(P, tmem_base, acc_cons.index,
                                                subtile, warp, lane),
                      r_x, packed_16b=(P.acc_dtype == f16))
            # instruction_selection: anchor uses two
            #   `tcgen05.ld.sync.aligned.32x32b.x32.b32`; extent: the gate and
            #   value block for each subtile pair. Source `1032-1038`; PTX
            #   `1059,1077`. FP16 accumulation uses the frozen packed-x16
            #   family in the specialization table.

            for element in source_thread_epilogue_elements(P):
                x_acc = acc_cast(r_x[element] * alpha_f32, P.acc_dtype)
                gate_acc = acc_cast(r_gate[element] * alpha_f32, P.acc_dtype)
                exp_arg = f32(-gate_acc) * LOG2_E_F32
                exp_value = exp2_approx_ftz_f32(exp_arg)
                denominator_acc = acc_cast(f32(1.0) + exp_value, P.acc_dtype)
                denominator_f32 = f32(denominator_acc)
                reciprocal = rcp_approx_ftz_f32(denominator_f32)
                silu_gate = reciprocal * f32(gate_acc)
                c_value = cast(f32(x_acc) * silu_gate, P.c_dtype)
                ab12_x = cast(x_acc, P.ab12_dtype)
                ab12_gate = cast(gate_acc, P.ab12_dtype)

                r_ab12_0[element] = ab12_x
                r_ab12_1[element] = ab12_gate
                r_c[element] = c_value

            # instruction_selection: the anchor unrolled body emits 80
            #   `mul.f32x2`, 16 `add.f32x2`, 32 `ex2.approx.ftz.f32`, 32
            #   `rcp.approx.ftz.f32`, and 48 `cvt.rn.bf16x2.f32`, in the
            #   source order alpha scaling -> negative/log2(e) multiply ->
            #   exp2 -> denominator add -> accumulator conversion -> FP32
            #   reciprocal -> gate/output multiplies -> AB12/C conversions.
            #   Extent: all per-thread elements. Source `1041-1064`; PTX
            #   arithmetic/conversions `1095-1460`, with exp2 `1149-1195`
            #   and reciprocal `1234-1310`. The accumulator conversion before
            #   reciprocal is semantically mandatory.

            ab12_stage0 = (previous_subtiles + subtile) % AB12_STAGES
            ab12_stage1 = (previous_subtiles + subtile + 1) % AB12_STAGES
            c_stage = (previous_subtiles + subtile // 2) % C_STAGES
            register_store_shared(
                r_ab12_0,
                sAB12 + sAB12_byte_offset(ab12_stage0,
                                          source_epi_row(P, warp, lane),
                                          source_epi_col0(P, warp, lane)),
            )
            register_store_shared(
                r_ab12_1,
                sAB12 + sAB12_byte_offset(ab12_stage1,
                                          source_epi_row(P, warp, lane),
                                          source_epi_col0(P, warp, lane)),
            )
            register_store_shared(
                r_c,
                sC + sC_byte_offset(c_stage,
                                    source_epi_row(P, warp, lane),
                                    source_c_col(P, warp, lane)),
            )
            # instruction_selection: anchor has twelve `st.shared.v4.b32`
            #   (four for each output region); extent: all 128 epilogue
            #   threads. Source `1066-1087`; PTX `1484-1514`.

            fence_async_shared_cta()
            named_barrier(id=1, threads=128)
            # instruction_selection: `fence.proxy.async.shared::cta` and
            #   `bar.sync 1,128`; extent: one pre-store handoff per pair.
            #   Source `1089-1098`; PTX `1516,1518`.

            if warp == 0:
                tma_store(map_AB12,
                          source_ab12_coord(P, tile_m, tile_n, batch, subtile),
                          sAB12 + source_ab12_stage_stride(P) * ab12_stage0)
                tma_store(map_AB12,
                          source_ab12_coord(P, tile_m, tile_n, batch,
                                            subtile + 1),
                          sAB12 + source_ab12_stage_stride(P) * ab12_stage1)
                tma_store(map_C,
                          source_c_coord(P, tile_m, tile_n, batch,
                                         subtile // 2),
                          sC + source_c_stage_stride(P) * c_stage)
                tma_store_commit()
                tma_store_wait(pending=3)
                # instruction_selection: each output store is
                #   `cp.async.bulk.tensor.3d.global.shared::cta.tile.
                #   bulk_group.L2::cache_hint`, followed exactly by
                #   `cp.async.bulk.commit_group` and
                #   `cp.async.bulk.wait_group.read 3`; extent: two AB12 stores
                #   plus one C store in one warp-collective triple per subtile
                #   pair. Source `1100-1120`;
                #   PTX `1528,1533,1539,1541,1543`.
            named_barrier(id=1, threads=128)
            # instruction_selection: `bar.sync 1,128`; extent: one post-store
            #   stage-reuse handoff per pair. Source `1121-1124`; PTX `1546`.

        if elect_one():
            release_accumulator_empty(ACC_EMPTY + 8 * acc_cons.index,
                                      source_acc_release_target(P, cluster_rank))
            # instruction_selection:
            #   `mbarrier.arrive.shared::cluster.b64`; extent: one elected
            #   remote-capable release per output tile. Source `1126-1131`;
            #   PTX `1566`.
        advance(acc_cons)
        sched.advance()

# ==========================================================================
# 7. TMEM teardown and output-store tail
# source 1139-1154; PTX 1600-1651
# ==========================================================================
    if warp == 0:
        relinquish_tmem_alloc_permit(cta_group=CTA_GROUP)
        # instruction_selection: `tcgen05.relinquish_alloc_permit.
        #   cta_group::2.sync.aligned`; extent: one warp-collective
        #   relinquish.
        #   Source `1140-1143`; PTX `1614`.
    named_barrier(id=1, threads=128)
    # instruction_selection: `bar.sync 1,128`; extent: epilogue teardown.
    #   Source `1144-1145`; PTX `1619`.
    if warp == 0:
        if CTA_GROUP == 2:
            remote_peer = cluster_rank ^ 1
            arrive_remote(TMEM_DEALLOC, remote_peer)
            wait_mbarrier(TMEM_DEALLOC, phase=0, token=False)
            # instruction_selection: `mbarrier.arrive.shared::cluster.b64`
            #   then `mbarrier.try_wait.parity.shared.b64`; extent:
            #   paired-CTA deallocation handshake. Source `1146-1149`; PTX
            #   `1629,1637`.
        dealloc_tmem(tmem_base, columns=TMEM_COLUMNS, cta_group=CTA_GROUP)
        # instruction_selection:
        #   `tcgen05.dealloc.cta_group::2.sync.aligned.b32`;
        #   extent: one warp-collective deallocation of the specialized
        #   column count, anchor 512. Source `1150,1480-1502`; PTX `1648`.
    tma_store_wait(pending=0)
    # instruction_selection: `cp.async.bulk.wait_group.read 0`; extent: all
    #   output stores complete before kernel exit. Source `1151-1154`; PTX
    #   `1651`.
```

## Source helper correspondence

The source helpers are not omitted by the compact body; each is lowered to
scalar formulas or hardware operations at the cited sketch sites.

| source range | source responsibility | frozen sketch responsibility |
| --- | --- | --- |
| `174-292` | dtype-dependent MMA, SMEM atoms, stage counts, stage storage | sections 1-2: static constants, rank-1 arena, scalar XOR offsets, raw descriptors |
| `295-484` | legality, TensorMaps, scheduler/grid, shared size, launch | sections 1 and 3: static validation, opaque TensorMaps, persistent grid, launch |
| `1156-1218` | TMEM load selection and accumulator/register partition | sections 2 and 6: scalar TMEM address and x32 versus packed-x16 load family |
| `1220-1266` | register-to-SMEM copy selection and partition | section 6: explicit per-thread register words and scalar output offsets |
| `1268-1337` | global AB12/C TMA store partition | section 6: scalar global coordinates plus two AB12 and one C TMA store |
| `1340-1444` | accumulator/A/B/AB12/C stage computation and byte capacity | sections 1-2: fixed output stages, specialized A/B stage count, linear byte intervals |
| `1447-1477` | persistent grid computation | sections 1 and 4-6: persistent cluster grid and identical role-local cursors |
| `1480-1502` | specialized TMEM allocation-column computation | sections 2, 6, and 7: `source_num_tmem_alloc_cols(P)` for allocation/address/deallocation |
| `blackwell_helpers.py:744-799,802-858,913-963` | source A/B/epilogue SMEM mapping and swizzle selection | section 2: scalar stage strides, XOR offsets, and raw descriptor fields over one rank-1 arena |
| `blackwell_helpers.py:435-632,273-432` | source TMEM-load and register-to-SMEM instruction selection | section 6: specialized x32/packed-x16 load and vector shared-store families |

## Synchronization inventory

| object | producer -> consumer | stages / arrivals | reuse proof |
| --- | --- | --- | --- |
| A/B full | TMA warp -> MMA warp/CTA group | dynamic AB stages; transaction bytes equal one A plus B stage | TMA completion flips full phase; MMA waits before descriptor use |
| A/B empty | MMA warp/CTA group -> TMA warp | dynamic AB stages; source multicast consumer count | UMMA commit follows all K phases; TMA waits before overwriting stage |
| accumulator full | MMA warp -> four epilogue warps across CTA group | 2; arrival 1 | UMMA commit follows final K block; epilogue waits before TMEM loads |
| accumulator empty | four epilogue warps -> MMA warp | 2; 4 arrivals for CTA1, 8 for CTA2 | elected release occurs after every subtile pair has staged and launched stores |
| named barrier 2 | epilogue warp 0 -> MMA plus epilogue | 160 threads | TMEM pointer is stored by synchronous alloc before any retrieval |
| named barrier 1 | epilogue warps -> warp 0 -> epilogue warps | 128 threads | first barrier publishes register stores; second protects stage reuse; final one gates teardown |
| teardown mbarrier | paired CTAs | CTA2 only; 32 arrivals | both CTAs relinquish and rendezvous before either deallocates shared TMEM allocation |
| TMA store group | epilogue warp 0 -> SMEM stage reuse / exit | four-stage AB12 ring, two-stage C ring | read-3 wait preserves rolling AB12 capacity; read-0 tail drains all outputs |

All lane election sites are explicit above. Every barrier count is uniform for
the threads that execute it. The scheduler loop and subtile-pair loop bounds
are compile-time or warp-uniform; no divergent lane may skip a named barrier.

## Numerical contract

For each paired N block, the source computes:

```text
X = accumulator_block_0 * alpha
G = accumulator_block_1 * alpha
D = cast_to_accumulator(1 + exp2(-G * log2(e)))
C = cast_to_c(X * (G * rcp.approx(float32(D))))
AB12_block_0 = cast_to_ab12(X)
AB12_block_1 = cast_to_ab12(G)
```

The cast of the denominator to the accumulator dtype occurs before FP32
reciprocal. It is observable for FP16 accumulation and must not be reassociated.
`exp2.approx.ftz` and `rcp.approx.ftz` are part of the contract. Output C has
shape `(M,N/2,L)` and preserves `[X0|G0|X1|G1] ->
[X0*silu(G0)|X1*silu(G1)]`. AB12 has shape `(M,N,L)` and preserves the scaled
pre-activation pair.

## Static specialization boundary

One parameterized kernel owns exactly the source-legal surface:

- A and B share a dtype in FP16, BF16, FP32, FP8 E4M3, or FP8 E5M2. FP32
  operands select TF32 UMMA.
- FP32 accumulation accepts all five operand types, AB12 FP32/FP16/BF16, and C
  FP16/BF16. FP16 accumulation accepts FP16/FP8 operands, AB12 FP16/BF16, and C
  FP16/BF16. These are 42 dtype combinations.
- A independently selects M-major or K-major; B independently selects N-major
  or K-major; AB12 and C share M-major or N-major: eight major combinations.
- tile M is 128 or 256. Tile N is 32 through 256 in steps of 32. Tile M 128
  permits only cluster `(1,1)`. Tile M 256 permits `(2,1)`, `(2,2)`, `(2,4)`,
  `(2,8)`, `(4,1)`, `(4,2)`, `(4,4)`, `(8,1)`, `(8,2)`, `(16,1)`: 88
  tile/cluster combinations and 29,568 total static modes.
- N is divisible by 64. Contiguous dimensions meet 16-byte alignment. M/N
  tails use TensorMap OOB/store predicates; K may be an aligned tail. L is a
  positive batch extent. Alpha is runtime FP32.

Supplemental writer exports freeze instruction-selection branches:

| branch anchor | evidence SHA256 | required emitted family |
| --- | --- | --- |
| CTA1, tile 128x32, FP16 outputs | `cbdf3cdba6d5894008228becc5a5308c142cc71818529cefab7be281f59423f3` | CTA-group-1 dense F16 UMMA; 9 AB stages |
| FP32 operands with alternate majors | `aa61cffbf13ad955259750cc08a8ac91d7334dfa31b7b62d14bf35efe5719d2c` | `kind::tf32`; 4 AB stages |
| FP16 accumulation | `4abb0eabbbb2cefca267af9f15b55274b12c2d4773ecc9df0927906ef5331ec2` | FP16 accumulator and packed x16 TMEM load; 7 AB stages |
| FP8 E4M3 | `6d1e85edffa1e4dc4588356e313f3dfceb74275697cafbf05c16de487163437e` | `kind::f8f6f4`; 6 AB stages |
| FP8 E5M2 plus FP16 accumulation | `f43d7064b144923bc2e40f7102759f0eb8b92a60879f32cd0255a48eba66dc1d` | `kind::f8f6f4`, packed x16 load; 8 AB stages |

Changing a static axis changes only `P`, constants, predicates, descriptor
fields, stage counts, and the frozen instruction families. It does not add a
sketch or registry entry.

## Executable module contract

The reviewed sketch is mechanically transcribed into one module exporting:

```text
KERNEL_META, CONFIGS, BENCH_CONFIGS, get_kernel,
prepare_data, run_test, prepare_bench, run_gpu, run_bench
```

`CONFIGS` is a deterministic covering set over every axis and valid pairwise
interaction, plus aligned M/N/K tail cases and L>1. It does not enumerate
29,568 workflows. `prepare_bench()` compiles only TIRx and does not import or
initialize the CuTeDSL reference. The reference is loaded only from
`CUDNN_FRONTEND_PATH` after CUDA context establishment. The timed closure
launches only the source kernel and uses separate AB12/C outputs over the same
inputs as TIRx.

Performance acceptance is deferred until correctness passes. The final
bench-suite matrix is a deterministic minimal structural representative set
crossed with square 1024, 2048, 4096, and 8192 at L=1 and L=2. Every row must
satisfy `mean(cudnn_frontend_us) / mean(tirx_us) > 0.99`; equality fails. No
other timing API may decide performance.
