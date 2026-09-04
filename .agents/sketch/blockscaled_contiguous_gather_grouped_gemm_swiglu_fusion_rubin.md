# blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin: coarse WASP pipeline sketch

This is a non-executable semantic execution sketch for the production SM107
specialization. The corresponding TIRx module
`tirx_kernels/flashinfer/fused_moe/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin.py`
is the executable source of truth.

The fixed specialization is FP4-E2M1 A/B/C, E4M3 vector-16 scale factors,
`N=4096`, `K=7168`, top-k 8, `(tileM,tileN,tileK)=(128,128,256)`,
`(instM,instN,instK)=(128,128,128)`, cluster `(1,1)`, cp.async A gather,
packed-FP32 SwiGLU, SFC generation, N-major output, persistent raster-N
scheduling, PDL, and alpha 1.0. Parameterized source variants are host-level
correctness obligations but are outside this instruction-selection sketch.

Writer line-info PTX:
`.porting/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin/kernel_sketch/source_export_production/cutlass___call___flashinferfused_moecute_dslrubinblockscaled_contiguous_gather_grouped_gemm_swiglu_fusionSm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel_object_at__Te.sm_107a.ptx`,
SHA256 `5f7c0e8b87a2d1dfa80d4c15e665050d9e93ca7f1efa3f2778b961ccc920a0da`.
It declares PTX 9.4, targets `sm_107a`, and requires 640 threads.

## Pipeline at a glance

| warps | role | tile program and publication/reuse edge |
| --- | --- | --- |
| 0-3 | epilogue consumers / TMEM allocator | wait accumulator-full; load paired 128x64 up/gate accumulator fragments; packed SwiGLU; derive/store SFC; quantize/store FP4 C through the nine-stage shared ring; four arrivals publish accumulator-empty |
| 4-7 | gathered-A producers | gather token rows and issue eight predicated 16-byte cp.async copies per thread for every K tile; the 128-thread producer group publishes one A-full stage |
| 8 | MMA consumer/producer | wait A-full, B/SFB-full, and transformed-SFA-full; copy SFB SMEM to TMEM; issue two block-scaled FP4 MMAs per K tile; release all three inputs and commit accumulator-full |
| 9 | B/SFB TMA producer | issue one rank-3 B TMA and one rank-4 SFB TMA into a shared completion barrier per K tile; one consumer arrival publishes stage reuse |
| 10 | persistent scheduler | publish `(tile_m,tile_n,expert,valid,mn_limit)` through the two-stage tile-info pipe, including one terminal invalid record |
| 11 | reserved | idle for this one-CTA specialization; the source reserves the slot for the two-CTA A relay |
| 12-15 | gathered-SFA producers | gather one routed token row per thread and issue one predicated 16-byte cp.async into linear SFA SMEM; publish SFA-SMEM-full |
| 16-19 | SFA transform producers | wait SFA-SMEM-full; each warp loads four row groups and issues four `tcgen05.st` operations into a rotating four-stage SFA TMEM ring; publish transformed-SFA-full and release SFA SMEM |

All roles consume the same tile-info stream except the scheduler and idle warp
11. Role branches occur in the source order shown in the complete sketch.

## Primitive vocabulary

`linear_smem(bytes, align)`, `linear_tmem(columns)`, `registers(dtype,count)`,
and `mbarrier_array(count)` allocate storage without first-class layouts.
Named scalar functions such as `A_smem(stage,row,k_byte)` and
`C_smem(stage,row,n_byte)` return explicit byte offsets. `copy_g2s`,
`copy_s2r`, `copy_s2t`, `copy_t2r`, `copy_r2s`, `copy_s2g`, `gemm`, `mul`,
`add`, `exp2`, `reciprocal`, `abs`, `reduce_max`, `min`, and `cast` are basic
movement or computation. Pipe operations expose acquire/wait/commit/release,
stage, phase, and ownership directly.

## Complete sketch

```python
# Static specialization and launch
CTA_M, CTA_N, K_TILE = 128, 128, 256
INST_M, INST_N, INST_K = 128, 128, 128
EPI_M, EPI_N = 128, 64
AB_STAGES, C_STAGES, ACC_STAGES = 8, 9, 2
TILE_STAGES, SFA_TMEM_STAGES = 2, 4
THREADS, CLUSTER, DYNAMIC_SMEM = 640, (1, 1, 1), 332800
launch(grid=min(problem_clusters, max_active_clusters), block=(640,1,1),
       cluster=CLUSTER, dynamic_smem=DYNAMIC_SMEM, pdl=True,
       min_blocks_per_sm=1)

# Runtime ABI: packed byte views preserve the source physical formats.
A_u8, B_u8, SFA_u8, SFB_u8, C_u8, SFC_u8
tile_idx_to_expert_i32, tile_idx_to_mn_limit_i32, token_id_mapping_i32
num_non_exiting_tiles_i32, alpha_f32, global_scale_f32

# One linear dynamic-SMEM arena. These are byte intervals and scalar mappings,
# never first-class layout-bearing objects.
smem = linear_smem(332800, align=1024)
INFO       = smem[0:40]            # 5 i32 * 2 stages
A_MBAR     = smem[40:168]          # full[8], empty[8]
B_MBAR     = smem[168:296]         # full[8], empty[8]
SFA_MBAR   = smem[296:424]         # full[8], empty[8]
SFA_T_MBAR = smem[424:488]         # full[4], empty[4]
ACC_MBAR   = smem[488:520]         # full[2], empty[2]
TILE_MBAR  = smem[520:552]         # full[2], empty[2]
TMEM_DEALLOC = smem[552:560]
TMEM_PTR     = smem[560:564]
C_SMEM     = smem[1024:37888]      # 9 * 4096 bytes
SFA_SMEM   = smem[37888:54272]     # 8 * 2048 bytes; deliberately before A
A_SMEM     = smem[54272:185344]    # 8 * 16384 packed-FP4 bytes
B_SMEM     = smem[185344:316416]   # 8 * 16384 packed-FP4 bytes
SFB_SMEM   = smem[316416:332800]   # 8 * 2048 E4M3 bytes

def info(stage, field): return 4 * (5 * stage + field)
def sfa_smem(stage, row, ksf): return 37888 + stage*2048 + row*16 + ksf
def a_smem(stage, row, packed_k):
    return 54272 + stage*16384 + swizzle_128b(row, packed_k, row_bytes=128)
def b_smem(stage, n, packed_k):
    return 185344 + stage*16384 + swizzle_128b(n, packed_k, row_bytes=128)
def sfb_smem(stage, n_group, ksf_group):
    return 316416 + stage*2048 + sfb_block16_swizzle(n_group, ksf_group)
def c_smem(stage, row, packed_n):
    return 1024 + stage*4096 + swizzle_128b(row, packed_n, row_bytes=32)

# TMEM allocator reserves 576 columns although this specialization uses 336.
tmem = linear_tmem(576)
ACC_TMEM = tmem[0:256]              # two 128-column FP32 accumulator stages
SFA_TMEM = tmem[256:320]            # four 16-column E4M3 stages
SFB_TMEM = tmem[320:336]            # one 16-column E4M3 operand region
def acc_tmem(stage, ncol): return stage*128 + ncol
def sfa_tmem(stage, gi): return 256 + stage*16 + gi*4
def sfb_tmem(gi): return 320 + gi*4

# Descriptor prefetch precedes storage allocation and pipe construction.
prefetch(B_map); prefetch(SFB_map); prefetch(C_map)
# instruction_selection: `prefetch.tensormap`; extent: three rank-3/rank-4
# descriptors. Source 1261-1267; PTX 102,105,108.

# Physical barrier initialization in source pipe-construction order.
init(SFA_MBAR.full[0:8], arrivals=128); init(SFA_MBAR.empty[0:8], arrivals=128)
init(SFA_T_MBAR.full[0:4], arrivals=128); init(SFA_T_MBAR.empty[0:4], arrivals=1)
init(A_MBAR.full[0:8], arrivals=128); init(A_MBAR.empty[0:8], arrivals=1)
init(B_MBAR.full[0:8], arrivals=1); init(B_MBAR.empty[0:8], arrivals=1)
init(ACC_MBAR.full[0:2], arrivals=1); init(ACC_MBAR.empty[0:2], arrivals=4)
init(TILE_MBAR.full[0:2], arrivals=32); init(TILE_MBAR.empty[0:2], arrivals=576)
# instruction_selection: elected `mbarrier.init.shared.b64`; extent: 64
# pipeline barriers. The one-CTA specialization leaves TMEM_DEALLOC uninitialized and unused.
# Source 1294-1459; PTX 113-287.

mbarrier_init_fence()
# instruction_selection: `fence.mbarrier_init.release.cluster`; extent: all 64
# initialized pipe barriers. Source pipe helper after 1432-1450; PTX 286.
named_barrier(id=0, threads=640)
# instruction_selection: `bar.sync 0`; extent: the full CTA after non-deferred
# tile-info pipe initialization. Source pipe helper after 1432-1450; PTX 287.

cluster_arrive_relaxed()
# instruction_selection: `fence.mbarrier_init.release.cluster`; extent:
# singleton-cluster initialization (no cluster-arrive instruction is emitted).
# Source 1459; PTX 289.
cluster_wait()
# instruction_selection: `bar.sync 0`; extent: singleton CTA (no cluster-wait
# instruction is emitted). Source 1619; PTX 308.
griddepcontrol_wait()
# instruction_selection: `griddepcontrol.wait`; extent: CTA. Source 1621.

# Tile-info consumer protocol is expanded here and invoked with the source field
# extent in each role; it is not a compound copy/synchronization primitive.
def consume_info_5(tile_cons):
    wait(TILE_MBAR.full[tile_cons.stage], tile_cons.phase)
    # instruction_selection: `mbarrier.try_wait.parity.shared.b64` retry loop;
    # extent: one tile-info full stage.
    for field in range(5):
        copy_s2r(INFO[info(tile_cons.stage,field)], tile[field])
        # instruction_selection: `ld.shared.b32`; extent: five words per
        # consumer thread (A, SFA gather, and SFA transform roles).
    fence_async_shared_cta()
    # instruction_selection: `fence.proxy.async.shared::cta`; extent: INFO.
    arrive(TILE_MBAR.empty[tile_cons.stage])
    # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one
    # arrival from each consumer thread.
    tile_cons.advance()
    return decode_tile(tile[0:5])

def consume_info_4(tile_cons):
    wait(TILE_MBAR.full[tile_cons.stage], tile_cons.phase)
    # instruction_selection: `mbarrier.try_wait.parity.shared.b64` retry loop;
    # extent: one tile-info full stage.
    for field in range(4):
        copy_s2r(INFO[info(tile_cons.stage,field)], tile[field])
        # instruction_selection: `ld.shared.b32`; extent: four words per
        # consumer thread (B, MMA, and epilogue roles).
    fence_async_shared_cta()
    # instruction_selection: `fence.proxy.async.shared::cta`; extent: INFO.
    arrive(TILE_MBAR.empty[tile_cons.stage])
    # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one
    # arrival from each consumer thread.
    tile_cons.advance()
    return decode_tile(tile[0:4])

# Source-order role 1: scheduler warp 10.
if warp == 10:
    tile_prod = producer_cursor(TILE_STAGES)
    work = persistent_initial(block_idx, grid_dim, raster="N")
    while work.valid and work.tile_m < num_non_exiting_tiles[0]:
        acquire(TILE_MBAR.empty[tile_prod.stage], tile_prod.phase)
        # instruction_selection: acquire parity try-wait loop; extent: one
        # tile-info stage. Source 1640-1700; PTX mbarrier try-wait family.
        if elect_one():
            store(INFO, info(tile_prod.stage, 0:5),
                  (work.tile_m, work.tile_n,
                   tile_idx_to_expert[work.tile_m],
                   1, tile_idx_to_mn_limit[work.tile_m]))
            # instruction_selection: scalar `st.shared.b32`; extent: five words.
        fence_async_shared_cta()
        # instruction_selection: `fence.proxy.async.shared::cta`; extent: INFO.
        named_barrier(id=4, threads=32)
        # instruction_selection: `bar.sync 4,32`; extent: scheduler warp.
        arrive(TILE_MBAR.full[tile_prod.stage])
        # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one
        # arrival per scheduler lane. Source 1640-1700.
        tile_prod.advance(); work = persistent_next(work)
    acquire(TILE_MBAR.empty[tile_prod.stage], tile_prod.phase)
    if elect_one(): store(INFO, info(tile_prod.stage,0:5), (work.m,work.n,-1,0,-1))
    fence_async_shared_cta()
    # instruction_selection: `fence.proxy.async.shared::cta`; extent: INFO.
    named_barrier(id=4, threads=32)
    # instruction_selection: `bar.sync 4,32`; extent: scheduler warp.
    arrive(TILE_MBAR.full[tile_prod.stage])
    # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one
    # terminal-record arrival per scheduler lane.
    tile_prod.advance()
    tail_wait(TILE_MBAR.empty, tile_prod)
    # instruction_selection: terminal shared stores, async-shared fence,
    # `bar.sync 4,32`, arrive, and parity waits; extent: one invalid record.
    # Source 1701-1721.

# Source-order role 2: gathered A warps 4-7.
if 4 <= warp <= 7:
    setmaxnreg_dec(80)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 80`; extent:
    # four-warp producer group. Source 1723-1726.
    tile_cons, a_prod = consumer_cursor(2), producer_cursor(8)
    tile = consume_info_5(tile_cons)
    while tile.valid:
        # Each of 128 lanes owns eight M rows spaced by 16; valid routing uses
        # token_id/topk and mn_limit, padded rows use the zero-fill predicate.
        rows[0:8] = gathered_rows(token_id_mapping, tile, lane128, topk=8)
        for kt in range(K // 256):
            acquire(A_MBAR.empty[a_prod.stage], a_prod.phase)
            for i in range(8):
                copy_g2s(A_u8[row=rows[i], packed_k=kt*128+lane128%8*16],
                         A_SMEM[a_smem(a_prod.stage,lane128//8+i*16,lane128%8*16)],
                         bytes=16, predicate=row_is_valid[i])
                # instruction_selection: `cp.async.cg.shared.global` with
                # 16-byte source-size predicate; extent: eight unrolled copies
                # per thread per K tile. Source 1725-1848; PTX 916-961.
            arrive_cpasync(A_MBAR.full[a_prod.stage])
            # instruction_selection: `cp.async.mbarrier.arrive.noinc.shared.b64`;
            # extent: one arrival per lane. Source 1846.
            a_prod.advance()
        tile = consume_info_5(tile_cons)
    tail_wait(A_MBAR.empty, a_prod)
    # instruction_selection: parity wait loop; extent: eight-stage A ring.

# Source-order role 3: gathered SFA warps 12-15.
if 12 <= warp <= 15:
    setmaxnreg_dec(80)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 80`; extent:
    # four-warp producer group. Source 1981-1984.
    tile_cons, sfa_prod = consumer_cursor(2), producer_cursor(8)
    tile = consume_info_5(tile_cons)
    while tile.valid:
        row = token_id_mapping[tile.m*128 + lane128] // 8
        pred = tile.m*128 + lane128 < tile.mn_limit
        for kt in range(K // 256):
            acquire(SFA_MBAR.empty[sfa_prod.stage], sfa_prod.phase)
            copy_g2s(SFA_u8[row, kt*16:kt*16+16],
                     SFA_SMEM[sfa_smem(sfa_prod.stage,lane128,0)],
                     bytes=16, predicate=pred)
            # instruction_selection: `cp.async.cg.shared.global`; extent: one
            # predicated 16-byte copy per lane per K tile. Source 1983-2088;
            # PTX 1307-1317.
            arrive_cpasync(SFA_MBAR.full[sfa_prod.stage])
            # instruction_selection: `cp.async.mbarrier.arrive.noinc.shared.b64`;
            # extent: one arrival per lane. Source 2087.
            sfa_prod.advance()
        tile = consume_info_5(tile_cons)
    tail_wait(SFA_MBAR.empty, sfa_prod)

# Source-order role 4: SFA transform warps 16-19.
if 16 <= warp <= 19:
    setmaxnreg_dec(48)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; extent:
    # four-warp transform group. Source 2102-2106.
    named_barrier(id=3, threads=288)
    # instruction_selection: `bar.sync 3,288`; extent: epilogue allocator,
    # MMA warp, and four transform warps.
    copy_s2r(TMEM_PTR, tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one TMEM base pointer
    # per participating thread.
    tile_cons, sfa_cons, st_prod = consumer_cursor(2), consumer_cursor(8), producer_cursor(4)
    tile = consume_info_5(tile_cons)
    while tile.valid:
        for kt in range(K // 256):
            wait(SFA_MBAR.full[sfa_cons.stage], sfa_cons.phase)
            for ci in range(4):
                copy_s2r(SFA_SMEM[sfa_smem(sfa_cons.stage,32*ci+lane,0)],
                         rSFA[ci,0:4], bytes=16)
                # instruction_selection: four vector `ld.shared.v4.b32` per
                # warp; extent: four row groups per K tile. Source 2149-2181.
            acquire(SFA_T_MBAR.empty[st_prod.stage], st_prod.phase)
            for gi in range(4):
                copy_r2t(rSFA[0:4,gi], SFA_TMEM[sfa_tmem(st_prod.stage,gi)])
                # instruction_selection:
                # `tcgen05.st.sync.aligned.32x32b.x4.b32`; extent: four static
                # stores per warp per K tile. Source 2182-2207; PTX 1658-1684.
            fence_tmem_store()
            # instruction_selection: `tcgen05.wait::st.sync.aligned` plus
            # async-TMEM store fence; extent: transform stage. Source 2208-2214.
            arrive(SFA_T_MBAR.full[st_prod.stage])
            # instruction_selection: `mbarrier.arrive.shared.b64`; extent:
            # 128 transform-lane arrivals. Source 2212-2214.
            arrive(SFA_MBAR.empty[sfa_cons.stage])
            # instruction_selection: `mbarrier.arrive.shared.b64`; extent:
            # one release from each of the 128 transform lanes after SMEM consumption.
            # Source 2215-2217; PTX 1699.
            st_prod.advance(); sfa_cons.advance()
        tile = consume_info_5(tile_cons)
    tail_wait(SFA_T_MBAR.empty, st_prod)

# Source-order role 5: warp 11 relay branch is statically absent for CTA-group 1.
if warp == 11:
    pass

# Source-order role 6: B/SFB TMA warp 9.
if warp == 9:
    tile_cons, b_prod = consumer_cursor(2), producer_cursor(8)
    tile = consume_info_4(tile_cons)
    while tile.valid:
        for kt in range(K // 256):
            acquire(B_MBAR.empty[b_prod.stage], b_prod.phase)
            # instruction_selection: `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`
            # followed by the plain retry loop when needed; extent: one B-empty
            # stage. Source 2351-2370; PTX 1885/1911.
            if elect_one():
                expect_tx(B_MBAR.full[b_prod.stage], bytes=18432)
                # instruction_selection: `mbarrier.arrive.expect_tx.shared.b64`;
                # extent: one elected 18,432-byte expectation before both TMA
                # issues. Source 2370; PTX 1927.
            if elect_one():
                copy_g2s(B_map[expert=tile.expert,n_tile=tile.n,k_tile=kt],
                         B_SMEM[b_smem(b_prod.stage,0,0)], mbar=B_MBAR.full[b_prod.stage])
                # instruction_selection:
                # `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
                # extent: one 128x256 packed-FP4 B tile. Source 2375-2382;
                # PTX 1933-1941.
                copy_g2s(SFB_map[expert=tile.expert,n_tile=tile.n,k_tile=kt],
                         SFB_SMEM[sfb_smem(b_prod.stage,0,0)], mbar=B_MBAR.full[b_prod.stage])
                # instruction_selection:
                # `cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
                # extent: one 128x16 E4M3 SFB tile. Source 2383-2391;
                # PTX 1944-1954.
            b_prod.advance()
        tile = consume_info_4(tile_cons)
    tail_wait(B_MBAR.empty, b_prod)


# Source-order role 7: MMA warp 8.
if warp == 8:
    named_barrier(id=3, threads=288)
    # instruction_selection: `bar.sync 3,288`; extent: epilogue allocator,
    # MMA warp, and four transform warps.
    copy_s2r(TMEM_PTR, tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one TMEM base pointer
    # per participating thread.
    tile_cons = consumer_cursor(2)
    a_cons, b_cons = consumer_cursor(8), consumer_cursor(8)
    st_cons, acc_prod = consumer_cursor(4), producer_cursor(2)
    tile = consume_info_4(tile_cons)
    while tile.valid:
        acquire(ACC_MBAR.empty[acc_prod.stage], acc_prod.phase)
        for kt in range(K // 256):
            wait(A_MBAR.full[a_cons.stage], a_cons.phase)
            wait(B_MBAR.full[b_cons.stage], b_cons.phase)
            wait(SFA_T_MBAR.full[st_cons.stage], st_cons.phase)
            # instruction_selection: acquire/plain parity try-wait loops;
            # extent: one stage from each input pipe. Source 2559-2636.
            for gi in range(4):
                copy_s2t(SFB_SMEM[sfb_smem(b_cons.stage,gi,0)], SFB_TMEM[sfb_tmem(gi)])
                # instruction_selection: `tcgen05.cp.cta_group::1.32x128b.warpx4`;
                # extent: four SFB copies per K tile. Source 2638-2641;
                # writer PTX has four static instructions.
            for kb in range(2):
                gemm(ACC_TMEM[acc_tmem(acc_prod.stage,0:128)],
                     A_SMEM[a_smem(a_cons.stage,0,kb*64)],
                     B_SMEM[b_smem(b_cons.stage,0,kb*64)],
                     SFA_TMEM[sfa_tmem(st_cons.stage,kb*2)],
                     SFB_TMEM[sfb_tmem(kb*2)],
                     accumulate=(kt != 0 or kb != 0))
                # instruction_selection:
                # `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard`;
                # extent: two 128x128x128 FP4 block-scaled MMAs per K tile.
                # Source 2643-2712; PTX 2462-2494.
            commit_mma(A_MBAR.empty[a_cons.stage])
            # instruction_selection: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
            # extent: one A-empty destination after both MMAs. Source 2713-2725;
            # PTX 2501.
            commit_mma(B_MBAR.empty[b_cons.stage])
            # instruction_selection: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
            # extent: one B-empty destination after both MMAs. Source 2713-2725;
            # PTX 2508.
            commit_mma(SFA_T_MBAR.empty[st_cons.stage])
            # instruction_selection: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
            # extent: one transformed-SFA-empty destination after both MMAs.
            # Source 2713-2725; PTX 2515.
            a_cons.advance(); b_cons.advance(); st_cons.advance()
        commit_mma(ACC_MBAR.full[acc_prod.stage])
        # instruction_selection: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
        # extent: one completed 128x128 accumulator tile. Source 2764; PTX 2575.
        acc_prod.advance()
        tile = consume_info_4(tile_cons)
    tail_wait(ACC_MBAR.empty, acc_prod)


# Source-order role 8: epilogue warps 0-3.
if 0 <= warp <= 3:
    setmaxnreg_inc(168)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 168`; extent:
    # epilogue warpgroup. Source 2790-2794.
    if warp == 0:
        alloc_tmem(columns=576); store_shared(TMEM_PTR)
        # instruction_selection: `tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32`;
        # extent: allocator warp. Source 2796; PTX 2669.
    named_barrier(id=3, threads=288)
    # instruction_selection: `bar.sync 3,288`; extent: allocator, MMA, and
    # four SFA-transform warps. Source 2801.
    tile_cons, acc_cons = consumer_cursor(2), consumer_cursor(2)
    c_count = 0
    tile = consume_info_4(tile_cons)
    while tile.valid:
        wait(ACC_MBAR.full[acc_cons.stage], acc_cons.phase)
        # instruction_selection: parity wait loop; extent: one accumulator.
        copy_t2r(ACC_TMEM[acc_tmem(acc_cons.stage,0:64)], rUp[0:64])
        # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x64.b32`;
        # extent: up half of one 128x64 epilogue tile. Source 2999; PTX 2774.
        copy_t2r(ACC_TMEM[acc_tmem(acc_cons.stage,64:128)], rGate[0:64])
        # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x64.b32`;
        # extent: gate half. Source 3000; PTX 2808.
        for i in range(0, len(rUp), 2):
            up = mul(rUp[i:i+2], alpha)
            gate = mul(rGate[i:i+2], alpha)
            neg = mul(gate, -1.4426950408889634)
            den = add(exp2(neg), 1.0)
            sig = reciprocal(den)
            rOut[i:i+2] = mul(mul(sig, gate), up)
            # instruction_selection: packed `mul.rn.f32x2`,
            # `add.rn.f32x2`, scalar `ex2.approx.ftz.f32`, and scalar
            # `rcp.approx.ftz.f32`; extent: two FP32 lanes per unrolled step.
            # Source 3021-3072; PTX 2844-3599.
        for group in groups_of(rOut, 16):
            scale = mul(mul(reduce_max(abs(group)), reciprocal_fp4_limit), global_scale)
            # instruction_selection: `abs.f32`, `max.NaN.f32`, and packed
            # `mul.rn.f32x2`; extent: one 16-value output scale group. Source
            # 3094-3144; PTX line-info locs 3117-3144.
        rScaleE4M3 = cast(scale, e4m3x2)
        # instruction_selection: two `cvt.rn.satfinite.e4m3x2.f32`;
        # extent: four scale values per epilogue thread. Source 3157;
        # PTX 3772/3774.
        copy_r2g(pack(rScaleE4M3), SFC_u8[sfc_offset(tile,lane)])
        # instruction_selection: `st.global.v2.b16`; extent: one four-byte
        # vector containing four E4M3 scales per thread. Source 3161;
        # PTX 3776.
        decoded_scale = cast(cast(rScaleE4M3, f16x2), f32)
        # instruction_selection: packed `cvt.rn.f16x2.e4m3x2` followed by
        # scalar `cvt.f32.f16`; extent: all four scale lanes. Source 3165;
        # PTX 3778-3785.
        qscale = min(mul(global_scale, reciprocal(decoded_scale)), FLT_MAX,
                     nan_semantics="propagate")
        rQuant = mul(rOut, qscale)
        # instruction_selection: `rcp.approx.ftz.f32`, packed
        # `mul.rn.f32x2`, and four scalar `min.NaN.f32` per thread;
        # extent: all output values. Source 3166-3192; PTX 3787-3849.
        rPackedFP4 = cast(rQuant, e2m1x2)
        # instruction_selection: `cvt.rn.satfinite.e2m1x2.f32`; extent:
        # packed FP4 output words. Source 3192-3210; PTX 3893-4059.
        c_stage = (c_count + 1) % 9
        copy_r2s(rPackedFP4, C_SMEM[c_smem(c_stage,0,0)])
        # instruction_selection: vector `st.shared.v4.b32`; extent: one
        # 128x64 packed-FP4 output tile. Source 3211-3216.
        fence_async_shared_cta()
        # instruction_selection: `fence.proxy.async.shared::cta`; extent:
        # C tile visibility.
        named_barrier(id=2, threads=128)
        # instruction_selection: `bar.sync 2,128`; extent: epilogue warpgroup. Source 3218-3224.
        if warp == 0:
            copy_s2g(C_SMEM[c_smem(c_stage,0,0)],
                     C_map[tile.m,tile.n], shape=(128,64))
            # instruction_selection:
            # `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint`;
            # extent: one 128x64 packed-FP4 C tile. Source 3225-3232;
            # PTX 4085-4093.
            tma_commit_group()
            # instruction_selection: `cp.async.bulk.commit_group`; extent:
            # one C bulk-copy group.
            tma_wait_group(8)
            # instruction_selection: `cp.async.bulk.wait_group.read 8`;
            # extent: nine-stage C ring.
            # Source 3233-3237; PTX 4094-4097.
        named_barrier(id=2, threads=128)
        # instruction_selection: `bar.sync 2,128`; extent: C stage reuse.
        arrive(ACC_MBAR.empty[acc_cons.stage])
        # instruction_selection: elected `mbarrier.arrive.shared.b64`; extent:
        # one arrival per epilogue warp, four total. Source 3240-3247.
        acc_cons.advance(); c_count += 1
        tile = consume_info_4(tile_cons)
    if warp == 0: relinquish_tmem()
    # instruction_selection:
    # `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`; extent:
    # allocator warp. Source 3264-3268.
    named_barrier(id=2, threads=128)
    if warp == 0: dealloc_tmem(columns=576)
    # instruction_selection: `tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32`;
    # extent: allocator warp. Source 3266; PTX 4168.
    tma_wait_group(0)
    # instruction_selection: `cp.async.bulk.wait_group.read 0`; extent:
    # drain all C stores. Source 3271.


griddepcontrol_launch_dependents()
# instruction_selection: `griddepcontrol.launch_dependents`; extent: CTA.
# Source 3273.
```

## Pipeline inventory

| pipe | stages | producer -> consumer | full publication | reuse publication |
| --- | ---: | --- | --- | --- |
| A gather | 8 | warps 4-7 -> warp 8 | 128 `cp.async.mbarrier.arrive.noinc` arrivals | MMA `tcgen05.commit` to A-empty |
| B/SFB | 8 | elected lane, warp 9 -> warp 8 | 18,432 expected TMA bytes | MMA `tcgen05.commit` to B-empty |
| SFA SMEM | 8 | warps 12-15 -> warps 16-19 | 128 `cp.async.mbarrier.arrive.noinc` arrivals | 128 transform-lane arrivals to SFA-empty |
| transformed SFA TMEM | 4 | warps 16-19 -> warp 8 | 128 transform-lane arrivals after `tcgen05.st` | MMA `tcgen05.commit` to transform-empty |
| accumulator | 2 | warp 8 -> warps 0-3 | MMA commit to accumulator-full | four elected epilogue arrivals to accumulator-empty |
| tile info | 2 | warp 10 -> 18 non-scheduler/non-idle warps | 32 scheduler-lane arrivals after shared fence | 576 consumer-lane arrivals after loading INFO |
| C store | 9 | warps 0-3 -> TMA engine | shared fence + named barrier 2 + bulk commit | `wait_group.read 8` + named barrier 2 |

Each cursor has `(stage,phase,count)`, complementary producer/consumer initial
phases, modulo-stage advancement, and parity flip on wrap. Every role consumes
the terminal invalid tile and drains its owned producer or consumer ring.

## TensorMap and tail contract

| descriptor | source rank | selected production region |
| --- | ---: | --- |
| B | 3 | packed FP4 `(K,N,expert)`, one 128x256 logical tile into 16 KiB SMEM |
| SFB | 4 | E4M3 block scales, one 128x16 region into 2 KiB SMEM |
| C | 3 | packed FP4 `(N/2,M,expert)`, one 128x64 logical output tile |

A and SFA do not use TensorMaps in this specialization: routed row IDs select
ordinary global pointers and predicated 16-byte cp.async copies. `mn_limit`
guards padded routing rows. B/SFB and C TensorMap bounds provide N/K tail
behavior. Production N/K are exact multiples of the selected tiles.

## Instruction-selection summary

The source export proves the low-offset SFA arena preserves
`cp.async.cg.shared.global`, the B/SFB pair lowers to rank-3/rank-4 TMA with a
shared byte-completion barrier, the four transform warps lower to four static
`tcgen05.st...x4` instructions, and each K tile lowers to four SFB TMEM copies
plus two block16 NVFP4 MMAs. The epilogue uses two x64 TMEM loads, packed FP32
SwiGLU arithmetic, E4M3 scale conversion/direct global scale stores, saturating
E2M1 pair conversion, and a nine-stage TMA C ring. The explicit byte/column
mappings and barrier counts above select those instructions without attaching
first-class layouts to any storage object.

## Executable and validation contract

The device definition imports only `tirx_kernels.kern as K`, uses one rank-one
dynamic `u8` shared arena with scalar offsets, and contains no inline CUDA
function-call escape. Correctness compares independent TIRx and frozen-source
outputs at physical-byte precision where the source is deterministic, checks
decoded FP4/SFC reconstruction against a mathematical oracle with the tightest
representable tolerances, exercises routing tails/empty experts/top-k and the
retained tactic/dtype/launch branches, and repeats launches for determinism.
Performance is determined only by `bench_suite`, on the 51 production profiles,
with source and TIRx both compiled for PTX 9.4 / `sm_107a`; every
`mean(flashinfer) / mean(tirx)` row must be greater than 0.99.
