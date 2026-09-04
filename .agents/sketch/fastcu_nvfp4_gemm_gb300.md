<!--
This file is a design sketch for a TIRx port of code from fast.cu
(https://github.com/pranjalssh/fast.cu @ 2dfe5e26aecfd9e5f27bf9d5837deea01acda24b),
Copyright (c) 2024 Pranjal Shankhdhar.
SPDX-License-Identifier: Apache-2.0 AND MIT
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# fast.cu GB300 NVFP4 GEMM r9: coarse WASP pipeline sketch

This is a non-executable operation-level sketch for
[`tirx_kernels/fastcu/nvfp4_gemm_gb300.py`](../../tirx_kernels/fastcu/nvfp4_gemm_gb300.py),
which becomes the executable source of truth. It freezes the source r9 kernel's
fixed two-CTA tile, seven-warp split, six-slot A/B feed, seven-slot scale feed,
two overlapping TMEM accumulators, exact K96/K64 issue order, direct FP16
epilogue, and L2-side-owned persistent schedule.

The scope is `gb300/nvfp4/gemm9.cuh::nvfp4::nvfp4_gemm_kernel` at commit
`2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`: packed E2M1 A/B, VEC16_UE4M3
block-16 scales, FP32 accumulation, and FP16 output on `sm_103a`. Other
`gemm*.cuh` rungs, other scale formats, other output dtypes, and other GPU
architectures are out of scope. The runtime domain is positive M/N and K >= 16
with K divisible by 16; the benchmark domain is the five requested square
shapes.

The writer export is
`.porting/fastcu_nvfp4_gemm_gb300/source_export/writer/gemm9.ptx`, SHA256
`25dd14c7142af7d5a31cfa8902a2d141c9bbffe15cab3f19437cb2d023ca891a`.
It was produced by CUDA 13.3.73 with `-O3 -DNDEBUG -lineinfo` and
`compute_103a`, contains 3,625 `.loc` records, and maps `.file 1` to the pinned
`gemm9.cuh`. The GEMM entry occupies PTX lines 22-11312. All instruction
selections below come from that entry. The intentional route-table address-space
adaptation is called out separately and must be checked against the generated
TIRx PTX and the performance gate.

After the independent sketch reviewer returns PASS, this file is immutable.

## Pipeline at a glance

| warp / CTA role | tile program | publication / reuse edges |
| --- | --- | --- |
| warps 0-3, both CTAs | each warp owns 32 rows; wait for one accumulator, drain eight 32-column bands in parity-dependent order, convert packed FP32 pairs to FP16, store directly to C, and release the accumulator after the shared 36-column edge is drained | `acc_ready` from warp 4; fused `tcgen05.wait::ld` then lane-0 remote `acc_free` arrive to the even CTA |
| warp 4, even CTA only | walk every tile assigned to the cluster; consume scale and A/B rings; stage nine scale tiles per scale slot to TMEM; issue the K96 stream and exact K64 tail subclass; publish one completed accumulator per tile | `sf_ready -> sf_free`, `ab_ready -> ab_free`, `acc_free -> acc_ready`; all tcgen05 copy/MMA/commit instructions have one lane issuer |
| warp 4, odd CTA | no mainloop work | participates in the prologue TMEM-allocation rendezvous, then remains idle/exits; epilogue warp 0 handles teardown |
| warp 5 lane 0, both CTAs | use the scalar route entry; per live A/B window wait for reuse, issue one self-multicast A TMA and one peer-local B TMA, then advance the six-slot cursor | `ab_free -> ab_ready`; even CTA declares 65,536 expected bytes, both CTAs contribute transfers |
| warp 6 lane 0, both CTAs | use the scalar route entry; per live scale slot issue one self-multicast SFA TMA and one two-peer SFB multicast, then advance the seven-slot cursor | `sf_free -> sf_ready`; even CTA declares 9,216 expected bytes, both CTAs contribute transfers |
| warp 0, both CTAs | lane 0 initializes 29 mbarriers and publishes them cluster-wide; the full warp allocates pair TMEM; after all epilogue work, all warp-0 lanes relinquish and arrive at the peer deallocation barrier, wait locally, then deallocate | init fence -> cluster release/acquire barrier; named barrier 2 publishes TMEM address to warps 0-4; named barrier 3 joins epilogue warps before teardown |

## Primitive vocabulary

The sketch uses no tile primitive and no first-class layout. All storage is a
rank-one byte/register allocation plus scalar index functions:

```python
linear_smem(name, bytes, alignment)
linear_gmem(name, dtype, elements)
reg_array(name, dtype, elements)
smem_byte_offset(region, slot, lane_or_atom)
tmem_column(base, buffer, band, row_band)
tensor_map(name, rank, scalar_fields)
```

Directional movement and computation are deliberately primitive:

```python
copy_g2s(src_map, coords, smem_byte, completion, multicast_mask)
copy_s2t(smem_descriptor, tmem_column)
copy_t2r(tmem_column, registers)
copy_r2g(registers, global_pointer, predicate)
cast(dst, src, rounding)
gemm(dst, a_descriptor, b_descriptor, sfa_column, sfb_column,
     instruction_descriptor, accumulate)
```

`init`, `wait`, `expect_bytes`, `arrive`, `commit`, `fence`, `barrier`,
`allocate_tmem`, `relinquish_tmem`, `deallocate_tmem`, and cursor updates are
schedule operations. Descriptor creation below is ordinary scalar shift/mask/OR
arithmetic yielding raw PTX operands; it creates no mapping object. One
`copy_g2s`, `copy_s2t`, `copy_t2r`, `copy_r2g`, or `gemm` denotes one
instruction or an explicitly stated loop of one instruction family.

## Complete sketch

```python
# Static target and runtime ABI.
@kernel(
    target="sm_103a",
    grid_in_clusters=(1, 1, 76),
    clusters=(2, 1, 1),
    block=(224, 1, 1),
    min_blocks_per_sm=1,
    dynamic_smem_bytes=230400,
)
# instruction_selection: `.version 9.3`, `.target sm_103a`,
#   `.reqntid 224,1,1`, `.minnctapersm 1`, `.reqnctapercluster 2,1,1`,
#   `.blocksareclusters`, and `.extern .shared .align 1024`; extent: one entry
def fastcu_nvfp4_gemm_gb300(
    A_tmap, B_tmap, SFA_tmap, SFB_tmap,  # four by-value 128-byte TensorMaps
    C,                                   # f16 [M,N]
    M, N, K,                             # runtime i32 source ABI
    route_table,                         # read-only i32 [4096]
    sm_side, cluster_side,               # read-only i32 [256], [128]
    placement_errors,                    # u32 [1]
):
    CTA_GROUP = 2
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_N_PEER = 128
    MMA_K = 96
    AB_RING = 6
    SF_RING = 7
    D_STRIDE = 220
    SF_TMEM_BASE = 476
    TMEM_COLUMNS = 512
    AB_TX = 65536
    SF_TX = 9216

    tid = thread_id()
    warp = tid >> 5
    lane = tid & 31
    crank = cluster_cta_rank()
    m_pair = crank & 1
    cluster_id = cluster_id_from_block()
    num_clusters = grid_cluster_count()

    cluster_grid_m = ceil_div(M, 256)
    grid_n = ceil_div(N, 256)
    total_tiles = cluster_grid_m * grid_n
    full_groups = K // 768
    tail_cells = ceil_div(K - full_groups * 768, 96)
    t_sf = ceil_div(tail_cells, 2)
    num_groups = full_groups + (tail_cells != 0)
    k_rem = K - full_groups * 768
    b64_try = 1 if k_rem % 96 == 64 else (2 if k_rem % 96 == 32 else 0)
    a64 = tail_cells - b64_try
    b64 = b64_try if b64_try != 0 and k_rem % 32 == 0 and a64 >= 0 and even(a64) else 0
    t_win = ceil_div(k_rem // 2, 128) if b64 else 3

    # One source SmemCD object becomes one rank-one K u8 allocation.
    smem = linear_smem("smem", 230400, alignment=1024)
    smem_base = shared_address(smem)
    AB_READY = [0 + 8*s for s in range(6)]
    AB_FREE = [48 + 8*s for s in range(6)]
    SF_READY = [96 + 8*s for s in range(7)]
    SF_FREE = [152 + 8*s for s in range(7)]
    ACC_READY = 208
    ACC_FREE = 216
    DEALLOC = 224
    TMEM_ADDR = 232
    A_BASE = 1024
    B_BASE = 99328
    SFA_BASE = 197632
    SFB_BASE = 208896

    def a_slot(slot): return A_BASE + slot * 16384
    def b_slot(slot): return B_BASE + slot * 16384
    def sfa_slot(slot): return SFA_BASE + slot * 1536
    def sfb_slot(slot): return SFB_BASE + slot * 3072

    # Scalar descriptor/index mappings; these produce raw integer operands,
    # never a layout value.
    def sf_cp_src_desc(region_base, slot, slot_u, tile_u):
        base = smem_base + region_base
        fixed = ((base >> 4) & 0x3fc0) | 0x400800010000
        return fixed + slot*slot_u + tile_u

    # The single reused TMEM SF area is overwritten only after its batch MMAs.
    SF_COPY = (
        # region, source tile, slot stride in 16B units, destination column
        (SFA_BASE, 0,  96, 476), (SFA_BASE, 1,  96, 480),
        (SFA_BASE, 2,  96, 484), (SFB_BASE, 0, 192, 488),
        (SFB_BASE, 3, 192, 492), (SFB_BASE, 1, 192, 496),
        (SFB_BASE, 4, 192, 500), (SFB_BASE, 2, 192, 504),
        (SFB_BASE, 5, 192, 508),
    )
    # instruction_selection: the nine entries above issue nine predicated
    #   `tcgen05.cp.cta_group::2.32x128b.warpx4` instructions in exact
    #   SFA 0,1,2; SFB 0,3,1,4,2,5 source order.

    def ab_desc(region_base, slot, atom):
        base = smem_base + region_base + slot*16384
        addr = ((base >> 4) & 0x7fc0) + atom
        lbo = (base << 12) & 0x7fc00000
        return addr | lbo | 0x4010404000000000

    def ab_desc_straddle(region_base, addr_slot, lbo_slot, atom):
        return ((ab_desc(region_base, addr_slot, atom) & 0xffff)
                | (ab_desc(region_base, lbo_slot, 0) & 0xffff0000)
                | (ab_desc(region_base, addr_slot, atom) & 0xffffffff00000000))

    def ab_desc_k64(region_base, slot, atom):
        d = ab_desc(region_base, slot, atom)
        return (d & 0xffff) | (((d >> 32) & ~(1 << 20)) << 32)

    def sf_id_bits(words):
        sfa, sfb = words
        return ((sfa >> 1) & 0x60000000) | ((sfb >> 26) & 48)

    # Cell -> (A/B descriptor form, SF form). W0/W1/W2 name consecutive
    # A/B ring slots at this group entry.
    K96_CELL = (
        (0, "plain(W0,0)",             "word0"),
        (1, "plain(W0,3)",             "word1-with-id"),
        (2, "straddle(W0,W1,6)",       "word0"),
        (3, "plain(W1,1)",             "word1-with-id"),
        (4, "plain(W1,4)",             "word0"),
        (5, "straddle(W1,W2,7)",       "word1-with-id"),
        (6, "plain(W2,2)",             "word0"),
        (7, "plain(W2,5)",             "word1-with-id"),
    )
    def sf_word0(taddr): return (taddr + 476, taddr + 488)
    def sf_word1(taddr):
        return ((taddr + 480) | 0x80000000,
                (taddr + 496) | 0x80000000)
    def idesc96_word0(taddr): return 0x90400480 | sf_id_bits(sf_word0(taddr))
    def idesc96_word1(taddr): return 0x90400480 | sf_id_bits(sf_word1(taddr))
    # K64 legacy operands mask descriptor lows to 16 bits and clear descriptor
    # bit 52 (high-word bit 20); idesc bit 31 is clear. Its two forms use plain
    # id-0 columns (476,488) and (480,496), respectively.
    def k64_word0(taddr): return (taddr + 476, taddr + 488)
    def k64_word1(taddr): return (taddr + 480, taddr + 496)
    def idesc64_word0(taddr): return 0x10400480 | sf_id_bits(k64_word0(taddr))
    def idesc64_word1(taddr): return 0x10400480 | sf_id_bits(k64_word1(taddr))

    def stage_sf(slot, phase):
        wait(SF_READY[slot], phase, acquire=None)
        # instruction_selection: retry loop around
        #   `mbarrier.try_wait.parity.shared.b64`; extent: one live SF batch
        for region, tile, slot_u, dst_col in SF_COPY:
            copy_s2t(sf_cp_src_desc(region, slot, slot_u, tile*32),
                     taddr + dst_col)
        commit(SF_FREE[slot], mask=0x3)
        # instruction_selection: one predicated multicast tcgen05 commit after
        #   the exact nine-copy sequence; extent: one live SF batch
        return advance_ring(slot, phase, 7)

    # The placement audit precedes all role dispatch.
    actual_side = load_readonly(sm_side[sm_id()])
    planned_side = load_readonly(cluster_side[cluster_id])
    # instruction_selection: source has two `ld.const.b32` at PTX 153/157;
    #   adaptation uses two `ld.global.nc.b32`; extent: two scalar loads/thread
    if tid == 0 and actual_side != planned_side:
        atomic_add(placement_errors[0], 1)
        # instruction_selection: predicated `atom.global.add.u32`; extent: one audit site

    # Warp 0 lane 0 initializes the source fields in declaration order.
    if warp == 0 and lane == 0:
        for s in static_range(6):
            init(AB_READY[s], 1)
            # instruction_selection: `mbarrier.init.shared.b64`; extent: one slot
            init(AB_FREE[s], 1)
            # instruction_selection: `mbarrier.init.shared.b64`; extent: one slot
        for s in static_range(7):
            init(SF_READY[s], 1)
            # instruction_selection: `mbarrier.init.shared.b64`; extent: one slot
            init(SF_FREE[s], 1)
            # instruction_selection: `mbarrier.init.shared.b64`; extent: one slot
        init(ACC_READY, 1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: one slot
        init(ACC_FREE, 8)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: one slot
        init(DEALLOC, 32)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: one slot
        fence_mbarrier_init_release_cluster()
        # instruction_selection: `fence.mbarrier_init.release.cluster`; extent: one fence
    cluster_barrier_release_acquire()
    # instruction_selection: `barrier.cluster.arrive.release.aligned` then
    #   `barrier.cluster.wait.acquire.aligned`; extent: all threads in both CTAs

    # Warps 0-4 see the TMEM grant; TMA producer warps do not wait here.
    if warp <= 4:
        if warp == 0:
            allocate_tmem(smem + TMEM_ADDR, 512)
            # instruction_selection: `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32`;
            #   extent: all warp-0 lanes as required by the instruction
        barrier(id=2, threads=160)
        # instruction_selection: `barrier.cta.sync`; extent: warps 0-4
        taddr = load_shared_u32(smem + TMEM_ADDR)
        # instruction_selection: `ld.shared.b32`; extent: one scalar/thread in warps 0-4

    def producer_tile(t):
        # Source PTX has one scalar `ld.const.b32` in each producer copy of this helper.
        tile = load_readonly(route_table[t])
        # instruction_selection: intentional adaptation `ld.global.nc.b32`;
        #   extent: one lane-0 scalar load per persistent tile in warp 5 or 6
        return tile

    def epilogue_tile(t):
        tile_lane = load_readonly(route_table[t])
        # instruction_selection: intentional adaptation `ld.global.nc.b32`;
        #   extent: one scalar load per lane before uniformization
        tile = warp_min(tile_lane, mask=0xffffffff)
        # instruction_selection: `redux.sync.min.u32`; extent: one warp collective
        return tile

    # Warp 5 lane 0: A/B TMA producer, six-slot cursor starts phase 1.
    if warp == 5 and lane == 0:
        slot, free_phase = 0, 1
        for t in range(cluster_id, total_tiles, num_clusters):
            tile = producer_tile(t)
            m_row, n_group = tile % cluster_grid_m, tile // cluster_grid_m
            m_block = 2*m_row + m_pair
            n_block = 2*n_group + m_pair
            for group in rolled_range(num_groups):
                for sub in static_range(3):
                    if group == full_groups and sub >= t_win:
                        break
                    wait(AB_FREE[slot], free_phase, acquire="cta")
                    # instruction_selection: retry loop around
                    #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                    #   extent: one live A/B window
                    if m_pair == 0:
                        expect_bytes(AB_READY[slot] with peer_bit_cleared, AB_TX)
                        # instruction_selection:
                        #   `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`;
                        #   extent: one even-CTA arrival per live A/B window
                    copy_g2s(A_tmap, (group*384 + sub*128, m_block*128, 0),
                             a_slot(slot), AB_READY[slot], 1 << m_pair)
                    # instruction_selection:
                    #   `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global`
                    #   `.mbarrier::complete_tx::bytes.multicast::cluster`;
                    #   extent: one 128x128-byte source box
                    copy_g2s(B_tmap, (group*384 + sub*128, n_block*128, 0),
                             b_slot(slot), AB_READY[slot], multicast_mask=None)
                    # instruction_selection:
                    #   `cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global`
                    #   `.mbarrier::complete_tx::bytes`; extent: one 128x128-byte source box
                    slot, free_phase = advance_ring(slot, free_phase, 6)
        last_slot, last_phase = previous_ring(slot, free_phase, 6)
        wait(AB_FREE[last_slot], last_phase, acquire="cta")
        # instruction_selection: same acquire CTA parity retry loop; extent: producer tail
        if m_pair == 0:
            expect_bytes(AB_READY[last_slot] with peer_bit_cleared, AB_TX)
            # instruction_selection: same expect-tx instruction; extent: clean-shutdown token

    # Warp 6 lane 0: scale TMA producer, seven-slot cursor starts phase 1.
    elif warp == 6 and lane == 0:
        slot, free_phase = 0, 1
        for t in range(cluster_id, total_tiles, num_clusters):
            tile = producer_tile(t)
            m_row, n_group = tile % cluster_grid_m, tile // cluster_grid_m
            m_block = 2*m_row + m_pair
            for group in rolled_range(num_groups):
                for sub in static_range(4):
                    if group == full_groups and sub >= t_sf:
                        break
                    wait(SF_FREE[slot], free_phase, acquire="cta")
                    # instruction_selection: retry loop around
                    #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                    #   extent: one live scale slot
                    if m_pair == 0:
                        expect_bytes(SF_READY[slot] with peer_bit_cleared, SF_TX)
                        # instruction_selection: same expect-tx instruction;
                        #   extent: one even-CTA arrival per live scale slot
                    sf_group = (group*4 + sub)*3
                    copy_g2s(SFA_tmap, (0, 0, sf_group, m_block),
                             sfa_slot(slot), SF_READY[slot], 1 << m_pair)
                    # instruction_selection:
                    #   `cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global`
                    #   `.mbarrier::complete_tx::bytes.multicast::cluster`;
                    #   extent: one 128x4x3 scale box
                    copy_g2s(SFB_tmap, (0, 0, sf_group, 2*n_group + m_pair),
                             sfb_slot(slot) + m_pair*1536, SF_READY[slot], 0x3)
                    # instruction_selection: same rank-4 multicast family;
                    #   extent: one peer half of the byte-identical SFB slot
                    slot, free_phase = advance_ring(slot, free_phase, 7)
        last_slot, last_phase = previous_ring(slot, free_phase, 7)
        wait(SF_FREE[last_slot], last_phase, acquire="cta")
        # instruction_selection: same acquire CTA parity retry loop; extent: producer tail
        if m_pair == 0:
            expect_bytes(SF_READY[last_slot] with peer_bit_cleared, SF_TX)
            # instruction_selection: same expect-tx instruction; extent: clean-shutdown token

    # Warp 4 lane 0 of the even CTA: the only copy/MMA issuer.
    elif warp == 4 and m_pair == 0 and lane == 0:
        state = (d_parity=1, ab_phase=0, ab_slot=0, sf_phase=0, sf_slot=0)
        for t in range(cluster_id, total_tiles, num_clusters):
            acc_wait_phase = state.d_parity
            state.d_parity ^= 1
            d_tmem = taddr + state.d_parity*220
            accumulate = False

            # Each complete group is rolled; SF staging and MMAs are
            # deliberately interleaved around the one reused TMEM SF area.
            for group in rolled_range(full_groups):
                W0, ph0 = state.ab_slot, state.ab_phase
                W1, ph1 = advance_ring(W0, ph0, 6)
                W2, ph2 = advance_ring(W1, ph1, 6)

                # Batch 0: cells 0/1, atoms 0/3.
                state.sf_slot, state.sf_phase = stage_sf(
                    state.sf_slot, state.sf_phase)
                wait(AB_READY[W0], ph0, acquire=None)
                # instruction_selection: relaxed parity mbarrier retry loop;
                #   extent: W0 once per complete group
                if group == 0:
                    wait(ACC_FREE, acc_wait_phase, acquire=None)
                    # instruction_selection: gated relaxed parity retry loop;
                    #   extent: exactly once per output tile
                gemm(cell=K96_CELL[0], windows=(W0,W1,W2),
                     idesc=idesc96_word0(taddr), sf=sf_word0(taddr),
                     accumulate=accumulate)
                gemm(cell=K96_CELL[1], windows=(W0,W1,W2),
                     idesc=idesc96_word1(taddr), sf=sf_word1(taddr), accumulate=True)
                accumulate = True

                # Batch 1: cells 2/3, atoms 6/1.
                state.sf_slot, state.sf_phase = stage_sf(
                    state.sf_slot, state.sf_phase)
                wait(AB_READY[W1], ph1, acquire=None)
                gemm(cell=K96_CELL[2], windows=(W0,W1,W2),
                     idesc=idesc96_word0(taddr), sf=sf_word0(taddr), accumulate=True)
                commit(AB_FREE[W0], mask=0x3)  # immediately after atom 6
                gemm(cell=K96_CELL[3], windows=(W0,W1,W2),
                     idesc=idesc96_word1(taddr), sf=sf_word1(taddr), accumulate=True)

                # Batch 2: atom 4 must issue before the W2 wait; atom 7 follows.
                state.sf_slot, state.sf_phase = stage_sf(
                    state.sf_slot, state.sf_phase)
                gemm(cell=K96_CELL[4], windows=(W0,W1,W2),
                     idesc=idesc96_word0(taddr), sf=sf_word0(taddr), accumulate=True)
                wait(AB_READY[W2], ph2, acquire=None)
                state.ab_slot, state.ab_phase = advance_ring(W2, ph2, 6)
                gemm(cell=K96_CELL[5], windows=(W0,W1,W2),
                     idesc=idesc96_word1(taddr), sf=sf_word1(taddr), accumulate=True)

                # Batch 3: r9 delays W1 release until after this SF copy.
                state.sf_slot, state.sf_phase = stage_sf(
                    state.sf_slot, state.sf_phase)
                commit(AB_FREE[W1], mask=0x3)
                gemm(cell=K96_CELL[6], windows=(W0,W1,W2),
                     idesc=idesc96_word0(taddr), sf=sf_word0(taddr), accumulate=True)
                gemm(cell=K96_CELL[7], windows=(W0,W1,W2),
                     idesc=idesc96_word1(taddr), sf=sf_word1(taddr), accumulate=True)
                commit(AB_FREE[W2], mask=0x3)
                # instruction_selection for the eight gemm sites: predicated
                #   `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X`;
                #   exact atom order 0,3,6,1,4,7,2,5. The three releases use
                #   predicated tcgen05 multicast mbarrier commits.

            if tail_cells != 0:
                W0, ph0 = state.ab_slot, state.ab_phase
                W1, ph1 = advance_ring(W0, ph0, 6)
                W2, ph2 = advance_ring(W1, ph1, 6)

                if b64 != 0:
                    # Every tuple is cell-indexed in issue order. 96p/96x use
                    # K96_CELL's plain/straddle descriptor and alternating SF
                    # word; 64a/64b use legacy low16 + bit52-clear descriptors,
                    # idesc64_word0/1, and k64_word0/1.
                    K64_VARIANT = {
                      (0,1): ("64a(W0,0)",),
                      (0,2): ("64a(W0,0)", "64b(W0,2)"),
                      (2,1): ("96p(W0,0,w0)", "96p(W0,3,w1)",
                              "64a(W0,6)"),
                      (2,2): ("96p(W0,0,w0)", "96p(W0,3,w1)",
                              "64a(W0,6)", "64b(W1,0)"),
                      (4,1): ("96p(W0,0,w0)", "96p(W0,3,w1)",
                              "96x(W0,W1,6,w0)", "96p(W1,1,w1)",
                              "64a(W1,4)"),
                      (4,2): ("96p(W0,0,w0)", "96p(W0,3,w1)",
                              "96x(W0,W1,6,w0)", "96p(W1,1,w1)",
                              "64a(W1,4)", "64b(W1,6)"),
                      (6,1): ("96p(W0,0,w0)", "96p(W0,3,w1)",
                              "96x(W0,W1,6,w0)", "96p(W1,1,w1)",
                              "96p(W1,4,w0)", "96x(W1,W2,7,w1)",
                              "64a(W2,2)"),
                      (6,2): ("96p(W0,0,w0)", "96p(W0,3,w1)",
                              "96x(W0,W1,6,w0)", "96p(W1,1,w1)",
                              "96p(W1,4,w0)", "96x(W1,W2,7,w1)",
                              "64a(W2,2)", "64b(W2,4)"),
                    }
                    variant = K64_VARIANT[(a64, b64)]
                    N_C = a64 + b64
                    T_WIN = 1 if (a64,b64) in ((0,1),(0,2),(2,1)) else (
                            2 if a64 in (2,4) else 3)

                    # Batch 0, cells 0/1.
                    state.sf_slot, state.sf_phase = stage_sf(
                        state.sf_slot, state.sf_phase)
                    wait(AB_READY[W0], ph0, acquire=None)
                    wait_gated(ACC_FREE, acc_wait_phase,
                               skip=(full_groups != 0))
                    issue_if_present(variant, 0, overwrite=(full_groups == 0))
                    issue_if_present(variant, 1, overwrite=False)

                    # Batch 1 then cells 2/3. The W1 wait is source-positioned
                    # before cell 2 whenever the variant consumes W1.
                    if N_C > 2:
                        state.sf_slot, state.sf_phase = stage_sf(
                            state.sf_slot, state.sf_phase)
                    if T_WIN >= 2:
                        wait(AB_READY[W1], ph1, acquire=None)
                    issue_if_present(variant, 2, overwrite=False)
                    commit(AB_FREE[W0], mask=0x3)
                    issue_if_present(variant, 3, overwrite=False)

                    # Batch 2 then cells 4/5. W2 is waited and the A/B cursor
                    # written back after cell 4 but before cell 5.
                    if N_C > 4:
                        state.sf_slot, state.sf_phase = stage_sf(
                            state.sf_slot, state.sf_phase)
                    issue_if_present(variant, 4, overwrite=False)
                    if T_WIN >= 3:
                        wait(AB_READY[W2], ph2, acquire=None)
                    if T_WIN == 1:
                        state.ab_slot, state.ab_phase = W1, ph1
                    elif T_WIN == 2:
                        state.ab_slot, state.ab_phase = W2, ph2
                    else:
                        state.ab_slot, state.ab_phase = advance_ring(W2, ph2, 6)
                    issue_if_present(variant, 5, overwrite=False)

                    # Batch 3, if live, precedes the deliberately delayed W1
                    # release; cells 6/7 and conditional W2 release follow.
                    if N_C > 6:
                        state.sf_slot, state.sf_phase = stage_sf(
                            state.sf_slot, state.sf_phase)
                    if T_WIN >= 2:
                        commit(AB_FREE[W1], mask=0x3)
                    issue_if_present(variant, 6, overwrite=False)
                    issue_if_present(variant, 7, overwrite=False)
                    if T_WIN >= 3:
                        commit(AB_FREE[W2], mask=0x3)
                    # instruction_selection: the selected tuple expands to
                    #   exactly N_C separately materialized mxf4nvf4 MMA sites;
                    #   96p/96x set idesc bit31, 64a/64b clear it.

                else:
                    # Ordinary tail: all three A/B windows exist. SF batches
                    # exist only for their live cell pair and skipped batches
                    # advance no SF cursor.
                    state.sf_slot, state.sf_phase = stage_sf(
                        state.sf_slot, state.sf_phase)
                    wait(AB_READY[W0], ph0, acquire=None)
                    wait_gated(ACC_FREE, acc_wait_phase,
                               skip=(full_groups != 0))
                    gemm(cell=K96_CELL[0], predicate=True,
                         accumulate=(full_groups != 0))
                    gemm(cell=K96_CELL[1], predicate=(tail_cells >= 2),
                         accumulate=True)

                    if tail_cells > 2:
                        state.sf_slot, state.sf_phase = stage_sf(
                            state.sf_slot, state.sf_phase)
                    wait(AB_READY[W1], ph1, acquire=None)
                    gemm(cell=K96_CELL[2], predicate=(tail_cells >= 3),
                         accumulate=True)
                    commit(AB_FREE[W0], mask=0x3)
                    gemm(cell=K96_CELL[3], predicate=(tail_cells >= 4),
                         accumulate=True)

                    if tail_cells > 4:
                        state.sf_slot, state.sf_phase = stage_sf(
                            state.sf_slot, state.sf_phase)
                    gemm(cell=K96_CELL[4], predicate=(tail_cells >= 5),
                         accumulate=True)
                    wait(AB_READY[W2], ph2, acquire=None)
                    state.ab_slot, state.ab_phase = advance_ring(W2, ph2, 6)
                    gemm(cell=K96_CELL[5], predicate=(tail_cells >= 6),
                         accumulate=True)

                    if tail_cells > 6:
                        state.sf_slot, state.sf_phase = stage_sf(
                            state.sf_slot, state.sf_phase)
                    commit(AB_FREE[W1], mask=0x3)
                    gemm(cell=K96_CELL[6], predicate=(tail_cells >= 7),
                         accumulate=True)
                    gemm(cell=K96_CELL[7], predicate=(tail_cells >= 8),
                         accumulate=True)
                    commit(AB_FREE[W2], mask=0x3)
                    # instruction_selection: eight predicated mxf4nvf4 K96
                    #   sites in atom order 0,3,6,1,4,7,2,5; liveness is the
                    #   issuer predicate and all waits/releases retain source order.

            commit(ACC_READY, mask=0x3)
            # instruction_selection:
            #   `tcgen05.commit.cta_group::2.mbarrier::arrive::one`
            #   `.shared::cluster.multicast::cluster.b64`; extent: one output tile
        wait(ACC_FREE, state.d_parity, acquire=None)
        # instruction_selection: relaxed parity retry loop; extent: MMA drain

    # Warps 0-3 in both CTAs: direct TMEM-to-global epilogue. All unmatched
    # lanes in warps 4-6 are idle, matching the source's outer wg==1 branch.
    elif warp <= 3:
        epi_warp = warp
        row_band_taddr = (epi_warp*32) << 16
        d_buffer = 0
        acc_phase = 0
        for t in range(cluster_id, total_tiles, num_clusters):
            tile = epilogue_tile(t)
            m_row, n_group = tile % cluster_grid_m, tile // cluster_grid_m
            wait(ACC_READY, acc_phase, acquire=None)
            # instruction_selection: relaxed parity retry loop; extent: one output tile
            acc_phase ^= 1
            row = (2*m_row + m_pair)*128 + epi_warp*32 + lane
            for k in rolled_range(8):
                band = 7-k if d_buffer == 0 else k
                words = reg_array("words", "u32", 32)
                copy_t2r(taddr + d_buffer*220 + band*32 + row_band_taddr, words)
                # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x32.b32`;
                #   extent: one 32-row x 32-column band per epilogue warp
                if k == 1:
                    wait_tmem_loads()
                    # instruction_selection: `tcgen05.wait::ld.sync.aligned`;
                    #   extent: one early-release site per tile/band walk
                    if lane == 0:
                        arrive_remote(ACC_FREE, even_cta_rank=crank & ~1, release="cta")
                        # instruction_selection:
                        #   `mbarrier.arrive.release.cta.shared::cluster.b64`;
                        #   extent: one lane per epilogue warp after the fused wait
                if row < M:
                    half_words = reg_array("half_words", "u32", 16)
                    for j in static_range(16):
                        cast(half_words[j], (words[2*j+1], words[2*j]), rounding="rn")
                        # instruction_selection: `cvt.rn.f16x2.f32`;
                        #   extent: one packed pair
                    col = n_group*256 + band*32
                    if output_pointer_and_row_are_32B_aligned and col % 16 == 0:
                        for q in static_range(2):
                            c16 = col + q*16
                            if c16 + 16 <= N:
                                copy_r2g(half_words[q*8:q*8+8],
                                         C[row,c16:c16+16], predicate=True)
                                # instruction_selection:
                                #   `st.global.L1::no_allocate.L2::evict_first.v8.b32`;
                                #   extent: one complete aligned 16-half segment
                            else:
                                for p in static_range(2):
                                    c8 = c16 + p*8
                                    if c8 + 8 <= N:
                                        copy_r2g(half_words[q*8+p*4:q*8+p*4+4],
                                                 C[row,c8:c8+8], predicate=True)
                                        # instruction_selection:
                                        #   `st.global.L1::no_allocate.v4.b32`;
                                        #   extent: one complete 8-half tail segment
                                    else:
                                        for h in static_range(8):
                                            copy_r2g(half_at(half_words,q,p,h),
                                                     C[row,c8+h],
                                                     predicate=(c8+h < N))
                                            # instruction_selection: predicated
                                            #   `st.global.b16`; extent: one half
                    else:
                        for q in static_range(4):
                            c8 = col + q*8
                            if c8 + 8 <= N:
                                copy_r2g(half_words[q*4:q*4+4],
                                         C[row,c8:c8+8], predicate=True)
                                # instruction_selection:
                                #   `st.global.L1::no_allocate.v4.b32`;
                                #   extent: one complete 8-half segment
                            else:
                                for h in static_range(8):
                                    copy_r2g(half_at(half_words,q,h),
                                             C[row,c8+h],
                                             predicate=(c8+h < N))
                                    # instruction_selection: predicated
                                    #   `st.global.b16`; extent: one half
            d_buffer ^= 1

    # Only epilogue warps rendezvous and deallocate.
    if warp <= 3:
        barrier(id=3, threads=128)
        # instruction_selection: `barrier.cta.sync`; extent: four epilogue warps
        if warp == 0:
            relinquish_tmem()
            # instruction_selection:
            #   `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned`;
            #   extent: warp 0
            arrive_remote(DEALLOC, peer_rank=crank ^ 1, release="cta", all_lanes=True)
            # instruction_selection: `mapa.shared::cluster.u32` plus
            #   `mbarrier.arrive.release.cta.shared::cluster.b64`;
            #   extent: all 32 lanes arrive at peer's count-32 barrier
            wait(DEALLOC, phase=0, acquire="cta")
            # instruction_selection: retry loop around
            #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            #   extent: one teardown wait
            deallocate_tmem(taddr, 512)
            # instruction_selection:
            #   `tcgen05.dealloc.cta_group::2.sync.aligned.b32`; extent: warp 0
```

## TensorMap fields and explicit physical mappings

| map | fastest-first global dimensions | byte strides for dimensions 1+ | box | swizzle / OOB |
| --- | --- | --- | --- | --- |
| A | `(K/2, M, 1)` | `(aligned(K/2,16), M*aligned(K/2,16))` | `(128,128,1)` u8 | 128 B / zero |
| B | `(K/2, N, 1)` | `(aligned(K/2,16), N*aligned(K/2,16))` | `(128,128,1)` u8 | 128 B / zero |
| SFA | `(128,4,ceil(K/64),ceil(M/128))` | `(128,512,ceil(K/64)*4*128)` | `(128,4,3,1)` u8 | none / zero |
| SFB | `(128,4,ceil(K/64),ceil(N/128))` | `(128,512,ceil(K/64)*4*128)` | `(128,4,3,1)` u8 | none / zero |

Scale storage uses the source VEC16 offset exactly:
`((sf_inner//4)*4 + outer_block*sf_inner_dim)*128 + (outer%32)*16 +
(outer//32)*4 + sf_inner%4`. A/B descriptor atom order is
`0,3,6,1,4,7,2,5`; atoms 6 and 7 splice the address half from the current
128-byte window with the absolute-LBO half from the next window.

## Pipeline inventory

| edge | slots / initial phase | producer | consumer | completion / release |
| --- | --- | --- | --- | --- |
| A/B | 6 / producer free phase 1, MMA ready phase 0 | warp 5 lane 0 in both CTAs | warp 4 lane 0 in even CTA | even CTA expect 65,536 bytes; tcgen05 commit multicast to both `ab_free` copies at each last reader |
| scale | 7 / producer free phase 1, MMA ready phase 0 | warp 6 lane 0 in both CTAs | warp 4 lane 0 in even CTA | even CTA expect 9,216 bytes; tcgen05 commit multicast after nine scale copies |
| accumulator | singleton barriers, two TMEM buffers | warp 4 | eight epilogue warps across two CTAs | tcgen05 commit to both `acc_ready`; each epilogue warp lane 0 remotely arrives at even `acc_free` after the shared edge drains |
| deallocation | singleton count 32 per CTA | all lanes of peer warp 0 | local warp 0 | release remote arrive, acquire local parity wait |

## Exact-K dispatch table

| remainder class | body | A/B windows | scale batches | instruction descriptor |
| --- | --- | ---: | ---: | --- |
| 0 after full K=768 groups | no tail | 0 | 0 | none |
| ordinary K%16 tail | 1..8 K96 cells with predicated dead sites | 3 | `ceil(tail_cells/2)` | `0x90400480` plus scale IDs |
| active K64 `a=0,2,4,6`, `b=1` | `a` K96 cells then one K64 cell | `ceil((96a+64)/256)` | `ceil((a+1)/2)` | K96 as above; K64 clears bit 31 and descriptor bit 52 |
| active K64 `a=0,2,4,6`, `b=2` | `a` K96 cells then two K64 cells | `ceil((96a+128)/256)` | `ceil((a+2)/2)` | second K64 uses the source's plain word-1 scale columns |

The K64 branch is active only when the derived `b64_try` is nonzero, the
remainder is divisible by 32, and `a` is nonnegative and even. All other
remainders take the ordinary K96 tail. Cursor movement follows issued windows
and batches, never the maximum body size.

## L2 schedule and measured address-space adaptation

The host performs the source's census and ownership construction before either
timed implementation. It supplies the exact selected route order:

- below 128 MiB operand reads: owned-pocket-8x8 only for a 32x32 tile grid,
  otherwise owned-plain;
- at or above 128 MiB: owned-pocket-8x4 for near-square grids with low side at
  least 48 and aspect at most 1.25, otherwise Hilbert-in-owned;
- no route may exceed 4096 tiles; every table must pass exactly-once coverage
  and side-ownership verification;
- the physical SM-to-side and cluster-to-side arrays accompany the route table,
  and a nonzero post-launch placement counter invalidates correctness or timing.

CUDA r9 declares the three read-only arrays in `.const`, producing five static
`ld.const.b32` sites in the writer PTX: two audit loads, one in each producer
body, and one epilogue load followed by `redux.sync.min.u32`. The public K API
has no module-scope constant declaration or `ld.const` operand. The port
therefore accepts read-only `K.gptr` buffers and uses `ld.global.nc.b32` while
preserving load ownership, location, cadence, uniformization, table contents,
and the placement atomic. This is the sole device-algorithm adaptation; it is
accepted only if the reviewer finds it auditable, bitwise correctness passes,
and every requested `bench_suite` ratio exceeds 0.99.

## TIRx module and benchmark contract

- The executable imports the device language only as
  `import tirx_kernels.kern as K`. It uses rank-one shared allocation, scalar
  offset functions, `K.TensorMap`, registers, K control flow, and raw `K.ptx`.
  It uses no tile primitive, layout object, inline CUDA device call, IR checker
  exemption, or change under `tirx_kernels/kern/`.
- Correctness runs source and TIRx on byte-identical packed inputs and requires
  bitwise equality of the complete FP16 output, finite results, complete poison
  overwrite, unchanged input canaries, a zero placement counter, and repeated
  deterministic output. The host/cublas checks retain the source's `atol=0.5`,
  `rtol=2e-3` as secondary checks and never replace the bitwise source comparison.
- The correctness matrix covers ragged K%16, both active K64 subclasses, an
  inactive K64-looking remainder, an aligned tail, M/N boundaries, and the five
  benchmark squares without repeatedly running the full matrix during tuning.
- `prepare_bench` compiles before GPU timing and returns a prepared benchmark.
  `run_gpu` allocates once, validates once, and gives the canonical Proton timer
  exactly one source or TIRx GEMM launch per closure. The lazy reference name is
  `fastcu_gemm9`; the implementation name is `tirx`.
- Final performance authority is only `bench_suite`, with external references,
  Proton, five rounds, one-second cooldown, and the five square configs. Both
  source and TIRx PTX must say `.version 9.3` and `.target sm_103a` under the
  same CUDA 13.3 nvcc toolchain.

## Instruction-selection summary

| source decision | emitted consequence in writer PTX |
| --- | --- |
| by-value tensor maps, fixed block/cluster | four aligned 128-byte params; `.reqntid 224`; `.reqnctapercluster 2,1,1`; `.blocksareclusters` |
| `SmemCD` extern allocation | `.extern .shared .align 1024`, with source offsets A=1024, B=99328, SFA=197632, SFB=208896 |
| lane-0 TMA producers | rank-3 cta-group-2 A multicast and B non-multicast; rank-4 cta-group-2 SFA self-multicast and SFB mask-3 multicast |
| same-warp scale copy and MMA stream | `tcgen05.cp.cta_group::2.32x128b.warpx4` immediately ordered with `tcgen05.mma...mxf4nvf4...scale_vec::4X` and tcgen05 commits |
| K96 versus K64 | identical MMA mnemonic; idesc bit 31 selects K=96, legacy descriptor/id-0 scale forms select K=64 |
| overlap-aware epilogue drain | one x32 TMEM load per 32-column band, packed RN FP16 conversion, high-to-low/low-to-high alternating order, fused load wait before remote buffer release |
| write-once output policy | aligned full segments use `st.global.L1::no_allocate.L2::evict_first.v8.b32`; smaller segments use no-L1 v4; scalar b16 only for N tails |
| constant route load in epilogue | source `ld.const.b32` followed by `redux.sync.min.u32`; adaptation replaces only the load with `ld.global.nc.b32` |
