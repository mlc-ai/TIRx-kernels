<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 MXFP8-input projection GEMM + YARN RoPE sketch

This is the non-executable execution sketch for
[`tirx_kernels/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_mxfp8in.py`](../../tirx_kernels/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_mxfp8in.py).
It freezes the direct TE-native ABI, block-scaled MXFP8 mainloop, 14-warp
split, static persistent scheduler, dual pipeline, FP32-TMEM-to-BF16-SMEM
drain, YARN RoPE, dual-direction MXFP8 quantization, and TMEM teardown. Once
the independent sketch reviewer returns PASS, this file is immutable.

The authoritative source is
`python/cudnn/gemm/cutedsl/dense/proj_rope_mxfp8/gemm_proj_rope_mxfp8_mxfp8in.py`
at commit `aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5`. Its SHA256 is
`c2754a908368055f0fcbc9f044f06808b204e730ea94723888493b594198c34e`.
Source citations below refer to that file. The writer's independent,
cache-disabled line-info exports are under
`.porting/gemm_proj_rope_mxfp8_mxfp8in/writer_export/`:

- `tokens128`: PTX SHA256
  `44b54abc473ebea981da2c241973dd60a7fdf3689a1d05a1bd639fbfcf3ca159`;
  `PTX-128 n` means physical line `n` of the 7309-line PTX.
- `tokens2048`: PTX SHA256
  `5c08f06fdd9bf8bbee9b5902c73f6215a295f1180674a0c840461d04129d904a`;
  `PTX-2048 n` means physical line `n` of the 7560-line PTX.

Both use `K=1536`, `num_heads=128`, execute once, and match all four
upstream outputs with row and column match fractions 1.0. The first export
covers the non-persistent x32 T2R branch; the second covers persistent x8.

## No first-class layout invariant

The executable device body imports only `tirx_kernels.kern as K`. It may not
construct, pass, return, or store a first-class layout, invoke a tile
primitive, allocate multidimensional shared memory, call `K.cuda.func_call`,
embed CUDA source, or rely on a low-level-IR exemption. All shared storage is
one rank-1 dynamic `u8` arena. TensorMap fields, swizzles, descriptor fields,
scheduler coordinates, TMEM addresses, and fragment placement are explicit
integer arithmetic. “Tile” below names an algorithmic rectangle only.

## Frozen roles and publication edges

| role | warps | responsibility |
| --- | --- | --- |
| epilogue | 0-11 | warp 0 allocates 512 TMEM columns; warps 0-3 drain one 32-row accumulator band each; all 12 own one 32-token by 64-feature cell, apply the optional RoPE cell, and quantize row/column outputs |
| MMA | 12 | copies staged compact SFA/SFB into TMEM, issues block-scaled E4M3 MMA, releases AB stages, and publishes accumulator stages |
| TMA | 13 | relays 128 SFA rows and paired 256 SFB rows, publishes them to async-shared view, then issues A/B TensorMap loads |

AB has three full/empty stages with one arrival on each side. ACC has two
full/empty stages: one MMA full arrival and four T2R empty arrivals. Named
barrier 2 joins MMA plus all epilogue warps (416 threads); named barrier 1
joins all epilogue warps (384 threads) twice per work tile.

## Scalar primitive vocabulary

The pseudocode uses only scalar values, raw pointers, opaque host-created
TensorMaps, and explicit byte/TMEM-column offsets:

```python
encode_tensormap(base, dtype, extents, byte_stride, box, swizzle)
arena_ptr(byte_offset)
raw_umma_descriptor(base_bits, shared_byte_address)
pipeline_state(stage_count, initial_phase)
wait_mbarrier(barrier_byte, phase)
arrive_expect_tx(barrier_byte, byte_count)
tma_load_2d_cta(map, coord0, coord1, arena_byte, barrier_byte)
scale_s2t(tmem_column, shared_descriptor)
blockscaled_mma(acc_column, a_desc, b_desc, instruction_desc,
                sfa_column, sfb_column, accumulate)
umma_commit(barrier_byte)
tmem_load_x32_or_x8(register_words, tmem_address)
packed_f32x2_mul_or_fma(...)
convert_e8m0_or_e4m3x2(...)
global_load_or_store(...)
```

## Complete sketch

```python
# S1 =======================================================================
# Fixed specialization, direct ABI, TensorMaps, and launch
# source 18-71, 557-649; PTX-128/PTX-2048 14-42
# ==========================================================================
TILE_M = 128
TILE_N = HEAD_DIM = 192
K_TILE = 128
K_DIM = 1536
NUM_HEADS = 128
QK_NOPE = 128
QK_ROPE = 64
BLOCK = SF_VEC = 32
AB_STAGES = 3
ACC_STAGES = 2
NUM_K_TILES = 12
NUM_EPI_WARPS = 12
T2R_WARPS = 4
MMA_WARP = 12
TMA_WARP = 13
TMEM_COLUMNS = 512
SACC_STRIDE_BF16 = 196
assert tokens in (128, 256, 2048, 4096)

# This is the only supported device ABI. Codes and scales are physically
# bytes; cos/sin and the staging conversion are BF16.
ABI = (
    x_code_u8_ptr, x_scale_u8_ptr,
    w_code_u8_ptr, w_scale_u8_ptr,
    cos_bf16_ptr, sin_bf16_ptr,
    qrow_u8_ptr, srow_u8_ptr,
    qcol_u8_ptr, scol_u8_ptr,
)

# x_code is [tokens,K] and w_code is [heads*192,K], both row-major. TensorMap
# coordinate 0 is contiguous K. No wrapper weight-transpose orientation exists.
map_A = encode_tensormap(
    x_code_u8_ptr, dtype=e4m3, extents=(K_DIM, tokens),
    byte_stride=K_DIM, box=(128, 128), swizzle=SW128,
)
map_B = encode_tensormap(
    w_code_u8_ptr, dtype=e4m3, extents=(K_DIM, NUM_HEADS * HEAD_DIM),
    byte_stride=K_DIM, box=(128, 192), swizzle=SW128,
)

M_TILES = tokens // 128
TOTAL_WORK = M_TILES * NUM_HEADS
NUM_CLUSTERS = min(TOTAL_WORK, 148)
SWIZZLE = {128: 4, 256: 4, 2048: 16, 4096: 32}[tokens]
T2R_X8 = tokens >= 2048
launch(
    grid=(1, 1, NUM_CLUSTERS), block_threads=448,
    cluster=(1, 1, 1), dynamic_smem_bytes=177792, arch="sm_100a",
)
# instruction_selection: `.reqntid 448,1,1`, singleton CTA/cluster, one
#   `.extern .shared .align 1024`; extent: every specialization. The B200
#   residency query caps the z grid at 148. Source 596-649; both PTX 14-42.

# S2 =======================================================================
# One rank-1 u8 shared arena, raw descriptors, and TMEM partition
# source 43-54, 74-80, 222-228, 325-366; both PTX 93-119
# ==========================================================================
smem = rank1_dynamic_u8(177792, alignment=1024)

AB_FULL = 0                  # three u64 barriers: [0,24)
AB_EMPTY = 24                # three u64 barriers: [24,48)
ACC_FULL = 48                # two u64 barriers: [48,64)
ACC_EMPTY = 64               # two u64 barriers: [64,80)
TMEM_DEALLOC = 80            # one u64: [80,88)
TMEM_PTR = 88                # one u32, header padded through 128
A_BASE = 128                 # 3 * 16384 bytes; ends 49280
B_BASE = 49280               # 3 * 24576 bytes; ends 123008
SFA_BASE = 123008            # 3 * 512 bytes; ends 124544
SFB_BASE = 124544            # 3 * 1024 bytes; ends 127616
SACC_BASE = 127616           # 128 * 196 BF16; ends 177792

def barrier(base, stage):
    return arena_ptr(base + 8 * stage)

def a_stage_byte(stage):
    return A_BASE + stage * 16384

def b_stage_byte(stage):
    return B_BASE + stage * 24576

def sfa_stage_byte(stage):
    return SFA_BASE + stage * 512

def sfb_stage_byte(stage):
    return SFB_BASE + stage * 1024

def sacc_byte(row, feature):
    return SACC_BASE + 2 * (row * 196 + feature)

def desc_address(base_bits, byte_address):
    return uint64(base_bits) | uint64((shared_address(byte_address) >> 4) & 0x3fff)

AB_DESC_BASE = 0x4000404000010000
SF_DESC_BASE = 0x400800010000
MMA_INSTR_BASE = 0x08B00000
A_DESC = desc_address(AB_DESC_BASE, A_BASE)
B_DESC = desc_address(AB_DESC_BASE, B_BASE)
SFA_DESC = desc_address(SF_DESC_BASE, SFA_BASE)
SFB_DESC = desc_address(SF_DESC_BASE, SFB_BASE)

def a_desc(stage, kphase):
    return A_DESC + stage * 1024 + 2 * kphase

def b_desc(stage, kphase):
    return B_DESC + stage * 1536 + 2 * kphase

def sf_desc_sfa(stage):
    return SFA_DESC + stage * 32

def sf_desc_sfb(stage, half):
    return SFB_DESC + stage * 64 + half * 32

# TMEM columns are scalar integer addresses: ACC 0-383, SFA 384-399, and
# SFB 400-431. Odd heads shift the SFB MMA view by exactly two columns.
ACC_COL = 0
SFA_COL = 384
SFB_COL = 400

def mma_instruction_desc(sfa_column, sfb_column):
    return ((MMA_INSTR_BASE & 0x9fffffcf)
            | ((sfa_column >> 1) & 0x60000000)
            | ((sfb_column >> 26) & 0x30))

# instruction_selection: A/B operands are raw 64-bit shared descriptors,
#   scale-copy operands use raw SF descriptors, the accumulator and scale
#   destinations are raw TMEM columns, and the MMA descriptor is the patched
#   integer above. Extent: four K phases x 12 K tiles for every work tile.
#   Source 327-374; PTX-128 433-725, PTX-2048 450-751.

# S3 =======================================================================
# Warp roles, barrier initialization, scheduler coordinate lowering
# source 213-220, 230-254, 275-300
# ==========================================================================
warp = warp_uniform(warp_id())
lane = lane_id()
roles = chain_dispatch(
    epilogue=warps(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    mma=warps(12),
    tma=warps(13),
)

if warp == TMA_WARP:
    prefetch_tensormap(map_A)
    prefetch_tensormap(map_B)
# instruction_selection: two `prefetch.tensormap`; extent: one per input map
#   by the active TMA warp. Source 230-232; both PTX 89,92.

if warp == 0 and elect_one():
    for stage in unroll(3):
        mbarrier_init(barrier(AB_FULL, stage), arrivals=1)
if warp == 0 and elect_one():
    for stage in unroll(3):
        mbarrier_init(barrier(AB_EMPTY, stage), arrivals=1)
fence_mbarrier_init_release_cluster()
cta_barrier_all()
if warp == 0 and elect_one():
    for stage in unroll(2):
        mbarrier_init(barrier(ACC_FULL, stage), arrivals=1)
if warp == 0 and elect_one():
    for stage in unroll(2):
        mbarrier_init(barrier(ACC_EMPTY, stage), arrivals=4)
fence_mbarrier_init_release_cluster()
cta_barrier_all()
# instruction_selection: exactly ten `mbarrier.init.shared.b64`, two
#   `fence.mbarrier_init.release.cluster`, and two `bar.sync 0`; arrivals are
#   1/1/1/4 for AB-full/AB-empty/ACC-full/ACC-empty. Source 275-294; both PTX
#   126-173. Each barrier family has an independent warp election.

def scheduler_coord(work):
    s = SWIZZLE
    head_minor = work % s
    m_idx = (work // s) % M_TILES
    head = (work // (s * M_TILES)) * s + head_minor
    return m_idx, head

def advance_work(work):
    return work + NUM_CLUSTERS

# instruction_selection: only integer shifts/masks/division express the
#   scheduler. For tokens=2048, `m=(work>>4)&15` and
#   `head=((work>>4)&112)|(work&15)`; extent: independent cursors in all three
#   roles. Source 298-305,316-317,377-378,550-551; PTX-2048 178-241 and tail.

# S4 =======================================================================
# Warp 13: compact-scale relay followed by three-stage A/B TMA
# source 147-183, 301-318; PTX-128 249-412, PTX-2048 251-429
# ==========================================================================
with role("tma"):
    state = pipeline_state(stages=3, phase=1)
    work = cta_id_z()
    while work < TOTAL_WORK:
        m_idx, head = scheduler_coord(work)
        for ktile in serial(12):
            handle = acquire_and_advance(state)
            if elect_one():
                arrive_expect_tx(handle.full_barrier, 40960)

            # Every lane relays four SFA rows as one v4 b32 store.
            sfa_row0 = m_idx * 128 + lane
            sfa_word = ktile * 4
            sfa4 = [
                global_load_u32(x_scale_u8_ptr,
                    (sfa_row0 + row_block * 32) * 48 + sfa_word)
                for row_block in unroll(4)
            ]
            store_shared_v4_u32(
                arena_ptr(sfa_stage_byte(handle.stage) + lane * 16), sfa4
            )

            # The paired SFB relay starts 64 rows earlier for odd heads.
            sfb_row0 = head * 192 - (head & 1) * 64 + lane
            sfb8 = [
                global_load_u32(w_scale_u8_ptr,
                    (sfb_row0 + row_block * 32) * 48 + sfa_word)
                for row_block in unroll(8)
            ]
            store_shared_v4_u32(
                arena_ptr(sfb_stage_byte(handle.stage) + lane * 16), sfb8[0:4]
            )
            store_shared_v4_u32(
                arena_ptr(sfb_stage_byte(handle.stage) + 512 + lane * 16),
                sfb8[4:8],
            )

            sync_warp()
            fence_proxy_async_shared_cta()
            if elect_one():
                tma_load_2d_cta(
                    map_A, coord0=ktile * 128, coord1=m_idx * 128,
                    arena_byte=a_stage_byte(handle.stage),
                    barrier_byte=handle.full_barrier,
                )
            if elect_one():
                tma_load_2d_cta(
                    map_B, coord0=ktile * 128, coord1=head * 192,
                    arena_byte=b_stage_byte(handle.stage),
                    barrier_byte=handle.full_barrier,
                )
        work = advance_work(work)
    for unused in unroll(3):
        wait_mbarrier(barrier(AB_EMPTY, state.stage), state.phase)
        state.advance()

# protocol_order: `acquire_and_advance` clones the current stage/phase, waits
#   AB-empty, and advances the live cursor before relay/TMA; every subsequent
#   operation uses the cloned handle. The three tail waits use the live cursor.
# instruction_selection: AB acquire and all three tail waits use
#   `mbarrier.try_wait.parity.shared.b64`; extent: 12 acquires/work and three
#   tail waits. `mbarrier.arrive.expect_tx.shared.b64` arms exactly 40960
#   bytes. Source 309-318; first PTX wait/expect 249/264 (128) and 251/266
#   (2048).
# instruction_selection: scale relay is twelve `ld.global.b32` plus three
#   `st.shared.v4.b32` static sites, dynamically four SFA and eight SFB words
#   per lane/stage. It is followed in order by `bar.warp.sync -1` and
#   `fence.proxy.async.shared::cta`. Source 147-183,311-313; PTX-128 276-307,
#   PTX-2048 278-309.
# instruction_selection: each map copy has its own `elect.sync` predicate and
#   emits
#   `cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`.
#   Extent: one A plus one B copy per K tile, 12 K tiles/work. Source 314-315;
#   PTX-128 317,327; PTX-2048 319,329.

# S5 =======================================================================
# Warp 12: scale S2T, block-scaled E4M3 MMA, ACC publication
# source 320-379; PTX-128 433-725, PTX-2048 450-751
# ==========================================================================
with role("mma"):
    named_barrier(id=2, threads=416)
    tmem_base = load_shared_u32(TMEM_PTR)
    ab = pipeline_state(stages=3, phase=0)
    acc = pipeline_state(stages=2, phase=1)
    work = cta_id_z()
    while work < TOTAL_WORK:
        acc_handle = acquire_and_advance(acc)
        m_idx, head = scheduler_coord(work)
        sfb_mma_column = tmem_base + SFB_COL + 2 * (head & 1)
        accumulate = False
        for ktile in serial(12):
            ab_handle = wait_and_advance(ab)
            if elect_one():
                scale_s2t(tmem_base + SFA_COL, sf_desc_sfa(ab_handle.stage))
            if elect_one():
                scale_s2t(tmem_base + SFB_COL, sf_desc_sfb(ab_handle.stage, 0))
            if elect_one():
                scale_s2t(tmem_base + SFB_COL + 4, sf_desc_sfb(ab_handle.stage, 1))
            for kphase in unroll(4):
                sfa_phase_column = tmem_base + SFA_COL + kphase * 0x40000000
                sfb_phase_column = sfb_mma_column + kphase * 0x40000000
                phase_instruction = mma_instruction_desc(
                    sfa_phase_column, sfb_phase_column
                )
                if elect_one():
                    blockscaled_mma(
                        acc_column=tmem_base + acc_handle.stage * 192,
                        a_desc=a_desc(ab_handle.stage, kphase),
                        b_desc=b_desc(ab_handle.stage, kphase),
                        instruction_desc=phase_instruction,
                        sfa_column=sfa_phase_column,
                        sfb_column=sfb_phase_column,
                        accumulate=accumulate,
                    )
                accumulate = True
            if elect_one():
                umma_commit(ab_handle.empty_barrier)
        if elect_one():
            umma_commit(acc_handle.full_barrier)
        work = advance_work(work)
    acc.advance()
    wait_mbarrier(barrier(ACC_EMPTY, acc.stage), acc.phase)

# protocol_order: ACC `acquire_and_advance` and AB `wait_and_advance` both
#   advance their live cursors immediately after cloning/waiting; S2T, MMA,
#   and commits use only the cloned handle. Source 356-376.
# instruction_selection: MMA entry uses `bar.sync 2,416` and `ld.shared.b32`;
#   ACC-empty and AB-full acquire plus the one-stage producer tail use
#   `mbarrier.try_wait.parity.shared.b64`. Extent: 1+12 waits/work and one tail
#   wait. Source 322-379; PTX-128 433,499,555 and PTX-2048 450,516,572.
# instruction_selection: three elected
#   `tcgen05.cp.cta_group::1.32x128b.warpx4` sites copy one SFA and two SFB
#   atoms per K tile to TMEM columns 384,400,404. Source 345-371; PTX-128
#   579,590,598; PTX-2048 596,607,615.
# instruction_selection: four elected
#   `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32` sites execute
#   per K tile. Every phase recomputes descriptor selector bits from its shifted
#   SFA/SFB TMEM addresses. Only the first phase of the first K tile has
#   accumulate false; all other 47 dynamic MMAs accumulate. Source 361-375;
#   PTX-128 620,633,
#   646,659; PTX-2048 637,650,663,676.
# instruction_selection: elected
#   `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`
#   releases AB-empty after every K tile and publishes ACC-full after all 12.
#   Source 375-376; PTX-128 666,678; PTX-2048 683,695.

# S6 =======================================================================
# Epilogue entry and warps 0-3 FP32 TMEM -> BF16 sACC drain
# source 381-427; PTX-128 744-1237, PTX-2048 770-1259
# ==========================================================================
with role("epilogue"):
    if warp == 0:
        tmem_alloc_cta_group_one(shared_u32=TMEM_PTR, columns=512)
    named_barrier(id=2, threads=416)
    tmem_base = load_shared_u32(TMEM_PTR)
    acc = pipeline_state(stages=2, phase=0)
    work = cta_id_z()
    while work < TOTAL_WORK:
        if warp < 4:
            acc_handle = wait_and_advance(acc)
            row = warp * 32 + lane
            if not T2R_X8:
                for group in unroll(6):
                    words_f32[0:32] = tmem_load_x32(
                        tmem_base + (warp << 21) + acc_handle.stage * 192 + group * 32
                    )
                    for pair in unroll(16):
                        packed_bf16[pair] = convert_bf16x2(
                            words_f32[2 * pair + 1], words_f32[2 * pair]
                        )
                    for vec in unroll(8):
                        store_shared_v2_u32(
                            arena_ptr(sacc_byte(row, group * 32 + vec * 4)),
                            packed_bf16[2 * vec], packed_bf16[2 * vec + 1],
                        )
            else:
                for group in unroll(6):
                    for chunk in unroll(4):
                        words_f32[0:8] = tmem_load_x8(
                            tmem_base + (warp << 21) + acc_handle.stage * 192
                            + group * 32 + chunk * 8
                        )
                        for pair in unroll(4):
                            packed_bf16[pair] = convert_bf16x2(
                                words_f32[2 * pair + 1], words_f32[2 * pair]
                            )
                        for vec in unroll(2):
                            store_shared_v2_u32(
                                arena_ptr(sacc_byte(
                                    row, group * 32 + chunk * 8 + vec * 4
                                )),
                                packed_bf16[2 * vec], packed_bf16[2 * vec + 1],
                            )
            if elect_one():
                arrive_mbarrier(acc_handle.empty_barrier, count=1)
        named_barrier(id=1, threads=384)

# instruction_selection: warp 0 issues
#   `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` with immediate
#   512; all 13 participating warps execute `bar.sync 2,416` and retrieve the
#   shared TMEM pointer with `ld.shared.b32`. Source 251-254,383-385; PTX-128
#   744,749,751; PTX-2048 770,775,777.
# instruction_selection: tokens 128/256 emit six static
#   `tcgen05.ld.sync.aligned.32x32b.x32.b32`; tokens 2048/4096 emit 24 static
#   `...x8.b32`. Each specialization has 96 static
#   `cvt.rn.bf16x2.f32` sites and 48 `st.shared.v2.b32` sites; dynamically each
#   32-column group performs 16 conversions and eight vector stores. Extent:
#   four T2R warps x six groups per work tile. Source 388-423; PTX-128
#   849-1223, PTX-2048 835-1245.
# protocol_order: `wait_and_advance` clones the ACC-full handle and advances
#   the T2R cursor before all loads; its cloned empty barrier receives release.
# instruction_selection: each T2R warp has one elected
#   `mbarrier.arrive.shared.b64`, totaling four ACC-empty arrivals, followed by
#   the first `bar.sync 1,384`. Source 424-427; PTX-128 1232,1237,
#   PTX-2048 1254,1259.

# S7 =======================================================================
# Twelve epilogue cells: BF16 load, branch-uniform YARN RoPE, column amax
# source 429-499; PTX-128 1243-4296, PTX-2048 1265-4355
# ==========================================================================
        m_idx, head = scheduler_coord(work)
        token_base = m_idx * 128
        cell = warp
        cb = cell // 3                 # token block 0..3
        fc = cell % 3                  # feature cell 0..2
        tok0 = cb * 32
        f0 = fc * 64 + 2 * lane
        f1 = f0 + 1
        scale_block = fc * 2 + lane // 16
        col_amax0 = 0.0f
        col_amax1 = 0.0f

        if fc == 2:
            lib = lane & 15
            side = lane >> 4
            pcol0 = 128 + 4 * lib
            pcol1 = pcol0 + 2
            trig_col = 2 * lib + 32 * side
            if side == 1:             # uniform half-warp branch
                for r in unroll(32):
                    p0,q0,p1,q1 = load_and_convert_shared_v4_bf16(
                        arena_ptr(sacc_byte(tok0 + r, pcol0))
                    )
                    c0,c1 = load_and_convert_global_v2_bf16(
                        cos_bf16_ptr, token_base + tok0 + r, trig_col
                    )
                    s0,s1 = load_and_convert_global_v2_bf16(
                        sin_bf16_ptr, token_base + tok0 + r, trig_col
                    )
                    ps0,ps1 = packed_mul_f32x2((p0,p1), (s0,s1))
                    v0,v1 = packed_fma_f32x2((q0,q1), (c0,c1), (ps0,ps1))
                    buf0[r],buf1[r] = v0,v1
                    col_amax0 = max(col_amax0, abs(v0))
                    col_amax1 = max(col_amax1, abs(v1))
            else:
                for r in unroll(32):
                    p0,q0,p1,q1 = load_and_convert_shared_v4_bf16(
                        arena_ptr(sacc_byte(tok0 + r, pcol0))
                    )
                    c0,c1 = load_and_convert_global_v2_bf16(
                        cos_bf16_ptr, token_base + tok0 + r, trig_col
                    )
                    s0,s1 = load_and_convert_global_v2_bf16(
                        sin_bf16_ptr, token_base + tok0 + r, trig_col
                    )
                    pc0,pc1 = packed_mul_f32x2((p0,p1), (c0,c1))
                    qs0,qs1 = packed_mul_f32x2((q0,q1), (s0,s1))
                    v0,v1 = packed_fma_f32x2(
                        (qs0,qs1), (-1.0,-1.0), (pc0,pc1)
                    )
                    buf0[r],buf1[r] = v0,v1
                    col_amax0 = max(col_amax0, abs(v0))
                    col_amax1 = max(col_amax1, abs(v1))
        else:
            for r in unroll(32):
                v0,v1 = load_and_convert_shared_v2_bf16(
                    arena_ptr(sacc_byte(tok0 + r, f0))
                )
                buf0[r],buf1[r] = v0,v1
                col_amax0 = max(col_amax0, abs(v0))
                col_amax1 = max(col_amax1, abs(v1))

# instruction_selection: the two half-warp RoPE branches contribute 64 static
#   `ld.shared.v4.b16` sites; each dynamic RoPE cell executes 32. The compiler
#   scalarizes the two cos and two sin values into 256 static `ld.global.b16`
#   sites (128 dynamic loads per RoPE cell), not vector global loads. The plain
#   path has 32 static/dynamic `ld.shared.v2.b16` sites. Across both static
#   RoPE branches and the plain path there are 576 `cvt.f32.bf16` sites. Each
#   RoPE row then uses packed `mul.rn.f32x2` plus
#   `fma.rn.f32x2`. Lanes 16-31 compute `p*s+q*c`; lanes 0-15 compute
#   `p*c-q*s` with two packed multiplies and one packed FMA. Extent: 32 rows
#   only for fc=2. Non-RoPE cells load two BF16 values per row. All paths use
#   `abs.f32`/`max.f32` for both columns. Source 441-499; PTX-128 1250-4296,
#   PTX-2048 1272-4355. The static export contains 160 packed multiplies and
#   64 packed FMAs.

# S8 =======================================================================
# Column E8M0 scale and two byte stores
# source 82-129, 500-503; PTX-128 4300-4346, PTX-2048 4359-4410
# ==========================================================================
        packed_col_scale = cvt_rp_satfinite_ue8m0x2_f32(
            col_amax0 / 448.0, col_amax1 / 448.0
        )
        scol0 = (packed_col_scale >> 8) & 0xff
        scol1 = packed_col_scale & 0xff
        inv_col0 = reinterpret_f32((254 - scol0) << 23)
        inv_col1 = reinterpret_f32((254 - scol1) << 23)
        scol_row = m_idx * 4 + cb
        global_store_u8(scol_u8_ptr,
            (scol_row * NUM_HEADS + head) * 192 + f0, scol0)
        global_store_u8(scol_u8_ptr,
            (scol_row * NUM_HEADS + head) * 192 + f1, scol1)
# instruction_selection: one `cvt.rp.satfinite.ue8m0x2.f32`, exact inverse
#   bit construction using `sub.s32`/`shl.b32 23`/`mov.b32`, and two
#   `st.global.b8`. Extent: one conversion and two column-scale stores per
#   epilogue warp/work tile. Source 82-129,500-503; PTX ranges above.

# S9 =======================================================================
# Two-pass row scale, dual E4M3x2 packs, four outputs
# source 82-110,132-143,504-546; PTX-128 4347-7267, PTX-2048 4411-7503
# ==========================================================================
        leader = (lane & 15) == 0
        # Pass one freezes all 32 row scales before any output-code pass.
        for r in unroll(32):
            v0,v1 = buf0[r],buf1[r]
            row_amax = max(abs(v0), abs(v1))
            for delta in (8,4,2,1):
                row_amax = max(row_amax, shfl_xor(row_amax, delta))
            packed_row_scale = cvt_rp_satfinite_ue8m0x2_f32(
                0.0, row_amax / 448.0
            )
            srow_buf[r] = packed_row_scale & 0xff
            inv_row_buf[r] = reinterpret_f32((254 - srow_buf[r]) << 23)

        # Pass two writes the leader-owned row scale and both quantizations.
        for r in unroll(32):
            token = token_base + tok0 + r
            v0,v1 = buf0[r],buf1[r]
            if leader:
                global_store_u8(srow_u8_ptr,
                    (token * NUM_HEADS + head) * 6 + scale_block,
                    srow_buf[r])
            vr0,vr1 = packed_mul_f32x2(
                (v0,v1), (inv_row_buf[r],inv_row_buf[r]))
            vc0,vc1 = packed_mul_f32x2(
                (v0,v1), (inv_col0,inv_col1))
            row_pair = cvt_rn_satfinite_e4m3x2_f32(vr1,vr0)
            col_pair = cvt_rn_satfinite_e4m3x2_f32(vc1,vc0)
            output_byte = (token * NUM_HEADS + head) * 192 + f0
            global_store_u16(qrow_u8_ptr, output_byte, row_pair)
            global_store_u16(qcol_u8_ptr, output_byte, col_pair)

# instruction_selection: pass one uses `abs.f32`/`max.f32`, four
#   `shfl.sync.bfly.b32` deltas 8/4/2/1, one E8M0 conversion, and exact inverse
#   bits for each of 32 rows. Pass two uses leader-predicated `st.global.b8`,
#   two packed `mul.rn.f32x2`, two
#   `cvt.rn.satfinite.e4m3x2.f32`, and two `st.global.b16` per row. Extent: 32
#   rows per epilogue warp/work tile; static totals are 128 shuffles, 64 E4M3
#   conversions, 34 byte stores, and 64 u16 stores. Source 510-546;
#   complete phase PTX-128 4347-7267; PTX-2048 4411-7503.

# S10 ======================================================================
# sACC protection, persistent advance, and TMEM release
# source 548-554; PTX-128 7269-7303, PTX-2048 7505-7554
# ==========================================================================
        named_barrier(id=1, threads=384)
        work = advance_work(work)

    if warp == 0:
        tmem_relinquish_alloc_permit_cta_group_one()
    if warp == 0:
        tmem_dealloc_cta_group_one(tmem_base, columns=512)

# instruction_selection: the second `bar.sync 1,384` protects sACC reuse and
#   precedes every role-local work advance. Warp 0 then issues
#   `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned` followed by
#   `tcgen05.dealloc.cta_group::1.sync.aligned.b32` immediate 512. Extent: two
#   named-barrier-1 rendezvous per work tile and one teardown per CTA. Source
#   548-554; PTX-128 7269,7300,7303; PTX-2048 7505,7551,7554.
```

## Resource and storage summary

| resource | frozen value | evidence |
| --- | ---: | --- |
| threads / warps | 448 / 14 | source 51-53,218-220,646; PTX `.reqntid` |
| cluster / CTA group | `(1,1,1)` / one | source 596-600,644-648 |
| AB / ACC stages | 3 / 2 | source 45-46,275-294 |
| dynamic SMEM | 177792 bytes | both exports; exact bases 128/49280/123008/124544/127616 |
| TMEM | 512 columns | source 47-50,325-366; PTX teardown immediate 512 |
| named barrier 1 | 384 epilogue threads | PTX-128 1237,7269; PTX-2048 1259,7505 |
| named barrier 2 | 416 MMA+epilogue threads | PTX-128 433,749; PTX-2048 450,775 |
| TMA transaction | 40960 bytes | A 16384 + B 24576; source 271-284 |
| T2R | four warps x six 32-column groups | source 405-423; x32 or x8 emission branch |
| cell ownership | 12 x `(32 tokens,64 features)` | source 429-440 |

No protocol interval overlaps A/B/SFA/SFB/sACC. The 196-BF16 sACC stride
includes four padding values per row; that padding is not reusable storage.

## Static module contract

- The module admits only the direct E4M3/E8M0 input ABI with `K=1536`, 128
  heads, head geometry `128+64`, and tokens 128/256/2048/4096. The wrapper
  weight-transpose ABI is explicitly absent.
- Correctness covers all four token counts. Required performance rows are
  2048 and 4096, both measured against the pinned source with bench-suite.
- `prepare_data` produces deterministic signed BF16 values, then applies the
  upstream rowwise MXFP8 quantizer. TIRx and source share read-only inputs but
  own independent four-output buffers. Row and column dequantized match
  fractions must each be at least 0.95 under
  `abs(diff) <= 0.1 + 0.1*abs(source)`.
- The public API is `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`, `get_kernel`,
  `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, and `run_bench`.
  Reference import, support check, compile, and correctness are outside timing;
  each timed closure launches the target kernel exactly once.

## Instruction-selection summary

| operation | required generated instruction |
| --- | --- |
| map warmup | `prefetch.tensormap` |
| scale relay | `ld.global.b32`, `st.shared.v4.b32`, `bar.warp.sync`, async-shared fence |
| AB publication | `mbarrier.arrive.expect_tx.shared.b64` with 40960 bytes |
| A/B G2S | CTA-scope 2D TensorMap `cp.async.bulk...L2::cache_hint` |
| scale S2T | `tcgen05.cp.cta_group::1.32x128b.warpx4` |
| GEMM | `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32` |
| pipeline commit | `tcgen05.commit...mbarrier::arrive::one.shared::cluster.b64` |
| FP32 TMEM drain | x32 below 2048 tokens, x8 at and above 2048 |
| BF16 staging | `cvt.rn.bf16x2.f32` plus `st.shared.v2.b32` |
| RoPE | packed `.f32x2` multiply and FMA |
| row reduction | `shfl.sync.bfly.b32` deltas 8/4/2/1 |
| E8M0 / E4M3 | `cvt.rp.satfinite.ue8m0x2.f32` / `cvt.rn.satfinite.e4m3x2.f32` |
| output stores | `st.global.b8` scales and `st.global.b16` adjacent code pairs |
| TMEM lifecycle | CTA-group-one alloc/relinquish/dealloc, 512 columns |

Any implementation requiring a multidimensional shared buffer, first-class
layout, tile primitive, embedded CUDA, `K.cuda.func_call`, change under
`tirx_kernels/kern/`, or low-level-IR exemption violates this frozen design.

## Bidirectional source / sketch / PTX map

The physical sketch ranges below are stable because this table is appended
after the complete sketch. Reading left-to-right proves every source phase has
an explicit scalar sketch phase and an emitted anchor; reading right-to-left
proves every critical emitted family has a source and sketch owner.

| phase | source lines | sketch lines | PTX-128 | PTX-2048 |
| --- | --- | --- | --- | --- |
| direct ABI and E8M0 scale reinterpret | `api.py:272-419`; 186-212,652-657 | 100-117,676-691 | parameter list 14-42 | parameter list 14-42 |
| fixed geometry, ABI, host launch | 18-71,557-649 | 86-142 | 14-42 | 14-42 |
| arena, descriptors, TMEM | 43-54,74-80,222-228,325-366 | 143-218 | 93-119,579-678 | 93-119,596-695 |
| roles, initialization, scheduler | 213-220,230-300 | 219-272 | 89-239 | 89-241 |
| scale relay and A/B TMA | 147-183,301-318 | 273-351 | 249-412 | 251-429 |
| scale S2T and block-scaled MMA | 320-379 | 352-422 | 433-725 | 450-751 |
| TMEM allocation and T2R drain | 381-427 | 423-492 | 744-1237 | 770-1259 |
| RoPE/plain staging and column amax | 429-499 | 493-572 | 1243-4296 | 1265-4355 |
| column E8M0 quantization | 82-129,500-503 | 573-593 | 4300-4346 | 4359-4410 |
| two-pass row quantization and E4M3 helpers | 82-110,132-143,504-546 | 594-637 | 4347-7267 | 4411-7503 |
| second rendezvous and teardown | 548-554 | 638-657 | 7269-7303 | 7505-7554 |
