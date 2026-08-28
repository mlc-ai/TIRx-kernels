<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 projection GEMM + YARN RoPE + MXFP8 sketch

This is the non-executable execution sketch for the single parameterized TIRx
module
[`tirx_kernels/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_bf16in.py`](../../tirx_kernels/cudnn/proj_rope_mxfp8/gemm_proj_rope_mxfp8_bf16in.py).
It freezes the source kernel's 14-warp split, persistent scheduler, A/B and
accumulator pipelines, FP32-TMEM-to-BF16-SMEM drain, YARN RoPE arithmetic, two
MXFP8 block directions, and TMEM teardown. After the independent reviewer
returns PASS this file is immutable.

The authoritative source is
`python/cudnn/gemm/cutedsl/dense/proj_rope_mxfp8/gemm_proj_rope_mxfp8_bf16in.py`
at commit `aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5`. Source citations below
refer to that file. The writer's two cache-disabled line-info exports are under
`.porting/gemm_proj_rope_mxfp8_bf16in/writer_export/`:

- `w_out_in_true`: SHA256
  `3296c8ec051146224f1b867cef1ecba286d302efab057fa4ce663f99de72fc39`;
  `PTX-T n` means line `n` of that 6090-line artifact.
- `w_out_in_false`: SHA256
  `468cf0c56ff18e305f15ef493ea68a1a2183f8aace27bd101c84ef6eee40e7c2`;
  `PTX-F n` means line `n` of that 6205-line artifact.

Both exports compile and execute the production anchor `tokens=2048`,
`K=1536`, `num_heads=128` and compare all four outputs exactly with the source
oracle. The only compile-time branch is the physical weight orientation:

| `w_out_in` | input weight | logical B major | B G2S per stage | raw B descriptor base |
| --- | --- | --- | --- | --- |
| true | `[24576,1536]` | K-major | one 24576-byte TMA | `0x4000404000010000` |
| false | `[1536,24576]` | N-major strided view | three 8192-byte TMAs | `0x4000404002000000` |

## No first-class layout invariant

The executable device body uses only `import tirx_kernels.kern as K`. It may
not construct, pass, return, or store a first-class layout value, use a tile
primitive, call `K.cuda.func_call`, embed CUDA source, or rely on a low-level-IR
exemption. All shared storage is one rank-1 dynamic `u8` arena. TensorMap
fields, shared swizzles, descriptor fields, scheduler coordinates, TMEM
addresses, and fragment placement are explicit scalar integer arithmetic.
“Tile” below is an algorithmic rectangle only.

## Pipeline at a glance

| role | warps | persistent work and publication edges |
| --- | --- | --- |
| epilogue | 0-11 | warp 0 allocates 512 TMEM columns; all 12 retrieve the pointer; warps 0-3 wait on ACC-full and drain one 32-row band each from TMEM to BF16 `sACC`; named barrier 1 publishes staging; every warp owns one 32-token by 64-feature cell, applies the optional trailing RoPE cell and emits row/column MXFP8; second named barrier protects `sACC`; one elected lane in warps 0-3 releases ACC-empty; warp 0 relinquishes and deallocates TMEM |
| MMA | 12 | wait for TMEM allocation through named barrier 2; consume four-stage AB; issue four BF16 UMMA K phases per 64-wide K block; commit AB-empty after each K block and ACC-full after all 24 K blocks |
| TMA | 13 | prefetch A/B maps; walk the same work list; acquire AB-empty, arm 40960-byte completion, issue one A and orientation-specific B G2S copies; tail-wait all four empty stages |

The AB full/empty pipeline has four stages and one arrival on each side. The
ACC full/empty pipeline has two stages: full arrival count one from MMA, empty
arrival count four from the four T2R warps. Source `200-265,267-284,286-433`;
PTX-T `120-169,244-647,782-1126,1164-6084`.

## Primitive vocabulary

Every helper below consumes and produces only scalar values, raw pointers, or
opaque TensorMaps:

```python
encode_tensormap(base, dtype, rank, extents, byte_strides, box,
                 element_strides, swizzle, interleave, oob_fill)
arena_ptr(byte_offset)
sw128_byte_offset(row, column, logical_row_bytes)
raw_umma_descriptor(base_bits, shared_byte_address)
pipeline_state(stage_count, initial_phase)
wait_mbarrier(barrier_byte, phase)
arrive_expect_tx(barrier_byte, byte_count)
tma_load_2d(map, coord0, coord1, arena_byte, barrier_byte, multicast_mask)
umma_f16(acc_tmem_column, a_desc, b_desc, instruction_desc, accumulate)
umma_commit(barrier_byte)
tmem_load_32x32b_x32(register_words, tmem_address)
named_barrier(barrier_id, thread_count)
pack_bf16x2(hi_f32, lo_f32)
pack_e8m0x2(hi_f32, lo_f32)
pack_e4m3x2(hi_f32, lo_f32)
global_store_u8/u16(...)
```

## Complete sketch

```python
# ==========================================================================
# 1. Fixed specialization, ABI, TensorMaps, and launch
# source 16-59, 436-508; PTX-T 14-42
# ==========================================================================
TILE_M = 128
TILE_N = 192
K_TILE = 64
INSTRUCTION_K = 16
K_DIM = 1536
NUM_HEADS = 128
HEAD_DIM = 192
QK_NOPE = 128
QK_ROPE = 64
BLOCK = 32
AB_STAGES = 4
ACC_STAGES = 2
WARPS = 14
EPI_WARPS = range(0, 12)
T2R_WARPS = range(0, 4)
MMA_WARP = 12
TMA_WARP = 13
TMEM_COLUMNS = 512
assert tokens > 0 and tokens % TILE_M == 0
assert w_out_in in (False, True)

# Runtime device ABI. x/w remain live because the host-created TensorMaps point
# to them. Output pointers are raw bytes: E4M3 data uses u8 storage and E8M0
# scale is a byte code.
ABI = (
    x_bf16_ptr, w_bf16_ptr, cos_bf16_ptr, sin_bf16_ptr,
    qrow_u8_ptr, srow_u8_ptr, qcol_u8_ptr, scol_u8_ptr,
)

# x is physically [tokens,K]. The TMA map exposes [K,tokens], with K the
# contiguous dimension and a [64,128] box.
map_A = encode_tensormap(
    base=x_bf16_ptr, dtype=bf16, rank=2,
    extents=(K_DIM, tokens), byte_strides=(K_DIM * 2,),
    box=(64, 128), element_strides=(1, 1), swizzle=SW128,
    interleave=NONE, oob_fill=NONE,
)
if w_out_in:
    # Packed [out,K] is logical B[K,N] for TMA.
    map_B = encode_tensormap(
        base=w_bf16_ptr, dtype=bf16, rank=2,
        extents=(K_DIM, NUM_HEADS * HEAD_DIM),
        byte_strides=(K_DIM * 2,), box=(64, 192),
        element_strides=(1, 1), swizzle=SW128,
        interleave=NONE, oob_fill=NONE,
    )
else:
    # Packed [K,out] is logical B[N,K]; three 64-column boxes fill 192 N.
    map_B = encode_tensormap(
        base=w_bf16_ptr, dtype=bf16, rank=2,
        extents=(NUM_HEADS * HEAD_DIM, K_DIM),
        byte_strides=(NUM_HEADS * HEAD_DIM * 2,), box=(64, 64),
        element_strides=(1, 1), swizzle=SW128,
        interleave=NONE, oob_fill=NONE,
    )

M_TILES = tokens // 128
TOTAL_TILES = M_TILES * NUM_HEADS
NUM_CLUSTERS = min(TOTAL_TILES, 148)
launch(
    grid=(1, 1, NUM_CLUSTERS), block_threads=448,
    cluster=(1, 1, 1), dynamic_smem_bytes=214144, arch="sm_100a",
)
# instruction_selection: `.reqntid 448,1,1`, `.extern .shared .align 1024`,
#   and `%ctaid.z` persistent id; extent: every specialization. Source
#   `248,480-508`; PTX-T `14,29,170-177`.

# ==========================================================================
# 2. One rank-1 u8 arena and explicit address spaces
# source 62-68,171-175,197-204; PTX-T 93-110,120-169,1186-1192
# ==========================================================================
smem = rank1_dynamic_u8(214144, alignment=1024)

AB_FULL = 0                 # four u64 mbarriers: [0,32)
AB_EMPTY = 32               # four u64 mbarriers: [32,64)
ACC_FULL = 64               # two u64 mbarriers: [64,80)
ACC_EMPTY = 80              # two u64 mbarriers: [80,96)
TMEM_DEALLOC = 96           # one u64, unused for CTA-group one
TMEM_PTR = 104              # one u32, header ends at 108
A_BASE = 128                # four 16384-byte stages, ends 65664
B_BASE = 65664              # four 24576-byte stages, ends 163968
SACC_BASE = 163968          # 128 rows * 196 BF16, ends 214144

def barrier(base, stage):
    return arena_ptr(base + 8 * stage)

def sw128_element(row, col, row_elements):
    # BF16 SW128 S<3,4,3>: XOR the 16-byte segment selector by low row bits.
    linear = row * row_elements + col
    byte = linear * 2
    return byte ^ ((byte >> 3) & 112)

def a_stage_byte(stage):
    return A_BASE + stage * 16384

def b_stage_byte(stage):
    return B_BASE + stage * 24576

def sacc_byte(row, feature):
    assert 0 <= row < 128 and 0 <= feature < 192
    return SACC_BASE + 2 * (row * 196 + feature)

def descriptor_with_address(base_bits, byte_address):
    return uint64(base_bits) | uint64((byte_address >> 4) & 0x3fff)

A_DESC_BASE = 0x4000404000010000
B_DESC_BASE = 0x4000404000010000 if w_out_in else 0x4000404002000000
A_DESC = descriptor_with_address(A_DESC_BASE, shared_address(a_stage_byte(0)))
B_DESC = descriptor_with_address(B_DESC_BASE, shared_address(b_stage_byte(0)))

def a_desc(stage, kphase):
    # Descriptor increments are in 16-byte units. BF16 K-major advances two
    # units for each 16-wide instruction phase.
    return A_DESC + stage * (16384 // 16) + 2 * kphase

def b_desc(stage, kphase):
    if w_out_in:
        return B_DESC + stage * (24576 // 16) + 2 * kphase
    # N-major B advances a 64-column source chunk per instruction phase.
    return B_DESC + stage * (24576 // 16) + 128 * kphase

def acc_tmem(stage, warp, column):
    # Warp-row selector is encoded in TMEM address bits 21+, not a layout.
    return tmem_base + (warp << 21) + stage * 192 + column

# instruction_selection: A/B UMMA operands are raw 64-bit descriptors and the
#   accumulator is a raw 32-bit TMEM address; extent: all 24 K blocks x four
#   phases. Source `194-199,270-279`; PTX-T `100-110,784-788,881-1126` and
#   PTX-F `899-903,996-1241`.

# ==========================================================================
# 3. Warp roles, barrier publication, and static persistent coordinates
# source 163-170,200-249; PTX-T 71-81,111-177
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
# instruction_selection: two `prefetch.tensormap`; extent: one per input map,
#   executed by the full active TMA warp under only the warp-13 role branch;
#   there is no `elect.sync` lane election. Source `177-180`; PTX-T/PTX-F
#   warp branch `77-80`, prefetches `83,86`.

if warp == 0 and elect_one():
    for stage in unroll(4):
        mbarrier_init(barrier(AB_FULL, stage), arrivals=1)
if warp == 0 and elect_one():
    for stage in unroll(4):
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
# instruction_selection: eight AB plus four ACC `mbarrier.init.shared.b64`,
#   two `fence.mbarrier_init.release.cluster`, and two `bar.sync 0`; exact
#   arrivals are 1/1/1/4. Each of AB-full, AB-empty, ACC-full, and ACC-empty
#   has its own warp-0 `elect.sync` predicate: four independent elections in
#   source order. Source `227-244` plus pipeline helper `260-352`; PTX-T/PTX-F
#   election/init groups `115/120-123`, `128/133-136`, `144/149-150`, and
#   `157/162-163`, with publication edges `139-140,168-169`.

def scheduler_coord(work):
    # Scalar form of source StaticPersistentTileScheduler(swizzle_size=8).
    # Within each 8-head stripe, M is rastered before the next stripe.
    head_minor = work & 7
    m_idx = (work // 8) % M_TILES
    head_major = work // (8 * M_TILES)
    head = head_major * 8 + head_minor
    return m_idx, head

def advance_work(work):
    return work + NUM_CLUSTERS

# instruction_selection: integer shifts/masks implement the swizzle; for the
#   anchor `m=(work>>3)&15`, `head=((work>>4)&120)|(work&7)`. Extent: all three
#   role-local cursors. Source `248-249,263-264,282-283,429-430`; PTX-T
#   `170-192,6043-6071`.

# ==========================================================================
# 4. Warp 13: four-stage A/B TMA producer
# source 251-265; PTX-T 244-647, PTX-F 246-762
# ==========================================================================
with role("tma"):
    state = pipeline_state(stages=4, phase=1)
    work = cta_id_z()
    while work < TOTAL_TILES:
        m_idx, head = scheduler_coord(work)
        for ktile in serial(24):
            wait_mbarrier(barrier(AB_EMPTY, state.stage), state.phase)
            if elect_one():
                arrive_expect_tx(barrier(AB_FULL, state.stage), 40960)
            if elect_one():
                tma_load_2d(
                    map_A, coord0=ktile * 64, coord1=m_idx * 128,
                    arena_byte=a_stage_byte(state.stage),
                    barrier_byte=barrier(AB_FULL, state.stage),
                    multicast_mask=1, cache_hint=0,
                )
            if w_out_in:
                if elect_one():
                    tma_load_2d(
                        map_B, coord0=ktile * 64, coord1=head * 192,
                        arena_byte=b_stage_byte(state.stage),
                        barrier_byte=barrier(AB_FULL, state.stage),
                        multicast_mask=1, cache_hint=0,
                    )
            else:
                for n_piece in unroll(3):
                    if elect_one():
                        tma_load_2d(
                            map_B,
                            coord0=head * 192 + n_piece * 64,
                            coord1=ktile * 64,
                            arena_byte=b_stage_byte(state.stage) + n_piece * 8192,
                            barrier_byte=barrier(AB_FULL, state.stage),
                            multicast_mask=1, cache_hint=0,
                        )
            state.advance()
        work = advance_work(work)
    for unused in unroll(4):
        wait_mbarrier(barrier(AB_EMPTY, state.stage), state.phase)
        state.advance()

# instruction_selection: every AB-empty acquire and producer-tail wait is
#   `mbarrier.try_wait.parity.shared.b64`; extent: 24 acquires per work tile and
#   four tail waits after the last tile. Source `260,265`; PTX-T first acquire
#   `244`, tail `692,715,738,761`; PTX-F `246` and `807,830,853,876`.
# instruction_selection: each guarded producer operation is predicated by a
#   warp `elect.sync`; the armed full stage is
#   `mbarrier.arrive.expect_tx.shared.b64`, followed by
#   `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint`.
#   Each expect or TMA instruction has its own immediately governing election.
#   Extent: 24 stages/tile; K-major B has three elections (expect/A/B), while
#   N-major B has five (expect/A/B0/B1/B2). Source `260-262`; PTX-T first
#   elected/operation pairs `254/259,269/279,282/289`; PTX-F
#   `256/261,271/281,284/291,293/299,301/307`.

# ==========================================================================
# 5. Warp 12: BF16 UMMA consumer and two-stage FP32 TMEM producer
# source 267-284; PTX-T 782-1126, PTX-F 897-1241
# ==========================================================================
with role("mma"):
    named_barrier(id=2, threads=416)       # TMEM pointer publication
    tmem_base = load_shared_u32(TMEM_PTR)
    ab = pipeline_state(stages=4, phase=0)
    acc = pipeline_state(stages=2, phase=1)
    work = cta_id_z()
    while work < TOTAL_TILES:
        wait_mbarrier(barrier(ACC_EMPTY, acc.stage), acc.phase)
        accumulate = False
        for ktile in serial(24):
            wait_mbarrier(barrier(AB_FULL, ab.stage), ab.phase)
            for kphase in unroll(4):
                if elect_one():
                    umma_f16(
                        acc_tmem(acc.stage, warp=0, column=0),
                        a_desc(ab.stage, kphase),
                        b_desc(ab.stage, kphase),
                        instruction=(M128, N192, K16, BF16, FP32,
                                     trans_a=False,
                                     trans_b=(not w_out_in), cta_group=1),
                        accumulate=accumulate,
                    )
                accumulate = True
            if elect_one():
                umma_commit(barrier(AB_EMPTY, ab.stage))
            ab.advance()
        if elect_one():
            umma_commit(barrier(ACC_FULL, acc.stage))
        acc.advance()
        work = advance_work(work)
    # Producer tail protects the last two accumulator buffers before exit.
    acc.advance()
    wait_mbarrier(barrier(ACC_EMPTY, acc.stage), acc.phase)

# instruction_selection: MMA entry uses `bar.sync 2,416`, then retrieves the
#   32-bit TMEM pointer with `ld.shared.b32`; extent: once per MMA warp. Source
#   `269-271`; PTX-T `782,796`, PTX-F `897,911`.
# instruction_selection: ACC-empty acquire, all 24 AB-full acquires, and the
#   one-stage ACC producer tail are `mbarrier.try_wait.parity.shared.b64`;
#   extent: 1+24 waits per work tile plus one tail wait. Source `273,276,284`;
#   PTX-T `816,851,1164`, PTX-F `931,966,1279`.
# instruction_selection: a warp `elect.sync` predicate guards each key issue;
#   four `tcgen05.mma.cta_group::1.kind::f16` execute per K stage, and the exact
#   publication/release instruction is
#   `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`.
#   Extent: 24 AB-empty commits plus one ACC-full commit per work tile. Source
#   `277-281`; PTX-T elected/MMA/commit sites `870-927,1122,1126`, PTX-F
#   `985-1042,1237,1241`.

# ==========================================================================
# 6. Warps 0-3: TMEM allocation and FP32-to-BF16 sACC drain
# source 286-332; PTX-T 1186-1671, PTX-F 1301-1786
# ==========================================================================
with role("epilogue"):
    if warp == 0:
        tmem_alloc_cta_group_one(shared_u32=TMEM_PTR, columns=512)
    named_barrier(id=2, threads=416)
    tmem_base = load_shared_u32(TMEM_PTR)
    acc = pipeline_state(stages=2, phase=0)
    work = cta_id_z()
    while work < TOTAL_TILES:
        if warp < 4:
            wait_mbarrier(barrier(ACC_FULL, acc.stage), acc.phase)
            row = warp * 32 + lane
            for group in unroll(6):
                words_f32[0:32] = tmem_load_32x32b_x32(
                    acc_tmem(acc.stage, warp, group * 32)
                )
                for pair in unroll(16):
                    packed_bf16[pair] = pack_bf16x2(
                        words_f32[2 * pair + 1], words_f32[2 * pair]
                    )
                for vec in unroll(8):
                    store_shared_v2_u32(
                        arena_ptr(sacc_byte(row, group * 32 + vec * 4)),
                        packed_bf16[2 * vec], packed_bf16[2 * vec + 1],
                    )
            if elect_one():
                arrive_mbarrier(barrier(ACC_EMPTY, acc.stage), count=1)
        named_barrier(id=1, threads=384)

# instruction_selection: warp 0 allocates with exact
#   `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` immediate 512;
#   all 13 participating warps execute `bar.sync 2,416` then retrieve the TMEM
#   pointer using `ld.shared.b32`. Extent: once per CTA. Source `288-290`;
#   PTX-T `1186,1191,1193`, PTX-F `1301,1306,1308`.
# instruction_selection: every T2R warp/work tile waits with
#   `mbarrier.try_wait.parity.shared.b64`, then issues six
#   `tcgen05.ld.sync.aligned.32x32b.x32.b32`; every load emits 16
#   `cvt.rn.bf16x2.f32` and eight `st.shared.v2.b32`. Extent: four T2R warps,
#   six loads and 48 vector stores per warp/work tile. Source `311-327`;
#   PTX-T `1272,1289-1657`, PTX-F `1387,1404-1772`.
# instruction_selection: each T2R warp's `elect.sync`-selected lane releases
#   ACC-empty with one `mbarrier.arrive.shared.b64`, so there are four dynamic
#   arrivals per tile; all 12 epilogue warps then execute the first
#   `bar.sync 1,384`. Source `329-332`; PTX-T `1659,1666,1671`, PTX-F
#   `1774,1781,1786`. The explicit row address is
#   `SACC_BASE + 2*((warp*32+lane)*196 + group*32)`.

# ==========================================================================
# 7. All epilogue warps: one 32-token x 64-feature cell
# source 334-390; PTX-T 1677-3589
# ==========================================================================
        m_idx, head = scheduler_coord(work)
        token_base = m_idx * 128
        cell = warp
        cb = cell // 3                   # token block 0..3
        fc = cell % 3                    # feature cell 0..2
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
            left_weight = float32(1 - side)
            right_weight = float32(side)
            for r in unroll(32):
                row = tok0 + r
                p0_b16, q0_b16, p1_b16, q1_b16 = load_shared_v4_b16(
                    arena_ptr(sacc_byte(row, pcol0))
                )
                p0, q0, p1, q1 = convert_each_bf16_to_f32(
                    p0_b16, q0_b16, p1_b16, q1_b16
                )
                cos_b16_0, cos_b16_1 = load_global_v2_b16(
                    cos_bf16_ptr, 2 * ((token_base + row) * 64 + trig_col)
                )
                sin_b16_0, sin_b16_1 = load_global_v2_b16(
                    sin_bf16_ptr, 2 * ((token_base + row) * 64 + trig_col)
                )
                c0, c1 = convert_each_bf16_to_f32(cos_b16_0, cos_b16_1)
                s0, s1 = convert_each_bf16_to_f32(sin_b16_0, sin_b16_1)
                pc0, pc1 = packed_mul_f32x2((p0, p1), (c0, c1))
                qs0, qs1 = packed_mul_f32x2((q0, q1), (s0, s1))
                left0, left1 = packed_fma_f32x2(
                    (qs0, qs1), (-1.0, -1.0), (pc0, pc1)
                )
                ps0, ps1 = packed_mul_f32x2((p0, p1), (s0, s1))
                qc0, qc1 = packed_mul_f32x2((q0, q1), (c0, c1))
                right0, right1 = packed_add_f32x2((ps0, ps1), (qc0, qc1))
                low0, low1 = packed_mul_f32x2(
                    (left0, left1), (left_weight, left_weight)
                )
                v0, v1 = packed_fma_f32x2(
                    (right0, right1), (right_weight, right_weight), (low0, low1)
                )
                buf0[r], buf1[r] = v0, v1
                col_amax0 = max(col_amax0, abs(v0))
                col_amax1 = max(col_amax1, abs(v1))
        else:
            for r in unroll(32):
                v0_b16, v1_b16 = load_shared_v2_b16(
                    arena_ptr(sacc_byte(tok0 + r, f0))
                )
                v0, v1 = convert_each_bf16_to_f32(v0_b16, v1_b16)
                buf0[r], buf1[r] = v0, v1
                col_amax0 = max(col_amax0, abs(v0))
                col_amax1 = max(col_amax1, abs(v1))

# instruction_selection: each of 32 RoPE rows loads four contiguous staged
#   values with `ld.shared.v4.b16` and two trig pairs with two
#   `ld.global.v2.b16`, then performs eight scalar `cvt.f32.bf16`; each of 32
#   non-RoPE rows uses `ld.shared.v2.b16` and two `cvt.f32.bf16`. Source
#   `357-368,383-385`; PTX-T first RoPE row `1688-1703`, first non-RoPE load
#   `3165`; PTX-F has the same sequence at `1803-1818,3280`.
# instruction_selection: every RoPE row uses packed `mul.rn.f32x2`,
#   `fma.rn.f32x2`, and `add.rn.f32x2`; extent: 32 rows for fc=2 only. Both
#   RoPE and non-RoPE paths update each column maximum with `abs.f32` and
#   `max.f32` for both values on every one of 32 rows. Source `369-389`; PTX-T
#   first packed row `1707-1725`, RoPE abs/max begins around `2905`, non-RoPE
#   abs/max around `3325`; PTX-F is shifted by 115 lines in this region.

# ==========================================================================
# 8. E8M0 helpers and columnwise scale
# source 70-139,390-399; PTX-T 3590-3621
# ==========================================================================
        packed_col_scale = cvt_rp_satfinite_ue8m0x2_f32(
            col_amax0 / 448.0, col_amax1 / 448.0
        )
        scol0 = (packed_col_scale >> 8) & 0xff
        scol1 = packed_col_scale & 0xff
        inv_col0 = reinterpret_f32((254 - scol0) << 23)
        inv_col1 = reinterpret_f32((254 - scol1) << 23)
        scol_row = m_idx * 4 + cb
        global_store_u8(
            scol_u8_ptr,
            (scol_row * NUM_HEADS + head) * HEAD_DIM + f0,
            scol0,
        )
        global_store_u8(
            scol_u8_ptr,
            (scol_row * NUM_HEADS + head) * HEAD_DIM + f1,
            scol1,
        )
# instruction_selection: one `cvt.rp.satfinite.ue8m0x2.f32` with two dynamic
#   operands is followed independently for each byte by exact inverse-bit
#   construction `sub.s32`, `shl.b32 ...,23`, `mov.b32`; two
#   `st.global.b8` store the column scales. Extent: one conversion, two inverse
#   constructions, and two byte stores per epilogue warp/tile. Source
#   `95-125,390-393`; PTX-T `3590,3602,3606,3619,3621`; PTX-F
#   `3705,3717,3721,3734,3736`.

# ==========================================================================
# 9. Rowwise 16-lane reduction, dual E4M3x2 pack, and global writes
# source 394-425; PTX-T 3622-6040
# ==========================================================================
        leader = (lane & 15) == 0
        for r in unroll(32):
            token = token_base + tok0 + r
            v0, v1 = buf0[r], buf1[r]
            row_amax = max(abs(v0), abs(v1))
            for delta in (8, 4, 2, 1):
                row_amax = max(row_amax, shfl_xor(row_amax, delta))
            # Zero in the high conversion input makes the low byte the scale.
            packed_row_scale = cvt_rp_satfinite_ue8m0x2_f32(
                0.0, row_amax / 448.0
            )
            srow = packed_row_scale & 0xff
            inv_row = reinterpret_f32((254 - srow) << 23)
            if leader:
                global_store_u8(
                    srow_u8_ptr,
                    (token * NUM_HEADS + head) * 6 + scale_block,
                    srow,
                )
            row_pair = cvt_rn_satfinite_e4m3x2_f32(
                v1 * inv_row, v0 * inv_row
            )
            col_pair = cvt_rn_satfinite_e4m3x2_f32(
                v1 * inv_col1, v0 * inv_col0
            )
            output_byte = (token * NUM_HEADS + head) * HEAD_DIM + f0
            global_store_u16(qrow_u8_ptr, output_byte, row_pair)
            global_store_u16(qcol_u8_ptr, output_byte, col_pair)

# instruction_selection: each row starts with `abs.f32`/`max.f32`, then four
#   `shfl.sync.bfly.b32` deltas 8/4/2/1 and max updates. One
#   `cvt.rp.satfinite.ue8m0x2.f32` is followed by exact inverse-bit
#   `sub.s32`, `shl.b32 ...,23`, `mov.b32`; the two half-warp leaders execute
#   the predicated `st.global.b8`. Two packed `mul.rn.f32x2`, two
#   `cvt.rn.satfinite.e4m3x2.f32`, and two `st.global.b16` complete the row.
#   Extent: 32 statically unrolled rows per epilogue warp/tile. Source
#   `400-425`; PTX-T first-row abs/max/shuffles/inverse/store/data
#   `3627-3707`, last row `5975-6040`; PTX-F first row `3742-3822`.

# ==========================================================================
# 10. sACC reuse, ACC release, persistent advance, and TMEM teardown
# source 427-433; PTX-T 6042-6084
# ==========================================================================
        named_barrier(id=1, threads=384)
        if warp < 4:
            acc.advance()
        work = advance_work(work)

# instruction_selection: this is the second and final `bar.sync 1,384` for
#   each work tile; it protects sACC reuse before all role-local cursors
#   advance. Source `427-430`; PTX-T `6042` and PTX-F `6157` for the anchor's
#   final tile.

    if warp == 0:
        tmem_relinquish_alloc_permit_cta_group_one()
    if warp == 0:
        tmem_dealloc_cta_group_one(tmem_base, columns=512)

# instruction_selection: warp 0 issues
#   `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned` immediately
#   followed, with no intervening barrier, by
#   `tcgen05.dealloc.cta_group::1.sync.aligned.b32` with immediate 512. Source
#   `432-433`; PTX-T `6081,6084`, PTX-F `6196,6199`.
```

## Resource and storage summary

| resource | frozen value | evidence |
| --- | ---: | --- |
| threads / warps | 448 / 14 | source 39-41,167-169,505; PTX `.reqntid` |
| cluster / CTA group | `(1,1,1)` / one | source 455,461-470,506 |
| AB / ACC stages | 4 / 2 | source 37-38,227-244 |
| dynamic SMEM | 214144 bytes | source allocation order plus PTX bases 128/65664/163968 |
| TMEM | 512 columns | PTX-T 1185-1186,6083-6084 |
| named barrier 1 | 384 epilogue threads | PTX-T 1671,6042 |
| named barrier 2 | 416 MMA+epilogue threads | PTX-T 782,1191 |
| T2R shape | four warps x six 32-column fragments | source 310-327; PTX six static `ld...x32` sites |
| cell ownership | 12 x `(32 tokens,64 features)` | source 334-344 |

No protocol interval overlaps A/B/sACC. `sACC`'s allocated stride is 196 BF16
values even though only features 0..191 are read; the four-value padding is
part of the source resource shape and is not reusable storage.

## Static specialization and module contract

- The module admits only BF16 input projection with `K=1536`, 128 heads,
  head geometry `128+64`, tokens a positive multiple of 128, and either source
  weight orientation. The separate MXFP8-input source sibling is out of scope.
- Correctness configs cover tokens 128/256/2048/4096 and both orientations.
  Required performance configs cover production tokens 2048/4096 and both
  orientations. Every required line is measured by `bench_suite` against the
  pinned cuDNN Frontend source.
- `prepare_data` creates deterministic nonzero signed BF16 x/w and nontrivial
  BF16 cos/sin, with independent TIRx/source output buffers. Row and column data
  are compared as E4M3 codes and both scale arrays as exact bytes; the source's
  accepted comparison contract is preserved in the numerical result artifact.
- The public API is `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`, `get_kernel`,
  `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, and `run_bench`.
  Reference import/compile/oracle work is lazy and outside timing. One timed
  closure invocation launches exactly one target kernel.

## Instruction-selection summary

| operation | required generated instruction |
| --- | --- |
| A/B map warmup | `prefetch.tensormap` |
| AB producer publication | `mbarrier.arrive.expect_tx.shared.b64` |
| A/B G2S | `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint` |
| GEMM | `tcgen05.mma.cta_group::1.kind::f16` |
| pipeline release/publication | `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64` |
| FP32 TMEM drain | `tcgen05.ld.sync.aligned.32x32b.x32.b32` |
| BF16 staging | `cvt.rn.bf16x2.f32` plus vector-two 32-bit shared stores |
| RoPE | packed `.f32x2` multiply/FMA/add |
| row reduction | `shfl.sync.bfly.b32` deltas 8/4/2/1 |
| E8M0 | `cvt.rp.satfinite.ue8m0x2.f32` |
| E4M3 data | `cvt.rn.satfinite.e4m3x2.f32` |
| data output | `st.global.b16` for each adjacent feature pair |
| scale output | `st.global.b8` |
| TMEM lifecycle | `tcgen05.alloc`, `relinquish_alloc_permit`, `dealloc`, all CTA-group one and 512 columns |

Any implementation that needs a multidimensional shared buffer, a layout or
tile primitive, an inline CUDA call, a change below `tirx_kernels/kern/`, or a
low-level-IR func-call exemption violates this frozen design.
