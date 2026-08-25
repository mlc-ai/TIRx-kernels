<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This design sketch documents a modified TIRx port of cuDNN Frontend's
python/cudnn/linear_attention/frost/kernel/gdn2_bprop_f16.py at commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5.
-->

# cuDNN SM100 GDN-2 BF16 backward: coarse WASP pipeline sketch

This is a non-executable execution sketch, not a Python module, builder API,
mathematical reference, or alternate implementation.  The implementation it
describes is maintained in
[`tirx_kernels/cudnn/linear_attention/gdn2_bprop_f16.py`](../../../tirx_kernels/cudnn/linear_attention/gdn2_bprop_f16.py),
which is the source of truth after the correctness gate.

The source specialization is fixed to `BT=16`, `DK=DV=128`, BF16 or FP16 I/O,
FP32 accumulation, one CTA per cluster, 512 main-kernel threads, 1024 prologue
threads, and 512 TMEM columns.  Capability is the following discrete,
source-verified set rather than a Cartesian product: `basic`, `tail` (ragged
lengths including a zero-length sequence), `grouped(HQ,HK,HV)=(4,1,1)`,
`l2norm`, `safe_gate`, `beta_sigmoid`, `state` (initial state + final-state
gradient + initial-state gradient), `dynamic`, `order_scratch`,
`order_generate`, and `fp16`.  Each label means the exact profile frozen in
`.porting/gdn2_bprop_f16/export_ptx.py`; combinations not present there are not
claimed.

GDN-2 differs from its KDA sibling in the gating algebra, and that difference is
what this sketch must carry: the erase gate `Beta` is per-key-channel and is
folded into the `K_decay` operand, a new per-value write gate `W` forms
`Y = W*V - S^T K_decay`, `T_inv` and `dM_strict` carry no Beta row scale, and
the kernel emits two additional gradients, `dW_out = V*dY` and a per-channel
`dBeta`.

Writer line-info evidence is under `.porting/gdn2_bprop_f16/source_export/`,
one directory per accepted branch.  Each holds two PTX files with both `.file`
and `.loc`: the main `cutlass_host_Gdn2BwdCfg...sm_100a.ptx` and the
`cutlass_prologue_...sm_100a.ptx`.  Unless stated otherwise, PTX line
numbers below refer to the `basic` branch's main PTX.  Static counts use
instruction lines minus predicated lines.

## Pipeline at a glance

| Warps | Register target | Role-local program | Main publication/reuse edges |
| --- | ---: | --- | --- |
| 0..3 | 128 | gate log2 prefix scan, optional Q/K L2 normalization, materialize `K_decay = eG*(Beta*K)`, `K_inv = K/eG`, `K_restore = (eGl/eG)*K`, `Q_decay = eG*scale*Q` and `diag(eGl)`; push raw Q/K into the TMEM ring; restage the entering state SMEM->TMEM | raw Q/K/Gate/Beta, state, decay operands, Q/K raw TMEM |
| 4..7 | 184 | value side: `Y = W*V - state_k`, U/dU/dY restages, `dV = W*dY` and `dW_out = V*dY`, dH capture, dht seed, dS0 drain | raw V/W, U/dY/dV/dW stages, dState |
| 8..11 | 136 | drain the four dQ/dK TMEM accumulators, assemble dQ/dK, per-channel `dBeta`, `dGate` plus its in-register reverse cumsum, L2-norm backward projection | dQ/dK/dGate/dBeta output stages, state and dState reuse |
| 12 | 64 | register `mma.sync` program: `KK = K_decay@K_inv^T`, three-round Neumann `T_inv`, `dM = dY@U^T` and the `+/-strict(dM)` tiles | intermediate tiles, decay operands, dY/U |
| 13 | 64 | allocate 512 TMEM columns and issue the 15 logical `tcgen05.mma` chains, then tear TMEM down | every TMEM accumulator/input edge |
| 14 | 64 | persistent work-stealing scheduler plus Q/K/Gate/Beta/V/W/dO/checkpoint TensorMap loads | seven raw rings, state ring, scheduler publication |
| 15 | 64 | register `A = tril_incl(Q_decay@K_inv^T)` and `dA = tril_incl(dO@U^T)`, then the one-behind six-store TensorMap ladder | A/dA intermediate tiles, dO reuse, all six output stages |

Role dispatch is the source-order `if/elif` chain: 14, 12, 13, 15, 0..3, 4..7,
then 8..11.  Every role constructs its own persistent scheduler cursor; the
eight scheduler stages are never reset between work items.

The register split is exact rather than approximate.  The launch base is 128
registers per thread over 512 threads (65536 total).  Warps 0..3 stay at the
base, warps 4..7 take `+56`, warps 8..11 take `+8`, and warps 12..15 release
`-64`; at 128 threads per group that is `+7168 + 1024 = 8192` taken against
`8192` released.  Warps 12..15 are one warpgroup and therefore must share a
single value.

## Primitive vocabulary

All storage is linear.  No declaration or operation carries a layout object,
layout value, tile primitive, or first-class view.  Logical matrices below are
only index ranges selected by scalar address functions.

```python
linear_buffer(space, dtype, elements, byte_offset, alignment, lifetime)
reg_array(dtype, elements)
smem_byte(base, stage, row, col, stage_bytes, row_bytes, elem_bytes, xor_bits)
tmem_cell(base, row, col)       # base + col + (row << 16)
gmem_element(base, scalar_index)
descriptor_slot(workspace, array, batch)  # workspace + (array*B+batch)*128
```

Directional movement primitives:

```python
copy_p2g(src, dst)                        # parameter descriptor -> GMEM
copy_g2s(src, dst, predicate, completion) # GMEM -> SMEM
copy_s2g(src, dst, predicate)             # SMEM -> GMEM
copy_g2r(src, dst, predicate=True)
copy_r2g(src, dst, predicate=True)
copy_s2r(src, dst)
copy_r2s(src, dst)
copy_t2r(src, dst)
copy_r2t(src, dst)
```

Computational primitives are only:

```python
fill(dst, value)
cast(dst, src)
add(dst, lhs, rhs)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
rcp(dst, src)
exp2(dst, src)
log2(dst, src)
tanh(dst, src)
rsqrt(dst, src)
min(dst, lhs, rhs)
max(dst, lhs, rhs)
select(dst, predicate, true_value, false_value)
shuffle_xor(dst, src, delta)
shuffle_up(dst, src, delta)
gemm(dst, lhs, rhs, accumulate=False)
```

`init`, `wait`, `arrive`, `expect_bytes`, `commit`, `release`, `fence`,
`barrier`, stage/phase advancement, register-budget changes, descriptor field
replacement, directional-copy groups, and TMEM allocation are schedule
operations.  Three of them lower uniformly everywhere they appear, so they are
annotated once here rather than at each of their many sites:
`wait(bar, stage, phase)` is
`mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` inside a spin loop (82
sites); `arrive(bar, stage)` is `mbarrier.arrive.shared.b64` (50 sites); and
`expect_bytes` is `mbarrier.arrive.expect_tx.shared.b64` (8 sites, one per TMA
tensor).  Each key movement, computation, or synchronization occurrence
below is immediately followed by the selected instruction or instruction
family observed in the writer export.

## Complete sketch

```python
# ==========================================================================
# Static parameters and the two-launch runtime ABI
# ==========================================================================
BT      = 16          # chunk-inner token tile
DK      = 128         # query/key head dim
DV      = 128         # value head dim
IO      = bf16        # or fp16; FP32 accumulation throughout
BPE     = 2
LOG2_E  = 1.4426950408889634
L2_EPS  = 1e-12       # compared against the squared sum, clamped at 1e-24

# Capability flags, each a separate compiled program.
L2NORM, SAFE_GATE, BETA_SIGMOID = ...
USE_INITIAL_STATE, USE_DSTATE_IN, USE_DSTATE0 = ...
DYN_SCHED, ORDER_IN_PROLOGUE, ORDER_GENERATE = ...
GATE_SCALE_LOG2 = gate_lower_bound * LOG2_E        # -5.0 * LOG2_E by default
Q_RATIO, K_RATIO, V_RATIO = HO // HQ, HO // HK, HO // HV
FIRST_STATE_CHUNK = 0 if USE_INITIAL_STATE else 1

# Launch 1 (prologue): grid 1, 1024 threads.
# Launch 2 (main):     grid = SM count, 512 threads, one CTA per cluster,
#                      min_blocks_per_sm = 1, persistent.

# GMEM tensors.  THD packed, stride-1 innermost.
q       : gmem[IO,  total_T * HQ * DK]
k       : gmem[IO,  total_T * HK * DK]
v       : gmem[IO,  total_T * HV * DV]
gate    : gmem[f32, total_T * HO * DK]     # natural log, or safe-gate logits
beta    : gmem[IO,  total_T * HO * DK]     # per-key erase gate  (GDN-2)
w       : gmem[IO,  total_T * HO * DV]     # per-value write gate (GDN-2)
do      : gmem[IO,  total_T * HO * DV]
ckpt    : gmem[IO,  rows * HO * DV * DK]   # entering-state series, V-major
dq, dk, dv, dgate, dbeta, dw               # outputs, pre-allocated, not zeroed
dstate_in  : gmem[f32, B * HO * DV * DK]   # optional dL/d(final state)
dstate0_out: gmem[f32, B * HO * DV * DK]   # optional dL/d(initial state)
cu_seqlens : gmem[i32, B + 1]
work_items : gmem[i32, rows * 8]           # (b, h, wstart, wend, cstart, cend, tok0, tok1)
work_count : gmem[i32, 1]
sched_ctr  : gmem[i32, 2]                  # dynamic-scheduler ticket
tmaps      : gmem[u64, 14 * B * 16]        # 14 descriptor arrays, 128 B slots

# ==========================================================================
# Descriptor arrays, in the source's own index order.  Arrays 0..12 are
# rank-3; Gate and dGate are FP32 with [32,1,16] boxes, the other 16-bit
# tensors use [64,1,16].  Only array 13, the checkpoint, is rank-4 with box
# [64,DK,1,1].  All use 128-byte swizzle.
# ==========================================================================
TMAP_Q, TMAP_K, TMAP_V, TMAP_GATE     = 0, 1, 2, 3
TMAP_DO, TMAP_BETA, TMAP_W            = 4, 5, 6
TMAP_DQ, TMAP_DK, TMAP_DV             = 7, 8, 9
TMAP_DGATE, TMAP_DWO, TMAP_DBETA      = 10, 11, 12
TMAP_CKPT                             = 13

# ==========================================================================
# Linear SMEM.  One arena; every region is a byte range.  Declaration order is
# semantic: tcgen05-descriptor operands occupy the low group, `BANK_FILL` pads,
# and generic-client buffers occupy the high group.  The pad makes SQ start at
# exactly 131072 (the 128 KB midpoint); this is verified, not assumed.
# ==========================================================================
arena = linear_buffer(smem, u8, 225280, 0, 1024, whole_kernel)

BARRIERS       =      0;  BARRIERS_END       =    984   # 123 slots x 8 B
TMEM_HOLD      =    984;  TMEM_HOLD_END      =    988
SCHED          =    992;  SCHED_END          =   1024   # 8 x i32 ring, align 16
RED1           =   1024;  RED1_END           =   1280   # f32 cross-warp scratch
NORM           =   1280;  NORM_END           =   1792   # f32 q/k inverse norms
STATE          =   2048;  STATE_END          =  34816   # [DK][DV]
DSTATE         =  34816;  DSTATE_END         =  67584
K_DECAY        =  67584;  K_DECAY_END        =  75776   # 2 x [BT][DK]
K_INV          =  75776;  K_INV_END          =  83968
K_RESTORE      =  83968;  K_RESTORE_END      =  92160
Q_DECAY        =  92160;  Q_DECAY_END        = 100352
STATE_DIAG     = 100352;  STATE_DIAG_END     = 108544   # 2 x 8 x [16][16], 32B swz
INTERMEDIATE   = 108544;  INTERMEDIATE_END   = 113664   # 2 x 5 x [16][16], 32B swz
DO             = 113664;  DO_END             = 121856
BETA           = 121856;  BETA_END           = 130048
BANK_FILL      = 130048;  BANK_FILL_END      = 131072   # pad, never addressed
SQ             = 131072;  SQ_END             = 139264   # high group starts here
SK             = 139264;  SK_END             = 147456
SV             = 147456;  SV_END             = 155648
SGATE          = 155648;  SGATE_END          = 172032   # f32
SW             = 172032;  SW_END             = 180224
SU             = 180224;  SU_END             = 184320
SDY            = 184320;  SDY_END            = 188416
SDQ            = 188416;  SDQ_END            = 192512
SDK            = 192512;  SDK_END            = 196608
SDV            = 196608;  SDV_END            = 204800
SDGATE         = 204800;  SDGATE_END         = 212992   # f32
SDWO           = 212992;  SDWO_END           = 221184
SDB            = 221184;  SDB_END            = 225280
assert SQ == 131072             # the pad places the high group at 128 KB exactly
assert SDB_END == 225280        # 7168 B of headroom under the 232448 B cap

# Scalar address functions.  Each selects bytes; none is a layout.
def swizzle_xor_128b(row, col, elem_bytes):
    elems = 128 // elem_bytes
    return (col ^ ((row % 8) * (16 // elem_bytes))) % elems

def raw_io_byte(base, stage, token, channel):        # [BT][DK] or [BT][DV], 128B swizzle
    seg, within = channel // 64, channel % 64
    return base + stage * 4096 + (seg * 1024 + token * 64
                                  + swizzle_xor_128b(token, within, 2)) * 2

def raw_f32_byte(base, stage, token, channel):       # Gate and dGate
    seg, within = channel // 32, channel % 32
    return base + stage * 8192 + (seg * 512 + token * 32
                                  + swizzle_xor_128b(token, within, 4)) * 4

def swizzle_xor_32b(row, col, elem_bytes):           # 16x16 intermediate tiles
    chunk = 16 // elem_bytes
    return ((col // chunk) ^ ((row >> 2) & 1)) * chunk + col % chunk

def inter_byte(stage, slot, row, col):               # slot 0=A 1=T_inv 2=dA 3=+dM 4=-dM
    return INTERMEDIATE + (stage * 5 + slot) * 512 + (row * 16
           + swizzle_xor_32b(row, col, 2)) * 2

def state_byte(row, col):                            # [DK][DV] entering state
    seg, within = col // 64, col % 64
    return STATE + (seg * 8192 + row * 64 + swizzle_xor_128b(row, within, 2)) * 2

# tcgen05 operand descriptors are packed integers, not layouts: the 14-bit
# shared address, the leading and stride byte offsets, bit 46, and the layout
# code at bit 61.  Aliased operand views differ only in `leading_bytes`:
#   state_alt (lead DV*128) vs state_direct (lead 16); do_lead16 vs do_amaj;
#   q_decay_trans; k_decay_trans; k_inv_amaj; dy_lead16.
def smem_desc(base_byte, leading_bytes, stride_bytes, layout_code): ...

# ==========================================================================
# TMEM: one 512-column allocation, used exactly.  Two aliases are load-bearing
# and are safe only under the in-order tcgen05 issue queue plus the barrier
# chain quoted beside each.
# ==========================================================================
TM_DSTATE_ACC   =   0    # 128 cols, fp32 dH [DV rows][DK cols]
TM_DSTATE_INP   = 128    #  64 cols, dH as an f16 A-operand
TM_STATE_INP    = 192    # 128 cols, 2 stages x 64, entering state
TM_STATE_K_ACC  = 320    #  16 cols
TM_DY_ACC       = 320    #  ALIAS of STATE_K_ACC: WG1 consumes state_k before
                         #  dY = dU @ T_inv writes, and the dY readback precedes
                         #  state_k(c+1) = state @ K_decay^T
TM_U_ACC        = 336    #  16 cols
TM_DU_ACC       = 352    #  16 cols
TM_DQ_ACC       = 368    #  16 cols
TM_DK_DECAY_ACC = 384    #  16 cols
TM_DK_INV_ACC   = 400    #  16 cols
TM_DK_RESTORE   = 416    #  16 cols
TM_Y_INP        = 432    #   8 cols
TM_NEG_DY_INP   = 432    #  ALIAS of Y_INP: U = Y @ T_inv consumed Y before the
                         #  dY block runs, and y_inp(c+1) is gated by
                         #  state_k_acc_ready(c+1)
TM_DU_INP       = 440    #   8 cols
TM_QRAW_INP     = 448    #  4 stages x 8
TM_KRAW_INP     = 480    #  4 stages x 8
assert TM_KRAW_INP + 4 * 8 == 512

def tmem_cell(base, row, col): return base + col + (row << 16)
```

```python
# ==========================================================================
# Launch 1: prologue, grid 1 x 1024 threads.
# ==========================================================================
def prologue_kernel():
    if ORDER_IN_PROLOGUE:
        # Longest-processing-time ordering over the work table, plus zeroing of
        # BOTH consumers' scheduler rings.  When a scratch buffer is supplied it
        # holds the built table and `work_items` is the destination; with
        # ORDER_GENERATE the body synthesizes the table instead.
        order_body(work_items, work_item_scratch, sched_all, n_threads=1024)
        # instruction_selection: ld.global.s32 / st.global.s32 pairs per field;
        # extent: loop over the item table
        barrier(id=0, threads=1024)

    for array in range(14):
        for batch in range(B):
            copy_p2g(param_descriptor(array), descriptor_slot(tmaps, array, batch))
            # instruction_selection: st.global.b64; extent: a constexpr-unrolled
            # 16 per descriptor, 14 arrays -> the 224 st.global.b64 in the
            # prologue export.  There is no st.global.v4.b32 there.
            replace_field(GLOBAL_ADDRESS, base + cu_seqlens[batch] * row_stride)
            replace_field(GLOBAL_DIM[2], seqlen_b)
            # instruction_selection: tensormap.replace.tile.global_address
            #   .global.b1024.b64 and .global_dim.global.b1024.b32; extent: scalar
            #   per field.  The export carries exactly 14 of each, one per array.
            fence_tensormap_release()
            # instruction_selection: fence.proxy.tensormap::generic.release.gpu;
            # extent: 14 sites, matching the descriptor-array count

# Varlen needs no masking loop anywhere in the main kernel: the patched
# GLOBAL_ADDRESS/GLOBAL_DIM make tail loads zero-fill and tail stores clip in
# hardware.  The checkpoint array additionally derives its per-sequence chunk
# offset on device from a running prefix of ceil(seqlen_b / BT).

# ==========================================================================
# Launch 2: main kernel, grid = SM count, 512 threads.
# ==========================================================================
def persistent_backward():
    arena = ...            # the 225280-byte linear allocation above

    # ---- barrier construction (one warp owns each contiguous run) ----------
    for (name, stages, count, owner_warp) in PROTOCOL:      # 70 objects
        if warp_id == owner_warp and elected():
            for stage in range(stages):
                init(barrier_slot(name, stage), arrival_count=count)
                # instruction_selection: mbarrier.init.shared.b64; extent: scalar
    fence_mbarrier_init()
    # instruction_selection: fence.mbarrier_init.release.cluster
    barrier(id=0, threads=512)
    # instruction_selection: barrier.sync 0, 512

    # ---- source-order role dispatch ---------------------------------------
    if warp_id == 14:  tma_warp()          # register budget 64 (DECREASE)
    elif warp_id == 12: super_mma_warp()   # 64
    elif warp_id == 13: tcgen05_warp()     # 64
    elif warp_id == 15: epilogue_warp()    # 64
    elif warp_id < 4:   compute_group_0()  # 128 (equals the launch base)
    elif warp_id < 8:   compute_group_1()  # 184 (INCREASE)
    else:               compute_group_2()  # 136 (INCREASE)
    # instruction_selection: setmaxnreg.inc/dec.sync.aligned.u32; extent: 7 sites
    # (three compute groups plus each of warps 12..15 separately)

# ==========================================================================
# Persistent tile scheduler.  The TMA warp owns the ticket; the other fifteen
# warps each elect one lane and consume through an eight-deep SMEM ring.
# ==========================================================================
# The ENTIRE ring protocol is compile-time gated.  With DYN_SCHED off there is
# no barrier traffic, no SMEM traffic and no atomic at all -- the export for the
# `basic` branch carries zero `atom.global`, and one in the `dynamic` branch.
def sched_publish_next(tile_idx, num_ctas):          # warp 14 only
    if not DYN_SCHED:
        return tile_idx + num_ctas
    wait("sched_done", sched.stage, sched.phase)
    if elected():
        ticket = atomic_add(sched_ctr, 1)
        # instruction_selection: atom.global.add.u32; extent: scalar, one lane
        copy_r2s(num_ctas + ticket, SCHED + sched.stage * 4)
        # instruction_selection: st.shared.b32; extent: scalar
    warp_sync()
    # instruction_selection: bar.warp.sync; extent: publishes the elected lane's
    # store to the whole warp before every lane reads it back
    copy_s2r(SCHED + sched.stage * 4, next_tile)     # ALL lanes read it back
    # instruction_selection: ld.shared.b32; extent: scalar
    if elected():
        arrive("sched_ready", sched.stage)
        # instruction_selection: mbarrier.arrive.shared.b64
    sched = advance(sched, 8)
    return next_tile

def sched_next_tile(tile_idx, num_ctas):             # the other fifteen warps
    if not DYN_SCHED:
        return tile_idx + num_ctas
    wait("sched_ready", sched.stage, sched.phase)
    copy_s2r(SCHED + sched.stage * 4, next_tile)
    # instruction_selection: ld.shared.b32; extent: scalar
    if elected():
        arrive("sched_done", sched.stage)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: one lane per
        # warp -- `sched_done` carries arrival count 15, every warp but the producer
    sched = advance(sched, 8)
    return next_tile

# ==========================================================================
# Warp 14: TMA loads.  Seven per-token tensors plus the entering state.
# ==========================================================================
def tma_warp():
    raw    = PipelineState.start(phase=1)      # 2 stages
    state  = PipelineState.start(phase=1)      # 1 stage
    tile_idx = cta_id
    while tile_idx < total_tiles:
        b, h, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = \
            decode_work_item(work_items, tile_idx)
        # instruction_selection: 2 x ld.global.v4.b32; extent: one 8-field item
        next_tile = sched_publish_next(tile_idx, num_ctas)

        # Head grouping is a load coordinate, never a descriptor patch.
        head_o = h; head_q = h // Q_RATIO; head_k = h // K_RATIO; head_v = h // V_RATIO
        if elected():
            for array in (Q, K, V, GATE, DO, BETA, W, CKPT):
                acquire_tensormap(descriptor_slot(tmaps, array, b))
                # instruction_selection: fence.proxy.tensormap::generic.acquire.gpu

        for rev in range(cend - wstart):               # unroll=1
            chunk = cend - 1 - rev                     # walks HIGH -> LOW
            tok0  = chunk * BT

            for (tensor, bar, head, stage_bytes) in (
                    (SQ, "q", head_q, 4096), (SK, "k", head_k, 4096),
                    (SGATE, "gate", head_o, 8192), (BETA, "beta", head_o, 4096)):
                wait(bar + "_done", raw.stage, raw.phase)
                if elected():
                    expect_bytes(bar + "_ready", raw.stage, stage_bytes)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64
                copy_g2s(gmem_tile(tensor, b, head, tok0), tensor + raw.stage * ...,
                         completion=bar + "_ready")
                # instruction_selection: cp.async.bulk.tensor.3d.shared::cta
                #   .global.tile.mbarrier::complete_tx::bytes; extent: DK//64 = 2
                #   issues per 16-bit tile, DK//32 = 4 for the FP32 gate.  The 16
                #   3d loads are Q2 + K2 + Gate4 + Beta2 + dO2 + V2 + W2.

            if chunk >= FIRST_STATE_CHUNK:
                wait("state_cg0_done", state.stage, state.phase)
                wait("state_done",     state.stage, state.phase)
                if elected():
                    expect_bytes("state_ready", state.stage, 32768)
                copy_g2s(gmem_checkpoint(b, head_o, chunk), STATE,
                         completion="state_ready")
                # instruction_selection: cp.async.bulk.tensor.4d.shared::cta.global
                #   .tile.mbarrier::complete_tx::bytes; extent: one [DV,DK]
                #   checkpoint, box [64,DK,1,1]; 2 static sites
                state = advance(state, 1)

            # dO waits on TWO consumers: the epilogue warp and the MMA warp.
            wait("do_done",     raw.stage, raw.phase)
            wait("do_mma_done", raw.stage, raw.phase)
            if elected():
                expect_bytes("do_ready", raw.stage, 4096)
            copy_g2s(gmem_tile(DO, b, head_o, tok0), DO + raw.stage * 4096,
                     completion="do_ready")
            # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global
            #   .tile.mbarrier::complete_tx::bytes

            for (tensor, bar, head) in ((SV, "v", head_v), (SW, "w", head_o)):
                wait(bar + "_done", raw.stage, raw.phase)
                if elected():
                    expect_bytes(bar + "_ready", raw.stage, 4096)
                copy_g2s(gmem_tile(tensor, b, head, tok0), tensor + raw.stage * 4096,
                         completion=bar + "_ready")
                # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global
                #   .tile.mbarrier::complete_tx::bytes
            raw = advance(raw, 2)
        tile_idx = next_tile
```

The load order is fixed and observable: Q, K, Gate, Beta, entering state, dO, V,
W.  `V` and `W` are released late in GDN-2 -- warp group 1 holds both until
after the scalar `dV`/`dW_out` pass -- so their two raw stages stay occupied
longer than the KDA sibling's do.

```python
# ==========================================================================
# Warp 12: register-MMA program.  KK, the Neumann inverse, and dM.
# Unlike the KDA sibling, NO Beta row scale is applied to L or to dM_strict:
# GDN-2 folds Beta into the K_decay operand instead.
# ==========================================================================
def super_mma_warp():
    while tile_idx < total_tiles:
        ...decode; for chunk in reverse_walk:
            wait("t_inv_done", inter_stage, inter_phase)
            wait("k_decay_inv_ready", decay_stage, decay_phase)

            # ---- KK = K_decay @ K_inv^T ----------------------------------
            fill(kk_acc[0:8], 0.0)
            for k_block in range(DK // 16):
                copy_s2r(K_DECAY + raw_io_byte(...), a_frag[0:4])
                copy_s2r(K_INV   + raw_io_byte(...), b_frag[0:4])
                # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16;
                # extent: one 16x16 operand fragment
                gemm(kk_acc, a_frag, b_frag, accumulate=True)
                # instruction_selection: mma.sync.aligned.m16n8k16.row.col
                #   .f32.bf16.bf16.f32; extent: 8 issues over the K walk

            # ---- L = strict_tril(KK), T_inv = (I + L)^-1 by 3 Neumann rounds
            select(l_regs, strict_lower_mask, kk_acc, 0.0)
            cast(l_packed, l_regs)
            # instruction_selection: cvt.rn.bf16x2.f32; extent: 8 lanes -> 4 packs
            sub(tinv_acc, eye_mask, l_packed)          # T_inv <- I - L
            transpose(mov_lpow, l_packed)          # once before the loop
            for _round in range(3):
                gemm(sq_acc, l_packed, mov_lpow)       # square the strict power
                cast(l_packed, sq_acc)                 # io round-trip each round
                transpose(mov_lpow, l_packed)          # and again after squaring
                # instruction_selection: movmatrix.sync.aligned.m8n8.trans.b16;
                # extent: 4 initial + 3 x 4 = the export's 16 sites
                gemm(upd_acc, tinv_packed, mov_lpow)   # T_inv @ L^(2^r)
                # The round ACCUMULATES into the series; assigning here would
                # never sum it and T_inv would be wrong on every chunk.
                add(tinv_acc, cast_f32(tinv_packed), upd_acc)
                cast(tinv_packed, tinv_acc)
                # instruction_selection: 2 x mma.sync...m16n8k16 per round, then
                # 4 x add.rn.f32x2 over the io round-tripped T_inv; the rounding
                # and the four-way bracketing are both part of the contract
            copy_r2s(tinv_packed, inter_byte(inter_stage, slot=1, ...))
            # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("t_inv_ready", inter_stage)

            # ---- dM = dY @ U^T, then the two strict tiles -----------------
            # U-readiness is transitive: CG1 arrives u_smem_ready before
            # dy_smem_ready, so warp 12 waits only the reverse edge and dY.
            wait("dm_done", inter_stage, inter_phase)
            wait("dy_smem_ready")
            for k_block in range(DV // 16):
                copy_s2r(SDY + ..., a_frag); copy_s2r(SU + ..., b_frag)
                # instruction_selection: ldmatrix...x4.shared.b16
                gemm(dm_acc, a_frag, b_frag, accumulate=True)
                # instruction_selection: 8 x mma.sync...m16n8k16; extent: dM
            select(dm_strict, strict_lower_mask, dm_acc, 0.0)
            copy_r2s(+dm_strict, inter_byte(inter_stage, slot=3, ...))
            copy_r2s(-dm_strict, inter_byte(inter_stage, slot=4, ...))
            # instruction_selection: 2 x stmatrix...x4.shared.b16; extent: the
            # +/- pair, so dk_inv and dk_decay can share one accumulator sign
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("dm_ready", inter_stage)

# ==========================================================================
# Warp 13: TMEM lifecycle and the fifteen tcgen05 GEMM chains.
# ==========================================================================
def tcgen05_warp():
    alloc_tmem(columns=512)
    # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned
    #   .shared::cta.b32; extent: one allocation
    relinquish_alloc_permit()
    # instruction_selection: tcgen05.relinquish_alloc_permit...
    barrier(id=3, threads=416)          # tcgen05 warp + the three compute groups
    # instruction_selection: barrier.sync 3, 416

    while tile_idx < total_tiles:
        ...decode; for chunk in reverse_walk:
            # Each logical GEMM below expands only to its repeated tcgen05.mma
            # family; publication is a separate, delayed operation.
            if chunk >= FIRST_STATE_CHUNK:
                wait("state_ready", state_stage, state_phase)
                for k in range(8):
                    gemm(tmem_cell(TM_STATE_K_ACC, row, 0),
                         state_operand(k), k_decay_operand(decay_stage, k),
                         accumulate=k > 0)
                # instruction_selection: 8 x tcgen05.mma.cta_group::1.kind::f16;
                # extent: GEMM 1
                commit("state_k_acc_ready"); commit("state_done")
                # instruction_selection: one delayed tcgen05.commit.cta_group::1
                #   .mbarrier::arrive::one.shared::cluster.b64 per published edge

            ...                        # GEMMs 2..15 in the order of the table below,
                                       # each followed by its own extent annotation
    # ---- teardown ------------------------------------------------------
    wait("tmem_done")
    dealloc_tmem()
    # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned...

# ==========================================================================
# Warp 15: two register GEMMs, then the one-behind six-store ladder.
# ==========================================================================
def epilogue_warp():
    while tile_idx < total_tiles:
        ...decode
        if elected():
            for array in (TMAP_DQ, TMAP_DK, TMAP_DV, TMAP_DGATE, TMAP_DBETA, TMAP_DWO):
                acquire_tensormap(descriptor_slot(tmaps, array, b))
                # instruction_selection: fence.proxy.tensormap::generic.acquire.gpu;
                # extent: 6 of the export's 14 acquires; warp 14 owns the other 8
        for chunk in reverse_walk:
            writes = chunk < wend       # [wend, cend) is the right warm-up

            # ---- A = tril_incl(Q_decay @ K_inv^T) ------------------------
            wait("a_done", inter_stage, inter_phase)      # reverse edge from warp 13
            wait("q_decay_k_restore_ready", decay_stage, decay_phase)
            for k_block in range(DK // 16):
                copy_s2r(Q_DECAY + ..., a_frag); copy_s2r(K_INV + ..., b_frag)
                # instruction_selection: ldmatrix...x4.shared.b16
                gemm(a_acc, a_frag, b_frag, accumulate=True)
                # instruction_selection: 8 x mma.sync...m16n8k16; extent: A
            select(a_tile, lower_incl_mask, a_acc, 0.0)
            copy_r2s(a_tile, inter_byte(inter_stage, slot=0, ...))
            # instruction_selection: stmatrix...x4.shared.b16
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("a_ready", inter_stage)

            # ---- dA = tril_incl(dO @ U^T) --------------------------------
            # Only warp 13 waits `do_ready`; the epilogue's edge is `u_smem_ready`.
            wait("da_done", inter_stage, inter_phase)     # reverse edge from warp 13
            wait("u_smem_ready")
            for k_block in range(DV // 16):
                copy_s2r(DO + ..., a_frag); copy_s2r(SU + ..., b_frag)
                gemm(da_acc, a_frag, b_frag, accumulate=True)
                # instruction_selection: 8 x mma.sync...m16n8k16; extent: dA
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("do_done", raw_stage)          # released BEFORE the mask/store
            select(da_tile, lower_incl_mask, da_acc, 0.0)
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta; extent: one of
            # the 17 generic-to-async-proxy publishes in the kernel
            copy_r2s(da_tile, inter_byte(inter_stage, slot=2, ...))
            arrive("da_ready", inter_stage)

            # ---- one-behind store ladder ---------------------------------
            # The ladder stores the PREVIOUS chunk (`pend_start`/`pend_writes`),
            # so it runs only from the second iteration on.  The six staging
            # waits are UNCONDITIONAL -- only the store and its commit sit under
            # `pend_writes`.  Gating the waits instead would desynchronise every
            # staging barrier's phase on a warm-up chunk.
            LADDER = ((SDQ, "dq", TMAP_DQ), (SDK, "dk", TMAP_DK),
                      (SDGATE, "dgate", TMAP_DGATE), (SDB, "db", TMAP_DBETA),
                      (SDV, "dv", TMAP_DV), (SDWO, "dwo", TMAP_DWO))
            if rev > 0:
                for (stage_buf, bar, desc) in LADDER:
                    wait(bar + "_tmastg_ready", cursor[bar].stage, cursor[bar].phase)
                    if pend_writes:
                        copy_s2g(stage_buf + cursor[bar].stage * ...,
                                 gmem_tile(desc, b, head_o, pend_start))
                        # instruction_selection: cp.async.bulk.tensor.3d.global
                        #   .shared::cta.tile.bulk_group; extent: one [BT,dim] tile
                        commit_store_group()
                        # instruction_selection: cp.async.bulk.commit_group;
                        # extent: ONE PER STORE, which is what makes the staged
                        # wait_group(5..0) below release exactly one buffer each
                for (slot, (_, bar, _)) in zip((5, 4, 3, 2, 1, 0), LADDER):
                    wait_store_group(slot)
                    # instruction_selection: cp.async.bulk.wait_group.read;
                    # extent: staged release, one staging buffer freed per step
                    arrive(bar + "_tmastg_done", cursor[bar].stage)
                    cursor[bar] = advance(cursor[bar], stages_of(bar))
            pend_start, pend_writes = tok0, writes
        # ---- tile tail: the same ladder once more, draining the last chunk --
        if sk_nt > 0:
            ...                     # identical six-store block over pend_start
```

Out-of-range rows need no predicate here: the prologue capped `GLOBAL_DIM[2]`
at `seqlen_b`, so a tail store clips in hardware.  A work item whose `wstart`
is 0 additionally drains `dstate0`, and a zero-length sequence takes a
pass-through branch that copies or zeros `d_initial_state` without entering the
chunk loop at all.

```python
# ==========================================================================
# Warps 0..3: gate prefix, normalization, and the four decay operands.
# ==========================================================================def compute_group_0():
    barrier(id=3, threads=416)      # TMEM-lifecycle rendezvous after setmaxnreg
    # instruction_selection: barrier.sync 3, 416; extent: one of four such sites
    while tile_idx < total_tiles:
        ...decode; for chunk in reverse_walk:
            # All four raw waits happen up front, before any gate arithmetic.
            wait("q_ready", raw_stage, raw_phase)
            wait("k_ready", raw_stage, raw_phase)
            wait("gate_ready", raw_stage, raw_phase)
            wait("beta_ready", raw_stage, raw_phase)
            copy_s2r(SGATE + raw_f32_byte(...), raw_gate[0:16])
            # instruction_selection: ld.shared.b32; extent: 16 tokens of one
            # channel.  The export carries no ld.shared.f32 at all -- 36 ld.shared.b32
            # and 32 st.shared.b32 cover every f32 SMEM access.

            # ---- gate -> log2 domain, then an in-chunk inclusive prefix ----
            if SAFE_GATE:
                mul(t, raw_gate, exp2(a_log * LOG2_E))
                tanh(s, (t + dt_bias) * 0.5); mul(g_log2, GATE_SCALE_LOG2, s*0.5+0.5)
                # instruction_selection: ex2.approx.ftz.f32 then
                #   tanh.approx.f32; extent: 1 + 16 per chunk (the `safe_gate`
                #   branch's whole delta over `basic`)
            else:
                mul(g_log2, raw_gate, LOG2_E)
            select(g_log2, token_idx < seqlen_b, g_log2, 0.0)   # tail rows decay by 1
            # A fixed pairwise bracketing, not a compiler-chosen reduction:
            #   prefix0 = acc + g0;  pair = g0 + g1;  prefix1 = acc + pair
            add_packed(prefix, acc, g_pair)
            # instruction_selection: add.rn.f32x2; extent: one token pair per step
            exp2(eG, prefix); exp2(eGl, chunk_total)
            # instruction_selection: ex2.approx.ftz.f32; extent: 16 tokens

            wait("decay_done", decay_stage, decay_phase)   # reverse edge, warp 13

            # eG MUST round-trip through sGate: the prefix scan runs on a channel
            # assignment (warp*32 + lane) that differs from the operand assignment
            # (decay_row, lane_in_row_group), and warps 8..11 read the same buffer.
            # That shared reader is why `gate_done` carries CG0+CG2 = 256.
            copy_r2s(eG, SGATE + raw_f32_byte(...))
            # instruction_selection: st.shared.b32; extent: 16 tokens
            fill(STATE_DIAG + inter..., diag(eGl))          # 8 k-atoms of [16][16]
            # instruction_selection: st.shared.b16; extent: 8 atoms

            # ---- raw Q/K into the TMEM ring, then the CG0-local rendezvous --
            wait("qk_raw_done", qk_stage, qk_phase)         # reverse edge from CG2
            copy_r2t(raw_q, tmem_cell(TM_QRAW_INP, row, qk_stage * 8))
            copy_r2t(raw_k, tmem_cell(TM_KRAW_INP, row, qk_stage * 8))
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x8.b32;
            # extent: 2 sites, the raw Q/K ring warps 8..11 later re-read
            tcgen05_wait_store()
            # instruction_selection: tcgen05.wait::st.sync.aligned
            arrive("qk_raw_ready", qk_stage)
            barrier(id=1, threads=128)
            # instruction_selection: barrier.sync 1, 128; extent: separates the ring
            # publish from the raw operand fetch below

            # ---- raw Q/K/Beta register fetch, then release those three stages
            copy_s2r(SQ + raw_io_byte(...), raw_q_regs)
            copy_s2r(SK + raw_io_byte(...), raw_k_regs)
            copy_s2r(BETA + raw_io_byte(...), raw_beta_regs)
            # instruction_selection: ld.shared.v4.b32; extent: 8 channels each
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("q_done", raw_stage); arrive("k_done", raw_stage)
            arrive("beta_done", raw_stage)   # the other half of beta_done's 256

            if L2NORM:
                # The squared sums use a FIXED four-accumulator ffma2 bracketing
                # seeded with opaque zeros so the optimizer cannot reassociate.
                fill(qacc[0:4], opaque_f32_zero()); fill(kacc[0:4], opaque_f32_zero())
                for pair in range(DK // 8):
                    fma_packed(qacc[pair % 4], raw_q_pair, raw_q_pair, qacc[pair % 4])
                    fma_packed(kacc[pair % 4], raw_k_pair, raw_k_pair, kacc[pair % 4])
                    # instruction_selection: fma.rn.f32x2; extent: the fixed 4-way
                    # interleave over this lane's channels
                for delta in (1, 2, 4, 8, 16, 32):
                    shuffle_xor(peer, sum_sq, delta); add(sum_sq, sum_sq, peer)
                    # instruction_selection: shfl.sync.bfly.b32; extent: six steps,
                    # three per gradient, reducing across the row group
                rsqrt(q_inv_norm, max(sum_sq_q, 1e-24)); rsqrt(k_inv_norm, ...)
                # instruction_selection: rsqrt.approx.ftz.f32; extent: 2 sites
                if lane_in_row_group == 0:
                    copy_r2s(q_inv_norm, NORM + ...); copy_r2s(k_inv_norm, NORM + BT + ...)
                    # instruction_selection: st.shared.b32; extent: one lane per row

            # ---- eG/eGl re-read under the OPERAND channel assignment --------
            copy_s2r(SGATE + raw_f32_byte(...), eG_operand)
            copy_s2r(SGATE + raw_f32_byte(row=BT-1, ...), eGl_operand)
            # instruction_selection: ld.shared.b32; extent: the re-read
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("gate_done", raw_stage)
            # gate_done releases the sGate stage for TMA reload, so it MUST come
            # after CG0 has finished reading eG and eGl back out of it.

            # ---- the four operands.  Beta enters K_decay ONLY. -------------
            mul(k_value, raw_k, k_inv_norm)
            mul(k_beta, k_value, raw_beta)                 # GDN-2: Beta folded in
            mul(K_decay_pack, cast(k_beta), cast(eG))
            # instruction_selection: mul.bf16x2; extent: 4 packs per dim half
            rcp(exp_neg_g, eG)                             # never a divide
            # instruction_selection: rcp.approx.ftz.f32; extent: 2 per pack
            mul(K_inv_pack, cast(k_value), cast(exp_neg_g))   # no Beta here
            copy_r2s(K_decay_pack, raw_io_byte(K_DECAY, decay_stage, row, dim))
            copy_r2s(K_inv_pack,   raw_io_byte(K_INV,   decay_stage, row, dim))
            # instruction_selection: st.shared.v4.b32; extent: 8 channels each
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("k_decay_inv_ready", decay_stage)

            # K_restore reuses the ALREADY-ROUNDED K_inv rather than re-deriving
            # from K, and the scale is folded into Q BEFORE the eG multiply.  The
            # rounding points differ from the algebraically equal forms.
            mul(K_restore_pack, K_inv_pack, cast(eGl))
            mul(Q_decay_pack, cast(q_value * scale), cast(eG))
            copy_r2s(K_restore_pack, raw_io_byte(K_RESTORE, decay_stage, row, dim))
            copy_r2s(Q_decay_pack,   raw_io_byte(Q_DECAY,   decay_stage, row, dim))
            # instruction_selection: st.shared.v4.b32; extent: 8 channels each
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("q_decay_k_restore_ready", decay_stage)

            # Only `state_ready` is guarded; the two reverse edges advance every
            # chunk so their phases stay aligned with warp 13 and warps 8..11.
            wait("state_inp_done", state_inp_stage, state_inp_phase)
            wait("state_inp_cg2_done", state_inp_stage, state_inp_phase)
            if chunk >= FIRST_STATE_CHUNK:                  # state SMEM -> TMEM
                wait("state_ready")
                copy_s2r(state_byte(...), state_regs)
                # instruction_selection: ld.shared.b16
                copy_r2t(state_regs, TM_STATE_INP + state_inp_stage * 64)
                # instruction_selection: tcgen05.st.sync.aligned.32x32b.x4.b32;
                # extent: 16 sites
                tcgen05_wait_store()
                # instruction_selection: tcgen05.wait::st.sync.aligned
                arrive("state_cg0_done")
            arrive("state_inp_ready", state_inp_stage)   # UNCONDITIONAL: the phase
            # advances even on a chunk that loads no state

def compute_group_1():
    barrier(id=3, threads=416)
    # instruction_selection: barrier.sync 3, 416
    while tile_idx < total_tiles:
        ...decode
        # dht seed runs at the TOP of the tile, before the chunk loop: a
        # non-terminal item seeds zeros but still walks the same path, so the
        # pipeline shape is uniform.
        if USE_DSTATE_IN and sk_nt > 0:
            wait("dstate_smem_done"); wait("dstate_smem_cg2_done")
            seed_true = (cend == num_chunks_b)     # non-terminal items seed zeros
            copy_g2r(dstate_in_tile, dh_in)
            select(dh_seed, seed_true, dh_in, 0.0)          # per element
            copy_r2t(dh_seed, TM_DSTATE_ACC)       # the FP32 seed: dH itself
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32
            copy_r2t(cast(dh_seed), TM_DSTATE_INP) # and the f16 A-operand form
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32
            tcgen05_wait_store()
            # instruction_selection: tcgen05.wait::st.sync.aligned
            arrive("dstate_inp_ready")
            copy_t2r(TM_DSTATE_INP, dh_reread)     # re-read, then publish to SMEM
            copy_r2s(cast(dh_reread), DSTATE)
            # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
            fence_proxy_async_shared_cta()
            arrive("dstate_smem_ready")
        for chunk in reverse_walk:
            # ---- Y: two variants.  Chunk 0 with no entering state is W*V only.
            wait("v_ready", raw_stage, raw_phase)
            wait("w_ready", raw_stage, raw_phase)
            copy_s2r(SV + ..., v_val); copy_s2r(SW + ..., w_val)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16;
            # extent: 4 sites, V and W in the transposed operand form
            if chunk >= FIRST_STATE_CHUNK:
                wait("state_k_acc_ready")
                copy_t2r(tmem_cell(TM_STATE_K_ACC, row, col), state_k)
                # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x2.b32;
                # extent: 2 sites
                sub(y_val, mul(w_val, v_val), state_k)      # GDN-2: W gates V
                # instruction_selection: sub.bf16x2; extent: 8 packs
            else:
                mul(y_val, w_val, v_val)                    # no entering state
            copy_r2t(cast(y_val), tmem_cell(TM_Y_INP, ...))
            # instruction_selection: tcgen05.st.sync.aligned.16x128b.x2.b32;
            # extent: 2 sites
            tcgen05_wait_store()
            # instruction_selection: tcgen05.wait::st.sync.aligned
            arrive("y_inp_ready")

            # dU is restaged BEFORE U: du_acc is finished by GEMMs 3 and 5, which
            # precede GEMM 7's u_acc in the in-order tcgen05 queue.
            wait("du_acc_ready"); copy_t2r(TM_DU_ACC, du_val)
            # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x2.b32
            copy_r2t(cast(du_val), TM_DU_INP)
            # instruction_selection: tcgen05.st.sync.aligned.16x128b.x2.b32
            tcgen05_wait_store(); arrive("du_inp_ready")

            wait("u_acc_ready"); copy_t2r(TM_U_ACC, u_val)
            # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x2.b32
            copy_r2s(cast(u_val), SU)
            # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("u_smem_ready")

            wait("dy_acc_ready"); copy_t2r(TM_DY_ACC, dy_val)   # ALIAS of state_k
            # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x2.b32
            copy_r2t(cast(-dy_val), TM_NEG_DY_INP)
            # instruction_selection: tcgen05.st.sync.aligned.16x128b.x2.b32
            tcgen05_wait_store(); arrive("neg_dy_inp_ready")
            copy_r2s(cast(dy_val), SDY)
            # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
            fence_proxy_async_shared_cta(); arrive("dy_smem_ready")
            # dY gains a tcgen05 consumer in GDN-2 (dk_decay uses plain dY, not
            # Beta*dY), hence the extra `dy_smem_done` MMA_COMMIT edge.

            # ---- dV and dW_out: one scalar pass over this lane's channel ---
            wait("dv_tmastg_done", dv_stage); wait("dwo_tmastg_done", dwo_stage)
            barrier(id=4, threads=128)
            # instruction_selection: barrier.sync 4, 128
            for t in range(BT):                              # 16 iterations
                copy_s2r(SDY + raw_io_byte(...), dy_v)
                copy_s2r(SW  + raw_io_byte(...), w_v)
                copy_s2r(SV  + raw_io_byte(...), v_v)
                # instruction_selection: ld.shared.b16; extent: three loads per token
                copy_r2s(cast(mul(w_v, dy_v)), SDV  + dv_stage  * 4096 + ...)
                copy_r2s(cast(mul(v_v, dy_v)), SDWO + dwo_stage * 4096 + ...)
                # instruction_selection: st.shared.b16; extent: dV and dW_out
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("dv_tmastg_ready"); arrive("dwo_tmastg_ready")
            arrive("v_done", raw_stage); arrive("w_done", raw_stage)
            # V and W are released only HERE, after the dW_out product needs V.

            # dH capture is UNCONDITIONALLY waited but only performed when a
            # next chunk exists; the SMEM copy re-reads TMEM rather than reusing
            # the registers already in hand.
            wait("dstate_acc_ready")
            if rev + 1 < sk_nt:
                wait("dstate_smem_done")
                wait("dstate_smem_cg2_done")
                copy_t2r(TM_DSTATE_ACC, dh)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32;
                # extent: 4 sites
                copy_r2t(cast(dh), TM_DSTATE_INP)
                # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32;
                # extent: 4 sites
                tcgen05_wait_store()
                copy_t2r(TM_DSTATE_INP, dh_reread)          # explicit re-read
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32;
                # extent: 4 sites
                copy_r2s(cast(dh_reread), DSTATE)
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
                fence_proxy_async_shared_cta()
                arrive("dstate_inp_ready"); arrive("dstate_smem_ready")
    # tail: seed d_final_state when cend == num_chunks_b, drain dstate0 when
    # wstart == 0, and pass through for a zero-length sequence.
    arrive("tmem_done")

# ==========================================================================
# Warps 8..11: gradient drain.  One lane owns one key channel x 16 tokens.
# ==========================================================================
def compute_group_2():
    barrier(id=3, threads=416)
    # instruction_selection: barrier.sync 3, 416
    while tile_idx < total_tiles:
        ...decode; for chunk in reverse_walk:
            wait("k_decay_inv_ready", decay_stage, decay_phase)
            copy_s2r(SGATE + raw_f32_byte(...), eG)      # written back by CG0
            # instruction_selection: ld.shared.b32; extent: 16 tokens
            fence_proxy_async_shared_cta(); arrive("gate_done", raw_stage)

            # `has_dstate` is a RUNTIME boolean, not the compile-time flag: a
            # dState exists once the reverse walk has produced one, so it is
            # `rev > 0`, forced True when USE_DSTATE_IN. The hdot is therefore
            # live even in the `basic` branch, which is why the basic export
            # carries 128 fma.rn.f32.bf16.
            has_dstate = (rev > 0) or USE_DSTATE_IN

            # ---- dGate_last hdot: sum_v(dH[v,c] * S[c,v]) -------------------
            fill(dgate_last_val, 0.0)
            wait("state_inp_ready", state_inp_stage, state_inp_phase)  # UNGUARDED
            if has_dstate:
                wait("dstate_smem_ready")
                copy_t2r(tmem_cell(TM_STATE_INP, ...), state_cols)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32;
                # extent: 4 sites
                copy_s2r(DSTATE + state_byte(...), dh_cols)
                for j in range(DV):
                    fma(hacc[(2 * j) % 8], dh_cols[j], state_cols[j], hacc[(2 * j) % 8])
                    # instruction_selection: fma.rn.f32.bf16; extent: 128 issues
                    # over a FIXED 8-way interleave, not a compiler-chosen tree
                reduce_fixed_tree(dgate_last_val, hacc[0:8])
                # instruction_selection: add.rn.f32x2; extent: the fixed fadd2 tree
                arrive("dstate_smem_cg2_done")
            arrive("state_inp_cg2_done", state_inp_stage)   # UNGUARDED, pairs
            # with CG0's state_inp_cg2_done wait on every chunk

            fill(dgate_last_acc[0:4], opaque_f32_zero())
            fill(dk_n[0:BT], 0.0)        # valid accumulator on the no-dstate path
            if has_dstate:
                wait("dk_restore_part_acc_ready")
                copy_t2r(tmem_cell(TM_DK_RESTORE, row, col), dk_restore_part)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32
                copy_t2r(tmem_cell(TM_KRAW_INP, ...), k_raw)   # ring read 1 of 4
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32
                for t in range(BT):
                    mul(dk_hat, eGl * rcp(eG[t]), dk_restore_part[t])
                    # instruction_selection: rcp.approx.ftz.f32 then mul.f32
                    fma(dgate_last_acc[t % 4], k_n[t], dk_hat, dgate_last_acc[t % 4])
                    # instruction_selection: fma.rn.f32; extent: a FIXED 4-way
                    # interleaved accumulation, not a compiler-chosen tree

            wait("dq_acc_ready"); copy_t2r(TM_DQ_ACC, dq_vec)
            mul(dq_n, mul_packed(eG, scale), dq_vec)
            # instruction_selection: mul.f32x2; extent: TWO packed multiplies per
            # token pair -- (eG*scale) then (*dq_vec); 40 sites in the export

            wait("dk_inv_part_acc_ready"); copy_t2r(TM_DK_INV_ACC, dk_inv_part)
            fma(dk_n, dk_inv_part, rcp(eG), dk_n)

            # ---- dK_decay drain seeds BOTH dBeta and dGate -----------------
            wait("dk_decay_part_acc_ready"); copy_t2r(TM_DK_DECAY_ACC, dk_decay_part)
            copy_t2r(tmem_cell(TM_KRAW_INP, ...), k_raw)     # raw K from the ring
            # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32; extent:
            # one of FOUR separate ring reads -- raw K three times (the restore
            # drain, here, and the dGate finalize) and raw Q once
            for t in range(BT):
                mul(dk_decay, -eG[t], dk_decay_part[t])
                mul(db_regs[t], k_n[t], dk_decay)            # GDN-2: per-channel dBeta
                copy_s2r(BETA + raw_io_byte(...), beta_v)
                if BETA_SIGMOID:
                    tanh(s, beta_v * 0.5); cast(beta_v, s * 0.5 + 0.5)   # io round-trip
                    # instruction_selection: tanh.approx.f32 then cvt.rn.bf16.f32
                    #   and back to f32 -- the rounding is part of the contract
                mul(dgate_regs[t], beta_v, dk_decay)
                add(dk_n[t], dk_n[t], dgate_regs[t])
            tcgen05_wait_load()
            # instruction_selection: tcgen05.wait::ld.sync.aligned; extent: one of
            # the export's two sites
            arrive("dqk_acc_done")

            # ---- dGate finalize: its OWN ring reads and its OWN beta reload -
            copy_t2r(tmem_cell(TM_QRAW_INP, ...), q_raw)     # ring read 3 of 4
            copy_t2r(tmem_cell(TM_KRAW_INP, ...), k_raw)     # ring read 4 of 4
            # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32;
            # extent: 2 sites; with the restore drain and the dk_decay drain these
            # are the four separate ring reads (raw K three times, raw Q once)
            for t in range(BT):
                mul(q_n[t], q_raw[t], q_inv_norm) if L2NORM else assign(q_n[t], q_raw[t])
                copy_s2r(BETA + raw_io_byte(...), beta_v)     # reloaded, not reused
                if BETA_SIGMOID:
                    tanh(s, beta_v * 0.5); cast(beta_v, s * 0.5 + 0.5)
                    # instruction_selection: tanh.approx.f32; extent: the THIRD of
                    # three sigmoid sites (CG0, the dk_decay drain, here), which is
                    # what makes the beta_sigmoid branch's delta 48 and not 16
                dgate_regs[t] = q_n[t]*dq_n[t] + beta_v*db_regs[t] \
                                - k_n[t]*(dk_n[t] - dgate_regs[t])
                # instruction_selection: fma.rn.f32 chain; extent: one token
                if BETA_SIGMOID:
                    mul(db_regs[t], db_regs[t], beta_v - beta_v*beta_v)
                    # ORDER IS LOAD-BEARING: dGate above consumed db_regs BEFORE
                    # the chain rule; swapping these corrupts dGate silently.
            add(dgate_regs[BT-1], dgate_regs[BT-1],
                (dgate_last_acc[0]+dgate_last_acc[1]) + (dgate_last_acc[2]+dgate_last_acc[3]))
            if has_dstate and chunk >= FIRST_STATE_CHUNK:
                fma(dgate_regs[BT-1], eGl, dgate_last_val, dgate_regs[BT-1])
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta
            arrive("beta_done", raw_stage)

            if L2NORM:      # row projection: grad <- (grad - a*norm*<a,grad>)*norm
                for (grad, qk_col, inv_off) in ((dq_n, TM_QRAW_INP, 0),
                                                (dk_n, TM_KRAW_INP, BT)):
                    copy_t2r(tmem_cell(qk_col, ...), raw_lo)   # two halves
                    copy_t2r(tmem_cell(qk_col, ...), raw_hi)
                    # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32;
                    # extent: 2 of FOUR reads per gradient -- the l2norm export
                    # takes tcgen05.ld from 28 to 36 for exactly this reason
                    mul(part, raw_pair, grad_pair)
                    # instruction_selection: mul.f32x2; extent: 2 per token pair
                    for off in range(5):           # ASCENDING: 1, 2, 4, 8, 16
                        shuffle_xor(peer, part, 1 << off); add(part, part, peer)
                        # instruction_selection: shfl.sync.bfly.b32; extent: a
                        # 5-step butterfly over the lanes of one row
                    copy_r2s(part, RED1 + (warp_in_group * BT + t) * 4)
                    # RED1 is 256 B = 4 warps x 16 tokens, not one scalar per warp
                    barrier(id=2, threads=128)
                    # instruction_selection: barrier.sync 2, 128; extent: the first
                    # of TWO such barriers per gradient
                    copy_s2r(RED1 + t * 4, four_warp_parts)   # the four warps' t
                    reduce_fixed_tree(dots, four_warp_parts)
                    # instruction_selection: add.rn.f32x2; extent: a fixed 4-term
                    # tree over the four warps, not a compiler-chosen reduction
                    copy_t2r(tmem_cell(qk_col, ...), raw_lo)  # reads 3 and 4, after
                    copy_t2r(tmem_cell(qk_col, ...), raw_hi)  # the barrier
                    # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32
                    copy_s2r(NORM + inv_off, inv_norm)
                    fma(grad, -raw_qk, dots * inv_norm, grad)
                    mul(grad, grad, inv_norm)          # the post-multiply
                    barrier(id=2, threads=128)
                    # instruction_selection: barrier.sync 2, 128; extent: the second
                    # barrier, before RED1 is reused by the next gradient
            # dQ and dK stage FIRST, before the cumsum, so their staging buffers
            # release as early as possible.
            for (regs, base, bar) in ((dq_n, SDQ, "dq"), (dk_n, SDK, "dk")):
                wait(bar + "_tmastg_done", ...)
                copy_r2s(cast(regs), base + ...)
                # instruction_selection: st.shared.b16; extent: 16 tokens
            fence_proxy_async_shared_cta()
            arrive("dq_tmastg_ready"); arrive("dk_tmastg_ready")

            tcgen05_wait_load()
            # instruction_selection: tcgen05.wait::ld.sync.aligned
            arrive("qk_raw_done", qk_stage)

            reverse_cumsum(dgate_regs)
            # instruction_selection: add.f32 chain; extent: a STRICTLY SEQUENTIAL
            # suffix accumulation from t = BT-1 down to 0, not a parallel scan

            for (regs, base, bar) in ((dgate_regs, SDGATE, "dgate"), (db_regs, SDB, "db")):
                wait(bar + "_tmastg_done", ...)
                copy_r2s(cast(regs), base + ...)
                # instruction_selection: st.shared.b32 for the FP32 dGate,
                # st.shared.b16 for dBeta; extent: 16 tokens
            fence_proxy_async_shared_cta()
            arrive("dgate_tmastg_ready"); arrive("db_tmastg_ready")
    arrive("tmem_done")
```

## Logical tcgen05 GEMMs

Warp 13 issues fifteen logical GEMMs per chunk.  Every one is
`tcgen05.mma.cta_group::1.kind::f16` with `tile_k_hw = 16` and FP32
accumulation, so a GEMM whose K extent is 128 expands to eight issues and one
whose K extent is 16 expands to one.  Two sites are duplicated for the
`accumulate=True`/`False` twin.

Six loop-level waits sit OUTSIDE any GEMM's guard, so their phases advance even
on a chunk whose MMA is skipped: `k_decay_inv_ready`, `dqk_acc_done` (the
loop-carried backpressure edge from warps 8..11, whose cursor starts at phase
1), `state_inp_ready`, `do_ready`, `q_decay_k_restore_ready`, and
`u_smem_ready`.  They are listed here rather than in the per-row waits column.
`dstate0_acc_stored` is waited once at teardown before the TMEM dealloc.  With
row 6 publishing nothing and `decay_done` attributed only to row 15, the
publishes column sums to exactly the export's 19 commits.

| # | FP32 TMEM destination | A operand | B operand | M x N x K | issues | acc | guard | waits | publishes |
| ---: | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| 1 | `state_k_acc` | `state` (SMEM) | `K_decay^T` | 128 x 16 x 128 | 8 | on k>0 | `chunk >= FIRST_STATE_CHUNK` | `state_ready` | `state_k_acc_ready`, `state_done` |
| 2 | `dq_acc` | `state` (TMEM) | `dO^T` | 128 x 16 x 128 | 8 | on k>0 | `chunk >= FIRST_STATE_CHUNK` | (`state_inp_ready`, `do_ready` hoisted) | none yet |
| 3 | `du_acc` | `dstate_inp` (TMEM) | `K_restore` | 128 x 16 x 128 | 8 | on k>0 | `has_dstate` | `dstate_inp_ready` | none yet |
| 4 | `dstate_acc` | `dstate_inp` (TMEM) | `diag(eGl)` | 128 x 16 x 16 | 8 | per k-atom | `has_dstate` | same stage | none yet |
| 5 | `du_acc` | `dO^T` (SMEM) | `A` | 128 x 16 x 16 | 1 | yes | `has_dstate` | `a_ready` | `du_acc_ready`, `a_done` |
| 6 | `dstate_acc` | `dO^T` (SMEM) | `Q_decay` | 128 x 128 x 16 | 1 | yes | `has_dstate` | (hoisted) | none yet |
| 7 | `u_acc` | `Y` (TMEM) | `T_inv` | 128 x 16 x 16 | 1 | no | none | `y_inp_ready`, `t_inv_ready` | `u_acc_ready`, `do_mma_done` |
| 8 | `dy_acc` | `dU` (TMEM) | `T_inv` | 128 x 16 x 16 | 1 | no | none | `du_inp_ready` | `dy_acc_ready`, `t_inv_done` |
| 9 | `dk_restore_acc` | `dH` (SMEM) | `U^T` | 128 x 16 x 128 | 8 | on k>0 | `has_dstate` | `dstate_smem_ready` | `dk_restore_part_acc_ready` |
| 10 | `dstate_acc` | `-dY` (TMEM) | `K_decay` | 128 x 128 x 16 | 1 | yes | none | `neg_dy_inp_ready` | `dstate_acc_ready` |
| 11 | `dk_inv_acc` | `Q_decay^T` (SMEM) | `dA` | 128 x 16 x 16 | 1 | no | none | `da_ready` | none yet |
| 12 | `dq_acc` | `K_inv^T` (SMEM) | `dA^T` | 128 x 16 x 16 | 1 | twin | `chunk >= FIRST_STATE_CHUNK` selects the twin | same stage | `dq_acc_ready`, `da_done` |
| 13 | `dk_decay_acc` | `state` (TMEM) | `dY^T` | 128 x 16 x 128 | 8 | on k>0 | `chunk >= FIRST_STATE_CHUNK` | `dy_smem_ready` | `dy_smem_done`, `state_inp_done` |
| 14 | `dk_inv_acc` | `K_decay^T` (SMEM) | `-dM_strict` | 128 x 16 x 16 | 1 | yes | none | `dm_ready` | `dk_inv_part_acc_ready` |
| 15 | `dk_decay_acc` | `K_inv^T` (SMEM) | `dM_strict^T` | 128 x 16 x 16 | 1 | twin | `chunk >= FIRST_STATE_CHUNK` selects the twin | same stage | `dk_decay_part_acc_ready`, `dm_done`, `decay_done` |

Issue vector `8,8,8,8,1,1,1,1,8,1,1,1,8,1,1` sums to 57 runtime issues; the two
accumulate twins add two more static sites, giving the **59**
`tcgen05.mma.cta_group::1.kind::f16` occurrences the export carries.  The count
is identical in every accepted branch, so the GEMM structure is not specialized
-- only the gating math around it is.

Commits are delayed rather than per-issue: the export holds **19**
`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64` against
those 59 issues, so a chain publishes once after all of its issues.

Register-MMA work sits outside this table: warp 12 runs `KK` (8 x
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`), three Neumann rounds
(2 each) and `dM` (8); warp 15 runs `A` (8) and `dA` (8).  The export's 76
`mma.sync` occurrences cover both warps.

## Static opcode evidence

Counting convention: instruction lines minus predicated lines.  Full per-branch
table in `.porting/gdn2_bprop_f16/static_opcode_evidence.md`.

| family | `basic` main kernel | note |
| --- | ---: | --- |
| `mbarrier.init.shared.b64` | 123 | equals the 123 physical slots derived independently from the barrier table |
| `tcgen05.mma.cta_group::1.kind::f16` | 59 | branch-invariant |
| `tcgen05.commit...shared::cluster.b64` | 19 | delayed, not per-issue |
| `tcgen05.alloc` / `relinquish` / `dealloc` | 1 / 1 / 1 | one 512-column allocation |
| `tcgen05.ld` / `tcgen05.st` | 28 / 28 | TMEM drains and restages |
| `mma.sync.aligned.m16n8k16...bf16` | 76 | warps 12 and 15 |
| `cp.async.bulk.tensor.3d` load / store | 16 / 28 | `shared::cta`, not `shared::cluster` |
| `cp.async.bulk.tensor.4d` load | 2 | the checkpoint tile |
| `ldmatrix` / `stmatrix` / `movmatrix` | 68 / 9 / 16 | |
| `rcp.approx.ftz.f32` | 48 | `1/eG` is a reciprocal, never a divide |
| `ex2.approx.ftz.f32` | 16 | the gate lives in log2 space |
| `barrier.sync` | 7 | named barriers at 512, 416 and 128 threads |
| `fence.proxy.async.shared::cta` | 17 | one per generic-to-async-proxy publish; the `state` branch has 18, the extra being the dht seed's, which `basic` does not compile |
| `setmaxnreg` | 7 | three compute groups plus four single warps |
| `atom.global` | 0 | the ticket atomic appears only in the `dynamic` branch, where it is 1 |

Branch deltas that a port must not flatten: `safe_gate` adds 16 `tanh.approx`
and one `ex2` (the `exp(a_log)`); `beta_sigmoid` adds 48 `tanh.approx`;
`l2norm` adds roughly 900 net instructions; and `state` *drops* `rcp.approx`
from 48 to 32, because an entering state at chunk 0 takes a different restore
path.

## Storage-alias lifetimes

Four aliases carry correctness arguments rather than convenience, and a port
must reproduce the producer/consumer order that makes each safe, not merely the
offsets.

| alias | over | why it is safe |
| --- | --- | --- |
| `dy_acc` | `state_k_acc` (TMEM 320) | warp group 1 consumes `state_k` into `Y` before the `dY = dU @ T_inv` MMA writes, and the `dY` readback precedes `state_k(c+1) = state @ K_decay^T`; the in-order tcgen05 queue plus `state_k_acc_ready` closes the loop |
| `neg_dy_inp` | `y_inp` (TMEM 432) | `U = Y @ T_inv` consumed `Y` before the `dY` block runs (`u_acc_ready`), and `y_inp(c+1)` is gated by `state_k_acc_ready(c+1)`, whose commit covers the `-dY @ K_decay` MMA of chunk `c` |
| `state_alt` / `state_direct` | `STATE` bytes | the same entering state read as two operand forms, differing only in `leading_bytes` (`DV*128` versus 16) |
| `do_lead16` / `do_amaj` | `DO` bytes | `dO` feeds both a register GEMM in warp 15 and two tcgen05 chains, under different descriptor leading offsets |

## TIRx module and validation contract

- The module imports only `tirx_kernels.kern as K`.  No `T`, `Tx`, `I`, no
  `tirx.tile.*`, no tile primitive, and no first-class layout anywhere.
- Every SMEM region is a byte range inside one flat `u8` arena.  `K.smem_pool`
  may own only the 984-byte barrier header; all data addressing is explicit
  scalar arithmetic over `arena.ptr_to([byte])`.
- TMEM is integer columns through `tmem_cell(base, row, col)`.
- `get_kernel` returns the prologue entry followed by the main entry; the timed
  region on both the TIRx and the reference side contains exactly those two
  launches and nothing else.  Checkpoint construction, the work table, and all
  workspace allocation happen outside timing.
- Numerical contract that the upstream test suite enforces directly, and that a
  port therefore may not "simplify": the log2-domain gate; `rcp`/`ex2`/`tanh`/
  `rsqrt` in their approximate forms; the fixed packed-`f32x2` prefix-scan
  bracketing and the fixed 4-way `dgate_last` accumulation; the beta-sigmoid
  round-trip through the I/O dtype; and dGate consuming `db_regs` before the
  beta chain rule.
- There are no FP32 atomics anywhere.  The only atomic is the 32-bit scheduler
  ticket, and it appears solely in the `dynamic` branch.  This is what makes the
  kernel bitwise deterministic, which upstream tests assert.
- The benchmark suite is the only performance authority.  PTX, SASS, NCU
  counters and the static tables above are diagnostic evidence; none of them can
  accept or reject a candidate.

## Instruction-selection summary

- **Placement selects the copy.**  A GMEM tile reaches SMEM as
  `cp.async.bulk.tensor.{3d,4d}.shared::cta.global.tile.mbarrier::complete_tx::bytes`
  and leaves as `...global.shared::cta.tile.bulk_group`, because every such
  region is a TensorMap-described tile whose descriptor the prologue patched.
  Register traffic to SMEM is `ldmatrix`/`stmatrix` where a 16x16 operand
  fragment is wanted and plain `ld.shared`/`st.shared` where a lane owns a
  single channel.
- **Shape selects the MMA.**  A K extent of 128 against `tile_k_hw = 16`
  expands one logical GEMM into eight `tcgen05.mma.cta_group::1.kind::f16`
  issues; a K extent of 16 gives one.  The small 16x16 products that must stay
  in registers (`KK`, the Neumann rounds, `A`, `dA`, `dM`) select
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instead, and the
  transposes between them select `movmatrix.sync.aligned.m8n8.trans.b16`.
- **The schedule selects the synchronization.**  A TMA producer publishes with
  `mbarrier.arrive.expect_tx`; a tcgen05 chain publishes with a delayed
  `tcgen05.commit...mbarrier::arrive::one`, which is why 59 issues carry only 19
  commits.  Cross-role rendezvous that are not producer/consumer edges use CTA
  named barriers, `barrier.sync` at 512, 416 and 128 threads.
- **The gate's domain selects its arithmetic.**  Carrying the decay in log2
  space makes every application an `ex2.approx.ftz.f32` and every inverse an
  `rcp.approx.ftz.f32`; neither a divide nor a natural-base exponential appears.
