<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ c5365737570a2a156d7cae0c4070fa3770ecc670),
Copyright (c) 2026 by FlashInfer team.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer Cake VSA ultrasparse_bsr SM100: coarse WASP pipeline sketch

This is the non-executable, mechanically translatable sketch for
[`tirx_kernels/flashinfer/cake_vsa/cake_vsa_ultrasparse_bsr_sm100.py`](../../../../tirx_kernels/flashinfer/cake_vsa/cake_vsa_ultrasparse_bsr_sm100.py).
The authoritative source is `csrc/cake_vsa/cake_vsa_ultrasparse_bsr_sm_100a.cu`
at FlashInfer commit `c5365737570a2a156d7cae0c4070fa3770ecc670` (file SHA256
`e850a27705a8ccde640b5417bea5e2e3bbc4efcda8a7e578845bd5e0b0c32a73`, 1524 lines,
last changed by upstream `adc49a85`, "perf(cake_vsa): refresh Blackwell
block-sparse WS kernel (#4804)"), launched by
`csrc/cake_vsa/cake_vsa_ultrasparse_bsr_host.cpp` and dispatched by
`flashinfer/cake_vsa.py` (`run_cake_vsa` lines 935-957). The port targets
`sm_100a` only.

First-class layouts are forbidden. Every shared-memory object below is a byte
range in one rank-one `u8` arena. Every alias, stage, descriptor word and TMEM
fragment is selected by scalar integer arithmetic. "Tile" means a logical
rectangle only; it is not a TIRx tile primitive or a layout-bearing value.

## Scope and specialization boundary

| axis | in-scope values | emitted-code consequence |
| --- | --- | --- |
| dtype / head dim / block | BF16, D=128, R=C=128 | fixed QK idesc `0x08200490` (M=128, N=128), PV idesc `0x08210490`, descriptor high word `0x40004040`, 32 KB Q box, 8 KB K/V boxes |
| heads | MHA, `num_q_heads == num_kv_heads == 8` (dispatch predicate); `num_kv_heads` is an ABI argument the device never reads | CTA uses `kv_head = q_head` |
| selected KV blocks per query block | exactly six, shared by all heads (`bsr_indices[mb][6]`, host-checked `selected_blocks == 6`) | three fully unrolled block pairs; instance `i` consumes slots `5-i`, `3-i`, `1-i` |
| scheduling | persistent: `grid_x = min(total_tiles, sm_count)`, `total_tiles = mb*num_q_heads` (host-checked) | rolled tile loop `tile_idx = %ctaid.x; += %nctaid.x`; unsigned `rem.u32`/`div.u32` by `mb` |
| `mb`, `nb`, `num_q_heads`, `softmax_scale_log2` | runtime scalars (`mb >= 625` by dispatch) | `q_tile = tile_idx % mb`, `q_head = tile_idx / mb`; `kv_len = nb*128` feeds one predicate |
| `return_softmax_lse`, `return_temperature_lse` | 0/1 runtime (upstream Python passes 0, 0; the device implements both stores) | two predicated scalar stores of the same `final_lse` |
| `selected_blocks`, `lse_temperature_scale` | ABI arguments never read by the device | none |
| partial tiles | `q_valid = 128 if q_tile < mb else 0`, `row_valid = my_row < q_valid`, `valid_cols = clamp(128*(nb - n_block), 0, 128)`: predicates present, never false/short in the dispatch domain | the export lowers the source's per-lane mask loops (706-775) to one predicate `!(row_valid && n_block < nb)` guarding a whole-fragment `-inf` fill (128 `mov.b32`); `q_valid`/`my_row < q_valid` selects survive as `setp`/`selp` |

Out of scope: fp16, GQA, D64/D96, blk64, blk128_compact, longseq, causal or
any other mask shape, clusters, PDL, non-persistent launch, temperature scaling
(the device reuses `final_lse`).

## Writer line-info evidence

Exports at `.porting/cake_vsa_ultrasparse_bsr_sm100/writer_exports/`:
`source_sm_100a.lineinfo.ptx` (2168 `.loc`, `.version 9.2`, `.target sm_100a`),
`source_sm_100a.ptx` (upstream flags, no line info), `source_sm_100a.cubin`,
`source_sm_100a.lineinfo.cubin`, `source_sm_100a.sass`,
`source_sm_100a.lineinfo.sass`, `source_ptxas_v.txt` (128 registers at entry, 9
barriers, 96 bytes stack frame, 164 bytes spill stores, 220 bytes spill loads),
`keyops_by_loc.txt`, `ptx_opcode_hist.txt`, `sass_opcode_hist.txt`,
`source.sha256`. Build: `nvcc -ptx|-cubin [-lineinfo] --std=c++17
--use_fast_math -arch=sm_100a`.

Static emitted counts (raw PTX instruction lines, including the `@leader`-predicated
`tcgen05.mma`/`tcgen05.commit` lines): 96 `tcgen05.mma.cta_group::1.kind::f16`
(12 chains x 8), 12 `tcgen05.ld...32x32b.x32`, 32 `tcgen05.ld...32x32b.x16`, 32
`tcgen05.ld...32x32b.x8`, 44 `tcgen05.st...32x32b.x16`, 20 `tcgen05.commit`, 7
`tcgen05.wait::st`, 1 `tcgen05.wait::ld`, 49 `cp.async.bulk.tensor.4d` (1 Q + 12
pushes x 4), 20 `mbarrier.init`, 29 `mbarrier.arrive`, 13
`mbarrier.arrive.expect_tx`, 58 `mbarrier.try_wait` loops, 46 `elect.sync`, 22
`shfl.sync.idx.b32`, 4 `vote.sync.any.pred`, 1 `barrier.sync 8, 32`, 3
`setmaxnreg`, 388 `ex2.approx.ftz.f32` (384 probabilities + 2 rescales + 2
combine scales), 192 `fma.rn.ftz.f32x2`, 448 `mul.rn.ftz.f32x2`, 192 `add.f32x2`,
391 `max.f32`, 132 `add.ftz.f32`, 13 `mul.ftz.f32`, 7 `fma.rn.ftz.f32`, 2
`sub.ftz.f32`, 6 `neg.ftz.f32`, 16 `selp.f32`, 256 `cvt.rn.bf16x2.f32`, 16
`st.global.v4.b32`, 2 `st.global.b32`, 1 `ld.global.nc.b32`, 23 `ld.shared.b32`,
6 `st.shared.b32`, 2 `ld.volatile.shared.b32`, 5 `fence.proxy.async.shared::cta`,
1 `lg2.approx.ftz.f32`, 1 `rcp.approx.ftz.f32`, 1 `rem.u32`, 2 `div.u32`, 1
`mov.u32 %nctaid.x`. SASS: 96 `UTCHMMA`, 76 `LDTM`, 44 `STTM`, 49
`UTMALDG.4D`, 20 `UTCBAR`, 388 `MUFU.EX2`, 1 `MUFU.LG2`, 448 `FMUL2.FTZ`, 192
`FFMA2.FTZ`, 192 `FADD2`, 137 `FADD.FTZ`, 195 `FMNMX3`, 256
`F2FP.BF16.F32.PACK_AB`, 16 `STG.E.128`, 2 `STG.E`, 1 `LDG.E`, 16 `FSEL`, 16
`ELECT`, plus register-pressure traffic (48 `LDL`, 34 `STL`, 23 `MOV.SPILL`, 23
`R2UR.FILL`).

## Pipeline at a glance

One persistent CTA per SM (grid `min(mb*8, sm_count)`); every role walks the same
`tile_idx` sequence and processes one `(query block of 128 rows, head)` per
iteration. Six KV blocks per tile are handled as three pairs; the two softmax
instances each own all 128 query rows and take one block of each pair.

| warps | role-local program | publication / reuse edges |
| --- | --- | --- |
| 0..3, 4..7 | two softmax instances (instance `i = warp/4`, blocks `5-i, 3-i, 1-i`): read the 128x128 f32 S tile from TMEM (128 columns per thread), row max with a `2^-8` rescale-skip decision, `ex2` probabilities, bf16 P written in place over the upper 64 S columns, running sum/max | consume `union_ready`, `s_full[i]`, `corr_done[i]`; produce `corr_sig[i]` (rescale factor, then final stats in SMEM), `p_full[i]` |
| 8..11 | correction + epilogue: rescale the TMEM O accumulator of an instance when a warp vote says any row needs it; finally merge both instances (combine scales, one reciprocal), store bf16 rows and the LSE | consume `union_ready`, `corr_sig[i]`, `o_full[i]`; produce `corr_done[i]`, `p_full[i]`, `q_empty`, `tile_done` |
| 12 | sole MMA issuer: `S_i = Q K^T` (8 issues, M=128) then `O_i += P_i V` (8 issues, A from TMEM) per block; order S5 S4 / PV5 S3 PV4 S2 / PV3 S1 PV2 S0 / PV1 PV0 | consume `union_ready`, `q_full`, `kv_full`, `p_full[i]`, `tile_done`; produce `s_full[i]`, `kv_empty`, `o_full[i]`; frees TMEM after the loop |
| 13, 14 | register-decreased idle warps | none |
| 15 | load warp: per tile wait for the epilogue, Q TMA (32 KB), copy the six BSR indices into SMEM, then the three-stage K/V TMA ring in the order K5 K4 V5 K3 V4 K2 V3 K1 V2 K0 V1 V0 | consume `q_empty`, `kv_empty`; produce `q_full`, `union_ready`, `kv_full` |

## Primitive vocabulary

Structural operations do not compute values:

```python
linear_smem_u8(bytes, alignment)   # the one rank-one arena
byte(...)                          # integer byte offset into the arena
reg_tile(...)                      # role-local register storage
tmem_addr(lane_origin, column)     # integer TMEM address (lane<<16 | column)
```

Copies always state their storage direction:

```python
copy_g2s_tma(map, coords, dst_byte, completion)   # global -> shared (TMA)
copy_g2r(src, dst)                                # global -> register
copy_r2g(src, dst)                                # register -> global
copy_s2r(byte, dst) / copy_r2s(src, byte)         # shared <-> register
copy_t2r(tmem_addr, dst, shape) / copy_r2t(src, tmem_addr, shape)
```

Compute vocabulary: `fill, cast, add, sub, mul, fma, max, exp2, log2, rcp,
select, shuffle_index, any, gemm, udiv, umod`. `ring(stages, stage, parity)` is a
`(stage, parity)` cursor: `stage, parity = r.stage, r.parity; r.advance()` captures
the wait operands before the advance (`advance`: `stage += 1`; at `stages` it wraps
to 0 and flips `parity`). In this kernel every ring wait has a trace-time-constant
stage and parity because 12 pushes per tile are exactly four ring turns. Packed forms carry `lanes=2` (one
`f32x2` instruction with two ordered results). Schedule operations:
`init_mbarrier, wait, arrive, arrive_expect_tx, tmem_commit, fence,
named_barrier, sync_warp, sync_cta, elect, setmaxnreg, tmem_alloc,
tmem_relinquish, tmem_dealloc, tmem_wait_ld, tmem_wait_st`.

## Fixed storage and integer maps

```python
THREADS=512; WARPS=16; D=128; BLOCK=128; KV_STAGES=3; TMEM_COLS=512
SMEM_BYTES=135424; SELECTED=6; PAIRS=3
REG_SOFTMAX=192; REG_CORRECTION=80; REG_OTHER=48      # 8*192+4*80+4*48 = 2048
launch(grid=(min(total_tiles, sm_count),1,1), block=(512,1,1), min_blocks_per_sm=1,
       dynamic_smem_bytes=SMEM_BYTES, target="sm_100a")
# instruction_selection: `.reqntid`-free `__launch_bounds__(512,1)` entry with
# 128 registers, `.extern .shared .align 1024`; the grid extent is the host's
# min(total_tiles, multi_processor_count) (152 on GB200); extent: one specialization.

ABI=(Qmap, Kmap, Vmap,                     # const __grid_constant__ CUtensorMap
     out_bf16[M*Hq*128], lse_f32, temperature_lse_f32, bsr_indices_i32[mb*6],
     mb_i32, nb_i32, selected_blocks_i32, total_tiles_i32, num_q_heads_i32,
     num_kv_heads_i32, softmax_scale_log2_f32, lse_temperature_scale_f32,
     return_softmax_lse_i32, return_temperature_lse_i32)
# TensorMaps (host): bf16, rank 4, SWIZZLE_128B, no L2 promotion, no OOB fill.
#   Qmap dims {64, M, Hq, 2}, byte strides {Hq*256, 256, 128}, box {64,128,1,2}
#   Kmap/Vmap dims {64, N, 2, Hkv}, byte strides {Hkv*256, 128, 256}, box {64,64,1,1}
# A Q coordinate (0, token, head, 0) moves 128 tokens x both dim halves = 32768 B
# laid out as [dim_half][128 rows][128 B]; a K/V coordinate (0, token, dim_half,
# head) moves 64 tokens x one half = 8192 B.

arena = linear_smem_u8(SMEM_BYTES, alignment=1024)
OFF = dict(
  q_full=0, q_empty=8, union_ready=16, kv_full=24, kv_empty=48, s_full=72, p_full=88,
  corr_sig=104, corr_done=120, o_full=136, tile_done=152, tmem_mailbox=160,
  q=1024, kv=33792, scale=132096, union_count=135168, union_blocks=135172)
def bar(name, index=0): return OFF[name] + 8*index          # 20 mbarriers in [0,160)
INIT_COUNT = dict(q_full=1, q_empty=128, union_ready=32, kv_full=1, kv_empty=1, s_full=1,
                  p_full=256, corr_sig=128, corr_done=128, o_full=1, tile_done=128)
# union_count (135168) is declared by the source and never accessed.

# Q: one 128-row tile; two 16384-byte dim halves of 128 rows x 128 B with the TMA
# 128-byte swizzle, consumed only through MMA descriptors.
def q_byte(dim_half): return OFF.q + 16384*dim_half
# K/V ring: stage = {tokens 0-63 half 0 @+0, tokens 64-127 half 0 @+8192,
#                    tokens 0-63 half 1 @+16384, tokens 64-127 half 1 @+24576}
def kv_byte(stage, token_half, dim_half): return OFF.kv + 32768*stage + 8192*token_half + 16384*dim_half
def scale_byte(kind, instance, row):   # kind: 0 acc_scale, 1 row_sum, 2 row_max
    return OFF.scale + 4*(256*kind + 128*instance + row)
def union_block_byte(slot): return OFF.union_blocks + 4*slot        # slots 0..5 used
# instruction_selection: scalar `ld.shared.b32`/`st.shared.b32`; extent: rank-one
# byte offsets, never a multidimensional shared tensor.

# UMMA shared descriptors are built as two 32-bit halves and packed with mov.b64.
DESC_HI = 0x40004040            # SBO 1024 B, base-offset/version bit 46, 128 B swizzle
def desc_lo(byte): return (smem_base_addr(byte) >> 4) & 0x3FFF
QK_IDESC = 0x08200490           # bf16 x bf16 -> f32, M=128, N=128, K-major A and B
PV_IDESC = 0x08210490           # same with MN-major B (V is [token, feature])
# Q/K low-word walk over K=128 in eight K16 steps: +2,+2,+2 inside a 64-column
# half, then +1018 (+16384 B) to the other half, then +2,+2,+2 (identical for A and B
# because the Q tile and the K tile share the [dim_half][128 rows] shape).
QK_STEPS = (2,2,2,1018,2,2,2)
# V descriptor: low word |= 0x4000000 encodes LBO = 16384 B (dim-half stride);
# +128 (2048 B = 16 tokens x 128 B) per K16 step.  P is read from TMEM at +8
# columns per K16 (16 bf16 columns packed two per 32-bit column).
PV_B_STEP = 128; PV_A_STEP = 8

# TMEM: 512 columns, lane field in bits 31..16.
S_COL=(0,128); P_COL=(64,192); O_COL=(256,384)          # per-instance column bases;
# S spans 128 columns, P overwrites S_COL+64..+128, O spans 128 columns
def tmem_addr(lane_origin, col): return taddr + col + (lane_origin << 16)
# 32x32b fragments: a warp covers 32 lanes = 32 rows; thread `lane` owns row
# lane_origin + lane and every column it loads.  Four x32 loads at col+0, +32,
# +64, +96 give one thread the full 128-column S row; x16/x8 loads read 16/8
# consecutive columns of the row.
```

## Exact prologue and pipeline state

```python
tid = thread_id(); warp = shuffle_index(tid // 32, lane=0); lane = tid % 32
# instruction_selection: `shr.u32`, `shfl.sync.idx.b32 ..., 0, 0x1f, 0xffffffff`, `and.b32`; extent: one.
smem_base = cvta_shared(arena)
bid = cta_id_x(); num_bids = grid_dim_x()
# instruction_selection: `mov.u32 %r, %ctaid.x` and `mov.u32 %r, %nctaid.x`; extent: one each.

if warp == 0:
    if elect():
        # instruction_selection: `elect.sync _|p, 0xFFFFFFFF`; extent: warp 0.
        for name in (q_full, q_empty, union_ready, kv_full x3, kv_empty x3, s_full x2,
                     p_full x2, corr_sig x2, corr_done x2, o_full x2, tile_done):
            init_mbarrier(bar(name, i), INIT_COUNT[name])
        # instruction_selection: 20 x `mbarrier.init.shared::cta.b64`; extent: one lane.
        fence("mbarrier_init_release_cluster")
        # instruction_selection: `fence.mbarrier_init.release.cluster`; inside the leader branch.
sync_warp()
# instruction_selection: `bar.warp.sync -1` executed by every warp.
if warp == 0:
    tmem_alloc(OFF.tmem_mailbox, columns=512); tmem_relinquish()
    # instruction_selection: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [mailbox], 512`
    # then `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`; extent: warp 0.
sync_cta()
# instruction_selection: `bar.sync 0`; extent: all 512 threads.
fence("tcgen05_after_thread_sync")
# instruction_selection: `tcgen05.fence::after_thread_sync`; extent: all threads.
taddr = copy_s2r(OFF.tmem_mailbox, volatile=True)
# instruction_selection: `ld.volatile.shared.b32`; extent: all threads.

# Phase state (every wait is parity-based); the scalars below live across tiles.
# Consumer parities start at 0; the producer-side q_empty and kv_empty start at 1.
PHASE = dict(union_ready=0, q_full=0, q_empty_producer=1, kv_empty_producer=1,
             s_full=[0,0], p_full=[0,0], corr_sig=[0,0], corr_done=[0,0],
             o_full=[0,0], tile_done=0)
# The KV ring cursor (stage 0..2, parity) advances once per K or V push, 12 pushes
# per tile = four full turns, so every tile starts at (stage 0, parity 0) in the MMA
# warp and (stage 0, parity 1) in the load warp.  No ring state crosses tiles and
# the export folds all 12 kv_full waits (MMA) and 12 kv_empty waits (load) to
# immediate stage addresses (+24/+32/+40 resp. +48/+56/+64) and immediate parities:
# MMA 0,0,0,1,1,1,0,0,0,1,1,1; load 1,1,1,0,0,0,1,1,1,0,0,0.  The only load-warp
# parity register that lives across tiles is q_empty's.
```

## Source-order role programs

The branch order below is source-exact: `dec(12..15) -> softmax(0..7) ->
correction(8..11) -> mma(12) -> idle(13,14) -> load(15)`. Every role runs the
rolled tile loop `for tile_idx in serial_range(bid, total_tiles, step=num_bids)`
(`#pragma unroll 1`; entry guard `setp.ge.u32`, back edge `add.s32`+`setp.lt.u32`)
and recomputes `q_tile = umod(tile_idx, mb)`, `q_head = udiv(tile_idx, mb)`,
`kv_head = q_head`, `query_base = q_tile*128`, `q_valid = 128 if q_tile < mb else 0`,
`kv_len = nb*128`. The softmax warps only materialise `q_tile` (`rem.u32`); the
correction and load warps materialise `q_head` (`div.u32`) and derive `q_tile` as
`tile_idx - q_head*mb` (`mul.lo.s32`, `sub.s32`); the MMA warp needs neither.

```python
if 12 <= warp <= 15:
    setmaxnreg(decrease, REG_OTHER)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; extent: warps 12..15,
    # emitted before any increase so the pool can grant the softmax request.

# 1. Warps 0..7: two four-warp softmax instances.
if warp <= 7:
    setmaxnreg(increase, REG_SOFTMAX)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 192`; extent: warps 0..7.
    for tile_idx in serial_range(bid, total_tiles, step=num_bids):       # rolled
        q_tile = umod(tile_idx, mb); q_valid = select(q_tile < mb, 128, 0)
        # instruction_selection: `rem.u32`, `setp.lt.s32` + `selp.b32`; extent: scalars.
        wait(bar(union_ready), PHASE.union_ready); PHASE.union_ready ^= 1
        # instruction_selection: `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` retry
        # loop with suspend hint 0x989680, `xor.b32`; extent: all 256 softmax threads.
        instance = shuffle_index(warp // 4, lane=0)
        instance_tmem_offset = shuffle_index(instance*128, 0)
        instance_row_offset = shuffle_index(instance*128, 0)
        # instruction_selection: three `shfl.sync.idx.b32`; extent: warp-uniform broadcasts.
        warp_in_instance = warp % 4; lane_origin = warp_in_instance*32
        my_row = lane_origin + lane; row_addr = lane_origin << 16
        row_valid = my_row < q_valid
        # instruction_selection: `shl.b32`, `setp.lt.s32`; extent: scalars.
        row_max = -inf; row_sum = 0.0
        score_addr = tmem_addr(lane_origin, S_COL[instance])
        p_addr = tmem_addr(lane_origin, P_COL[instance])

        for pair in static_range(3):                                   # `#pragma unroll`
            n_block = copy_s2r(union_block_byte(5 - instance - 2*pair))
            # instruction_selection: `ld.shared.b32` at `union_blocks + 4*(5-2*pair) - 4*instance`;
            # extent: one scalar per pair.
            if instance == 0: wait(bar(s_full, 0), PHASE.s_full[0]); PHASE.s_full[0] ^= 1
            else:             wait(bar(s_full, 1), PHASE.s_full[1]); PHASE.s_full[1] ^= 1
            # instruction_selection: branch on `instance`, parity retry wait on one of two
            # static barrier addresses, `xor.b32`; extent: 128 threads per instance.
            valid_cols = clamp(kv_len - n_block*128, 0, 128) if row_valid else 0
            s = reg_tile([128], "f32")
            for quarter in static_range(4):
                copy_t2r(score_addr + 32*quarter, s[32*quarter : 32*quarter+32], shape="32x32b.x32")
            # instruction_selection: four `tcgen05.ld.sync.aligned.32x32b.x32.b32 {32}, [addr]`;
            # extent: one 128-row x 128-column S tile, 128 f32 per thread.
            if not (row_valid and n_block < nb):                         # valid_cols < 128
                for j in static_range(128): s[j] = -inf
            # instruction_selection: `setp.gt.s32 nb, n_block` + `and.pred` with row_valid,
            # branch over one `mov.b32 0fFF800000` immediate plus 128 register `mov.b32`
            # copies of it; the source's per-32-column mask
            # arithmetic (706-775) is folded: valid_cols is provably 0 or 128, so every
            # lane mask is 0 and the fill covers the whole fragment; never taken in-domain.
            acc = (-inf, -inf)
            for quarter in static_range(4):
                for j in static_range(16):                       # two accumulators alternate
                    acc[j % 2] = max(acc[j % 2], max(s[32*quarter+2*j], s[32*quarter+2*j+1]))
            tile_max = max(acc[0], acc[1])
            # instruction_selection: 128 + 1 `max.f32` (no ftz), SASS fuses pairs into
            # `FMNMX3`; extent: one 128-value fragment, no cross-lane shuffle.
            new_max = max(tile_max, row_max)
            safe_max = select(new_max == -inf, 0.0, new_max)
            new_max_scaled = mul(safe_max, softmax_scale_log2)
            acc_scale_log2 = fma(row_max, softmax_scale_log2, -new_max_scaled)
            # instruction_selection: `max.f32`, `setp.eq.ftz.f32`+`selp.f32`, `mul.ftz.f32`,
            # `neg.ftz.f32` + `fma.rn.ftz.f32` (pair 0 folds row_max to the `0fFF800000`
            # immediate); extent: scalars per thread.
            if acc_scale_log2 >= -8.0:
                selected_max = row_max; safe_max = select(row_max == -inf, 0.0, row_max)
                acc_scale = 1.0
                new_max_scaled = mul(safe_max, softmax_scale_log2)
                # instruction_selection (pairs 1, 2): `setp.ltu.ftz.f32 .., 0fC1000000` + `@p bra`;
                # this arm: `setp.eq.ftz.f32`+`selp.f32`, `mul.ftz.f32`, `mov.b32 acc_scale, 0f3F800000`.
            else:
                selected_max = new_max
                acc_scale = select(row_max > -inf, exp2(acc_scale_log2), 1.0)
                # instruction_selection (pairs 1, 2): `ex2.approx.ftz.f32`, `setp.ne.ftz.f32` +
                # `selp.f32`, `mov.b32 row_max, new_max`.
            # Pair 0 (row_max is the -inf immediate): no branch.  `setp.ltu.ftz.f32` then two
            # `selp.f32`: new_max_scaled = p ? safe_max*scale : (softmax_scale_log2 * 0.0, one
            # `mul.ftz.f32` hoisted out of the tile loop) and row_max = p ? new_max : -inf;
            # acc_scale is the immediate 1.0 (`st.shared.b32 [..], 1065353216` below); the
            # 793/795/798/799 instructions do not exist for pair 0.
            row_max = selected_max
            copy_r2s(acc_scale, scale_byte(0, instance, my_row))
            # instruction_selection: `st.shared.b32`; extent: 128 rows per instance.
            fence("proxy_async_shared_cta")
            # instruction_selection: `fence.proxy.async.shared::cta`; extent: every softmax thread.
            if instance == 0: arrive(bar(corr_sig, 0))
            else:             arrive(bar(corr_sig, 1))
            # instruction_selection: `mbarrier.arrive.release.cta.shared::cta.b64` on one of two
            # static addresses; extent: 128 arrivals per instance (barrier count 128).
            for j in static_range(64):
                s[2*j:2*j+2] = fma(s[2*j:2*j+2], (softmax_scale_log2,)*2, (-new_max_scaled,)*2, lanes=2)
            # instruction_selection: `neg.ftz.f32` + `mov.b64` packs, 64 `fma.rn.ftz.f32x2`;
            # extent: 128 values.
            packed_p = reg_tile([64], "b32")
            for quarter in static_range(4):
                for j in static_range(32): s[32*quarter+j] = exp2(s[32*quarter+j])
                for j in static_range(16):
                    packed_p[16*quarter+j] = cast(pack(s[32*quarter+2*j], s[32*quarter+2*j+1]), "bf16x2")
                copy_r2t(packed_p[16*quarter : 16*quarter+16], p_addr + 16*quarter, shape="32x32b.x16")
            # instruction_selection: per quarter 32 `ex2.approx.ftz.f32`, 16 `cvt.rn.bf16x2.f32`,
            # one `tcgen05.st.sync.aligned.32x32b.x16.b32 [addr], {16}`; extent: P columns
            # P_COL..P_COL+64 of this instance, written in place over the upper S half.
            tmem_wait_st()
            # instruction_selection: `tcgen05.wait::st.sync.aligned`.
            if instance == 0:
                arrive(bar(p_full, 0)); wait(bar(corr_done, 0), PHASE.corr_done[0]); PHASE.corr_done[0] ^= 1
            else:
                arrive(bar(p_full, 1)); wait(bar(corr_done, 1), PHASE.corr_done[1]); PHASE.corr_done[1] ^= 1
            # instruction_selection: `mbarrier.arrive.release.cta.shared::cta.b64` (128 of the 256
            # expected arrivals; the other 128 come from the correction warps), parity retry
            # wait, `xor.b32`; extent: 128 threads per instance.
            acc2 = (0.0, 0.0)
            for j in static_range(64): acc2 = add(acc2, s[2*j:2*j+2], lanes=2)
            block_sum = add(acc2[0], acc2[1])
            # instruction_selection: 64 `add.f32x2` (no ftz) into one accumulator, `add.ftz.f32`;
            # extent: one row sum, computed after the corr_done wait as in the source.
            row_sum = fma(row_sum, acc_scale, block_sum)
            # instruction_selection: `fma.rn.ftz.f32` (pairs 1, 2); pair 0 folds `0*acc_scale +
            # block_sum` to `add.ftz.f32 block_sum, 0f00000000`.

        copy_r2s(row_sum, scale_byte(1, instance, my_row))
        copy_r2s(row_max, scale_byte(2, instance, my_row))
        # instruction_selection: two `st.shared.b32`; extent: 128 rows per instance.
        fence("proxy_async_shared_cta")
        if instance == 0: arrive(bar(corr_sig, 0))
        else:             arrive(bar(corr_sig, 1))
        # instruction_selection: `fence.proxy.async.shared::cta` then
        # `mbarrier.arrive.release.cta.shared::cta.b64`; the final corr_sig generation of the tile.

# 2. Warps 8..11: correction and epilogue.
if 8 <= warp <= 11:
    setmaxnreg(decrease, REG_CORRECTION)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 80`; extent: warps 8..11.
    for tile_idx in serial_range(bid, total_tiles, step=num_bids):       # rolled
        q_head = udiv(tile_idx, mb); q_tile = tile_idx - q_head*mb
        query_base = q_tile*128; q_valid = select(q_tile < mb, 128, 0)
        # instruction_selection: `div.u32`, `mul.lo.s32`, `sub.s32`, `shl.b32`, `setp.lt.s32`+`selp.b32`.
        wait(bar(union_ready), PHASE.union_ready); PHASE.union_ready ^= 1
        # instruction_selection: parity retry wait, `xor.b32`; extent: 128 threads.
        warp_in_role = warp - 8; lane_origin = warp_in_role*32
        my_row = lane_origin + lane; row_addr = lane_origin << 16
        arrive(bar(p_full, 0)); arrive(bar(p_full, 1))
        # instruction_selection: two `mbarrier.arrive.release.cta.shared::cta.b64`; the
        # correction half of the first p_full generation needs no O rescale.
        wait(bar(corr_sig, 0), PHASE.corr_sig[0]); PHASE.corr_sig[0] ^= 1; arrive(bar(corr_done, 0))
        wait(bar(corr_sig, 1), PHASE.corr_sig[1]); PHASE.corr_sig[1] ^= 1; arrive(bar(corr_done, 1))
        # instruction_selection: parity wait then ordinary arrive, per instance; pair 0
        # never rescales.
        for pair in static_range(1, 3):                                # `#pragma unroll`
            for instance in static_range(2):
                wait(bar(corr_sig, instance), PHASE.corr_sig[instance]); PHASE.corr_sig[instance] ^= 1
                # instruction_selection: parity retry wait; extent: 128 threads.
                acc_scale = copy_s2r(scale_byte(0, instance, my_row))
                # instruction_selection: `ld.shared.b32`; extent: one row scale.
                if any(acc_scale < 1.0):
                    # instruction_selection: `setp.lt.ftz.f32` + `vote.sync.any.pred`; one warp vote.
                    for chunk in static_range(8):
                        o = reg_tile([16], "f32")
                        copy_t2r(tmem_addr(lane_origin, O_COL[instance]) + 16*chunk, o, shape="32x32b.x16")
                        for j in static_range(8): o[2*j:2*j+2] = mul(o[2*j:2*j+2], (acc_scale,)*2, lanes=2)
                        copy_r2t(o, tmem_addr(lane_origin, O_COL[instance]) + 16*chunk, shape="32x32b.x16")
                    # instruction_selection: per chunk one `tcgen05.ld.sync.aligned.32x32b.x16.b32`,
                    # eight `mul.rn.ftz.f32x2`, one `tcgen05.st.sync.aligned.32x32b.x16.b32`;
                    # extent: 8 chunks = 128 O columns of one instance.
                    tmem_wait_st()
                    # instruction_selection: `tcgen05.wait::st.sync.aligned`.
                arrive(bar(p_full, instance)); arrive(bar(corr_done, instance))
                # instruction_selection: two `mbarrier.arrive.release.cta.shared::cta.b64`.
        wait(bar(o_full, 0), PHASE.o_full[0]); PHASE.o_full[0] ^= 1
        wait(bar(o_full, 1), PHASE.o_full[1]); PHASE.o_full[1] ^= 1
        wait(bar(corr_sig, 0), PHASE.corr_sig[0]); PHASE.corr_sig[0] ^= 1
        wait(bar(corr_sig, 1), PHASE.corr_sig[1]); PHASE.corr_sig[1] ^= 1
        # instruction_selection: four parity retry waits; extent: both O accumulators and
        # both final statistics generations.
        fence("tcgen05_after_thread_sync")
        # instruction_selection: `tcgen05.fence::after_thread_sync`.
        final_sum0 = copy_s2r(scale_byte(1, 0, my_row)); final_sum1 = copy_s2r(scale_byte(1, 1, my_row))
        final_max0 = copy_s2r(scale_byte(2, 0, my_row)); final_max1 = copy_s2r(scale_byte(2, 1, my_row))
        # instruction_selection: four `ld.shared.b32`.
        valid0 = final_sum0 > 0.0; valid1 = final_sum1 > 0.0
        max0 = select(valid0, final_max0, -inf); max1 = select(valid1, final_max1, -inf)
        # instruction_selection: two `setp.gt.ftz.f32` (the `sum == sum` NaN test is folded),
        # two `selp.f32`.
        final_max = max(max0, max1); safe_max = select(final_max == -inf, 0.0, final_max)
        # instruction_selection: `max.f32`, `setp.eq.ftz.f32` + `selp.f32`.
        combine_scale0 = select(valid0, exp2(mul(sub(max0, safe_max), softmax_scale_log2)), 0.0)
        combine_scale1 = select(valid1, exp2(mul(sub(max1, safe_max), softmax_scale_log2)), 0.0)
        # instruction_selection: per instance `sub.ftz.f32`, `mul.ftz.f32`, `ex2.approx.ftz.f32`,
        # `selp.f32`; extent: scalars.
        final_sum = fma(final_sum0, combine_scale0, mul(final_sum1, combine_scale1))
        # instruction_selection: `mul.ftz.f32` (sum1*cs1) then `fma.rn.ftz.f32` fusing sum0*cs0.
        inv_sum = select(final_sum > 0.0, rcp(final_sum), 0.0)
        # instruction_selection: `rcp.approx.ftz.f32`, `setp.gt.ftz.f32`, `selp.f32`.
        output_scale0 = mul(combine_scale0, inv_sum); output_scale1 = mul(combine_scale1, inv_sum)
        # instruction_selection: two `mul.ftz.f32`, then `mov.b64` packs of each scale pair.
        query = query_base + my_row; output_row = (query*num_q_heads + q_head)*128
        # instruction_selection: `add.s32`, `mad.lo.s32`, `shl.b32 7`.
        if my_row < q_valid:
            # instruction_selection: `setp.ge.s32` branch over the whole store block.
            for chunk in static_range(16):
                o0 = reg_tile([8], "f32"); o1 = reg_tile([8], "f32")
                copy_t2r(tmem_addr(lane_origin, O_COL[0]) + 8*chunk, o0, shape="32x32b.x8")
                copy_t2r(tmem_addr(lane_origin, O_COL[1]) + 8*chunk, o1, shape="32x32b.x8")
                # instruction_selection: two `tcgen05.ld.sync.aligned.32x32b.x8.b32`.
                for j in static_range(4): o0[2*j:2*j+2] = mul(o0[2*j:2*j+2], (output_scale0,)*2, lanes=2)
                for j in static_range(4): o1[2*j:2*j+2] = mul(o1[2*j:2*j+2], (output_scale1,)*2, lanes=2)
                for j in static_range(8): o0[j] = add(o0[j], o1[j])
                for j in static_range(4): o0[2*j:2*j+2] = mul(o0[2*j:2*j+2], (1.0,)*2, lanes=2)
                # instruction_selection: 4 + 4 `mul.rn.ftz.f32x2`, 8 scalar `add.ftz.f32`, then
                # 4 `mul.rn.ftz.f32x2` by the packed 1.0 (kept: the ftz multiply flushes
                # denormals and is inline asm in the source); extent: 8 columns.
                words = [cast(pack(o0[2*j], o0[2*j+1]), "bf16x2") for j in static_range(4)]
                copy_r2g(words, out[output_row + 8*chunk : +8])
                # instruction_selection: four `cvt.rn.bf16x2.f32`, one `st.global.v4.b32` at
                # `[base + 16*chunk]` (one `mul.wide.s32` + `add.s64` base per row);
                # extent: 16 stores per thread = one 128-column output row.
            stat_idx = query*num_q_heads + q_head
            final_lse = select(final_sum > 0.0,
                               fma(mul(final_max, softmax_scale_log2), LN2, mul(log2(final_sum), LN2)),
                               -inf)
            # instruction_selection: `lg2.approx.ftz.f32` (asm volatile: emitted regardless of
            # the flags), `mul.ftz.f32` x2, `fma.rn.ftz.f32` with the ln2 immediate,
            # `setp.gt.ftz.f32`, `selp.f32`; extent: one scalar per live row.
            if return_softmax_lse != 0:     copy_r2g(final_lse, lse[stat_idx])
            if return_temperature_lse != 0: copy_r2g(final_lse, temperature_lse[stat_idx])
            # instruction_selection: two `setp.eq.b32`-guarded `st.global.b32` (each with
            # `mul.wide.s32` + `add.s64`); the same value goes to both tensors.
        tmem_wait_ld()
        fence("tcgen05_before_thread_sync")
        arrive(bar(q_empty)); arrive(bar(tile_done))
        # instruction_selection: `tcgen05.wait::ld.sync.aligned`, `tcgen05.fence::before_thread_sync`,
        # two `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 128 arrivals each.

# 3. Warp 12: MMA issuer and TMEM owner.
if warp == 12:
    for tile_idx in serial_range(bid, total_tiles, step=num_bids):       # rolled
        wait(bar(union_ready), PHASE.union_ready); PHASE.union_ready ^= 1
        # instruction_selection: parity retry wait, `xor.b32`; extent: warp 12.
        kv = ring(stages=3, stage=0, parity=0)                           # per tile; every stage/parity below is a trace-time constant
        wait(bar(q_full), PHASE.q_full); PHASE.q_full ^= 1
        # instruction_selection: parity retry wait, `xor.b32`; extent: one Q generation.
        first_pv = [True, True]

        def qk_chain(instance, k_stage):                                 # S_instance = Q K^T
            a_lo = shuffle_index(desc_lo(q_byte(0)), 0)
            b_lo = shuffle_index(desc_lo(kv_byte(k_stage, 0, 0)), 0)
            # instruction_selection: two `shfl.sync.idx.b32` uniform broadcasts.
            leader = elect()
            for k16 in static_range(8):
                gemm(tmem_addr(0, S_COL[instance]), desc64(a_lo, DESC_HI), desc64(b_lo, DESC_HI),
                     QK_IDESC, accumulate=(k16 != 0), predicate=leader)
                a_lo += QK_STEPS[k16]; b_lo += QK_STEPS[k16]
            # instruction_selection: one `elect.sync`, `setp.ne.b32` x2, `mov.b32` immediates
            # (0x40004040, 136316048), eight `mov.b64 {lo,hi}` pairs and eight
            # `@leader tcgen05.mma.cta_group::1.kind::f16 [d], da, db, id, p` with `add.u32`
            # low-word steps; extent: K=128 in eight K16 issues, M=128 x N=128.
            tmem_commit(bar(s_full, instance), elect=True)
            tmem_commit(bar(kv_empty, k_stage), elect=True)
            # instruction_selection: two elected
            # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`.

        def pv_chain(instance, v_stage, last_pair):                      # O_instance += P_instance V
            wait(bar(p_full, instance), PHASE.p_full[instance]); PHASE.p_full[instance] ^= 1
            # instruction_selection: parity retry wait, `xor.b32`; extent: one P generation.
            b_lo = shuffle_index(desc_lo(kv_byte(v_stage, 0, 0)) | 0x4000000, 0)
            # instruction_selection: `shfl.sync.idx.b32`; LBO bit set for the MN-major V.
            ta = tmem_addr(0, P_COL[instance]); leader = elect()
            for k16 in static_range(8):
                gemm(tmem_addr(0, O_COL[instance]), tmem_operand(ta), desc64(b_lo, DESC_HI),
                     PV_IDESC, accumulate=(k16 != 0 or not first_pv[instance]), predicate=leader)
                ta += PV_A_STEP; b_lo += PV_B_STEP
            # instruction_selection: one `elect.sync`, `setp.ne.b32` x2 (p0 from first_pv),
            # `mov.b32` immediates (0x40004040, 136381584), eight `@leader tcgen05.mma.cta_group::1.kind::f16
            # [d], [ta], db, id, p` with `add.u32 ta, ta, 8` and `add.u32 blo, blo, 128`;
            # the first issue of the instance's first block in the tile passes enable_input_d = 0.
            first_pv[instance] = False
            if last_pair:
                tmem_commit(bar(o_full, instance), elect=True)
                # instruction_selection: elected `tcgen05.commit...mbarrier::arrive::one`; pair 2 only.
            tmem_commit(bar(kv_empty, v_stage), elect=True)
            # instruction_selection: elected `tcgen05.commit...mbarrier::arrive::one`; V release.

        for instance in static_range(2):                                 # S5, S4
            k_stage, k_parity = kv.stage, kv.parity; kv.advance()
            wait(bar(kv_full, k_stage), k_parity)
            # instruction_selection: parity retry wait with immediate stage address and
            # immediate parity (sites 1-2 of the 12: +24/0, +32/0); extent: one K stage.
            qk_chain(instance, k_stage)
        for pair in static_range(3):
            for instance in static_range(2):
                v_stage, v_parity = kv.stage, kv.parity; kv.advance()
                wait(bar(kv_full, v_stage), v_parity)
                # instruction_selection: parity retry wait with immediate stage/parity (the 12
                # MMA kv_full waits in push order carry parities 0,0,0,1,1,1,0,0,0,1,1,1);
                # extent: one V stage.
                pv_chain(instance, v_stage, last_pair=(pair == 2))
                if pair < 2:                                             # prefetch S for slot 5-instance-2*(pair+1)
                    k_stage, k_parity = kv.stage, kv.parity; kv.advance()
                    wait(bar(kv_full, k_stage), k_parity)
                    # instruction_selection: parity retry wait with immediate stage/parity; extent: one K stage.
                    qk_chain(instance, k_stage)
        wait(bar(tile_done), PHASE.tile_done); PHASE.tile_done ^= 1
        # instruction_selection: parity retry wait, `xor.b32`; the epilogue has released TMEM.
    taddr_again = copy_s2r(OFF.tmem_mailbox, volatile=True)
    tmem_dealloc(taddr_again, columns=512)
    # instruction_selection: `ld.volatile.shared.b32`,
    # `tcgen05.dealloc.cta_group::1.sync.aligned.b32 addr, 512`; extent: warp 12, after the loop.

# 4. Warps 13, 14: idle after the register decrease.
if 13 <= warp <= 14:
    pass

# 5. Warp 15: Q load, BSR index publish, K/V ring producer.
if warp == 15:
    for tile_idx in serial_range(bid, total_tiles, step=num_bids):       # rolled
        wait(bar(q_empty), PHASE.q_empty_producer); PHASE.q_empty_producer ^= 1
        # instruction_selection: parity retry wait (parity register starts at 1, `xor.b32` per
        # tile; the only load-warp parity that lives across tiles); the previous tile's
        # epilogue has finished with Q and union_blocks.
        q_head = udiv(tile_idx, mb); q_tile = tile_idx - q_head*mb; query_base = q_tile*128
        kv_head = q_head
        # instruction_selection: `div.u32`, `mul.lo.s32`, `sub.s32`, `shl.b32`.
        if elect():
            arrive_expect_tx(bar(q_full), 32768)
            copy_g2s_tma(Qmap, (0, query_base, q_head, 0), q_byte(0), bar(q_full))
            # instruction_selection: `elect.sync`, `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`,
            # one `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes`;
            # extent: 32768 bytes (both dim halves of 128 rows in one issue).
        q_block = query_base // 128
        if lane < 6:
            copy_r2s(copy_g2r(bsr_indices[q_block*6 + lane]), union_block_byte(lane))
            # instruction_selection: `setp.gt.u32 lane, 5` + `@p bra` (skip when lane > 5),
            # `mad.lo.s32`, `mul.wide.s32`, `add.s64`, `ld.global.nc.b32`, `st.shared.b32`;
            # extent: six lanes.
        named_barrier(8, threads=32)
        fence("proxy_async_shared_cta")
        arrive(bar(union_ready))
        # instruction_selection: `barrier.sync 8, 32`, `fence.proxy.async.shared::cta`,
        # `mbarrier.arrive.release.cta.shared::cta.b64` by all 32 lanes (count 32).
        load = ring(stages=3, stage=0, parity=1)                        # per tile; 12 pushes return it to (0, 1)

        def push(map, slot):
            n_block = copy_s2r(union_block_byte(slot)); token0 = n_block*128; token1 = token0 + 64
            # instruction_selection: `ld.shared.b32`, `shl.b32`; extent: one scalar.
            stage, parity = load.stage, load.parity
            wait(bar(kv_empty, stage), parity)
            # instruction_selection: parity retry wait by all 32 lanes with immediate stage
            # address (+48/+56/+64) and immediate parity (push order 1,1,1,0,0,0,1,1,1,0,0,0).
            if elect():
                arrive_expect_tx(bar(kv_full, stage), 32768)
                copy_g2s_tma(map, (0, token0, 0, kv_head), kv_byte(stage, 0, 0), bar(kv_full, stage))
                copy_g2s_tma(map, (0, token1, 0, kv_head), kv_byte(stage, 1, 0), bar(kv_full, stage))
                copy_g2s_tma(map, (0, token0, 1, kv_head), kv_byte(stage, 0, 1), bar(kv_full, stage))
                copy_g2s_tma(map, (0, token1, 1, kv_head), kv_byte(stage, 1, 1), bar(kv_full, stage))
                # instruction_selection: `elect.sync`, `mbarrier.arrive.expect_tx...` 32768,
                # four `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes`
                # of 8192 bytes; extent: one 128-token K or V tile per push, 12 static sites.
            load.advance()

        for instance in static_range(2): push(Kmap, 5 - instance)      # K5, K4
        for pair in static_range(3):
            for instance in static_range(2):
                slot = 5 - instance - 2*pair
                push(Vmap, slot)                                         # V for the block being consumed
                if slot - 2 >= 0: push(Kmap, slot - 2)                   # K prefetch for the next pair
```

## Bidirectional source / sketch / PTX map

| semantic edge | source lines | sketch section | exported `.loc` / opcode evidence |
| --- | --- | --- | --- |
| helpers: elect, mbarrier init/wait/arrive/expect_tx, commit, TMA, warp-uniform, TMEM ld/st, approx math, f32x2 | 44-61, 92-105, 193-216, 219-274, 277-295, 314-327, 362-392, 395-407, 495-529 | primitive vocabulary | `.loc 53` (14 elect), `60` (20 init), `104` (58 waits), `201` (20 commits), `208` (29 arrives), `215` (13 expect_tx), `234/247/260/273/520` (12/32/12/32/32 TMEM ops), `279` (388 ex2), `286` (rcp), `293` (391 max), `390` (192 add.f32x2), `400` (192 fma.f32x2), `406` (448 mul.f32x2), `502` (49 TMA), `527` (22 shfl.idx), `122` (4 votes) |
| dead helpers (try_wait/cluster/token waits, `tcgen05_mma_f16`, `desc_encode`, `mma_ss/ts_step`, `warp_reduce_*`, `ex2_emulation_f32x2`, unused f32x2 wrappers, `fma_scale_x32`, `fence_async_shared`, `make_smem_desc`, `tcgen05_commit`) | 64-90, 107-132, 135-190, 298-311, 330-360, 409-476, 481-492, 506-511 | omitted | no `.loc` |
| ABI, smem carve-up, blockIdx/gridDim | 533-559 | fixed storage / prologue | `.loc 536-538` (`shr`, `shfl.idx`, `and`), `544-545` (`%ctaid.x`, `%nctaid.x`) |
| mbarrier init, warp sync, TMEM alloc, CTA sync, TMEM base | 561-634 | exact prologue | `.loc 564-601` (20 inits + `fence.mbarrier_init`), `603` `bar.warp.sync`, `609-610` alloc/relinquish, `613` `bar.sync 0`, `614` fence, `628` `ld.volatile.shared` |
| register redistribution | 638-644, 857-858 | role entries | `.loc 639` dec 48, `644` inc 192, `858` dec 80 |
| softmax tile loop, tile split, union_ready, uniform broadcasts, rows | 652-676 | role 1 head | `.loc 652` (`setp.ge.u32`/`add.s32`/`setp.lt.u32`), `654` `rem.u32`, `659` `selp`, `667-668` wait+`xor`, `669-672` `shfl.idx` x3, `675-676` `shl`/`setp.lt` |
| softmax per pair: slot read, s_full wait, S load, folded mask fill, max tree, rescale decision, acc_scale publish | 680-808 | role 1 pair body | `.loc 682` `ld.shared`, `683-689` waits, `702-705` 4 x `32x32b.x32`, `706` `setp.gt.s32`+`and.pred`+128 `mov -inf`, `776-782` (via `.loc 293`), `785-787` `selp`/`mul.ftz`/`neg`+`fma.rn.ftz`, `791` `setp.ltu` (pairs 1-2: `@p bra` + arms at `793/795/798/799`; pair 0: two `selp.f32` under `.loc 0`, hoisted `mul.ftz.f32 scale, 0f00000000`, `st.shared.b32 .., 1065353216`), `802-808` `st.shared`, fence, arrives; no `.loc` for 707-775 |
| softmax probabilities: fma, exp2, cvt, P store, wait, p_full/corr_done, block sum, running sum | 809-843 | role 1 pair tail | `.loc 810` `neg`+`mov.b64`, `400` (64 fma x2 per pair), `279`/`373-377`/`260` (per quarter 32 ex2, 16 cvt, one x16 store), `827` `wait::st`, `828-836` arrive/wait, `390`/`842` (64 add.f32x2 + `add.ftz`), `843` `fma.rn.ftz` (pairs 1-2) / `add.ftz 0.0` (pair 0) |
| softmax final stats | 845-852 | role 1 tile tail | `.loc 845-849` two `st.shared`, fence, arrive |
| correction tile head, handshake, rescale pairs | 866-937 | role 2 loop | `.loc 866/868/871/873` (loop, `div.u32`, `shl`, `selp`), `881-894` waits/arrives, `896-936` (`.loc 899/919` `ld.shared`, `900/920` `setp.lt.ftz`, `122` votes, `247/406/273` x16 ld / mul / x16 st, `913/933` `wait::st`) |
| epilogue: waits, merge statistics, normalize, store, LSE, release | 938-1022 | role 2 epilogue | `.loc 938-946` waits + fence, `947-966` (`ld.shared` x4, `setp.gt.ftz` x2, `selp`, `max`, `sub.ftz`/`mul.ftz` x2, ex2 x2, `mul.ftz`+`fma.rn.ftz`, `rcp`, `selp`, `mul.ftz` x2), `967-969` row math + branch, `520` (32 x8 loads), `406` (192 mul.f32x2), `988` (128 `add.ftz`), `1001-1005` (64 cvt, 16 `st.global.v4.b32`), `1010-1016` lg2 + LSE + 2 predicated `st.global.b32`, `1019-1022` `wait::ld`, fence, arrives |
| MMA warp: tile loop, waits, S chains, PV chains, tile_done, dealloc | 1035-1418 | role 3 | `.loc 1035-1041` loop/waits, `1050/1178/1292` 12 kv_full waits with immediate stage/parity (0,0,0,1,1,1,0,0,0,1,1,1), `1052-1053/1110-1111` `shfl.idx`, asm chains closing at `1107/1165/1227/1280/1349/1407` (96 `tcgen05.mma`), `201` commits, `1180/1233` p_full waits, `1414-1418` `tile_done` wait, `ld.volatile.shared`, `tcgen05.dealloc` |
| idle warps | 1422-1424 | role 4 | none |
| load warp: q_empty, tile split, Q TMA, BSR publish, ring | 1431-1516 | role 5 | `.loc 1431-1435` loop/wait (parity register init 1 + `xor.b32`)/`div.u32`, `1448-1450` elect/expect_tx/TMA, `1453` `setp.gt.u32 lane, 5`, `1454-1455` `ld.global.nc.b32`/`st.shared`, `1457-1459` `barrier.sync`/fence/arrive, `1464-1516` (`ld.shared` x12, `shl`, 12 kv_empty waits with immediate stage/parity 1,1,1,0,0,0,1,1,1,0,0,0, `502` 48 TMA) |

Reading in reverse, every TMA, TMEM load/store, MMA, commit, mbarrier, named
barrier, warp vote, shuffle, transcendental, packed f32x2, conversion, vector
global store, special-register read, unsigned division and register-budget
instruction in the export maps to a concrete occurrence above. Address arithmetic
maps to the integer functions of "Fixed storage and integer maps", never to a
layout.

## TIRx, correctness and performance contract

The public module exports `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`,
`get_kernel`, `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, and
`run_bench`. Device code imports only `tirx_kernels.kern as K`; low-level
instruction families use `K.ptx[...]` / `K.ptx.*`, the grid stride reads
`%nctaid.x` through `K.cuda.mov_sreg`, and the grid extent is
`K.min(total_tiles, sm_count)` with the SM count read once at kernel build. There
is no inline CUDA function call, tile primitive, first-class layout or
multidimensional SMEM allocation. The kernel is compiled once, shape-generic,
through nvcc 13.2 (`compile_kernel(..., cuda_compile_mode="nvcc")`) so TIRx and
the upstream `nvcc -cubin` build share PTX ISA 9.2.

The frozen correctness matrix has 6 rows inside the dispatch domain (8 heads,
`mb >= 625`, six distinct shared blocks per row): an uneven persistent tail
without LSE, the six-block dense case with LSE, 1024 KV blocks with unsorted
caller-order columns, a banded K-amplified case that forces the O rescale and
unequal combine scales, a diagonal window over 54 waves, and first/last-block
edges with both LSE stores. TIRx and the pinned upstream kernel share immutable
inputs and independent outputs; `out`, `lse` and `temperature_lse` must be
bitwise identical between them, a second TIRx launch must reproduce the first
bit for bit, no-LSE sentinels stay untouched, NaN is forbidden, the public
`plan_cake_vsa`/`run_cake_vsa` route must dispatch here and agree bitwise on
`out`, and both implementations are checked against one FP64 block-gather
oracle with tolerances frozen from measurement.

Performance truth is exclusively `python -m tirx_kernels.bench_suite` over the
frozen 6-row benchmark matrix with the pinned FlashInfer reference. Each row
must have five finite positive Proton samples for both implementations and
strict `mean(flashinfer)/mean(tirx) > 0.99`.
