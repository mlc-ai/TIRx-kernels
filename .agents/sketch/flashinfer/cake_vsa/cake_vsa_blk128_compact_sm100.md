<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ c5365737570a2a156d7cae0c4070fa3770ecc670),
Copyright (c) 2026 by FlashInfer team.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer Cake VSA blk128_compact SM100: coarse WASP pipeline sketch

This is the non-executable, mechanically translatable sketch for
[`tirx_kernels/flashinfer/cake_vsa/cake_vsa_blk128_compact_sm100.py`](../../../../tirx_kernels/flashinfer/cake_vsa/cake_vsa_blk128_compact_sm100.py).
The authoritative source is `csrc/cake_vsa/cake_vsa_blk128_compact_sm_100a.cu`
at FlashInfer commit `c5365737570a2a156d7cae0c4070fa3770ecc670` (file SHA256
`e220b84c36930737d9da00676950a547402ba34e40eaa2f33061102bb6b4f7b6`, last
changed by upstream `adc49a85`, "perf(cake_vsa): refresh Blackwell block-sparse
WS kernel (#4804)"), launched by `csrc/cake_vsa/cake_vsa_blk128_compact_host.cpp`
and dispatched by `flashinfer/cake_vsa.py`. The port targets `sm_100a` only.

First-class layouts are forbidden. Every shared-memory object below is a byte
range in one rank-one `u8` arena. Every alias, stage, descriptor word and TMEM
fragment is selected by scalar integer arithmetic. "Tile" means a logical
rectangle only; it is not a TIRx tile primitive or a layout-bearing value.

## Scope and specialization boundary

| axis | in-scope values | emitted-code consequence |
| --- | --- | --- |
| dtype / head dim / block | BF16, D=128, R=C=128 | fixed QK idesc `0x04200490`, PV idesc `0x04210490`, descriptor high word `0x40004040`, 16 KB / 8 KB TMA boxes |
| heads | MHA (`num_q_heads == num_kv_heads`); `num_kv_heads` is an ABI argument the device never reads | CTA uses `kv_head = q_head` |
| selected KV blocks per query block | 1..64 (compact list capacity); a zero-count row degrades to block 0 with count 1 | one rolled `union_index` loop per role |
| `mb`, `nb`, `num_q_heads`, `softmax_scale_log2` | runtime scalars | grid `(mb, num_q_heads, 1)`; ballot loop over `nb` in 32-block steps |
| `return_softmax_lse` | 0/1 runtime | predicate on the scalar LSE store |
| `return_temperature_lse`, `lse_temperature_scale` | 0/1 and f32 runtime (upstream Python passes 0 and 1.0) | selects the temperature branch that re-reads S from TMEM and keeps a second running sum |
| partial KV / Q tiles | source predicates present, never true (planner requires `M % 128 == N % 128 == 0`, grid x = `mb`) | `valid_cols`, `q_valid`, `instance_valid`, `row_valid` selects stay; the per-element `-inf` fill loops (source 683-717, 794-829) are compiler-dead: `valid_cols = 128*(nb - n_block)` clamps to exactly 0 or 128, so `0 < half_valid < 64` is unsatisfiable and the export has no `.loc` for those lines |

Out of scope: fp16, GQA, D64/D96, blk64, ultrasparse BSR, longseq, causal or
any other mask shape, clusters, PDL, persistent scheduling.

## Writer line-info evidence

Exports at `.porting/cake_vsa_blk128_compact_sm100/writer_exports/`:
`source_sm_100a.lineinfo.ptx` (1051 `.loc`, `.version 9.2`, `.target sm_100a`),
`source_sm_100a.ptx` (upstream flags, no line info), `source_sm_100a.cubin`,
`source_sm_100a.lineinfo.cubin`, `source_sm_100a.sass`, `source_ptxas_v.txt`
(128 registers, 0 spill bytes, 9 barriers), `keyops_by_loc.txt`,
`ptx_opcode_hist.txt`, `ptx_scalar_glue.txt`, `sass_opcode_hist.txt`. Build:
`nvcc -ptx|-cubin [-lineinfo] --std=c++17 --use_fast_math -arch=sm_100a`.

Static emitted counts (raw PTX instruction lines, including the `@leader`-predicated
`tcgen05.mma`/`tcgen05.commit` lines):
32 `tcgen05.mma.cta_group::1.kind::f16`, 12 `tcgen05.ld...16x32bx2.x32`,
4 `tcgen05.st...16x32bx2.x16`, 2 `tcgen05.st...16x32bx2.x64`, 8
`tcgen05.commit`, 18 `cp.async.bulk.tensor.4d`, 19 `mbarrier.init`, 16
`mbarrier.arrive`, 5 `mbarrier.arrive.expect_tx`, 27 `mbarrier.try_wait`
loops, 18 `elect.sync`, 194 `ex2.approx.ftz.f32`, 96 `fma.rn.ftz.f32x2`, 128
`mul.rn.ftz.f32x2`, 96 `add.f32x2`, 67 `max.f32`, 128 `cvt.rn.bf16x2.f32`, 16
`st.global.v4.b32`, 4 `lg2.approx.ftz.f32`, 2 `rcp.approx.ftz.f32`, 11
`shfl.sync.idx.b32`, 4 `shfl.sync.bfly.b32`, 1 `vote.sync.ballot.b32`, 2
`vote.sync.any.pred`. SASS: 32 `UTCHMMA`, 24 `LDTM.16`, 12 `STTM.16`, 18
`UTMALDG.4D`, 8 `UTCBAR`, 194 `MUFU.EX2`, 128 `FMUL2.FTZ`, 96 `FFMA2.FTZ`, 96
`FADD2`, 34 `FMNMX3`/`FMNMX`, 128 `F2FP.BF16.F32.PACK_AB`, 16 `STG.E.128`.

## Pipeline at a glance

One CTA per `(query block of 128 rows, head)`; nothing is persistent.

| warps | role-local program | publication / reuse edges |
| --- | --- | --- |
| 0..3, 4..7 | two softmax instances (64 query rows each): read the 64x128 f32 S tile from TMEM, row max with a `2^-8` rescale-skip decision, `ex2` probabilities, bf16 P written in place over the upper S columns, running sum/max | consume `s_full[i]`, `corr_done[i]`; produce `corr_sig[i]` (rescale factor and final stats in SMEM), `p_full[i]` |
| 8..11 | correction + epilogue: rescale the TMEM O accumulator of either instance when a warp vote says any row needs it; finally normalize O, store bf16 rows and the LSE | consume `corr_sig[i]`, `o_full[i]`; produce `corr_done[i]`, `p_full[i]`, `tmem_dealloc` |
| 12 | sole MMA issuer: `S_i = Q_i K^T` (8 issues) then `O_i += P_i V` (8 issues, A from TMEM) per selected block and instance | consume `q_full`, `kv_full`, `p_full[i]`; produce `s_full[i]`, `kv_empty`, `o_full[i]`; frees TMEM after `tmem_dealloc` |
| 13, 14 | register-decreased idle warps | none |
| 15 | load warp: Q TMA, ballot compaction of the block-mask row into the shared selected-block list, then the three-stage K/V TMA ring in order K0, K1, V0, V1 per selected block | produce `q_full`, `union_ready`, `kv_full`; consume `kv_empty` |

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
select, shuffle_xor, shuffle_index, ballot, any, popc, gemm`. Packed forms carry
`lanes=2` (one `f32x2` instruction with two ordered results). Schedule
operations: `init_mbarrier, wait, arrive, arrive_expect_tx, tmem_commit,
fence, named_barrier, sync_warp, sync_cta, elect, setmaxnreg, tmem_alloc,
tmem_relinquish, tmem_dealloc, tmem_wait_ld, tmem_wait_st`.

## Fixed storage and integer maps

```python
THREADS=512; WARPS=16; D=128; BLOCK=128; KV_STAGES=3; TMEM_COLS=512
SMEM_BYTES=134784; MAX_COMPACT=64
REG_SOFTMAX=192; REG_CORRECTION=80; REG_OTHER=48      # 8*192+4*80+4*48 = 2048
launch(grid=(mb,num_q_heads,1), block=(512,1,1), min_blocks_per_sm=1,
       dynamic_smem_bytes=SMEM_BYTES, target="sm_100a")
# instruction_selection: `.reqntid`-free `__launch_bounds__(512,1)` entry with
# 128 registers, `.extern .shared .align 1024`; extent: one specialization.

ABI=(Qmap, Kmap, Vmap,                     # const __grid_constant__ CUtensorMap
     out_bf16[M*Hq*128], lse_f32, temperature_lse_f32, block_mask_u8[Hq*mb*nb],
     mb_i32, nb_i32, num_q_heads_i32, num_kv_heads_i32,
     softmax_scale_log2_f32, lse_temperature_scale_f32,
     return_softmax_lse_i32, return_temperature_lse_i32)
# TensorMaps (host): bf16, rank 4, SWIZZLE_128B, no L2 promotion, no OOB fill.
#   Qmap dims {64, M, Hq, 2}, byte strides {Hq*256, 256, 128}, box {64,64,1,2}
#   Kmap/Vmap dims {64, N, 2, Hkv}, byte strides {Hkv*256, 128, 256}, box {64,64,1,1}
# A Q coordinate (0, token, head, 0) moves 64 tokens x both dim halves = 16384 B;
# a K/V coordinate (0, token, dim_half, head) moves 64 tokens x one half = 8192 B.

arena = linear_smem_u8(SMEM_BYTES, alignment=1024)
OFF = dict(
  q_full=0, union_ready=8, kv_full=16, kv_empty=40, s_full=64, p_full=80,
  corr_sig=96, corr_done=112, o_full=128, tmem_dealloc=144, tmem_mailbox=152,
  q0=1024, q1=17408, kv=33792, scale=132096, union_count=134144, union_blocks=134152)
def bar(name, index=0): return OFF[name] + 8*index          # 19 mbarriers
INIT_COUNT = dict(q_full=1, union_ready=32, kv_full=1, kv_empty=1, s_full=1,
                  p_full=256, corr_sig=128, corr_done=128, o_full=1, tmem_dealloc=128)

# Q: two 64-row tiles; each tile is two 8192-byte dim halves of 64 rows x 128 B
# with the TMA 128-byte swizzle, consumed only through MMA descriptors.
def q_byte(tile, dim_half): return OFF.q0 + 16384*tile + 8192*dim_half       # tile 1 lands at OFF.q1 = 17408
# K/V ring: stage = {tokens 0-63 half 0 @+0, tokens 64-127 half 0 @+8192,
#                    tokens 0-63 half 1 @+16384, tokens 64-127 half 1 @+24576}
def kv_byte(stage, token_half, dim_half): return OFF.kv + 32768*stage + 8192*token_half + 16384*dim_half
def scale_byte(kind, instance, row):   # kind: 0 acc_scale, 1 row_sum, 2 row_max, 3 temperature_sum
    return OFF.scale + 4*(128*kind + 64*instance + row)
def union_count_byte(instance): return OFF.union_count + 4*instance
def union_block_byte(instance, slot): return OFF.union_blocks + 4*(64*instance + slot)
# instruction_selection: scalar `ld.shared.b32`/`st.shared.b32` (the two
# union counts are read together as one `ld.shared.v2.b32`); extent: rank-one
# byte offsets, never a multidimensional shared tensor.

# UMMA shared descriptors are built as two 32-bit halves and packed with mov.b64.
DESC_HI = 0x40004040            # SBO 1024 B, base-offset/version bit 46, 128 B swizzle
def desc_lo(byte): return (smem_base_addr(byte) >> 4) & 0x3FFF
QK_IDESC = 0x04200490           # bf16 x bf16 -> f32, M=64, N=128, K-major A and B
PV_IDESC = 0x04210490           # same with MN-major B (V is [token, feature])
# Q/K low-word walk over K=128 in eight K16 steps: +2,+2,+2 inside a 64-column
# half, then +506 (Q: +8192 B) / +1018 (K: +16384 B) to the other half, then +2,+2,+2.
QK_A_STEPS = (2,2,2,506,2,2,2); QK_B_STEPS = (2,2,2,1018,2,2,2)
# V descriptor: low word |= 0x4000000 encodes LBO = 16384 B (dim-half stride);
# +128 (2048 B = 16 tokens x 128 B) per K16 step.  P is read from TMEM at +8
# columns per K16 (16 bf16 columns packed two per 32-bit column).
PV_B_STEP = 128; PV_A_STEP = 8

# TMEM: 512 columns, lane field in bits 31..16.
S_COL=(0,256); P_COL=(64,320); O_COL=(128,384)          # per-instance column bases;
# S spans 128 columns (S_COL..S_COL+128), P overwrites S_COL+64..+128, O spans O_COL..O_COL+128
def tmem_addr(lane_origin, col): return taddr + col + (lane_origin << 16)
# 16x32bx2 fragments: a warp covers 16 lanes x two 64-column halves; thread
# (lane%16) owns row lane_origin/2 + lane%16 of the 64-row instance, thread
# (lane/16) owns column half 0 or 1; x32 = 32 consecutive columns per thread.
# Two x32 loads at col+0 and col+32 give one thread its 64 columns
# [64*col_half, 64*col_half+64) of one 64x128 tile.
```

## Exact prologue and pipeline state

```python
tid = thread_id(); warp = shuffle_index(tid // 32, lane=0); lane = tid % 32
# instruction_selection: `shfl.sync.idx.b32 ..., 0, 0x1f, 0xffffffff`; extent: one.
smem_base = cvta_shared(arena)

if warp == 0:
    if elect():
        # instruction_selection: `elect.sync _|p, 0xFFFFFFFF`; extent: warp 0.
        for name in (q_full, union_ready, kv_full x3, kv_empty x3, s_full x2,
                     p_full x2, corr_sig x2, corr_done x2, o_full x2, tmem_dealloc):
            init_mbarrier(bar(name, i), INIT_COUNT[name])
        # instruction_selection: 19 x `mbarrier.init.shared::cta.b64`; extent: one lane.
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

# Phase state (every wait is parity-based). Producer-side kv_empty starts at 1.
PHASE = dict(union_ready=0, q_full=0, kv_full=0, kv_empty_producer=1,
             s_full=[0,0], p_full=[0,0], corr_sig=[0,0], corr_done=[0,0],
             o_full=[0,0], tmem_dealloc=0)
# The KV ring cursor (stage 0..2, parity) advances once per K or V push;
# the MMA warp and the load warp walk the identical K0,K1,V0,V1 sequence.
```

## Source-order role programs

The branch order below is source-exact: `dec(12..15) -> softmax(0..7) ->
correction(8..11) -> mma(12) -> idle(13,14) -> load(15)`. Every role recomputes
its own `q_tile = blockIdx.x`, `q_head = blockIdx.y`, `kv_head = q_head`,
`query_base = q_tile*128`, `q_valid = 128 if q_tile < mb else 0`,
`kv_len = nb*128` scalars.

```python
if 12 <= warp <= 15:
    setmaxnreg(decrease, REG_OTHER)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; extent: warps 12..15,
    # emitted before any increase so the pool can grant the softmax request.

# 1. Warps 0..7: two four-warp softmax instances.
if warp <= 7:
    setmaxnreg(increase, REG_SOFTMAX)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 192`; extent: warps 0..7.
    wait(bar(union_ready), PHASE.union_ready)
    # instruction_selection: `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` retry
    # loop with suspend hint 0x989680; extent: all 256 softmax threads.
    instance = shuffle_index(warp // 4, lane=0)
    instance_row_offset = shuffle_index(instance*64, 0)
    instance_token_offset = shuffle_index(instance*64, 0)
    instance_tmem_offset = shuffle_index(instance*256, 0)
    # instruction_selection: four `shfl.sync.idx.b32`; extent: warp-uniform broadcasts.
    union_count = copy_s2r(union_count_byte(instance))
    # instruction_selection: `ld.shared.b32`; extent: one scalar.
    warp_in_instance = warp % 4; lane_origin = warp_in_instance*32
    my_row = warp_in_instance*16 + lane % 16; col_half = lane // 16
    instance_valid = clamp(q_valid - instance_token_offset, 0, 64)
    row_valid = my_row < instance_valid
    row_max = -inf; row_sum = 0.0; temperature_sum = 0.0
    score_addr = tmem_addr(lane_origin, S_COL[instance])
    p_addr = tmem_addr(lane_origin, P_COL[instance])

    for union_index in serial_range(union_count):            # rolled (`#pragma unroll 1`)
        n_block = copy_s2r(union_block_byte(instance, union_index))
        # instruction_selection: `ld.shared.b32`; extent: one scalar.
        wait(bar(s_full, instance), PHASE.s_full[instance]); PHASE.s_full[instance] ^= 1
        # instruction_selection: parity retry wait, one of two static barrier
        # addresses selected by `instance`; extent: 128 threads per instance.
        valid_cols = clamp(kv_len - n_block*128, 0, 128) if row_valid else 0
        s = reg_tile([64], "f32")
        copy_t2r(score_addr, s[0:32], shape="16x32bx2.x32")
        copy_t2r(score_addr + 32, s[32:64], shape="16x32bx2.x32")
        # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {32}, [addr], 64`;
        # extent: one 64-row x 128-column S tile, 64 f32 per thread.
        half_valid = clamp(valid_cols - col_half*64, 0, 64)
        # Source lines 683-717 (`-inf` fill of columns >= half_valid) are dead:
        # half_valid is provably 0 or 64; the export has no `.loc` for them.
        acc = (-inf, -inf)
        for j in static_range(32):                       # two accumulators alternate
            acc[j % 2] = max(acc[j % 2], max(s[2*j], s[2*j+1]))
        tile_max = max(acc[0], acc[1])
        # instruction_selection: 64 + 1 `max.f32` (no ftz), SASS fuses pairs into
        # `FMNMX3`; extent: one 64-value fragment.
        tile_max = select(half_valid <= 0, -inf, tile_max)
        tile_max = max(tile_max, shuffle_xor(tile_max, 16))
        # instruction_selection: `selp.f32`, `shfl.sync.bfly.b32 ..., 16, 31, -1`,
        # `max.f32`; extent: one row across the two column halves.
        new_max = max(tile_max, row_max)
        safe_max = select(new_max == -inf, 0.0, new_max)
        new_max_scaled = mul(safe_max, softmax_scale_log2)
        acc_scale_log2 = fma(row_max, softmax_scale_log2, -new_max_scaled)
        # instruction_selection: `max.f32`, `setp.eq.ftz.f32`+`selp.f32`, `mul.ftz.f32`,
        # `fma.rn.ftz.f32`; extent: scalars per thread.
        if acc_scale_log2 >= -8.0:
            selected_max = row_max; safe_max = select(row_max == -inf, 0.0, row_max)
            acc_scale = 1.0; temperature_acc_scale = 1.0
            new_max_scaled = mul(safe_max, softmax_scale_log2)
            # instruction_selection: `setp.ltu.ftz.f32` branch, `selp.f32`, `mul.ftz.f32`.
        else:
            selected_max = new_max
            acc_scale = select(row_max > -inf, exp2(acc_scale_log2), 1.0)
            temperature_acc_scale = select(row_max > -inf,
                                           exp2(mul(acc_scale_log2, lse_temperature_scale)), 1.0)
            # instruction_selection: two `ex2.approx.ftz.f32`, `mul.ftz.f32`,
            # `setp.ne.ftz.f32` + two `selp.f32`; extent: rescale path only.
        row_max = selected_max
        if col_half == 0:
            copy_r2s(acc_scale, scale_byte(0, instance, my_row))
            # instruction_selection: branch-guarded `st.shared.b32`; extent: 64 rows per instance.
        fence("proxy_async_shared_cta")
        # instruction_selection: `fence.proxy.async.shared::cta`; extent: every softmax thread.
        arrive(bar(corr_sig, instance))
        # instruction_selection: `mbarrier.arrive.release.cta.shared::cta.b64`; extent:
        # 128 arrivals per instance (barrier count 128).
        score_bias = select(valid_cols > 0, -new_max_scaled, -inf)
        # instruction_selection: `selp.f32`; extent: one scalar.
        block_sum = 0.0; block_temperature_sum = 0.0
        if return_temperature_lse != 0:
            tau_scale = mul(softmax_scale_log2, lse_temperature_scale)
            tau_bias = mul(score_bias, lse_temperature_scale)
            for j in static_range(32):
                s[2*j:2*j+2] = fma(s[2*j:2*j+2], (tau_scale,)*2, (tau_bias,)*2, lanes=2)
            # instruction_selection: 32 `fma.rn.ftz.f32x2`; extent: 64 values.
            for j in static_range(64): s[j] = exp2(s[j])
            # instruction_selection: 64 `ex2.approx.ftz.f32`.
            acc2 = (0.0, 0.0)
            for j in static_range(32): acc2 = add(acc2, s[2*j:2*j+2], lanes=2)
            half = add(acc2[0], acc2[1])
            block_temperature_sum = add(half, shuffle_xor(half, 16))
            # instruction_selection: 32 `add.f32x2` (no ftz), `add.ftz.f32`,
            # `shfl.sync.bfly.b32 16`, `add.ftz.f32`; extent: one row.
            copy_t2r(score_addr, s[0:32], shape="16x32bx2.x32")
            copy_t2r(score_addr + 32, s[32:64], shape="16x32bx2.x32")
            # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`;
            # S is re-read because the first copy was consumed by the temperature sum.
            # Source lines 794-829 (`-inf` fill) are dead as above.
        for j in static_range(32):
            s[2*j:2*j+2] = fma(s[2*j:2*j+2], (softmax_scale_log2,)*2, (score_bias,)*2, lanes=2)
        # instruction_selection: 32 `fma.rn.ftz.f32x2`; extent: 64 values.
        for j in static_range(64): s[j] = exp2(s[j])
        # instruction_selection: 64 `ex2.approx.ftz.f32`.
        acc2 = (0.0, 0.0)
        for j in static_range(32): acc2 = add(acc2, s[2*j:2*j+2], lanes=2)
        half = add(acc2[0], acc2[1]); block_sum = add(half, shuffle_xor(half, 16))
        # instruction_selection: 32 `add.f32x2`, `add.ftz.f32`, `shfl.sync.bfly.b32 16`,
        # `add.ftz.f32`; extent: one row sum.
        packed_p = reg_tile([32], "b32")
        for j in static_range(32): packed_p[j] = cast(pack(s[2*j], s[2*j+1]), "bf16x2")
        # instruction_selection: 32 `cvt.rn.bf16x2.f32`; extent: 64 probabilities.
        copy_r2t(packed_p[0:16], p_addr, shape="16x32bx2.x16")
        copy_r2t(packed_p[16:32], p_addr + 16, shape="16x32bx2.x16")
        # instruction_selection: two `tcgen05.st.sync.aligned.16x32bx2.x16.b32 [addr], 32, {16}`;
        # extent: P columns P_COL..P_COL+64 of this instance, written in place over S.
        # (The temperature and plain branches each emit their own copy of the
        # fma/exp2/sum/cvt/st sequence: 3 x 32 fma, 3 x 64 ex2, 4 x16 stores total.)
        tmem_wait_st()
        # instruction_selection: `tcgen05.wait::st.sync.aligned`.
        arrive(bar(p_full, instance))
        # instruction_selection: `mbarrier.arrive.release.cta.shared::cta.b64`; 128 of the
        # 256 expected arrivals (the other 128 come from the correction warps).
        wait(bar(corr_done, instance), PHASE.corr_done[instance]); PHASE.corr_done[instance] ^= 1
        # instruction_selection: parity retry wait; extent: 128 threads.
        row_sum = fma(row_sum, acc_scale, block_sum)
        if return_temperature_lse != 0:
            temperature_sum = fma(temperature_sum, temperature_acc_scale, block_temperature_sum)
        # instruction_selection: `fma.rn.ftz.f32` (x2, second predicated); extent: scalars.

    if col_half == 0:
        copy_r2s(row_sum, scale_byte(1, instance, my_row))
        copy_r2s(row_max, scale_byte(2, instance, my_row))
        copy_r2s(temperature_sum, scale_byte(3, instance, my_row))
        # instruction_selection: three branch-guarded `st.shared.b32`; extent: 64 rows.
    fence("proxy_async_shared_cta")
    arrive(bar(corr_sig, instance))
    # instruction_selection: `fence.proxy.async.shared::cta` then
    # `mbarrier.arrive.release.cta.shared::cta.b64`; the final corr_sig generation.

# 2. Warps 8..11: correction and epilogue.
if 8 <= warp <= 11:
    setmaxnreg(decrease, REG_CORRECTION)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 80`; extent: warps 8..11.
    wait(bar(union_ready), PHASE.union_ready)
    # instruction_selection: parity retry wait; extent: 128 threads.
    count0, count1 = copy_s2r(union_count_byte(0), width=2)
    max_union_count = max(count0, count1)
    # instruction_selection: `ld.shared.v2.b32` + `max.s32`; extent: two scalars.
    warp_in_role = warp - 8; lane_origin = warp_in_role*32
    my_row = warp_in_role*16 + lane % 16; col_half = lane // 16
    row_addr = lane_origin << 16
    arrive(bar(p_full, 0)); arrive(bar(p_full, 1))
    # instruction_selection: two `mbarrier.arrive.release.cta.shared::cta.b64`; the
    # correction half of the first p_full generation needs no O rescale.
    wait(bar(corr_sig, 0), PHASE.corr_sig[0]); PHASE.corr_sig[0] ^= 1; arrive(bar(corr_done, 0))
    wait(bar(corr_sig, 1), PHASE.corr_sig[1]); PHASE.corr_sig[1] ^= 1; arrive(bar(corr_done, 1))
    # instruction_selection: parity wait then ordinary arrive, per instance; block 0
    # never rescales.
    for union_index in serial_range(1, max_union_count):     # rolled
        for instance in static_range(2):
            count = copy_s2r(union_count_byte(instance))
            # instruction_selection: `ld.shared.b32`; re-read every iteration.
            if count > union_index:
                wait(bar(corr_sig, instance), PHASE.corr_sig[instance]); PHASE.corr_sig[instance] ^= 1
                # instruction_selection: parity retry wait; extent: 128 threads.
                acc_scale = copy_s2r(scale_byte(0, instance, my_row))
                # instruction_selection: `ld.shared.b32`; extent: one row scale.
                if any(acc_scale < 1.0):
                    # instruction_selection: `setp.lt.ftz.f32` + `vote.sync.any.pred`; one warp vote.
                    o = reg_tile([64], "f32")
                    copy_t2r(tmem_addr(lane_origin, O_COL[instance]), o[0:32], shape="16x32bx2.x32")
                    copy_t2r(tmem_addr(lane_origin, O_COL[instance]) + 32, o[32:64], shape="16x32bx2.x32")
                    # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`.
                    for j in static_range(32): o[2*j:2*j+2] = mul(o[2*j:2*j+2], (acc_scale,)*2, lanes=2)
                    # instruction_selection: 32 `mul.rn.ftz.f32x2`; extent: 64 values.
                    copy_r2t(o, tmem_addr(lane_origin, O_COL[instance]), shape="16x32bx2.x64")
                    # instruction_selection: one `tcgen05.st.sync.aligned.16x32bx2.x64.b32 [addr], 64, {64}`.
                    tmem_wait_st()
                    # instruction_selection: `tcgen05.wait::st.sync.aligned`.
                arrive(bar(p_full, instance)); arrive(bar(corr_done, instance))
                # instruction_selection: two `mbarrier.arrive.release.cta.shared::cta.b64`.
    wait(bar(o_full, 0), PHASE.o_full[0]); wait(bar(o_full, 1), PHASE.o_full[1])
    wait(bar(corr_sig, 0), PHASE.corr_sig[0]); wait(bar(corr_sig, 1), PHASE.corr_sig[1])
    # instruction_selection: four parity retry waits; extent: both O accumulators and
    # both final statistics generations.
    fence("tcgen05_after_thread_sync")
    # instruction_selection: `tcgen05.fence::after_thread_sync`.
    for instance in static_range(2):
        final_sum = copy_s2r(scale_byte(1, instance, my_row))
        final_max = copy_s2r(scale_byte(2, instance, my_row))
        final_temperature_sum = copy_s2r(scale_byte(3, instance, my_row))
        # instruction_selection: three `ld.shared.b32`.
        inv_sum = select(final_sum > 0 and final_sum == final_sum, rcp(final_sum), 0.0)
        # instruction_selection: `rcp.approx.ftz.f32`, `setp.gt.ftz.f32` x2, `selp.f32`.
        instance_valid = clamp(q_valid - 64*instance, 0, 64)
        query = query_base + 64*instance + my_row
        output_row = (query*num_q_heads + q_head)*128
        o = reg_tile([64], "f32")
        copy_t2r(tmem_addr(lane_origin, O_COL[instance]), o[0:32], shape="16x32bx2.x32")
        copy_t2r(tmem_addr(lane_origin, O_COL[instance]) + 32, o[32:64], shape="16x32bx2.x32")
        # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`.
        if my_row < instance_valid:
            for chunk in static_range(8):
                for j in static_range(4):
                    o[8*chunk+2*j : 8*chunk+2*j+2] = mul(o[8*chunk+2*j : 8*chunk+2*j+2], (inv_sum,)*2, lanes=2)
                words = [cast(pack(o[8*chunk+2*j], o[8*chunk+2*j+1]), "bf16x2") for j in static_range(4)]
                copy_r2g(words, out[output_row + 64*col_half + 8*chunk : +8])
                # instruction_selection: four `mul.rn.ftz.f32x2`, four `cvt.rn.bf16x2.f32`,
                # one `st.global.v4.b32`; extent: 8 chunks x 2 instances = 16 stores/thread.
            if col_half == 0:
                stat_idx = query*num_q_heads + q_head
                if return_softmax_lse != 0:
                    lse_value = select(final_sum > 0,
                                       fma(mul(final_max, softmax_scale_log2), LN2, mul(log2(final_sum), LN2)),
                                       -inf)
                    copy_r2g(lse_value, lse[stat_idx])
                    # instruction_selection: `lg2.approx.ftz.f32`, `mul.ftz.f32` x2,
                    # `fma.rn.ftz.f32` with the ln2 immediate, `setp.gt.ftz.f32`, `selp.f32`,
                    # `st.global.b32`; extent: one scalar per live row.
                if return_temperature_lse != 0:
                    t_value = select(final_temperature_sum > 0,
                                     fma(lse_temperature_scale,
                                         mul(mul(final_max, softmax_scale_log2), LN2),
                                         mul(log2(final_temperature_sum), LN2)),
                                     -inf)
                    copy_r2g(t_value, temperature_lse[stat_idx])
                    # instruction_selection: `lg2.approx.ftz.f32`, `mul.ftz.f32` x3
                    # (final_max*scale, *LN2 immediate, log2*LN2 immediate), one
                    # `fma.rn.ftz.f32` whose multiplier is `lse_temperature_scale`,
                    # `setp.leu.ftz.f32` branching over an unconditional `mov.b32 -inf`,
                    # `st.global.b32`; extent: one scalar per live row.
    tmem_wait_ld()
    fence("tcgen05_before_thread_sync")
    arrive(bar(tmem_dealloc))
    # instruction_selection: `tcgen05.wait::ld.sync.aligned`, `tcgen05.fence::before_thread_sync`,
    # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 128 arrivals.

# 3. Warp 12: MMA issuer and TMEM owner.
if warp == 12:
    wait(bar(union_ready), PHASE.union_ready)
    count0, count1 = copy_s2r(union_count_byte(0), width=2); max_union_count = max(count0, count1)
    # instruction_selection: parity wait, `ld.shared.v2.b32`, `max.s32`.
    kv = ring(stages=3, stage=0, parity=0)
    wait(bar(q_full), PHASE.q_full)
    # instruction_selection: parity retry wait; extent: one Q generation.
    first_pv = [True, True]
    for union_index in serial_range(max_union_count):        # rolled
        count = [None, None]
        for instance in static_range(2):                     # S_i = Q_i K^T
            count[instance] = copy_s2r(union_count_byte(instance))
            # instruction_selection: `ld.shared.b32`; one count read per instance per
            # iteration, reused by the PV guard and the last-block test below.
            if count[instance] > union_index:
                # instruction_selection: `setp` + branch on the count just read.
                k_stage = kv.stage; kv.advance()
                wait(bar(kv_full, k_stage), kv.parity_at(k_stage))
                # instruction_selection: parity retry wait; extent: one K stage.
                a_lo = shuffle_index(desc_lo(q_byte(instance, 0)), 0)
                b_lo = shuffle_index(desc_lo(kv_byte(k_stage, 0, 0)), 0)
                # instruction_selection: two `shfl.sync.idx.b32` uniform broadcasts.
                leader = elect()
                for k16 in static_range(8):
                    gemm(tmem_addr(0, S_COL[instance]), desc64(a_lo, DESC_HI), desc64(b_lo, DESC_HI),
                         QK_IDESC, accumulate=(k16 != 0), predicate=leader)
                    a_lo += QK_A_STEPS[k16]; b_lo += QK_B_STEPS[k16]
                # instruction_selection: one `elect.sync`, `setp.ne.b32` x2, `mov.b32` immediates,
                # eight `mov.b64 {lo,hi}` pairs and eight `@leader tcgen05.mma.cta_group::1.kind::f16
                # [d], da, db, id, p` with `add.u32` low-word steps; extent: K=128 in eight K16 issues.
                tmem_commit(bar(s_full, instance), elect=True)
                tmem_commit(bar(kv_empty, k_stage), elect=True)
                # instruction_selection: two elected
                # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`.
        for instance in static_range(2):                     # O_i += P_i V
            if count[instance] > union_index:                # no second shared read
                v_stage = kv.stage; kv.advance()
                wait(bar(kv_full, v_stage), kv.parity_at(v_stage))
                wait(bar(p_full, instance), PHASE.p_full[instance]); PHASE.p_full[instance] ^= 1
                # instruction_selection: two parity retry waits; extent: one V stage, one P.
                b_lo = shuffle_index(desc_lo(kv_byte(v_stage, 0, 0)) | 0x4000000, 0)
                # instruction_selection: `shfl.sync.idx.b32`; LBO bit set for the MN-major V.
                ta = tmem_addr(0, P_COL[instance]); leader = elect()
                for k16 in static_range(8):
                    gemm(tmem_addr(0, O_COL[instance]), tmem_operand(ta), desc64(b_lo, DESC_HI),
                         PV_IDESC, accumulate=(k16 != 0 or not first_pv[instance]), predicate=leader)
                    ta += PV_A_STEP; b_lo += PV_B_STEP
                # instruction_selection: one `elect.sync`, eight `@leader tcgen05.mma.cta_group::1.kind::f16
                # [d], [ta], db, id, p` with `add.u32 ta, ta, 8` and `add.u32 blo, blo, 128`;
                # the first issue of the instance's first block passes enable_input_d = 0.
                first_pv[instance] = False
                if union_index + 1 == count[instance]:
                    tmem_commit(bar(o_full, instance), elect=True)
                    # instruction_selection: elected `tcgen05.commit...mbarrier::arrive::one`; last block only.
                tmem_commit(bar(kv_empty, v_stage), elect=True)
                # instruction_selection: elected `tcgen05.commit...mbarrier::arrive::one`; V release.
    wait(bar(tmem_dealloc), PHASE.tmem_dealloc)
    taddr_again = copy_s2r(OFF.tmem_mailbox, volatile=True)
    tmem_dealloc(taddr_again, columns=512)
    # instruction_selection: parity retry wait, `ld.volatile.shared.b32`,
    # `tcgen05.dealloc.cta_group::1.sync.aligned.b32 addr, 512`; extent: warp 12.

# 4. Warps 13, 14: idle after the register decrease.
if 13 <= warp <= 14:
    pass

# 5. Warp 15: Q load, mask compaction, K/V ring producer.
if warp == 15:
    if elect():
        arrive_expect_tx(bar(q_full), 32768)
        copy_g2s_tma(Qmap, (0, query_base, q_head, 0), q_byte(0, 0), bar(q_full))
        copy_g2s_tma(Qmap, (0, query_base + 64, q_head, 0), q_byte(1, 0), bar(q_full))
        # instruction_selection: `elect.sync`, `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`,
        # two `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes`;
        # extent: 2 x 16384 bytes (both dim halves of 64 rows per issue).
    q_block = query_base // 128; mask_base = (q_head*mb + q_block)*nb; selected_count = 0
    for block_base in serial_range(0, nb, 32):                # rolled
        n_block = block_base + lane
        selected = (copy_g2r(block_mask[mask_base + n_block]) != 0) if n_block < nb else 0
        # instruction_selection: branch-guarded `ld.global.nc.b8` + `setp.ne.b16`; extent: one byte per lane.
        vote = ballot(selected != 0)
        slot = selected_count + popc(vote & ((1 << lane) - 1))
        # instruction_selection: `vote.sync.ballot.b32`, `popc.b32` x2, `shl.b32`/`add.s32`.
        if selected:
            copy_r2s(n_block, union_block_byte(0, slot)); copy_r2s(n_block, union_block_byte(1, slot))
            # instruction_selection: two branch-guarded `st.shared.b32`; both instances get the list.
        selected_count += popc(vote)
    if selected_count == 0:
        if lane == 0:
            copy_r2s(0, union_block_byte(0, 0)); copy_r2s(0, union_block_byte(1, 0))
            # instruction_selection: two `st.shared.b32 [addr], 0`; extent: lane 0.
        selected_count = 1
    if lane < 2:
        copy_r2s(selected_count, union_count_byte(lane))
        # instruction_selection: branch-guarded `st.shared.b32`; extent: lanes 0 and 1.
    named_barrier(8, threads=32)
    fence("proxy_async_shared_cta")
    arrive(bar(union_ready))
    # instruction_selection: `barrier.sync 8, 32`, `fence.proxy.async.shared::cta`,
    # `mbarrier.arrive.release.cta.shared::cta.b64` by all 32 lanes (count 32).
    load = ring(stages=3, stage=0, parity=1)
    count0, count1 = copy_s2r(union_count_byte(0), width=2); max_union_count = max(count0, count1)
    # instruction_selection: `ld.shared.v2.b32`, `max.s32`.
    for union_index in serial_range(max_union_count):        # rolled
        for map in (Kmap, Vmap):                             # K for both instances, then V
            for instance in static_range(2):
                if copy_s2r(union_count_byte(instance)) > union_index:
                    n_block = copy_s2r(union_block_byte(instance, union_index))
                    # instruction_selection: two `ld.shared.b32`.
                    token0 = n_block*128; token1 = token0 + 64
                    wait(bar(kv_empty, load.stage), load.parity)
                    # instruction_selection: parity retry wait by all 32 lanes; producer parity starts at 1.
                    if elect():
                        arrive_expect_tx(bar(kv_full, load.stage), 32768)
                        copy_g2s_tma(map, (0, token0, 0, kv_head), kv_byte(load.stage, 0, 0), bar(kv_full, load.stage))
                        copy_g2s_tma(map, (0, token1, 0, kv_head), kv_byte(load.stage, 1, 0), bar(kv_full, load.stage))
                        copy_g2s_tma(map, (0, token0, 1, kv_head), kv_byte(load.stage, 0, 1), bar(kv_full, load.stage))
                        copy_g2s_tma(map, (0, token1, 1, kv_head), kv_byte(load.stage, 1, 1), bar(kv_full, load.stage))
                        # instruction_selection: `elect.sync`, `mbarrier.arrive.expect_tx...` 32768,
                        # four `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes`
                        # of 8192 bytes; extent: one 128-token K or V tile per push, 16 static sites.
                    load.advance()
```

## Bidirectional source / sketch / PTX map

| semantic edge | source lines | sketch section | exported `.loc` / opcode evidence |
| --- | --- | --- | --- |
| helpers: elect, mbarrier init/wait/arrive/expect_tx, commit, TMA, warp-uniform, approx math | 47-260, 445-483 | primitive vocabulary | `.loc 56/63/107/204/211/218/466/481/243/250/257` inlined at every call site |
| f32x2 helpers (fma/mul in place, block sum), max accumulators | 278-372 | softmax/correction/epilogue bodies | `.loc 287/289` (64 `max.f32`), `364` (96 `fma.rn.ftz.f32x2`), `370` (128 `mul.rn.ftz.f32x2`), `354` (96 `add.f32x2`) |
| dead helpers (cluster waits, tokens, `mma_ss/ts_step`, `tmem_ld_x32`, `warp_reduce_*`, `ex2_emulation`, `softmax_frag_exp2_cast`, unused f32x2 wrappers, `make_smem_desc`, `tcgen05_commit`) | 67-135, 138-193, 222-238, 262-275, 294-343, 373-440, 450-456, 470-475 | omitted | no `.loc` |
| ABI and smem carve-up | 487-515 | fixed storage | `cvta.to.global` x7, `.extern .shared .align 1024` |
| mbarrier init, warp sync, TMEM alloc, CTA sync, TMEM base | 520-587 | exact prologue | `.loc 521-553` (19 inits + fence), `557` `bar.warp.sync`, `563/564` alloc/relinquish, `567` `bar.sync 0`, `568` fence, `581` `ld.volatile.shared` |
| register redistribution | 591-597, 922 | role entries | `.loc 592` dec 48, `597` inc 192, `922` dec 80 |
| softmax scalars, union_ready, uniform broadcasts | 599-642 | role 1 head | `.loc 614` wait, `616-619` `shfl.sync.idx`, `620` `ld.shared` |
| softmax per-block: wait, S load, (dead mask), max tree, rescale decision, acc_scale publish | 644-762 | role 1 loop | `.loc 645/647/650/670/675/720-734/739-750/754/756/758/760`; no `.loc` for 683-717 |
| temperature branch (fma, exp2, sum, S re-read, fma, exp2, sum, cvt, P store) | 766-859 | role 1 temperature branch | `.loc 771/774/777-782/788/793/834/837/840-845/849/855/859`; no `.loc` for 794-829 |
| plain branch (fma, exp2, sum, cvt, P store) | 860-891 | role 1 plain branch | `.loc 865/868/871-876/880/886/890` |
| P publish, corr_done, running sums, final stats | 892-918 | role 1 tail | `.loc 892/894/895/898/899/902/903/908-916` |
| correction prologue and rescale loop | 921-1026 | role 2 loop | `.loc 939/941/952-961/964-969/976/981/985/989/990/992/993/995-1024` |
| epilogue: waits, stats, normalize, store, LSE | 1027-1112 | role 2 epilogue | `.loc 1028-1037/1043-1047/1065/1070/1080/1087-1091/1098/1099/1103/1104/1109-1111` |
| MMA warp: waits, QK chains, PV chains, dealloc | 1115-1394 | role 3 | `.loc 1118/1120/1128/1136/1142-1144/1198-1200/1202-1210/1264-1266/1273-1276/1321-1326/1333-1336/1381-1386/1390-1393` |
| idle warps | 1397-1399 | role 4 | none |
| load warp: Q TMA, ballot compaction, union publish, K/V ring | 1401-1507 | role 5 | `.loc 1417-1420/1426-1457/1459/1469-1484/1489-1504` |

Reading in reverse, every TMA, TMEM load/store, MMA, commit, mbarrier, named
barrier, warp vote, shuffle, transcendental, packed f32x2, conversion, vector
global store and register-budget instruction in the export maps to a concrete
occurrence above. Address arithmetic maps to the integer functions of "Fixed
storage and integer maps", never to a layout.

## TIRx, correctness and performance contract

The public module exports `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`,
`get_kernel`, `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, and
`run_bench`. Device code imports only `tirx_kernels.kern as K`; low-level
instruction families use `K.ptx[...]` / `K.ptx.*`. There is no inline CUDA
function call, tile primitive, first-class layout or multidimensional SMEM
allocation. The kernel is compiled once, shape-generic, through nvcc 13.2
(`compile_kernel(..., cuda_compile_mode="nvcc")`) so TIRx and the upstream
`nvcc -cubin` build share PTX ISA 9.2.

The frozen correctness matrix has 9 rows: the upstream test shape, one selected
block without LSE, all 64 blocks, an nb=40 ballot tail with variable per-row
counts, N=16384 with 16 heads, an amplified-Q rescale case, a single CTA, an
empty block row, and the temperature-LSE branch. TIRx and the pinned upstream
kernel share immutable inputs and independent outputs; `out` and `lse` must be
bitwise identical between them, a second TIRx launch must reproduce the first
bit for bit, no-LSE sentinels stay untouched, NaN is forbidden, and both are
checked against one FP32 dense oracle with tolerances frozen from measurement.

Performance truth is exclusively `python -m tirx_kernels.bench_suite` over the
frozen 6-row benchmark matrix with the pinned FlashInfer reference. Each row
must have five finite positive Proton samples for both implementations and
strict `mean(flashinfer)/mean(tirx) > 0.99`.
