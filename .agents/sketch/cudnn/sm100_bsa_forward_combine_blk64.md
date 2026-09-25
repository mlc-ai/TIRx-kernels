<!--
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This design sketch documents a TIRx port of cuDNN Frontend's
block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_combine.py.
-->

# cudnn_sm100_bsa_forward_combine_blk64: split-KV reduction sketch

This non-executable canonical sketch describes the direct CuTeDSL combine ABI,
physical storage, thread maps, pipeline, synchronization, and instruction
selection of
[`block_sparse_attention_forward_combine_sm100_blk64.py`](../../../tirx_kernels/cudnn/bsa/block_sparse_attention_forward_combine_sm100_blk64.py).
The module is the authoritative pure-`K` implementation.

Pinned source: `bsa_fwd_combine.py` at cuDNN Frontend commit
`aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5`, SHA256
`ff5a6e8205453a2d6b2b04b3a2ee30b24992906f97fa1f601a3e6fbd5e0c3169`.
Fresh line-info exports for logical split counts 2, 3, 8, and 256 are under
`.porting/block_sparse_attention_forward_combine_sm100_blk64/source_export/`.
The export manifest records exact PTX hashes and anchors.

## Scope and ABI

The port covers `BlockSparseAttnForwardCombine` (`:22-594`) with the production
constructor values fixed by its BSA host: output BF16, partial FP32,
`head_dim=128`, `tile_m=16`, `k_block_size=64`, `num_threads=128`, and
`stages=4`.  Its only compile-time axis is `log_max_splits=ceil(log2(S))`, with
`2 <= S <= 256`; the physical LSE extent is `max(8, 2**log_max_splits)`.

The direct, TE-native public ABI is:

```text
O_partial   f32 [B, S*H, SQ, 128]  read-only
LSE_partial f32 [B, S*H, SQ]       read-only
O           bf16[B, SQ, H, 128]    write-only
LSE         f32 [B, SQ, H]         write-only
```

The source internally creates `(SQ,D,S,H,B)` / `(SQ,S,H,B)` views
(`:181-200`) but does not transpose storage.  In particular the public output
is BSHD, not the BHSD copy performed later by the surrounding wrapper.  The
port takes runtime `B,H,SQ,S` and the three FastDivmod operands for `SQ` while
specializing only the physical split extent.

Out of scope: the source's varlen buffers, `seqused`, dynamic split pointer,
virtual-batch remap, semaphore reset, uneven head-dim predicate, and one-split
early-exit variant.  The direct BSA invocation uses dense batch, fixed
`head_dim=128`, always supplies LSE output, and launches combine only for
`S>1`.  There is no weight transpose, wrapper output transpose, TMA, TMEM,
cluster, PDL, or warp-specialized role.

## Launch, roles, and ownership

```python
launch(grid=(ceil_div(SQ*H, 16), 2, B),
       block=(128, 1, 1), arch="sm_100a",
       dynamic_smem_bytes=storage_bytes(PHYS_SPLITS))
# instruction_selection: `.reqntid 128,1,1` and
#   `.extern .shared .align 1024`; extent: one ordinary CTA.
```

All four warps run the same program; the labels below are temporal phases, not
warp roles.  CTA `(row_tile, dim_tile, batch)` owns 16 flattened `(head,query)`
rows and one 64-column half of D.  Flattening is `flat=head*SQ+query`, so
`head,query = fast_divmod(flat,SQ)`.  The direct stores then linearize
`((batch*SQ+query)*H+head)*128 + 64*dim_tile + col`.

Each thread owns:

```python
lse_stage_row   = tid & 15
lse_stage_split = (tid >> 4) + 8*slot
stats_row       = tid >> 3
stats_split     = (tid & 7) + 8*slot
row0            = tid >> 4
row1            = row0 + 8
col0            = (tid & 15) * 4
```

Thus LSE stage and LSE readback are transposed views: 16 row lanes x 8 split
lanes becomes 8 reduction lanes x 16 rows.  Every thread handles two rows and
four adjacent FP32/BF16 values for partial-O and output.  Define
`SLOTS=PHYS_SPLITS/8`; its exact extent is 1/2/4/8/16/32 for physical
8/16/32/64/128/256 respectively.

## Rank-one shared arena

The implementation allocates one rank-one dynamic `u8` arena aligned to 1024
and creates only flat scalar aliases with explicit byte offsets.  It does not
materialize a layout object or multidimensional SMEM.

| region | byte offset | bytes |
| --- | ---: | ---: |
| `sLSE` | 0 | `PHYS_SPLITS*16*4` |
| `sMaxValidSplit` | `align_up(end(sLSE),128)` | 64 |
| padding | end max-valid to next 128 | at most 64 |
| `sO[4][16][64]` | `align_up(end(max-valid),128)` | 16384 |

For `PHYS_SPLITS=8`: `sLSE=[0,512)`, `sMax=[512,576)`, padding
`[576,640)`, `sO=[640,17024)`, total 17024 bytes.  For 256 splits the same
formula yields 16384 bytes of LSE and total 32896 bytes.

The source's `S<4,0,4>` composed swizzle over ordered `(8,16)` resolves to:

```python
def lse_index(split, row):
    linear = 16 * split + row
    return linear ^ ((linear >> 4) & 15)

def o_index(stage, row, col):
    return stage * 1024 + row * 64 + col
```

## Complete execution sketch

```python
tid = thread_id()                                  # source :275
row_tile, dim_tile, batch = cta_id()               # source :276
# instruction_selection: `%tid.x`, `%ctaid.{x,y,z}`; extent: scalar

# ---- 1. LSE GMEM -> SMEM (:323-365) -------------------------------
for slot in unroll(PHYS_SPLITS // 8):
    split = (tid >> 4) + 8*slot
    row = tid & 15
    flat = 16*row_tile + row
    if flat < SQ*H and split < S:
        head, query = fast_divmod(flat, SQ)
        src = ((batch*(S*H) + split*H + head)*SQ + query)
        cp_async_ca(sLSE[lse_index(split,row)], LSE_partial[src], 4, 4)
        # instruction_selection: `cp.async.ca.shared.global ...,4,4`
        #   (`.loc 1 351`); extent: one scalar per physical slot.
    else:
        st_shared_b32(sLSE[lse_index(split,row)], NEG_INF_BITS)
        # instruction_selection: `st.shared.b32` (`:357`/`:364`);
        #   extent: masked slot, including all invalid tail rows.
commit_group()
# instruction_selection: `cp.async.commit_group` (`.loc 1 365`); extent: one.

# ---- 2. Prime three of four O stages (:367-418, loader :568-594) ---
for stage in unroll(3):
    if stage < S:
        for row in (row0,row1):
            flat = 16*row_tile + row
            if flat < SQ*H:
                head, query = fast_divmod(flat,SQ)
                src = (((batch*(S*H) + stage*H + head)*SQ + query)*128
                       + dim_tile*64 + col0)
                cp_async_cg(sO[o_index(stage,row,col0)], O_partial[src],16,16)
                # instruction_selection: `cp.async.cg.shared.global ...,16,16`
                #   (`.loc 1 590`); extent: two FP32x4 rows per thread/stage.
    commit_group()
    # instruction_selection: `cp.async.commit_group` (`.loc 1 418`);
    #   extent: three initial O generations, including empty generations.

wait_group(3)
# instruction_selection: `cp.async.wait_group 3` (`.loc 1 425`); extent: one.
barrier()
# instruction_selection: `bar.sync 0` (`.loc 1 426`); extent: CTA.

# ---- 3/4. Transposed LSE stats and normalized weights (:429-477) ---
for slot in unroll(PHYS_SPLITS // 8):
    lse_reg[slot] = ld_shared_f32(
        sLSE[lse_index((tid & 7) + 8*slot, tid >> 3)])
# instruction_selection: scalar `ld.shared.b32` (`.loc 1 431`);
#   extent: one per physical slot.

if SLOTS == 1:
    local_max = lse_reg[0]
    # instruction_selection: no local max instruction; extent: one slot.
else:
    local_max = generated_max_nan_tree(lse_reg[0:SLOTS])
    # instruction_selection: `SLOTS/2 == PHYS_SPLITS/16` `max.NaN.f32`
    #   sites (`.loc 1 447`); extent: 1/2/4/8/16 sites for physical
    #   16/32/64/128/256.  Physical 16 has a two-input root; larger generated
    #   trees use the exported mixed/three-input reduction forms.
peer4 = shfl_bfly(local_max, 4)
pick4 = peer4 if isnan(peer4) or local_max <= peer4 else local_max
# instruction_selection: `shfl.sync.bfly.b32` mask 4, `setp.le.f32`,
#   `setp.nan.f32`, and two `selp.f32` (`.loc 1 446`); extent: first
#   cross-lane step of each 8-lane row group.
peer2 = shfl_bfly(pick4, 2)
max2 = max_f32(pick4, peer2)
peer1 = shfl_bfly(max2, 1)
lse_max = max_f32(max2, peer1)
# instruction_selection: two `shfl.sync.bfly.b32` plus two `max.f32`
#   (`.loc 1 446`); extent: masks 2 then 1 for one 8-lane row group.

last = -1
for slot in unroll(PHYS_SPLITS // 8):
    split = (tid & 7) + 8*slot
    last = split if lse_reg[slot] != -inf else last
    # instruction_selection: `setp.neu.f32` plus `selp.b32`
    #   (`.loc 1 453-454`); extent: 1 update/thread for physical 8 and
    #   32 updates/thread for physical 256.
max_valid = butterfly_max_s32(last, masks=(4,2,1), width=8)
# instruction_selection: three `shfl.sync.bfly.b32` plus three `max.s32`
#   (`.loc 1 455`); extent: masks 4,2,1 for one 8-lane row group.

safe_max = 0.0 if lse_max == -inf else lse_max
# instruction_selection: `setp.eq.f32` plus helper-located `selp.f32`
#   (`.loc 1 457`); extent: one safe-max selection per thread.
safe_max_log2 = safe_max * LOG2_E
# instruction_selection: one loop-invariant `mul.f32` (`.loc 1 461`);
#   extent: once per thread for every supported physical extent.
for slot in unroll(PHYS_SPLITS // 8):
    scale[slot] = exp2(lse_reg[slot]*LOG2_E - safe_max_log2)
    local_sum += scale[slot]
    # instruction_selection: one `mul.f32`, `sub.f32`,
    #   `ex2.approx.ftz.f32`, `add.f32` (`.loc 1 461-462`);
    #   extent: one physical slot.
sum_exp = butterfly_sum(local_sum, masks=(4,2,1), width=8)
# instruction_selection: three shuffles plus `add.f32` (`.loc 1 464`).
final_lse = lg2(sum_exp)*LN2 + lse_max
# instruction_selection: `lg2.approx.ftz.f32`; `fma.rn.f32`
#   (`.loc 1 465`); extent: one row group.
inv = 0.0 if sum_exp == 0.0 or isnan(sum_exp) else rcp_rn(sum_exp)
# instruction_selection: unordered `setp.equ.f32`, `rcp.rn.f32`, `selp.f32`
#   (`.loc 1 467`); extent: one row group.
if SLOTS == 1:
    scale[0] = mul_f32(scale[0], inv)
    # instruction_selection: one scalar `mul.f32` (`.loc 1 468`).
else:
    for pair in unroll(SLOTS // 2):
        scale[2*pair:2*pair+2] = mul_f32x2(
            scale[2*pair:2*pair+2], broadcast(inv))
        # instruction_selection: `mul.f32x2` (`.loc 1 468`); extent:
        #   `SLOTS/2 == PHYS_SPLITS/16` packed pairs: 1/2/4/8/16 for physical
        #   16/32/64/128/256, covering every slot exactly once.
for slot in unroll(PHYS_SPLITS // 8):
    st_shared_f32(sLSE[lse_index((tid&7)+8*slot,tid>>3)], scale[slot])
# instruction_selection: scalar `st.shared.b32` (`.loc 1 470`);
#   extent: `SLOTS=1/2/4/8/16/32` stores per thread for physical
#   8/16/32/64/128/256 respectively.
if (tid & 7) == 0:
    st_shared_b32(sMaxValidSplit[tid >> 3], bitcast_b32(max_valid))
    # instruction_selection: `st.shared.b32` (`.loc 1 477`); one writer/row.
    if dim_tile == 0 and 16*row_tile + (tid >> 3) < SQ*H:
        head, query = fast_divmod(16*row_tile + (tid >> 3), SQ)
        st_global_f32(LSE[(batch*SQ + query)*H + head], final_lse)
        # instruction_selection: `st.global.b32` (`.loc 1 499`);
        #   extent: one direct BSH row.

barrier()
# instruction_selection: `bar.sync 0` (`.loc 1 505`); extent: CTA.  This
#   publishes both weights and max-valid across the transposed thread maps.

# ---- 5/6. Four-stage O pipeline and weighted accumulation (:508-543) ----
thread_max = max(ld_shared_i32(sMaxValidSplit[row0]),
                 ld_shared_i32(sMaxValidSplit[row1]))
# instruction_selection: two `ld.shared.b32` plus one `max.s32`
#   (`.loc 1 508/510`); extent: once per thread before the live-split loop.
acc0 = fp32x4(0); acc1 = fp32x4(0)
load_stage = 3; compute_stage = 0
for split in serial(thread_max + 1, unroll=4):
    weight0 = ld_shared_f32(sLSE[lse_index(split,row0)])
    weight1 = ld_shared_f32(sLSE[lse_index(split,row1)])
    # instruction_selection: two scalar `ld.shared.b32` (`.loc 1 524`);
    #   extent: once per live split iteration, one weight per owned row.
    next_split = split + 3
    if next_split <= thread_max:
        for row in (row0,row1):
            flat = 16*row_tile + row
            if flat < SQ*H:
                head, query = fast_divmod(flat,SQ)
                src = (((batch*(S*H) + next_split*H + head)*SQ + query)*128
                       + dim_tile*64 + col0)
                cp_async_cg(
                    sO[o_index(load_stage,row,col0)], O_partial[src], 16, 16)
                # instruction_selection: `cp.async.cg.shared.global ...,16,16`
                #   (`.loc 1 590`); extent: one copy per valid owned row,
                #   exactly two possible copies/thread/iteration.
    commit_group()
    # instruction_selection: `cp.async.commit_group` (`.loc 1 530`);
    #   extent: one generation per live loop iteration.
    load_stage = (load_stage + 1) & 3
    wait_group(3)
    # instruction_selection: `cp.async.wait_group 3` (`.loc 1 534`).
    partial0 = ld_shared_v2_b64(sO[o_index(compute_stage,row0,col0)])
    partial1 = ld_shared_v2_b64(sO[o_index(compute_stage,row1,col0)])
    # instruction_selection: two `ld.shared.v2.b64` (`.loc 1 537`);
    #   extent: two FP32x4 owned rows.
    compute_stage = (compute_stage + 1) & 3
    if row0 valid and weight0 > 0:
        for pair in unroll(2):
            weighted = mul_f32x2(partial0[pair], broadcast(weight0))
            acc0[pair] = add_f32x2(acc0[pair], weighted)
            # instruction_selection: separate `mul.f32x2`, `add.f32x2`
            #   (`.loc 1 543`); extent: two pairs.  No FMA: source rounding.
    if row1 valid and weight1 > 0: same two packed pairs

# ---- 7. Direct BSHD output (:550-565) ------------------------------
for owned row in (row0,row1):
    packed0 = cvt_rn_bf16x2(acc[1],acc[0])
    packed1 = cvt_rn_bf16x2(acc[3],acc[2])
    # instruction_selection: `cvt.rn.bf16x2.f32` (`.loc 1 550`);
    #   extent: two words per row, four words per thread.
    if row valid:
        head, query = fast_divmod(16*row_tile + row,SQ)
        dst = ((batch*SQ + query)*H + head)*128 + dim_tile*64 + col0
        st_global_v2_b32(O[dst], packed0, packed1)
        # instruction_selection: `st.global.v2.b32` (`.loc 1 565`);
        #   extent: one BF16x4 store per valid owned row.
```

The accumulation loop needs no per-iteration CTA barrier: each thread reads
exactly the O-ring addresses written by its own `cp.async`, while the only
cross-thread values (normalized LSE weights and max-valid bounds) were already
published by the second barrier.  The four-stage ring cursors are power-of-two
masked integers, preserving source modulo without division.

Rows past `SQ*H` write `-inf` into every physical LSE slot before the first
barrier, so they contribute `max_valid=-1` and never consume stale O.  A dead
split has exactly zero normalized weight.  For an all-dead row, `sum_exp=0`,
`inv=0`, output remains exact zero, and final LSE remains `-inf`.

## Bidirectional evidence map

| semantic edge | source | sketch section | R8 PTX evidence | R256 PTX evidence |
| --- | --- | --- | --- | --- |
| LSE partition/swizzle | `:89-139,323-365` | phase 1, `lse_index` | 64-123; `.loc 331,351,357,364,365` | 64-747; same `.loc` family x32 |
| O partition/ring prime | `:55-88,367-418,568-594` | phase 2 | 124-260; `.loc 374,402,410,418,583,590` | 748-884; same `.loc` family |
| generation publish | `:425-426` | phase 2/3 boundary | 261-264 | 885-888 |
| LSE register loads | `:428-431` | phase 3/4 | 265-275 | 889-934 |
| local/butterfly max | `:437-449` | phase 3/4 | 276-285; no local tree | 935-993; endpoint 16 `max.NaN.f32` plus butterfly; intermediate physical 16/32/64/128 have 1/2/4/8 sites |
| last-live scan/reduce | `:450-455` | phase 3/4 | 286-296; one local update | 994-1190; 32 local updates |
| exp/sum/log/reciprocal | `:456-467` | phase 3/4 | 297-326 | 1191-1436 |
| weights/max publication | `:468-505` | phase 4 | 327-368; scalar normalization | 1437-1562; endpoint 16 packed normalizations; intermediate physical 16/32/64/128 have 1/2/4/8 packed pairs |
| ring overlap | `:508-537,568-594` | phase 5/6 | 404-849; `.loc 524,530,534,537,590` | 1609-2089; same `.loc` family |
| exact packed arithmetic | `:542-543` | phase 5/6 | 466-470 and repeats | 1669-1673 and repeats |
| BF16/direct stores | `:549-565` | phase 7 | 852-891 | 2092-2131 |

Reading in the opposite direction, every selected asynchronous copy, commit,
wait, barrier, shuffle, transcendental, reciprocal, vector shared load, packed
arithmetic operation, conversion, and vector global store in the preserved PTX
maps to exactly one phase above.  Address-only integer instructions map to the
explicit flat ABI and FastDivmod formulas rather than to an implicit layout.

## Correctness and integration contract

The module exports `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`, `get_kernel`,
`prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, and `run_bench`.
Inputs are deterministic mixed-sign FP32 with independently generated LSE and
explicit all-dead/poisoned-dead cases.  TIRx and the pinned source share
read-only inputs and own disjoint outputs.  Correctness compares O BF16 bits
exactly and LSE FP32 bits/classification exactly; any relaxation must be backed
by a specific source/codegen mismatch and remain tighter than a numerical
allclose.

Coverage includes row tails 1/15/17/65, non-power-of-two logical split 3, the
two physical-specialization families (8 and 256), multi-batch/head addressing,
all-dead rows, and production sizes.  Before performance, the same code hash
must pass the independent sketch and correctness reviews, all configs,
memcheck/synccheck guards, low-level IR inspection, `func_calls == ()`, no tile
primitive/layout, and rank-one-SMEM inspection.

Performance truth is only `python -m tirx_kernels.bench_suite`; every required
row must have five finite positive Proton samples for both implementations and
strict `mean(cudnn_frontend)/mean(tirx) > 0.99`.
