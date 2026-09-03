<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839),
Copyright (c) 2026 by FlashInfer team.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer Cake VSA longseq SM103: coarse WASP pipeline sketch

This is the non-executable, mechanically translatable sketch for
[`tirx_kernels/flashinfer/cake_vsa/cake_vsa_longseq_sm103.py`](../../../../tirx_kernels/flashinfer/cake_vsa/cake_vsa_longseq_sm103.py).
The authoritative source is
`csrc/cake_vsa/cake_vsa_longseq_sm_103a.cu` at FlashInfer commit
`cc6e8794c49bf66172627bdb9742fcb17d18b839` (SHA256
`647951f777c17fa4e74e892843ce5ba8ff25d4abf1b0e68b19a264e25532c71d`),
launched by `csrc/cake_vsa/cake_vsa_longseq_host.cpp` and dispatched by
`flashinfer/cake_vsa.py`. The port targets `sm_103a` only.

First-class layouts are forbidden. Every shared-memory object below is a byte
range in one rank-one `u8` arena. Every stage, alias, swizzle, descriptor word,
and TMEM fragment is selected by scalar integer arithmetic. “Tile” means a
logical rectangle only; it is not a TIRx tile primitive or layout-bearing value.

## Scope and specialization boundary

| axis | in-scope values | emitted-code consequence |
| --- | --- | --- |
| dtype / head dim / block | BF16, D=128, R=C=128 | QK idesc `0x04200490`, PV idesc `0x04210490`, descriptor high word `0x40004040` |
| heads | MHA, `num_q_heads == num_kv_heads == 8` | CTA uses `kv_head=q_head`; grid Y is eight |
| selected blocks | fixed uniform count `1..192` per mask row | rolled loops bounded by runtime `selected_blocks`; two 192-entry shared lists |
| scheduling | one CTA per `(query block, head)` | grid `(mb,8,1)`, no persistent scheduler or cluster |
| sequence | `M,N` multiples of 128, `N>=16384` | mask scan covers `nb=N/128` in 32-bit ballot groups |
| LSE | two independent runtime flags in the direct ABI | optional natural-log LSE and temperature-LSE scalar stores |
| partial tiles | source keeps Q/K bounds but the public domain is exact-block aligned | predicates remain source-shaped; no padding route is added |

The public wrapper passes both LSE flags as zero and selects this profile after
the ultrasparse branch. Correctness may call the direct ABI to exercise the
source's two LSE branches. Out of scope: FP16, GQA, other head dimensions or
block sizes, variable top-k, more than 192 blocks, clusters, persistent
scheduling, and the compact/ultrasparse device kernels.

## Writer line-info evidence

Artifacts are under `.porting/cake_vsa_longseq_sm103/writer_exports/`:
`source_sm_103a.lineinfo.ptx` (2100 `.loc`, `.version 9.3`, `.target sm_103a`),
plain PTX, line-info and plain cubins, SASS, ptxas output, key-op locations, and
PTX/SASS histograms. Build flags are `nvcc -ptx|-cubin [-lineinfo]
--std=c++17 --use_fast_math -arch=sm_103a` using CUDA 13.3 V13.3.73.
The native cubin uses 128 registers, nine hardware barriers, zero stack frame,
and zero spill loads/stores.

Static PTX instruction-line counts include predicated instruction lines: 48
`tcgen05.mma.cta_group::1.kind::f16`, 12 TMEM x32 loads, four TMEM x16 stores,
two TMEM x64 stores, 12 `tcgen05.commit`, 18 TMA 4-D loads, 23 mbarrier
initializations, five arrive-expect-tx sites, 30 try-wait sites, 24 elect sites,
194 `ex2.approx.ftz.f32`, two `rcp.approx.ftz.f32`, four
`lg2.approx.ftz.f32`, 16 vector output stores, and three setmaxnreg sites.
The SASS contains 48 `UTCHMMA`, 24 `LDTM.16`, 12 `STTM.16`, 18 `UTMALDG.4D`,
12 `UTCBAR`, 192 `MUFU.EX2`, and 16 `STG.E.128`.

## Pipeline at a glance

| warps | role-local program | publication / reuse edges |
| --- | --- | --- |
| 0..3, 4..7 | two 64-row online-softmax instances: load S from TMEM, select running max with the `2^-8` skip rule, form BF16 P, maintain sums | consume `union_ready`, `s_full[i]`, `corr_done[i]`; produce `corr_sig[i]`, `p_full[i]` |
| 8..11 | prime the P/correction handshake, correct the two TMEM O accumulators when any row rescales, then normalize/store output and optional statistics | consume `union_ready`, `corr_sig[i]`, `o_full[i]`; produce `corr_done[i]`, `p_full[i]`, `tmem_dealloc` |
| 12 | sole MMA issuer, alternating instance-0/1 QK and PV work through separate K/V rings | consume `union_ready`, `q_full`, `k_full`, `v_full`, `p_full`; produce `s_full`, `k_empty`, `v_empty`, `o_full` |
| 13 | load Q, ballot-compact the mask into both union lists, then stream K through three stages | produce `q_full`, `union_ready`, `k_full`; consume `k_empty` |
| 14 | stream V through two stages after the union list is published | consume `union_ready`, `v_empty`; produce `v_full` |
| 15 | register-decreased idle warp | none |

## Primitive vocabulary

```python
linear_smem_u8(bytes, alignment)            # the only shared allocation
byte(base, scalar_offset)                   # scalar byte addressing
reg_fragment(count, dtype)                  # ordinary register scalars
tmem_addr(lane_origin, column)              # taddr + (lane_origin<<16) + column

copy_g2s_tma(map, coords, dst, completion)  # global -> shared
copy_g2r(src, dst) / copy_r2g(src, dst)     # global <-> register
copy_s2r(src, dst) / copy_r2s(src, dst)     # shared <-> register
copy_t2r(src, dst, shape) / copy_r2t(src, dst, shape)  # TMEM <-> register

gemm, fill, cast, add, sub, mul, fma, max, exp2, log2, rcp, select
shuffle_xor, shuffle_index, ballot, any, popc
init_mbarrier, wait, arrive, arrive_expect_tx, commit_mma
fence, sync_warp, sync_cta, named_barrier, elect
setmaxnreg, tmem_alloc, tmem_relinquish, tmem_dealloc, tmem_wait_ld, tmem_wait_st
```

## Complete sketch

### Fixed storage, descriptors, and scalar maps

```python
THREADS=512; WARPS=16; D=128; BLOCK=128; K_STAGES=3; V_STAGES=2
SMEM_BYTES=201344; TMEM_COLS=512; MAX_SELECTED=192
REG_SOFTMAX=192; REG_CORRECTION=80; REG_OTHER=48
launch(grid=(mb,num_q_heads,1), block=(512,1,1), min_blocks_per_sm=1,
       dynamic_smem_bytes=SMEM_BYTES, target="sm_103a")
# instruction_selection: `.maxntid 512`, `.minnctapersm 1`,
# `.extern .shared .align 1024`; extent: one runtime-shape specialization.

ABI=(Qmap,Kmap,Vmap,out_bf16,lse_f32,temperature_lse_f32,block_mask_u8,
     mb_i32,nb_i32,selected_blocks_i32,num_q_heads_i32,num_kv_heads_i32,
     softmax_scale_log2_f32,lse_temperature_scale_f32,
     return_softmax_lse_i32,return_temperature_lse_i32)

# bf16 TensorMaps, rank four, SWIZZLE_128B, no L2 promotion or OOB fill.
# Q dims {64,M,Hq,2}, strides bytes {Hq*256,256,128}, box {64,64,1,2}.
# K/V dims {64,N,2,Hkv}, strides bytes {Hkv*256,128,256}, box {64,64,1,1}.

arena = linear_smem_u8(201344, alignment=1024)
OFF=dict(q_full=0,union_ready=8,k_full=16,k_empty=40,v_full=64,v_empty=80,
         s_full=96,p_full=112,corr_sig=128,corr_done=144,o_full=160,
         tmem_dealloc=176,tmem_mailbox=184,q0=1024,q1=17408,k=33792,
         v=132096,scale=197632,union_count=199680,union_blocks=199688)
def bar(name,i=0): return OFF[name]+8*i
INIT=dict(q_full=1,union_ready=32,k_full=1,k_empty=1,v_full=1,v_empty=1,
          s_full=1,p_full=256,corr_sig=128,corr_done=128,o_full=1,
          tmem_dealloc=128)

def q_byte(instance,dim_half): return OFF.q0+instance*16384+dim_half*8192
def k_byte(stage,token_half,dim_half):
    return OFF.k+stage*32768+token_half*8192+dim_half*16384
def v_byte(stage,token_half,dim_half):
    return OFF.v+stage*32768+token_half*8192+dim_half*16384
def scale_byte(kind,instance,row):
    return OFF.scale+4*(128*kind+64*instance+row)
def union_block_byte(instance,slot): return OFF.union_blocks+4*(192*instance+slot)
def mask_index(head,q_tile,n_block): return (head*mb+q_tile)*nb+n_block
def query_index(q_tile,instance,row): return q_tile*128+instance*64+row
def output_index(query,head,column): return (query*num_q_heads+head)*128+column
def statistic_index(query,head): return query*num_q_heads+head

DESC_HI=0x40004040
QK_IDESC=0x04200490; PV_IDESC=0x04210490
QK_A_STEPS=(2,2,2,506,2,2,2)
QK_B_STEPS=(2,2,2,1018,2,2,2)
PV_A_STEP=8; PV_B_STEP=128
S_COL=(0,256); P_COL=(64,320); O_COL=(128,384)
def tmem_addr(lane_origin,col): return taddr+(lane_origin<<16)+col
```

Each 32-KB K/V stage comprises four 8-KB TMA boxes: token half 0/1 crossed
with feature half 0/1. Two TMEM x32 loads give each lane 64 FP32 elements of a
64x128 logical tile. P overwrites columns 64..127 of each S region; O occupies
the adjacent 128 columns.

### Prologue

```python
q_tile,q_head=cta_id(); warp=warp_id(); lane=lane_id(); smem=cvta_shared(arena)
if warp == 0:
    if elect():
        for offset,count in source_order_23_barriers:
            init_mbarrier(smem+offset,count)
            # instruction_selection: `mbarrier.init.shared::cta.b64`; extent: 23, one lane.
        fence("mbarrier_init_release_cluster")
        # instruction_selection: `fence.mbarrier_init.release.cluster`; extent: one lane.
sync_warp()
# instruction_selection: `bar.warp.sync -1`; extent: all warps.
if warp == 0:
    tmem_alloc(smem+OFF.tmem_mailbox,512); tmem_relinquish()
    # instruction_selection: `tcgen05.alloc...b32 [mailbox],512` then
    # `tcgen05.relinquish_alloc_permit...`; extent: warp 0.
sync_cta(); fence("tcgen05_after_thread_sync")
# instruction_selection: `bar.sync 0`, `tcgen05.fence::after_thread_sync`; extent: CTA.
taddr=copy_s2r(OFF.tmem_mailbox,volatile=True)
# instruction_selection: `ld.volatile.shared.b32`; extent: all threads.

if 12 <= warp <= 15:
    setmaxnreg(decrease,48)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; extent: warps 12..15.
```

### Warps 0..7: online softmax

```python
if warp <= 7:
    setmaxnreg(increase,192)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 192`; extent: warps 0..7.
    wait(bar(union_ready),phase_union_ready=0)
    # instruction_selection: parity `mbarrier.try_wait...cta.shared::cta.b64` retry loop.
    instance=shuffle_index(warp//4,0); row_base=shuffle_index(instance*64,0)
    tmem_base=shuffle_index(instance*256,0)
    # instruction_selection: `shfl.sync.idx.b32`; extent: source warp-uniform broadcasts.
    warp_in_instance=warp%4; lane_origin=warp_in_instance*32
    my_row=warp_in_instance*16+lane%16; col_half=lane//16
    row_valid = my_row < clamp(128-instance*64,0,64)
    row_max=-inf; row_sum=0.0; temperature_sum=0.0

    for u in serial_range(selected_blocks):
        n_block=copy_s2r(union_block_byte(instance,u))
        # instruction_selection: `ld.shared.b32`; extent: one list entry.
        wait(bar(s_full,instance),phase_s[instance]); phase_s[instance]^=1
        # instruction_selection: parity mbarrier retry loop; extent: 128 threads/instance.
        scores=reg_fragment(64,f32)
        copy_t2r(tmem_addr(lane_origin,S_COL[instance]),scores[0:32],"16x32bx2.x32")
        copy_t2r(tmem_addr(lane_origin,S_COL[instance])+32,scores[32:64],"16x32bx2.x32")
        # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32 ... ,64`.
        valid_cols=clamp(nb*128-n_block*128,0,128) if row_valid else 0
        half_valid=clamp(valid_cols-col_half*64,0,64)
        if 0 < half_valid < 64:
            fill(scores[half_valid:64],-inf)
            # instruction_selection: predicated `mov.b32` fill selected by scalar masks.
        tile_max=max(scores)
        # instruction_selection: local `max.f32` tree; extent: one lane's 64 values.
        tile_max=select(half_valid<=0,-inf,tile_max)
        # instruction_selection: predicate and `selp.f32`; extent: one scalar.
        tile_max=max(tile_max,shuffle_xor(tile_max,16))
        # instruction_selection: `shfl.sync.bfly.b32` then `max.f32`; extent: two column halves.
        new_max=max(tile_max,row_max); safe_max=select(new_max==-inf,0,new_max)
        new_max_scaled=mul(safe_max,softmax_scale_log2)
        acc_log2=fma(row_max,softmax_scale_log2,-new_max_scaled)
        # instruction_selection: `max.f32`, `selp.f32`, `mul.ftz.f32`, `fma.rn.ftz.f32`.
        if acc_log2 >= -8.0:
            selected_max=row_max; acc_scale=1.0; temperature_acc_scale=1.0
            new_max_scaled=mul(select(row_max==-inf,0,row_max),softmax_scale_log2)
        else:
            selected_max=new_max; acc_scale=select(row_max>-inf,exp2(acc_log2),1.0)
            temperature_acc_scale=select(row_max>-inf,exp2(acc_log2*lse_temperature_scale),1.0)
            # instruction_selection: `ex2.approx.ftz.f32`; extent: two scalar rescale values.
        row_max=selected_max
        if col_half == 0: copy_r2s(acc_scale,scale_byte(0,instance,my_row))
        # instruction_selection: `st.shared.b32`; extent: one value per logical row.
        fence("proxy_async_shared_cta"); arrive(bar(corr_sig,instance))
        # instruction_selection: `fence.proxy.async.shared::cta`, `mbarrier.arrive.release...`.

        score_bias=select(valid_cols>0,-new_max_scaled,-inf)
        if return_temperature_lse:
            temp_scores=fma(scores,softmax_scale_log2*lse_temperature_scale,
                            score_bias*lse_temperature_scale,lanes=2)
            temp_probs=exp2(temp_scores)
            # instruction_selection: 32 `fma.rn.ftz.f32x2`, 64 `ex2.approx.ftz.f32`.
            temp_half=add_tree(temp_probs[0:32],temp_probs[32:64],lanes=2)
            block_temperature_sum=add(temp_half,shuffle_xor(temp_half,16))
            # instruction_selection: `add.f32x2` tree and `shfl.sync.bfly.b32`.
            copy_t2r(S_again, scores, "2x16x32bx2.x32")
            # instruction_selection: two more x32 TMEM loads; extent: source re-read branch.
            if 0 < half_valid < 64:
                fill(scores[half_valid:64],-inf)
                # instruction_selection: the source-repeated predicated `mov.b32` fill.
        probs=exp2(fma(scores,softmax_scale_log2,score_bias,lanes=2))
        # instruction_selection: 32 `fma.rn.ftz.f32x2`, 64 `ex2.approx.ftz.f32`.
        prob_half=add_tree(probs[0:32],probs[32:64],lanes=2)
        block_sum=add(prob_half,shuffle_xor(prob_half,16))
        # instruction_selection: `add.f32x2` tree and `shfl.sync.bfly.b32`.
        copy_r2t(cast_bf16x2(probs[0:32]),P_COL[instance],"16x32bx2.x16")
        copy_r2t(cast_bf16x2(probs[32:64]),P_COL[instance]+32,"16x32bx2.x16")
        # instruction_selection: `cvt.rn.bf16x2.f32` then two
        # `tcgen05.st.sync.aligned.16x32bx2.x16.b32`; extent: one P tile.
        tmem_wait_st(); arrive(bar(p_full,instance))
        wait(bar(corr_done,instance),phase_corr_done[instance]); phase_corr_done[instance]^=1
        # instruction_selection: `tcgen05.wait::st.sync.aligned`, mbarrier arrival,
        # then parity mbarrier retry loop in source order.
        row_sum=fma(row_sum,acc_scale,block_sum)
        # instruction_selection: scalar `fma.rn.ftz.f32`.
        if return_temperature_lse:
            temperature_sum=fma(temperature_sum,temperature_acc_scale,block_temperature_sum)
            # instruction_selection: conditional scalar `fma.rn.ftz.f32` and select.

    if col_half == 0:
        copy_r2s(row_sum,scale_byte(1,instance,my_row))
        copy_r2s(row_max,scale_byte(2,instance,my_row))
        copy_r2s(temperature_sum,scale_byte(3,instance,my_row))
        # instruction_selection: three `st.shared.b32`; extent: one lane per logical row.
    fence("proxy_async_shared_cta"); arrive(bar(corr_sig,instance))
    # instruction_selection: `fence.proxy.async.shared::cta` then final
    # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 128 threads/instance.
```

### Warps 8..11: correction and epilogue

```python
if 8 <= warp <= 11:
    setmaxnreg(decrease,80)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 80`; extent: warps 8..11.
    my_row=(warp-8)*16+lane%16; col_half=lane//16; row_addr=(warp-8)*32<<16
    wait(bar(union_ready),phase_union_ready=0)
    # instruction_selection: parity mbarrier retry loop.
    arrive(bar(p_full,0)); arrive(bar(p_full,1))
    # instruction_selection: two `mbarrier.arrive.release.cta.shared::cta.b64`.
    wait(bar(corr_sig,0),phase_corr_sig[0]); phase_corr_sig[0]^=1
    arrive(bar(corr_done,0))
    wait(bar(corr_sig,1),phase_corr_sig[1]); phase_corr_sig[1]^=1
    arrive(bar(corr_done,1))
    # instruction_selection: two source-ordered parity waits and two corr_done arrivals.
    for u in serial_range(1,selected_blocks):
        for instance in static_range(2):
            wait(bar(corr_sig,instance),phase_corr_sig[instance]); phase_corr_sig[instance]^=1
            # instruction_selection: parity mbarrier retry loop.
            acc_scale=copy_s2r(scale_byte(0,instance,my_row))
            # instruction_selection: `ld.shared.b32`.
            if any(acc_scale < 1.0):
                # instruction_selection: `vote.sync.any.pred`; extent: one warp.
                accum=copy_t2r(tmem_addr(row_addr,O_COL[instance]),64,"2x16x32bx2.x32")
                # instruction_selection: two x32 TMEM loads.
                accum=mul(accum,acc_scale,lanes=2)
                # instruction_selection: 32 `mul.rn.ftz.f32x2`.
                copy_r2t(accum,tmem_addr(row_addr,O_COL[instance]),"16x32bx2.x64")
                # instruction_selection: one `tcgen05.st.sync.aligned.16x32bx2.x64.b32`.
                tmem_wait_st()
                # instruction_selection: `tcgen05.wait::st.sync.aligned`.
            arrive(bar(p_full,instance)); arrive(bar(corr_done,instance))
            # instruction_selection: two mbarrier arrivals.

    wait(bar(o_full,0),phase_o[0]); wait(bar(o_full,1),phase_o[1])
    wait(bar(corr_sig,0),phase_corr_sig[0]); wait(bar(corr_sig,1),phase_corr_sig[1])
    # instruction_selection: four parity mbarrier retry waits.
    fence("tcgen05_after_thread_sync")
    # instruction_selection: `tcgen05.fence::after_thread_sync`.
    for instance in static_range(2):
        final_sum=copy_s2r(scale_byte(1,instance,my_row))
        final_max=copy_s2r(scale_byte(2,instance,my_row))
        final_temperature_sum=copy_s2r(scale_byte(3,instance,my_row))
        # instruction_selection: scalar shared loads.
        inv_sum=select(final_sum>0 and final_sum==final_sum,rcp(final_sum),0)
        # instruction_selection: `rcp.approx.ftz.f32`, predicates and `selp.f32`.
        accum=copy_t2r(tmem_addr(row_addr,O_COL[instance]),64,"2x16x32bx2.x32")
        # instruction_selection: two x32 TMEM loads.
        query=query_index(q_tile,instance,my_row)
        row_valid=my_row < clamp(128-instance*64,0,64)
        if row_valid:
            for chunk in static_range(8):
                normalized=mul(accum[chunk*8:chunk*8+8],inv_sum,lanes=2)
                packed=cast_bf16x2(normalized)
                # instruction_selection: four `mul.rn.ftz.f32x2` followed by four
                # `cvt.rn.bf16x2.f32`; extent: one eight-element output chunk.
                column=col_half*64+chunk*8
                copy_r2g(packed[0:4],out_bf16[output_index(query,q_head,column)])
                # instruction_selection: one `st.global.v4.b32`; extent: four packed words.
            if col_half == 0 and return_softmax_lse:
                scaled_max=mul(final_max,softmax_scale_log2)
                log_term=mul(log2(final_sum),ln2)
                lse_value=select(final_sum>0,fma(scaled_max,ln2,log_term),-inf)
                copy_r2g(lse_value,lse_f32[statistic_index(query,q_head)])
                # instruction_selection: `lg2.approx.ftz.f32`, two scalar `mul.ftz.f32`,
                # `fma.rn.ftz.f32`, predicate/select, and `st.global.b32`.
            if col_half == 0 and return_temperature_lse:
                scaled_max=mul(final_max,softmax_scale_log2)
                scaled_max_ln2=mul(scaled_max,ln2)
                log_term=mul(log2(final_temperature_sum),ln2)
                tlse_value=select(final_temperature_sum>0,
                    fma(lse_temperature_scale,scaled_max_ln2,log_term),-inf)
                copy_r2g(tlse_value,temperature_lse_f32[statistic_index(query,q_head)])
                # instruction_selection: `lg2.approx.ftz.f32`, three scalar `mul.ftz.f32`,
                # `fma.rn.ftz.f32`, predicate/select, and `st.global.b32`.
    tmem_wait_ld(); fence("tcgen05_before_thread_sync"); arrive(bar(tmem_dealloc))
    # instruction_selection: `tcgen05.wait::ld.sync.aligned`,
    # `tcgen05.fence::before_thread_sync`, one mbarrier arrival per thread.
```

### Warp 12: QK/PV MMA pipeline

```python
if warp == 12:
    wait(bar(union_ready),0); wait(bar(q_full),0)
    # instruction_selection: parity mbarrier retry waits.
    k_stage=0; k_phase=0; v_stage=0; v_phase=0; first_pv=[True,True]
    for instance in static_range(2):
        current_k_stage=k_stage; current_k_phase=k_phase
        advance(k_stage,k_phase,3)
        wait(bar(k_full,current_k_stage),current_k_phase)
        # instruction_selection: parity mbarrier retry wait.
        gemm(S[instance],Q[instance],K[current_k_stage],QK_IDESC,enable_d=False,steps=8)
        # instruction_selection: eight elected `tcgen05.mma.cta_group::1.kind::f16`;
        # descriptor low-word steps are Q `(2,2,2,506,2,2,2)` and
        # K `(2,2,2,1018,2,2,2)`.
        commit_mma(bar(s_full,instance)); commit_mma(bar(k_empty,current_k_stage))
        # instruction_selection: elected `tcgen05.commit...mbarrier::arrive::one` twice.

    for u in serial_range(selected_blocks):
        for instance in static_range(2):
            current_v_stage=v_stage; current_v_phase=v_phase
            advance(v_stage,v_phase,2)
            wait(bar(v_full,current_v_stage),current_v_phase)
            wait(bar(p_full,instance),phase_p[instance]); phase_p[instance]^=1
            # instruction_selection: two parity mbarrier retry waits.
            gemm(O[instance],P[instance],V[current_v_stage],PV_IDESC,
                 enable_d=not first_pv[instance],steps=8)
            # instruction_selection: eight elected `tcgen05.mma.cta_group::1.kind::f16`;
            # TMEM A advances +8 columns and V descriptor advances +128 low words.
            first_pv[instance]=False
            if u+1 == selected_blocks: commit_mma(bar(o_full,instance))
            commit_mma(bar(v_empty,current_v_stage))
            # instruction_selection: elected tcgen05 commits to O/V barriers.
            if u+1 < selected_blocks:
                current_k_stage=k_stage; current_k_phase=k_phase
                advance(k_stage,k_phase,3)
                wait(bar(k_full,current_k_stage),current_k_phase)
                gemm(S[instance],Q[instance],K[current_k_stage],QK_IDESC,enable_d=False,steps=8)
                commit_mma(bar(s_full,instance)); commit_mma(bar(k_empty,current_k_stage))
                # instruction_selection: pre-advance, parity wait, eight elected QK MMAs, two commits.
    wait(bar(tmem_dealloc),0); tmem_dealloc(taddr,512)
    # instruction_selection: parity wait and
    # `tcgen05.dealloc.cta_group::1.sync.aligned.b32 ...,512`.
```

### Warp 13: Q, compaction, and K producer

```python
if warp == 13:
    if elect():
        arrive_expect_tx(bar(q_full),32768)
        copy_g2s_tma(Qmap,(0,q_tile*128,q_head,0),OFF.q0,bar(q_full))
        copy_g2s_tma(Qmap,(0,q_tile*128+64,q_head,0),OFF.q1,bar(q_full))
        # instruction_selection: one arrive-expect-tx and two
        # `cp.async.bulk.tensor.4d...complete_tx::bytes` operations.
    selected_count=0
    for block_base in serial_range(0,nb,32):
        n_block=block_base+lane
        selected=(n_block<nb and copy_g2r(block_mask[mask_index(q_head,q_tile,n_block)])!=0)
        # instruction_selection: predicated `ld.global.nc.b8`.
        votes=ballot(selected); slot=selected_count+popc(votes & ((1<<lane)-1))
        # instruction_selection: `vote.sync.ballot.b32` and `popc.b32`.
        if selected:
            copy_r2s(n_block,union_block_byte(0,slot));
            copy_r2s(n_block,union_block_byte(1,slot))
            # instruction_selection: two `st.shared.b32`.
        selected_count += popc(votes)
    if selected_count == 0:
        if lane == 0:
            copy_r2s(0,union_block_byte(0,0)); copy_r2s(0,union_block_byte(1,0))
        selected_count=1
    if lane < 2: copy_r2s(selected_count,OFF.union_count+lane*4)
    # instruction_selection: scalar shared stores; public domain count equals selected_blocks.
    named_barrier(id=8,count=32); fence("proxy_async_shared_cta"); arrive(bar(union_ready))
    # instruction_selection: `barrier.sync 8,32`, proxy fence, mbarrier arrival.
    k_stage=0; k_empty_phase=1
    for u in serial_range(selected_blocks):
        for instance in static_range(2):
            n_block=copy_s2r(union_block_byte(instance,u))
            wait(bar(k_empty,k_stage),k_empty_phase)
            # instruction_selection: shared load and parity mbarrier retry wait.
            if elect():
                arrive_expect_tx(bar(k_full,k_stage),32768)
                for token_half,dim_half in static_product((0,1),(0,1)):
                    copy_g2s_tma(Kmap,(0,n_block*128+token_half*64,dim_half,q_head),
                                 k_byte(k_stage,token_half,dim_half),bar(k_full,k_stage))
                # instruction_selection: one arrive-expect-tx plus four TMA 4-D loads.
            advance(k_stage,k_empty_phase,3)
```

### Warp 14: V producer; warp 15 idle

```python
if warp == 14:
    wait(bar(union_ready),0)
    # instruction_selection: parity mbarrier retry wait.
    v_stage=0; v_empty_phase=1
    for u in serial_range(selected_blocks):
        for instance in static_range(2):
            n_block=copy_s2r(union_block_byte(instance,u))
            wait(bar(v_empty,v_stage),v_empty_phase)
            # instruction_selection: shared load and parity mbarrier retry wait.
            if elect():
                arrive_expect_tx(bar(v_full,v_stage),32768)
                for token_half,dim_half in static_product((0,1),(0,1)):
                    copy_g2s_tma(Vmap,(0,n_block*128+token_half*64,dim_half,q_head),
                                 v_byte(v_stage,token_half,dim_half),bar(v_full,v_stage))
                # instruction_selection: one arrive-expect-tx plus four TMA 4-D loads.
            advance(v_stage,v_empty_phase,2)
if warp == 15:
    pass
```

The role conditions are independent sibling guards in source order. They must
not become an `if/elif` chain, and the separate K and V producers/rings must not
be fused even though both consume the same compact list.

## TIRx module and benchmark contract

- The module imports only `tirx_kernels.kern as K` as its device language and
  exposes the repository-standard metadata, config, preparation, test, and
  benchmark entry points.
- Correctness uses the exact pinned FlashInfer cubin as the primary reference,
  requires bitwise source/TIRx equality, deterministic repeats, and adds a
  chunked sparse FP64 oracle and guard checks.
- Benchmark setup, compilation, data generation, and validation occur outside
  the no-argument source/TIRx launch closures. The curated sweep uses Proton
  through `tvm.tirx.bench.bench`; `bench_suite` is the only performance gate.
- Both binaries must declare PTX 9.3 targeting `sm_103a` before a comparison is
  valid. Every required source-time/TIRx-time ratio must be strictly above 0.99.

## Instruction-selection summary

The source performance structure is selected by the 16-warp role split,
independent register scopes, the one linear 201344-byte arena, exact 128-byte
TMA descriptors, separate three-stage K and two-stage V pipelines, two
64x128-score/64x128-output TMEM instances, eight-instruction QK/PV MMA chains,
packed FP32x2 softmax arithmetic, approximate base-2 math, and 128-bit output
stores. These choices are source requirements rather than optional tuning
ideas; later performance work may alter lowering controls only while preserving
this execution skeleton.
