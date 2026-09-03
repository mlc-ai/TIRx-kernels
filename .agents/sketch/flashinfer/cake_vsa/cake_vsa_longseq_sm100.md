<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ c5365737570a2a156d7cae0c4070fa3770ecc670),
Copyright (c) 2026 by FlashInfer team.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer Cake VSA longseq SM100: coarse WASP pipeline sketch

This is the non-executable, mechanically translatable execution sketch for
[`tirx_kernels/flashinfer/cake_vsa/cake_vsa_longseq_sm100.py`](../../../../tirx_kernels/flashinfer/cake_vsa/cake_vsa_longseq_sm100.py).
The authoritative source is `csrc/cake_vsa/cake_vsa_longseq_sm_100a.cu` at
FlashInfer commit `c5365737570a2a156d7cae0c4070fa3770ecc670`, file SHA-256
`647951f777c17fa4e74e892843ce5ba8ff25d4abf1b0e68b19a264e25532c71d`,
launched by `csrc/cake_vsa/cake_vsa_longseq_host.cpp`. The specialization is
SM100a only: BF16, D=128, R=C=128, MHA with eight heads, `N >= 16384`, and one
uniform selected-block count in `[1,192]`. Other dtypes, dimensions, GQA,
clusters, persistent scheduling, and the compact/ultrasparse profiles are out
of scope.

The writer export is
`.porting/cake_vsa_longseq_sm100/writer_exports/source_sm_100a.lineinfo.ptx`:
PTX 9.2, target `sm_100a`, 1046 `.loc` records. The paired cubin uses 128
registers, zero spill bytes, nine hardware barriers. Instruction counts below
are static PTX instruction lines, including predicated lines: 48 `tcgen05.mma`,
12 `tcgen05.commit`, 18 four-dimensional TMA loads, 23 barrier initializations,
30 parity-wait loops, 12 TMEM loads, six TMEM stores, 194 `ex2`, 96 packed FMA,
96 packed add, 128 packed multiply, and 20 global stores.

Every shared object is an interval in one rank-one byte arena. Every logical
coordinate, stage, swizzle, alias, descriptor, and TMEM reference is represented
by scalar integer arithmetic. No first-class layout or tile primitive exists.

## Pipeline at a glance

| warps | role-local program | publication and reuse edges |
| --- | --- | --- |
| 0–7 | two four-warp, 64-row instances read S from TMEM, update row max/sums, form BF16 P, and publish rescale/final statistics | consume `union_ready`, `s_full[i]`, `corr_done[i]`; produce `corr_sig[i]`, `p_full[i]` |
| 8–11 | rescale each instance's TMEM O when required, then load/scale/store output and optional statistics | consume `union_ready`, `corr_sig[i]`, `o_full[i]`; produce `corr_done[i]`, half of `p_full[i]`, `tmem_dealloc` |
| 12 | sole QK/PV MMA issuer; interleaves independent three-stage K and two-stage V rings | consume `q_full`, `k_full[*]`, `v_full[*]`, `p_full[i]`, `tmem_dealloc`; produce `s_full[i]`, `k_empty[*]`, `v_empty[*]`, `o_full[i]` |
| 13 | loads Q, ballot-compacts the mask, and streams K | produce `q_full`, `union_ready`, `k_full[*]`; consume `k_empty[*]` |
| 14 | waits for the compact list and streams V independently | consume `union_ready`, `v_empty[*]`; produce `v_full[*]` |
| 15 | register-decreased idle warp | none |

## Primitive vocabulary

```python
linear_smem_u8(bytes, alignment)
byte(base, scalar_offset)
reg_tile(count, dtype)
tmem_addr(lane_origin, column)

copy_g2s_tma(tensor_map, coordinates, shared_byte, completion_barrier)
copy_g2r(global_byte, register)
copy_s2r(shared_byte, register)
copy_r2s(register, shared_byte)
copy_t2r(tmem_address, registers, instruction_shape)
copy_r2t(registers, tmem_address, instruction_shape)
copy_r2g(registers, global_byte)

fill, cast, add, sub, mul, fma, max, exp2, log2, rcp, select
shuffle_xor, shuffle_index, ballot, any, popc, gemm
init_mbarrier, wait, arrive, arrive_expect_tx, commit, fence
named_barrier, sync_warp, sync_cta, elect, setmaxnreg
tmem_alloc, tmem_relinquish, tmem_dealloc, tmem_wait_ld, tmem_wait_st
```

Each occurrence below denotes one instruction, one fixed tile of a single
instruction family, or one explicit loop of that family.

## Fixed resources and scalar maps

```python
THREADS=512; WARPS=16; BLOCK=128; D=128; K_STAGES=3; V_STAGES=2
SMEM_BYTES=201344; MAX_SELECTED=192; TMEM_COLS=512
REG_SOFTMAX=192; REG_CORRECTION=80; REG_OTHER=48
launch(grid=(mb,num_q_heads,1), block=(512,1,1), min_blocks_per_sm=1,
       dynamic_smem_bytes=201344, target="sm_100a")
# instruction_selection: `__launch_bounds__(512,1)`, `.extern .shared .align 1024`;
# extent: one runtime-shape specialization.

ABI=(Qmap,Kmap,Vmap,out_bf16,lse_f32,temperature_lse_f32,block_mask_u8,
     mb_i32,nb_i32,selected_blocks_i32,num_q_heads_i32,num_kv_heads_i32,
     softmax_scale_log2_f32,lse_temperature_scale_f32,
     return_softmax_lse_i32,return_temperature_lse_i32)
# Q map: bf16 rank 4, dims {64,M,Hq,2}, byte strides {Hq*256,256,128},
# box {64,64,1,2}; K/V maps: dims {64,N,2,Hkv}, strides
# {Hkv*256,128,256}, box {64,64,1,1}; all use SWIZZLE_128B.

arena = linear_smem_u8(201344, alignment=1024)
OFF=dict(q_full=0,union_ready=8,k_full=16,k_empty=40,v_full=64,v_empty=80,
         s_full=96,p_full=112,corr_sig=128,corr_done=144,o_full=160,
         tmem_dealloc=176,tmem_mailbox=184,q0=1024,q1=17408,k=33792,
         v=132096,scale=197632,union_count=199680,union_blocks=199688)
def bar(name,i=0): return OFF[name] + 8*i
INIT_COUNT=dict(q_full=1,union_ready=32,k_full=1,k_empty=1,v_full=1,v_empty=1,
                s_full=1,p_full=256,corr_sig=128,corr_done=128,o_full=1,
                tmem_dealloc=128)
def q_byte(instance,dim_half): return OFF.q0 + 16384*instance + 8192*dim_half
def k_byte(stage,token_half,dim_half):
    return OFF.k + 32768*stage + 8192*token_half + 16384*dim_half
def v_byte(stage,token_half,dim_half):
    return OFF.v + 32768*stage + 8192*token_half + 16384*dim_half
def scale_byte(kind,instance,row): return OFF.scale + 4*(128*kind+64*instance+row)
def union_count_byte(instance): return OFF.union_count + 4*instance
def union_block_byte(instance,slot): return OFF.union_blocks + 4*(192*instance+slot)
# instruction_selection: scalar `ld.shared.b32` / `st.shared.b32`; extent:
# explicit rank-one byte addresses only.

DESC_HI=0x40004040
QK_IDESC=0x04200490; PV_IDESC=0x04210490
QK_A_STEPS=(2,2,2,506,2,2,2); QK_B_STEPS=(2,2,2,1018,2,2,2)
PV_A_STEP=8; PV_B_STEP=128; V_LBO_BIT=0x4000000
S_COL=(0,256); P_COL=(64,320); O_COL=(128,384)
def desc_lo(shared_byte): return (shared_address(shared_byte)>>4) & 0x3fff
def tmem_addr(lane_origin,column): return tmem_base + (lane_origin<<16) + column
```

## Complete operation sketch

### Prologue

```python
q_tile,q_head = cta_id(); warp=shuffle_index(thread_id()//32,0); lane=thread_id()%32
# instruction_selection: `%ctaid.x/y`, `%tid.x`, one `shfl.sync.idx.b32`;
# extent: scalar special-register reads and one warp-uniform broadcast.

if warp == 0 and elect():
    for name in (q_full,union_ready,k_full*3,k_empty*3,v_full*2,v_empty*2,
                 s_full*2,p_full*2,corr_sig*2,corr_done*2,o_full*2,tmem_dealloc):
        init_mbarrier(bar(name),INIT_COUNT[name])
    # instruction_selection: `elect.sync`, 23 `mbarrier.init.shared::cta.b64`;
    # extent: elected lane of warp 0.
    fence("mbarrier_init_release_cluster")
    # instruction_selection: `fence.mbarrier_init.release.cluster`; extent: one.
sync_warp()
# instruction_selection: `bar.warp.sync -1`; extent: every warp.
if warp == 0:
    tmem_alloc(OFF.tmem_mailbox,512); tmem_relinquish()
    # instruction_selection: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`
    # then `tcgen05.relinquish_alloc_permit...`; extent: warp 0.
sync_cta(); fence("tcgen05_after_thread_sync")
# instruction_selection: `bar.sync 0`, `tcgen05.fence::after_thread_sync`.
tmem_base=copy_s2r(OFF.tmem_mailbox,volatile=True)
# instruction_selection: `ld.volatile.shared.b32`; extent: all threads.

if 12 <= warp <= 15: setmaxnreg(decrease,48)
# instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; extent: warps 12–15.
```

Every parity wait is the source retry loop around
`mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` with timeout `0x989680`.
Producer empty phases start at one; other phases start at zero and flip after
each completed wait.

### Warps 0–7: two 64-row probability/statistics instances

```python
if warp <= 7:
    setmaxnreg(increase,192)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 192`; extent: eight warps.
    wait(bar(union_ready),phase_union_ready)
    # instruction_selection: parity retry loop; extent: 256 threads.
    instance=shuffle_index(warp//4,0); lane_origin=(warp%4)*32
    my_row=(warp%4)*16 + lane%16; col_half=lane//16
    # instruction_selection: source's four `shfl.sync.idx.b32` broadcasts;
    # extent: instance/row/token/TMEM offsets.
    row_max=-inf; row_sum=0.0; temperature_sum=0.0
    for block_slot in serial_range(selected_blocks):
        n_block=copy_s2r(union_block_byte(instance,block_slot))
        # instruction_selection: `ld.shared.b32`; extent: one selected index.
        wait(bar(s_full,instance),phase_s[instance]); phase_s[instance]^=1
        # instruction_selection: parity retry loop; extent: 128 threads per instance.
        s=reg_tile(64,"f32")
        copy_t2r(tmem_addr(lane_origin,S_COL[instance]),s[0:32],"16x32bx2.x32")
        copy_t2r(tmem_addr(lane_origin,S_COL[instance]+32),s[32:64],"16x32bx2.x32")
        # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`;
        # extent: 64 FP32 values per thread.
        acc=(-inf,-inf)
        for j in static_range(32): acc[j%2]=max(acc[j%2],max(s[2*j],s[2*j+1]))
        tile_max=max(max(acc[0],acc[1]),shuffle_xor(max(acc[0],acc[1]),16))
        # instruction_selection: 67 static `max.f32` plus `shfl.sync.bfly.b32`;
        # extent: one 128-column row split between lane halves.
        new_max=max(tile_max,row_max); safe_max=select(new_max == -inf,0.0,new_max)
        scaled_max=mul(safe_max,softmax_scale_log2)
        delta=fma(row_max,softmax_scale_log2,-scaled_max)
        # instruction_selection: `max.f32`, `selp.f32`, `mul.ftz.f32`,
        # `fma.rn.ftz.f32`; extent: row scalars.
        if delta >= -8.0:
            acc_scale=1.0; temperature_acc_scale=1.0; selected_max=row_max
            safe_max=select(row_max == -inf,0.0,row_max)
            scaled_max=mul(safe_max,softmax_scale_log2)
        else:
            acc_scale=select(row_max>-inf,exp2(delta),1.0)
            temperature_acc_scale=select(row_max>-inf,exp2(mul(delta,lse_temperature_scale)),1.0)
            selected_max=new_max
        # instruction_selection: `setp.ltu.ftz.f32` branch, `ex2.approx.ftz.f32`,
        # `mul.ftz.f32`, and `selp.f32`; extent: one rescale decision.
        row_max=selected_max
        if col_half == 0: copy_r2s(acc_scale,scale_byte(0,instance,my_row))
        # instruction_selection: predicated `st.shared.b32`; extent: one row scale.
        fence("proxy_async_shared_cta"); arrive(bar(corr_sig,instance))
        # instruction_selection: `fence.proxy.async.shared::cta` then
        # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 128 arrivals.
        bias=select(True,-scaled_max,-inf)
        if return_temperature_lse != 0:
            for j in static_range(32):
                s[2*j:2*j+2]=fma(s[2*j:2*j+2],(softmax_scale_log2*lse_temperature_scale,)*2,
                                   (bias*lse_temperature_scale,)*2,lanes=2)
            for j in static_range(64): s[j]=exp2(s[j])
            # instruction_selection: 32 `fma.rn.ftz.f32x2`, 64
            # `ex2.approx.ftz.f32`; extent: the optional temperature fragment.
            pair_acc=(0.0,0.0)
            for j in static_range(32): pair_acc=add(pair_acc,s[2*j:2*j+2],lanes=2)
            half=add(pair_acc[0],pair_acc[1])
            temperature_block_sum=add(half,shuffle_xor(half,16))
            # instruction_selection: 32 `add.f32x2`, two scalar `add.ftz.f32`,
            # one `shfl.sync.bfly.b32`; extent: one row.
            copy_t2r(tmem_addr(lane_origin,S_COL[instance]),s[0:32],"16x32bx2.x32")
            copy_t2r(tmem_addr(lane_origin,S_COL[instance]+32),s[32:64],"16x32bx2.x32")
            # instruction_selection: two `tcgen05.ld...x32`; extent: re-read S.
        for j in static_range(32):
            s[2*j:2*j+2]=fma(s[2*j:2*j+2],(softmax_scale_log2,)*2,(bias,)*2,lanes=2)
        for j in static_range(64): s[j]=exp2(s[j])
        pair_acc=(0.0,0.0)
        for j in static_range(32): pair_acc=add(pair_acc,s[2*j:2*j+2],lanes=2)
        half=add(pair_acc[0],pair_acc[1])
        block_sum=add(half,shuffle_xor(half,16))
        # instruction_selection: 32 packed FMA, 64 `ex2`, 32 packed add, two
        # scalar add and one shuffle; extent: one probability row.
        packed=[cast((s[2*j],s[2*j+1]),"bf16x2") for j in static_range(32)]
        copy_r2t(packed[0:16],tmem_addr(lane_origin,P_COL[instance]),"16x32bx2.x16")
        copy_r2t(packed[16:32],tmem_addr(lane_origin,P_COL[instance]+16),"16x32bx2.x16")
        # instruction_selection: 32 `cvt.rn.bf16x2.f32` and two
        # `tcgen05.st.sync.aligned.16x32bx2.x16.b32`; extent: 64 probabilities.
        tmem_wait_st(); arrive(bar(p_full,instance))
        wait(bar(corr_done,instance),phase_corr_done[instance]); phase_corr_done[instance]^=1
        # instruction_selection: TMEM store wait, mbarrier arrive, parity retry wait.
        row_sum=fma(row_sum,acc_scale,block_sum)
        if return_temperature_lse: temperature_sum=fma(temperature_sum,temperature_acc_scale,
                                                        temperature_block_sum)
        # instruction_selection: `fma.rn.ftz.f32`; extent: one or two running sums.
    if col_half == 0:
        copy_r2s(row_sum,scale_byte(1,instance,my_row))
        copy_r2s(row_max,scale_byte(2,instance,my_row))
        copy_r2s(temperature_sum,scale_byte(3,instance,my_row))
    # instruction_selection: three predicated `st.shared.b32`; extent: final row state.
    fence("proxy_async_shared_cta"); arrive(bar(corr_sig,instance))
    # instruction_selection: proxy fence and mbarrier arrive; extent: final generation.
```

### Warps 8–11: correction and output

```python
if 8 <= warp <= 11:
    setmaxnreg(decrease,80)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 80`; extent: four warps.
    wait(bar(union_ready),phase_union_ready)
    warp_in_role=warp-8; lane_origin=warp_in_role*32
    logical_row_origin=warp_in_role*16
    my_row=logical_row_origin+lane%16; col_half=lane//16
    arrive(bar(p_full,0)); arrive(bar(p_full,1))
    for instance in static_range(2):
        wait(bar(corr_sig,instance),phase_corr_sig[instance]); phase_corr_sig[instance]^=1
        arrive(bar(corr_done,instance))
    # instruction_selection: parity waits and mbarrier arrives; first block never rescales O.
    for block_slot in serial_range(1,selected_blocks):
        for instance in static_range(2):
            wait(bar(corr_sig,instance),phase_corr_sig[instance]); phase_corr_sig[instance]^=1
            acc_scale=copy_s2r(scale_byte(0,instance,my_row))
            # instruction_selection: parity wait and `ld.shared.b32`; extent: one row.
            if any(acc_scale < 1.0):
                # instruction_selection: `setp.lt.ftz.f32`, `vote.sync.any.pred`; one warp vote.
                o=reg_tile(64,"f32")
                copy_t2r(tmem_addr(lane_origin,O_COL[instance]),o[0:32],"16x32bx2.x32")
                copy_t2r(tmem_addr(lane_origin,O_COL[instance]+32),o[32:64],"16x32bx2.x32")
                for j in static_range(32):
                    o[2*j:2*j+2]=mul(o[2*j:2*j+2],(acc_scale,)*2,lanes=2)
                copy_r2t(o,tmem_addr(lane_origin,O_COL[instance]),"16x32bx2.x64")
                # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`,
                # 32 `mul.rn.ftz.f32x2`, one
                # `tcgen05.st.sync.aligned.16x32bx2.x64.b32`; extent: 64 registers.
                tmem_wait_st()
                # instruction_selection: `tcgen05.wait::st.sync.aligned`; extent: one.
            arrive(bar(p_full,instance)); arrive(bar(corr_done,instance))
            # instruction_selection: two mbarrier arrives; extent: one instance/block.
    wait(bar(o_full,0),phase_o[0]); wait(bar(o_full,1),phase_o[1])
    wait(bar(corr_sig,0),phase_corr_sig[0]); wait(bar(corr_sig,1),phase_corr_sig[1])
    fence("tcgen05_after_thread_sync")
    # instruction_selection: four parity waits and `tcgen05.fence::after_thread_sync`.
    for instance in static_range(2):
        final_sum=copy_s2r(scale_byte(1,instance,my_row))
        final_max=copy_s2r(scale_byte(2,instance,my_row))
        final_temperature_sum=copy_s2r(scale_byte(3,instance,my_row))
        inv_sum=select(final_sum>0.0,rcp(final_sum),0.0)
        # instruction_selection: three unconditional `ld.shared.b32`,
        # `rcp.approx.ftz.f32`, `selp.f32`; extent: one instance row.
        o=reg_tile(64,"f32")
        copy_t2r(tmem_addr(lane_origin,O_COL[instance]),o[0:32],"16x32bx2.x32")
        copy_t2r(tmem_addr(lane_origin,O_COL[instance]+32),o[32:64],"16x32bx2.x32")
        # instruction_selection: two `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`;
        # extent: 64 columns in the lane-selected half.
        for chunk in static_range(8):
            for j in static_range(4):
                o[8*chunk+2*j:8*chunk+2*j+2]=mul(
                    o[8*chunk+2*j:8*chunk+2*j+2],(inv_sum,)*2,lanes=2)
            words=[cast((o[8*chunk+2*j],o[8*chunk+2*j+1]),"bf16x2")
                   for j in static_range(4)]
            copy_r2g(words,out_row+col_half*64+8*chunk)
        # instruction_selection: 32 packed FTZ multiplies, 32 packed BF16
        # conversions, eight `st.global.v4.b32`; extent: one 64-column lane half.
        if col_half == 0:
            if return_softmax_lse:
                lse_core=fma(mul(final_max,softmax_scale_log2),ln2,
                             mul(log2(final_sum),ln2))
                softmax_lse=select(final_sum>0.0,lse_core,-inf)
                copy_r2g(softmax_lse,lse_index)
                # instruction_selection: `lg2.approx.ftz.f32`, ordered mul/FMA,
                # positive-sum `selp.f32`, and one `st.global.b32`.
            if return_temperature_lse:
                temperature_core=fma(
                    lse_temperature_scale,
                    mul(mul(final_max,softmax_scale_log2),ln2),
                    mul(log2(final_temperature_sum),ln2))
                temperature_value=select(final_temperature_sum>0.0,temperature_core,-inf)
                copy_r2g(temperature_value,temperature_lse_index)
                # instruction_selection: `lg2.approx.ftz.f32`, source-ordered nested
                # mul/FMA, positive-sum `selp.f32`, and one `st.global.b32`.
    tmem_wait_ld(); fence("tcgen05_before_thread_sync"); arrive(bar(tmem_dealloc))
    # instruction_selection: `tcgen05.wait::ld.sync.aligned`,
    # `tcgen05.fence::before_thread_sync`, mbarrier arrive; extent: 128 threads.
```

### Warp 12: MMA issuer

```python
if warp == 12:
    wait(bar(union_ready),phase_union_ready); wait(bar(q_full),phase_q_full)
    # instruction_selection: two parity retry loops; extent: warp 12.
    k_ring=(stage=0,phase=0); v_ring=(stage=0,phase=0); first_pv=(True,True)
    for instance in static_range(2):
        wait(bar(k_full,k_ring.stage),k_ring.phase)
        for k16 in static_range(8):
            gemm(tmem_addr(0,S_COL[instance]),q_descriptor(instance,k16),
                 k_descriptor(k_ring.stage,k16),QK_IDESC,accumulate=(k16!=0))
        # instruction_selection: parity wait, `elect.sync`, eight predicated
        # `tcgen05.mma.cta_group::1.kind::f16`; extent: one 64x128x128 QK tile.
        commit(bar(s_full,instance)); commit(bar(k_empty,k_ring.stage)); k_ring.advance()
        # instruction_selection: two elected `tcgen05.commit...mbarrier::arrive::one`.
    for block_slot in serial_range(selected_blocks):
        for instance in static_range(2):
            wait(bar(v_full,v_ring.stage),v_ring.phase)
            wait(bar(p_full,instance),phase_p[instance]); phase_p[instance]^=1
            for k16 in static_range(8):
                gemm(tmem_addr(0,O_COL[instance]),tmem_addr(0,P_COL[instance]+8*k16),
                     v_descriptor(v_ring.stage,k16),PV_IDESC,
                     accumulate=(k16!=0 or not first_pv[instance]))
            # instruction_selection: two parity waits, elect, eight predicated
            # `tcgen05.mma...kind::f16`; extent: one 64x128x128 PV tile.
            first_pv[instance]=False
            if block_slot+1 == selected_blocks: commit(bar(o_full,instance))
            commit(bar(v_empty,v_ring.stage)); v_ring.advance()
            # instruction_selection: elected tcgen05 commit(s); output commit only on last block.
            if block_slot+1 < selected_blocks:
                wait(bar(k_full,k_ring.stage),k_ring.phase)
                for k16 in static_range(8):
                    gemm(tmem_addr(0,S_COL[instance]),q_descriptor(instance,k16),
                         k_descriptor(k_ring.stage,k16),QK_IDESC,accumulate=(k16!=0))
                commit(bar(s_full,instance)); commit(bar(k_empty,k_ring.stage)); k_ring.advance()
                # instruction_selection: parity wait, eight QK `tcgen05.mma`, two commits;
                # extent: prefetch the next score tile between PV instances.
    wait(bar(tmem_dealloc),phase_tmem_dealloc)
    tmem_dealloc(copy_s2r(OFF.tmem_mailbox,volatile=True),512)
    # instruction_selection: parity wait, `ld.volatile.shared.b32`,
    # `tcgen05.dealloc.cta_group::1.sync.aligned.b32`; extent: warp 12.
```

### Warp 13: Q, mask, and K producer

```python
if warp == 13:
    if elect():
        arrive_expect_tx(bar(q_full),32768)
        copy_g2s_tma(Qmap,(0,query_base,q_head,0),q_byte(0,0),bar(q_full))
        copy_g2s_tma(Qmap,(0,query_base+64,q_head,0),q_byte(1,0),bar(q_full))
    # instruction_selection: elect, one expect-tx, two
    # `cp.async.bulk.tensor.4d.shared::cta.global...`; extent: two 16 KiB Q halves.
    selected_count=0
    for block_base in serial_range(0,nb,32):
        selected=copy_g2r(block_mask[mask_base+block_base+lane]) != 0 if block_base+lane<nb else False
        bits=ballot(selected); lower=popc(bits & ((1<<lane)-1)); count=popc(bits)
        # instruction_selection: predicated `ld.global.nc.b8`, `vote.sync.ballot.b32`,
        # two `popc.b32`; extent: one 32-column mask chunk.
        if selected:
            copy_r2s(block_base+lane,union_block_byte(0,selected_count+lower))
            copy_r2s(block_base+lane,union_block_byte(1,selected_count+lower))
        # instruction_selection: two predicated `st.shared.b32`; extent: one selected lane.
        selected_count+=count
    if lane<2: copy_r2s(selected_count,union_count_byte(lane))
    named_barrier(8,32); fence("proxy_async_shared_cta"); arrive(bar(union_ready))
    # instruction_selection: `barrier.sync 8,32`, proxy fence, mbarrier arrive.
    k_ring=(stage=0,phase=1)
    for block_slot in serial_range(selected_blocks):
        for instance in static_range(2):
            n_block=copy_s2r(union_block_byte(instance,block_slot))
            wait(bar(k_empty,k_ring.stage),k_ring.phase)
            if elect():
                arrive_expect_tx(bar(k_full,k_ring.stage),32768)
                for token_half,dim_half in static_product(2,2):
                    copy_g2s_tma(Kmap,(0,n_block*128+64*token_half,dim_half,q_head),
                                 k_byte(k_ring.stage,token_half,dim_half),bar(k_full,k_ring.stage))
            # instruction_selection: shared index load, parity wait, elect, expect-tx,
            # four 8 KiB four-dimensional TMA loads; extent: one K tile.
            k_ring.advance()
```

### Warp 14 and warp 15

```python
if warp == 14:
    wait(bar(union_ready),phase_union_ready)
    # instruction_selection: parity retry loop; extent: warp 14.
    v_ring=(stage=0,phase=1)
    for block_slot in serial_range(selected_blocks):
        for instance in static_range(2):
            n_block=copy_s2r(union_block_byte(instance,block_slot))
            wait(bar(v_empty,v_ring.stage),v_ring.phase)
            if elect():
                arrive_expect_tx(bar(v_full,v_ring.stage),32768)
                for token_half,dim_half in static_product(2,2):
                    copy_g2s_tma(Vmap,(0,n_block*128+64*token_half,dim_half,q_head),
                                 v_byte(v_ring.stage,token_half,dim_half),bar(v_full,v_ring.stage))
            # instruction_selection: shared index load, parity wait, elect, expect-tx,
            # four 8 KiB four-dimensional TMA loads; extent: one V tile.
            v_ring.advance()
if warp == 15: pass
```

## Bidirectional source/sketch/PTX map

| source region | sketch region | line-info PTX evidence |
| --- | --- | --- |
| helpers 44–481 | primitive vocabulary | helper `.loc` records for elect, waits, TMA, TMEM, packed arithmetic, and approximate math |
| ABI/storage/init 489–596 | fixed resources and prologue | 23 `mbarrier.init`, alloc/relinquish, warp/CTA sync, dynamic shared declaration |
| probability/statistics 598–933 | warps 0–7 | `.loc` 624/657/660/905/909 waits; 12 TMEM x32 loads; 194 ex2; packed FMA/add; four TMEM x16 stores |
| correction/output 935–1118 | warps 8–11 | `.loc` 949–1030 waits/rescale stores; 1043 fence; 1049–1110 shared loads, TMEM loads, 16 vector output stores and four statistic stores |
| MMA 1120–1529 | warp 12 | `.loc` 1198/1264/1325/1394/1451/1519 six static eight-issue chains and 12 static commits |
| Q/mask/K 1531–1609 | warp 13 | `.loc` 1549–1588 Q TMA, mask load/ballot/shared stores/publication; 1595 K list loads and K pipeline waits/TMA |
| V 1611–1659 | warp 14 | `.loc` 1617/1639/1641 V publication wait, list loads, V pipeline waits/TMA |
| idle 1661–1663 | warp 15 | no emitted operation |

Reading the mapping in reverse, every TMA, TMEM load/store, MMA, tcgen commit,
mbarrier operation, named barrier, vote, shuffle, transcendental, packed FP32
operation, conversion, global store, and register-budget instruction in the
writer export has an occurrence above.

## TIRx and validation contract

The implementation imports only `tirx_kernels.kern as K`, uses a single
rank-one shared allocation and scalar offsets, compiles with nvcc for PTX 9.2,
and contains no tile primitive, first-class layout, `K.cuda.func_call`, verifier
exception, or edit under `tirx_kernels/kern/`.

Correctness compares independent outputs from this exact TIRx implementation
and FlashInfer's direct `longseq/sm_100a` module bit-for-bit for BF16 output and
enabled FP32 statistics, repeats TIRx deterministically, checks disabled-output
sentinels, exercises selected counts 1/8/16/32/65/128/192 plus both statistic
branches and an amplified-Q rescale case, and applies a separate strict FP32
oracle. Performance truth is exclusively the six-row `bench_suite` matrix with
the pinned FlashInfer reference and requires every mean ratio to be strictly
greater than 0.99.
