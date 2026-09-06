<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 2497bee508e6337c086f0993f68b41cabce97436),
Copyright (c) 2025 Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms,
Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun,
Yong Li, and Wei Lin.
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100/SM103 FlexAttention forward HD256: operation-level sketch

This is the non-executable sketch for
[`tirx_kernels/cudnn/flex_attention/forward_hd256_sm100.py`](../../tirx_kernels/cudnn/flex_attention/forward_hd256_sm100.py).
It freezes
`python/cudnn/flex_attention/kernels/sm100/fwd/forward_hd256.py::
BlackwellFusedMultiHeadAttentionForward` from cuDNN Frontend commit
`2497bee508e6337c086f0993f68b41cabce97436`. The source SHA-256 is
`ed05d9a992c6853cc9e8180d4557e0ac711a0bc7eb7aa22c83e1c94e602f9671`.

Writer line-info exports are
`.porting/forward_hd256_sm100/source_cta1.lineinfo.ptx` (SHA-256
`66e180cb5895e249e31ec7b5c58ce0f540d7a57ca57af62434c4f2d7b42b2963`)
and `source_cta2.lineinfo.ptx` (SHA-256
`8948f8dc99d56fbcd8e3ef5d380fbb8b7a492de22d6706a7127e2b57c6e5aad7`).
CUDA 13.3.27 emitted PTX ISA 9.3 for `sm_103a`; the exports contain 1,958
and 2,064 `.loc` records. All selections below are derived from those
exports and the pinned source.

After the independent sketch reviewer returns PASS, this file is immutable.

## Scope and invariants

The port covers the source's dedicated HD256 path: FP16 or BF16 Q/K/V/O,
FP32 QK/PV accumulation, Dqk=Dv=256, physical M128 x N128 work per CTA,
one-CTA and two-CTA-group instructions, fixed BSHD and true packed THD,
unpacked MHA/GQA with `Hq % Hkv == 0`, shared or per-query-head compact
plans, optional FP32 LSE, partial/full N128 blocks, and empty plan rows.
`pack_gqa` is false. SM100a and SM103a are runtime targets; writer evidence
is SM103a.

The executable imports the device language only as
`import tirx_kernels.kern as K`. Every shared tensor is a rank-one linear
allocation with explicit scalar byte offsets. No tile primitive, first-class
layout/mapping object, `layout=` argument, direct TVM script namespace,
`K.cuda.func_call`, inline-CUDA function-call exemption, or edit under
`tirx_kernels/kern/` is allowed.

## Source-order warp program

| warps | role | program and edges |
| --- | --- | --- |
| 0-3 | softmax | consume score TMEM; partial-mask only the partial prefix; online max/sum; publish old/new max in TMEM and BF16/FP16 P; publish final sum in SMEM and optional LSE |
| 4-7 | correction/epilogue | consume max-change tokens; rescale prior O in TMEM; consume final sum; normalize and use fixed TMA S2G or packed predicated R2G |
| 8 | MMA plus TMEM lifetime | allocate 512 TMEM columns; issue two K128 QK iterations per N block; interleave prior P x V; commit score/O tokens; release Q; drain and deallocate |
| 9 | Q/K TMA producer | prefetch Q/K maps; load two Q K128 stages once per work tile and two K stages per selected N128 block |
| 10 | CLC scheduler | issue try-cancel/prefetch and publish the next persistent work descriptor |
| 11 | V TMA producer | prefetch V map; load two Dv128 V stages per selected N128 block |

All roles independently decode the same CLC work record. In CTA2,
`physical_m_block = logical_m_block*2 + cluster_cta_rank`; the leader MMA
warp issues CTA-group-2 operations while both CTAs execute their local loader,
softmax, correction, and scheduler-consumer state. The nonleader MMA warp
drains descriptors but does not issue duplicate MMA.

## Primitive vocabulary

The sketch uses scalar names for all logical mappings:

```python
linear_smem(bytes, alignment)
reg_array(dtype, elements)
smem_byte(base, stage, scalar)
tmem_addr(base, column, warp_row_band)
tensor_map(rank, extents, byte_strides, box, dtype, swizzle)
g2s_tma(map, coords, destination, completion_barrier, multicast)
s2g_tma(map, coords, source)
t2r(address, registers); r2t(registers, address)
r2g(registers, pointer, predicate)
mma_ss(destination, a_desc, b_desc, descriptor, accumulate)
mma_ts(destination, a_tmem, b_desc, descriptor, accumulate)
init(barrier, count); wait(barrier, phase); arrive(barrier)
expect_bytes(barrier, bytes); commit(barrier)
```

Descriptor construction, swizzles, row ownership, stages, and phases are
integer expressions. Operations never hide a compute, move, wait, or store.

## Complete sketch

~~~python
# 1. Compile-time specialization and host-side integer mappings.
def make_kernel(dtype, cta_group_size, sequence_mode, has_lse, strides):
    assert dtype in (BF16, FP16)
    assert cta_group_size in (1, 2)
    assert DQK == 256 and DV == 256
    assert Hq > 0 and Hkv > 0 and Hq % Hkv == 0
    assert pack_gqa is False
    q_per_kv = Hq // Hkv

    # These scalar functions are the only global-layout authority. "token" is
    # b*S+s for fixed BSHD and the packed token for THD.
    def q_element(token, q_head, d):
        return token*strides.q_token + q_head*strides.q_head + d*strides.q_d
    def k_element(token, kv_head, d):
        return token*strides.k_token + kv_head*strides.k_head + d*strides.k_d
    def v_element(token, kv_head, d):
        return token*strides.v_token + kv_head*strides.v_head + d*strides.v_d
    def o_element(token, q_head, d):
        return token*strides.o_token + q_head*strides.o_head + d*strides.o_d
    # Tensor-map sequence coordinates are sample-local in fixed mode and
    # cumulative in packed mode. Fixed has a real batch stride; packed keeps
    # the batch axis for the source rank but encodes its byte stride as zero.
    map_q_tokens = Sq if sequence_mode == "fixed" else total_q
    map_k_tokens = Sk if sequence_mode == "fixed" else total_k
    q_batch_stride = Sq*Hq*256*2 if sequence_mode == "fixed" else 0
    kv_batch_stride = Sk*Hkv*256*2 if sequence_mode == "fixed" else 0

    # Source-normalized coordinates. Q/O have axes
    # (d,q-head-within-kv-head,token,kv-head,batch); K/V have (d,token,kv-head,batch).
    q_map = tensor_map(
        rank=5, dtype=dtype, swizzle=SWIZZLE_128B,
        extents=(256,q_per_kv,map_q_tokens,Hkv,B),
        byte_strides=(512,Hq*512,q_per_kv*512,q_batch_stride),
        box=(64,1,128*cta_group_size,1,1))
    k_map = tensor_map(
        rank=4, dtype=dtype, swizzle=SWIZZLE_128B,
        extents=(256,map_k_tokens,Hkv,B), byte_strides=(Hkv*512,512,kv_batch_stride),
        box=(64,128,1,1))
    v_map = tensor_map(
        rank=4, dtype=dtype, swizzle=SWIZZLE_128B,
        extents=(256,map_k_tokens,Hkv,B), byte_strides=(Hkv*512,512,kv_batch_stride),
        box=((64,128,1,1) if cta_group_size == 1 else (128,128,1,1)))
    if sequence_mode == "fixed":
        o_map = tensor_map(
            rank=5, dtype=dtype, swizzle=SWIZZLE_128B,
            extents=(256,q_per_kv,map_q_tokens,Hkv,B),
            byte_strides=(512,Hq*512,q_per_kv*512,q_batch_stride),
            box=(64,1,128,1,1))
    # instruction_selection: runtime.cuTensorMapEncodeTiled in prepare;
    # extent: Q/K/V and fixed-only O, outside launch and timed region.

@kernel(
    target=("sm_100a","sm_103a"),
    grid=round_up(plan_clc_grid,(cta_group_size,1,1)),
    block=(384,1,1),
    cluster=((2,1,1) if cta_group_size == 2 else None),
    min_blocks_per_sm=1,
    dynamic_smem_bytes={
        (1,"fixed"):230272, (1,"varlen"):197504,
        (2,"fixed"):164736, (2,"varlen"):131968,
    }[cta_group_size,sequence_mode],
)
# instruction_selection: PTX 9.3 sm_103a, .reqntid 384,1,1,
# .minnctapersm 1 and .extern .shared .align 1024.
def flex_attention_forward_hd256(
    q,k,v,out,lse,cu_q,cu_k,sequence_desc,
    mask_block_cnt,mask_block_offset,mask_block_idx,
    full_block_cnt,full_block_offset,full_block_idx,
    mask_payload,fwd_work_desc,clc_grid,
    softmax_scale_log2,softmax_scale,
):
    tid=thread_id(); warp=warp_uniform(tid>>5); lane=tid&31; tid128=tid&127
    cta_rank=cluster_cta_rank_x() if cta_group_size == 2 else 0
    is_leader_cta=cta_rank == 0

    # 2. All source descriptor prefetches precede pipeline/barrier creation,
    # shared-tensor address construction, and cluster wait.
    if warp == 9:
        prefetch_tensormap(q_map); prefetch_tensormap(k_map)
    if warp == 11: prefetch_tensormap(v_map)
    if sequence_mode == "fixed" and warp == 4:
        prefetch_tensormap(o_map)

    # 3. Exact rank-one shared arena and scalar address functions.
    arena=linear_smem(dynamic_smem_bytes,alignment=1024)
    Q_FULL=(0,8); Q_EMPTY=(16,24)
    K_FULL=(32,40); K_EMPTY=(48,56)
    V_FULL=(64,72); V_EMPTY=(80,88)
    S_FULL=(96,104); S_EMPTY=(112,120)
    P_FULL=(128,136); P_EMPTY=(144,152)
    STATS_FULL=(160,168); STATS_EMPTY=(176,184)
    SUM_FULL=192; SUM_EMPTY=200; O_FULL=208; O_EMPTY=216
    TMEM_DEALLOC=224; TMEM_MAILBOX=232
    CLC_FULL=240; CLC_EMPTY=248; CLC_RESPONSE=256
    SQ=384; SK=65920; SV=131456 if cta_group_size == 1 else 98688
    if sequence_mode == "fixed":
        SO=196992 if cta_group_size == 1 else 131456
        SSUM=229760 if cta_group_size == 1 else 164224
    else:
        SO=-1; SSUM=196992 if cta_group_size == 1 else 131456
    # Q=2*32768 B. K/V stage=32768 B CTA1 or 16384 B/CTA CTA2.
    # Fixed sO=32768 B/CTA; sSum=512 B.
    q_stage_bytes=32768
    kv_stage_bytes=32768 if cta_group_size == 1 else 16384

    def sw128_band_byte(base,major,minor):
        # One 64-BF16-column x 128-row band is 16384 B.
        raw=major*128+minor*2
        return base+(raw ^ ((raw>>3)&0x70))
    def q_smem_byte(stage,row,k):
        band=k//64; minor=k-band*64
        return sw128_band_byte(SQ+stage*q_stage_bytes+band*16384,row,minor)
    def k_smem_byte(stage,n,k):
        band=k//64; minor=k-band*64
        rows=128 if cta_group_size == 1 else 64
        local_n=n if cta_group_size == 1 else n-cta_rank*64
        return sw128_band_byte(SK+stage*kv_stage_bytes+band*rows*128,local_n,minor)
    def v_smem_byte(stage,n,d):
        if cta_group_size == 1:
            band=d//64; minor=d-band*64
            return sw128_band_byte(SV+stage*kv_stage_bytes+band*16384,n,minor)
        # Group-2 V is one collective 128x128 transaction; each CTA owns one
        # 64-column 16-KiB slice selected by cluster rank.
        return sw128_band_byte(SV+stage*kv_stage_bytes,n,d-cta_rank*64)
    def o_smem_byte(row,d):
        band=d//64; minor=d-band*64
        return sw128_band_byte(SO+band*16384,row,minor)

    # Raw 16-B-unit descriptors; each TCGEN phase advances K by 32 elements,
    # hence +2. Stage addresses, not layout objects, select Q/K/V halves.
    def smem_desc(base_byte,ldo16,sdo16):
        addr16=cvta_generic_to_shared(base_byte)>>4
        return ((addr16&0x3fff)|((ldo16&0x3fff)<<16)|
                ((sdo16&0x3fff)<<32)|(1<<46)|(3<<61))
    # The exact SW128 selector is 3 in descriptor bits [61,64).
    def q_desc(stage,kphase):
        return smem_desc(SQ+stage*q_stage_bytes,1024,64)+2*kphase
    def k_desc(stage,kphase):
        return smem_desc(SK+stage*kv_stage_bytes,1024,64)+2*kphase
    def v_desc(stage,kphase):
        return smem_desc(SV+stage*kv_stage_bytes,1024,64)+2*kphase
    def tmem_score(stage,row,col32):
        return tmem_base+stage*128+(row<<16)+col32
    def tmem_stats(score_stage,row):
        return tmem_base+score_stage*128+64+(row<<16)
    def tmem_o(dv_iter,row,col32):
        return tmem_base+256+dv_iter*128+(row<<16)+col32

    # Exact tensor-map coordinates are distinct from flat element offsets.
    def map_sequence_origins(info):
        q_s=(cu_q[info.batch] if sequence_mode == "varlen" else 0)+info.physical_m*128
        k_s=cu_k[info.batch] if sequence_mode == "varlen" else 0
        qh=info.head; kh=info.head//q_per_kv
        # packed descriptors encode batch stride zero, fixed descriptors use
        # sample-local q_s/k_s and the explicit batch coordinate.
        return q_s,k_s,qh,kh
    def q_coords(info,qk_iter):
        q_s,_,qh,kh=map_sequence_origins(info)
        return (qk_iter*128,qh-kh*q_per_kv,q_s,kh,info.batch)
    def k_coords(info,nblock,qk_iter):
        _,k_s,_,kh=map_sequence_origins(info)
        return (qk_iter*128,k_s+nblock*128,kh,info.batch)
    def v_coords(info,nblock,dv_iter):
        _,k_s,_,kh=map_sequence_origins(info)
        return (dv_iter*128,k_s+nblock*128,kh,info.batch)
    def o_coords(info,dv_iter):
        q_s,_,qh,kh=map_sequence_origins(info)
        return (dv_iter*128,qh-kh*q_per_kv,q_s,kh,info.batch)
    def flat_q_token(info,row):
        if sequence_mode == "varlen": return cu_q[info.batch]+info.physical_m*128+row
        return info.batch*Sq+info.physical_m*128+row
    def packed_o_element(info,row,d):
        return o_element(flat_q_token(info,row),info.head,d)
    def lse_element(info,row):
        token=flat_q_token(info,row)
        if sequence_mode == "varlen": return info.head*total_q+token
        return (info.batch*Hq+info.head)*Sq+info.physical_m*128+row
    def row_store_valid(info,row):
        return cta_rank*128+row < info.q_valid

    # 4. Independent pipeline cursors. Two-stage parity changes only on wrap;
    # one-stage parity changes on every reuse.
    def advance2(index,phase):
        index+=1
        if index == 2: index=0; phase^=1
        return index,phase
    def advance1(phase): return phase^1
    q_prod=(0,1); k_prod=(0,1); v_prod=(0,1)
    q_cons=(0,0); k_cons=(0,0); v_cons=(0,0)
    s_prod=(0,1); s_cons=(0,0); p_prod=(0,1); p_cons=(0,0)
    stats_prod=(0,1); stats_cons=(0,0)
    sum_prod=1; sum_cons=0; o_prod=1; o_cons=0
    clc_prod=1; clc_cons=0

    if warp == 0 and elect_one():
        for stage in static_range(2):
            init(Q_FULL[stage],1); init(Q_EMPTY[stage],1)
            init(K_FULL[stage],1); init(K_EMPTY[stage],1)
            init(V_FULL[stage],1); init(V_EMPTY[stage],1)
            init(S_FULL[stage],1); init(S_EMPTY[stage],128*cta_group_size)
            init(P_FULL[stage],128*cta_group_size); init(P_EMPTY[stage],1)
            init(STATS_FULL[stage],128); init(STATS_EMPTY[stage],128)
        init(SUM_FULL,128); init(SUM_EMPTY,128)
        init(O_FULL,1); init(O_EMPTY,128*cta_group_size)
        init(CLC_FULL,1); init(CLC_EMPTY,384*cta_group_size)
    if cta_group_size == 2 and warp == 8 and elect_one():
        init(TMEM_DEALLOC,32)
    fence_mbarrier_init_release_cluster()
    if cta_group_size == 2: cluster_arrive_relaxed()

    # Source warp 10 register deallocation is before pipeline_init_wait.
    if warp == 10: setmaxnreg_decrease(72)
    if cta_group_size == 2: cluster_wait()
    else: cta_sync()
    # All other role changes occur at role entry after cluster wait.
    if warp in (8,9,11): setmaxnreg_decrease(72)
    if 0 <= warp < 4: setmaxnreg_increase(240)
    if 4 <= warp < 8: setmaxnreg_decrease(128)

    # 5. The CTA2 plan row is logical M256; physical_m only selects one M128
    # Q/O slice and one mask-payload subtile.
    def decode_plan_task(hw_tile_x,hw_valid):
        task=hw_tile_x//cta_group_size
        valid=hw_valid and task < fwd_work_desc.shape[0]
        safe=min(task,fwd_work_desc.shape[0]-1)
        logical_m=select(valid,fwd_work_desc[safe,0],0)
        head=select(valid,fwd_work_desc[safe,1],0)
        batch=select(valid,fwd_work_desc[safe,2],0)
        q_valid=select(valid,fwd_work_desc[safe,3],0)
        physical_m=logical_m*cta_group_size+cta_rank
        plan_head=0 if mask_block_cnt.shape[0] == 1 else head
        outer_row=sequence_desc[batch,4]+logical_m
        plan_row=plan_head*mask_block_cnt.shape[1]+outer_row
        partial_count=mask_block_cnt[plan_head,outer_row]
        full_count=full_block_cnt[plan_head,outer_row]
        partial_base=mask_block_offset[plan_row]
        full_base=full_block_offset[plan_row]
        return Work(valid,logical_m,physical_m,head,batch,q_valid,
                    partial_count,full_count,partial_base,full_base)
    def nblock(info,step):
        if info.partial_count+info.full_count == 0: return 0
        if step < info.partial_count:
            return mask_block_idx[info.partial_base+step]
        return full_block_idx[info.full_base+step-info.partial_count]
    def partial_payload_v4(info,step,row):
        # Runtime tensor is [partial_block,cta_subtile,row,4]. One aligned
        # ld.global.v4.b32 supplies all 128 predicate bits.
        linear=(((info.partial_base+step)*cta_group_size+cta_rank)*128+row)*4
        words=reg_array(U32,4)
        ptx_ld_global_v4_b32(words,mask_payload+4*linear)
        return words
    initial=decode_plan_task(block_idx_x(),True)

    # TCGEN producer commits and consumer releases use the same exact family.
    # CTA2 always multicasts to both peer ranks (mask 0b11), including P_EMPTY.
    def tcgen_signal(barrier,stage):
        if cta_group_size == 1:
            ptx_tcgen05_commit_group1_arrive_one_shared_cluster_b64(barrier[stage])
        else:
            ptx_tcgen05_commit_group2_arrive_one_shared_cluster_multicast_b64(
                barrier[stage],mask_u16=3)

    # 6. Q/K producer, warp 9.
    with role(warp == 9):
        work=initial
        while work.valid:
            if work.logical_m*(128*cta_group_size) < q_length(work.batch):
                for qk_iter in static_range(2):
                    qs,qp=q_prod; wait(Q_EMPTY[qs],qp)
                    expect_bytes(Q_FULL[qs],32768 if cta_group_size == 1 else 65536)
                    for band in static_range(2):
                        qc=q_coords(work,qk_iter); qc[0]=qc[0]+band*64
                        g2s_tma(q_map,qc,
                                SQ+qs*q_stage_bytes+band*16384,Q_FULL[qs],
                                multicast=(cta_group_size==2))
                    q_prod=advance2(qs,qp)
                loop_blocks=max(work.partial_count+work.full_count,1)
                for step in rolled_range(loop_blocks):
                    nb=nblock(work,step)
                    for qk_iter in static_range(2):
                        ks,kp=k_prod; wait(K_EMPTY[ks],kp)
                        expect_bytes(K_FULL[ks],32768)
                        for band in static_range(2):
                            kc=k_coords(work,nb,qk_iter); kc[0]=kc[0]+band*64
                            band_bytes=16384 if cta_group_size == 1 else 8192
                            g2s_tma(k_map,kc,
                                    SK+ks*kv_stage_bytes+band*band_bytes,K_FULL[ks],
                                    multicast=(cta_group_size==2))
                        k_prod=advance2(ks,kp)
            # Explicit CLC consumer sequence for this role.
            wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
            words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                   load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
            next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
            arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
            clc_cons=advance1(clc_cons); work=next_work
        # Source tail order: K then Q.
        wait(K_EMPTY[k_prod[0]],k_prod[1]); wait(Q_EMPTY[q_prod[0]],q_prod[1])
    # instruction_selection CTA1:
    # cp.async.bulk.tensor.{5d|4d}.shared::cta.global.tile.mbarrier::
    # complete_tx::bytes.L2::cache_hint
    # CTA2: same spelling with shared::cluster and trailing cta_group::2.
    # Static sites/transfer: Q and K each issue two SW128 bands.
    # Expected bytes: Q CTA1 32768, Q CTA2 65536, K 32768 both.

    # 7. V producer, warp 11.
    with role(warp == 11):
        work=initial
        while work.valid:
            if work.logical_m*(128*cta_group_size) < q_length(work.batch):
                for step in rolled_range(max(work.partial_count+work.full_count,1)):
                    nb=nblock(work,step)
                    for dv_iter in static_range(2):
                        vs,vp=v_prod; wait(V_EMPTY[vs],vp)
                        expect_bytes(V_FULL[vs],32768)
                        if cta_group_size == 1:
                            for band in static_range(2):
                                vc=v_coords(work,nb,dv_iter); vc[0]=vc[0]+band*64
                                g2s_tma(v_map,vc,
                                        SV+vs*kv_stage_bytes+band*16384,V_FULL[vs],
                                        multicast=False)
                        else:
                            g2s_tma(v_map,v_coords(work,nb,dv_iter),
                                    SV+vs*kv_stage_bytes,V_FULL[vs],multicast=True)
                        v_prod=advance2(vs,vp)
            wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
            words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                   load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
            next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
            arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
            clc_cons=advance1(clc_cons); work=next_work
        wait(V_EMPTY[v_prod[0]],v_prod[1])
    # instruction_selection: exact 4-D family; CTA1 issues two 64-column
    # bands, CTA2 one collective V transaction; expected bytes V=32768 both.

    # 8. TMEM allocation and MMA, warp 8.
    with role(warp == 8):
        alloc_tmem(TMEM_MAILBOX,columns=512,cta_group=cta_group_size)
        tmem_base=volatile_load_u32(TMEM_MAILBOX)
        work=initial
        while work.valid and is_leader_cta:
            if work.logical_m*(128*cta_group_size) < q_length(work.batch):
                loop_blocks=max(work.partial_count+work.full_count,1)
                q_release=q_cons.clone(); o_has_prior=False
                for step in rolled_range(loop_blocks):
                    ss,sp=s_prod; wait(S_EMPTY[ss],sp)
                    # Exactly 2 K128 iterations x 4 K32 phases = 8 QK MMA.
                    for qk_iter in static_range(2):
                        if step == 0:
                            qs,qp=q_cons; wait(Q_FULL[qs],qp)
                            q_stage=qs; q_cons=advance2(qs,qp)
                        else:
                            q_stage=q_release_stage(q_release,qk_iter)
                        ks,kp=k_cons; wait(K_FULL[ks],kp)
                        for kphase in static_range(4):
                            mma_ss(tmem_score(ss,0,0),q_desc(q_stage,kphase),
                                   k_desc(ks,kphase),QK_IDESC[dtype,cta_group_size],
                                   accumulate=(qk_iter != 0 or kphase != 0))
                        tcgen_signal(K_EMPTY,ks)
                        k_cons=advance2(ks,kp)
                    tcgen_signal(S_FULL,ss)
                    s_prod=advance2(ss,sp)

                    if step != 0:  # P/V for previous score block.
                        ps,pp=p_cons; wait(P_FULL[ps],pp); wait(O_EMPTY,o_prod)
                        for dv_iter in static_range(2):
                            vs,vp=v_cons; wait(V_FULL[vs],vp)
                            for kphase in static_range(4):
                                mma_ts(tmem_o(dv_iter,0,0),tmem_score(ps,0,0),
                                       v_desc(vs,kphase),PV_IDESC[dtype,cta_group_size],
                                       accumulate=(o_has_prior or kphase != 0))
                            tcgen_signal(V_EMPTY,vs)
                            v_cons=advance2(vs,vp)
                        tcgen_signal(O_FULL,0); o_prod=advance1(o_prod)
                        tcgen_signal(P_EMPTY,ps)
                        p_cons=advance2(ps,pp); o_has_prior=True

                for qk_iter in static_range(2):
                    qs,qp=q_release
                    tcgen_signal(Q_EMPTY,qs)
                    q_release=advance2(qs,qp)

                # Final P x V: again 2 Dv iterations x 4 phases = 8 MMA.
                ps,pp=p_cons; wait(P_FULL[ps],pp); wait(O_EMPTY,o_prod)
                for dv_iter in static_range(2):
                    vs,vp=v_cons; wait(V_FULL[vs],vp)
                    for kphase in static_range(4):
                        mma_ts(tmem_o(dv_iter,0,0),tmem_score(ps,0,0),
                               v_desc(vs,kphase),PV_IDESC[dtype,cta_group_size],
                               accumulate=(o_has_prior or kphase != 0))
                    tcgen_signal(V_EMPTY,vs)
                    v_cons=advance2(vs,vp)
                tcgen_signal(O_FULL,0); o_prod=advance1(o_prod)
                tcgen_signal(P_EMPTY,ps); p_cons=advance2(ps,pp)

            wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
            words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                   load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
            next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
            arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
            clc_cons=advance1(clc_cons); work=next_work
        # CTA2 nonleader only drains CLC; it issues no duplicate MMA.
        if not is_leader_cta:
            while work.valid:
                wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
                words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                       load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
                next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
                arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
                clc_cons=advance1(clc_cons); work=next_work
        wait(S_EMPTY[s_prod[0]],s_prod[1]); wait(O_EMPTY,o_prod)
        relinquish_tmem_allocation_permit(cta_group_size)
        named_barrier_arrive_wait(id=1,threads=288)
        dealloc_tmem(tmem_base,512,cta_group_size,TMEM_DEALLOC)
    # instruction_selection: tcgen05.mma.cta_group::{1|2}.kind::f16.
    # Per executed N block: 8 QK + 8 PV. Writer's 32 static sites are
    # 8 QK0 + 8 QKi + 8 PV-loop + 8 PV-final.

    # 9. Exact source/PTX scalar arithmetic (all extents frozen).
    def exp2_approx(x):
        y=local_f32(); ptx_ex2_approx_ftz_f32(y,x); return y
    def reciprocal_approx(x):
        y=local_f32(); ptx_rcp_approx_ftz_f32(y,x); return y
    def fast_log(x):
        y=local_f32(); ptx_lg2_approx_ftz_f32(y,x)
        return y*0.6931471805599453
    def reduce_max_exact_128(x,initial):
        a=reg_array(F32,4)
        ptx_max_f32(a[0],initial,x[0],x[1])
        ptx_max_f32(a[1],x[2],x[3]); ptx_max_f32(a[2],x[4],x[5])
        ptx_max_f32(a[3],x[6],x[7])
        for group in static_range(1,16):
            base=group*8
            for lane4 in static_range(4):
                ptx_max_f32(a[lane4],a[lane4],x[base+2*lane4],x[base+2*lane4+1])
        ptx_max_f32(a[0],a[0],a[1]); ptx_max_f32(a[0],a[0],a[2],a[3])
        return a[0]
    def exp2_source_mix(x):
        # The pinned source comment says polynomial emulation, but operation
        # evidence is authoritative: writer PTX has 262 ex2.approx.ftz.f32,
        # 129 fma.rn.f32x2, and zero add.rm/sub polynomial instructions. The
        # score path therefore emits one SFU exp2 for every scaled score.
        for i in static_range(128): x[i]=exp2_approx(x[i])
    def packed_sum_exact_128(x,old_sum,old_scale):
        carried=old_sum*old_scale
        quarter0=(carried,carried)
        quarter1=(0.0,0.0); quarter2=(0.0,0.0); quarter3=(0.0,0.0)
        # logical_divide(fragment,32): four contiguous 32-value quarters.
        for j in static_range(0,32,2):
            quarter0=add_rn_f32x2(quarter0,(x[0*32+j],x[0*32+j+1]))
            quarter1=add_rn_f32x2(quarter1,(x[1*32+j],x[1*32+j+1]))
            quarter2=add_rn_f32x2(quarter2,(x[2*32+j],x[2*32+j+1]))
            quarter3=add_rn_f32x2(quarter3,(x[3*32+j],x[3*32+j+1]))
        quarter0=add_rn_f32x2(quarter0,quarter1)
        quarter2=add_rn_f32x2(quarter2,quarter3)
        quarter0=add_rn_f32x2(quarter0,quarter2)
        return quarter0[0]+quarter0[1]


    # 10. One source-faithful softmax step. Score/P/stats cursors remain
    # independent; at this fixed-rate point their stage indices must agree.
    def softmax_step(work,mask_kind,step,row_max,row_sum):
        ss,sp=s_cons; wait(S_FULL[ss],sp)
        score=reg_array(F32,128)
        for frag in static_range(4):
            score[frag*32:(frag+1)*32]=t2r(tmem_score(ss,tid128,frag*32),f32x32)
        wait_tmem_loads(); arrive_remote(S_EMPTY[ss],cta_rank=0,count=1)
        s_cons=advance2(ss,sp)
        if mask_kind == EMPTY:
            for i in static_range(128): score[i]=-inf
        elif mask_kind == PARTIAL:
            payload=partial_payload_v4(work,step,tid128)
            for i in static_range(128):
                if ((payload[i>>5]>>(i&31))&1)==0: score[i]=-inf
        # FULL is called from its own loop and emits no payload load, bit-test,
        # predicate, or masked-score select.
        old_max=row_max; row_max=reduce_max_exact_128(score,row_max)
        safe_max=0.0 if row_max == -inf else row_max
        sts,stp=stats_prod; assert sts == ss; wait(STATS_EMPTY[sts],stp)
        r2t((old_max,safe_max),tmem_stats(ss,tid128)); wait_tmem_stores()
        commit(STATS_FULL[sts]); stats_prod=advance2(sts,stp)
        ps,pp=p_prod; assert ps == ss; wait(P_EMPTY[ps],pp)
        for pair in static_range(64):
            score[2*pair:2*pair+2]=fma_f32x2(
                score[2*pair:2*pair+2],(softmax_scale_log2,)*2,
                (-safe_max*softmax_scale_log2,)*2)
        exp2_source_mix(score)
        probability=convert_pairs(score,dtype,rounding="rn")
        r2t(probability,tmem_score(ps,tid128,0)); wait_tmem_stores()
        commit(P_FULL[ps]); p_prod=advance2(ps,pp)
        acc_scale=exp2_approx(softmax_scale_log2*(old_max-safe_max))*0.5
        row_sum=packed_sum_exact_128(score,row_sum,acc_scale)
        return row_max,row_sum

    # 11. Softmax, warps 0..3: distinct empty/partial/full control flow.
    with role(0 <= warp < 4):
        tmem_base=retrieve_tmem(TMEM_MAILBOX); work=initial
        while work.valid:
            if work.logical_m*(128*cta_group_size) < q_length(work.batch):
                row_max=-inf; row_sum=0.0
                total=work.partial_count+work.full_count
                if total == 0:
                    row_max,row_sum=softmax_step(work,EMPTY,0,row_max,row_sum)
                else:
                    for step in rolled_range(0,work.partial_count):
                        row_max,row_sum=softmax_step(work,PARTIAL,step,row_max,row_sum)
                    for step in rolled_range(work.partial_count,total):
                        row_max,row_sum=softmax_step(work,FULL,step,row_max,row_sum)
                wait(SUM_EMPTY,sum_prod); store_shared_f32(SSUM+4*tid128,row_sum)
                fence_proxy_async_shared_cta(); commit(SUM_FULL); sum_prod=advance1(sum_prod)
                if has_lse and row_store_valid(work,tid128):
                    value=(-inf if row_sum==0.0 or isnan(row_sum) else
                           softmax_scale*row_max+fast_log(row_sum))
                    store_global_f32(lse_element(work,tid128),value)
            wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
            words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                   load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
            next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
            arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
            clc_cons=advance1(clc_cons); work=next_work
        wait(P_EMPTY[p_prod[0]],p_prod[1]); wait(STATS_EMPTY[stats_prod[0]],stats_prod[1])
        named_barrier_arrive(id=1,threads=288)

    # 12. Correction/output, warps 4..7.
    with role(4 <= warp < 8):
        tmem_base=retrieve_tmem(TMEM_MAILBOX); corr_row=tid128; work=initial
        while work.valid:
            if work.logical_m*(128*cta_group_size) < q_length(work.batch):
                loop_blocks=max(work.partial_count+work.full_count,1)
                # First stats token is consumed without rescale.
                sts,stp=stats_cons; wait(STATS_FULL[sts],stp)
                arrive(STATS_EMPTY[sts]); stats_cons=advance2(sts,stp)
                for step in rolled_range(1,loop_blocks):
                    sts,stp=stats_cons; wait(STATS_FULL[sts],stp)
                    old_max,new_max=t2r(tmem_stats(sts,corr_row),f32x2)
                    wait_tmem_loads()
                    scale=exp2_approx(softmax_scale_log2*(old_max-new_max))
                    arrive(STATS_EMPTY[sts]); stats_cons=advance2(sts,stp)
                    wait(O_FULL,o_cons)
                    for dv_iter in static_range(2):
                        for frag16 in static_range(8):
                            addr=tmem_o(dv_iter,corr_row,frag16*16)
                            values=mul_f32x2(t2r(addr,f32x16),scale); r2t(values,addr)
                    wait_tmem_stores(); arrive_remote(O_EMPTY,cta_rank=0,count=1)
                    o_cons=advance1(o_cons)

                wait(SUM_FULL,sum_cons); row_sum=load_shared_f32(SSUM+4*corr_row)
                fence_proxy_async_shared(); arrive(SUM_EMPTY); sum_cons=advance1(sum_cons)
                out_scale=0.0 if row_sum==0.0 or isnan(row_sum) else reciprocal_approx(row_sum)
                wait(O_FULL,o_cons)
                if sequence_mode == "fixed":
                    # Exactly 8 x32 TCGEN loads total: 4 per Dv iteration.
                    for dv_iter in static_range(2):
                        for frag32 in static_range(4):
                            values=mul_f32x2(
                                t2r(tmem_o(dv_iter,corr_row,frag32*32),f32x32),out_scale)
                            words=convert_pairs(values,dtype,rounding="rn")
                            for vec in static_range(4):
                                # Eight dtype values = four packed u32 = 16 B.
                                ptx_st_shared_v4_b32(
                                    o_smem_byte(corr_row,frag32*32+vec*8),
                                    words[vec*4+0],words[vec*4+1],
                                    words[vec*4+2],words[vec*4+3])
                        # Exact one-stage order; no acquire before initial fill.
                        fence_proxy_async_shared_cta()
                        named_barrier_arrive_wait(id=2,threads=128)
                        if warp == 4:
                            for band in static_range(2):
                                oc=o_coords(work,dv_iter); oc[0]=oc[0]+band*64
                                s2g_tma(o_map,oc,SO+band*16384)
                            commit_bulk_group()
                            acquire_wait_bulk_group_read(stage=0)
                        named_barrier_arrive_wait(id=2,threads=128)
                else:
                    for dv_iter in static_range(2):
                        for frag32 in static_range(4):
                            values=mul_f32x2(
                                t2r(tmem_o(dv_iter,corr_row,frag32*32),f32x32),out_scale)
                            words=convert_pairs(values,dtype,rounding="rn")
                            for j in static_range(32):
                                r2g_scalar(words,j,
                                    packed_o_element(work,corr_row,dv_iter*128+frag32*32+j),
                                    predicate=row_store_valid(work,corr_row))
                arrive_remote(O_EMPTY,cta_rank=0,count=1); o_cons=advance1(o_cons)
            wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
            words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                   load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
            next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
            arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
            clc_cons=advance1(clc_cons); work=next_work
        if sequence_mode == "fixed" and warp == 4: tail_bulk_group()
        named_barrier_arrive(id=1,threads=288)
    # fixed instruction_selection: tcgen05.ld...x32.b32, SW128 stores,
    # fence.proxy.async.shared::cta, named bar 2, 5-D TMA S2G.

    # 13. CLC scheduler warp. It is both leader producer and normal consumer;
    # its producer and consumer parities are independent.
    with role(warp == 10):
        work=initial
        if is_leader_cta:
            while work.valid:
                # producer_acquire, expect, try-cancel, producer advance.
                wait(CLC_EMPTY,clc_prod); expect_bytes(CLC_FULL,16)
                if elect_one():
                    try_cancel_async_multicast_all(CLC_RESPONSE,CLC_FULL,
                                                   cluster=cta_group_size)
                clc_prod=advance1(clc_prod)
                # consumer wait/read/decode/release/advance.
                wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
                words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                       load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
                next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
                arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
                clc_cons=advance1(clc_cons); work=next_work
            wait(CLC_EMPTY,clc_prod)  # producer_tail
        else:
            while work.valid:
                wait(CLC_FULL,clc_cons); fence_proxy_async_shared()
                words=(load_u32(CLC_RESPONSE),load_u32(CLC_RESPONSE+4),
                       load_u32(CLC_RESPONSE+8),load_u32(CLC_RESPONSE+12))
                next_work=decode_plan_task(query_cancel_tile_x(words),query_cancel_valid(words))
                arrive_remote(CLC_EMPTY,cta_rank=0,count=1)
                clc_cons=advance1(clc_cons); work=next_work
    # instruction_selection:
    # clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::
    # complete_tx::bytes.multicast::cluster::all.b128 plus query_cancel.
    return
~~~

## Source-to-target operation map

| source operation | target operation |
| --- | --- |
| `PipelineTmaUmma` Q/K/V | explicit two-stage full/empty mbarriers and raw TMA G2S |
| `PipelineUmmaAsync` score | explicit S full/empty plus TCGEN commit |
| `PipelineAsyncUmma` P | explicit P full/empty around TMEM probability stores |
| `PipelineAsync` stats/sum | explicit per-thread arrivals and phase waits |
| `PipelineUmmaAsync` O | explicit O ownership full/empty token |
| `TmemAllocator` | CTA1/CTA2 alloc, mailbox, named lifetime barrier, relinquish/dealloc |
| `PlanClcPersistentTileSchedulerSm100` | preserved CLC cancel/response persistent loop |
| `load_mask_payload/apply_loaded_arbitrary_mask` | one vector u32x4 load and 128 explicit bit predicates |
| fixed `correction_epilog_tma` | explicit TMEM-to-linear-SMEM conversion and TMA S2G |
| varlen `correction_epilog` | explicit TMEM-to-register conversion and predicated R2G |
| `store_sum_max` | exact row sum publication and optional FP32 LSE store |

## Correctness and performance contract

Correctness compares TIRx directly with the pinned source using identical plan
tensors, output layouts, and PTX 9.3 target on GB300. O and enabled LSE are
first checked bitwise; any mismatch is reported with exact max ULP/absolute/
relative error and location, with no permissive tolerance hiding it. A small
independent FP32 oracle, guard zones, input/plan immutability, deterministic
replay, fixed/packed tails, real ragged batches, per-head masks, MHA/GQA,
partial/full/empty rows, both dtypes, both CTA modes, and no-LSE sentinel
preservation are required before reviewer PASS.

The performance gate starts only after numerical correctness and the required
correctness reviewer both pass. From that point no reviewer/subagent is
launched and this sketch is never edited. The sole acceptance measurement is
`bench_suite` with source and TIRx both compiling PTX 9.3 for `sm_103a`.
The final LSE-on matrix is 8 masks x 2 dtypes x 2 CTA modes x 2 sequence modes
x 2 head modes = 128 rows at 131072 tokens, and every source/TIRx ratio must be
strictly greater than 0.99. Targeted worst-case loops precede the final full
suite; profiler data is diagnostic only.
