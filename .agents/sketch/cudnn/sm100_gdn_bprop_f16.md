<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This design sketch documents a modified TIRx port of cuDNN Frontend's
python/cudnn/linear_attention/frost/kernel/gdn_bprop_f16.py at commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5.
-->

# cuDNN SM100 GDN BF16/FP16 backward: coarse WASP pipeline sketch

This is a non-executable execution sketch, not Python, a builder API, a
mathematical reference, or an alternate implementation.  The implementation it
describes belongs in
[`tirx_kernels/cudnn/linear_attention/gdn_bprop_f16.py`](../../../tirx_kernels/cudnn/linear_attention/gdn_bprop_f16.py),
which becomes the executable source of truth after the correctness gate.

The frozen source is the two-launch `chunk_gdn_bwd_sm100` / `run_bwd` path.
The anchor specialization is BF16 I/O, `log_gate=True`, `(HQ,HK,HV)=(4,1,1)`,
`BT=64`, `DK=DV=128`, FP32 state accumulation, no initial/final-state option,
one CTA per cluster, 384 main threads, 1024 prologue threads, 512 TMEM columns,
232448 bytes of main dynamic SMEM, and a persistent grid of 152 CTAs on the
writer's GB200.  Accepted compiled variants are BF16/FP16, GQA/GVA head ratios,
ragged and zero-length sequences, raw/log/safe gate, beta post-sigmoid or
in-kernel sigmoid, optional initial/final state gradients, static/dynamic
scheduling, scratch/generated prologue ordering, non-default scale, and i32/i64
`cu_seqlens`.  Engine-owned Q/K normalization, checkpoint recomputation,
grouped-head reduction, and safe-gate parameter reduction are outside this
direct-kernel port and outside both timed closures.

Writer line-info evidence is preserved in
`.porting/gdn_bprop_f16/source_lineinfo/`.  The main PTX has SHA256
`12c3003283614044b611ece116dfdbf481d994e285cc34cf66775aa4bea1be75`,
20086 lines and 5636 `.loc` records; the prologue PTX has SHA256
`b0f531ac428d1e5b16a0c38107f9c2232a934202f02ba5c606b59cd605847a7b`,
764 lines and 166 `.loc` records.  Both `.file` tables point directly into the
pinned Python source and its helpers, so no generated intermediate source is
missing.  Counts below are instruction lines minus predicated lines.

## Pipeline at a glance

| Warps | Register target | Source-order role | Principal publication/reuse edges |
| --- | ---: | --- | --- |
| 0..3 | 192 | CG0: T-pairwise decay, KK/A epilogues, 8→16→32→64 inverse, dQ/dK scaling and folding, dA/dM masks, scalar dGate/dBeta M-terms | gate/beta, KK/A/dA/dM TMEM, T-inverse, dQ/dK shared staging |
| 4..7 | 248 | CG1: dstate seed/rescale/restage, dO'/dU/dY' TMEM inputs, `Y=V-cumprod*(K@state)`, dY/dV, dGate/dBeta V-terms, dQ dot, dK fold, state drain | V/dO/state, five sequential dV/dK accumulations, dstate and output staging |
| 8 | 64 | sole tcgen05 issuer: allocate 512 columns, issue the complete 17-phase reverse-chunk GEMM schedule, then deallocate | all TMEM accumulator/input barriers and split raw-buffer releases |
| 9 | 64 | persistent scheduler and TensorMap loads for Q/K/V/dO/checkpoint | Q/K/V/dO/state TMA full edges and dynamic scheduler ring |
| 10 | 64 | first/next scalar gate/beta prefetch, gate-domain prefix, reverse dGate suffix, and in-place dGate/dBeta stores | two-stage gate/beta rings; final `gate_done`/`beta_done` reuse edges |
| 11 | 64 | current-chunk TensorMap stores in dV -> dQ -> dK order | three output staging rings and immediate bulk-group 2/1/0 releases |

The source dispatch order is warps 0..3, warps 4..7, warp 8, warp 9, warp 10,
then warp 11.  Every role owns independent pipeline cursors that are initialized
once per kernel and continue across persistent work items.  Dynamic scheduling
uses the warp-9 producer and eleven consumer arrivals; the static variant emits
no ticket atomic and advances by `gridDim.x`.

## Primitive vocabulary and translation boundary

All storage is one-dimensional.  Matrix names below are comments on scalar
byte/column formulas, never layout values, tile primitives, fragments, or
first-class views.

```python
linear_buffer(space, dtype, elements, byte_offset, alignment, lifetime)
reg_array(dtype, elements)
smem_byte(base, stage, row, col, stage_bytes, segment_bytes, elem_bytes)
tmem_cell(base_column, row, column)      # base + column + (row << 16)
gmem_element(base, scalar_index)
descriptor_slot(workspace, array, batch) # (array*B+batch)*128 bytes
raw_smem_descriptor(base_byte, leading_bytes, stride_bytes, swizzle_code)
```

Directional movement operations are `copy_p2g`, `copy_g2s`, `copy_s2g`,
`copy_g2r`, `copy_r2g`, `copy_s2r`, `copy_r2s`, `copy_t2r`, and `copy_r2t`.
Basic computation operations are `fill`, `cast`, `add`, `sub`, `mul`, `fma`,
`rcp`, `exp2`, `tanh`, `select`, `shuffle`, and `gemm`.  Barrier construction,
wait/arrive/expect/commit, proxy fences, CTA synchronization, stage/phase
advancement, register-budget changes, TensorMap replacement, store-group
commit/wait, and TMEM allocation are schedule operations.

The implementation imports device language only as
`import tirx_kernels.kern as K`.  It uses no `T`, `Tx`, `I`, `tirx.tile.*`,
`TilePrimitiveCall`, first-class layout, rank>1 SMEM, inline CUDA source/call,
or function-call exemption.  A single rank-1 `u8` arena is the only main shared
allocation; `K.smem_pool(base=arena)` owns only the protocol prefix and all
payload addresses are explicit scalar byte arithmetic.

## Complete non-executable sketch

```python
@specialize(
    IO={"bf16", "f16"}, ACC="f32", BT=64, DK=128, DV=128,
    Q_STAGES=1, K_STAGES=2, V_STAGES=1, DO_STAGES=1, STATE_STAGES=1,
    GATE_STAGES=2, BETA_STAGES=2, TINV_STAGES=1, A_STAGES=1,
    DQ_STAGES=1, DK_STAGES=1, DV_STAGES=1, SCHED_STAGES=2,
    TMEM_DSTATE_ACC_STAGES=1, TMEM_DVDK_ACC_STAGES=1,
    TMEM_DSTATE_INP_STAGES=1, TMEM_SHARED_ACC_STAGES=2,
    TMEM_SHARED_INP_STAGES=2, CLUSTER=(1,1,1),
    REGS=(CG0=192, CG1=248, SERVICE=64),
)
def gdn_bprop_factory(...):
    return descriptor_order_prologue, persistent_backward

# -----------------------------------------------------------------------
# Runtime tensors and head specialization
# -----------------------------------------------------------------------
q       : gmem[IO, total_T * HQ * DK]
k       : gmem[IO, total_T * HK * DK]
v       : gmem[IO, total_T * HV * DV]
gate    : gmem[f32, total_T * HO]       # raw alpha, ln(alpha), or safe logits
beta    : gmem[f32_or_IO, total_T * HO] # post-sigmoid or IO logits
do      : gmem[IO, total_T * HO * DV]
ckpt    : gmem[IO, rows * HO * DV * DK]
dq, dk, dv : gmem[IO, total_T * HO * {DK,DK,DV}]
dgate   : gmem[f32, total_T * HO]
dbeta   : gmem[f32_or_IO, total_T * HO]
dstate_in, dstate0_out : optional gmem[f32, B * HO * DV * DK]
cu_seqlens : gmem[i32_or_i64, B+1]
work_items : gmem[i32, rows*8]
work_count : gmem[i32, 1]
sched_ctr  : optional gmem[i32, 2]
tmaps      : gmem[u64, 8*B*16]

HO=max(HQ,HV); Q_RATIO=HO//HQ; K_RATIO=HO//HK; V_RATIO=HO//HV
FIRST_STATE_CHUNK = 0 if USE_INITIAL_STATE else 1
```

### Prologue launch

```python
@kernel(grid=(1,1,1), block=(1024,1,1), warps=32, target="sm_100a")
def descriptor_order_prologue(base_maps[8], workspace, cu_seqlens,
                              q,k,v,do,ckpt,dq,dk,dv,
                              item_scratch,work_count,work_items,sched_all,
                              batch_count,row_strides,checkpoint_every_n):
    tid=thread_id(); warp=tid//32
    if ORDER_IN_PROLOGUE:
        # Variant semantics only: this branch is inactive in the reviewed
        # anchor, so it carries no instruction-selection annotation.
        order_arena=linear_buffer(smem,u8,32776,0,16,"order pass")
        SKEY=0; SKEY_END=16384                    # 4096 i32 keys
        SIDX=16384; SIDX_END=32768                # 4096 i32 source indices
        SSPREAD=32768; SSPREAD_END=32776          # min/max i32 pair
        n=num_generated_rows if ORDER_GENERATE else work_count[0]
        if ORDER_GENERATE and tid==0: work_count[0]=num_generated_rows
        if n>4096:
            for dst in thread_strided_range(n):
                # Eight fields go directly from generation/global staging to
                # the final global work table; they never occupy shared memory.
                for field in range(8):
                    work_items[dst,field]=(generate_work_field(dst,field)
                                           if ORDER_GENERATE
                                           else item_scratch[dst,field])
        else:
            if tid==0:
                store_i32(SSPREAD+0,INT_MAX); store_i32(SSPREAD+4,INT_MIN)
            padded=next_power_of_two(n); cta_sync()
            local_min=INT_MAX; local_max=INT_MIN
            for i in thread_strided_fixed_capacity(4096):
                if i<n:
                    key=(generated_num_chunks(i) if ORDER_GENERATE
                         else item_scratch[i,5]-item_scratch[i,4])
                    store_i32(SKEY+4*i,key); store_i32(SIDX+4*i,i)
                    local_min=min(local_min,key); local_max=max(local_max,key)
                elif i<padded:
                    store_i32(SKEY+4*i,INT_MIN); store_i32(SIDX+4*i,i)
            atomic_shared_min(SSPREAD+0,local_min)
            atomic_shared_max(SSPREAD+4,local_max); cta_sync()
            if load_i32(SSPREAD+0)!=load_i32(SSPREAD+4):
                width=2
                while width<=padded:
                    distance=width//2
                    while distance>0:
                        for i in thread_strided_fixed_capacity(4096):
                            partner=i^distance
                            if i<padded and partner>i:
                                ki=load_i32(SKEY+4*i); kp=load_i32(SKEY+4*partner)
                                ascending=((i//width)&1)==0
                                if (ascending and ki<kp) or (not ascending and ki>kp):
                                    swap_i32(SKEY+4*i,SKEY+4*partner)
                                    swap_i32(SIDX+4*i,SIDX+4*partner)
                        cta_sync(); distance//=2
                    width*=2
            for dst in thread_strided_range(n):
                src=dst if load_i32(SSPREAD+0)==load_i32(SSPREAD+4) \
                    else load_i32(SIDX+4*dst)
                for field in range(8):
                    work_items[dst,field]=(generate_work_field(src,field)
                                           if ORDER_GENERATE
                                           else item_scratch[src,field])
        if tid==0:
            sched_all[0]=0; sched_all[1]=0; sched_all[2]=0; sched_all[3]=0

    # Arrays 0..3 load Q/K/V/dO, array 4 loads checkpoint, arrays 5..7 store
    # dQ/dK/dV.  Gate/Beta and their outputs are scalar GMEM traffic.
    for array in static_range(8):
        if warp==array and elected_lane():
            checkpoint_prefix=0
            for batch in dynamic_range(batch_count):
                bos=copy_g2r(cu_seqlens[batch]); eos=copy_g2r(cu_seqlens[batch+1])
                # instruction_selection: ld.global.s64 in the reviewed anchor;
                # extent: selected sequence endpoints (i32 is a non-annotated variant)
                copy_p2g(base_maps[array], descriptor_slot(workspace,array,batch))
                # instruction_selection: sixteen st.global.b64; extent: one 128-B map
                if array==4:
                    count=0 if eos==bos else ceil_div(eos-bos,64)
                    replace_address_and_dim2(checkpoint_prefix,count)
                    checkpoint_prefix += count
                else:
                    replace_address_and_dim2(bos,eos-bos)
                # instruction_selection: tensormap.replace.tile.global_address
                # and tensormap.replace.tile.global_dim; extent: one field each
            fence_tensormap_release()
            # instruction_selection: fence.proxy.tensormap::generic.release.gpu
```

Descriptor boxes are `[64,1,64]` for Q/K, `[64,1,64]` for V/dO/dQ/dK/dV in
their channel-major view, and `[64,128,1,1]` for checkpoints.  Every descriptor
uses 128-byte swizzle and a 128-byte workspace slot.

### Main launch, linear storage, and protocol

```python
@kernel(grid=(num_sms,1,1), block=(384,1,1), warps=12,
        cluster=(1,1,1), min_blocks_per_sm=1, target="sm_100a")
def persistent_backward(workspace,n_desc,a_log,dt_bias,gate,beta,dgate,dbeta,
                        cu_seqlens,dstate0,dstate_in,work_items,work_count,
                        sched_ctr,scale):
    tid=thread_id(); warp=warp_uniform(tid//32); lane=tid&31
    arena=linear_buffer(smem,u8,232448,0,1024,"whole CTA")

    # Reconciled from the writer PTX's concrete shared operands.
    BARRIERS=0;       BARRIERS_LAST_WORD=1048
    SCHED=1056;       SCHED_END=1064
    TMEM_MAILBOX=1072; TMEM_MAILBOX_END=1076
    CUMSUMLOG=1152;   CUMSUMLOG_END=1664       # 2 x 64 f32
    CUMPROD=1664;     CUMPROD_END=2176         # 2 x 64 f32
    BETA_S=2176;      BETA_S_END=2688           # 2 x 64 f32
    Q=3072;           Q_END=19456               # 1 x 64 x 128 IO
    K=19456;          K_END=52224               # 2 x 64 x 128 IO
    DO=52224;         DO_END=68608               # 1 x 64 x 128 IO
    STATE=68608;      STATE_END=101376           # 128 x 128 IO
    TINV=101376;      TINV_END=109568             # 64 x 64 IO
    KK=109568;        KK_END=117760               # 64 x 64 IO
    A_DA=117760;      A_DA_END=125952             # A then dA alias
    DM=125952;        DM_END=134144                # 64 x 64 IO
    V_U=134144;       V_U_END=150528              # V then U alias
    DSTATE=150528;    DSTATE_END=183296            # 128 x 128 IO
    DQ=183296;        DQ_END=199680                # 64 x 128 IO
    DK=199680;        DK_END=216064
    DV_DY=216064;     DV_DY_END=232448             # dV stage and dY operand
    assert DV_DY_END==232448

    # Scalar-reduction aliases are byte-addressed subranges of live payloads.
    # Their source barriers delimit overwrite from the primary tile lifetime.
    CG0_DGATE_SCRATCH=109568; CG0_DGATE_SCRATCH_END=110592 # 256 f32 in sKK
    CG1_V_SCRATCH=183296; CG1_V_SCRATCH_END=185344         # 512 f32 in current sDQ
    CG1_QDOT_SCRATCH=150528; CG1_QDOT_SCRATCH_END=151552   # 256 f32 in sDstate


    def xor128(row,col,elem_bytes):
        return col ^ ((row&7)*(16//elem_bytes))
    def io_byte(base,stage,row,channel):
        segment=channel//64; within=channel%64
        return base + stage*16384 + 2*(segment*4096 + row*64 + xor128(row,within,2))
    def f32_ring_byte(base,stage,row): return base + stage*256 + 4*row
    def state_byte(base,key,value):
        segment=value//64; within=value%64
        return base + 2*(segment*8192 + key*64 + xor128(key,within,2))
    def tmem_cell(base,row,col): return base+col+(row<<16)

    # Descriptor words are raw integer fields: 14-bit shared address,
    # lead/stride byte fields, bit 46, and swizzle code in bits 61..63.
    # No helper returns a first-class layout.

    # 75 physical init instructions span the declaration-ordered protocol.
    # Stages/arrival owners below are the source's exact values.
    PROTOCOL=(
      (q_ready,1,1,TMA),(q_mma_done,1,1,MMA),(q_cg1_done,1,128,CG1),
      (k_ready,2,1,TMA),(k_mma_done,2,1,MMA),(k_cg0_done,2,128,CG0),
      (v_ready,1,1,TMA),(v_mma_done,1,1,MMA),
      (do_ready,1,1,TMA),(do_mma_done,1,1,MMA),
      (state_ready,1,1,TMA),(state_mma_done,1,1,MMA),
      (gate_ready,2,32,GATE),(gate_done,2,256,CG0_CG1),
      (beta_ready,2,32,GATE),(beta_done,2,256,CG0_CG1),
      (dstate_acc_ready,1,1,MMA),(dstate_scale_done,1,128,CG1),
      (du_scale_ready,1,1,MMA),(du_scale_done,1,128,CG1),
      (du_total_ready,1,1,MMA),
      (dk_scale_ready,1,1,MMA),(dk_scale_done,1,128,CG0),
      (dk_attn_ready,1,1,MMA),(dk_attn_done,1,128,CG0),
      (dk_total_ready,1,1,MMA),(dk_total_done,1,128,CG1),
      (dq_scale_ready,1,1,MMA),(dq_scale_done,1,128,CG0),
      (dq_total_ready,1,1,MMA),(dq_total_done,1,128,CG1),
      (kk_acc_ready,1,1,MMA),(kk_acc_done,1,128,CG0),
      (a_acc_ready,1,1,MMA),(k_state_ready,1,1,MMA),
      (u_acc_ready,1,1,MMA),(dy_acc_ready,1,1,MMA),
      (da_acc_ready,1,1,MMA),(dm_acc_ready,1,1,MMA),(dm_acc_done,1,128,CG0),
      (dk_state_ready,1,1,MMA),
      (dstate_inp_ready,1,128,CG1),(dstate_inp_done,1,1,MMA),
      (do_prime_ready,1,128,CG1),(du_inp_ready,1,128,CG1),
      (dyp_inp_ready,1,128,CG1),(y_ready,1,128,CG1),
      (tinv_ready,1,128,CG0),(a_ready,1,128,CG0),(a_done,1,1,MMA),
      (u_ready,1,128,CG1),(dstate_smem_ready,1,128,CG1),
      (state_dot_done,1,128,CG0),(da_ready,1,128,CG0),
      (dbeta_cg1_ready,1,128,CG1),(dgate_cg1_ready,1,128,CG1),
      (dq_stage_ready,1,128,CG1),(dq_stage_done,1,32,EPI),
      (dk_stage_ready,1,128,CG1),(dk_stage_done,1,32,EPI),
      (dv_stage_ready,1,128,CG1),(dv_stage_done,1,32,EPI),
      (sdv_done,1,1,MMA),(tmem_done,1,256,CG0_CG1),
      (sched_ready,2,1,SCHED_PROD),(sched_done,2,11,SCHED_CONS),
    )
    for edge in PROTOCOL:
        if owner_warp(edge) and elected_lane(): init_all_stages(edge)
        # instruction_selection: mbarrier.init.shared.b64; extent: 75 words
    fence_mbarrier_init(); cta_sync()
    # instruction_selection: fence.mbarrier_init.release.cluster then bar.sync 0

    # Physical offsets are declaration-ordered 16-B-aligned allocations; only
    # the second element of a two-stage ring occupies base+8.
    BARRIER_OFFSETS=(
      (q_ready,(0,)),(q_mma_done,(16,)),(q_cg1_done,(32,)),
      (k_ready,(48,56)),(k_mma_done,(64,72)),(k_cg0_done,(80,88)),
      (v_ready,(96,)),(v_mma_done,(112,)),(do_ready,(128,)),
      (do_mma_done,(144,)),(state_ready,(160,)),(state_mma_done,(176,)),
      (gate_ready,(192,200)),(gate_done,(208,216)),
      (beta_ready,(224,232)),(beta_done,(240,248)),
      (dstate_acc_ready,(256,)),(dstate_scale_done,(272,)),
      (du_scale_ready,(288,)),(du_scale_done,(304,)),(du_total_ready,(320,)),
      (dk_scale_ready,(336,)),(dk_scale_done,(352,)),
      (dk_attn_ready,(368,)),(dk_attn_done,(384,)),
      (dk_total_ready,(400,)),(dk_total_done,(416,)),
      (dq_scale_ready,(432,)),(dq_scale_done,(448,)),
      (dq_total_ready,(464,)),(dq_total_done,(480,)),
      (kk_acc_ready,(496,)),(kk_acc_done,(512,)),(a_acc_ready,(528,)),
      (k_state_ready,(544,)),(u_acc_ready,(560,)),(dy_acc_ready,(576,)),
      (da_acc_ready,(592,)),(dm_acc_ready,(608,)),(dm_acc_done,(624,)),
      (dk_state_ready,(640,)),(dstate_inp_ready,(656,)),
      (dstate_inp_done,(672,)),(do_prime_ready,(688,)),
      (du_inp_ready,(704,)),(dyp_inp_ready,(720,)),(y_ready,(736,)),
      (tinv_ready,(752,)),(a_ready,(768,)),(a_done,(784,)),
      (u_ready,(800,)),(dstate_smem_ready,(816,)),(state_dot_done,(832,)),
      (da_ready,(848,)),(dbeta_cg1_ready,(864,)),(dgate_cg1_ready,(880,)),
      (dq_stage_ready,(896,)),(dq_stage_done,(912,)),
      (dk_stage_ready,(928,)),(dk_stage_done,(944,)),
      (dv_stage_ready,(960,)),(dv_stage_done,(976,)),(sdv_done,(992,)),
      (tmem_done,(1008,)),(sched_ready,(1024,1032)),
      (sched_done,(1040,1048)),
    )

    # One 512-column TMEM allocation, exactly filled.
    DSTATE_ACC=0       # 128 FP32 columns
    DVDK_ACC=128       # 64 FP32 columns, five sequential productions
    DSTATE_INP=192     # 64 packed-IO columns
    SHARED_ACC0=256    # 64 FP32 columns: KK -> K_state -> dY -> dM
    SHARED_ACC1=320    # 64 FP32 columns: A -> U -> dA
    SHARED_INP0=384    # 32 packed-IO columns: dO' then dK-state accumulator alias
    SHARED_INP1=416    # 32 packed-IO columns: dU then dY'
    Y=448               # packed Y columns 448..479
    G_K_STATE=480       # packed g_k_state columns 480..511
    Y_GK_QT=448         # later Q^T aliases the full 448..511 range
```

Every cursor below advances immediately after capturing its current slot. A
one-stage cursor therefore flips parity on every use; a two-stage cursor flips
parity after two uses. This table is the complete source declaration set.

| role | cursors `(initial phase; stages)` | advancement point |
| --- | --- | --- |
| warp 11 | `dq,dk,dv=(0;1)`, `sched=(0;2)` | after each current-chunk ready wait; scheduler after each work item |
| warp 10 | `gate_load,beta_load=(1;2)`, `gate_store,beta_store=(0;2)`, `sched=(0;2)` | load cursors after reserving first/next prefetch slot; store cursors after current done wait; scheduler after item |
| warp 9 | `q=(1;1)`, `k=(1;2)`, `v=(1;1)`, `do=(1;1)`, `state=(1;1)`, `sched=(1;2)` | after its exact split-done/reuse wait and before the corresponding issue; scheduler is published after all item loads |
| warp 8 | `kk=(0;1)`, `dk_total=(1;1)`, `du_scale,dk_scale,dk_attn=(0;1)`, `dstate_acc=(0 if dstate input else 1;1)`, `k=(0;2)`, `q,state,v,tinv,do,y,u,da,dv,dm,dstate_smem,dq_scale=(0;1)`, `dq_total=(1;1)`, `a,do_prime,du,dyp,dstate_inp=(0;1)`, `sched=(0;2)` | immediately after each wait/selected slot in the 17-operation issuer sequence; scheduler after item |
| CG0 | `gate,beta=(0;2)`, `a,tinv=(1;1)`, `k=(0;2)`, `da,dm,cg0_dbeta,dgate,kk,a_ready,dk_scale,dq_scale,dstate_smem,dk_attn=(0;1)`, `sched=(0;2)` | immediately after each matching full/ready wait; `a` also advances at the source reuse wait; scheduler after item |
| CG1 | `gate,beta=(0;2)`, `v,do,k_state,u,dy,du_scale,du_total,dk_total,dstate_acc,state_dot,dk_state,dq_total=(0;1)`, `dq,sdv,dstate_inp,dk,dv=(1;1)`, `sched=(0;2)` | immediately after the corresponding full/reuse/store-done wait at its separate staging point; scheduler after item |

### Persistent scheduler

The anchor takes the static early return. The ticket atomic and two-stage dynamic
ring below are explicit variant semantics without an anchor instruction claim.

```python
def producer_next(tile,cursor):                    # warp 9
    if not DYNAMIC: return tile+gridDim.x,cursor
    wait(sched_done[cursor.stage],cursor.phase)
    if elected_lane():
        ticket=atomic_add(sched_ctr[0],1)
        copy_r2s(gridDim.x+ticket,SCHED+4*cursor.stage)
    warp_sync(); next_tile=copy_s2r(SCHED+4*cursor.stage)
    if elected_lane(): arrive(sched_ready[cursor.stage])
    return next_tile,advance(cursor,2)

def consumer_next(tile,cursor):                    # other eleven warps
    if not DYNAMIC: return tile+gridDim.x,cursor
    wait(sched_ready[cursor.stage],cursor.phase)
    next_tile=copy_s2r(SCHED+4*cursor.stage)
    if elected_lane(): arrive(sched_done[cursor.stage])
    return next_tile,advance(cursor,2)
```

### Warp 9: TensorMap loads

```python
if warp==9:
    set_register_budget(decrease,64)
    raw_q=cursor(1,phase=1); raw_k=cursor(2,phase=1)
    raw_v=cursor(1,phase=1); raw_do=cursor(1,phase=1)
    state=cursor(1,phase=1); sched=cursor(2,phase=1)
    while tile<total_tiles:
        item=decode_eight_i32_fields(work_items[tile])
        # instruction_selection: two ld.global.v4.b32; extent: one work row
        if elected_lane():
            for array in (Q,K,V,DO,CKPT):
                acquire_tensormap(descriptor(array,item.batch))
            # instruction_selection: five
            # fence.proxy.tensormap::generic.acquire.gpu operations
        for rev in range(item.cend-item.wstart):
            chunk=item.cend-1-rev

            # K has two consumers and is deliberately loaded before Q.
            wait(k_mma_done[raw_k.stage],raw_k.phase)
            wait(k_cg0_done[raw_k.stage],raw_k.phase)
            k_slot=raw_k.stage; raw_k=advance(raw_k,2)
            if elected_lane(): expect_bytes(k_ready[k_slot],16384)
            for channel_half in (0,64):
                copy_g2s(map_tile(K,item,chunk,channel_half),
                         arena+io_byte(K,k_slot,0,channel_half),k_ready[k_slot])
            # instruction_selection: two 3-D TMA loads into one 16-KiB stage

            wait(q_mma_done[raw_q.stage],raw_q.phase)
            wait(q_cg1_done[raw_q.stage],raw_q.phase)
            q_slot=raw_q.stage; raw_q=advance(raw_q,1)
            if elected_lane(): expect_bytes(q_ready[q_slot],16384)
            for channel_half in (0,64):
                copy_g2s(map_tile(Q,item,chunk,channel_half),
                         arena+io_byte(Q,q_slot,0,channel_half),q_ready[q_slot])

            wait(v_mma_done[raw_v.stage],raw_v.phase)
            v_slot=raw_v.stage; raw_v=advance(raw_v,1)
            if elected_lane(): expect_bytes(v_ready[v_slot],16384)
            for channel_half in (0,64):
                copy_g2s(map_tile(V,item,chunk,channel_half),
                         arena+io_byte(V_U,v_slot,0,channel_half),v_ready[v_slot])

            wait(do_mma_done[raw_do.stage],raw_do.phase)
            do_slot=raw_do.stage; raw_do=advance(raw_do,1)
            if elected_lane(): expect_bytes(do_ready[do_slot],16384)
            for channel_half in (0,64):
                copy_g2s(map_tile(DO,item,chunk,channel_half),
                         arena+io_byte(DO,do_slot,0,channel_half),do_ready[do_slot])
            # instruction_selection for Q/V/dO: six more 3-D TMA sites

            if chunk>=FIRST_STATE_CHUNK:
                wait(state_mma_done[state.stage],state.phase)
                state_slot=state.stage; state=advance(state,1)
                if elected_lane(): expect_bytes(state_ready[state_slot],32768)
                for value_half in (0,64):
                    copy_g2s(map_checkpoint(item,chunk,value_half),
                             arena+state_byte(STATE,0,value_half),state_ready[state_slot])
                # instruction_selection: two 4-D TMA loads

        # The producer publishes a scheduler ticket only after every load for
        # this work item has been issued.
        tile,sched=producer_next(tile,sched)

    # Producer tail, in source order. These waits prevent a persistent CTA
    # from exiting while a split consumer or MMA still owns a raw stage.
    for _ in range(1):
        wait(q_mma_done[raw_q.stage],raw_q.phase)
        wait(q_cg1_done[raw_q.stage],raw_q.phase); raw_q=advance(raw_q,1)
    for _ in range(2):
        wait(k_mma_done[raw_k.stage],raw_k.phase)
        wait(k_cg0_done[raw_k.stage],raw_k.phase); raw_k=advance(raw_k,2)
    for _ in range(1):
        wait(v_mma_done[raw_v.stage],raw_v.phase); raw_v=advance(raw_v,1)
    for _ in range(1):
        wait(do_mma_done[raw_do.stage],raw_do.phase); raw_do=advance(raw_do,1)
```

### Warp 10: first/next gate-beta prefetch and in-place scalar stores

The instruction annotations in this role describe only the reviewed anchor
(`log_gate=True`, FP32 post-sigmoid beta). The optional formulas after the code
state specialization semantics but deliberately make no PTX claim.

```python
def anchor_prefetch_scalar_chunk(chunk,gate_load,beta_load,item):
    offset=item.batch_start+chunk*64
    valid=[offset+lane+32*col < item.batch_end for col in (0,1)]

    gslot=gate_load.stage; gate_load=advance(gate_load,2)
    vals=[copy_g2r(gate[offset+lane+32*col]) if valid[col] else 0.0
          for col in (0,1)]
    # instruction_selection: two predicated ld.global.b32 values per lane
    for col in (0,1): vals[col] *= RCP_LN2
    for distance in (1,2,4,8,16):
        for col in (0,1):
            prior=shuffle_up(vals[col],distance)
            if lane>=distance: vals[col]+=prior
    vals[1]+=shuffle_index(vals[0],31)
    # instruction_selection: ten shfl.sync.up.b32 and one shfl.sync.idx.b32
    for col in (0,1):
        token=lane+32*col
        copy_r2s(vals[col],CUMSUMLOG+f32_ring_byte(0,gslot,token))
        copy_r2s(exp2(vals[col]),CUMPROD+f32_ring_byte(0,gslot,token))
    fence_proxy_async_shared(); arrive(gate_ready[gslot],32)

    bslot=beta_load.stage; beta_load=advance(beta_load,2)
    for col in (0,1):
        token=lane+32*col
        cp_async_g2s_4B(beta[offset+token],BETA_S+f32_ring_byte(0,bslot,token),
                        copy_bytes=4 if valid[col] else 0)
    cp_async_arrive_noinc(beta_ready[bslot])
    # instruction_selection: two cp.async.ca.shared.global sites and one
    # cp.async.mbarrier.arrive.noinc.shared.b64 per producer iteration
    return gate_load,beta_load

if warp==10:
    set_register_budget(decrease,64)
    gate_load=cursor(2,phase=1); beta_load=cursor(2,phase=1)
    gate_store=cursor(2,phase=0); beta_store=cursor(2,phase=0)
    sched=cursor(2,phase=0)
    while tile<total_tiles:
        item=decode_work(tile); count=item.cend-item.wstart
        write_end=min(item.batch_start+item.wend*64,item.batch_end)

        # First prefetch is outside the reverse loop.
        if count>0:
            gate_load,beta_load=anchor_prefetch_scalar_chunk(
                item.cend-1,gate_load,beta_load,item)

        for rev in range(count):
            # Prefetch the next backward chunk before storing the current one.
            if rev+1<count:
                gate_load,beta_load=anchor_prefetch_scalar_chunk(
                    item.cend-2-rev,gate_load,beta_load,item)

            chunk=item.cend-1-rev; offset=item.batch_start+chunk*64
            gslot=gate_store.stage
            wait(gate_done[gslot],gate_store.phase)
            gate_store=advance(gate_store,2)
            dg=[copy_s2r(CUMSUMLOG+f32_ring_byte(0,gslot,lane+32*col))
                for col in (0,1)]
            for distance in (1,2,4,8,16):
                for col in (0,1):
                    later=shuffle_down(dg[col],distance)
                    if lane<32-distance: dg[col]+=later
            dg[0]+=shuffle_index(dg[1],0)
            # instruction_selection: ten shfl.sync.down.b32 and one
            # shfl.sync.idx.b32; this is the only reverse suffix scan
            for col in (0,1):
                token=lane+32*col
                if offset+token<write_end: copy_r2g(dg[col],dgate[offset+token])

            bslot=beta_store.stage
            wait(beta_done[bslot],beta_store.phase)
            beta_store=advance(beta_store,2)
            for col in (0,1):
                token=lane+32*col
                if offset+token<write_end:
                    copy_s2r_then_g(BETA_S+f32_ring_byte(0,bslot,token),
                                    dbeta[offset+token])
            # instruction_selection: shared b32 loads plus predicated
            # st.global.b32 for both final scalar arrays
        tile,sched=consumer_next(tile,sched)
```

Optional gate specializations replace only the anchor's initial transform:
safe gate uses
`a_l2=-exp2(a_log*RCP_LN2)*RCP_LN2` and
`g_log2=a_l2*softplus(gate+dt_bias)`, where softplus is the source's stable
piecewise `max(x,0)+log1p(exp(-abs(x)))`; raw gate uses
`log2(gate+1e-10)`. Optional beta-sigmoid loads IO logits, evaluates the tanh
sigmoid identity, rounds through the IO dtype, and stores FP32 in `BETA_S`;
the final `dbeta` store casts back to IO. These branches require their own
line-info export before instruction-level annotations are added.

### Warp 8: TMEM lifetime and source-order GEMM schedule

```python
if warp==8:
    set_register_budget(decrease,64)
    alloc_tmem(TMEM_MAILBOX,512); named_barrier(1,288); tmem=copy_s2r(TMEM_MAILBOX)
    # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32,
    # bar.sync 1,288, then ld.shared.b32
    while tile<total_tiles:
      item=decode_work(tile)
      for rev in range(item.cend-item.wstart):
        chunk=item.cend-1-rev; has_dstate=USE_DSTATE_IN or rev>0

        wait(k_ready); gemm(SHARED_ACC0,K,K.T,accumulate=False); commit(kk_acc_ready)
        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16;
        # extent: 8 K=16 issues for [64,64,128], then one delayed tcgen05.commit
        wait(q_ready); gemm(SHARED_ACC1,Q,K.T,False); commit(a_acc_ready)
        # instruction_selection: same family/extent, then one delayed commit

        if chunk>=FIRST_STATE_CHUNK:
            wait(state_ready); wait(kk_acc_done)
            gemm(SHARED_ACC0,STATE.T,K.T,False); commit(k_state_ready)
            # instruction_selection: eight tcgen05.mma f16 issues and one commit;
            # checkpoint ownership intentionally continues through dK-state

        if has_dstate:
            wait(dstate_inp_ready); wait(dk_total_done)
            gemm(DVDK_ACC,DSTATE_INP,K.T,False)
            commit(du_scale_ready); commit(dstate_inp_done)
            # instruction_selection: eight tcgen05.mma f16 issues and two commits
        else: wait(dk_total_done)

        wait(do_ready)
        if chunk>=FIRST_STATE_CHUNK:
            wait(dq_total_done); gemm(DSTATE_INP,STATE,dO.T,False); commit(dq_scale_ready)
            # instruction_selection: eight tcgen05.mma f16 issues plus one commit

        wait(a_ready); if has_dstate: wait(du_scale_done)
        gemm(DVDK_ACC,dO.T,A,accumulate=has_dstate); commit(du_total_ready)
        # instruction_selection: four tcgen05.mma f16 issues for K=64 plus commit

        wait(do_prime_ready); wait(dstate_scale_done)
        gemm(DSTATE_ACC,dO_prime.T,Q,accumulate=has_dstate)
        # instruction_selection: four tcgen05.mma f16 issues; publication delayed

        wait(tinv_ready); wait(y_ready)
        gemm(SHARED_ACC1,Y.T,TINV.T,False); commit(u_acc_ready)
        # instruction_selection: four tcgen05.mma f16 issues plus commit

        wait(du_inp_ready)
        gemm(SHARED_ACC0,dU.T,TINV,False); commit(dy_acc_ready)
        # instruction_selection: four tcgen05.mma f16 issues plus commit

        wait(u_ready)
        if has_dstate:
            wait(dstate_smem_ready); gemm(DVDK_ACC,DSTATE,U.T,False); commit(dk_scale_ready)
            # instruction_selection: eight tcgen05.mma f16 issues plus commit

        gemm(SHARED_ACC1,dO,U.T,False); commit(da_acc_ready); commit(do_mma_done)
        # instruction_selection: eight tcgen05.mma f16 issues plus two commits
        wait(dv_stage_ready)
        gemm(SHARED_ACC0,dY,U.T,False); commit(dm_acc_ready); commit(v_mma_done)
        # instruction_selection: eight tcgen05.mma f16 issues plus two commits

        wait(dyp_inp_ready)
        gemm(DSTATE_ACC,dY_prime.T,K,True); commit(dstate_acc_ready)
        # instruction_selection: four tcgen05.mma f16 issues plus commit

        if has_dstate: wait(dk_scale_done)
        wait(da_ready); gemm(DVDK_ACC,Q.T,dA,accumulate=has_dstate)
        commit(dk_attn_ready); commit(q_mma_done)
        # instruction_selection: four tcgen05.mma f16 issues plus two commits

        if chunk>=FIRST_STATE_CHUNK: wait(dq_scale_done)
        else: wait(dq_total_done)
        gemm(DSTATE_INP,K.T,dA.T,accumulate=(chunk>=FIRST_STATE_CHUNK))
        commit(dq_total_ready); commit(a_done)
        # instruction_selection: four tcgen05.mma f16 issues; separate static
        # accumulate true/false sites, then two commits

        if chunk>=FIRST_STATE_CHUNK:
            gemm(SHARED_INP0,STATE,dY.T,False); commit(dk_state_ready); commit(state_mma_done)
            # instruction_selection: eight tcgen05.mma f16 issues plus commits
        commit(sdv_done)

        wait(dm_acc_done); wait(dk_attn_done)
        gemm(DVDK_ACC,K.T,dM.T,True); gemm(DVDK_ACC,K.T,dM,True)
        commit(dk_total_ready); commit(k_mma_done)
        # instruction_selection: two four-issue tcgen05.mma chains and two commits
      tile=consumer_next(tile)
    wait(dk_total_done); wait(tmem_done)
    relinquish_tmem_alloc_permit(); dealloc_tmem(512)
    # instruction_selection: tcgen05.relinquish_alloc_permit then
    # tcgen05.dealloc.cta_group::1.sync.aligned.b32
```

The anchor PTX contains 112 static `tcgen05.mma.cta_group::1.kind::f16`
instructions and 24 delayed `tcgen05.commit...mbarrier::arrive::one` sites.
Those totals include the accumulate-true/false twins but no hidden computation.

### Warps 0..3: CG0 inverse, dot terms, and final scalar fold

```python
if warp<4:
    set_register_budget(increase,192); named_barrier(1,288)
    DSTATE_IN0=1 if USE_DSTATE_IN else 0
    # This zero fill is once per CTA, before the persistent work loop.
    for word in thread_strided_range(TINV,8192,128): copy_r2s(0,word)
    # instruction_selection: lane-strided st.shared.b32 zero fill

    while tile<total_tiles:
      item=decode_work(tile)
      for rev in range(item.cend-item.wstart):
        chunk=item.cend-1-rev; has_dstate=USE_DSTATE_IN or rev>0
        gate_slot=gate_cursor.stage; wait(gate_ready[gate_slot],gate_cursor.phase)
        gate_cursor=advance(gate_cursor,2)
        beta_slot=beta_cursor.stage; wait(beta_ready[beta_slot],beta_cursor.phase)
        beta_cursor=advance(beta_cursor,2)

        # Explicit lane-owned 64x64 decay fragments.
        for owned_i_j in 32_values_per_thread:
            gi=copy_s2r(CUMSUMLOG+f32_ring_byte(0,gate_slot,i))
            gj=copy_s2r(CUMSUMLOG+f32_ring_byte(0,gate_slot,j))
            decay[i,j]=exp2(gi-gj) if i>=j else 0
            strict[i,j]=0 if i==j else decay[i,j]
        last_gate=copy_s2r(CUMSUMLOG+f32_ring_byte(0,gate_slot,63))
        for owned_j in 16_column_values:
            decay_scale[j]=exp2(last_gate-copy_s2r(
                CUMSUMLOG+f32_ring_byte(0,gate_slot,j)))
        cumprod_total=copy_s2r(CUMPROD+f32_ring_byte(0,gate_slot,63))

        wait(kk_acc_ready,kk_cursor.phase); kk_cursor=advance(kk_cursor,1)
        kk=copy_t2r(SHARED_ACC0)
        for owned_i_j: copy_r2s(cast(kk[i,j]*strict[i,j]*beta[i]),KK[i,j])
        # Source predicates on the reverse-loop counter, not decoded chunk.
        # The first K-state wait consumes the initial phase; each qualifying
        # CG0 epilogue arrival primes the following one-stage reuse phase.
        if rev < item.cend-FIRST_STATE_CHUNK: arrive(kk_acc_done,128)

        a_slot=a_cursor.stage; a_phase=a_cursor.phase; a_cursor=advance(a_cursor,1)
        wait(a_acc_ready,a_ready_cursor.phase); a_ready_cursor=advance(a_ready_cursor,1)
        avec=copy_t2r(SHARED_ACC1)
        for owned_i_j: a_pack[i,j]=cast(avec[i,j]*decay[i,j]*scale)
        wait(a_done[a_slot],a_phase)             # mandatory A overwrite guard
        for owned_i_j: copy_r2s(a_pack[i,j],A_DA[i,j])
        fence_proxy_async_shared(); arrive(a_ready[a_slot],128)

        # Exact synchronous hierarchical inverse. There are no async-proxy
        # fences between stages: group barriers make each SMEM stage visible.
        named_barrier(2,128)                     # pre-inverse
        if cg0_warp<2: invert_eight_diagonal_8x8_blocks(KK,TINV)
        named_barrier(2,128)                     # after width 8
        invert_8x8_to_16x16_with_two_static_m16n8k8(TINV,KK)
        named_barrier(2,128)                     # after width 16
        if cg0_warp<2: invert_16x16_to_32x32_with_m16n8k16(TINV,KK)
        named_barrier(2,128)                     # after width 32
        if cg0_warp<2:
            load_two_32x32_halves()
            named_barrier(3,64)                  # inner 32->64 helper barrier
            invert_32x32_to_64x64_with_m16n8k16(TINV,KK)
        named_barrier(2,128)                     # after width 64
        # instruction_selection: two static
        # mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 sites at 8->16;
        # twenty m16n8k16 sites across later inverse work
        for owned_i_j:
            copy_r2s(cast(copy_s2r(TINV[i,j])*beta[j]),TINV[i,j])
        fence_proxy_async_shared(); arrive(tinv_ready,128)

        if chunk>=FIRST_STATE_CHUNK:
            wait(dq_scale_ready,dq_scale_cursor.phase)
            dq_scale_cursor=advance(dq_scale_cursor,1)
            for fragment in dq_inter_fragments:
                copy_r2t(copy_t2r(fragment)*cumprod[token]*scale,fragment)
            tmem_store_wait(); arrive(dq_scale_done,128)

        # Explicit state-dot-dstate loop, including the overwrite-protection
        # publication consumed by CG1 before it restages sDstate.
        if has_dstate:
            wait(dstate_smem_ready,dstate_smem_cursor.phase)
            dstate_smem_cursor=advance(dstate_smem_cursor,1)
        state_dot_dstate=0
        for octet in range(16):
            for packed_word in range(4):
                state_dot_dstate=fma(unpack(DSTATE[octet,packed_word]),
                                     unpack(STATE[octet,packed_word]),
                                     state_dot_dstate)
        for distance in (1,2,4,8,16):
            state_dot_dstate+=shuffle_bfly(state_dot_dstate,distance)
        arrive(state_dot_done,128)

        # Explicit K-dot-dK-inter loop is fused with dK inter scaling.
        k_slot=k_cursor.stage; k_cursor=advance(k_cursor,2)
        k_dot_dk_inter=0
        if has_dstate:
            wait(dk_scale_ready,dk_scale_cursor.phase)
            dk_scale_cursor=advance(dk_scale_cursor,1)
            for fragment in dk_inter_fragments:
                scaled=copy_t2r(fragment)*decay_scale[token]
                copy_r2t(scaled,fragment)
                for packed_k in corresponding_K_fragment(k_slot):
                    k_dot_dk_inter=fma(scaled,unpack(packed_k),k_dot_dk_inter)
            tmem_store_wait(); arrive(dk_scale_done,128)
            for distance in (1,2,4,8,16):
                k_dot_dk_inter+=shuffle_bfly(k_dot_dk_inter,distance)

        wait(da_acc_ready,da_cursor.phase); da_cursor=advance(da_cursor,1)
        for owned_i_j:
            copy_r2s(cast(copy_t2r(SHARED_ACC1[i,j])*decay[i,j]*scale),A_DA[i,j])
        fence_proxy_async_shared(); arrive(da_ready,128)

        wait(dm_acc_ready,dm_cursor.phase); dm_cursor=advance(dm_cursor,1)
        for owned_i_j:
            dm_value=-copy_t2r(SHARED_ACC0[i,j])*strict[i,j]
            copy_r2s(cast(dm_value),DM[i,j])
        fence_proxy_async_shared(); arrive(dm_acc_done,128)

        wait(dk_attn_ready,dk_attn_cursor.phase); dk_attn_cursor=advance(dk_attn_cursor,1)
        dks=copy_t2r(DVDK_ACC); tmem_load_wait(); arrive(dk_attn_done,128)

        # M-term E=dM_core*M_kk/beta. Row and column reductions remain
        # separate fixed loops rather than a compound reduction placeholder.
        row_acc=[0]*8; col_part=[0]*16
        for pair in range(16):
            e=unpack(DM[pair])*unpack(KK[pair])*rcp(beta[row(pair)]+1e-10)
            row_acc[row_bucket(pair)]+=e.lo+e.hi
            col_part[column_pair(pair)]+=e
        for distance in (1,2):
            row_acc[0]+=shuffle_bfly(row_acc[0],distance)
            row_acc[1]+=shuffle_bfly(row_acc[1],distance)

        # CG1 owns the V partial first; CG0 consumes it in place and adds M.
        wait(dbeta_cg1_ready,cg0_dbeta_cursor.phase)
        cg0_dbeta_cursor=advance(cg0_dbeta_cursor,1)
        for owned_token:
            db=copy_s2r(BETA_S[beta_slot,token])-row_acc[token]*rcp(beta[token]+1e-10)
            if BETA_SIGMOID: db*=beta[token]*(1-beta[token])
            copy_r2s(db,BETA_S[beta_slot,token])

        # K-dot-dKattn is explicit and closes the split raw-K edge exactly once.
        part_k=[0]*16
        for sub in (0,1):
          for matrix_row in range(4):
            for packed_word in range(4):
                part_k[fragment_token]+=dks[sub]*unpack(K[k_slot,sub,matrix_row,packed_word])
        fence_proxy_async_shared(); arrive(k_cg0_done[k_slot],128)
        dgate_part=col_part-part_k
        dgate_last=k_dot_dk_inter
        if (rev+DSTATE_IN0>=1) and (rev < item.cend-FIRST_STATE_CHUNK):
            dgate_last+=cumprod_total*state_dot_dstate
        for offset in (4,8,16):
          for pair in paired_fragment_tokens:
            send=pair.low if lane_in_high_half(offset) else pair.high
            pair=pair.keep+shuffle_bfly(send,offset)
        dgate_partial=(paired_fragment_tokens[0],paired_fragment_tokens[1])
        if lane==31: dgate_partial.last+=dgate_last
        copy_r2s(dgate_partial,CG0_DGATE_SCRATCH)

        wait(dgate_cg1_ready,dgate_cursor.phase); dgate_cursor=advance(dgate_cursor,1)
        for owned_row: CUMSUMLOG[gate_slot,owned_row]-=row_acc[owned_row]
        named_barrier(2,128)
        for token in first_64_threads:
            CUMSUMLOG[gate_slot,token]+=sum(
                CG0_DGATE_SCRATCH[group,token] for group in range(4))

        # These are CG0's 128 arrivals; together with CG1 they satisfy the
        # 256-count barriers that warp 10 waits before suffix/store.
        arrive(gate_done[gate_slot],128); arrive(beta_done[beta_slot],128)
      tile,sched=consumer_next(tile,sched)

    # A has one stage; drain its final outstanding MMA consumer before exit.
    for _ in range(1):
        wait(a_done[a_cursor.stage],a_cursor.phase); a_cursor=advance(a_cursor,1)
    arrive(tmem_done,128)
```

### Warps 4..7: CG1 state/value path and separate dV/dQ/dK staging

```python
if 4<=warp<8:
    set_register_budget(increase,248); named_barrier(1,288)
    while tile<total_tiles:
      item=decode_work(tile); count=item.cend-item.wstart

      if USE_DSTATE_IN and count>0:
        wait(dstate_inp_done,dstate_inp_cursor.phase)
        dstate_inp_cursor=advance(dstate_inp_cursor,1)
        for owned_state_value:
            seed=copy_g2r(dstate_in[item]) if item.cend==item.num_chunks else 0
            copy_r2t(seed,DSTATE_ACC); copy_r2t(cast(seed),DSTATE_INP)
        tmem_store_wait(); arrive(dstate_inp_ready,128)
        for owned_state_value: copy_t2r_then_s(DSTATE_ACC,DSTATE)
        fence_proxy_async_shared(); arrive(dstate_smem_ready,128)

      for rev in range(count):
        chunk=item.cend-1-rev; have_dstate=USE_DSTATE_IN or rev>0
        gate_slot=gate_cursor.stage; wait(gate_ready[gate_slot],gate_cursor.phase)
        gate_cursor=advance(gate_cursor,2)
        beta_slot=beta_cursor.stage; wait(beta_ready[beta_slot],beta_cursor.phase)
        beta_cursor=advance(beta_cursor,2)

        # There is deliberately no false-path arrival on the first anchor
        # chunk. Only a real dstate rescale releases this accumulator phase.
        if have_dstate:
            for owned_state_value:
                h=copy_t2r(DSTATE_ACC)*CUMPROD[gate_slot,63]
                copy_r2t(h,DSTATE_ACC)
            tmem_store_wait(); arrive(dstate_scale_done,128)

        do_slot=do_cursor.stage; wait(do_ready[do_slot],do_cursor.phase)
        do_cursor=advance(do_cursor,1)
        for owned_do_value:
            dop=unpack(DO[do_slot])*CUMPROD[gate_slot,token]*scale
            copy_r2t(cast(dop),SHARED_INP0)
        tmem_store_wait(); arrive(do_prime_ready,128)

        if have_dstate:
            wait(du_scale_ready,du_scale_cursor.phase)
            du_scale_cursor=advance(du_scale_cursor,1)
            for fragment in du_inter_fragments:
                copy_r2t(copy_t2r(fragment)*decay_scale[token],fragment)
            tmem_store_wait(); arrive(du_scale_done,128)

        # Y formation has three distinct lowering steps: FP32 K-state scaling,
        # rounding that product to packed IO, then packed BF16 subtraction.
        v_slot=v_cursor.stage; wait(v_ready[v_slot],v_cursor.phase)
        v_cursor=advance(v_cursor,1)
        if chunk>=FIRST_STATE_CHUNK:
            wait(k_state_ready,k_state_cursor.phase); k_state_cursor=advance(k_state_cursor,1)
            for packed_pair:
                scaled_fp32=copy_t2r(SHARED_ACC0)*CUMPROD[gate_slot,token]
                scaled_bf16x2=pack_bf16x2_rn(scaled_fp32)
                y_bf16x2=sub_bf16x2(copy_s2r(V_U[v_slot,packed_pair]),scaled_bf16x2)
                copy_r2t(y_bf16x2,Y)
                copy_r2t(scaled_bf16x2,G_K_STATE)
        else:
            for packed_pair: copy_s2r_then_t(V_U[v_slot,packed_pair],Y)
        tmem_store_wait(); arrive(y_ready,128)
        # instruction_selection in anchor: FP32 mul, cvt.rn.bf16x2.f32,
        # sub.bf16x2, then two distinct tcgen05.st.sync.aligned.16x128b
        # sites: 32 packed-IO columns at Y=448..479 and 32 at
        # G_K_STATE=480..511

        wait(du_total_ready,du_total_cursor.phase); du_total_cursor=advance(du_total_cursor,1)
        for fragment: copy_r2t(cast(copy_t2r(DVDK_ACC)),SHARED_INP1)
        tmem_store_wait(); arrive(du_inp_ready,128)

        wait(u_acc_ready,u_cursor.phase); u_cursor=advance(u_cursor,1)
        for fragment: copy_t2r_then_s_cast(SHARED_ACC1,V_U)
        fence_proxy_async_shared(); arrive(u_ready,128)

        wait(dy_acc_ready,dy_cursor.phase); dy_cursor=advance(dy_cursor,1)
        dy=copy_t2r(SHARED_ACC0)
        for fragment: copy_r2t(cast(-CUMPROD[gate_slot,token]*dy),SHARED_INP1)
        tmem_store_wait(); arrive(dyp_inp_ready,128)

        # dV is published first and exactly once.
        dv_slot=dv_cursor.stage; wait(dv_stage_done[dv_slot],dv_cursor.phase)
        dv_cursor=advance(dv_cursor,1)
        wait(sdv_done,sdv_cursor.phase); sdv_cursor=advance(sdv_cursor,1)
        for fragment: copy_r2s(cast(dy),DV_DY[dv_slot])
        fence_proxy_async_shared(); arrive(dv_stage_ready[dv_slot],128)

        # Reserve dQ now, but publish only after the final dQ TMEM read below.
        dq_slot=dq_cursor.stage; wait(dq_stage_done[dq_slot],dq_cursor.phase)
        dq_cursor=advance(dq_cursor,1)

        # CG1 V partials: fixed per-channel products, the explicit 4-warp
        # reduction, then in-place writes to the real scalar rings.
        part_y=[0]*16; part_g=[0]*16
        for sub in (0,1):
          yvec=copy_t2r(Y[sub])
          for channel_pair in range(16):
            part_y[fragment_token]+=dy[sub,channel_pair]*unpack(yvec[channel_pair])
            if chunk>=FIRST_STATE_CHUNK:
                part_g[fragment_token]+=dy[sub,channel_pair]*unpack(copy_t2r(G_K_STATE[sub,channel_pair]))
        for offset in (4,8,16):
          for pair in paired_fragment_tokens:
            send=pair.low if lane_in_high_half(offset) else pair.high
            pair=pair.keep+shuffle_bfly(send,offset)
        copy_r2s(part_y_and_part_g,CG1_V_SCRATCH)
        named_barrier(CG1_ID,128)
        for token in first_64_threads:
            ysum=sum(CG1_V_SCRATCH[group,token] for group in range(4))
            gsum=sum(CG1_V_SCRATCH[group,token+256] for group in range(4))
            BETA_S[beta_slot,token]=ysum*rcp(BETA_S[beta_slot,token]+1e-10)
            CUMSUMLOG[gate_slot,token]=-gsum if chunk>=FIRST_STATE_CHUNK else 0
        arrive(beta_done[beta_slot],128); arrive(dbeta_cg1_ready,128)
        named_barrier(CG1_ID,128)

        # Q is copied to the Y/Q alias for the explicit dQ dot and its split
        # raw-buffer release occurs only after that copy.
        for fragment: copy_s2r_then_t(Q,Y_GK_QT)
        tmem_store_wait(); arrive(q_cg1_done,128)
        wait(dq_total_ready,dq_total_cursor.phase); dq_total_cursor=advance(dq_total_cursor,1)
        dq=copy_t2r(DSTATE_INP)
        for fragment: copy_r2s(cast(dq),DQ[dq_slot])
        fence_proxy_async_shared(); arrive(dq_total_done,128)
        arrive(dq_stage_ready[dq_slot],128)          # dQ published second

        part_q=[0]*16
        for sub in (0,1):
          for matrix_row in range(4):
            for packed_word in range(4):
                part_q[fragment_token]+=dq[sub]*unpack(Q_fragment[sub,matrix_row,packed_word])
        for offset in (4,8,16):
          for pair in paired_fragment_tokens:
            send=pair.low if lane_in_high_half(offset) else pair.high
            pair=pair.keep+shuffle_bfly(send,offset)
        copy_r2s(part_q,CG1_QDOT_SCRATCH); named_barrier(CG1_ID,128)
        for token in first_64_threads:
            CUMSUMLOG[gate_slot,token]+=sum(
                CG1_QDOT_SCRATCH[group,token] for group in range(4))
        arrive(gate_done[gate_slot],128); arrive(dgate_cg1_ready,128)

        if chunk>=item.wstart+1:
            wait(dstate_acc_ready,dstate_acc_cursor.phase)
            dstate_acc_cursor=advance(dstate_acc_cursor,1)
            wait(dstate_inp_done,dstate_inp_cursor.phase)
            dstate_inp_cursor=advance(dstate_inp_cursor,1)
            for owned_state_value:
                h=copy_t2r(DSTATE_ACC); copy_r2t(cast(h),DSTATE_INP)
            tmem_store_wait(); arrive(dstate_inp_ready,128)

        # dK is reserved/folded last. The state path is multiplied by the
        # per-token negative cumprod before it is added to the accumulated dK.
        dk_slot=dk_cursor.stage; wait(dk_stage_done[dk_slot],dk_cursor.phase)
        dk_cursor=advance(dk_cursor,1)
        if chunk>=FIRST_STATE_CHUNK:
            wait(dk_state_ready,dk_state_cursor.phase); dk_state_cursor=advance(dk_state_cursor,1)
            state_raw=copy_t2r(SHARED_INP0)
            tmem_load_wait()                    # state-path load completion
            state_part=state_raw*(-CUMPROD[gate_slot,token])
        else: state_part=0
        wait(dk_total_ready,dk_total_cursor.phase); dk_total_cursor=advance(dk_total_cursor,1)
        total_dk=copy_t2r(DVDK_ACC)
        tmem_load_wait()                        # distinct total-dK completion
        arrive(dk_total_done,128); dk=total_dk+state_part
        for fragment: copy_r2s(cast(dk),DK[dk_slot])
        fence_proxy_async_shared(); arrive(dk_stage_ready[dk_slot],128) # third

        # Before overwriting the shared dstate image, wait for CG0's explicit
        # state-dot-dstate loop. The false first-chunk path advances parity only.
        if chunk>=item.wstart+1:
            wait(state_dot_done,state_dot_cursor.phase); state_dot_cursor=advance(state_dot_cursor,1)
            for owned_state_value: copy_t2r_then_s_cast(DSTATE_ACC,DSTATE)
            fence_proxy_async_shared(); arrive(dstate_smem_ready,128)
        else: state_dot_cursor=advance(state_dot_cursor,1)

      # Final dstate drain. In the no-dstate-input specialization this is the
      # single tail arrival that balances the accumulator reuse protocol.
      if count>0:
        wait(dstate_acc_ready,dstate_acc_cursor.phase); dstate_acc_cursor=advance(dstate_acc_cursor,1)
        if USE_DSTATE0 and item.wstart==0: copy_t2r_then_g(DSTATE_ACC,dstate0_out)
        if not USE_DSTATE_IN: arrive(dstate_scale_done,128)
      elif USE_DSTATE0 and item.wstart==0:
        copy_g2g(dstate_in,dstate0_out) if USE_DSTATE_IN else fill_gmem(dstate0_out,0)
      tile,sched=consumer_next(tile,sched)

    arrive(tmem_done,128)
    for _ in range(1):
        wait(dstate_inp_done,dstate_inp_cursor.phase); dstate_inp_cursor=advance(dstate_inp_cursor,1)
    for _ in range(1):
        wait(dk_stage_done[dk_cursor.stage],dk_cursor.phase); dk_cursor=advance(dk_cursor,1)
    for _ in range(1):
        wait(dv_stage_done[dv_cursor.stage],dv_cursor.phase); dv_cursor=advance(dv_cursor,1)
```

### Warp 11: direct current-chunk dV -> dQ -> dK TensorMap stores

```python
if warp==11:
    set_register_budget(decrease,64)
    dq=cursor(1,phase=0); dk=cursor(1,phase=0); dv=cursor(1,phase=0)
    sched=cursor(2,phase=0)
    while tile<total_tiles:
      item=decode_work(tile)
      if elected_lane():
        for array in (DV,DQ,DK): acquire_tensormap(descriptor(array,item.batch))
      for rev in range(item.cend-item.wstart):
        chunk=item.cend-1-rev; writes=chunk<item.wend

        dv_slot=dv.stage; wait(dv_stage_ready[dv_slot],dv.phase); dv=advance(dv,1)
        if writes:
            copy_s2g(DV_DY[dv_slot],map_tile(DV,item,chunk)); commit_store_group()

        dq_slot=dq.stage; wait(dq_stage_ready[dq_slot],dq.phase); dq=advance(dq,1)
        if writes:
            copy_s2g(DQ[dq_slot],map_tile(DQ,item,chunk)); commit_store_group()

        dk_slot=dk.stage; wait(dk_stage_ready[dk_slot],dk.phase); dk=advance(dk,1)
        if writes:
            copy_s2g(DK[dk_slot],map_tile(DK,item,chunk)); commit_store_group()
        # instruction_selection: six static 3-D TMA store sites (two subtiles
        # for each logical dV/dQ/dK tile) and three commit-group sites

        wait_store_group(2); arrive(dv_stage_done[dv_slot],32)
        wait_store_group(1); arrive(dq_stage_done[dq_slot],32)
        wait_store_group(0); arrive(dk_stage_done[dk_slot],32)
        # instruction_selection: cp.async.bulk.wait_group.read 2/1/0 followed
        # immediately by the corresponding 32-lane reuse arrival
      tile,sched=consumer_next(tile,sched)
```

There is no pending item and no post-loop store drain. Tail rows are clipped by
the patched TensorMap dimensions; warm-up chunks execute all three ready waits
and phase advances but predicate the stores off. A zero-length item skips the
chunk loop while preserving persistent scheduler parity.

## Logical GEMM inventory

| phase | destination | operation | shape / instruction extent | guard |
| --- | --- | --- | --- | --- |
| KK | shared acc 0 | `K @ K^T` | `64x64x128`, 8 tcgen05 issues | always |
| QK | shared acc 1 | `Q @ K^T` | `64x64x128`, 8 | always |
| K-state | shared acc 0 | `state^T @ K^T` | `128x64x128`, 8 | entering state |
| dV inter | dV/dK acc | `dstate^T @ K^T` | `128x64x128`, 8 | has dstate |
| dQ inter | dQ acc | `state @ dO^T` | `128x64x128`, 8 | entering state |
| dU intra | dV/dK acc | `dO^T @ A` | `128x64x64`, 4 | always, accumulates inter |
| dstate Q | dstate acc | `dO'^T @ Q` | `128x128x64`, 4 | always; acc flag by dstate |
| U | shared acc 1 | `Y^T @ T_inv^T` | `128x64x64`, 4 | always |
| dY | shared acc 0 | `dU^T @ T_inv` | `128x64x64`, 4 | always |
| dK inter | dV/dK acc | `dstate_entry @ U^T` | `128x64x128`, 8 | has dstate |
| dA | shared acc 1 | `dO @ U^T` | `64x64x128`, 8 | always |
| dM | shared acc 0 | `dY @ U^T` | `64x64x128`, 8 | always |
| dstate dY | dstate acc | `dY'^T @ K` | `128x128x64`, 4 | always |
| dK attention | dV/dK acc | `Q^T @ dA` | `128x64x64`, 4 | always |
| dQ attention | dQ acc | `K^T @ dA^T` | `128x64x64`, 4 | true/false accumulate twin |
| dK state | shared input alias | `state @ dY^T` | `128x64x128`, 8 | entering state |
| dK dM pair | dV/dK acc | `K^T@dM^T + K^T@dM` | two `128x64x64`, 4+4 | always |

Every row selects `tcgen05.mma.cta_group::1.kind::f16`; a K extent 128 expands
to eight issues and K extent 64 to four.  Publications are delayed to the edge
shown in the role program rather than committed per issue.  The inverse emits two static `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32`
sites in the 8-to-16 stage and 20 static
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` sites in later stages.

## Storage aliases and lifetimes

| alias | physical region | safety edge |
| --- | --- | --- |
| CG0 dGate fold scratch / sKK | `109568..110591` (256 f32) | written after dK/K-dot reduction, consumed after `dgate_cg1_ready` and group-2 rendezvous before the next KK payload reuse |
| CG1 V-term scratch / current sDQ | `183296..185343` (512 f32) | `dq_stage_done` reserves the current dQ stage before V partials; reduction completes before final dQ overwrites and publishes that stage |
| CG1 Q-dot scratch / sDstate | `150528..151551` (256 f32) | Q-dot reduction completes before `state_dot_done` is waited and the next dstate image overwrites sDstate |
| A then dA | `117760..125951` | dU consumes A before warp 8 publishes `a_done`; dA is written only after `da_acc_ready` and consumed before next A generation |
| V then U | `134144..150527` | CG1 reads V into Y, then publishes/reuses the same bytes for U; `v_mma_done` and `u_ready` close both consumers |
| dV output / dY operand | `216064..232447` | CG1 stages dY as dV; warp 8 reads it for dM/state-path before `sdv_done`, and warp 11 drains the same bytes as dV |
| shared acc 0 | TMEM `256..319` | ordered `KK -> K_state -> dY -> dM`; each publication is paired with the prior consumer/reuse wait |
| shared acc 1 | TMEM `320..383` | ordered `A -> U -> dA` under `a_acc_ready/u_acc_ready/da_acc_ready` |
| shared input 0 | TMEM `384..415` | dO' is consumed by dstate-Q before the dK-state accumulator overwrites it |
| shared input 1 | TMEM `416..447` | dU is consumed by dY before dY' overwrites it for the dstate update |
| Y/g_k/QT | TMEM `448..511` | Y is consumed by U, gate-state intermediates finish, then Q transpose is staged for dQ dot |

## Static specialization boundary

| axis | accepted values | emitted-code consequence |
| --- | --- | --- |
| I/O dtype | BF16, FP16 | selects operand conversion and tcgen05 f16 dtype encoding; accumulation remains FP32 |
| `(HQ,HK,HV)` | each divides `HO=max(HQ,HV)` | compile-time head ratios choose TMA coordinates; direct outputs always have HO heads |
| gate mode | raw alpha, natural log, safe | selects `log2`, multiply by `LOG2_E`, or exp2/tanh safe transform |
| beta mode | FP32 post-sigmoid, IO logits | logits add tanh sigmoid and the beta chain rule after dGate consumption |
| state | initial checkpoint, final gradient, initial-gradient output independently represented by validated profiles | changes chunk-0 state path, dstate seed/drain, and zero-length pass-through |
| scheduler | static, dynamic | dynamic emits the sole global atomic and the two-stage 11-consumer ring |
| ordering | none, caller scratch, generated | changes only the 1024-thread prologue order body |
| `cu_seqlens` | i32, i64 | selects endpoint load width and address arithmetic; work items remain i32 |
| tails/splits | arbitrary nonnegative lengths and `[wstart,wend,cend)` | TensorMap clipping and scalar predicates; warm-up chunks compute but do not write |

## TIRx and validation contract

- Registry name `cudnn_sm100_gdn_bprop_f16`, category `cudnn`, compute
  capability 10.  `get_kernel` returns prologue then main; both launches and
  only those launches are timed on each side.
- Checkpoint construction, work-table creation, allocation, compilation,
  source import/JIT, and validation stay outside timing.
- Correctness compares the same input tensors against the pinned source with
  `torch.assert_close` at BF16 `(rtol=2^-7, atol=2^-10)`, FP16
  `(2^-10,2^-14)`, and FP32 `(2^-16,2^-20)`, plus exact shape/dtype, NaN/Inf
  classification, zero-length behavior, and redzones.  An independent FP64
  recurrence oracle uses the source test suite's RMS caps as a second check.
- The benchmark suite is the only performance authority.  The complete frozen
  21-row matrix must report `mean(cudnn_frontend_us)/mean(tirx_us)>0.99` on
  every row.  PTX/SASS/NCU/codegen evidence is diagnostic only.
- The representation gate requires a single rank-1 SMEM allocation, no
  low-level function calls or exemptions, and no changes under
  `tirx_kernels/kern/`.

## Instruction-selection evidence

| family | anchor static count | consequence |
| --- | ---: | --- |
| `mbarrier.init.shared.b64` | 75 | exact physical protocol words |
| `mbarrier.try_wait.parity...` | 86 | all producer/reuse and consumer/full waits remain explicit |
| `mbarrier.arrive.shared.b64` | 42 | thread/warp publications and releases |
| `mbarrier.arrive.expect_tx.shared.b64` | 5 | Q/K/V/dO/checkpoint transaction barriers |
| `tcgen05.mma.cta_group::1.kind::f16` | 112 | complete logical GEMM table including static twins |
| `tcgen05.commit...mbarrier::arrive::one` | 24 | delayed chain publications |
| `mma.sync.aligned.m16n8k8...bf16` | 2 | 8-to-16 inverse stage |
| `mma.sync.aligned.m16n8k16...bf16` | 20 | later hierarchical inverse stages |
| 3-D TMA loads / 4-D TMA loads | 8 / 2 | Q/K/V/dO and two checkpoint halves |
| 3-D TMA stores | 6 | two static sites each for dQ/dK/dV ladder |
| `ldmatrix` / `stmatrix` | 80 / 64 | register-SMEM operand and epilogue movement |
| `setmaxnreg` | 6 | two compute groups plus four service warps |
| `bar.sync` | 15 | CTA and named group rendezvous |
| `fence.proxy.async.shared::cta` | 12 | generic-to-async SMEM publication |

Placement selects TMA versus register/shared traffic; shape selects tcgen05
issue multiplicity versus register MMA; and the producer/consumer schedule
selects transaction completion, delayed tcgen05 commits, or plain arrivals.
Those are lowering facts from the writer export, not inferences from source
declarations.
