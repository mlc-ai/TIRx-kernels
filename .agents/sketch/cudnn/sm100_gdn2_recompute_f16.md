<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This design sketch documents a modified TIRx port of cuDNN Frontend's
python/cudnn/linear_attention/frost/kernel/gdn2_recompute_f16.py at commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5.
-->

# cuDNN SM100 GDN2 FP16/BF16 recompute: coarse WASP pipeline sketch

This is a non-executable operation sketch, not Python, a builder API, a new IR,
or a mathematical reference. It freezes the source program before device-code
transcription. The executable source of truth belongs in
[`tirx_kernels/cudnn/linear_attention/gdn2_recompute_f16.py`](../../../tirx_kernels/cudnn/linear_attention/gdn2_recompute_f16.py).

The frozen source is the standalone `chunk_gdn2_recompute_sm100` two-launch
entry at the commit above. It is the state/checkpoint-only GDN2 path: BF16 or
FP16 I/O, BF16 or FP32 state, `BT=16`, `DK=DV=128`, one CTA per cluster, 512
main threads, 1024 prologue threads, and no Q or O operand. It supports absent
or present initial/final state, optional checkpoints at a positive multiple of
16 tokens, grouped K/V heads, ragged and zero-length sequences, natural-log or
safe gate, optional K L2 normalization, optional beta sigmoid, split work rows,
generated or caller-staged ordering, and static grid-stride or dynamic ticket
scheduling. At least checkpoint or final-state output must be enabled.

Writer evidence is under
`.porting/gdn2_recompute_f16/source_export_ptx92/`. Both accepted source
specializations declare PTX 9.2 and `sm_100a`; the main launch declares 512
threads and the prologue 1024. The no-state/state main digests are respectively
`4b94c20c6df31d8895dbc0d92cba2ec5a159ddc8fb5840bf6ce7831f5b3521f5`
and `3c9f74f76c125330ddfc7414d502db8706329ce8374b28ea9717715631f1a98c`.
The common prologue digest is
`5bff2d1fbd4b6f6bfa58019af97dca70096d0472442db1811fd740e04468e1cd`.

After the independent sketch reviewer returns PASS, this file is immutable.

## Pipeline at a glance

| Warps | Register cap | Source-order role | Principal publications |
| --- | ---: | --- | --- |
| 0..7 | 160 | two four-warp ping-pong CG0 groups; gate prefix, K normalization/beta, K inverse/decay/restore, state diagonal | raw done, decay ready, diagonal/QK ready |
| 8..11 | 136 | CG1 state owner; seed/repack state, form `Y=W*V-state*(beta*K)`, repack U, stage checkpoints, store final state | state/Y/U inputs, checkpoint ready, TMEM done |
| 12 | 56 | register MMA for `KK`, strict-lower `L`, and three Neumann-doubling rounds for `T_inv` | T-inverse and decay-super done |
| 13 | 56 | allocate TMEM and issue four logical `tcgen05.mma` chains | state-K, state decay, U, state update |
| 14 | 56 | persistent scheduler and five TensorMap GMEM-to-SMEM input loads | K/V/Gate/Beta/W ready |
| 15 | 56 | checkpoint TensorMap store drain | checkpoint done |

Dispatch order is exactly source order: warp 14, warp 12, warp 13, warp 15,
warps 0..7, then warps 8..11. Each role has an independent persistent cursor.
Its rolling stage/phase state starts once, advances across work-item boundaries,
and is not reset for a new row.

## Primitive vocabulary and transcription boundary

All storage is one-dimensional. Matrix names below are semantic labels on
scalar integer byte/row/column functions; they are never layout objects, views,
tiles, fragments, or first-class mappings.

```python
linear_buffer(space, dtype, elements, byte_offset, alignment, lifetime)
reg_array(dtype, elements)
smem_byte(arena, base, stage, row, col,
          stage_bytes, row_bytes, elem_bytes, xor_mask)
tmem_cell(base_column, row, column)  # base_column + column + (row << 16)
gmem_element(base, scalar_index)
descriptor_slot(workspace, array, batch)  # (array*B+batch)*128 bytes
```

Movement uses only `copy_p2g`, `copy_g2s`, `copy_s2g`, `copy_g2r`,
`copy_r2g`, `copy_s2r`, `copy_r2s`, `copy_t2r`, and `copy_r2t`. Computation
uses only `fill`, `cast`, `add`, `sub`, `mul`, `fma`, `max`, `select`, `exp2`,
`tanh`, `rsqrt`, `rcp`, `shuffle_xor`, and `gemm`. Barrier actions, fences,
waits, commits, cursor advancement, register-budget changes, TensorMap field
replacement, and TMEM lifecycle operations remain explicit schedule actions.

The executable module imports device language only as
`import tirx_kernels.kern as K`. It may use scalar K control flow, opaque
TensorMaps, raw `K.ptx[...]`, rank-one shared storage, and integer descriptor
assembly. It must not use direct TVM script namespaces, a tile primitive,
`TilePrimitiveCall`, a first-class layout, rank>1 SMEM, `K.cuda.func_call`, or
an inline-CUDA exemption. Nothing under `tirx_kernels/kern/` is modified.

## Complete non-executable sketch

```python
@specialize(
    IO_DTYPE={"bf16", "f16"}, STATE_DTYPE={"f32", "bf16"},
    USE_INITIAL_STATE={False,True}, STORE_FINAL_STATE={False,True},
    CHECKPOINTS={False,True}, L2NORM={False,True},
    SAFE_GATE={False,True}, BETA_SIGMOID={False,True},
    BT=16, DK=128, DV=128, ACC_DTYPE="f32",
    RAW_STAGES=(4 if CHECKPOINTS else 5),
    RAW_READY_STAGES=(4 if CHECKPOINTS else 6),
    CHECKPOINT_STAGES=(2 if CHECKPOINTS else 1),
    DECAY_STAGES=2, INTERMEDIATE_STAGES=2,
    DIAG_STAGES=4, QK_READY_STAGES=4,
    SCHED_STAGES=8, CLUSTER=(1,1,1),
)
def gdn2_recompute_factory(...):
    return descriptor_order_prologue, persistent_gdn2_recompute

# ========================================================================
# Launch 1: source-order work table and six TensorMap descriptor arrays
# ========================================================================

@kernel(grid=(1,1,1), block=(1024,1,1), warps=32, target="sm_100a")
def descriptor_order_prologue(
    RUN_ORDER, ORDER_GENERATE, HAS_DYNAMIC_SCHEDULER, BT,
    base_k, base_v, base_gate, base_beta, base_w, base_checkpoint,
    descriptor_workspace, cu_seqlens,
    k, v, gate, beta, w, checkpoints_or_dummy,
    staging_items_or_dummy, work_count, work_items,
    sched_all_or_dummy, batch_count,
    k_row_stride, v_row_stride, gate_row_stride,
    beta_row_stride, w_row_stride, checkpoint_row_stride,
    checkpoint_every_n_tokens,
):
    tid = thread_id()
    warp = tid // 32
    arena = linear_buffer("smem", "u8", 32776, 0, 16, "whole launch")
    order_key = integer_region(arena, 0, 4096, "i32")
    order_idx = integer_region(arena, 16384, 4096, "i32")
    order_spread = integer_region(arena, 32768, 2, "i32")

    if tid == 0 and HAS_DYNAMIC_SCHEDULER:
        for counter in range(sched_all_or_dummy.shape[0]):
            copy_r2g(i32(0),sched_all_or_dummy[counter])
            # instruction_selection: st.global.b32; extent: one scalar per
            # iteration, serially covering the complete scheduler array

    if RUN_ORDER:
        n = batch_count*HO if ORDER_GENERATE else copy_g2r(work_count[0])
        if ORDER_GENERATE and tid == 0:
            copy_r2g(n, work_count[0])
            # instruction_selection: st.global.b32; extent: one scalar
        if n > 4096:
            for item in range(tid,n,1024):
                row = generate_uncut_row(item) if ORDER_GENERATE else copy_g2r(
                    staging_items[item,0:8])
                copy_r2g(row,work_items[item,0:8])
                # instruction_selection: generated rows may use two
                # st.global.v4.b32; staged rows retain scalar global loads and
                # stores; extent: one eight-i32 work row
        else:
            if tid == 0:
                copy_r2s(i32_max,order_spread[0])
                copy_r2s(i32_min,order_spread[1])
            padded = next_power_of_two(n)
            barrier(0,1024)
            # instruction_selection: barrier.sync 0,1024; extent: whole CTA
            for element in range(4):
                item=tid+1024*element
                if item<padded:
                    key=i32_min
                    if item<n:
                        row=generate_or_load_eight_fields(item)
                        key=row_span(row)
                    copy_r2s(key,order_key[item])
                    copy_r2s(item,order_idx[item])
            shared_atomic_minmax(order_spread,local_min,local_max)
            # instruction_selection: atom.shared::cta.min/max.s32;
            # extent: one pair per thread after four local items
            barrier(0,1024)
            # instruction_selection: barrier.sync 0,1024; extent: whole CTA
            stable_lpt_bitonic_sort(order_key,order_idx,n,order_spread)
            barrier(0,1024)
            # instruction_selection: barrier.sync 0,1024; extent: whole CTA
            for rank in range(tid,n,1024):
                row=generate_or_load_eight_fields(copy_s2r(order_idx[rank]))
                copy_r2g(row,work_items[rank,0:8])
                # instruction_selection: global i32 load/store family;
                # extent: one eight-i32 work row

    base_maps=(base_k,base_v,base_gate,base_beta,base_w,base_checkpoint)
    checkpoint_prefix=0
    if warp<6 and elected_lane():
        for batch in range(batch_count):
            bos=copy_g2r(cu_seqlens[batch])
            eos=copy_g2r(cu_seqlens[batch+1])
            length=eos-bos
            dst=descriptor_slot(descriptor_workspace,warp,batch)
            copy_p2g(base_maps[warp],dst)
            # instruction_selection: grid-constant parameter loads and sixteen
            # st.global.b64 stores; extent: one 128-byte descriptor
            if warp==5:
                count=0 if length==0 else (length-1)//checkpoint_every_n_tokens+1
                replace_global_address(dst,checkpoints_or_dummy+
                                       checkpoint_prefix*checkpoint_row_stride)
                replace_global_dim(dst,checkpoint_ordinal,count)
                checkpoint_prefix+=count
            else:
                replace_global_address(dst,tensor_base(warp)+bos*row_stride(warp))
                replace_global_dim(dst,sequence_ordinal(warp),length)
        fence("tensormap_generic_release_gpu",warp)
        # instruction_selection: tensormap.replace.tile.global_address/global_dim
        # followed by fence.proxy.tensormap::generic.release.gpu;
        # extent: one serial per-batch descriptor-array walk

# ========================================================================
# Launch 2 ABI, rank-one arena, exact offsets, and protocol
# ========================================================================

@kernel(grid=(MAX_ACTIVE_CLUSTERS,1,1), block=(512,1,1), warps=16,
        cluster=(1,1,1), min_blocks_per_sm=1, target="sm_100a")
def persistent_gdn2_recompute(
    descriptor_workspace, n_desc,
    k, v, gate, a_log_or_dummy, dt_bias_or_dummy, beta, w,
    cu_seqlens, initial_state_or_dummy, final_state_or_dummy,
    work_items, work_count, sched_counter_or_dummy,
    checkpoint_every_n_tokens,
):
    tid=thread_id(); warp=warp_uniform(tid//32); lane=tid&31
    cta=block_id_x(); grid_x=grid_dim_x(); total_tiles=copy_g2r(work_count[0])

    def decode_work(tile):
        row=copy_g2r(work_items[tile,0:8])
        batch,head,wstart,wend,cstart,cend,bos,eos=row
        return row_with_derived_fields(row,num_chunks=ceil_div(eos-bos,16))
        # instruction_selection: role-projected global i32 loads; extent: one
        # eight-field work row with only downstream-used fields retained

    arena = linear_buffer(
        "smem", "u8", 206848 if CHECKPOINTS else 165888,
        0, 1024, "whole launch")

    # Declaration-order protocol offsets for checkpoint / no-checkpoint.
    # Tuple: name, ready offset, done offset, ready/done stages,
    # ready/done arrivals, init owner.
    CHECKPOINT_PROTOCOL=(
      ("k",0,32,4,4,1,128,14), ("v",64,96,4,4,1,128,14),
      ("w",128,160,4,4,1,128,14), ("gate",192,224,4,4,1,128,14),
      ("beta",256,288,4,4,1,128,14),
      ("state_k",320,None,1,0,1,None,13),
      ("u_acc",328,None,1,0,1,None,13),
      ("state_inp",336,None,1,0,128,None,13),
      ("y_inp",344,None,1,0,128,None,13),
      ("u_inp",352,None,1,0,128,None,13),
      ("k_decay",360,376,2,2,128,1,12),
      ("decay_super",392,None,2,0,32,None,13),
      ("k_restore",408,None,2,0,1,None,13),
      ("qk_scale",424,None,4,0,128,None,12),
      ("diag_done",456,None,4,0,1,None,13),
      ("t_inv",488,504,2,2,32,1,12),
      ("state_read",520,None,1,0,128,None,15),
      ("tmem_done",528,None,1,0,128,None,13),
      ("checkpoint",536,552,2,2,128,32,15),
      ("scheduler",568,632,8,8,1,15,15),
    )
    # With checkpoints disabled the same declaration order uses raw-ready
    # depth six, raw data depth five, checkpoint depth one, and ends at byte
    # 800 before the mailbox. The exact consecutive offsets are:
    NO_CHECKPOINT_PROTOCOL=(
      ("k",0,48),("v",88,136),("w",176,224),
      ("gate",264,312),("beta",352,400),
      ("state_k",440,None),("u_acc",448,None),("state_inp",456,None),
      ("y_inp",464,None),("u_inp",472,None),
      ("k_decay",480,496),("decay_super",512,None),
      ("k_restore",528,None),("qk_scale",544,None),
      ("diag_done",576,None),("t_inv",608,624),
      ("state_read",640,None),("tmem_done",648,None),
      ("checkpoint",656,664),("scheduler",672,736),
    )

    TMEM_MAILBOX=696 if CHECKPOINTS else 800
    SCHED_BASE=704 if CHECKPOINTS else 816
    if CHECKPOINTS:
        K_DECAY_BASE=1024; K_RESTORE_BASE=9216; INTERMEDIATE_BASE=17408
        K_RAW_BASE=18432; V_RAW_BASE=34816; GATE_RAW_BASE=51200
        DIAG_BASE=83968; K_INV_BASE=100352; BETA_RAW_BASE=108544
        W_RAW_BASE=124928; CHECKPOINT_BASE=141312
    else:
        K_DECAY_BASE=1024; K_RESTORE_BASE=9216; INTERMEDIATE_BASE=17408
        K_RAW_BASE=18432; V_RAW_BASE=38912; GATE_RAW_BASE=59392
        DIAG_BASE=100352; K_INV_BASE=116736; BETA_RAW_BASE=124928
        W_RAW_BASE=145408; CHECKPOINT_BASE=V_RAW_BASE

    # Payload bytes: K-decay 8192, K-restore 8192, intermediate 1024,
    # K/V/Beta/W each RAW_STAGES*4096, Gate RAW_STAGES*8192,
    # diagonal 16384, K-inverse 8192, checkpoint 65536 when enabled.
    # All physical mappings use scalar SW128/SW32 XOR offset functions.

    protocol=CHECKPOINT_PROTOCOL if CHECKPOINTS else NO_CHECKPOINT_PROTOCOL
    pool=smem_pool(base=arena)
    allocate_protocol_in_declaration_order(pool,protocol)
    for i in range(tid,8192,512):
        copy_r2s(cast(IO_DTYPE,0),arena[DIAG_BASE+2*i])
        # instruction_selection: st.shared.u16; extent: strided diagonal arena

    if warp==14: init_raw_ready_and_done(protocol)
    elif warp==13: init_tcgen_owned_edges(protocol)
    elif warp==12: init_register_mma_owned_edges(protocol)
    elif warp==15: init_scheduler_checkpoint_and_state_read(protocol)
    # instruction_selection: mbarrier.init.shared.b64; extent: 87 static
    # sites in each accepted checkpoint PTX
    fence("mbarrier_init_release_cluster")
    # instruction_selection: fence.mbarrier_init.release.cluster;
    # extent: one CTA initialization publication
    barrier(0,512)
    # instruction_selection: barrier.sync 0,512;
    # extent: one CTA-wide synchronization

    STATE_ACC=0; STATE_INPUT=128; STATE_K_ACC=192
    U_ACC=208; Y_INPUT=224; U_INPUT=232
    assert U_INPUT+8==240 and 240<=512

    def wait_edge(name,stage,phase):
        wait(protocol_ptr(name,stage),phase)
        # instruction_selection:
        # mbarrier.try_wait.parity.acquire.cta.shared::cta.b64;
        # extent: one rolling edge wait
    def arrive_edge(name,stage=0):
        arrive(protocol_ptr(name,stage))
        # instruction_selection: mbarrier.arrive.shared.b64;
        # extent: one publication/return
    def expect_edge(name,stage,nbytes):
        expect_bytes(protocol_ptr(name,stage),nbytes)
        # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
        # extent: one TMA transaction start
    def commit_edge(name,stage=0):
        commit(protocol_ptr(name,stage))
        # instruction_selection:
        # tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64;
        # extent: one logical GEMM publication

    def scheduler_publish(cursor,tile):
        if DYNAMIC:
            wait_edge("scheduler_done",cursor.stage,cursor.phase)
            if elected_lane():
                ticket=atomic_add(sched_counter[0],1)
                copy_r2s(grid_x+ticket,arena[SCHED_BASE+4*cursor.stage])
                # instruction_selection: st.shared.b32;
                # extent: one scheduler-ring cell
            warp_sync()
            # instruction_selection: bar.warp.sync 0xffffffff;
            # extent: the scheduler-producer warp
            next_tile=copy_s2r(arena[SCHED_BASE+4*cursor.stage])
            # instruction_selection: ld.shared.b32;
            # extent: one scheduler-ring cell after warp synchronization
            if elected_lane(): arrive_edge("scheduler",cursor.stage)
            return next_tile,advance(cursor)
        return tile+grid_x,cursor
        # instruction_selection: atom.global.add.u32 in dynamic mode;
        # extent: one elected-lane ticket per completed work row

    def scheduler_consume(cursor,tile):
        if DYNAMIC:
            wait_edge("scheduler",cursor.stage,cursor.phase)
            next_tile=copy_s2r(arena[SCHED_BASE+4*cursor.stage])
            # instruction_selection: ld.shared.b32;
            # extent: one scheduler-ring cell after the ready wait
            if elected_lane(): arrive_edge("scheduler_done",cursor.stage)
            return next_tile,advance(cursor)
        return tile+grid_x,cursor

    # ====================================================================
    # Warp 14: five raw TensorMap loads and scheduler producer
    # ====================================================================
    if warp==14:
        set_register_budget("decrease",56)
        raw=producer_cursor(RAW_STAGES,phase=1)
        ready=producer_cursor(RAW_READY_STAGES,phase=0)
        sched=producer_cursor(8,phase=1); tile=cta
        while tile<total_tiles:
            item=decode_work(tile)
            kh=item.head if K_RATIO==1 else item.head//K_RATIO
            vh=item.head if V_RATIO==1 else item.head//V_RATIO
            for array in (K,V,GATE,BETA,W):
                if elected_lane():
                    acquire_descriptor(array,item.batch)
                    # instruction_selection:
                    # fence.proxy.tensormap::generic.acquire.gpu [descriptor],128;
                    # extent: one 128-byte elected-lane acquire per active descriptor
            for chunk in range(item.cstart,item.wend):
                for name,array,head,nbytes,subtiles in (
                    ("k",K,kh,4096,2),("v",V,vh,4096,2),
                    ("beta",BETA,item.head,4096,2),
                    ("w",W,item.head,4096,2),
                    ("gate",GATE,item.head,8192,4)):
                    wait_edge(name+"_done",raw.stage,raw.phase)
                    if elected_lane(): expect_edge(name,ready.stage,nbytes)
                    copy_g2s(descriptor(array,item.batch),
                             raw_stage_base(name,raw.stage),
                             (0,head,chunk*16),
                             completion=protocol_ptr(name,ready.stage))
                    # instruction_selection:
                    # cp.async.bulk.tensor.3d.shared::cta.global.tile
                    # .mbarrier::complete_tx::bytes; extent: two 64-channel
                    # subtiles for K/V/Beta/W, four 32-channel subtiles for Gate
                raw=advance(raw); ready=advance(ready)
            tile,sched=scheduler_publish(sched,tile)

    # ====================================================================
    # Warp 12: register KK and T-inverse
    # ====================================================================
    elif warp==12:
        set_register_budget("decrease",56)
        sched=consumer_cursor(8,phase=0); serial_base=0; tile=cta
        while tile<total_tiles:
            item=decode_work(tile)
            for local_chunk in range(item.wend-item.cstart):
                serial=serial_base+local_chunk
                decay_stage=serial%2; inter_stage=serial%2
                wait_edge("k_decay",decay_stage,(serial//2)&1)
                kk=fill(reg_array("f32",8),0)
                for k_block in range(8):
                    lhs=copy_s2r(k_decay_ldmatrix_address(decay_stage,k_block,lane))
                    rhs=copy_s2r(k_inverse_ldmatrix_address(decay_stage,k_block,lane))
                    # instruction_selection:
                    # ldmatrix.sync.aligned.m8n8.x4.shared.b16;
                    # extent: one A and one B fragment per k block
                    gemm(kk,lhs,rhs,accumulate=k_block>0)
                    # instruction_selection: two
                    # mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32;
                    # extent: sixteen instructions over eight k blocks
                L=select(strict_lower_fragment(lane),kk,0)
                Lpacked=cast(IO_DTYPE,L)
                Tinv=identity_fragment(lane)-cast("f32",Lpacked)
                Lpow=Lpacked
                moved=[move_matrix(Lpow[pair]) for pair in range(4)]
                # instruction_selection: four
                # movmatrix.sync.aligned.m8n8.trans.b16; extent: initial Lpow
                for round in range(3):
                    Lpow=cast(IO_DTYPE,gemm(Lpow,moved))
                    moved=[move_matrix(Lpow[pair]) for pair in range(4)]
                    Tround=cast(IO_DTYPE,Tinv)
                    Tinv=cast("f32",Tround)+gemm(Tround,moved)
                    # instruction_selection: four movmatrix and four
                    # mma.sync sites per round plus packed conversion/add;
                    # extent: twelve MMA and twelve moved fragments
                wait_edge("t_inv_done",inter_stage,(serial//2+1)&1)
                copy_r2s(cast(IO_DTYPE,Tinv),tinv_stmatrix_address(inter_stage,lane))
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16;
                # extent: one complete 16x16 T-inverse
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta;
                # extent: T-inverse shared-store publication
                arrive_edge("t_inv",inter_stage)
                arrive_edge("decay_super",decay_stage)
            serial_base+=item.wend-item.cstart
            tile,sched=scheduler_consume(sched,tile)

    # ====================================================================
    # Warp 13: TMEM allocation and four logical tcgen05 chains
    # ====================================================================
    elif warp==13:
        set_register_budget("decrease",56)
        allocate_tmem_warp_level(TMEM_MAILBOX,columns=512,cta_group=1)
        # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32;
        # extent: one warp-level allocation
        barrier(3,160)
        # instruction_selection: barrier.sync 3,160; extent: warp 13 and all
        # four CG1 warps
        tmem_base=copy_s2r(arena[TMEM_MAILBOX:TMEM_MAILBOX+4])
        # instruction_selection: ld.shared.b32; extent: one mailbox load per
        # warp-13 lane after allocation publication
        sched=consumer_cursor(8,phase=0); serial_base=0; tile=cta
        state_inp=consumer_cursor(1,phase=0)
        state_read=consumer_cursor(1,phase=0)
        y_inp=consumer_cursor(1,phase=0); u_inp=consumer_cursor(1,phase=0)
        diag=consumer_cursor(4,phase=0)
        while tile<total_tiles:
            item=decode_work(tile)
            for local_chunk in range(item.wend-item.cstart):
                serial=serial_base+local_chunk
                have_state=USE_INITIAL_STATE or local_chunk>0
                decay_stage=serial%2; inter_stage=serial%2
                wait_edge("k_decay",decay_stage,(serial//2)&1)
                if have_state:
                    wait_edge("state_inp",0,state_inp.phase); state_inp=advance(state_inp)
                    for kphase in range(8):
                        gemm(tmem_cell(tmem_base,0,STATE_K_ACC),
                             tmem_state_input_subtile(kphase),
                             smem_k_decay_subtile(decay_stage,kphase),
                             accumulate=kphase>0)
                    # instruction_selection: eight
                    # tcgen05.mma.cta_group::1.kind::f16.collector::a::discard;
                    # extent: 128x16x128
                    commit_edge("state_k")
                commit_edge("decay_tcgen05",decay_stage)

                if CHECKPOINTS and have_state:
                    wait_edge("state_read",0,state_read.phase); state_read=advance(state_read)
                wait_edge("qk_scale",diag.stage,diag.phase)
                if have_state:
                    for key_block in range(8):
                        gemm(tmem_cell(tmem_base,0,STATE_ACC+16*key_block),
                             tmem_state_input_block(key_block),
                             smem_diag_block(diag.stage,key_block),False)
                    # instruction_selection: eight
                    # tcgen05.mma.cta_group::1.kind::f16.collector::a::discard;
                    # extent: eight
                    # independent 128x16x16 diagonal-block products
                commit_edge("diag_done",diag.stage)

                wait_edge("t_inv",inter_stage,(serial//2)&1)
                wait_edge("y_inp",0,y_inp.phase); y_inp=advance(y_inp)
                gemm(tmem_cell(tmem_base,0,U_ACC),tmem_y_input(),
                     smem_tinv(inter_stage),False)
                # instruction_selection: one
                # tcgen05.mma.cta_group::1.kind::f16.collector::a::discard;
                # extent: 128x16x16
                commit_edge("u_acc"); commit_edge("t_inv_done",inter_stage)

                wait_edge("u_inp",0,u_inp.phase); u_inp=advance(u_inp)
                gemm(tmem_cell(tmem_base,0,STATE_ACC),tmem_u_input(),
                     smem_k_restore(decay_stage),accumulate=have_state)
                # instruction_selection: one
                # tcgen05.mma.cta_group::1.kind::f16.collector::a::discard;
                # extent: 128x128x16
                commit_edge("k_restore",decay_stage)
                diag=advance(diag)
            serial_base+=item.wend-item.cstart
            tile,sched=scheduler_consume(sched,tile)
        wait_edge("tmem_done",0,0)
        relinquish_tmem_alloc_permit(cta_group=1)
        deallocate_tmem(tmem_base,columns=512,cta_group=1)
        # instruction_selection: tcgen05.relinquish_alloc_permit and
        # tcgen05.dealloc.cta_group::1.sync.aligned.b32; extent: one teardown

    # ====================================================================
    # Warp 15: checkpoint TensorMap store drain
    # ====================================================================
    elif warp==15:
        set_register_budget("decrease",56)
        sched=consumer_cursor(8,phase=0)
        ready=consumer_cursor(CHECKPOINT_STAGES,phase=0); tile=cta
        while tile<total_tiles:
            item=decode_work(tile)
            if CHECKPOINTS:
                if elected_lane():
                    acquire_descriptor(CHECKPOINT,item.batch)
                    # instruction_selection:
                    # fence.proxy.tensormap::generic.acquire.gpu [descriptor],128;
                    # extent: one 128-byte elected-lane acquire per work row
                cadence_chunks=checkpoint_every_n_tokens//16
                quotient=(item.cstart+1)//cadence_chunks
                modulo=(item.cstart+1)%cadence_chunks
                if item.wend>item.cstart and item.wstart==0:
                    stage=ready.stage; wait_edge("checkpoint",stage,ready.phase)
                    ready=advance(ready)
                    copy_s2g(checkpoint_stage_base(stage),
                             checkpoint_coord(item.head,0))
                    # instruction_selection:
                    # cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group;
                    # extent: two 64-key subtiles over a 128x128 state
                    store_commit()
                    # instruction_selection: cp.async.bulk.commit_group;
                    # extent: one checkpoint store group
                    store_wait_group(0)
                    # instruction_selection: cp.async.bulk.wait_group.read 0;
                    # extent: drain the checkpoint store group
                    arrive_edge("checkpoint_done",stage)
                for local_chunk in range(1,item.wend-item.cstart):
                    chunk=item.cstart+local_chunk
                    if modulo==0 and chunk>=item.wstart:
                        stage=ready.stage; wait_edge("checkpoint",stage,ready.phase)
                        ready=advance(ready)
                        copy_s2g(checkpoint_stage_base(stage),
                                 checkpoint_coord(item.head,quotient))
                        # instruction_selection: same rank-4 bulk-group TMA
                        # store family; extent: two 64-key subtiles
                        store_commit()
                        # instruction_selection: cp.async.bulk.commit_group;
                        # extent: one checkpoint store group
                        store_wait_group(0)
                        # instruction_selection: cp.async.bulk.wait_group.read 0;
                        # extent: drain the checkpoint store group
                        arrive_edge("checkpoint_done",stage)
                    modulo+=1
                    if modulo==cadence_chunks: modulo=0; quotient+=1
            tile,sched=scheduler_consume(sched,tile)

    # ====================================================================
    # Warps 0..7: two four-warp CG0 groups
    # ====================================================================
    elif 0<=warp<=7:
        set_register_budget("increase",160)
        group=warp//4; local_warp=warp%4
        prefix_dim=local_warp*32+lane
        sched=consumer_cursor(8,phase=0); serial_base=0; tile=cta
        while tile<total_tiles:
            item=decode_work(tile)
            if SAFE_GATE and item.wend>item.cstart:
                aexp=exp2(copy_g2r(a_log[item.head])*LOG2_E)
                dbias=copy_g2r(dt_bias[item.head,prefix_dim])
            barrier(5,256)
            # instruction_selection: barrier.sync 5,256; extent: both CG0
            # ping-pong groups at work-item entry
            for local_chunk in range(group,item.wend-item.cstart,2):
                serial=serial_base+local_chunk
                raw_stage=rolling_data_stage_from_group(serial,RAW_STAGES)
                ready_stage=rolling_ready_stage_from_group(serial,RAW_READY_STAGES)
                decay_stage=serial%2; diag_stage=serial%4
                wait_edge("gate",ready_stage,ready_phase(serial))
                gates=copy_s2r(gate_fragment(raw_stage,local_warp,lane))
                if SAFE_GATE:
                    gates=select(valid_rows,
                        GATE_SCALE_LOG2*(tanh(aexp*(gates+dbias)*0.5)*0.5+0.5),0)
                else:
                    gates=select(valid_rows,gates*LOG2_E,0)
                # instruction_selection: vector shared f32 loads, fma/mul,
                # tanh.approx where enabled, and predicates; extent: 16 rows
                prefix=inclusive_scan_16_in_source_pair_order(gates)
                # instruction_selection: packed add.rn.f32x2 plus scalar
                # add.f32 in eight row pairs; extent: one channel scan
                exp_prefix=exp2(prefix); exp_last=exp_prefix[15]
                # instruction_selection: ex2.approx.ftz.f32; extent: 16 values
                copy_r2s(exp_prefix,gate_fragment(raw_stage,local_warp,lane))
                # instruction_selection: vector shared f32 stores; extent: 16
                wait_edge("diag_done",diag_stage,(serial//4+1)&1)
                copy_r2s(cast(IO_DTYPE,exp_last),diagonal_cell(diag_stage,prefix_dim))
                # instruction_selection: cvt.rn.{bf16|f16}.f32 and st.shared.u16;
                # extent: one diagonal element per lane
                barrier(1+group,128)
                # instruction_selection: barrier.sync {1|2},128; extent: the
                # selected four-warp CG0 group

                wait_edge("k",ready_stage,ready_phase(serial))
                wait_edge("beta",ready_stage,ready_phase(serial))
                raw_k,raw_beta=copy_two_x8_vectors(raw_stage,local_warp,lane)
                # instruction_selection: ld.shared.v4.b32; extent: two
                # 64-channel halves for K and Beta
                if BETA_SIGMOID:
                    raw_beta=cast("f32",cast(IO_DTYPE,
                        tanh(raw_beta*0.5)*0.5+0.5))
                if L2NORM:
                    norm=rsqrt(max(group_sum_source_order(raw_k*raw_k),EPS2))
                    # instruction_selection: fma.rn.f32, shfl.sync.bfly.b32 at
                    # XOR 4/2/1, max.f32, rsqrt.approx.ftz.f32; extent: one
                    # 128-channel norm per token row
                else: norm=opaque_one()

                wait_edge("decay_super",decay_stage,(serial//2+1)&1)
                wait_edge("decay_tcgen05",decay_stage,(serial//2+1)&1)
                k_norm=cast(IO_DTYPE,raw_k*norm)
                k_decay=packed_mul(cast(IO_DTYPE,cast("f32",k_norm)*raw_beta),
                                   cast(IO_DTYPE,exp_prefix))
                k_inv=packed_mul(k_norm,cast(IO_DTYPE,rcp(exp_prefix)))
                copy_r2s(k_decay,k_decay_sw128(decay_stage))
                copy_r2s(k_inv,k_inverse_sw128(decay_stage))
                # instruction_selection: packed f16x2/bf16x2 multiply,
                # st.shared.v4.b32; extent: two eight-value halves
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta;
                # extent: K-decay and K-inverse shared-store publication
                arrive_edge("k_decay",decay_stage)
                arrive_edge("k_done",raw_stage)
                arrive_edge("gate_done",raw_stage)
                arrive_edge("beta_done",raw_stage)

                wait_edge("k_restore",decay_stage,(serial//2+1)&1)
                k_restore=packed_mul(k_inv,cast(IO_DTYPE,exp_last))
                copy_r2s(k_restore,k_restore_sw128_transposed(decay_stage))
                # instruction_selection: packed f16x2/bf16x2 multiply and
                # st.shared.v4.b32; extent: two eight-value halves
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta;
                # extent: K-restore shared-store publication
                arrive_edge("qk_scale",diag_stage)
                advance_group_owned_raw_and_diag_cursors_by_two()
            serial_base+=item.wend-item.cstart
            tile,sched=scheduler_consume(sched,tile)

    # ====================================================================
    # Warps 8..11: state, Y/U, checkpoint staging, and final state
    # ====================================================================
    elif 8<=warp<=11:
        set_register_budget("increase",136)
        barrier(3,160)
        # instruction_selection: barrier.sync 3,160; extent: warp 13 and all
        # four CG1 warps
        tmem_base=copy_s2r(arena[TMEM_MAILBOX:TMEM_MAILBOX+4])
        # instruction_selection: ld.shared.b32; extent: one mailbox load per
        # CG1 lane after allocation publication
        value_dim=(warp-8)*32+lane
        sched=consumer_cursor(8,phase=0)
        checkpoint_done=producer_cursor(CHECKPOINT_STAGES,phase=1)
        state_k=consumer_cursor(1,phase=0); u_acc=consumer_cursor(1,phase=0)
        k_restore=consumer_cursor(2,phase=0)
        raw=consumer_cursor(RAW_STAGES,phase=0)
        ready=consumer_cursor(RAW_READY_STAGES,phase=0)
        tile=cta
        while tile<total_tiles:
            item=decode_work(tile); chunks=item.wend-item.cstart
            if chunks>0:
                if USE_INITIAL_STATE:
                    if item.cstart==0:
                        state=copy_g2r(initial_state[
                            item.batch,item.head,value_dim,0:128])
                    else: state=fill(reg_array("f32",128),0)
                    copy_r2t(state,tmem_state_rows(value_dim))
                    # instruction_selection: vector global loads and four
                    # tcgen05.st.sync.aligned.32x32b.x32.b32; extent: 128 FP32
                    wait_tmem_store()
                    # instruction_selection: tcgen05.wait::st.sync.aligned;
                    # extent: all preceding seed stores
                    state=copy_t2r(tmem_state_rows(value_dim))
                    # instruction_selection: eight
                    # tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: state row
                    packed=cast(IO_DTYPE,state,source_pair_xor=4)
                    copy_r2t(packed,tmem_state_input_rows(value_dim))
                    # instruction_selection: eight
                    # tcgen05.st.sync.aligned.32x32b.x8.b32; extent: packed row
                    wait_tmem_store()
                    # instruction_selection: tcgen05.wait::st.sync.aligned;
                    # extent: all preceding packed-state stores
                    arrive_edge("state_inp")
                    if CHECKPOINTS and item.wstart==0:
                        cp=checkpoint_done.stage
                        wait_edge("checkpoint_done",cp,checkpoint_done.phase)
                        checkpoint_done=advance(checkpoint_done)
                        copy_r2s(cast(IO_DTYPE,state),checkpoint_stage_rows(cp,value_dim))
                        # instruction_selection: st.shared.v4.b32; extent:
                        # 128 state values per value row in SW128 order
                        wait_tmem_load()
                        # instruction_selection: tcgen05.wait::ld.sync.aligned;
                        # extent: all preceding state loads
                        fence("async_shared_cta")
                        # instruction_selection: fence.proxy.async.shared::cta;
                        # extent: checkpoint shared-store publication
                        arrive_edge("checkpoint",cp)
                    if CHECKPOINTS: arrive_edge("state_read")
                elif CHECKPOINTS and item.wstart==0:
                    cp=checkpoint_done.stage
                    wait_edge("checkpoint_done",cp,checkpoint_done.phase)
                    checkpoint_done=advance(checkpoint_done)
                    copy_r2s(cast(IO_DTYPE,0),checkpoint_stage_rows(cp,value_dim))
                    # instruction_selection: st.shared.v4.b32 zero stores;
                    # extent: one complete zero checkpoint
                    fence("async_shared_cta")
                    # instruction_selection: fence.proxy.async.shared::cta;
                    # extent: zero-checkpoint shared-store publication
                    arrive_edge("checkpoint",cp)

                for local_chunk in range(chunks):
                    serial=global_serial_for_role(local_chunk)
                    if local_chunk>0:
                        state_repack=copy_t2r(
                            tmem_state_rows_x16(value_dim,blocks=8))
                        # instruction_selection:
                        # tcgen05.ld.sync.aligned.32x32b.x16.b32; extent:
                        # eight 16-key blocks for one value row
                        packed=cast(IO_DTYPE,state_repack,source_pair_xor=4)
                        copy_r2t(packed,tmem_state_input_rows(value_dim))
                        # instruction_selection:
                        # tcgen05.st.sync.aligned.32x32b.x8.b32; extent:
                        # eight 16-key blocks
                        wait_tmem_store()
                        # instruction_selection: tcgen05.wait::st.sync.aligned;
                        # extent: all preceding packed-state stores
                        arrive_edge("state_inp")
                        if CHECKPOINTS and checkpoint_boundary(item.cstart+local_chunk) \
                           and item.cstart+local_chunk>=item.wstart:
                            cp=checkpoint_done.stage
                            wait_edge("checkpoint_done",cp,checkpoint_done.phase)
                            checkpoint_done=advance(checkpoint_done)
                            checkpoint_state=copy_t2r(
                                tmem_state_rows_x32(value_dim,blocks=4))
                            # instruction_selection:
                            # tcgen05.ld.sync.aligned.32x32b.x32.b32;
                            # extent: four distinct 32-key rereads after the
                            # state-input publication
                            copy_r2s(cast(IO_DTYPE,checkpoint_state),
                                     checkpoint_stage_rows(cp,value_dim))
                            wait_tmem_load()
                            # instruction_selection:
                            # tcgen05.wait::ld.sync.aligned; extent: all four
                            # checkpoint rereads
                            arrive_edge("state_read")
                            fence("async_shared_cta")
                            # instruction_selection:
                            # fence.proxy.async.shared::cta; extent:
                            # checkpoint shared-store publication
                            arrive_edge("checkpoint",cp)
                        elif CHECKPOINTS: arrive_edge("state_read")

                    wait_edge("v",ready.stage,ready.phase)
                    raw_v=copy_s2r(v_ldmatrix_transposed(raw.stage,value_dim,lane))
                    wait_edge("w",ready.stage,ready.phase)
                    raw_w=copy_s2r(w_ldmatrix_transposed(raw.stage,value_dim,lane))
                    # instruction_selection: four
                    # ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 sites;
                    # extent: two value halves for V and W
                    if USE_INITIAL_STATE or local_chunk>0:
                        wait_edge("state_k",0,state_k.phase); state_k=advance(state_k)
                        erase=copy_t2r(tmem_state_k_rows(value_dim))
                        # instruction_selection: two
                        # tcgen05.ld.sync.aligned.16x256b.x2.b32; extent: 16 tokens
                        y=packed_sub(packed_mul(raw_w,raw_v),cast(IO_DTYPE,erase))
                    else: y=packed_mul(raw_w,raw_v)
                    copy_r2t(y,tmem_y_input_rows(value_dim))
                    # instruction_selection: two
                    # tcgen05.st.sync.aligned.16x128b.x2.b32; extent: 16 tokens
                    wait_tmem_store()
                    # instruction_selection: tcgen05.wait::st.sync.aligned;
                    # extent: the two preceding Y stores
                    arrive_edge("v_done",raw.stage)
                    arrive_edge("w_done",raw.stage); arrive_edge("y_inp")

                    wait_edge("u_acc",0,u_acc.phase); u_acc=advance(u_acc)
                    u=copy_t2r(tmem_u_rows(value_dim))
                    # instruction_selection:
                    # tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: 16 FP32
                    copy_r2t(cast(IO_DTYPE,u,source_pair_xor=4),
                             tmem_u_input_rows(value_dim))
                    # instruction_selection:
                    # tcgen05.st.sync.aligned.32x32b.x8.b32; extent: 16 packed
                    wait_tmem_store()
                    # instruction_selection: tcgen05.wait::st.sync.aligned;
                    # extent: the preceding packed-U store
                    arrive_edge("u_inp")
                    wait_edge("k_restore",k_restore.stage,k_restore.phase)
                    k_restore=advance(k_restore); raw=advance(raw); ready=advance(ready)

                if STORE_FINAL_STATE and item.wend==item.num_chunks:
                    state=copy_t2r(tmem_state_rows(value_dim))
                    copy_r2g(cast(STATE_DTYPE,state),
                             final_state[item.batch,item.head,value_dim,0:128])
                    # instruction_selection: four tcgen05 32x32 state loads
                    # followed by 16-byte vector global stores; extent: one
                    # final-state row
            elif STORE_FINAL_STATE:
                for key in range(128):
                    value=(copy_g2r(initial_state[item.batch,item.head,value_dim,key])
                           if USE_INITIAL_STATE else cast(STATE_DTYPE,0))
                    copy_r2g(value,final_state[item.batch,item.head,value_dim,key])
                    # instruction_selection: scalar global load/store or zero
                    # store; extent: one empty-sequence state row
            tile,sched=scheduler_consume(sched,tile)
        arrive_edge("tmem_done")
```

## Logical tcgen05 GEMMs

| # | FP32 TMEM destination | A operand | B operand | logical MxNxK | issues | accumulate |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | state-K col 192 | packed state TMEM | K-decay SMEM | 128x16x128 | 8 | no |
| 2 | state cols 0..127 | packed state TMEM | four-stage block diagonal | 128x128x16 per key block | 8 | no |
| 3 | U col 208 | packed Y TMEM | T-inverse SMEM | 128x16x16 | 1 | no |
| 4 | state cols 0..127 | packed U TMEM | K-restore SMEM transposed | 128x128x16 | 1 | runtime `have_state` |

The accepted PTX therefore has 18 static `tcgen05.mma` sites and six static
commit sites. The super-MMA warp has 28 register-MMA sites: sixteen for KK and
twelve for the three Neumann rounds. Publication remains separate from compute.

## Numerical order frozen by the schedule

For each token, the source recurrence represented by the chunk factorization is

```text
S_decayed = exp(g_t) * S
erase     = (beta_t * k_t) @ S_decayed
v_new     = w_t * v_t - erase
S         = S_decayed + outer(k_t, v_new)
```

The exact source order is observable: gate is transformed to log2 and
prefix-scanned in packed pair order; K normalization reduces even/odd FMA
chains and then XOR 4/2/1; beta sigmoid is rounded through I/O dtype;
K-decay/inverse/restore are rounded and multiplied in packed pairs; T-inverse
uses register MMA; state and U accumulate in FP32 TMEM. Tail gate rows become
zero increments and TensorMap OOB fill supplies zero K/V/Beta/W values.

## Source / PTX / sketch correspondence

| Source action | Source lines | Accepted PTX evidence | Sketch owner |
| --- | ---: | --- | --- |
| dynamic scheduler | 234..261 | `atom.global.add.u32`, rolling scheduler barriers | scheduler helpers |
| five raw TensorMap loads | 274..417 | 12 static rank-3 G2S sites, five expect-tx sites | warp 14 |
| KK and Neumann inverse | 419..599 | 28 register MMA, 16 movmatrix, one stmatrix site | warp 12 |
| four tcgen chains/lifecycle | 601..806 | 18 tcgen MMA, six commits, alloc/relinquish/dealloc | warp 13 |
| checkpoint store drain | 808..882 | four static rank-4 S2G sites and bulk-group waits | warp 15 |
| gate and K operands | 884..1218 | packed f32/I/O arithmetic and shared traffic | warps 0..7 |
| state/value/checkpoint/final | 1220..1720 | TMEM load/store, ldmatrix, shared/global stores | warps 8..11 |
| main ABI/storage/init/dispatch | 1770..2040 | `.reqntid 512`, 87 mbarrier init sites | launch 2 |
| configuration and physical sizes | 2040..2204 | folded constants and shared offsets | specialization/storage |
| six descriptor bodies | 2205..2268 | 96 `st.global.b64`, 12 replace sites, six release fences | launch 1 descriptors |
| order/generation prologue | 2269..2456 plus split/order helper | shared min/max and work-row traffic | launch 1 order |
| initial/final state difference | compile-time branch | state PTX adds one wait and two arrivals | CG1 / warp 13 |

## Static instruction evidence for the BF16 checkpoint anchor

| Instruction family | No state | Initial+final state | Owner |
| --- | ---: | ---: | --- |
| `mbarrier.init.shared.b64` | 87 | 87 | distributed initialization |
| `mbarrier.try_wait*` | 41 | 42 | role-local waits |
| `mbarrier.arrive.shared.b64` | 29 | 31 | publications/returns |
| `mbarrier.arrive.expect_tx*` | 5 | 5 | warp 14 |
| `tcgen05.mma*kind::f16` | 18 | 18 | warp 13 |
| `tcgen05.commit*mbarrier` | 6 | 6 | warp 13 |
| `mma.sync.aligned.m16n8k16*` | 28 | 28 | warp 12 |
| rank-3 TMA GMEM-to-SMEM | 12 | 12 | warp 14 |
| rank-4 TMA SMEM-to-GMEM | 4 | 4 | warp 15 |
| `ldmatrix*` / `stmatrix*` / `movmatrix*` | 24 / 1 / 16 | 24 / 1 / 16 | warps 12 and CG1 |
| `tcgen05.alloc/relinquish/dealloc` | 1 / 1 / 1 | 1 / 1 / 1 | warp 13 |

## Validation and benchmark contract

`get_kernel` returns `[descriptor_order_prologue, persistent_gdn2_recompute]`
in launch order. Source and TIRx receive identical immutable logical inputs but
independent descriptor workspaces, work tables, scheduler counters, checkpoint
buffers, and final-state buffers. Correctness first requires exact tensor
equality against the pinned source; any tolerance fallback must be no larger
than an observed output ULP and separately justified. Canary regions guard all
outputs, and lowered IR is rejected if it contains an inline CUDA function call.

Benchmark timing includes both launches for each implementation. Descriptor
construction, reference/TIRx compilation, and data preparation stay outside
the timed closures. `prepare_bench` is CPU-only. Only `bench_suite` reference
rows decide performance, with matched source/TIRx PTX versions (GB200 9.2,
GB300 9.3, VR200 9.4). Every frozen row must have finite positive raw samples
and satisfy strict `mean(cudnn_frontend) / mean(tirx) > 0.99`.
