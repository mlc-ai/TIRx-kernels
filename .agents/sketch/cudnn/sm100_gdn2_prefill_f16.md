<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This design sketch documents a modified TIRx port of cuDNN Frontend's
python/cudnn/linear_attention/frost/kernel/gdn2_prefill_f16.py at commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5.
-->

# cuDNN SM100 GDN2 FP16/BF16 prefill: coarse WASP pipeline sketch

This is a non-executable execution sketch, not Python, a builder API, a new IR,
or a mathematical reference.  It freezes the source program before device-code
transcription.  The implementation it describes belongs in
[`tirx_kernels/cudnn/linear_attention/gdn2_prefill_f16.py`](../../../tirx_kernels/cudnn/linear_attention/gdn2_prefill_f16.py).

The frozen source is the standalone `chunk_gdn2_sm100` two-launch entry at the
commit above.  The main anchor is BF16 I/O, FP32 state, final-state enabled,
`BT=16`, `DK=DV=128`, one CTA per cluster, 512 main threads, 1024 prologue
threads, five raw-data stages, six raw-ready stages, two O/decay/intermediate
stages, four diagonal/QK-ready stages, two Q-state TMEM stages, eight scheduler
stages, 203776 bytes of main dynamic shared memory, and 512 TMEM columns.
Checkpoint specializations use three raw stages, four raw-ready stages, two
checkpoint stages, and 211968 bytes of main dynamic shared memory.

The accepted capability set is the source-tested set frozen in
`.porting/gdn2_prefill_f16/capability_manifest.yaml`: BF16 and FP16 I/O; FP32
or BF16 state; absent/present initial state; absent/present final state;
checkpoint cadence 16, 32, or 48 (plus zero for disabled); grouped Q/K/V heads; ragged/tail/zero-length
sequences; L2 Q/K normalization; safe gate with `a_log`, `dt_bias`, and frozen
lower bound; beta sigmoid; static grid-stride or dynamic ticket scheduler;
generated uncut or caller-staged/sorted work items; i32 or i64 cumulative
sequence lengths; and overlap-recompute split work rows.  These are supported
only under the validated head divisibility, positive 16-token checkpoint
cadence, and work-item ownership guards described below.

Writer line-info evidence is in
`.porting/gdn2_prefill_f16/source_export/`.  Unless a branch profile is named,
PTX evidence refers to the sole `cutlass_host_Gdn2Cfg...sm_100a.ptx` and
`cutlass_prologue_...sm_100a.ptx` files in `basic_001`.  In the main anchor,
`.file 1` is `gdn2_prefill_f16.py`, `.file 2` is the barrier helper, `.file 3`
is split-K/order, `.file 4` is TMA, `.file 5` is handle/address conversion,
`.file 6` is swizzle, `.file 7` is MMA, and `.file 8` is pointwise arithmetic.

## Pipeline at a glance

| Warps | Registers | Source-order role | Principal publications |
| --- | ---: | --- | --- |
| 0..7 | 160 | two four-warp ping-pong CG0 groups; gate prefix, optional Q/K L2 norm, beta, K inverse/decay/restore, Q decay, state diagonal | raw-done, decay-ready, QK/diag-ready |
| 8..11 | 136 | CG1 state owner; seed/repack state, form `Y=W*V-state*(beta*K)`, repack U, stage checkpoint/O/final state | state/Y/U inputs, O/checkpoint-ready, TMEM-done |
| 12 | 56 | register MMA for `KK`, strict lower `L`, and three Neumann-doubling rounds for `T_inv` | T-inverse and decay-super-done |
| 13 | 56 | allocate TMEM and issue six logical block-scaled-by-schedule `tcgen05.mma` chains | state-K, state decay, U, state update, O |
| 14 | 56 | persistent scheduler and six TensorMap GMEM-to-SMEM input loads | Q/K/V/Gate/Beta/W ready |
| 15 | 56 | register MMA for causal `A`, one-behind O and checkpoint TensorMap stores | A ready, O/checkpoint done |

Dispatch order is exactly source order: warp 14, warp 12, warp 13, warp 15,
warps 0..7, then warps 8..11.  Every role owns an independent scheduler cursor
that starts once per kernel, advances once per completed work item, and is not
reset at work-item boundaries.

## Primitive vocabulary and translation boundary

All storage is one-dimensional.  Names that look like matrices below are only
logical comments on integer offsets and strides; they are never layout values,
views, tiles, fragments, or first-class descriptors.

```python
linear_buffer(space, dtype, elements, byte_offset, alignment, lifetime)
reg_array(dtype, elements)
smem_byte(arena, base, stage, row, col,
          stage_bytes, row_bytes, elem_bytes, xor_mask)
tmem_cell(base, row, column)       # base + column + (row << 16)
gmem_element(base, scalar_index)
descriptor_slot(workspace, array, batch)  # (array*B+batch)*128 bytes
```

Directional movements are only `copy_p2g`, `copy_g2s`, `copy_s2g`,
`copy_g2r`, `copy_r2g`, `copy_s2r`, `copy_r2s`, `copy_t2r`, and `copy_r2t`.
Computations are only `fill`, `cast`, `add`, `sub`, `mul`, `fma`, `min`,
`max`, `select`, `exp2`, `tanh`, `rsqrt`, `rcp`, `shuffle_xor`, and `gemm`.
Barrier init/wait/arrive/expect-tx, commit, fence, synchronization, cursor
advancement, register-budget changes, TensorMap replacement, and TMEM lifetime
operations are scheduling actions rather than computation primitives.

The implementation may import device language only as
`import tirx_kernels.kern as K`.  It may not use `T`, `Tx`, `I`,
`tirx.tile.*`, a categorized tile primitive, `TilePrimitiveCall`, any
first-class layout, or rank>1 SMEM.  The single rank-1 `u8` arena is the only
shared-memory declaration.  `K.smem_pool(base=arena)` allocates only the
protocol/header prefix; all payload mappings use integer byte arithmetic and
raw SMEM/TMEM descriptors.

## Complete non-executable sketch

```python
@specialize(
    IO_DTYPE={"bf16", "f16"}, STATE_DTYPE={"f32", "bf16"},
    BT=16, DK=128, DV=128, ACC_DTYPE="f32",
    RAW_STAGES=(3 if CHECKPOINTS else 5),
    RAW_READY_STAGES=(4 if CHECKPOINTS else 6),
    CHECKPOINT_STAGES=(2 if CHECKPOINTS else 1),
    O_STAGES=2, DECAY_STAGES=2, INTERMEDIATE_STAGES=2,
    DIAG_STAGES=4, QK_READY_STAGES=4, QSTATE_STAGES=2,
    SCHED_STAGES=8, CLUSTER=(1,1,1),
)
def gdn2_prefill_factory(...):
    return descriptor_order_prologue, persistent_gdn2

# ========================================================================
# Launch 1: source-order work table and eight TensorMap descriptor arrays
# ========================================================================

@kernel(grid=(1,1,1), block=(1024,1,1), warps=32, target="sm_100a")
def descriptor_order_prologue(
    ORDER_GENERATE, HAS_SCHEDULER, BT,
    base_q, base_k, base_v, base_gate,
    base_beta, base_w, base_o, base_checkpoint,
    descriptor_workspace, cu_seqlens,
    q, k, v, gate, beta, w, o, checkpoints_or_dummy,
    staging_items_or_dummy, work_count, work_items,
    sched_counter_or_dummy, batch_count,
    q_row_stride, k_row_stride, v_row_stride, gate_row_stride,
    beta_row_stride, w_row_stride, o_row_stride,
    checkpoint_row_stride, checkpoint_every_n_tokens,
):
    tid = thread_id()
    warp = tid // 32

    # This launch has one rank-1 u8 arena.  Order uses two 4096-i32 regions
    # plus a two-i32 spread region; descriptor building needs no SMEM payload.
    arena = linear_buffer("smem", "u8", 32776, 0, 16, "whole launch")
    order_key = integer_region(arena, 0, 4096, "i32")
    order_idx = integer_region(arena, 16384, 4096, "i32")
    order_spread = integer_region(arena, 32768, 2, "i32")

    if tid == 0 and HAS_DYNAMIC_SCHEDULER:
        copy_r2g(i32(0), sched_counter[0])
        # instruction_selection: st.global.u32; extent: one scalar

    if RUN_ORDER:
        n = batch_count * HO if GENERATE_UNCUT else copy_g2r(work_count[0])
        if GENERATE_UNCUT and tid == 0:
            copy_r2g(n, work_count[0])
        if n > 4096:
            for item in range(tid, n, 1024):
                if GENERATE_UNCUT:
                    batch = item // HO
                    head = item - batch * HO
                    bos = copy_g2r(cu_seqlens[batch])
                    eos = copy_g2r(cu_seqlens[batch+1])
                    chunks = ceil_div(eos-bos, 16)
                    row = [batch,head,0,chunks,0,chunks,bos,eos]
                else:
                    row = copy_g2r(staging_items[item,0:8])
                copy_r2g(row, work_items[item,0:8])
                # instruction_selection: generated rows may use two
                # st.global.v4.b32 stores; scratch rows use eight scalar
                # ld.global.b32 plus eight scalar st.global.b32 operations
        else:
            # Source LPT ordering: load/generate rows, compute chunk-span key,
            # find spread, stable bitonic sort by descending span and index,
            # then write every eight-i32 row.  All 1024 threads participate in
            # each CTA barrier even when n is small.
            if tid == 0:
                copy_r2s(i32_max,order_spread[0])
                copy_r2s(i32_min,order_spread[1])
            padded_count = next_power_of_two(n)
            barrier(0,1024)
            local_min = i32_max
            local_max = i32_min
            for element in range(4):
                item = tid + element*1024
                if item < padded_count:
                    key = i32_min
                    if item < n:
                        row = generate_or_load_eight_fields(item)
                        key = row_span(row)
                        local_min = min(local_min,key)
                        local_max = max(local_max,key)
                    copy_r2s(key,order_key[item])
                    copy_r2s(item,order_idx[item])
            shared_atomic_min(order_spread[0],local_min)
            shared_atomic_max(order_spread[1],local_max)
            # instruction_selection: atom.shared::cta.min/max.s32;
            # extent: one pair per thread after four local items
            barrier(0,1024)
            spread_min = copy_s2r(order_spread[0])
            spread_max = copy_s2r(order_spread[1])
            stable_lpt_bitonic_sort(order_key, order_idx, n, order_spread)
            barrier(0,1024)
            for rank in range(tid, n, 1024):
                source = copy_s2r(order_idx[rank])
                row = generate_or_load_eight_fields(source)
                copy_r2g(row, work_items[rank,0:8])
                # instruction_selection: generated rows may use two v4
                # stores; scratch rows retain eight scalar global loads/stores

    # Exactly one descriptor array belongs to each warp 0..7.  Its elected
    # lane walks all batches serially.  Slots are Q,K,V,Gate,Beta,W,O,
    # Checkpoint in that exact warp/array order.  Basic has only arrays 0..6;
    # warp 7 is compile-time inactive without checkpoints.
    base_maps = (base_q,base_k,base_v,base_gate,
                 base_beta,base_w,base_o,base_checkpoint)
    checkpoint_prefix = 0
    if warp < 8 and elected_lane() and (warp < 7 or CHECKPOINTS):
        array = warp
        for batch in range(batch_count):
            bos = copy_g2r(cu_seqlens[batch])
            eos = copy_g2r(cu_seqlens[batch+1])
            length = eos - bos
            dst = descriptor_slot(descriptor_workspace, array, batch)
            copy_p2g(base_maps[array], dst)
            # instruction_selection: grid-constant ld.param.v2.b64 payload
            # loading followed by sixteen scalar st.global.b64 stores;
            # extent: exactly 128 bytes per descriptor slot
            if array == 7:
                checkpoint_count = (0 if length == 0 else
                                    (length-1)//checkpoint_every_n_tokens+1)
                replace_global_address(
                    dst, checkpoints_or_dummy +
                         checkpoint_prefix*checkpoint_row_stride)
                replace_global_dim(dst, checkpoint_ordinal, checkpoint_count)
                checkpoint_prefix += checkpoint_count
            else:
                replace_global_address(
                    dst, tensor_base(array) + bos*row_stride(array))
                replace_global_dim(dst, sequence_ordinal(array), length)
        fence("tensormap_generic_release_gpu", array)
        # instruction_selection: tensormap.replace tile fields followed by
        # one fence.proxy.tensormap::generic.release.gpu per active warp

# ========================================================================
# Launch 2 ABI, scalar maps, arena and protocol
# ========================================================================

@kernel(grid=(num_sms,1,1), block=(512,1,1), warps=16,
        cluster=(1,1,1), min_blocks_per_sm=1, target="sm_100a")
def persistent_gdn2(
    descriptor_workspace, n_desc,
    q, k, v, gate, a_log_or_dummy, dt_bias_or_dummy, beta, w,
    cu_seqlens, initial_state_or_dummy, o, final_state_or_dummy,
    work_items, work_count, sched_counter_or_dummy,
    scale, checkpoint_every_n_tokens,
):
    tid = thread_id()
    warp = warp_uniform(tid // 32)
    lane = tid & 31
    cta = block_id_x()
    grid_x = grid_dim_x()
    total_tiles = copy_g2r(work_count[0])

    def decode_work(tile):
        row = copy_g2r(work_items[tile,0:8])
        batch, head, wstart, wend, cstart, cend, bos, eos = row
        num_chunks_b = ceil_div(eos-bos,16)
        return row_with_derived_fields(row,num_chunks_b=num_chunks_b)
        # instruction_selection: role-projected work-row decode, normally one
        # ld.global.v4.b32 with a used-byte mask plus required scalar loads;
        # it is intentionally not asserted as two full-vector loads

    def input_head(head, ratio):
        return head if ratio == 1 else head // ratio

    def token_valid(item, chunk, row):
        return chunk*16 + row < item.eos-item.bos

    def descriptor(array, batch):
        return descriptor_slot(descriptor_workspace, array, batch)

    # One rank-1 u8 declaration.  All offsets are bytes and declaration order
    # is identical to source.  Normal anchor offsets follow; checkpoint mode
    # keeps the first 1024 bytes and early payload bases but shortens the raw
    # rings and appends a 65536-byte two-stage checkpoint ring.
    arena = linear_buffer(
        "smem", "u8", 211968 if CHECKPOINTS else 203776,
        0, 1024, "whole launch")

    BARRIER_BASE = 0
    TMEM_MAILBOX = 784 if CHECKPOINTS else 960
    SCHED_BASE = 800 if CHECKPOINTS else 976
    K_DECAY_BASE = 1024
    Q_DECAY_BASE = 9216
    K_RESTORE_BASE = 17408
    INTERMEDIATE_BASE = 25600
    Q_RAW_BASE = 27648
    K_RAW_BASE = 48128 if not CHECKPOINTS else 39936
    V_RAW_BASE = 68608 if not CHECKPOINTS else 52224
    GATE_RAW_BASE = 89088 if not CHECKPOINTS else 64512
    DIAG_BASE = 130048 if not CHECKPOINTS else 89088
    K_INV_BASE = 146432 if not CHECKPOINTS else 105472
    O_BASE = 154624 if not CHECKPOINTS else 113664
    BETA_RAW_BASE = 162816 if not CHECKPOINTS else 121856
    W_RAW_BASE = 183296 if not CHECKPOINTS else 134144
    CHECKPOINT_BASE = 146432  # checkpoint specialization only

    # Normal protocol byte offsets.  Each tuple is
    # (name, ready, done-or-None, ready stages, done stages,
    #  ready arrivals, done arrivals, init owner).
    NORMAL_PROTOCOL = (
      ("q",       0,  48, 6,5,   1,128,14),
      ("k",      88, 136, 6,5,   1,128,14),
      ("v",     176, 224, 6,5,   1,128,14),
      ("w",     264, 312, 6,5,   1,128,14),
      ("gate",  352, 400, 6,5,   1,128,14),
      ("beta",  440, 488, 6,5,   1,128,14),
      ("o_acc", 528, 536, 1,2,   1,128,13),
      ("state_k",552,None,1,0,    1,None,13),
      ("u_acc", 560,None,1,0,     1,None,13),
      ("state_inp",568,None,1,0,128,None,13),
      ("y_inp", 576,None,1,0,    128,None,13),
      ("u_inp", 584,None,1,0,    128,None,13),
      ("t_inv", 592,608,2,2,      32,1,12),
      ("a",     624,None,2,0,      32,None,12),
      ("k_decay",640,656,2,2,    128,1,12),
      ("decay_super",672,None,2,0, 64,None,13),
      ("qk_scale",688,None,4,0,  128,None,12),
      ("k_restore",720,None,2,0,   1,None,13),
      ("diag_done",736,None,4,0,   1,None,13),
      ("tmem_done",768,None,1,0, 128,None,13),
      ("state_read",776,None,1,0,128,None,15),
      ("o_stage",784,800,2,2,    128,32,15),
      ("checkpoint",816,824,1,1,128,32,15),
      ("scheduler",832,896,8,8,    1,15,15),
    )

    # Checkpoint mode has its own exact declaration-ordered physical table.
    # The tuples have the same fields as NORMAL_PROTOCOL.  Header allocation
    # ends at byte 784, the TMEM mailbox occupies 784..787, the scheduler
    # payload occupies 800..831, and payload storage still begins at 1024.
    CHECKPOINT_PROTOCOL = (
      ("q",       0,  32, 4,3,   1,128,14),
      ("k",      56,  88, 4,3,   1,128,14),
      ("v",     112, 144, 4,3,   1,128,14),
      ("w",     168, 200, 4,3,   1,128,14),
      ("gate",  224, 256, 4,3,   1,128,14),
      ("beta",  280, 312, 4,3,   1,128,14),
      ("o_acc", 336, 344, 1,2,   1,128,13),
      ("state_k",360,None,1,0,    1,None,13),
      ("u_acc", 368,None,1,0,     1,None,13),
      ("state_inp",376,None,1,0,128,None,13),
      ("y_inp", 384,None,1,0,    128,None,13),
      ("u_inp", 392,None,1,0,    128,None,13),
      ("t_inv", 400,416,2,2,      32,1,12),
      ("a",     432,None,2,0,      32,None,12),
      ("k_decay",448,464,2,2,    128,1,12),
      ("decay_super",480,None,2,0, 64,None,13),
      ("qk_scale",496,None,4,0,  128,None,12),
      ("k_restore",528,None,2,0,   1,None,13),
      ("diag_done",544,None,4,0,   1,None,13),
      ("tmem_done",576,None,1,0, 128,None,13),
      ("state_read",584,None,1,0,128,None,15),
      ("o_stage",592,608,2,2,    128,32,15),
      ("checkpoint",624,640,2,2,128,32,15),
      ("scheduler",656,720,8,8,    1,15,15),
    )
    PROTOCOL = CHECKPOINT_PROTOCOL if CHECKPOINTS else NORMAL_PROTOCOL

    # Integer offset generation is compile-time; no layout object exists.
    pool = smem_pool(base=arena)
    for edge in PROTOCOL:
        allocate_edge_from_pool(edge)

    # The diagonal region is explicitly zeroed by all 512 threads before any
    # role runs.  Only diagonal cells are overwritten by CG0.
    for i in range(tid, DIAG_BYTES // 2, 512):
        copy_r2s(cast(IO_DTYPE,0), arena[DIAG_BASE + 2*i])

    if warp == 14: init_raw_ready_and_done(PROTOCOL)
    elif warp == 13:
        init_tcgen_owned_edges(PROTOCOL)  # includes decay_super_done
    elif warp == 12: init_register_mma_owned_edges(PROTOCOL)
    elif warp == 15:
        init_o_and_scheduler_edges(PROTOCOL)
        if CHECKPOINTS:
            init_checkpoint_and_state_read_edges(PROTOCOL)
    # Static initialized physical sites: basic=117, checkpoint=98.
    fence("mbarrier_init_release_cluster")
    barrier(0,512)

    # TMEM columns.  BF16/F16 packed input regions use two scalar elements per
    # 32-bit TMEM cell; all accumulator regions are FP32.
    tmem_allocated = integer_base_from_mailbox(TMEM_MAILBOX)
    tmem_col = tmem_allocated & 0xffff
    tmem_row = tmem_allocated >> 16
    STATE_ACC = 0                 # columns 0..127
    STATE_INPUT = 128             # packed columns 128..191
    Q_STATE_ACC = 192             # two stages, columns 192..223
    STATE_K_ACC = 224             # columns 224..239
    U_ACC = 240                   # columns 240..255
    Y_INPUT = 256                 # packed columns 256..263
    U_INPUT = 264                 # packed columns 264..271
    assert U_INPUT + 8 <= 512

    def wait_edge(name, stage, phase):
        wait(protocol_ptr(name, stage), phase)
        # instruction_selection:
        # mbarrier.try_wait.parity.acquire.cta.shared::cta.b64

    def arrive_edge(name, stage=0):
        arrive(protocol_ptr(name, stage))
        # instruction_selection: mbarrier.arrive.shared.b64

    def expect_edge(name, stage, nbytes):
        expect_bytes(protocol_ptr(name, stage), nbytes)
        # instruction_selection: mbarrier.arrive.expect_tx.shared.b64

    def commit_edge(name, stage=0):
        commit(protocol_ptr(name, stage))
        # instruction_selection:
        # tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64

    def scheduler_publish(cursor, tile):
        if DYNAMIC:
            wait_edge("scheduler_done", cursor.stage, cursor.phase)
            if elected_lane():
                ticket = atomic_add(sched_counter[0], 1)
                copy_r2s(grid_x + ticket, arena[SCHED_BASE+4*cursor.stage])
            warp_sync()
            next_tile = copy_s2r(arena[SCHED_BASE+4*cursor.stage])
            if elected_lane(): arrive_edge("scheduler", cursor.stage)
            return next_tile, advance(cursor)
        return tile + grid_x, cursor

    def scheduler_consume(cursor, tile):
        if DYNAMIC:
            wait_edge("scheduler", cursor.stage, cursor.phase)
            next_tile = copy_s2r(arena[SCHED_BASE+4*cursor.stage])
            if elected_lane(): arrive_edge("scheduler_done", cursor.stage)
            return next_tile, advance(cursor)
        return tile + grid_x, cursor

    # ====================================================================
    # Warp 14: six raw TensorMap loads and scheduler producer
    # ====================================================================
    if warp == 14:
        set_register_budget("decrease",56)
        raw = producer_cursor(RAW_STAGES, phase=1)
        raw_ready = producer_cursor(RAW_READY_STAGES, phase=0)
        sched = producer_cursor(8, phase=1)
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            qh = input_head(item.head, Q_RATIO)
            kh = input_head(item.head, K_RATIO)
            vh = input_head(item.head, V_RATIO)
            for array in (Q,K,V,GATE,BETA,W):
                if elected_lane():
                    fence("tensormap_generic_acquire_gpu",
                          descriptor(array,item.batch))
            for chunk in range(item.cstart, item.wend):
                coords = (0, mapped_head, chunk*16)
                for name, array, bytes, subtiles in (
                    ("q",Q,4096,2), ("k",K,4096,2),
                    ("v",V,4096,2), ("beta",BETA,4096,2),
                    ("w",W,4096,2), ("gate",GATE,8192,4)):
                    wait_edge(name+"_done", raw.stage, raw.phase)
                    if elected_lane():
                        expect_edge(name,raw_ready.stage,bytes)
                    copy_g2s(descriptor(array,item.batch),
                             raw_stage_base(name,raw.stage),
                             coords, completion=protocol_ptr(name,raw_ready.stage))
                    # instruction_selection: rank-3 TMA GMEM->SMEM;
                    # extent: two 64-channel subtiles except four gate subtiles
                raw = advance(raw)
                raw_ready = advance(raw_ready)
            # The dynamic ticket is acquired only after this work item's
            # complete raw-load loop, matching source sched_publish_next.
            next_tile, sched = scheduler_publish(sched, tile)
            tile = next_tile

    # ====================================================================
    # Warp 12: register KK and T-inverse
    # ====================================================================
    elif warp == 12:
        set_register_budget("decrease",56)
        sched = consumer_cursor(8,phase=0)
        serial_base = 0
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            for local_chunk in range(item.wend-item.cstart):
                serial = serial_base + local_chunk
                decay_stage = serial % 2
                inter_stage = serial % 2
                wait_edge("k_decay",decay_stage,(serial//2)&1)

                kk = fill(reg_array("f32",8),0)
                for k_block in range(8):
                    lhs = copy_s2r(k_decay_ldmatrix_address(decay_stage,k_block,lane))
                    rhs = copy_s2r(k_inverse_ldmatrix_address(decay_stage,k_block,lane))
                    gemm(kk,lhs,rhs,accumulate=k_block>0)
                    # instruction_selection: two
                    # mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32
                    # per k block; extent: 16 instructions

                L = select(strict_lower_fragment(lane),kk,0)
                Lpacked = cast(IO_DTYPE,L)
                Tinv = identity_fragment(lane) - cast("f32",Lpacked)
                Lpow = Lpacked
                moved_Lpow = [move_matrix(Lpow[pair]) for pair in range(4)]
                # instruction_selection: four movmatrix.sync.aligned.m8n8.trans.b16
                # before the first round
                for round in range(3):
                    Lpow = cast(IO_DTYPE,gemm(Lpow,moved_Lpow))
                    moved_Lpow = [move_matrix(Lpow[pair]) for pair in range(4)]
                    # instruction_selection: exactly four movmatrix sites
                    # refreshed after this squaring MMA
                    Tround = cast(IO_DTYPE,Tinv)
                    Tinv = cast("f32",Tround) + gemm(Tround,moved_Lpow)
                    # instruction_selection: four mma.sync per round plus
                    # cvt/fadd.rn.f32x2; extent: 12 MMA instructions and a
                    # total of 4 + 3*4 = 16 movmatrix instructions
                wait_edge("intermediate_done",inter_stage,(serial//2+1)&1)
                copy_r2s(cast(IO_DTYPE,Tinv),
                         intermediate_tinv_stmatrix_address(inter_stage,lane))
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16
                fence("async_shared_cta")
                arrive_edge("t_inv",inter_stage)
                arrive_edge("decay_super",decay_stage)
            serial_base += item.wend-item.cstart
            tile,sched = scheduler_consume(sched,tile)

    # ====================================================================
    # Warp 13: TMEM allocation and six logical tcgen05 chains
    # ====================================================================
    elif warp == 13:
        set_register_budget("decrease",56)
        allocate_tmem_warp_level(TMEM_MAILBOX,columns=512,cta_group=1)
        barrier(3,160)  # warp 13 plus CG1 warps 8..11
        sched = consumer_cursor(8,phase=0)
        state_inp_cursor = consumer_cursor(1,phase=0)
        state_read_cursor = consumer_cursor(1,phase=0)
        y_cursor = consumer_cursor(1,phase=0)
        u_cursor = consumer_cursor(1,phase=0)
        serial_base = 0
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            for local_chunk in range(item.wend-item.cstart):
                serial = serial_base + local_chunk
                have_state = USE_INITIAL_STATE or local_chunk > 0
                decay_stage = serial % 2
                qstate_stage = serial % 2
                inter_stage = serial % 2
                diag_stage = serial % 4

                wait_edge("k_decay",decay_stage,(serial//2)&1)
                if have_state:
                    wait_edge("state_inp",0,state_inp_cursor.phase)
                    state_inp_cursor = advance(state_inp_cursor)
                    for k_phase in range(8):
                        gemm(tmem_cell(tmem_col,tmem_row,STATE_K_ACC),
                             tmem_state_input_subtile(k_phase),
                             smem_k_decay_subtile(decay_stage,k_phase),
                             accumulate=k_phase>0)
                    # instruction_selection: 8 x tcgen05.mma...kind::f16
                    commit_edge("state_k",0)

                wait_edge("qk_scale",diag_stage,(serial//4)&1)
                wait_edge("o_acc_done",qstate_stage,(serial//2+1)&1)
                if have_state:
                    for k_phase in range(8):
                        gemm(tmem_cell(tmem_col,tmem_row,Q_STATE_ACC+16*qstate_stage),
                             tmem_state_input_subtile(k_phase),
                             smem_q_decay_subtile(decay_stage,k_phase),
                             accumulate=k_phase>0)
                    # instruction_selection: 8 x tcgen05.mma...kind::f16
                commit_edge("decay_tcgen05",decay_stage)

                if CHECKPOINTS and have_state:
                    wait_edge("state_read",0,state_read_cursor.phase)
                    state_read_cursor = advance(state_read_cursor)
                if have_state:
                    for key_block in range(8):
                        gemm(tmem_cell(tmem_col,tmem_row,STATE_ACC+16*key_block),
                             tmem_state_input_block(key_block),
                             smem_diag_block(diag_stage,key_block),
                             accumulate=False)
                    # instruction_selection: 8 x tcgen05.mma...kind::f16
                commit_edge("diag_done",diag_stage)

                wait_edge("t_inv",inter_stage,(serial//2)&1)
                wait_edge("y_inp",0,y_cursor.phase)
                y_cursor = advance(y_cursor)
                gemm(tmem_cell(tmem_col,tmem_row,U_ACC),
                     tmem_y_input(),smem_tinv(inter_stage),accumulate=False)
                # instruction_selection: 1 x tcgen05.mma...kind::f16
                commit_edge("u_acc",0)

                wait_edge("u_inp",0,u_cursor.phase)
                u_cursor = advance(u_cursor)
                gemm(tmem_cell(tmem_col,tmem_row,STATE_ACC),
                     tmem_u_input(),smem_k_restore(decay_stage),
                     accumulate=have_state)
                # instruction_selection: 1 x tcgen05.mma...kind::f16
                commit_edge("k_restore",decay_stage)

                wait_edge("a",inter_stage,(serial//2)&1)
                gemm(tmem_cell(tmem_col,tmem_row,Q_STATE_ACC+16*qstate_stage),
                     tmem_u_input(),smem_a(inter_stage),
                     accumulate=have_state)
                # instruction_selection: 1 x tcgen05.mma...kind::f16
                commit_edge("o_acc",0)
                commit_edge("intermediate_done",inter_stage)
            serial_base += item.wend-item.cstart
            tile,sched = scheduler_consume(sched,tile)
        wait_edge("tmem_done",0,0)
        relinquish_tmem_alloc_permit(cta_group=1)
        deallocate_tmem(tmem_col,columns=512,cta_group=1)

    # ====================================================================
    # Warp 15: causal A and one-behind stores
    # ====================================================================
    elif warp == 15:
        set_register_budget("decrease",56)
        sched = consumer_cursor(8,phase=0)
        checkpoint_ready = consumer_cursor(CHECKPOINT_STAGES,phase=0)
        serial_base = 0
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            if CHECKPOINTS:
                acquire_descriptor(CHECKPOINT,item.batch)
                if item.wstart == 0 and item.wend>item.cstart:
                    checkpoint_stage = checkpoint_ready.stage
                    wait_edge("checkpoint",checkpoint_stage,
                              checkpoint_ready.phase)
                    checkpoint_ready = advance(checkpoint_ready)
                    copy_s2g(checkpoint_stage_base(checkpoint_stage),
                             checkpoint_coord(item.head,0))
                    store_commit_and_wait(0)
                    arrive_edge("checkpoint_done",checkpoint_stage)
            acquire_descriptor(O,item.batch)
            for local_chunk in range(item.wend-item.cstart):
                serial = serial_base+local_chunk
                decay_stage=serial%2
                inter_stage=serial%2
                diag_stage=serial%4
                wait_edge("qk_scale",diag_stage,(serial//4)&1)
                A=fill(reg_array("f32",8),0)
                for k_block in range(8):
                    lhs=copy_s2r(q_decay_ldmatrix_address(decay_stage,k_block,lane))
                    rhs=copy_s2r(k_inverse_ldmatrix_address(decay_stage,k_block,lane))
                    gemm(A,lhs,rhs,accumulate=k_block>0)
                    # instruction_selection: 2 x mma.sync per block; extent 16
                A=select(causal_lower_or_diagonal_fragment(lane),A,0)
                wait_edge("intermediate_done",inter_stage,(serial//2+1)&1)
                copy_r2s(cast(IO_DTYPE,A),intermediate_a_stmatrix_address(inter_stage,lane))
                # instruction_selection: stmatrix...m8n8.x4.shared.b16
                fence("async_shared_cta")
                arrive_edge("a",inter_stage)
                arrive_edge("decay_super",decay_stage)

                if local_chunk>0:
                    previous=item.cstart+local_chunk-1
                    did_checkpoint = False
                    did_o = False
                    chunk_idx = item.cstart+local_chunk
                    do_checkpoint = (CHECKPOINTS and
                                     is_checkpoint_boundary(chunk_idx) and
                                     chunk_idx >= item.wstart)
                    if do_checkpoint:
                        checkpoint_stage = checkpoint_ready.stage
                        wait_edge("checkpoint",checkpoint_stage,
                                  checkpoint_ready.phase)
                        checkpoint_ready = advance(checkpoint_ready)
                        copy_s2g(checkpoint_stage_base(checkpoint_stage),
                                 checkpoint_coord(item.head,
                                                  checkpoint_index(chunk_idx)))
                        store_commit()
                        did_checkpoint = True
                    o_stage=(serial-1)%2
                    wait_edge("o_stage",o_stage,((serial-1)//2)&1)
                    if previous>=item.wstart:
                        copy_s2g(o_stage_base(o_stage),
                                 output_coord(item.head,previous*16))
                        store_commit()
                        did_o = True
                    # Checkpoint is always issued/committed before O.  The
                    # four cases are separate scheduling actions, not a helper.
                    if CHECKPOINTS and did_checkpoint and did_o:
                        store_wait_group(1)
                        arrive_edge("checkpoint_done",checkpoint_stage)
                        store_wait_group(0)
                        arrive_edge("o_stage_done",o_stage)
                    elif CHECKPOINTS and did_checkpoint and not did_o:
                        store_wait_group(0)
                        arrive_edge("checkpoint_done",checkpoint_stage)
                        arrive_edge("o_stage_done",o_stage)
                    elif did_o:
                        store_wait_group(0)
                        arrive_edge("o_stage_done",o_stage)
                    else:
                        arrive_edge("o_stage_done",o_stage)
                    # instruction_selection: four static rank-3 O S2G sites
                    # in basic and four static rank-4 checkpoint S2G sites in
                    # the checkpoint specialization; bulk-group waits above
                    # establish the source checkpoint-before-O publication.
            if item.wend>item.cstart:
                last=item.wend-1
                last_serial=serial_base+item.wend-item.cstart-1
                o_stage=last_serial%2
                wait_edge("o_stage",o_stage,(last_serial//2)&1)
                copy_s2g(o_stage_base(o_stage),output_coord(item.head,last*16))
                store_commit_and_wait(0)
                arrive_edge("o_stage_done",o_stage)
            serial_base+=item.wend-item.cstart
            tile,sched=scheduler_consume(sched,tile)

    # ====================================================================
    # Warps 0..7: two four-warp CG0 groups
    # ====================================================================
    elif 0 <= warp <= 7:
        set_register_budget("increase",160)
        group=warp//4
        local_warp=warp%4
        prefix_dim=local_warp*32+lane
        sched=consumer_cursor(8,phase=0)
        serial_base=0
        tile=cta
        while tile<total_tiles:
            item=decode_work(tile)
            if SAFE_GATE and item.wend>item.cstart:
                aexp=exp2(copy_g2r(a_log[item.head])*LOG2_E)
                dbias=copy_g2r(dt_bias[item.head,prefix_dim])
            barrier(5,256)  # both ping-pong groups exchange parity proof
            for local_chunk in range(group,item.wend-item.cstart,2):
                serial=serial_base+local_chunk
                raw_stage=rolling_data_stage_from_group(serial,RAW_STAGES)
                raw_ready_stage=rolling_ready_stage_from_group(serial,RAW_READY_STAGES)
                decay_stage=serial%2
                diag_stage=serial%4
                wait_edge("gate",raw_ready_stage,ready_phase(serial))

                # Each warp owns 32 key dimensions and four token rows.  Gate
                # TMA is f32 [16,128]; safe-gate applies exp(a_log)*(g+dt_bias),
                # tanh sigmoid and the compile-time log2 lower-bound scale.
                gates=copy_s2r(gate_stage_fragment(raw_stage,local_warp,lane))
                gates=apply_safe_or_natural_log_gate(gates,aexp,dbias,valid_rows)
                prefix=inclusive_scan_16_in_source_pair_order(gates)
                exp_prefix=exp2(prefix)
                copy_r2s(exp_prefix,gate_stage_fragment(raw_stage,local_warp,lane))
                exp_last=exp_prefix[15]

                wait_edge("diag_done",diag_stage,(serial//4+1)&1)
                copy_r2s(cast(IO_DTYPE,exp_last),
                         diagonal_cell(diag_stage,prefix_dim))
                barrier(1+group,128)

                wait_edge("q",raw_ready_stage,ready_phase(serial))
                wait_edge("k",raw_ready_stage,ready_phase(serial))
                wait_edge("beta",raw_ready_stage,ready_phase(serial))
                raw_q,raw_k,raw_beta=copy_two_x8_vectors(raw_stage,local_warp,lane)
                if BETA_SIGMOID:
                    raw_beta=cast("f32",cast(IO_DTYPE,
                        tanh(raw_beta*0.5)*0.5+0.5))
                if L2NORM:
                    qnorm=rsqrt(max(group_sum(raw_q*raw_q),EPS2))
                    knorm=rsqrt(max(group_sum(raw_k*raw_k),EPS2))
                else:
                    qnorm=knorm=opaque_one()

                # K_decay = round((K*knorm)*beta) * round(exp(prefix));
                # K_inv = round(K*knorm) * round(rcp(exp(prefix))).
                wait_edge("decay_super",decay_stage,(serial//2+1)&1)
                wait_edge("decay_tcgen05",decay_stage,(serial//2+1)&1)
                k_decay=packed_mul(cast(IO_DTYPE,raw_k*knorm*raw_beta),
                                   cast(IO_DTYPE,exp_prefix))
                k_inv=packed_mul(cast(IO_DTYPE,raw_k*knorm),
                                 cast(IO_DTYPE,rcp(exp_prefix)))
                copy_r2s(k_decay,k_decay_integer_swizzle(decay_stage))
                copy_r2s(k_inv,k_inverse_integer_swizzle(decay_stage))
                fence("async_shared_cta")
                arrive_edge("k_decay",decay_stage)
                arrive_edge("q_done",raw_stage)
                arrive_edge("k_done",raw_stage)
                arrive_edge("gate_done",raw_stage)
                arrive_edge("beta_done",raw_stage)

                # Q_decay = round(Q*qnorm) * round(exp(prefix));
                # K_restore = K_inv * round(exp(last-prefix)).
                q_decay=packed_mul(cast(IO_DTYPE,raw_q*qnorm),cast(IO_DTYPE,exp_prefix))
                k_restore=packed_mul(k_inv,cast(IO_DTYPE,exp_last))
                copy_r2s(q_decay,q_decay_integer_swizzle(decay_stage))
                copy_r2s(k_restore,k_restore_integer_swizzle(decay_stage))
                fence("async_shared_cta")
                arrive_edge("qk_scale",diag_stage)
                advance_group_owned_raw_diag_cursors_by_two()
            serial_base+=item.wend-item.cstart
            tile,sched=scheduler_consume(sched,tile)

    # ====================================================================
    # Warps 8..11: CG1 state/Y/U/O/checkpoint/final owner
    # ====================================================================
    elif 8 <= warp <= 11:
        set_register_budget("increase",136)
        barrier(3,160)
        subpartition=warp-8
        value_dim=subpartition*32+lane
        sched=consumer_cursor(8,phase=0)
        checkpoint_done=producer_cursor(CHECKPOINT_STAGES,phase=1)
        state_k_cursor=consumer_cursor(1,phase=0)
        u_acc_cursor=consumer_cursor(1,phase=0)
        o_acc_cursor=consumer_cursor(1,phase=0)
        k_restore_cursor=consumer_cursor(DECAY_STAGES,phase=0)
        raw_data_cursor=consumer_cursor(RAW_STAGES,phase=0)
        raw_ready_cursor=consumer_cursor(RAW_READY_STAGES,phase=0)
        serial_base=0
        tile=cta
        while tile<total_tiles:
            item=decode_work(tile)
            chunks=item.wend-item.cstart
            if chunks>0:
                # Source has no first-chunk state transaction when initial
                # state is absent: the first state update uses accumulate=false
                # and creates the accumulator from U*K_restore.  When initial
                # state is enabled, a work row owning cstart==0 loads it;
                # otherwise the source's guarded seed path writes zero.
                if USE_INITIAL_STATE:
                    if item.cstart==0:
                        state=copy_g2r(initial_state[
                            item.batch,item.head,value_dim,0:128])
                    else:
                        state=fill(reg_array("f32",128),0)
                    copy_r2t(state,tmem_state_acc_rows(value_dim))
                    wait_tmem_store()

                    # Pack state FP32->IO for the first state-K/Q-state/diag.
                    state=copy_t2r(tmem_state_acc_rows(value_dim))
                    packed=cast(IO_DTYPE,state,source_pair_xor=4)
                    copy_r2t(packed,tmem_state_input_rows(value_dim))
                    wait_tmem_store()
                    arrive_edge("state_inp")
                    if CHECKPOINTS and item.wstart==0:
                        cp_stage=checkpoint_done.stage
                        wait_edge("checkpoint_done",cp_stage,
                                  checkpoint_done.phase)
                        checkpoint_done=advance(checkpoint_done)
                        copy_r2s(cast(IO_DTYPE,state),
                                 checkpoint_stage_rows(cp_stage,value_dim))
                        wait_tmem_load(); fence("async_shared_cta")
                        arrive_edge("checkpoint",cp_stage)
                    if CHECKPOINTS: arrive_edge("state_read")
                elif CHECKPOINTS and item.wstart==0:
                    # The absent-initial-state checkpoint branch stages an
                    # explicit zero checkpoint but does not publish state_inp.
                    cp_stage=checkpoint_done.stage
                    wait_edge("checkpoint_done",cp_stage,
                              checkpoint_done.phase)
                    checkpoint_done=advance(checkpoint_done)
                    copy_r2s(cast(IO_DTYPE,0),
                             checkpoint_stage_rows(cp_stage,value_dim))
                    fence("async_shared_cta")
                    arrive_edge("checkpoint",cp_stage)

                for local_chunk in range(chunks):
                    serial=serial_base+local_chunk
                    raw_stage=raw_data_cursor.stage
                    raw_ready_stage=raw_ready_cursor.stage
                    if local_chunk>0:
                        # One-behind: load state and prior O accumulator, stage
                        # the prior output, optionally snapshot the incoming
                        # state, then publish the repacked state for this chunk.
                        state=copy_t2r(tmem_state_acc_rows(value_dim))
                        packed=cast(IO_DTYPE,state,source_pair_xor=4)
                        copy_r2t(packed,tmem_state_input_rows(value_dim))
                        wait_tmem_store()
                        arrive_edge("state_inp")
                        if (CHECKPOINTS and
                            checkpoint_boundary(item.cstart+local_chunk) and
                            item.cstart+local_chunk >= item.wstart):
                            cp_stage=checkpoint_done.stage
                            wait_edge("checkpoint_done",cp_stage,
                                      checkpoint_done.phase)
                            checkpoint_done=advance(checkpoint_done)
                            copy_r2s(cast(IO_DTYPE,state),checkpoint_stage_rows(cp_stage,value_dim))
                            wait_tmem_load()
                            arrive_edge("state_read")
                            fence("async_shared_cta")
                            arrive_edge("checkpoint",cp_stage)
                        elif CHECKPOINTS:
                            arrive_edge("state_read")
                        wait_edge("o_acc",0,o_acc_cursor.phase)
                        o_acc_cursor=advance(o_acc_cursor)
                        oval=copy_t2r(tmem_qstate_rows((serial-1)%2,value_dim))
                        oval=cast(IO_DTYPE,oval*scale)
                        wait_edge("o_stage_done",(serial-1)%2,
                                  (((serial-1)//2)+1)&1)
                        copy_r2s(oval,o_stage_rows((serial-1)%2,value_dim))
                        arrive_edge("o_acc_done",(serial-1)%2)
                        fence("async_shared_cta")
                        arrive_edge("o_stage",(serial-1)%2)

                    wait_edge("v",raw_ready_stage,raw_ready_cursor.phase)
                    raw_v=copy_s2r(v_ldmatrix_transpose(raw_stage,value_dim,lane))
                    wait_edge("w",raw_ready_stage,raw_ready_cursor.phase)
                    raw_w=copy_s2r(w_ldmatrix_transpose(raw_stage,value_dim,lane))
                    if USE_INITIAL_STATE or local_chunk>0:
                        wait_edge("state_k",0,state_k_cursor.phase)
                        state_k_cursor=advance(state_k_cursor)
                        erase=copy_t2r(tmem_state_k_rows(value_dim))
                        y=packed_sub(packed_mul(raw_w,raw_v),cast(IO_DTYPE,erase))
                    else:
                        y=packed_mul(raw_w,raw_v)
                    copy_r2t(y,tmem_y_input_rows(value_dim))
                    wait_tmem_store()
                    arrive_edge("v_done",raw_stage)
                    arrive_edge("w_done",raw_stage)
                    arrive_edge("y_inp")

                    wait_edge("u_acc",0,u_acc_cursor.phase)
                    u_acc_cursor=advance(u_acc_cursor)
                    u=copy_t2r(tmem_u_rows(value_dim))
                    copy_r2t(cast(IO_DTYPE,u,source_pair_xor=4),tmem_u_input_rows(value_dim))
                    wait_tmem_store(); arrive_edge("u_inp")
                    wait_edge("k_restore",k_restore_cursor.stage,
                              k_restore_cursor.phase)
                    k_restore_cursor=advance(k_restore_cursor)
                    raw_data_cursor=advance(raw_data_cursor)
                    raw_ready_cursor=advance(raw_ready_cursor)

                # Final O drain occurs before final-state reads/stores.
                wait_edge("o_acc",0,o_acc_cursor.phase)
                o_acc_cursor=advance(o_acc_cursor)
                oval=copy_t2r(tmem_qstate_rows(last_serial%2,value_dim))
                oval=cast(IO_DTYPE,oval*scale)
                wait_edge("o_stage_done",last_serial%2,
                          ((last_serial//2)+1)&1)
                copy_r2s(oval,o_stage_rows(last_serial%2,value_dim))
                arrive_edge("o_acc_done",last_serial%2)
                fence("async_shared_cta")
                arrive_edge("o_stage",last_serial%2)

                if STORE_FINAL_STATE and item.wend==item.num_chunks_b:
                    state=copy_t2r(tmem_state_acc_rows(value_dim))
                    copy_r2g(cast(STATE_DTYPE,state),
                             final_state[item.batch,item.head,value_dim,0:128])
            elif STORE_FINAL_STATE:
                # Empty sequence: the source never touches TMEM.  Preserve an
                # initial state when present, otherwise write state-dtype zero.
                for key in range(128):
                    value = (copy_g2r(initial_state[
                                item.batch,item.head,value_dim,key])
                             if USE_INITIAL_STATE else cast(STATE_DTYPE,0))
                    copy_r2g(value,final_state[
                        item.batch,item.head,value_dim,key])
            serial_base+=chunks
            tile,sched=scheduler_consume(sched,tile)
        arrive_edge("tmem_done")
```

## Numerical order frozen by the schedule

For one token, the source recurrence represented by the chunk factorization is:

```text
S_decayed = exp(g_t) * S
erase     = (beta_t * k_t) @ S_decayed
v_new     = w_t * v_t - erase
S         = S_decayed + outer(k_t, v_new)
o_t       = (scale * q_t) @ S
```

Within a chunk the exact source order matters: gate is converted to log2 and
prefix-scanned in packed `fadd.rn.f32x2` order; Q/K normalization uses the
three XOR reductions `4,2,1`; beta sigmoid is rounded through IO dtype;
K-decay/inverse/Q-decay/restore operands are rounded and multiplied in packed
IO pairs; both inverse construction and A use register MMA; state, U, and O
accumulate in FP32 TMEM.  Tail rows get zero gate contribution and TMA OOB
zero fill; output stores remain predicated by descriptor sequence extent.

## Logical tcgen05 GEMMs

| # | FP32 TMEM destination | A operand | B operand | logical MxNxK | issues | accumulate |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | state-K col 224 | packed state TMEM | K-decay SMEM | 128x16x128 | 8 | no |
| 2 | Q-state col 192/208 | packed state TMEM | Q-decay SMEM | 128x16x128 | 8 | no |
| 3 | state cols 0..127 | packed state TMEM | four-stage block diagonal | 128x128x16 per key block | 8 | no |
| 4 | U col 240 | packed Y TMEM | T-inverse SMEM | 128x16x16 | 1 | no |
| 5 | state cols 0..127 | packed U TMEM | K-restore SMEM transposed | 128x128x16 | 1 | runtime `have_state` |
| 6 | Q-state col 192/208 | packed U TMEM | causal A SMEM | 128x16x16 | 1 | runtime `have_state` |

The source anchor therefore has 27 static `tcgen05.mma` sites.  Publication is
separate and follows the exact delayed commit sites in warp 13.

## Source / PTX / sketch correspondence

The table uses source action granularity.  PTX ranges are identified by `.loc`
because generated filenames contain the full specialization and line numbers
vary across accepted compile-time branches.  `basic-main` and
`basic-prologue` mean the two files in `source_export/basic_001`.

| Source action | Source lines | PTX evidence | Sketch section |
| --- | ---: | --- | --- |
| scheduler publish/consume | 252..279 | dynamic-main `.loc 1 252..279`; `atom.global.add.u32` and scheduler mbarriers | `scheduler_publish`, `scheduler_consume` |
| six raw TensorMap loads | 281..447 | basic-main `.loc 1 281..447`; 14 rank-3 G2S sites | warp 14 |
| KK and Neumann inverse | 449..629 | basic-main `.loc 1 449..629`; 28 register MMA, movmatrix, stmatrix | warp 12 |
| six tcgen chains and teardown | 630..869 | basic-main `.loc 1 630..869`; 27 tcgen MMA, commit, alloc/dealloc | warp 13 |
| causal A and store drain | 870..1101 | basic-main `.loc 1 870..1101`; 16 register MMA, S2G and waits | warp 15 |
| gate transform and CG0 operands | 1102..1490 | basic-main `.loc 1 1102..1490`; packed f32/IO and shared traffic | warps 0..7 |
| CG1 state/value/output | 1492..2097 | basic-main `.loc 1 1492..2097`; TMEM ld/st and shared/global stores | warps 8..11 |
| main ABI, arena, init, dispatch | 2152..2480 | basic-main `.loc 1 2152..2480`; 512-thread launch and 117 init sites | ABI/protocol/dispatch |
| cfg stage/offset derivation | 2480..2652 | compile-time constants in basic/checkpoint PTX | specialization and arena |
| seven base descriptor bodies | 2653..2735 | basic-prologue `.loc 1 2653..2735`; arrays Q/K/V/Gate/Beta/W/O | launch 1 descriptor loop |
| eighth checkpoint descriptor body | 2653..2735 | checkpoint-prologue `.loc 1 2653..2735`; checkpoint prefix address/extent | launch 1 descriptor loop |
| work order/generation | 2736..2953 and `split_k.py` | order-scratch/dynamic prologue line-info | launch 1 order branch |
| FP16 instruction selection | compile branch | fp16-main: `mma.sync...f16`, FP16 converts | `IO_DTYPE` specialization |
| grouped head mapping | runtime ratios | grouped-main signed ratio divisions before descriptor coordinates | warp 14 / decode |
| checkpoint rings and stores | checkpoint branches | checkpoint-main/prologue `.loc 1`; rank-4 S2G and raw3 rings | warp 13/15/CG1 |
| safe gate, L2, beta sigmoid | accepted optional branches | feature-main `.loc 1 1102..1490` | CG0 |
| initial/final BF16 state | accepted state branches | feature/state-bf16 exports, CG1 `.loc 1 1492..2097` | CG1 |

Reverse lookup is exhaustive at action granularity: each accepted compile-time
branch, storage family, role, logical GEMM, scheduler operation, descriptor
operation, and pipeline family has one owner in the table and one concrete
section above.

## Static instruction evidence for the BF16 basic anchor

| Instruction family | Static sites | Owner |
| --- | ---: | --- |
| `mbarrier.init.shared.b64` | 117 basic / 98 checkpoint | distributed protocol init |
| `mbarrier.try_wait.parity.acquire...` | 42 | role-local waits |
| `mbarrier.arrive.shared.b64` | 26 | role publications/reuse |
| `mbarrier.arrive.expect_tx.shared.b64` | 6 | warp-14 TMA transaction starts |
| `tcgen05.mma...kind::f16` | 27 | warp 13 |
| `tcgen05.commit...mbarrier` | 7 | warp 13 publications |
| `mma.sync...m16n8k16...bf16` | 44 | warps 12 and 15 |
| rank-3 TMA GMEM-to-SMEM | 14 | warp 14 |
| rank-3 TMA SMEM-to-GMEM | 4 | warp 15 O store |
| rank-4 TMA SMEM-to-GMEM | 4 in checkpoint specialization | warp 15 checkpoint store |
| `tcgen05.alloc/relinquish/dealloc` | one each | warp 13 lifecycle |
| `ldmatrix...x4` non-transposed/transposed | 32 / 8 | warp 12/15 and CG1 |
| `stmatrix...x4` transposed/non-transposed | 4 / 2 | CG1 O/Y and register roles |
| `movmatrix` | 16 | three inverse rounds |

## Validation contract

`get_kernel` returns `[descriptor_order_prologue, persistent_gdn2]` in launch
order.  The host adapter gives TIRx and standalone source the same immutable
logical inputs but independent work tables, scheduler counters, descriptor
workspaces, O/final-state/checkpoint buffers.  Correctness compares both to the
FP64 recurrence, with relative RMS limits 0.02 for BF16 and 0.01 for FP16,
covering every frozen capability profile and the explicit BT/head/state/order
boundaries in `CONFIGS`.

Benchmark timing includes both launches for each implementation.  Work-table,
TensorMap, reference compilation, and data preparation are outside the timed
closure.  `prepare_bench` is CPU-only.  The benchmark suite is the only
performance authority; PTX/SASS/NCU can explain a candidate but cannot select
or pass it.  Every final raw row must contain five finite positive source and
TIRx samples from one complete run and satisfy strict
`mean(cudnn_frontend) / mean(tirx) > 0.99`.
