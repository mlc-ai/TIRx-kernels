<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This design sketch documents a modified TIRx port of cuDNN Frontend's
python/cudnn/linear_attention/frost/kernel/gdn_prefill_f16.py at commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5.
-->

# cuDNN SM100 GDN (v1) FP16/BF16 prefill: coarse WASP pipeline sketch

This is a non-executable execution sketch, not Python, a builder API, a new IR,
or a mathematical reference.  It freezes the source program before device-code
transcription.  The implementation it describes belongs in
[`tirx_kernels/cudnn/linear_attention/gdn_prefill_f16.py`](../../../tirx_kernels/cudnn/linear_attention/gdn_prefill_f16.py).

The frozen source is the standalone `chunk_gdn_sm100` two-launch entry at the
commit above.  The main anchor is BF16 I/O, FP32 state, final-state enabled,
`BT=64`, `DK=DV=128`, one CTA per cluster, 384 main threads (12 warps), 1024
prologue threads, four interleaved K+Q stages, two V stages, three T-inverse
stages, three A stages, one O stage, three gate/cumprod/beta stages, two
scheduler stages, 512 TMEM columns, and 232448 bytes of main dynamic shared
memory.  Checkpoint specializations trim K+Q to three stages and append one
32768-byte checkpoint staging buffer at the same total.

The accepted capability set is the set the source entry dispatches: BF16 and
FP16 I/O; FP32 or BF16 state; absent/present initial state; absent/present
final state; checkpoint cadence 64, 128, or 192 (plus zero for disabled) — a
positive multiple of the 64-token chunk; grouped Q/K/V heads under
`HO = max(HQ, HV)` with `k_heads` equal to `q_heads` or `v_heads`;
ragged/tail/zero-length sequences; gate interpretations raw-alpha,
natural-log (`log_gate`, the engine's production path), and safe gate with
per-head `a_log`/`dt_bias`; beta post-sigmoid FP32 or IO-dtype logits with
in-kernel sigmoid (`use_beta_sigmoid`); static grid-stride or dynamic ticket
scheduler; generated uncut or caller-staged/sorted work items; i32 or i64
cumulative sequence lengths; and overlap-recompute split work rows.  Q/K L2
normalization is NOT part of this kernel: the source engine runs it as a
separate pre-kernel and the production gdn benchmark runs unfused.

Writer line-info evidence is in `.porting/gdn_prefill_f16/source_export/`:
`dump_nostate/` holds the BF16 basic anchor (`cutlass_host_GdnCfg...sm_100a.ptx`,
2652 `.loc` sites, and `cutlass_prologue_...sm_100a.ptx`), `dump_state/` the
checkpoint + initial-state specialization.  PTX line evidence below refers to
the nostate anchor unless a branch profile is named.

## Pipeline at a glance

| Warps | Registers | Source-order role | Principal publications |
| --- | ---: | --- | --- |
| 0..3 | 224 | CG0 chunk-pair owner: per-pair T-pairwise decay fragments, KK epilogue into the T-inverse buffer, four-level hierarchical pair inverse (8→16→32→64), post-inverse beta column scaling, causal A epilogue | t_inv-ready, a-ready, cg0-acc-done, gate/beta-done |
| 4..7 | 256 (232 without initial state) | CG1 state owner: initial-state TMEM seed, state repack FP32→IO, checkpoint snapshot, state rescale by chunk cumprod, per-row gate registers, `Y = V − cumprod·(K·S)`, Q·S scale, U repack + decayed-U, O TMEM→SMEM, final-state TMEM→GMEM | state-inp/y-inp/u-inp/decay-u-inp, state-scale-done, o-ready, checkpoint-ready, tmem-done |
| 8 | 24 (48 w/o initial state) | gate/beta producer: predicated gate loads, log2 transform (raw/log/safe), warp prefix scan → cumsum-log and cumprod SMEM rings, beta cp.async or in-register sigmoid | gate-ready, beta-ready |
| 9 | 24 (48) | TMA-LDG: per-batch descriptor acquire, interleaved K+Q tile and one-behind V tile TensorMap loads, dynamic-ticket producer | kq-ready (expect-tx), v-ready (expect-tx), sched-ready |
| 10 | 24 (48) | sole tcgen05 issuer: TMEM alloc/dealloc, fused KK/QK pairs with member-parity lookahead, K·S, Q·S, U, O, state-update chains | cg0/state/o/k-state/u-acc-ready, t_inv/a/kq-done |
| 11 | 24 (48) | epilogue: one-behind O TensorMap stores and state-checkpoint TensorMap stores | o-done, checkpoint-done |

Dispatch order is exactly source order: warps 0..3, warps 4..7, warp 8,
warp 10, warp 9, then warp 11.  Every role owns independent pipeline cursors
that start once per kernel, advance per acquired stage, and are not reset at
work-item boundaries.  Chunks execute in PAIRS on CG0: warps 0..1 invert the
pair's member-0 matrix while warps 2..3 invert member 1; an odd tail pair runs
member 0 only with warps 2..3 idling through the per-warp inverse steps while
still arriving every group barrier.

## Primitive vocabulary and translation boundary

All storage is one-dimensional.  Names that look like matrices below are only
logical comments on integer offsets and strides; they are never layout values,
views, tiles, fragments, or first-class descriptors.

```python
linear_buffer(space, dtype, elements, byte_offset, alignment, lifetime)
reg_array(dtype, elements)
smem_byte(arena, base, stage, row, col,
          stage_bytes, row_bytes, elem_bytes, xor128)  # col ^ ((row&7)*8) elems
tmem_cell(base, row, column)       # base + column + (row << 16)
gmem_element(base, scalar_index)
descriptor_slot(workspace, array, batch)  # (array*B+batch)*128 bytes
smem_matrix_desc(base_byte, lead_bytes, stride_bytes, swizzle128)  # tcgen05 B/A operand
```

Directional movements are only `copy_p2g`, `copy_g2s`, `copy_s2g`,
`copy_g2r`, `copy_r2g`, `copy_s2r`, `copy_r2s`, `copy_t2r`, and `copy_r2t`.
Computations are only `fill`, `cast`, `add`, `sub`, `mul`, `fma`, `min`,
`max`, `select`, `exp2`, `log2`, `tanh`, `rcp`, `shuffle`, and `gemm`.
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
    IO_DTYPE={"bf16", "f16"}, STATE_DTYPE={"f32", "bf16"}, ACC_DTYPE="f32",
    BT=64, DK=128, DV=128,
    KQ_STAGES=(3 if CHECKPOINTS else 4), V_STAGES=2,
    TINV_STAGES=3, A_STAGES=3, O_STAGES=1,
    GATE_STAGES=3, BETA_STAGES=3, CHECKPOINT_STAGES=1,
    STATE_ACC_STAGES=1, QSTATE_ACC_STAGES=1, STATE_INP_STAGES=1,
    CG0_ACC_STAGES=2, CG1_ACC_STAGES=1, SCHED_STAGES=2,
    CLUSTER=(1,1,1),
    REGS=(224, 256 if USE_INITIAL_STATE else 232,
          24 if USE_INITIAL_STATE else 48),
)
def gdn_prefill_factory(...):
    return descriptor_order_prologue, persistent_gdn

# ========================================================================
# Launch 1: source-order work table and five TensorMap descriptor arrays
# ========================================================================

@kernel(grid=(1,1,1), block=(1024,1,1), warps=32, target="sm_100a")
def descriptor_order_prologue(
    base_q, base_k, base_v, base_o, base_checkpoint,
    descriptor_workspace, cu_seqlens,
    q, k, v, o, checkpoints_or_dummy,
    staging_items_or_dummy, work_count, work_items,
    sched_counter_or_dummy, batch_count,
    q_row_stride, k_row_stride, v_row_stride,
    o_row_stride, checkpoint_row_stride, checkpoint_every_n_tokens,
):
    tid = thread_id()
    warp = tid // 32

    # One rank-1 u8 arena: two 4096-i32 order regions plus a two-i32 spread
    # region; descriptor building needs no SMEM payload.
    arena = linear_buffer("smem", "u8", 32776, 0, 16, "whole launch")
    order_key = integer_region(arena, 0, 4096, "i32")
    order_idx = integer_region(arena, 16384, 4096, "i32")
    order_spread = integer_region(arena, 32768, 2, "i32")

    if tid == 0 and HAS_DYNAMIC_SCHEDULER:
        copy_r2g(i32(0), sched_counter[0])
        # instruction_selection: st.global.u32; extent: one scalar

    # Source LPT order pass (common/split_k.py order_body): generate uncut
    # rows (batch*HO, chunk-span key from cu_seqlens) or copy caller-staged
    # rows; above 4096 items copy through unsorted; otherwise spread-detect
    # via shared atomics and stable bitonic sort descending by span, then
    # write every eight-i32 row.
    n = batch_count * HO if GENERATE_UNCUT else copy_g2r(work_count[0])
    if GENERATE_UNCUT and tid == 0:
        copy_r2g(n, work_count[0])
    if n > 4096:
        for item in range(tid, n, 1024):
            row = generate_or_load_eight_fields(item)
            copy_r2g(row, work_items[item,0:8])
            # instruction_selection: generated rows use two st.global.v4.b32;
            # scratch rows use eight ld.global.b32 + eight st.global.b32
    else:
        if tid == 0:
            copy_r2s(i32_max, order_spread[0]); copy_r2s(i32_min, order_spread[1])
        padded_count = next_power_of_two(n)
        barrier(0,1024)
        local_min, local_max = i32_max, i32_min
        for element in range(4):
            item = tid + element*1024
            if item < padded_count:
                key = i32_min
                if item < n:
                    row = generate_or_load_eight_fields(item)
                    key = row_chunk_span(row)
                    local_min = min(local_min,key); local_max = max(local_max,key)
                copy_r2s(key, order_key[item]); copy_r2s(item, order_idx[item])
        shared_atomic_min(order_spread[0], local_min)
        shared_atomic_max(order_spread[1], local_max)
        # instruction_selection: atom.shared::cta.min/max.s32; extent: one
        # pair per thread after four local items
        barrier(0,1024)
        stable_lpt_bitonic_sort(order_key, order_idx, n, order_spread)
        barrier(0,1024)
        for rank in range(tid, n, 1024):
            source = copy_s2r(order_idx[rank])
            row = generate_or_load_eight_fields(source)
            copy_r2g(row, work_items[rank,0:8])
            # instruction_selection: two st.global.v4.b32 generated rows /
            # eight scalar load-store scratch rows

    # Exactly one descriptor array belongs to each warp 0..4.  Its elected
    # lane walks all batches serially.  Slots are Q,K,V,O,Checkpoint in that
    # exact warp/array order; warp 4 is compile-time inactive without
    # checkpoints.  Q/K/V/O base maps carry (channel, head, token) with the
    # token axis at TMA ordinal 2; the checkpoint map carries
    # (dv, dk, checkpoint, head) with the checkpoint count at ordinal 2.
    base_maps = (base_q, base_k, base_v, base_o, base_checkpoint)
    checkpoint_prefix = 0
    if warp < 5 and elected_lane() and (warp < 4 or CHECKPOINTS):
        array = warp
        for batch in range(batch_count):
            bos = copy_g2r(cu_seqlens[batch])
            eos = copy_g2r(cu_seqlens[batch+1])
            length = eos - bos
            dst = descriptor_slot(descriptor_workspace, array, batch)
            copy_p2g(base_maps[array], dst)
            # instruction_selection: sixteen b64 copies of the 128-byte base
            # descriptor per slot
            if array == 4:
                count = 0 if length == 0 else (length-1)//checkpoint_every_n_tokens + 1
                replace_global_address(dst,
                    checkpoints_base + checkpoint_prefix*checkpoint_row_stride)
                replace_global_dim(dst, ordinal=2, count)
                checkpoint_prefix += count
            else:
                replace_global_address(dst, tensor_base(array) + bos*row_stride(array))
                replace_global_dim(dst, ordinal=2, length)
            # instruction_selection: tensormap.replace.tile.global_address /
            # .global_dim per slot
        fence("tensormap_generic_release_gpu")
        # instruction_selection: fence.proxy.tensormap::generic.release.gpu

# ========================================================================
# Launch 2 ABI, scalar maps, arena and protocol
# ========================================================================

@kernel(grid=(num_sms,1,1), block=(384,1,1), warps=12,
        cluster=(1,1,1), min_blocks_per_sm=1, target="sm_100a")
def persistent_gdn(
    descriptor_workspace, n_desc,
    q, k, v, gate, a_log_or_dummy, dt_bias_or_dummy, beta,
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
        # fields: batch, head, wstart, wend, cstart, cend, bos, eos;
        # num_chunks_b = ceil_div(eos-bos, 64).  An item computes chunks
        # [cstart, wend) and writes O/checkpoints only for [wstart, wend).
        row = copy_g2r(work_items[tile,0:8])
        return row_with_derived_fields(row)
        # instruction_selection: role-projected work-row decode, normally
        # ld.global.v4.b32 plus required scalar loads; it is intentionally
        # not asserted as two full-vector loads

    def input_head(head, ratio):
        return head if ratio == 1 else head // ratio

    def descriptor(array, batch):
        return descriptor_slot(descriptor_workspace, array, batch)

    # One rank-1 u8 declaration, 232448 bytes, 1024-byte aligned.  The first
    # 3072 bytes are the protocol header plus the three FP32 gate rings; the
    # payloads follow at 1024-aligned bases.  Checkpoint mode drops one K+Q
    # stage and inserts the checkpoint staging buffer after O at the same
    # total footprint.
    arena = linear_buffer("smem", "u8", 232448, 0, 1024, "whole launch")

    BARRIER_BASE = 0          # 60 x 8-byte mbarriers, declaration-ordered
    TMEM_MAILBOX = 480        # tcgen05.alloc result, one i32
    SCHED_BASE = 496          # dynamic-ticket ring, SCHED_STAGES x i32
    CUMSUMLOG_BASE = 512      # f32 [64 x 3 stages]
    CUMPROD_BASE = 1280       # f32 [64 x 3 stages]
    BETA_BASE = 2048          # f32 [64 x 3 stages]
    O_BASE = 3072             # IO [64 x 128], 1 stage, 16384 B
    CHECKPOINT_BASE = 19456   # checkpoint mode only: IO [128 x 128], 32768 B
    KQ_BASE = 19456 + (32768 if CHECKPOINTS else 0)
                              # IO [2 x 64 x 128] x KQ_STAGES (32768 B/stage)
    TINV_BASE = KQ_BASE + KQ_STAGES*32768        # IO [64 x 64] x 3, 8192 B/stage
    A_BASE = TINV_BASE + 3*8192                  # IO [64 x 64] x 3
    V_BASE = A_BASE + 3*8192                     # IO [64 x 128] x 2, 16384 B/stage
    # nostate: KQ 19456, TINV 150528, A 175104, V 199680, end 232448.

    # Every IO payload tile is stored 128-byte XOR-swizzled:
    # phys_col_elems = col ^ ((row & 7) * 8) within each 64-element row
    # segment; tcgen05 operand descriptors carry lead 16 bytes, stride 1024
    # bytes, swizzle-128B.  The K^T view of the same K+Q bytes uses lead
    # 16384 bytes for the state-update GEMM.

    # Protocol table, declaration-ordered physical mbarriers.  Each tuple is
    # (name, ready, done-or-None, ready stages, done stages,
    #  ready arrivals, done arrivals).
    PROTOCOL = (
      ("kq",           0,  32, KQ_STAGES,KQ_STAGES, 1,1),     # TMA expect-tx / MMA commit
      ("v",           64,  80, 2,2,   1,128),                 # TMA expect-tx / CG1 threads
      ("gate",        96, 120, 3,3,  32,256),                 # warp 8 / CG0+CG1
      ("beta",       144, 168, 3,3,  32,128),                 # warp 8 (cp.async) / CG0
      ("state_acc",  192, 200, 1,1,   1,128),                 # MMA commit / CG1 scale-done
      ("o_acc",      208,None, 1,0,   1,None),                # MMA commit (Q*state)
      ("o_final_acc",216,None, 1,0,   1,None),                # MMA commit (O)
      ("o_scale_done",224,None,1,0, 128,None),                # CG1 Q*state scale done
      ("cg0_acc",    232, 248, 2,2,   1,64),                  # MMA commit / half-CG0
      ("state_inp",  264,None, 1,0, 128,None),                # CG1 packed state
      ("y_inp",      272,None, 1,0, 128,None),                # CG1 packed Y
      ("u_inp",      280,None, 1,0, 128,None),                # CG1 packed U
      ("decay_u_inp",288,None, 1,0, 128,None),                # CG1 packed decayed U
      ("t_inv",      296, 320, 3,3, 128,1),                   # CG0 / MMA commit
      ("a",          344, 368, 3,3,  64,1),                   # half-CG0 / MMA commit
      ("k_state_acc",392,None, 1,0,   1,None),                # MMA commit
      ("u_acc",      400,None, 1,0,   1,None),                # MMA commit
      ("o_stage",    408, 416, 1,1, 128,32),                  # CG1 / epilogue warp
      ("checkpoint", 424, 432, 1,1,   4,32),                  # CG1 warp-elects / epilogue
      ("tmem_done",  440,None, 1,0, 128,None),                # CG1 -> MMA dealloc
      ("sched",      448, 464, 2,2,   1,11),                  # warp 9 / 11 consumers
    )

    pool = smem_pool(base=arena)   # allocates only the header prefix
    for edge in PROTOCOL:
        allocate_edge_from_pool(edge)

    # All 60 physical barriers are initialized before the init fence; the
    # port distributes the init sites across the producer-owning warps.
    init_protocol_edges(PROTOCOL)
    # instruction_selection: mbarrier.init.shared.b64; extent: 60 sites
    fence("mbarrier_init_release_cluster")
    barrier(0,384)
    # instruction_selection: fence.mbarrier_init.release.cluster; bar.sync 0

    # TMEM columns (512 total).  Packed IO regions hold two elements per
    # 32-bit cell; accumulators are FP32.  All row bases are warp-relative
    # tmem_cell(base, warp_row, col) = base + col + (warp_row << 16).
    STATE_ACC = 0        # S^T accumulator, 128 rows x cols 0..127
    QSTATE_ACC = 128     # (Q*S)^T then O^T accumulator, cols 128..191
    STATE_INP = 192      # packed IO S^T staging (GEMM 3/4 A), cols 192..255
    CG0_ACC = 256        # fused KK/QK ring, 2 stages x 64, cols 256..383
    CG1_ACC = 384        # (K*S)^T then U^T accumulator, cols 384..447
    YU_INP = 448         # packed Y then U (GEMM 5/6 A), cols 448..479
    DECAY_U_INP = 480    # packed decayed U (GEMM 7 A), cols 480..511

    def wait_edge(name, stage, phase):
        wait(protocol_ptr(name, stage), phase)
        # instruction_selection:
        # mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 in a retry loop

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

    def scheduler_publish(cursor, tile):      # warp 9 side
        if DYNAMIC:
            wait_edge("sched_done", cursor.stage, cursor.phase)
            if elected_lane():
                ticket = atomic_add(sched_counter[0], 1, scope="gpu", order="relaxed")
                copy_r2s(grid_x + ticket, arena[SCHED_BASE + 4*cursor.stage])
                # instruction_selection: atom.global.add.s32 (relaxed, gpu)
            warp_sync()
            next_tile = copy_s2r(arena[SCHED_BASE + 4*cursor.stage])
            if elected_lane(): arrive_edge("sched", cursor.stage)
            return next_tile, advance(cursor)
        return tile + grid_x, cursor

    def scheduler_consume(cursor, tile):      # all consumer roles
        if DYNAMIC:
            wait_edge("sched", cursor.stage, cursor.phase)
            next_tile = copy_s2r(arena[SCHED_BASE + 4*cursor.stage])
            if elected_lane(): arrive_edge("sched_done", cursor.stage)
            return next_tile, advance(cursor)
        return tile + grid_x, cursor

    # ====================================================================
    # Warps 0..3 (CG0): T-pairwise, KK epilogue, pair inverse, A epilogue
    # ====================================================================
    if 0 <= warp <= 3:
        set_register_budget("increase", 224)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32
        barrier(1, 288)      # TMEM-alloc rendezvous with warps 4..7 and 10
        tmem = copy_s2r(arena[TMEM_MAILBOX])
        gate_cur = consumer_cursor(GATE_STAGES, phase=0)
        beta_cur = consumer_cursor(BETA_STAGES, phase=0)
        cg0_cur = consumer_cursor(CG0_ACC_STAGES, phase=0)
        tinv_cur = producer_cursor(TINV_STAGES, phase=1)
        a_cur = producer_cursor(A_STAGES, phase=1)
        sched = consumer_cursor(SCHED_STAGES, phase=0)
        pair_half = warp // 2          # warps 0..1 = member 0, 2..3 = member 1
        inv_warp = warp % 2
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            n_local = item.wend - item.cstart
            for pair in range(ceil_div(n_local, 2)):
                # An odd chunk count leaves the last pair with member 0 only;
                # have_m1 is uniform across CG0 so shared barriers stay aligned.
                have_m1 = pair*2 + 1 < n_local
                do_kk = have_m1 or pair_half == 0
                do_a  = have_m1 or pair_half == 1

                # -- T-pairwise decay fragments for both members ----------
                # Acquire gate stage 0 (and stage 1 when the pair has two
                # members); this warp's KK member is stage pair_half, its A
                # member the opposite one.
                g0 = acquire(gate_cur); g1 = acquire(gate_cur) if have_m1 else g0
                wait_edge("gate", g0.stage, g0.phase)
                if have_m1: wait_edge("gate", g1.stage, g1.phase)
                kk_g, a_g = (g1, g0) if pair_half == 1 else (g0, g1)
                # Per accumulator fragment cell (i,j):
                #   decay[i,j] = exp2(cumsumlog[i] - cumsumlog[j]) if i>=j else 0
                # built from 4 row reads + 16 column reads of the cumsum ring
                # for each member; 64 FP32 values per member per thread pair
                # of chunk = 128 registers.
                decay_kk = [exp2(sub(cumsumlog_row(kk_g,i), cumsumlog_col(kk_g,j)))
                            if lower(i,j) else opaque_zero() for (i,j) in acc_frag()]
                decay_a  = [exp2(sub(cumsumlog_row(a_g,i), cumsumlog_col(a_g,j)))
                            if lower(i,j) else opaque_zero() for (i,j) in acc_frag()]
                # instruction_selection: ld.shared.b32 pairs, sub.f32,
                # ex2.approx.ftz.f32, selp.f32 against an opaque zero;
                # extent: 128 fused cells per thread
                arrive_edge("gate_done", g0); arrive_edge("gate_done", g1) if have_m1

                b0 = acquire(beta_cur); b1 = acquire(beta_cur) if have_m1 else b0
                wait_edge("beta", b0.stage, b0.phase)
                if have_m1: wait_edge("beta", b1.stage, b1.phase)
                kk_b = (b1 if pair_half == 1 else b0)
                kk_beta_rows = copy_s2r(beta_ring_rows(kk_b))

                # -- KK epilogue into the T-inverse buffer ----------------
                kk_acc = acquire(cg0_cur); a_acc = acquire(cg0_cur) if have_m1 else kk_acc
                if pair_half == 1: kk_acc, a_acc = a_acc, kk_acc
                t0 = acquire(tinv_cur); t1 = acquire(tinv_cur) if have_m1 else t0
                kk_t = (t1 if pair_half == 1 else t0)
                if do_kk:
                    wait_edge("cg0_acc", kk_acc.stage, kk_acc.phase)
                    kk = copy_t2r(tmem_cell(tmem, warp*32, CG0_ACC + kk_acc.stage*64))
                    # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32;
                    # extent: two 16-row halves per warp
                    wait_edge("t_inv_done", kk_t.stage, kk_t.phase)
                    m = cast(IO_DTYPE, mul(mul(kk, decay_kk), kk_beta_rows))
                    # instruction_selection: packed mul.f32 pairs +
                    # cvt.rn.{bf16x2|f16x2}.f32
                    copy_r2s(m, smem_byte(arena, TINV_BASE, kk_t.stage,
                                          row=member_row(warp,lane),
                                          col=xor128, elem_bytes=2))
                    # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16;
                    # extent: 4 fragments x 2 halves

                # -- four-level hierarchical pair inverse -----------------
                # Warps 0..1 own matrix 0, warps 2..3 matrix 1; without
                # member 1 all four warps take matrix 0 and the duplicate
                # band is identical.  Diagonal cells become I via lane
                # identity injection; result overwrites the same buffer.
                barrier(2, 128)
                if have_m1 or warp < 2:
                    gauss_jordan_8x8(arena, my_tinv_base, d=(inv_warp*32+lane)//8)
                    # instruction_selection: 1 x ld.shared.v4.b32 8-element
                    # IO-pair row load, shfl.sync.idx.b32 with clustered
                    # member mask, rcp-free scaled row elimination in f32,
                    # 1 x st.shared.v4.b32 row store
                barrier(2, 128)
                blockwise_8_to_16(arena, t0.base, d0=warp*16, lane)
                if have_m1: blockwise_8_to_16(arena, t1.base, d0=warp*16, lane)
                # instruction_selection per call: 3 x ldmatrix.sync.aligned
                # .m8n8.x1{,.trans}.shared.b16, 2 x mma.sync.aligned
                # .m16n8k8.row.col.f32.{bf16|f16}.f32, neg.f32, packed cvt,
                # 1 x stmatrix.m8n8.x1
                barrier(2, 128)
                if have_m1 or warp < 2:
                    blockwise_16_to_32(arena, my_tinv_base, d0=inv_warp*32, lane)
                    # instruction_selection: 3 x ldmatrix.x4, 4 x mma.sync
                    # .m16n8k16, packed cvt, 1 x stmatrix.x4
                barrier(2, 128)
                blockwise_32_to_64(arena, my_tail_base, band=inv_warp, lane)
                # instruction_selection: 10 x ldmatrix.x4, 16 x mma.sync
                # .m16n8k16, packed cvt, barrier(2,128), 2 x stmatrix.x4
                barrier(2, 128)

                # -- post-inverse beta column scaling + publish -----------
                # T_inv[:, j] *= beta[j] for member 0 then member 1:
                # ldmatrix the finished inverse, multiply by the beta ring
                # columns, store back, fence, publish.
                for (t, b) in ((t0, b0),) + (((t1, b1),) if have_m1 else ()):
                    frag = copy_s2r(smem_byte(arena, TINV_BASE, t.stage, xor128))
                    # instruction_selection: ldmatrix.m8n8.x4.shared.b16 x4
                    scaled = cast(IO_DTYPE, mul(cast("f32",frag), beta_cols(b)))
                    copy_r2s(scaled, smem_byte(arena, TINV_BASE, t.stage, xor128))
                    # instruction_selection: stmatrix.m8n8.x4 x4
                    fence("async_shared_cta")
                    arrive_edge("t_inv", t.stage)
                    arrive_edge("beta_done", b)

                # -- causal A epilogue (opposite member) ------------------
                a0 = acquire(a_cur); a1 = acquire(a_cur) if have_m1 else a0
                my_a = (a0 if pair_half == 1 else a1)
                if do_a:
                    av = copy_t2r(tmem_cell(tmem, warp*32, CG0_ACC + a_acc.stage*64))
                    # instruction_selection: tcgen05.ld.16x256b.x8 x2 halves
                    wait_tmem_load()
                    # instruction_selection: tcgen05.wait::ld.sync.aligned
                    arrive_edge("cg0_acc_done", a_acc.stage)
                    wait_edge("a_done", my_a.stage, my_a.phase)
                    ascaled = cast(IO_DTYPE, mul(mul(av, decay_a), scale))
                    copy_r2s(ascaled, smem_byte(arena, A_BASE, my_a.stage, xor128))
                    # instruction_selection: packed mul.f32 + cvt +
                    # stmatrix.m8n8.x4 x4 x 2 halves
                    fence("async_shared_cta")
                    arrive_edge("a", my_a.stage)
            tile, sched = scheduler_consume(sched, tile)
        drain_producer("t_inv_done", tinv_cur); drain_producer("a_done", a_cur)

    # ====================================================================
    # Warps 4..7 (CG1): state seed/restage/rescale, Y, Q*S scale, U, O, final
    # ====================================================================
    if 4 <= warp <= 7:
        set_register_budget("increase", 256 if USE_INITIAL_STATE else 232)
        barrier(1, 288)
        tmem = copy_s2r(arena[TMEM_MAILBOX])
        v_cur = consumer_cursor(V_STAGES, phase=0)
        gate_cur = consumer_cursor(GATE_STAGES, phase=0)
        state_cur = consumer_cursor(1, phase=0)      # state_acc ready
        seed_cur = producer_cursor(1, phase=1)       # state-scale-done reuse
        oacc_cur = consumer_cursor(1, phase=0)
        ofin_cur = consumer_cursor(1, phase=0)
        kst_cur = consumer_cursor(1, phase=0)
        uacc_cur = consumer_cursor(1, phase=0)
        o_cur = producer_cursor(O_STAGES, phase=1)
        sched = consumer_cursor(SCHED_STAGES, phase=0)
        value_dim = (warp-4)*32 + lane               # 0..127 output row
        checkpoint_serial = 0
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            n_local = item.wend - item.cstart
            ckpt_mod = item.cstart % (checkpoint_every_n_tokens // 64) if CHECKPOINTS
            if n_local > 0:
                if USE_INITIAL_STATE:
                    # -- initial-state seed: GMEM (or zero) -> state TMEM ---
                    wait_edge("state_scale_done", 0, seed_cur.phase); advance(seed_cur)
                    if item.cstart == 0:
                        rows = copy_g2r(initial_state[item.batch, item.head,
                                                      value_dim, 0:128])
                        # instruction_selection: 128 scalar ld.global with
                        # cast for BF16 state
                    else:
                        rows = fill(reg_array("f32",128), 0)
                    copy_r2t(rows, tmem_cell(tmem, value_dim_row, STATE_ACC))
                    # instruction_selection: tcgen05.st.sync.aligned
                    # .32x32b.x32.b32 x4 subtiles; tcgen05.wait::st
                    barrier(4, 128)

                for local in range(n_local):
                    chunk = item.cstart + local
                    if CHECKPOINTS:
                        do_ckpt_now = ckpt_mod == 0
                        ckpt_mod = 0 if ckpt_mod+1 == ckpt_chunks else ckpt_mod+1
                    if CHECKPOINTS and not USE_INITIAL_STATE and chunk == 0 and item.wstart == 0:
                        # explicit zero checkpoint before the first chunk
                        stage = acquire_checkpoint_done(checkpoint_serial)
                        copy_r2s(fill(0), arena[CHECKPOINT_BASE + stage_bytes])
                        # instruction_selection: st.shared.b32 strided zero fill
                        fence("async_shared_cta")
                        if elected_lane(): arrive_edge("checkpoint", stage)
                        checkpoint_serial += 1
                    have_state = USE_INITIAL_STATE or local > 0
                    if USE_INITIAL_STATE:
                        # source line 1708: the seed cursor also advances once
                        # per chunk so it tracks every scale-done arrival
                        advance(seed_cur)

                    g = acquire(gate_cur)
                    wait_edge("gate", g.stage, g.phase)
                    cumprod_total = copy_s2r(arena[CUMPROD_BASE + g.stage*256 + 63*4])

                    # -- state restage + optional checkpoint + rescale ------
                    if have_state:
                        st = acquire(state_cur)
                        wait_edge("state_acc", st.stage, st.phase)
                        srows = copy_t2r(tmem_cell(tmem, value_dim_row, STATE_ACC))
                        # instruction_selection: tcgen05.ld.32x32b.x32 x4
                        packed = cast(IO_DTYPE, srows, packed_pairs)
                        copy_r2t(packed, tmem_cell(tmem, value_dim_row, STATE_INP))
                        # instruction_selection: cvt.rn.{bf16x2|f16x2}.f32 x64;
                        # tcgen05.st.32x32b.x16 x4; tcgen05.wait::st
                        arrive_edge("state_inp")
                        if CHECKPOINTS and do_ckpt_now and item.wstart <= chunk < item.wend:
                            stage = acquire_checkpoint_done(checkpoint_serial)
                            ck = copy_t2r(tmem_cell(tmem, value_dim_row, STATE_ACC))
                            copy_r2s(cast(IO_DTYPE, ck),
                                     arena[CHECKPOINT_BASE + xor128_row(value_dim)])
                            # instruction_selection: tcgen05.ld.32x32b.x32 x4;
                            # packed cvt; 16-byte st.shared vector stores
                            fence("async_shared_cta")
                            if elected_lane(): arrive_edge("checkpoint", stage)
                            checkpoint_serial += 1
                        scaled = mul(srows, cumprod_total)
                        copy_r2t(scaled, tmem_cell(tmem, value_dim_row, STATE_ACC))
                        # instruction_selection: packed mul.f32;
                        # tcgen05.st.32x32b.x32 x4; tcgen05.wait::st
                        arrive_edge("state_scale_done", st.stage)

                    # -- per-row gate registers -----------------------------
                    cumprod_cols = copy_s2r(cumprod_ring_cols(g.stage))
                    last_log = copy_s2r(arena[CUMSUMLOG_BASE + g.stage*256 + 63*4])
                    decay_scale = exp2(sub(last_log, cumsumlog_ring_cols(g.stage)))
                    # instruction_selection: ld.shared.b32, add.rn.f32x2 pair
                    # sums, ex2.approx.ftz.f32; extent: 32 columns
                    arrive_edge("gate_done", g.stage)

                    # -- Y = V - cumprod * (K*S), packed 16-bit -------------
                    vv = acquire(v_cur)
                    wait_edge("v", vv.stage, vv.phase)
                    vfrag = copy_s2r(smem_byte(arena, V_BASE, vv.stage,
                                     row=token_row, col=xor128, trans=True))
                    # instruction_selection: ldmatrix.m8n8.x4.trans.shared.b16
                    # x8 (two 64-row subtiles x four 16-token bands)
                    if have_state:
                        wait_edge("k_state_acc", 0, kst_cur.phase); advance(kst_cur)
                        ks = copy_t2r(tmem_cell(tmem, value_dim_row, CG1_ACC))
                        # instruction_selection: tcgen05.ld.16x256b.x8 x2
                        y = packed_sub(vfrag, cast(IO_DTYPE, mul(ks, cumprod_cols)))
                        # instruction_selection: packed mul.f32, cvt,
                        # sub.{bf16x2|f16x2}; extent: 32 packed cells
                    else:
                        y = vfrag
                    copy_r2t(y, tmem_cell(tmem, value_dim_row, YU_INP))
                    # instruction_selection: tcgen05.st.16x128b.x8 x2;
                    # tcgen05.wait::st
                    arrive_edge("y_inp")

                    # -- Q*S epilogue: acc *= cumprod * scale ---------------
                    if have_state:
                        qs = acquire(oacc_cur)
                        wait_edge("o_acc", 0, qs.phase)
                        qsv = copy_t2r(tmem_cell(tmem, value_dim_row, QSTATE_ACC))
                        # instruction_selection: tcgen05.ld.16x256b.x8 x2
                        qsv = mul(mul(qsv, cumprod_cols), scale)
                        copy_r2t(qsv, tmem_cell(tmem, value_dim_row, QSTATE_ACC))
                        # instruction_selection: packed mul.f32;
                        # tcgen05.st.16x256b.x8 x2; tcgen05.wait::st
                        arrive_edge("o_scale_done", 0)

                    # -- U epilogue + decayed-U publish ---------------------
                    wait_edge("u_acc", 0, uacc_cur.phase); advance(uacc_cur)
                    arrive_edge("v_done", vv.stage)
                    u = copy_t2r(tmem_cell(tmem, value_dim_row, CG1_ACC))
                    # instruction_selection: tcgen05.ld.16x256b.x8 x2
                    copy_r2t(cast(IO_DTYPE, u), tmem_cell(tmem, value_dim_row, YU_INP))
                    # instruction_selection: packed cvt; tcgen05.st.16x128b.x8
                    # x2; tcgen05.wait::st
                    arrive_edge("u_inp")
                    du = cast(IO_DTYPE, mul(u, decay_scale))
                    copy_r2t(du, tmem_cell(tmem, value_dim_row, DECAY_U_INP))
                    # instruction_selection: packed mul.f32, cvt;
                    # tcgen05.st.16x128b.x8 x2; tcgen05.wait::st
                    arrive_edge("decay_u_inp")

                    # -- O epilogue: acc TMEM -> sO SMEM --------------------
                    of = acquire(ofin_cur)
                    wait_edge("o_final_acc", 0, of.phase)
                    ov = copy_t2r(tmem_cell(tmem, value_dim_row, QSTATE_ACC))
                    # instruction_selection: tcgen05.ld.16x256b.x8 x2;
                    # tcgen05.wait::ld
                    oslot = acquire(o_cur)
                    wait_edge("o_done", oslot.stage, oslot.phase)
                    copy_r2s(cast(IO_DTYPE, ov),
                             smem_byte(arena, O_BASE, oslot.stage, xor128, trans=True))
                    # instruction_selection: packed cvt +
                    # stmatrix.m8n8.x4.trans.shared.b16 x8
                    fence("async_shared_cta")
                    arrive_edge("o_stage", oslot.stage)

                # -- final state: TMEM -> GMEM after the last chunk ---------
                st = acquire(state_cur)
                wait_edge("state_acc", st.stage, st.phase)
                if STORE_FINAL_STATE and item.wend == item.num_chunks_b:
                    rows = copy_t2r(tmem_cell(tmem, value_dim_row, STATE_ACC))
                    # instruction_selection: tcgen05.ld.32x32b.x32 x4
                    copy_r2g(cast(STATE_DTYPE, rows),
                             final_state[item.batch, item.head, value_dim, 0:128])
                    # instruction_selection: st.global.v2.b64 vector stores
                    # (FP32 state) / packed converts then stores (BF16 state)
                arrive_edge("state_scale_done", st.stage)
            elif STORE_FINAL_STATE and item.wend == item.num_chunks_b:
                # Empty sequence: never touches TMEM; passthrough initial
                # state when present, else state-dtype zero.
                for key in range(128):
                    val = (copy_g2r(initial_state[item.batch,item.head,value_dim,key])
                           if USE_INITIAL_STATE else cast(STATE_DTYPE,0))
                    copy_r2g(val, final_state[item.batch,item.head,value_dim,key])
                    # instruction_selection: st.global.v2.b64 vectorized
                    # zero stores in the FP32 nostate anchor (.loc 1 1977,
                    # 32 sites); the initial-state profile loads each value
                    # via ld.global (+ cvt for BF16 state) before the store
            tile, sched = scheduler_consume(sched, tile)
        arrive_edge("tmem_done")
        drain_producer("o_done", o_cur)
        if CHECKPOINTS: drain_checkpoint_done(checkpoint_serial)

    # ====================================================================
    # Warp 8: gate/beta producer
    # ====================================================================
    elif warp == 8:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32
        gate_cur = producer_cursor(GATE_STAGES, phase=1)
        beta_cur = producer_cursor(BETA_STAGES, phase=1)
        sched = consumer_cursor(SCHED_STAGES, phase=0)
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            if SAFE_GATE and item.wend > item.cstart:
                a_l2 = mul(neg(exp2(mul(copy_g2r(a_log[item.head]), LOG2_E))), LOG2_E)
                bias = copy_g2r(dt_bias[item.head])
                # instruction_selection: ld.global.b32 x2, ex2.approx.ftz.f32
            for chunk in range(item.cstart, item.wend):
                # -- gate: predicated GMEM loads, log2 transform, prefix ----
                # Each lane owns two token columns (lane, lane+32); the OOB
                # neutral is 0 in log domain, 1 in alpha domain.
                g = acquire(gate_cur)
                vals = [copy_g2r(gate[clamped_token(col), item.head])
                        if token_valid(col) else oob_neutral() for col in (0,1)]
                # instruction_selection: branch-predicated ld.global.b32 with
                # min.s32 index clamp and mov-immediate OOB neutral
                if SAFE_GATE:
                    vals = [mul(a_l2, softplus(add(vc, bias))) if valid else 0
                            for vc in vals]
                    # instruction_selection: ex2/lg2.approx.ftz chain per lane
                elif LOG_GATE:
                    vals = mul(vals, RCP_LN2)          # natural log -> log2
                else:
                    vals = log2(add(vals, 1e-10))
                    # instruction_selection: lg2.approx.ftz.f32
                prefix = warp_inclusive_scan(vals)
                # instruction_selection: shfl.sync.up.b32 offsets 1,2,4,8,16
                # per column plus one shfl.sync.idx.b32 lane-31 carry;
                # extent: 5+5+1 shuffles for the 64-token scan
                wait_edge("gate_done", g.stage, g.phase)
                copy_r2s(prefix, arena[CUMSUMLOG_BASE + g.stage*256 + col*4])
                copy_r2s(exp2(prefix), arena[CUMPROD_BASE + g.stage*256 + col*4])
                # instruction_selection: st.shared.b32 x2 per column,
                # ex2.approx.ftz.f32
                arrive_edge("gate", g.stage)

                # -- beta: cp.async per element or in-register sigmoid ------
                b = acquire(beta_cur)
                wait_edge("beta_done", b.stage, b.phase)
                if BETA_SIGMOID:
                    for col in (0,1):
                        bv = 0
                        if token_valid(col):
                            bv = cast("f32", cast(IO_DTYPE,
                                 add(mul(tanh(mul(copy_g2r(beta[token,item.head]),0.5)),0.5),0.5)))
                        copy_r2s(bv, arena[BETA_BASE + b.stage*256 + col*4])
                    # instruction_selection: ld.global.b16, tanh.approx.f32,
                    # round through IO dtype, st.shared.b32
                    arrive_edge("beta", b.stage)
                else:
                    for col in (0,1):
                        copy_g2s(beta[token, item.head],
                                 arena[BETA_BASE + b.stage*256 + col*4],
                                 bytes=4 if token_valid(col) else 0)
                    # instruction_selection: cp.async.ca.shared.global 4B with
                    # zero-fill cp-size predication
                    cp_async_arrive(protocol_ptr("beta", b.stage), noinc=True)
                    # instruction_selection: cp.async.mbarrier.arrive.noinc
            tile, sched = scheduler_consume(sched, tile)
        drain_producer("gate_done", gate_cur); drain_producer("beta_done", beta_cur)

    # ====================================================================
    # Warp 10: sole tcgen05 issuer and TMEM lifecycle
    # ====================================================================
    elif warp == 10:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        allocate_tmem(TMEM_MAILBOX, columns=512, cta_group=1)
        # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned
        # .shared::cta.b32
        barrier(1, 288)
        tmem = copy_s2r(arena[TMEM_MAILBOX])
        kq_cur = consumer_cursor(KQ_STAGES, phase=0)        # per-chunk operand
        kqf_cur = consumer_cursor(KQ_STAGES, phase=0)       # fused-pair lookahead
        cg0_cur = consumer_cursor(CG0_ACC_STAGES, phase=1)  # cg0 acc done
        tinv_cur = consumer_cursor(TINV_STAGES, phase=0)
        a_cur = consumer_cursor(A_STAGES, phase=0)
        sinp_cur = consumer_cursor(1, phase=0)
        y_cur = consumer_cursor(1, phase=0)
        u_cur = consumer_cursor(1, phase=0)
        du_cur = consumer_cursor(1, phase=0)
        oacc_cur = producer_cursor(1, phase=1)              # q_state acc
        oscale_cur = consumer_cursor(1, phase=0)
        state_cur = producer_cursor(1, phase=1)             # state acc
        sched = consumer_cursor(SCHED_STAGES, phase=0)

        # SMEM operand descriptors are integer immediates: base>>4 plus
        # lead 16 B, stride 1024 B, swizzle-128B; the fused pair uses
        # half-tile offsets KQ_BOX = 64*64*2 B and half-K segment
        # KQ_SEG = 128*64*2 B >> 4; the state-update B operand re-views the
        # same K+Q bytes with lead 16384 B.
        def fused_kk_qk(kqf, acc_stage, same_operand):
            wait_edge("cg0_acc_done", acc_stage.stage, acc_stage.phase)
            wait_edge("kq", kqf.stage, kqf.phase)
            a = smem_matrix_desc(KQ_BASE + kqf.stage*32768)
            b = a if same_operand else a + KQ_BOX
            gemm(tmem_cell(tmem, 0, CG0_ACC + acc_stage.stage*64), a, b,
                 M=128, N=64, K=64, accumulate=False)
            gemm(tmem_cell(tmem, 0, CG0_ACC + acc_stage.stage*64),
                 a + KQ_SEG, b + KQ_SEG, M=128, N=64, K=64, accumulate=True)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 with
            # SMEM A/B descriptors; extent: 2 x 4 K-phases = 8 instructions
            if elected_lane(): commit_edge("cg0_acc", acc_stage.stage)

        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            n_local = item.wend - item.cstart
            # fused pair 0 and (when present) pair 1 are issued ahead
            if n_local > 0: fused_kk_qk(acquire(kqf_cur), acquire(cg0_cur), same=True)
            if n_local > 1: fused_kk_qk(acquire(kqf_cur), acquire(cg0_cur), same=False)
            for local in range(n_local):
                member = local & 1
                have_state = USE_INITIAL_STATE or local > 0
                if USE_INITIAL_STATE and local == 0:
                    if elected_lane(): commit_edge("state_acc", 0)
                    advance(state_cur)
                kq = acquire(kq_cur)
                desc_k = smem_matrix_desc(KQ_BASE + kq.stage*32768) + member*KQ_BOX
                desc_q = smem_matrix_desc(KQ_BASE + kq.stage*32768) + (KQ_BOX - member*KQ_BOX)
                desc_kt = smem_matrix_desc(KQ_BASE + kq.stage*32768,
                                           lead=16384) + member*KQ_BOX

                # member-parity lookahead keeps one fused pair in flight
                if member == 1 and local+2 < n_local:
                    fused_kk_qk(acquire(kqf_cur), acquire(cg0_cur), same=False)

                # -- GEMM 3: (K*S)^T = S^T_packed @ K^T ---------------------
                if have_state:
                    wait_edge("state_inp", 0, sinp_cur.phase); advance(sinp_cur)
                    gemm(tmem_cell(tmem, 0, CG1_ACC),
                         tmem_a(STATE_INP), desc_k, M=128, N=64, K=128,
                         accumulate=per_k_phase)
                    # instruction_selection: tcgen05.mma.kind::f16 TMEM-A;
                    # extent: 2 half-K segments x 4 = 8 instructions
                    if elected_lane(): commit_edge("k_state_acc", 0)

                # -- GEMM 4: (Q*S)^T = S^T_packed @ Q^T ---------------------
                qacc = acquire(oacc_cur)
                if have_state:
                    gemm(tmem_cell(tmem, 0, QSTATE_ACC),
                         tmem_a(STATE_INP), desc_q, M=128, N=64, K=128,
                         accumulate=per_k_phase)
                    # instruction_selection: 8 x tcgen05.mma TMEM-A
                    if elected_lane(): commit_edge("o_acc", 0)

                # -- GEMM 5: U^T = Y^T_packed @ T_inv^T ---------------------
                tv = acquire(tinv_cur)
                wait_edge("t_inv", tv.stage, tv.phase)
                wait_edge("y_inp", 0, y_cur.phase); advance(y_cur)
                gemm(tmem_cell(tmem, 0, CG1_ACC),
                     tmem_a(YU_INP), smem_matrix_desc(TINV_BASE + tv.stage*8192),
                     M=128, N=64, K=64, accumulate=per_k_phase_first_false)
                # instruction_selection: 4 x tcgen05.mma TMEM-A
                if elected_lane():
                    commit_edge("u_acc", 0); commit_edge("t_inv_done", tv.stage)

                if member == 0 and local+2 < n_local:
                    fused_kk_qk(acquire(kqf_cur), acquire(cg0_cur), same=True)

                # -- GEMM 6: O^T += U^T_packed @ A^T ------------------------
                av = acquire(a_cur)
                wait_edge("a", av.stage, av.phase)
                if have_state:
                    wait_edge("o_scale_done", 0, oscale_cur.phase); advance(oscale_cur)
                wait_edge("u_inp", 0, u_cur.phase); advance(u_cur)
                gemm(tmem_cell(tmem, 0, QSTATE_ACC),
                     tmem_a(YU_INP), smem_matrix_desc(A_BASE + av.stage*8192),
                     M=128, N=64, K=64, accumulate=have_state_on_first_k)
                # instruction_selection: 4 x tcgen05.mma TMEM-A
                if elected_lane():
                    commit_edge("a_done", av.stage); commit_edge("o_final_acc", 0)

                # -- GEMM 7: S^T += decayedU^T_packed @ K -------------------
                wait_edge("decay_u_inp", 0, du_cur.phase); advance(du_cur)
                advance(state_cur)
                gemm(tmem_cell(tmem, 0, STATE_ACC),
                     tmem_a(DECAY_U_INP), desc_kt, M=128, N=128, K=64,
                     accumulate=have_state_on_first_k)
                # instruction_selection: 4 x tcgen05.mma TMEM-A, B transposed
                # through the lead-16384 K^T descriptor view
                if elected_lane():
                    commit_edge("state_acc", 0); commit_edge("kq_done", kq.stage)
            tile, sched = scheduler_consume(sched, tile)
        wait_edge("tmem_done", 0, 0)
        relinquish_tmem_alloc_permit(cta_group=1)
        deallocate_tmem(tmem, columns=512, cta_group=1)
        # instruction_selection: tcgen05.relinquish_alloc_permit /
        # tcgen05.dealloc.cta_group::1.sync.aligned.b32

    # ====================================================================
    # Warp 9: TMA-LDG and scheduler producer
    # ====================================================================
    elif warp == 9:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        kq_cur = producer_cursor(KQ_STAGES, phase=1)
        v_cur = producer_cursor(V_STAGES, phase=1)
        sched = producer_cursor(SCHED_STAGES, phase=1)
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            qh = input_head(item.head, Q_RATIO)
            kh = input_head(item.head, K_RATIO)
            vh = input_head(item.head, V_RATIO)
            if elected_lane():
                for array in (Q, K, V):
                    acquire_descriptor(array, item.batch)
                    # instruction_selection:
                    # fence.proxy.tensormap::generic.acquire.gpu
            if item.wend > item.cstart:
                # first K+Q tile of the item: K in the low half, Q high
                s = acquire(kq_cur)
                wait_edge("kq_done", s.stage, s.phase)
                if elected_lane(): expect_edge("kq", s.stage, 32768)
                copy_g2s(descriptor(K, item.batch), arena[KQ_BASE + s.stage*32768],
                         coords=(0, kh, item.cstart*64))
                copy_g2s(descriptor(Q, item.batch), arena[KQ_BASE + s.stage*32768 + 8192],
                         coords=(0, qh, item.cstart*64))
                # instruction_selection: cp.async.bulk.tensor.3d.shared::cta
                # .global.tile.mbarrier::complete_tx::bytes; extent: each
                # half-tile is two 8-KB 64-channel boxes at +off and
                # +off+16384 (interleaved [K_lo, Q_lo, K_hi, Q_hi] stage;
                # four instructions per 32-KB stage)
                for chunk in range(item.cstart+1, item.wend):
                    member = (chunk - item.cstart) & 1
                    s = acquire(kq_cur)
                    wait_edge("kq_done", s.stage, s.phase)
                    if elected_lane(): expect_edge("kq", s.stage, 32768)
                    first, second = (K, Q) if member == 0 else (Q, K)
                    copy_g2s(descriptor(first, item.batch),
                             arena[KQ_BASE + s.stage*32768], (0, head(first), chunk*64))
                    copy_g2s(descriptor(second, item.batch),
                             arena[KQ_BASE + s.stage*32768 + 8192],
                             (0, head(second), chunk*64))
                    # instruction_selection: same rank-3 TMA pair, two 8-KB
                    # boxes per half-tile (subtile stride 16384 B); member
                    # parity swaps which operand owns the +0 half
                    vs = acquire(v_cur)
                    wait_edge("v_done", vs.stage, vs.phase)
                    if elected_lane(): expect_edge("v", vs.stage, 16384)
                    copy_g2s(descriptor(V, item.batch), arena[V_BASE + vs.stage*16384],
                             (0, vh, (chunk-1)*64))
                    # instruction_selection: rank-3 TMA, one-behind V tile
                vs = acquire(v_cur)
                wait_edge("v_done", vs.stage, vs.phase)
                if elected_lane(): expect_edge("v", vs.stage, 16384)
                copy_g2s(descriptor(V, item.batch), arena[V_BASE + vs.stage*16384],
                         (0, vh, (item.wend-1)*64))
            tile, sched = scheduler_publish(sched, tile)
        drain_producer("kq_done", kq_cur); drain_producer("v_done", v_cur)

    # ====================================================================
    # Warp 11: epilogue O and checkpoint TensorMap stores
    # ====================================================================
    if warp == 11:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        o_cur = consumer_cursor(O_STAGES, phase=0)
        sched = consumer_cursor(SCHED_STAGES, phase=0)
        checkpoint_serial = 0
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            n_local = item.wend - item.cstart
            if elected_lane():
                acquire_descriptor(O, item.batch)
                if CHECKPOINTS: acquire_descriptor(CHECKPOINT, item.batch)
            if CHECKPOINTS:
                ckpt_coord = ceil_div(item.wstart, ckpt_chunks)
                ckpt_mod = (item.cstart + 1) % ckpt_chunks
            if n_local > 0:
                if CHECKPOINTS and item.wstart == 0:
                    stage = checkpoint_serial % CHECKPOINT_STAGES
                    wait_edge("checkpoint", stage, phase_of(checkpoint_serial))
                    copy_s2g(arena[CHECKPOINT_BASE + stage*32768],
                             checkpoint_coord(0, 0, ckpt_coord, item.head))
                    # instruction_selection: cp.async.bulk.tensor.4d.global
                    # .shared::cta.tile.bulk_group; extent: two 64-channel
                    # subtiles
                    store_commit(); store_wait(0)
                    # instruction_selection: cp.async.bulk.commit_group;
                    # cp.async.bulk.wait_group 0
                    arrive_edge("checkpoint_done", stage)
                    ckpt_coord += 1; checkpoint_serial += 1
                for local in range(n_local):
                    chunk = item.cstart + local
                    did_o = did_ckpt = False
                    os = acquire(o_cur)
                    wait_edge("o_stage", os.stage, os.phase)
                    if item.wstart <= chunk < item.wend:
                        copy_s2g(arena[O_BASE + os.stage*16384],
                                 output_coord(0, item.head, chunk*64))
                        # instruction_selection: rank-3 TMA S2G, two 8-KB
                        # subtiles; extent capped by the per-batch sequence dim
                        store_commit(); did_o = True
                    if CHECKPOINTS:
                        stage = checkpoint_serial % CHECKPOINT_STAGES
                        if item.wstart-1 <= chunk < item.wend-1 and ckpt_mod == 0:
                            wait_edge("checkpoint", stage, phase_of(checkpoint_serial))
                            copy_s2g(arena[CHECKPOINT_BASE + stage*32768],
                                     checkpoint_coord(0,0,ckpt_coord,item.head))
                            store_commit(); ckpt_coord += 1; did_ckpt = True
                        ckpt_mod = 0 if ckpt_mod+1 == ckpt_chunks else ckpt_mod+1
                    # source completion order: O group first, checkpoint after
                    if CHECKPOINTS and did_o and did_ckpt:
                        store_wait(1); arrive_edge("o_done", os.stage)
                        store_wait(0); arrive_edge("checkpoint_done", stage)
                        checkpoint_serial += 1
                    elif CHECKPOINTS and did_o:
                        store_wait(0); arrive_edge("o_done", os.stage)
                    elif CHECKPOINTS:
                        if did_ckpt:
                            store_wait(0); arrive_edge("checkpoint_done", stage)
                            checkpoint_serial += 1
                        arrive_edge("o_done", os.stage)
                    else:
                        if did_o: store_wait(0)
                        arrive_edge("o_done", os.stage)
            tile, sched = scheduler_consume(sched, tile)
```

## Numerical order frozen by the schedule

For one token, the source recurrence represented by the chunk factorization is:

```text
S_decayed = alpha_t * S            # alpha = exp(g), gates kept in log2 domain
erase     = k_t @ S_decayed
y_t       = v_t - erase
S         = S_decayed + outer(k_t, beta-folded y)
o_t       = (scale * q_t) @ S_intra+inter
```

Within a chunk the exact source order matters: the gate warp converts to log2
(safe gate through `-exp(a_log·log2e)·softplus(g+bias)·log2e`, natural log
through `·1/ln2`, raw alpha through `lg2(x+1e-10)`) and prefix-scans with warp
shuffles; T-pairwise cells are `exp2(cumsumlog_i − cumsumlog_j)` masked by an
opaque zero; the KK epilogue folds `decay·beta_row` before IO rounding; the
64×64 inverse runs Gauss-Jordan on 8×8 diagonals then three blockwise
correction levels in register MMA with IO-dtype rounding at each level; beta
column scaling rounds through packed IO pairs after the inverse; U, Q·S, and O
accumulate in FP32 TMEM; `Y = V − cumprod·(K·S)` is computed in packed 16-bit
subtraction; the decayed U multiplies `exp2(last−prefix)` before IO rounding.
Tail tokens get log-domain zero gate contribution and TMA OOB zero fill;
output and checkpoint stores clip in hardware through per-batch descriptor
sequence extents.

## Logical tcgen05 GEMMs

TMEM accumulators hold transposed results with tokens (or DK for the state)
in the N dimension.

| # | FP32 TMEM destination | A operand | B operand | logical MxNxK | issues | accumulate |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1+2 | fused KK/QK ring col 256/320 | K+Q SMEM stage (SS) | same-stage K SMEM | 128x64x128 | 8 | K-phase only |
| 3 | (K·S)^T col 384 | packed S^T TMEM | K half of the K+Q stage | 128x64x128 | 8 | K-phase only |
| 4 | (Q·S)^T col 128 | packed S^T TMEM | Q half of the K+Q stage | 128x64x128 | 8 | K-phase only |
| 5 | U^T col 384 | packed Y^T TMEM | beta-scaled T_inv SMEM | 128x64x64 | 4 | K-phase only |
| 6 | O^T col 128 | packed U^T TMEM | causal A SMEM | 128x64x64 | 4 | runtime `have_state` |
| 7 | S^T cols 0..127 | packed decayed-U^T TMEM | K^T lead-16384 view | 128x128x64 | 4 | runtime `have_state` |

The fused KK/QK op has four static sites (two pre-loop, one per member
parity), so the BF16 anchor has `4*8 + 8 + 8 + 4 + 4 + 4 = 60` static
`tcgen05.mma` sites, matching the export.  GEMMs 3 and 5 share the CG1
accumulator columns sequentially within one chunk; publication follows the
exact delayed commit sites in warp 10.

## Storage-alias lifetimes

- The T-inverse buffer stage holds, in order: the beta-row-scaled masked KK
  product `M`, the in-place hierarchical inverse of `I + M` (identity injected
  by lane predicates), then the beta-column-scaled `T_inv` consumed by GEMM 5.
- The CG1 TMEM accumulator (col 384) holds `(K·S)^T` until CG1 consumes it
  into `Y`, then `U^T` from GEMM 5 in the same chunk.
- The YU input slot (col 448) holds packed `Y^T` for GEMM 5, then packed
  `U^T` for GEMM 6; the decayed-U slot (col 480) feeds GEMM 7.
- The Q-state accumulator (col 128) receives GEMM 4, is scaled in place by
  `cumprod·scale`, then accumulates GEMM 6 before the O epilogue drains it.
- One K+Q SMEM stage feeds the fused pair, GEMM 3 (K half), GEMM 4 (Q half),
  and GEMM 7 (K^T view) before `kq_done` releases it.

## Source / PTX / sketch correspondence

`nostate-main`/`nostate-prologue` mean the two files in
`source_export/dump_nostate`, `state-*` the checkpoint + initial-state files.

| Source action | Source lines | PTX evidence | Sketch section |
| --- | ---: | --- | --- |
| mbarrier inventory and arrive counts | 125..231 | nostate-main 60 `mbarrier.init` sites | PROTOCOL table |
| 8x8 Gauss-Jordan diagonal inverse | 233..263 | `.loc 1 233..263`; shfl.sync.idx with clustered mask, v4.b32 vector row load/stores (`.loc 1 251/262`) | CG0 `gauss_jordan_8x8` |
| 8→16 blockwise correction | 265..315 | `.loc 1 265..315`; `mma.sync.m16n8k8` x4, ldmatrix.x1 | CG0 `blockwise_8_to_16` |
| 16→32 blockwise correction | 316..374 | `.loc 1 316..374`; `mma.sync.m16n8k16`, ldmatrix.x4 | CG0 `blockwise_16_to_32` |
| 32→64 blockwise correction | 375..458 | `.loc 1 375..458`; 16 `mma.sync.m16n8k16`, barrier(2) | CG0 `blockwise_32_to_64` |
| dynamic ticket publish/consume | 463..488 | dyn-sched profile; `atom.global.add.s32`, sched mbarriers | `scheduler_publish/consume` |
| O and checkpoint TensorMap stores | 491..629 | `.loc 1 548..628`; 2 rank-3 S2G sites (nostate), rank-4 checkpoint sites in state-main | warp 11 |
| gate/beta loads, scan, transforms | 631..756 | `.loc 1 662..748`; predicated ld.global, shfl.up x10, ex2, cp.async.ca x2, cp.async.mbarrier.arrive | warp 8 |
| tcgen05 alloc + GEMM descriptors | 758..903 | `.loc 1 792..900`; tcgen05.alloc, descriptor immediates | warp 10 preamble |
| fused KK/QK pairs + lookahead | 904..981, 1016..1030 | `.loc 1 908..935`; 32 of 60 `tcgen05.mma` sites | warp 10 `fused_kk_qk` |
| GEMMs 3..7 chain | 983..1054 | `.loc 1 983..1054`; 28 `tcgen05.mma` sites, commit pairs | warp 10 loop |
| TMEM teardown | 1058..1064 | relinquish + dealloc pair | warp 10 tail |
| K+Q interleaved and V TMA loads | 1067..1196 | `.loc 1 1138..1187`; 16 rank-3 G2S sites, expect-tx x4 | warp 9 |
| T-pairwise decay fragments | 1263..1317 | `.loc 1 1287..1317`; ex2/sub/selp blocks | CG0 pair prologue |
| KK epilogue | 1332..1388 | `.loc 1 1362..1388`; tcgen05.ld.16x256b, packed mul, stmatrix | CG0 `do_kk` |
| pair inverse orchestration | 1390..1434 | `.loc 1 1399..1434`; bar.sync id-2 sites | CG0 inverse ladder |
| beta column scaling + publish | 1436..1507 | `.loc 1 1440..1506`; ldmatrix/stmatrix x4, fence.proxy | CG0 publish |
| A epilogue | 1509..1552 | `.loc 1 1522..1552`; tcgen05.ld + packed mul + stmatrix | CG0 `do_a` |
| initial-state seed | 1650..1683 | state-main `.loc 1 1652..1683`; tcgen05.st.32x32b, bar.sync id-4 | CG1 seed |
| state restage/checkpoint/rescale | 1715..1782 | `.loc 1 1722..1782`; tcgen05.ld/st.32x32b, packed cvt | CG1 state block |
| per-row gate registers | 1784..1797 | `.loc 1 1785..1796`; add.rn.f32x2, ex2 | CG1 gate regs |
| Y = V − cumprod·KS | 1799..1842 | `.loc 1 1804..1842`; ldmatrix.trans x8, sub.bf16x2 x32, tcgen05.st.16x128b | CG1 Y |
| Q·S scale | 1844..1866 | `.loc 1 1850..1866`; tcgen05.ld/st.16x256b | CG1 Q·S |
| U repack + decayed U | 1868..1908 | `.loc 1 1873..1907`; tcgen05.ld.16x256b, st.16x128b x2 waves | CG1 U |
| O epilogue | 1910..1942 | `.loc 1 1915..1941`; tcgen05.ld, stmatrix.trans x8 | CG1 O |
| final state store + passthrough | 1944..1977 | `.loc 1 1950..1977`; st.global.v2.b64 x64 | CG1 tail |
| descriptor arrays (5) | 1994..2048 | nostate-prologue; b64 copies + tensormap.replace + release fence | launch 1 arrays |
| order pass | 2051..2130, split_k.py | nostate-prologue; sort/atomics/v4 stores | launch 1 order |
| base maps and launch | 2133..2228 | host-built descriptors; box (64,1,64) token-ordinal-2 | launch 1 ABI |
| GQA head folds | 2259..2356 | grouped profile; ratio divisions at issue sites | `input_head` |
| arena, SmemTile bases, barrier init | 2420..2633 | `.loc 1 2586..2632`; 60 init + fence + bar.sync | arena/protocol |
| dispatch | 2637..2754 | warp-guard branch structure | role dispatch |
| cfg stage/offset derivation | 2757..2895 | compile-time constants in both profiles | specialization header |

Reverse lookup is exhaustive at action granularity: each accepted compile-time
branch, storage family, role, logical GEMM, scheduler operation, descriptor
operation, and pipeline family has one owner in the table and one concrete
section above.

## Static instruction evidence for the BF16 nostate anchor

Counts are static sites in the exported main-kernel PTX (instruction lines,
unpredicated convention).

| Instruction family | Static sites | Owner |
| --- | ---: | --- |
| `mbarrier.init.shared.b64` | 60 | protocol init |
| `mbarrier.try_wait.parity.acquire...` | 58 | role-local waits |
| `mbarrier.arrive.shared.b64` | 21 | thread publications |
| `mbarrier.arrive.expect_tx.shared.b64` | 4 | warp-9 TMA transactions |
| `tcgen05.mma.cta_group::1.kind::f16` | 60 | warp 10 |
| `tcgen05.commit...mbarrier::arrive::one` | 12 | warp 10 publications |
| `tcgen05.ld` 16x256b.x8 / 32x32b.x32 | 12 / 8 | CG0 + CG1 |
| `tcgen05.st` 16x128b.x8 / 32x32b / 16x256b.x8 | 6 / 8 / 2 | CG1 |
| `mma.sync.m16n8k16` / `m16n8k8` (bf16) | 20 / 4 | CG0 inverse ladder |
| rank-3 TMA G2S / S2G | 16 / 2 | warp 9 / warp 11 |
| `cp.async.ca.shared.global` + mbarrier arrive | 2 + 1 | warp 8 beta |
| `ldmatrix` x4 / x4.trans / x1 / x1.trans | 11 / 18 / 2 / 4 | CG0 + CG1 |
| `stmatrix` x4 / x4.trans / x1 | 27 / 8 / 2 | CG0 + CG1 |
| `tcgen05.alloc/relinquish/dealloc` | one each | warp 10 lifecycle |
| `setmaxnreg` dec / inc | 4 / 2 | role budgets |
| `ex2.approx.ftz.f32` | 122 | gates and decay fragments |
| `cvt.rn.bf16x2.f32` | 321 | packed IO rounding |
| `sub.bf16x2` | 32 | CG1 Y subtraction |
| `add.rn.f32x2` | 8 | CG1 decay-scale pair sums |
| `shfl.sync.up/idx.b32` | 10 / 25 | warp-8 scan, inverse ladder |
| `st.global.v2.b64` | 64 | CG1 final state |
| `bar.sync` named ids 1,2 | 9 (3 + 6) | TMEM rendezvous, inverse ladder |
| `barrier.sync 0` CTA-wide | 1 | post-init sync; id 4 appears only in the state profile |
| `fence.proxy.async.shared::cta` | 4 | SMEM publications |
| `fence.mbarrier_init.release.cluster` | 1 | protocol init |

## Validation contract

`get_kernel` returns `[descriptor_order_prologue, persistent_gdn]` in launch
order.  The host adapter gives TIRx and the standalone source the same
immutable logical inputs but independent work tables, scheduler counters,
descriptor workspaces, and O/final-state/checkpoint buffers.  Correctness
compares TIRx against the source kernel on identical inputs with relative RMS
limits 0.02 for BF16 and 0.01 for FP16 over output, final state, and the
source-written checkpoint slots, covering every frozen capability profile in
`CONFIGS`.  Benchmark timing includes both launches for each implementation;
work-table, TensorMap, reference compilation, and data preparation stay
outside the timed closure.  `prepare_bench` is CPU-only.  The benchmark suite
is the only performance authority; PTX/SASS/NCU can explain a candidate but
cannot select or pass it.
