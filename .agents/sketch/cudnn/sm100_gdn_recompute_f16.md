<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This design sketch documents a modified TIRx port of cuDNN Frontend's
python/cudnn/linear_attention/frost/kernel/gdn_recompute_f16.py at commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5.
-->

# cuDNN SM100 GDN (v1) FP16/BF16 state recompute: coarse WASP pipeline sketch

This is a non-executable execution sketch, not Python, a builder API, a new IR,
or a mathematical reference.  It freezes the source program before device-code
transcription.  The implementation it describes belongs in
[`tirx_kernels/cudnn/linear_attention/gdn_recompute_f16.py`](../../../tirx_kernels/cudnn/linear_attention/gdn_recompute_f16.py).

The frozen source is the standalone `chunk_gdn_recompute_sm100` two-launch entry
at the commit above.  The kernel re-runs the chunked delta-rule forward
recurrence for its recurrent state alone and publishes the per-chunk state
checkpoint series a backward pass consumes; it carries no query tensor, no
attention output, and no attention scale.  The main anchor is BF16 I/O, FP32
state, checkpoints enabled at the per-chunk cadence, no initial state, no final
state, dynamic ticket scheduler, order pass owned by this prologue, `BT=64`,
`DK=DV=128`, one CTA per cluster, 384 main threads (12 warps), 1024 prologue
threads, three K stages, two V stages, three T-inverse stages, one checkpoint
staging stage, three gate/cumprod/beta stages, two scheduler stages, 512
allocated TMEM columns of which 448 are used, and 191488 bytes of main dynamic
shared memory.  Specializations without checkpoints carry four K stages and no
staging buffer at exactly the same total.

The accepted capability set is the set the source entry dispatches: BF16 and
FP16 I/O; FP32 or BF16 state (FP32 when neither state tensor is present);
absent/present initial state; absent/present final state; checkpoint cadence
64, 128, or 192 — a positive multiple of the 64-token chunk — with the source
requiring at least one of the checkpoint series and the final state; grouped
K/V heads under `HO = gate.shape[1]` with `k_ratio = HO//HK` and
`v_ratio = HO//HV`; ragged/tail/zero-length sequences; gate interpretations
raw-alpha, natural-log (`log_gate`, the backward plan's production path), and
safe gate with per-head `a_log`/`dt_bias`; beta post-sigmoid FP32 or IO-dtype
logits with in-kernel sigmoid (`use_beta_sigmoid`); static grid-stride or
dynamic ticket scheduler; the prologue order pass owned or not owned by this
kernel (`run_order`); generated uncut or caller-staged/sorted work items; i32 or
i64 cumulative sequence lengths; and overlap-recompute split work rows.  Q/K L2
normalization is not part of this kernel.

Writer line-info evidence is in `.porting/gdn_recompute_f16/source_export/`:
`dump_nostate/` holds the BF16 checkpoint-only anchor
(`cutlass_host_GdnRecomputeCfg...sm_100a.ptx`, 2259 `.loc` sites, and
`cutlass_prologue_...sm_100a.ptx`, 309 `.loc` sites), `dump_state/` the
initial-state + final-state specialization (2649 `.loc`).  PTX line evidence
below refers to the nostate anchor unless a branch profile is named.  Both were
produced by `driver_gdn_recompute.py`, which validates every produced checkpoint
row against the repository's FP64 per-token recurrence over the matching token
prefix before the dump is kept.

## Pipeline at a glance

| Warps | Registers | Source-order role | Principal publications |
| --- | ---: | --- | --- |
| 0..3 | 224 | CG0 chunk-pair owner: per-pair T-pairwise decay fragments for its KK member, KK epilogue into the T-inverse buffer, four-level hierarchical pair inverse (8→16→32→64), post-inverse beta column scaling for both members | t_inv-ready, cg0-acc-done, gate/beta-done |
| 4..7 | 256 (232 without initial state) | CG1 state owner: initial-state TMEM seed, state repack FP32→IO, pre-rescale checkpoint snapshot, state rescale by chunk cumprod, per-row gate registers, `Y = V − cumprod·(K·S)`, decayed-U repack, final-state TMEM→GMEM | state-inp/y-inp/decay-u-inp, state-scale-done, checkpoint-ready, tmem-done |
| 8 | 24 (48 w/o initial state) | gate/beta producer: predicated gate loads, log2 transform (raw/log/safe), warp prefix scan → cumsum-log and cumprod SMEM rings, beta cp.async or in-register sigmoid | gate-ready, beta-ready |
| 9 | 24 (48) | TMA-LDG: per-batch K/V descriptor acquire, one K box per chunk into the alternating pair stage, one-behind V tile, dynamic-ticket producer | kq-ready (expect-tx), v-ready (expect-tx), sched-ready |
| 10 | 24 (48) | sole tcgen05 issuer: TMEM alloc/dealloc, fused M=128 KK pairs with member-parity lookahead, K·S, U, state-update chains | cg0/state/k-state/u-acc-ready, t_inv/kq-done |
| 11 | 24 (48) | epilogue: state-checkpoint TensorMap stores | checkpoint-done |

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
`max`, `select`, `exp2`, `log2`, `tanh`, `shuffle`, and `gemm`.
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
    TINV_STAGES=3, GATE_STAGES=3, BETA_STAGES=3, CHECKPOINT_STAGES=1,
    STATE_ACC_STAGES=1, STATE_INP_STAGES=1,
    CG0_ACC_STAGES=2, CG1_ACC_STAGES=1, SCHED_STAGES=2,
    CLUSTER=(1,1,1),
    REGS=(224, 256 if USE_INITIAL_STATE else 232,
          24 if USE_INITIAL_STATE else 48),
)
def gdn_recompute_factory(...):
    return descriptor_order_prologue, persistent_gdn_recompute

# ========================================================================
# Launch 1: optional work table order pass and three TensorMap arrays
# ========================================================================

@kernel(grid=(1,1,1), block=(1024,1,1), warps=32, target="sm_100a")
def descriptor_order_prologue(
    base_k, base_v, base_checkpoint,
    descriptor_workspace, cu_seqlens,
    k, v, checkpoints_or_dummy,
    staging_items_or_dummy, work_count, work_items,
    sched_ring_or_dummy, batch_count,
    k_row_stride, v_row_stride,
    checkpoint_row_stride, checkpoint_every_n_tokens,
):
    tid = thread_id()
    warp = tid // 32

    # One rank-1 u8 arena: two 4096-i32 order regions plus a two-i32 spread
    # region; descriptor building needs no SMEM payload.  The arena exists
    # only under RUN_ORDER.
    arena = linear_buffer("smem", "u8", 32776, 0, 16, "whole launch")
    order_key = integer_region(arena, 0, 4096, "i32")
    order_idx = integer_region(arena, 16384, 4096, "i32")
    order_spread = integer_region(arena, 32768, 2, "i32")

    if RUN_ORDER:
        # Source LPT order pass (common/split_k.py order_body): this kernel is
        # the backward pair's first table consumer, so it zeroes EVERY cell of
        # the shared sched region, then generates uncut rows (batch*HO,
        # chunk-span key from cu_seqlens) or copies caller-staged rows; above
        # 4096 items it copies through unsorted; otherwise it spread-detects
        # via shared atomics and stable bitonic sorts descending by span.
        if tid == 0:
            for cell in range(sched_ring_cells):
                copy_r2g(i32(0), sched_ring[cell])
                # instruction_selection: st.global.b32 (.loc 2 635); extent: a
                # serial thread-0 loop over the ring, unrolled to 29 sites
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

    # Exactly one descriptor array belongs to each warp 0..2.  Its elected lane
    # walks all batches serially.  Slots are K,V,Checkpoint in that exact
    # warp/array order; warp 2 is compile-time inactive without checkpoints.
    # K and V base maps carry (channel, head, token) with the token axis at TMA
    # ordinal 2; the checkpoint map carries (dv, dk, checkpoint, head) with the
    # checkpoint count at ordinal 2.  Without the series the source aliases the
    # checkpoint base map onto V's.
    base_maps = (base_k, base_v, base_checkpoint)
    checkpoint_prefix = 0
    if warp < 3 and elected_lane() and (warp < 2 or CHECKPOINTS):
        array = warp
        for batch in range(batch_count):
            bos = copy_g2r(cu_seqlens[batch])
            eos = copy_g2r(cu_seqlens[batch+1])
            length = eos - bos
            dst = descriptor_slot(descriptor_workspace, array, batch)
            copy_p2g(base_maps[array], dst)
            # instruction_selection: st.global.b64; extent: sixteen b64 copies
            # of the 128-byte base descriptor per slot, 48 sites across the
            # three arrays, imprecisely attributed to `.loc 1 1813`
            if array == 2:
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
        # instruction_selection: fence.proxy.tensormap::generic.release.gpu;
        # extent: 3 static sites, one per array warp
        # (.loc 1 1726 / 1730 / 1737)

# ========================================================================
# Launch 2 ABI, scalar maps, arena and protocol
# ========================================================================

@kernel(grid=(num_sms,1,1), block=(384,1,1), warps=12,
        cluster=(1,1,1), min_blocks_per_sm=1, target="sm_100a")
def persistent_gdn_recompute(
    descriptor_workspace, n_desc,
    k, v, gate, a_log_or_dummy, dt_bias_or_dummy, beta,
    cu_seqlens, initial_state_or_dummy, final_state_or_dummy,
    work_items, work_count, sched_counter_or_dummy,
    checkpoint_every_n_tokens,
):
    # k and v are carried for source ABI parity only: the source kernel takes
    # mK/mV and never reads them, because every K/V byte arrives through the
    # prologue-built TMA descriptors.
    tid = thread_id()
    warp = warp_uniform(tid // 32)
    lane = tid & 31
    cta = block_id_x()
    grid_x = grid_dim_x()
    total_tiles = copy_g2r(work_count[0])
    # instruction_selection: ld.global.b32 (.loc 1 2077)

    def decode_work(tile):
        # fields: batch, head, wstart, wend, cstart, cend, bos, eos;
        # num_chunks_b = ceil_div(eos-bos, 64).  An item computes chunks
        # [cstart, wend) and writes checkpoints only for [wstart, wend).
        row = copy_g2r(work_items[tile,0:8])
        return row_with_derived_fields(row)
        # instruction_selection: role-projected work-row decode, normally
        # ld.global.v4.b32 plus required scalar loads (.loc 5 144..148)

    def input_head(head, ratio):
        return head if ratio == 1 else head // ratio

    def descriptor(array, batch):
        return descriptor_slot(descriptor_workspace, array, batch)

    # One rank-1 u8 declaration, 191488 bytes, 1024-byte aligned.  The first
    # 3072 bytes are the protocol header plus the three FP32 gate rings; the
    # payloads follow at 1024-aligned bases.  Checkpoint mode drops one K stage
    # and inserts the checkpoint staging buffer at the same total footprint.
    arena = linear_buffer("smem", "u8", 191488, 0, 1024, "whole launch")

    BARRIER_BASE = 0          # 48 x 8-byte mbarriers, declaration-ordered
    TMEM_MAILBOX = 384        # tcgen05.alloc result, one i32
    SCHED_BASE = 400          # dynamic-ticket ring, SCHED_STAGES x i32
    CUMSUMLOG_BASE = 512      # f32 [64 x 3 stages]
    CUMPROD_BASE = 1280       # f32 [64 x 3 stages]
    BETA_BASE = 2048          # f32 [64 x 3 stages]
    CHECKPOINT_BASE = 3072    # checkpoint mode only: IO [128 x 128], 32768 B
    KQ_BASE = 3072 + (32768 if CHECKPOINTS else 0)
                              # IO [2 boxes x 64 x 128] x KQ_STAGES (32768 B/stage)
    TINV_BASE = KQ_BASE + KQ_STAGES*32768        # IO [64 x 64] x 3, 8192 B/stage
    V_BASE = TINV_BASE + 3*8192                  # IO [128 x 64] x 2, 16384 B/stage
    # checkpoints: KQ 35840, TINV 134144, V 158720, end 191488.
    # nostate:     KQ  3072, TINV 134144, V 158720, end 191488.

    # Every IO payload tile is stored 128-byte XOR-swizzled:
    # phys_col_elems = col ^ ((row & 7) * 8) within each 64-element row
    # segment; tcgen05 operand descriptors carry lead 16 bytes, stride 1024
    # bytes, swizzle-128B.  The K^T view of the same K stage bytes uses lead
    # 16384 bytes for the state-update GEMM; the V descriptor uses lead 8192.
    #
    # One K stage holds the pair's two chunks as four 8192-byte boxes:
    #   +0     K(even chunk) channels 0..63     <- TMA subtile 0 of box 0
    #   +8192  K(odd  chunk) channels 0..63     <- TMA subtile 0 of box 1
    #   +16384 K(even chunk) channels 64..127   <- TMA subtile 1 of box 0
    #   +24576 K(odd  chunk) channels 64..127   <- TMA subtile 1 of box 1
    # so the member offset is KQ_BOX = 8192 B and the high-channel segment
    # offset is KQ_SEG = 16384 B.

    # Protocol table, declaration-ordered physical mbarriers.  Each tuple is
    # (name, ready, done-or-None, ready stages, done stages,
    #  ready arrivals, done arrivals).
    PROTOCOL = (
      ("kq",           0,  32, KQ_STAGES,KQ_STAGES, 1,1),   # TMA expect-tx / MMA commit
      ("v",           64,  80, 2,2,   1,128),               # TMA expect-tx / CG1 threads
      ("gate",        96, 120, 3,3,  32,256),               # warp 8 / CG0+CG1
      ("beta",       144, 168, 3,3,  32,128),               # warp 8 (cp.async) / CG0
      ("state_acc",  192, 200, 1,1,   1,128),               # MMA commit / CG1 scale-done
      ("cg0_acc",    208, 224, 2,2,   1,64),                # MMA commit / half-CG0
      ("k_state_acc",240,None, 1,0,   1,None),              # MMA commit
      ("u_acc",      248,None, 1,0,   1,None),              # MMA commit
      ("t_inv",      256, 280, 3,3, 128,1),                 # CG0 / MMA commit
      ("state_inp",  304,None, 1,0, 128,None),              # CG1 packed state
      ("y_inp",      312,None, 1,0, 128,None),              # CG1 packed Y
      ("decay_u_inp",320,None, 1,0, 128,None),              # CG1 packed decayed U
      ("checkpoint", 328, 336, 1,1,   4,32),                # CG1 warp-elects / epilogue
      ("tmem_done",  344,None, 1,0, 128,None),              # CG1 -> MMA dealloc
      ("sched",      352, 368, 2,2,   1,11),                # warp 9 / 11 consumers
    )

    pool = smem_pool(base=arena)   # allocates only the header prefix
    for edge in PROTOCOL:
        allocate_edge_from_pool(edge)

    # All 46 physical barriers are initialized before the init fence (48 when
    # the K ring keeps four stages); the port distributes the init sites across
    # the producer-owning warps.
    init_protocol_edges(PROTOCOL)
    # instruction_selection: mbarrier.init.shared.b64; extent: 46 sites
    fence("mbarrier_init_release_cluster")
    barrier(0,384)
    # instruction_selection: fence.mbarrier_init.release.cluster (.loc 1 2219);
    # barrier.sync 0 (.loc 1 2220)

    # TMEM columns: 512 allocated, 448 used.  Packed IO regions hold two
    # elements per 32-bit cell; accumulators are FP32.  All row bases are
    # warp-relative tmem_cell(base, warp_row, col) = base + col + (warp_row<<16).
    STATE_ACC = 0        # S^T accumulator, 128 rows x cols 0..127
    STATE_INP = 128      # packed IO S^T staging (GEMM 3 A), cols 128..191
    CG0_ACC = 192        # KK ring, 2 stages x 64, cols 192..319
    CG1_ACC = 320        # (K*S)^T then U^T accumulator, cols 320..383
    Y_INP = 384          # packed Y^T (GEMM 5 A), cols 384..415
    DECAY_U_INP = 416    # packed decayed U^T (GEMM 7 A), cols 416..447

    def wait_edge(name, stage, phase):
        wait(protocol_ptr(name, stage), phase)
        # instruction_selection:
        # mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 in a retry loop
        # (.loc 2 39; 55 static sites)

    def arrive_edge(name, stage=0):
        arrive(protocol_ptr(name, stage))
        # instruction_selection: mbarrier.arrive.shared.b64 (.loc 2 45)

    def expect_edge(name, stage, nbytes):
        expect_bytes(protocol_ptr(name, stage), nbytes)
        # instruction_selection: mbarrier.arrive.expect_tx.shared.b64 (.loc 2 51)

    def commit_edge(name, stage=0):
        commit(protocol_ptr(name, stage))
        # instruction_selection:
        # tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
        # (.loc 2 118; 9 static sites, 10 with the initial-state seed)

    def scheduler_publish(cursor, tile):      # warp 9 side
        if DYNAMIC:
            wait_edge("sched_done", cursor.stage, cursor.phase)
            if elected_lane():
                ticket = atomic_add(sched_counter[0], 1, scope="gpu", order="relaxed")
                # instruction_selection: atom.global.add.u32 (.loc 1 419)
                copy_r2s(grid_x + ticket, arena[SCHED_BASE + 4*cursor.stage])
                # instruction_selection: st.shared.b32 (attributed .loc 1 1896)
            warp_sync()
            next_tile = copy_s2r(arena[SCHED_BASE + 4*cursor.stage])
            # instruction_selection: ld.shared.b32 (.loc 1 422)
            if elected_lane(): arrive_edge("sched", cursor.stage)
            return next_tile, advance(cursor)
        return tile + grid_x, cursor

    def scheduler_consume(cursor, tile):      # all consumer roles
        if DYNAMIC:
            wait_edge("sched", cursor.stage, cursor.phase)
            next_tile = copy_s2r(arena[SCHED_BASE + 4*cursor.stage])
            # instruction_selection: ld.shared.b32 (.loc 1 434; 5 role sites)
            if elected_lane(): arrive_edge("sched_done", cursor.stage)
            return next_tile, advance(cursor)
        return tile + grid_x, cursor

    # ====================================================================
    # Warps 0..3 (CG0): T-pairwise, KK epilogue, pair inverse, beta scaling
    # ====================================================================
    if 0 <= warp <= 3:
        set_register_budget("increase", 224)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 (.loc 1 1064)
        barrier(1, 288)      # TMEM-alloc rendezvous with warps 4..7 and 10
        # instruction_selection: bar.sync 1 (.loc 1 1070)
        tmem = copy_s2r(arena[TMEM_MAILBOX])
        # instruction_selection: ld.shared.b32 (.loc 1 1074)
        gate_cur = consumer_cursor(GATE_STAGES, phase=0)
        beta_cur = consumer_cursor(BETA_STAGES, phase=0)
        cg0_cur = consumer_cursor(CG0_ACC_STAGES, phase=0)
        tinv_cur = producer_cursor(TINV_STAGES, phase=1)
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

                # -- T-pairwise decay fragments for this warp's KK member --
                # Acquire gate stage 0 (and stage 1 when the pair has two
                # members); this warp's KK member is stage pair_half.  Unlike
                # the prefill sibling only ONE decay fragment set is built,
                # because there is no A epilogue.
                g0 = acquire(gate_cur); g1 = acquire(gate_cur) if have_m1 else g0
                wait_edge("gate", g0.stage, g0.phase)
                if have_m1: wait_edge("gate", g1.stage, g1.phase)
                kk_g = g1 if pair_half == 1 else g0
                rows = copy_s2r(cumsumlog_rows(kk_g))       # 4 row reads
                # instruction_selection: ld.shared.b32 x4 (.loc 1 1128)
                cols = copy_s2r(cumsumlog_cols(kk_g))       # 16 column reads
                # instruction_selection: ld.shared.v2.b32 x8 (.loc 1 1133)
                # Per accumulator fragment cell (i,j):
                #   decay[i,j] = exp2(cumsumlog[i] - cumsumlog[j]) if i>=j else 0
                decay_kk = [exp2(sub(rows[r(i)], cols[c(j)])) if lower(i,j)
                            else opaque_zero() for (i,j) in acc_frag()]
                # instruction_selection: sub.f32, ex2.approx.ftz.f32 (52 sites,
                # .loc 1 1146), selp.f32 against an opaque zero; extent: 64
                # cells per thread over two 32-value halves
                arrive_edge("gate_done", g0); arrive_edge("gate_done", g1) if have_m1

                b0 = acquire(beta_cur); b1 = acquire(beta_cur) if have_m1 else b0
                wait_edge("beta", b0.stage, b0.phase)
                if have_m1: wait_edge("beta", b1.stage, b1.phase)
                kk_b = (b1 if pair_half == 1 else b0)
                kk_beta_rows = copy_s2r(beta_ring_rows(kk_b))
                # instruction_selection: ld.shared.b32 x4 (.loc 1 1162)

                # -- KK epilogue into the T-inverse buffer ----------------
                kk_acc = acquire(cg0_cur); a_acc = acquire(cg0_cur) if have_m1 else kk_acc
                if pair_half == 1: kk_acc = a_acc
                t0 = acquire(tinv_cur); t1 = acquire(tinv_cur) if have_m1 else t0
                kk_t = (t1 if pair_half == 1 else t0)
                if do_kk:
                    wait_edge("cg0_acc", kk_acc.stage, kk_acc.phase)
                    # Only the 64-row half matching this warp's member is read
                    # out of the M=128 accumulator; the other half multiplies
                    # the stage box this member does not own and is dropped.
                    kk = copy_t2r(tmem_cell(tmem, warp*32, CG0_ACC + kk_acc.stage*64))
                    # instruction_selection: tcgen05.ld.sync.aligned.16x256b
                    # .x8.b32; extent: two 16-row halves per warp
                    # (.loc 1 1194, 1197)
                    wait_tmem_load()
                    # instruction_selection: tcgen05.wait::ld.sync.aligned
                    # (.loc 1 1200; the only load wait in the kernel)
                    arrive_edge("cg0_acc_done", kk_acc.stage)
                    wait_edge("t_inv_done", kk_t.stage, kk_t.phase)
                    m = cast(IO_DTYPE, mul(mul(kk, decay_kk), kk_beta_rows))
                    # instruction_selection: 2 x mul.f32x2 (.loc 3 244) +
                    # 1 x cvt.rn.{bf16x2|f16x2}.f32 (.loc 3 120) per
                    # (half, pair) = 64 + 32 over the two 16-row halves
                    copy_r2s(m, smem_byte(arena, TINV_BASE, kk_t.stage,
                                          row=member_row(warp,lane),
                                          col=xor128, elem_bytes=2))
                    # instruction_selection: stmatrix.sync.aligned.m8n8.x4
                    # .shared.b16; extent: 4 fragments x 2 halves = 8 sites
                    # (.loc 1 1213)

                # -- four-level hierarchical pair inverse -----------------
                # Warps 0..1 own matrix 0, warps 2..3 matrix 1; without
                # member 1 all four warps take matrix 0 and the duplicate
                # band is identical.  Diagonal cells become I via lane
                # identity injection; result overwrites the same buffer.
                barrier(2, 128)
                if have_m1 or warp < 2:
                    gauss_jordan_8x8(arena, my_tinv_base, d=(inv_warp*32+lane)//8)
                    # instruction_selection: 1 x ld.shared.v4.b32 8-element
                    # IO-pair row load (.loc 1 217), 21 x shfl.sync.idx.b32
                    # (.loc 1 223), scaled row elimination in f32,
                    # 1 x st.shared.v4.b32 (.loc 1 228)
                barrier(2, 128)
                blockwise_8_to_16(arena, t0.base, d0=warp*16, lane)
                if have_m1: blockwise_8_to_16(arena, t1.base, d0=warp*16, lane)
                # instruction_selection per call: ldmatrix.m8n8.x1{,.trans}
                # .shared.b16 (.loc 1 236/241/257), 2 x mma.sync.aligned
                # .m16n8k8.row.col.f32.bf16.bf16.f32 (.loc 7 488), neg.f32,
                # packed cvt, 1 x stmatrix.m8n8.x1 (.loc 1 269)
                barrier(2, 128)
                if have_m1 or warp < 2:
                    blockwise_16_to_32(arena, my_tinv_base, d0=inv_warp*32, lane)
                    # instruction_selection: 3 x ldmatrix.x4{,.trans}
                    # (.loc 1 282/291/309), 4 x mma.sync.m16n8k16 (.loc 7 436),
                    # packed cvt, 1 x stmatrix.x4 (.loc 1 322)
                barrier(2, 128)
                blockwise_32_to_64(arena, my_tail_base, band=inv_warp, lane)
                # instruction_selection: 10 x ldmatrix.x4{,.trans}
                # (.loc 1 338/351/376), 16 x mma.sync.m16n8k16 (.loc 7 436),
                # packed cvt, bar.sync 2 (.loc 1 394), 2 x stmatrix.x4
                # (.loc 1 398/403)
                barrier(2, 128)
                # instruction_selection: bar.sync 2; extent: 5 ladder sites
                # (.loc 1 1232, 1238, 1247, 1255, 1264)

                # -- post-inverse beta column scaling + publish -----------
                # T_inv[:, j] *= beta[j] for member 0 then member 1, both
                # performed by all four warps: ldmatrix the finished inverse,
                # multiply by the beta ring columns, store back, fence, publish.
                for (t, b) in ((t0, b0),) + (((t1, b1),) if have_m1 else ()):
                    beta_cols = copy_s2r(beta_ring_cols(b))
                    # instruction_selection: ld.shared.v2.b32 x8
                    # (.loc 1 1272 / 1309)
                    frag = copy_s2r(smem_byte(arena, TINV_BASE, t.stage, xor128))
                    # instruction_selection: ldmatrix.m8n8.x4.shared.b16 x4
                    # (.loc 1 1276 / 1313)
                    scaled = cast(IO_DTYPE, mul(cast("f32",frag), beta_cols))
                    # instruction_selection: f16x2_to_f32 unpack x16,
                    # mul.f32x2 x16 (.loc 3 244), cvt.rn.bf16x2.f32 x16
                    # (.loc 3 120) per member block, over 2 blocks
                    copy_r2s(scaled, smem_byte(arena, TINV_BASE, t.stage, xor128))
                    # instruction_selection: stmatrix.m8n8.x4 x4
                    # (.loc 1 1292 / 1329)
                    fence("async_shared_cta")
                    # instruction_selection: fence.proxy.async.shared::cta
                    # (.loc 1 1301 / 1338)
                    arrive_edge("t_inv", t.stage)
                    arrive_edge("beta_done", b)
            tile, sched = scheduler_consume(sched, tile)
        drain_producer("t_inv_done", tinv_cur)

    # ====================================================================
    # Warps 4..7 (CG1): state seed/restage/checkpoint/rescale, Y, U, final
    # ====================================================================
    if 4 <= warp <= 7:
        set_register_budget("increase", 256 if USE_INITIAL_STATE else 232)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 (.loc 1 1383)
        barrier(1, 288)
        # instruction_selection: bar.sync 1 (.loc 1 1384)
        tmem = copy_s2r(arena[TMEM_MAILBOX])
        # instruction_selection: ld.shared.b32 (.loc 1 1388)
        v_cur = consumer_cursor(V_STAGES, phase=0)
        gate_cur = consumer_cursor(GATE_STAGES, phase=0)
        state_cur = consumer_cursor(1, phase=0)      # state_acc ready
        seed_cur = producer_cursor(1, phase=1)       # state-scale-done reuse
        kst_cur = consumer_cursor(1, phase=0)
        uacc_cur = consumer_cursor(1, phase=0)
        sched = consumer_cursor(SCHED_STAGES, phase=0)
        value_dim = (warp-4)*32 + lane               # 0..127 state row
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
                        # instruction_selection: ld.global.v2.b64 x32, each
                        # moving four FP32 of the row (state-main .loc 1 1437);
                        # BF16 state routes through packed converts first
                        copy_r2t(rows, tmem_cell(tmem, value_dim_row, STATE_ACC))
                        # instruction_selection: tcgen05.st.32x32b.x32.b32 x4
                        # (state-main .loc 1 1441)
                    else:
                        copy_r2t(fill(reg_array("f32",128), 0),
                                 tmem_cell(tmem, value_dim_row, STATE_ACC))
                        # instruction_selection: tcgen05.st.32x32b.x32.b32 x4
                        # (state-main .loc 1 1448)
                    wait_tmem_store()
                    # instruction_selection: tcgen05.wait::st (state .loc 1 1453)
                    barrier(4, 128)
                    # instruction_selection: bar.sync 4 (state .loc 1 1455);
                    # this named barrier appears only in the seeded profile

                for local in range(n_local):
                    chunk = item.cstart + local
                    if CHECKPOINTS:
                        do_ckpt_now = ckpt_mod == 0
                        ckpt_mod = 0 if ckpt_mod+1 == ckpt_chunks else ckpt_mod+1
                    if CHECKPOINTS and not USE_INITIAL_STATE and chunk == 0 and item.wstart == 0:
                        # Explicit zero checkpoint before the first chunk: the
                        # entering state of an unseeded sequence is zero and no
                        # TMEM read can supply it.
                        stage = acquire_checkpoint_done(checkpoint_serial)
                        fill_zero(arena[CHECKPOINT_BASE + stage*32768], per_thread=64)
                        # instruction_selection: st.shared.b32; extent: 64
                        # strided stores per thread (attributed .loc 1 1896)
                        fence("async_shared_cta")
                        # instruction_selection: fence.proxy.async.shared::cta
                        # (.loc 1 1476)
                        if elected_lane(): arrive_edge("checkpoint", stage)
                        checkpoint_serial += 1
                    have_state = USE_INITIAL_STATE or local > 0
                    if USE_INITIAL_STATE:
                        # the seed cursor also advances once per chunk so it
                        # tracks every scale-done arrival
                        advance(seed_cur)

                    g = acquire(gate_cur)
                    wait_edge("gate", g.stage, g.phase)
                    cumprod_total = copy_s2r(arena[CUMPROD_BASE + g.stage*256 + 63*4])
                    # instruction_selection: ld.shared.b32 (.loc 1 1488)

                    # -- state restage + checkpoint + rescale ---------------
                    if have_state:
                        st = acquire(state_cur)
                        wait_edge("state_acc", st.stage, st.phase)
                        srows = copy_t2r(tmem_cell(tmem, value_dim_row, STATE_ACC))
                        # instruction_selection: tcgen05.ld.32x32b.x32.b32 x4
                        # (.loc 1 1502)
                        packed = cast(IO_DTYPE, srows, packed_pairs)
                        copy_r2t(packed, tmem_cell(tmem, value_dim_row, STATE_INP))
                        # instruction_selection: cvt.rn.{bf16x2|f16x2}.f32 x64;
                        # tcgen05.st.32x32b.x16.b32 x4 (.loc 1 1508);
                        # tcgen05.wait::st (.loc 1 1513)
                        arrive_edge("state_inp")
                        if CHECKPOINTS and do_ckpt_now and item.wstart <= chunk < item.wend:
                            # The snapshot is taken BEFORE the rescale, so row j
                            # of the series is the state ENTERING token j*N.
                            # The source re-reads the same TMEM cells here even
                            # though `srows` still holds them.
                            stage = acquire_checkpoint_done(checkpoint_serial)
                            ck = copy_t2r(tmem_cell(tmem, value_dim_row, STATE_ACC))
                            # instruction_selection: tcgen05.ld.32x32b.x32.b32
                            # x4, a second read of the cells already in
                            # registers (.loc 1 1527)
                            copy_r2s(cast(IO_DTYPE, ck),
                                     arena[CHECKPOINT_BASE + xor128_row(value_dim)])
                            # instruction_selection: packed cvt; 16 x
                            # st.shared.v4.b32 16-byte vector stores
                            # (attributed .loc 1 1896)
                            fence("async_shared_cta")
                            # instruction_selection: fence.proxy.async
                            # .shared::cta (.loc 1 1541)
                            if elected_lane(): arrive_edge("checkpoint", stage)
                            checkpoint_serial += 1
                        scaled = mul(srows, cumprod_total)
                        copy_r2t(scaled, tmem_cell(tmem, value_dim_row, STATE_ACC))
                        # instruction_selection: mul.f32x2 x64 (.loc 3 244),
                        # four subs x sixteen pairs;
                        # tcgen05.st.32x32b.x32.b32 x4 (.loc 1 1551);
                        # tcgen05.wait::st (.loc 1 1556)
                        arrive_edge("state_scale_done", st.stage)

                    # -- per-row gate registers -----------------------------
                    cumprod_cols = copy_s2r(cumprod_ring_cols(g.stage))
                    # instruction_selection: ld.shared.v2.b32 x8 (.loc 1 1562)
                    last_log = copy_s2r(arena[CUMSUMLOG_BASE + g.stage*256 + 63*4])
                    # instruction_selection: ld.shared.b32 (.loc 1 1563)
                    decay_scale = exp2(sub(last_log, cumsumlog_ring_cols(g.stage)))
                    # instruction_selection: ld.shared.v2.b32 x8 (.loc 1 1566);
                    # neg.f32 x16 (.loc 1 1569); add.rn.f32x2 x8 pair sums
                    # against those negated columns (attributed .loc 1 1896);
                    # ex2.approx.ftz.f32 x16 (.loc 1 1570/1571)
                    arrive_edge("gate_done", g.stage)

                    # -- Y = V - cumprod * (K*S), packed 16-bit -------------
                    vv = acquire(v_cur)
                    wait_edge("v", vv.stage, vv.phase)
                    vfrag = copy_s2r(smem_byte(arena, V_BASE, vv.stage,
                                     row=token_row, col=xor128, trans=True))
                    # instruction_selection: ldmatrix.m8n8.x4.trans.shared.b16
                    # x8 (two 64-channel subtiles x four 16-token bands)
                    # (.loc 1 1583)
                    if have_state:
                        wait_edge("k_state_acc", 0, kst_cur.phase); advance(kst_cur)
                        ks = copy_t2r(tmem_cell(tmem, value_dim_row, CG1_ACC))
                        # instruction_selection: tcgen05.ld.16x256b.x8.b32 x2
                        # (.loc 1 1601)
                        y = packed_sub(vfrag, cast(IO_DTYPE, mul(ks, cumprod_cols)))
                        # instruction_selection: mul.f32x2 x32 (.loc 3 244),
                        # cvt.rn.bf16x2.f32 x32 (.loc 3 120),
                        # sub.{bf16x2|f16x2} x32 (.loc 3 297)
                    else:
                        y = vfrag
                    copy_r2t(y, tmem_cell(tmem, value_dim_row, Y_INP))
                    # instruction_selection: tcgen05.st.16x128b.x8.b32 x2
                    # (.loc 1 1611); tcgen05.wait::st (.loc 1 1616)
                    arrive_edge("y_inp")

                    # -- U epilogue: decayed-U publish ----------------------
                    wait_edge("u_acc", 0, uacc_cur.phase); advance(uacc_cur)
                    arrive_edge("v_done", vv.stage)
                    u = copy_t2r(tmem_cell(tmem, value_dim_row, CG1_ACC))
                    # instruction_selection: tcgen05.ld.16x256b.x8.b32 x2
                    # (.loc 1 1628)
                    du = cast(IO_DTYPE, mul(u, decay_scale))
                    copy_r2t(du, tmem_cell(tmem, value_dim_row, DECAY_U_INP))
                    # instruction_selection: mul.f32x2 x32 (.loc 3 244),
                    # cvt.rn.bf16x2.f32 x32 (.loc 3 120);
                    # tcgen05.st.16x128b.x8.b32 x2 (.loc 1 1644);
                    # tcgen05.wait::st (.loc 1 1649)
                    # Unlike the prefill sibling there is no undecayed U
                    # republish: nothing consumes U except the state update.
                    arrive_edge("decay_u_inp")

                # -- final state: TMEM -> GMEM after the last chunk ---------
                # The wait/arrive pair runs whether or not the final state is
                # stored, because it also closes the last chunk's state edge.
                st = acquire(state_cur)
                wait_edge("state_acc", st.stage, st.phase)
                if STORE_FINAL_STATE and item.wend == item.num_chunks_b:
                    rows = copy_t2r(tmem_cell(tmem, value_dim_row, STATE_ACC))
                    # instruction_selection: tcgen05.ld.32x32b.x32.b32 x4
                    # (state-main .loc 1 1661)
                    copy_r2g(cast(STATE_DTYPE, rows),
                             final_state[item.batch, item.head, value_dim, 0:128])
                    # instruction_selection: st.global.v2.b64 x32, 16 B per
                    # access (FP32 state) / packed converts then stores for
                    # BF16 state (state-main .loc 1 1668)
                arrive_edge("state_scale_done", st.stage)
            elif STORE_FINAL_STATE and item.wend == item.num_chunks_b:
                # Empty sequence: never touches TMEM; passthrough initial
                # state when present, else state-dtype zero.
                for key in range(128):
                    val = (copy_g2r(initial_state[item.batch,item.head,value_dim,key])
                           if USE_INITIAL_STATE else cast(STATE_DTYPE,0))
                    copy_r2g(val, final_state[item.batch,item.head,value_dim,key])
                    # instruction_selection: ld.global + st.global pairs in the
                    # seeded profile (state-main .loc 1 1680); plain st.global
                    # zero stores otherwise
            tile, sched = scheduler_consume(sched, tile)
        arrive_edge("tmem_done")
        if CHECKPOINTS: drain_checkpoint_done(checkpoint_serial)

    # ====================================================================
    # Warp 8: gate/beta producer
    # ====================================================================
    elif warp == 8:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (.loc 1 544)
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
                # instruction_selection: branch-predicated ld.global.b32 x2 with
                # min.s32 index clamp and mov-immediate OOB neutral (.loc 1 581)
                if SAFE_GATE:
                    vals = [mul(a_l2, softplus(add(vc, bias))) if valid else 0
                            for vc in vals]
                    # instruction_selection: ex2/lg2.approx.ftz chain per lane
                elif LOG_GATE:
                    vals = mul(vals, RCP_LN2)          # natural log -> log2
                    # instruction_selection: mul.f32 x2, one per owned token
                    # column (.loc 1 590)
                else:
                    vals = log2(add(vals, 1e-10))
                    # instruction_selection: lg2.approx.ftz.f32
                prefix = warp_inclusive_scan(vals)
                # instruction_selection: shfl.sync.up.b32 offsets 1,2,4,8,16 per
                # column (10 sites, .loc 1 596) plus one shfl.sync.idx.b32
                # lane-31 carry (.loc 1 600)
                wait_edge("gate_done", g.stage, g.phase)
                copy_r2s(prefix, arena[CUMSUMLOG_BASE + g.stage*256 + col*4])
                copy_r2s(exp2(prefix), arena[CUMPROD_BASE + g.stage*256 + col*4])
                # instruction_selection: st.shared.b32 x2 each (.loc 1 612/613),
                # ex2.approx.ftz.f32 x2 (.loc 1 613)
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
                    # zero-fill cp-size predication x2 (.loc 1 638)
                    cp_async_arrive(protocol_ptr("beta", b.stage), noinc=True)
                    # instruction_selection: cp.async.mbarrier.arrive.noinc
                    # .shared.b64 (.loc 1 639)
            tile, sched = scheduler_consume(sched, tile)
        drain_producer("gate_done", gate_cur); drain_producer("beta_done", beta_cur)

    # ====================================================================
    # Warp 10: sole tcgen05 issuer and TMEM lifecycle
    # ====================================================================
    elif warp == 10:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (.loc 1 669)
        allocate_tmem(TMEM_MAILBOX, columns=512, cta_group=1)
        # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned
        # .shared::cta.b32 (.loc 1 679)
        barrier(1, 288)
        # instruction_selection: bar.sync 1 (.loc 1 680)
        tmem = copy_s2r(arena[TMEM_MAILBOX])
        # instruction_selection: ld.shared.b32 (.loc 1 776)
        kq_cur = consumer_cursor(KQ_STAGES, phase=0)        # per-chunk operand
        kqf_cur = consumer_cursor(KQ_STAGES, phase=0)       # fused-pair lookahead
        cg0_cur = consumer_cursor(CG0_ACC_STAGES, phase=1)  # cg0 acc done
        tinv_cur = consumer_cursor(TINV_STAGES, phase=0)
        sinp_cur = consumer_cursor(1, phase=0)
        y_cur = consumer_cursor(1, phase=0)
        du_cur = consumer_cursor(1, phase=0)
        state_cur = producer_cursor(1, phase=1)             # state acc
        sched = consumer_cursor(SCHED_STAGES, phase=0)

        # SMEM operand descriptors are integer immediates: base>>4 plus lead
        # 16 B, stride 1024 B, swizzle-128B; the member box offset is
        # KQ_BOX = 64*64*2 B >> 4 and the high-channel segment offset is
        # KQ_SEG = 2*64*64*2 B >> 4; the state-update B operand re-views the
        # same K stage bytes with lead 16384 B.
        def fused_kk(kqf, acc_stage, member_one):
            # M = 128 spans BOTH pair boxes of the stage, N = 64 spans only the
            # box of the member being computed, so exactly half of each
            # accumulator is consumed by CG0 and the other half is dead work
            # the source keeps.
            wait_edge("cg0_acc_done", acc_stage.stage, acc_stage.phase)
            wait_edge("kq", kqf.stage, kqf.phase)
            a = smem_matrix_desc(KQ_BASE + kqf.stage*32768)
            b = a + KQ_BOX if member_one else a
            gemm(tmem_cell(tmem, 0, CG0_ACC + acc_stage.stage*64), a, b,
                 M=128, N=64, K=64, accumulate=False)
            gemm(tmem_cell(tmem, 0, CG0_ACC + acc_stage.stage*64),
                 a + KQ_SEG, b + KQ_SEG, M=128, N=64, K=64, accumulate=True)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 with
            # SMEM A/B descriptors; extent: 2 x 4 K-phases = 8 instructions
            # (.loc 7 143; 4 static sites x 8 = 32 of the 48 total)
            if elected_lane(): commit_edge("cg0_acc", acc_stage.stage)

        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            n_local = item.wend - item.cstart
            # pair member 0 and (when present) member 1 are issued ahead
            if n_local > 0: fused_kk(acquire(kqf_cur), acquire(cg0_cur), member_one=False)
            if n_local > 1: fused_kk(acquire(kqf_cur), acquire(cg0_cur), member_one=True)
            for local in range(n_local):
                member = local & 1
                have_state = USE_INITIAL_STATE or local > 0
                if USE_INITIAL_STATE and local == 0:
                    # the seeded state is already in TMEM; publish it so CG1's
                    # first restage can proceed without an MMA
                    if elected_lane(): commit_edge("state_acc", 0)
                    advance(state_cur)
                kq = acquire(kq_cur)
                desc_k = smem_matrix_desc(KQ_BASE + kq.stage*32768) + member*KQ_BOX
                desc_kt = smem_matrix_desc(KQ_BASE + kq.stage*32768,
                                           lead=16384) + member*KQ_BOX

                # member-parity lookahead keeps one fused pair in flight
                if member == 1 and local+2 < n_local:
                    fused_kk(acquire(kqf_cur), acquire(cg0_cur), member_one=True)

                # -- GEMM 3: (K*S)^T = S^T_packed @ K^T ---------------------
                if have_state:
                    wait_edge("state_inp", 0, sinp_cur.phase); advance(sinp_cur)
                    gemm(tmem_cell(tmem, 0, CG1_ACC),
                         tmem_a(STATE_INP), desc_k, M=128, N=64, K=128,
                         accumulate=per_k_phase)
                    # instruction_selection: tcgen05.mma.kind::f16 TMEM-A;
                    # extent: 2 half-K segments x 4 = 8 instructions
                    # (.loc 7 223), the second segment advancing the TMEM A
                    # pointer by 32 columns and the B descriptor by KQ_SEG
                    if elected_lane(): commit_edge("k_state_acc", 0)

                # -- GEMM 5: U^T = Y^T_packed @ T_inv -----------------------
                tv = acquire(tinv_cur)
                wait_edge("t_inv", tv.stage, tv.phase)
                wait_edge("y_inp", 0, y_cur.phase); advance(y_cur)
                gemm(tmem_cell(tmem, 0, CG1_ACC),
                     tmem_a(Y_INP), smem_matrix_desc(TINV_BASE + tv.stage*8192),
                     M=128, N=64, K=64, accumulate=per_k_phase)
                # instruction_selection: 4 x tcgen05.mma TMEM-A (.loc 7 223)
                if elected_lane():
                    commit_edge("u_acc", 0); commit_edge("t_inv_done", tv.stage)

                if member == 0 and local+2 < n_local:
                    fused_kk(acquire(kqf_cur), acquire(cg0_cur), member_one=False)

                # -- GEMM 7: S^T += decayedU^T_packed @ K -------------------
                wait_edge("decay_u_inp", 0, du_cur.phase); advance(du_cur)
                advance(state_cur)
                gemm(tmem_cell(tmem, 0, STATE_ACC),
                     tmem_a(DECAY_U_INP), desc_kt, M=128, N=128, K=64,
                     accumulate=have_state_on_first_k)
                # instruction_selection: 4 x tcgen05.mma TMEM-A (.loc 7 223),
                # B transposed through the lead-16384 K^T descriptor view and
                # the b_major instruction-descriptor field
                if elected_lane():
                    commit_edge("state_acc", 0); commit_edge("kq_done", kq.stage)
            tile, sched = scheduler_consume(sched, tile)
        wait_edge("tmem_done", 0, 0)
        relinquish_tmem_alloc_permit(cta_group=1)
        deallocate_tmem(tmem, columns=512, cta_group=1)
        # instruction_selection: tcgen05.relinquish_alloc_permit (.loc 1 912) /
        # tcgen05.dealloc.cta_group::1.sync.aligned.b32 (.loc 1 913)

    # ====================================================================
    # Warp 9: TMA-LDG and scheduler producer
    # ====================================================================
    elif warp == 9:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (.loc 1 939)
        kq_cur = producer_cursor(KQ_STAGES, phase=1)
        v_cur = producer_cursor(V_STAGES, phase=1)
        sched = producer_cursor(SCHED_STAGES, phase=1)
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            kh = input_head(item.head, K_RATIO)
            vh = input_head(item.head, V_RATIO)
            if elected_lane():
                for array in (K, V):
                    acquire_descriptor(array, item.batch)
                    # instruction_selection:
                    # fence.proxy.tensormap::generic.acquire.gpu (.loc 8 272)
            if item.wend > item.cstart:
                # first chunk of the item always lands in box 0 of its stage
                s = acquire(kq_cur)
                wait_edge("kq_done", s.stage, s.phase)
                if elected_lane(): expect_edge("kq", s.stage, 16384)
                copy_g2s(descriptor(K, item.batch), arena[KQ_BASE + s.stage*32768],
                         coords=(0, kh, item.cstart*64))
                # instruction_selection: cp.async.bulk.tensor.3d.shared::cta
                # .global.tile.mbarrier::complete_tx::bytes; extent: two
                # 8-KB 64-channel boxes at +0 and +16384 (.loc 8 77)
                for chunk in range(item.cstart+1, item.wend):
                    member = (chunk - item.cstart) & 1
                    s = acquire(kq_cur)
                    wait_edge("kq_done", s.stage, s.phase)
                    if elected_lane(): expect_edge("kq", s.stage, 16384)
                    if member == 0:
                        copy_g2s(descriptor(K, item.batch),
                                 arena[KQ_BASE + s.stage*32768],
                                 (0, kh, chunk*64))
                    else:
                        copy_g2s(descriptor(K, item.batch),
                                 arena[KQ_BASE + s.stage*32768 + 8192],
                                 (0, kh, chunk*64))
                    # instruction_selection: cp.async.bulk.tensor.3d...; extent:
                    # 2 static sites x two 8-KB subtiles 16384 B apart; 4 of the
                    # 10 rank-3 `.loc 8 77` instructions.  Member parity selects
                    # box 0 or box 1 of the stage at compile time, not through a
                    # runtime byte offset.
                    vs = acquire(v_cur)
                    wait_edge("v_done", vs.stage, vs.phase)
                    if elected_lane(): expect_edge("v", vs.stage, 16384)
                    copy_g2s(descriptor(V, item.batch), arena[V_BASE + vs.stage*16384],
                             (0, vh, (chunk-1)*64))
                    # instruction_selection: rank-3 TMA, one-behind V tile, two
                    # 8-KB subtiles 8192 B apart (.loc 8 77)
                vs = acquire(v_cur)
                wait_edge("v_done", vs.stage, vs.phase)
                if elected_lane(): expect_edge("v", vs.stage, 16384)
                copy_g2s(descriptor(V, item.batch), arena[V_BASE + vs.stage*16384],
                         (0, vh, (item.wend-1)*64))
            tile, sched = scheduler_publish(sched, tile)
        drain_producer("kq_done", kq_cur); drain_producer("v_done", v_cur)

    # ====================================================================
    # Warp 11: epilogue checkpoint TensorMap stores
    # ====================================================================
    if warp == 11:
        set_register_budget("decrease", 24 if USE_INITIAL_STATE else 48)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (.loc 1 459)
        sched = consumer_cursor(SCHED_STAGES, phase=0)
        checkpoint_serial = 0
        tile = cta
        while tile < total_tiles:
            item = decode_work(tile)
            n_local = item.wend - item.cstart
            if CHECKPOINTS:
                if elected_lane(): acquire_descriptor(CHECKPOINT, item.batch)
                # instruction_selection:
                # fence.proxy.tensormap::generic.acquire.gpu (.loc 8 272)
                ckpt_coord = ceil_div(item.wstart, ckpt_chunks)
                ckpt_mod = item.cstart % ckpt_chunks
            if n_local > 0:
                for local in range(n_local):
                    chunk = item.cstart + local
                    did_ckpt = False
                    if CHECKPOINTS:
                        stage = checkpoint_serial % CHECKPOINT_STAGES
                        if item.wstart <= chunk < item.wend and ckpt_mod == 0:
                            wait_edge("checkpoint", stage, phase_of(checkpoint_serial))
                            copy_s2g(arena[CHECKPOINT_BASE + stage*32768],
                                     checkpoint_coord(0, 0, ckpt_coord, item.head))
                            # instruction_selection: cp.async.bulk.tensor.4d
                            # .global.shared::cta.tile.bulk_group; extent: two
                            # 64-channel subtiles (.loc 8 119)
                            store_commit()
                            # instruction_selection: cp.async.bulk.commit_group
                            # (.loc 8 152)
                            ckpt_coord += 1; did_ckpt = True
                        ckpt_mod = 0 if ckpt_mod+1 == ckpt_chunks else ckpt_mod+1
                        if did_ckpt:
                            store_wait(0)
                            # instruction_selection: cp.async.bulk.wait_group
                            # .read 0 (.loc 8 157)
                            arrive_edge("checkpoint_done", stage)
                            checkpoint_serial += 1
            tile, sched = scheduler_consume(sched, tile)
```

## Numerical order frozen by the schedule

For one token, the source recurrence represented by the chunk factorization is:

```text
S_decayed = alpha_t * S            # alpha = exp(g), gates kept in log2 domain
erase     = k_t @ S_decayed
y_t       = v_t - erase
S         = S_decayed + outer(k_t, beta-folded y)
```

Within a chunk the exact source order matters: the gate warp converts to log2
(safe gate through `-exp(a_log·log2e)·softplus(g+bias)·log2e`, natural log
through `·1/ln2`, raw alpha through `lg2(x+1e-10)`) and prefix-scans with warp
shuffles; T-pairwise cells are `exp2(cumsumlog_i − cumsumlog_j)` masked by an
opaque zero; the KK epilogue folds `decay·beta_row` before IO rounding; the
64×64 inverse runs Gauss-Jordan on 8×8 diagonals then three blockwise
correction levels in register MMA with IO-dtype rounding at each level; beta
column scaling rounds through packed IO pairs after the inverse; U accumulates
in FP32 TMEM; `Y = V − cumprod·(K·S)` is computed in packed 16-bit subtraction;
the decayed U multiplies `exp2(last−prefix)` before IO rounding; the checkpoint
snapshot rounds the pre-rescale FP32 state to IO dtype.  Tail tokens get
log-domain zero gate contribution and TMA OOB zero fill; checkpoint stores clip
in hardware through the per-batch descriptor's checkpoint-count extent.

## Logical tcgen05 GEMMs

TMEM accumulators hold transposed results with tokens (or DK for the state) in
the N dimension.

| # | FP32 TMEM destination | A operand | B operand | logical MxNxK | issues | accumulate |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | KK ring col 192/256 | K stage, both boxes (SS) | the member's box of the same stage | 128x64x128 | 8 | K-phase only |
| 3 | (K·S)^T col 320 | packed S^T TMEM | the member's K box | 128x64x128 | 8 | K-phase only |
| 5 | U^T col 320 | packed Y^T TMEM | beta-scaled T_inv SMEM | 128x64x64 | 4 | K-phase only |
| 7 | S^T cols 0..127 | packed decayed-U^T TMEM | K^T lead-16384 view of the member's box | 128x128x64 | 4 | runtime `have_state` |

GEMM 1 has four static sites (two pre-loop, one per member parity), so the BF16
anchor has `4*8 + 8 + 4 + 4 = 48` static `tcgen05.mma` sites, matching the
export exactly.  GEMMs 3 and 5 share the CG1 accumulator columns sequentially
within one chunk; publication follows the exact delayed commit sites in warp 10.
The M=128 shape of GEMM 1 makes half of every KK accumulator dead: the source
keeps it because the shape is inherited from the prefill kernel's genuinely
fused KK/QK pair, and the port keeps it because dropping it would change the
issue count and the accumulator ring geometry.

## Storage-alias lifetimes

- The T-inverse buffer stage holds, in order: the beta-row-scaled masked KK
  product `M`, the in-place hierarchical inverse of `I + M` (identity injected
  by lane predicates), then the beta-column-scaled `T_inv` consumed by GEMM 5.
- The CG1 TMEM accumulator (col 320) holds `(K·S)^T` until CG1 consumes it into
  `Y`, then `U^T` from GEMM 5 in the same chunk.
- The Y input slot (col 384) feeds GEMM 5; the decayed-U slot (col 416) feeds
  GEMM 7.  Unlike the prefill sibling the Y slot is never rewritten with an
  undecayed U, because no second consumer exists.
- One K SMEM stage carries two chunks: box 0 the pair's even member, box 1 the
  odd member.  It feeds the fused KK issue for both members, GEMM 3, and GEMM 7
  (K^T view) before `kq_done` releases it.
- The checkpoint staging buffer holds one 128x128 IO snapshot between the CG1
  publish and the warp-11 TMA store; its single stage serializes CG1 against the
  store, which is why the CG1 side waits `checkpoint_done` before filling.

## Source / PTX / sketch correspondence

`nostate-main`/`nostate-prologue` mean the two files in
`source_export/dump_nostate`, `state-*` the initial-state + final-state files.

| Source action | Source lines | PTX evidence | Sketch section |
| --- | ---: | --- | --- |
| mbarrier inventory and arrive counts | 115..196 | nostate-main 46 `mbarrier.init` sites (`.loc 2 183`) | PROTOCOL table |
| 8x8 Gauss-Jordan diagonal inverse | 199..228 | `.loc 1 217/223/228`; ld.shared.v4.b32, 21 shfl.sync.idx.b32, st.shared.v4.b32 | CG0 `gauss_jordan_8x8` |
| 8→16 blockwise correction | 231..273 | `.loc 1 236/241/257/269`, `.loc 7 488`; 4 `mma.sync.m16n8k8`, ldmatrix.x1, stmatrix.x1 | CG0 `blockwise_8_to_16` |
| 16→32 blockwise correction | 276..326 | `.loc 1 282/291/309/322`, `.loc 7 436`; ldmatrix.x4, stmatrix.x4 | CG0 `blockwise_16_to_32` |
| 32→64 blockwise correction | 329..407 | `.loc 1 338/351/376/394/398/403`; 16 `mma.sync.m16n8k16`, internal bar.sync 2 | CG0 `blockwise_32_to_64` |
| dynamic ticket publish/consume | 413..438 | `.loc 1 419` atom.global.add.u32, `.loc 1 422/434` ld.shared.b32 | `scheduler_publish/consume` |
| checkpoint TensorMap stores | 441..519 | `.loc 8 119/152/157/272`; 2 rank-4 S2G sites, commit, wait_group, descriptor acquire | warp 11 |
| gate/beta loads, scan, transforms | 522..647 | `.loc 1 581/596/600/612/613/638/639`; predicated ld.global, shfl.up x10, ex2, cp.async.ca x2, cp.async.mbarrier.arrive | warp 8 |
| tcgen05 alloc + GEMM descriptors | 650..787 | `.loc 1 679/776`; tcgen05.alloc, descriptor immediates | warp 10 preamble |
| fused KK pairs + lookahead | 792..819, 843..858, 883..897 | `.loc 7 143`; 32 of 48 `tcgen05.mma` sites | warp 10 `fused_kk` |
| GEMMs 3, 5, 7 chain | 860..907 | `.loc 7 223`; 16 `tcgen05.mma` sites, commit pairs (`.loc 2 118`) | warp 10 loop |
| TMEM teardown | 911..917 | `.loc 1 912/913`; relinquish + dealloc pair | warp 10 tail |
| K and V TMA loads | 920..1040 | `.loc 8 77`; 10 rank-3 G2S sites, expect-tx x4 (`.loc 2 51`) | warp 9 |
| T-pairwise decay fragments | 1104..1149 | `.loc 1 1128/1133/1146`; ld.shared.b32 x4 + v2.b32 x8, 52 ex2 sites | CG0 pair prologue |
| KK epilogue | 1151..1221 | `.loc 1 1162/1194/1197/1200/1213`, `.loc 3 244/120`; tcgen05.ld.16x256b x2, tcgen05.wait::ld, 64 mul.f32x2 + 32 cvt.rn.bf16x2.f32, 8 stmatrix.x4 | CG0 `do_kk` |
| pair inverse orchestration | 1223..1267 | `.loc 1 1232/1238/1247/1255/1264`; 5 bar.sync id-2 sites | CG0 inverse ladder |
| beta column scaling + publish | 1269..1340 | `.loc 1 1272/1276/1292/1301` and 1309/1313/1329/1338; ldmatrix/stmatrix x4 each, fence.proxy | CG0 publish |
| initial-state seed | 1426..1458 | state-main `.loc 1 1437/1441/1448/1453/1455`; ld.global.v2.b64 x32, tcgen05.st.32x32b x8, bar.sync id-4 | CG1 seed |
| zero checkpoint before chunk 0 | 1466..1479 | `.loc 1 1476` fence; 64 st.shared.b32 (attributed `.loc 1 1896`) | CG1 zero checkpoint |
| state restage | 1491..1515 | `.loc 1 1502/1508/1513`; tcgen05.ld.32x32b x4, tcgen05.st.32x32b.x16 x4, wait::st | CG1 restage |
| pre-rescale checkpoint snapshot | 1517..1544 | `.loc 1 1527/1541`; tcgen05.ld.32x32b x4 (a redundant second read), 16 st.shared.v4.b32 (attributed `.loc 1 1896`), fence.proxy | CG1 checkpoint |
| state rescale | 1546..1557 | `.loc 1 1551/1556`, `.loc 3 244`; mul.f32x2 x64, tcgen05.st.32x32b x4, wait::st | CG1 rescale |
| per-row gate registers | 1559..1572 | `.loc 1 1562/1563/1566/1570/1571`; ld.shared.v2.b32 x16, neg.f32 x16 (`.loc 1 1569`), add.rn.f32x2 x8 (attributed `.loc 1 1896`), 16 ex2 | CG1 gate regs |
| Y = V − cumprod·KS | 1574..1617 | `.loc 1 1583/1601/1611/1616`, `.loc 3 297`; ldmatrix.trans x8, tcgen05.ld.16x256b x2, 32 sub.bf16x2, tcgen05.st.16x128b x2 | CG1 Y |
| U epilogue + decayed U | 1619..1650 | `.loc 1 1628/1644/1649`, `.loc 3 244/120`; tcgen05.ld.16x256b x2, mul.f32x2 x32, cvt x32, tcgen05.st.16x128b x2 | CG1 U |
| final state store + passthrough | 1652..1684 | state-main `.loc 1 1661/1668/1680`; tcgen05.ld.32x32b x4, st.global.v2.b64 x32, passthrough ld.global.b32/st.global.b32 x16 | CG1 tail |
| checkpoint drain | 1690..1695 | `.loc 2 39` try_wait sites | CG1 drain |
| descriptor arrays (3) | 1698..1737 | nostate-prologue; 48 st.global.b64 base copies (imprecisely attributed `.loc 1 1813`), tensormap.replace at `.loc 3 56/61/103/108`, 3 release fences at `.loc 1 1726/1730/1737` | launch 1 arrays |
| order pass and its sched zeroing | 1740..1794, split_k.py 602..660 | nostate-prologue; thread-0 serial ring zeroing `st.global.b32` (`.loc 2 635`), row synth/copy (`.loc 2 567/568/591/595`), shared min/max atomics, bitonic sort; `run_order` gates the whole block | launch 1 order |
| base maps and prologue launch | 1813..1893 | host-built descriptors; K/V box (64,1,64) token-ordinal-2, checkpoint box (64,128,1,1) | launch 1 ABI |
| GQA head folds and cfg stamping | 1915..2015 | ratio divisions at the TMA coordinate sites (`.loc 1 978/979`, elided at ratio 1, so the anchor emits nothing here) | `input_head` |
| arena, SmemTile bases, barrier init | 2050..2220 | `.loc 1 2219/2220`; 46 init + fence + barrier.sync 0 | arena/protocol |
| dispatch | 2222..2333 | warp-guard branch structure (`.loc 1 2073` warp_uniform) | role dispatch |
| cfg stage/offset derivation | 2336..2469 | compile-time constants in both profiles | specialization header |
| host entry, cache key, validation | 2475..2823 | n/a | validation contract |
| `run_recompute` replay: prologue launch then main launch | 2825..2881 | n/a | launch order, validation contract |

Reverse lookup is exhaustive at action granularity: each accepted compile-time
branch, storage family, role, logical GEMM, scheduler operation, descriptor
operation, and pipeline family has one owner in the table and one concrete
section above.

Four unrelated operations carry imprecise line attribution in the export and are
recorded as such rather than silently mapped: the checkpoint zero fill (source
1475, 64 `st.shared.b32`), the dynamic-scheduler ring publish (source 420, 1
`st.shared.b32`), the checkpoint snapshot's SMEM vector stores (source 1538, 16
`st.shared.v4.b32`), and the CG1 decay-scale pair sums (source 1569, 8
`add.rn.f32x2`) all surface under `.loc 1 1896`, the `@cute.jit` decorator line
of `host`.  Their counts (64, 1, 16, 8) and their operand widths identify them
unambiguously; the 64/1 split is confirmed by the store addresses, the zero fill
walking the 32768-byte checkpoint buffer at a 512-byte stride and the scheduler
publish landing on the `sSched` base.

## Static instruction evidence for the BF16 checkpoint-only anchor

Counts are static sites in the exported main-kernel PTX (instruction lines,
unpredicated convention).  `fmul2` in `pointwise.py` is emitted as a braced
inline-PTX block, so its `mul.f32x2` -- and the `mov.b64` register-pair packs
around it, which dominate the raw opcode histogram and usually fold away in
SASS -- sit mid-line and a line-anchored grep returns zero for them; the 224
below is a per-mnemonic occurrence count.  The other packed mnemonics
(`add.rn.f32x2` 8, `sub.bf16x2` 32, `cvt.rn.bf16x2.f32` 289) are one-per-line
and count normally.

| Instruction family | Static sites | Owner |
| --- | ---: | --- |
| `mbarrier.init.shared.b64` | 46 | protocol init |
| `mbarrier.try_wait.parity.acquire...` | 55 | role-local waits |
| `mbarrier.arrive.shared.b64` | 25 | thread publications |
| `mbarrier.arrive.expect_tx.shared.b64` | 4 | warp-9 TMA transactions |
| `tcgen05.mma.cta_group::1.kind::f16` | 48 | warp 10 |
| `tcgen05.commit...mbarrier::arrive::one` | 9 (10 seeded) | warp 10 publications |
| `tcgen05.ld` 16x256b.x8 / 32x32b.x32 | 6 / 8 (12 seeded) | CG0 + CG1 |
| `tcgen05.st` 32x32b.x32 / 32x32b.x16 / 16x128b.x8 | 4 / 4 / 4 (12 / 4 / 4 seeded) | CG1 |
| `tcgen05.wait::st` / `tcgen05.wait::ld` | 4 / 1 (5 / 1 seeded) | CG1 / CG0 |
| `mma.sync.m16n8k16` / `m16n8k8` (bf16) | 20 / 4 | CG0 inverse ladder |
| rank-3 TMA G2S / rank-4 TMA S2G | 10 / 2 | warp 9 / warp 11 |
| `cp.async.bulk.commit_group` / `.wait_group.read` | 1 / 1 | warp 11 |
| `cp.async.ca.shared.global` + mbarrier arrive | 2 + 1 | warp 8 beta |
| `ldmatrix` x4 / x4.trans / x1 / x1.trans | 11 / 18 / 2 / 4 | CG0 + CG1 |
| `stmatrix` x4 / x1 | 19 / 2 | CG0 |
| `tcgen05.alloc/relinquish/dealloc` | one each | warp 10 lifecycle |
| `setmaxnreg` dec / inc | 4 / 2 | role budgets |
| `ex2.approx.ftz.f32` | 70 | gates and decay fragments |
| `cvt.rn.bf16x2.f32` | 289 | packed IO rounding |
| `mul.f32x2` (inline PTX, `.loc 3 244`) | 224 | every packed FP32 multiply |
| `sub.bf16x2` | 32 | CG1 Y subtraction |
| `add.rn.f32x2` | 8 | CG1 decay-scale pair sums |
| `shfl.sync.up/idx.b32` | 10 / 25 | warp-8 scan, 8x8 inverse |
| `st.shared.b32` / `st.shared.v4.b32` | 69 / 17 | gate rings, checkpoint staging |
| `ld.shared.b32` / `.v2.b32` / `.v4.b32` | 19 / 40 / 1 | rings, TMEM mailbox, inverse |
| `bar.sync` named ids 1,2 | 9 (3 + 6) | TMEM rendezvous, inverse ladder |
| `barrier.sync 0` CTA-wide | 1 | post-init sync; id 4 appears only in the seeded profile |
| `fence.proxy.async.shared::cta` | 4 | SMEM publications |
| `fence.proxy.tensormap::generic.acquire.gpu` | 3 | K, V, checkpoint descriptors |
| `fence.mbarrier_init.release.cluster` | 1 | protocol init |
| `atom.global.add.u32` | 1 | dynamic ticket |

## Validation contract

`get_kernel` returns `[descriptor_order_prologue, persistent_gdn_recompute]` in
launch order.  The host adapter gives TIRx and the standalone source the same
immutable logical inputs but independent work tables, scheduler rings,
descriptor workspaces, and checkpoint/final-state buffers.  Correctness compares
TIRx against the source kernel on identical inputs with relative RMS limits 0.02
for BF16 and 0.01 for FP16 over the source-written checkpoint slots and the
final state, covering every frozen capability profile in `CONFIGS`.  Benchmark
timing includes both launches for each implementation; work-table, TensorMap,
reference compilation, and data preparation stay outside the timed closure.
`prepare_bench` is CPU-only.  The benchmark suite is the only performance
authority; PTX/SASS/NCU can explain a candidate but cannot select or pass it.
