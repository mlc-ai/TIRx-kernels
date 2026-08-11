<!--
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2026 The TIRx Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Selective-state-update STP horizontal SM100: execution sketch

This is a non-executable transcription sketch of FlashInfer's CUDA
`selective_state_update_kernel_producer_consumer_horizontal`. It freezes the
full-DIM column-tile ring, one-producer/many-consumer warp split, three-phase
TMA protocol, B/C publication rendezvous, conflict-free shared-column
permutation, optional Philox conversion, two-lane row reduction, and output
epilogue that the TIRx port must preserve. The target module is
[`tirx_kernels/flashinfer/mamba/selective_state_update_stp_horizontal.py`](../../tirx_kernels/flashinfer/mamba/selective_state_update_stp_horizontal.py).
It may become executable only after this sketch passes independent review.

The frozen FlashInfer commit is
`f2e04400e330fb2debe0bf8730d9424a1d37927f`; the primary CUDA header SHA256 is
`c0e13b64bf42f4f8155058dc9f5877f7aca90832f50a1e7602863894908e89fd`.
The target is SM100a/B200. Input, B, C, z, and output are BF16. State is BF16,
FP16, or FP32; scaled int16 state is rejected by the horizontal launcher.
Weight/dt/D/dt_bias is FP32 or BF16, A is FP32, and indices are int32 or int64.
`DIM in {64,128}`, `DSTATE in {64,96,128,256}`, group ratio in
`{1,2,4,8,16,32,64}`, and `PHILOX_ROUNDS in {0,10}` are in scope. Every source
oracle launch explicitly requests `algorithm="horizontal"`; auto, simple,
vertical, and MTP paths are out of scope.

## Pipeline at a glance

The physical block is `(threadIdx.x,threadIdx.y)=(32,5)` for DIM 64 and
`(32,9)` for DIM 128. The last warp is the producer; every preceding warp is a
consumer. The axes stay independent. Both DIM values have exactly two lanes
per output row and 16 rows per consumer warp. Thus DIM 64 uses four consumer
warps and DIM 128 uses eight; the terminal reduction is one full-mask
shuffle-down by 16 in both cases.

Each state stage is `[DIM,STAGE_COLS]`, where `STAGE_COLS=32` for BF16/FP16
state and 16 for FP32 state. The stage always contains `DIM*64` bytes. The ring
has `min(4,DSTATE/STAGE_COLS)` stages. Each stage owns an empty/full barrier
pair with arrival count `1+32*CONSUMER_WARPS`; a separate B/C publication
barrier has count `32*CONSUMER_WARPS`.

| Physical warp | Role | Exact work | Publication/reuse edge |
| --- | --- | --- | --- |
| `0 .. CONSUMER_WARPS-1` | consumers | each owns 16 rows, two lanes per row; warp 0 first publishes B and warp 1 first publishes C | initially arrives on every empty barrier; B/C token rendezvous; full-token wait per serial stage; empty arrival after shared state writes |
| `CONSUMER_WARPS`, elected lane | producer | runtime TT/TF/FF state dispatch; unrolled fill, steady, and drain state-TMA loops | empty token wait; proxy fence before ring overwrite/store; full expect/bare arrival; store commit and read-wait |
| remaining producer lanes | inactive after leader election | no state transfer or arithmetic | no pipeline arrivals after initialization |

The producer's fill and drain trip counts are `NUM_STAGES`; steady has
`DSTATE/STAGE_COLS-NUM_STAGES` iterations. All three CUDA loops carry
`#pragma unroll`. The consumer DSTATE traversal deliberately has no unroll
pragma and remains a source/TIRx serial loop with loop-carried stage and column
indices. Normal O3 PTX retains that loop except for one physical-code-shape
caveat: the BF16-state/F32-weight/i64-index, DIM64/DSTATE64/PR0 padded helper is
fully expanded into two stage bodies. The matching cached helper retains its
backedge and folds its cursor update to `xor.b32 stage,1` plus
`mov.b32 iBegin,32`. Its item loop is fully unrolled in every shape.

CUDA spells `exp(A*dt)` in each source inner-element body. Normal
`-O3 --use_fast_math` SM100a PTX performs invariant CSE: each valid-state and
padded-state helper instance emits one `mul.ftz`, one LOG2E `mul.ftz`, and one
`ex2.approx.ftz` before its source-serial stage traversal. The padded helper
then emits one additional invariant `mul.ftz(dA,0)` and reuses that result as
every update FMA's addend; it is not replaced by a literal zero. This remains
before the two physically expanded stage bodies in the unique padded DS64
shape described above. This emitted placement, as well as the source mapping,
is explicit below.

## Primitive vocabulary

Structural operations do not move or compute data:

```python
specialize(...)              # compile-time dtype/shape/ratio variant
launch(...)                  # physical grid/block and shared-memory metadata
tensor_map(...)              # host-encoded rank-four CUtensorMap
raw_shared(...)              # one dynamic shared allocation
view(...)                    # typed storage view without a copy
reg_tile(...)                # lane-private registers
dispatch(...)                # runtime branch into a compile-time helper body
```

Copies always expose direction and completion:

```python
copy_g2r(src, dst=None, predicate=None)
copy_s2r(src, dst=None)
copy_r2s(src, dst)
copy_r2g(src, dst, predicate=None)
copy_tmap_g2s(tensor_map, coords, dst_shared, completion_barrier)
copy_tmap_s2g(src_shared, tensor_map, coords)
```

Synchronization and warp control are explicit:

```python
mbarrier_init(shared_barrier, arrival_count)
mbarrier_arrive(shared_barrier) -> token
mbarrier_arrive_expect_tx(shared_barrier, bytes) -> token
mbarrier_wait_token(shared_barrier, token)
fence_proxy_async_shared_cta()
bulk_commit_group()
bulk_wait_group_read_zero()
cta_sync()
elect_sync() -> predicate
shuffle_down(src, lane_delta, clamp, member_mask) -> (dst, pred)
```

The compute vocabulary is primitive:

```python
fill(dst, value)
cast(dtype, src, rounding=None)
add(lhs, rhs)
sub(lhs, rhs)
mul(lhs, rhs)
fma(lhs, rhs, acc)
exp2(src)
log2(src)
div(lhs, rhs)
bit_and(lhs, rhs)
bit_xor(lhs, rhs)
mul_hi_u32(lhs, rhs)
mul_lo_s32(lhs, rhs)
add_s32(lhs, rhs)
prmt_b32(lhs, rhs, selector)
```

Predicates, static indices, pointer choice, address expressions, and integer
column permutation are control operations. There is no compound `pipeline`,
`softplus`, `state_update`, `philox`, `silu`, or `reduce` primitive: every key
copy, compute, and synchronization instruction is expanded below.

## Complete sketch

```python
# ===========================================================================
# Static specializations, host descriptor, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    INPUT_DTYPE="bf16",
    STATE_DTYPE=("bf16", "f16", "f32"),
    WEIGHT_DTYPE=("f32", "bf16"),
    MATRIX_A_DTYPE="f32",
    INDEX_DTYPE=("i32", "i64"),
    DIM=(64, 128),
    DSTATE=(64, 96, 128, 256),
    HEADS_GROUP_RATIO=(1, 2, 4, 8, 16, 32, 64),
    PHILOX_ROUNDS=(0, 10),
    target="sm_100a",
)

host_assert(STATE_DTYPE in ("bf16", "f16", "f32"))
host_assert(PHILOX_ROUNDS == 0 or
            (PHILOX_ROUNDS == 10 and STATE_DTYPE == "f16"))
host_assert(nheads % ngroups == 0)
host_assert(nheads // ngroups == HEADS_GROUP_RATIO)
host_assert(HEADS_GROUP_RATIO in (1, 2, 4, 8, 16, 32, 64))

CONSUMER_WARPS = (DIM // 64) * 4
PRODUCER_WARPS = 1
NUM_WARPS = CONSUMER_WARPS + PRODUCER_WARPS
SECTOR_BYTES = 32
STAGE_COLS = 2 * SECTOR_BYTES // sizeof(STATE_DTYPE)
TOTAL_STAGES = DSTATE // STAGE_COLS
NUM_STAGES = min(4, TOTAL_STAGES)
STAGES_READ_ONLY = NUM_STAGES
STAGES_BOTH = TOTAL_STAGES - NUM_STAGES
STAGES_WRITE_ONLY = NUM_STAGES
STATE_STAGE_BYTES = DIM * STAGE_COLS * sizeof(STATE_DTYPE)
STATE_VALUES_PER_BANK = 4 // sizeof(STATE_DTYPE)
LANES_PER_ROW = CONSUMER_WARPS * 32 // DIM
ROWS_PER_WARP = 32 // LANES_PER_ROW
ITEMS_PER_THREAD = STAGE_COLS // LANES_PER_ROW

host_assert(DSTATE % STAGE_COLS == 0 and DSTATE >= STAGE_COLS)
host_assert(STAGE_COLS == (32 if sizeof(STATE_DTYPE) == 2 else 16))
host_assert(STATE_STAGE_BYTES == DIM * 64)
host_assert(LANES_PER_ROW == 2)
host_assert(ROWS_PER_WARP == 16)

state_map = tensor_map(
    base=state,
    rank=4,
    shape=(DSTATE, DIM, nheads, state_cache_size),
    element_strides=(1, DSTATE, DSTATE * DIM, state_stride_batch),
    box=(STAGE_COLS, DIM, 1, 1),
    element_stride=(1, 1, 1, 1),
    interleave="none",
    swizzle="none",
    l2_promotion="128B",
    oob_fill="none",
)
# instruction_selection: host CUtensorMap encoding consumed by rank-4
# cp.async.bulk.tensor instructions; extent: one by-value descriptor per launch

host_assert(address(state) % 128 == 0)
INPUT_VECTOR_BYTES = sizeof(PackedAligned(INPUT_DTYPE))
host_assert(INPUT_VECTOR_BYTES == 16)
host_assert(address(x) % INPUT_VECTOR_BYTES == 0)
host_assert(x_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
if z is present:
    host_assert(address(z) % INPUT_VECTOR_BYTES == 0)
    host_assert(z_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
host_assert(address(B) % INPUT_VECTOR_BYTES == 0)
host_assert(address(C) % INPUT_VECTOR_BYTES == 0)
host_assert(B_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
host_assert(C_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)

STATE_RING_BYTES = NUM_STAGES * STATE_STAGE_BYTES
B_BYTES = DSTATE * sizeof(INPUT_DTYPE)
C_BYTES = DSTATE * sizeof(INPUT_DTYPE)
OFF_STATE = 0
OFF_B = STATE_RING_BYTES
OFF_C = OFF_B + B_BYTES
OFF_EMPTY = OFF_C + C_BYTES
OFF_FULL = OFF_EMPTY + NUM_STAGES * 8
OFF_CONSUMERS = OFF_FULL + NUM_STAGES * 8
SHARED_BYTES_USED = OFF_CONSUMERS + 8
DYNAMIC_SMEM_BYTES = align_up(SHARED_BYTES_USED, 128)

launch_config = launch(
    grid=(batch, nheads, 1),
    block=(32, NUM_WARPS, 1),
    threads=32 * NUM_WARPS,
    dynamic_smem_bytes=DYNAMIC_SMEM_BYTES,
    max_dynamic_shared_memory_attribute=DYNAMIC_SMEM_BYTES,
)

def selective_state_update_stp_horizontal(
    state_map,
    state, x, dt, A, B, C, D, z, dt_bias,
    state_batch_indices, dst_state_batch_indices, rand_seed, output,
    state_stride_batch, x_stride_batch, dt_stride_batch,
    B_stride_batch, C_stride_batch, z_stride_batch, out_stride_batch,
    state_batch_indices_stride_batch,
    dst_state_batch_indices_stride_batch,
    nheads, ngroups, state_cache_size,
    dt_softplus, update_state, pad_slot_id,
):
    if PHILOX_ROUNDS > 0:
        random_seed = 0
        # instruction_selection: mov.b64 zero; extent: one nullable-seed fallback per thread
        if rand_seed is present:
            random_seed = copy_g2r(rand_seed[0])
            # instruction_selection: guarded ordinary ld.global.b64 after a pointer-null branch; extent: one scalar per thread. The entire seed expression is DCE for PHILOX_ROUNDS=0.

    batch_i = cta_id(axis="x", extent=batch)
    # instruction_selection: mov.u32 from %ctaid.x; extent: one physical CTA coordinate
    head = cta_id(axis="y", extent=nheads)
    # instruction_selection: mov.u32 from %ctaid.y; extent: one physical CTA coordinate
    raw_lane = thread_id(axis="x", extent=32)
    # instruction_selection: mov.u32 from %tid.x; extent: one independent physical x coordinate
    lane = bit_and(raw_lane, 31)
    # instruction_selection: and.b32 with 31; extent: one source-exact threadIdx.x % warpSize lane mask
    warp = thread_id(axis="y", extent=NUM_WARPS)
    # instruction_selection: mov.u32 from %tid.y; extent: one independent physical warp coordinate
    group_i = head // HEADS_GROUP_RATIO
    # instruction_selection: power-of-two integer division folded to identity or shift; extent: one head-group coordinate

    if state_batch_indices is present:
        raw_state_batch = copy_g2r(
            state_batch_indices[
                batch_i * state_batch_indices_stride_batch])
        # instruction_selection: i32 uses sign-extending ld.global.s32 directly into a b64 destination; i64 uses ld.global.b64; extent: one source slot per thread
        state_batch = cast("i64", raw_state_batch)
        # instruction_selection: logical i64 value only; there is no standalone cvt.s64.s32 after the i32 load, while i64 is identity
    else:
        state_batch = cast("i64", batch_i)
        # instruction_selection: cvt.u64.u32; extent: one fallback source slot

    if dst_state_batch_indices is present:
        raw_dst_batch = copy_g2r(
            dst_state_batch_indices[
                batch_i * dst_state_batch_indices_stride_batch])
        # instruction_selection: i32 uses ld.global.b32 and remains b32 in the reviewed PR0 shape; i64 uses ld.global.b64; extent: one destination slot per thread
        dst_state_batch = cast("i64", raw_dst_batch)
        # instruction_selection: logical i64 pseudo value only; there is no standalone cvt.s64.s32, and i64 is identity until a TMA coordinate narrows it
    else:
        dst_state_batch = state_batch

    state_ptr_offset = (
        state_batch * state_stride_batch + head * DIM * DSTATE)
    # instruction_selection: mul/add wide integer arithmetic; extent: one source-state element offset per thread, live only in Philox specializations

    # =======================================================================
    # Exact dynamic-shared C++ ABI layout
    # =======================================================================

    smem = raw_shared(DYNAMIC_SMEM_BYTES, alignment=128)
    sState = view(
        smem, offset=OFF_STATE, dtype=STATE_DTYPE,
        shape=(NUM_STAGES, DIM, STAGE_COLS), alignment=128)
    sB = view(
        smem, offset=OFF_B, dtype="bf16", shape=(DSTATE,), alignment=16)
    sC = view(
        smem, offset=OFF_C, dtype="bf16", shape=(DSTATE,), alignment=16)
    bar_empty = view(
        smem, offset=OFF_EMPTY, dtype="mbarrier.b64", shape=(NUM_STAGES,))
    bar_full = view(
        smem, offset=OFF_FULL, dtype="mbarrier.b64", shape=(NUM_STAGES,))
    bar_consumers = view(
        smem, offset=OFF_CONSUMERS, dtype="mbarrier.b64", shape=(1,))
    # OFF_B/OFF_C are already 128/16-byte aligned for the declared DSTATE
    # domain. The final align_up is the C++ struct's 128-byte tail padding.

    # =======================================================================
    # Barrier initialization by exact source warps
    # =======================================================================

    for init_stage in serial_range(warp, NUM_STAGES, NUM_WARPS):
        if lane == 0:
            mbarrier_init(
                bar_empty[init_stage], 1 + CONSUMER_WARPS * 32)
            # instruction_selection: mbarrier.init.shared.b64 count 129 for DIM64 or 257 for DIM128; extent: one empty barrier per stage
            mbarrier_init(
                bar_full[init_stage], 1 + CONSUMER_WARPS * 32)
            # instruction_selection: mbarrier.init.shared.b64 with the same count; extent: one full barrier per stage
            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta; extent: one barrier-pair publication per initializing lane. Because NUM_STAGES<NUM_WARPS, optimized PTX predicates warp<NUM_STAGES and has no loop backedge.
    if warp == 0 and lane == 0:
        mbarrier_init(bar_consumers[0], CONSUMER_WARPS * 32)
        # instruction_selection: mbarrier.init.shared.b64 count 128 for DIM64 or 256 for DIM128; extent: one B/C publication barrier
    cta_sync()
    # instruction_selection: bar.sync 0; extent: all 160 or 288 physical threads after initialization

    # =======================================================================
    # Producer warp: elected lane, runtime TT/TF/FF dispatch
    # =======================================================================

    if warp == CONSUMER_WARPS:
        read_state = state_batch != pad_slot_id
        write_state = read_state and update_state
        if elect_sync():
            # instruction_selection: activemask.b32 plus elect.sync and predicated leader materialization; extent: one elected lane in the producer warp
            dispatch producer_program(
                READ_STATE=(True if read_state else False),
                WRITE_STATE=(True if write_state else False),
            ):
                # READ_STATE/WRITE_STATE instances are exactly TT, TF, and FF;
                # FT is unreachable and never instantiated.

                # -----------------------------------------------------------
                # Phase 1: read-only pipeline fill, source pragma-unrolled
                # -----------------------------------------------------------
                for fill_iter in static_range(STAGES_READ_ONLY):
                    stage = fill_iter % NUM_STAGES
                    i_read = fill_iter * STAGE_COLS
                    token_empty = mbarrier_arrive(bar_empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning b64 token; extent: one elected-producer arrival per fill stage
                    mbarrier_wait_token(bar_empty[stage], token_empty)
                    # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 on that exact token; extent: wait for every consumer's initial empty arrival

                    if READ_STATE:
                        copy_tmap_g2s(
                            state_map,
                            coords=(i_read, 0, head, state_batch),
                            dst_shared=sState[stage],
                            completion_barrier=bar_full[stage])
                        # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes; extent: one full-DIM column tile of STATE_STAGE_BYTES. Its 32-bit source-slot coordinate uses cvt.u32.u64 for the logical i64 source/fallback value.
                        mbarrier_arrive_expect_tx(
                            bar_full[stage], STATE_STAGE_BYTES)
                        # instruction_selection: mbarrier.arrive.expect_tx.release.cta.shared::cta.b64; extent: exact DIM*64 transaction bytes per fill stage
                    else:
                        mbarrier_arrive(bar_full[stage])
                        # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: one bare full arrival per padded fill stage, with no expected transaction

                # -----------------------------------------------------------
                # Phase 2: store-before-load steady state, pragma-unrolled
                # -----------------------------------------------------------
                for steady_iter in static_range(STAGES_BOTH):
                    stage = (STAGES_READ_ONLY + steady_iter) % NUM_STAGES
                    i_read = (STAGES_READ_ONLY + steady_iter) * STAGE_COLS
                    i_write = steady_iter * STAGE_COLS
                    token_empty = mbarrier_arrive(bar_empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning b64 token; extent: one producer arrival per reused stage
                    mbarrier_wait_token(bar_empty[stage], token_empty)
                    # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 on the token; extent: wait until every consumer has completed this stage

                    if READ_STATE or WRITE_STATE:
                        fence_proxy_async_shared_cta()
                        # instruction_selection: fence.proxy.async.shared::cta; extent: one consumer-shared-to-async reuse edge in TT and TF, absent in FF
                        if WRITE_STATE:
                            copy_tmap_s2g(
                                sState[stage], state_map,
                                coords=(i_write, 0, head, dst_state_batch))
                            # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group; extent: one completed state tile to the destination slot. A present i32 destination pointer loads an already-b32 coordinate and emits no conversion. A null i32 destination pointer falls back to logical-b64 state_batch and therefore emits cvt.u32.u64; present/fallback i64 paths likewise narrow when forming the TMA b32 coordinate.
                            bulk_commit_group()
                            # instruction_selection: cp.async.bulk.commit_group; extent: one state-store group
                            bulk_wait_group_read_zero()
                            # instruction_selection: cp.async.bulk.wait_group.read 0; extent: store has finished reading shared before any next G2S overwrite

                        if READ_STATE:
                            copy_tmap_g2s(
                                state_map,
                                coords=(i_read, 0, head, state_batch),
                                dst_shared=sState[stage],
                                completion_barrier=bar_full[stage])
                            # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes; extent: one next full-DIM state tile. Its 32-bit source-slot coordinate uses cvt.u32.u64 for the logical i64 source/fallback value.
                            mbarrier_arrive_expect_tx(
                                bar_full[stage], STATE_STAGE_BYTES)
                            # instruction_selection: mbarrier.arrive.expect_tx.release.cta.shared::cta.b64; extent: exact next-tile bytes
                        else:
                            mbarrier_arrive(bar_full[stage])
                            # instruction_selection: mbarrier.arrive.shared::cta.b64; extent: one full phase without a next load
                    else:
                        mbarrier_arrive(bar_full[stage])
                        # instruction_selection: mbarrier.arrive.shared::cta.b64; extent: one FF full phase, with no proxy fence or transfer

                # -----------------------------------------------------------
                # Phase 3: write-only drain, source pragma-unrolled
                # -----------------------------------------------------------
                for drain_iter in static_range(STAGES_WRITE_ONLY):
                    stage = (
                        STAGES_READ_ONLY + STAGES_BOTH + drain_iter
                    ) % NUM_STAGES
                    i_write = (STAGES_BOTH + drain_iter) * STAGE_COLS
                    token_empty = mbarrier_arrive(bar_empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64 returning b64 token; extent: one producer arrival per drain stage
                    mbarrier_wait_token(bar_empty[stage], token_empty)
                    # instruction_selection: polling mbarrier.try_wait.shared::cta.b64; extent: one final consumer-completion wait per drain stage
                    if WRITE_STATE:
                        fence_proxy_async_shared_cta()
                        # instruction_selection: fence.proxy.async.shared::cta; extent: one final consumer-shared-to-async publication edge
                        copy_tmap_s2g(
                            sState[stage], state_map,
                            coords=(i_write, 0, head, dst_state_batch))
                        # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group; extent: one final state tile. A present i32 destination pointer loads an already-b32 coordinate and emits no conversion. A null i32 destination pointer falls back to logical-b64 state_batch and therefore emits cvt.u32.u64; present/fallback i64 paths likewise narrow when forming the TMA b32 coordinate.
                        bulk_commit_group()
                        # instruction_selection: cp.async.bulk.commit_group; extent: one final state-store group
                        bulk_wait_group_read_zero()
                        # instruction_selection: cp.async.bulk.wait_group.read 0; extent: one final shared-read completion

    # =======================================================================
    # Consumer warps: initial empty arrivals, scalar setup, and B/C publication
    # =======================================================================

    else:
        for initial_stage in static_range(NUM_STAGES):
            mbarrier_arrive(bar_empty[initial_stage])
            # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: one initial arrival from every consumer thread to every stage

        A_raw = copy_g2r(A[head])
        # instruction_selection: ordinary ld.global.b32; extent: one A scalar per consumer thread
        A_value = cast("f32", A_raw)
        # instruction_selection: identity f32; extent: one scalar

        d_value = 0.0
        # instruction_selection: runtime nullable-pointer fallback selects a zero multiplier when D is absent; extent: one D scalar
        if D is present:
            D_raw = copy_g2r(D[head])
            # instruction_selection: ordinary ld.global.b32 or ld.global.b16; extent: one D scalar
            d_value = cast("f32", D_raw)
            # instruction_selection: identity f32 or cvt.f32.bf16; extent: one D scalar

        dt_raw = copy_g2r(dt[batch_i * dt_stride_batch + head])
        # instruction_selection: ordinary ld.global.b32 or ld.global.b16; extent: one dt scalar
        dt_value = cast("f32", dt_raw)
        # instruction_selection: identity f32 or cvt.f32.bf16; extent: one dt scalar
        if dt_bias is present:
            bias_raw = copy_g2r(dt_bias[head])
            # instruction_selection: ordinary ld.global.b32 or ld.global.b16; extent: one bias scalar
            bias_value = cast("f32", bias_raw)
            # instruction_selection: identity f32 or cvt.f32.bf16; extent: one bias scalar
            dt_value = add(dt_value, bias_value)
            # instruction_selection: add.ftz.f32; extent: one optional-bias scalar
        if dt_softplus:
            if dt_value <= 20.0:
                dt_exp_arg = mul(dt_value, LOG2_E)
                # instruction_selection: mul.ftz.f32; extent: one scalar on the <=20 branch
                dt_exp = exp2(dt_exp_arg)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar on the <=20 branch
                dt_one_plus = add(1.0, dt_exp)
                # instruction_selection: add.ftz.f32; extent: one scalar on the <=20 branch
                dt_log2 = log2(dt_one_plus)
                # instruction_selection: lg2.approx.ftz.f32; extent: one scalar on the <=20 branch
                dt_value = mul(dt_log2, LN2)
                # instruction_selection: mul.ftz.f32; extent: one scalar on the <=20 branch
            # instruction_selection: setp.gtu.ftz.f32 plus branch around the complete exp/log chain; dt>20 preserves its original value

        if warp == 0:
            for b_col in serial_range(
                    lane * 8, DSTATE, 32 * 8):
                b_vec = copy_g2r(
                    B[
                        batch_i * B_stride_batch +
                        group_i * DSTATE + b_col : b_col + 8])
                # instruction_selection: ld.global.v4.b32; extent: one aligned 16-byte vector for each participating warp-0 lane, at most one iteration in the DSTATE<=256 domain
                copy_r2s(b_vec, sB[b_col : b_col + 8])
                # instruction_selection: st.shared.v4.b32; extent: one aligned 16-byte B vector after its global load
        elif warp == 1:
            for c_col in serial_range(
                    lane * 8, DSTATE, 32 * 8):
                c_vec = copy_g2r(
                    C[
                        batch_i * C_stride_batch +
                        group_i * DSTATE + c_col : c_col + 8])
                # instruction_selection: ld.global.v4.b32; extent: one aligned 16-byte vector for each participating warp-1 lane, at most one iteration in the DSTATE<=256 domain
                copy_r2s(c_vec, sC[c_col : c_col + 8])
                # instruction_selection: st.shared.v4.b32; extent: one aligned 16-byte C vector after its global load

        row_group = lane % ROWS_PER_WARP
        member = lane // ROWS_PER_WARP
        d = warp * ROWS_PER_WARP + row_group
        # instruction_selection: and/shift/add integer mapping from independent %tid.x/%tid.y; extent: one row and one member per consumer thread

        x_raw = copy_g2r(
            x[batch_i * x_stride_batch + head * DIM + d])
        # instruction_selection: ordinary ld.global.b16; extent: one row input per consumer thread
        x_value = cast("f32", x_raw)
        # instruction_selection: cvt.f32.bf16; extent: one row input
        z_value = 0.0
        # instruction_selection: mov.b32 zero fallback; extent: one optional gate scalar
        if z is present:
            z_raw = copy_g2r(
                z[batch_i * z_stride_batch + head * DIM + d])
            # instruction_selection: ordinary ld.global.b16; extent: one row gate per consumer thread
            z_value = cast("f32", z_raw)
            # instruction_selection: cvt.f32.bf16; extent: one row gate

        token_consumers = mbarrier_arrive(bar_consumers[0])
        # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning b64 token; extent: one B/C-publication arrival per consumer thread
        mbarrier_wait_token(bar_consumers[0], token_consumers)
        # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 on that exact token; extent: all consumers wait for warp-0 B and warp-1 C stores

        out_value = 0.0
        # instruction_selection: mov.b32 zero accumulator; extent: one lane partial
        dispatch consumer_program(
            USE_STATE_CACHE=(state_batch != pad_slot_id),
        ):
            # The next three operations map source lines 963/1008. Although
            # CUDA spells them in every element body, normal O3 PTX CSEs the
            # invariant sequence once here, separately in each helper instance
            # and before its source/TIRx serial traversal. In the unique padded
            # DS64 physical-unroll shape, it precedes the two expanded bodies.
            a_dt = mul(A_value, dt_value)
            # instruction_selection: mul.ftz.f32; optimized extent: one scalar before the source-serial stage traversal
            a_dt_exp_arg = mul(a_dt, LOG2_E)
            # instruction_selection: mul.ftz.f32; optimized extent: one scalar before the source-serial stage traversal
            dA = exp2(a_dt_exp_arg)
            # instruction_selection: ex2.approx.ftz.f32; optimized extent: one scalar before the source-serial stage traversal
            if not USE_STATE_CACHE:
                padded_state_dA = mul(dA, 0.0)
                # instruction_selection: mul.ftz.f32; optimized extent: one invariant padded-state product before the source-serial traversal (or before the unique DS64 helper's two physically expanded bodies). It remains live to preserve 0*Inf/NaN behavior and is reused by every update FMA.

            i_begin = 0
            stage = 0
            random_words = reg_tile("u32", [4])
            while i_begin < DSTATE:
                token_full = mbarrier_arrive(bar_full[stage])
                # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning b64 token; extent: one arrival per consumer thread/stage
                mbarrier_wait_token(bar_full[stage], token_full)
                # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 on that token; extent: wait for producer and any G2S transaction

                for item in static_range(
                        0, ITEMS_PER_THREAD, STATE_VALUES_PER_BANK):
                    base_col = item + member * ITEMS_PER_THREAD
                    sequence_index = row_group * STAGE_COLS + base_col
                    bank_cycle = (
                        sequence_index // STATE_VALUES_PER_BANK) // 32
                    ii = (
                        base_col + STATE_VALUES_PER_BANK * bank_cycle
                    ) % STAGE_COLS
                    i = i_begin + ii
                    # instruction_selection: compile-time-sized integer add/shift/and/or permutation implementing conflict_free_column; extent: one permuted bank column per unrolled item

                    state_word = copy_s2r(
                        word_view(
                            sState[
                                stage, d,
                                ii : ii + STATE_VALUES_PER_BANK],
                            "u32"))
                    # instruction_selection: ld.shared.v2.b16 for BF16/FP16 or ld.shared.b32 for FP32; extent: one old state bank word in USE_STATE_CACHE=true. The load is DCE in the padded helper because the old state is never consumed.
                    r_state = view(
                        state_word, dtype=STATE_DTYPE,
                        shape=(STATE_VALUES_PER_BANK,))

                    if sizeof(STATE_DTYPE) == sizeof(INPUT_DTYPE):
                        b_word = copy_s2r(
                            word_view(
                                sB[i : i + STATE_VALUES_PER_BANK], "u32"))
                        # instruction_selection: ld.shared.v2.b16; extent: one BF16 B pair, issued after the live state-word load
                        c_word = copy_s2r(
                            word_view(
                                sC[i : i + STATE_VALUES_PER_BANK], "u32"))
                        # instruction_selection: ld.shared.v2.b16; extent: one BF16 C pair, issued after the B-pair load
                        r_B = view(b_word, dtype="bf16", shape=(2,))
                        r_C = view(c_word, dtype="bf16", shape=(2,))

                    flat_item = item
                    if PHILOX_ROUNDS > 0 and flat_item % 4 == 0:
                        random_offset = (
                            state_ptr_offset + d * DSTATE + i)
                        c0 = low_u32(random_offset)
                        c1 = high_u32(random_offset)
                        c2 = 0
                        c3 = 0
                        k0 = low_u32(random_seed)
                        k1 = high_u32(random_seed)
                        for philox_i in static_range(10):
                            old_c0 = c0
                            old_c2 = c2
                            c0_hi = mul_hi_u32(0xCD9E8D57, old_c2)
                            # instruction_selection: mul.hi.u32; logical extent: one source dependency per round. The round-0 PHILOX_ROUND_B*0 instruction is physically retained and CSE-shared by the four refreshes in one stage body, not constant-folded.
                            c0_xor_c1 = bit_xor(c0_hi, c1)
                            # instruction_selection: xor.b32; logical extent: one source dependency per round. The round-0 XOR consuming the retained c0_hi also remains; the corresponding next-round XOR folds only because round-0 c1 comes from a folded zero low-multiply.
                            next_c0 = bit_xor(c0_xor_c1, k0)
                            # instruction_selection: xor.b32; logical extent: one source dependency per round
                            c2_hi = mul_hi_u32(0xD2511F53, old_c0)
                            # instruction_selection: mul.hi.u32; logical extent: one source dependency per round
                            c2_xor_c3 = bit_xor(c2_hi, c3)
                            # instruction_selection: xor.b32; logical extent: one source dependency per round, with round-0 c3==0 folded
                            next_c2 = bit_xor(c2_xor_c3, k1)
                            # instruction_selection: xor.b32; logical extent: one source dependency per round
                            next_c1 = mul_lo_s32(old_c2, 0xCD9E8D57)
                            # instruction_selection: mul.lo.s32; logical extent: one source dependency per round, with round-0 old_c2==0 folded to c1=0; that zero also removes the corresponding XOR in the next round
                            next_c3 = mul_lo_s32(old_c0, 0xD2511F53)
                            # instruction_selection: mul.lo.s32; logical extent: one source dependency per round
                            next_k0 = add_s32(k0, 0x9E3779B9)
                            # instruction_selection: add.s32; logical extent: one key update per round. Normal PSR64/PSR128 PTX precomputes all nine consumed cumulative low-key values before the serial-stage loop and DCEs the unused final update; final low-13-bit liveness folds the round-9 low-key operand to immediate 1921.
                            next_k1 = add_s32(k1, 0xBB67AE85)
                            # instruction_selection: add.s32; logical extent: one key update per round. Cached PTX computes nine consumed high-key values once per stage iteration and shares them across its four refreshes. Padded PTX keeps seven there but hoists round-8 add(high_seed,-616729560) and the final low-13-bit add(high_seed,685) before the serial-stage loop; the unused final update is DCE.
                            c0, c1, c2, c3 = (
                                next_c0, next_c1, next_c2, next_c3)
                            k0, k1 = next_k0, next_k1
                        random_words = (c0, c1, c2, c3)
                        # instruction_selection optimized-extent rule: one stage body has four logical refreshes and physically issues 77 mul.hi.u32, 76 mul.lo.s32, and 152 xor.b32 in both PSR64/PSR128 cached/padded helpers. The retained mul.hi(B,0) is CSE-shared once across the four calls; only mul.lo(B,0), the round-0 c3-zero XOR, and the next-round XOR fed by c1=0 fold. Final-key updates DCE. Final low-13-bit liveness folds operands to low-key 1921, high-key 685, and low-multiply immediates 3415/8019. Final c0/c1 and c2/c3 all remain live because the current bank uses outputs 0/1 and the next bank reuses outputs 2/3.

                    raw_sr_words = reg_tile(
                        "u32", [STATE_VALUES_PER_BANK])
                    for e in static_range(STATE_VALUES_PER_BANK):
                        if USE_STATE_CACHE:
                            state_value = cast("f32", r_state[e])
                            # instruction_selection: cvt.f32.bf16, cvt.f32.f16, or identity f32; extent: one live state element
                        else:
                            state_value = 0.0
                            # instruction_selection: compile-specialized zero; old-state load/conversion are DCE

                        if sizeof(STATE_DTYPE) == sizeof(INPUT_DTYPE):
                            B_value = cast("f32", r_B[e])
                            # instruction_selection: cvt.f32.bf16; extent: one element from the already-loaded B pair
                            C_value = cast("f32", r_C[e])
                            # instruction_selection: cvt.f32.bf16; extent: one element from the already-loaded C pair
                        else:
                            B_raw = copy_s2r(sB[i + e])
                            # instruction_selection: ld.shared.b16; extent: one scalar B element after the state load/conversion
                            B_value = cast("f32", B_raw)
                            # instruction_selection: cvt.f32.bf16; extent: one B scalar before C load
                            C_raw = copy_s2r(sC[i + e])
                            # instruction_selection: ld.shared.b16; extent: one scalar C element after B conversion
                            C_value = cast("f32", C_raw)
                            # instruction_selection: cvt.f32.bf16; extent: one C scalar before arithmetic

                        dB = mul(B_value, dt_value)
                        # instruction_selection: mul.ftz.f32; extent: one scalar
                        if USE_STATE_CACHE:
                            state_dA = mul(state_value, dA)
                            # instruction_selection: mul.ftz.f32; extent: one live old-state contribution
                        else:
                            state_dA = padded_state_dA
                            # instruction_selection: reuse of the single loop-external mul.ftz(dA,0) result; there is no per-element old-state load or multiply
                        new_state = fma(x_value, dB, state_dA)
                        # instruction_selection: fma.rn.ftz.f32; extent: one state update, retaining the loop-external padded product as addend in USE_STATE_CACHE=false

                        if PHILOX_ROUNDS > 0:
                            random13 = bit_and(
                                random_words[(flat_item + e) % 4], 0x1FFF)
                            # instruction_selection: and.b32; extent: one 13-bit random operand per state element
                            raw_sr_words[e] = cast(
                                "raw_f16x2_u32", new_state,
                                rounding=(
                                    "stochastic", random13,
                                    "zero_second_f32"))
                            # instruction_selection: cvt.rs.f16x2.f32 producing a full b32 whose low half is the desired scalar FP16 and whose high half converts a zero dummy; extent: one per element
                        else:
                            r_state[e] = cast(
                                STATE_DTYPE, new_state, rounding="rn")
                            # instruction_selection: cvt.rn.bf16.f32, cvt.rn.f16.f32, or identity f32; extent: one state element

                        out_value = fma(
                            C_value, new_state, out_value)
                        # instruction_selection: fma.rn.ftz.f32; extent: one serial lane accumulation after state conversion

                    if PHILOX_ROUNDS > 0:
                        packed_state = prmt_b32(
                            raw_sr_words[0], raw_sr_words[1],
                            selector=0x5410)
                        # instruction_selection: prmt.b32 selector 0x5410; extent: one packed stochastic-FP16 pair
                        copy_r2s(
                            packed_state,
                            word_view(
                                sState[
                                    stage, d,
                                    ii : ii + STATE_VALUES_PER_BANK],
                                "u32"))
                        # instruction_selection: st.shared.b32; extent: one packed stochastic-FP16 bank word
                    elif sizeof(STATE_DTYPE) == 2:
                        copy_r2s(
                            r_state[0:2],
                            sState[stage, d, ii : ii + 2])
                        # instruction_selection: st.shared.v2.b16; extent: one ordinary BF16/FP16 state pair
                    else:
                        copy_r2s(
                            r_state[0], sState[stage, d, ii])
                        # instruction_selection: st.shared.b32; extent: one FP32 state value

                mbarrier_arrive(bar_empty[stage])
                # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: one completion arrival per consumer thread/stage. There is deliberately no consumer-side async-proxy fence.
                i_begin = i_begin + STAGE_COLS
                stage = (stage + 1) % NUM_STAGES
                # instruction_selection: normally an optimized cursor update plus a real loop backedge; retained DSTATE64 loops fold this to xor.b32 stage,1 plus mov.b32 iBegin,32. Only the BF16-state/F32-weight/i64-index DIM64/DSTATE64/PR0 padded helper physically expands both stages and emits neither this update nor a backedge. The source/TIRx while remains unchanged.

        # ===================================================================
        # Two-lane row reduction and source-order output epilogue
        # ===================================================================

        peer_value, peer_pred = shuffle_down(
            out_value, lane_delta=16, clamp=31,
            member_mask=0xFFFFFFFF)
        # instruction_selection: shfl.sync.down.b32 dst|pred, src, 16, 31, -1; extent: one full-mask shuffle per consumer lane, returned predicate unused
        out_value = add(out_value, peer_value)
        # instruction_selection: add.ftz.f32; extent: one two-member row reduction
        # The source's lanesPerRow==4 shuffle-by-8 branch is compile-time false
        # for both DIM64 and DIM128. Horizontal emits no standalone warp_sync.

        if member == 0:
            out_value = fma(d_value, x_value, out_value)
            # instruction_selection: one unconditional fma.rn.ftz.f32 on every runtime nullable-D branch; D=null supplies a zero multiplier but does not fold or remove the FMA; extent: one member-zero row epilogue

            if z is present:
                neg_z = sub(0.0, z_value)
                # instruction_selection: sub.ftz.f32; extent: one row gate
                z_exp_arg = mul(neg_z, LOG2_E)
                # instruction_selection: mul.ftz.f32; extent: one row gate
                exp_neg_z = exp2(z_exp_arg)
                # instruction_selection: ex2.approx.ftz.f32; extent: one row gate
                denominator = add(1.0, exp_neg_z)
                # instruction_selection: add.ftz.f32; extent: one row gate
                sigmoid_z = div(1.0, denominator)
                # instruction_selection: div.approx.ftz.f32; extent: one row gate
                silu_z = mul(z_value, sigmoid_z)
                # instruction_selection: mul.ftz.f32; extent: one row gate
                out_value = mul(out_value, silu_z)
                # instruction_selection: mul.ftz.f32; extent: one gated output

            output_value = cast("bf16", out_value, rounding="rn")
            # instruction_selection: cvt.rn.bf16.f32; extent: one output scalar
            copy_r2g(
                output_value,
                output[
                    batch_i * out_stride_batch + head * DIM + d])
            # instruction_selection: st.global.b16; extent: one member-zero output
```

## Static specialization and launch boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| DIM, DSTATE, state/weight/index dtype, Philox rounds, group ratio | static | exact CTA size, stage width/count, unrolled producer/item loops, address types, conversion family, and group division |
| consumer-warps | derived static 4 or 8 | block is 5 or 9 physical warps; arrival counts are 129/128 or 257/256 |
| stage count | derived static 2-4 | exact dynamic layout and producer unrolled trip counts; source/TIRx consumer traversal stays serial. Normal O3 retains its backedge except for the one padded BF16/F32-weight/i64 DIM64/DSTATE64/PR0 helper, whose two stages are physically expanded |
| batch and nheads | static for each TIRx launch | exactly one CTA per batch/head |
| source/destination slots, update, pad, z/D/bias/seed pointers, softplus | runtime or host-specialized pointer/boolean state | preserves ordinary scalar branches, TT/TF/FF producer helpers, and cached/padded consumer helpers |

The host builds the same rank-four descriptor over the mutable state base with
element shapes `(DSTATE,DIM,nheads,cache)`, byte strides derived from
`(1,DSTATE,DSTATE*DIM,state_stride_batch)`, and box
`(STAGE_COLS,DIM,1,1)`. It sets `cudaFuncAttributeMaxDynamicSharedMemorySize`
to the 128-byte-tail-padded C++ struct size before launch. Horizontal rejects
scaled int16 state before descriptor construction.

## TIRx module and benchmark contract

- `KERNEL_META` names `selective_state_update_stp_horizontal`, category
  `flashinfer`, compute capability 10.
- `CONFIGS` contains 35 correctness rows and `BENCH_CONFIGS` contains 26 timed
  rows. They cover DIM 64/128; DSTATE 64/96/128/256; BF16/FP16/FP32 state;
  FP32/BF16 weight; every ratio; z/D/bias/softplus/update/pad branches;
  absent/i32/i64/rank-one/rank-two source and destination indices; strided
  state; output allocation modes; and FP16 Philox-10 at DSTATE 64/128.
- Every reference invocation uses the frozen FlashInfer horizontal oracle and
  independently allocated mutable state/output. Descriptor construction,
  source JIT, allocation, correctness preflight, and warmup stay outside timed
  closures.
- `bench_suite` is the only performance authority. Every final timed row must
  satisfy `source_time / tirx_time > 0.99`; no aggregate can hide a failing
  row.
- The executable implementation must use plain TIRx directional copies,
  mbarriers, rank-four TMA, proxy fence, elect, shuffle, and scalar operations
  corresponding to this sketch; it must not introduce a tile primitive or a
  different algorithm.

## Instruction-selection summary

- Empty/full barriers have counts 129 for DIM64 and 257 for DIM128; the B/C
  barrier has count 128 or 256. Every source wait is token-based
  `mbarrier.arrive.shared::cta.b64` plus polling
  `mbarrier.try_wait.shared::cta.b64`, never a parity-only wait.
- Producer fill/steady/drain loops are fully unrolled. G2S state uses
  `cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes`
  followed by exact-byte `mbarrier.arrive.expect_tx`. S2G state uses
  `cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group`, commit, and
  `wait_group.read 0`. Steady performs store/read-wait before the next load.
- Warp 0 and warp 1 publish B and C through paired 16-byte
  `ld.global.v4.b32` then `st.shared.v4.b32` operations before the all-consumer
  barrier. There is no bulk-copy proxy for B/C.
- Every row has two lanes. A 16-bit-state item issues one
  `ld.shared.v2.b16` state pair, then B pair, then C pair; ordinary stores use
  `st.shared.v2.b16`. FP32 state uses `ld/st.shared.b32` and scalar
  `ld.shared.b16` B/C operations. Padded helpers DCE old-state reads but still
  compute and write the discarded shared state tile.
- Normal O3 PTX emits one invariant A*dt exponential per cached/padded helper
  before its source-serial stage traversal. The padded helper also emits one
  invariant `dA*0` multiply and reuses it as the update-FMA addend. The unique
  BF16/F32-weight/i64 DIM64/DSTATE64/PR0 padded helper physically expands its
  two stages after those invariants; all other reviewed helpers retain a
  backedge, with retained DSTATE64 cursor updates folded to `xor stage,1` plus
  `mov iBegin,32`. Each element emits dB multiply, cached-state dA multiply
  when applicable, update FMA, state conversion, and output FMA in source
  order.
- Horizontal Philox refreshes once per four logical elements. All four final
  outputs are consumed across two consecutive packed state words. Each stage
  body's four refreshes physically issue `77 mul.hi / 76 mul.lo / 152 xor`:
  round-0 `mul.hi(B,0)` remains and is CSE-shared across the four calls, while
  `mul.lo(B,0)`, the c3-zero XOR, and its next-round zero dependency fold.
  All nine consumed low-key values are loop-external. Cached helpers compute
  nine high-key values once per stage iteration and share them across four
  calls; padded helpers keep seven there and hoist `-616729560` plus final
  low-13-bit `685` before the loop. Final low-13-bit liveness also folds the
  operands to `1921/685/3415/8019`, and unused final key updates DCE. Each
  element uses `cvt.rs.f16x2.f32` with a dummy zero; each pair uses
  `prmt.b32 0x5410` and `st.shared.b32`.
- Reduction is exactly one `shfl.sync.down.b32` with delta 16, clamp 31, mask
  -1, and unused predicate, followed by one `add.ftz.f32`. There is no
  shuffle-by-8 and no standalone warp synchronization in the reviewed domain.
- Member zero always performs one `fma.rn.ftz.f32` after the runtime nullable-D
  select; D=null supplies zero but does not remove the FMA. It then performs
  the optional explicit fast-math SILU chain, BF16 conversion, and one
  `st.global.b16` output. Horizontal has no separate state-scale epilogue.
