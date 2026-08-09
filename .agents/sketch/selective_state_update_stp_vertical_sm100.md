<!--
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2026 The TIRX Authors.

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

# Selective-state-update STP vertical SM100: coarse WASP execution sketch

This is a non-executable sketch of FlashInfer's CUDA
`selective_state_update_kernel_producer_consumer_vertical`. It records the
exact one-producer/four-consumer warp split, three-stage TMA state ring,
mbarrier protocol, shared bank-word traversal, reduction order, optional state
paths, and output publication that the TIRx port must preserve. The target
module is
[`tirx_kernels/flashinfer/selective_state_update_stp_vertical.py`](../../tirx_kernels/flashinfer/selective_state_update_stp_vertical.py),
which becomes the source of truth only after this sketch passes independent
review.

The target is SM100/B200. Input, B, C, z and output are BF16. State is BF16,
FP16, FP32, or int16 with FP32 row scales; weight/dt/D/dt_bias is FP32 or BF16;
A is FP32; indices are int32 or int64. `DIM in {64,128}`,
`DSTATE in {64,96,128,256}`, and `PHILOX_ROUNDS in {0,10}` are in scope.
The selected int16 matrix uses DSTATE 64/128/256, and stochastic FP16 uses
DSTATE 64/128. The simple, horizontal, MTP, and automatic algorithm-selection
paths are out of scope. Every source-oracle launch explicitly requests
`algorithm="vertical"`.

## Pipeline at a glance

The physical block is `(threadIdx.x,threadIdx.y)=(32,5)`. Warps 0-3 are
consumers and warp 4 is the producer; the two physical thread axes are not
flattened. Shared state is a three-stage ring of `[16,DSTATE]` tiles. Each stage
has one `bar_empty` and one `bar_full`, both initialized with arrival count 129:
128 consumer threads plus the single producer lane. A separate barrier with
arrival count 128 joins the consumers before the epilogue.

| Warp | Role | Exact work | Publication/reuse edge |
| --- | --- | --- | --- |
| 0 | consumer | rows `dd=0,4,8,12` in each 16-row stage | arrives every empty barrier initially; full-token wait; async-proxy fence then empty arrival; consumer-completion token wait |
| 1 | consumer | rows `dd=1,5,9,13` | same |
| 2 | consumer | rows `dd=2,6,10,14` | same |
| 3 | consumer | rows `dd=3,7,11,15` | same |
| 4 lane 0 | producer | stage-0 vector copies, state TMA fill/steady/drain and runtime state/z dispatch | empty-token wait; full expect/arrival; store commit/read-wait before ring reuse |
| 4 lanes 1-31 | inactive after initialization barrier | no producer copy or compute | no arrivals after initialization |

The producer executes three source-order phases. Phase 1 fills stages 0, 1 and
2. Phase 2 has `DIM/16-3` store-before-load steady iterations: one iteration for
DIM 64 and five for DIM 128. Phase 3 drains three final stores. Stage 0 alone
also carries x/B/C and optional z/scale bulk-copy transaction bytes.

For int16 state, each consumer row retains its new FP32 state values in
registers, reduces the absolute maximum, broadcasts lane 0, quantizes in a
second pass, and writes packed int16 words into the current shared TMA tile.
The full-mask lane-0 broadcast is the synchronization operation represented by
the source `__shfl_sync`; vertical has no separate warp barrier at that edge.

## Primitive vocabulary

Structural operations do not move or compute data:

```python
specialize(...)              # static dtype, DIM, DSTATE and Philox variant
launch(...)                  # grid/block and dynamic-shared metadata
tensor_map(...)              # host-encoded CUtensorMap descriptor
raw_shared(...)              # one dynamic shared allocation
view(...)                    # typed view without a copy
reg_tile(...)                # lane-private scalar/vector registers
```

Copies always name their storage direction and completion mechanism:

```python
copy_g2r(src, dst, predicate=None)
copy_s2r(src, dst)
copy_r2s(src, dst)
copy_r2g(src, dst, predicate=None)
copy_g2s_bulk(src, dst, bytes, completion_barrier)
copy_tmap_g2s(tensor_map, coords, dst_shared, completion_barrier)
copy_tmap_s2g(src_shared, tensor_map, coords)
```

Synchronization primitives are explicit:

```python
mbarrier_init(shared_barrier, arrival_count)
mbarrier_arrive(shared_barrier) -> token
mbarrier_arrive_expect_tx(shared_barrier, bytes) -> token
mbarrier_wait_token(shared_barrier, token)
fence_proxy_async_shared_cta()
bulk_commit_group()
bulk_wait_group_read_zero()
cta_sync()
```

The computation vocabulary is deliberately primitive:

```python
fill(dst, value)
cast(dst, src, rounding=None)
add(dst, lhs, rhs)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
exp2(dst, src)
log2(dst, src)
div(dst, lhs, rhs)
abs(dst, src)
min(dst, lhs, rhs)
max(dst, lhs, rhs)
select(dst, predicate, true_value, false_value)
bit_and(dst, lhs, rhs)
bit_xor(dst, lhs, rhs)
mul_hi_u32(dst, lhs, rhs)
mul_lo_s32(dst, lhs, rhs)
add_s32(dst, lhs, rhs)
prmt_b32(dst, lhs, rhs, selector)
shuffle_down(src, lane_delta, clamp, member_mask) -> (dst, pred)
shuffle_index(src, source_lane, clamp, member_mask) -> (dst, pred)
```

Predicates, pointer selection, static loop indices and address expressions are
control operations. There is no compound `pipeline`, `update_state`,
`softplus`, `quantize`, `gate`, or `reduce` operation: all relevant operations
are expanded below.

## Complete sketch

```python
# ===========================================================================
# Static specializations, host descriptor, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    INPUT_DTYPE="bf16",
    STATE_DTYPE=("bf16", "f16", "f32", "i16"),
    WEIGHT_DTYPE=("f32", "bf16"),
    MATRIX_A_DTYPE="f32",
    INDEX_DTYPE=("i32", "i64"),
    SCALE_STATE=(False, True),
    DIM=(64, 128),
    DSTATE=(64, 96, 128, 256),
    CONSUMER_WARPS=4,
    ROWS_PER_STAGE=16,
    NUM_STAGES=3,
    PHILOX_ROUNDS=(0, 10),
    target="sm_100a",
)

host_assert(SCALE_STATE == (STATE_DTYPE == "i16"))
host_assert(STATE_DTYPE != "i16" or DSTATE in (64, 128, 256))
host_assert(PHILOX_ROUNDS == 0 or
            (PHILOX_ROUNDS == 10 and STATE_DTYPE == "f16" and
             DSTATE in (64, 128) and not SCALE_STATE))
host_assert(DIM % ROWS_PER_STAGE == 0)
host_assert(DIM // ROWS_PER_STAGE >= NUM_STAGES)

state_map = tensor_map(
    base=state,
    rank=4,
    shape=(DSTATE, DIM, nheads, state_cache_size),
    element_strides=(1, DSTATE, DSTATE * DIM, state_stride_batch),
    box=(DSTATE, ROWS_PER_STAGE, 1, 1),
    element_stride=(1, 1, 1, 1),
    interleave="none",
    swizzle="none",
    l2_promotion="128B",
    oob_fill="none",
)
# instruction_selection: host CUtensorMap encoding consumed by rank-4
# cp.async.bulk.tensor instructions; extent: one by-value descriptor per launch

host_assert(address(state) % 128 == 0)
host_assert(address(x) % 128 == 0)
host_assert(address(B) % 128 == 0)
host_assert(address(C) % 128 == 0)
if z is present:
    host_assert(address(z) % 128 == 0)
if SCALE_STATE:
    host_assert(address(state_scale) % 128 == 0)

INPUT_VECTOR_BYTES = sizeof(PackedAligned(INPUT_DTYPE))
host_assert(INPUT_VECTOR_BYTES == 16)
host_assert(
    x_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
if z is present:
    host_assert(
        z_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
host_assert(
    B_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
host_assert(
    C_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)

STATE_STAGE_BYTES = ROWS_PER_STAGE * DSTATE * sizeof(STATE_DTYPE)
X_BYTES = DIM * sizeof(INPUT_DTYPE)
B_BYTES = DSTATE * sizeof(INPUT_DTYPE)
C_BYTES = DSTATE * sizeof(INPUT_DTYPE)
Z_BYTES = DIM * sizeof(INPUT_DTYPE) if z is present else 0
SCALE_BYTES = DIM * sizeof("f32") if SCALE_STATE else 0
INPUT_BYTES = X_BYTES + B_BYTES + C_BYTES + Z_BYTES + SCALE_BYTES
TOTAL_ROW_STAGES = DIM // ROWS_PER_STAGE
STAGES_READ_ONLY = NUM_STAGES
STAGES_BOTH = TOTAL_ROW_STAGES - NUM_STAGES
STAGES_WRITE_ONLY = NUM_STAGES

launch_config = launch(
    grid=(batch, nheads, 1),
    block=(32, 5, 1),
    threads=160,
    dynamic_smem_bytes=sizeof_shared_storage_vertical(
        INPUT_DTYPE, STATE_DTYPE, SCALE_STATE, DIM, DSTATE,
        ROWS_PER_STAGE, NUM_STAGES),
)

def selective_state_update_stp_vertical(
    state_map,
    state, state_scale,
    x, dt, A, B, C, D, z, dt_bias,
    state_batch_indices, dst_state_batch_indices, rand_seed,
    output,
    state_stride_batch, state_scale_stride_batch,
    x_stride_batch, dt_stride_batch, B_stride_batch, C_stride_batch,
    z_stride_batch, out_stride_batch,
    state_batch_indices_stride_batch,
    dst_state_batch_indices_stride_batch,
    nheads, ngroups, state_cache_size,
    dt_softplus, update_state, pad_slot_id,
):
    if PHILOX_ROUNDS > 0 and not SCALE_STATE:
        random_seed = 0
        # instruction_selection: mov.b64 zero; extent: one nullable-seed fallback
        if rand_seed is present:
            random_seed = copy_g2r(rand_seed[0])
            # instruction_selection: guarded ld.global.b64 after setp.eq.b64 and a pointer-null branch; extent: one scalar per thread at kernel entry. The complete expression is dead-code eliminated in non-Philox specializations.

    batch_i = cta_id(axis="x", extent=batch)
    # instruction_selection: mov.u32 from %ctaid.x; extent: one physical CTA coordinate
    head = cta_id(axis="y", extent=nheads)
    # instruction_selection: mov.u32 from %ctaid.y; extent: one physical CTA coordinate
    raw_lane = thread_id(axis="x", extent=32)
    # instruction_selection: mov.u32 from %tid.x; extent: one independent physical x coordinate
    lane = bit_and(raw_lane, 31)
    # instruction_selection: and.b32 with immediate 31; extent: one source-exact threadIdx.x % warpSize lane mask
    warp = thread_id(axis="y", extent=5)
    # instruction_selection: mov.u32 from %tid.y; extent: one independent physical warp coordinate
    group = head // (nheads // ngroups)
    # instruction_selection: integer division specialized/folded according to runtime nheads/ngroups; extent: one group index

    if state_batch_indices is present:
        raw_state_batch = copy_g2r(
            state_batch_indices[
                batch_i * state_batch_indices_stride_batch])
        # instruction_selection: present i32 emits ld.global.nc.s32 followed by cvt.s64.s32; present i64 emits ld.global.nc.s64; extent: one source slot per thread
        state_batch = cast("i64", raw_state_batch)
        # instruction_selection: the i32 sign extension above forms the logical b64 slot, while i64 is identity; extent: one source slot
    else:
        state_batch = cast("i64", batch_i)
        # instruction_selection: cvt.u64.u32; extent: one fallback source slot
    # instruction_selection: source b64-to-b32 cvt.u32.u64 is hoisted once before all G2S operations in each READ_STATE producer helper branch and reused by every G2S in that helper; extent: four static TT/TF x z helper copies, at most one executed at runtime, and none in FF

    if dst_state_batch_indices is present:
        raw_dst_batch = copy_g2r(
            dst_state_batch_indices[
                batch_i * dst_state_batch_indices_stride_batch])
        # instruction_selection: present i32 emits ld.global.nc.s32 directly to a b32 TMA coordinate; present i64 emits ld.global.nc.s64; extent: one destination slot per thread
        dst_state_batch = cast("i64", raw_dst_batch)
        # instruction_selection: logical i64 pseudo cast only; the present-i32 cast is DCE and present-i64 remains b64 until producer-helper coordinate preparation
    else:
        dst_state_batch = state_batch
        # instruction_selection: logical b64 alias of the source slot; extent: one null-destination fallback
    # instruction_selection: present i32 uses its loaded b32 directly; null i32 emits one cvt.u32.u64 in the outer fallback and all S2G operations reuse it; present/null i64 emits one cvt.u32.u64 per WRITE_STATE producer helper branch and reuses it throughout that helper; extent: two static TT x z helper copies for i64, at most one executed at runtime, and none in TF/FF

    state_ptr_offset = (
        state_batch * state_stride_batch + head * DIM * DSTATE)
    dst_state_ptr_offset = (
        dst_state_batch * state_stride_batch + head * DIM * DSTATE)
    # instruction_selection: mul/add wide integer address arithmetic; extent: one source and destination state-head offset per thread

    x_head = x[batch_i * x_stride_batch + head * DIM :]
    B_group = B[batch_i * B_stride_batch + group * DSTATE :]
    C_group = C[batch_i * C_stride_batch + group * DSTATE :]
    z_head = (z[batch_i * z_stride_batch + head * DIM :] if z is present
              else null)
    source_scale_rows = (
        state_scale[state_batch * state_scale_stride_batch + head * DIM :]
        if SCALE_STATE else null)
    destination_scale_rows = (
        state_scale[dst_state_batch * state_scale_stride_batch + head * DIM :]
        if SCALE_STATE else null)

    # =======================================================================
    # Exact dynamic-shared C++ ABI layout
    # =======================================================================

    smem = raw_shared(dynamic_smem_bytes, alignment=128)
    off_state = 0
    off_x = off_state + NUM_STAGES * STATE_STAGE_BYTES
    off_z = off_x + X_BYTES
    off_B = off_z + DIM * sizeof(INPUT_DTYPE)
    off_C = off_B + B_BYTES
    off_out = off_C + C_BYTES
    off_scale = align_up(off_out + DIM * sizeof("f32"), 128)
    off_empty = off_scale + (DIM * sizeof("f32") if SCALE_STATE else 0)
    off_full = off_empty + NUM_STAGES * 8
    off_consumers = off_full + NUM_STAGES * 8

    sState = view(
        smem, offset=off_state, dtype=STATE_DTYPE,
        shape=(NUM_STAGES, ROWS_PER_STAGE, DSTATE),
        alignment=128)
    sX = view(smem, offset=off_x, dtype="bf16", shape=(DIM,), alignment=16)
    sZ = view(smem, offset=off_z, dtype="bf16", shape=(DIM,), alignment=16)
    sB = view(smem, offset=off_B, dtype="bf16", shape=(DSTATE,), alignment=16)
    sC = view(smem, offset=off_C, dtype="bf16", shape=(DSTATE,), alignment=16)
    sOut = view(smem, offset=off_out, dtype="f32", shape=(DIM,), alignment=4)
    sScale = view(smem, offset=off_scale, dtype="f32", shape=(DIM,), alignment=128)
    bar_empty = view(smem, offset=off_empty, dtype="mbarrier.b64", shape=(3,))
    bar_full = view(smem, offset=off_full, dtype="mbarrier.b64", shape=(3,))
    bar_consumers = view(
        smem, offset=off_consumers, dtype="mbarrier.b64", shape=(1,))
    # The no-scale C++ conditional array has zero extent; barriers begin at the
    # 128-byte-aligned off_scale. There is no extra scale tile or dummy byte.

    # =======================================================================
    # Barrier initialization by exact source warps
    # =======================================================================

    if warp < NUM_STAGES and lane == 0:
        mbarrier_init(bar_empty[warp], 1 + CONSUMER_WARPS * 32)
        # instruction_selection: mbarrier.init.shared.b64 count 129; extent: one empty barrier for each of warps 0,1,2
        mbarrier_init(bar_full[warp], 1 + CONSUMER_WARPS * 32)
        # instruction_selection: mbarrier.init.shared.b64 count 129; extent: one full barrier for each of warps 0,1,2
        fence_proxy_async_shared_cta()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one barrier-pair publication by each initializing lane
    if warp == 0 and lane == 0:
        mbarrier_init(bar_consumers[0], CONSUMER_WARPS * 32)
        # instruction_selection: mbarrier.init.shared.b64 count 128; extent: one consumer-completion barrier
    cta_sync()
    # instruction_selection: bar.sync 0; extent: all 160 physical threads after barrier initialization

    # =======================================================================
    # Producer warp: one lane, runtime read/write/z dispatch
    # =======================================================================

    if warp == CONSUMER_WARPS:
        read_state = state_batch != pad_slot_id
        write_state = read_state and update_state
        if lane == 0:
            # The source instantiates exactly TT, TF, or FF state helpers and
            # crosses those with hasZ=true/false. It never instantiates FT.
            dispatch producer_program(
                READ_STATE=(True if read_state else False),
                WRITE_STATE=(True if write_state else False),
                HAS_Z=(True if z_head is not null else False),
            ):
                # -----------------------------------------------------------
                # Phase 1, stage 0: all vectors plus optional first state tile
                # -----------------------------------------------------------
                stage = 0
                token_empty_0 = mbarrier_arrive(bar_empty[stage])
                # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning a b64 token; extent: producer arrival for stage 0
                mbarrier_wait_token(bar_empty[stage], token_empty_0)
                # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 on that exact token; extent: producer waits until 128 initial consumer arrivals complete the phase

                copy_g2s_bulk(x_head[0:DIM], sX, X_BYTES, bar_full[stage])
                # instruction_selection: cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes; extent: one exact DIM*2-byte transfer
                copy_g2s_bulk(B_group[0:DSTATE], sB, B_BYTES, bar_full[stage])
                # instruction_selection: cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes; extent: one exact DSTATE*2-byte transfer
                copy_g2s_bulk(C_group[0:DSTATE], sC, C_BYTES, bar_full[stage])
                # instruction_selection: cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes; extent: one exact DSTATE*2-byte transfer
                if HAS_Z:
                    copy_g2s_bulk(z_head[0:DIM], sZ, Z_BYTES, bar_full[stage])
                    # instruction_selection: cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes; extent: one exact DIM*2-byte transfer, absent in no-z specialization
                if SCALE_STATE:
                    copy_g2s_bulk(
                        source_scale_rows[0:DIM], sScale,
                        SCALE_BYTES, bar_full[stage])
                    # instruction_selection: cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes; extent: one exact DIM*4-byte transfer

                if READ_STATE:
                    copy_tmap_g2s(
                        state_map, coords=(0, 0, head, state_batch),
                        dst_shared=sState[stage],
                        completion_barrier=bar_full[stage])
                    # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes; extent: one [16,DSTATE] tile
                    mbarrier_arrive_expect_tx(
                        bar_full[stage], STATE_STAGE_BYTES + INPUT_BYTES)
                    # instruction_selection: mbarrier.arrive.expect_tx.release.cta.shared::cta.b64; extent: exact state+issued-vector transaction bytes
                else:
                    mbarrier_arrive_expect_tx(bar_full[stage], INPUT_BYTES)
                    # instruction_selection: same expect-tx form with only issued-vector transaction bytes; no state TMA

                # -----------------------------------------------------------
                # Phase 1, stages 1 and 2: state only
                # -----------------------------------------------------------
                for fill_iter in static_range(1, STAGES_READ_ONLY):
                    stage = fill_iter % NUM_STAGES
                    d_read = fill_iter * ROWS_PER_STAGE
                    token_empty = mbarrier_arrive(bar_empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning token; extent: one producer arrival per fill stage
                    mbarrier_wait_token(bar_empty[stage], token_empty)
                    # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 with that token; extent: one wait per fill stage
                    if READ_STATE:
                        copy_tmap_g2s(
                            state_map,
                            coords=(0, d_read, head, state_batch),
                            dst_shared=sState[stage],
                            completion_barrier=bar_full[stage])
                        # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes; extent: one state tile
                        mbarrier_arrive_expect_tx(
                            bar_full[stage], STATE_STAGE_BYTES)
                        # instruction_selection: mbarrier.arrive.expect_tx.release.cta.shared::cta.b64; extent: exact tile bytes
                    else:
                        mbarrier_arrive(bar_full[stage])
                        # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: one bare full arrival, no expected transaction

                # -----------------------------------------------------------
                # Phase 2: store-before-load steady state
                # -----------------------------------------------------------
                for steady_iter in static_range(STAGES_BOTH):
                    stage = (STAGES_READ_ONLY + steady_iter) % NUM_STAGES
                    d_read = (
                        STAGES_READ_ONLY + steady_iter) * ROWS_PER_STAGE
                    d_write = steady_iter * ROWS_PER_STAGE

                    token_empty = mbarrier_arrive(bar_empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64 returning token; extent: one producer arrival per reused stage
                    mbarrier_wait_token(bar_empty[stage], token_empty)
                    # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 with that token; extent: wait until all four consumer warps finish this tile

                    if READ_STATE or WRITE_STATE:
                        fence_proxy_async_shared_cta()
                        # instruction_selection: fence.proxy.async.shared::cta; extent: one reuse edge after consumer shared writes and before any state store/load
                        if WRITE_STATE:
                            copy_tmap_s2g(
                                sState[stage], state_map,
                                coords=(0, d_write, head, dst_state_batch))
                            # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group consuming the already prepared b32 destination coordinate; extent: one [16,DSTATE] state tile
                            bulk_commit_group()
                            # instruction_selection: cp.async.bulk.commit_group; extent: one store group
                            bulk_wait_group_read_zero()
                            # instruction_selection: cp.async.bulk.wait_group.read 0; extent: store has finished reading shared before stage overwrite

                        if READ_STATE:
                            copy_tmap_g2s(
                                state_map,
                                coords=(0, d_read, head, state_batch),
                                dst_shared=sState[stage],
                                completion_barrier=bar_full[stage])
                            # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes; extent: one next state tile after store read-wait
                            mbarrier_arrive_expect_tx(
                                bar_full[stage], STATE_STAGE_BYTES)
                            # instruction_selection: mbarrier.arrive.expect_tx.release.cta.shared::cta.b64; extent: exact next-tile bytes
                        else:
                            mbarrier_arrive(bar_full[stage])
                            # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: one full phase without a next load
                    else:
                        mbarrier_arrive(bar_full[stage])
                        # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: pad-slot full phase with no proxy fence or state transfer

                # -----------------------------------------------------------
                # Phase 3: drain the final three optional state stores
                # -----------------------------------------------------------
                for drain_iter in static_range(STAGES_WRITE_ONLY):
                    stage = (
                        STAGES_READ_ONLY + STAGES_BOTH + drain_iter
                    ) % NUM_STAGES
                    d_write = (STAGES_BOTH + drain_iter) * ROWS_PER_STAGE
                    token_empty = mbarrier_arrive(bar_empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64 returning token; extent: one producer arrival per drain stage
                    mbarrier_wait_token(bar_empty[stage], token_empty)
                    # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 with that token; extent: one final consumer-completion wait
                    if WRITE_STATE:
                        fence_proxy_async_shared_cta()
                        # instruction_selection: fence.proxy.async.shared::cta; extent: one final shared-to-async edge
                        copy_tmap_s2g(
                            sState[stage], state_map,
                            coords=(0, d_write, head, dst_state_batch))
                        # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group consuming the already prepared b32 destination coordinate; extent: one final state tile
                        bulk_commit_group()
                        # instruction_selection: cp.async.bulk.commit_group; extent: one final store group
                        bulk_wait_group_read_zero()
                        # instruction_selection: cp.async.bulk.wait_group.read 0; extent: one final shared-read completion

    # =======================================================================
    # Consumer warps: initial empty arrivals and scalar setup
    # =======================================================================

    else:
        for stage in static_range(NUM_STAGES):
            mbarrier_arrive(bar_empty[stage])
            # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: one arrival from each of 128 consumers for each of three stages

        A_value_raw = copy_g2r(A[head])
        # instruction_selection: ld.global.nc.f32; extent: one read-only A scalar per consumer thread
        A_value = cast("f32", A_value_raw)
        # instruction_selection: identity for matrix-A f32; extent: one scalar

        d_value = 0.0
        # instruction_selection: mov.b32 zero fallback; extent: one optional-D scalar
        if D is present:
            d_raw = copy_g2r(D[head])
            # instruction_selection: ld.global.nc.f32 or ld.global.nc.b16; extent: one read-only D scalar
            d_value = cast("f32", d_raw)
            # instruction_selection: cvt.f32.bf16 for BF16 or identity for f32; extent: one scalar

        dt_raw = copy_g2r(dt[batch_i * dt_stride_batch + head])
        # instruction_selection: ld.global.nc.f32 or ld.global.nc.b16; extent: one read-only dt scalar
        dt_value = cast("f32", dt_raw)
        # instruction_selection: cvt.f32.bf16 for BF16 or identity for f32; extent: one scalar
        if dt_bias is present:
            bias_raw = copy_g2r(dt_bias[head])
            # instruction_selection: ld.global.nc.f32 or ld.global.nc.b16; extent: one read-only bias scalar
            bias_value = cast("f32", bias_raw)
            # instruction_selection: cvt.f32.bf16 for BF16 or identity for f32; extent: one scalar
            dt_value = add(dt_value, bias_value)
            # instruction_selection: add.ftz.f32; extent: one scalar

        if dt_softplus:
            if dt_value <= 20.0:
                exp_arg = mul(dt_value, LOG2_E)
                # instruction_selection: mul.ftz.f32; extent: one scalar on the <=20 branch
                exp_dt = exp2(exp_arg)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar on the <=20 branch
                one_plus_exp = add(1.0, exp_dt)
                # instruction_selection: add.ftz.f32; extent: one scalar on the <=20 branch
                log_value = log2(one_plus_exp)
                # instruction_selection: lg2.approx.ftz.f32; extent: one scalar on the <=20 branch
                dt_value = mul(log_value, LN_2)
                # instruction_selection: mul.ftz.f32; extent: one scalar on the <=20 branch
            # instruction_selection: setp/branch around the full exp/log path; dt>20 preserves the original value

        dA_arg = mul(A_value, dt_value)
        # instruction_selection: mul.ftz.f32; extent: one scalar
        dA_exp_arg = mul(dA_arg, LOG2_E)
        # instruction_selection: mul.ftz.f32; extent: one scalar
        dA = exp2(dA_exp_arg)
        # instruction_selection: ex2.approx.ftz.f32; extent: one scalar

        USE_STATE_CACHE = state_batch != pad_slot_id
        # source runtime dispatch creates true/false consumer helper instances

        lane_is_zero = select(lane == 0, 1.0, 0.0)
        # instruction_selection: one setp.eq.b32 plus one selp.f32 hoisted before the serial stage traversal; extent: one invariant FP32 lane indicator per consumer thread

        # ===================================================================
        # Consumer stage traversal
        # ===================================================================

        d_begin = 0
        stage = 0
        while d_begin < DIM:
            # This is the source's non-unrolled serial counted loop. Both
            # d_begin and stage are loop-carried in optimized PTX.

            token_full = mbarrier_arrive(bar_full[stage])
            # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning a b64 token; extent: one arrival per consumer thread/stage
            mbarrier_wait_token(bar_full[stage], token_full)
            # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 on that exact token; extent: wait for producer plus async transactions

            for dd in static_range(warp, ROWS_PER_STAGE, CONSUMER_WARPS):
                d = d_begin + dd
                x_raw = copy_s2r(sX[d])
                # instruction_selection: ld.shared.b16; extent: one row input
                x_value = cast("f32", x_raw)
                # instruction_selection: cvt.f32.bf16; extent: one scalar
                d_times_x = mul(d_value, x_value)
                # instruction_selection: mul.ftz.f32; extent: the first of two source row-seed multiplies
                out_value = mul(d_times_x, lane_is_zero)
                # instruction_selection: mul.ftz.f32; extent: the second row-seed multiply using the hoisted FP32 lane indicator

                STATE_VALUES_PER_BANK = 4 // sizeof(STATE_DTYPE)
                state_decode_scale = 1.0
                new_state_max = NEGATIVE_FLT_MAX
                if SCALE_STATE:
                    state_decode_scale = copy_s2r(sScale[d])
                    # instruction_selection: ld.shared.b32; extent: one old row scale before any new scale publication
                rNewState = reg_tile(
                    "f32", [DSTATE // 32] if SCALE_STATE else [1])
                random_words = reg_tile("u32", [4])

                for iteration, i in lane_vector_range_with_iteration(
                        lane * STATE_VALUES_PER_BANK,
                        DSTATE,
                        32 * STATE_VALUES_PER_BANK):
                    rStateWord = copy_s2r(
                        word_view(
                            sState[stage, dd, i : i + STATE_VALUES_PER_BANK],
                            "u32"))
                    # instruction_selection: ld.shared.v2.b16 for two-byte state or ld.shared.b32 for f32; extent: one exact four-byte bank word
                    rState = view(
                        rStateWord, dtype=STATE_DTYPE,
                        shape=(STATE_VALUES_PER_BANK,))

                    if sizeof(STATE_DTYPE) == sizeof(INPUT_DTYPE):
                        rBWord = copy_s2r(
                            word_view(sB[i : i + STATE_VALUES_PER_BANK], "u32"))
                        # instruction_selection: ld.shared.v2.b16; extent: one exact BF16 bank word issued after the state-word load
                        rCWord = copy_s2r(
                            word_view(sC[i : i + STATE_VALUES_PER_BANK], "u32"))
                        # instruction_selection: ld.shared.v2.b16; extent: one exact BF16 bank word issued after the B-word load
                        rB = view(rBWord, dtype="bf16", shape=(2,))
                        rC = view(rCWord, dtype="bf16", shape=(2,))

                    if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                        # STATE_VALUES_PER_BANK is two for FP16, so e==0 is the
                        # sole refresh in this bank word.
                        offset = state_ptr_offset + d * DSTATE + i
                        c0 = low_u32(offset)
                        c1 = high_u32(offset)
                        c2 = 0
                        c3 = 0
                        k0 = low_u32(random_seed)
                        k1 = high_u32(random_seed)
                        # The graph below is the complete ten-round source
                        # dependency graph. Optimized placement is shape
                        # specific: PSR128 precomputes the observed complete
                        # seed-derived immediate-key add chain outside the
                        # consumer serial stage loop, whereas PSR64 hoists only
                        # part of that chain and retains the remaining
                        # key-derived add.s32 instructions in every serial-stage
                        # iteration. It is written in-line here only to preserve
                        # the source dependencies.
                        for philox_i in static_range(10):
                            old_c0 = c0
                            old_c2 = c2
                            c0_hi = mul_hi_u32(0xCD9E8D57, old_c2)
                            # instruction_selection: mul.hi.u32; logical extent: one source op per round; optimized emitted extent is shape-specific after the round-0 zero fold and round-9 output DCE described below
                            c0_xor_c1 = bit_xor(c0_hi, c1)
                            # instruction_selection: xor.b32; logical extent: one source op per round; optimized emitted extent is shape-specific after constant propagation
                            next_c0 = bit_xor(c0_xor_c1, k0)
                            # instruction_selection: xor.b32; logical extent: one source op per round; optimized emitted extent retains only non-folded dependencies needed by consumed outputs
                            c2_hi = mul_hi_u32(0xD2511F53, old_c0)
                            # instruction_selection: mul.hi.u32; logical extent: one source op per round; optimized emitted extent omits the final-round c2 path because random_words[2] is not consumed
                            c2_xor_c3 = bit_xor(c2_hi, c3)
                            # instruction_selection: xor.b32; logical extent: one source op per round; optimized emitted extent folds round-0 c3==0 and deletes the unused final c2 path
                            next_c2 = bit_xor(c2_xor_c3, k1)
                            # instruction_selection: xor.b32; logical extent: one source op per round; optimized emitted extent deletes the unused final c2 result
                            next_c1 = mul_lo_s32(old_c2, 0xCD9E8D57)
                            # instruction_selection: mul.lo.s32; logical extent: one source op per round; optimized emitted extent folds round-0 old_c2==0
                            next_c3 = mul_lo_s32(old_c0, 0xD2511F53)
                            # instruction_selection: mul.lo.s32; logical extent: one source op per round; optimized emitted extent deletes the unused final c3 result
                            next_k0 = add_s32(k0, 0x9E3779B9)
                            # instruction_selection: add.s32; logical extent: one source key update per round; optimized emitted placement is fully loop-external for the observed PSR128 immediate-key chain but split between loop-external and per-stage-loop issues in PSR64, with the final unused update DCE
                            next_k1 = add_s32(k1, 0xBB67AE85)
                            # instruction_selection: add.s32; logical extent: one source key update per round; optimized emitted placement is fully loop-external for the observed PSR128 immediate-key chain but split between loop-external and per-stage-loop issues in PSR64, with the final unused update DCE
                            c0, c1, c2, c3 = (
                                next_c0, next_c1, next_c2, next_c3)
                            k0, k1 = next_k0, next_k1
                        # instruction_selection optimized-extent rule: round 0
                        # folds operations fed by c2==c3==0; round 9 removes
                        # unconsumed c2/c3 and final key updates because this
                        # vertical bank consumes only random_words[0:2]. The
                        # frozen normal-JIT PSR64 and PSR128 PTX each retain
                        # their verified shape-specific constant folds; no
                        # annotation above promises ten physical issues.
                        random_words = (c0, c1, c2, c3)

                    raw_sr_f16x2 = reg_tile(
                        "u32", [STATE_VALUES_PER_BANK])
                    for e in static_range(STATE_VALUES_PER_BANK):
                        if USE_STATE_CACHE:
                            state_value = cast("f32", rState[e])
                            # instruction_selection: cvt.f32.bf16, cvt.f32.f16, cvt.rn.f32.s16, or identity f32; extent: one state element
                            if SCALE_STATE:
                                state_value = mul(
                                    state_value, state_decode_scale)
                                # instruction_selection: mul.ftz.f32; extent: one decoded int16 state element
                        else:
                            state_value = 0.0
                            # instruction_selection: compile-specialized zero value; source state-word load may be dead-code eliminated in this helper

                        if sizeof(STATE_DTYPE) == sizeof(INPUT_DTYPE):
                            B_value = cast("f32", rB[e])
                            # instruction_selection: cvt.f32.bf16; extent: one element from the already-loaded bank word
                            C_value = cast("f32", rC[e])
                            # instruction_selection: cvt.f32.bf16; extent: one element from the already-loaded bank word
                        else:
                            b_raw = copy_s2r(sB[i + e])
                            # instruction_selection: ld.shared.b16; extent: one scalar B element after this e's state conversion
                            B_value = cast("f32", b_raw)
                            # instruction_selection: cvt.f32.bf16; extent: one scalar completed before C load
                            c_raw = copy_s2r(sC[i + e])
                            # instruction_selection: ld.shared.b16; extent: one scalar C element after B conversion
                            C_value = cast("f32", c_raw)
                            # instruction_selection: cvt.f32.bf16; extent: one scalar completed before compute

                        dB = mul(B_value, dt_value)
                        # instruction_selection: mul.ftz.f32; extent: one scalar
                        dB_x = mul(dB, x_value)
                        # instruction_selection: mul.ftz.f32; extent: one scalar
                        new_state = fma(state_value, dA, dB_x)
                        # instruction_selection: fma.rn.ftz.f32; extent: one scalar

                        if SCALE_STATE:
                            magnitude = abs(new_state)
                            # instruction_selection: abs.ftz.f32; extent: one scalar
                            new_state_max = max(new_state_max, magnitude)
                            # instruction_selection: max.ftz.f32; extent: one scalar
                            rNewState[
                                iteration * STATE_VALUES_PER_BANK + e
                            ] = new_state
                        elif PHILOX_ROUNDS > 0:
                            random13 = bit_and(random_words[e], 0x1FFF)
                            # instruction_selection: and.b32; extent: one 13-bit random operand
                            raw_sr_f16x2[e] = cast(
                                "raw_f16x2_u32", new_state,
                                rounding=(
                                    "stochastic", random13,
                                    "zero_second_f32"))
                            # instruction_selection: cvt.rs.f16x2.f32 producing one full b32 result; extent: desired FP16 low half plus zero dummy
                        else:
                            rState[e] = cast(
                                STATE_DTYPE, new_state, rounding="rn")
                            # instruction_selection: cvt.rn.bf16.f32, cvt.rn.f16.f32, or identity f32; extent: one scalar

                        out_value = fma(
                            new_state, C_value, out_value)
                        # instruction_selection: fma.rn.ftz.f32, with source-order seed folding permitted; extent: one scalar accumulator

                    if not SCALE_STATE:
                        if PHILOX_ROUNDS > 0:
                            packed_state_word = prmt_b32(
                                raw_sr_f16x2[0], raw_sr_f16x2[1],
                                selector=0x5410)
                            # instruction_selection: prmt.b32 selector 0x5410; extent: one FP16 pair
                            copy_r2s(
                                packed_state_word,
                                word_view(
                                    sState[
                                        stage, dd,
                                        i : i + STATE_VALUES_PER_BANK],
                                    "u32"))
                            # instruction_selection: st.shared.b32; extent: one packed stochastic-FP16 bank word
                        else:
                            if sizeof(STATE_DTYPE) == 2:
                                copy_r2s(
                                    rState[0:2],
                                    sState[stage, dd, i : i + 2])
                                # instruction_selection: st.shared.v2.b16 for ordinary BF16/FP16 state; extent: one typed two-element bank word
                            else:
                                copy_r2s(
                                    rState[0], sState[stage, dd, i])
                                # instruction_selection: st.shared.b32 for ordinary FP32 state; extent: one scalar bank word

                for delta in (16, 8, 4, 2, 1):
                    peer_out, unused_pred = shuffle_down(
                        out_value, delta,
                        clamp=31, member_mask=0xFFFFFFFF)
                    # instruction_selection: shfl.sync.down.b32 dst|pred, src, delta, 31, -1; extent: one scalar per lane, returned predicate unused
                    out_value = add(out_value, peer_out)
                    # instruction_selection: add.ftz.f32; extent: one scalar per lane
                if lane == 0:
                    copy_r2s(out_value, sOut[d])
                    # instruction_selection: st.shared.b32; extent: one row result

                if SCALE_STATE and USE_STATE_CACHE:
                    for delta in (16, 8, 4, 2, 1):
                        peer_max, unused_pred = shuffle_down(
                            new_state_max, delta,
                            clamp=31, member_mask=0xFFFFFFFF)
                        # instruction_selection: shfl.sync.down.b32 dst|pred, src, delta, 31, -1; extent: one scalar per lane, returned predicate unused
                        new_state_max = max(new_state_max, peer_max)
                        # instruction_selection: max.ftz.f32; extent: one scalar per lane
                    new_state_max, unused_pred = shuffle_index(
                        new_state_max, source_lane=0,
                        clamp=31, member_mask=0xFFFFFFFF)
                    # instruction_selection: shfl.sync.idx.b32 dst|pred, src, 0, 31, -1; extent: one lane-0 broadcast, returned predicate unused

                    encode_scale = 1.0
                    # instruction_selection: mov.b32 1.0 default; extent: one scalar
                    if new_state_max != 0.0:
                        # instruction_selection: setp.eq.ftz.f32 plus branch over division when zero; extent: one scalar control edge
                        encode_scale = div(32767.0, new_state_max)
                        # instruction_selection: div.approx.ftz.f32; extent: nonzero branch only
                    new_decode_scale = div(1.0, encode_scale)
                    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar

                    for iteration, i in lane_vector_range_with_iteration(
                            lane * STATE_VALUES_PER_BANK,
                            DSTATE,
                            32 * STATE_VALUES_PER_BANK):
                        quantized_s32 = reg_tile(
                            "i32", [STATE_VALUES_PER_BANK])
                        for e in static_range(STATE_VALUES_PER_BANK):
                            scaled = mul(
                                rNewState[
                                    iteration * STATE_VALUES_PER_BANK + e],
                                encode_scale)
                            # instruction_selection: mul.ftz.f32; extent: one scalar
                            clipped_low = max(scaled, -32767.0)
                            # instruction_selection: max.ftz.f32; extent: one scalar
                            clipped = min(clipped_low, 32767.0)
                            # instruction_selection: min.ftz.f32; extent: one scalar
                            quantized_s32[e] = cast(
                                "i32", clipped, rounding="rni")
                            # instruction_selection: cvt.rni.ftz.s32.f32; extent: one scalar whose low 16 bits hold int16
                        packed_i16 = prmt_b32(
                            quantized_s32[0], quantized_s32[1],
                            selector=0x5410)
                        # instruction_selection: prmt.b32 selector 0x5410; extent: one int16 pair
                        copy_r2s(
                            packed_i16,
                            word_view(
                                sState[
                                    stage, dd,
                                    i : i + STATE_VALUES_PER_BANK],
                                "u32"))
                        # instruction_selection: st.shared.b32; extent: one packed int16 bank word in the TMA tile

                    if lane == 0:
                        copy_r2s(new_decode_scale, sScale[d])
                        # instruction_selection: st.shared.b32; extent: one new row scale, even when update_state is false (global publication is suppressed later)

            fence_proxy_async_shared_cta()
            # instruction_selection: fence.proxy.async.shared::cta; extent: one publication instruction from each consumer thread after its warp has completed all four owned rows in this stage
            mbarrier_arrive(bar_empty[stage])
            # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1; extent: one arrival from each consumer thread after shared writes

            d_begin = d_begin + ROWS_PER_STAGE
            stage = (stage + 1) % NUM_STAGES
            # instruction_selection: add.s32 updates plus the optimized modulo-3 wrap and a loop backedge; extent: one pair of loop-carried updates per serial stage iteration

        # ===================================================================
        # Consumer-only completion barrier and source-order epilogue
        # ===================================================================

        token_consumers = mbarrier_arrive(bar_consumers[0])
        # instruction_selection: mbarrier.arrive.shared::cta.b64 count 1 returning token; extent: one arrival per consumer thread
        mbarrier_wait_token(bar_consumers[0], token_consumers)
        # instruction_selection: polling mbarrier.try_wait.shared::cta.b64 on that token; extent: all 128 consumers wait for every sOut/sScale publication

        d = warp * 32 + lane
        # instruction_selection: shl/add or mad integer arithmetic using independent %tid.y/%tid.x; extent: one epilogue row
        if d < DIM:
            out_value = copy_s2r(sOut[d])
            # instruction_selection: ld.shared.b32; extent: one row result
            if z is present:
                z_raw = copy_s2r(sZ[d])
                # instruction_selection: ld.shared.b16; extent: one row gate
                z_value = cast("f32", z_raw)
                # instruction_selection: cvt.f32.bf16; extent: one scalar
                neg_z = sub(0.0, z_value)
                # instruction_selection: sub.ftz.f32; extent: one scalar
                exp_arg = mul(neg_z, LOG2_E)
                # instruction_selection: mul.ftz.f32; extent: one scalar
                exp_neg_z = exp2(exp_arg)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
                denominator = add(1.0, exp_neg_z)
                # instruction_selection: add.ftz.f32; extent: one scalar
                sigmoid_z = div(1.0, denominator)
                # instruction_selection: div.approx.ftz.f32; extent: one scalar
                silu_z = mul(z_value, sigmoid_z)
                # instruction_selection: mul.ftz.f32; extent: one scalar
                out_value = mul(out_value, silu_z)
                # instruction_selection: mul.ftz.f32; extent: one scalar
            output_value = cast("bf16", out_value, rounding="rn")
            # instruction_selection: cvt.rn.bf16.f32; extent: one scalar
            copy_r2g(
                output_value,
                output[
                    batch_i * out_stride_batch + head * DIM + d])
            # instruction_selection: st.global.b16; extent: one guarded row result

        # This is a separate source-order branch after the complete output
        # branch, not a fused output+scale publication.
        if SCALE_STATE:
            if update_state and state_batch != pad_slot_id:
                if d < DIM:
                    new_scale = copy_s2r(sScale[d])
                    # instruction_selection: ld.shared.b32; extent: one row scale
                    copy_r2g(new_scale, destination_scale_rows[d])
                    # instruction_selection: st.global.b32; extent: one row scale
```

## Static specialization and launch boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| DIM, DSTATE, state/weight/index dtype, scale-state, Philox rounds | static | producer loops and four-row/element bodies specialize; consumer stage and lane-bank counted loops retain their emitted loop shape while bounds, registers, conversions and transaction bytes specialize |
| consumer warps, rows per stage, ring stages | fixed static 4/16/3 | exact block shape, stage offsets, arrival counts and fill/steady/drain trip counts |
| batch and nheads grid extents | static for each TIRx launch | exactly one CTA per batch/head |
| nheads/ngroups and tensor strides | runtime kernel values | preserves source integer group division and address arithmetic |
| state/destination indices, z/D/bias/seed presence, softplus/update/pad | runtime or host-specialized pointer/boolean state | preserves source branches and TT/TF/FF producer dispatch |

The host always builds the rank-4 state descriptor from the same base pointer,
shape, byte strides, `[DSTATE,16,1,1]` box and cache/swizzle policy as
FlashInfer. It sets the exact dynamic-shared-memory launch attribute before
launch. DIM 256 is intentionally absent: the frozen vertical dispatch and
upstream vertical test domain use DIM 64/128 with four consumer warps.

## TIRx module and benchmark contract

- `KERNEL_META` names `selective_state_update_stp_vertical`, category
  `flashinfer`, compute capability 10.
- The frozen FlashInfer commit is
  `f2e04400e330fb2debe0bf8730d9424a1d37927f`; the primary CUDA header SHA256 is
  `c0e13b64bf42f4f8155058dc9f5877f7aca90832f50a1e7602863894908e89fd`.
- `CONFIGS` covers every declared source branch/shape; `BENCH_CONFIGS` is its
  branch-representative timed subset. Every source call uses
  `algorithm="vertical"` and verifies the expected dispatch.
- TIRx and source receive independent mutable state/output/scale buffers.
  Compilation, descriptor construction, allocations, source JIT, warmups and
  correctness preflight remain outside timed closures.
- The implementation and all pre-dispatch specializations contain no tile
  primitives. The implementation must use the basic directional copy,
  mbarrier, TMA, proxy-fence, shuffle and arithmetic operations represented
  here.

## Instruction-selection summary

- Three barrier pairs use counts 129; the consumer-completion barrier uses 128.
  Every source wait is token-based `mbarrier.arrive` plus
  `mbarrier.try_wait`, not parity-based waiting.
- Stage 0 issues one-dimensional bulk x/B/C and optional z/scale copies plus
  optional rank-4 state G2S TMA under one full barrier. Later fills issue only
  state TMA. Steady/drain stores use rank-4 S2G TMA, commit, and
  `wait_group.read 0` before shared reuse.
- Each consumer warp owns four rows per 16-row stage. Each lane traverses one
  32-bit state bank word at a time: two 16-bit state/B/C values or one FP32
  state plus scalar BF16 B/C loads.
- Every row executes five `shfl.sync.down` sum steps. Int16 rows with valid
  state also execute five max shuffles, one full-mask lane-0 broadcast, a
  register-resident second pass, symmetric max-then-min clamp,
  round-to-nearest integer conversion, one `prmt.b32` and one shared b32 store
  per lane word.
- Philox state preserves ten logical multiply-high/multiply/xor/add dependency
  rounds per bank word. Normal `-O3` PSR128 PTX hoists the observed complete
  immediate-key add chain outside the serial stage loop; PSR64 hoists only a
  subset and leaves the remaining key-derived adds inside each stage iteration.
  Both fold round-0 zeros and DCE unused round-9 c2/c3 and final key updates with
  their verified shape-specific folds. The consumed values use
  `cvt.rs.f16x2.f32`, `prmt.b32`, and `st.shared.b32`.
- The epilogue performs the BF16 output branch first, including the explicit
  exp/reciprocal/multiply SILU path only when z is present, then independently
  performs the optional destination-scale load/store branch.
