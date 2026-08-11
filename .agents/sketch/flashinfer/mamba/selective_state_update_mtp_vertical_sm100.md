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

# Selective-state-update MTP vertical SM100: execution sketch

This document is a non-executable transcription sketch of FlashInfer's CUDA
`selective_state_update_kernel_vertical_mtp`.  It freezes the 16-warp CTA,
three independent head groups, one TMA producer per group, four recurrence
warps per group, shared epilogue warp, one-stage state input, and direct
intermediate/final-state stores that the TIRx port must preserve.  The target
module is
[`tirx_kernels/flashinfer/mamba/selective_state_update_mtp_vertical.py`](../../tirx_kernels/flashinfer/mamba/selective_state_update_mtp_vertical.py).
It remains scaffolding until this sketch passes independent review.

The frozen FlashInfer commit is
`f2e04400e330fb2debe0bf8730d9424a1d37927f`.  The primary source is
`include/flashinfer/mamba/kernel_selective_state_update_mtp_vertical.cuh`,
SHA256
`8c8d292c08cc29eb0db47d231bcab47bdb86e285ff7b02f6135709e3a5058cca`.
Launch and descriptor construction come from
`include/flashinfer/mamba/invoke_selective_state_update_mtp.cuh`, SHA256
`5be5da574adc8b148064ee1214a9eda99881ec5163ef244e462cb53ae54c4bf8`.
The reviewed target is SM100a/B200.  Every reference call explicitly requests
`algorithm="vertical"`.

Input, B, C, z, and output are BF16 in the represented API domain.  State is
BF16, FP16, or FP32; weight/dt are FP32 or BF16; A is FP32; state indices are
int32 or int64.  `DIM in {64,128}`, `DSTATE in {64,96,128}`, `NTOKENS in
{1,2,4,6,8}`, ratio in `{1,2,4,8,16,32,64}`, and Philox rounds in `{0,10}`
are represented.  Scaled state and varlen are host-side rejection cases, not
kernel specializations.

## Pipeline and ownership

The grid is `(batch, ceil(nheads/3))`.  A CTA contains 512 threads as
`threadIdx=(lane 0..31, warp 0..15)`.  It owns up to three consecutive heads.
The final CTA head chunk can have one or two active groups; every role tests the
same `g < num_active_groups` predicate, so inactive group storage and barriers
are never touched.

| Warps | Role | Group | Exact work |
| --- | --- | ---: | --- |
| 0..3 | compute | 0 | DIM rows 0..DIM-1 in four warp stripes |
| 4..7 | compute | 1 | same for the second active head |
| 8..11 | compute | 2 | same for the third active head |
| 12 | TMA load | 0 | B, C, state and x descriptors for group 0 |
| 13 | TMA load | 1 | same for group 1 |
| 14 | TMA load | 2 | same for group 2 |
| 15 | epilogue | all active | drains groups in increasing order, applies optional z, stores output |

There is no `setmaxnreg` and no warp-group alignment operation.  Each compute
warp owns `DIM/4` rows.  It processes four adjacent rows per register-pressure
pass, so `numPasses=(DIM/4)/4`: four passes for DIM64 and eight for DIM128.
Each lane owns `DSTATE/32` contiguous state columns: 2, 3, or 4 values for
DSTATE64, 96, or 128.  The entire input state is loaded by one TMA transaction
before any pass; passes are register-lifetime partitions, not TMA stages.

## Primitive vocabulary

The sketch uses only explicit structural, copy, synchronization, and scalar or
packed operations.  These names describe lowering obligations; they are not
tile primitives.

```python
specialize(...)
launch(...)
shared_region(...)
reg(...)
view(...)

ptx(exact_mnemonic, exact_operands, issue_extent)
cta_sync()
```

There is no compound recurrence, softplus, sigmoid, reduction, stochastic
conversion, TMA pipeline, or epilogue operation.  The TIRx transcription must
spell out every address, load, arithmetic operation, shuffle, store, barrier
arrival, transaction byte count, and control predicate.  It must not use a
tile primitive.

## Complete sketch

```python
# ===========================================================================
# Host specialization, validation, descriptors, and launch
# source: invoke_selective_state_update_mtp.cuh vertical dispatch
# ===========================================================================

FROZEN_MODULE_KEYS = (
    # (STATE, WEIGHT, STATE_INDEX, DIM, DSTATE, NTOKENS, PHILOX_ROUNDS)
    ("bf16","bf16","i64",  64,128,4,0),
    ("bf16","f32", "i32",  64,128,4,0),
    ("bf16","f32", "i64", 128,128,4,0),
    ("bf16","f32", "i64",  64,128,1,0),
    ("bf16","f32", "i64",  64,128,2,0),
    ("bf16","f32", "i64",  64,128,4,0),
    ("bf16","f32", "i64",  64,128,6,0),
    ("bf16","f32", "i64",  64,128,8,0),
    ("bf16","f32", "i64",  64, 64,4,0),
    ("bf16","f32", "i64",  64, 96,4,0),
    ("f16", "f32", "i64",  64,128,4,10),
    ("f16", "f32", "i64",  64,128,4,0),
    ("f32", "f32", "i64",  64,128,4,0),
    ("f32", "f32", "i64",  64,128,6,0),
)

# Direct-store evidence families for the tuples above, in order:
# A,A,A,B,A,A,A,A,C,D,F,A,E,E.  A is ordinary 16-bit/DSTATE128/T>1;
# B is its NTOKENS1 exception; C is BF16/DSTATE64; D is BF16/DSTATE96;
# E is FP32/DSTATE128; F is FP16/DSTATE128/Philox10.

variant = specialize(
    module_key=one_of(FROZEN_MODULE_KEYS),
    INPUT_DTYPE="bf16",
    MATRIX_A_DTYPE="f32",
    HEADS_PER_GROUP=one_of(1,2,4,8,16,32,64),
    NUM_IN_STAGES=1,
    target="sm_100a",
)

NUM_GROUPS = 3
COMPUTE_WARPS_PER_GROUP = 4
NUM_WARPS = 16
ROWS_PER_WARP = DIM // COMPUTE_WARPS_PER_GROUP
ROWS_PER_PASS = 4
NUM_PASSES = ROWS_PER_WARP // ROWS_PER_PASS
STATE_VALUES_PER_THREAD = DSTATE // 32
OUTPUT_VECTOR = min(16 // sizeof(INPUT_DTYPE), DIM // 32)

# Source order: validate the enum, normalize async-horizontal, and resolve auto;
# this benchmark explicitly enters with vertical, so selection remains vertical.
host_assert(algorithm in (AUTO,SIMPLE,VERTICAL,HORIZONTAL,ASYNC_HORIZONTAL))
if algorithm == ASYNC_HORIZONTAL:
    algorithm = SIMPLE
if algorithm == AUTO:
    algorithm = SIMPLE if (state_scale is present or cu_seqlens is present) else (
        HORIZONTAL if batch >= 32 else SIMPLE)
host_assert(algorithm == VERTICAL)

# Frozen common alignment gates run after algorithm selection but before the
# vertical branch.  For represented BF16 input, PackedAligned<input_t> has 8
# elements and sizeof=alignof=16.
INPUT_PACK_BYTES = 16
host_assert(address(x) % INPUT_PACK_BYTES == 0)
host_assert((x_stride_batch*sizeof(INPUT_DTYPE)) % INPUT_PACK_BYTES == 0)
if z is present:
    host_assert(address(z) % INPUT_PACK_BYTES == 0)
    host_assert((z_stride_batch*sizeof(INPUT_DTYPE)) % INPUT_PACK_BYTES == 0)
host_assert(address(B) % INPUT_PACK_BYTES == 0)
host_assert(address(C) % INPUT_PACK_BYTES == 0)
host_assert((B_stride_batch*sizeof(INPUT_DTYPE)) % INPUT_PACK_BYTES == 0)
host_assert((C_stride_batch*sizeof(INPUT_DTYPE)) % INPUT_PACK_BYTES == 0)

STATE_VECTOR = min(16//sizeof(STATE_DTYPE), DSTATE//32)
STATE_VECTOR_BYTES = STATE_VECTOR*sizeof(STATE_DTYPE)
STATE_VECTOR_ALIGN = largest_power_of_two_divisor(STATE_VECTOR_BYTES)
host_assert(address(state) % STATE_VECTOR_ALIGN == 0)
host_assert((DIM*DSTATE*sizeof(STATE_DTYPE)) % STATE_VECTOR_BYTES == 0)
host_assert(address(output) % alignof(PackedAligned(INPUT_DTYPE,OUTPUT_VECTOR)) == 0)
if intermediate_states is present:
    host_assert(address(intermediate_states) % STATE_VECTOR_ALIGN == 0)

# Only now enter the vertical branch and execute its private validation.
host_assert(nheads % ngroups == 0)
host_assert(DIM % 32 == 0)
host_assert(DSTATE % 32 == 0)
host_assert(state_scale is absent)
host_assert(cu_seqlens is absent)
host_assert(PHILOX_ROUNDS == 0 or STATE_DTYPE == "f16")

tensor_state = tma_descriptor_4d(
    base=state,
    shape=(DSTATE, DIM, nheads, state_cache_size),
    stride_elements=(1, DSTATE, DSTATE*DIM, state_stride_batch),
    tile=(DSTATE, DIM, 1, 1),
    dtype=STATE_DTYPE,
    tensor_map_dtype=(BFLOAT16 if STATE_DTYPE=="bf16" else
                      FLOAT16 if STATE_DTYPE=="f16" else FLOAT32),
    strides_are_bytes_after_dimension_zero=True,
    box_strides=(1,1,1,1),
    interleave=NONE, swizzle=NONE, l2_promotion=L2_128B, oob_fill=NONE,
    base_alignment_bytes=128,
)
tensor_B = tma_descriptor_4d(
    base=B,
    shape=(DSTATE, ngroups, ntokens_mtp, batch),
    stride_elements=(1, DSTATE, B_stride_mtp, B_stride_batch),
    tile=(DSTATE, 1, NTOKENS, 1),
    dtype=INPUT_DTYPE,
    tensor_map_dtype=BFLOAT16,
    strides_are_bytes_after_dimension_zero=True,
    box_strides=(1,1,1,1),
    interleave=NONE, swizzle=NONE, l2_promotion=L2_128B, oob_fill=NONE,
    base_alignment_bytes=128,
)
tensor_C = tma_descriptor_4d(
    base=C,
    shape=(DSTATE, ngroups, ntokens_mtp, batch),
    stride_elements=(1, DSTATE, C_stride_mtp, C_stride_batch),
    tile=(DSTATE, 1, NTOKENS, 1),
    dtype=INPUT_DTYPE,
    tensor_map_dtype=BFLOAT16,
    strides_are_bytes_after_dimension_zero=True,
    box_strides=(1,1,1,1),
    interleave=NONE, swizzle=NONE, l2_promotion=L2_128B, oob_fill=NONE,
    base_alignment_bytes=128,
)
tensor_x = tma_descriptor_4d(
    base=x,
    shape=(DIM, nheads, ntokens_mtp, batch),
    stride_elements=(1, DIM, x_stride_mtp, x_stride_batch),
    tile=(DIM, 1, NTOKENS, 1),
    dtype=INPUT_DTYPE,
    tensor_map_dtype=BFLOAT16,
    strides_are_bytes_after_dimension_zero=True,
    box_strides=(1,1,1,1),
    interleave=NONE, swizzle=NONE, l2_promotion=L2_128B, oob_fill=NONE,
    base_alignment_bytes=128,
)

# Descriptor builder checks rank=4, every shape in [1,2^32], stride[0]=1,
# every supplied tile dimension after dimension zero <=256, and converts
# element strides 1..3 to byte strides.  The four kernel parameters are placed
# as independent `__grid_constant__ CUtensorMap` values in this exact order:
# tensor_state, tensor_B, tensor_C, tensor_x.  They are not ordinary parameter
# pointers and are not copied to per-thread local storage.

launch_config = launch(
    grid=(batch, ceil_div(nheads, 3), 1),
    block=(32, 16, 1),
    threads=512,
    launch_bounds=(512, 2),
    dynamic_smem_bytes=sizeof(SharedStorageVertical),
    max_dynamic_shared_memory_attribute=sizeof(SharedStorageVertical),
    arguments=(params,
               grid_constant(tensor_state), grid_constant(tensor_B),
               grid_constant(tensor_C), grid_constant(tensor_x)),
)

# ===========================================================================
# Exact per-group shared storage and non-overlapping lifetimes
# source: kernel_selective_state_update_mtp_vertical.cuh:67-90
# ===========================================================================

GroupStorage = struct(alignment=128):
    B = shared_region(INPUT_DTYPE, [NTOKENS,DSTATE], alignment=128,
                      lifetime="TMA B completion through all recurrence steps")
    C = shared_region(INPUT_DTYPE, [NTOKENS,DSTATE], alignment=128,
                      lifetime="TMA C completion through all recurrence steps")
    dt = shared_region("f32", [NTOKENS],
                       lifetime="compute-warp scalar publication through all passes")
    state_in = shared_region(STATE_DTYPE, [1,DIM*DSTATE], alignment=128,
                             lifetime="one state TMA through all row passes")
    x = shared_region(INPUT_DTYPE, [1,NTOKENS,DIM], alignment=128,
                      lifetime="one x TMA through all row passes")
    out = shared_region("f32", [NTOKENS,DIM],
                        lifetime="lane-zero recurrence publication through epilogue")
    bar_BC_full = shared_region("mbarrier", [1])
    bar_state_in_empty = shared_region("mbarrier", [1])
    bar_state_in_full = shared_region("mbarrier", [1])
    bar_out_ready = shared_region("mbarrier", [1])
    bar_epilogue_done = shared_region("mbarrier", [1])

sram = shared_region(GroupStorage, [3], alignment=128)

# Every active group owns distinct B/C/dt/state/x/out and all five barriers.
# There is no aliasing or cross-group reuse.  Struct padding and barrier offsets
# must follow the CUDA ABI exactly; the TIRx body uses one raw aligned allocation
# with mechanically computed byte offsets.

# ===========================================================================
# Entry coordinates, active tail, index selection, barrier initialization
# source: vertical.cuh:399-432
# ===========================================================================

batch_idx, chunk_idx = cta_id(extents=(batch, ceil_div(nheads,3)))
lane, warp = thread_id(extents=(32,16))

heads = [chunk_idx*3 + g for g in static_range(3)]
num_active_groups = clamp(nheads - chunk_idx*3, 0, 3)

if state_batch_indices is present:
    if STATE_INDEX_DTYPE == "i64":
        state_batch = ptx(
            "ld.global.b64", address(state_batch_indices[batch_idx]))
    else:
        state_batch_i32 = ptx(
            "ld.global.s32", address(state_batch_indices[batch_idx]))
        state_batch = ptx("cvt.s64.s32", state_batch_i32)
else:
    state_batch = ptx("cvt.u64.u32", batch_idx)
pad_slot_i64 = ptx("cvt.s64.s32", pad_slot_id)
is_pad = ptx("setp.eq.s64", state_batch, pad_slot_i64)

# Extent: one selected index load/conversion per CTA thread when the pointer is
# present; otherwise one CTA-coordinate widening.  Frozen i64 modules issue
# `ld.global.b64`; the one i32-index module issues `ld.global.s32` followed by
# `cvt.s64.s32`.

if warp == 0 and lane == 0:
    for g in runtime_range(num_active_groups):
        ptx("mbarrier.init.shared.b64", address(sram[g].bar_BC_full), 160)
        ptx("mbarrier.init.shared.b64", address(sram[g].bar_state_in_empty), 160)
        ptx("mbarrier.init.shared.b64", address(sram[g].bar_state_in_full), 129)
        ptx("mbarrier.init.shared.b64", address(sram[g].bar_out_ready), 160)
        ptx("mbarrier.init.shared.b64", address(sram[g].bar_epilogue_done), 160)
ptx("bar.sync", barrier_id=0)

# Exactly five `mbarrier.init.shared.b64` issues per active group by one thread;
# then exactly one `bar.sync 0` issue per thread.  Inactive groups have no
# initialization and no later arrival.

if warp < 12:
    role = "compute"
    g = warp // 4
    compute_warp = warp % 4
elif warp < 15:
    role = "tma"
    g = warp - 12
else:
    role = "epilogue"

dispatch_runtime_pad_specialization(is_pad):
    if role == "compute" and g < num_active_groups:
        role_update_state(IS_PAD=is_pad, group=g, compute_warp=compute_warp)
    elif role == "tma" and g < num_active_groups:
        kv_group = heads[g] // HEADS_PER_GROUP
        role_load(IS_PAD=is_pad, group=g, kv_group=kv_group)
    elif role == "epilogue":
        role_epilogue(num_active_groups)

# ===========================================================================
# TMA producer role: warp 12+g
# source: vertical.cuh:96-138
# ===========================================================================

def role_load(IS_PAD, group, kv_group):
    gs = sram[group]
    head = heads[group]

    # B and C are valid and required even for pad state slots.
    if lane == 0:
        ptx("cp.async.bulk.tensor.4d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes",
            shared_dst=address(gs.B[0,0]), grid_constant_desc=tensor_B,
            coords=(0,kv_group,0,batch_idx),
            shared_cta_barrier=address(gs.bar_BC_full))
        ptx("cp.async.bulk.tensor.4d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes",
            shared_dst=address(gs.C[0,0]), grid_constant_desc=tensor_C,
            coords=(0,kv_group,0,batch_idx),
            shared_cta_barrier=address(gs.bar_BC_full))
        bytes_BC = 2*NTOKENS*DSTATE*sizeof(INPUT_DTYPE)
        ptx("mbarrier.expect_tx.relaxed.cta.shared::cta.b64",
            address(gs.bar_BC_full), bytes_BC)
        bc_token = ptx("mbarrier.arrive.release.cta.shared::cta.b64",
                       address(gs.bar_BC_full), drop_count=32)
        # One expect-tx and one release-arrive issue by TMA lane zero.  The
        # returned token is dead here; transaction completion advances bytes.

    # All 32 load-warp lanes arrive at the empty barrier.  Together with the
    # four compute warps' 128 pre-arrivals, this completes its 160 arrivals.
    empty_token = ptx("mbarrier.arrive.shared::cta.b64",
                      address(gs.bar_state_in_empty), arrival_count=1)
    empty_ready = 0
    while empty_ready == 0:
        empty_pred = ptx("mbarrier.try_wait.shared::cta.b64",
                         address(gs.bar_state_in_empty), empty_token)
        empty_ready = ptx("selp.b32", 1, 0, empty_pred)
    # All 32 producer lanes issue one arrive and a token-based tight try-wait
    # loop.  No parity wait and no nanosleep is emitted.

    if lane == 0:
        if not IS_PAD:
            ptx("cp.async.bulk.tensor.4d.shared::cluster.global.tile."
                "mbarrier::complete_tx::bytes",
                shared_dst=address(gs.state_in[0,0]),
                grid_constant_desc=tensor_state,
                coords=(0,0,head,state_batch),
                shared_cta_barrier=address(gs.bar_state_in_full))
        ptx("cp.async.bulk.tensor.4d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes",
            shared_dst=address(gs.x[0,0,0]), grid_constant_desc=tensor_x,
            coords=(0,head,0,batch_idx),
            shared_cta_barrier=address(gs.bar_state_in_full))
        transaction_bytes = NTOKENS*DIM*sizeof(INPUT_DTYPE)
        if not IS_PAD:
            transaction_bytes += DIM*DSTATE*sizeof(STATE_DTYPE)
        full_token_from_producer = ptx(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
            address(gs.bar_state_in_full), transaction_bytes)
        # One combined count-one arrive/expect-tx issue by lane zero.  This is
        # not the split expect_tx + drop-count-32 BC sequence.

# Each explicit `cp.async.bulk.tensor.4d...` above is one issue by lane zero.
# Operands are `[shared_dst], [grid_constant_descriptor,
# {coord0,coord1,coord2,coord3}], [shared_cta_barrier]`.  B and C therefore issue
# two instructions; x issues one; non-pad state issues one.  The two B/C
# transactions share the split expect-tx/release-arrive byte count.  State and x
# share the combined full-barrier byte count; pad omits state and its bytes.

# ===========================================================================
# Compute role prologue and synchronization
# source: vertical.cuh:185-248
# ===========================================================================

def role_update_state(IS_PAD, group, compute_warp):
    gs = sram[group]
    head = heads[group]

    rand_seed = 0
    if PHILOX_ROUNDS > 0 and not IS_PAD:
        if rand_seed_ptr is present:
            rand_seed = ptx("ld.global.b64", address(rand_seed_ptr[0]))
        # One optional `ld.global.b64` per active non-pad compute lane.
    # All thirteen PHILOX_ROUNDS=0 frozen modules have no rand_seed pointer
    # access at all; the family-F pad body also DCEs it.

    # icache_idx fallback is zero-instruction semantic bookkeeping.  Every pad
    # body DCEs the pointer test/load/conversion because it cannot store state.
    icache_idx = state_batch
    if not IS_PAD and intermediate_state_indices is present:
        if STATE_INDEX_DTYPE == "i64":
            icache_idx = ptx(
                "ld.global.b64", address(intermediate_state_indices[batch_idx]))
        else:
            icache_i32 = ptx(
                "ld.global.s32", address(intermediate_state_indices[batch_idx]))
            icache_idx = ptx("cvt.s64.s32", icache_i32)
    philox_head_term_i32 = absent
    philox_state_batch_term_i64 = absent
    if PHILOX_ROUNDS == 10 and not IS_PAD:
        # The source i64 state_ptr_offset is implicitly narrowed by the helper.
        # Before either barrier wait, frozen PTX emits only this head narrowing,
        # i64 state-batch multiply, and head*8192 shift.  Product narrowing and
        # the final i32 base add deliberately remain on the far side of the
        # full state/x wait to bound their live ranges.
        head_i32 = ptx("cvt.u32.u64", head)
        philox_state_batch_term_i64 = ptx(
            "mul.lo.s64", state_stride_batch, state_batch)
        philox_head_term_i32 = ptx("shl.b32", head_i32, 13)

    # These ordinary global loads are intentionally issued before waits.
    A_value = ptx("ld.global.b32", address(A[head]))
    D_value = 0.0
    if D is present:
        if WEIGHT_DTYPE == "f32":
            D_value = ptx("ld.global.b32", address(D[head]))
        else:
            D_bits = ptx("ld.global.b16", address(D[head]))
            D_value = ptx("cvt.f32.bf16", D_bits)
    dt_bias_value = 0.0
    if dt_bias is present:
        if WEIGHT_DTYPE == "f32":
            dt_bias_value = ptx("ld.global.b32", address(dt_bias[head]))
        else:
            bias_bits = ptx("ld.global.b16", address(dt_bias[head]))
            dt_bias_value = ptx("cvt.f32.bf16", bias_bits)

    # Per active compute lane: one A `ld.global.b32`; optional D and bias each
    # issue either one `ld.global.b32` or `ld.global.b16` + `cvt.f32.bf16`.

    # Each of the 128 compute lanes contributes one early arrival, releasing
    # the load warp only after it contributes the remaining 32 arrivals.
    empty_release_token = ptx("mbarrier.arrive.shared::cta.b64",
                              address(gs.bar_state_in_empty), arrival_count=1)

    # Each compute lane arrives and waits.  Combined with the load warp's
    # arrival_count=32 expect-tx arrival, this waits for both B and C bytes.
    bc_wait_token = ptx("mbarrier.arrive.shared::cta.b64",
                        address(gs.bar_BC_full), arrival_count=1)
    bc_ready = 0
    while bc_ready == 0:
        bc_pred = ptx("mbarrier.try_wait.shared::cta.b64",
                      address(gs.bar_BC_full), bc_wait_token)
        bc_ready = ptx("selp.b32", 1, 0, bc_pred)

    # Token steps are striped over the four compute warps; only lane zero writes.
    for step in range(compute_warp, NTOKENS, 4):
        if lane == 0:
            dt_address = address(
                dt[batch_idx*dt_stride_batch + step*dt_stride_mtp + head])
            if WEIGHT_DTYPE == "f32":
                dt_value = ptx("ld.global.b32", dt_address)
            else:
                dt_bits = ptx("ld.global.b16", dt_address)
                dt_value = ptx("cvt.f32.bf16", dt_bits)
            dt_value = ptx("add.ftz.f32", dt_value, dt_bias_value)
            if dt_softplus:
                if dt_value <= 20.0:
                    softplus_log2_arg = ptx(
                        "mul.ftz.f32", dt_value, 1.4426950408889634)
                    softplus_exp = ptx(
                        "ex2.approx.ftz.f32", softplus_log2_arg)
                    softplus_one_plus = ptx(
                        "add.ftz.f32", 1.0, softplus_exp)
                    softplus_log2 = ptx(
                        "lg2.approx.ftz.f32", softplus_one_plus)
                    dt_value = ptx(
                        "mul.ftz.f32", softplus_log2, 0.6931471805599453)
            ptx("st.shared.b32", address(gs.dt[step]), dt_value)

    # For every lane-zero-owned token: one scalar global dt load (b32, or b16
    # followed by cvt.f32.bf16), one add.ftz, the threshold predicate, and—only
    # on the <=20 softplus arm—the exact five fast-math issues above.  Publication
    # is one `st.shared.b32`.

    # All 128 compute lanes arrive and wait.  The load warp lane-zero expect-tx
    # arrival is the 129th arrival and completes only after state/x bytes land.
    full_wait_token = ptx("mbarrier.arrive.shared::cta.b64",
                          address(gs.bar_state_in_full), arrival_count=1)
    full_ready = 0
    while full_ready == 0:
        full_pred = ptx("mbarrier.try_wait.shared::cta.b64",
                        address(gs.bar_state_in_full), full_wait_token)
        full_ready = ptx("selp.b32", 1, 0, full_pred)

    philox_state_base_i32 = absent
    philox_seed_lo = absent
    philox_seed_hi = absent
    if PHILOX_ROUNDS == 10 and not IS_PAD:
        # Frozen schedule immediately after the full wait: signed widening of
        # head*8192 for address plumbing, then state-product narrowing and the
        # i32 Philox base.  Seed split is exactly cvt-low plus mov-high.
        philox_head_term_i64 = ptx(
            "cvt.s64.s32", philox_head_term_i32)
        philox_state_batch_term_i32 = ptx(
            "cvt.u32.u64", philox_state_batch_term_i64)
        philox_state_base_i32 = ptx(
            "add.s32", philox_head_term_i32, philox_state_batch_term_i32)
        philox_seed_lo = ptx("cvt.u32.u64", rand_seed)
        discard, philox_seed_hi = ptx("mov.b64", split=rand_seed)

    # The frozen CSE extent is shape-selected.  Every PHILOX0 entry has two
    # static sites, one in each emitted pad/non-pad compute body; family F has
    # only the one pad-body site here.  Its twelve non-pad sites live in the
    # three split recurrence/store CFG regions documented below.
    if PHILOX_ROUNDS == 0 or IS_PAD:
        lane_zero_pred = ptx("setp.eq.b32", lane, 0)
        lane_zero_value = ptx("selp.f32", 1.0, 0.0, lane_zero_pred)
    else:
        lane_zero_value = absent

    compute_passes(IS_PAD, gs, head, state_batch, icache_idx,
                   A_value, D_value, philox_seed_lo, philox_seed_hi,
                   philox_state_base_i32,
                   lane_zero_value)

# ===========================================================================
# Four-row recurrence passes and state stores
# source: vertical.cuh:249-315
# ===========================================================================

def compute_passes(IS_PAD, gs, head, state_batch, icache_idx,
                   A_value, D_value, philox_seed_lo, philox_seed_hi,
                   philox_state_base_i32,
                   lane_zero_value):
    pass_idx = ptx("mov.b32", 0)
    PASS_LOOP:  # retained runtime-counter loop, extent NUM_PASSES
        if PHILOX_ROUNDS == 10 and not IS_PAD:
            # Exact family-F pass/compute-row and lane front end is formed
            # before the four shared state-row loads.  DIM64 gives
            # rowsPerWarp16; the mask is the frozen compiler form.
            pass_times_4 = ptx("shl.b32", pass_idx, 2)
            compute_times_16 = ptx("shl.b32", compute_warp, 4)
            compute_row = ptx("and.b32", compute_times_16, 48)
            row_offset = ptx("add.s32", pass_times_4, compute_row)
            first_row_times_128 = ptx("shl.b32", row_offset, 7)
            lane_times_4 = ptx("shl.b32", lane, 2)
            row_lane_i32 = ptx(
                "add.s32", first_row_times_128, lane_times_4)
        else:
            # This includes the family-F pad specialization: it needs the
            # semantic recurrence row but emits no Philox `.loc 251` front end.
            row_offset = (
                compute_warp*ROWS_PER_WARP + pass_idx*ROWS_PER_PASS)
            row_lane_i32 = absent
        r_state = reg("f32", [ROWS_PER_PASS,STATE_VALUES_PER_THREAD])

        for wr in static_range(4):
            dd = row_offset + wr
            if IS_PAD:
                for ii in static_range(STATE_VALUES_PER_THREAD):
                    r_state[wr,ii] = 0.0
            elif DSTATE == 64:
                static_assert(STATE_DTYPE == "bf16")
                state_pack = ptx(
                    "ld.shared.v2.b16",
                    address(gs.state_in[0,dd*DSTATE + lane*2]))
                for ii in static_range(2):
                    r_state[wr,ii] = ptx(
                        "cvt.f32.bf16", extract_u16(state_pack,ii))
            elif DSTATE == 96:
                static_assert(STATE_DTYPE == "bf16")
                # BF16 DSTATE96 issues three
                # independent loads, never a synthetic six-byte vector load.
                for ii in static_range(3):
                    state_bits = ptx(
                        "ld.shared.b16",
                        address(gs.state_in[0,dd*DSTATE + lane*3 + ii]))
                    r_state[wr,ii] = ptx("cvt.f32.bf16", state_bits)
            elif STATE_DTYPE in ("bf16","f16"):
                state_pack = ptx(
                    "ld.shared.v4.b16",
                    address(gs.state_in[0,dd*DSTATE + lane*4]))
                for ii in static_range(4):
                    r_state[wr,ii] = ptx(
                        "cvt.f32.bf16" if STATE_DTYPE=="bf16" else "cvt.f32.f16",
                        extract_u16(state_pack,ii))
            else:  # frozen FP32 state has DSTATE128
                state_pack = ptx(
                    "ld.shared.v4.b32",
                    address(gs.state_in[0,dd*DSTATE + lane*4]))
                for ii in static_range(4):
                    r_state[wr,ii] = extract_b32_as_f32(state_pack,ii)

        # Family F performs one four-counter/CSE batch per runtime pass after
        # all four rows are resident and before either the token loop or any
        # state-destination predicate.  For non-pad it is unconditional even
        # when both destinations are absent.  PHILOX0 and pad issue no batch.
        rand_by_wr = absent
        philox_token_step_zero = absent
        if PHILOX_ROUNDS == 10 and not IS_PAD:
            rand_by_wr, philox_token_step_zero = generate_four_row_rand_for_pass(
                philox_seed_lo, philox_seed_hi,
                philox_state_base_i32, row_lane_i32)

        if PHILOX_ROUNDS == 10 and not IS_PAD:
            # Cross-region CSE: the table's one physical r1830 zero is both
            # Philox counter word c2 and the runtime token-loop initializer.
            step = philox_token_step_zero
        else:
            # PHILOX0 and the family-F pad body have no four-counter table.
            step = ptx("mov.b32", 0)
        TOKEN_LOOP:  # one retained runtime-counter loop when NTOKENS > 1
            dt_value = ptx("ld.shared.b32", address(gs.dt[step]))
            dA_product = ptx("mul.ftz.f32", A_value, dt_value)
            dA_log2_arg = ptx(
                "mul.ftz.f32", dA_product, 1.4426950408889634)
            dA = ptx("ex2.approx.ftz.f32", dA_log2_arg)

            # B and C issue once per token/lane and are reused for all four wr.
            col_base = lane*STATE_VALUES_PER_THREAD
            if DSTATE == 64:
                B_bits = ptx("ld.shared.v2.b16", address(gs.B[step,col_base]))
                C_bits = ptx("ld.shared.v2.b16", address(gs.C[step,col_base]))
                B_values = [ptx("cvt.f32.bf16",extract_u16(B_bits,i)) for i in static_range(2)]
                C_values = [ptx("cvt.f32.bf16",extract_u16(C_bits,i)) for i in static_range(2)]
            elif DSTATE == 96:
                B_values = [
                    ptx("cvt.f32.bf16",ptx("ld.shared.b16",address(gs.B[step,col_base+i])))
                    for i in static_range(3)]
                C_values = [
                    ptx("cvt.f32.bf16",ptx("ld.shared.b16",address(gs.C[step,col_base+i])))
                    for i in static_range(3)]
            else:
                B_bits = ptx("ld.shared.v4.b16", address(gs.B[step,col_base]))
                C_bits = ptx("ld.shared.v4.b16", address(gs.C[step,col_base]))
                B_values = [ptx("cvt.f32.bf16",extract_u16(B_bits,i)) for i in static_range(4)]
                C_values = [ptx("cvt.f32.bf16",extract_u16(C_bits,i)) for i in static_range(4)]

            for wr in static_range(4):
                dd = row_offset + wr
                x_bits = ptx("ld.shared.b16", address(gs.x[0,step,dd]))
                x_value = ptx("cvt.f32.bf16", x_bits)
                D_times_x = ptx("mul.ftz.f32", D_value, x_value)

                if PHILOX_ROUNDS == 10 and not IS_PAD:
                    # This notation denotes the exact emitted CFG, not a
                    # dynamically repeated helper: -O3 creates three split
                    # non-pad recurrence/store regions, and static wr unrolling
                    # puts one setp/selp pair in each region for each wr.
                    # Thus the entry has 3*4=12 non-pad static sites.
                    local_lane_zero_pred = ptx("setp.eq.b32", lane, 0)
                    local_lane_zero_value = ptx(
                        "selp.f32", 1.0, 0.0, local_lane_zero_pred)
                else:
                    local_lane_zero_value = lane_zero_value

                for ii in static_range(STATE_VALUES_PER_THREAD):
                    dB = ptx("mul.ftz.f32", B_values[ii], dt_value)
                    bx = ptx("mul.ftz.f32", dB, x_value)
                    r_state[wr,ii] = ptx(
                        "fma.rn.ftz.f32", r_state[wr,ii], dA, bx)
                    if ii == 0:
                        state_C = ptx(
                            "mul.ftz.f32", r_state[wr,ii], C_values[ii])
                        out_value = ptx(
                            "fma.rn.ftz.f32",
                            D_times_x, local_lane_zero_value, state_C)
                    else:
                        out_value = ptx(
                            "fma.rn.ftz.f32",
                            r_state[wr,ii], C_values[ii], out_value)

                # Full-warp sum; values from lower lanes remain live but only
                # lane zero publishes the completed reduction.
                for delta in (16,8,4,2,1):
                    peer, valid = ptx(
                        "shfl.sync.down.b32", out_value, delta,
                        clamp=31, member_mask=-1)
                    out_value = ptx("add.ftz.f32", out_value, peer)
                if lane == 0:
                    ptx("st.shared.b32", address(gs.out[step,dd]), out_value)

                write_intermediate = intermediate_states is present and not IS_PAD
                write_final = (
                    step == NTOKENS-1 and update_state and not IS_PAD)
                if write_intermediate or write_final:
                    intermediate_destination = absent
                    if write_intermediate:
                        intermediate_destination = (
                            intermediate_states
                            + icache_idx*intermediate_state_stride_batch
                            + step*nheads*DIM*DSTATE
                            + head*DIM*DSTATE + dd*DSTATE
                            + lane*STATE_VALUES_PER_THREAD)
                    final_destination = absent
                    if write_final:
                        # The CUDA vertical kernel writes state_batch;
                        # dst_state_batch_indices is not read.
                        final_destination = (
                            state + state_batch*state_stride_batch
                            + head*DIM*DSTATE + dd*DSTATE
                            + lane*STATE_VALUES_PER_THREAD)
                    store_state_row(
                        values=r_state[wr],
                        intermediate_destination=intermediate_destination,
                        final_destination=final_destination,
                        wr=wr,
                        rand_for_wr=(rand_by_wr[wr]
                                     if PHILOX_ROUNDS == 10 and not IS_PAD
                                     else absent))

            if NTOKENS > 1:
                step = ptx("add.s32", step, 1)
                more_tokens = ptx("setp.ne.b32", step, NTOKENS)
                ptx("@pred bra", TOKEN_LOOP, predicate=more_tokens)
            # NTOKENS==1 emits one token body; the token add, predicate, and
            # backedge are completely DCE.  It does not become a static loop.

        pass_idx = ptx("add.s32", pass_idx, 1)
        more_passes = ptx("setp.ne.b32", pass_idx, NUM_PASSES)
        ptx("@pred bra", PASS_LOOP, predicate=more_passes)

# Lane-indicator extent per frozen entry is exact: PHILOX0 has two invariant
# `setp.eq.b32`/`selp.f32` sites (pad and non-pad bodies).  PHILOX10 has one
# pad-body site plus twelve non-pad sites: three destination-split recurrence
# CFG regions times four statically unrolled wr rows, for 13 total.  Runtime
# pass/token iterations revisit their region's instructions; they do not add
# static sites.  Per `(pass,step,wr,lane)`: one scalar BF16 x load/conversion,
# one all-lane `mul.ftz.f32 D*x`,
# STATE_VALUES_PER_THREAD dB muls, bx muls, and state FMAs, one state*C mul plus
# one first-output FMA, then STATE_VALUES_PER_THREAD-1 output FMAs.  Reduction
# is exactly five `shfl.sync.down.b32` with delta 16/8/4/2/1, clamp 31 and member
# mask -1, each immediately followed by `add.ftz.f32`.

# ===========================================================================
# Direct state conversion and vector store
# source: vertical.cuh:140-177 and conversion.cuh:95-249
# ===========================================================================

def store_state_row(values, intermediate_destination, final_destination,
                    wr, rand_for_wr):
    I = intermediate_destination is present
    F = final_destination is present
    if PHILOX_ROUNDS == 0:
        if STATE_DTYPE in ("bf16","f16") and DSTATE == 128 and NTOKENS > 1:
            # Family A has three separate emitted conversion bodies.  The
            # conversions below are deliberately inside each path.
            if I and not F:
                packed = reg(STATE_DTYPE, [4])
                for k in static_range(4):
                    packed[k] = ptx(
                        "cvt.rn.bf16.f32" if STATE_DTYPE=="bf16"
                        else "cvt.rn.f16.f32", values[k])
                ptx("st.v4.b16", address(intermediate_destination), tuple(packed))
            elif F and not I:
                packed = reg(STATE_DTYPE, [4])
                for k in static_range(4):
                    packed[k] = ptx(
                        "cvt.rn.bf16.f32" if STATE_DTYPE=="bf16"
                        else "cvt.rn.f16.f32", values[k])
                ptx("st.global.v4.b16", address(final_destination), tuple(packed))
            else:  # I and F: distinct last-step body; one result feeds both
                packed = reg(STATE_DTYPE, [4])
                for k in static_range(4):
                    packed[k] = ptx(
                        "cvt.rn.bf16.f32" if STATE_DTYPE=="bf16"
                        else "cvt.rn.f16.f32", values[k])
                word1 = ptx("mov.b32", pair=(packed[2],packed[3]))
                word0 = ptx("mov.b32", pair=(packed[0],packed[1]))
                ptx("st.v2.b32", address(intermediate_destination), (word0,word1))
                ptx("st.global.v2.b32", address(final_destination), (word0,word1))

        elif STATE_DTYPE == "bf16" and DSTATE == 128:  # family B, NTOKENS1
            static_assert(NTOKENS == 1)
            if not I:  # final-only body
                packed = reg("bf16", [4])
                for k in static_range(4):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                ptx("st.global.v4.b16", address(final_destination), tuple(packed))
            else:
                # One intermediate-present body serves I&&!F and I&&F.  Its
                # exact pack order is low word first, then high word, in both
                # cases; family B never falls through to family A's order.
                packed = reg("bf16", [4])
                for k in static_range(4):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                word0 = ptx("mov.b32", pair=(packed[0],packed[1]))
                word1 = ptx("mov.b32", pair=(packed[2],packed[3]))
                ptx("st.v2.b32", address(intermediate_destination), (word0,word1))
                if F:
                    ptx("st.global.v2.b32", address(final_destination), (word0,word1))

        elif STATE_DTYPE == "bf16" and DSTATE == 64:  # family C
            if I and not F:
                packed = reg("bf16", [2])
                for k in static_range(2):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                ptx("st.v2.b16", address(intermediate_destination), tuple(packed))
            elif F and not I:
                packed = reg("bf16", [2])
                for k in static_range(2):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                ptx("st.global.v2.b16", address(final_destination), tuple(packed))
            else:
                packed = reg("bf16", [2])
                for k in static_range(2):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                word0 = ptx("mov.b32", pair=(packed[0],packed[1]))
                ptx("st.b32", address(intermediate_destination), word0)
                ptx("st.global.b32", address(final_destination), word0)

        elif STATE_DTYPE == "bf16" and DSTATE == 96:  # family D
            if I and not F:
                packed = reg("bf16", [3])
                for k in static_range(3):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                for k in static_range(3):
                    ptx("st.b16", address(intermediate_destination)+2*k, packed[k])
            elif F and not I:
                packed = reg("bf16", [3])
                for k in static_range(3):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                for k in static_range(3):
                    ptx("st.global.b16", address(final_destination)+2*k, packed[k])
            else:
                packed = reg("bf16", [3])
                for k in static_range(3):
                    packed[k] = ptx("cvt.rn.bf16.f32", values[k])
                for k in static_range(3):
                    ptx("st.b16", address(intermediate_destination)+2*k, packed[k])
                for k in static_range(3):
                    ptx("st.global.b16", address(final_destination)+2*k, packed[k])

        else:  # family E: frozen FP32/DSTATE128
            static_assert(STATE_DTYPE == "f32" and DSTATE == 128)
            if I and not F:
                packed = tuple(values[k] for k in static_range(4))
                ptx("st.v4.b32", address(intermediate_destination), tuple(packed))
            elif F and not I:
                packed = tuple(values[k] for k in static_range(4))
                ptx("st.global.v4.b32", address(final_destination), tuple(packed))
            else:
                packed = tuple(values[k] for k in static_range(4))
                word1 = ptx("mov.b64", pair=(packed[2],packed[3]))
                word0 = ptx("mov.b64", pair=(packed[0],packed[1]))
                ptx("st.v2.b64", address(intermediate_destination), (word0,word1))
                ptx("st.global.v2.b64", address(final_destination), (word0,word1))
    else:
        # Family F has separate final-only and intermediate-present conversion
        # bodies.  rand_for_wr was generated unconditionally before TOKEN_LOOP;
        # neither body calls Philox, and both reuse that per-pass row value.
        static_assert(STATE_DTYPE == "f16" and DSTATE == 128)
        if F and not I:  # final-only has no half extraction or b32 repack
            pair0 = ptx("cvt.rs.f16x2.f32",
                        ptx_a=values[1], ptx_b=values[0], rbits=rand_for_wr[0])
            pair1 = ptx("cvt.rs.f16x2.f32",
                        ptx_a=values[3], ptx_b=values[2], rbits=rand_for_wr[1])
            tmp0 = ptx("mov.b64", pair=(pair0,discard))
            word0, discard = ptx("mov.b64", split=tmp0)
            ptx("st.global.v2.b32", address(final_destination), (word0,pair1))
        else:  # intermediate-present episode, reused by final when F
            pair0 = ptx("cvt.rs.f16x2.f32",
                        ptx_a=values[1], ptx_b=values[0], rbits=rand_for_wr[0])
            pair1 = ptx("cvt.rs.f16x2.f32",
                        ptx_a=values[3], ptx_b=values[2], rbits=rand_for_wr[1])
            lo1 = ptx("cvt.u16.u32", pair1)
            hi1 = ptx("mov.b32", high_u16_destination=True, source=pair1)
            tmp0 = ptx("mov.b64", pair=(pair0,discard))
            word0, discard = ptx("mov.b64", split=tmp0)
            ptx("st.v2.b32", address(intermediate_destination), (word0,pair1))
            if F:
                word1 = ptx("mov.b32", pair=(lo1,hi1))
                if wr < 3:
                    tmp0f = ptx("mov.b64", pair=(pair0,discard))
                    word0f, discard = ptx("mov.b64", split=tmp0f)
                else:
                    word0f = pair0
                ptx("st.global.v2.b32", address(final_destination), (word0f,word1))

        # Per HEADS_PER_GROUP entry, frozen family F has four intermediate
        # st.v2.b32, eight final st.global.v2.b32 and sixteen static cvt.rs
        # sites.  A both path reuses one branch-local conversion result.

def generate_four_row_rand_for_pass(seed_lo, seed_hi,
                                    philox_state_base_i32, row_lane_i32):
    # The function body is the instruction-by-instruction table below; it is
    # not a runtime table interpreter or a compound helper.  Each table row is
    # mechanically one `ptx(mnemonic,dst,src...)` issue in listed order.  The
    # physical names are from the first frozen HPG entry; all seven entries are
    # register-renamed isomorphic.  Input bindings are r38=state base,
    # r39=seed_lo (`cvt.u32.u64`), r40=seed_hi (`mov.b64 {_,hi}`), and
    # r467=row/lane term.  Four segments are deliberately interleaved as
    # counter/sign-high, folded rounds, next counter/sign-high, ... .
    FROZEN_FOUR_COUNTER_PTX_32562_32857 = r"""
add.s32 r472, r467, r38
shr.s32 r473, r472, 31
xor.b32 r474, r473, r39
mov.b32 r1830, 0
mov.b32 r475, -845247145
mul.hi.u32 r476, r475, r1830
xor.b32 r477, r474, r476
mov.b32 r478, -766435501
mul.hi.u32 r479, r478, r472
xor.b32 r480, r479, r40
mul.lo.s32 r481, r472, -766435501
mul.hi.u32 r482, r475, r480
add.s32 r483, r39, -1640531527
xor.b32 r484, r482, r483
mul.hi.u32 r485, r478, r477
xor.b32 r486, r485, r481
add.s32 r487, r40, -1150833019
xor.b32 r488, r486, r487
mul.lo.s32 r489, r480, -845247145
mul.lo.s32 r490, r477, -766435501
mul.hi.u32 r491, r475, r488
xor.b32 r492, r489, r491
add.s32 r493, r39, 1013904242
xor.b32 r494, r492, r493
mul.hi.u32 r495, r478, r484
xor.b32 r496, r495, r490
add.s32 r497, r40, 1993301258
xor.b32 r498, r496, r497
mul.lo.s32 r499, r488, -845247145
mul.lo.s32 r500, r484, -766435501
mul.hi.u32 r501, r475, r498
xor.b32 r502, r499, r501
add.s32 r503, r39, -626627285
xor.b32 r504, r502, r503
mul.hi.u32 r505, r478, r494
xor.b32 r506, r505, r500
add.s32 r507, r40, 842468239
xor.b32 r508, r506, r507
mul.lo.s32 r509, r498, -845247145
mul.lo.s32 r510, r494, -766435501
mul.hi.u32 r511, r475, r508
xor.b32 r512, r509, r511
add.s32 r513, r39, 2027808484
xor.b32 r514, r512, r513
mul.hi.u32 r515, r478, r504
xor.b32 r516, r515, r510
add.s32 r517, r40, -308364780
xor.b32 r518, r516, r517
mul.lo.s32 r519, r508, -845247145
mul.lo.s32 r520, r504, -766435501
mul.hi.u32 r521, r475, r518
xor.b32 r522, r519, r521
add.s32 r523, r39, 387276957
xor.b32 r524, r522, r523
mul.hi.u32 r525, r478, r514
xor.b32 r526, r525, r520
add.s32 r527, r40, -1459197799
xor.b32 r528, r526, r527
mul.lo.s32 r529, r518, -845247145
mul.lo.s32 r530, r514, -766435501
mul.hi.u32 r531, r475, r528
xor.b32 r532, r529, r531
add.s32 r533, r39, -1253254570
xor.b32 r534, r532, r533
mul.hi.u32 r535, r478, r524
xor.b32 r536, r535, r530
add.s32 r537, r40, 1684936478
xor.b32 r538, r536, r537
mul.lo.s32 r539, r528, -845247145
mul.lo.s32 r540, r524, -766435501
mul.hi.u32 r541, r475, r538
xor.b32 r542, r539, r541
add.s32 r543, r39, 1401181199
xor.b32 r544, r542, r543
mul.hi.u32 r545, r478, r534
xor.b32 r546, r545, r540
add.s32 r547, r40, 534103459
xor.b32 r548, r546, r547
mul.lo.s32 r549, r534, -766435501
mul.hi.u32 r550, r478, r544
xor.b32 r551, r550, r549
add.s32 r552, r40, -616729560
xor.b32 r553, r551, r552
mul.lo.s32 r554, r548, -845247145
mul.hi.u32 r555, r475, r553
xor.b32 r556, r554, r555
add.s32 r557, r39, -1879881855
xor.b32 r815, r556, r557
mul.lo.s32 r817, r553, -845247145
add.s32 r558, r472, 128
shr.s32 r559, r558, 31
xor.b32 r560, r559, r39
xor.b32 r561, r560, r476
mul.hi.u32 r562, r478, r558
xor.b32 r563, r562, r40
add.s32 r564, r481, 680503680
mul.hi.u32 r565, r475, r563
xor.b32 r566, r565, r483
mul.hi.u32 r567, r478, r561
xor.b32 r568, r567, r564
xor.b32 r569, r568, r487
mul.lo.s32 r570, r563, -845247145
mul.lo.s32 r571, r561, -766435501
mul.hi.u32 r572, r475, r569
xor.b32 r573, r570, r572
xor.b32 r574, r573, r493
mul.hi.u32 r575, r478, r566
xor.b32 r576, r575, r571
xor.b32 r577, r576, r497
mul.lo.s32 r578, r569, -845247145
mul.lo.s32 r579, r566, -766435501
mul.hi.u32 r580, r475, r577
xor.b32 r581, r578, r580
xor.b32 r582, r581, r503
mul.hi.u32 r583, r478, r574
xor.b32 r584, r583, r579
xor.b32 r585, r584, r507
mul.lo.s32 r586, r577, -845247145
mul.lo.s32 r587, r574, -766435501
mul.hi.u32 r588, r475, r585
xor.b32 r589, r586, r588
xor.b32 r590, r589, r513
mul.hi.u32 r591, r478, r582
xor.b32 r592, r591, r587
xor.b32 r593, r592, r517
mul.lo.s32 r594, r585, -845247145
mul.lo.s32 r595, r582, -766435501
mul.hi.u32 r596, r475, r593
xor.b32 r597, r594, r596
xor.b32 r598, r597, r523
mul.hi.u32 r599, r478, r590
xor.b32 r600, r599, r595
xor.b32 r601, r600, r527
mul.lo.s32 r602, r593, -845247145
mul.lo.s32 r603, r590, -766435501
mul.hi.u32 r604, r475, r601
xor.b32 r605, r602, r604
xor.b32 r606, r605, r533
mul.hi.u32 r607, r478, r598
xor.b32 r608, r607, r603
xor.b32 r609, r608, r537
mul.lo.s32 r610, r601, -845247145
mul.lo.s32 r611, r598, -766435501
mul.hi.u32 r612, r475, r609
xor.b32 r613, r610, r612
xor.b32 r614, r613, r543
mul.hi.u32 r615, r478, r606
xor.b32 r616, r615, r611
xor.b32 r617, r616, r547
mul.lo.s32 r618, r606, -766435501
mul.hi.u32 r619, r478, r614
xor.b32 r620, r619, r618
xor.b32 r621, r620, r552
mul.lo.s32 r622, r617, -845247145
mul.hi.u32 r623, r475, r621
xor.b32 r624, r622, r623
xor.b32 r874, r624, r557
mul.lo.s32 r876, r621, -845247145
add.s32 r625, r472, 256
shr.s32 r626, r625, 31
xor.b32 r627, r626, r39
xor.b32 r628, r627, r476
mul.hi.u32 r629, r478, r625
xor.b32 r630, r629, r40
add.s32 r631, r481, 1361007360
mul.hi.u32 r632, r475, r630
xor.b32 r633, r632, r483
mul.hi.u32 r634, r478, r628
xor.b32 r635, r634, r631
xor.b32 r636, r635, r487
mul.lo.s32 r637, r630, -845247145
mul.lo.s32 r638, r628, -766435501
mul.hi.u32 r639, r475, r636
xor.b32 r640, r637, r639
xor.b32 r641, r640, r493
mul.hi.u32 r642, r478, r633
xor.b32 r643, r642, r638
xor.b32 r644, r643, r497
mul.lo.s32 r645, r636, -845247145
mul.lo.s32 r646, r633, -766435501
mul.hi.u32 r647, r475, r644
xor.b32 r648, r645, r647
xor.b32 r649, r648, r503
mul.hi.u32 r650, r478, r641
xor.b32 r651, r650, r646
xor.b32 r652, r651, r507
mul.lo.s32 r653, r644, -845247145
mul.lo.s32 r654, r641, -766435501
mul.hi.u32 r655, r475, r652
xor.b32 r656, r653, r655
xor.b32 r657, r656, r513
mul.hi.u32 r658, r478, r649
xor.b32 r659, r658, r654
xor.b32 r660, r659, r517
mul.lo.s32 r661, r652, -845247145
mul.lo.s32 r662, r649, -766435501
mul.hi.u32 r663, r475, r660
xor.b32 r664, r661, r663
xor.b32 r665, r664, r523
mul.hi.u32 r666, r478, r657
xor.b32 r667, r666, r662
xor.b32 r668, r667, r527
mul.lo.s32 r669, r660, -845247145
mul.lo.s32 r670, r657, -766435501
mul.hi.u32 r671, r475, r668
xor.b32 r672, r669, r671
xor.b32 r673, r672, r533
mul.hi.u32 r674, r478, r665
xor.b32 r675, r674, r670
xor.b32 r676, r675, r537
mul.lo.s32 r677, r668, -845247145
mul.lo.s32 r678, r665, -766435501
mul.hi.u32 r679, r475, r676
xor.b32 r680, r677, r679
xor.b32 r681, r680, r543
mul.hi.u32 r682, r478, r673
xor.b32 r683, r682, r678
xor.b32 r684, r683, r547
mul.lo.s32 r685, r673, -766435501
mul.hi.u32 r686, r478, r681
xor.b32 r687, r686, r685
xor.b32 r688, r687, r552
mul.lo.s32 r689, r684, -845247145
mul.hi.u32 r690, r475, r688
xor.b32 r691, r689, r690
xor.b32 r942, r691, r557
mul.lo.s32 r944, r688, -845247145
add.s32 r692, r472, 384
shr.s32 r693, r692, 31
xor.b32 r694, r693, r39
xor.b32 r695, r694, r476
mul.hi.u32 r696, r478, r692
xor.b32 r697, r696, r40
add.s32 r698, r481, 2041511040
mul.hi.u32 r699, r475, r697
xor.b32 r700, r699, r483
mul.hi.u32 r701, r478, r695
xor.b32 r702, r701, r698
xor.b32 r703, r702, r487
mul.lo.s32 r704, r697, -845247145
mul.lo.s32 r705, r695, -766435501
mul.hi.u32 r706, r475, r703
xor.b32 r707, r704, r706
xor.b32 r708, r707, r493
mul.hi.u32 r709, r478, r700
xor.b32 r710, r709, r705
xor.b32 r711, r710, r497
mul.lo.s32 r712, r703, -845247145
mul.lo.s32 r713, r700, -766435501
mul.hi.u32 r714, r475, r711
xor.b32 r715, r712, r714
xor.b32 r716, r715, r503
mul.hi.u32 r717, r478, r708
xor.b32 r718, r717, r713
xor.b32 r719, r718, r507
mul.lo.s32 r720, r711, -845247145
mul.lo.s32 r721, r708, -766435501
mul.hi.u32 r722, r475, r719
xor.b32 r723, r720, r722
xor.b32 r724, r723, r513
mul.hi.u32 r725, r478, r716
xor.b32 r726, r725, r721
xor.b32 r727, r726, r517
mul.lo.s32 r728, r719, -845247145
mul.lo.s32 r729, r716, -766435501
mul.hi.u32 r730, r475, r727
xor.b32 r731, r728, r730
xor.b32 r732, r731, r523
mul.hi.u32 r733, r478, r724
xor.b32 r734, r733, r729
xor.b32 r735, r734, r527
mul.lo.s32 r736, r727, -845247145
mul.lo.s32 r737, r724, -766435501
mul.hi.u32 r738, r475, r735
xor.b32 r739, r736, r738
xor.b32 r740, r739, r533
mul.hi.u32 r741, r478, r732
xor.b32 r742, r741, r737
xor.b32 r743, r742, r537
mul.lo.s32 r744, r735, -845247145
mul.lo.s32 r745, r732, -766435501
mul.hi.u32 r746, r475, r743
xor.b32 r747, r744, r746
xor.b32 r748, r747, r543
mul.hi.u32 r749, r478, r740
xor.b32 r750, r749, r745
xor.b32 r751, r750, r547
mul.lo.s32 r752, r740, -766435501
mul.hi.u32 r753, r478, r748
xor.b32 r754, r753, r752
xor.b32 r755, r754, r552
mul.lo.s32 r756, r751, -845247145
mul.hi.u32 r757, r475, r755
xor.b32 r758, r756, r757
xor.b32 r1317, r758, r557
mul.lo.s32 r1319, r755, -845247145
"""
    # Exact nine-value live-out map.  Eight values feed every later stochastic
    # conversion body; r1830 crosses the region boundary as token step zero.
    rand_by_wr = (
        (r815, r817), (r874, r876),
        (r942, r944), (r1317, r1319),
    )
    token_step_zero = r1830
    # Segment checksums (mul.hi,mul.lo,xor,add,shr) are respectively
    # (18,16,34,17,1), (17,15,34,2,1), (17,15,34,2,1),
    # (17,15,34,2,1), totaling (69,61,136,23,4).
    return rand_by_wr, token_step_zero

# The generator above appears once statically and executes once per runtime
# pass on the non-pad family-F path, before TOKEN_LOOP and before I/F tests.
# Its four row-result pairs and shared token-zero remain live across the region
# boundary and every token iteration.  Later store control contains sixteen
# static `cvt.rs.f16x2.f32` sites per entry and never recomputes Philox.
# SASS families are `F2FP.F16.F32.PACK_AB.RS` for the two conversions and
# `STG.E.64` for the eight-byte store.  Ordinary BF16/F16 conversions map to
# scalar `F2FP.{BF16,F16}.F32` before the tabled stores; FP32 has no conversion.

# ===========================================================================
# Compute-to-epilogue handoff
# source: vertical.cuh:317-321
# ===========================================================================

    # Executed once after all runtime pass/token loops by all four compute warps.
    out_arrive_token = ptx("mbarrier.arrive.shared::cta.b64",
                           address(gs.bar_out_ready), arrival_count=1)
    done_wait_token = ptx("mbarrier.arrive.shared::cta.b64",
                          address(gs.bar_epilogue_done), arrival_count=1)
    done_ready = 0
    while done_ready == 0:
        done_pred = ptx("mbarrier.try_wait.shared::cta.b64",
                        address(gs.bar_epilogue_done), done_wait_token)
        done_ready = ptx("selp.b32", 1, 0, done_pred)

# bar_out_ready completes when 128 compute lanes plus 32 epilogue lanes arrive.
# bar_epilogue_done completes when 32 epilogue lanes plus 128 compute lanes
# arrive.  The final wait prevents CTA exit while warp 15 still reads gs.out.

# ===========================================================================
# Shared epilogue warp
# source: vertical.cuh:328-376
# ===========================================================================

def role_epilogue(num_active_groups):
    for group in runtime_range(num_active_groups):
        gs = sram[group]
        head = heads[group]
        out_wait_token = ptx("mbarrier.arrive.shared::cta.b64",
                             address(gs.bar_out_ready), arrival_count=1)
        out_ready = 0
        while out_ready == 0:
            out_pred = ptx("mbarrier.try_wait.shared::cta.b64",
                           address(gs.bar_out_ready), out_wait_token)
            out_ready = ptx("selp.b32", 1, 0, out_pred)

        for step in static_range(NTOKENS):
            out_base = (
                batch_idx*out_stride_batch + step*out_stride_mtp + head*DIM)
            z_base = (
                batch_idx*z_stride_batch + step*z_stride_mtp + head*DIM)

            for ii in static_range(0, DIM//32, OUTPUT_VECTOR):
                d = lane*OUTPUT_VECTOR + (ii//OUTPUT_VECTOR)*32*OUTPUT_VECTOR
                z_pack = reg(INPUT_DTYPE, [OUTPUT_VECTOR])
                if z is present:
                    if DIM == 64:
                        z_pack = ptx("ld.global.v2.b16", address(z[z_base+d]))
                    else:
                        z_pack = ptx("ld.global.v4.b16", address(z[z_base+d]))
                out_pack = reg(INPUT_DTYPE, [OUTPUT_VECTOR])

                if DIM == 64:
                    out_values = ptx(
                        "ld.shared.v2.b32", address(gs.out[step,d]))
                else:
                    out_values = ptx(
                        "ld.shared.v4.b32", address(gs.out[step,d]))

                for k in static_range(OUTPUT_VECTOR):
                    out_value = extract_b32_as_f32(out_values,k)
                    if z is present:
                        z_value = ptx("cvt.f32.bf16", extract_u16(z_pack,k))
                        neg_z = ptx("sub.ftz.f32", 0.0, z_value)
                        silu_log2_arg = ptx(
                            "mul.ftz.f32", neg_z, 1.4426950408889634)
                        exp_neg_z = ptx(
                            "ex2.approx.ftz.f32", silu_log2_arg)
                        denominator = ptx("add.ftz.f32", 1.0, exp_neg_z)
                        sigmoid = ptx("div.approx.ftz.f32", 1.0, denominator)
                        z_sigmoid = ptx("mul.ftz.f32", z_value, sigmoid)
                        out_value = ptx("mul.ftz.f32", out_value, z_sigmoid)
                    out_pack[k] = ptx("cvt.rn.bf16.f32", out_value)
                if DIM == 64:
                    ptx("st.global.v2.b16", address(output[out_base+d]), out_pack)
                else:
                    ptx("st.global.v4.b16", address(output[out_base+d]), out_pack)

        epilogue_arrive_token = ptx("mbarrier.arrive.shared::cta.b64",
                                    address(gs.bar_epilogue_done), arrival_count=1)

# OUTPUT_VECTOR is exactly 2 BF16 values for DIM64 and 4 for DIM128.  Thus every
# token/group/lane issues one v2/v4 global z load when present, one v2/v4 shared
# FP32 out load, 2/4 fully expanded SiLU sequences and scalar BF16 conversions,
# then one v2/v4 global BF16 output store.  Warp 15 drains group 0, then 1, then
# 2; it does not overlap epilogues from different active groups in source.
```

## Barrier phase ledger

Each active group owns independent phase-zero barriers.  There is one head and
one state stage, so no barrier is reused for another head or parity phase.

| Barrier | Init arrivals | Producers/arrivals | Transaction bytes | Consumer edge |
| --- | ---: | --- | ---: | --- |
| `bar_BC_full` | 160 | 128 compute token-arrive/try-waits + load lane 0 split `expect_tx.relaxed` then release-arrive drop-count 32 | `2*T*DSTATE*sizeof(input)` | compute begins dt/state work only after B and C land |
| `bar_state_in_empty` | 160 | 128 compute pre-arrivals + 32 load-lane arrive-and-waits | 0 | TMA warp may issue state/x only after compute releases stage 0 |
| `bar_state_in_full` | 129 | 128 compute token-arrive/try-waits + load lane 0 combined release `arrive.expect_tx` count 1 | `T*DIM*sizeof(input)` plus non-pad `DIM*DSTATE*sizeof(state)` | recurrence reads state/x only after TMA completion |
| `bar_out_ready` | 160 | 128 compute arrivals + 32 epilogue arrive-and-waits | 0 | epilogue reads all `out` rows for the group |
| `bar_epilogue_done` | 160 | 32 epilogue arrivals + 128 compute arrive-and-waits | 0 | compute warps and CTA may exit |

## Control-flow and dataflow invariants

1. Barrier initialization and use have identical active-group predicates.
2. All 512 threads cross the one CTA barrier after barrier initialization.
3. Pad specialization still loads B/C and x, initializes register state to
   zero, and produces output; it omits state TMA and all state writes.
4. B/C and x are batch-indexed; state is state-slot-indexed.  The head-to-KV
   group mapping is integer division by the compile-time heads-per-group ratio.
5. The vertical CUDA implementation reads one `state_batch_indices[batch]`.
   It does not consume `dst_state_batch_indices`; final state returns to the
   source state slot.  Intermediate destinations use their own optional index.
6. Each recurrence row stays in registers across all token steps.  Direct state
   writes occur after the corresponding token update and never round-trip via
   shared memory.
7. `out` contains complete warp-reduced FP32 values.  Only warp 15 converts,
   gates, and stores final output.
8. No scaled-state buffer, state scale, intermediate scale, or cu-seqlens
   pointer reaches the kernel.

## Source-to-sketch mapping

| CUDA source | Sketch section |
| --- | --- |
| `vertical.cuh:48-61` | Pipeline and ownership |
| `vertical.cuh:67-90` | Exact per-group shared storage |
| `vertical.cuh:96-138` | TMA producer role |
| `vertical.cuh:146-177` | Direct state conversion and vector store |
| `vertical.cuh:185-248` | Compute role prologue and synchronization |
| `vertical.cuh:249-315` | Four-row recurrence passes and state stores |
| `vertical.cuh:317-321` | Compute-to-epilogue handoff |
| `vertical.cuh:328-376` | Shared epilogue warp |
| `vertical.cuh:382-464` | Entry coordinates, barriers, role and pad dispatch |
| `invoke...mtp.cuh:42-51` | dtype/Philox specialization gates |
| `invoke...mtp.cuh:76-94` | common/state/output/intermediate alignment gates |
| `invoke...mtp.cuh:96-164` | vertical validation, descriptors, grid/block, SMEM and launch |
| `create_tensor_map.cuh:15-88` | dtype encoding, 128B base, byte strides and descriptor modes |
| `common.cuh:41-90,184-207` | packed vectors, warp sum, thresholded softplus and input alignment |
| `conversion.cuh:95-249` | Philox and FP16 stochastic conversion |

## Independent reviewer obligations

The reviewer must use FlashInfer's normal JIT route for SM100a with
`-DNDEBUG -O3 -lineinfo`, freeze one PTX/SASS artifact set per unique compiled
code shape, and establish bidirectional CUDA-source ↔ sketch ↔ PTX/SASS
mapping.  Review is not complete until roles, storage, synchronization,
operation dataflow, and instruction selection all pass.  At minimum it must
enumerate exact TMA/mbarrier forms and transaction counts, standard
`cuda::barrier::wait` lowering, BF16/FP16/FP32 conversion and direct-store
widths for DSTATE64/96/128, fast exp/log/div lowering, five-step shuffle sum,
output vector widths for DIM64/128, inactive-group tails, and both ordinary and
Philox FP16 stores.  A FAIL returns only to this sketch; no TIRx body may be
implemented before PASS.
