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

# Selective-state-update MTP simple SM100: execution sketch

This is a non-executable transcription sketch of FlashInfer's CUDA
`selective_state_update_kernel_simple_mtp`. It freezes the `(sequence, head,
CTA-within-head)` launch, four-compute-warp cooperative input copy, direct
register recurrence, one- or two-stage state traversal, fixed and variable
length addressing, destination-slot selection, intermediate-cache writes,
scaled int16 state, Philox conversion, and output epilogue that the TIRx port
must preserve. The target module is
[`tirx_kernels/flashinfer/mamba/selective_state_update_mtp_simple.py`](../../tirx_kernels/flashinfer/mamba/selective_state_update_mtp_simple.py).
It remains an integration scaffold until this sketch passes independent
review.

The frozen FlashInfer commit is
`f2e04400e330fb2debe0bf8730d9424a1d37927f`. The primary CUDA source is
`include/flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh`, SHA256
`64b189b642d05970202964bdd3accf7055d34d6a50b7953a64c8a05a3e0a0dbe`.
The reviewed target is SM100a/B200 even though the source simple kernel also
supports SM80+. Input, B, C, z, and output are BF16. State is BF16, FP16,
FP32, or scaled int16; weights and dt are FP32 or BF16; A is FP32; state and
accepted-token indices are int32 or int64. `DIM in {64,128}`, `DSTATE in
{64,96,128}`, `NTOKENS in {1,2,4,6,8}`, group ratio in
`{1,2,4,8,16,32,64}`, `CTAS_PER_HEAD in {1,2,4}`, and Philox rounds in
`{0,10}` are represented. Every reference call explicitly requests
`algorithm="simple"`; `async_horizontal` is only an API alias for this same
implementation and is not a fourth kernel.

## Pipeline at a glance

All four warps are compute warps. Warp 0 preloads B, warp 1 preloads C, every
warp preloads token-assigned x rows and cooperatively preloads the first
16-row state pass, while ordinary global loads publish dt and destination
slots. One copy-group wait and CTA barrier publish those values. Each pass maps
eight adjacent lanes to one DIM row, keeps that row's state in FP32 register
pairs for all token steps, and publishes partial outputs to shared memory.
Multi-pass variants copy the next 16 state rows after the current pass, wait,
and cross a CTA barrier before reusing the alternate state stage. A final CTA
barrier precedes z gating and output stores.

| Physical threads | Role | Exact ownership | Publication/reuse edge |
| --- | --- | --- | --- |
| warp 0 | B producer plus x/state/dt/slot participant and compute | all B tokens; four rows per pass | initial copy-group wait and CTA barrier; per-pass state-copy wait/barrier |
| warp 1 | C producer plus x/state/dt/slot participant and compute | all C tokens; four rows per pass | same |
| warps 2..3 | x/state/dt/slot participant and compute | token-striped x; four rows per pass each | same |
| every 8-lane subgroup | recurrence owner | one DIM row, DSTATE columns striped across its eight members | three shuffle-down reductions; member zero writes shared output and optional scale |

`ROWS_PER_PASS=4 warps * 4 rows/warp=16`. `DIM_PER_CTA=DIM /
CTAS_PER_HEAD`, `NUM_PASSES=DIM_PER_CTA/16`, and `STATE_STAGES` is one only
when `NUM_PASSES==1`, otherwise two. Thus the represented code shapes are:

| DIM | CTAS_PER_HEAD | DIM_PER_CTA | passes | state stages |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 64 | 4 | 2 |
| 64 | 2 | 32 | 2 | 2 |
| 64 | 4 | 16 | 1 | 1 |
| 128 | 1 | 128 | 8 | 2 |
| 128 | 2 | 64 | 4 | 2 |
| 128 | 4 | 32 | 2 | 2 |

The host chooses the largest legal value in `(4,2,1)` not exceeding
`clamp((num_sms*10)/(batch*nheads),1,DIM/16)`. This selection, not an
independent user knob, defines `grid.z`.

## Primitive vocabulary

Structural operations do not move or compute values:

```python
specialize(...)       # compile-time dtype, shape, token, ratio and CTA shape
launch(...)           # physical grid/block and dynamic-shared metadata
tile(...)             # GMEM, SMEM, or register storage declaration
view(...)             # typed/addressing view without moving values
alias(...)            # same physical storage with an ordered lifetime
reg_tile(...)         # lane-private scalar or vector registers
```

Copies always expose their direction:

```python
copy_g2r(src, dst=None, predicate=None)
copy_g2s(src, dst, bytes, predicate=None)
copy_s2r(src, dst)
copy_r2s(src, dst)
copy_r2g(src, dst, predicate=None)
```

Synchronization and copy scheduling are explicit:

```python
copy_group_commit()
copy_group_wait_zero()
cta_sync()
```

The compute vocabulary is primitive:

```python
fill(dst, value)
cast(dtype, src, rounding=None)
add(dst, lhs, rhs, lanes=1, rounding=None)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs, lanes=1, rounding=None)
fma(dst, lhs, rhs, acc, lanes=1, rounding=None)
exp2(dst, src)
log2(dst, src)
div(dst, lhs, rhs)
abs(dst, src)
min(dst, lhs, rhs)
max(dst, lhs, rhs)
bit_and(dst, lhs, rhs)
bit_xor(dst, lhs, rhs)
mul_hi_u32(dst, lhs, rhs)
mul_lo_s32(dst, lhs, rhs)
add_s32(dst, lhs, rhs)
shuffle_down(dst, src, delta, clamp, member_mask)
shuffle_index(dst, src, source_lane, clamp, member_mask)
```

Predicates, pointer selection, loop indices, static address expressions, and
stage/cursor updates are control operations. There is no compound `pipeline`,
`state_update`, `softplus`, `quantize`, `philox`, `silu`, or `reduce`
operation. No tile primitive appears in this sketch or in the eventual kernel.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    INPUT_DTYPE="bf16",
    STATE_DTYPE=("bf16", "f16", "f32", "i16"),
    WEIGHT_DTYPE=("f32", "bf16"),
    MATRIX_A_DTYPE="f32",
    STATE_INDEX_DTYPE=("i32", "i64"),
    ACCEPTED_INDEX_DTYPE=("i32", "i64"),
    DIM=(64, 128),
    DSTATE=(64, 96, 128),
    NTOKENS=(1, 2, 4, 6, 8),
    HEADS_PER_GROUP=(1, 2, 4, 8, 16, 32, 64),
    CTAS_PER_HEAD=(1, 2, 4),
    PHILOX_ROUNDS=(0, 10),
    target="sm_100a",
)

NUM_WARPS = 4
LANES_PER_ROW = 8
ROWS_PER_WARP = 4
ROWS_PER_PASS = 16
DIM_PER_CTA = DIM // CTAS_PER_HEAD
NUM_PASSES = DIM_PER_CTA // ROWS_PER_PASS
STATE_STAGES = 1 if NUM_PASSES == 1 else 2
DSTATE_PAD = align_up(DSTATE * sizeof(INPUT_DTYPE), 128) // sizeof(INPUT_DTYPE)

host_assert(DIM % CTAS_PER_HEAD == 0)
host_assert(DIM_PER_CTA % ROWS_PER_PASS == 0)
host_assert(DSTATE % (16 // sizeof(INPUT_DTYPE)) == 0)
host_assert(DSTATE % (16 // sizeof(STATE_DTYPE)) == 0)
host_assert(PHILOX_ROUNDS == 0 or STATE_DTYPE == "f16")
host_assert((STATE_DTYPE == "i16") == (state_scale is present))

launch_config = launch(
    grid=(batch_or_num_sequences, nheads, CTAS_PER_HEAD),
    block=(32, 4, 1),
    threads=128,
    dynamic_smem_bytes=SHARED_BYTES,
    max_dynamic_shared_memory_attribute=SHARED_BYTES,
)

def selective_state_update_mtp_simple(
    state, state_scale,
    x, dt, A, B, C, D, z, dt_bias,
    state_batch_indices, dst_state_batch_indices,
    intermediate_states, intermediate_state_scales,
    intermediate_state_indices,
    cu_seqlens, num_accepted_tokens, rand_seed,
    output,
    all_runtime_strides,
    nheads, ngroups, cache_steps,
    dt_softplus, update_state, pad_slot_id,
):
    seq_idx, head, cta_z = cta_id(
        extents=(batch_or_num_sequences, nheads, CTAS_PER_HEAD))
    # instruction_selection: mov.u32 from %ctaid.x/y/z; extent: three physical CTA coordinates
    lane, warp = thread_id(extents=(32, 4))
    # instruction_selection: mov.u32 from independent %tid.x/%tid.y; extent: one lane and one warp coordinate
    flat_tid = warp * 32 + lane
    dim_offset = cta_z * DIM_PER_CTA
    kv_group = head // HEADS_PER_GROUP

    # Fixed-length uses every compile-time token. Varlen reads consecutive
    # endpoints and returns the whole CTA before any copy or barrier when empty.
    if cu_seqlens is present:
        bos = copy_g2r(cu_seqlens[seq_idx])
        # instruction_selection: read-only ld.global.s32; extent: one BOS scalar per thread
        eos = copy_g2r(cu_seqlens[seq_idx + 1])
        # instruction_selection: read-only ld.global.s32; extent: one EOS scalar per thread
        seq_len = sub(eos, bos)
        # instruction_selection: sub.s32; extent: one runtime length
        if seq_len <= 0:
            return
    else:
        bos = 0
        seq_len = NTOKENS

    init_token_idx = 0
    if num_accepted_tokens is present:
        accepted = copy_g2r(num_accepted_tokens[seq_idx])
        # instruction_selection: read-only ld.global.s32 or ld.global.b64 followed by the needed narrowing; extent: one accepted-token scalar per thread
        init_token_idx = max(accepted - 1, 0)
        # instruction_selection: sub.s32 plus max.s32; extent: one clamped token coordinate

    if state_batch_indices is present:
        state_batch = copy_g2r(
            state_batch_indices[
                seq_idx * state_indices_stride_batch
                + init_token_idx * state_indices_stride_T])
        # instruction_selection: ld.global.s32 with sign extension or ld.global.b64; extent: one 1D/2D-selected source slot per thread
        state_batch = cast("i64", state_batch)
        # instruction_selection: sign-extension is part of the i32 load/lowering; i64 is identity
    else:
        state_batch = cast("i64", seq_idx)
        # instruction_selection: cvt.u64.u32; extent: one fallback source slot

    is_not_pad = state_batch != i64(pad_slot_id)
    # instruction_selection: setp.ne.b64 followed by the specialization branch; extent: one runtime pad-dispatch predicate

    A_value = cast("f32", copy_g2r(A[head]))
    # instruction_selection: ld.global.b32 and identity f32 conversion; extent: one A scalar per thread before the initial barrier
    D_value = 0.0
    if D is present:
        D_value = cast("f32", copy_g2r(D[head]))
        # instruction_selection: ld.global.b32 or ld.global.b16 plus cvt.f32.bf16; extent: one optional D scalar per thread before the initial barrier

    dispatch run_simple(IS_PAD=not is_not_pad):
        preload_and_prepare_slots(IS_PAD)
        cta_sync()
        # instruction_selection: bar.sync 0; extent: all 128 threads after initial copies, dt stores, and slot stores
        compute_all_passes(IS_PAD, A_value, D_value)

    # =======================================================================
    # Exact shared-memory ABI and lifetimes
    # =======================================================================

    OFF_B = 0
    OFF_C = align_up(OFF_B + NTOKENS * DSTATE_PAD * 2, 128)
    OFF_X = align_up(OFF_C + NTOKENS * DSTATE_PAD * 2, 128)
    OFF_DT = align_up(OFF_X + NTOKENS * DIM_PER_CTA * 2, 4)
    OFF_OUT = OFF_DT + NTOKENS * 4
    OFF_DST_SLOTS = align_up(OFF_OUT + NTOKENS * DIM_PER_CTA * 4, 8)
    OFF_STATE_IN = align_up(OFF_DST_SLOTS + NTOKENS * 8, 128)
    SHARED_BYTES = align_up(
        OFF_STATE_IN
        + STATE_STAGES * ROWS_PER_PASS * DSTATE_PAD * sizeof(STATE_DTYPE),
        128)

    shared_raw = tile("smem", "u8", [SHARED_BYTES], alignment=128)
    sB = view(shared_raw, "bf16", [NTOKENS,DSTATE_PAD], OFF_B,
              lifetime="initial copy through every pass/token recurrence")
    sC = view(shared_raw, "bf16", [NTOKENS,DSTATE_PAD], OFF_C,
              lifetime="initial copy through every pass/token recurrence")
    sX = view(shared_raw, "bf16", [NTOKENS,DIM_PER_CTA], OFF_X,
              lifetime="initial copy through every pass/token recurrence")
    sDt = view(shared_raw, "f32", [NTOKENS], OFF_DT,
               lifetime="ordinary global-load publication through every pass")
    sOut = view(shared_raw, "f32", [NTOKENS,DIM_PER_CTA], OFF_OUT,
                lifetime="member-zero pass publication through final epilogue")
    sDstSlot = view(shared_raw, "i64", [NTOKENS], OFF_DST_SLOTS,
                    lifetime="precomputed mode selection through every state write")
    sState = view(
        shared_raw, STATE_DTYPE,
        [STATE_STAGES,ROWS_PER_PASS,DSTATE_PAD], OFF_STATE_IN,
        lifetime="one pass load; alternate stages across multi-pass traversal")

    # =======================================================================
    # Initial cooperative copy and destination-slot preparation
    # source: kernel_selective_state_update_mtp_simple.cuh:126-268
    # =======================================================================

    def preload_and_prepare_slots(IS_PAD):
        if cu_seqlens is present:
            B_base = bos * B_stride_batch
            C_base = bos * C_stride_batch
            x_base = bos * x_stride_batch
            dt_base = bos * dt_stride_batch
            B_tstride, C_tstride = B_stride_batch, C_stride_batch
            x_tstride, dt_tstride = x_stride_batch, dt_stride_batch
        else:
            B_base = seq_idx * B_stride_batch
            C_base = seq_idx * C_stride_batch
            x_base = seq_idx * x_stride_batch
            dt_base = seq_idx * dt_stride_batch
            B_tstride, C_tstride = B_stride_mtp, C_stride_mtp
            x_tstride, dt_tstride = x_stride_mtp, dt_stride_mtp

        INPUT_PACK = 16 // sizeof(INPUT_DTYPE)
        STATE_PACK = 16 // sizeof(STATE_DTYPE)

        if warp == 0:
            for packed_item in lane_strided_range(
                    0, NTOKENS * DSTATE // INPUT_PACK, 32):
                step = packed_item // (DSTATE // INPUT_PACK)
                col = (packed_item % (DSTATE // INPUT_PACK)) * INPUT_PACK
                if step < seq_len:
                    copy_g2s(
                        B[B_base + step*B_tstride + kv_group*DSTATE + col : +INPUT_PACK],
                        sB[step,col:col+INPUT_PACK], bytes=16)
                    # instruction_selection: cp.async.cg.shared.global with 16-byte extent; one warp-0 copy per live B pack

        elif warp == 1:
            for packed_item in lane_strided_range(
                    0, NTOKENS * DSTATE // INPUT_PACK, 32):
                step = packed_item // (DSTATE // INPUT_PACK)
                col = (packed_item % (DSTATE // INPUT_PACK)) * INPUT_PACK
                if step < seq_len:
                    copy_g2s(
                        C[C_base + step*C_tstride + kv_group*DSTATE + col : +INPUT_PACK],
                        sC[step,col:col+INPUT_PACK], bytes=16)
                    # instruction_selection: cp.async.cg.shared.global with 16-byte extent; one warp-1 copy per live C pack

        for step in warp_strided_range(warp, seq_len, NUM_WARPS):
            for col in lane_vector_range(0, DIM_PER_CTA, INPUT_PACK):
                copy_g2s(
                    x[x_base + step*x_tstride + head*DIM + dim_offset + col : +INPUT_PACK],
                    sX[step,col:col+INPUT_PACK], bytes=16)
                # instruction_selection: cp.async.cg.shared.global with 16-byte extent; token ownership is warp-striped and columns are lane-vectorized

        if not IS_PAD:
            state_base = state_batch*state_stride_batch + head*DIM*DSTATE
            NUM_STATE_PACKS = ROWS_PER_PASS * DSTATE // STATE_PACK
            for packed_item in flat_thread_strided_range(
                    flat_tid, NUM_STATE_PACKS, NUM_WARPS*32):
                row = packed_item // (DSTATE // STATE_PACK)
                col = (packed_item % (DSTATE // STATE_PACK)) * STATE_PACK
                copy_g2s(
                    state[state_base + (dim_offset+row)*DSTATE + col : +STATE_PACK],
                    sState[0,row,col:col+STATE_PACK], bytes=16)
                # instruction_selection: cp.async.cg.shared.global with 16-byte extent; all 128 threads cooperatively cover the first 16 state rows

        if flat_tid < seq_len:
            step = flat_tid
            dt_value = cast("f32", copy_g2r(dt[dt_base + step*dt_tstride + head]))
            # instruction_selection: ld.global.b32 or b16 plus cvt.f32.bf16; extent: one live dt scalar
            if dt_bias is present:
                bias = cast("f32", copy_g2r(dt_bias[head]))
                # instruction_selection: ld.global.b32 or b16 plus cvt.f32.bf16; extent: one optional bias scalar
                dt_value = add(dt_value, bias)
                # instruction_selection: add.ftz.f32; extent: one scalar
            if dt_softplus and dt_value <= 20.0:
                dt_exp = exp2(mul(dt_value, LOG2_E))
                # instruction_selection: mul.ftz.f32 then ex2.approx.ftz.f32; extent: one <=20 softplus exponential
                dt_log = log2(add(1.0, dt_exp))
                # instruction_selection: add.ftz.f32 then lg2.approx.ftz.f32; extent: one <=20 softplus logarithm
                dt_value = mul(dt_log, LN_2)
                # instruction_selection: mul.ftz.f32; extent: one final softplus scalar
            copy_r2s(dt_value, sDt[step])
            # instruction_selection: st.shared.b32; extent: one live token dt

        if flat_tid < NTOKENS:
            step = flat_tid
            SKIP = -1
            if IS_PAD or step >= seq_len:
                dst_slot = SKIP
            elif dst_state_batch_indices is present:
                dst_slot = cast("i64", copy_g2r(
                    dst_state_batch_indices[
                        seq_idx*dst_indices_stride_batch + step*dst_indices_stride_T]))
                # instruction_selection: ld.global.s32 with sign extension or ld.global.b64; extent: one live per-step destination index
                if dst_slot == i64(pad_slot_id):
                    dst_slot = SKIP
            elif intermediate_states is present:
                icache_idx = state_batch
                if intermediate_state_indices is present:
                    icache_idx = cast("i64", copy_g2r(intermediate_state_indices[seq_idx]))
                    # instruction_selection: ld.global.s32 with sign extension or ld.global.b64; extent: one intermediate-cache index
                dst_slot = icache_idx * cache_steps + step
            else:
                dst_slot = state_batch if (step == seq_len-1 and update_state) else SKIP
            copy_r2s(dst_slot, sDstSlot[step])
            # instruction_selection: st.shared.b64; extent: one slot decision for every compile-time token

        copy_group_commit()
        # instruction_selection: cp.async.commit_group; extent: one group containing this thread's B/C/x/initial-state copies
        copy_group_wait_zero()
        # instruction_selection: cp.async.wait_group 0; extent: each thread waits for all issued copies before the enclosing CTA barrier

    # =======================================================================
    # Register recurrence and multi-pass state traversal
    # source: kernel_selective_state_update_mtp_simple.cuh:274-497
    # =======================================================================

    def compute_all_passes(IS_PAD, A_value, D_value):
        member = lane % LANES_PER_ROW
        row_group = lane // LANES_PER_ROW
        # instruction_selection: and.b32 with 7 plus shr.u32 by 3; extent: one eight-lane row mapping

        STATE_PAD_POW2 = next_power_of_two(DSTATE)
        STATE_VALUES_PER_THREAD = STATE_PAD_POW2 // LANES_PER_ROW
        STATE_VALUES_PER_BANK = 4 // sizeof(STATE_DTYPE)
        READS_PER_MEMBER = 32 // LANES_PER_ROW
        ELEMS_PER_TILE_MEMBER = READS_PER_MEMBER * STATE_VALUES_PER_BANK
        ELEMS_PER_TILE = ELEMS_PER_TILE_MEMBER * LANES_PER_ROW
        NUM_TILES = STATE_VALUES_PER_THREAD // ELEMS_PER_TILE_MEMBER
        PAIRS_PER_TILE_MEMBER = ELEMS_PER_TILE_MEMBER // 2

        def base_col(tile_idx, element_idx):
            return tile_idx*ELEMS_PER_TILE + member*ELEMS_PER_TILE_MEMBER + element_idx

        if intermediate_states is present:
            write_state = intermediate_states
            write_state_stride = nheads*DIM*DSTATE
            write_scale = intermediate_state_scales
            write_scale_stride = nheads*DIM
        else:
            write_state = state
            write_state_stride = state_stride_batch
            write_scale = state_scale
            write_scale_stride = state_scale_stride_batch

        source_state_base = state_batch*state_stride_batch + head*DIM*DSTATE
        source_state_base_i32 = cast("i32", source_state_base)
        # instruction_selection: cvt.u32.u64; extent: one deliberate narrowing to the frozen helper's signed-int state_ptr_offset ABI
        random_seed = 0
        if PHILOX_ROUNDS > 0 and rand_seed is present:
            random_seed = copy_g2r(rand_seed[0])
            # instruction_selection: guarded ld.global.b64; extent: one seed per thread, live only in Philox specializations

        for pass_idx in static_range(NUM_PASSES):
            pass_row = warp*ROWS_PER_WARP + row_group
            local_row = pass_idx*ROWS_PER_PASS + pass_row
            d = dim_offset + local_row
            stage = pass_idx % STATE_STAGES

            state_decode_scale = 1.0
            if STATE_DTYPE == "i16" and not IS_PAD:
                state_decode_scale = cast("f32", copy_g2r(
                    state_scale[state_batch*state_scale_stride_batch + head*DIM + d]))
                # instruction_selection: ld.global.b32; extent: one old decode scale per row/pass before all token steps

            rState = reg_tile("f32x2", [NUM_TILES,PAIRS_PER_TILE_MEMBER])
            for tile_idx in static_range(NUM_TILES):
                member_col0 = base_col(tile_idx, 0)
                if IS_PAD or member_col0 >= DSTATE:
                    for pair_idx in static_range(PAIRS_PER_TILE_MEMBER):
                        fill(rState[tile_idx,pair_idx], (0.0,0.0))
                        # instruction_selection: paired zero register moves or predicate-selected zero; extent: one inactive state pair
                else:
                    state_member = copy_s2r(
                        sState[stage,pass_row,
                               member_col0:member_col0+ELEMS_PER_TILE_MEMBER],
                        bytes=16)
                    # instruction_selection: ld.shared.v4.b32; extent: exactly one 16-byte state member tile for every state dtype
                    if STATE_DTYPE == "bf16":
                        for packed_word in static_range(4):
                            low_f32_bits = shift_left_b32(
                                state_member.word[packed_word],16)
                            high_f32_bits = bit_and(
                                state_member.word[packed_word],0xFFFF0000)
                            rState[tile_idx,packed_word] = (
                                bitcast_f32(low_f32_bits),
                                bitcast_f32(high_f32_bits))
                            # instruction_selection: one shl.b32 by 16 and one and.b32 with 0xffff0000 per packed word; extent: four shl plus four and instructions for the eight-element tile
                    elif STATE_DTYPE == "f16":
                        for element_idx in static_range(8):
                            state_f32[element_idx] = cast(
                                "f32", state_member.f16[element_idx])
                            # instruction_selection: cvt.f32.f16; extent: eight scalar conversions for the eight-element tile
                        for pair_idx in static_range(4):
                            rState[tile_idx,pair_idx] = (
                                state_f32[pair_idx*2],state_f32[pair_idx*2+1])
                    elif STATE_DTYPE == "f32":
                        for element_idx in static_range(4):
                            state_f32[element_idx] = state_member.f32[element_idx]
                            # instruction_selection: identity b32 register interpretation; extent: four FP32 elements from the 16-byte tile, regrouped as two f32x2 pairs
                        for pair_idx in static_range(2):
                            rState[tile_idx,pair_idx] = (
                                state_f32[pair_idx*2],state_f32[pair_idx*2+1])
                    else:  # STATE_DTYPE == "i16"
                        for element_idx in static_range(8):
                            state_f32[element_idx] = cast(
                                "f32", state_member.i16[element_idx], rounding="rn")
                            # instruction_selection: cvt.rn.f32.s16; extent: eight scalar conversions for the eight-element tile
                        for pair_idx in static_range(4):
                            rState[tile_idx,pair_idx] = (
                                state_f32[pair_idx*2],state_f32[pair_idx*2+1])
                        for pair_idx in static_range(4):
                            mul(rState[tile_idx,pair_idx], rState[tile_idx,pair_idx],
                                (state_decode_scale,state_decode_scale), lanes=2)
                            # instruction_selection: mul.f32x2; extent: four packed decode operations for the eight-element int16 tile

            # Keep five shared-memory cursors and strength-reduce every
            # token-dependent address exactly as the source does.  Across all
            # frozen specializations ptxas folds the two logical B/C cursors
            # into one physical moving shared-address base plus their static
            # storage separation, and source line 434 never emits an independent
            # add.  Frozen shapes that retain the source association place the
            # single row-stride add.s32 at line 433; DSTATE64/DPC16/CTAS4 instead
            # coalesces its folded-B/C +128, x +32, and out +64 induction adds
            # under loop line 392 while dt +4 remains at line 436.  No path
            # recomputes step * row_stride.
            B_step = address_of(sB[0,0])
            C_step = address_of(sC[0,0])
            x_step = address_of(sX[0,0])
            dt_step = address_of(sDt[0])
            out_step = address_of(sOut[0,0])

            for step in serial_range(0, NTOKENS):
                if step >= seq_len:
                    break
                dst_slot = copy_s2r(sDstSlot[step])
                # instruction_selection: ld.shared.b64 scheduled before recurrence arithmetic; extent: one destination decision per row/token
                dt_value = copy_s2r(dt_step[0])
                # instruction_selection: ld.shared.b32; extent: one dt scalar per row/token
                a_dt = mul(A_value, dt_value)
                # instruction_selection: mul.ftz.f32; extent: one row/token scalar
                dA = exp2(mul(a_dt, LOG2_E))
                # instruction_selection: mul.ftz.f32 then ex2.approx.ftz.f32; extent: one recurrence decay per row/token
                x_value = cast("f32", copy_s2r(x_step[local_row]))
                # instruction_selection: ld.shared.b16 plus cvt.f32.bf16; extent: one row/token x scalar
                dtx = mul(dt_value, x_value)
                # instruction_selection: mul.ftz.f32; extent: one row/token scalar
                out_pair = (0.0,0.0)

                for tile_idx in static_range(NUM_TILES):
                    member_col0 = base_col(tile_idx, 0)
                    if member_col0 < DSTATE:
                        if STATE_DTYPE == "f32":
                            B_member_bits = copy_s2r(
                                B_step[member_col0:member_col0+4], bytes=8)
                            C_member_bits = copy_s2r(
                                C_step[member_col0:member_col0+4], bytes=8)
                            # instruction_selection: ld.shared.v2.b32; extent: one eight-byte/four-BF16 member tile for each of B and C in the FP32-state shape
                            for packed_word in static_range(2):
                                B_member[packed_word] = (
                                    bitcast_f32(shift_left_b32(
                                        B_member_bits.word[packed_word],16)),
                                    bitcast_f32(bit_and(
                                        B_member_bits.word[packed_word],
                                        0xFFFF0000)))
                                C_member[packed_word] = (
                                    bitcast_f32(shift_left_b32(
                                        C_member_bits.word[packed_word],16)),
                                    bitcast_f32(bit_and(
                                        C_member_bits.word[packed_word],
                                        0xFFFF0000)))
                            # instruction_selection: one shl.b32 by 16 and one and.b32 with 0xffff0000 per packed word; extent: two shl plus two and instructions for B and independently two plus two for C
                        else:
                            B_member_bits = copy_s2r(
                                B_step[member_col0:member_col0+8], bytes=16)
                            C_member_bits = copy_s2r(
                                C_step[member_col0:member_col0+8], bytes=16)
                            # instruction_selection: ld.shared.v4.b32; extent: one 16-byte/eight-BF16 member tile for each of B and C in BF16/FP16/int16-state shapes
                            for packed_word in static_range(4):
                                B_member[packed_word] = (
                                    bitcast_f32(shift_left_b32(
                                        B_member_bits.word[packed_word],16)),
                                    bitcast_f32(bit_and(
                                        B_member_bits.word[packed_word],
                                        0xFFFF0000)))
                                C_member[packed_word] = (
                                    bitcast_f32(shift_left_b32(
                                        C_member_bits.word[packed_word],16)),
                                    bitcast_f32(bit_and(
                                        C_member_bits.word[packed_word],
                                        0xFFFF0000)))
                            # instruction_selection: one shl.b32 by 16 and one and.b32 with 0xffff0000 per packed word; extent: four shl plus four and instructions for B and independently four plus four for C
                    for pair_idx in static_range(PAIRS_PER_TILE_MEMBER):
                        col0 = base_col(tile_idx, pair_idx*2)
                        if col0 < DSTATE:
                            B_pair = B_member[pair_idx]
                            C_pair = C_member[pair_idx]
                            dBx_pair = mul(B_pair, (dtx,dtx), lanes=2, rounding="rn")
                            # instruction_selection: mul.f32x2; extent: one adjacent state pair
                            rState[tile_idx,pair_idx] = fma(
                                (dA,dA), rState[tile_idx,pair_idx], dBx_pair,
                                lanes=2, rounding="rn")
                            # instruction_selection: fma.rn.f32x2; extent: one adjacent recurrence pair
                            out_pair = fma(
                                rState[tile_idx,pair_idx], C_pair, out_pair,
                                lanes=2, rounding="rn")
                            # instruction_selection: fma.rn.f32x2; extent: one adjacent output-accumulation pair

                out_value = add(out_pair.x, out_pair.y)
                # instruction_selection: add.ftz.f32; extent: one pair collapse per lane
                for delta in (4,2,1):
                    peer = shuffle_down(
                        out_value, delta, clamp=31, member_mask=0xFFFFFFFF)
                    # instruction_selection: shfl.sync.down.b32 with delta 4/2/1; extent: one eight-lane subgroup reduction step
                    out_value = add(out_value, peer)
                    # instruction_selection: add.ftz.f32; extent: one scalar reduction step
                if member == 0:
                    row_output = fma(D_value, x_value, out_value)
                    # instruction_selection: fma.rn.ftz.f32; extent: exactly one member-zero row/token epilogue
                    copy_r2s(row_output, out_step[local_row])
                    # instruction_selection: st.shared.b32; extent: one row/token output publication

                B_step += DSTATE_PAD * sizeof(INPUT_DTYPE)
                C_step += DSTATE_PAD * sizeof(INPUT_DTYPE)
                x_step += DIM_PER_CTA * sizeof(INPUT_DTYPE)
                dt_step += sizeof(f32)
                out_step += DIM_PER_CTA * sizeof(f32)
                # instruction_selection: add.s32 shared-address cursor updates.  Every frozen specialization folds logical B/C into one physical moving base with a static B/C separation, and source line 434 emits no independent add.  In shapes retaining the normal source association, line 433 owns the symbolic folded-B/C row-stride add.  The DSTATE64/DPC16/CTAS4 exception emits folded-B/C +128, x +32, and out +64 under loop line 392, with dt +4 under line 436.  Static extent is zero for NTOKENS=1 because the entire advance set is DCE; NTOKENS>1 emits the folded-B/C, x, dt, and out add.s32 updates for each emitted loop-back body.  For the frozen DPC16/DSTATE128 shape the line-associated PTX advances the folded B/C base by 256 bytes, x by 32, dt by 4, and out by 64; no frozen shape issues step*stride address multiplies.

                if dst_slot != -1:
                    encode_scale = 1.0
                    if STATE_DTYPE == "i16":
                        local_max = -FLT_MAX
                        for tile_idx in static_range(NUM_TILES):
                            for pair_idx in static_range(PAIRS_PER_TILE_MEMBER):
                                col0 = base_col(tile_idx,pair_idx*2)
                                if col0 < DSTATE:
                                    local_max = max(
                                        local_max,
                                        max(abs(rState[tile_idx,pair_idx].x),
                                            abs(rState[tile_idx,pair_idx].y)))
                                    # instruction_selection: abs.ftz.f32 and max.ftz.f32 chain; extent: every live register pair
                        for delta in (4,2,1):
                            peer_max = shuffle_down(
                                local_max, delta, clamp=31,
                                member_mask=0xFFFFFFFF)
                            # instruction_selection: shfl.sync.down.b32 with delta 4/2/1; extent: one subgroup max step
                            local_max = max(local_max, peer_max)
                            # instruction_selection: max.ftz.f32; extent: one subgroup max step
                        leader_lane = lane & ~7
                        local_max = shuffle_index(
                            local_max, leader_lane, clamp=31,
                            member_mask=0xFFFFFFFF)
                        # instruction_selection: shfl.sync.idx.b32 from the first lane of this eight-lane group; extent: one max broadcast
                        if local_max != 0.0:
                            encode_scale = div(32767.0, local_max)
                            # instruction_selection: div.approx.ftz.f32; extent: one nonzero row/token encode scale

                    dst_base = (
                        dst_slot*write_state_stride + head*DIM*DSTATE + d*DSTATE)
                    for tile_idx in static_range(NUM_TILES):
                        col0 = base_col(tile_idx,0)
                        if col0 < DSTATE:
                            packed_out = reg_tile(STATE_DTYPE,
                                [ELEMS_PER_TILE_MEMBER])
                            random_words = reg_tile("u32", [4])
                            for element_idx in static_range(
                                    0,ELEMS_PER_TILE_MEMBER,2):
                                pair_value = rState[tile_idx,element_idx//2]
                                if STATE_DTYPE == "i16":
                                    pair_value = mul(
                                        pair_value,(encode_scale,encode_scale),lanes=2)
                                    # instruction_selection: mul.f32x2; extent: one quantized state pair
                                    pair_value = min(max(pair_value,-32767.0),32767.0)
                                    # instruction_selection: max/min.ftz.f32 per component; extent: one symmetric int16 clamp pair
                                    cast(packed_out[element_idx:element_idx+2],
                                         pair_value,rounding="rni")
                                    # instruction_selection: cvt.rni.ftz.s32.f32; extent: two scalar conversions per pair, eight conversions per 16-byte/eight-int16 member tile
                                elif PHILOX_ROUNDS > 0:
                                    if element_idx % 4 == 0:
                                        random_offset_i32 = mad_lo_s32(
                                            d,DSTATE,source_state_base_i32)
                                        # instruction_selection: mad.lo.s32; extent: helper-ABI signed-i32 row-offset accumulation after the deliberate base narrowing
                                        random_offset_i32 = add_s32(
                                            random_offset_i32,col0+element_idx)
                                        # instruction_selection: add.s32; extent: helper-ABI signed-i32 member offset for one four-element Philox group
                                        c0 = bitcast_u32(random_offset_i32)
                                        # instruction_selection: direct i32 bit-pattern reuse with no conversion instruction; extent: the low offset counter word
                                        c1 = arithmetic_shift_right_s32(
                                            random_offset_i32,31)
                                        # instruction_selection: shr.s32 by 31; extent: the high sign-fill offset counter word
                                        c2 = 0
                                        c3 = 0
                                        k0 = low_u32(random_seed)
                                        k1 = high_u32(random_seed)
                                        for philox_round in static_range(10):
                                            old_c0 = c0
                                            old_c2 = c2
                                            c0_hi = mul_hi_u32(
                                                0xCD9E8D57,old_c2)
                                            # instruction_selection: mul.hi.u32; logical extent: one source dependency per unrolled round, subject to normal round-zero constant folding
                                            c0_xor_c1 = bit_xor(c0_hi,c1)
                                            # instruction_selection: xor.b32; logical extent: one source dependency per unrolled round
                                            next_c0 = bit_xor(c0_xor_c1,k0)
                                            # instruction_selection: xor.b32; logical extent: one source dependency per unrolled round
                                            c2_hi = mul_hi_u32(
                                                0xD2511F53,old_c0)
                                            # instruction_selection: mul.hi.u32; logical extent: one source dependency per unrolled round
                                            c2_xor_c3 = bit_xor(c2_hi,c3)
                                            # instruction_selection: xor.b32; logical extent: one source dependency per unrolled round, with the round-zero c3 input constant
                                            next_c2 = bit_xor(c2_xor_c3,k1)
                                            # instruction_selection: xor.b32; logical extent: one source dependency per unrolled round
                                            next_c1 = mul_lo_s32(
                                                old_c2,0xCD9E8D57)
                                            # instruction_selection: mul.lo.s32; logical extent: one source dependency per unrolled round
                                            next_c3 = mul_lo_s32(
                                                old_c0,0xD2511F53)
                                            # instruction_selection: mul.lo.s32; logical extent: one source dependency per unrolled round
                                            next_k0 = add_s32(k0,0x9E3779B9)
                                            # instruction_selection: add.s32; logical extent: one key update per unrolled round
                                            next_k1 = add_s32(k1,0xBB67AE85)
                                            # instruction_selection: add.s32; logical extent: one key update per unrolled round
                                            c0,c1,c2,c3 = (
                                                next_c0,next_c1,next_c2,next_c3)
                                            k0,k1 = next_k0,next_k1
                                        random_words = (c0,c1,c2,c3)
                                    stochastic_bits = random_words[
                                        element_idx//2 % 2]
                                    # instruction_selection: no mask; extent: one full u32 whose bits 12:0 and 28:16 drive the two stochastic halves
                                    cast(packed_out[element_idx:element_idx+2],
                                         pair_value,
                                         rounding=("stochastic-f16x2",stochastic_bits))
                                    # instruction_selection: cvt.rs.f16x2.f32; extent: one complete adjacent FP16 pair
                                else:
                                    if STATE_DTYPE == "bf16":
                                        packed_low = cast(
                                            "bf16",pair_value.x,rounding="rn")
                                        packed_high = cast(
                                            "bf16",pair_value.y,rounding="rn")
                                        packed_out.word[element_idx//2] = \
                                            pack_b16x2_b32(packed_low,packed_high)
                                        # instruction_selection: cvt.rn.bf16.f32; extent: two scalar conversions per pair, eight conversions per 16-byte/eight-element member tile
                                        # instruction_selection: mov.b32 d,{lo,hi}; extent: one register-group pack per pair, four packs per member tile; keeping both 16-bit conversion results as the source vector is required for ptxas to select F2FP.BF16.F32.PACK_AB rather than scalar F2F plus PRMT
                                    elif STATE_DTYPE == "f16":
                                        packed_low = cast(
                                            "f16",pair_value.x,rounding="rn")
                                        packed_high = cast(
                                            "f16",pair_value.y,rounding="rn")
                                        packed_out.word[element_idx//2] = \
                                            pack_b16x2_b32(packed_low,packed_high)
                                        # instruction_selection: cvt.rn.f16.f32; extent: two scalar conversions per pair, eight conversions per 16-byte/eight-element member tile
                                        # instruction_selection: mov.b32 d,{lo,hi}; extent: one register-group pack per pair, four packs per member tile; this is the exact frozen CUDA PTX pack shape following the scalar conversions
                                    else:  # STATE_DTYPE == "f32"
                                        packed_out[element_idx] = pair_value.x
                                        packed_out[element_idx+1] = pair_value.y
                                        # instruction_selection: identity b32 register placement; extent: two FP32 values per pair, four values per 16-byte member tile
                            if STATE_DTYPE == "i16":
                                for packed_word in static_range(4):
                                    packed_out.word[packed_word] = permute_b32(
                                        packed_out.i32[packed_word*2],
                                        packed_out.i32[packed_word*2+1],
                                        selector=0x5410)
                                    # instruction_selection: prmt.b32 with selector 0x5410; extent: exactly four packing permutations per 16-byte/eight-int16 member tile
                            copy_r2g(
                                packed_out,
                                write_state[dst_base+col0:
                                            dst_base+col0+ELEMS_PER_TILE_MEMBER])
                            # instruction_selection: st.global.v4.b32; extent: exactly one 16-byte store for every member tile (eight BF16/FP16/int16 values or four FP32 values), repeated once per NUM_TILES tile owned by the thread

                    if STATE_DTYPE == "i16" and member == 0:
                        decode_scale = div(1.0,encode_scale)
                        # instruction_selection: rcp.approx.ftz.f32; extent: exactly one row/token decode scale
                        copy_r2g(
                            decode_scale,
                            write_scale[dst_slot*write_scale_stride + head*DIM + d])
                        # instruction_selection: st.global.b32; extent: one destination/intermediate row scale

            if NUM_PASSES > 1 and pass_idx < NUM_PASSES-1:
                next_stage = (pass_idx+1) % STATE_STAGES
                next_dim_base = dim_offset + (pass_idx+1)*ROWS_PER_PASS
                if not IS_PAD:
                    NUM_STATE_PACKS = ROWS_PER_PASS*DSTATE // (16//sizeof(STATE_DTYPE))
                    for packed_item in flat_thread_strided_range(
                            flat_tid,NUM_STATE_PACKS,NUM_WARPS*32):
                        row = packed_item // (DSTATE // (16//sizeof(STATE_DTYPE)))
                        col = (packed_item % (DSTATE // (16//sizeof(STATE_DTYPE)))) * \
                              (16//sizeof(STATE_DTYPE))
                        copy_g2s(
                            state[source_state_base + (next_dim_base+row)*DSTATE + col:
                                  +(16//sizeof(STATE_DTYPE))],
                            sState[next_stage,row,col:col+(16//sizeof(STATE_DTYPE))],
                            bytes=16)
                        # instruction_selection: cp.async.cg.shared.global with 16-byte extent; all 128 threads cooperatively cover the next 16-row pass
                copy_group_commit()
                # instruction_selection: cp.async.commit_group; extent: one next-pass state group per thread
                copy_group_wait_zero()
                # instruction_selection: cp.async.wait_group 0; extent: complete next-pass state visibility before the barrier
                cta_sync()
                # instruction_selection: bar.sync 0; extent: all 128 threads before reading the alternate state stage

        cta_sync()
        # instruction_selection: bar.sync 0; extent: all 128 threads after every pass has published sOut

        # ===================================================================
        # Fixed/varlen output epilogue and optional z gate
        # source: kernel_selective_state_update_mtp_simple.cuh:499-561
        # ===================================================================

        for step in warp_strided_range(warp,seq_len,NUM_WARPS):
            if cu_seqlens is present:
                out_base = (bos+step)*out_stride_batch + head*DIM + dim_offset
                z_base = (bos+step)*z_stride_batch + head*DIM + dim_offset
            else:
                out_base = (seq_idx*out_stride_batch + step*out_stride_mtp
                            + head*DIM + dim_offset)
                z_base = (seq_idx*z_stride_batch + step*z_stride_mtp
                          + head*DIM + dim_offset)

            if DIM_PER_CTA >= 32:
                EPILOGUE_ELEMS = DIM_PER_CTA // 32  # statically one, two, or four
                element_base = lane*EPILOGUE_ELEMS
                if DIM_PER_CTA == 32:
                    out_values = copy_s2r(
                        sOut[step,element_base:element_base+1], bytes=4)
                    # instruction_selection: ld.shared.b32; extent: one FP32 output element (four bytes) per lane
                elif DIM_PER_CTA == 64:
                    if OFF_OUT % 8 == 0:
                        out_values = copy_s2r(
                            sOut[step,element_base:element_base+2], bytes=8)
                        # instruction_selection: ld.shared.v2.b32; extent: two FP32 output elements (eight bytes) per lane for even-NTOKENS shapes
                    else:
                        out_values[0] = copy_s2r(
                            sOut[step,element_base], bytes=4)
                        out_values[1] = copy_s2r(
                            sOut[step,element_base+1], bytes=4)
                        # instruction_selection: ld.shared.b32; extent: exactly two scalar four-byte loads per lane for the NTOKENS=1 shape, whose source struct places OFF_OUT at byte offset 644 and whose frozen PTX scalarizes the unaligned PackedAligned request
                else:  # DIM_PER_CTA == 128
                    out_values = copy_s2r(
                        sOut[step,element_base:element_base+4], bytes=16)
                    # instruction_selection: ld.shared.v4.b32; extent: four FP32 output elements (16 bytes) per lane

                if z is present:
                    if DIM_PER_CTA == 32:
                        z_bits = copy_g2r(
                            z[z_base+element_base:z_base+element_base+1],bytes=2)
                        # instruction_selection: ld.global.b16; extent: one BF16 z element (two bytes) per lane
                    elif DIM_PER_CTA == 64:
                        z_bits = copy_g2r(
                            z[z_base+element_base:z_base+element_base+2],bytes=4)
                        # instruction_selection: ld.global.v2.b16; extent: two BF16 z elements (four bytes) per lane
                    else:
                        z_bits = copy_g2r(
                            z[z_base+element_base:z_base+element_base+4],bytes=8)
                        # instruction_selection: ld.global.v4.b16; extent: four BF16 z elements (eight bytes) per lane
                    for element in static_range(EPILOGUE_ELEMS):
                        z_value = cast("f32",z_bits[element])
                        # instruction_selection: cvt.f32.bf16; extent: one scalar conversion per z element, statically 1/2/4 conversions for DPC=32/64/128
                        exp_neg_z = exp2(mul(sub(0.0,z_value),LOG2_E))
                        # instruction_selection: sub.ftz/mul.ftz/ex2.approx.ftz.f32; extent: one z element
                        sigmoid_z = div(1.0,add(1.0,exp_neg_z))
                        # instruction_selection: add.ftz plus div.approx.ftz.f32; extent: one z element
                        out_values[element] = mul(
                            out_values[element],mul(z_value,sigmoid_z))
                        # instruction_selection: two mul.ftz.f32; extent: one SILU-gated output element

                output_bits = reg_tile("bf16",[EPILOGUE_ELEMS])
                for element in static_range(EPILOGUE_ELEMS):
                    output_bits[element] = cast(
                        "bf16",out_values[element],rounding="rn")
                    # instruction_selection: cvt.rn.bf16.f32; extent: one scalar conversion per output, statically 1/2/4 conversions for DPC=32/64/128
                if DIM_PER_CTA == 32:
                    copy_r2g(output_bits,output[out_base+element_base:
                                                 out_base+element_base+1],bytes=2)
                    # instruction_selection: st.global.b16; extent: one BF16 output element (two bytes) per lane
                elif DIM_PER_CTA == 64:
                    copy_r2g(output_bits,output[out_base+element_base:
                                                 out_base+element_base+2],bytes=4)
                    # instruction_selection: st.global.v2.b16; extent: two BF16 output elements (four bytes) per lane
                else:
                    copy_r2g(output_bits,output[out_base+element_base:
                                                 out_base+element_base+4],bytes=8)
                    # instruction_selection: st.global.v4.b16; extent: four BF16 output elements (eight bytes) per lane
            elif lane < DIM_PER_CTA:
                out_value = copy_s2r(sOut[step,lane])
                # instruction_selection: ld.shared.b32; extent: one active narrow-path output lane
                if z is present:
                    z_value = cast("f32",copy_g2r(z[z_base+lane]))
                    # instruction_selection: generic-state-space ld.b16 plus cvt.f32.bf16; extent: one active narrow-path z lane (ptxas resolves the generic pointer to a global SASS load)
                    exp_neg_z = exp2(mul(sub(0.0,z_value),LOG2_E))
                    # instruction_selection: sub.ftz/mul.ftz/ex2.approx.ftz.f32; extent: one z scalar
                    sigmoid_z = div(1.0,add(1.0,exp_neg_z))
                    # instruction_selection: add.ftz plus div.approx.ftz.f32; extent: one z scalar
                    out_value = mul(out_value,mul(z_value,sigmoid_z))
                    # instruction_selection: two mul.ftz.f32; extent: one gated output scalar
                output_value = cast("bf16",out_value,rounding="rn")
                # instruction_selection: cvt.rn.bf16.f32; extent: one active narrow-path output
                copy_r2g(output_value,output[out_base+lane])
                # instruction_selection: st.global.b16; extent: one active narrow-path output
```

## Source-to-sketch coverage

| Frozen source region | Sketch region | Represented facts |
| --- | --- | --- |
| lines 49-84 | pipeline constants and shared ABI | eight-lane rows, padding, B/C/x/dt/out/slot/state order, one/two stages |
| lines 90-119 | `preload_and_prepare_slots` | exact 16-byte state copies, flat 128-thread partition |
| lines 126-268 | initial preload | warp-0 B, warp-1 C, all-warp x/state, ordinary dt, slot precedence, commit/wait |
| lines 274-383 | pass setup and register state | eight-lane row mapping, bank-derived tiles, decode scale, pad zeros |
| lines 385-478 | token recurrence and writes | packed f32x2 recurrence, three shuffles, D epilogue, destination/intermediate/final state and scale, Philox |
| lines 480-497 | pass boundary | alternating state stage, next-pass copy, commit/wait, CTA barrier |
| lines 499-561 | output epilogue | final barrier, fixed/varlen addresses, wide/narrow paths, z gate |
| lines 570-663 | kernel entry | grid/block, varlen early return, accepted-token source slot, pad dispatch, A/D overlap |
| invoke lines 249-300 | host launch boundary | four warps, occupancy heuristic, ratio dispatch, dynamic shared size, CTA selection order |

## State-write mode ordering

The shared `sDstSlot[step]` is the only state-write decision consumed by the
recurrence. Its precedence is exact and intentionally differs from a single
`update_state` boolean:

1. pad CTA or inactive token: `-1`, never write;
2. `dst_state_batch_indices`: use that step's 1D/2D destination unless it is
   `pad_slot_id`;
3. `intermediate_states`: use `icache_idx*cache_steps+step`, and write the
   parallel intermediate scale for int16;
4. otherwise: write the original state slot only at the last live token and
   only when `update_state` is true.

All `CTAS_PER_HEAD` CTAs compute disjoint DIM rows. A destination slot may be
shared across them without a race because `cta_z` changes only the row interval.
Pad CTAs still load B/C/x/dt and produce valid outputs from zero initial state.

## Static and runtime boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| DIM, DSTATE, NTOKENS, dtype, group ratio, Philox rounds, CTAS_PER_HEAD | static | shared footprint, pass/stage count, register tile width, conversion and vector instruction families specialize |
| batch/sequence and nheads grid extents | static for a TIRx specialization | launch exactly matches the source grid |
| fixed versus varlen pointer presence | host-specialized pointer presence plus runtime lengths | base/stride formula and empty-sequence early return specialize without changing recurrence |
| source/destination/intermediate/accepted index presence and rank | host-specialized pointer shape | exact slot and initial-token address paths specialize |
| strides, softplus, update, pad slot | runtime values | source address arithmetic and final-state fallback remain visible |

## TIRx module and benchmark contract

- `KERNEL_META` names `selective_state_update_mtp_simple`, category
  `flashinfer`, compute capability 10.
- `CONFIGS` covers the one-variable correctness matrix, including fixed and
  uniform/variable/empty varlen, accepted-token int32/int64, 1D/2D indices,
  destination and intermediate caches, int16 state scales, and Philox-10.
- `BENCH_CONFIGS` includes the official powers-of-two batch sweep at T=6 for
  BF16/FP32 state and guard workloads for every remaining physical code shape
  or execution-body branch.
- TIRx and FlashInfer receive independent mutable state, scale, intermediate,
  and output buffers. Every source launch uses `algorithm="simple"`.
- Compilation, allocation, source JIT, warmup, and correctness preflight stay
  outside timed closures. `bench_suite` is the only performance authority and
  every row must satisfy `flashinfer_cuda/tirx > 0.99`.
- The implementation and every pre-dispatch specialization contain no tile
  primitive. CUDA helper calls are transcribed as normal TIRx or native
  `T.ptx` operations.

## Instruction-selection summary

- Initial B/C/x/state movement and every multi-pass state refill select exact
  16-byte `cp.async.cg.shared.global`, followed by one
  `cp.async.commit_group`, `cp.async.wait_group 0`, and the source-order CTA
  barrier.
- Each row is owned by eight adjacent lanes. State is converted to FP32 register
  pairs once per pass and remains there through all token steps. Recurrence and
  C accumulation select native `mul.f32x2` and `fma.rn.f32x2`; dA and z use the
  source fast-math exp/log/div families.
- Output reduction is exactly three `shfl.sync.down.b32` steps with deltas
  4, 2, and 1. Int16 scale reduction repeats those three max shuffles and uses
  an eight-lane-group leader broadcast before symmetric clipping and scale
  publication.
- Philox-10 preserves the helper ABI's signed-i32 offset narrowing, the full
  unrolled four-counter dependency graph, and the unmasked packed random word
  consumed by `cvt.rs.f16x2.f32`. Non-stochastic BF16/FP16 state stores use two
  scalar nearest-even conversions per pair; FP32 is identity. Every member-tile
  state store is exactly one 16-byte `st.global.v4.b32`.
- `DIM_PER_CTA>=32` selects the wide output path. DPC=32 uses one scalar shared
  load, DPC=64 normally uses `ld.shared.v2.b32`, and DPC=128 uses
  `ld.shared.v4.b32`. The unique `NTOKENS=1,DPC=64` shape has `OFF_OUT=644`,
  so the frozen compiler emits two scalar `ld.shared.b32` instructions instead
  of an unaligned shared vector load; its global z/output operations remain
  `v2.b16`. Only the `DIM64,CTAS_PER_HEAD=4` shape has DPC=16 and selects the
  guarded narrow scalar path.
